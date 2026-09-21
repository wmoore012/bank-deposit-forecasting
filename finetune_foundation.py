"""Train-only forecast-head adaptation; validation selects checkpoint, never 2024.

Prerequisite: prepare_finetune_data.py exports the notebook's exact eligible splits.
Fixed protocol: seed 42, 256 batches of 32, Adam 1e-5, log-balance pinball loss.
Backbones stay frozen. Checkpoints 0/128/256 compete on 2023 log-balance MAE.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark_timesfm import ROOT, SOURCE, build_contexts

STEPS = 256
BATCH = 32
RATE = 1e-5
LOCAL = ROOT / '.finetune'


def get_split(name, raw):
    frame = pd.read_csv(LOCAL / f'{name}.csv', parse_dates=['date', 'target_date'])
    contexts, _ = build_contexts(raw, frame)
    labels = np.log(frame.next_deposits.to_numpy()).astype('float32')
    return frame, contexts, labels


def batches(contexts, size=64):
    # Exact-length batching preserves TimesFM's original normalization and padding.
    lengths = np.array(list(map(len, contexts)))
    for length in np.unique(lengths):
        indices = np.flatnonzero(lengths == length)
        for offset in range(0, len(indices), size):
            ids = indices[offset:offset + size]
            yield ids, np.stack([contexts[i] for i in ids])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=['timesfm', 'chronos'])
    args = parser.parse_args()
    raw = pd.read_csv(SOURCE, usecols=['CERT', 'REPDTE', 'DEPDOM'])
    train, contexts, target = get_split('train', raw)
    valid, vcontexts, vtarget = get_split('valid', raw)
    assert train.target_date.max() < valid.date.min()
    assert train.target_date.max() <= pd.Timestamp('2022-12-31')
    assert valid.target_date.max() <= pd.Timestamp('2023-12-31')
    rng = np.random.default_rng(42)
    lengths = np.array(list(map(len, contexts)))
    choices = {n: np.flatnonzero(lengths == n) for n in np.unique(lengths)}
    out = ROOT / 'growth_outputs' / f'{args.model}_finetuned_predictions.csv'

    if args.model == 'chronos':
        import torch
        from chronos import BaseChronosPipeline
        torch.manual_seed(42)
        torch.set_num_threads(4)
        revision = json.loads((ROOT / 'growth_outputs/chronos_zero_shot_predictions.json').read_text())['checkpoint_revision']
        pipeline = BaseChronosPipeline.from_pretrained('amazon/chronos-bolt-small', revision=revision,
                                                       device_map='cpu', torch_dtype=torch.float32, local_files_only=True)
        model = pipeline.model
        for param in model.parameters():
            param.requires_grad = False
        head = model.output_patch_embedding
        for param in head.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(head.parameters(), lr=RATE)
        initial = {k: v.detach().clone() for k, v in head.state_dict().items()}
        trainable = sum(p.numel() for p in head.parameters())
        total = sum(p.numel() for p in model.parameters())

        def predict(xs):
            model.eval()
            result = np.empty(len(xs))
            with torch.no_grad():
                for ids, batch in batches(xs):
                    result[ids] = model(context=torch.tensor(batch)).quantile_preds[:, 4, 0].numpy()
            return result

        def update(batch, y):
            # eval keeps dropout fixed; gradients still update the head.
            model.eval()
            optimizer.zero_grad()
            pred = model(context=torch.tensor(batch)).quantile_preds[:, :, 0]
            difference = torch.tensor(y)[:, None] - pred
            q = model.quantiles.detach()[None, :]
            loss = torch.maximum(q * difference, (q - 1) * difference).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
            return float(loss.detach())

        def snapshot():
            return {k: v.detach().clone() for k, v in head.state_dict().items()}

        def restore(state):
            head.load_state_dict(state)

        def save(state):
            torch.save(state, LOCAL / 'chronos_head.pt')

        def changed(state):
            return any(not torch.equal(initial[k], v) for k, v in state.items())
    else:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten
        from timesfm3.mlx import TimesFM3Forecaster
        revision = json.loads((ROOT / 'growth_outputs/timesfm_zero_shot_predictions.json').read_text())['checkpoint_revision']
        forecaster = TimesFM3Forecaster.from_pretrained('google/timesfm-3.0-pytorch', revision=revision,
                                                       local_files_only=True, compile=False)
        # Horizon-statistic refinement contains unused masked branches whose zero-
        # variance derivatives are undefined. Freeze that normalization gradient,
        # preserving exactly the original forward calculation and head gradients.
        from timesfm3.mlx import cpm_revin_refine
        original_refine = cpm_revin_refine.cpm_iterative_revin_refine
        def fixed_refine(raw_logits, *args, **kwargs):
            return original_refine(mx.stop_gradient(raw_logits), *args, **kwargs)
        cpm_revin_refine.cpm_iterative_revin_refine = fixed_refine
        model = forecaster.model
        model.freeze()
        model.output_head.unfreeze()
        optimizer = optim.Adam(learning_rate=RATE)
        initial = {k: np.array(v) for k, v in tree_flatten(model.trainable_parameters())}
        trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
        total = sum(v.size for _, v in tree_flatten(model.parameters()))

        def predict(xs):
            result = np.empty(len(xs))
            for ids, batch in batches(xs, 32):
                pred = model.decode(mx.array(batch)[:, None, :], horizon=1)[:, 0, 0, :]
                result[ids] = np.sort(np.array(pred), axis=-1)[:, 4]
            return result

        def loss_fn(network, batch, y):
            pred = network.decode(batch[:, None, :], horizon=1)[:, 0, 0, :]
            difference = y[:, None] - pred
            q = mx.array(network.config.quantiles)[None, :]
            return mx.maximum(q * difference, (q - 1) * difference).mean()

        value_grad = nn.value_and_grad(model, loss_fn)

        def update(batch, y):
            loss, gradients = value_grad(model, mx.array(batch), mx.array(y))
            gradients, _ = optim.clip_grad_norm(gradients, 1.)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            return float(loss)

        def snapshot():
            return {k: np.array(v) for k, v in tree_flatten(model.trainable_parameters())}

        def restore(state):
            model.load_weights([(k, mx.array(v)) for k, v in state.items()], strict=False)

        def save(state):
            mx.save_safetensors(str(LOCAL / 'timesfm_head.safetensors'), {k: mx.array(v) for k, v in state.items()})

        def changed(state):
            return any(not np.array_equal(initial[k], v) for k, v in state.items())

    print(f'{args.model}: {trainable:,}/{total:,} parameters trainable; {len(train):,} eligible training rows', flush=True)
    baseline = float(np.abs(predict(vcontexts) - vtarget).mean())
    best, best_step, best_state = baseline, 0, snapshot()
    records = [{'step': 0, 'validation_log_MAE': baseline}]
    seen = set()
    for step in range(1, STEPS + 1):
        anchor = int(rng.integers(len(contexts)))
        ids = rng.choice(choices[lengths[anchor]], BATCH, replace=True)
        seen.update(map(int, ids))
        loss = update(np.stack([contexts[i] for i in ids]), target[ids])
        assert np.isfinite(loss), f'Nonfinite loss at step {step}'
        if step % 32 == 0:
            print(f'{args.model} step {step}/{STEPS}; training loss {loss:.5f}', flush=True)
        if step % 128 == 0:
            score = float(np.abs(predict(vcontexts) - vtarget).mean())
            assert np.isfinite(score)
            records.append({'step': step, 'validation_log_MAE': score})
            print(f'Validation log MAE: {score:.6f}', flush=True)
            if score < best:
                best, best_step, best_state = score, step, snapshot()
    candidate_changed = changed(snapshot())
    assert candidate_changed, 'No weight update occurred'
    restore(best_state)
    save(best_state)
    # Freeze the decision before opening any holdout outcomes.
    metadata = dict(model=args.model, checkpoint_revision=revision, seed=42,
                    steps=STEPS, batch_size=BATCH, learning_rate=RATE,
                    trainable_parameters=trainable, total_parameters=total,
                    training_rows=len(train), training_unique_rows_seen=len(seen),
                    train_last_outcome=str(train.target_date.max().date()),
                    validation_last_outcome=str(valid.target_date.max().date()),
                    selection_metric='2023 mean absolute log-balance error',
                    selected_step=best_step, validation=records,
                    candidate_weights_changed=candidate_changed,
                    selected_weights_changed=changed(best_state),
                    adaptation='forecast head only; frozen backbone; 256 sampled updates, not a full epoch',
                    normalization_gradient='TimesFM horizon refinement treated as fixed during backpropagation; forward calculation unchanged' if args.model == 'timesfm' else 'original implementation',
                    pretrained_overlap='unknown; no claim of globally leakage-free pretraining')
    decision = LOCAL / f'{args.model}_selection.json'
    decision.write_text(json.dumps(metadata, indent=2) + '\n')
    metadata['selection_sha256'] = hashlib.sha256(decision.read_bytes()).hexdigest()
    test, tcontexts, _ = get_split('holdout', raw)
    assert valid.target_date.max() < test.date.min()
    predicted_logs = predict(tcontexts)
    growth = np.expm1(predicted_logs - np.log(test.DEPDOM.to_numpy()))
    assert np.isfinite(growth).all()
    result = test[['CERT', 'date', 'DEPDOM', 'target_date']].copy()
    result['Prediction'] = growth
    result.to_csv(out, index=False)
    error = 100 * (growth - test.growth.to_numpy())
    metadata['rows'] = len(test)
    metadata['MAE (pp)'] = float(np.abs(error).mean())
    metadata['RMSE (pp)'] = float(np.sqrt(np.square(error).mean()))
    out.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
