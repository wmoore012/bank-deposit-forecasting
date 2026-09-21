"""A prespecified second zero-shot comparison; no fitting on the reused holdout."""
import json
from importlib.metadata import version
import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline
from huggingface_hub import HfApi
from benchmark_timesfm import ROOT, SOURCE, HOLDOUT, build_contexts


def main():
    torch.set_num_threads(4)
    checkpoint = 'amazon/chronos-bolt-small'
    revision = HfApi().model_info(checkpoint).sha
    holdout = pd.read_csv(HOLDOUT, parse_dates=['date', 'target_date'])
    source = pd.read_csv(SOURCE, usecols=['REPDTE', 'CERT', 'DEPDOM'])
    contexts, starts = build_contexts(source, holdout)
    model = BaseChronosPipeline.from_pretrained(
        checkpoint, revision=revision, device_map='cpu', torch_dtype=torch.float32,
    )
    predictions = []
    for start in range(0, len(contexts), 64):
        batch = [torch.tensor(x) for x in contexts[start:start + 64]]
        quantiles, _ = model.predict_quantiles(
            batch, prediction_length=1, quantile_levels=[0.5],
        )
        predictions.extend(quantiles[:, 0, 0].numpy().tolist())
        if start % 1024 == 0:
            print(f'Completed {len(predictions):,} forecasts', flush=True)
    logs = np.asarray(predictions, dtype=float)
    growth = np.expm1(logs - np.log(holdout.DEPDOM.to_numpy()))
    assert np.isfinite(growth).all()
    result = holdout[['CERT', 'date', 'DEPDOM', 'target_date']].copy()
    result['Context start'] = starts
    result['Context quarters'] = list(map(len, contexts))
    result['Predicted log deposits'] = logs
    result['Chronos-Bolt'] = growth
    path = ROOT / 'growth_outputs/chronos_zero_shot_predictions.csv'
    result.to_csv(path, index=False)
    error = 100 * (growth - holdout.growth.to_numpy())
    metadata = dict(
        model='Chronos-Bolt Small', checkpoint=checkpoint, checkpoint_revision=revision,
        package_version=version('chronos-forecasting'), rows=len(result),
        mode='zero-shot; median forecast; CPU float32; no fitting or holdout tuning',
        input='same consecutive positive log-deposit contexts as TimesFM',
        retrospective=True, pretraining_overlap='not independently ruled out',
    )
    metadata['MAE (pp)'] = float(np.abs(error).mean())
    metadata['RMSE (pp)'] = float(np.sqrt(np.square(error).mean()))
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
