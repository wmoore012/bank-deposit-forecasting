"""Run the predeclared, retrospective eight-quarter review-list comparison."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from deposit_experiment import FEATURES, prepare_population
from freeze_forecasts import fingerprint, selected_keys
from review_history import (
    feature_histories,
    standardize_histories,
    reconstruction_error,
    fit_pca,
    fit_isolation_forest,
    fit_autoencoder,
    review_flags,
    list_overlaps,
    outcome_capture,
    choose_examples,
)


def run_study(root, output):
    output.mkdir(parents=True, exist_ok=True)
    models_dir = output / "models"
    models_dir.mkdir(exist_ok=True)
    raw_path = root / "data/fdic_financials_2013_2024.csv"
    metadata_path = root / "data/fdic_reporting_2010_2024.csv"
    inputs, _ = prepare_population(pd.read_csv(raw_path), pd.read_csv(metadata_path))
    rows, histories, audit = feature_histories(inputs)
    evaluation_audit = audit.loc[audit.date.between("2024-01-01", "2024-09-30")]
    observed = evaluation_audit["Outcome status"].eq("Usable adjacent outcome")
    counts = {
        "original": int(observed.sum()),
        "expanded": len(evaluation_audit),
        "original_complete": int((observed & evaluation_audit.history_eligible).sum()),
        "expanded_complete": int(evaluation_audit.history_eligible.sum()),
    }
    assert counts == {
        "original": 13532,
        "expanded": 13618,
        "original_complete": 13490,
        "expanded_complete": 13576,
    }, counts
    evaluation_audit.to_csv(output / "eligibility_audit.csv", index=False)
    evaluation_audit.groupby(["date", "exclusion_reason"]).size().rename(
        "rows"
    ).reset_index().to_csv(output / "coverage.csv", index=False)
    masks = [
        rows.date.le("2022-09-30"),
        rows.date.between("2023-01-01", "2023-09-30"),
        rows.date.between("2024-01-01", "2024-09-30"),
    ]
    populations = [rows.loc[mask].reset_index(drop=True) for mask in masks]
    train, valid, test = populations
    history_splits = [histories[mask] for mask in masks]
    extended_path = root / "growth_outputs/frozen_forecasts/expanded_predictions.csv"
    forecast = pd.read_csv(extended_path, parse_dates=["date"])
    forecasts = test[["CERT", "date"]].merge(
        forecast[["CERT", "date", "Ridge", "MLP"]], on=["CERT", "date"], validate="one_to_one"
    )
    assert len(forecasts) == len(test)
    score = pd.DataFrame(
        {
            "Ridge": -forecasts.Ridge,
            "MLP": -forecasts.MLP,
            "Low equity/assets": -test.equity_ratio,
            "Largest deposits": test.DEPDOM,
        }
    )
    plan = {
        "question": "At 10% review capacity, does eight-quarter financial history improve selection for low next-quarter deposit growth?",
        "status": "Exploratory retrospective comparison; 2024 outcomes already inspected.",
        "population_counts": counts,
        "features": list(FEATURES),
        "flatten_order": "oldest quarter first, feature order within quarter",
        "seed_primary": 42,
        "stochastic_seeds": [42, 7, 99],
        "capacity": 0.10,
        "tie_break": "CERT ascending",
        "pca": {"variance": 0.90, "solver": "full"},
        "isolation_forest": {
            "trees": 300,
            "max_samples": 256,
            "contamination": "auto",
            "score": "negative score_samples",
        },
        "autoencoder": {
            "dimensions": [40, 16, 4, 16, 40],
            "without_size_dimensions": [32, 16, 4, 16, 32],
            "hidden_activation": "relu",
            "bottleneck_output": "linear",
            "loss": "standardized MSE",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch": 256,
            "max_epochs": 100,
            "patience": 10,
            "restore_best": True,
        },
        "rule_origin": "post_hoc_failed_bank_exploration",
        "split_rows": {
            name: len(frame)
            for name, frame in zip(["training", "validation", "evaluation"], populations)
        },
        "training_dates": sorted(train.date.dt.strftime("%Y-%m-%d").unique().tolist()),
        "source_hashes": {
            str(p.relative_to(root)): fingerprint(p)
            for p in [
                raw_path,
                metadata_path,
                extended_path,
                root / "review_history.py",
                root / "experiments/review_history/run.py",
                root / "uv.lock",
            ]
        },
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name, frame in zip(["training", "validation", "evaluation"], populations):
        frame[["CERT", "date"]].to_csv(output / f"{name}_rows.csv", index=False)
    fitted, shares = [], []
    for representation, feature_indices in [
        ("full", list(range(5))),
        ("without_size", list(range(1, 5))),
    ]:
        blocks = [values[:, :, feature_indices] for values in history_splits]
        scaler, (X_train, X_valid, X_test) = standardize_histories(*blocks)
        joblib.dump(scaler, models_dir / f"{representation}_scaler.joblib")
        suffix = "" if representation == "full" else " without size"
        start = time.perf_counter()
        pca = fit_pca(X_train)
        predicted = pca.inverse_transform(pca.transform(X_test))
        error, contributions = reconstruction_error(X_test, predicted, len(feature_indices))
        score["PCA" + suffix] = error
        joblib.dump(pca, models_dir / f"{representation}_pca.joblib")
        fitted.append(
            {
                "method": "PCA" + suffix,
                "components": int(pca.n_components_),
                "training_variance": float(pca.explained_variance_ratio_.sum()),
                "seconds": time.perf_counter() - start,
            }
        )
        if representation == "full":
            share = test[["CERT", "date"]].copy()
            share["method"] = "PCA"
            for j, name in enumerate(FEATURES):
                share[name] = contributions[:, j]
            shares.append(share)
        print(representation, "PCA complete", pca.n_components_, "components", flush=True)
        for seed in [42, 7, 99]:
            start = time.perf_counter()
            model = fit_isolation_forest(X_train, seed)
            name = f"Isolation Forest {seed}" + suffix
            score[name] = -model.score_samples(X_test)
            joblib.dump(model, models_dir / f"{representation}_isolation_{seed}.joblib")
            fitted.append({"method": name, "seed": seed, "seconds": time.perf_counter() - start})
            print(name, "complete", flush=True)
        for seed in [42, 7, 99]:
            start = time.perf_counter()
            model, history = fit_autoencoder(X_train, X_valid, seed)
            predicted = np.asarray(model(X_test.astype("float32"), training=False))
            error, contributions = reconstruction_error(X_test, predicted, len(feature_indices))
            name = f"Autoencoder {seed}" + suffix
            score[name] = error
            model.save(models_dir / f"{representation}_autoencoder_{seed}.keras")
            (output / f"{representation}_learning_{seed}.json").write_text(
                json.dumps(history) + "\n"
            )
            fitted.append(
                {
                    "method": name,
                    "seed": seed,
                    "epochs": len(history["loss"]),
                    "best_epoch": int(np.argmin(history["val_loss"]) + 1),
                    "seconds": time.perf_counter() - start,
                }
            )
            if representation == "full" and seed == 42:
                share = test[["CERT", "date"]].copy()
                share["method"] = name
                for j, feature in enumerate(FEATURES):
                    share[feature] = contributions[:, j]
                shares.append(share)
            print(name, "complete", len(history["loss"]), "epochs", flush=True)
    # Freeze continuous scores before opening the observed-outcome comparison.
    scored = pd.concat([test[["CERT", "NAME", "date"]].reset_index(drop=True), score], axis=1)
    scored.to_csv(output / "scores.csv", index=False)
    pd.DataFrame(fitted).to_csv(output / "fits.csv", index=False)
    pd.concat(shares, ignore_index=True).to_csv(output / "reconstruction_shares.csv", index=False)
    manifest = {
        "plan_sha256": fingerprint(output / "plan.json"),
        "scores_sha256": fingerprint(output / "scores.csv"),
        "models": {p.name: fingerprint(p) for p in sorted(models_dir.iterdir())},
    }
    (output / "score_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    assert manifest["scores_sha256"] == fingerprint(output / "scores.csv")

    primary_flags = review_flags(test, score)
    examples = choose_examples(test, score, primary_flags)
    examples.to_csv(output / "examples.csv", index=False)
    example_rows = []
    for record in examples.dropna(subset=["CERT"]).itertuples(index=False):
        bank = inputs.loc[inputs.CERT.eq(record.CERT) & inputs.date.le("2024-03-31")].tail(8).copy()
        bank["group"] = record.group
        example_rows.append(bank[["group", "CERT", "NAME", "date", "DEPDOM", *FEATURES]])
    pd.concat(example_rows).to_csv(output / "example_histories.csv", index=False)
    for population, mask in [
        ("expanded", np.ones(len(test), dtype=bool)),
        ("original", test.growth.notna().to_numpy()),
    ]:
        common = test.loc[mask].reset_index(drop=True)
        scores = score.loc[mask].reset_index(drop=True)
        flags = review_flags(common, scores)
        pd.concat([common[["CERT", "date"]], flags], axis=1).to_csv(
            output / f"{population}_selections.csv", index=False
        )
        list_overlaps(common, flags).to_csv(output / f"{population}_overlap.csv", index=False)
        if population == "original":
            outcome_capture(common, flags).to_csv(output / "outcome_capture.csv", index=False)
    # Preserve original membership; attach available scores instead of reranking it.
    original = pd.read_csv(
        root / "growth_outputs/masterclass/predictions.csv", parse_dates=["date"]
    )
    attached = original[["CERT", "NAME", "date"]].copy()
    for method in ["Ridge", "MLP"]:
        keys = selected_keys(original, original[method])
        attached[method + " original selected"] = [
            (str(d.date()), int(c)) in keys for c, d in zip(original.CERT, original.date)
        ]
    attached = attached.merge(
        scored.drop(columns=["NAME", "Ridge", "MLP"]),
        on=["CERT", "date"],
        how="left",
        validate="one_to_one",
    )
    attached["anomaly_score_available"] = attached.PCA.notna()
    attached.to_csv(output / "original_list_audit.csv", index=False)
    print("Completed benchmark:", counts, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/review_history/outputs")
    args = parser.parse_args()
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    run_study(ROOT, args.output)
