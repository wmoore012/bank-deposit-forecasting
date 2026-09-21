"""Zero-shot TimesFM 3.0 check on the notebook's frozen 2024 bank-quarters.

Run with ``uv run python benchmark_timesfm.py`` on an Apple Silicon Mac.
The project pins ``timesfm[mlx]==3.0.2``. The checkpoint comes from Google's Hugging Face
repository; set HF_HOME outside this project to keep weights out of bundles.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CHECKPOINT = "google/timesfm-3.0-pytorch"
SOURCE = ROOT / "data/fdic_financials_2013_2024.csv"
HOLDOUT = ROOT / "growth_outputs/submission/predictions.csv"
DEFAULT_OUTPUT = ROOT / "growth_outputs/timesfm_zero_shot_predictions.csv"


def build_contexts(source: pd.DataFrame, holdout: pd.DataFrame):
    """Use only each bank's consecutive positive reports through its predictor date."""
    source = source.copy()
    source["date"] = pd.to_datetime(source.REPDTE.astype(str), format="%Y%m%d")
    source["quarter"] = source.date.dt.to_period("Q").astype("int64")
    source = source.sort_values(["CERT", "quarter"])
    assert not source.duplicated(["CERT", "quarter"]).any()

    histories = {
        int(cert): (
            group.quarter.to_numpy(),
            group.DEPDOM.to_numpy(dtype=np.float64),
        )
        for cert, group in source.groupby("CERT", sort=False)
    }

    contexts = []
    starts = []
    for row in holdout.itertuples(index=False):
        quarter = pd.Period(row.date, freq="Q").ordinal
        quarters, deposits = histories[int(row.CERT)]
        end = int(np.searchsorted(quarters, quarter, side="right"))
        assert end and quarters[end - 1] == quarter
        assert deposits[end - 1] == row.DEPDOM

        start = end - 1
        while (
            start > 0
            and deposits[start - 1] > 0
            and quarters[start] - quarters[start - 1] == 1
        ):
            start -= 1

        balances = deposits[start:end]
        assert len(balances) >= 2 and (balances > 0).all()
        assert quarters[end - 1] == quarter
        contexts.append(np.log(balances).astype(np.float32))
        starts.append(pd.Period(ordinal=int(quarters[start]), freq="Q").end_time.date())

    return contexts, starts


def main():
    from huggingface_hub import HfApi
    from timesfm3.mlx import TimesFM3Forecaster

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Smoke-test the first N rows")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--revision", help="Checkpoint commit SHA; defaults to current SHA")
    args = parser.parse_args()
    assert args.batch_size > 0

    holdout = pd.read_csv(HOLDOUT, parse_dates=["date", "target_date"])
    if args.limit is not None:
        assert args.limit > 0
        holdout = holdout.head(args.limit).copy()
    assert holdout.date.between("2024-01-01", "2024-09-30").all()
    assert holdout.target_date.gt(holdout.date).all()

    source = pd.read_csv(SOURCE, usecols=["REPDTE", "CERT", "DEPDOM"])
    contexts, starts = build_contexts(source, holdout)
    revision = args.revision or HfApi().model_info(CHECKPOINT).sha
    print(
        f"TimesFM {version('timesfm')}; checkpoint {CHECKPOINT}@{revision}; "
        f"{len(contexts):,} one-step forecasts",
        flush=True,
    )
    print(
        f"History length: min {min(map(len, contexts))}, "
        f"median {np.median(list(map(len, contexts))):.0f}, "
        f"max {max(map(len, contexts))}",
        flush=True,
    )

    forecaster = TimesFM3Forecaster.from_pretrained(
        CHECKPOINT,
        revision=revision,
        per_core_batch_size=args.batch_size,
    )
    forecast_logs = []
    for start in range(0, len(contexts), args.batch_size):
        batch = contexts[start : start + args.batch_size]
        outputs = list(
            forecaster.predict_batch(batch, horizon=1, return_quantiles=False)
        )
        assert len(outputs) == len(batch)
        forecast_logs.extend(float(out.forecast[0]) for out in outputs)
        if (start // args.batch_size) % 50 == 0:
            print(f"Forecasts completed: {len(forecast_logs):,}", flush=True)

    forecast_logs = np.asarray(forecast_logs, dtype=np.float64)
    assert np.isfinite(forecast_logs).all()
    growth = np.expm1(forecast_logs - np.log(holdout.DEPDOM.to_numpy()))
    assert np.isfinite(growth).all()

    result = holdout[["CERT", "date", "DEPDOM", "target_date"]].copy()
    result["Context start"] = starts
    result["Context quarters"] = list(map(len, contexts))
    result["Predicted log deposits"] = forecast_logs
    result["TimesFM"] = growth
    assert not result.duplicated(["CERT", "date"]).any()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    error = 100 * (growth - holdout.growth.to_numpy())
    metadata = {
        "model": "TimesFM 3.0",
        "package_version": version("timesfm"),
        "checkpoint": CHECKPOINT,
        "checkpoint_revision": revision,
        "mode": "zero-shot univariate; no covariates or holdout tuning",
        "input": "log of each bank's consecutive positive quarterly DEPDOM reports through predictor date",
        "output": "one-step predicted log deposit level, converted to ordinary growth",
        "rows": len(result),
        "context_min": int(min(map(len, contexts))),
        "context_median": float(np.median(list(map(len, contexts)))),
        "context_max": int(max(map(len, contexts))),
        "MAE (pp)": float(np.mean(np.abs(error))),
        "RMSE (pp)": float(np.sqrt(np.mean(error**2))),
        "license": "TimesFM 3.0 pretrained weights: noncommercial, nonproduction research use",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
