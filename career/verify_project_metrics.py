"""Recalculate the figures used in the FDIC project resume entry.

FDIC DEPDOM balances are reported in USD thousands. A deposit decline is a
change in an account balance, not a loss to the FDIC or money saved by a model.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "growth_outputs/submission/predictions.csv"
SCORES = ROOT / "growth_outputs/submission/historical_scores.csv"


def read_predictions() -> list[dict[str, str]]:
    with PREDICTIONS.open(newline="") as source:
        return list(csv.DictReader(source))


def read_scores() -> dict[str, dict[str, float]]:
    with SCORES.open(newline="") as source:
        return {
            row["Model"]: {
                "mae_pp": float(row["MAE (pp)"]),
                "rmse_pp": float(row["RMSE (pp)"]),
            }
            for row in csv.DictReader(source)
        }


def dollar_context(rows: list[dict[str, str]]) -> dict[str, object]:
    deposits_by_quarter = defaultdict(float)
    gross_declines_thousands = 0.0

    for row in rows:
        current = float(row["DEPDOM"])
        following = float(row["next_deposits"])

        deposits_by_quarter[row["date"]] += current
        gross_declines_thousands += max(current - following, 0.0)

    return {
        "holdout_bank_quarters": len(rows),
        "deposits_by_predictor_quarter_usd_trillion": {
            quarter: round(balance / 1_000_000_000, 3)
            for quarter, balance in sorted(deposits_by_quarter.items())
        },
        "gross_observed_declines_usd_billion": round(
            gross_declines_thousands / 1_000_000, 3
        ),
    }


def verify_growth_error(
    rows: list[dict[str, str]], scores: dict[str, dict[str, float]]
) -> float:
    """Check saved RMSE against ordinary-growth predictions in percentage points."""
    for model in ("Zero growth", "MLP"):
        squared_errors = [
            (100 * (float(row[model]) - float(row["growth"]))) ** 2
            for row in rows
        ]
        calculated_rmse = sqrt(sum(squared_errors) / len(rows))
        assert abs(calculated_rmse - scores[model]["rmse_pp"]) < 1e-6

    zero_rmse = scores["Zero growth"]["rmse_pp"]
    mlp_rmse = scores["MLP"]["rmse_pp"]
    return round(100 * (zero_rmse - mlp_rmse) / zero_rmse, 2)


def main() -> None:
    rows = read_predictions()
    scores = read_scores()

    summary = dollar_context(rows)
    summary["mlp_rmse_reduction_vs_zero_growth_percent"] = verify_growth_error(
        rows, scores
    )
    summary["zero_growth_mae_pp"] = round(scores["Zero growth"]["mae_pp"], 3)
    summary["mlp_rmse_pp"] = round(scores["MLP"]["rmse_pp"], 3)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
