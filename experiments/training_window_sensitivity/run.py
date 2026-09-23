"""Reproduce the exploratory fixed-holdout training-window comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
ROOT = Path(__file__).resolve().parents[2]
# Direct script execution resolves the same local module as the notebooks.
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import tensorflow as tf
from deposit_experiment import (
    FEATURES, prepare_population, chronological_splits, fit_preprocessor,
    transform_features, choose_ridge, fit_network, predict_log_growth,
    scores, review_precision, compare_training_windows,
)


def run_experiment(root, output):
    """Train all declared windows/seeds; save complete, comparable results."""
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(root / 'data/fdic_financials_2013_2024.csv')
    metadata = pd.read_csv(root / 'data/fdic_reporting_2010_2024.csv')
    _, rows = prepare_population(raw, metadata)
    _, valid, test = chronological_splits(rows)
    assert len(valid) == 13834 and len(test) == 13532
    original = pd.read_csv(root / 'growth_outputs/masterclass/predictions.csv')
    assert list(zip(test.CERT, test.date.dt.strftime('%Y-%m-%d'))) == list(zip(original.CERT, original.date))
    np.testing.assert_allclose(test.growth, original.growth)
    plan = {
        'status': 'Exploratory: 2024 already inspected; no fresh holdout',
        'training_starts': [2013, 2020, 2021], 'training_end': '2022-09-30',
        'validation_predictors': '2023 Q1-Q3', 'test_predictors': '2024 Q1-Q3',
        'seeds': [42, 7, 99], 'features': list(FEATURES),
        'architecture': [5, 32, 16, 1], 'epochs_max': 200, 'patience': 10,
        'batch_size': 512, 'learning_rate': 0.001, 'ridge_alphas': [.01, .1, 1, 10, 100],
        'rationale': 'Test more recent training while holding 2023 selection and 2024 scoring fixed. Recency and sample size both change.',
        'source_sha256': hashlib.sha256((root / 'data/fdic_financials_2013_2024.csv').read_bytes()).hexdigest(),
        'reporting_sha256': hashlib.sha256((root / 'data/fdic_reporting_2010_2024.csv').read_bytes()).hexdigest(),
        'core_sha256': hashlib.sha256((root / 'deposit_experiment.py').read_bytes()).hexdigest(),
    }
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    result, predictions = compare_training_windows(rows)
    reference = result.loc[result.training_start.eq(2013) & result.model.eq('MLP') & result.seed.eq(42)].iloc[0]
    assert abs(reference.MAE_pp - 3.627317978) < .0001
    result.to_csv(output / 'scores.csv', index=False)
    predictions.to_csv(output / 'predictions.csv', index=False)
    zero_scores = scores(test.growth, np.zeros(len(test)))
    zero = {'MAE_pp': zero_scores['MAE (pp)'], 'RMSE_pp': zero_scores['RMSE (pp)']}
    zero['bottom_decile_precision'] = None
    (output / 'zero_baseline.json').write_text(json.dumps(zero, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run_experiment(ROOT, args.output)
