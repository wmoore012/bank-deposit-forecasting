"""Reproduce Study 1, verify it, and save the fitted objects for expanded scoring."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import deposit_experiment as experiment


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_keys(rows, values):
    ranked = rows[['CERT', 'date']].copy()
    ranked['forecast'] = np.asarray(values)
    return {
        (str(date.date()), int(cert))
        for date, quarter in ranked.groupby('date')
        for cert in quarter.sort_values(['forecast', 'CERT']).head(int(np.ceil(len(quarter) * .1))).CERT
    }


def freeze_models(root, output):
    """Accept extra predictions only after reproduction and reload checks pass."""
    raw_path = root / 'data/fdic_financials_2013_2024.csv'
    metadata_path = root / 'data/fdic_reporting_2010_2024.csv'
    inputs, observed = experiment.prepare_population(pd.read_csv(raw_path), pd.read_csv(metadata_path))
    train, valid, test = experiment.chronological_splits(observed)
    imputer, scaler = experiment.fit_preprocessor(train)
    X_train, X_valid, X_test = [experiment.transform_features(frame, imputer, scaler)
                              for frame in (train, valid, test)]
    y_train = train.log_growth.to_numpy(dtype='float32')
    y_valid = valid.log_growth.to_numpy(dtype='float32')
    ridge, alpha, _ = experiment.choose_ridge(X_train, y_train, X_valid, y_valid, [.01, .1, 1, 10, 100])
    mlp, history, _ = experiment.fit_network(42, X_train, y_train, X_valid, y_valid)
    predictions = {'Ridge': np.expm1(ridge.predict(X_test)),
                   'MLP': np.expm1(experiment.predict_log_growth(mlp, X_test))}
    reference_path = root / 'growth_outputs/masterclass/predictions.csv'
    reference = pd.read_csv(reference_path, parse_dates=['date'])
    pd.testing.assert_frame_equal(test[['CERT', 'date']].reset_index(drop=True), reference[['CERT', 'date']])
    differences = {}
    for name, values in predictions.items():
        np.testing.assert_allclose(values, reference[name], rtol=1e-6, atol=1e-8)
        assert selected_keys(test, values) == selected_keys(test, reference[name])
        measured = experiment.scores(test.growth, values)
        saved = experiment.scores(test.growth, reference[name])
        np.testing.assert_allclose(list(measured.values()), list(saved.values()), rtol=1e-6, atol=1e-8)
        differences[name] = {'max_prediction_difference': float(np.max(np.abs(values-reference[name]))),
                             'metrics': measured, 'original_list_membership_identical': True}

    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({'imputer': imputer, 'scaler': scaler, 'ridge': ridge}, output / 'ridge_preprocessing.joblib')
    mlp.save(output / 'mlp.keras')
    loaded = joblib.load(output / 'ridge_preprocessing.joblib')
    loaded_mlp = tf.keras.models.load_model(output / 'mlp.keras')
    reloaded_X = experiment.transform_features(test, loaded['imputer'], loaded['scaler'])
    reloaded = {'Ridge': np.expm1(loaded['ridge'].predict(reloaded_X)),
                'MLP': np.expm1(experiment.predict_log_growth(loaded_mlp, reloaded_X))}
    for name in predictions:
        np.testing.assert_array_equal(predictions[name], reloaded[name])
        differences[name]['reload_max_difference'] = float(np.max(np.abs(predictions[name]-reloaded[name])))

    for name, frame in [('training', train), ('validation', valid)]:
        frame[['CERT', 'date', 'target_date']].to_csv(output / f'{name}_rows.csv', index=False)
    expanded = inputs.loc[inputs.date.between('2024-01-01', '2024-09-30')].copy()
    X_expanded = experiment.transform_features(expanded, loaded['imputer'], loaded['scaler'])
    extended = expanded[['CERT', 'NAME', 'date', 'growth', 'Outcome status']].copy()
    extended['Ridge'] = np.expm1(loaded['ridge'].predict(X_expanded))
    extended['MLP'] = np.expm1(experiment.predict_log_growth(loaded_mlp, X_expanded))
    for name in predictions:
        accepted = extended.merge(reference[['CERT', 'date']], on=['CERT', 'date'], validate='one_to_one')
        np.testing.assert_allclose(accepted[name], predictions[name], rtol=1e-6, atol=1e-8)
        assert selected_keys(test, accepted[name]) == selected_keys(test, reference[name])
    extended.to_csv(output / 'expanded_predictions.csv', index=False)
    manifest = {
        'status': 'Verified reproduction; historical weight identity is unknown.',
        'features': list(experiment.FEATURES), 'seed': 42, 'ridge_alpha': alpha,
        'settings': {'ridge_alphas': [.01, .1, 1, 10, 100], 'architecture': [5,32,16,1],
                     'batch_size': 512, 'epochs_max': 200, 'patience': 10, 'adam_learning_rate': .001},
        'best_epoch': int(np.argmin(history['val_loss']) + 1), 'epochs': len(history['loss']),
        'verification': differences, 'expanded_rows': len(extended),
        'versions': {name: importlib.metadata.version(name) for name in
                     ['tensorflow', 'numpy', 'pandas', 'scikit-learn', 'joblib']},
        'source_hashes': {str(p.relative_to(root)): fingerprint(p) for p in
                         [raw_path, metadata_path, reference_path, root/'deposit_experiment.py', root/'freeze_forecasts.py', root/'uv.lock']},
        'artifact_hashes': {p.name: fingerprint(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'manifest.json'},
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest['verification'], indent=2), flush=True)
    return manifest


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root/'growth_outputs/frozen_forecasts')
    args = parser.parse_args()
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    freeze_models(root, args.output)
