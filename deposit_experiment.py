"""Shared, explicit-input routines for the deposit-growth experiments.

The notebook builder embeds these definitions as visible code. Command-line
experiments import them normally. Importing this module never reads data,
trains a model, creates output directories, or changes TensorFlow settings.
"""
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = ('log_deposits', 'cash_ratio', 'loan_ratio', 'equity_ratio', 'prior_growth')

def scores(actual, predicted):
    """Report ordinary-growth errors in percentage points."""
    return {
        'MAE (pp)': 100 * mean_absolute_error(actual, predicted),
        'RMSE (pp)': 100 * np.sqrt(mean_squared_error(actual, predicted)),
    }

def log_scores(y_log, pred_log):
    """Score log-growth forecasts in ordinary percentage-point units."""
    return scores(np.expm1(y_log), np.expm1(pred_log))

def add_adjacent_reports(frame):
    """Attach prior and next-quarter values within each FDIC certificate."""
    result = frame.sort_values(['CERT', 'REPDTE']).copy()
    result['date'] = pd.to_datetime(result['REPDTE'].astype(str))
    result['quarter_number'] = result['date'].dt.to_period('Q').astype('int64')

    grouped = result.groupby('CERT', sort=False)
    shifts = {
        'prior_deposits': ('DEPDOM', 1),
        'next_deposits': ('DEPDOM', -1),
        'prior_quarter': ('quarter_number', 1),
        'next_quarter': ('quarter_number', -1),
        'target_date': ('date', -1),
        'next_name': ('NAME', -1),
        'next_event': ('ACTEVT', -1),
    }

    for new_column, (source_column, periods) in shifts.items():
        result[new_column] = grouped[source_column].shift(periods)

    return result

def label_next_report(frame):
    """Separate ordinary missing outcomes from the end of the dataset."""
    last_quarter = frame['quarter_number'].max()

    return np.select(
        [
            frame['quarter_number'].eq(last_quarter),
            frame['next_quarter'].isna(),
            frame['next_quarter'].sub(frame['quarter_number']).ne(1),
        ],
        [
            'Dataset end',
            'Institution stops before dataset end',
            'Gap before later report',
        ],
        default='Adjacent next report',
    )

def apply_rules(frame, rules):
    """Apply eligibility rules one at a time and record their effect."""
    keep = pd.Series(True, index=frame.index)
    ledger = [
        {
            'Rule': 'All source rows',
            'Remaining': len(frame),
            'Removed': 0,
        }
    ]

    for rule_name, rule_mask in rules.items():
        rows_before = int(keep.sum())
        keep &= rule_mask
        rows_after = int(keep.sum())

        ledger.append(
            {
                'Rule': rule_name,
                'Remaining': rows_after,
                'Removed': rows_before - rows_after,
            }
        )

    return frame.loc[keep].copy(), pd.DataFrame(ledger)

def add_model_fields(frame):
    """Create the three candidate targets and five current-quarter inputs."""
    result = frame.copy()

    usable = result['next_quarter'].sub(result['quarter_number']).eq(1) & result['next_deposits'].gt(0)
    next_balance = result['next_deposits'].where(usable)
    result['growth'] = next_balance.div(result['DEPDOM']).sub(1)
    result['log_growth'] = np.log(next_balance.div(result['DEPDOM']))
    result['dollar_change_m'] = next_balance.sub(result['DEPDOM']).div(1000)

    result['log_deposits'] = np.log(result['DEPDOM'])
    result['cash_ratio'] = result['CHBAL'].div(result['ASSET'])
    result['loan_ratio'] = result['LNLSNET'].div(result['ASSET'])
    result['equity_ratio'] = result['EQ'].div(result['ASSET'])
    result['prior_growth'] = np.log(result['DEPDOM'].div(result['prior_deposits']))

    return result

def fit_preprocessor(training_frame, features=FEATURES):
    """Fit median imputation and standardization on training rows only."""
    fitted_imputer = SimpleImputer(strategy='median')
    fitted_scaler = StandardScaler()

    imputed = fitted_imputer.fit_transform(training_frame[list(features)])
    fitted_scaler.fit(imputed)

    return fitted_imputer, fitted_scaler

def transform_features(frame, fitted_imputer, fitted_scaler):
    """Apply the frozen preprocessing steps to one time split."""
    imputed = fitted_imputer.transform(frame[list(fitted_imputer.feature_names_in_)])
    transformed = fitted_scaler.transform(imputed)
    return transformed.astype('float32')

def choose_ridge(X_fit, y_fit, X_check, y_check, alphas):
    """Select Ridge strength using validation MAE in percentage points."""
    rows = []
    models = {}

    for alpha in alphas:
        candidate = Ridge(alpha=alpha).fit(X_fit, y_fit)
        models[alpha] = candidate
        rows.append(
            {
                'Alpha': alpha,
                **log_scores(y_check, candidate.predict(X_check)),
            }
        )

    results = pd.DataFrame(rows)
    chosen_alpha = float(
        results.sort_values(['MAE (pp)', 'Alpha']).iloc[0]['Alpha']
    )
    return models[chosen_alpha], chosen_alpha, results

def build_network(seed):
    """Create the fixed 5 → 32 → 16 → 1 neural network."""
    tf.keras.utils.set_random_seed(seed)

    network = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(5,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1),
        ]
    )
    # Create optimizer state eagerly before tracing the training graph.
    optimizer = tf.keras.optimizers.Adam(0.001)
    optimizer.build(network.trainable_variables)
    network.compile(
        optimizer=optimizer,
        loss='mse',
    )

    assert network.count_params() == 737
    return network

def fit_network(seed, X_train, y_train, X_valid, y_valid):
    """Train one fixed network and restore its best validation weights."""
    network = build_network(seed)

    options = tf.data.Options()
    options.threading.private_threadpool_size = 2

    training_data = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train.reshape(-1, 1)))
        .shuffle(len(y_train), seed=seed)
        .batch(512)
        .with_options(options)
    )
    validation_data = (
        tf.data.Dataset.from_tensor_slices((X_valid, y_valid.reshape(-1, 1)))
        .batch(512)
        .with_options(options)
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
    )

    started = time.perf_counter()
    fitted = network.fit(
        training_data,
        validation_data=validation_data,
        epochs=200,
        callbacks=[early_stopping],
        verbose=0,
        shuffle=False,
    )
    elapsed = time.perf_counter() - started

    return network, fitted.history, elapsed

def predict_log_growth(model, features):
    """Return a flat NumPy array from a Keras model."""
    return np.asarray(model(features, training=False)).ravel()

def scoring_eligibility(panel):
    """Use only current and prior reports to decide who can receive a forecast."""
    return (
        panel['BKCLASS'].isin(['N', 'NM', 'SM', 'SB', 'SI', 'SL'])
        & panel['CALLFORM'].isin([31, 41, 51])
        & panel['quarter_number'].sub(panel['prior_quarter']).eq(1)
        & panel['prior_deposits'].gt(0)
        & panel['DEPDOM'].gt(0)
        & panel['ASSET'].gt(0)
    )


def outcome_status(panel):
    """Describe observed coverage; absence is not an event diagnosis."""
    return pd.Series(np.select(
        [panel['next_quarter'].isna(),
         panel['next_quarter'].sub(panel['quarter_number']).ne(1),
         ~panel['next_deposits'].gt(0)],
        ['No later report in snapshot', 'Gap before later report',
         'Nonpositive or missing next deposits'],
        default='Usable adjacent outcome'), index=panel.index)


def prepare_population(financials, reporting):
    """Return all scoreable inputs and the subset with observable targets."""
    panel = financials.merge(reporting, on=['CERT', 'REPDTE'], how='left',
                             validate='one_to_one', indicator=True)
    if not panel['_merge'].eq('both').all():
        raise ValueError('Reporting metadata is missing for financial rows')
    panel = add_adjacent_reports(panel)
    scoring = panel.loc[scoring_eligibility(panel)].copy()
    scoring['Outcome status'] = outcome_status(scoring)
    # add_model_fields retains unknown targets as NaN; inputs use current/prior only.
    scoring = add_model_fields(scoring)
    observed = scoring.loc[scoring['Outcome status'].eq('Usable adjacent outcome')].copy()
    return scoring, observed


def chronological_splits(rows, training_start='2013-01-01'):
    """Preserve the original date gaps and fixed validation/evaluation periods."""
    train = rows.loc[rows['date'].between(training_start, '2022-09-30')].copy()
    valid = rows.loc[rows['date'].between('2023-01-01', '2023-09-30')].copy()
    test = rows.loc[rows['date'].between('2024-01-01', '2024-09-30')].copy()
    if any(frame.empty for frame in (train, valid, test)):
        raise ValueError('A required chronological split is empty')
    assert train['target_date'].max() < valid['date'].min()
    assert valid['target_date'].max() < test['date'].min()
    return train, valid, test


def outcome_coverage(scoring_rows):
    """Count all eligible inputs before restricting evaluation to known targets."""
    frame = scoring_rows.copy()
    frame['Observed'] = frame['Outcome status'].eq('Usable adjacent outcome')
    receipt = frame.groupby('date').agg(
        Eligible=('CERT', 'size'), Observed=('Observed', 'sum')).reset_index()
    receipt['Unresolved'] = receipt['Eligible'] - receipt['Observed']
    return receipt


def review_precision(rows, prediction, fraction=0.1):
    """Equal-quarter precision at fixed capacity, with CERT tie-breaking."""
    audit = rows[['CERT', 'date', 'growth']].copy()
    audit['prediction'] = np.asarray(prediction)
    if audit['growth'].isna().any():
        raise ValueError('Review precision requires observed outcomes')
    precision = []
    for _, quarter in audit.groupby('date'):
        k = int(np.ceil(len(quarter) * fraction))
        actual = set(quarter.sort_values(['growth', 'CERT']).head(k)['CERT'])
        selected = set(quarter.sort_values(['prediction', 'CERT']).head(k)['CERT'])
        precision.append(len(actual & selected) / k)
    return float(np.mean(precision))


def compare_training_windows(rows, starts=(2013, 2020, 2021), seeds=(42, 7, 99),
                             alphas=(0.01, 0.1, 1, 10, 100)):
    """Compare all declared windows on identical validation and evaluation rows."""
    _, valid, test = chronological_splits(rows)
    results, predictions = [], []

    def measure(prediction):
        measured = scores(test.growth.to_numpy(), prediction)
        return {'MAE_pp': measured['MAE (pp)'], 'RMSE_pp': measured['RMSE (pp)'],
                'bottom_decile_precision': review_precision(test, prediction)}

    for year in starts:
        train, _, _ = chronological_splits(rows, f'{year}-01-01')
        imputer, scaler = fit_preprocessor(train)
        X_train = transform_features(train, imputer, scaler)
        X_valid = transform_features(valid, imputer, scaler)
        X_test = transform_features(test, imputer, scaler)
        y_train = train.log_growth.to_numpy(dtype='float32')
        y_valid = valid.log_growth.to_numpy(dtype='float32')
        ridge, alpha, _ = choose_ridge(X_train, y_train, X_valid, y_valid, alphas)
        results.append({'training_start': year, 'model': 'Ridge', 'seed': None,
                        'training_rows': len(train), 'alpha': alpha,
                        **measure(np.expm1(ridge.predict(X_test)))})
        for seed in seeds:
            model, history, elapsed = fit_network(seed, X_train, y_train, X_valid, y_valid)
            prediction = np.expm1(predict_log_growth(model, X_test))
            metrics = measure(prediction)
            results.append({'training_start': year, 'model': 'MLP', 'seed': seed,
                            'training_rows': len(train), 'best_epoch': int(np.argmin(history['val_loss']) + 1),
                            'epochs': len(history['loss']), 'seconds': elapsed, **metrics})
            saved = test[['CERT', 'date', 'growth']].copy()
            saved['prediction'] = prediction
            saved['training_start'], saved['seed'] = year, seed
            predictions.append(saved)
    return pd.DataFrame(results), pd.concat(predictions, ignore_index=True)
