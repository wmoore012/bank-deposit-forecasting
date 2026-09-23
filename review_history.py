"""Eight-quarter histories and fixed-capacity review comparisons.

Quarter slots run oldest to newest; features keep deposit_experiment.FEATURES order.
All functions receive their data explicitly. Importing this module performs no work.
"""

from itertools import combinations
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from deposit_experiment import FEATURES


def feature_histories(scoring_rows):
    """Keep eight consecutive, finite feature vectors, without filling gaps."""
    rows = scoring_rows.sort_values(["CERT", "date"]).reset_index(drop=True)
    grouped = rows.groupby("CERT", sort=False)
    quarters = np.column_stack([grouped.quarter_number.shift(lag) for lag in range(7, -1, -1)])
    blocks = [grouped[list(FEATURES)].shift(lag).to_numpy() for lag in range(7, -1, -1)]
    values = np.stack(blocks, axis=1)
    consecutive = np.isfinite(quarters).all(axis=1) & (np.diff(quarters, axis=1) == 1).all(axis=1)
    finite = np.isfinite(values).all(axis=(1, 2))
    eligible = consecutive & finite
    audit = rows[["CERT", "NAME", "date", "Outcome status"]].copy()
    audit["history_eligible"] = eligible
    audit["exclusion_reason"] = np.select(
        [~consecutive, ~finite],
        [
            "Fewer than eight consecutive input-eligible quarters",
            "Nonfinite feature in eight-quarter history",
        ],
        default="Eligible",
    )
    return rows.loc[eligible].reset_index(drop=True), values[eligible], audit


def standardize_histories(training, validation, evaluation):
    """Fit each quarter-feature slot using training histories alone."""
    scaler = StandardScaler().fit(training.reshape(len(training), -1))
    arrays = [
        scaler.transform(values.reshape(len(values), -1))
        for values in (training, validation, evaluation)
    ]
    return scaler, arrays


def reconstruction_error(actual, reconstructed, feature_count):
    """Return mean squared error and its contribution by original input."""
    squared = np.square(actual - reconstructed)
    by_input = squared.reshape(len(actual), 8, feature_count).sum(axis=1)
    total = by_input.sum(axis=1, keepdims=True)
    shares = np.divide(by_input, total, out=np.zeros_like(by_input), where=total != 0)
    return squared.mean(axis=1), shares


def fit_pca(training):
    """Use training variance to choose the smallest sufficient dimension."""
    model = PCA(svd_solver="full").fit(training)
    count = int(np.searchsorted(np.cumsum(model.explained_variance_ratio_), 0.90) + 1)
    model = PCA(n_components=count, svd_solver="full").fit(training)
    return model


def build_autoencoder(input_count, seed):
    """A dense reconstruction model on fixed quarter-feature slots."""
    tf.keras.utils.set_random_seed(seed)
    encoder = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(input_count,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(4),
        ],
        name="encoder",
    )
    decoder = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(4,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(input_count),
        ],
        name="decoder",
    )
    model = tf.keras.Sequential([tf.keras.Input(shape=(input_count,)), encoder, decoder])
    optimizer = tf.keras.optimizers.Adam(0.001)
    optimizer.build(model.trainable_variables)
    model.compile(optimizer=optimizer, loss="mse")
    return model


def fit_autoencoder(training, validation, seed):
    """Restore the best validation reconstruction weights, never event-selected."""
    model = build_autoencoder(training.shape[1], seed)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    train = training.astype("float32")
    valid = validation.astype("float32")
    batches = (
        tf.data.Dataset.from_tensor_slices((train, train))
        .shuffle(len(train), seed=seed)
        .batch(256)
        .with_options(options)
    )
    checks = tf.data.Dataset.from_tensor_slices((valid, valid)).batch(256).with_options(options)
    stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    history = model.fit(
        batches, validation_data=checks, epochs=100, callbacks=[stopping], verbose=0, shuffle=False
    )
    return model, history.history


def fit_isolation_forest(training, seed):
    model = IsolationForest(
        n_estimators=300,
        max_samples=min(256, len(training)),
        contamination="auto",
        random_state=seed,
        n_jobs=2,
    )
    return model.fit(training)


def review_flags(rows, scores):
    """All scores point upward toward review priority; CERT resolves ties."""
    flags = pd.DataFrame(False, index=rows.index, columns=scores.columns)
    for _, quarter in rows.groupby("date"):
        capacity = int(np.ceil(0.10 * len(quarter)))
        for method in scores.columns:
            ranked = pd.DataFrame(
                {"score": scores.loc[quarter.index, method], "CERT": quarter.CERT}
            )
            if not np.isfinite(ranked.score).all():
                raise ValueError(
                    f"Unavailable scores cannot enter a common-population list: {method}"
                )
            chosen = (
                ranked.sort_values(["score", "CERT"], ascending=[False, True]).head(capacity).index
            )
            flags.loc[chosen, method] = True
    return flags


def list_overlaps(rows, flags):
    """Keep exact quarterly denominators; pool counts rather than percentages."""
    records = []
    for date, quarter in rows.groupby("date"):
        for left, right in combinations(flags.columns, 2):
            a, b = flags.loc[quarter.index, left], flags.loc[quarter.index, right]
            shared, union = int((a & b).sum()), int((a | b).sum())
            records.append(
                {
                    "date": str(date.date()),
                    "left": left,
                    "right": right,
                    "eligible": len(quarter),
                    "left_count": int(a.sum()),
                    "right_count": int(b.sum()),
                    "shared": shared,
                    "left_share": shared / a.sum(),
                    "right_share": shared / b.sum(),
                    "jaccard": shared / union,
                    "random_expected": a.sum() * b.sum() / len(quarter),
                }
            )
    result = pd.DataFrame(records)
    sums = (
        result.groupby(["left", "right"])[
            ["eligible", "left_count", "right_count", "shared", "random_expected"]
        ]
        .sum()
        .reset_index()
    )
    sums["date"] = "All three quarters"
    sums["left_share"] = sums.shared / sums.left_count
    sums["right_share"] = sums.shared / sums.right_count
    sums["jaccard"] = sums.shared / (sums.left_count + sums.right_count - sums.shared)
    return pd.concat([result, sums], ignore_index=True)


def outcome_capture(rows, flags):
    """Evaluate the same quarter-specific realized bottom decile for every list."""
    if not np.isfinite(rows.growth).all():
        raise ValueError("Observed outcomes are required for this comparison")
    records = []
    for date, quarter in rows.groupby("date"):
        k = int(np.ceil(0.10 * len(quarter)))
        target = quarter.sort_values(["growth", "CERT"]).head(k).index
        for method in flags.columns:
            chosen = flags.loc[quarter.index, method]
            hits = int(flags.loc[target, method].sum())
            records.append(
                {
                    "date": str(date.date()),
                    "method": method,
                    "eligible": len(quarter),
                    "selected": int(chosen.sum()),
                    "target_count": k,
                    "captured": hits,
                    "capture_per_selected": hits / chosen.sum(),
                }
            )
    result = pd.DataFrame(records)
    total = (
        result.groupby("method")[["eligible", "selected", "target_count", "captured"]]
        .sum()
        .reset_index()
    )
    total["date"] = "All three quarters"
    total["capture_per_selected"] = total.captured / total.selected
    return pd.concat([result, total], ignore_index=True)


def choose_examples(rows, scores, flags):
    """Choose March cases before inspecting identities or subsequent events."""
    march = rows.loc[rows.date.eq("2024-03-31")].copy()
    groups = {
        "Both": flags.Ridge & flags.PCA,
        "Forecast only": flags.Ridge & ~flags.PCA,
        "Anomaly only": ~flags.Ridge & flags.PCA,
    }
    selected = []
    for group, mask in groups.items():
        candidates = march.loc[mask.reindex(march.index)].copy()
        method = "Ridge" if group == "Forecast only" else "PCA"
        candidates["priority"] = scores.loc[candidates.index, method]
        if candidates.empty:
            selected.append({"group": group, "CERT": None, "NAME": "Category empty"})
        else:
            bank = candidates.sort_values(["priority", "CERT"], ascending=[False, True]).iloc[0]
            selected.append({"group": group, "CERT": int(bank.CERT), "NAME": bank.NAME})
    return pd.DataFrame(selected)
