# Does financial history improve the review list?

At a 10% review capacity, Ridge still captured the most low-growth outcomes in
this historical comparison. On the original common-history population, it
captured 296 in 1,351 selections (21.9%). The primary autoencoder captured 274
(20.3%), Isolation Forest 237 (17.5%), and PCA 236 (17.5%). MLP captured 293.
None of the declared methods or seeds exceeded Ridge on this outcome.

The lists were different. Across the expanded common-history population, Ridge
and PCA shared 285 of 1,359 bank-quarter selections (21.0%; Jaccard 0.117).
The primary autoencoder shared 331 with Ridge. These are bank-quarter counts,
not distinct institutions or measured improvements in analyst productivity.

The full-input Isolation Forest lists shared 90.2%–93.2% across seed pairs;
the autoencoder lists shared 85.9%–90.3%. Stability concerns repeatability,
not whether a selection serves the review objective.

## Data and method

The generated eligibility audit contains 13,618 current input-eligible rows,
including 86 unknown future outcomes. Eight consecutive finite feature vectors
leave 13,576 rows. The original observed population has 13,532 rows; its common
history subset contains 13,490. Original list membership is retained separately,
with five Ridge selections and seven MLP selections lacking anomaly scores.

Each history contains the original five inputs over eight quarters. Its earliest
`prior_growth` needs a ninth source deposit report. Zero prior deposits produce an
undefined feature; no imputation or invented balance is used. This accounts for
the distinction between nine-report presence and complete feature eligibility.

Standardization uses complete training sequences only, with endpoints through
September 2022. Validation endpoints are Q1–Q3 2023. Training contains 169,298
histories spanning 31 distinct endpoints. PCA selects the minimum components
explaining 90% of training variance: nine for both representations. The dense
autoencoder uses fixed, flattened quarter-feature slots, without recurrence or
attention. All fixed settings and fingerprints are in `outputs/plan.json`.

The full five-feature version and the version without log deposits use identical
rows. Seeds 42, 7, and 99 are retained for both stochastic methods. Scores are saved
and fingerprinted before the observed-outcome comparison. Every method selects
ceil(10% of eligible rows) per quarter, with certificate-number tie-breaking.

## Read and reproduce

Open `FDIC_Review_History.ipynb` at the repository root. It shows the business
question, modeling definitions, deterministic disagreement examples, seed
stability, size-removal check, and all outcome receipts. Both Study 1 editions
contain a concise continuation after their original ranking lesson.

The main README supplies the shared Python 3.12 setup. Run the benchmark with:

```sh
.venv/bin/python experiments/review_history/run.py
```

To preserve delivered outputs during an independent reproduction, pass
`--output /tmp/review-history-reproduction`. To rebuild and execute the notebook:

```sh
.venv/bin/python build_review_history.py
.venv/bin/python execute_masterclass.py FDIC_Review_History.ipynb
```

The notebook reruns the declared benchmark by default. For a presentation-only rerun of already verified saved results, set `REVIEW_HISTORY_REFIT=0`; this does not fit new models. `review_history.py` contains ordinary
explicit-input functions shared by the runner and visible notebook definitions.
Fitted models, scores, coverage, selections, overlaps, reconstruction-error shares,
and training choices are preserved under `outputs/`.

## What this result supports

Keep Ridge for this historical deposit-growth review comparison. The anomaly
methods identify unusual histories and choose different banks, but do not capture
more of the specified outcomes at the same review capacity. Lower reconstruction
error alone does not make an anomaly detector better.

These three 2024 quarters were already examined. Historical publication timing
and report revisions are not reconstructed. No operational savings or failure
prediction is established. The raw equity rule originated in earlier failure
exploration; its historical performance is not independent confirmation.

The separate FDIC coverage audit supplies event context for missing future
balances. Event labels do not enter modeling, and an event cannot supply an
unobserved deposit-growth target. Peer-adjusted comparisons, earlier-period
backtests, and warning-time analyses remain outside this study.
