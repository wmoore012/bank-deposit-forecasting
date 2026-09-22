# But wait. What if we train on only the recent years?

In volatile economic times, older examples might be less useful. So I tried starting the training data in 2020, then 2021.

And... the errors got bigger.

The original 2013–2022 training window had lower 2024 mean absolute error than windows starting in 2020 or 2021. This held for Ridge and for each of three neural-network seeds. This is an exploratory result on already-inspected 2024 outcomes, not evidence about a unique current economic regime.

## Same comparison, less training history

All windows end with September 2022 predictor rows (December 2022 outcomes). Validation uses January–September 2023 predictor dates; evaluation uses January–September 2024 predictor dates. The same 13,532 bank-quarter evaluation rows and 13,834 validation rows are used every time.

The five inputs, log-growth target, eligibility rules, 5→32→16→1 neural network, optimizer, batch size, maximum epochs, early-stopping rule, and Ridge alpha grid are unchanged. Imputation and scaling are fitted separately on each training window. Ridge alpha and neural-network stopping epoch use 2023 validation only.

| Training predictors | Rows | Ridge MAE | MLP MAE, seed 42 | MLP MAE range, 3 seeds |
|---|---:|---:|---:|---:|
| 2013–Sep 2022 | 214,425 | 3.641 | 3.627 | 3.601–3.669 |
| 2020–Sep 2022 | 53,731 | 4.183 | 4.322 | 4.114–4.322 |
| 2021–Sep 2022 | 33,671 | 3.656 | 3.776 | 3.751–3.799 |

MAE is in percentage points; lower is better. The fixed zero-growth forecast has MAE **3.601** on these same rows and does not depend on a training window. Seeds are 42, 7, and 99; the range measures training randomness, not a confidence interval. Seed 7 with the full window very slightly beats zero (3.600729 versus 3.601064); the carousel’s original comparison uses seed 42.

## Other outcomes

| Training start | Ridge RMSE | MLP RMSE, seed 42 | Ridge review precision | MLP review precision, seed 42 |
|---|---:|---:|---:|---:|
| 2013 | 6.512 | 6.429 | 21.90% | 21.77% |
| 2020 | 6.724 | 6.836 | 19.11% | 17.49% |
| 2021 | 6.494 | 6.495 | 18.51% | 17.34% |

Review precision is the equal-quarter average of the share of selected banks actually in that quarter’s lowest-growth decile; review capacity is ceil(10% of banks), ties broken by FDIC certificate. Higher is better. The 2021 Ridge has slightly lower RMSE than full-history Ridge, but worse MAE and review precision.

## What this does and does not establish

Keep the full window for this historical comparison. Fewer older rows did not help the tested settings. The comparison changes both recency and sample size, so it cannot isolate an economic-regime effect. The dates were chosen after reviewing the existing story; do not describe them as a preregistered selection or the 2024 data as fresh.

The steep decline in the single-bank chart extends into 2023–2026. Those observations were not moved into training: doing so would contaminate this comparison. Testing adaptation to that later period requires a new chronological design, newer data across the bank population, and later evaluation quarters. One bank’s chart alone does not establish population-wide volatility.

## Verification and reproducibility

The full-window seed-42 MLP reproduces the original MAE 3.6273179778953386 and RMSE 6.429240270152202. Evaluation row keys and actual growth are asserted equal to the saved original predictions. Split separation and original row counts are checked. The three full-window neural-network fits and six shorter-window fits completed.

The existing data-preparation and model routines are reused from build_masterclass.py. The runner preserves scalar targets as (batch, 1) to handle a final training batch containing one observation; this fixes a Keras shape error without changing targets or the architecture. The original scores reproduce exactly after this adjustment.

Run from the project root: `.venv/bin/python experiments/training_window_sensitivity/run.py`. The plan records source-data SHA-256 and experiment settings; scores.csv records every model/window/seed; predictions.csv saves every neural-network prediction. Core notebook files and their result files are not overwritten.
