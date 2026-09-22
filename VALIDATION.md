# Validation

Both generated notebooks executed fully in separate fresh Python 3.12 CPU kernels.
Their historical predictions agree within 1e-6 relative tolerance. Six automated
checks cover target arithmetic, consecutive quarters, chronological boundaries,
frozen settings, score units, ranking counts, executed outputs, numeric histogram
axes, equal-unit scatter configuration, and readable tooltip settings.

The original 94.5443558% decline-dollar concentration was reproduced. The original
assignment was not edited in this task. All 33 cell sources match its pre-existing
ZIP; saved outputs/metadata differ, so byte identity is not claimed. The original masterclass
and results are archived separately.

The local exported HTML was inspected in the Codex browser. Observed populated
Plotly charts, a readable formula card, exact score tables, a 3.627 pp MLP tooltip,
the learning curve, actual/predicted scatter, realized-tail errors, and predicted
ranking lines. Fixed subtitle/subplot overlap and dark-on-dark tooltips. Compact
ranking columns retain complete evidence in the CSV. Browser console inspection
returned no warning/error entries during these checks.

Narrow-width opening prose was visually readable. DOM inspection confirmed
contained horizontal chart scrolling, but subsequent narrow screenshot capture
returned a blank image despite visible DOM geometry. Complete narrow-chart visual
verification is therefore not claimed. Native VS Code rendering and cloud Colab
execution were not tested. Private browser QA records remain ignored locally.

The longer historical dataset did not pass the form-comparability gate; this is
an explicit analytical limitation, not a missing download. Institution changes
behind extreme growth and disappearing reports remain only partially identified.
The holdout is reused, and the bank-cluster interval is conditional on three
historical quarters and fixed fitted models.

## TimesFM research extension, 2026-09-21

TimesFM 3.0.2 was run through the official MLX backend on all 13,532 frozen 2024
bank-quarters. Checkpoint revision and context counts are recorded in
`growth_outputs/timesfm_zero_shot_predictions.json`. MAE is 6.171345 pp; RMSE is
294.540591 pp. Bank CERT 59324 (three context quarters) contributes 99.841981% of
squared error. Every row remains in the result; no holdout-based clipping or
history-length exclusion was applied. The full benchmark is a masterclass bonus
after the core conclusion. It compares different input information and pretrained
weights, not architecture alone. The checkpoint postdates the historical evaluation.

Both editions executed in fresh kernels. Regression tests recompute TimesFM scores,
check exact holdout joins, verify that future balances cannot change a historical
context, and preserve the core model score checks. Personal CS50 transcript references
and the teaching skill are stored outside this repository and excluded from bundles.

### Forecast-head fine-tuning (2026-09-21)

Both foundation models received 256 sampled updates (batch 32, seed 42, Adam 1e-5),
using only eligible training rows with outcomes through 2022-12-31. Their backbones
stayed frozen. There were 8,020 unique training rows sampled, not a full epoch.
All 13,834 validation rows chose among steps 0, 128, 256 by log-balance MAE.
Both selected step 256 before any evaluation outcomes were scored. All 13,532
2024 evaluation rows remain. TimesFM head adaptation: MAE 6.140479 pp,
RMSE 296.334609 pp. Chronos head adaptation: MAE 3.999659 pp, RMSE 7.417518 pp.
Neither beats the best core baselines. Improved validation did not assure improved
ordinary-growth errors in 2024.

The initial TimesFM backward pass produced nonfinite gradients. It was stopped.
The successful retry freezes the gradient through horizon normalization refinement,
without changing its forward computation. This is a custom bounded head adaptation,
not an official turnkey full-model training run. Pretraining overlap is unknown;
chronological adaptation does not remove that limitation or repeated holdout reuse.
Selection metadata, checkpoint revision, changed-weight checks, and complete predictions
are saved. Local head weights and intermediate split exports are ignored under `.finetune/`.

### Recruiter reading path (2026-09-21)

Both editions now open with a project summary and give every section a plain-language
preview. The source section states the 2013–2024 choice before its ordered reasons,
defines all nine source columns before the first preview, and explains the 815 missing
EQ values immediately after the pandas missingness output. A computed assertion confirms
none of those EQ gaps belong to the selected domestic-bank/form population. Source blanks
remain unknown. The 48 missing state labels are not model inputs.

Both notebooks executed successfully in fresh kernels after adding the source checks.
The subsequent summary edits changed Markdown only and preserved those outputs. Fifteen
regression tests pass, including the source-population and lead-first checks. Desktop HTML
inspection confirmed the ordered rationale, wrapped dictionary, and missingness card.
Previously recorded narrow-screen verification limits still apply.

## September 21 teaching and chart revision

Both editions were regenerated from shared teaching sources with full EDA retained.
The 38/3/3 split chart previously selected bank-count columns by position; named
calendar columns now fix that presentation error. The underlying split and model
predictions were unaffected. New regression checks cover calendar coordinates,
finding-led chart titles, capacity denominators, illustrative arithmetic, real-bank
input mapping, source provenance, follow-up baselines, and paired uncertainty.

The local design uses white WM cards and cyan accents, retaining the approved fonts.
Question, big-number, and pictogram highlights supplement the ordinary charts and
pandas outputs. The pictogram displays an approximate proportion and explicitly does
not assign an individual bank to each grid dot.

Final verification: both fresh-kernel executions completed; 28 unittest checks passed.
Desktop browser checks confirmed the standard WM chips, white cards, new question/
number/pictogram highlights, readable error lesson, and corrected calendar labels.
At 389 CSS pixels, measured card widths were 343 pixels and the document width stayed
within the viewport. The browser screenshot compositor returned blank/distorted images
under viewport overrides, so narrow-screen visual inspection remains limited to DOM
geometry; desktop screenshots were available normally. All twelve PNGs and twelve PDF
pages were rendered and reviewed as contact sheets. The PDF has twelve portrait pages.
