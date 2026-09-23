# Validation

## Study 2 numerical verification, September 23, 2026

**Ridge remains the strongest measured review list in this comparison.** On the
13,490 common-history rows with observed outcomes, each method selected 1,351
bank-quarter cases across three quarters. Ridge captured 296 realized lowest-growth
cases, MLP 293, primary autoencoder 274, primary Isolation Forest 237, PCA 236,
raw equity/assets 170, and size alone 86. Every stochastic seed (42, 7, 99) and
both feature representations remain in the record; none exceeded Ridge.

The expanded comparison has 13,576 finite eight-quarter histories out of 13,618
input-eligible rows. The original comparison has 13,490 out of 13,532 observed
rows. Nine source deposit reports are necessary but do not make undefined growth
valid. Integro's zero preceding balance stays excluded. Original selections stay
in their separate audit, including five Ridge and seven MLP selections without
anomaly scores. The 86 unknown future balances remain unknown.

Training-only standardization uses 169,298 complete histories across 31 endpoints.
Both deterministic PCA representations select nine components for 90% training
variance. The dense autoencoder uses fixed quarter-feature positions, without
recurrence or attention. Events do not enter fitting or evaluation. The official
register and separate coverage ledger preserve verified later events without
assigning a cause to an unavailable report.

A newly extracted allowlisted package installed with locked Python 3.12 dependencies
from the local cache and executed all three notebooks. Independent Study 2
reproduction matched all 14 learned models' saved scores exactly (maximum absolute
difference 0), including every seed and the representation without deposit size.
Selections, overlaps, outcome capture, coverage, and reconstruction shares agreed.
Both Study 1 editions reproduced predictions, splits, original scores/rankings,
and training-window results within rtol 1e-6 / atol 1e-8; elapsed fit time is not a
statistical result. Presentation-only reruns use these verified saved outputs.

The regression suite contains 47 passing checks. It covers observed/input population
separation, consecutive histories, future-outcome independence, model reloads,
training-only scaling, PCA dimension choice, review capacities and certificate ties,
matched outcome calculations, reconstruction shares, visible notebook implementation,
score/model fingerprints, and publication image dimensions. Earlier dated test totals
below describe earlier versions.

Static figure inspection found and corrected crowded history labels, cramped seed
labels, transparent chart backgrounds, and exported figures defaulting to the wrong
height. Current PNGs preserve the specified dimensions at 2x resolution. The shared
HTML exporter removes external script dependencies. Browser inspection of the final
Study 2 HTML and repaired GitHub previews remains pending because automatic browser
approval review failed with a service usage-limit error. Static artifact inspection
and successful execution do not close that browser acceptance check. Study 2 stays
local until that remaining check is completed.

This is exploratory evidence from three previously inspected 2024 quarters. It does
not reconstruct report vintages or historical publication timing, demonstrate
operational savings, establish general anomaly quality, or evaluate failure prediction.

## Current Study 1 release, September 22, 2026

Both notebook editions executed fully in separate Python 3.12 CPU kernels. A fresh
extracted ZIP installed from the locked environment without network access and
executed both editions. The original predictions, MAE/RMSE, ranking, and split
receipts agree with the checkpoint within rtol 1e-6 / atol 1e-8. All original
review-list memberships are preserved. The independent extracted training-window
runner reproduced all nine seed/window prediction sets and all twelve score rows.

The shared module supplies ordinary importable routines and the builder keeps
those definitions visible in notebook cells. The training-window runner no longer
extracts or rewrites code. Both editions include the completed shorter-window
comparison after the deposit-size comparison. Singleton training batches retain
an explicit one-column target shape.

Scoring eligibility is independent of future outcome availability: 13,618 current
input-eligible 2024 rows, 13,532 observed outcomes, and 86 unresolved outcomes
(30, 22, and 34 by quarter). The unresolved rows remain in coverage receipts;
missing balances are not set to zero or assigned an economic cause.

`growth_outputs/frozen_forecasts/manifest.json` records verified reproductions of
the original Ridge and seed-42 MLP. Maximum absolute growth-prediction differences
from saved CSVs were 2.988e-9 and 2.649e-8, respectively. Saved preprocessing and
models reload with zero prediction change. Expanded predictions use those loaded
objects. Historical weight identity cannot be established from saved predictions.

Browser verification observed 50 Masterclass and 41 Submission plots with no
page errors or failed requests. The coverage-axis overlap and target-panel label
overlap were corrected; dollars, percentage growth, and log growth have separate
axes. Narrow-width prose was visually readable with contained code scrolling.
Saved HTML embeds Plotly and removes redundant external script loaders, including
the module-import loader. VS Code and Colab execution remain untested.

The current LinkedIn PDF has 11 landscape pages. Its text and rendered pages were
reviewed against the verified results; earlier dated records below describe older
exports. Resume bullets were updated in ignored private storage. ZIPs use an
explicit allowlist and exclude personal notes, local archives, and browser evidence.
Git history is preserved, including older editorial/methodological drafts that were
subsequently removed from the current public tree.

## Earlier validation records

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

## GitHub preview repair, September 23, 2026

GitHub loaded the notebooks but suppressed JavaScript-only charts. The first static fallback restored images; reader screenshots exposed cropping inside fixed-height containers and low-resolution text. The revised publication output places a 2x-resolution image outside those containers and preserves the original interactive markup for trusted HTML. Static PNG inspection confirms complete labels and axes for the reported examples. Card labels now retain whitespace without their styles. Model code, predictions, and execution counts are unchanged.

Live browser verification of this second preview repair remains pending: automatic approval review could not complete because its service reported a usage limit. This is a verification gap, not a passed browser check.
