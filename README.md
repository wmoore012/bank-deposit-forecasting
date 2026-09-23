# Bank Deposit Forecasting

A reproducible research project testing whether machine learning improves next-quarter deposit forecasts and helps prioritize a limited bank-review queue.

## The result in 30 seconds

- **The simple baseline matters:** among the original four forecasts, predicting zero growth has the lowest mean absolute error, **3.601 percentage points**.
- **The neural network helps on a different metric:** its **6.429 pp RMSE** is 4.3% below zero growth, but does not establish a dependable overall advantage.
- **Review capacity changes the question:** selecting 10% of banks using Ridge or MLP predictions identifies realized bottom-decile banks with **16.7%–24.7% precision** across three quarters, versus roughly 10% from random selection.
- **More complex is not automatically better:** zero-shot and forecast-head-fine-tuned TimesFM and Chronos models did not beat the best core baselines in this experiment.

**Scope:** 214,425 training bank-quarters; 13,532 reused 2024 evaluation rows. Research prototype, no measured operational savings, no claim that unusual reports prove distress.

**Stack:** Python · pandas · scikit-learn · TensorFlow · Plotly · Jupyter · uv. Foundation-model extensions use PyTorch and MLX.

## Start here

1. [Concise experiment](FDIC_Deep_Learning_Submission.ipynb): data, model, results, and decision.
2. [Detailed walkthrough](FDIC_Deep_Learning_Masterclass.ipynb): worked examples, explanatory cards, pretrained models, and fine-tuning.
3. [Validation record](VALIDATION.md): checks, limitations, and reproducibility.

## Data decisions before modeling

1. Use **2013–2024** after the 2012 reporting conversion. Older forms still need a verified field-by-field mapping.
2. Keep **2010–2012** in the audit; matching column names alone cannot prove comparable definitions.
3. Preserve the **815 missing source EQ values** as unknown. They all occur outside the domestic-bank/form eligibility group and are excluded by that population rule, not filled with zero.
4. Leave missing state labels alone because state is not a model input. Fit any feature-imputation parameters on training data only.

## Run

```sh
uv sync --python 3.12 --locked
.venv/bin/python build_masterclass.py
.venv/bin/python execute_masterclass.py
.venv/bin/python execute_masterclass.py FDIC_Deep_Learning_Submission.ipynb
.venv/bin/python -m unittest discover -s tests -v
```

Alternatively, select the project `.venv` kernel and Run All from this folder.
No data network calls occur during the analysis. CPU training is sufficient.
The committed public CSVs and bundled WM notecards wheel are required inputs.

`build_masterclass.py` is the canonical authoring source. It rebuilds both notebooks
without outputs; execute them afterward. `build_professor_submission.py` delegates
to that same builder so it cannot restore the old experiment by accident.
`execute_masterclass.py` saves only after all cells succeed and makes saved Plotly
outputs independent of a CDN. Notebook readers may need to trust the notebook to
allow interactive JavaScript. The HTML preview provides a separate reading surface.

## What is preserved

- `experiments/experiment_0/`: original dollar-runoff masterclass, builder, and outputs.
- `growth_outputs/masterclass/` and `growth_outputs/submission/`: frozen plan,
  predictions, scores, source fingerprints, coverage, and decision evidence.
- `sources/README.md`: primary sources and the historical comparability decision.
- `notebooks/source/`: visible experiment and teaching sources.

The 2010–2024 file is audited but not used directly for primary training: 5,738
early form-100 reports cross the 2012 TFR-to-Call-Report transition without a
verified five-field historical crosswalk. The core begins in 2013, after that
conversion. Its 2013–2024 rows match the long audit extract exactly. Missing equity
concentrates entirely in form-2 reporting; the revised population uses insured
domestic-bank classes and forms 31, 41, and 51.

Run `download_fdic.py` only to refresh the long financial extract, and
`download_supplement.py` to refresh reporting metadata. Such refreshes may reflect
FDIC revisions and change results. The committed snapshots define this run.

## Bundles and Colab

`package_project.py` creates ignored ZIP bundles from an explicit file list.
Upload and extract a bundle, change into its folder, and install dependencies:

```python
%pip install tensorflow==2.18.1 pandas==3.0.5 scikit-learn==1.9.0 plotly==7.1.0 matplotlib ./vendor/wm_notecards-0.1.0-py3-none-any.whl
```

Then open the chosen notebook and Run All. Colab execution is not claimed as
verified. The local CPU environment and saved HTML presentation are the validation targets.

## Executed result

On 13,532 historical bank-quarters (2024 Q1–Q3 predictors), zero growth has the
lowest MAE: **3.601 pp**, versus **3.641** for Ridge, **3.627** for the MLP,
and **5.110** for persistence. The MLP has the lowest RMSE, **6.429 pp**.
The MLP-minus-Ridge MAE difference is **-0.014 pp**, with a paired bank-cluster
95% interval of **-0.043 to +0.013 pp**. The interval includes zero.

The result does not establish added nonlinear value. In the realized bottom
decile, MLP MAE rises to **6.781 pp**. Its predicted bottom-decile selection has
about **21.8% precision** on an equal-quarter average. The original 94.54%
dollar concentration reproduces exactly.

A particularly instructive validation pair changes from PLUS INTERNATIONAL BANK
to EMIGRANT BANK on the same certificate; reported deposits grow by roughly
14,380%. It contributes 99.6% of the MLP validation squared error. The notebook
retains this outcome and flags the unresolved institutional change. It explains
why squared loss can be dominated by one observation; it does not justify
silently deleting a difficult example or tuning to historical performance.

## Concrete teaching and follow-up checks

Customers can request cash before bank loans are repaid. Both editions now begin
with a USD 100 asset illustration and a USD 20 withdrawal requiring USD 5 more cash.
Worked examples explain missing values, log growth, MAE/RMSE, and the difference
between a forecast and an ordering of banks. The complete EDA remains visible.

The original four-model scorecard remains identifiable. Additional comparisons,
developed after seeing 2024, give seasonal median **3.500 pp MAE** and show
**21.9%** equal-quarter precision for Ridge versus **18.8%** for size-only Ridge.
The paired bank-cluster interval for that precision difference is **+0.77 to
+5.40 pp**; the MLP–Ridge interval includes either ordering. These intervals are
conditional on fixed forecasts, fixed lists, and the three shared 2024 dates.
They do not measure uncertainty across future economic periods.

Capacity checks vary the selected fraction at 5%, 10%, and 20% while holding the
outcome group at the lowest-growth 10%. A verified October 1, 2023 merger and a
bounded Silvergate wind-down window support a separate, incomplete-event
sensitivity. Foundation-model checks use identical rows within history bands.
The final lesson describes a proposed shadow-mode trial with frozen models and
separate scoring-time and outcome-time coverage counts.

Generate the posting visuals after executing the notebooks:

```sh
.venv/bin/python export_story.py
.venv/bin/python package_project.py
```

The export command creates a fresh versioned folder under `exports/` with twelve
numbered portrait PNGs, one matching PDF, and a source manifest. Shared teaching
prose lives in `notebook_lessons.py` and `concrete_teaching.py`; experiment sources
under `notebooks/source/` are copied into visible notebook cells. The builder uses
Ruff for formatting when it is available and preserves runnable Python otherwise.

## Shared code and the shorter-window experiment

`deposit_experiment.py` contains the ordinary importable data, split, preprocessing,
model, and scoring functions. The builder embeds those same function definitions
as visible notebook code. No experiment executes Python extracted from prose strings.

Both editions include the shorter-window comparison after the deposit-size comparison.
It holds 2023 validation and the same 13,532 observed 2024 evaluation rows fixed.
The windows begin in 2013, 2020, and 2021 and all end in September 2022.
Three neural-network seeds and Ridge produced higher MAE with both shorter windows.
This follow-up was developed after inspecting 2024; it does not test 2026 conditions.

To reproduce it separately, without regenerating the notebooks:

```sh
.venv/bin/python experiments/training_window_sensitivity/run.py
# Optional: keep a verification run separate from saved results.
.venv/bin/python experiments/training_window_sensitivity/run.py --output /tmp/window-check
```

The existing evaluation has 13,618 input-eligible bank-quarters. Of these,
30 in March, 22 in June, and 34 in September lack a later report in the snapshot.
The scores and historical review lists describe the remaining 13,532 observed
outcomes. The coverage audit retains those 86 unresolved cases. Their absence
alone does not establish merger, failure, or distress. A deployed review list
could contain unresolved cases; the current precision does not measure that list.

## Rebuilding the optional foundation-model results

Normal notebook execution reads the saved predictions and needs no model download.
Regenerating those optional results is a separate, network-using workflow. The
recorded TimesFM backend is MLX on Apple Silicon:

```sh
uv run python benchmark_timesfm.py --revision 43046b85ec22d584a13f8098c2ed39c889e129c2
uv run python prepare_finetune_data.py
uv run python finetune_foundation.py chronos
uv run python finetune_foundation.py timesfm
```

The adapted backbones remain frozen; 2023 validation selects the saved forecast-head
checkpoint. See the source provenance and saved selection metadata for the exact
checkpoint and usage terms. Normal reproduction does not retrain these extensions.

## Sharing the project

Run `package_project.py` only after both notebooks execute successfully. It refuses
unexecuted or error-bearing notebooks. The packages include the shared module,
source data, reproducible experiments, saved results, and their reader-facing notes.
Internal editing notes, personal files, local archives, and browser QA are excluded.
The complete original experiment remains available because the notebooks use its evidence.

The current edited carousel is under `communications/simple-models-fight-back/`.
Older portrait exports are archived locally; `export_story.py` remains an optional
portrait-export workflow and does not rebuild the edited landscape carousel.

## Reproduced core models

`freeze_forecasts.py` verifies the original Ridge and seed-42 MLP, saves their
fitted preprocessing and models, reloads them, and scores all 13,618 input-eligible
2024 rows. Run it with `.venv/bin/python freeze_forecasts.py`. The manifest in
`growth_outputs/frozen_forecasts/` records numerical differences, exact original
list membership, row order, environment, and hashes. These are verified
reproductions; historical weight identity is unknown. Extended predictions stay
separate from the original observed-outcome predictions.

Build the offline HTML and ZIP deliverables after execution and tests:
` .venv/bin/python package_project.py`. Both bundles contain both executed editions,
the shared module, all three training-window seeds, and the frozen-model receipts.
