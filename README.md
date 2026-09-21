# Can a small neural network forecast bank deposit growth?

Open **FDIC_Deep_Learning_Masterclass.ipynb** for the teaching notebook, or
**FDIC_Deep_Learning_Submission.ipynb** for the concise professor version.
Both run the same five-feature regression experiment with visible code.

The comparison is zero growth → persistence → Ridge → a 5–32–16–1 MLP.
Errors use percentage points. The 2024 evaluation is explicitly a **reused historical holdout**.
The original `Assign1.ipynb` is unchanged; the old two-head masterclass and its
results live in `experiments/experiment_0/`.

## Run

```sh
uv sync --python 3.12 --locked
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
- `notebooks/source/`: scratch and takeover notes for future work.

The 2010–2024 file is audited but not used for primary training: 5,738 early
form-100 reports cross the 2012 TFR-to-Call-Report transition without a verified
five-field historical crosswalk. The core retains 2020–2024. Missing equity
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
lowest MAE: **3.601 pp**, versus **3.944** for Ridge, **4.140** for the MLP,
and **5.110** for persistence. Ridge has the lowest RMSE, **6.617 pp**.
The MLP-minus-Ridge MAE difference is **+0.196 pp**, with a conditional paired
bank-cluster 95% interval of **+0.165 to +0.230 pp**. Positive favors Ridge.

The result does not establish added nonlinear value. In the realized bottom
decile, MLP MAE rises to **8.778 pp**. Its predicted bottom-decile selection has
about **14.9% precision** on an equal-quarter average. The original 94.54%
dollar concentration reproduces exactly.

A particularly instructive validation pair changes from PLUS INTERNATIONAL BANK
to EMIGRANT BANK on the same certificate; reported deposits grow by roughly
14,380%. It contributes 99.6% of the MLP validation squared error. The notebook
retains this outcome and flags the unresolved institutional change. It explains
why squared loss can be dominated by one observation; it does not justify
silently deleting a difficult example or tuning to historical performance.
