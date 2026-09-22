# Simple Models Fight Back — accuracy review

**Current design: `simple_models_fight_back_data_science_story_v5.html` and its matching PDF.**

The user-supplied nine-slide HTML design supersedes the six-slide Matplotlib layout below. Slides 4 and 6 clarify squared-error training and show the first ten entries of the actual Ridge review queue. The full 454-row queue is saved in `ridge_review_queue_2024-03-31.csv`; forecasts are ordinary-growth fractions. Sort order is Ridge prediction ascending, then FDIC certificate number, matching the notebook. The 112 matches were recomputed against the realized 454-bank bottom decile. The HTML prints at 1792 × 1024 CSS pixels per page. A small slide 8 chart-height adjustment prevents its conclusion from colliding with its footnote.

The supplied PDF and generator are preserved as `supplied_original.pdf` and
`supplied_generator.py.txt`. They are archival inputs, not the recommended posting
version. The original uses custom HTML/SVG styled like cards; it does not import
wm-notecards. The reviewed PDF is a source-backed Matplotlib rendering in a white,
cyan, teal and blue layout. Its layout differs from the supplied deck.

## Current six-slide story

The posting sequence follows the flat-line hook, training histogram, MAE/RMSE
reversal, fixed-capacity review list, one-input Ridge challenger, and seasonal-rule
ending. Foundation-model comparisons and the monitoring epilogue are omitted from
this shorter carousel; their source results remain in the repository. Every slide
uses an analytical object: saved model predictions on identical axes, an actual
training histogram, score dotplots, zero-baseline precision bars, a Ridge challenger
dumbbell, or seasonal frequencies with a historical score comparison.

The prediction panels use `predictions.csv`, whose outcomes and predictions are
already ordinary growth fractions. Scores are recomputed after multiplying by 100
and checked against the saved MAE and RMSE. Both panels show the same rows within
a shared square window defined by the actual-growth 1st and 99th percentiles;
rows outside either coordinate are omitted from both displays, never from scoring.

## Accuracy review history

- Scope zero growth's MAE win to the original four-model comparison. The subsequent
  seasonal median has a lower historical MAE.
- Label the histogram as **100 × log growth**, not ordinary percentage points.
  Use the notebook's already-trimmed central-98% trace without trimming it again.
  This shows training concentration; it does not establish why a forecast wins later.
- Include March-to-June 2024, 4,536 eligible banks, and hypothetical 454 review slots
  in the hit comparison. The current version uses zero-baseline precision bars. Random expected hits are
  454 × 454 / 4,536 = 45.440035.
- Lift uses equal-quarter means of precision divided by actual quarter prevalence:
  persistence 1.26005×, size-only 1.87931×, Ridge 2.18758×, MLP 2.17384×.
- Correct Chronos RMSE to **7.224** zero-shot and **7.418** head-adapted.
  The supplied 7.220 and 7.420 were not correct three-decimal values.
- Describe head adaptation as **256 sampled updates**, rather than implying a full
  training epoch or comprehensive tuning. Checkpoints were selected on earlier
  validation; pretraining overlap remains unknown.
- Restrict foundation-model conclusions to the tested checkpoints, data and budget.
- Remove the unsupported claim that the slide generator itself uses wm-notecards.
  The research notebook does use wm-notecards.
- Include the reused evaluation, post-hoc baseline, uncertainty and no-savings limits.

## Rebuild

From the repository root:

```sh
.venv/bin/python communications/simple-models-fight-back/build.py
```

Requires NumPy, pandas and Matplotlib, already used by the project. Produces six
PNGs, a six-page PDF, and `manifest.json` with SHA-256 hashes of numerical inputs.
`training_histogram.json` freezes the central-98% histogram from the notebook's
ignored Plotly chart output so the carousel can rebuild from a clone without running
models. Counts total 210,135 displayed rows from 214,425 training examples.

The reviewed PDF was rendered through Poppler and all six pages were visually
checked. Original notebooks and other working-tree edits were left untouched.
