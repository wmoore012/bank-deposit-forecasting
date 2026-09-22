# Simple Models Fight Back — accuracy review

**Use `simple_models_fight_back_reviewed.pdf` for the corrected version.**

The supplied PDF and generator are preserved as `supplied_original.pdf` and
`supplied_generator.py.txt`. They are archival inputs, not the recommended posting
version. The original uses custom HTML/SVG styled like cards; it does not import
wm-notecards. The reviewed PDF is a source-backed Matplotlib rendering in a white,
cyan, teal and blue layout. Its layout differs from the supplied deck.

## Corrections

- Scope zero growth's MAE win to the original four-model comparison. The subsequent
  seasonal median has a lower historical MAE.
- Label the histogram as **100 × log growth**, not ordinary percentage points.
  Use the notebook's already-trimmed central-98% trace without trimming it again.
  This shows training concentration; it does not establish why a forecast wins later.
- Add March-to-June 2024, 4,536 eligible banks, hypothetical 454 review slots, and
  approximate grid semantics to the hit comparison. Random expected hits are
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

Requires NumPy, pandas and Matplotlib, already used by the project. Produces eight
PNGs, an eight-page PDF, and `manifest.json` with SHA-256 hashes of numerical inputs.
`training_histogram.json` freezes the central-98% histogram from the notebook's
ignored Plotly chart output so the carousel can rebuild from a clone without running
models. Counts total 210,135 displayed rows from 214,425 training examples.

The reviewed PDF was rendered through Poppler and all eight pages were visually
checked. Original notebooks and other working-tree edits were left untouched.
