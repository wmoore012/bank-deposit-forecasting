# Simple Models Fight Back

The current carousel is `simple_models_fight_back_data_science_story_v6.pdf`,
a 13-page edition. It adds the approved anomaly-detection setup, forecast and feature-space mini-plots with explicit selection rules
at equal review capacity, and the measured outcome
comparison after the deposit-size page. The calendar-rule ending is preserved. Page 11 now uses horizontal stacks showing
full-window MAE plus the increase for each shorter window, using the saved
training-window sensitivity scores.

`extend_story.py` builds this edition from the preserved 11-page v5 PDF and
`experiments/review_history/outputs/outcome_capture.csv`. It requires Python with
ReportLab, pypdf, and the macOS Arial fonts. Run from any working directory:

```sh
python communications/simple-models-fight-back/extend_story.py
```

The v5 HTML and PDF remain the earlier source edition. The HTML does not contain
the new pages. Both PDF editions use 1344 × 768 PDF points per page.

The notebook is the full computational walkthrough. This carousel presents the
original four-model result, deposit-growth intuition, the actual historical Ridge
review queue, the size-only comparison, shorter training windows, and a calendar
rule selected for a future test. The shorter-window result is reproduced by
`experiments/training_window_sensitivity/run.py` from the project root.

The one-bank timeline is illustrative. The 2013 labels deliberately show a stable
year. All-model scores cover the matched observed evaluation population, not every
bank that could have received a forecast. The reused 2024 evaluation has only three
predictor quarters; later experiments remain exploratory.

The saved FDIC history, historical review queue and forecast snapshot in this
folder retain provenance for the corresponding graphics. Earlier six-page and
portrait designs are superseded; they are not the current output.

The Study 2 outcome chart pools 1,351 bank-quarter selections across three 2024
quarters on 13,490 complete histories with observed outcomes. It is distinct from
the original population and equal-quarter averages used earlier in the story.
The chart shows seed 42 for stochastic methods; the full experiment preserves
all seeds and the deposit-size-removal comparison. Reused outcomes remain
exploratory. The v6 additions were checked as rendered PDF pages, not as HTML.

Pages 8 and 10 distinguish the review-list task (higher capture is better) from
forecast error (lower MAE is better). The Study 2 chart includes both simple
controls as well as all five learned methods. Mini-plots on page 9 are labeled
schematic illustrations; they are not empirical projections or forecasts.
