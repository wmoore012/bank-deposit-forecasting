# Communication revision: what the evidence lets a reader do

Status: proposed revision after reviewing the new communication notes on September 21, 2026. Existing notebook edits and generated exports remain uncommitted. The next implementation should finish verification before committing locally.

## The story in one sentence

Bank reports helped concentrate low-growth deposit outcomes in a small historical review list; a future shadow-mode trial would test whether that advantage survives new reports, changing coverage, and simple competing forecasts.

## Reader sequence

1. **Customers need cash. Loans take time.** Use the $15 cash / $20 withdrawal illustration. Explain why deposit changes deserve attention.
2. **4,536 banks. 454 places to look.** State the assumed March capacity, 45.44 expected random hits, 112 Ridge hits, and 109 MLP hits. Define the lowest-growth 10% as an outcome group.
3. **Twelve years have three different jobs.** Show 48 source quarters, 38 training predictor quarters, three validation quarters, three reused evaluation quarters. Separate bank counts from bank-quarter examples.
4. **What can a report tell us?** Five financial inputs, next-quarter growth, and an abbreviated worked target example. Full arithmetic remains in the notebook.
5. **The error measure changes the leader.** Original four-model comparison: zero growth leads MAE, MLP leads RMSE. Label seasonal median as a subsequent comparison and show its 3.500 pp MAE.
6. **Did the list concentrate the intended outcomes?** Model rows, quarterly dots, exact prevalence-adjusted lift, equal-quarter averages. Keep March counts separate from three-quarter averages.
7. **How much did complexity add?** Size-only versus full Ridge and MLP versus Ridge. Use paired difference intervals; explain shared banks and only three economic dates. Separate intervals for individual scores are insufficient to test a paired difference.
8. **What happened between these reports?** Merger timeline, +14,380% reported deposit growth, $32.44M to $4.70B. Effective October 1, 2023 falls one day after the September predictor report. Preserve true date spacing with an offset label.
9. **Who could we score—and who could we later evaluate?** Two separate coverage receipts. At scoring: reports available, in-scope banks, usable features/history, scored banks, reason codes. At evaluation: scored cases with observable next outcomes, unresolved outcomes, and coverage by model. Avoid using future-report availability to choose a prospective list.
10. **What changes when review time changes?** Historical sensitivity at 5%, 10%, 20%, with every model on matching populations and a defined outcome group. Keep the outcome target fixed at the bottom decile while changing review capacity, so precision and recall answer distinct questions. Label this follow-up; do not select the best capacity by 2024 performance.
11. **How would we test next quarter?** Proposed shadow mode: freeze rules and model versions → wait for reports to become available → save timestamped scores and lists → wait for next outcomes → compare with permanent challengers. Trial rankings do not drive operational decisions during this evaluation.
12. **Should anything change?** Review data changes, coverage, forecast errors, selection results, challengers, and large misses together. Investigate before deciding whether retraining or another change is justified. State the next untouched evaluation as the concrete next step.

## Visual grammar

- Warm off-white background and dark navy text support long-form reading.
- Muted blue marks measured balances/report dates. Amber marks a structural event requiring explanation. Teal marks the primary question or selected evidence.
- Reserve saturation for one focal comparison per card. Keep reference data neutral.
- Use shapes and direct text labels alongside color. Avoid assigning red/green bank-health meanings to growth outcomes.
- Keep MAE/RMSE units, selection denominators, predictor/outcome dates, and retrospective status visible on exported images.
- The merger KPI is a reported change, not a benefit or saving. Include the source publication date to distinguish retrospective annotation from information available when a forecast could have been issued.
- Illustrative monitoring sequences must carry an explicit illustration label. Do not render invented future lift values as observed results.

## Evidence safeguards

The NYDFS bulletin records Emigrant merging into Plus International under the Emigrant name effective October 1, 2023. The bulletin was published November 22, 2024. Approval and DOJ-review dates serve different purposes.

Source: https://www.dfs.ny.gov/reports-and-publications/weekly-bulletins/wb20241122

The deposit change comes from the project extract: CERT57083, September 30 to December31, 2023; DEPDOM 32,437 to 4,696,889 in USD thousands. The event establishes relevant structural context without allocating every dollar of change to the merger.

Silvergate's bounded wind-down window remains a separate timeline/event receipt. Two documented cases form an incomplete event list. Preserve all eligible observations in primary scores and report the same forecasts on the sensitivity population separately.

For shadow mode, distinguish report accounting date, publication/availability date, scoring timestamp, and outcome availability. The existing historical files do not establish a complete point-in-time information set.

Permanent forecast challengers: zero growth, training median, seasonal median, persistence, size-only Ridge, full Ridge, MLP. Constant-within-quarter forecasts have no within-quarter ranking signal; show them on forecast scorecards only.

Do not infer drift causes from a metric alone. Predefine review thresholds and what counts as sustained deterioration before the prospective trial. Retraining is a governed choice supported by evidence.

## Caption direction

Keep the LinkedIn caption focused on the discovery and next test. Let the carousel carry the definitions and evidence. Describe the quarterly loop as proposed work; the current deliverable is a reproducible historical experiment with follow-up checks.

## Implementation and acceptance

Retain both notebook editions and visible code. Integrate the timeline through the existing card helpers, with a static export matching its content. Add capacity sensitivity as visible cells, then document the shadow-mode protocol without claiming deployment. Export the revised twelve-page sequence, inspect every page and both responsive HTML editions, rerun evidence checks, reconcile shared results, and review a local commit. Publishing remains separate.
