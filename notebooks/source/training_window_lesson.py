# Canonical model definitions remain visible; these are the runner's exact routines.
# SHARED ROUTINE: review_precision

# SHARED ROUTINE: compare_training_windows

# %% NOTEBOOK CELL
window_scores, window_predictions = compare_training_windows(rows)
window_scores.to_csv(OUT / 'training_window_scores.csv', index=False)
window_predictions.to_csv(OUT / 'training_window_predictions.csv', index=False)
shown_windows = window_scores.loc[
    window_scores['model'].eq('MLP') & window_scores['seed'].eq(42)
].copy()
shown_windows['Training window'] = shown_windows['training_start'].astype(str) + '–Sep 2022'

# EXEMPLAR: missingness-evidence
# Adapt the paired chart and exact receipt to the measured model comparison.
fig = px.bar(shown_windows, x='MAE_pp', y='Training window', orientation='h', text='MAE_pp')
fig.update_traces(marker_color=['#0B6F75', '#A86223', '#A86223'],
                  texttemplate='%{text:.3f}', textposition='outside')
fig.update_xaxes(title='Mean absolute error (percentage points), lower is better', range=[0, 5])
fig.update_yaxes(title='', autorange='reversed')
chart(fig, 'training_window_errors', 'And... the errors got bigger.',
      'Original seed 42. Same 13,532 observed 2024 outcomes; only the training start changed.', height=420)
table(shown_windows[['Training window', 'training_rows', 'MAE_pp', 'RMSE_pp']],
      'The same network, trained on fewer years', {'MAE_pp': '{:.3f}', 'RMSE_pp': '{:.3f}'})

# %% NOTEBOOK CELL
# Check every seed rather than selecting the best result from 2024.
all_window_receipt = window_scores[['training_start', 'model', 'seed', 'MAE_pp', 'RMSE_pp', 'bottom_decile_precision']]
table(all_window_receipt, 'All three random seeds and the Ridge comparison',
      {'MAE_pp': '{:.3f}', 'RMSE_pp': '{:.3f}', 'bottom_decile_precision': '{:.1%}'})
mlp_by_window = window_scores.loc[window_scores.model.eq('MLP')].pivot(
    index='seed', columns='training_start', values='MAE_pp')
assert (mlp_by_window[2020] > mlp_by_window[2013]).all()
assert (mlp_by_window[2021] > mlp_by_window[2013]).all()
# EXEMPLAR: bounded-takeaway
takeaway('Keep the full window for this historical comparison',
         'Both shorter windows raised MAE with all three random seeds. Ridge MAE rose too. '
         'The shorter windows also contain fewer examples, so this does not isolate an economic effect. '
         'We already inspected 2024; this follow-up cannot tell us which window works best in 2026.')
