# EXEMPLAR: missingness-evidence
history_output = ROOT / 'experiments/review_history/outputs'
history_plan = json.loads((history_output / 'plan.json').read_text())
history_capture = pd.read_csv(history_output / 'outcome_capture.csv')
history_overlap = pd.read_csv(history_output / 'expanded_overlap.csv')
history_methods = ['Ridge', 'MLP', 'Low equity/assets', 'Largest deposits',
                   'PCA', 'Isolation Forest 42', 'Autoencoder 42']
history_main = history_capture.loc[
    history_capture.date.eq('All three quarters') & history_capture.method.isin(history_methods)
].sort_values('capture_per_selected')
fig = px.bar(history_main, x='capture_per_selected', y='method', orientation='h', text='captured')
fig.update_traces(marker_color=['#0B6F75' if method == 'Ridge' else '#8098AA' for method in history_main.method])
fig.update_xaxes(title='Share of selections with lowest-decile growth', tickformat='.0%', range=[0, .27])
fig.update_yaxes(title='')
chart(fig, 'history_review_outcomes', 'Ridge still captured the most low-growth outcomes',
      'Eight-quarter history comparison. Same matched population and 10% capacity; primary seed 42.', height=510)
table(history_main[['method', 'captured', 'selected', 'capture_per_selected']],
      'New matched-population lists; original Study 1 lists remain unchanged',
      {'capture_per_selected': '{:.1%}'})

# %% NOTEBOOK CELL
history_pair = history_overlap.loc[
    history_overlap.date.eq('All three quarters') & history_overlap.left.eq('Ridge') & history_overlap.right.eq('PCA')
].iloc[0]
history_pooled = history_capture.loc[history_capture.date.eq('All three quarters')].set_index('method')
history_ridge = history_pooled.loc['Ridge']
history_auto = history_pooled.loc['Autoencoder 42']
# EXEMPLAR: bounded-takeaway
takeaway('Different histories led to different lists',
         f'Ridge and PCA shared {int(history_pair.shared)} of {int(history_pair.left_count)} selections '
         f'across the expanded common-history population ({history_pair.left_share:.1%}). '
         f'On {history_plan["population_counts"]["original_complete"]:,} rows with observed outcomes, '
         f'Ridge captured {int(history_ridge.captured)} low-growth outcomes in {int(history_ridge.selected)} selections; '
         f'the primary autoencoder captured {int(history_auto.captured)}. '
         'None of the declared history detectors exceeded Ridge on this historical outcome. '
         'The separate Study 2 notebook shows every seed, the deposit-size removal check, and three actual histories.')
