# %% NOTEBOOK CELL
# Follow-up: change how many banks we review, holding the desired outcome group fixed.
# The bottom decile still contains ceil(10% * banks) at every review capacity.
capacity_rows = []
for quarter, quarter_rows in extended.groupby("date"):
    bank_count = len(quarter_rows)
    outcome_count = int(np.ceil(0.10 * bank_count))
    actual_lowest = set(quarter_rows.sort_values(["growth", "CERT"]).head(outcome_count).CERT)
    for capacity in [0.05, 0.10, 0.20]:
        selected_count = int(np.ceil(capacity * bank_count))
        for model in ["Persistence", "Size-only Ridge", "Ridge", "MLP"]:
            selected = set(quarter_rows.sort_values([model, "CERT"]).head(selected_count).CERT)
            hits = len(selected & actual_lowest)
            capacity_rows.append({
                "Quarter": quarter, "Model": model, "Capacity": capacity,
                "Banks": bank_count, "Outcome count": outcome_count,
                "Selected": selected_count, "Hits": hits,
                "Precision": hits / selected_count,
                "Recall": hits / outcome_count,
                "Expected random hits": selected_count * outcome_count / bank_count,
                "Lift": (hits / selected_count) / (outcome_count / bank_count),
            })
capacity_results = pd.DataFrame(capacity_rows)
capacity_results.to_csv(OUT / "capacity_results.csv", index=False)
capacity_means = capacity_results.groupby(["Model", "Capacity"], as_index=False)[["Precision", "Recall", "Lift"]].mean()
capacity_means.to_csv(OUT / "capacity_means.csv", index=False)

# %% NOTEBOOK CELL
# Precision asks about our list. Recall asks about all the outcomes we wanted to find.
fig = make_subplots(rows=1, cols=3, subplot_titles=["Share of selections found", "Share of target group found", "Concentration vs random"])
capacity_colors = {"Persistence": "#627381", "Size-only Ridge": "#76528B", "Ridge": "#3F6294", "MLP": "#0B6F75"}
for model, frame in capacity_means.groupby("Model", sort=False):
    for column, metric in enumerate(["Precision", "Recall", "Lift"], start=1):
        fig.add_trace(go.Scatter(x=100*frame.Capacity, y=frame[metric], mode="lines+markers",
            name=model, legendgroup=model, showlegend=column==1,
            line_color=capacity_colors[model],
            hovertemplate=model+"<br>Review capacity: %{x}%<br>"+metric+": %{y:.3f}<extra></extra>"),row=1,col=column)
        fig.update_xaxes(tickvals=[5,10,20],ticksuffix="%",title="Review capacity",row=1,col=column)
for column in [1,2]:
    fig.update_yaxes(tickformat=".0%",rangemode="tozero",row=1,col=column)
fig.add_hline(y=1,line_dash="dash",line_color="#5E6871",row=1,col=3)
chart(fig,"capacity_check","Larger review lists find more cases with lower concentration",
    "Equal-quarter means; fixed bottom-decile outcome; follow-up using three reused 2024 quarters",height=640)
display(capacity_means.round(3))
for capacity in [0.05,0.10,0.20]:
    row = capacity_means.loc[capacity_means.Model.eq("Ridge") & capacity_means.Capacity.eq(capacity)].iloc[0]
    print(f"Ridge at {capacity:.0%} capacity: {row.Precision:.1%} of selections found the target; {row.Recall:.1%} of the target group was found; {row.Lift:.2f}x random concentration.")
takeaway("More review slots change two different fractions",
    "Precision asks: of the banks we selected, how many finished in the lowest-growth group? "
    "Recall asks: of all banks in that group, how many did we select? "
    "Inspect both before deciding whether additional review time is worthwhile. "
    "We retain the original 10% assumption and do not select a new capacity from these reused outcomes.")
