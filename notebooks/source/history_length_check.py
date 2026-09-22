# %% NOTEBOOK CELL
# Join stored predictions by bank and date, preserving the same evaluation population.
history_comparison = extended[["CERT", "date", "growth"] + comparison_names].copy()
for file_name, name, prediction_column in [
    ("timesfm_zero_shot_predictions.csv", "TimesFM zero-shot", "TimesFM"),
    ("chronos_zero_shot_predictions.csv", "Chronos zero-shot", "Chronos-Bolt"),
    ("timesfm_finetuned_predictions.csv", "TimesFM head adapted", "Prediction"),
    ("chronos_finetuned_predictions.csv", "Chronos head adapted", "Prediction"),
]:
    loaded = pd.read_csv(ROOT / "growth_outputs" / file_name, parse_dates=["date"])
    columns = ["CERT", "date", prediction_column]
    if name == "TimesFM zero-shot":
        columns.append("Context quarters")
    history_comparison = history_comparison.merge(
        loaded[columns].rename(columns={prediction_column: name}),
        on=["CERT", "date"],
        how="left",
        validate="one_to_one",
    )
assert len(history_comparison) == len(extended)
assert history_comparison.notna().all().all()
history_comparison["History band"] = pd.cut(
    history_comparison["Context quarters"],
    bins=[1, 3, 7, 15, 31, np.inf],
    labels=["2–3", "4–7", "8–15", "16–31", "32+"],
)
foundation_names = [
    "TimesFM zero-shot",
    "Chronos zero-shot",
    "TimesFM head adapted",
    "Chronos head adapted",
]
history_scores = []
for band, part in history_comparison.groupby("History band", observed=True):
    for name in comparison_names + foundation_names:
        history_scores.append(
            {
                "History quarters": band,
                "Model": name,
                "Rows": len(part),
                **scores(part["growth"], part[name]),
            }
        )
history_scores = pd.DataFrame(history_scores)
history_scores.to_csv(OUT / "history_length_scores.csv", index=False)
display(history_scores.round(3))
fig = px.line(
    history_scores.loc[history_scores["Model"].isin(["Zero growth", "MLP"] + foundation_names)],
    x="History quarters",
    y="RMSE (pp)",
    color="Model",
    markers=True,
    log_y=True,
    hover_data=["Rows"],
)
chart(
    fig,
    "history_length",
    "History-length bands expose uneven foundation-model errors",
    "Same banks within each band; logarithmic error axis; bands chosen after viewing 2024",
    height=600,
)
takeaway(
    "History length describes the cases where errors occur",
    "These groups also differ in bank age, size, and circumstances. "
    "A large error in a short history motivates controlled tests using truncated histories of the same banks. "
    "The full-population scores remain the main foundation-model comparison.",
)
