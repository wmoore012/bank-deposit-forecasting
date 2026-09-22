# %% NOTEBOOK CELL
# Official event dates are annotations for a retrospective sensitivity.
# They are never fed into a forecast or used to choose model settings.
events = pd.read_csv(
    ROOT / "sources/structural_events.csv",
    parse_dates=["Window start", "Window end"],
)
display(events[["CERT", "Event", "Window start", "Window end", "Date meaning"]])


def known_event_mask(frame):
    """Flag a bank's forecast interval when it overlaps a verified event window."""
    flagged = pd.Series(False, index=frame.index)
    for _, event in events.iterrows():
        flagged |= (
            frame["CERT"].eq(event["CERT"])
            & frame["date"].lt(event["Window end"])
            & frame["target_date"].ge(event["Window start"])
        )
    return flagged


validation_frame = valid[
    ["CERT", "NAME", "date", "target_date", "DEPDOM", "next_deposits", "growth"]
].copy()
for name, prediction in validation_predictions.items():
    validation_frame[name] = np.expm1(prediction)

sensitivity_rows = []
flagged_examples = []
for period, frame in [("Validation", validation_frame), ("Reused 2024", result_frame)]:
    flagged = known_event_mask(frame)
    flagged_examples.append(frame.loc[flagged].assign(Period=period))
    for population, keep in [
        ("All eligible rows", pd.Series(True, index=frame.index)),
        ("Verified event intervals omitted", ~flagged),
    ]:
        for name in MODEL_ORDER:
            sensitivity_rows.append(
                {
                    "Period": period,
                    "Population": population,
                    "Model": name,
                    "Rows": int(keep.sum()),
                    "Flagged event rows": int(flagged.sum()),
                    **scores(frame.loc[keep, "growth"], frame.loc[keep, name]),
                }
            )
structural_scores = pd.DataFrame(sensitivity_rows)
structural_scores.to_csv(OUT / "structural_sensitivity.csv", index=False)
flagged_examples = pd.concat(flagged_examples, ignore_index=True)
flagged_examples.to_csv(OUT / "verified_event_rows.csv", index=False)
display(structural_scores.round(3))

fig = px.scatter(
    structural_scores.loc[structural_scores["Period"].eq("Validation")],
    x="RMSE (pp)",
    y="Model",
    color="Population",
    symbol="Population",
    hover_data=["Rows", "Flagged event rows"],
    log_x=True,
    color_discrete_sequence=["#A86223", "#0B6F75"],
)
fig.update_traces(marker_size=13)
chart(
    fig,
    "structural_sensitivity",
    "Verified structural events change the validation error comparison",
    "Fixed models; logarithmic RMSE axis; the primary scores retain every eligible row",
    height=530,
)
takeaway(
    "Reported deposit changes can include institutional restructuring",
    "The two-source event list identifies a merger interval and a bounded liquidation period. "
    "It is incomplete and was assembled retrospectively. Unchanged 2024 scores mean "
    "these particular windows flag no 2024 rows; other structural events may remain.",
)

# %% NOTEBOOK CELL
# Classify reporting gaps by observable dates. Their economic causes remain unknown.
comparable_panel = panel.loc[conditions["Insured domestic bank; comparable Call Report"]].copy()
comparable_panel["Prior report status"] = np.select(
    [
        comparable_panel["quarter_number"].eq(panel["quarter_number"].min()),
        comparable_panel["prior_quarter"].isna(),
        comparable_panel["quarter_number"].sub(comparable_panel["prior_quarter"]).ne(1),
    ],
    ["Dataset start", "First observed bank report", "Gap after earlier report"],
    default="Adjacent prior report",
)
gap_rows = []
for side, frame, column in [
    ("Prior", comparable_panel, "Prior report status"),
    (
        "Next after prior-adjacency filter",
        comparable_panel.loc[conditions["Adjacent prior quarter"]],
        "Next report status",
    ),
]:
    counts = frame.groupby(column).size().sort_values(ascending=False)
    for status, count in counts.items():
        gap_rows.append({"Stage": side, "Observed status": status, "Rows": count})
report_gaps = pd.DataFrame(gap_rows)
report_gaps.to_csv(OUT / "report_gap_classification.csv", index=False)
display(report_gaps)
takeaway(
    "An absent next report leaves the forecast without an observed answer",
    "Dataset endpoints, first or last appearances, and gaps are distinguishable from dates alone. "
    "Merger, closure, failure, and reporting explanations require additional records. "
    "This study evaluates bank-quarters with adjacent positive balances; institutions missing that outcome remain outside its scored population.",
)
