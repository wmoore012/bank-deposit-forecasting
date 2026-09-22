# This source is copied into both notebooks as visible code cells.
# %% NOTEBOOK CELL
# Fit two fixed-rule forecasts using earlier outcomes only.
# A median is the middle training value. The seasonal rule uses one per quarter.
training_median = float(train["log_growth"].median())
quarter_medians = train.groupby(train["date"].dt.quarter)["log_growth"].median()
assert set(quarter_medians.index) == {1, 2, 3, 4}

# Bank size gets its own benchmark so we can assess the five-feature comparison.
size_scaler = StandardScaler().fit(train[["log_deposits"]])
size_train = size_scaler.transform(train[["log_deposits"]])
size_valid = size_scaler.transform(valid[["log_deposits"]])
size_model, size_alpha, size_tuning = choose_ridge(
    size_train,
    y_train,
    size_valid,
    y_valid,
    settings["ridge_alphas"],
)
size_tuning.to_csv(OUT / "size_only_validation.csv", index=False)


def followup_forecasts(frame):
    """Use the already-fitted rules; this function never learns from frame."""
    return {
        "Training median": np.full(len(frame), np.expm1(training_median)),
        "Seasonal median": np.expm1(frame["date"].dt.quarter.map(quarter_medians)),
        "Size-only Ridge": np.expm1(
            size_model.predict(size_scaler.transform(frame[["log_deposits"]]))
        ),
    }


# %% NOTEBOOK CELL
# Give validation and historical evaluation their own rows. No setting is selected here.
followup_score_rows = []
for period, frame in [("Validation", valid), ("Reused 2024", holdout)]:
    predictions_for_period = followup_forecasts(frame)
    for name, prediction in predictions_for_period.items():
        followup_score_rows.append(
            {
                "Period": period,
                "Model": name,
                "Rows": len(frame),
                **scores(frame["growth"], prediction),
            }
        )
followup_scores = pd.DataFrame(followup_score_rows)
followup_scores.to_csv(OUT / "followup_scores.csv", index=False)
display(followup_scores.round(3))

extended = result_frame.copy()
for name, prediction in followup_forecasts(holdout).items():
    extended[name] = np.asarray(prediction)
extended.to_csv(OUT / "followup_predictions.csv", index=False)
followup_names = ["Training median", "Seasonal median", "Size-only Ridge"]
comparison_names = MODEL_ORDER + followup_names
combined_scores = pd.DataFrame(
    [{"Model": name, **scores(extended["growth"], extended[name])} for name in comparison_names]
).sort_values("MAE (pp)")
combined_scores.to_csv(OUT / "extended_scores.csv", index=False)

fig = go.Figure()
for _, row in combined_scores.iterrows():
    fig.add_trace(
        go.Scatter(
            x=[row["MAE (pp)"]],
            y=[row["Model"]],
            mode="markers+text",
            text=[f"{row['MAE (pp)']:.3f} pp"],
            textposition="middle right",
            showlegend=False,
            marker=dict(size=13, color="#0B6F75" if row["Model"] in followup_names else "#627381"),
        )
    )
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(title="Mean absolute error (percentage points)", range=[3.3, 5.7])
chart(
    fig,
    "followup_errors",
    "The seasonal median improves historical MAE in a follow-up check",
    "Teal = follow-up comparison; all 13,532 reused-2024 rows; lower is better",
    height=600,
)
winning_followup = combined_scores.iloc[0]
takeaway(
    f"{winning_followup['Model']} has the lowest MAE in this expanded comparison",
    f"Its historical MAE is {winning_followup['MAE (pp)']:.3f} pp. "
    "The additional comparisons were developed after 2024 had been examined. "
    "The next evaluation should freeze these rules before new outcomes arrive.",
)

# %% NOTEBOOK CELL
# Pool training bank-quarters to answer: what fraction declined after each quarter?
# This differs from the earlier chart's median across separate years.
seasonal_rates = (
    train.assign(Quarter=train["date"].dt.quarter, Declined=train["growth"].lt(0))
    .groupby("Quarter")
    .agg(Rows=("CERT", "size"), Declines=("Declined", "sum"))
)
seasonal_rates["Decline share"] = seasonal_rates["Declines"] / seasonal_rates["Rows"]
seasonal_rates["Median log growth"] = quarter_medians
seasonal_rates["Median growth (%)"] = 100 * np.expm1(quarter_medians)
seasonal_rates.to_csv(OUT / "seasonal_rules.csv")
display(seasonal_rates.round(4))
takeaway(
    "Calendar timing deserves a frozen baseline in the next experiment",
    "The table pools bank-quarter observations within each predictor quarter of the year. "
    "The earlier dots keep individual years visible. A repeating pattern can motivate "
    "a forecast rule; performance on a later untouched period will test its usefulness.",
)


# %% NOTEBOOK CELL
# Freeze the two sets before counting overlap. Equal sizes make precision = recall here.
def selection_evidence(frame, models):
    """Return per-quarter counts and row-level flags for the fitted ranking rules."""
    summaries = []
    flags = []
    for quarter, part in frame.groupby("date"):
        count = int(np.ceil(0.10 * len(part)))
        actual_ids = set(part.sort_values(["growth", "CERT"]).head(count)["CERT"])
        for name in models:
            # An identical forecast cannot meaningfully order banks within a quarter.
            if part[name].nunique() == 1:
                continue
            chosen_ids = set(part.sort_values([name, "CERT"]).head(count)["CERT"])
            selected = part["CERT"].isin(chosen_ids)
            hit = selected & part["CERT"].isin(actual_ids)
            prevalence = len(actual_ids) / len(part)
            precision_value = hit.sum() / count
            summaries.append(
                {
                    "Quarter": quarter,
                    "Model": name,
                    "Banks": len(part),
                    "Selected": count,
                    "Hits": int(hit.sum()),
                    "Precision": precision_value,
                    "Random expectation": prevalence,
                    "Expected random hits": count * prevalence,
                    "Lift": precision_value / prevalence,
                }
            )
            flags.append(
                pd.DataFrame(
                    {
                        "CERT": part["CERT"],
                        "Quarter": quarter,
                        "Model": name,
                        "Selected": selected.astype(int),
                        "Hit": hit.astype(int),
                    }
                )
            )
    return pd.DataFrame(summaries), pd.concat(flags, ignore_index=True)


ranking_models = ["Persistence", "Size-only Ridge", "Ridge", "MLP"]
followup_ranking, selection_flags = selection_evidence(extended, ranking_models)
followup_ranking.to_csv(OUT / "followup_ranking.csv", index=False)
selection_flags.to_csv(OUT / "selection_flags.csv", index=False)
mean_ranking = followup_ranking.groupby("Model")[["Precision", "Lift"]].mean()
mean_ranking = mean_ranking.sort_values("Lift", ascending=False)
mean_ranking.to_csv(OUT / "mean_ranking.csv")

fig = go.Figure(
    go.Scatter(
        x=mean_ranking["Lift"],
        y=mean_ranking.index,
        mode="markers+text",
        text=[f"{v:.2f}×" for v in mean_ranking["Lift"]],
        textposition="middle right",
        marker=dict(size=16, color="#0B6F75"),
    )
)
fig.add_vline(x=1, line_dash="dash", line_color="#707B87")
fig.update_xaxes(title="Concentration relative to random selection", range=[0.8, 2.7])
fig.update_yaxes(autorange="reversed")
chart(
    fig,
    "review_lift",
    "Five-feature rankings concentrate more cases than size alone",
    "Equal weight for each of three reused quarters; dashed line = random selection",
    height=520,
)
takeaway(
    "Bank size supplies useful ranking information in these quarters",
    f"Size-only Ridge: {mean_ranking.loc['Size-only Ridge', 'Precision']:.1%} precision. "
    f"Five-feature Ridge: {mean_ranking.loc['Ridge', 'Precision']:.1%}. "
    "This comparison measures association and a fitted selection rule. "
    "The paired intervals below show how much the difference varies across resampled banks.",
)

# %% NOTEBOOK CELL
# Resample banks, carrying each bank's repeated quarters together.
# Keep fitted models and the original review lists fixed: these are conditional intervals.
bank_ids = np.sort(extended["CERT"].unique())
quarters = sorted(extended["date"].unique())
selection_arrays = {}
for name in ranking_models:
    model_flags = selection_flags.loc[selection_flags["Model"].eq(name)]
    selection_arrays[name] = [
        model_flags.pivot(index="CERT", columns="Quarter", values=column)
        .reindex(index=bank_ids, columns=quarters)
        .fillna(0)
        .to_numpy()
        for column in ["Hit", "Selected"]
    ]

error_arrays = {}
for name in comparison_names:
    bank_errors = (
        extended.assign(Error=100 * (extended[name] - extended["growth"]).abs())
        .groupby("CERT")["Error"]
        .agg(["sum", "count"])
        .reindex(bank_ids)
    )
    error_arrays[name] = bank_errors.to_numpy()

rng = np.random.default_rng(42)
bootstrap_rows = []
for repeat in range(1000):
    multiplicity = rng.multinomial(len(bank_ids), np.full(len(bank_ids), 1 / len(bank_ids)))
    sampled_precision = {}
    for name, (hit_array, selected_array) in selection_arrays.items():
        quarter_hits = multiplicity @ hit_array
        quarter_selections = multiplicity @ selected_array
        sampled_precision[name] = np.mean(quarter_hits / quarter_selections)
    sampled_mae = {
        name: (multiplicity @ values[:, 0]) / (multiplicity @ values[:, 1])
        for name, values in error_arrays.items()
    }
    for first, second in [("Ridge", "Size-only Ridge"), ("MLP", "Ridge")]:
        bootstrap_rows.append(
            {
                "Repeat": repeat,
                "Metric": "Precision difference (pp)",
                "Comparison": f"{first} minus {second}",
                "Difference": 100 * (sampled_precision[first] - sampled_precision[second]),
            }
        )
    for first, second in [
        ("MLP", "Zero growth"),
        ("MLP", "Ridge"),
        ("Seasonal median", "Zero growth"),
    ]:
        bootstrap_rows.append(
            {
                "Repeat": repeat,
                "Metric": "MAE difference (pp)",
                "Comparison": f"{first} minus {second}",
                "Difference": sampled_mae[first] - sampled_mae[second],
            }
        )
bootstrap_samples = pd.DataFrame(bootstrap_rows)
bootstrap_samples.to_csv(OUT / "paired_bootstrap_samples.csv", index=False)

interval_rows = []
score_lookup = combined_scores.set_index("Model")["MAE (pp)"]
for (metric_name, label), samples in bootstrap_samples.groupby(["Metric", "Comparison"]):
    first, second = label.split(" minus ")
    observed = (
        100 * (mean_ranking.loc[first, "Precision"] - mean_ranking.loc[second, "Precision"])
        if metric_name.startswith("Precision")
        else score_lookup[first] - score_lookup[second]
    )
    low, high = samples["Difference"].quantile([0.025, 0.975])
    interval_rows.append(
        {
            "Metric": metric_name,
            "Comparison": label,
            "Observed": observed,
            "95% lower": low,
            "95% upper": high,
            "Banks": len(bank_ids),
        }
    )
paired_intervals = pd.DataFrame(interval_rows)
paired_intervals.to_csv(OUT / "paired_intervals.csv", index=False)
display(paired_intervals.round(3))

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["MAE: lower favors first model", "Precision: higher favors first model"],
    horizontal_spacing=0.32,
)
for column, metric_name in enumerate(["MAE difference (pp)", "Precision difference (pp)"], 1):
    for _, row in paired_intervals.loc[paired_intervals["Metric"].eq(metric_name)].iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["Observed"]],
                y=[row["Comparison"]],
                mode="markers",
                marker=dict(size=12, color="#0B6F75"),
                showlegend=False,
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[row["95% upper"] - row["Observed"]],
                    arrayminus=[row["Observed"] - row["95% lower"]],
                ),
            ),
            row=1,
            col=column,
        )
    fig.add_vline(x=0, line_dash="dash", line_color="#707B87", row=1, col=column)
    fig.update_xaxes(title="Difference (percentage points)", row=1, col=column)
chart(
    fig,
    "paired_uncertainty",
    "Paired intervals show which historical differences remain uncertain",
    "1,000 paired bank resamples; fixed predictions and lists; three historical quarters",
    height=590,
)
takeaway(
    "Thousands of banks still share only three evaluation dates",
    "An interval crossing zero includes both directions of the measured difference. "
    "An interval excluding zero describes this conditional comparison. "
    "These resamples keep the 2024 environment and trained models fixed; "
    "a later untouched period is needed to test whether the result persists.",
)

# %% NOTEBOOK CELL
# Save a machine-readable account of what was fitted and when.
followup_design = {
    "status": "Follow-up analyses developed after examining 2024",
    "training_rows": len(train),
    "validation_rows": len(valid),
    "evaluation_rows": len(holdout),
    "training_last_outcome": str(train["target_date"].max()),
    "validation_last_outcome": str(valid["target_date"].max()),
    "training_median_log_growth": training_median,
    "seasonal_medians": quarter_medians.to_dict(),
    "size_ridge_alpha": size_alpha,
    "bootstrap": "1000 paired bank clusters; fixed fitted models and original selection lists",
    "average_ranking": "Arithmetic mean of three quarterly values; exact k/n random reference",
}
(OUT / "followup_design.json").write_text(json.dumps(followup_design, indent=2))
