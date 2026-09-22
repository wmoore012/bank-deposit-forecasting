# %% NOTEBOOK CELL
# EXEMPLAR: formula-card
# Illustration: three absolute errors, expressed in percentage points.
# Compare the average miss with a score that puts extra weight on large misses.
toy_errors = pd.DataFrame({"Forecast A": [1., 1., 7.], "Forecast B": [4., 4., 4.]})
toy_scores = pd.DataFrame({
    "MAE (pp)": toy_errors.mean(),
    "RMSE (pp)": np.sqrt(toy_errors.pow(2).mean()),
})
display(toy_errors.rename_axis("Illustrative case").round(2))
display(toy_scores.round(3))
toy_scores.to_csv(OUT / "metric_illustration.csv")
question_card(title="Would you prefer two tiny misses and one large miss?", theme=theme,
    body="Forecast A misses by 1, 1, and 7 percentage points. Forecast B misses by 4 points every time. The score you choose changes the answer.",
    kicker="Three invented forecasts", chip_text="TRY IT")
wm_counterintuitive_card(title="The seven-point miss changes the ordering", theme=theme,
    why_misread="A is closer on two of the three cases: 1 point versus 4 points.",
    ordinary_process="MAE adds the misses: A totals 9 points; B totals 12. Divide by three: A scores 3, B scores 4. A has the smaller average absolute miss.",
    conclusion_boundary="RMSE squares each miss first. A: 1 + 1 + 49 = 51. B: 16 + 16 + 16 = 48. Divide by three and take the square root: A scores 4.123, B scores 4. B now scores lower.",
    kicker="MAE versus RMSE", chip_text="LOOK TWICE")
big_number_card(title="One large miss can change the comparison", theme=theme,
    value="49 of 51", value_label="A’s squared-error total comes mostly from one case",
    body="The seven-point miss contributes 49 after squaring.\nThe two one-point misses contribute only 2 together.\nThese are invented examples. The next chart evaluates the real bank forecasts.")
