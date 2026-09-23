# %% NOTEBOOK CELL
# EXEMPLAR: formula-card
# Imagine 100 dollars of assets. Separate cash today from repayment later.
illustration = pd.DataFrame({
    "Item": ["Cash", "Loans", "Customer deposits", "Equity"],
    "Dollars": [15, 85, 90, 10],
    "Meaning": ["Available for payment", "Repaid over time", "Money owed to customers", "Assets minus liabilities"],
})
fig = go.Figure()
for amount, left, label, color in [
    (15, 0, "Cash now: USD 15", "#3F6294"),
    (85, 15, "Loans repaid over time: USD 85", "#C5CDD6"),
]:
    fig.add_trace(go.Bar(x=[amount], base=[left], y=["Assets"], orientation="h",
        text=[label], textposition="inside", marker_color=color, showlegend=False))
fig.update_layout(barmode="overlay")
fig.update_xaxes(range=[0, 100], title="Dollars of assets")
chart(fig, "cash_balance_sheet", "Example: a bank with USD 100 worth of assets",
      "Invented numbers for one balance-sheet example. USD 15 is cash today; USD 85 is loans repaid later.", height=410)

# %% NOTEBOOK CELL
# The request arrives before the loans have been repaid.
cash_available, withdrawal_request = 15, 20
cash_shortfall = withdrawal_request - cash_available
assert cash_shortfall == 5
fig = go.Figure()
fig.add_trace(go.Bar(x=[cash_available], y=["How to pay"], orientation="h",
    text=["15 available"], textposition="inside", marker_color="#3F6294", name="Cash available"))
fig.add_trace(go.Bar(x=[cash_shortfall], y=["How to pay"], orientation="h",
    text=["5 needed"], textposition="inside", marker_color="#A86223", name="Additional cash"))
fig.update_layout(barmode="stack", showlegend=False)
fig.update_xaxes(range=[0, 20], title="Invented customer request: USD 20")
chart(fig, "withdrawal_shortfall", "Example: USD 20 requested, USD 15 available, USD 5 to obtain",
    "Invented numbers. Blue is cash already available. Amber is the extra cash this example needs.", height=410)
takeaway("In this example, the bank still needs USD 5 in cash",
    "This is a teaching example, not a real withdrawal or a claim about a real bank. "
    "The USD 85 in loans is valuable, but it may be repaid years from now. "
    "For this example, the remaining USD 5 could come from borrowing or selling an asset. The cost and availability would need evidence.")

# %% NOTEBOOK CELL
# Now introduce how the assets are financed, after the cash problem is clear.
question_card(title="Okay, so where is equity in our made-up bank?", theme=theme,
    body="We are still using the same invented USD 100 example. The bank owes customers USD 90 in deposits. "
         "That leaves USD 10 of equity. The cash asset is still USD 15. Equity explains the difference. "
         "It does not create another USD 10 of cash.")
wm_formula_card(
    title="So what fills the USD 5 gap in this example?",
    subtitle="Still invented numbers. Equity explains the balance-sheet difference. It is not another pile of cash.",
    theme=theme,
    items=[
        {
            "label": "Cash gap",
            "latex": r"\$20 - \$15 = \$5",
            "fallback": "USD 20 requested minus USD 15 cash available equals USD 5 to obtain.",
        },
        {
            "label": "Equity",
            "latex": r"\$100 - \$90 = \$10",
            "fallback": "USD 100 in assets minus USD 90 in deposits equals USD 10 in equity.",
        },
        {
            "label": "After borrowing and paying",
            "latex": r"\$85 = \$70 + \$5 + \$10",
            "fallback": "USD 85 in assets equals USD 70 in deposits, USD 5 in borrowing, and USD 10 in equity.",
        },
    ],
)
wm_counterintuitive_card(
    title="Can the USD 10 equity balance pay the withdrawal?",
    theme=theme,
    why_misread="Equity sounds like a separate pile of spare cash.",
    ordinary_process="Equity is the difference between assets and liabilities. In this example the cash asset is USD 15; the loans total USD 85.",
    conclusion_boundary="The bank needs another USD 5 of cash, for example from borrowing or selling an asset. Equity absorbs losses if assets lose value. Funding cost and availability require their own investigation.",
    kicker="Cash and loss absorption",
    chip_text="LOOK TWICE",
)

# %% NOTEBOOK CELL
# Read the saved original run for an early preview; fresh execution verifies it later.
opening_ranking = pd.read_csv(ROOT / "growth_outputs/masterclass/ranking.csv")
opening_march = opening_ranking.loc[opening_ranking["Quarter"].eq("2024-03-31")].set_index("Model")
opening_banks = int(opening_march.loc["Ridge", "Banks"])
opening_slots = int(opening_march.loc["Ridge", "Selected"])
opening_counts = pd.DataFrame(
    {
        "Selection": ["Random expectation", "Ridge", "Neural network"],
        "Hits": [
            opening_slots**2 / opening_banks,
            opening_march.loc["Ridge", "Hits"],
            opening_march.loc["MLP", "Hits"],
        ],
    }
)
fig = go.Figure(
    go.Bar(
        x=opening_counts["Hits"],
        y=opening_counts["Selection"],
        orientation="h",
        text=[
            f"{v:.2f} expected" if i == 0 else f"{int(v)} found"
            for i, v in enumerate(opening_counts["Hits"])
        ],
        textposition="outside",
        marker_color=["#C5CDD6", "#3F6294", "#0B6F75"],
    )
)
fig.update_xaxes(title="Selected banks later in the lowest-growth 10%", range=[0, 140])
fig.update_yaxes(autorange="reversed", title="")
chart(
    fig,
    "opening_review",
    f"{opening_banks:,} banks. {opening_slots:,} places to look.",
    "March 2024 reports → June outcomes; fixed 10% capacity is an experiment assumption",
    height=480,
)
takeaway(
    "Imagine you can examine only 10 of every 100 banks",
    "After the next quarter, identify the 10 banks with the lowest deposit growth. "
    "Selecting 10 banks randomly would include about one of those banks on average. "
    "The Ridge and neural-network lists included about two, averaged across our three historical quarters. "
    "This is a simplified explanation of measured rates. The chart gives the exact March counts. "
    "The same amount of review time brought more intended cases into the list. "
    "A person would investigate why their deposits changed; the lowest-growth group can even include positive growth.",
)

# %% NOTEBOOK CELL
# Repeat the decisive counts in cards, beside the full chart above.
question_card(title="Which 454 banks would you examine first?", theme=theme,
    body="The experiment sorts banks from lowest to highest predicted deposit growth. After June’s reports arrive, we count how many selections actually belong to that quarter’s lowest-growth tenth.",
    kicker="March 2024 example", chip_text="THE DECISION")
big_number_card(title="Ridge found 112 of the intended cases", theme=theme,
    value=f"{int(opening_march.loc['Ridge', 'Hits']):,}", value_label=f"Among {opening_slots:,} selected banks",
    body=f"Random selection: {opening_slots**2 / opening_banks:.2f} expected. Neural network: {int(opening_march.loc['MLP', 'Hits'])} found. March-to-June 2024; hypothetical 10% review capacity.")
pictogram_card(percent=float(opening_march.loc['Ridge', 'Hits']) / opening_slots,
    headline="About one in four Ridge selections reached the lowest-growth group",
    big_text=f"{int(opening_march.loc['Ridge', 'Hits'])} / {opening_slots}", icon="dot", theme=theme,
    subtitle="March-to-June 2024: 24.7% of selected banks. Filled dots show an approximate share; each dot is a unit of the grid, rather than an individual bank. A person still investigates the cause.",
    kicker="Historical selection precision")

# %% NOTEBOOK CELL
# Keep the source history and the model's three jobs visible at the beginning.
opening_periods = pd.DataFrame(
    {
        "Job": ["Learn patterns", "Choose settings", "Measure historical performance"],
        "First predictor": pd.to_datetime(["2013-06-30", "2023-03-31", "2024-03-31"]),
        "Last predictor": pd.to_datetime(["2022-09-30", "2023-09-30", "2024-09-30"]),
        "Predictor quarters": [38, 3, 3],
        "Examples": [214425, 13834, 13532],
    }
)
fig = go.Figure()
for position, row in opening_periods.iterrows():
    fig.add_trace(
        go.Scatter(
            x=[row["First predictor"], row["Last predictor"]],
            y=[position, position],
            mode="lines+markers",
            showlegend=False,
            line=dict(width=18, color=["#0B6F75", "#A86223", "#3F6294"][position]),
            marker=dict(size=10),
            hovertemplate=f"{row['Job']}: {row['Predictor quarters']} quarters; {row['Examples']:,} examples<extra></extra>",
        )
    )
fig.update_yaxes(
    tickvals=[0, 1, 2],
    ticktext=["38 quarters · learn", "3 quarters · choose", "3 quarters · evaluate"],
    autorange="reversed",
)
fig.update_xaxes(title="Predictor report date", range=["2013-01-01", "2025-01-01"])
chart(
    fig,
    "history_roles",
    "Twelve years of reports have three different jobs",
    "48 source quarters; 38 training predictor quarters; three validation and three reused evaluation quarters",
    height=500,
)
display(opening_periods)
(OUT / "opening_periods.csv").write_text(opening_periods.to_csv(index=False))
