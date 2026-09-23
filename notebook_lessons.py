"""Teaching prose and assembly of visible notebook experiments.

Source files are copied into cells at build time, so learners can inspect every
calculation in the notebook. This module changes presentation, not fitted models.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

OPENING = """**Why examine a bank’s deposits?**

Customers expect their bank to complete payments and withdrawals when needed. A bank therefore needs cash and other reliable ways to obtain cash.

Some assets are loans that customers will repay over years. Payments and withdrawals can arrive much sooner. When deposits leave, the bank may need to use available cash, sell assets, or obtain replacement funding. Those choices can carry costs.

**That makes deposit changes worth understanding.** A person examining the bank would also consider its available funding, assets, customers, and institutional history.

**So here is the question. Can bank reports help that person decide where to look first?**

### A made-up USD 100 bank

For one teaching example, give the bank **USD 100 in assets**. It has **USD 15 in cash**. Borrowers owe it **USD 85 in loans**, which they will repay over time.

Now, in this invented example, a customer asks to withdraw **USD 20**.

**How much extra cash does this example need?** **USD 5.** The first visual separates money available today from loans repaid later. These numbers are not a real bank, a recorded withdrawal, or a forecast.

Keep the same made-up bank for one more step. It owes customers USD 90 in deposits, so USD 10 is equity. That explains the balance-sheet difference. It does not add USD 10 to the cash pile.

The example isolates cash availability. Lending can itself create deposits; the [Bank of England explains that mechanism](https://www.bankofengland.co.uk/quarterly-bulletin/2014/q1/money-creation-in-the-modern-economy). The [Federal Reserve’s funding study](https://www.federalreserve.gov/econres/notes/feds-notes/assessing-bank-resilience-to-a-funding-shock-20260217.html) explains how replacement funding can raise costs in a modeled funding shock. Neither source evaluates this project's forecasts.

**The answer from the original experiment:** assume time to examine **10 of every 100 banks**. Random selection would find about **one** bank whose next-quarter deposit growth finishes in the lowest 10%. Ridge and the neural network found about **two**, averaged across three historical quarters of 2024. A human would investigate the movement and decide whether it needs attention. The 10% examination capacity is hypothetical; no operational savings were measured.

**Project in one minute.** The code learns from 214,425 earlier bank-quarter examples. Predicting zero growth has the lowest average absolute error among the original four models; the network has the lowest RMSE, which gives larger mistakes more weight. The review-list comparison asks a separate question: how many low-growth outcomes enter a fixed-size list? The expanded follow-up comparison gives the seasonal-median rule the lowest MAE: **3.500 percentage points**, versus **3.601** for zero growth. This rule was added after 2024 had been examined. Ridge’s review precision exceeds size-only Ridge in the conditional bank-resampling interval; the MLP–Ridge interval includes either ordering. Detailed checks follow the original conclusion.

### Twelve years of reports have three different jobs

| Job | Predictor reports | Bank-quarter examples |
|---|---|---:|
| Learn patterns | Q2 2013–Q3 2022: **38 quarters** | **214,425** |
| Choose settings | Q1–Q3 2023: **3 quarters** | **13,834** |
| Measure historical performance | Q1–Q3 2024: **3 quarters** | **13,532** |

The source contains **48 quarters of reports, 2013–2024**. Most earlier reports have a next-quarter answer, subject to bank-specific gaps. We assign earlier periods to learning and model selection. The previous-quarter input starts the usable training examples in June 2013.

**Why three evaluation quarters?** March 2024 predicts June; June predicts September; September predicts December. A December 2024 forecast would need March 2025, outside this dataset. December 2022 and December 2023 predictor rows are omitted so their outcomes do not cross into the following stage.

**How fresh is the evaluation?** The core models fit earlier data and use 2023 to choose settings. The project has examined 2024 repeatedly. Later comparisons were developed with those results known, so we call this a **reused historical evaluation period**. Thousands of bank examples share three evaluation dates. A later untouched period would test performance under new conditions.

**Tools:** Python, pandas, scikit-learn, TensorFlow, Plotly, Jupyter, and uv. The foundation-model extensions use PyTorch and MLX.
"""

FOLLOWUP_PROSE = """**Three extra baselines test whether the original comparison overlooked simple information.** Predict the middle training growth, predict the middle growth for the same calendar quarter, and fit Ridge using deposit size alone. Each uses earlier data for fitting. These follow-up analyses were developed after examining 2024.

The original result remains identifiable. The expanded scorecard answers whether those additional rules change the comparison on the same historical rows. Calendar patterns motivate a seasonal rule; they do not by themselves establish dependable forecasting skill.

**What does 21.9% precision mean?** About 22 out of every 100 selected banks later belonged to the lowest-growth group. Lift divides that share by the share expected from random selection. The exact random share is the rounded bottom-group count divided by the quarter's bank count, approximately 10%.

**How certain is a small difference?** We repeatedly resample banks, carrying their quarterly records together. We recalculate errors and precision on the existing forecasts and lists. The intervals describe variation across those banks, conditional on the fitted models and three 2024 dates. New periods and retraining bring additional uncertainty.
"""

EVENT_PROSE = """**The September-to-December 2023 Plus International jump spans a verified merger.** New York’s regulator records **October 1, 2023** as the effective date of Emigrant Bank merging into Plus International, under the Emigrant name. The bulletin was published in November 2024, so this source supports retrospective annotation. [Effective-date record](https://www.dfs.ny.gov/reports-and-publications/weekly-bulletins/wb20241122).

Silvergate announced its wind-down on **March 8, 2023** and reported fewer than **$10,000** of remaining deposit liabilities on **November 22, 2023**. These dates define our bounded wind-down sensitivity window. [March announcement](https://www.sec.gov/Archives/edgar/data/1312109/000131210923000058/ex991sipressrelease3x8x23.htm), [November repayment announcement](https://www.sec.gov/Archives/edgar/data/1312109/000095015723001160/ex99-1.htm).

Keep all eligible rows in the primary scores. Then omit intervals overlapping these two verified cases and recalculate using unchanged forecasts. This is a sensitivity to a small documented event list, with no claim that all mergers or closures have been identified. The reported balance change can combine depositor activity and institutional restructuring.
"""

CONTEXT_PROSE = """**A selected bank gives a human a place to begin asking questions.** Examine deposit composition, uninsured concentration, wholesale funding, available assets, borrowing capacity, pricing, and institution history. The current model uses five balance-sheet and deposit-history features. Its target is reported next-quarter deposit growth.

The [FFIEC’s UBPR](https://cdr.ffiec.gov/public/HelpFiles/FAQ.htm) compares a bank with its own history and peer banks. A future experiment could compare growth within documented historical peer groups. That changes the question and may change the size signal. We have not measured the benefit of that alternative. Peer definitions must match the historical period; current rules should not be silently applied to old observations.

**What would make the next evaluation stronger?** Freeze the candidate models, event rules, and selection capacity, then evaluate outcomes from a later period that has not guided development. Account for when reports actually became available; this dataset contains accounting dates and lacks historical publication vintages.

The agencies’ [April 2026 SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm) replaced SR 11-7. Its benchmarking and outcome-analysis principles provide context; the Fed states it is most relevant to organizations above $30 billion. This class project makes no compliance claim.
"""


def visible_source(name):
    return (ROOT / "notebooks/source" / name).read_text()


def teach_story(sections):
    title, _, setup, advanced = sections[0]
    setup = setup[: setup.index("# %% NOTEBOOK CELL\npreview_card(")]
    sections[0] = (
        "Can bank reports help us decide which banks to examine first?",
        "",
        setup + visible_source("banking_opening.py").replace("# Read the saved original run", visible_source("real_bank_bridge.py") + "\n# %% NOTEBOOK CELL\n# Read the saved original run"),
        advanced,
    )
    # Preserve the original experiment and append clearly labelled new checks.
    conclusion_index = next(i for i, item in enumerate(sections) if item[0].startswith("14 ·"))
    sections.insert(
        conclusion_index + 1,
        (
            "Follow-up · Do simpler forecasts change the answer?",
            FOLLOWUP_PROSE,
            visible_source("review_followups.py"),
            False,
        ),
    )
    sections.insert(
        conclusion_index + 2,
        (
            "Follow-up · What do the merger and liquidation change?",
            EVENT_PROSE,
            visible_source("merger_timeline.py") + "\n# %% NOTEBOOK CELL\n" + visible_source("structural_sensitivity.py"),
            False,
        ),
    )
    sections.insert(
        conclusion_index + 3,
        (
            "Next decision · What would a person investigate?",
            CONTEXT_PROSE,
            "",
            False,
        ),
    )
    sections.insert(conclusion_index + 2, (
        "Follow-up · But wait. What if we train on only the recent years?",
        "In volatile economic times, older examples might be less useful. So I tried data starting in 2020, then 2021. "
        "Same model settings. Same 2023 validation. Same 2024 evaluation. And... the errors got bigger. "
        "This is an exploratory follow-up after examining 2024, not a fresh test of 2026 conditions.",
        visible_source("training_window_lesson.py"), False,
    ))
    adapted_index = next(i for i, item in enumerate(sections) if item[0].startswith("Bonus · Does"))
    sections.insert(
        adapted_index + 1,
        (
            "Bonus · Does a short history change the foundation-model comparison?",
            "**One short-history case dominates the TimesFM error.** We group the same banks by available history and score every model on identical rows within each group. These bands are a descriptive follow-up. Bank characteristics can vary with history length, so the comparison alone cannot identify the cause.",
            visible_source("history_length_check.py"),
            True,
        ),
    )
    event_index = next(i for i, item in enumerate(sections) if item[0].startswith("Follow-up · What"))
    sections.insert(event_index, (
        "Follow-up · What if we can examine 5%, 10%, or 20%?",
        "**Imagine your team has half as much review time, or twice as much.** We change the number selected while keeping the desired outcome group fixed at the lowest-growth 10%. Precision measures the concentration in your list; recall measures how much of the entire target group your list reaches. This is a descriptive follow-up using the reused 2024 period.",
        visible_source("capacity_check.py"), False,
    ))
    sections.append((
        "Next trial · Should the model change next quarter?",
        """**First save a forecast before its answer is available.** That is the next evidence we need. A proposed shadow-mode trial would record lists while people continue their existing decision process.

1. **Freeze the rules:** model versions, earlier-data-fitted preprocessing, competing forecasts, review capacity, and event handling.
2. **Wait for reports to become available:** store publication/availability dates and the scoring timestamp, alongside accounting dates. This historical extract does not supply complete publication vintages.
3. **Count who can receive a forecast:** reports available, in-scope banks, usable features/history, scored banks, and reasons for exclusion. A future outcome must never determine who gets scored today.
4. **Save predictions and lists:** retain zero growth, training median, seasonal median, persistence, size-only Ridge, Ridge, and MLP as permanent forecast comparisons. Constants within a quarter provide no ranking signal.
5. **Wait for outcomes:** separately count which issued forecasts can now be evaluated and which remain unresolved. Measure forecast errors, review precision/recall/lift, and paired differences on matching populations.
6. **Investigate changes before retraining:** did coverage fall, inputs shift, errors grow, or a simpler method catch up? Did a merger explain a large miss? One unusual quarter and a persistent change call for different investigations.

**Should we retrain automatically?** New data alone supplies no evidence that replacing the model will help. Predefine review triggers, investigate what changed, and validate a proposed improvement before replacing the frozen system. The project has designed this next trial; it has not deployed or measured operational savings from it.""",
        "", False,
    ))
    return sections
