"""Concrete-first prose copied into generated notebooks; no hidden model calculations."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

LESSONS = {
'1 ·': '''**One row is one bank's report for one quarter.** Imagine finding the same bank in March and June. Those are two examples from one institution. CERT identifies the bank; REPDTE identifies the report date. Deposits, assets, loans, cash, and equity describe that report.

**Would a blank equity entry mean the bank has zero equity?** A blank means this extract gives us no number. A recorded zero supplies a number whose meaning still needs checking against the reporting definition. The 815 EQ blanks all fall outside our chosen domestic-bank/reporting-form population. We preserve the source blanks and show that exclusion beside the pandas output.

**Why begin in 2013?** We chose 2013–2024 because it follows the 2012 reporting conversion. Matching column names across older forms do not establish matching definitions. The older reports remain in the detailed audit. Read the column dictionary, a few actual rows, types, missing counts, and summaries before using a ratio.''',
'2 ·': '''**A historical forecast needs a before, a now, and an after.** Imagine June is the report we use. March supplies previous growth; September supplies the answer we later score. A gap in either neighboring report prevents this particular historical example from being constructed.

**Does a missing September answer stop us forecasting in June?** We could still issue a forecast using information available in June. The September requirement belongs to this historical evaluation population. For a future trial, record scorable banks first and track missing outcomes later. The exclusion receipt counts what happened; disappearance alone supplies no merger or failure diagnosis.''',
'3 ·': '''**The time plots ask whether deposit declines recur at similar points in the year.** Imagine March reports repeatedly precede more declines than December reports. A model might benefit from knowing the calendar, so we first examine whether that pattern recurs across individual years.

Read left to right through training dates. In the Q1–Q4 view, compare the separate year markers as well as the summary. A high rate means more bank-quarter examples had a negative next-quarter change. The seasonal forecast later in the notebook is a follow-up experiment developed after inspecting 2024.''',
'4 ·': '''**The models receive five clues about each bank.** Imagine assets of USD 100 and cash of USD 15: cash/assets is 15%. Repeat that division for loans and equity. Deposit size and previous deposit growth complete the five inputs.

**Would one ratio cleanly separate banks whose deposits later fall?** Compare the two cash-ratio boxes. Their middle halves overlap substantially. Next, inspect every input's distribution: the median shows its center, the spread shows how different banks can be, and the tails show the unusual reports a mean can conceal.

**Are these five copies of the same clue?** The correlation plot measures pairwise rank relationships. Modest correlations suggest limited pairwise repetition; they do not establish independence or rule out nonlinear relationships. The prior-versus-next plot then asks whether simply repeating the previous change has useful signal.''',
'5 ·': '''**Counting dollars gives large banks enormous influence.** Imagine one bank loses 1% of USD 100 billion and another loses 10% of USD 100 million. The first contributes USD 1 billion to a dollar-loss score; the second contributes USD 10 million.

The archived experiment's size-only rule nearly matched the network's dollar capture. That motivates asking about proportional growth. We retain the original calculation so readers can see why the question changed.''',
'6 ·': '''**Start with USD 100 million in deposits. Next quarter there is USD 90 million.** The dollar change is minus USD 10 million. Divide by the starting USD 100 million: ordinary growth is minus 10%.

**How can the same calculation produce 522,646%?** The real training example starts at USD 1.15 million and ends near USD 6.01 billion. The small starting balance makes the ratio enormous. Its arithmetic is valid; the economic cause needs separate evidence.

The model learns **y = log(next deposits / current deposits)**. Here, log means the natural logarithm. For 100 to 90, log(0.9) is about −0.105. To interpret a prediction, use **100 × (exp(y_hat) − 1)**. Exp reverses the logarithm; subtracting one gives growth; multiplying by 100 expresses it as a percentage. We score the resulting forecasts in percentage points. Back-transforming a mean log forecast generally differs from estimating mean ordinary growth.''',
'7 ·': '''**A December prediction needs the following March to reveal its answer.** If we train through December 2022 predictors, those answers would spill into 2023. We stop training predictors in September 2022, whose December outcomes remain inside the training stage. The December 2023 predictor is omitted for the same boundary reason.

The timeline separates 38 training predictor quarters from three validation and three historical evaluation quarters. Earlier reports generally also have answers. Those answers serve learning or validation. The 2024 outcomes have been examined repeatedly during development, so the evaluation is reused.''',
'8 ·': '''**Imagine a forecast is too high. Which number should the network change?** A weight controls how strongly an input affects the prediction. The small worked example calculates an error, adjusts a weight, and checks whether the error shrinks.

The full network repeats that kind of adjustment across 737 parameters. Its five inputs feed 32 hidden units, then 16, then one log-growth prediction. A successful training update tells us the optimization worked. Later observations test whether what it learned travels through time.''',
'9 ·': '''**Give every bank 0% growth. You have built the first forecast.** It has no weights to fit. Persistence copies the previous quarter's growth. Ridge learns a regularized linear relationship from the five inputs. The MLP learns weights in its two hidden layers.

**How do we decide when to stop changing a model?** Use the later validation period. Ridge chooses alpha by validation MAE in ordinary percentage points. The original MLP chooses its stopping point by validation mean squared error in log growth, restoring the best weights. These are different selection rules, and we keep that history visible. Imputation and scaling learn their parameters from training rows only.''',
'10 ·': '''**A forecast can make smaller average misses and still make a worse large miss.** Work the three-error example below before reading the real scores. MAE averages the absolute misses. RMSE squares each miss first, giving a large error much more influence, then takes a square root.

Among the original four forecasts, zero growth has the lowest MAE and the MLP has the lowest RMSE. The follow-up seasonal median later lowers historical MAE further. The error plots show where the original RMSE difference comes from; a metric label alone cannot tell us which individual forecasts improved.''',
'11 ·': '''**Imagine two models have similar average error. Do they miss the same banks?** The actual-versus-predicted plot shows where each forecast sits relative to the correct-answer diagonal. The quarter and tail views then separate dates and realized outcomes.

Groups defined by what happened next help explain mistakes afterward. They could not have been used to choose banks when the forecasts were issued. A lower average can coexist with serious misses in particular cases.''',
'12 ·': '''**Bank A: 0%. Bank B: 0%. Bank C: 0%. Which goes first?** Those forecasts provide no ordering. Zero growth remains useful for testing numerical forecast accuracy.

Now sort banks from lowest to highest predicted growth and select the first 10%. After the next quarter, count how many selected banks actually belong to that quarter's lowest-growth 10%. **Precision** is that count divided by the number selected. **Random selection** answers how many we would expect with no informative ordering. These are two different baseline jobs.

In March, Ridge selected 454 banks and found 112 of the realized lowest-growth group; the network found 109. The random expectation is 45.44. Lower growth describes a relative deposit outcome. A person must investigate its reason.''',
'13 ·': '''**Imagine a bank reports an unusual cash ratio. Is that enough to predict next quarter's deposit change?** An anomaly detector can identify a report that differs from familiar reports. Its usefulness for finding future low-growth outcomes still needs evaluation.

A later experiment could compare an anomaly-based list with the forecast-based list at the same capacity and on the same available banks. This project has not measured that anomaly detector's performance. An unusual report provides a reason to inspect the source and context.''',
'14 ·': '''**The original network reduced RMSE while zero growth retained lower MAE.** That tells us the network's extra complexity helped one error criterion in this historical sample. It did not establish a clear advantage across the jobs we tested.

**Did sorting help us choose cases to examine?** Ridge and the MLP placed more eventual low-growth outcomes in the assumed 10% list than random selection in each of three quarters. The follow-up comparisons now ask how much size alone explains, how uncertainty affects the difference, and whether simple seasonal forecasts change the error comparison.''',
'Deeper check ·': '''**Two reports can use the same column name while meaning different things.** Imagine comparing a ratio collected under two reporting forms. First check how each form defines the fields. The older-report tables keep that unresolved comparability issue visible, alongside missing equity and report coverage.''',
'Deeper lesson · Does': '''**Start the same network with different weights. Will it finish in the same place?** These three fixed seeds show how the training path changes validation error. We retain seed 42 for the original fit rather than choosing whichever seed looks best on 2024.''',
'Deeper lesson · How': '''**Would a different collection of banks change the small MLP–Ridge gap?** Resample entire banks, carrying each bank's quarterly records together. The resulting interval includes either ordering. It describes these fixed forecasts and historical dates; future economic periods add uncertainty this resampling cannot measure.''',
'Deeper lesson · Can': '''**An API field is a named column returned by the data service.** Imagine asking for uninsured deposits and receiving a blank, zero, or positive value. A filled cell tells us a value arrived. It does not establish that every bank had the same reporting obligation.

The coverage chart asks how often a value is present. The zero chart asks how often recorded values equal zero, with its denominator stated beside the calculation. Follow the smaller-bank series through the reporting changes. We keep these fields outside the five-feature comparison while their historical definitions and eligibility remain unresolved.''',
'Bonus · Can': '''**Imagine giving a pretrained forecaster only a bank's deposit history.** Its weights already learned from other time series. In a zero-shot test, we keep those weights fixed and ask for the next value. TimesFM and Chronos receive numeric histories, while the original Ridge and MLP receive five financial features.

We test all eligible evaluation rows and keep the predictions. The scores below assess those particular zero-shot runs. Pretraining overlap with this history is unknown. Poor performance does not establish that the model has never encountered similar patterns.''',
'Bonus · Does a short': '''**Imagine forecasting from three observations, then from forty.** More history could reveal a different pattern, but banks with short histories may also be different institutions. We compare every available model on exactly the same rows in each history band.

One short-history TimesFM case dominates its squared error. That observation motivates this check. A controlled experiment truncating the same banks' histories would be needed to isolate the effect of context length.''',
'Bonus · Does': '''**Could these pretrained forecasts improve after learning from our earlier bank data?** We updated each model's forecast head while keeping its backbone frozen. The notebook shows the adaptation budget, training dates, changed weights, and validation-selected checkpoint.

This was one bounded adaptation recipe. Validation selected the checkpoint before scoring 2024; the broader adaptation experiment was developed after that historical period had already been inspected. Compare zero-shot and adapted results separately. Neither adapted run beat the best core baselines.''',
'Appendix · Rebuild': '''**If you mostly select large banks, how many decline dollars can you capture?** Rebuild the archived experiment and compare the network with the size-only rule. The near tie explains why the main experiment moved to proportional growth. These results use the archived experiment's own population and outcome.''',
'Appendix · Keep': '''**What exactly does a large capture percentage count?** The archived worked examples separate dollar volume, banks selected, and predicted probabilities. Keep their denominators visible when interpreting the earlier experiment.''',
}

GUIDES = {
'cash_balance_sheet':'Read the entire block as USD 100 of assets. The blue USD 15 is cash available now; the other USD 85 represents loans that borrowers will repay over time.',
'withdrawal_shortfall':'Read the bar as the USD 20 request. Blue supplies USD 15. Amber marks the remaining USD 5 the bank must obtain.',
'opening_review':'Each bar counts selected banks that later fell in the lowest-growth group. All methods have 454 March selections. The random bar is an expectation, so its count can be fractional.',
'history_roles':'Read left to right through predictor dates. The long earlier interval is training. The two short later intervals choose settings and measure reused historical performance; their labels count bank-quarter examples.',
'reporting_banks_time':'The horizontal axis is the reporting quarter; height counts banks with a source report. A falling count changes the population represented by later summaries. It does not identify why individual banks disappear.',
'median_deposits_time':'The horizontal axis is report date. Height is the middle reporting bank’s deposits in USD millions, not the total banking system. Changing membership can also change this median.',
'next_report':'Each category describes the observable next-report situation. A dataset boundary and an institution’s final observed report are different facts. Neither category alone supplies an economic cause.',
'decline_share_time':'Each point is a training predictor quarter. Height is the share of eligible bank-quarter examples whose deposits decline next quarter. Compare the recurring rhythm with periods that differ.',
 'training_seasonality':'Q1–Q4 refer to predictor quarters. Each year has separate markers; the summary describes their pattern. Compare years before treating a quarter-of-year difference as stable.',
'cash_by_outcome_box':'Compare cash/assets before the next outcome. The line inside each box is the median; the box covers the middle half. The groups overlap. The displayed window is zoomed; the quartiles use all training rows.',
'feature_correlations':'Each square compares two training inputs using Spearman rank correlation. Values near zero show little monotonic pairwise association, not proof of independence. Color strength shows relationship size.',
'persistence_eda':'Compare previous growth on the horizontal axis with next growth vertically. The training points show how far the next change can depart from simply repeating the last one. Read the displayed window before judging the tails.',
 'target_growth_side_by_side':'Both panels use the same training bank-quarter population. Height counts observations. Each horizontal scale shows its own central 98% window; the maximum labels retain the full-range extremes.',
'small_denominator_growth':'Move right for larger starting deposits and up for larger positive percentage growth. Both axes are logarithmic. Labels identify the largest cases; the scatter explains the ratio, while institutional causes need separate evidence.',
'split':'Predictor dates provide inputs; following-quarter dates provide answers. Read the gaps between stages and the number of predictor quarters. The reused 2024 rows were excluded from fitting the original models.',
'architecture':'Follow five inputs through the 32-unit and 16-unit hidden layers to one output. That output is log growth; the later conversion expresses the forecast as ordinary percentage growth.',
'validation':'Compare candidate Ridge settings using validation error. Smaller MAE chooses alpha. This is the settings-selection period, separate from the reused 2024 score.',
'learning':'Move right through training epochs. Compare training and validation loss in log-growth space. The restored checkpoint comes from the best validation epoch, not automatically the final epoch.',
'comparison':'Read each panel horizontally: farther left means smaller error. MAE averages absolute percentage-point misses. RMSE puts more weight on large misses. These are the original four models on identical 2024 rows.',
'error_tradeoff':'Compare absolute-error quantiles on identical 2024 rows. Higher quantiles describe increasingly large misses. This helps locate the MLP’s RMSE advantage instead of assuming every forecast improved.',
'actual_predicted':'A point on the diagonal would have the correct predicted growth. Distance from it is a miss. Both axes use matching display windows; the scores still include every eligible evaluation row.',
'quarter_errors':'Compare models within each predictor quarter. The three dates are shared historical conditions, so thousands of bank rows do not amount to thousands of independent economic periods.',
'tail_errors':'Compare errors after grouping banks by realized growth. Those outcome groups are known afterward. This chart diagnoses mistakes and cannot show what we knew before the quarter ended.',
'ranking':'Each model has one marker per predictor quarter. Farther right means more selected banks later belonged to the lowest-growth group. Hover shows hits and selections. The reference marks roughly 10% random precision.',
'followup_errors':'Each dot is MAE on the same reused-2024 rows. Teal identifies the additional earlier-data-fitted rules. Seasonal median has the lowest observed MAE in this expanded comparison.',
'review_lift':'A lift of 2 means twice the random concentration of the intended outcomes. Compare individual quarters and their equal-quarter means. The exact reference uses the rounded bottom-group prevalence.',
'paired_uncertainty':'A difference of zero means equal scores. The interval for MLP versus Ridge includes both orderings. Ridge minus size-only precision stays positive in this fixed-list bank-cluster resampling. Future quarters remain untested.',
'structural_sensitivity':'Compare unchanged forecasts on all eligible rows with the subset omitting the documented event intervals. The event list flags validation rows and no 2024 rows. Its incomplete coverage limits the interpretation.',
'foundation_models':'Compare ordinary-growth errors on matching evaluation rows. The original feature models and pretrained history models receive different inputs. The large TimesFM miss remains in the full score.',
'finetuning_validation':'The horizontal axis counts adaptation steps. Validation chooses among saved checkpoints using its stated log-balance criterion. The backbone stays frozen while the head changes.',
'adaptation_comparison':'Compare each frozen-weight forecast with its adapted-head forecast on the same rows. Improving the adaptation criterion did not guarantee lower 2024 ordinary-growth error.',
'history_length':'Within each history-length band, every model is scored on identical rows. Read the sample count: the shortest band is small. The logarithmic error scale keeps a very large miss visible.',
'missingness':'These counts describe missing source entries in the broader audit population. Use the reporting-form receipt to see which groups contain the blanks before deciding how they affect the main population.',
'coverage_time':'The height is the share of audit reports with missing equity at each date. The denominator is all reports in that audit quarter. Compare reporting forms before attributing a change to bank finances.',
'seeds':'Each point uses the same data and architecture with different initial weights. The horizontal error comes from validation. The original seed remains fixed for the historical comparison.',
'bootstrap':'Each draw resamples whole banks and recalculates the MLP-minus-Ridge MAE difference. Negative values favor the MLP; positive values favor Ridge. This conditional distribution crosses zero.',
'uninsured_states':'Read the populated, zero, and nonzero states using the denominator defined in the adjacent code. A zero can require reporting-context investigation. These fields are excluded from the core model.',
'experiment0':'Compare the archived model’s captured decline dollars with the size-only rule at the same capacity. A large dollar share can arise from choosing large institutions.',
'old_capture':'This archived calculation uses its own population and dollar outcome. Compare size-only with the learned forecast before attributing captured dollars to nonlinear learning.',
}

GUIDES.update({
    'real_bank_assets': 'The entire bar is the bank’s reported assets. Blue marks cash and balances due; the other segments show net loans and the remaining assets. The exact receipt below supplies values for the small cash segment.',
    'capacity_check': 'Each panel changes review capacity from 5% to 20% while holding the lowest-growth outcome group at 10%. Lines are equal-quarter means. Precision describes the list, recall describes the target group, and lift compares concentration with random selection.',
    'merger_timeline': 'The blue endpoints are reported domestic-deposit balances. The amber diamond marks the official effective merger date, one day after the first report. Actual calendar spacing keeps the event close to September 30.',
})

def apply_concrete_teaching(sections):
    for i, (title, prose, source, advanced) in enumerate(sections):
        match = next((k for k in sorted(LESSONS, key=len, reverse=True) if title.startswith(k)), None)
        if match:
            prose = LESSONS[match]
        if title.startswith('10 ·'):
            source = (ROOT/'notebooks/source/metric_lesson.py').read_text() + '\n# %% NOTEBOOK CELL\n' + source
        if title.startswith('1 ·'):
            # Place the concrete distinction before the source's missingness audit.
            source = source.replace('post_conversion.isna().sum()', 'post_conversion.isna().sum()')
        sections[i] = title, prose, source, advanced
    return sections
