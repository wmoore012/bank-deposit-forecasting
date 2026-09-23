"""Build the visible-code Study 2 notebook from the shared implementation."""

import inspect
from pathlib import Path
import nbformat
import review_history

ROOT = Path(__file__).resolve().parent


def build():
    cells = []

    def markdown(text):
        cells.append(nbformat.v4.new_markdown_cell(text.strip()))

    def code(text):
        cells.append(nbformat.v4.new_code_cell(text.strip()))

    markdown("""# We can review 10% of the banks. Would looking at their recent history help us choose?

Study 1 ranked banks by predicted next-quarter deposit growth. Here we keep those forecasts and ask whether eight quarters of the same five inputs give us useful additional information.

First, compare the lists. Then look at three actual banks. Finally, check how many banks in each list went on to have growth in the lowest 10% of their quarter.

**This is a retrospective comparison of three already-inspected 2024 quarters.** Report publication dates and historical revisions are not reconstructed. The assumed review budget is 10%; operational benefit has not been measured.""")
    code("""# EXEMPLAR: bootstrap
import os
from pathlib import Path
import textwrap

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mpl-cache"))
import json
from itertools import combinations
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from IPython.display import display, HTML
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
from wm_notecards import WMTheme, init_notebook
from wm_notecards.cards import preview_card, takeaway_card
from wm_notecards.charts import style_fig_wm, wm_render_figure_card
from wm_notecards.tables import wm_render_styler
from deposit_experiment import FEATURES
from experiments.review_history.run import run_study

ROOT = Path.cwd()
OUT = ROOT / "experiments/review_history/outputs"
assert (ROOT / "growth_outputs/frozen_forecasts/manifest.json").is_file()
tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.experimental.enable_op_determinism()
theme = WMTheme(width=860, height=480, card_bg="#FFFFFF", plot_bg="#FFFFFF")
init_notebook()
display(HTML("<script>" + get_plotlyjs() + "</script>"))
display(
    HTML(
        "<style>.jp-RenderedMarkdown{max-width:940px;margin:auto;line-height:1.65}.jp-OutputArea-output{overflow-x:auto}</style>"
    )
)


# Adapt the existing notebook table and chart helpers.
def table(frame, title, formats=None):
    styled = frame.style.hide(axis="index").format(formats or {}, na_rep="Unavailable")
    wm_render_styler(styled, theme=theme, title=title)


def chart(figure, name, title, subtitle, height=480):
    style_fig_wm(figure, title=title, subtitle=subtitle, theme=theme, category_policy="preserve")
    title_lines = textwrap.wrap(title, 55)
    subtitle_lines = textwrap.wrap(subtitle, 105)
    heading = (
        "<b>"
        + "<br>".join(title_lines)
        + '</b><br><span style="font-size:14px">'
        + "<br>".join(subtitle_lines)
        + "</span>"
    )
    figure.update_layout(
        height=height,
        width=860,
        font=dict(family="Arial", size=13, color="#222222"),
        title=dict(
            text=heading,
            x=0.025,
            xanchor="left",
            y=0.975,
            yanchor="top",
            pad=dict(t=0),
            font=dict(size=24, family="Arial"),
        ),
        margin=dict(l=100, r=40, t=60 + 30 * len(title_lines) + 20 * len(subtitle_lines), b=110),
        legend=dict(font=dict(size=12), title_text="", orientation="h", y=-0.18, x=0),
    )
    figure.update_xaxes(
        tickfont=dict(size=12, family="Arial"),
        title_font=dict(size=13, family="Arial"),
        automargin=True,
    )
    figure.update_yaxes(
        tickfont=dict(size=12, family="Arial"),
        title_font=dict(size=13, family="Arial"),
        automargin=True,
    )
    figure.update_annotations(font=dict(size=12, family="Arial"))
    figure.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
    display(
        HTML(
            figure.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={"responsive": True, "displaylogo": False},
            )
        )
    )
""")
    markdown("""## The same banks, the same review budget

The original forecasts used five current-quarter inputs. Each anomaly detector receives eight quarters of those inputs. Eight growth calculations require nine underlying deposit reports. Undefined values stay unavailable.

We preserve the original lists, then build separate matched-population lists. Future outcomes do not decide who can enter anomaly training.""")
    for name in ["feature_histories", "standardize_histories"]:
        code(inspect.getsource(getattr(review_history, name)))
    markdown("""## Start with the simple rules

Ridge and the MLP select the lowest forecast growth. The equity rule selects the lowest current equity/assets ratio. A size-only control selects the largest deposit balances.

PCA learns a linear reconstruction of the histories. Isolation Forest scores histories that are easier to isolate with random splits. The dense autoencoder learns a nonlinear reconstruction. Larger reconstruction errors receive higher priority for PCA and the autoencoder.

The autoencoder uses a flattened history: quarter slots run oldest to newest, and each slot contains the five features in the listed order. It has no recurrence or attention. Its settings are fixed before evaluation.""")
    for name in [
        "reconstruction_error",
        "fit_pca",
        "fit_isolation_forest",
        "build_autoencoder",
        "fit_autoencoder",
    ]:
        code(inspect.getsource(getattr(review_history, name)))
    markdown("""## Fit the declared methods, then freeze their scores

The runner uses the definitions shown above and ordinary imports. It saves the eligibility audit, fitted models, training histories, and a score fingerprint before evaluating deposit outcomes. Seed 42 is primary; 7 and 99 remain in every receipt. The same methods also run without `log_deposits`.

Training history endpoints stop in September 2022. Validation endpoints are Q1–Q3 of 2023. Evaluation endpoints are Q1–Q3 of 2024. The runner source is included in the package at `experiments/review_history/run.py`.""")
    code("""if os.environ.get("REVIEW_HISTORY_REFIT", "1") == "1":
    run_study(ROOT, OUT)
plan = json.loads((OUT / "plan.json").read_text())
coverage = pd.read_csv(OUT / "coverage.csv")
# EXEMPLAR: missingness-evidence
counts = pd.DataFrame(
    {
        "Population": list(plan["population_counts"]),
        "Bank-quarter rows": list(plan["population_counts"].values()),
    }
)
table(counts, "The original sample and the two complete-history comparisons")
fig = px.bar(
    coverage.loc[coverage.exclusion_reason.ne("Eligible")],
    x="date",
    y="rows",
    color="exclusion_reason",
    text="rows",
)
fig.update_xaxes(
    type="category",
    title="Report quarter",
    tickvals=coverage.date.unique(),
    ticktext=pd.to_datetime(coverage.date.unique()).strftime("%b %Y"),
)
fig.update_yaxes(title="Excluded bank-quarter rows")
chart(
    fig,
    "history_coverage",
    "Some forecast rows lack a complete financial history",
    "Exclusions stay in the audit. They do not receive invented anomaly scores.",
)
table(coverage, "Every quarter and exclusion reason")
original_audit = pd.read_csv(OUT / "original_list_audit.csv")
unavailable = original_audit.loc[~original_audit.anomaly_score_available]
print(
    "Original selected rows without anomaly scores:",
    unavailable[["Ridge original selected", "MLP original selected"]].sum().to_dict(),
)
""")
    markdown("""## Do the lists agree?

Both methods can select 10% and still choose different banks. Start with Ridge and PCA in March 2024. Each bank belongs in exactly one of four groups.""")
    for name in ["review_flags", "list_overlaps", "choose_examples"]:
        code(inspect.getsource(getattr(review_history, name)))
    code("""# EXEMPLAR: missingness-evidence
selections = pd.read_csv(OUT / "expanded_selections.csv")
march = selections.loc[selections.date.eq("2024-03-31")]
four = pd.DataFrame(
    {
        "Group": ["Both", "Forecast only", "Anomaly only", "Neither"],
        "Banks": [
            int((march.Ridge & march.PCA).sum()),
            int((march.Ridge & ~march.PCA).sum()),
            int((~march.Ridge & march.PCA).sum()),
            int((~march.Ridge & ~march.PCA).sum()),
        ],
    }
)
fig = go.Figure(
    go.Heatmap(
        z=[[four.Banks.iloc[0], four.Banks.iloc[2]], [four.Banks.iloc[1], four.Banks.iloc[3]]],
        x=["Ridge selected", "Ridge not selected"],
        y=["PCA selected", "PCA not selected"],
        text=[
            ["Both: " + str(four.Banks.iloc[0]), "Anomaly only: " + str(four.Banks.iloc[2])],
            ["Forecast only: " + str(four.Banks.iloc[1]), "Neither: " + str(four.Banks.iloc[3])],
        ],
        texttemplate="%{text}",
        colorscale=[[0, "#EFF3F4"], [1, "#3F8290"]],
        showscale=False,
    )
)
fig.update_yaxes(autorange="reversed")
chart(
    fig,
    "four_groups",
    "Ridge and PCA choose different review lists",
    f"March 2024: {len(march):,} common eligible banks. Each list has {int(march.Ridge.sum()):,} places.",
)
table(four, "Each bank appears in exactly one group")
overlap = pd.read_csv(OUT / "expanded_overlap.csv")
primary = [
    "MLP",
    "Low equity/assets",
    "Largest deposits",
    "PCA",
    "Isolation Forest 42",
    "Autoencoder 42",
]
shown = overlap.loc[
    overlap.date.eq("All three quarters") & overlap.left.eq("Ridge") & overlap.right.isin(primary)
]
fig = px.bar(shown, x="left_share", y="right", orientation="h", text="shared")
fig.update_xaxes(title="Share of Ridge selections also chosen", tickformat=".0%", range=[0, 1])
fig.update_yaxes(title="")
chart(
    fig,
    "ridge_overlap",
    "How much of Ridge’s list does each method also select?",
    "Pooled bank-quarter selections across three quarters; seed 42 for stochastic methods.",
)
table(
    shown[["right", "shared", "left_count", "left_share", "jaccard", "random_expected"]],
    "Shared selections and independent-random expectation",
    {"left_share": "{:.1%}", "jaccard": "{:.3f}", "random_expected": "{:.2f}"},
)
""")
    markdown("""## What did each method select?

Choose the shared bank with the highest PCA score, the forecast-only bank with the lowest Ridge forecast, and the anomaly-only bank with the highest PCA score. These rules were fixed before inspecting the bank names. The examples explain these selections; they do not represent every bank in their group.""")
    code("""examples = pd.read_csv(OUT / "examples.csv")
table(examples, "Three examples selected by the declared rules")
histories = pd.read_csv(OUT / "example_histories.csv", parse_dates=["date"])
labels = {
    "DEPDOM": "Deposits · USD million",
    "cash_ratio": "Cash / assets · %",
    "loan_ratio": "Loans / assets · %",
    "equity_ratio": "Equity / assets · %",
    "prior_growth": "Prior-quarter log growth × 100",
}
for example in examples.dropna(subset=["CERT"]).itertuples(index=False):
    bank = histories.loc[histories.CERT.eq(example.CERT)]
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    for position, (feature, label) in enumerate(labels.items(), start=1):
        values = bank[feature] / 1000 if feature == "DEPDOM" else bank[feature] * 100
        fig.add_trace(
            go.Scatter(x=bank.date, y=values, mode="lines+markers", name=label, showlegend=False),
            row=position,
            col=1,
        )
        fig.update_yaxes(title="", row=position, col=1)
        axis = "" if position == 1 else str(position)
        fig.add_annotation(
            x=0,
            y=1,
            xref="x" + axis + " domain",
            yref="y" + axis + " domain",
            text=label,
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            yshift=8,
        )
    chart(
        fig,
        "history_" + str(int(example.CERT)),
        example.group + ": " + example.NAME,
        "Eight observed quarters. Deposits use USD millions; the model receives log deposits.",
        height=1000,
    )
shares = pd.read_csv(OUT / "reconstruction_shares.csv")
case_shares = shares.merge(examples[["CERT", "group"]], on="CERT").loc[
    lambda f: f.date.eq("2024-03-31")
]
long = case_shares.melt(
    id_vars=["group", "method"], value_vars=list(FEATURES), var_name="Input", value_name="Share"
)
fig = px.bar(
    long,
    x="Share",
    y="group",
    color="Input",
    facet_col="method",
    facet_col_spacing=0.09,
    orientation="h",
)
fig.update_xaxes(tickformat=".0%", range=[0, 1], title="")
fig.update_yaxes(title="")
chart(
    fig,
    "reconstruction_shares",
    "Which inputs were harder to reconstruct?",
    "Share of squared reconstruction error across eight quarters. These are not causes of risk.",
    height=500,
)
table(
    case_shares[["group", "method", *FEATURES]],
    "Exact reconstruction-error shares",
    {feature: "{:.1%}" for feature in FEATURES},
)
""")
    markdown("""## Did the models mostly react to deposit size?

Remove `log_deposits` and repeat the learned methods on the same banks and dates. Other inputs can still correlate with size. Here we check dependence on the explicit size feature.""")
    code("""pairs = []
for method in [
    "PCA",
    "Isolation Forest 42",
    "Isolation Forest 7",
    "Isolation Forest 99",
    "Autoencoder 42",
    "Autoencoder 7",
    "Autoencoder 99",
]:
    row = overlap.loc[
        overlap.date.eq("All three quarters")
        & (
            ((overlap.left == method) & (overlap.right == method + " without size"))
            | ((overlap.right == method) & (overlap.left == method + " without size"))
        )
    ].iloc[0]
    pairs.append(
        {
            "Method": method,
            "Shared selections": int(row.shared),
            "Share retained": row.left_share,
            "Jaccard": row.jaccard,
        }
    )
size_check = pd.DataFrame(pairs)
fig = px.bar(size_check, x="Share retained", y="Method", orientation="h", text="Shared selections")
fig.update_xaxes(range=[0, 1], tickformat=".0%")
fig.update_yaxes(title="")
chart(
    fig,
    "size_check",
    "How much of each list survives removing deposit size?",
    "Matched rows, fixed settings, 10% review capacity.",
)
table(
    size_check,
    "Full-input versus no-deposit-size lists",
    {"Share retained": "{:.1%}", "Jaccard": "{:.3f}"},
)
""")
    markdown("""## Does a different seed change who gets reviewed?

Use every pair among seeds 42, 7, and 99. Agreement between seeds concerns repeatability of the list; it does not establish that the list serves the review objective.""")
    code("""stable = []
for representation in ["", " without size"]:
    for family in ["Isolation Forest", "Autoencoder"]:
        methods = [f"{family} {seed}" + representation for seed in [42, 7, 99]]
        part = overlap.loc[
            overlap.date.eq("All three quarters")
            & overlap.left.isin(methods)
            & overlap.right.isin(methods)
        ].copy()
        version = "without size" if representation else "full inputs"
        part["Comparison"] = [
            family
            + " · "
            + version
            + " · "
            + a.split()[1 if family == "Autoencoder" else 2]
            + "/"
            + b.split()[1 if family == "Autoencoder" else 2]
            for a, b in zip(part.left, part.right)
        ]
        stable.append(part)
stability = pd.concat(stable)
fig = px.bar(stability, x="left_share", y="Comparison", orientation="h")
fig.update_xaxes(range=[0, 1], tickformat=".0%", title="Share of selections shared")
fig.update_yaxes(title="")
chart(
    fig,
    "seed_stability",
    "How repeatable are the stochastic review lists?",
    "Every seed pair is retained, including the comparisons without deposit size.",
    height=720,
)
table(
    stability[["left", "right", "shared", "left_count", "left_share", "jaccard"]],
    "Exact seed overlap",
    {"left_share": "{:.1%}", "jaccard": "{:.3f}"},
)
""")
    markdown("""## Which lists captured more of the low-growth outcomes?

Now use the original common-history population with observed outcomes. Recalculate every list and the realized lowest-growth decile on that same population. These are new matched-population results; the original Study 1 lists remain unchanged.

This checks the deposit-growth objective we started with. An unusual history can have other explanations or uses that this outcome does not measure.""")
    code(inspect.getsource(review_history.outcome_capture))
    code("""capture = pd.read_csv(OUT / "outcome_capture.csv")
main_methods = [
    "Ridge",
    "MLP",
    "Low equity/assets",
    "Largest deposits",
    "PCA",
    "Isolation Forest 42",
    "Autoencoder 42",
]
main = capture.loc[
    capture.date.eq("All three quarters") & capture.method.isin(main_methods)
].sort_values("capture_per_selected")
fig = px.bar(main, x="capture_per_selected", y="method", orientation="h", text="captured")
fig.update_traces(
    marker_color=["#0B6F75" if name == "Ridge" else "#8098AA" for name in main.method]
)
fig.update_xaxes(
    title="Lowest-growth banks captured per selected bank", tickformat=".0%", range=[0, 0.27]
)
fig.update_yaxes(title="")
chart(
    fig,
    "outcome_capture",
    "Ridge captured the most low-growth outcomes in this comparison",
    "Same original common-history population, same 10% capacity. Labels count bank-quarter selections.",
)
table(
    main[["method", "captured", "selected", "capture_per_selected"]],
    "Primary methods, pooled selections",
    {"capture_per_selected": "{:.1%}"},
)
table(capture, "Every quarter, seed, and size-removal result", {"capture_per_selected": "{:.1%}"})
# EXEMPLAR: bounded-takeaway
pooled = capture.loc[capture.date.eq("All three quarters")].set_index("method")
ridge = pooled.loc["Ridge"]
auto = pooled.loc["Autoencoder 42"]
pca = pooled.loc["PCA"]
assert ridge.captured == pooled.captured.max()
takeaway_card(
    title="Keep Ridge for the historical deposit-growth review task",
    theme=theme,
    body=f"Ridge captured {int(ridge.captured)} of the lowest-growth bank-quarter outcomes in {int(ridge.selected)} selections. The primary autoencoder captured {int(auto.captured)}; PCA captured {int(pca.captured)}. The history methods produced different lists, but none captured more of this outcome at the same capacity. These three reused quarters do not settle future performance.",
)
""")
    markdown("""## Reproduction and boundaries

The five-feature histories and their no-deposit-size counterparts use the same rows. All fitted models, eligibility reasons, seed results, original-list annotations, and score fingerprints live in `experiments/review_history/outputs/`. The runner saves scores before calculating the observed-outcome comparison.

The raw equity rule originated in earlier failure exploration. No failure labels enter these models or the outcome comparison. Official events are kept separately in the unresolved-outcome coverage audit. They do not supply missing deposit balances.

A historical event study would require its own operational question, publication timing, chronological model fits, and outcome coverage. It is outside this experiment.""")
    code("""fits = pd.read_csv(OUT / "fits.csv")
table(fits, "All fitted models and recorded component or stopping choices")
manifest = json.loads((OUT / "score_manifest.json").read_text())
from freeze_forecasts import fingerprint

assert fingerprint(OUT / "scores.csv") == manifest["scores_sha256"]
assert fingerprint(OUT / "plan.json") == manifest["plan_sha256"]
assert plan["population_counts"]["expanded_complete"] == len(selections)
print("Verified saved score fingerprint and generated population counts.")
""")
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
    )
    nbformat.write(notebook, ROOT / "FDIC_Review_History.ipynb")
    print("Built Study 2:", len(cells), "cells")


if __name__ == "__main__":
    build()
