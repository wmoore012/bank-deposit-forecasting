"""Export twelve numbered, source-backed images and one LinkedIn-ready PDF.

Run after executing the notebooks: uv run python export_story.py
Each invocation creates a new version. Figures use the notebook's saved evidence.
"""

from pathlib import Path
from datetime import datetime
import hashlib
import json
import textwrap

import os
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "growth_outputs/masterclass"
DEST = ROOT / "exports" / ("deposit-story-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
DEST.mkdir(parents=True, exist_ok=False)
TEAL, BLUE, GOLD, GRAY, INK = "#0B6F75", "#3F6294", "#A86223", "#C5CDD6", "#172F3E"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "text.parse_math": False,
        "font.size": 14,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
pages = []
manifest = {"created": datetime.now().isoformat(), "size_px": [1080, 1350], "pages": []}


def read(name):
    return pd.read_csv(OUT / name)


def page(number, title, subtitle, note, sources):
    fig = plt.figure(figsize=(9, 11.25), facecolor="#FFFFFF")
    fig.text(
        0.075,
        0.954,
        f"BANK DEPOSIT FORECASTING   /   {number:02d}",
        fontsize=10,
        color=TEAL,
        weight="bold",
    )
    fig.text(
        0.075,
        0.905,
        textwrap.fill(title, 33),
        fontsize=29,
        weight="bold",
        va="top",
        linespacing=1.12,
    )
    fig.text(0.075, 0.765, textwrap.fill(subtitle, 67), fontsize=13, va="top", linespacing=1.4)
    fig.text(0.075, 0.17, textwrap.fill(note, 65), fontsize=13, va="top", linespacing=1.5)
    fig.text(0.075, 0.045, "wmoore012 / bank-deposit-forecasting", fontsize=10, color="#576B78")
    fig.text(0.925, 0.045, f"{number:02d} / 12", ha="right", fontsize=10, color="#576B78")
    paths = [ROOT / s for s in sources]
    manifest["pages"].append(
        {
            "number": number,
            "notebook": "FDIC_Deep_Learning_Masterclass.ipynb",
            "section": {1: "Can bank reports help us decide which banks to examine first?", 2: "Can bank reports help us decide which banks to examine first?", 3: "7 · Did a future outcome enter training?", 4: "4 · What do the five inputs look like?", 5: "6 · What should the model predict?", 6: "10 · Which forecast errs least?", 7: "Follow-up · Do simpler forecasts change the answer?", 8: "Follow-up · Do simpler forecasts change the answer?", 9: "Follow-up · Do simpler forecasts change the answer?", 10: "Follow-up · What do the merger and liquidation change?", 11: "Bonus · Does fine-tuning earn its place?", 12: "Next decision · What would a person investigate?"}[number],
            "title": title,
            "file": f"{number:02d}-"
            + "".join(c.lower() if c.isalnum() else "-" for c in title).strip("-")
            + ".png",
            "sources": [
                {
                    "path": str(p.relative_to(ROOT)),
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for p in paths
            ],
        }
    )
    pages.append(fig)
    return fig


def axis(fig, left=0.28, bottom=0.30, width=0.62, height=0.38):
    ax = fig.add_axes([left, bottom, width, height], facecolor="#FFFFFF")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#DEE4E8", linewidth=0.7)
    return ax


# 01: one accounting idea per visual step.
f = page(1, "Imagine a bank with $100 of assets.",
    "It has $15 in cash. Borrowers owe it $85 in loans, which they will repay over time.",
    "Customers request $20 today. The bank needs another $5 in cash. It could borrow or sell an asset. The next notebook step explains why equity is separate from available cash.",
    ["notebooks/source/banking_opening.py", "sources/banking_context.md"])
a = fig_assets = f.add_axes([.12,.49,.78,.18])
a.barh([0],[15],color=BLUE,height=.45)
a.barh([0],[85],left=[15],color=GRAY,height=.45)
a.text(7.5,0,"$15\ncash",ha="center",va="center",color="white",weight="bold",size=13)
a.text(57,0,"$85 loans\nrepaid over time",ha="center",va="center",color=INK,size=15)
a.set_xlim(0,100); a.set_ylim(-.6,.6); a.axis("off")
f.text(.12,.47,"Now customers ask to withdraw $20.",size=17,weight="bold",color=INK)
a = f.add_axes([.12,.29,.78,.14])
a.barh([0],[15],color=BLUE,height=.5)
a.barh([0],[5],left=[15],color=GOLD,height=.5)
a.text(7.5,0,"$15 available",ha="center",va="center",color="white",weight="bold",size=15)
a.text(17.5,0,"$5 needed",ha="center",va="center",color="white",weight="bold",size=14)
a.set_xlim(0,20); a.set_ylim(-.6,.6); a.axis("off")

# 02: one quarter, exact capacity and counts.
r = read("ranking.csv")
march = r.loc[r.Quarter.eq("2024-03-31")].set_index("Model")
k, n = int(march.loc["Ridge", "Selected"]), int(march.loc["Ridge", "Banks"])
f = page(
    2,
    f"{n:,} banks. {k} places to look.",
    "Can bank reports help us decide which banks to examine first? March 2024 reports predict June outcomes.",
    "The experiment assumes room to examine 10% of banks. These counts show selected banks later in the lowest-growth 10%. A person would investigate the reason for the movement.",
    ["growth_outputs/masterclass/ranking.csv"],
)
a = axis(f)
v = [k * k / n, march.loc["Ridge", "Hits"], march.loc["MLP", "Hits"]]
a.barh([2, 1, 0], v, color=[GRAY, BLUE, TEAL], height=0.48)
a.set_yticks([2, 1, 0], ["Random\nexpectation", "Ridge", "Neural\nnetwork"])
a.set_xlim(0, 145)
a.set_xlabel("Lowest-growth outcomes found")
for y, value in zip([2, 1, 0], v):
    a.text(value + 2, y, f"{value:.1f}" if y == 2 else f"{value:.0f}", va="center", weight="bold")

# 03: separate source coverage from the three modeling roles.
f = page(
    3,
    "Twelve years. Three different jobs.",
    "48 source quarters from 2013–2024. Training uses 38 predictor quarters; validation and evaluation use three each.",
    "2024 has already been examined during development. Thousands of bank examples share three evaluation dates. A later untouched period would test performance under new conditions.",
    ["growth_outputs/masterclass/splits.csv"],
)
a = axis(f, left=0.22, width=0.69)
splits = read("splits.csv")
for i, row in splits.iterrows():
    start, end = pd.to_datetime(row["First predictor"]), pd.to_datetime(row["Last predictor"])
    a.plot([start, end], [2 - i, 2 - i], lw=16, solid_capstyle="butt", color=[TEAL, GOLD, BLUE][i])
    a.annotate(
        f"{int(row.Rows):,} examples",
        (start, 2 - i),
        xytext=(0, -28),
        textcoords="offset points",
        fontsize=11,
    )
a.set_yticks([2, 1, 0], ["Learn", "Choose\nsettings", "Evaluate"])
a.set_ylim(-0.5, 2.5)
import matplotlib.dates as mdates

a.xaxis.set_major_locator(mdates.YearLocator(3))
a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
a.set_xlim(pd.Timestamp("2013-01-01"), pd.Timestamp("2025-09-01"))
# Explain the deliberate gaps directly on the exported timeline page.
f.text(0.075, 0.255,
       textwrap.fill("December 2022 and December 2023 are excluded as predictor dates because their next-quarter answers cross into the following stage.", 76),
       fontsize=11, va="top", linespacing=1.4, color=INK)


# 04: teach inputs before equations.
f = page(
    4,
    "Five inputs describe the bank today.",
    "Each prediction uses current balance-sheet information and the previous quarter’s deposit change.",
    "The output is next-quarter deposit growth. These five inputs give a partial view of a bank; a human would examine funding composition and institutional context too.",
    ["sources/fdic_fields.yaml", "build_masterclass.py"],
)
for i, (name, meaning) in enumerate(
    [
        ("Deposit size", "Log of domestic deposits"),
        ("Cash / assets", "Cash and balances due, relative to assets"),
        ("Loans / assets", "Net loans and leases, relative to assets"),
        ("Equity / assets", "Capitalization"),
        ("Previous growth", "Prior-quarter log deposit growth"),
    ]
):
    y = 0.65 - i * 0.082
    f.text(0.09, y, name, weight="bold", fontsize=17, color=TEAL)
    f.text(0.09, y - 0.029, meaning, fontsize=12)

# 05: same training observations, clearly disclosed central windows.
# The saved histograms contain the exact displayed observations, so exports do not require private preparation files.
import base64

hist = json.loads((OUT / "charts/target_growth_side_by_side.json").read_text())


def plot_values(value):
    if isinstance(value, dict) and "bdata" in value:
        return np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
    return np.asarray(value)


summary = read("target_summary.csv") if (OUT / "target_summary.csv").exists() else None
f = page(
    5,
    "A tiny starting balance makes a huge percentage.",
    "Training example: $1.15 million → $6.01 billion. The reported increase is about 522,646%; log growth is about 8.56.",
    "Both panels use the same training population and show their own central 98% windows. Extreme valid observations stay in training. The economic cause of this example remains unverified.",
    [
        "growth_outputs/masterclass/charts/target_growth_side_by_side.json",
        "growth_outputs/masterclass/extreme_growth.csv",
    ]
    if (OUT / "extreme_growth.csv").exists()
    else ["growth_outputs/masterclass/charts/target_growth_side_by_side.json"],
)
for i, (trace, label, col) in enumerate(
    zip(hist["data"], ["Ordinary growth (%)", "Log growth"], [GOLD, TEAL])
):
    a = axis(f, left=0.10 + i * 0.46, width=0.36, height=0.31, bottom=0.33)
    a.hist(plot_values(trace["x"]), bins=40, color=col)
    a.set_xlabel(label, fontsize=12)
    a.tick_params(labelsize=10)
    a.set_title("Central 98% display", fontsize=12)
    if i == 0:
        a.set_ylabel("Training examples", fontsize=11)

# 06: original four-model comparison with exact labels.
scores = read("historical_scores.csv")
f = page(
    6,
    "The error measure changes the leader.",
    "Original four-model comparison. All models are scored on the same 13,532 reused-2024 bank-quarter examples.",
    "Zero growth has the lowest mean absolute error. The neural network has the lowest RMSE, where large misses receive more weight. Follow-up baselines appear later.",
    ["growth_outputs/masterclass/historical_scores.csv"],
)
for i, metric in enumerate(["MAE (pp)", "RMSE (pp)"]):
    a = axis(f, left=0.23 + i * 0.43, width=0.29, height=0.32, bottom=0.33)
    ordered = scores.sort_values(metric)
    a.scatter(ordered[metric], range(4), s=95, c=[TEAL] + [GRAY] * 3)
    a.set_yticks(
        range(4),
        ordered.Model.replace({"Persistence": "Previous quarter", "Zero growth": "Zero growth"}),
        fontsize=10,
    )
    a.invert_yaxis()
    a.set_title(metric, fontsize=15)
    a.set_xlim(ordered[metric].min() - 0.2, ordered[metric].max() + 1)
    for j, (_, r0) in enumerate(ordered.iterrows()):
        a.text(r0[metric] + 0.06, j, f"{r0[metric]:.2f}", va="center", fontsize=10)

# 07: actual quarterly lift with an equal-quarter mean.
r = read("followup_ranking.csv")
mean = read("mean_ranking.csv").set_index("Model")
f = page(
    7,
    "About twice the concentration of low-growth outcomes.",
    "Select the lowest predicted-growth 10% each quarter. Lift compares the share found with the exact random expectation.",
    "Large dots show equal-quarter averages; small dots show individual quarters. Size-only Ridge is a follow-up benchmark developed after viewing 2024. March alone has different results from this average.",
    [
        "growth_outputs/masterclass/followup_ranking.csv",
        "growth_outputs/masterclass/mean_ranking.csv",
    ],
)
a = axis(f, left=0.28, width=0.63)
for j, name in enumerate(["Persistence", "Size-only Ridge", "Ridge", "MLP"]):
    values = r.loc[r.Model.eq(name), "Lift"]
    a.scatter(values, [j] * len(values), s=30, color=GRAY, zorder=2)
    a.scatter([mean.loc[name, "Lift"]], [j], s=110, color=TEAL, zorder=3)
    a.text(2.85, j, f"{mean.loc[name, 'Lift']:.2f}×", va="center", fontsize=13, weight="bold")
a.axvline(1, ls="--", color=INK)
a.set_xlim(0.8, 3.3)
a.set_yticks(
    range(4), ["Previous quarter", "Size-only Ridge", "Ridge", "Neural network"], fontsize=12
)
a.invert_yaxis()
a.set_xlabel("Lift relative to random selection")

# 08: paired precision difference, without declaring tiny gaps settled.
intervals = read("paired_intervals.csv")
part = intervals.loc[intervals.Metric.str.startswith("Precision")]
f = page(
    8,
    "The MLP–Ridge interval includes either ordering.",
    "Resample banks 1,000 times, keeping each bank’s repeated quarters together. Keep the fitted models and review lists fixed.",
    "A range crossing zero includes either ordering. These intervals describe variation across the sampled banks in the same three quarters. They do not measure uncertainty across future economic periods.",
    ["growth_outputs/masterclass/paired_intervals.csv"],
)
a = axis(f, left=0.34, width=0.55, height=0.30, bottom=0.34)
for j, (_, r0) in enumerate(part.iterrows()):
    a.plot([r0["95% lower"], r0["95% upper"]], [j, j], lw=3, color=TEAL)
    a.scatter([r0.Observed], [j], s=95, color=TEAL)
    a.text(r0.Observed, j + 0.17, f"{r0.Observed:+.2f} pp", ha="center", fontsize=12)
a.axvline(0, color=INK, ls="--")
a.set_yticks(
    range(len(part)), [s.replace(" minus ", "\nminus ") for s in part.Comparison], fontsize=12
)
a.set_ylim(-0.5, len(part) - 0.5)
a.set_xlabel("Precision difference (percentage points)", fontsize=11)

# 09: preserve individual years and distinguish aggregation methods.
season = read("training_seasonality.csv")
rules = read("seasonal_rules.csv")
f = page(
    9,
    "A seasonal baseline improves historical MAE.",
    "Training data only. Each faint line is a year. Large dots pool bank-quarter decline rates within each predictor quarter.",
    "The seasonal forecast uses training-period median growth for that quarter of the year. Its reused-2024 score is a follow-up result; the rule needs a later untouched evaluation.",
    [
        "growth_outputs/masterclass/training_seasonality.csv",
        "growth_outputs/masterclass/seasonal_rules.csv",
        "growth_outputs/masterclass/followup_scores.csv",
    ],
)
a = axis(f, left=0.15, width=0.75)
season["Q"] = season.Quarter.str[-1].astype(int)
for year, part0 in season.groupby("Year"):
    part0 = part0.sort_values("Q")
    a.plot(part0.Q, 100 * part0["Decline share"], color=GRAY, marker="o", ms=3, alpha=0.65)
a.scatter(rules.Quarter, 100 * rules["Decline share"], s=110, color=TEAL, zorder=4)
for _, r0 in rules.iterrows():
    a.text(
        r0.Quarter,
        100 * r0["Decline share"] + 2,
        f"{100 * r0['Decline share']:.1f}%",
        ha="center",
        fontsize=12,
        weight="bold",
    )
a.set_xticks([1, 2, 3, 4], ["Q1", "Q2", "Q3", "Q4"])
a.set_ylabel("Next-quarter decline rate (%)")
a.set_ylim(0, 65)

# 10: an actual event inside the forecast interval, with true date spacing.
merger = read("merger_timeline.csv").iloc[0]
f = page(10, "A merger falls inside the forecast interval.",
    "Plus International Bank, CERT 57083. Emigrant merged into Plus under the Emigrant name on October 1, 2023.",
    "The two-case event list is incomplete. Primary scores retain all eligible observations. The date source was published in November 2024 and supports retrospective annotation.",
    ["sources/structural_events.csv", "growth_outputs/masterclass/merger_timeline.csv", "sources/banking_context.md"])
f.text(.09,.65,f"+{merger['Reported growth (%)']:,.0f}%",size=42,weight="bold",color=INK)
f.text(.09,.607,"Reported deposit growth",size=16,color=GOLD)
a=f.add_axes([.12,.28,.76,.29]);a.set_xlim(-5,97);a.set_ylim(-1,1);a.axis("off")
a.plot([0,92],[0,0],color=GRAY,lw=4)
a.scatter([0,92],[0,0],s=100,color=BLUE)
a.scatter([1],[0],s=150,color=GOLD,marker="D",zorder=3)
a.annotate("OCT 1, 2023\nMerger effective",xy=(1,0),xytext=(27,.50),color=GOLD,size=14,weight="bold",
    arrowprops={"arrowstyle":"-","color":GOLD,"connectionstyle":"angle,angleA=0,angleB=90,rad=4"})
a.text(0,-.3,"SEP 30",ha="left",color=BLUE,size=13,weight="bold")
a.text(0,-.55,f"${merger['Starting deposits USD']/1e6:.2f}M",ha="left",size=20,weight="bold")
a.text(92,-.3,"DEC 31",ha="right",color=BLUE,size=13,weight="bold")
a.text(92,-.55,f"${merger['Next deposits USD']/1e9:.2f}B",ha="right",size=20,weight="bold")
f.text(.09,.26,"Actual calendar spacing. A merger falls inside the interval.",size=12,color=INK)

# 11: include all foundation statuses and retain all rows.
f = page(
    11,
    "Pretraining and adaptation still need a benchmark.",
    "TimesFM and Chronos were tested with frozen weights, then with their forecast heads adapted using earlier data.",
    "One bounded head-adaptation recipe was selected using validation. Neither adapted model beat the original baselines. TimesFM’s large error motivates the matching-history checks in the notebook.",
    [
        "growth_outputs/masterclass/historical_scores.csv",
        "growth_outputs/timesfm_finetuned_predictions.json",
        "growth_outputs/chronos_finetuned_predictions.json",
        "growth_outputs/timesfm_zero_shot_predictions.json",
        "growth_outputs/chronos_zero_shot_predictions.json",
        "growth_outputs/timesfm_zero_shot_predictions.csv",
        "growth_outputs/chronos_zero_shot_predictions.csv",
        "growth_outputs/timesfm_finetuned_predictions.csv",
        "growth_outputs/chronos_finetuned_predictions.csv",
        "growth_outputs/masterclass/predictions.csv",
    ],
)
a = axis(f, left=0.34, width=0.56, height=0.36)
values = [
    ("Zero growth", scores.set_index("Model").loc["Zero growth", "RMSE (pp)"], GRAY),
    ("MLP", scores.set_index("Model").loc["MLP", "RMSE (pp)"], TEAL),
]
for key, label in [("chronos", "Chronos"), ("timesfm", "TimesFM")]:
    for suffix, status, color in [
        ("zero_shot", "zero-shot", GOLD),
        ("finetuned", "head adapted", BLUE),
    ]:
        meta = json.loads((ROOT / f"growth_outputs/{key}_{suffix}_predictions.json").read_text())
        value = meta.get("RMSE (pp)")
        if value is None:
            # Recompute from saved forecasts if metadata stores scores under a nested key.
            preds = pd.read_csv(ROOT / f"growth_outputs/{key}_{suffix}_predictions.csv")
            actual = read("predictions.csv")[["CERT", "date", "growth"]]
            joined = actual.merge(preds, on=["CERT", "date"], validate="one_to_one")
            column = (
                "Prediction"
                if suffix == "finetuned"
                else ("TimesFM" if key == "timesfm" else "Chronos-Bolt")
            )
            value = 100 * np.sqrt(np.mean((joined[column] - joined.growth) ** 2))
        values.append((label + "\n" + status, value, color))
for j, (label, v, color) in enumerate(values):
    a.scatter(v, j, c=color, s=90)
    a.text(v * 1.12, j, f"{v:.2f}", va="center", fontsize=11)
a.set_yticks(range(len(values)), [v[0] for v in values], fontsize=11)
a.invert_yaxis()
a.set_xscale("log")
a.set_xticks([10, 100, 300], labels=["10", "100", "300"])
a.minorticks_off()
a.set_xlim(5, 550)
a.set_xlabel("Historical RMSE (pp), logarithmic scale", fontsize=11)

# 12: show the result and the next decision without a fabricated operational gain.
extra = read("extended_scores.csv").sort_values("MAE (pp)").iloc[0]
f = page(
    12,
    "Freeze the rules. Test the next unseen quarter.",
    "Use the experiment to decide which approaches deserve a stronger test.",
    "Freeze the candidate models, event rules, and review capacity. Evaluate a later period that has not guided development. Human investigation remains essential; no operational savings were measured.",
    [
        "growth_outputs/masterclass/extended_scores.csv",
        "growth_outputs/masterclass/mean_ranking.csv",
        "growth_outputs/masterclass/paired_intervals.csv",
    ],
)
for y, title, body in [
    (
        0.65,
        "Keep simple forecasts in the comparison.",
        f"{extra.Model} has {extra['MAE (pp)']:.3f} pp MAE in the expanded historical comparison.",
    ),
    (
        0.49,
        "Test whether the review list helps.",
        f"Ridge averaged {mean.loc['Ridge', 'Lift']:.2f}× random concentration. Size alone averaged {mean.loc['Size-only Ridge', 'Lift']:.2f}×.",
    ),
    (
        0.33,
        "Investigate the underlying bank.",
        "Check funding, deposit composition, mergers, and institution history before drawing conclusions.",
    ),
]:
    f.text(0.08, y, textwrap.fill(title, 40), fontsize=18, weight="bold", color=TEAL, va="top")
    f.text(0.08, y - 0.055, textwrap.fill(body, 64), fontsize=13, va="top", linespacing=1.3)

with PdfPages(DEST / "bank-deposit-story.pdf") as pdf:
    for fig, item in zip(pages, manifest["pages"]):
        fig.savefig(DEST / item["file"], dpi=120, facecolor=fig.get_facecolor())
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)
(DEST / "manifest.json").write_text(json.dumps(manifest, indent=2))
(DEST / "README.md").write_text(
    "Twelve numbered 1080 × 1350 PNGs and a matching PDF. Upload in numerical order. Every page is linked to local evidence in manifest.json. No measured financial savings are claimed.\n"
)
print(DEST)
