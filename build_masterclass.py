"""Build two visible-code notebooks from one experiment; preserve Experiment 0."""
from pathlib import Path
from textwrap import dedent
import hashlib
import ast
import shutil
import subprocess
from notebook_lessons import OPENING, teach_story
from concrete_teaching import apply_concrete_teaching, GUIDES
import nbformat as nbf
ROOT=Path(__file__).resolve().parent
sections=[]
def section(title, prose, code='', *, advanced=False):
    sections.append((title,dedent(prose).strip(),dedent(code).strip(),advanced))

section('Can a small neural network anticipate next quarter’s deposit growth?', '''
You put money in a bank. The bank uses deposits to help fund loans and other assets.
Now imagine some of that funding leaves. Replacing it can cost more. That is a real forecasting problem.

The Federal Reserve’s [February 2026 funding study](https://www.federalreserve.gov/econres/notes/feds-notes/assessing-bank-resilience-to-a-funding-shock-20260217.html)
puts deposits at roughly two-thirds of U.S. bank liabilities and explains the cost of replacing them with wholesale funding.
The FDIC’s [May 2026 study](https://www.fdic.gov/news/press-releases/2026/fdic-releases-staff-study-deposit-flows-three-failed-banks-spring-2023)
examines rapid depositor flight at three failed banks using transaction-level records.

**Our question is smaller and testable: can today’s financial report help forecast next quarter’s deposit growth?**
We compare a small neural network with zero growth, repeating last quarter, and Ridge regression.
Our quarterly balances describe report-to-report changes. A bank run requires much more detailed evidence, and these balances do not separate withdrawals from mergers.

The original experiment asked where decline dollars would concentrate. We preserve that discovery,
then ask how large each change was relative to the bank’s own deposits.

**Reading route:** inspect the banks → check the time pattern → define one prediction → protect time → train → compare → inspect weak outcomes.
All modeling code stays visible. Advanced lessons follow the core answer.

**Evidence boundary:** 2024 was already examined in the original project. It is a reused historical holdout.
Report publication times and historical vintages are unavailable. Treat this as a retrospective forecast study.
The same banks may appear in earlier and later periods.
''',r'''
# Keep imports, frozen settings, and rendering tools together.
import os, sys, json, hashlib, platform, time, html, textwrap, io
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('MPLCONFIGDIR',str(Path.cwd()/'.mpl-cache'))
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
from IPython.display import display, HTML
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from wm_notecards import WMTheme, init_notebook
from wm_notecards.cards import (
    preview_card,
    big_number_card,
    question_card,
    takeaway_card,
    wm_counterintuitive_card,
    wm_formula_card,
)
from wm_notecards.charts import (
    plot_shell_html,
    style_fig_wm,
    wm_render_figure_card,
)
from wm_notecards.pictogram import pictogram_card
from wm_notecards.eda import display_data_chips, wm_compare_fields
from wm_notecards.tables import (
    display_cols_by_dtype,
    style_describe_wm,
    wm_render_micro_profile_cards,
    wm_render_styler,
)
SEED = 42
FEATURES = [
    'log_deposits',
    'cash_ratio',
    'loan_ratio',
    'equity_ratio',
    'prior_growth',
]
MODEL_COLORS = {
    'Zero growth': '#627381',
    'Persistence': '#A86223',
    'Ridge': '#3F6294',
    'MLP': '#0B6F75',
}
MODEL_ORDER = ['Zero growth', 'Persistence', 'Ridge', 'MLP']
SIZE_ORDER = ['Smallest', 'Lower middle', 'Upper middle', 'Largest']
SLICE_ORDER = ['All', 'Realized bottom 25%', 'Realized bottom 10%']

ROOT = Path.cwd()
assert (ROOT / 'data/fdic_financials_2020_2024.csv').is_file(), (
    'Run from the project folder.'
)

OUT = ROOT / 'growth_outputs' / NOTEBOOK_EDITION
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'charts').mkdir(exist_ok=True)

tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.experimental.enable_op_determinism()
tf.keras.utils.set_random_seed(SEED)

# Use the repository's standard card, chip, typography, and accent colors.
# White plotting surfaces match the user's preferred white cards.
theme = WMTheme(width=860, height=480, card_bg='#FFFFFF', plot_bg='#FFFFFF')
init_notebook(expand_colab_outputs=True)

# VS Code places native Plotly outputs against the left edge of the output
# region. Center every renderer output once here so individual charts cannot
# drift away from the card column.
display(
    HTML(
        """
        <style>
        body { background: #EFF1F6; color: #172F3E; }
        .jp-RenderedMarkdown { color: #172F3E; }
        .jp-RenderedMarkdown h1, .jp-RenderedMarkdown h2, .jp-RenderedMarkdown h3 { color: #172F3E; }
        .jp-RenderedMarkdown a { color: #3F6294; }
        .jp-RenderedMarkdown { max-width: 940px; margin: auto; line-height: 1.65; }

        .output_container .output {
            display: flex !important;
            justify-content: center !important;
        }
        .output_container .output > div {
            margin-left: auto !important;
            margin-right: auto !important;
        }
        .wm-micro-rail {
            max-width: 860px !important;
            grid-auto-flow: row !important;
            grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            grid-auto-columns: auto !important;
            overflow-x: visible !important;
        }
        .wm-micro-rail .wm-micro-card:last-child {
            grid-column: 1 / -1;
        }
        @media (max-width: 720px) {
            .wm-micro-rail {
                grid-template-columns: minmax(0, 1fr) !important;
            }
            .wm-micro-rail .wm-micro-card:last-child {
                grid-column: auto;
            }
            .wm-table-card:has(td.col4) {
                overflow-x: auto !important;
            }
            .wm-table-card:has(td.col4) table {
                min-width: 700px !important;
            }
            .wm-table-card:has(td.col6) table {
                min-width: 780px !important;
            }
        }
        </style>
        """
    )
)

# One inline library makes saved chart outputs independent of a CDN.
display(HTML('<script>' + get_plotlyjs() + '</script>'))


# %% NOTEBOOK CELL
# Shared display helpers keep all plots centered and all exact receipts wrapped.
def table(frame, title, formats=None, wrap_columns=None):
    """Render a dataframe with the same centered teaching-card treatment."""
    default_formats = {
        column: (lambda value: value.strftime('%Y-%m-%d'))
        for column in frame.select_dtypes(include=['datetime'])
    }
    default_formats.update(
        {
            column: '{:,.3f}'
            for column in frame.select_dtypes(include=['floating'])
        }
    )
    default_formats.update(formats or {})

    styled = frame.style.hide(axis='index').format(
        default_formats,
        na_rep='Missing',
    )
    default_wrap = {
        column: 290
        for column in frame
        if pd.api.types.is_string_dtype(frame[column])
    }
    default_wrap.update(wrap_columns or {})
    styled = styled.set_table_styles(
        [
            {
                'selector': f'td.col{frame.columns.get_loc(column)}',
                'props': [
                    ('white-space', 'normal !important'),
                    ('overflow-wrap', 'anywhere !important'),
                    ('word-break', 'normal !important'),
                    ('text-align', 'left'),
                ],
            }
            for column in default_wrap
        ],
        overwrite=False,
    )
    wm_render_styler(
        styled,
        theme=theme,
        title=title,
        wrap_columns=default_wrap,
    )


def ordered_rows(frame, columns, category_orders=None, ascending=True):
    """Return rows in the order a reader expects to scan them."""
    result = frame.copy()

    for column, values in (category_orders or {}).items():
        result[column] = pd.Categorical(
            result[column],
            categories=values,
            ordered=True,
        )

    return result.sort_values(
        columns,
        ascending=ascending,
        kind='stable',
    ).reset_index(drop=True)


# %% NOTEBOOK CELL
def chart(fig, name, title, subtitle='', height=540, legend_y=-0.24):
    """Apply one visual system and one centered renderer to every chart."""
    default_palette = {'#636efa':'#3F6294', '#EF553B':'#A86223', '#00cc96':'#0B6F75',
                       '#ab63fa':'#76528B', '#FFA15A':'#A86223', '#19d3f3':'#0B6F75'}
    for trace in fig.data:
        for component in ['marker', 'line']:
            obj = getattr(trace, component, None)
            if obj is not None and isinstance(getattr(obj, 'color', None), str):
                obj.color = default_palette.get(obj.color, obj.color)
    title = title.replace(chr(36), "USD ")
    subtitle = subtitle.replace(chr(36), "USD ")
    style_fig_wm(
        fig,
        title=title,
        subtitle=subtitle,
        theme=theme,
        normalize_legacy_colors=False,
        category_policy='preserve',
        allow_dense_categories=True,
    )

    title_html = '<b>' + '<br>'.join(
        html.escape(line)
        for line in textwrap.wrap(title, 52)
    ) + '</b>'
    if subtitle:
        wrapped_subtitle = '<br>'.join(
            html.escape(line)
            for line in textwrap.wrap(subtitle, 88)
        )
        title_html += (
            '<br><span style="font-size:14px">'
            + wrapped_subtitle
            + '</span>'
        )

    fig.update_layout(
        width=860,
        height=height,
        font=dict(size=14, family='Inter, Arial, sans-serif', color=theme.text_main),
        title=dict(
            text=title_html,
            font=dict(size=24),
            x=0.035,
            y=0.98,
            xanchor='left',
            yanchor='top',
        ),
        hovermode='closest',
        hoverlabel=dict(
            bgcolor='white',
            bordercolor='#B8C2CC',
            font=dict(
                size=14,
                color='#172F3E',
                family='Inter, Arial, sans-serif',
            ),
        ),
        legend=dict(
            y=legend_y,
            yanchor='top',
            x=0.5,
            xanchor='center',
            orientation='h',
            font=dict(size=14, family='Inter, Arial, sans-serif'),
            title_text='',
        ),
        margin=dict(l=85, r=45, t=155, b=155 if legend_y < -0.24 else 120),
        paper_bgcolor=theme.card_bg,
        plot_bgcolor=theme.plot_bg,
    )
    fig.update_xaxes(
        automargin=True,
        showline=True,
        linecolor='#ADB5BD',
        gridcolor='#ECEFF1',
        tickfont=dict(size=14, family='Inter, Arial, sans-serif'),
        title_font=dict(size=15, family='Inter, Arial, sans-serif'),
    )
    fig.update_yaxes(
        automargin=True,
        showline=True,
        linecolor='#ADB5BD',
        gridcolor='#ECEFF1',
        tickfont=dict(size=14, family='Inter, Arial, sans-serif'),
        title_font=dict(size=15, family='Inter, Arial, sans-serif'),
    )
    fig.for_each_yaxis(
        lambda axis: axis.update(dtick=1) if axis.type == 'log' else None
    )
    fig.write_json(OUT / 'charts' / f'{name}.json')

    if 'google.colab' in sys.modules:
        wm_render_figure_card(fig, theme=theme, file_stub=name)
        return

    # VS Code chooses Plotly's native MIME renderer before the centered WM
    # shell. Render one self-contained HTML output locally so the shell owns
    # alignment. Plotly.js was loaded once in the setup cell above.
    figure_html = fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={
            'displaylogo': False,
            'responsive': False,
            'displayModeBar': False,
        },
        default_width='860px',
        default_height=f'{height}px',
    )
    shell = plot_shell_html(
        figure_html,
        theme,
        figure_width=860,
    )
    display(HTML(shell))
    guide = READING_GUIDES.get(name)
    if guide:
        display(HTML('<p style="max-width:860px;margin:12px auto 28px;line-height:1.6;text-align:left"><strong>Read this chart:</strong> ' + html.escape(guide) + '</p>'))


# %% NOTEBOOK CELL
def takeaway(title, body, metric=None):
    """Show the answer immediately after its evidence."""
    takeaway_card(title=title, body=body, metric=metric, theme=theme)


def scores(actual, predicted):
    """Report ordinary-growth errors in percentage points."""
    return {
        'MAE (pp)': 100 * mean_absolute_error(actual, predicted),
        'RMSE (pp)': 100 * np.sqrt(mean_squared_error(actual, predicted)),
    }


def log_scores(y_log, pred_log):
    """Score log-growth forecasts in ordinary percentage-point units."""
    return scores(np.expm1(y_log), np.expm1(pred_log))


# %% NOTEBOOK CELL
preview_card(
    title='Can the network earn its extra complexity?',
    theme=theme,
    body=(
        'Every model gets the same dates and outcomes. We will judge the '
        'neural network by the forecast errors it reduces.'
    ),
    bullets=[
        'One row: one institution at one reporting quarter.',
        'One target: next-quarter domestic deposit growth.',
        'One honest comparison: simple rules, Ridge, and a small MLP.',
    ],
)
''')
section('1 · Can fifteen years of reports tell one consistent story?', '''
**We chose 2013–2024 for the main experiment.** The 2010–2012 reports stay in a separate audit. Here is how we decided:
1. **Start with the same kind of report.** Some older institutions used a different reporting form. The FDIC documented their switch to Call Reports in 2012.
2. **Check that the numbers mean the same thing.** Matching column names alone cannot prove that older forms measured all five financial inputs the same way. That historical mapping is still unverified.
3. **Check the newer file.** Its 48 quarters match the corresponding rows in the longer file exactly. Starting in 2013 gives us a practical boundary after the reporting change.
4. **Keep the older evidence.** We retain 2010–2012 for inspection, but do not use it to train this comparison. This is a cautious comparability choice, not proof that every older row is wrong.

See the [FDIC's 2012 reporting conversion notice](https://www.fdic.gov/news/inactive-financial-institution-letters/2012/fil12010.html). The detailed receipts appear after the result; you already have the decision.
''',r'''
# EXEMPLAR: analytical-question
# Give the decision before the audit that supports it.
question_card(
    title='Can reports from 2013 and 2024 live in one experiment?',
    theme=theme,
    body=(
        'We use 2013–2024. Earlier reports cross a 2012 reporting-form change; '
        'their five-field historical mapping remains unverified. The newer '
        '48 quarters match the long file exactly.'
    ),
    kicker='Data history',
    chip_text='QUESTION',
)


def ordered(frame):
    """Return bank-quarter rows in a stable order for exact comparisons."""
    return frame.sort_values(['CERT', 'REPDTE']).reset_index(drop=True)


def schema_table(frame):
    """Summarize the basic structure a reader checks at the start of EDA."""
    return pd.DataFrame(
        {
            'Column': frame.columns,
            'Data type': frame.dtypes.astype(str).to_numpy(),
            'Missing rows': frame.isna().sum().to_numpy(),
            'Distinct values': frame.nunique(dropna=False).to_numpy(),
        }
    )


# Keep the audit history separate from the post-conversion modeling history.
long_history = pd.read_csv(ROOT / 'data/fdic_financials_2010_2024.csv')
post_conversion = pd.read_csv(ROOT / 'data/fdic_financials_2013_2024.csv')
recent_extract = pd.read_csv(ROOT / 'data/fdic_financials_2020_2024.csv')
reporting = pd.read_csv(ROOT / 'data/fdic_reporting_2010_2024.csv')
keys = ['CERT', 'REPDTE']

# Teach the source vocabulary before asking the reader to inspect numbers.
source_dictionary = pd.DataFrame([
    ('REPDTE', 'Report date', 'The quarter-end date for this report'),
    ('CERT', 'Institution ID', 'FDIC certificate; links reports for one institution'),
    ('NAME', 'Institution name', 'Reported name; can change over time'),
    ('STALP', 'State abbreviation', 'Location label; not a model input'),
    ('DEPDOM', 'Domestic deposits', 'Deposits in domestic offices; USD thousands'),
    ('ASSET', 'Total assets', 'Total reported assets; USD thousands'),
    ('LNLSNET', 'Net loans and leases', 'Reported net loan and lease balance; USD thousands'),
    ('CHBAL', 'Cash and balances due', 'Cash and balances due from depository institutions; USD thousands'),
    ('EQ', 'Equity capital', 'Reported equity capital; USD thousands'),
], columns=['Column', 'Plain name', 'Meaning and units'])
table(source_dictionary, 'Read the column names before reading the numbers')
# %% NOTEBOOK CELL
# First contact with the data: size, sample rows, types, missingness, and keys.
shape_receipt = pd.DataFrame(
    {
        'Rows': [len(post_conversion)],
        'Columns': [post_conversion.shape[1]],
        'Banks': [post_conversion['CERT'].nunique()],
        'Quarters': [post_conversion['REPDTE'].nunique()],
        'First quarter': [post_conversion['REPDTE'].min()],
        'Last quarter': [post_conversion['REPDTE'].max()],
        'Duplicate bank-quarters': [
            int(post_conversion.duplicated(keys).sum())
        ],
    }
)
table(shape_receipt, 'What is in the post-conversion extract?')
table(
    post_conversion[
        ['REPDTE', 'CERT', 'NAME', 'STALP']
    ].head(5),
    'Five source rows: identity and reporting date',
)
table(
    post_conversion[
        ['CERT', 'DEPDOM', 'ASSET', 'LNLSNET', 'CHBAL', 'EQ']
    ].head(5),
    'The same five rows: financial balances in USD thousands',
)
table(schema_table(post_conversion), 'Column types, missing values, and cardinality')

# The familiar pandas checks come first. The cards below explain their results.
post_conversion.head()
# %% NOTEBOOK CELL
info_buffer = io.StringIO()
post_conversion.info(buf=info_buffer, memory_usage='deep')
print(info_buffer.getvalue())
# %% NOTEBOOK CELL
post_conversion.isna().sum().sort_values(ascending=False)
# %% NOTEBOOK CELL
# EXEMPLAR: missingness-evidence
# These two illustrative entries show why a missing value cannot be read as zero.
display(pd.DataFrame({
    "Illustrative EQ entry": [np.nan, 0.0],
    "Reading": ["No value supplied in this extract", "A recorded numeric zero; verify its reporting meaning"],
}))
# Explain the missingness result while it is still on screen.
source_audit = post_conversion.merge(reporting, on=keys, validate='one_to_one')
missing_eq = source_audit['EQ'].isna()
inside_population = (source_audit['BKCLASS'].isin(['N', 'NM', 'SM', 'SB', 'SI', 'SL'])
                     & source_audit['CALLFORM'].isin([31, 41, 51]))
assert not (missing_eq & inside_population).any()
assert source_audit.loc[missing_eq, 'CALLFORM'].eq(2).all()
preview_card(
    title='Missing equity stays unknown; these reports are outside our population',
    theme=theme,
    body=f'The 2013–2024 source has {missing_eq.sum():,} missing EQ values. All occur on form 2; none belong to the domestic-bank/report-form group used for modeling.',
    bullets=[
        '1. A blank means unknown. We do not replace source equity with zero.',
        '2. We select domestic-bank classes N, NM, SM, SB, SI, SL on forms 31, 41, 51. This eligibility rule removes these reports before modeling.',
        '3. We do not borrow a domestic-bank median to invent equity for a different reporting group.',
        f'4. STALP has {post_conversion.STALP.isna().sum():,} blanks. State is not one of the five inputs, so we leave those labels missing.',
        '5. The later median imputer is fitted only on training features. It is a safeguard, not the treatment used for these source EQ gaps.',
    ],
)
# %% NOTEBOOK CELL
source_summary = post_conversion[
    ['DEPDOM', 'ASSET', 'LNLSNET', 'CHBAL', 'EQ']
].describe()
display(source_summary)
# %% NOTEBOOK CELL
display_data_chips(
    post_conversion,
    theme=theme,
    identifier_columns=['CERT'],
    datetime_columns=['REPDTE'],
    categorical_columns=['STALP', 'NAME'],
    group_label='What each source column means',
)
display_cols_by_dtype(
    post_conversion.dtypes,
    theme=theme,
    group_label='How pandas stores these columns',
)
wm_render_styler(
    style_describe_wm(source_summary, theme),
    theme=theme,
    title='Source balances: the familiar describe() summary',
)

# A bank-quarter panel has a history. Count reporting banks and track the
# median deposit balance so changes in the population are visible in time.
quarterly_panel = (
    post_conversion.assign(
        Date=pd.to_datetime(post_conversion['REPDTE'].astype(str))
    )
    .groupby('Date', as_index=False)
    .agg(
        Banks=('CERT', 'nunique'),
        Median_deposits=('DEPDOM', 'median'),
    )
    .sort_values('Date')
)
quarterly_panel['Median deposits (USD million)'] = (
    quarterly_panel['Median_deposits'] / 1000
)

fig = px.line(quarterly_panel, x='Date', y='Banks', markers=True)
fig.update_traces(line=dict(color='#0B6F75', width=3), marker=dict(size=5))
fig.update_yaxes(title='Reporting banks', rangemode='tozero')
chart(fig, 'reporting_banks_time', 'The reporting population shrinks across 2013–2024',
      '2013–2024; distinct bank certificates')

fig = px.line(
    quarterly_panel,
    x='Date',
    y='Median deposits (USD million)',
    markers=True,
)
fig.update_traces(line=dict(color='#A86223', width=3), marker=dict(size=5))
fig.update_yaxes(title='Median deposits (USD million)', rangemode='tozero')
chart(fig, 'median_deposits_time', 'Median reported deposits rise as the bank population changes',
      'Median domestic deposits per reporting bank; USD million')

assert not long_history.duplicated(keys).any()
assert not post_conversion.duplicated(keys).any()
assert not reporting.duplicated(keys).any()

# The bundled extracts must reproduce the same rows in the long audit file.
source_columns = list(recent_extract.columns)
recent_rows_match = ordered(recent_extract).equals(
    ordered(
        long_history.loc[
            long_history['REPDTE'].ge(20200331),
            source_columns,
        ]
    )
)
post_conversion_rows_match = ordered(post_conversion).equals(
    ordered(
        long_history.loc[
            long_history['REPDTE'].ge(20130331),
            source_columns,
        ]
    )
)

assert recent_rows_match
assert post_conversion_rows_match
assert long_history['REPDTE'].nunique() == 60
assert post_conversion['REPDTE'].nunique() == 48

# Reporting metadata lets us explain missing equity instead of treating it as random.
long_audit = long_history.merge(
    reporting,
    on=keys,
    how='left',
    validate='one_to_one',
    indicator=True,
)
coverage = (
    long_audit.groupby('REPDTE')
    .agg(
        Rows=('CERT', 'size'),
        Equity_missing=('EQ', lambda values: values.isna().mean()),
        Metadata_matched=('_merge', lambda values: values.eq('both').mean()),
    )
    .reset_index()
)
coverage['Date'] = pd.to_datetime(coverage['REPDTE'].astype(str))
coverage.to_csv(OUT / 'history_coverage.csv', index=False)

missing = (
    long_history.isna()
    .sum()
    .rename('Missing rows')
    .rename_axis('Field')
    .reset_index()
)
missing = ordered_rows(
    missing,
    ['Missing rows', 'Field'],
    ascending=[False, True],
)
table(missing, 'Missing values in the 2010–2024 audit file')

fig = px.bar(
    missing,
    x='Missing rows',
    y='Field',
    orientation='h',
    color_discrete_sequence=['#A86223'],
)
chart(
    fig,
    'missingness',
    'Equity accounts for the missing financial values',
    '2010–2024 audit file; no values filled',
)

by_form = (
    long_audit.groupby(['BKCLASS', 'CALLFORM'], dropna=False)
    .agg(
        Rows=('CERT', 'size'),
        Missing_equity=('EQ', lambda values: values.isna().sum()),
    )
    .reset_index()
)
by_form.to_csv(OUT / 'reporting_forms.csv', index=False)
table(
    ordered_rows(
        by_form.loc[by_form['Missing_equity'].gt(0)],
        ['Missing_equity', 'BKCLASS', 'CALLFORM'],
        ascending=[False, True, True],
    ),
    'Missing equity is concentrated in particular reporting groups',
)

fig = px.line(
    coverage,
    x='Date',
    y='Equity_missing',
    markers=True,
    color_discrete_sequence=['#A86223'],
)
fig.update_yaxes(tickformat='.2%', title='Reports missing equity')
chart(
    fig,
    'coverage_time',
    'Missing equity varies across report dates',
    'All reports in the 2010–2024 audit file',
)

wm_counterintuitive_card(
    title='What a novice might overlook',
    theme=theme,
    why_misread=(
        'The column names match across fifteen years, and the newer extracts '
        'match the long file exactly.'
    ),
    ordinary_process=(
        'A regulatory form can change while a downloaded field keeps the same '
        'name. The 2012 TFR-to-Call-Report conversion creates that risk.'
    ),
    conclusion_boundary=(
        'Use the 48 quarters from 2013–2024. Keep 2010–2012 in the audit until '
        'a field-level historical crosswalk is verified.'
    ),
    kicker='Comparability check',
    chip_text='LOOK TWICE',
)

gate = {
    'all_60_audit_quarters': True,
    'duplicate_keys': 0,
    'recent_extract_exact': recent_rows_match,
    'post_2012_extract_exact': post_conversion_rows_match,
    'historical_form_crosswalk_verified': False,
    'selected_start': 2013,
    'reason': (
        'The 2013–2024 extract begins after the documented 2012 conversion and '
        'matches the long extract exactly. Pre-2013 rows remain in the audit only.'
    ),
}
(OUT / 'comparability_gate.json').write_text(json.dumps(gate, indent=2))

raw = post_conversion.copy()
source_path = ROOT / 'data/fdic_financials_2013_2024.csv'
source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
raw_fingerprint = pd.util.hash_pandas_object(raw, index=True).sum()

takeaway(
    'Use 2013–2024 for the primary experiment',
    (
        'The 48 post-conversion quarters match the long extract exactly. Reports '
        'from 2010–2012 stay in the audit because the field-level crosswalk '
        'remains unresolved.'
    ),
)
''')
section('2 · What happened when the first experiment counted dollars?', '''
**Large banks can dominate a dollar-based score even when the model learns little beyond size.**
A 1% decline at a bank with USD 100 billion in deposits is USD 1 billion.
A 10% decline at a bank with USD 100 million is USD 10 million.
The first bank contributes 100 times as many decline dollars despite its smaller proportional decline.

Reproduce the old eligibility and training-size boundaries before quoting the old concentration.
The archived masterclass preserves the two-head network, PR/calibration lessons, and review-capacity analysis.
''',r'''
# Test the simple size explanation before crediting the old network.
question_card(
    title='Could a model look smart by learning bank size?',
    theme=theme,
    body='Rebuild the original dollar target, then compare its network with a rule that ranks banks by deposits alone.',
    kicker='Experiment 0',
    chip_text='QUESTION',
)

# First, rebuild the exact rows used by the original dollar experiment.
def build_experiment_zero_rows(source):
    """Recreate adjacent-quarter rows under the original eligibility rule."""
    rows = source.sort_values(keys).copy()
    rows['date'] = pd.to_datetime(rows.REPDTE.astype(str))
    rows['q'] = rows.date.dt.to_period('Q').astype('int64')

    grouped = rows.groupby('CERT')
    shifted_columns = [
        ('prev', 'DEPDOM', 1),
        ('next', 'DEPDOM', -1),
        ('prev_q', 'q', 1),
        ('next_q', 'q', -1),
    ]
    for new_column, source_column, periods in shifted_columns:
        rows[new_column] = grouped[source_column].shift(periods)

    eligible = (
        rows.q.sub(rows.prev_q).eq(1)
        & rows.next_q.sub(rows.q).eq(1)
        & rows.DEPDOM.ge(10_000)
        & rows.prev.gt(0)
        & rows.next.ge(0)
        & rows.ASSET.gt(0)
    )
    rows = rows.loc[eligible].copy()
    rows['runoff_m'] = rows.DEPDOM.sub(rows.next).clip(lower=0).div(1_000)

    return rows


def summarize_dollars_by_size(rows):
    """Learn size quartiles on training rows and apply them to 2024 rows."""
    training_rows = rows.loc[rows.date.le('2022-09-30')]
    test_rows = rows.loc[
        rows.date.between('2024-01-01', '2024-09-30')
    ].copy()

    quartiles = training_rows.DEPDOM.quantile([0.25, 0.50, 0.75])
    size_edges = np.r_[-np.inf, quartiles, np.inf]
    size_labels = [
        'Smallest',
        'Lower middle',
        'Upper middle',
        'Largest',
    ]
    test_rows['Size group'] = pd.cut(
        test_rows.DEPDOM,
        bins=size_edges,
        labels=size_labels,
    )

    concentration = (
        test_rows.groupby('Size group', observed=True)
        .agg(
            Rows=('CERT', 'size'),
            Decline_m=('runoff_m', 'sum'),
        )
        .reset_index()
    )
    concentration['Dollar share'] = concentration.Decline_m.div(
        concentration.Decline_m.sum()
    )

    return concentration


# %% NOTEBOOK CELL
# Run the two small helpers, then inspect where the decline dollars landed.
old = build_experiment_zero_rows(recent_extract)
concentration = summarize_dollars_by_size(old)
concentration = ordered_rows(
    concentration,
    ['Size group'],
    category_orders={'Size group': SIZE_ORDER},
)

table(
    concentration,
    'Experiment 0: 2024 Q1–Q3 predictor rows',
    {
        'Dollar share': '{:.2%}',
        'Decline_m': '{:,.1f}',
    },
)

fig = px.bar(
    concentration,
    x='Size group',
    y='Dollar share',
    text=concentration['Dollar share'].map('{:.2%}'.format),
    color_discrete_sequence=['#0B6F75'],
)
fig.update_yaxes(
    tickformat='.0%',
    range=[0, 1.08],
    title='Share of decline dollars',
)
chart(
    fig,
    'experiment0',
    'Large banks dominate the dollar-decline totals',
    'Size cutoffs learned from the original training period',
)

# %% NOTEBOOK CELL
# Compare the archived network with the size-only rule.
share = float(
    concentration.loc[
        concentration['Size group'].eq('Largest'),
        'Dollar share',
    ].iloc[0]
)
legacy_path = ROOT / 'experiments/experiment_0/outputs/run_summary.json'
legacy = json.loads(legacy_path.read_text())

comparison = pd.DataFrame(
    {
        'Method': ['Size rule', 'Original base network'],
        'Quarter-average capture (%)': [
            100 * legacy['size_capture'],
            100 * legacy['network_capture'],
        ],
    }
)
comparison = ordered_rows(comparison, ['Quarter-average capture (%)'])
table(
    comparison,
    'Archived Experiment 0: 10% review capacity',
    {'Quarter-average capture (%)': '{:.2f}'},
)

size_capture = comparison.loc[
    comparison.Method.eq('Size rule'),
    'Quarter-average capture (%)',
].iloc[0]
network_capture = comparison.loc[
    comparison.Method.eq('Original base network'),
    'Quarter-average capture (%)',
].iloc[0]
delta = float(network_capture - size_capture)

fig = go.Figure(
    go.Scatter(
        x=comparison['Quarter-average capture (%)'],
        y=[0, 0],
        mode='lines+markers+text',
        line=dict(color='#ADB5BD', width=5),
        marker=dict(
            size=16,
            color=['#627381', '#0B6F75'],
        ),
        text=[
            f'{value:.2f}%'
            for value in comparison['Quarter-average capture (%)']
        ],
        textposition=['bottom center', 'top center'],
        hovertemplate='%{text}<extra></extra>',
    )
)
fig.update_yaxes(visible=False, range=[-0.45, 0.45])
fig.update_xaxes(
    title='Quarter-average capture (%)',
    range=[84.7, 85.2],
)
chart(
    fig,
    'old_capture',
    f'The network added {delta:.2f} percentage points',
    'Size rule on the left; original base network on the right',
    height=340,
)

wm_counterintuitive_card(
    title='What a novice might overlook',
    theme=theme,
    why_misread='An 85% capture rate sounds like strong pattern recognition.',
    ordinary_process='The largest banks hold most of the decline dollars. Ranking by deposits alone captured 84.82%.',
    conclusion_boundary=f'The original network added {delta:.2f} percentage points over the size rule. That result mainly reflects the target’s dollar weighting.',
    kicker='Interpretation check',
    chip_text='LOOK TWICE',
)
takeaway(
    'Bank size explains much of the dollar result',
    (
        f'The largest training-size group contains {share:.2%} of historical '
        'decline dollars. The original network added '
        f'{legacy["network_difference_pp"]:.2f} percentage points of '
        'quarter-average capture over size alone.'
    ),
    f'{share:.2%}',
)
concentration.to_csv(
    OUT / 'experiment0_concentration.csv',
    index=False,
)
''')
section('3 · Which rows can actually teach us about next quarter?', '''
**A missing next report is an unknown outcome.** It can reflect a merger, closure, reporting gap,
or the end of our file. We separate these cases as far as the data allows.

A certificate connects reports filed under the same institution identifier. Mergers, name changes, and other
structural events can still change the business behind that identifier, so we preserve those clues for review.
All 1,018 missing equity values in the long extract occur on form 2, across classes OI and NC.
Keep the primary population to insured domestic-bank classes N, NM, SM, SB, SI, and SL
using forms 31, 41, or 51. This avoids filling foreign-branch structural gaps with a domestic-bank median.
''',r'''
# Decide which bank-quarters can receive a measured outcome.
question_card(
    title='Which bank-quarters have a real next-quarter answer?',
    theme=theme,
    body=(
        'Trace each certificate through adjacent quarters, then count every row '
        'removed by the eligibility rules.'
    ),
    kicker='Forecast population',
    chip_text='QUESTION',
)


def add_adjacent_reports(frame):
    """Attach prior and next-quarter values within each FDIC certificate."""
    result = frame.sort_values(['CERT', 'REPDTE']).copy()
    result['date'] = pd.to_datetime(result['REPDTE'].astype(str))
    result['quarter_number'] = result['date'].dt.to_period('Q').astype('int64')

    grouped = result.groupby('CERT', sort=False)
    shifts = {
        'prior_deposits': ('DEPDOM', 1),
        'next_deposits': ('DEPDOM', -1),
        'prior_quarter': ('quarter_number', 1),
        'next_quarter': ('quarter_number', -1),
        'target_date': ('date', -1),
        'next_name': ('NAME', -1),
        'next_event': ('ACTEVT', -1),
    }

    for new_column, (source_column, periods) in shifts.items():
        result[new_column] = grouped[source_column].shift(periods)

    return result


def label_next_report(frame):
    """Separate ordinary missing outcomes from the end of the dataset."""
    last_quarter = frame['quarter_number'].max()

    return np.select(
        [
            frame['quarter_number'].eq(last_quarter),
            frame['next_quarter'].isna(),
            frame['next_quarter'].sub(frame['quarter_number']).ne(1),
        ],
        [
            'Dataset end',
            'Institution stops before dataset end',
            'Gap before later report',
        ],
        default='Adjacent next report',
    )


def apply_rules(frame, rules):
    """Apply eligibility rules one at a time and record their effect."""
    keep = pd.Series(True, index=frame.index)
    ledger = [
        {
            'Rule': 'All source rows',
            'Remaining': len(frame),
            'Removed': 0,
        }
    ]

    for rule_name, rule_mask in rules.items():
        rows_before = int(keep.sum())
        keep &= rule_mask
        rows_after = int(keep.sum())

        ledger.append(
            {
                'Rule': rule_name,
                'Remaining': rows_after,
                'Removed': rows_before - rows_after,
            }
        )

    return frame.loc[keep].copy(), pd.DataFrame(ledger)


def add_model_fields(frame):
    """Create the three candidate targets and five current-quarter inputs."""
    result = frame.copy()

    result['growth'] = result['next_deposits'].div(result['DEPDOM']).sub(1)
    result['log_growth'] = np.log(result['next_deposits'].div(result['DEPDOM']))
    result['dollar_change_m'] = result['next_deposits'].sub(result['DEPDOM']).div(1000)

    result['log_deposits'] = np.log(result['DEPDOM'])
    result['cash_ratio'] = result['CHBAL'].div(result['ASSET'])
    result['loan_ratio'] = result['LNLSNET'].div(result['ASSET'])
    result['equity_ratio'] = result['EQ'].div(result['ASSET'])
    result['prior_growth'] = np.log(result['DEPDOM'].div(result['prior_deposits']))

    return result


# %% NOTEBOOK CELL
# Merge reporting metadata before building the bank timeline.
panel = raw.merge(
    reporting,
    on=['CERT', 'REPDTE'],
    how='left',
    validate='one_to_one',
    indicator=True,
)
assert panel['_merge'].eq('both').all()

panel = add_adjacent_reports(panel)
panel['Next report status'] = label_next_report(panel)

# Count every reason that a future balance is unavailable.
missing_next = (
    panel.groupby('Next report status')
    .size()
    .rename('Rows')
    .reset_index()
)

next_report_order = [
    'Adjacent next report',
    'Dataset end',
    'Institution stops before dataset end',
    'Gap before later report',
]
missing_next = ordered_rows(
    missing_next,
    ['Next report status'],
    category_orders={'Next report status': next_report_order},
)

table(missing_next, 'A missing future balance stays missing')

fig = px.bar(
    missing_next.loc[missing_next['Next report status'].ne('Adjacent next report')],
    x='Rows',
    y='Next report status',
    orientation='h',
    text='Rows',
    color_discrete_sequence=['#0B6F75'],
)
fig.update_traces(texttemplate='%{text:,}', textposition='outside')
fig.update_layout(margin=dict(l=230, r=100, t=110, b=65))
chart(fig, 'next_report', 'The dataset ends before some banks have a next report',
      'Counts exclude the adjacent reports shown in the table')

unobserved_columns = [
    'CERT',
    'REPDTE',
    'NAME',
    'date',
    'ACTEVT',
    'Next report status',
]
panel.loc[
    panel['Next report status'].ne('Adjacent next report'),
    unobserved_columns,
].to_csv(OUT / 'unobserved_outcomes.csv', index=False)

# %% NOTEBOOK CELL
# The log-growth experiment requires positive balances in all three quarters.
domestic_classes = ['N', 'NM', 'SM', 'SB', 'SI', 'SL']
comparable_forms = [31, 41, 51]

conditions = {
    'Insured domestic bank; comparable Call Report': (
        panel['BKCLASS'].isin(domestic_classes)
        & panel['CALLFORM'].isin(comparable_forms)
    ),
    'Adjacent prior quarter': (
        panel['quarter_number'].sub(panel['prior_quarter']).eq(1)
    ),
    'Adjacent next quarter': (
        panel['next_quarter'].sub(panel['quarter_number']).eq(1)
    ),
    'Positive deposits in prior, current, and next quarter': (
        panel['prior_deposits'].gt(0)
        & panel['DEPDOM'].gt(0)
        & panel['next_deposits'].gt(0)
    ),
    'Positive current assets': panel['ASSET'].gt(0),
}

rows, eligibility_ledger = apply_rules(panel, conditions)
rows = add_model_fields(rows)

eligibility_ledger = ordered_rows(
    eligibility_ledger,
    ['Rule'],
    category_orders={'Rule': ['All source rows', *conditions.keys()]},
)

assert np.isfinite(rows['growth']).all()
assert np.isfinite(rows['log_growth']).all()
assert np.isfinite(rows['prior_growth']).all()
assert rows['growth'].ge(-1).all()
assert len(panel) - len(rows) == eligibility_ledger['Removed'].sum()

table(eligibility_ledger, 'Every exclusion has a count')

takeaway(
    'The forecasting rows have three consecutive positive balances',
    (
        f'{len(rows):,} eligible bank-quarters remain. Small deposit bases stay '
        'in the sample, so the target comparison must show how strongly they '
        'stretch ordinary percentage growth.'
    ),
)
''')
section('4 · How do we keep tomorrow out of today’s inputs?', '''
**Give each model only information from the predictor quarter or earlier.**
If a row describes September, its label describes December. December must not sneak into September’s features.

We leave 2022 Q4 and 2023 Q4 predictor rows out at the boundaries. Training outcomes therefore precede
validation predictor dates; validation outcomes precede the historical evaluation predictor dates.
Banks can appear in more than one period. The evaluation therefore covers later quarters for the monitored
population; it does not measure performance on a completely new set of institutions.
''',r'''
# Check the date boundaries before fitting any model.
question_card(
    title='Has any future quarter leaked into training?',
    theme=theme,
    body=(
        'The last training outcome must occur before the first validation '
        'predictor. Apply the same check at the holdout boundary.'
    ),
    kicker='Time split',
    chip_text='QUESTION',
)


def describe_split(name, frame):
    """Return one readable row for the chronological split receipt."""
    return {
        'Period': name,
        'Rows': len(frame),
        'Predictor quarters': frame['date'].nunique(),
        'Banks': frame['CERT'].nunique(),
        'First predictor': frame['date'].min(),
        'Last predictor': frame['date'].max(),
        'Last outcome': frame['target_date'].max(),
    }


# The one-quarter gaps keep each split's outcomes behind the next split's inputs.
train = rows.loc[rows['date'].le('2022-09-30')].copy()
valid = rows.loc[
    rows['date'].between('2023-01-01', '2023-09-30')
].copy()
holdout_mask = rows['date'].between('2024-01-01', '2024-09-30')
holdout = rows.loc[holdout_mask].copy()

assert train['target_date'].max() < valid['date'].min()
assert valid['target_date'].max() < holdout['date'].min()
assert set(FEATURES).isdisjoint(
    {
        'growth',
        'log_growth',
        'next_deposits',
        'target_date',
        'dollar_change_m',
    }
)

split_summary = pd.DataFrame(
    [
        describe_split('Training', train),
        describe_split('Validation', valid),
        describe_split('Reused holdout', holdout),
    ]
)
split_summary = ordered_rows(
    split_summary,
    ['Period'],
    category_orders={
        'Period': ['Training', 'Validation', 'Reused holdout'],
    },
)
table(split_summary, 'Accounting dates define this retrospective experiment')

fig = go.Figure()
period_colors = {
    'Training': '#0B6F75',
    'Validation': '#A86223',
    'Reused holdout': '#3F6294',
}

# Named columns keep bank counts out of the calendar axis.
for record in split_summary.to_dict('records'):
    period = record['Period']
    dates = rows.loc[
        rows['date'].between(record['First predictor'], record['Last predictor']), 'date'
    ].drop_duplicates().sort_values()
    fig.add_trace(go.Scatter(
        x=dates, y=[period] * len(dates), mode='lines+markers', name=period,
        line={'width': 7, 'color': period_colors[period]},
        marker={'size': 8},
        hovertemplate='%{x|%b %Y}<br>Predictor report<extra>%{fullData.name}</extra>',
    ))
    fig.add_annotation(x=record['First predictor'], y=period, yshift=26,
        text=f"{record['Predictor quarters']} quarters · {record['Rows']:,} examples",
        showarrow=False, xanchor='left' if period == 'Training' else 'right')
fig.update_xaxes(title='Predictor report date', type='date', dtick='M12', tickformat='%Y',
                 range=['2013-01-01', '2025-03-01'])
fig.update_yaxes(categoryorder='array', categoryarray=['Reused holdout','Validation','Training'])
fig.update_layout(showlegend=False)
chart(fig, 'split', '38 quarters teach the model; later quarters select and evaluate it',
      'Each dot is a predictor quarter. December boundary gaps keep outcomes in their assigned stage.')

split_summary.to_csv(OUT / 'splits.csv', index=False)
takeaway(
    'Fit on the past; judge on later reports',
    (
        'Training rows set imputation and scaling. Validation chooses Ridge '
        'strength and training duration. The reused 2024 holdout measures the '
        'previously selected design. Later research choices were informed by viewing 2024.'
    ),
)
''')
section('5 · Dollars, percentages, or logs: what are we asking?', '''
**Choose the meaning before looking for a better score.** Start with USD 100 million in deposits.
If next quarter has USD 90 million, the change is −USD 10 million, growth is −10%, and log growth is about −0.105.
Each describes the same observation. Each asks the model to care about something different.

Dollar change emphasizes funding magnitude. Percentage growth measures movement relative to the bank’s own base.
Log growth describes multiplicative movement and requires both deposit balances to be positive.
A zero next balance is −100% growth but has no finite log growth.

The paired histograms show the middle 98% of the training distribution; their titles
keep the full-data maxima visible. The next scatter shows what tiny starting balances
do to ordinary percentage growth. These displays use the 2013–2022 training period.
''',r'''
# Work one small change by hand before comparing training targets.
question_card(
    title='What should one prediction mean?',
    theme=theme,
    body=(
        'Compare dollars, ordinary percentage growth, and log growth on training '
        'rows before choosing the target.'
    ),
    kicker='Target choice',
    chip_text='QUESTION',
)

wm_formula_card(
    title='From a balance to a change',
    theme=theme,
    items=[
        {
            'label': 'Signed growth',
            'fallback': '(90 million − 100 million) / 100 million = −0.10 = −10%',
        },
        {
            'label': 'Log growth',
            'fallback': 'ln(90 / 100) ≈ −0.1054',
        },
    ],
)


def summarize_target(series, label, multiplier=1):
    """Return a compact full-range summary without trimming observations."""
    values = series.dropna().mul(multiplier)
    quantiles = values.quantile([0, 0.01, 0.50, 0.99, 1.00])

    return {
        'Target': label,
        'Defined rows': len(values),
        'Undefined rows': len(series) - len(values),
        'Minimum': quantiles.loc[0.00],
        '1st percentile': quantiles.loc[0.01],
        'Median': quantiles.loc[0.50],
        '99th percentile': quantiles.loc[0.99],
        'Maximum': quantiles.loc[1.00],
    }


def central_target_view(values):
    """Return the untouched values inside a display-only 1st–99th-percentile window."""
    low, high = values.quantile([0.01, 0.99])
    return values.loc[values.between(low, high)]


# %% NOTEBOOK CELL
# All three summaries use only training rows. The model choice comes later.
target_specs = [
    ('Dollar change (USD million)', 'dollar_change_m', 1),
    ('Signed growth (%)', 'growth', 100),
    ('Log growth', 'log_growth', 1),
]

target_summary = []
for target_label, column, multiplier in target_specs:
    target_summary.append(
        summarize_target(
            train[column],
            target_label,
            multiplier=multiplier,
        )
    )

ordinary_percent = 100 * train['growth']
log_values = train['log_growth']
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        f'Ordinary growth · max {ordinary_percent.max():,.0f}%',
        f'Log growth · max {log_values.max():.2f}',
    ],
    horizontal_spacing=0.16,
)
for column, values, color in [
    (1, ordinary_percent, '#A86223'),
    (2, log_values, '#0B6F75'),
]:
    central = central_target_view(values)
    fig.add_trace(
        go.Histogram(x=central, nbinsx=42, marker_color=color, showlegend=False),
        row=1,
        col=column,
    )
fig.update_xaxes(title='Next-quarter growth (%)', row=1, col=1)
fig.update_xaxes(title='Log growth', row=1, col=2)
fig.update_yaxes(title='Bank-quarter count', row=1, col=1)
fig.update_layout(bargap=0.02)
chart(
    fig,
    'target_growth_side_by_side',
    'One change, two very different numerical scales',
    'Training rows; each panel shows its 1st–99th-percentile window; maxima use the full data',
    height=570,
)

# %% NOTEBOOK CELL
# Tiny starting balances create enormous ordinary percentages. Sample the
# background for display speed, then include and label every top-five case.
positive_growth = train.loc[train['growth'].gt(0)].copy()
largest_growth = positive_growth.nlargest(5, 'growth')
background = positive_growth.drop(index=largest_growth.index).sample(
    n=min(8000, len(positive_growth) - len(largest_growth)),
    random_state=SEED,
)
fig = go.Figure()
fig.add_trace(go.Scattergl(
    x=background['DEPDOM'] / 1000,
    y=100 * background['growth'],
    mode='markers',
    name='Other positive-growth rows',
    marker=dict(color='#A5B5BC', size=5, opacity=0.22),
    hovertemplate='Starting deposits: $%{x:,.2f}M<br>Growth: %{y:,.1f}%<extra></extra>',
))
fig.add_trace(go.Scatter(
    x=largest_growth['DEPDOM'] / 1000,
    y=100 * largest_growth['growth'],
    mode='markers',
    text=largest_growth['NAME'].str.title(),
    name='Five largest percentages',
    marker=dict(color='#A86223', size=11),
    hovertemplate='%{text}<br>Starting deposits: $%{x:,.2f}M<br>Growth: %{y:,.1f}%<extra></extra>',
))

# Leader lines separate five unusually close labels without hiding the dots.
short_names = [
    'Schwab Signature',
    'Ameriprise',
    'Mitsubishi UFJ',
    'Wells Fargo Financial',
    'Stifel Trust',
]
label_offsets = [(135, -55), (45, -92), (-55, -60), (165, 5), (-35, 48)]
for (_, bank), name, (x_offset, y_offset) in zip(
    largest_growth.iterrows(), short_names, label_offsets
):
    fig.add_annotation(
        # Plotly annotations use log coordinates on logarithmic axes.
        x=np.log10(bank['DEPDOM'] / 1000),
        y=np.log10(100 * bank['growth']),
        text=name,
        ax=x_offset,
        ay=y_offset,
        arrowhead=2,
        arrowsize=.6,
        arrowwidth=1,
        arrowcolor='#A86223',
        font={'size': 12, 'color': '#6B4A1E'},
        bgcolor='rgba(255,255,255,.9)',
    )
fig.update_xaxes(type='log', title='Starting domestic deposits (USD million, log scale)')
fig.update_yaxes(type='log', title='Positive next-quarter growth (%, log scale)')
chart(
    fig,
    'small_denominator_growth',
    'Tiny starting balances magnify percentage growth',
    f'{len(background):,} sampled background rows plus the five largest; negative and zero growth remain in the experiment',
    height=640,
)

# %% NOTEBOOK CELL
# The table is an exact-value receipt after the two visual explanations.
target_summary = pd.DataFrame(target_summary)
target_summary = ordered_rows(
    target_summary,
    ['Target'],
    category_orders={
        'Target': [
            'Dollar change (USD million)',
            'Signed growth (%)',
            'Log growth',
        ],
    },
)
table(
    target_summary[['Target', '1st percentile', 'Median', '99th percentile', 'Maximum']],
    'Training targets: full-range arithmetic',
    {
        column: '{:,.4f}'
        for column in [
            'Minimum',
            '1st percentile',
            'Median',
            '99th percentile',
            'Maximum',
        ]
    },
)
target_summary.to_csv(OUT / 'target_comparison.csv', index=False)

wm_counterintuitive_card(
    title='What a novice might overlook',
    theme=theme,
    why_misread=(
        'Percentage growth sounds fair because every change is divided by the '
        'bank’s own starting balance.'
    ),
    ordinary_process=(
        'A very small starting balance can turn an ordinary dollar increase into '
        'a percentage change of thousands of percent. The training maximum exceeds '
        f'{train["growth"].max():,.0%}.'
    ),
    conclusion_boundary=(
        'Log growth keeps the proportional interpretation while compressing those '
        'multiplicative jumps. The model will predict log growth, and evaluation '
        'will convert predictions back to ordinary percentage points.'
    ),
    kicker='Target interpretation',
    chip_text='LOOK TWICE',
)

# %% NOTEBOOK CELL
# Preserve the largest changes for audit. Extreme does not mean erroneous.
extreme_columns = [
    'CERT',
    'NAME',
    'date',
    'target_date',
    'DEPDOM',
    'next_deposits',
    'growth',
    'next_name',
    'next_event',
]
extreme_index = train['growth'].abs().nlargest(6).index
extremes = train.loc[extreme_index, extreme_columns].copy()
extremes['Name changed'] = extremes['NAME'].ne(extremes['next_name'])

shown_extremes = extremes[
    [
        'CERT',
        'NAME',
        'date',
        'DEPDOM',
        'next_deposits',
        'growth',
        'Name changed',
    ]
]
shown_extremes = shown_extremes.assign(
    _absolute_growth=shown_extremes['growth'].abs()
).sort_values(
    ['_absolute_growth', 'CERT'],
    ascending=[False, True],
).drop(columns='_absolute_growth')
shown_extremes = shown_extremes.drop(
    columns=['CERT', 'Name changed']
).rename(columns={
    'NAME': 'Bank',
    'date': 'Report date',
    'DEPDOM': 'Start (USD k)',
    'next_deposits': 'Next (USD k)',
    'growth': 'Growth',
})
table(
    shown_extremes,
    'Six largest training changes · balances in USD thousands',
    {
        'Start (USD k)': '{:,.0f}',
        'Next (USD k)': '{:,.0f}',
        'Growth': '{:+.2%}',
    },
    wrap_columns={'Bank': 280},
)

extremes.to_csv(OUT / 'extreme_review.csv', index=False)
panel.loc[
    panel['CERT'].isin(extremes['CERT']),
    ['CERT', 'REPDTE', 'NAME', 'date', 'DEPDOM', 'ASSET', 'ACTEVT', 'CALLFORM'],
].to_csv(OUT / 'extreme_bank_histories.csv', index=False)

wm_formula_card(
    title='The same bank-quarter, a learnable scale',
    theme=theme,
    items=[
        {'label': 'Starting deposits', 'fallback': 'USD 1.15 million'},
        {'label': 'Next quarter', 'fallback': 'USD 6.01 billion'},
        {'label': 'Ordinary growth', 'fallback': '(6,011,577 − 1,150) / 1,150 = 522,645.83%'},
        {'label': 'Training target', 'fallback': 'y = ln(D next / D now) = 8.56'},
        {'label': 'Interpret a forecast', 'fallback': 'ordinary growth (%) = 100 × (exp(predicted y) − 1)'},
    ],
)

TARGET = 'log_growth'
target_decision = {
    'target': TARGET,
    'model_units': 'log change',
    'reported_error_units': 'ordinary percentage points',
    'reason': (
        'The 2013–2022 training distribution contains percentage growth above '
        '5,000x because some starting balances are tiny. Log growth preserves '
        'multiplicative movement and keeps those rows in the experiment.'
    ),
    'clipping': None,
    'extreme_treatment': 'Retain and audit the surrounding reports.',
    'training_only': True,
}
(OUT / 'target_decision.json').write_text(json.dumps(target_decision, indent=2))

takeaway(
    'Train on log growth; translate forecasts back to percentage growth',
    (
        'The decision comes from the training distribution and the meaning of the '
        'target. No row with valid positive current and next-quarter deposits '
        'is removed simply because its growth is extreme. A same-name report '
        'does not establish why a bank’s balance changed.'
    ),
)
''')
section('6 · What does a network actually learn?', '''
**It adjusts numbers so its predictions make smaller mistakes.** Start with one neuron:
multiply each input by a weight, add the results, then add a bias.
With inputs 2 and 3, weights 0.5 and −0.2, and bias 0.1, the output is 0.5.
ReLU keeps positive outputs and replaces negative outputs with zero.

One layer creates combinations of the five financial inputs. A second layer combines those learned features.
The final layer produces one signed growth forecast. A linear output allows negative growth.

**Why two hidden layers?** They let the relationship bend. Whether those bends help is an empirical question.
Ridge gives us the same inputs with an additive linear relationship as the comparison.

Training repeats four steps: predict → measure squared error → calculate gradients → update weights.
A gradient tells us how a small weight change affects loss. Adam uses those gradients to update the weights.
A batch of 512 rows gives one update. An epoch is one pass through training rows.
Early stopping keeps the weights from the best validation-loss epoch.
''',r'''
# Start with one neuron before showing the full network.
question_card(
    title='What changes when a neural network learns?',
    theme=theme,
    body='Follow one weighted sum, one ReLU, and one gradient step before looking at the full 5–32–16–1 architecture.',
    kicker='Network mechanics',
    chip_text='QUESTION',
)

# Use tiny numbers first. The full network repeats this same arithmetic across
# many connected weights.
wm_formula_card(title='One neuron, one visible calculation',theme=theme,items=[
    {'label':'Weighted sum','fallback':'z = 2 × 0.5 + 3 × (−0.2) + 0.1 = 0.5'},
    {'label':'ReLU','fallback':'max(0, z) = 0.5'},
    {'label':'Squared error','fallback':'If the answer is 0.8: (0.5 − 0.8)² = 0.09'}])
# This hand-checkable update teaches the gradient mechanics.
x_demo = np.array([2.0, 3.0])
w_demo = np.array([0.5, -0.2])
bias_demo = 0.1
answer_demo = 0.8
pred_demo=x_demo@w_demo+bias_demo
error_demo=pred_demo-answer_demo
w_after=w_demo-.01*(2*error_demo*x_demo)
bias_after=bias_demo-.01*(2*error_demo)
new_error=(x_demo@w_after+bias_after-answer_demo)**2
assert new_error<error_demo**2
table(pd.DataFrame({'Stage':['Before one update','After one update'],
    'Squared error':[error_demo**2,new_error]}),'A gradient step we can check by hand',{'Squared error':'{:.6f}'})
fig=go.Figure(go.Scatter(x=[0,1,2,3],y=[0,0,0,0],mode='lines+markers+text',
    text=['5 financial inputs','32 ReLU units','16 ReLU units','1 linear output'],textposition='top center',
    marker=dict(size=22,color='#0B6F75'),line=dict(color='#3F6294')))
fig.update_xaxes(visible=False, range=[-0.5, 3.5])
fig.update_yaxes(visible=False, range=[-0.3, 0.5])
chart(fig,'architecture','Five inputs become one growth forecast','737 trainable weights and biases',height=380)
''')
section('7 · Put the five inputs on comparable scales', '''
**The five inputs use very different units.** Scaling puts a deposit balance and an equity ratio on comparable numerical footing.
Standardization subtracts a training mean and divides by a training standard deviation.
A missing predictor gets its training median. Validation and historical rows reuse those exact values.

Before transforming anything, inspect the five inputs on the training period. The profile cards show
their shapes and missing values. A box plot compares cash ratios for banks whose deposits later fell
and banks whose deposits did not. The quarterly line checks whether declines cluster in time.
These are descriptions of past rows, not evidence that cash caused a decline. The correlation view
asks whether inputs repeat information; the prior-versus-next scatter checks recent-history signal.

Our comparison ladder has four rungs: zero growth; repeat last quarter’s growth; Ridge; the small MLP.
Ridge strength is selected from a fixed grid using validation MAE. The MLP stops on validation MSE.
MAE describes the average absolute miss. RMSE gives large misses extra weight.
A forecast of +2% when reality is −3% misses by **5 percentage points**.
''',r'''
# Give each input a name, calculation, and reporting date.
question_card(
    title='What do the five inputs look like before preprocessing?',
    theme=theme,
    body=(
        'Inspect their ranges, correlations, and the relationship between prior '
        'and next-quarter growth using training rows only.'
    ),
    kicker='Model inputs',
    chip_text='QUESTION',
)

# Begin with a plain-language map from FDIC fields to model inputs.
feature_ledger = pd.DataFrame(
    {
        'Input': FEATURES,
        'Calculation': [
            'ln(DEPDOM)',
            'CHBAL / ASSET',
            'LNLSNET / ASSET',
            'EQ / ASSET',
            'ln(DEPDOM / prior DEPDOM)',
        ],
        'Timing': [
            'Current quarter',
            'Current quarter',
            'Current quarter',
            'Current quarter',
            'Current and adjacent prior quarter',
        ],
    }
)
table(feature_ledger, 'Five inputs; no future balance')

# Describe every feature before imputation or scaling changes its units.
feature_summary = (
    train[FEATURES]
    .describe(percentiles=[0.01, 0.25, 0.50, 0.75, 0.99])
    .T
    .reset_index(names='Feature')
)
feature_summary = feature_summary[
    ['Feature', 'count', 'mean', '1%', '50%', '99%', 'max']
]
feature_summary['Feature'] = feature_summary['Feature'].map({
    'log_deposits': 'Bank size',
    'cash_ratio': 'Cash / assets',
    'loan_ratio': 'Loans / assets',
    'equity_ratio': 'Equity / assets',
    'prior_growth': 'Prior growth',
})
table(
    feature_summary,
    'What do the five training inputs look like?',
    {'count': '{:,.0f}', **{
        column: '{:,.4f}'
        for column in ['mean', '1%', '50%', '99%', 'max']
    }},
    wrap_columns={'Feature': 180},
)

# The notebook's original WM profile rail is the visual companion to describe().
# Each card keeps missingness and skew beside the field's typical value.
wm_render_micro_profile_cards(
    train[FEATURES],
    theme=theme,
    columns=FEATURES,
    visible_cards=5,
    max_cards=5,
    skew_threshold=1.0,
)

# %% NOTEBOOK CELL
# Compare cash ratios by what actually happened next quarter. Both groups
# remain in the training data; a difference here is an association.
cash_groups = train[['cash_ratio', 'growth']].copy()
cash_groups['Next quarter'] = np.where(
    cash_groups['growth'].lt(0),
    'Deposits fell',
    'No decline',
)
cash_group_summary = (
    cash_groups.groupby('Next quarter', sort=False)['cash_ratio']
    .agg(Rows='count', Median='median', Q1=lambda s: s.quantile(.25),
         Q3=lambda s: s.quantile(.75))
    .reset_index()
)
cash_comparison = wm_compare_fields(
    cash_groups.drop(columns='growth'),
    fields=['cash_ratio', 'Next quarter'],
    kind='numeric_by_category',
)
cash_comparison.figure.update_traces(boxpoints=False)
cash_comparison.figure.update_layout(showlegend=False)
cash_window = float(train['cash_ratio'].quantile(.99))
cash_comparison.figure.update_xaxes(
    title='Cash / assets',
    tickformat='.0%',
    range=[0, cash_window],
)
chart(
    cash_comparison.figure,
    'cash_by_outcome_box',
    'Banks with later declines held a higher median cash ratio',
    f'Training rows; view ends at the 99th percentile ({cash_window:.1%}); boxes use all rows',
)
table(
    cash_group_summary,
    'Cash-ratio quartiles · exact values',
    {'Median': '{:.1%}', 'Q1': '{:.1%}', 'Q3': '{:.1%}'},
)

# %% NOTEBOOK CELL
# A time line tests whether one period drives the overall decline rate.
quarterly_declines = (
    train.assign(Decline=train['growth'].lt(0))
    .groupby('date', as_index=False)
    .agg(Rows=('CERT', 'size'), Declines=('Decline', 'sum'))
    .sort_values('date')
)
quarterly_declines['Decline share'] = (
    quarterly_declines['Declines'] / quarterly_declines['Rows']
)
fig = px.line(
    quarterly_declines,
    x='date',
    y='Decline share',
    markers=True,
)
fig.update_traces(line=dict(color='#A86223', width=3), marker=dict(size=5))
fig.update_yaxes(title='Bank-quarters with deposit decline', tickformat='.0%')
fig.update_xaxes(title='Predictor quarter')
chart(
    fig,
    'decline_share_time',
    'Deposit declines became much less common in early 2020',
    'Training period only; next-quarter decline divided by eligible rows',
)

# Keep each year visible when checking whether the quarterly rhythm repeats.
seasonality = quarterly_declines.copy()
seasonality['Year'] = seasonality['date'].dt.year
seasonality['Quarter'] = 'Q' + seasonality['date'].dt.quarter.astype(str)
# A small fixed offset exposes individual years that would otherwise overlap.
seasonality['Quarter position'] = (
    seasonality['date'].dt.quarter
    + (seasonality['Year'] - seasonality['Year'].mean()) * .025
)
seasonality.to_csv(OUT / 'training_seasonality.csv', index=False)

fig = px.scatter(
    seasonality,
    x='Quarter position',
    y='Decline share',
    color='Year',
    hover_data=['Quarter', 'date', 'Rows', 'Declines'],
    color_continuous_scale='Teal',
)
fig.update_traces(marker={'size': 10, 'opacity': .8})
fig.update_yaxes(title='Bank-quarters with deposit decline', tickformat='.0%')
fig.update_xaxes(
    title='Predictor quarter of year',
    tickvals=[1, 2, 3, 4],
    ticktext=['Q1', 'Q2', 'Q3', 'Q4'],
    range=[.7, 4.3],
)
chart(
    fig,
    'training_seasonality',
    'Declines were more common after Q1; individual years varied',
    'Each dot is one training year; descriptive check only',
)
seasonal_medians = seasonality.groupby('Quarter')['Decline share'].median()
takeaway(
    'The usual Q1–Q4 rhythm breaks in 2020',
    f"The training-year median decline rate is {seasonal_medians['Q1']:.1%} "
    f"after Q1 reports and {seasonal_medians['Q4']:.1%} after Q4 reports. "
    'The 2020 points show why this remains descriptive. Quarter of year is not a model input.',
)

# Correlation answers whether the five fixed inputs repeat the same information.
friendly_names = [
    'Bank size',
    'Cash / assets',
    'Loans / assets',
    'Equity / assets',
    'Prior deposit growth',
]
correlations = train[FEATURES].corr(method='spearman')
lower_triangle = correlations.mask(
    np.triu(np.ones_like(correlations, dtype=bool))
)
heat_text = np.where(
    lower_triangle.notna(),
    lower_triangle.round(2).astype(str),
    '',
)

fig = go.Figure(
    go.Heatmap(
        z=lower_triangle.to_numpy(),
        x=friendly_names,
        y=friendly_names,
        zmin=-1,
        zmax=1,
        colorscale=[
            [0, '#A86223'],
            [0.5, '#F4F6F7'],
            [1, '#0B6F75'],
        ],
        text=heat_text,
        texttemplate='%{text}',
        hovertemplate=(
            '%{y} vs %{x}<br>Spearman %{z:.3f}<extra></extra>'
        ),
        colorbar_title='Spearman',
    )
)
fig.update_yaxes(autorange='reversed')
chart(
    fig,
    'feature_correlations',
    'Pairwise rank correlations are modest across all five inputs',
    'Training rows only; lower triangle; Spearman correlation',
)

# A display-only zoom keeps small-balance jumps from flattening the scatter.
sample_source = train[['prior_growth', 'log_growth']].dropna()
sample = sample_source.sample(
    n=min(5000, len(sample_source)),
    random_state=SEED,
)

x_lower, x_upper = sample_source['prior_growth'].quantile([0.01, 0.99])
y_lower, y_upper = sample_source['log_growth'].quantile([0.01, 0.99])
visible = (
    sample['prior_growth'].between(x_lower, x_upper)
    & sample['log_growth'].between(y_lower, y_upper)
)

fig = px.scatter(
    sample.loc[visible],
    x='prior_growth',
    y='log_growth',
    opacity=0.20,
    color_discrete_sequence=['#0B6F75'],
)
fig.add_hline(y=0, line_color='#627381')
fig.add_vline(x=0, line_color='#627381')
fig.update_xaxes(title='Prior-quarter log growth')
fig.update_yaxes(title='Next-quarter log growth')
chart(
    fig,
    'persistence_eda',
    'Similar prior growth leads to widely different next-quarter growth',
    (
        'Random training sample; middle 98% chart window; '
        f'{len(sample) - visible.sum():,} sampled points outside the view'
    ),
)


# %% NOTEBOOK CELL
# Learn missing-value replacements and scales from training rows only.
def fit_preprocessor(training_frame):
    """Fit median imputation and standardization on training rows only."""
    fitted_imputer = SimpleImputer(strategy='median')
    fitted_scaler = StandardScaler()

    imputed = fitted_imputer.fit_transform(training_frame[FEATURES])
    fitted_scaler.fit(imputed)

    return fitted_imputer, fitted_scaler


def transform_features(frame, fitted_imputer, fitted_scaler):
    """Apply the frozen preprocessing steps to one time split."""
    imputed = fitted_imputer.transform(frame[FEATURES])
    transformed = fitted_scaler.transform(imputed)
    return transformed.astype('float32')


imputer, scaler = fit_preprocessor(train)
X_train = transform_features(train, imputer, scaler)
X_valid = transform_features(valid, imputer, scaler)
y_train = train[TARGET].to_numpy(dtype='float32')
y_valid = valid[TARGET].to_numpy(dtype='float32')

assert np.isfinite(X_train).all()
assert np.isfinite(X_valid).all()
assert np.allclose(imputer.statistics_, train[FEATURES].median())

preprocessing = pd.DataFrame(
    {
        'Feature': FEATURES,
        'Training median': imputer.statistics_,
        'Training mean after imputation': scaler.mean_,
        'Training scale': scaler.scale_,
    }
)
table(preprocessing, 'Parameters learned only from training')
preprocessing.to_csv(OUT / 'preprocessing.csv', index=False)

# Freeze every choice that could otherwise drift after seeing 2024.
settings = {
    'target': target_decision,
    'features': FEATURES,
    'seed': 42,
    'additional_seeds': [7, 99],
    'architecture': [5, 32, 16, 1],
    'optimizer': 'Adam',
    'learning_rate': 0.001,
    'loss': 'MSE on log growth',
    'batch_size': 512,
    'epochs_max': 200,
    'early_stopping_patience': 10,
    'ridge_alphas': [0.01, 0.1, 1, 10, 100],
    'ridge_selection': 'validation MAE in percentage points',
    'historical_status': 'reused 2024 holdout',
    'clipping': None,
    'train_predictor_end': '2022-09-30',
    'validation_predictors': ['2023-01-01', '2023-09-30'],
    'holdout_predictors': ['2024-01-01', '2024-09-30'],
    'tail_definition': (
        'realized bottom 25% and 10% separately within each holdout quarter'
    ),
    'ranking': (
        'lowest ceil(10% of rows) per quarter; ties broken by CERT'
    ),
    'bootstrap': (
        '500 paired bank-cluster resamples; MAE difference MLP minus Ridge; '
        'fixed trained models'
    ),
    'eligibility': list(conditions),
    'source_sha256': source_hash,
}
(OUT / 'design_plan.json').write_text(json.dumps(settings, indent=2))


# %% NOTEBOOK CELL
# Tune the linear comparison on validation data.
def choose_ridge(X_fit, y_fit, X_check, y_check, alphas):
    """Select Ridge strength using validation MAE in percentage points."""
    rows = []
    models = {}

    for alpha in alphas:
        candidate = Ridge(alpha=alpha).fit(X_fit, y_fit)
        models[alpha] = candidate
        rows.append(
            {
                'Alpha': alpha,
                **log_scores(y_check, candidate.predict(X_check)),
            }
        )

    results = pd.DataFrame(rows)
    chosen_alpha = float(
        results.sort_values(['MAE (pp)', 'Alpha']).iloc[0]['Alpha']
    )
    return models[chosen_alpha], chosen_alpha, results


ridge, best_alpha, ridge_tuning = choose_ridge(
    X_train,
    y_train,
    X_valid,
    y_valid,
    settings['ridge_alphas'],
)


# %% NOTEBOOK CELL
# Build one fixed neural-network architecture, then let early stopping choose
# how long it trains.
def build_network(seed):
    """Create the fixed 5 → 32 → 16 → 1 neural network."""
    tf.keras.utils.set_random_seed(seed)

    network = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(5,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1),
        ]
    )
    network.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='mse',
    )

    assert network.count_params() == 737
    return network


def fit_network(seed):
    """Train one fixed network and restore its best validation weights."""
    network = build_network(seed)

    options = tf.data.Options()
    options.threading.private_threadpool_size = 2

    training_data = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(y_train), seed=seed)
        .batch(512)
        .with_options(options)
    )
    validation_data = (
        tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
        .batch(512)
        .with_options(options)
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
    )

    started = time.perf_counter()
    fitted = network.fit(
        training_data,
        validation_data=validation_data,
        epochs=200,
        callbacks=[early_stopping],
        verbose=0,
        shuffle=False,
    )
    elapsed = time.perf_counter() - started

    return network, fitted.history, elapsed


def predict_log_growth(model, features):
    """Return a flat NumPy array from a Keras model."""
    return np.asarray(model(features, training=False)).ravel()


# %% NOTEBOOK CELL
# Fit once with the frozen seed; validation decides when to stop.
mlp, history, training_seconds = fit_network(SEED)

validation_predictions = {
    'Zero growth': np.zeros(len(valid)),
    'Persistence': valid['prior_growth'].to_numpy(),
    'Ridge': ridge.predict(X_valid),
    'MLP': predict_log_growth(mlp, X_valid),
}
validation_scores = pd.DataFrame(
    [
        {
            'Model': model_name,
            **log_scores(y_valid, prediction),
        }
        for model_name, prediction in validation_predictions.items()
    ]
)
validation_scores = ordered_rows(
    validation_scores,
    ['Model'],
    category_orders={'Model': MODEL_ORDER},
)

table(
    validation_scores,
    'Validation chooses; historical evaluation waits',
    {'MAE (pp)': '{:.4f}', 'RMSE (pp)': '{:.4f}'},
)

fig = px.scatter(
    validation_scores,
    x='MAE (pp)',
    y='Model',
    color='Model',
    color_discrete_map=MODEL_COLORS,
)
fig.update_layout(showlegend=False)
chart(
    fig,
    'validation',
    'Validation errors compare every method on the same bank-quarters',
    '2023 Q1–Q3 predictor rows; lower MAE is better',
)

best_epoch = int(np.argmin(history['val_loss']) + 1)
frozen = {
    **settings,
    'chosen_ridge_alpha': best_alpha,
    'mlp_best_epoch': best_epoch,
    'validation_winner': (
        validation_scores.sort_values('MAE (pp)').iloc[0]['Model']
    ),
    'selection_complete_before_holdout_predictions': True,
}
(OUT / 'frozen_plan.json').write_text(json.dumps(frozen, indent=2))
validation_scores.to_csv(OUT / 'validation_scores.csv', index=False)
ridge_tuning.to_csv(OUT / 'ridge_tuning.csv', index=False)

# %% NOTEBOOK CELL
# Inspect which observed outcomes dominate validation MSE.
history_frame = (
    pd.DataFrame(history)
    .rename_axis('Epoch')
    .reset_index()
)
history_frame['Epoch'] += 1
history_frame.to_csv(OUT / 'learning_curve.csv', index=False)

validation_audit = valid[
    ['CERT', 'NAME', 'date', 'DEPDOM', 'next_deposits', 'growth', 'next_name']
].copy()
validation_mlp_growth = np.expm1(validation_predictions['MLP'])
validation_audit['Squared MLP error'] = (
    validation_audit['growth'].to_numpy() - validation_mlp_growth
) ** 2
validation_audit['Share of squared error'] = (
    validation_audit['Squared MLP error']
    / validation_audit['Squared MLP error'].sum()
)
validation_audit.nlargest(10, 'Squared MLP error').to_csv(
    OUT / 'validation_extremes.csv',
    index=False,
)

table(
    validation_audit.nlargest(3, 'Squared MLP error')[
        [
            'CERT',
            'NAME',
            'date',
            'DEPDOM',
            'next_deposits',
            'growth',
            'Share of squared error',
        ]
    ],
    'One extreme balance jump dominates validation squared error',
    {'growth': '{:+.1%}', 'Share of squared error': '{:.1%}'},
)

worst = validation_audit.nlargest(1, 'Squared MLP error').iloc[0]
takeaway(
    'One extreme outcome can dominate MSE',
    (
        f'{worst.NAME} at {worst.date:%Y-%m-%d} contributes '
        f'{worst["Share of squared error"]:.1%} of validation squared error. '
        f'Its growth is {worst.growth:.1%}. The source history remains available '
        'for review, and the frozen model stays unchanged.'
    ),
)

# %% NOTEBOOK CELL
# The learning curve shows the validation-selected training duration.
fig = go.Figure()
for column, label, color, dash in [
    ('loss', 'Training', '#0B6F75', 'solid'),
    ('val_loss', 'Validation', '#A86223', 'dash'),
]:
    fig.add_trace(
        go.Scatter(
            x=history_frame['Epoch'],
            y=history_frame[column],
            name=label,
            line={'color': color, 'dash': dash},
        )
    )

fig.update_xaxes(title='Epoch')
fig.update_yaxes(title='Mean squared log-growth error', type='log')
chart(
    fig,
    'learning',
    'Validation loss determines when neural-network training stops',
    'Logarithmic loss axis; best validation weights restored',
)

takeaway(
    'Training duration comes from validation',
    (
        f'The MLP restored epoch {best_epoch} after {len(history_frame)} epochs. '
        f'Training took {training_seconds:.1f} seconds on this run. The 2024 '
        'period played no role in choosing the weights.'
    ),
)
''')
section('8 · Does the neural network improve the forecast?', '''
**Now open the frozen historical evaluation.** Every model predicts the same bank-quarters.
The two dot panels carry the exact scores; the full score table is saved as a CSV.
The scatter asks a different question: do individual predictions move with reality?

**Two scores can disagree without either being wrong.** MAE averages the size of each miss.
RMSE squares each miss first, so a few large misses exert more influence. After the
scorecard, we will group errors by size to see where the MLP helped and hurt.

**What would make us investigate a bank?** A 4.3% improvement in an aggregate
error score cannot answer that. We need an advance ranking at a stated review
capacity, followed by a count of how many selected banks actually had weak
growth. Even a useful ranking is a lead for an analyst, not evidence of a
problem at any particular bank.

The masterclass tests a pretrained sequence model after the core conclusion.
That extension uses different inputs and does not replace this five-feature experiment.

On the diagonal, prediction equals outcome. Above it, the model predicts too much growth.
Below it, the model predicts too little. Both axes use the same units and scale.
Every eligible evaluation row remains in the scores and saved predictions.
''',r'''
# Compare the four forecasts on the same later bank-quarters.
question_card(
    title='Did the MLP reduce error on later quarters?',
    theme=theme,
    body=(
        'Score every model on the same bank-quarters. Read MAE and RMSE '
        'together because they reward different error behavior.'
    ),
    kicker='Historical evaluation',
    chip_text='QUESTION',
)

# Open the reused holdout after every modeling choice has been frozen.
test = rows.loc[holdout_mask].copy()
X_test = transform_features(test, imputer, scaler)
y_test = test[TARGET].to_numpy()

log_predictions = {
    'Zero growth': np.zeros(len(test)),
    'Persistence': test['prior_growth'].to_numpy(),
    'Ridge': ridge.predict(X_test),
    'MLP': predict_log_growth(mlp, X_test),
}

for prediction in log_predictions.values():
    assert len(prediction) == len(test)
    assert np.isfinite(prediction).all()

metrics = pd.DataFrame(
    [
        {
            'Model': model_name,
            **log_scores(y_test, prediction),
        }
        for model_name, prediction in log_predictions.items()
    ]
)
metrics = ordered_rows(
    metrics,
    ['Model'],
    category_orders={'Model': MODEL_ORDER},
)
# %% NOTEBOOK CELL
# The same four models appear in both panels. A dot's position carries the
# score, while its label keeps small differences readable.
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        'Average absolute miss',
        'Large-error-sensitive miss',
    ],
    horizontal_spacing=0.22,
)

for column_number, metric_name in enumerate(
    ['MAE (pp)', 'RMSE (pp)'],
    start=1,
):
    metric_values = metrics.set_index('Model')[metric_name].reindex(MODEL_ORDER)
    winner_name = metric_values.idxmin()
    for model_name, value in metric_values.items():
        fig.add_trace(
            go.Scatter(
                x=[value],
                y=[model_name],
                mode='markers+text',
                marker={
                    'size': 14 if model_name == winner_name else 10,
                    'color': '#0B6F75' if model_name == winner_name else '#9AA6B2',
                },
                text=[f'{value:.3f} pp'],
                textposition='middle right',
                textfont={'color': '#26323A'},
                showlegend=False,
                hovertemplate=f'{model_name}<br>{metric_name}: {value:.4f} pp<extra></extra>',
            ),
            row=1,
            col=column_number,
        )
    fig.update_xaxes(
        title='Percentage points; lower is better',
        range=[max(0, metric_values.min() - .25), metric_values.max() + .9],
        row=1,
        col=column_number,
    )
    fig.update_yaxes(
        categoryorder='array',
        categoryarray=list(reversed(MODEL_ORDER)),
        row=1,
        col=column_number,
    )

chart(
    fig,
    'comparison',
    'Zero growth leads MAE; the neural network leads RMSE',
    'Zero growth has the lowest MAE; the MLP has the lowest RMSE among the four core models.',
    height=500,
)

# Check the large misses directly before interpreting the RMSE result.
large_error_check = pd.DataFrame([
    {
        'Model': model_name,
        '99th percentile absolute error (pp)': float(np.quantile(
            np.abs(100 * (np.expm1(y_test) - np.expm1(prediction))), .99
        )),
    }
    for model_name, prediction in log_predictions.items()
])
large_error_check.to_csv(OUT / 'large_error_quantiles.csv', index=False)
assert large_error_check.loc[
    large_error_check.Model.eq('MLP'),
    '99th percentile absolute error (pp)',
].iloc[0] < large_error_check.loc[
    large_error_check.Model.eq('Zero growth'),
    '99th percentile absolute error (pp)',
].iloc[0]

# %% NOTEBOOK CELL
# EXEMPLAR: analytical-question
question_card(
    title='How can the MLP win RMSE but lose MAE?',
    theme=theme,
    body=(
        'A 10-point miss contributes 10 to absolute error but 100 to '
        'squared error. Group the same 2024 rows by the zero-growth miss '
        'and see where the MLP traded accuracy.'
    ),
    kicker='Read the scorecard',
    chip_text='QUESTION',
)

# %% NOTEBOOK CELL
# Group by realized baseline error for diagnosis only. This grouping uses
# the future outcome and cannot be used to select a bank in advance.
zero_absolute_error = np.abs(
    100 * (np.expm1(y_test) - np.expm1(log_predictions['Zero growth']))
)
mlp_absolute_error = np.abs(
    100 * (np.expm1(y_test) - np.expm1(log_predictions['MLP']))
)

cutoffs = np.quantile(zero_absolute_error, [.90, .99])
error_bands = pd.DataFrame({
    'Zero-growth miss (pp)': zero_absolute_error,
    'MLP miss (pp)': mlp_absolute_error,
})
error_bands['Band'] = pd.cut(
    error_bands['Zero-growth miss (pp)'],
    bins=[-np.inf, *cutoffs, np.inf],
    labels=['Smallest 90%', 'Next 9%', 'Largest 1%'],
)

tradeoff = (
    error_bands.groupby('Band', observed=True)
    .agg(
        Rows=('Band', 'size'),
        Zero_growth_MAE_pp=('Zero-growth miss (pp)', 'mean'),
        MLP_MAE_pp=('MLP miss (pp)', 'mean'),
    )
    .reset_index()
)
tradeoff['MLP minus zero (pp)'] = (
    tradeoff['MLP_MAE_pp'] - tradeoff['Zero_growth_MAE_pp']
)
tradeoff.to_csv(OUT / 'error_tradeoff.csv', index=False)

fig = go.Figure()
fig.add_bar(
    x=tradeoff['MLP minus zero (pp)'],
    y=tradeoff['Band'].astype(str),
    orientation='h',
    marker_color=[
        '#A86223' if delta > 0 else '#0B6F75'
        for delta in tradeoff['MLP minus zero (pp)']
    ],
    text=[f'{delta:+.2f} pp' for delta in tradeoff['MLP minus zero (pp)']],
    textposition='outside',
    customdata=tradeoff['Rows'],
    hovertemplate=(
        '%{y}<br>MLP minus zero-growth MAE: %{x:+.3f} pp'
        '<br>%{customdata:,} bank-quarters<extra></extra>'
    ),
)
fig.add_vline(x=0, line_color='#343B43', line_width=1.5)
fig.update_xaxes(
    title='MLP minus zero-growth mean absolute miss (pp)',
    range=[-4.5, 1.2],
)
fig.update_yaxes(
    title='',
    categoryorder='array',
    categoryarray=['Largest 1%', 'Next 9%', 'Smallest 90%'],
)
chart(
    fig,
    'error_tradeoff',
    'The MLP helps most where zero growth misses most',
    'Teal left of zero: MLP helped. Amber right: MLP hurt. Outcome-defined groups.',
    height=480,
)

common = tradeoff.loc[tradeoff.Band.eq('Smallest 90%')].iloc[0]
largest = tradeoff.loc[tradeoff.Band.eq('Largest 1%')].iloc[0]
takeaway(
    'The MLP trades small misses for fewer large misses',
    (
        f'On {int(common.Rows):,} smaller-miss rows, its average absolute '
        f'error was {common["MLP minus zero (pp)"]:+.2f} pp relative to '
        f'zero growth. On the largest {int(largest.Rows):,} baseline misses, '
        f'it was {largest["MLP minus zero (pp)"]:+.2f} pp. '
        'These groups use realized outcomes, so this chart explains the '
        'scores after the fact; it cannot choose banks for review beforehand.'
    ),
)

rmse_so_far = metrics.set_index('Model')['RMSE (pp)']
relative_rmse_drop = 100 * (
    rmse_so_far['Zero growth'] - rmse_so_far['MLP']
) / rmse_so_far['Zero growth']
takeaway(
    'The RMSE improvement changes large-error scoring',
    (
        f'The MLP lowered RMSE by {relative_rmse_drop:.1f}% relative to zero '
        'growth, yet zero growth still had the lower MAE. This measures '
        'forecast error across many bank-quarters, not the chance that any '
        'one bank needs investigation. The next check asks whether predicted '
        'rankings put more truly weak-growth banks into a fixed-size review '
        'list. An analyst must still verify the source report and context.'
    ),
)

# %% NOTEBOOK CELL

# Convert log predictions back to ordinary growth for interpretation.
growth_predictions = {
    model_name: np.expm1(prediction)
    for model_name, prediction in log_predictions.items()
}

scatter = test[['CERT', 'NAME', 'date', 'growth']].copy()
scatter['Actual (%)'] = 100 * scatter['growth']
scatter['Predicted (%)'] = 100 * growth_predictions['MLP']

actual_limits = scatter['Actual (%)'].quantile([0.01, 0.99])
predicted_limits = scatter['Predicted (%)'].quantile([0.01, 0.99])
central_mask = (
    scatter['Actual (%)'].between(*actual_limits)
    & scatter['Predicted (%)'].between(*predicted_limits)
)
central_scatter = scatter.loc[central_mask].copy()

fig = px.scatter(
    central_scatter,
    x='Actual (%)',
    y='Predicted (%)',
    hover_data=['NAME', 'CERT', 'date'],
    opacity=0.30,
    color_discrete_sequence=['#0B6F75'],
)

lower_bound = min(
    central_scatter['Actual (%)'].min(),
    central_scatter['Predicted (%)'].min(),
)
upper_bound = max(
    central_scatter['Actual (%)'].max(),
    central_scatter['Predicted (%)'].max(),
)
span = max(upper_bound - lower_bound, 1)
bounds = [
    lower_bound - 0.04 * span,
    upper_bound + 0.04 * span,
]

fig.add_trace(
    go.Scatter(
        x=bounds,
        y=bounds,
        mode='lines',
        name='Ideal forecast',
        line={'color': '#343B43', 'dash': 'dash'},
    )
)
fig.update_xaxes(range=bounds)
fig.update_yaxes(
    range=bounds,
    scaleanchor='x',
    scaleratio=1,
)
chart(
    fig,
    'actual_predicted',
    'Most MLP forecasts stay close to zero growth',
    (
        f'Central 98% view; {len(scatter) - len(central_scatter):,} extreme '
        'rows remain in every score and table'
    ),
    height=700,
)

# Save one row per bank-quarter with ordinary-growth predictions.
result_frame = test[
    [
        'CERT',
        'NAME',
        'date',
        'target_date',
        'growth',
        'DEPDOM',
        'next_deposits',
    ]
].reset_index(drop=True)

for model_name, prediction in growth_predictions.items():
    result_frame[model_name] = prediction

result_frame.to_csv(OUT / 'predictions.csv', index=False)
metrics.to_csv(OUT / 'historical_scores.csv', index=False)

mae = metrics.set_index('Model')['MAE (pp)']
winner = mae.idxmin()
difference = float(mae['MLP'] - mae['Ridge'])

takeaway(
    'Zero growth wins MAE; the MLP wins RMSE',
    (
        f'Zero growth: {mae["Zero growth"]:.3f} pp MAE. '
        f'MLP: {mae["MLP"]:.3f} pp. '
        f'Ridge: {mae["Ridge"]:.3f} pp. '
        f'Persistence: {mae["Persistence"]:.3f} pp. '
        f'The MLP minus Ridge difference is {difference:+.3f} pp. '
        'The MLP reduces some large misses, but its typical absolute miss is '
        'larger than zero growth on this reused holdout.'
    ),
    f'{mae[winner]:.3f} pp',
)
''')
section('9 · Does the average hide weak quarters or weak outcomes?', '''
**A small average miss can coexist with poor forecasts when deposits fall sharply.**
First compare quarters. Then look at the realized bottom quarter and bottom tenth of growth in each quarter.
These groups are defined after the outcome occurs. They diagnose errors; they do not prove advance warning.

Each bank-quarter gets equal weight in pooled MAE. We also average the three quarter-level scores, giving
each quarter equal influence. Three evaluation quarters provide a narrow view of temporal stability.
''',r'''
# Check errors by quarter and by realized outcome before interpreting an average.
question_card(
    title='Where does the average score hide the largest misses?',
    theme=theme,
    body='Compare all rows with each quarter’s realized bottom quartile and bottom decile, then inspect the quarter-by-quarter pattern.',
    kicker='Error diagnosis',
    chip_text='QUESTION',
)

# Keep time stability and weak-outcome severity as separate checks. One groups
# by quarter; the other groups by what actually happened inside each quarter.
quarter_rows = []
tail_rows = []

for quarter, part in result_frame.groupby('date'):
    for n in growth_predictions:
        quarter_rows.append({'Quarter':quarter,'Model':n,'Rows':len(part),**scores(part.growth,part[n])})

    # Include ties at the realized quantile boundary and show resulting sample sizes.
    for label, mask in [('All',pd.Series(True,index=part.index)),
        ('Realized bottom 25%',part.growth.le(part.growth.quantile(.25))),
        ('Realized bottom 10%',part.growth.le(part.growth.quantile(.10)))]:
        for n in growth_predictions:
            for idx in part.index[mask]:
                tail_rows.append({
                    'Index':idx,
                    'Slice':label,
                    'Model':n,
                    'Absolute error':abs(part.loc[idx,'growth']-part.loc[idx,n]),
                    'Squared error':(part.loc[idx,'growth']-part.loc[idx,n])**2,
                })

quarter_scores = ordered_rows(
    pd.DataFrame(quarter_rows),
    ['Quarter', 'Model'],
    category_orders={'Model': MODEL_ORDER},
)
table(quarter_scores,'Three quarters, separate error checks',{'MAE (pp)':'{:.3f}','RMSE (pp)':'{:.3f}'})
fig=px.line(quarter_scores,x='Quarter',y='MAE (pp)',color='Model',symbol='Model',markers=True,color_discrete_map=MODEL_COLORS)
fig.update_xaxes(tickvals=sorted(result_frame.date.unique()),tickformat='%b %Y')
chart(fig,'quarter_errors','Quarter-level errors expose variation hidden by the average','Predictor quarters; each outcome is one quarter later')
equal_quarter = (
    quarter_scores.groupby('Model', sort=False, observed=True)['MAE (pp)']
    .mean()
    .reset_index(name='Equal-quarter MAE (pp)')
)
equal_quarter = ordered_rows(
    equal_quarter,
    ['Model'],
    category_orders={'Model': MODEL_ORDER},
)
table(equal_quarter,'Give each evaluation quarter equal weight',{'Equal-quarter MAE (pp)':'{:.3f}'})
tail_errors=pd.DataFrame(tail_rows).groupby(['Slice','Model'],sort=False).agg(
    Rows=('Index','size'),MAE=('Absolute error','mean'),MSE=('Squared error','mean')).reset_index()
tail_errors['MAE (pp)'] = 100 * tail_errors['MAE']
tail_errors['RMSE (pp)'] = 100 * np.sqrt(tail_errors['MSE'])
tail_errors = ordered_rows(
    tail_errors,
    ['Slice', 'Model'],
    category_orders={
        'Slice': SLICE_ORDER,
        'Model': MODEL_ORDER,
    },
)
table(tail_errors[['Slice','Model','Rows','MAE (pp)','RMSE (pp)']],'How wrong are forecasts when growth is weak?',{'MAE (pp)':'{:.3f}','RMSE (pp)':'{:.3f}'})
tail_errors['Slice'] = pd.Categorical(
    tail_errors['Slice'],
    categories=SLICE_ORDER,
    ordered=True,
)

# A slope chart makes the direction visible: follow each model from all rows
# toward increasingly weak realized outcomes.
fig = px.line(
    tail_errors.sort_values('Slice'),
    x='Slice',
    y='MAE (pp)',
    color='Model',
    markers=True,
    text='MAE (pp)',
    category_orders={'Slice': SLICE_ORDER},
    color_discrete_map=MODEL_COLORS,
    hover_data=['Rows'],
)
fig.update_traces(texttemplate='%{text:.1f}', textposition='top center')
fig.update_xaxes(title='Realized outcome group')
fig.update_yaxes(title='Mean absolute error (percentage points)')
chart(
    fig,
    'tail_errors',
    'Forecast errors rise as realized growth gets weaker',
    'Follow each model from all rows to the bottom quartile and bottom decile',
)
quarter_scores.to_csv(OUT / 'quarter_scores.csv', index=False)
tail_errors.to_csv(OUT / 'tail_errors.csv', index=False)
weak=tail_errors.loc[(tail_errors.Slice=='Realized bottom 10%')&(tail_errors.Model=='MLP')].iloc[0]
takeaway('Check the weak-outcome error before using the forecast',f'The MLP misses by {weak["MAE (pp)"]:.3f} pp on average across {int(weak.Rows):,} bottom-decile bank-quarters, compared with {mae["MLP"]:.3f} pp overall. These groups were identified using realized outcomes.')
''')
section('10 · Could we identify weak growth beforehand?', '''
**Rank using predictions first; compare with outcomes second.** For each quarter, select the lowest
predicted-growth 10% of banks. Then count how many actually belong to the lowest-growth 10%.
This is the decision check the 4.3% RMSE figure cannot provide: at a capacity
of about 450 banks per quarter, does the model give analysts a better starting
list than chance? The answer is a historical prioritization result, not a
finding of misconduct, distress, or an economic cause.

Precision asks: of the banks selected, how many were in the realized bottom decile?
Recall asks: of the realized bottom decile, how many were selected?
Here both sets have the same size, so precision and recall are equal by construction.
Ties are broken by certificate for a reproducible, fixed-capacity comparison.
The zero-growth rule has no economic ranking signal; its certificate ordering is arbitrary.

Spearman correlation compares the order of all forecasts with the order of outcomes.
A constant forecast has no defined rank correlation. Leave that value missing.
''',r'''
# Fix the quarterly review capacity before looking at the rankings.
question_card(
    title='Which selected banks later had the lowest deposit growth?',
    theme=theme,
    body='Rank banks from predicted growth first. Then compare that list with the realized bottom decile inside the same quarter.',
    kicker='Identification',
    chip_text='QUESTION',
)

# Fix the review capacity at 10% within each quarter. Certificate number breaks
# ties so the result can be reproduced exactly.
ranking = []
for quarter, part in result_frame.groupby('date'):
    selected_count = int(np.ceil(0.10 * len(part)))
    realized_bottom = set(
        part.sort_values(['growth', 'CERT'])
        .head(selected_count)
        .CERT
    )

    for model_name in growth_predictions:
        predicted_bottom = set(
            part.sort_values([model_name, 'CERT'])
            .head(selected_count)
            .CERT
        )
        hits = len(realized_bottom & predicted_bottom)
        has_ranking_signal = part[model_name].nunique() > 1
        spearman = (
            part.growth.corr(part[model_name], method='spearman')
            if has_ranking_signal
            else np.nan
        )

        ranking.append(
            {
                'Quarter': quarter,
                'Model': model_name,
                'Banks': len(part),
                'Selected': selected_count,
                'Hits': hits if has_ranking_signal else np.nan,
                'Precision': hits / selected_count if has_ranking_signal else np.nan,
                'Ranking status': 'Forecast ordering' if has_ranking_signal else 'All predictions tied; no ranking signal',
                'Recall': hits / len(realized_bottom) if has_ranking_signal else np.nan,
                'Spearman': spearman,
                'Random expectation': selected_count / len(part),
            }
        )

ranking = pd.DataFrame(ranking)
# %% NOTEBOOK CELL
# One model gets one row; each quarter gets one marker. Hover retains the
# exact count behind the percentage, and the CSV retains every audit column.
ranked_models = ranking.loc[
    ranking.Model.ne('Zero growth')
].copy()
ranked_models['Quarter label'] = (
    pd.to_datetime(ranked_models.Quarter)
    .dt.strftime('%b %Y')
)
quarter_colors = {
    'Mar 2024': '#0B6F75',
    'Jun 2024': '#3F6294',
    'Sep 2024': '#A86223',
}
quarter_symbols = {'Mar 2024': 'circle', 'Jun 2024': 'diamond', 'Sep 2024': 'square'}

fig = go.Figure()
model_rows = {'Persistence': 2, 'Ridge': 1, 'MLP': 0}
quarter_offsets = {'Mar 2024': .16, 'Jun 2024': 0, 'Sep 2024': -.16}

for quarter_label in ['Mar 2024', 'Jun 2024', 'Sep 2024']:
    part = ranked_models.loc[
        ranked_models['Quarter label'].eq(quarter_label)
    ]
    fig.add_trace(go.Scatter(
        x=part['Precision'],
        y=part['Model'].map(model_rows) + quarter_offsets[quarter_label],
        mode='markers',
        name=quarter_label,
        marker={
            'size': 17,
            'symbol': quarter_symbols[quarter_label],
            'color': quarter_colors[quarter_label],
            'line': {'color': 'white', 'width': 1},
        },
        customdata=part[['Model', 'Hits', 'Selected']].to_numpy(),
        hovertemplate=(
            '%{customdata[0]}<br>' + quarter_label
            + '<br>%{customdata[1]} of %{customdata[2]} selected banks'
            + '<br>Precision: %{x:.1%}<extra></extra>'
        ),
    ))

fig.update_xaxes(
    title='Selected banks with realized bottom-decile growth',
    tickformat='.0%',
    range=[0, .30],
)
fig.update_yaxes(
    title='',
    tickvals=[0, 1, 2],
    ticktext=['MLP', 'Ridge', 'Persistence'],
    range=[-.45, 2.45],
)
fig.add_vline(
    x=0.10,
    line_dash='dash',
    line_color='#343B43',
)
chart(
    fig,
    'ranking',
    'Ridge leads in March and June; the network leads in September',
    'Dashed line: about 10% by chance. Ridge leads Mar and Jun; MLP leads Sep.',
    height=520,
    legend_y=-0.47,
)
ranking.to_csv(OUT/'ranking.csv',index=False)
precision=ranking.loc[ranking.Model.eq('MLP'),'Precision'].mean()
takeaway(
    'Both feature models beat chance in these three quarters',
    'Ridge leads in March and June; MLP leads in September. '
    'Hover over each point for hits out of banks selected. '
    'Three reused quarters cannot establish which model would lead later.',
)
''')
section('Can anomaly detection help decide where to look?', '''
**Yes, it is a useful next experiment.** The forecast asks which bank may have weak
growth next quarter. An anomaly score asks whose *observed* report looks unusual
relative to its own past or comparable banks. Those are different review questions.

An unusual balance can reflect a merger, a reporting change, or genuine funding
pressure. A useful review queue must surface relevant cases at a capacity analysts
can handle. This notebook has not measured anomaly-detection precision or the cost
of false alerts. Future-quarter balances cannot enter a score used to decide what
to review today.
''',r'''
# EXEMPLAR: analytical-question
question_card(
    title='Would anomaly detection give us a better review list?',
    theme=theme,
    body=(
        'Possibly. First decide whether the job is to anticipate a future '
        'decline or to investigate an unusual report already in hand.'
    ),
    kicker='Next decision',
    chip_text='QUESTION',
)

# %% NOTEBOOK CELL
# EXEMPLAR: decision-ledger
# Each row names the evidence available when an analyst makes the decision.
review_questions = pd.DataFrame([
    {
        'Question': 'Who may have weak growth next quarter?',
        'Method': 'Forecast ranking',
        'Evidence here': 'Tested on three 2024 quarters',
        'Next check': 'Repeat on a later untouched period',
    },
    {
        'Question': 'Whose report looks unusual today?',
        'Method': 'Anomaly detection',
        'Evidence here': 'Not tested',
        'Next check': 'Review top-ranked cases and false alerts',
    },
    {
        'Question': 'Does this bank need follow-up?',
        'Method': 'Human source review',
        'Evidence here': 'Reporting changes can mimic events',
        'Next check': 'Check filings, entity changes, and context',
    },
])
table(
    review_questions,
    'Three questions, three different kinds of evidence',
    wrap_columns={
        'Question': 230,
        'Method': 150,
        'Evidence here': 190,
        'Next check': 230,
    },
)

# %% NOTEBOOK CELL
# EXEMPLAR: counterintuitive-boundary
wm_counterintuitive_card(
    title='An unusual report starts an investigation',
    theme=theme,
    why_misread=(
        'A sudden deposit change can look like a warning signal.'
    ),
    ordinary_process=(
        'Mergers, name changes, and reporting differences can also make a '
        'bank look unusual. The validation audit found a same-certificate '
        'name change beside an extreme balance jump.'
    ),
    conclusion_boundary=(
        'Test an anomaly score on information available at the review date. '
        'At a fixed review capacity, measure useful cases and false alerts '
        'against analyst-reviewed records before recommending it.'
    ),
    kicker='Interpretation check',
    chip_text='CHECK',
)
takeaway(
    'Anomaly detection deserves a separate trial',
    'The current ranking found weak-growth banks better than chance in three '
    'historical quarters. That does not tell us whether an anomaly score would '
    'send analysts to better cases. Evaluate it with the same review capacity '
    'and a clear definition of a useful review.',
)
''')
section('11 · What did we learn, and what would we do next?', '''
**The small neural network did not establish a dependable forecasting advantage on the reused 2024
holdout.** A zero-growth forecast had the lowest MAE at **3.60 percentage points**. The MLP scored **3.63
MAE** and had the lowest RMSE at **6.43 points**. Ridge scored **3.64 MAE** and **6.51 RMSE**; persistence
scored **5.11 MAE** and **9.46 RMSE**. The MLP's small edge over Ridge does not establish dependable
nonlinear value. The deeper bank-cluster check appears after this conclusion.

Weak quarters tell a more practical story. The MLP's MAE rose from **3.63 points overall** to **6.78 points
in the realized bottom decile**. At a 10% quarterly review capacity, **21.8%** of its selections landed in
the realized bottom decile. Ridge reached **21.9%**. Both rankings beat the roughly 10% random expectation,
although three reused quarters provide limited evidence about how the ranking would behave later.

The evidence supports a simple operating choice. Keep zero growth as the accuracy benchmark, retain Ridge
as the transparent feature model, and treat the MLP as an unproven research candidate. Audit large misses
for mergers and institutional changes, then evaluate all three on a later untouched period.

**Should we try something else?** Yes. Anomaly detection may help prioritize reports
that look unusual *now*; it has not been tested here. The masterclass separately evaluates TimesFM using
bank histories after this core conclusion. That extension does not change the measured
MAE, RMSE, or review precision of the four evaluated methods.

**Why can zero growth win MAE while the MLP wins RMSE?** They reward different behavior.
An error of 10 percentage points contributes 10 to absolute error and 100 to squared error.
Large changes therefore pull an MSE-trained model harder. The prediction that minimizes expected
absolute error is a conditional median; squared error targets a conditional mean.
Our network minimizes MSE in log-growth space. After conversion to ordinary growth, that does not directly minimize percentage-point RMSE or MAE. The observed error distribution, rather than the training objective alone, establishes the tradeoff here.

The validation audit also exposes a same-certificate name change from PLUS INTERNATIONAL BANK
to EMIGRANT BANK with a huge balance jump. Institutional restructuring can dominate quarterly
balance changes. That is a limitation of this outcome definition that the next experiment must resolve.
We preserve it here instead of quietly changing the question after seeing the errors.
''',r'''
# Build the conclusion from the saved scores and the three-quarter limit.
rmse = metrics.set_index('Model')['RMSE (pp)']
assert mae.idxmin() == 'Zero growth'
assert rmse.idxmin() == 'MLP'
conclusion = (
    f'The MLP had higher mean absolute error than zero growth '
    f'({mae["MLP"]:.3f} versus {mae["Zero growth"]:.3f} pp MAE). '
    f'It had the lowest RMSE ({rmse["MLP"]:.3f} pp), and its 99th-percentile '
    'absolute error was below zero growth. Its small edge over Ridge does not '
    'establish dependable nonlinear value. Anomaly detection was not evaluated; '
    'the masterclass evaluates TimesFM separately after this core conclusion. '
    'Training used 214,425 examples across 38 predictor quarters. Evaluation used 13,532 examples across three reused 2024 predictor quarters. '
    'All four comparisons: '
    + '; '.join(
        f'{model}: MAE {mae[model]:.3f}, RMSE {rmse[model]:.3f} pp'
        for model in MODEL_ORDER
    )
    + '.'
)
takeaway('Does this experiment establish nonlinear value?', conclusion)
takeaway('For a bank analyst: use the result to guide research',
    'Inspect the source reports and institution changes behind large forecast misses. These quarterly balances do not establish withdrawals, bank runs, or the benefit of an intervention.')
takeaway('For the next experiment: earn genuinely new evidence',
    'Resolve reporting-vintage and release-date availability, verify the historical form crosswalk, and evaluate a later period that has not guided development. Keep the strongest simple baseline.')
assert source_hash==hashlib.sha256((ROOT/'data/fdic_financials_2013_2024.csv').read_bytes()).hexdigest()
assert raw_fingerprint==pd.util.hash_pandas_object(raw,index=True).sum()
assert np.allclose(rows.growth,(rows.next_deposits-rows.DEPDOM)/rows.DEPDOM)
assert np.allclose(rows.log_growth,np.log(rows.next_deposits/rows.DEPDOM))
import importlib.metadata
summary={'source_sha256':source_hash,'target':'log growth; errors reported in ordinary percentage points','history_gate':gate,
    'rows':{'train':len(train),'validation':len(valid),'holdout':len(test)},'winner_by_historical_MAE':winner,
    'mlp_minus_ridge_MAE_pp':difference,'conclusion':conclusion,'training_seconds':training_seconds,
    'frozen_plan':frozen,'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','tensorflow','scikit-learn','plotly','wm-notecards']},
    'python':platform.python_version()}
(OUT/'run_summary.json').write_text(json.dumps(summary,indent=2))
print('Audit passed: source unchanged, target arithmetic reconciled, predictions finite, results saved.')
''')
section('Deeper lesson · Does a different random start change the answer?', '''
**A seed changes the network’s initial weights.** Fit the same architecture with seeds 7 and 99,
in addition to the primary seed 42. Compare their validation errors only. Keep seed 42 as the primary
historical model even if another start looks better.

This sensitivity check measures optimization variation across three starting points. All three fits use the same banks and quarters.
''',r'''
# Repeat the same training run with two other starting weights.
seed_results=[{'Seed':42,**log_scores(y_valid, validation_predictions['MLP']),'Best epoch':frozen['mlp_best_epoch']}]
for seed in [7,99]:
    other,h,seconds=fit_network(seed)
    seed_results.append({'Seed':seed,**log_scores(y_valid, predict_log_growth(other, X_valid)),'Best epoch':int(np.argmin(h['val_loss'])+1)})
seed_results = ordered_rows(
    pd.DataFrame(seed_results),
    ['Seed'],
)
table(seed_results,'Same data, different starting weights',{'MAE (pp)':'{:.3f}','RMSE (pp)':'{:.3f}'})
fig=px.scatter(seed_results,x='MAE (pp)',y=seed_results.Seed.astype(str),color_discrete_sequence=['#0B6F75'])
fig.update_yaxes(title='Random seed')
chart(fig,'seeds','Three starting weights produce different validation errors','Primary historical model remains seed 42')
seed_results.to_csv(OUT/'seed_sensitivity.csv',index=False)
takeaway('Three starts, one dataset',f'Validation MAE ranges from {seed_results["MAE (pp)"].min():.3f} to {seed_results["MAE (pp)"].max():.3f} pp across the three fixed starts. Seed 42 remains the primary fit.')
''',advanced=True)
section('Deeper lesson · How uncertain is the MLP–Ridge difference?', '''
**Resample whole banks so their repeated reports stay together.** Sampling individual rows would pretend
that several observations of one bank are independent.

For each of 500 resamples, draw certificates with replacement and calculate the paired difference in MAE.
A negative difference favors the MLP. The 95% percentile interval describes resampling uncertainty
conditional on these fitted models and these three quarters. New economic regimes remain outside this interval
or all uncertainty from retraining the models.
''',r'''
# Resample certificates so repeated bank reports stay together.
errors=result_frame.assign(delta=np.abs(result_frame.growth-result_frame.MLP)-np.abs(result_frame.growth-result_frame.Ridge))
clusters=errors.groupby('CERT').delta.agg(['sum','count'])
rng = np.random.default_rng(42)
deltas = []
for _ in range(500):
    sampled=rng.integers(0,len(clusters),len(clusters))
    sample=clusters.iloc[sampled]
    deltas.append(100*sample['sum'].sum()/sample['count'].sum())
low,high=np.quantile(deltas,[.025,.975])
interval=pd.DataFrame({'Comparison':['MLP minus Ridge'],'Observed MAE difference (pp)':[difference],
    '95% lower (pp)':[low],'95% upper (pp)':[high],'Bank clusters':[len(clusters)]})
table(interval,'Paired bank-cluster bootstrap',{'Observed MAE difference (pp)':'{:+.4f}','95% lower (pp)':'{:+.4f}','95% upper (pp)':'{:+.4f}'})
fig = px.histogram(
    x=deltas,
    nbins=35,
    color_discrete_sequence=['#0B6F75'],
)
fig.add_vline(x=0, line_dash='dash', line_color='#343B43')
fig.update_xaxes(title='MLP minus Ridge MAE (pp)')
fig.update_yaxes(title='Bootstrap resamples')
chart(fig,'bootstrap','The MLP–Ridge uncertainty interval crosses zero','500 paired cluster resamples; negative favors MLP')
interval.to_csv(OUT/'bootstrap_interval.csv',index=False)
takeaway('Uncertainty belongs beside the difference',f'The observed difference is {difference:+.4f} pp; the conditional 95% interval is [{low:+.4f}, {high:+.4f}] pp. '+('The interval crosses zero.' if low<=0<=high else 'The interval stays on one side of zero.')+' Three quarters still limit temporal generalization.')
''',advanced=True)
section('Deeper lesson · Can uninsured deposits enter the model yet?', '''
**Missing reporting can encode eligibility.** The FDIC’s [reporting guidance](https://www.fdic.gov/news/financial-institution-letters/2023/estimated-uninsured-deposits-reporting-expectations)
notes that institutions below USD 1 billion may not report estimated uninsured deposits.
The [2024 RC-O instructions](https://www.fdic.gov/system/files/2024-05/031-041-324-rc-o.pdf) define the reporting item.
Historical eligibility can depend on an earlier June balance, so current asset size is a descriptive grouping,
not a reconstruction of legal reporting eligibility.

The API dictionary exposes two related fields, DEPUNA and DEPUNINS. Inspect both separately.
The API often supplies zeros even below the reporting threshold. A populated API field
is not proof that the bank filed that regulatory item, and a zero is not proof of zero exposure.
Do not splice the fields or impute either into the model without a verified historical mapping.
''',r'''
# EXEMPLAR: analytical-question
question_card(
    title='Does a value in a downloaded column prove the bank reported it?',
    theme=theme,
    body='No. An API is the FDIC service our code asks for records. A field is a named column in its response. DEPUNA and DEPUNINS are two uninsured-deposit-related columns; their historical mapping still needs verification.',
    kicker='Read the source before interpreting the number',
)

# %% NOTEBOOK CELL
# EXEMPLAR: formula-card
wm_formula_card(
    title='A blank, a zero, and a positive number answer different questions',
    theme=theme,
    items=[
        {'label': 'Coverage', 'fallback': 'nonmissing records / all records'},
        {'label': 'Zeros among populated records', 'fallback': 'zero records / nonmissing records'},
        {'label': 'Illustration only', 'fallback': '100 reports: 70 blank + 20 zero + 10 nonzero → 30% coverage; 66.7% of populated values are zero'},
    ],
)

# %% NOTEBOOK CELL
# Inspect source values as received. Current assets create descriptive groups.
coverage_source = long_audit.copy()
coverage_source['Asset group'] = np.where(
    coverage_source.ASSET.ge(1_000_000), 'At least USD 1bn', 'Below USD 1bn',
)
uninsured = []
for field in ['DEPUNA', 'DEPUNINS']:
    groups = coverage_source.groupby(['REPDTE', 'Asset group', 'CALLFORM'], dropna=False)
    for (quarter, size, form), part in groups:
        uninsured.append({
            'Quarter': quarter, 'Asset group': size, 'Call form': str(form),
            'Field': field, 'Rows': len(part),
            'API populated': part[field].notna().sum(),
            'Zero values': part[field].eq(0).sum(),
        })
uninsured = pd.DataFrame(uninsured)
uninsured['Coverage'] = uninsured['API populated'] / uninsured['Rows']
uninsured['Zero share'] = uninsured['Zero values'] / uninsured['API populated'].replace(0, np.nan)
uninsured.to_csv(OUT / 'uninsured_coverage.csv', index=False)

# %% NOTEBOOK CELL
# Each report belongs to exactly one state. This exposes what a coverage dip means.
aggregate = uninsured.groupby(
    ['Quarter', 'Asset group', 'Field'], as_index=False,
)[['Rows', 'API populated', 'Zero values']].sum()
aggregate['Date'] = pd.to_datetime(aggregate['Quarter'].astype(str))
aggregate['Coverage'] = aggregate['API populated'] / aggregate['Rows']
aggregate['Zero share'] = aggregate['Zero values'] / aggregate['API populated'].replace(0, np.nan)
aggregate['Missing'] = aggregate['Rows'] - aggregate['API populated']
aggregate['Zero'] = aggregate['Zero values']
aggregate['Nonzero'] = aggregate['API populated'] - aggregate['Zero values']
assert aggregate[['Missing', 'Zero', 'Nonzero']].sum(axis=1).eq(aggregate['Rows']).all()
aggregate.to_csv(OUT / 'uninsured_states.csv', index=False)

# %% NOTEBOOK CELL
# Separate panels eliminate the four-line legend puzzle.
fig = make_subplots(
    rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
    subplot_titles=['DEPUNA · Below USD 1bn', 'DEPUNINS · Below USD 1bn',
                    'DEPUNA · At least USD 1bn', 'DEPUNINS · At least USD 1bn'],
    vertical_spacing=.16,
)
for row, size in enumerate(['Below USD 1bn', 'At least USD 1bn'], start=1):
    for col, field in enumerate(['DEPUNA', 'DEPUNINS'], start=1):
        part = aggregate.loc[aggregate['Asset group'].eq(size) & aggregate['Field'].eq(field)].sort_values('Date')
        for state, color in [('Nonzero', '#0B6F75'), ('Zero', '#A86223'), ('Missing', '#D8DDE3')]:
            fig.add_trace(go.Scatter(
                x=part['Date'], y=part[state] / part['Rows'], name=state,
                stackgroup=f'{row}-{col}', mode='lines',
                line=dict(width=.5, color=color), fillcolor=color,
                legendgroup=state, showlegend=(row == 1 and col == 1),
                customdata=np.column_stack([part[state], part['Rows']]),
                hovertemplate=state + ': %{y:.1%}<br>%{customdata[0]:,} of %{customdata[1]:,} reports<extra></extra>',
            ), row=row, col=col)
        fig.update_yaxes(range=[0, 1], tickformat='.0%', row=row, col=col)
chart(fig, 'uninsured_states', 'Reported zeros and missing values follow different patterns',
      'Every panel sums to 100% of reports. Gray = missing; amber = stored zero; teal = nonzero.', height=630)

# %% NOTEBOOK CELL
# Read the deepest dip directly from the data instead of guessing from the line.
small = aggregate.loc[aggregate['Asset group'].eq('Below USD 1bn') & aggregate['Field'].eq('DEPUNA')]
dip = small.loc[small['Coverage'].idxmin()]
takeaway(
    'The dip describes the downloaded column, not disappearing deposits',
    f'At the lowest small-bank DEPUNA coverage in {dip.Date:%b %Y}, '
    f'{dip.Coverage:.1%} of reports contain a value and {dip["Zero share"]:.1%} '
    'of those populated values are zero. The earlier chart divided zero counts by all reports; '
    'that shared denominator made its shape resemble coverage. A return to 100% coverage can consist mostly of zeros.',
)
wm_counterintuitive_card(
    title='Could we fill the blanks and use this as a risk signal?', theme=theme,
    why_misread='A complete numeric column looks ready for a model.',
    ordinary_process='Reporting eligibility, field mappings, and source-system conventions can change which cells are blank or zero. These charts alone do not identify the cause of the 2017–2019 change.',
    conclusion_boundary='Keep the original values. Verify historical definitions and filing eligibility before adding either column. Current asset groups do not reconstruct regulatory eligibility, and missingness is not evidence of bank distress.',
    kicker='Source quality determines feature quality', chip_text='CHECK',
)

''',advanced=True)
section('Appendix · Keep the useful questions from Experiment 0', '''
The [original masterclass](experiments/experiment_0/FDIC_Deep_Learning_Masterclass.ipynb) preserves the complete
executed dollar-runoff experiment and its exact outputs. `Assign1.ipynb` is unchanged.

- **Two stages:** estimate whether deposits decline, then estimate severity conditional on decline.
  An expected dollar loss requires the conditional mean severity; the median answers a different question.
- **Precision–recall:** how many selected cases have a decline, and how many decline cases are found?
- **Calibration:** among cases assigned a given decline probability, how often does decline occur?
- **Review capacity:** selecting 10% each quarter is an explicit workload assumption. Captured dollars measure ranking coverage.
- **Loss choice:** MSE makes extreme growth influential. A future robust-loss comparison belongs in a newly frozen experiment,
  not an unreported repair after seeing this holdout.

**Try explaining these aloud:** Why can a high dollar-capture score come from size alone?
Why is a missing next report different from zero growth? Why does scaling use training rows?
Why can a model have low average error but poor bottom-decile precision?
What does the bootstrap leave uncertain?

Visual choices follow the questions: histograms for target shape, a timeline for leakage,
dots for close model errors, lines for quarters and optimization, and a scatter for individual forecasts.
Maps and pies add no evidence to this question. Classification graphics stay with the classification experiment.
The lower-triangle correlation heatmap describes overlap among the five fixed inputs. Feature selection was frozen before the holdout evaluation.
''',advanced=True)


def timesfm_extension():
    """A measured sequence-model extension after the core assignment concludes."""
    prose = r'''**Can a bank's own history improve the next forecast?** The full TimesFM
benchmark did worse on both aggregate errors. Before explaining the result,
consider the different evidence each model receives:

- **Ridge / MLP:** five financial features → predicted log growth.
- **TimesFM:** consecutive historical log deposit balances → predicted log balance.

We compare the same 13,532 bank-quarters and ordinary growth outcome. This compares
complete approaches with different inputs and pretraining, not architecture alone.
TimesFM uses a median-quantile point forecast. If its predicted log balance is
<code>ẑ</code>, predicted log growth is <code>ẑ − ln(Dₜ)</code> and ordinary percentage
growth is <code>100 × [exp(ẑ − ln(Dₜ)) − 1]</code>.

The eight-row smoke test established that inference runs. Only the full holdout
below supports a comparison. The saved forecast artifact is generated with
`uv run python benchmark_timesfm.py`, using the public
`from timesfm3.mlx import TimesFM3Forecaster` API and a recorded checkpoint revision.
Notebook execution reads this artifact so students need not download the model.

**What could go wrong with a short history?** TimesFM 3.0 processes numeric patches
(32 input observations per patch); histories here range from 2 to 47 quarters.
Short histories are retained. We do not exclude a difficult bank after seeing its error.
The error audit below identifies what happened; history length as its cause remains
a hypothesis. A future minimum-history rule needs validation data and new evaluation.

This is a retrospective zero-shot research benchmark. Input sequences end at each
predictor date, but a 2026 checkpoint was not available in 2024. Pretraining overlap
with these records has not been independently ruled out. The checkpoint's current
noncommercial, nonproduction license also prevents treating it as a deployment recommendation.
[Official implementation](https://github.com/google-research/timesfm) ·
[Checkpoint and license](https://huggingface.co/google/timesfm-3.0-pytorch).

The second benchmark uses [Amazon Chronos-Bolt Small](https://huggingface.co/amazon/chronos-bolt-small),
a pretrained numerical forecasting model, not a chat LLM. Run `uv run python benchmark_chronos.py`
to reproduce its saved forecasts. This checkpoint also postdates the evaluated quarters; pretraining
overlap has not been independently ruled out. These initial benchmarks use unchanged weights. The next bonus section tests head fine-tuning.
'''
    source = r'''
# EXEMPLAR: bounded-takeaway
# EXEMPLAR: analytical-question
question_card(
    title='Did we train TimesFM on these banks?', theme=theme,
    body='For this first benchmark, no. Its authors pretrained its weights; we loaded them and asked for forecasts. Passing a bank history into the model is inference. The next section then updates forecasting-head weights using training examples and a loss.',
    kicker='Pretraining → optional fine-tuning → inference',
)
wm_counterintuitive_card(
    title='Does a bad forecast prove the model never saw this pattern?', theme=theme,
    why_misread='A huge miss makes unfamiliar data sound like the obvious explanation.',
    ordinary_process='Short histories, balance jumps, an unsuitable input representation, or poor transfer can all produce bad forecasts. A model can also fail on a pattern resembling its pretraining data.',
    conclusion_boundary='We do not know whether similar bank patterns were in pretraining. We measured a failure of this frozen zero-shot configuration; we have not established its cause.',
    kicker='Observation versus explanation', chip_text='CHECK',
)

# %% NOTEBOOK CELL
# Reconcile saved sequence forecasts to the exact core evaluation rows.
sequence = pd.read_csv(ROOT / 'growth_outputs/timesfm_zero_shot_predictions.csv')
reference = pd.read_csv(ROOT / 'growth_outputs/submission/predictions.csv')
joined = reference.merge(sequence, on=['CERT', 'date', 'DEPDOM', 'target_date'],
                         validate='one_to_one', how='outer', indicator=True)
assert joined['_merge'].eq('both').all() and len(joined) == len(test)
sequence_errors = 100 * (joined['TimesFM'] - joined['growth'])
sequence_scores = pd.DataFrame([{
    'Model': 'TimesFM',
    'MAE (pp)': np.abs(sequence_errors).mean(),
    'RMSE (pp)': np.sqrt(np.square(sequence_errors).mean()),
}])
comparison = pd.concat([metrics, sequence_scores], ignore_index=True)
comparison.to_csv(OUT / 'timesfm_comparison.csv', index=False)

# %% NOTEBOOK CELL
takeaway('TimesFM did not improve either aggregate error',
         f'TimesFM MAE is {sequence_scores.iloc[0]["MAE (pp)"]:.3f} pp and RMSE is '
         f'{sequence_scores.iloc[0]["RMSE (pp)"]:.3f} pp. The core baseline comparison remains essential.')

# %% NOTEBOOK CELL
# Keep every row; identify the error driving the aggregate instead of clipping it.
worst = sequence_errors.abs().idxmax()
case = joined.loc[worst]
squared_share = float(sequence_errors.loc[worst] ** 2 / np.square(sequence_errors).sum())
# One subtraction in log space becomes a multiplier after exponentiation.
current_log_balance = np.log(case.DEPDOM)
predicted_log_growth = case['Predicted log deposits'] - current_log_balance
predicted_balance_multiplier = np.exp(predicted_log_growth)

wm_formula_card(
    title='Why does one log-balance miss become an enormous growth forecast?',
    theme=theme,
    items=[
        {'label': 'Log growth', 'fallback': f'{case["Predicted log deposits"]:.3f} − {current_log_balance:.3f} = {predicted_log_growth:.3f}'},
        {'label': 'Predicted balance multiplier', 'fallback': f'exp({predicted_log_growth:.3f}) = {predicted_balance_multiplier:,.2f} times the current balance'},
        {'label': 'Ordinary growth', 'fallback': f'100 × ({predicted_balance_multiplier:,.2f} − 1) = {100 * case.TimesFM:,.2f}%'},
    ],
)

# %% NOTEBOOK CELL
receipt = pd.DataFrame([{
    'Bank CERT': int(case.CERT), 'Predictor date': case.date,
    'History quarters': int(case['Context quarters']),
    'Predicted growth (%)': 100 * case.TimesFM,
    'Actual growth (%)': 100 * case.growth,
    'Share of squared error': squared_share,
}])
table(receipt, 'One short-history bank dominates TimesFM squared error',
      {'Predicted growth (%)': '{:,.2f}', 'Actual growth (%)': '{:,.2f}', 'Share of squared error': '{:.2%}'})
takeaway('The full evaluation reveals a failure the smoke test missed',
         f'Bank {int(case.CERT)} has {int(case["Context quarters"])} historical quarters and contributes '
         f'{squared_share:.2%} of TimesFM squared error. That observation remains in the scores. '
         'Short history and domain mismatch are hypotheses to investigate, not established causes. '
         'A large forecast error alone does not establish a problem at the bank.')

# %% NOTEBOOK CELL
# EXEMPLAR: analytical-question
question_card(
    title='Would a different pretrained forecaster help?', theme=theme,
    body='We test Chronos-Bolt Small with the same historical log balances, one-quarter horizon, median point forecast, and all 13,532 outcomes. Its T5-based numerical forecasting architecture gives us a second approach. We do not select settings using these outcomes.',
    kicker='A new experiment, with the same measurement contract',
)

# %% NOTEBOOK CELL
# Join by row identity before comparing model errors.
chronos = pd.read_csv(ROOT / 'growth_outputs/chronos_zero_shot_predictions.csv')
chronos_joined = reference.merge(
    chronos, on=['CERT', 'date', 'DEPDOM', 'target_date'],
    how='outer', validate='one_to_one', indicator=True,
)
assert chronos_joined['_merge'].eq('both').all()
assert len(chronos_joined) == len(joined)
chronos_error = 100 * (chronos_joined['Chronos-Bolt'] - chronos_joined['growth'])
chronos_scores = pd.DataFrame([{
    'Model': 'Chronos-Bolt',
    'MAE (pp)': chronos_error.abs().mean(),
    'RMSE (pp)': np.sqrt(np.square(chronos_error).mean()),
}])
all_scores = pd.concat([comparison, chronos_scores], ignore_index=True)
all_scores.to_csv(OUT / 'foundation_model_comparison.csv', index=False)

# %% NOTEBOOK CELL
# Position shows rank; direct numbers preserve differences hidden by the large TimesFM miss.
fig = make_subplots(rows=1, cols=2, subplot_titles=['Mean absolute error', 'Root mean squared error'])
for col, metric in enumerate(['MAE (pp)', 'RMSE (pp)'], start=1):
    ordered = all_scores.sort_values(metric)
    fig.add_trace(go.Scatter(
        x=ordered[metric], y=ordered['Model'], mode='markers+text',
        text=[f'{value:.3f}' for value in ordered[metric]],
        textposition=['middle left' if model == 'TimesFM' and col == 2 else 'middle right' for model in ordered['Model']],
        marker=dict(size=12, color=['#0B6F75' if i == 0 else '#3F6294' if model == 'Chronos-Bolt' else '#9AA6B2' for i, model in enumerate(ordered['Model'])]),
        showlegend=False, hovertemplate='%{y}: %{x:.4f} pp<extra></extra>',
    ), row=1, col=col)
    fig.update_xaxes(type='log', title='Error (pp), log scale',
                     tickvals=[4, 6, 10] if col == 1 else [10, 100, 300],
                     range=[np.log10(ordered[metric].min())-.1, np.log10(ordered[metric].max())+.35], row=1, col=col)
    fig.update_yaxes(categoryorder='array', categoryarray=ordered.Model.tolist(), autorange='reversed', row=1, col=col)
chart(fig, 'foundation_models', 'Pretrained forecasts face the same historical scoring rules',
      'Lower is better; every score uses all holdout rows. Foundation models receive histories, not the five tabular features.', height=490)
mae_winner = all_scores.loc[all_scores['MAE (pp)'].idxmin(), 'Model']
rmse_winner = all_scores.loc[all_scores['RMSE (pp)'].idxmin(), 'Model']
takeaway(
    'Chronos improves on TimesFM; the simple baselines still matter',
    f'Chronos-Bolt MAE is {chronos_scores.iloc[0]["MAE (pp)"]:.3f} pp; RMSE is '
    f'{chronos_scores.iloc[0]["RMSE (pp)"]:.3f} pp. Across these approaches, {mae_winner} '
    f'has the lowest MAE and {rmse_winner} has the lowest RMSE. '
    'This is an exploratory comparison on reused outcomes, not fresh confirmation of a model selected today. '
    'Any fine-tuning must use earlier training and validation periods, followed by a genuinely new evaluation.',
)

'''
    return ('Bonus · Can a pretrained model learn from deposit history?', prose, source, True)


def finetuning_extension():
    prose = '''**Can adaptation improve the pretrained forecast?** We actually update pretrained
forecasting-head weights using earlier bank examples. The large backbones remain frozen.
This is a bounded head-fine-tuning experiment, not full-model training or an exhaustive search.

Picture the calendar: training answers end in December 2022; validation answers end
in December 2023. The 2024 answers cannot choose a weight update, a learning rate, or a checkpoint.
We keep step zero in the comparison, so fine-tuning has to earn its place on validation.

The fixed recipe is 256 sampled batches of 32, seed 42, Adam at 0.00001, and pinball loss
on next-quarter log balances. This is 8,192 sampled examples with replacement, not an epoch over
all 214,425 eligible rows. TimesFM initially produced non-finite gradients through
its horizon normalization refinement. We stopped that run, then held the refinement
statistics fixed during backpropagation while preserving the forward calculation.
This is a custom head-training path, not an official turnkey TimesFM 3.0 trainer.
Checkpoints at steps 0, 128, and 256 compete on all 13,834
validation rows using mean absolute log-balance error. This selection metric differs from
the ordinary percentage-point errors reported below.

**What actually changes a weight?** This is the Chronos update from the training
script. `batch` contains past log balances; `y` contains training-only next-quarter
log balances. Each column of `pred` is a predicted quantile.

```python
# First measure how far each predicted quantile is from the training answer.
pred = model(context=torch.tensor(batch)).quantile_preds[:, :, 0]
difference = torch.tensor(y)[:, None] - pred
q = model.quantiles.detach()[None, :]
loss = torch.maximum(q * difference, (q - 1) * difference).mean()
```

Now use that loss to change the head weights. The frozen backbone stays untouched.

```python
# Clear old gradients, calculate new ones, and update only the unfrozen head.
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
optimizer.step()
```

The loss treats underprediction and overprediction differently for each quantile.
The 0.5 quantile is the median used for our point forecast. The gradient tells the
optimizer which way to move the head weights. Validation decides whether those
updates deserve to be kept.

Reproduce the data export and local runs from a terminal:

```bash
uv run python prepare_finetune_data.py
uv run python finetune_foundation.py chronos
uv run python finetune_foundation.py timesfm
```

The complete update loops live in `finetune_foundation.py`. The notebook loads their
saved results rather than silently retraining a foundation model on every Run All.
The original zero-shot forecasts remain available beside the adapted versions.

**Boundary:** we enforce chronology in this adaptation pipeline. Original pretraining
membership is unknown, and the checkpoints postdate 2024. We have also examined this
holdout repeatedly. This remains retrospective exploratory evidence, not a globally
leakage-free prospective test.
'''
    source = r'''
# EXEMPLAR: analytical-question
question_card(
    title='Which models should learn new weights?', theme=theme,
    body='A pretrained model can adapt. A constant forecasting rule has no weights to learn. Keep both in the comparison: adaptation must beat a useful baseline, not merely beat its own starting point.',
    kicker='Choose the training procedure before seeing the result',
)

# %% NOTEBOOK CELL
# Explain every model's training status instead of leaving a blank scorecard entry.
training_decisions = pd.DataFrame([
    {'Model': 'Zero growth', 'Training decision': 'No fitting', 'Why': 'Always predicts 0%; changing this rule would remove the baseline.'},
    {'Model': 'Persistence', 'Training decision': 'No fitting', 'Why': 'Repeats prior growth; there are no learned weights.'},
    {'Model': 'Ridge', 'Training decision': 'Fit on banks', 'Why': 'Coefficients fit on training rows; regularization chosen on validation. No pretrained checkpoint.'},
    {'Model': 'MLP', 'Training decision': 'Train from scratch', 'Why': 'Weights fit on training rows; validation chooses stopping. No pretrained checkpoint.'},
    {'Model': 'TimesFM', 'Training decision': 'Fine-tune forecast head', 'Why': 'Adapt output weights; freeze backbone to limit computation and degrees of freedom.'},
    {'Model': 'Chronos-Bolt', 'Training decision': 'Fine-tune forecast head', 'Why': 'Same bounded adaptation question; retain step zero if validation prefers it.'},
])
table(training_decisions, 'What learned, what stayed fixed, and why')

# %% NOTEBOOK CELL
# Read the decision record saved before 2024 predictions were evaluated.
fine_scores = []
selection_rows = []
curves = []
for key, label in [('timesfm', 'TimesFM'), ('chronos', 'Chronos-Bolt')]:
    path = ROOT / f'growth_outputs/{key}_finetuned_predictions.csv'
    metadata = json.loads(path.with_suffix('.json').read_text())
    adapted = pd.read_csv(path)
    checked = reference.merge(
        adapted, on=['CERT', 'date', 'DEPDOM', 'target_date'],
        how='outer', validate='one_to_one', indicator=True,
    )
    assert checked['_merge'].eq('both').all() and len(checked) == len(reference)
    assert pd.Timestamp(metadata['train_last_outcome']) < pd.Timestamp('2023-03-31')
    assert pd.Timestamp(metadata['validation_last_outcome']) < pd.Timestamp('2024-03-31')
    assert metadata['candidate_weights_changed']
    errors = 100 * (checked['Prediction'] - checked['growth'])
    fine_scores.append({
        'Model': label + ' · selected',
        'MAE (pp)': errors.abs().mean(),
        'RMSE (pp)': np.sqrt(np.square(errors).mean()),
    })
    selection_rows.append({
        'Model': label, 'Chosen step': metadata['selected_step'],
        'Unique training rows seen': metadata['training_unique_rows_seen'],
        'Trainable weights': metadata['trainable_parameters'],
        'Status': 'Head adapted' if metadata['selected_weights_changed'] else 'Original weights retained',
    })
    curves.extend(dict(Model=label, **point) for point in metadata['validation'])
finetuned_scores = pd.DataFrame(fine_scores)
selection = pd.DataFrame(selection_rows)
finetuned_scores.to_csv(OUT / 'finetuned_scores.csv', index=False)
selection.to_csv(OUT / 'finetuning_selection.csv', index=False)

# %% NOTEBOOK CELL
# Validation makes the choice. Holdout scores cannot change it.
validation_curve = pd.DataFrame(curves)
fig = px.line(validation_curve, x='step', y='validation_log_MAE', color='Model', markers=True)
fig.update_xaxes(title='Training updates', tickvals=[0, 128, 256])
fig.update_yaxes(title='Validation MAE in log balance; lower is better')
chart(fig, 'finetuning_validation', 'Validation scores measure the effect of adaptation',
      'Step zero is the untouched pretrained model. Only these 2023 scores select the checkpoint.', height=400)
table(selection, 'The checkpoint decision was frozen before scoring 2024')

# %% NOTEBOOK CELL
# Keep the zero-shot result visible; report every row, with no post-hoc clipping.
adaptation_scores = pd.concat([all_scores, finetuned_scores], ignore_index=True)
adaptation_scores.to_csv(OUT / 'adaptation_comparison.csv', index=False)
fig = make_subplots(rows=1, cols=2, subplot_titles=['Mean absolute error', 'Root mean squared error'])
for col, metric in enumerate(['MAE (pp)', 'RMSE (pp)'], start=1):
    ordered = adaptation_scores.sort_values(metric)
    fig.add_trace(go.Scatter(
        x=ordered[metric], y=ordered.Model, mode='markers+text',
        text=[f'{v:.3f}' for v in ordered[metric]],
        textposition=['middle left' if m.startswith('TimesFM') and col == 2 else 'middle right' for m in ordered.Model],
        marker=dict(size=11, color=['#0B6F75' if i == 0 else '#3F6294' if 'selected' in m else '#A0A8B0' for i, m in enumerate(ordered.Model)]),
        showlegend=False, hovertemplate='%{y}: %{x:.4f} pp<extra></extra>',
    ), row=1, col=col)
    fig.update_xaxes(type='log', title='Error (pp), log scale',
                     tickvals=[4, 6, 10] if col == 1 else [10, 100, 300],
                     range=[np.log10(ordered[metric].min())-.1, np.log10(ordered[metric].max())+.4], row=1, col=col)
    fig.update_yaxes(categoryorder='array', categoryarray=ordered.Model.tolist(), autorange='reversed', row=1, col=col)
chart(fig, 'adaptation_comparison', 'Adapted models remain accountable to the simple baselines',
      'Blue = validation-selected foundation-model checkpoint; teal = lowest score. All 13,532 evaluation rows remain.', height=590)
for label in ['TimesFM', 'Chronos-Bolt']:
    chosen = finetuned_scores.set_index('Model').loc[label + ' · selected']
    original = all_scores.set_index('Model').loc[label]
    decision = selection.set_index('Model').loc[label]
    takeaway(
        f'{label}: {decision["Status"].lower()}',
        f'Validation selected step {decision["Chosen step"]}. '
        f'2024 MAE: {original["MAE (pp)"]:.3f} → {chosen["MAE (pp)"]:.3f} pp; '
        f'RMSE: {original["RMSE (pp)"]:.3f} → {chosen["RMSE (pp)"]:.3f} pp. '
        'This tests one bounded head-adaptation recipe. It does not establish the best achievable fine-tuned performance.',
    )
wm_counterintuitive_card(
    title='Would more training automatically make the review list better?', theme=theme,
    why_misread='Lower forecast error sounds like better bank selection.',
    ordinary_process='Average error and finding the weakest banks reward different behavior. A model can improve one and worsen the other.',
    conclusion_boundary='These adaptation scores measure forecast error. They do not establish improved review precision, bank distress, or money saved. Any review policy needs its own fixed-capacity evaluation.',
    kicker='Keep the decision tied to its metric', chip_text='CHECK',
)
'''
    return ('Bonus · Does fine-tuning earn its place?', prose, source, True)


def arrange_story():
    """Put the short experiment first and the detailed audits after its answer."""
    (
        introduction,
        history,
        experiment_zero,
        eligibility,
        split,
        target,
        mechanics,
        features_and_model,
        evaluation,
        tail_errors,
        ranking,
        anomaly_section,
        conclusion,
        seeds,
        bootstrap,
        uninsured,
        experiment_zero_notes,
    ) = sections

    # The 2013-onward EDA stays up front. The older-report investigation uses
    # the same loaded data, but appears only after the core model result.
    history_title, history_prose, history_code, _ = history
    audit_start = '# Reporting metadata lets us explain missing equity'
    audit_end = '\ngate = {'
    early_history, audit_and_gate = history_code.split(audit_start, 1)
    detailed_audit, gate_tail = audit_and_gate.split(audit_end, 1)
    history = (
        '1 · What is in the 2013–2024 data?',
        history_prose,
        early_history + '\ngate = {' + gate_tail,
        False,
    )
    comparability_appendix = (
        'Deeper check · What changed in older reports?',
        'The pre-2013 rows remain an audit population. Inspect their missingness and reporting forms here, after the forecasting result.',
        audit_start + detailed_audit,
        False,
    )

    # Define the time windows as soon as eligible rows exist. Present their
    # visual receipt after the target decision, where the reader needs it.
    split_title, split_prose, split_code, _ = split
    setup_start = '# The one-quarter gaps keep each split'
    setup_end = '\nassert train'
    before_setup, setup_and_display = split_code.split(setup_start, 1)
    setup_body, display_rest = setup_and_display.split(setup_end, 1)
    eligibility = (
        '2 · Which bank-quarters have an observable outcome?',
        eligibility[1],
        eligibility[2] + '\n# %% NOTEBOOK CELL\n' + setup_start + setup_body,
        False,
    )
    split = (
        '7 · Did a future outcome enter training?',
        split_prose,
        before_setup + '\nassert train' + display_rest,
        False,
    )

    # The training-only EDA precedes the target choice. Preprocessing and
    # fitting remain together later, after the split has been explained.
    eda_code, model_code = features_and_model[2].split(
        '# %% NOTEBOOK CELL\n# Learn missing-value replacements', 1
    )
    features_before_time, time_and_correlations = eda_code.split(
        '# %% NOTEBOOK CELL\n# A time line', 1
    )
    time_body, feature_correlations = time_and_correlations.split(
        '# Correlation answers whether', 1
    )
    time_section = (
        '3 · Do deposit declines repeat over time?',
        'Training quarters only. Each dot in the quarter-of-year view is a separate year; this is a descriptive check, not a new model input.',
        '# A time line' + time_body,
        False,
    )
    feature_section = (
        '4 · What do the five inputs look like?',
        features_and_model[1].split('Our comparison ladder')[0].strip(),
        features_before_time + '# %% NOTEBOOK CELL\n# Correlation answers whether' + feature_correlations,
        False,
    )
    model_section = (
        '9 · How are the four forecasts fitted?',
        '''**Training changes weights; validation chooses settings and stopping time.** Ridge is selected using validation MAE after conversion to percentage growth. MLP early stopping monitors MSE in log growth and restores its best validation weights. Those selection criteria differ. We preserve this original comparison and report both percentage-point metrics afterward. Exponentiating a mean log forecast gives a back-transformed point forecast; it generally differs from expected arithmetic growth. The 2024 evaluation has already been examined during this project.''',
        '# Learn missing-value replacements' + model_code,
        False,
    )

    # One short finding motivates the target choice. The full replication of
    # Experiment 0 remains available with the deeper checks.
    discovery = (
        '5 · Why revisit the original dollar target?',
        'The archived dollar experiment was dominated by institution size. This is the clue that motivates a proportional target.',
        '''
legacy = json.loads(
    (ROOT / 'experiments/experiment_0/outputs/run_summary.json').read_text()
)
wm_counterintuitive_card(
    title='What a novice might overlook',
    theme=theme,
    why_misread='Capturing about 85% of decline dollars sounds impressive.',
    ordinary_process=(
        f'Ranking by bank size alone captured {100 * legacy["size_capture"]:.2f}% '
        'at the same review capacity.'
    ),
    conclusion_boundary=(
        f'The original network added {legacy["network_difference_pp"]:.2f} '
        'percentage points. The dollar weighting made size a strong baseline.'
    ),
    kicker='Original experiment',
    chip_text='LOOK TWICE',
)
''',
        False,
    )
    experiment_zero = (
        'Appendix · Rebuild Experiment 0',
        experiment_zero[1],
        experiment_zero[2],
        True,
    )

    target = ('6 · What should the model predict?', target[1], target[2], False)
    mechanics = ('8 · What does the network learn?', mechanics[1], mechanics[2], False)
    evaluation = ('10 · Which forecast errs least?', evaluation[1], evaluation[2], False)
    tail_errors = ('11 · Where are the large misses?', tail_errors[1], tail_errors[2], False)
    ranking = ('12 · Which selected banks had weak growth?', ranking[1], ranking[2], False)
    anomaly_section = (
        '13 · Would unusual reports give us a better review list?',
        anomaly_section[1],
        anomaly_section[2],
        False,
    )
    conclusion = ('14 · What did we learn?', conclusion[1], conclusion[2], False)

    sections[:] = [
        introduction,
        history,
        eligibility,
        time_section,
        feature_section,
        discovery,
        target,
        split,
        mechanics,
        model_section,
        evaluation,
        tail_errors,
        ranking,
        anomaly_section,
        conclusion,
        timesfm_extension(),
        finetuning_extension(),
        comparability_appendix,
        seeds,
        bootstrap,
        uninsured,
        experiment_zero,
        experiment_zero_notes,
    ]


SECTION_SUMMARIES = {'1 ·': 'We chose 2013–2024. First learn the column names, then see why older reports stay in an audit and how we handle missing values.', '2 ·': 'A forecast needs a real next-quarter answer. We keep comparable domestic-bank reports with neighboring quarters and positive balances, and count every exclusion.', '3 ·': 'We check when deposit declines happened and whether the same quarter of the year tends to repeat a pattern. Only training-period outcomes are used.', '4 ·': 'Five inputs describe bank size, cash, loans, equity, and recent deposit growth. Their distributions show what the model receives before any scaling.', '5 ·': 'The first dollar-based score mostly rewarded choosing large banks. A size-only rule nearly matched the neural network, motivating a proportional-growth target.', '6 ·': 'We predict log growth so very small starting balances do not create enormous training targets. The same bank appears in two representations; predictions are converted back to percentage growth for scoring.', '7 ·': 'Training comes first, validation comes later, and 2024 comes last. A timeline shows the gaps that keep future outcomes out of earlier training.', '8 ·': 'A network learns by adjusting weights. One small calculation shows how an update reduces a mistake before we examine the full network.', '9 ·': 'We fit four forecasts and use validation to choose settings. The learning curve shows when more training stops helping.', '10 ·': 'Zero growth has the lowest average absolute error; the small MLP has the lowest RMSE. We compare both scores because large mistakes receive extra weight in RMSE.', '11 ·': 'The models make larger mistakes on the weakest-growth banks. These after-the-fact groups explain errors; they cannot prove advance warning.', '12 ·': 'With room to review only 10% of banks, Ridge leads in March and June and the MLP leads in September. Both find weak-growth banks more often than the roughly 10% random reference.', '13 ·': 'Anomaly detection could flag unusual reports for review. It answers a different question from forecasting, and this project has not tested its review-list quality.', '14 ·': 'The simple baseline remains hard to beat. The MLP improves RMSE but does not establish dependable added value; review-list usefulness needs its own evidence.', 'Bonus · Can': 'We tested TimesFM and Chronos on every evaluation row. Neither zero-shot approach beat the best core baselines; one very large TimesFM miss dominates its RMSE.', 'Bonus · Does': 'We fine-tuned both forecasting heads using earlier data and selected checkpoints with validation. Neither adapted model beat the core baselines on the reused 2024 evaluation.', 'Deeper check ·': 'We kept 2010–2012 out of the main comparison because reporting definitions still need a historical mapping. Here are the counts and reporting-form checks behind that choice.', 'Deeper lesson · Does': 'Changing starting weights changes the validation score. Three fixed seeds show that sensitivity without choosing a winner from 2024.', 'Deeper lesson · How': 'We resample whole banks to see how much the MLP–Ridge difference varies. The interval crosses zero, so the small observed advantage is uncertain.', 'Deeper lesson · Can': 'A blank, a stored zero, and a nonzero deposit value mean different things. We keep these uninsured-deposit columns out until their reporting definitions are verified.', 'Appendix · Rebuild': 'This reproduces the original dollar-capture experiment. The network adds only 0.27 percentage points over selecting banks by size.', 'Appendix · Keep': 'The original experiment preserves useful lessons about probabilities, review capacity, and dollar outcomes. Use these as deeper questions after finishing the core comparison.'}
PROJECT_SUMMARY = '**Project in one minute.** Can a bank’s reports help us predict next-quarter deposit growth and choose which banks to review? We train on 214,425 bank-quarter records and evaluate 13,532 later records. Zero growth wins average absolute error; the small neural network wins RMSE, which gives large misses extra weight. Ridge and the neural network identify weak-growth banks more often than random selection in three historical quarters. This is a research prototype, with a reused 2024 holdout and no measured savings.\n\n**What you will see:** data definitions and cleaning decisions → time patterns → five inputs and a growth target → time-separated training → forecast errors → a capacity-limited review list. Pretrained models, fine-tuning, and deeper audits follow the main conclusion.\n\n**Tools:** Python, pandas, scikit-learn, TensorFlow, Plotly, Jupyter, and uv; PyTorch and MLX for the foundation-model extensions.'

SECTION_SUMMARIES.update({
    'Follow-up · Do': 'The original four-model comparison left simple median and size-only rules untested. Here we fit them using earlier data and measure their results on the reused 2024 period.',
    'Follow-up · What': 'A verified merger and a bounded liquidation window change which balance movements we are measuring. We retain primary scores and calculate a separate sensitivity.',
    'Next decision ·': 'A selected bank gives a human a place to begin investigating its funding, deposit composition, and history.',
    'Bonus · Does a short': 'A single short-history case dominates TimesFM error. Compare identical bank populations across models within each history band.',
})

def section_summary(title):
    return next((SECTION_SUMMARIES[prefix] for prefix in sorted(SECTION_SUMMARIES, key=len, reverse=True) if title.startswith(prefix)), '')

def readable_blocks(source):
    """Preserve statements while giving long cells visible stopping points."""
    formatter = shutil.which("ruff")
    if formatter:
        source = subprocess.run(
            [formatter, "format", "--stdin-filename", "lesson.py", "-"],
            input=source, text=True, capture_output=True, check=True,
        ).stdout
    # Keep each chart's reading guide with the visible calculation it explains.
    calls = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'chart'
             and len(n.args) > 1 and isinstance(n.args[1], ast.Constant)]
    prefix = "\n".join(f"READING_GUIDES[{n.args[1].value!r}] = {GUIDES[n.args[1].value]!r}"
                       for n in calls if n.args[1].value in GUIDES)
    if prefix: source = prefix + "\n" + source
    lines = source.splitlines()
    if len(lines) <= 80:
        return [source.strip()]
    statements = ast.parse(source).body
    blocks, start, last_end = [], 0, 0
    for statement in statements:
        if statement.end_lineno - start > 65 and last_end > start:
            blocks.append("\n".join(lines[start:last_end]).strip())
            start = last_end
        last_end = statement.end_lineno
    blocks.append("\n".join(lines[start:]).strip())
    return [block for block in blocks if block]

def build():
    arrange_story()
    teach_story(sections)
    apply_concrete_teaching(sections)
    for edition,name in [('masterclass','FDIC_Deep_Learning_Masterclass.ipynb'),('submission','FDIC_Deep_Learning_Submission.ipynb')]:
        cells=[]
        for index,(title,prose,source,advanced) in enumerate(sections):
            if advanced and edition=='submission':continue
            # Keep the same analysis code in both deliverables; reduce teaching prose only.
            lead = OPENING if index == 0 else '**In plain terms:** ' + prose
            if index > 0: prose = ''
            cells.append(nbf.v4.new_markdown_cell(('# ' if index==0 else '## ')+title+'\n\n'+lead+'\n\n'+prose))
            if source:
                if index==0:
                    source=f"NOTEBOOK_EDITION = {edition!r}\nREADING_GUIDES = {{}}\n# %% NOTEBOOK CELL\n"+source

                # A teaching notebook should reveal one idea at a time. Source
                # sections can opt into visible pauses without duplicating the
                # surrounding prose or changing execution order.
                code_blocks = source.split('# %% NOTEBOOK CELL')
                cells.extend(
                    nbf.v4.new_code_cell(part)
                    for block in code_blocks
                    if block.strip()
                    for part in readable_blocks(block.strip())
                )
        for i,c in enumerate(cells):c.id=hashlib.sha256(f'{edition}:{i}:{c.source}'.encode()).hexdigest()[:12]
        nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'DL Assignment (.venv)','language':'python','name':'python3'},'language_info':{'name':'python'}})
        nbf.validate(nb)
        nbf.write(nb, ROOT / name)
        print(name,len(cells),'cells')
if __name__=='__main__':build()
