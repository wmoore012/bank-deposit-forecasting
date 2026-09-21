"""Build two visible-code notebooks from one experiment; preserve Experiment 0."""
from pathlib import Path
from textwrap import dedent
import hashlib
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
Our quarterly balances describe report-to-report changes. They cannot identify a bank run or distinguish withdrawals from mergers.

The original experiment asked where decline dollars would concentrate. We preserve that discovery,
then ask how large each change was relative to the bank’s own deposits.

**Reading route:** trust the rows → diagnose dollar size → choose the target → protect time → train → compare → inspect weak outcomes.
All modeling code stays visible. Advanced lessons follow the core answer.

**Evidence boundary:** 2024 was already examined in the original project. It is a reused historical holdout.
Report publication times and historical vintages are unavailable; this is retrospective forecasting,
not a demonstrated live warning system. The same banks may appear in earlier and later periods.
''',r'''
# EXEMPLAR: bootstrap
import os, sys, json, hashlib, platform, time, html, textwrap
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
from wm_notecards.cards import preview_card, takeaway_card, wm_formula_card
from wm_notecards.charts import style_fig_wm, wm_render_figure_card
from wm_notecards.tables import wm_render_styler
SEED=42
FEATURES=['log_deposits','cash_ratio','loan_ratio','equity_ratio','prior_growth']
MODEL_COLORS={'Zero growth':'#737B86','Persistence':'#AF7721','Ridge':'#526EAA','MLP':'#007F89'}
ROOT=Path.cwd()
assert (ROOT/'data/fdic_financials_2020_2024.csv').is_file(), 'Run from the project folder.'
OUT=ROOT/'growth_outputs'/NOTEBOOK_EDITION
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'charts').mkdir(exist_ok=True)
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.experimental.enable_op_determinism()
tf.keras.utils.set_random_seed(SEED)
theme=WMTheme(width=860,height=480,accent='#007F89',plot_bg='#FFFFFF')
init_notebook(expand_colab_outputs=True)
# One inline library makes saved chart outputs independent of a CDN.
display(HTML('<script>'+get_plotlyjs()+'</script>'))
def table(frame,title,formats=None):
    default_formats={c: (lambda v: v.strftime('%Y-%m-%d')) for c in frame.select_dtypes(include=['datetime'])}
    default_formats.update({c:'{:,.3f}' for c in frame.select_dtypes(include=['floating'])})
    default_formats.update(formats or {})
    styled=frame.style.hide(axis='index').format(default_formats,na_rep='Missing')
    wm_render_styler(styled,theme=theme,title=title,
        wrap_columns={c:290 for c in frame if pd.api.types.is_string_dtype(frame[c])})
def chart(fig,name,title,subtitle='',height=540):
    style_fig_wm(fig,title=title,subtitle=subtitle,theme=theme,normalize_legacy_colors=False,
                 category_policy="preserve",allow_dense_categories=True)
    title_html='<b>'+'<br>'.join(html.escape(t) for t in textwrap.wrap(title,52))+'</b>'
    if subtitle:
        title_html+='<br><span style="font-size:14px">'+'<br>'.join(html.escape(t) for t in textwrap.wrap(subtitle,88))+'</span>'
    fig.update_layout(width=860,height=height,font=dict(size=14,family='Inter, Arial, sans-serif'),
        title=dict(text=title_html,font=dict(size=24),x=.035,y=.98,xanchor='left',yanchor='top'),
        hovermode='closest',hoverlabel=dict(bgcolor='white',bordercolor='#B8C2CC',font=dict(size=14,color='#182E3A',family='Inter, Arial, sans-serif')),legend=dict(y=-.24,yanchor='top',font=dict(size=14,family='Inter, Arial, sans-serif'),title_text=''),
        margin=dict(l=85,r=45,t=155,b=120),paper_bgcolor='white',plot_bgcolor='white')
    fig.update_xaxes(automargin=True,showline=True,linecolor='#ADB5BD',gridcolor='#ECEFF1',
        tickfont=dict(size=14,family='Inter, Arial, sans-serif'),title_font=dict(size=15,family='Inter, Arial, sans-serif'))
    fig.update_yaxes(automargin=True,showline=True,linecolor='#ADB5BD',gridcolor='#ECEFF1',
        tickfont=dict(size=14,family='Inter, Arial, sans-serif'),title_font=dict(size=15,family='Inter, Arial, sans-serif'))
    fig.for_each_yaxis(lambda axis: axis.update(dtick=1) if axis.type=='log' else None)
    fig.write_json(OUT/'charts'/f'{name}.json')
    wm_render_figure_card(fig,theme=theme,file_stub=name)
def takeaway(title,body,metric=None):
    takeaway_card(title=title,body=body,metric=metric,theme=theme)
def scores(y,pred):
    return {'MAE (pp)':100*mean_absolute_error(y,pred),
            'RMSE (pp)':100*np.sqrt(mean_squared_error(y,pred))}
preview_card(title='Can the network earn its extra complexity?',theme=theme,
    body='Every model gets the same dates and outcomes. The neural network must improve the forecast, not just the vocabulary.',
    bullets=['One row: one institution at one reporting quarter.',
             'One target: next-quarter domestic deposit growth.',
             'One honest comparison: simple rules, Ridge, and a small MLP.'])
''')
section('1 · Can fifteen years of reports tell one consistent story?', '''
**More rows help only when they describe comparable things.** The long extract already exists.
We check its keys, all 60 quarters, missingness, and exact agreement with the original five-year extract.
Then we check the reporting context. A field with the same name can still cross a reporting change.

The source dictionary describes domestic deposits, assets, net loans, cash, and equity.
The extract includes 5,738 form-100 reports in 2010–2011. The FDIC documents the
[2012 conversion from Thrift Financial Reports to Call Reports](https://www.fdic.gov/news/inactive-financial-institution-letters/2012/fil12010.html).
A five-field historical crosswalk has not been verified. The gate below records
what was verified and what remains unresolved before choosing the modeling period.
''',r'''
# EXEMPLAR: missingness-evidence
long=pd.read_csv(ROOT/'data/fdic_financials_2010_2024.csv')
short=pd.read_csv(ROOT/'data/fdic_financials_2020_2024.csv')
reporting=pd.read_csv(ROOT/'data/fdic_reporting_2010_2024.csv')
keys=['CERT','REPDTE']
assert not long.duplicated(keys).any()
assert not reporting.duplicated(keys).any()
cols=list(short)
ordered=lambda d:d.sort_values(keys).reset_index(drop=True)
overlap_equal=ordered(short).equals(ordered(long.loc[long.REPDTE>=20200331,cols]))
assert overlap_equal and long.REPDTE.nunique()==60
long_audit=long.merge(reporting,on=keys,how='left',validate='one_to_one',indicator=True)
coverage=long_audit.groupby('REPDTE').agg(Rows=('CERT','size'),
    Equity_missing=('EQ',lambda s:s.isna().mean()),
    Metadata_matched=('_merge',lambda s:s.eq('both').mean())).reset_index()
coverage['Date']=pd.to_datetime(coverage.REPDTE.astype(str))
coverage.to_csv(OUT/'history_coverage.csv',index=False)
missing=long.isna().sum().rename('Missing rows').rename_axis('Field').reset_index()
table(missing,'All source fields: missing counts in 359,497 reports')
fig=px.bar(missing,x='Missing rows',y='Field',orientation='h',color_discrete_sequence=['#AF7721'])
chart(fig,'missingness','Where are source values missing?','2010–2024; no values filled')
by_form=long_audit.groupby(['BKCLASS','CALLFORM'],dropna=False).agg(
    Rows=('CERT','size'),Missing_equity=('EQ',lambda s:s.isna().sum())).reset_index()
by_form.to_csv(OUT/'reporting_forms.csv',index=False)
table(by_form.loc[by_form.Missing_equity.gt(0)],'Which reporting groups lack equity?')
fig=px.line(coverage,x='Date',y='Equity_missing',markers=True,color_discrete_sequence=['#AF7721'])
fig.update_yaxes(tickformat='.2%',title='Reports missing equity')
chart(fig,'coverage_time','Does missing equity change over time?','All long-history source reports')
# A current dictionary and identical overlap do not establish pre-2020 form comparability.
gate={'all_60_quarters':True,'duplicate_keys':0,'overlap_exact':overlap_equal,
      'historical_form_crosswalk_verified':False,'selected_start':2020,
      'reason':'5738 form-100 reports precede the documented 2012 TFR-to-Call-Report conversion. A verified five-field crosswalk is unavailable. Retain 2020–2024.'}
(OUT/'comparability_gate.json').write_text(json.dumps(gate,indent=2))
raw=short.copy()
source_hash=hashlib.sha256((ROOT/'data/fdic_financials_2020_2024.csv').read_bytes()).hexdigest()
raw_fingerprint=pd.util.hash_pandas_object(raw,index=True).sum()
takeaway('Use 2020–2024 for the primary experiment',
    'All 60 quarters are present and the overlapping rows match exactly. Historical reporting-form continuity remains unresolved, so the longer history stays an audited extension.')
''')
section('2 · What happened when the first experiment counted dollars?', '''
**Large banks can dominate a dollar-based score even when the model learns little beyond size.**
A 1% decline at a bank with USD 100 billion in deposits is USD 1 billion.
A 10% decline at a bank with USD 100 million is USD 10 million.
The first bank contributes 100 times as many decline dollars despite its smaller proportional decline.

Reproduce the old eligibility and training-size boundaries before quoting the old concentration.
The archived masterclass preserves the two-head network, PR/calibration lessons, and review-capacity analysis.
''',r'''
# EXEMPLAR: counterintuitive-boundary
old=short.sort_values(keys).copy()
old['date']=pd.to_datetime(old.REPDTE.astype(str))
old['q']=old.date.dt.to_period('Q').astype('int64')
g=old.groupby('CERT')
for new,col,shift in [('prev','DEPDOM',1),('next','DEPDOM',-1),('prev_q','q',1),('next_q','q',-1)]:
    old[new]=g[col].shift(shift)
old=old.loc[(old.q-old.prev_q).eq(1)&(old.next_q-old.q).eq(1)&old.DEPDOM.ge(10000)&old.prev.gt(0)&old.next.ge(0)&old.ASSET.gt(0)].copy()
old['runoff_m']=(old.DEPDOM-old.next).clip(lower=0)/1000
old_train=old.loc[old.date.le('2022-09-30')]
old_test=old.loc[old.date.between('2024-01-01','2024-09-30')].copy()
edges=np.r_[-np.inf,old_train.DEPDOM.quantile([.25,.5,.75]),np.inf]
old_test['Size group']=pd.cut(old_test.DEPDOM,edges,labels=['Smallest','Lower middle','Upper middle','Largest'])
concentration=old_test.groupby('Size group',observed=True).agg(Rows=('CERT','size'),Decline_m=('runoff_m','sum')).reset_index()
concentration['Dollar share']=concentration.Decline_m/concentration.Decline_m.sum()
table(concentration,'Experiment 0: 2024 Q1–Q3 predictor rows',{'Dollar share':'{:.2%}','Decline_m':'{:,.1f}'})
fig=px.bar(concentration,x='Size group',y='Dollar share',text=concentration['Dollar share'].map('{:.2%}'.format),color_discrete_sequence=['#007F89'])
fig.update_yaxes(tickformat='.0%',range=[0,1.08],title='Share of decline dollars')
chart(fig,'experiment0','Where did the decline dollars sit?','Size cutoffs learned from the original training period')
share=float(concentration.loc[concentration['Size group'].eq('Largest'),'Dollar share'].iloc[0])
legacy=json.loads((ROOT/'experiments/experiment_0/outputs/run_summary.json').read_text())
comparison=pd.DataFrame({'Method':['Size rule','Original base network'],
    'Quarter-average capture (%)':[100*legacy['size_capture'],100*legacy['network_capture']]})
table(comparison,'Archived Experiment 0: 10% review capacity',{'Quarter-average capture (%)':'{:.2f}'})
fig=px.scatter(comparison,x='Quarter-average capture (%)',y='Method',color_discrete_sequence=['#007F89'])
chart(fig,'old_capture','How much did the network add?','Focused scale; exact percentages in the adjacent table')
takeaway('Bank size explains much of the dollar result',
    f'The largest training-size group contains {share:.2%} of historical decline dollars. The original network added {legacy["network_difference_pp"]:.2f} percentage points of quarter-average capture over size alone.',f'{share:.2%}')
concentration.to_csv(OUT/'experiment0_concentration.csv',index=False)
''')
section('3 · Which rows can actually teach us about next quarter?', '''
**A missing next report is an unknown outcome.** It can reflect a merger, closure, reporting gap,
or the end of our file. We separate these cases as far as the data allows.

A certificate connects reports from the same recorded institution. It does not guarantee the business
stayed unchanged. We preserve name changes and structural event codes for review.
All 1,018 missing equity values in the long extract occur on form 2, across classes OI and NC.
Keep the primary population to insured domestic-bank classes N, NM, SM, SB, SI, and SL
using forms 31, 41, or 51. This avoids filling foreign-branch structural gaps with a domestic-bank median.
''',r'''
# EXEMPLAR: decision-ledger
panel=raw.merge(reporting,on=keys,how='left',validate='one_to_one',indicator=True).sort_values(keys).copy()
assert panel['_merge'].eq('both').all(), 'Reporting metadata must match every source row.'
panel['date']=pd.to_datetime(panel.REPDTE.astype(str))
panel['q']=panel.date.dt.to_period('Q').astype('int64')
g=panel.groupby('CERT',sort=False)
for new,col,shift in [('prior_deposits','DEPDOM',1),('next_deposits','DEPDOM',-1),
    ('prior_q','q',1),('next_q','q',-1),('target_date','date',-1),('next_name','NAME',-1),('next_event','ACTEVT',-1)]:
    panel[new]=g[col].shift(shift)
panel['Next report status']=np.select([
    panel.q.eq(panel.q.max()),panel.next_q.isna(),panel.next_q.sub(panel.q).ne(1)],
    ['Dataset end','Institution stops before dataset end','Gap before later report'],default='Adjacent next report')
missing_next=panel.groupby('Next report status').size().reset_index(name='Rows')
table(missing_next,'A missing future balance stays missing')
fig=px.bar(missing_next,x='Rows',y='Next report status',orientation='h',color_discrete_sequence=['#007F89'])
chart(fig,'next_report','Which outcomes are observable?')
panel.loc[panel['Next report status'].ne('Adjacent next report'),keys+['NAME','date','ACTEVT','Next report status']].to_csv(OUT/'unobserved_outcomes.csv',index=False)
conditions={
    'Insured domestic bank; comparable Call Report':panel.BKCLASS.isin(['N','NM','SM','SB','SI','SL'])&panel.CALLFORM.isin([31,41,51]),
    'Adjacent prior quarter':panel.q.sub(panel.prior_q).eq(1),
    'Adjacent next quarter':panel.next_q.sub(panel.q).eq(1),
    'Positive current and prior deposits':panel.DEPDOM.gt(0)&panel.prior_deposits.gt(0),
    'Nonnegative next deposits and positive assets':panel.next_deposits.ge(0)&panel.ASSET.gt(0)}
keep=pd.Series(True,index=panel.index);ledger=[{'Rule':'All source rows','Remaining':len(panel),'Removed':0}]
for rule,mask in conditions.items():
    before=int(keep.sum());keep &= mask
    ledger.append({'Rule':rule,'Remaining':int(keep.sum()),'Removed':before-int(keep.sum())})
rows=panel.loc[keep].copy()
rows['growth']=(rows.next_deposits-rows.DEPDOM)/rows.DEPDOM
rows['dollar_change_m']=(rows.next_deposits-rows.DEPDOM)/1000
rows['log_growth']=np.log(rows.next_deposits.where(rows.next_deposits.gt(0))/rows.DEPDOM)
rows['prior_growth']=(rows.DEPDOM-rows.prior_deposits)/rows.prior_deposits
rows['log_deposits']=np.log(rows.DEPDOM)
for name,col in [('cash_ratio','CHBAL'),('loan_ratio','LNLSNET'),('equity_ratio','EQ')]:rows[name]=rows[col]/rows.ASSET
assert np.isfinite(rows.growth).all()
assert rows.growth.ge(-1).all()
assert len(panel)-len(rows)==sum(r['Removed'] for r in ledger)
table(pd.DataFrame(ledger),'Every exclusion has a count')
chart(px.bar(pd.DataFrame(ledger),x='Remaining',y='Rule',orientation='h',color_discrete_sequence=['#007F89']),
      'eligibility','Which rows enter the forecasting population?')
takeaway('The new experiment keeps small deposit bases',
    f'{len(rows):,} eligible bank-quarters remain. Positive denominators replace the old USD 10 million minimum. Extreme proportional changes stay visible; disappearance causes remain unresolved without event history.')
''')
section('4 · How do we keep tomorrow out of today’s inputs?', '''
**Give each model only information from the predictor quarter or earlier.**
If a row describes September, its label describes December. December must not sneak into September’s features.

We leave 2022 Q4 and 2023 Q4 predictor rows out at the boundaries. Training outcomes therefore precede
validation predictor dates; validation outcomes precede the historical evaluation predictor dates.
Repeated banks are expected. We are testing future quarters for this population, not entirely new institutions.
''',r'''
# EXEMPLAR: quality-check
train=rows.loc[rows.date.le('2022-09-30')].copy()
valid=rows.loc[rows.date.between('2023-01-01','2023-09-30')].copy()
holdout_mask=rows.date.between('2024-01-01','2024-09-30')
assert train.target_date.max()<valid.date.min()
assert valid.target_date.max()<rows.loc[holdout_mask,'date'].min()
assert set(FEATURES).isdisjoint({'growth','next_deposits','target_date','log_growth','dollar_change_m'})
split_summary=pd.DataFrame([{'Period':name,'Rows':len(f),'Banks':f.CERT.nunique(),
    'First predictor':f.date.min(),'Last predictor':f.date.max(),'Last outcome':f.target_date.max()}
    for name,f in [('Training',train),('Validation',valid),('Reused holdout',rows.loc[holdout_mask])]])
table(split_summary,'Accounting dates define this retrospective experiment')
fig=go.Figure()
for i,r in split_summary.iterrows():
    fig.add_trace(go.Scatter(x=[r['First predictor'],r['Last predictor']],y=[r.Period]*2,
        mode='lines+markers',name=r.Period,line=dict(width=9,color=['#007F89','#AF7721','#526EAA'][i])))
fig.update_xaxes(title='Predictor report date');fig.update_layout(showlegend=False)
chart(fig,'split','The model must live in time','Outcomes occur one quarter after each predictor report')
split_summary.to_csv(OUT/'splits.csv',index=False)
takeaway('Fit on the past; judge on later reports','Only training rows may set imputation values and scaling parameters. Validation selects training duration and Ridge strength. The reused holdout measures the frozen design.')
''')
section('5 · Dollars, percentages, or logs: what are we asking?', '''
**Choose the meaning before looking for a better score.** Start with USD 100 million in deposits.
If next quarter has USD 90 million, the change is −USD 10 million, growth is −10%, and log growth is about −0.105.
Each describes the same observation. Each asks the model to care about something different.

Dollar change emphasizes funding magnitude. Percentage growth measures movement relative to the bank’s own base.
Log growth describes multiplicative movement and requires both deposit balances to be positive.
A zero next balance is −100% growth but has no finite log growth.

First inspect the full training distributions. Then zoom into the middle 98% for readability.
That zoom changes the chart window only. **Every eligible training outcome stays in the model.**
''',r'''
# EXEMPLAR: formula-card
wm_formula_card(title='From a balance to a change',theme=theme,items=[
    {'label':'Signed growth','fallback':'(90 million − 100 million) / 100 million = −0.10 = −10%'},
    {'label':'Log growth','fallback':'ln(90 / 100) ≈ −0.1054'}])
targets={'Dollar change (USD million)':'dollar_change_m','Signed growth (%)':'growth','Log growth':'log_growth'}
summary=[]
for label,col in targets.items():
    s=train[col].dropna(); multiplier=100 if col=='growth' else 1
    q=s.quantile([0,.01,.5,.99,1])*multiplier
    summary.append({'Target':label,'Defined rows':len(s),'Undefined rows':len(train)-len(s),
        'Minimum':q.iloc[0],'1st percentile':q.iloc[1],'Median':q.iloc[2],'99th percentile':q.iloc[3],'Maximum':q.iloc[4]})
    counts,bins=np.histogram(s*multiplier,bins=70)
    f=go.Figure(go.Bar(x=(bins[:-1]+bins[1:])/2,y=counts,width=np.diff(bins)*.98,marker_color='#007F89'))
    f.update_xaxes(title=label);f.update_yaxes(title='Bank-quarter count',type='log')
    chart(f,'target_full_'+col,'How far does '+label.lower()+' extend?','Full untouched training range; logarithmic count axis')
    lo,hi=s.quantile([.01,.99]);central=s[s.between(lo,hi)]*multiplier
    f=px.histogram(x=central,nbins=50,color_discrete_sequence=['#007F89'])
    f.update_layout(bargap=.02);f.update_xaxes(title=label);f.update_yaxes(title='Bank-quarter count')
    chart(f,'target_middle_'+col,'Where do most '+label.lower()+' values sit?',f'Display-only 1st–99th percentile view; {len(s)-len(central):,} values outside this view')
summary=pd.DataFrame(summary);table(summary,'Training targets: full-range arithmetic', {c:'{:,.4f}' for c in ['Minimum','1st percentile','Median','99th percentile','Maximum']})
summary.to_csv(OUT/'target_comparison.csv',index=False)
extremes=train.loc[train.growth.abs().nlargest(12).index,
    ['CERT','NAME','date','target_date','DEPDOM','next_deposits','growth','next_name','next_event']].copy()
extremes['Name changed']=extremes.NAME.ne(extremes.next_name)
extremes['Review']='Adjacent same-certificate reports; economic cause unresolved. Retained without clipping.'
table(extremes[['CERT','NAME','date','DEPDOM','next_deposits','growth','Name changed']],
    'Largest training changes: balances in USD thousands',{'growth':'{:+.2%}'})
extremes.to_csv(OUT/'extreme_review.csv',index=False)
# Preserve the surrounding observations so the change can be audited beyond a single pair.
panel.loc[panel.CERT.isin(extremes.CERT),keys+['NAME','date','DEPDOM','ASSET','ACTEVT','CALLFORM']].to_csv(OUT/'extreme_bank_histories.csv',index=False)
TARGET='growth'
target_decision={'target':TARGET,'units':'fraction; errors reported in percentage points',
    'reason':'Relative signed movement directly answers the business question; zero next deposits remain defined.',
    'clipping':None,'extreme_treatment':'Retain; same-certificate continuity does not establish economic cause.',
    'training_only':True}
(OUT/'target_decision.json').write_text(json.dumps(target_decision,indent=2))
takeaway('Predict signed percentage growth','The target preserves direction and compares change with each bank’s own deposit base. Small bases can produce enormous growth, so the full distribution and extreme histories remain part of the evidence.')
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
# EXEMPLAR: formula-card
wm_formula_card(title='One neuron, one visible calculation',theme=theme,items=[
    {'label':'Weighted sum','fallback':'z = 2 × 0.5 + 3 × (−0.2) + 0.1 = 0.5'},
    {'label':'ReLU','fallback':'max(0, z) = 0.5'},
    {'label':'Squared error','fallback':'If the answer is 0.8: (0.5 − 0.8)² = 0.09'}])
# A hand-checkable gradient step, illustrative rather than model evidence.
x_demo=np.array([2.,3.]);w_demo=np.array([.5,-.2]);bias_demo=.1;answer_demo=.8
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
    marker=dict(size=22,color='#007F89'),line=dict(color='#526EAA')))
fig.update_xaxes(visible=False,range=[-.5,3.5]);fig.update_yaxes(visible=False,range=[-.3,.5])
chart(fig,'architecture','Five inputs become one growth forecast','737 trainable weights and biases',height=380)
''')
section('7 · Give every model the same information', '''
**Transform the rulers, not the answer key.** A deposit balance and an equity ratio have different scales.
Standardization subtracts a training mean and divides by a training standard deviation.
A missing predictor gets its training median. Validation and historical rows reuse those exact values.

Before transforming anything, inspect the five inputs on the training period. The summary shows scale and
extremes. The correlation view asks whether the inputs repeat the same information. The prior-versus-next
scatter asks whether “repeat last quarter” has visible signal before we give that rule a formal score.

Our comparison ladder has four rungs: zero growth; repeat last quarter’s growth; Ridge; the small MLP.
Ridge strength is selected from a fixed grid using validation MAE. The MLP stops on validation MSE.
MAE describes the average absolute miss. RMSE gives large misses extra weight.
A forecast of +2% when reality is −3% misses by **5 percentage points**.
''',r'''
# EXEMPLAR: decision-ledger
feature_ledger=pd.DataFrame({'Input':FEATURES,'Calculation':['ln(DEPDOM)','CHBAL / ASSET','LNLSNET / ASSET','EQ / ASSET','(DEPDOM − prior DEPDOM) / prior DEPDOM'],
    'Timing':['Current quarter']*4+['Current and adjacent prior quarter']})
table(feature_ledger,'Five inputs; no future balance')
feature_summary=train[FEATURES].describe(percentiles=[.01,.25,.5,.75,.99]).T.reset_index(names='Feature')
table(feature_summary,'What do the five training inputs look like?',
    {c:'{:,.4f}' for c in feature_summary.select_dtypes(include='number').columns})
corr=train[FEATURES].corr(method='spearman')
shown=corr.mask(np.triu(np.ones_like(corr,dtype=bool)))
fig=go.Figure(go.Heatmap(z=shown.to_numpy(),x=FEATURES,y=FEATURES,zmin=-1,zmax=1,
    colorscale=[[0,'#AF7721'],[.5,'#F4F6F7'],[1,'#007F89']],text=shown.round(2).astype(str),
    texttemplate='%{text}',hovertemplate='%{y} vs %{x}<br>Spearman %{z:.3f}<extra></extra>',colorbar_title='Spearman'))
fig.update_yaxes(autorange='reversed')
chart(fig,'feature_correlations','Do the five inputs repeat the same information?',
    'Training rows only; lower triangle; Spearman correlation')
sample=train[['prior_growth','growth']].dropna().sample(n=min(5000,len(train)),random_state=SEED)
fig=px.scatter(sample,x='prior_growth',y='growth',opacity=.20,color_discrete_sequence=['#007F89'])
fig.add_hline(y=0,line_color='#737B86');fig.add_vline(x=0,line_color='#737B86')
fig.update_xaxes(title='Prior-quarter deposit growth',tickformat='.0%')
fig.update_yaxes(title='Next-quarter deposit growth',tickformat='.0%')
chart(fig,'persistence_eda','Does last quarter point toward next quarter?',
    'Random sample of 5,000 training rows; the formal persistence score comes later')
imputer=SimpleImputer(strategy='median')
scaler=StandardScaler()
X_train=scaler.fit_transform(imputer.fit_transform(train[FEATURES])).astype('float32')
X_valid=scaler.transform(imputer.transform(valid[FEATURES])).astype('float32')
y_train=train.growth.to_numpy(dtype='float32');y_valid=valid.growth.to_numpy(dtype='float32')
assert np.isfinite(X_train).all() and np.isfinite(X_valid).all()
assert np.allclose(imputer.statistics_,train[FEATURES].median())
preprocessing=pd.DataFrame({'Feature':FEATURES,'Training median':imputer.statistics_,
    'Training mean after imputation':scaler.mean_,'Training scale':scaler.scale_})
table(preprocessing,'Parameters learned only from training')
preprocessing.to_csv(OUT/'preprocessing.csv',index=False)
settings={'target':target_decision,'features':FEATURES,'seed':42,'additional_seeds':[7,99],
    'architecture':[5,32,16,1],'optimizer':'Adam','learning_rate':.001,'loss':'MSE',
    'batch_size':512,'epochs_max':200,'early_stopping_patience':10,
    'ridge_alphas':[.01,.1,1,10,100],'ridge_selection':'validation MAE',
    'historical_status':'reused 2024 holdout','clipping':None,
    'train_predictor_end':'2022-09-30','validation_predictors':['2023-01-01','2023-09-30'],
    'holdout_predictors':['2024-01-01','2024-09-30'],
    'tail_definition':'realized bottom 25% and 10% separately within each holdout quarter',
    'ranking':'lowest ceil(10% of rows) per quarter; ties broken by CERT',
    'bootstrap':'500 paired bank-cluster resamples; MAE difference MLP minus Ridge; fixed trained models',
    'eligibility':list(conditions),'source_sha256':source_hash}
(OUT/'design_plan.json').write_text(json.dumps(settings,indent=2))
ridge_candidates=[];ridge_models={}
for alpha in settings['ridge_alphas']:
    model=Ridge(alpha=alpha).fit(X_train,y_train);ridge_models[alpha]=model
    ridge_candidates.append({'Alpha':alpha,**scores(y_valid,model.predict(X_valid))})
ridge_tuning=pd.DataFrame(ridge_candidates)
best_alpha=float(ridge_tuning.sort_values(['MAE (pp)','Alpha']).iloc[0].Alpha)
ridge=ridge_models[best_alpha]
def build_network(seed):
    tf.keras.utils.set_random_seed(seed)
    network=tf.keras.Sequential([tf.keras.Input(shape=(5,)),tf.keras.layers.Dense(32,activation='relu'),
        tf.keras.layers.Dense(16,activation='relu'),tf.keras.layers.Dense(1)])
    network.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='mse')
    assert network.count_params()==737
    return network
def fit_network(seed):
    network=build_network(seed)
    # Explicit datasets bound CPU threads and preserve chronological validation.
    options=tf.data.Options();options.threading.private_threadpool_size=2
    ds=tf.data.Dataset.from_tensor_slices((X_train,y_train)).shuffle(len(y_train),seed=seed).batch(512).with_options(options)
    vs=tf.data.Dataset.from_tensor_slices((X_valid,y_valid)).batch(512).with_options(options)
    start=time.perf_counter()
    history=network.fit(ds,validation_data=vs,epochs=200,verbose=0,shuffle=False,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)])
    return network,history.history,time.perf_counter()-start
mlp,history,training_seconds=fit_network(SEED)
predict=lambda model,X:np.asarray(model(X,training=False)).ravel()
validation_predictions={'Zero growth':np.zeros(len(valid)),'Persistence':valid.prior_growth.to_numpy(),
    'Ridge':ridge.predict(X_valid),'MLP':predict(mlp,X_valid)}
validation_scores=pd.DataFrame([{'Model':n,**scores(y_valid,p)} for n,p in validation_predictions.items()])
table(validation_scores,'Validation chooses; historical evaluation waits',{'MAE (pp)':'{:.4f}','RMSE (pp)':'{:.4f}'})
fig=px.scatter(validation_scores,x='MAE (pp)',y='Model',color='Model',color_discrete_map=MODEL_COLORS)
fig.update_layout(showlegend=False)
chart(fig,'validation','Which method has the smallest validation miss?','2023 Q1–Q3 predictor rows; lower MAE is better')
frozen={**settings,'chosen_ridge_alpha':best_alpha,'mlp_best_epoch':int(np.argmin(history['val_loss'])+1),
    'validation_winner':validation_scores.sort_values('MAE (pp)').iloc[0].Model,
    'selection_complete_before_holdout_predictions':True}
(OUT/'frozen_plan.json').write_text(json.dumps(frozen,indent=2))
validation_scores.to_csv(OUT/'validation_scores.csv',index=False)
ridge_tuning.to_csv(OUT/'ridge_tuning.csv',index=False)
history_frame=pd.DataFrame(history).rename_axis('Epoch').reset_index();history_frame.Epoch+=1
history_frame.to_csv(OUT/'learning_curve.csv',index=False)
validation_audit=valid[['CERT','NAME','date','DEPDOM','next_deposits','growth','next_name']].copy()
validation_audit['Squared MLP error']=(validation_audit.growth-validation_predictions['MLP'])**2
validation_audit['Share of squared error']=validation_audit['Squared MLP error']/validation_audit['Squared MLP error'].sum()
validation_audit.nlargest(10,'Squared MLP error').to_csv(OUT/'validation_extremes.csv',index=False)
table(validation_audit.nlargest(3,'Squared MLP error')[['CERT','NAME','date','DEPDOM','next_deposits','growth','Share of squared error']],
    'Which validation outcomes dominate squared error?',{'growth':'{:+.1%}','Share of squared error':'{:.1%}'})
worst=validation_audit.nlargest(1,'Squared MLP error').iloc[0]
takeaway('One extreme outcome can dominate MSE',
    f'{worst.NAME} at {worst.date:%Y-%m-%d} contributes {worst["Share of squared error"]:.1%} of validation squared error. Its growth is {worst.growth:.1%}. We retain it and keep the frozen loss choice; this diagnostic explains the loss scale, not a new model selection.')
fig=go.Figure()
for col,label,color,dash in [('loss','Training','#007F89','solid'),('val_loss','Validation','#AF7721','dash')]:
    fig.add_trace(go.Scatter(x=history_frame.Epoch,y=history_frame[col]*10000,name=label,line=dict(color=color,dash=dash)))
fig.update_xaxes(title='Epoch');fig.update_yaxes(title='Mean squared error (pp²)',type='log')
chart(fig,'learning','Did learning improve held-out predictions?','Logarithmic loss axis; best validation weights restored')
takeaway('Training duration comes from validation',f'The MLP restored epoch {frozen["mlp_best_epoch"]} after {len(history_frame)} epochs. Training took {training_seconds:.1f} seconds on this run. The historical period did not choose the weights.')
''')
section('8 · Does the neural network improve the forecast?', '''
**Now open the frozen historical evaluation.** Every model predicts the same bank-quarters.
The score table gives exact errors; the dots let your eye compare them.
The scatter asks a different question: do individual predictions move with reality?

On the diagonal, prediction equals outcome. Above it, the model predicts too much growth.
Below it, the model predicts too little. Both axes use the same units and scale.
The full-range scatter retains every eligible evaluation row.
''',r'''
# EXEMPLAR: bounded-takeaway
test=rows.loc[holdout_mask].copy()
X_test=scaler.transform(imputer.transform(test[FEATURES])).astype('float32')
y_test=test.growth.to_numpy()
predictions={'Zero growth':np.zeros(len(test)),'Persistence':test.prior_growth.to_numpy(),
    'Ridge':ridge.predict(X_test),'MLP':predict(mlp,X_test)}
assert all(np.isfinite(p).all() and len(p)==len(test) for p in predictions.values())
metrics=pd.DataFrame([{'Model':n,**scores(y_test,p)} for n,p in predictions.items()])
table(metrics,'Reused 2024 holdout: every eligible row counts',{'MAE (pp)':'{:.4f}','RMSE (pp)':'{:.4f}'})
fig=make_subplots(rows=1,cols=2,subplot_titles=['Average absolute miss','Large-error-sensitive miss'],horizontal_spacing=.22)
for n,p in predictions.items():
    result=scores(y_test,p)
    for j,col in enumerate(['MAE (pp)','RMSE (pp)'],1):
        fig.add_trace(go.Scatter(x=[result[col]],y=[n],mode='markers',marker=dict(size=12,color=MODEL_COLORS[n]),name=n,showlegend=False,hovertemplate='%{y}<br>'+col+': %{x:.3f}<extra></extra>'),row=1,col=j)
        fig.update_xaxes(title='Percentage points',row=1,col=j)
chart(fig,'comparison','Does the MLP beat simpler forecasts?','Focused dot-plot scales; lower is better',height=560)
scatter=test[['CERT','NAME','date','growth']].copy();scatter['Actual (%)']=100*y_test;scatter['Predicted (%)']=100*predictions['MLP']
fig=px.scatter(scatter,x='Actual (%)',y='Predicted (%)',hover_data=['NAME','CERT','date'],opacity=.3,color_discrete_sequence=['#007F89'])
lo=min(scatter['Actual (%)'].min(),scatter['Predicted (%)'].min());hi=max(scatter['Actual (%)'].max(),scatter['Predicted (%)'].max())
span=max(hi-lo,1);bounds=[lo-.04*span,hi+.04*span]
fig.add_trace(go.Scatter(x=bounds,y=bounds,mode='lines',name='Ideal forecast',line=dict(color='#343B43',dash='dash')))
fig.update_xaxes(range=bounds);fig.update_yaxes(range=bounds,scaleanchor='x',scaleratio=1)
chart(fig,'actual_predicted','Do forecasts follow the outcomes?','All historical evaluation rows; equal axis units',height=700)
result_frame=test[['CERT','NAME','date','target_date','growth','DEPDOM','next_deposits']].reset_index(drop=True)
for n,p in predictions.items():result_frame[n]=p
result_frame.to_csv(OUT/'predictions.csv',index=False)
metrics.to_csv(OUT/'historical_scores.csv',index=False)
mae=metrics.set_index('Model')['MAE (pp)'];winner=mae.idxmin()
difference=float(mae['MLP']-mae['Ridge'])
takeaway(f'{winner} has the lowest historical MAE',
    f'MLP: {mae["MLP"]:.3f} pp. Ridge: {mae["Ridge"]:.3f} pp. Persistence: {mae["Persistence"]:.3f} pp. The MLP-minus-Ridge difference is {difference:+.3f} pp; negative favors the MLP. This is descriptive evidence from a reused holdout.',f'{mae[winner]:.3f} pp')
''')
section('9 · Does the average hide weak quarters or weak outcomes?', '''
**A small average miss can coexist with poor forecasts when deposits fall sharply.**
First compare quarters. Then look at the realized bottom quarter and bottom tenth of growth in each quarter.
These groups are defined after the outcome occurs. They diagnose errors; they do not prove advance warning.

Each bank-quarter gets equal weight in pooled MAE. The equal-quarter mean is also reported so a quarter
with more rows does not silently carry more influence. With only three evaluation quarters,
we cannot establish stability across every economic regime.
''',r'''
# EXEMPLAR: bounded-takeaway
quarter_rows=[];tail_rows=[]
for quarter,part in result_frame.groupby('date'):
    for n in predictions:
        quarter_rows.append({'Quarter':quarter,'Model':n,'Rows':len(part),**scores(part.growth,part[n])})
    # Include ties at the realized quantile boundary and show resulting sample sizes.
    for label,mask in [('All',pd.Series(True,index=part.index)),
        ('Realized bottom 25%',part.growth.le(part.growth.quantile(.25))),
        ('Realized bottom 10%',part.growth.le(part.growth.quantile(.10)))]:
        for n in predictions:
            for idx in part.index[mask]:tail_rows.append({'Index':idx,'Slice':label,'Model':n,
                'Absolute error':abs(part.loc[idx,'growth']-part.loc[idx,n]),
                'Squared error':(part.loc[idx,'growth']-part.loc[idx,n])**2})
quarter_scores=pd.DataFrame(quarter_rows)
table(quarter_scores,'Three quarters, separate error checks',{'MAE (pp)':'{:.3f}','RMSE (pp)':'{:.3f}'})
fig=px.line(quarter_scores,x='Quarter',y='MAE (pp)',color='Model',symbol='Model',markers=True,color_discrete_map=MODEL_COLORS)
fig.update_xaxes(tickvals=sorted(result_frame.date.unique()),tickformat='%b %Y')
chart(fig,'quarter_errors','Is one quarter driving the pooled error?','Predictor quarters; each outcome is one quarter later')
equal_quarter=quarter_scores.groupby('Model',sort=False)['MAE (pp)'].mean().reset_index(name='Equal-quarter MAE (pp)')
table(equal_quarter,'Give each evaluation quarter equal weight',{'Equal-quarter MAE (pp)':'{:.3f}'})
tail_errors=pd.DataFrame(tail_rows).groupby(['Slice','Model'],sort=False).agg(
    Rows=('Index','size'),MAE=('Absolute error','mean'),MSE=('Squared error','mean')).reset_index()
tail_errors['MAE (pp)']=100*tail_errors.MAE;tail_errors['RMSE (pp)']=100*np.sqrt(tail_errors.MSE)
table(tail_errors[['Slice','Model','Rows','MAE (pp)','RMSE (pp)']],'How wrong are forecasts when growth is weak?',{'MAE (pp)':'{:.3f}','RMSE (pp)':'{:.3f}'})
fig=px.scatter(tail_errors,x='MAE (pp)',y='Slice',color='Model',symbol='Model',color_discrete_map=MODEL_COLORS,hover_data=['Rows'])
chart(fig,'tail_errors','Does weak deposit growth expose larger mistakes?','Slices use realized within-quarter growth; this is error analysis')
quarter_scores.to_csv(OUT/'quarter_scores.csv',index=False);tail_errors.to_csv(OUT/'tail_errors.csv',index=False)
weak=tail_errors.loc[(tail_errors.Slice=='Realized bottom 10%')&(tail_errors.Model=='MLP')].iloc[0]
takeaway('Check the weak-outcome error before using the forecast',f'The MLP misses by {weak["MAE (pp)"]:.3f} pp on average across {int(weak.Rows):,} bottom-decile bank-quarters, compared with {mae["MLP"]:.3f} pp overall. These groups were identified using realized outcomes.')
''')
section('10 · Could we identify weak growth beforehand?', '''
**Rank using predictions first; compare with outcomes second.** For each quarter, select the lowest
predicted-growth 10% of banks. Then count how many actually belong to the lowest-growth 10%.

Precision asks: of the banks selected, how many were in the realized bottom decile?
Recall asks: of the realized bottom decile, how many were selected?
Here both sets have the same size, so precision and recall are equal by construction.
Ties are broken by certificate for a reproducible, fixed-capacity comparison.
The zero-growth rule has no economic ranking signal; its certificate ordering is arbitrary.

Spearman correlation compares the order of all forecasts with the order of outcomes.
A constant forecast has no defined rank correlation. Leave that value missing.
''',r'''
# EXEMPLAR: bounded-takeaway
ranking=[]
for quarter,part in result_frame.groupby('date'):
    k=int(np.ceil(.1*len(part)))
    actual=set(part.sort_values(['growth','CERT']).head(k).CERT)
    for n in predictions:
        selected=set(part.sort_values([n,'CERT']).head(k).CERT)
        hits=len(actual&selected)
        rho=part.growth.corr(part[n],method='spearman') if part[n].nunique()>1 else np.nan
        ranking.append({'Quarter':quarter,'Model':n,'Banks':len(part),'Selected':k,'Hits':hits,
            'Precision':hits/k,'Recall':hits/len(actual),'Spearman':rho,'Random expectation':k/len(part)})
ranking=pd.DataFrame(ranking)
ranking_display=ranking[['Quarter','Model','Selected','Hits','Precision','Spearman']].copy()
ranking_display['Quarter']=pd.to_datetime(ranking_display.Quarter).dt.to_period('Q').astype(str)
table(ranking_display,'Predicted selection compared with realized weak growth',{'Precision':'{:.1%}','Spearman':'{:.3f}'})
fig=px.line(ranking,x='Quarter',y='Precision',color='Model',symbol='Model',markers=True,color_discrete_map=MODEL_COLORS)
fig.update_yaxes(tickformat='.0%',range=[0,1]);fig.update_xaxes(tickvals=sorted(result_frame.date.unique()),tickformat='%b %Y')
fig.add_hline(y=.1,line_dash='dash',line_color='#343B43')
chart(fig,'ranking','How many selected banks actually had weak growth?','Fixed 10% quarterly capacity; dashed line approximates random selection')
ranking.to_csv(OUT/'ranking.csv',index=False)
precision=ranking.loc[ranking.Model.eq('MLP'),'Precision'].mean()
takeaway('Forecast error and identification answer different questions',f'The MLP selection has {precision:.1%} precision on an equal-quarter average. A useful average forecast does not automatically produce a useful review ranking.')
''')
section('11 · What did we learn, and what would we do next?', '''
**The answer is no: this small neural network did not add useful nonlinear forecasting value on the reused
2024 holdout.** Zero growth had the lowest MAE at **3.60 percentage points**. Ridge had the lowest RMSE at
**6.62 percentage points**. The MLP reached **4.14 MAE** and **6.72 RMSE**, so it did not beat both the
simple historical rules and the linear model.

The weak-outcome results sharpen that answer. The MLP's MAE rose from **4.14 points overall** to **8.78
points in the realized bottom decile**. When the model selected the 10% of banks it expected to have the
weakest growth, only **14.9%** were actually in that quarter's bottom decile on average. Ridge produced the
stronger ranking. Average error therefore hid exactly the weakness a bank analyst would care about most.

That is still a useful result. Current balance-sheet ratios contain some signal—the Ridge model slightly
reduced RMSE—but the extra bends available to the MLP did not pay for their complexity. A practical analyst
should keep the simple benchmark, audit large misses for mergers and institutional changes, and collect a
later untouched period before treating any ranking as operational evidence.

**Why can zero growth win MAE while Ridge wins RMSE?** They reward different behavior.
An error of 10 percentage points contributes 10 to absolute error and 100 to squared error.
Large changes therefore pull an MSE-trained model harder. The prediction that minimizes expected
absolute error is a conditional median; squared error targets a conditional mean.
A network trained with MSE is not directly optimizing MAE. That explains a possible tradeoff,
not a guarantee that the network learned a useful conditional mean. Compare both actual scores.

The validation audit also exposes a same-certificate name change from PLUS INTERNATIONAL BANK
to EMIGRANT BANK with a huge balance jump. Institutional restructuring can dominate quarterly
balance changes. That is a limitation of this outcome definition that the next experiment must resolve.
We preserve it here instead of quietly changing the question after seeing the errors.
''',r'''
# EXEMPLAR: bounded-takeaway
conclusion=('The MLP improves historical MAE over both persistence and Ridge.' if mae['MLP']<min(mae['Persistence'],mae['Ridge'])
            else 'The MLP does not beat both persistence and Ridge on historical MAE.')
takeaway('Does this experiment establish nonlinear value?',conclusion+
    ' Compare RMSE, weak-outcome errors, and quarter-level results before deciding whether any improvement is useful. A reused three-quarter holdout cannot establish deployment reliability.')
takeaway('For a bank analyst: use the result to guide research',
    'Inspect the source reports and institution changes behind large forecast misses. These quarterly balances do not establish withdrawals, bank runs, or the benefit of an intervention.')
takeaway('For the next experiment: earn genuinely new evidence',
    'Resolve reporting-vintage and release-date availability, verify the historical form crosswalk, and evaluate a later period that has not guided development. Keep the strongest simple baseline.')
assert source_hash==hashlib.sha256((ROOT/'data/fdic_financials_2020_2024.csv').read_bytes()).hexdigest()
assert raw_fingerprint==pd.util.hash_pandas_object(raw,index=True).sum()
assert np.allclose(rows.growth,(rows.next_deposits-rows.DEPDOM)/rows.DEPDOM)
import importlib.metadata
summary={'source_sha256':source_hash,'target':'signed percentage growth','history_gate':gate,
    'rows':{'train':len(train),'validation':len(valid),'holdout':len(test)},'winner_by_historical_MAE':winner,
    'mlp_minus_ridge_MAE_pp':difference,'conclusion':conclusion,'training_seconds':training_seconds,
    'frozen_plan':frozen,'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','tensorflow','scikit-learn','plotly','wm-notecards']},
    'python':platform.python_version()}
(OUT/'run_summary.json').write_text(json.dumps(summary,indent=2))
print('Audit passed: source unchanged, target arithmetic reconciled, predictions finite, results saved.')
''')
section('Deeper lesson · Does a different random start change the answer?', '''
**The seed changes the starting weights, not the banks.** Fit the same architecture with seeds 7 and 99,
in addition to the primary seed 42. Compare their validation errors only. Keep seed 42 as the primary
historical model even if another start looks better.

This is a sensitivity check on optimization. It does not supply independent economic evidence.
''',r'''
# EXEMPLAR: bounded-takeaway
seed_results=[{'Seed':42,**scores(y_valid,validation_predictions['MLP']),'Best epoch':frozen['mlp_best_epoch']}]
for seed in [7,99]:
    other,h,seconds=fit_network(seed)
    seed_results.append({'Seed':seed,**scores(y_valid,predict(other,X_valid)),'Best epoch':int(np.argmin(h['val_loss'])+1)})
seed_results=pd.DataFrame(seed_results)
table(seed_results,'Same data, different starting weights',{'MAE (pp)':'{:.3f}','RMSE (pp)':'{:.3f}'})
fig=px.scatter(seed_results,x='MAE (pp)',y=seed_results.Seed.astype(str),color_discrete_sequence=['#007F89'])
fig.update_yaxes(title='Random seed')
chart(fig,'seeds','How sensitive is validation error to initialization?','Primary historical model remains seed 42')
seed_results.to_csv(OUT/'seed_sensitivity.csv',index=False)
takeaway('A seed is not a new sample',f'Validation MAE ranges from {seed_results["MAE (pp)"].min():.3f} to {seed_results["MAE (pp)"].max():.3f} pp across the three fixed starts. No seed is chosen using historical performance.')
''',advanced=True)
section('Deeper lesson · How uncertain is the MLP–Ridge difference?', '''
**Resample whole banks so their repeated reports stay together.** Sampling individual rows would pretend
that several observations of one bank are independent.

For each of 500 resamples, draw certificates with replacement and calculate the paired difference in MAE.
A negative difference favors the MLP. The 95% percentile interval describes resampling uncertainty
conditional on these fitted models and these three quarters. It does not cover new economic regimes
or all uncertainty from retraining the models.
''',r'''
# EXEMPLAR: bounded-takeaway
errors=result_frame.assign(delta=np.abs(result_frame.growth-result_frame.MLP)-np.abs(result_frame.growth-result_frame.Ridge))
clusters=errors.groupby('CERT').delta.agg(['sum','count'])
rng=np.random.default_rng(42);deltas=[]
for _ in range(500):
    sampled=rng.integers(0,len(clusters),len(clusters))
    sample=clusters.iloc[sampled]
    deltas.append(100*sample['sum'].sum()/sample['count'].sum())
low,high=np.quantile(deltas,[.025,.975])
interval=pd.DataFrame({'Comparison':['MLP minus Ridge'],'Observed MAE difference (pp)':[difference],
    '95% lower (pp)':[low],'95% upper (pp)':[high],'Bank clusters':[len(clusters)]})
table(interval,'Paired bank-cluster bootstrap',{'Observed MAE difference (pp)':'{:+.4f}','95% lower (pp)':'{:+.4f}','95% upper (pp)':'{:+.4f}'})
fig=px.histogram(x=deltas,nbins=35,color_discrete_sequence=['#007F89']);fig.add_vline(x=0,line_dash='dash',line_color='#343B43')
fig.update_xaxes(title='MLP minus Ridge MAE (pp)');fig.update_yaxes(title='Bootstrap resamples')
chart(fig,'bootstrap','Does the comparison survive resampling banks?','500 paired cluster resamples; negative favors MLP')
interval.to_csv(OUT/'bootstrap_interval.csv',index=False)
takeaway('Uncertainty belongs beside the difference',f'The observed difference is {difference:+.4f} pp; the conditional 95% interval is [{low:+.4f}, {high:+.4f}] pp. '+('The interval crosses zero.' if low<=0<=high else 'The interval stays on one side of zero.')+' Three quarters still limit temporal generalization.')
''',advanced=True)
section('Deeper lesson · Why not add uninsured deposits immediately?', '''
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
# EXEMPLAR: missingness-evidence
coverage_source=long_audit.copy()
coverage_source['Asset group']=np.where(coverage_source.ASSET.ge(1_000_000),'At least USD 1bn','Below USD 1bn')
uninsured=[]
for field in ['DEPUNA','DEPUNINS']:
    for (quarter,size,form),part in coverage_source.groupby(['REPDTE','Asset group','CALLFORM'],dropna=False):
        uninsured.append({'Quarter':quarter,'Asset group':size,'Call form':str(form),'Field':field,
            'Rows':len(part),'API populated':part[field].notna().sum(),'Zero values':part[field].eq(0).sum()})
uninsured=pd.DataFrame(uninsured);uninsured['Coverage']=uninsured['API populated']/uninsured.Rows
uninsured['Zero share']=uninsured['Zero values']/uninsured.Rows
uninsured.to_csv(OUT/'uninsured_coverage.csv',index=False)
aggregate=uninsured.groupby(['Quarter','Asset group','Field'],as_index=False)[['Rows','API populated','Zero values']].sum()
aggregate['Coverage']=aggregate['API populated']/aggregate.Rows
aggregate['Date']=pd.to_datetime(aggregate.Quarter.astype(str))
aggregate['Zero share']=aggregate['Zero values']/aggregate.Rows
table(uninsured.loc[uninsured.Quarter.eq(uninsured.Quarter.max())],'Latest quarter: coverage by size and reporting form',{'Coverage':'{:.1%}'})
fig=px.line(aggregate,x='Date',y='Coverage',color='Asset group',line_dash='Field',
    color_discrete_map={'At least USD 1bn':'#007F89','Below USD 1bn':'#AF7721'})
fig.update_yaxes(tickformat='.0%',range=[0,1.05])
chart(fig,'uninsured','Which API fields contain a value?','2010–2024; fields inspected separately; current assets define descriptive groups')
fig=px.line(aggregate,x='Date',y='Zero share',color='Asset group',line_dash='Field',
    color_discrete_map={'At least USD 1bn':'#007F89','Below USD 1bn':'#AF7721'})
fig.update_yaxes(tickformat='.0%',range=[0,1.05])
chart(fig,'uninsured_zeros','How often does a populated field contain zero?',
    'Zeros are ambiguous; field mappings and reporting eligibility need verification')
takeaway('Keep uninsured deposits out of the core model',
    'API population reaches 100% even where reporting is optional. Zeros and field mapping prevent us from equating population with regulatory reporting. Keep these fields outside the five-feature comparison.')
''',advanced=True)
section('Appendix · Keep the useful questions from Experiment 0', '''
The [original masterclass](experiments/experiment_0/FDIC_Deep_Learning_Masterclass.ipynb) preserves the complete
executed dollar-runoff experiment and its exact outputs. `Assign1.ipynb` is unchanged.

- **Two stages:** estimate whether deposits decline, then estimate severity conditional on decline.
  Multiplying a probability by an estimated median severity does not automatically produce an expected dollar loss.
- **Precision–recall:** how many selected cases have a decline, and how many decline cases are found?
- **Calibration:** among cases assigned a given decline probability, how often does decline occur?
- **Review capacity:** selecting 10% each quarter is an explicit workload assumption; captured dollars are not money saved.
- **Loss choice:** MSE makes extreme growth influential. A future robust-loss comparison belongs in a newly frozen experiment,
  not an unreported repair after seeing this holdout.

**Try explaining these aloud:** Why can a high dollar-capture score come from size alone?
Why is a missing next report different from zero growth? Why does scaling use training rows?
Why can a model have low average error but poor bottom-decile precision?
What does the bootstrap leave uncertain?

Visual choices follow the questions: histograms for target shape, a timeline for leakage,
dots for close model errors, lines for quarters and optimization, and a scatter for individual forecasts.
Maps and pies add no evidence to this question. Classification graphics stay with the classification experiment.
The lower-triangle correlation heatmap is descriptive: it checks whether the five fixed inputs repeat one another;
it does not select or remove features.
''',advanced=True)


def build():
    for edition,name in [('masterclass','FDIC_Deep_Learning_Masterclass.ipynb'),('submission','FDIC_Deep_Learning_Submission.ipynb')]:
        cells=[]
        for index,(title,prose,source,advanced) in enumerate(sections):
            if advanced and edition=='submission':continue
            # Keep the same analysis code in both deliverables; reduce teaching prose only.
            if edition=='submission' and index not in [0]:
                prose=prose.split('\n\n')[0]
            cells.append(nbf.v4.new_markdown_cell(('# ' if index==0 else '## ')+title+'\n\n'+prose))
            if source:
                if index==0:source=f"NOTEBOOK_EDITION = {edition!r}\n"+source
                cells.append(nbf.v4.new_code_cell(source))
        for i,c in enumerate(cells):c.id=hashlib.sha256(f'{edition}:{i}:{c.source}'.encode()).hexdigest()[:12]
        nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'DL Assignment (.venv)','language':'python','name':'python3'},'language_info':{'name':'python'}})
        nbf.validate(nb);nbf.write(nb,ROOT/name)
        print(name,len(cells),'cells')
if __name__=='__main__':build()
