"""Export the six-slide story from verified notebook results."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
OUT = ROOT / 'growth_outputs/masterclass'
scores = pd.read_csv(OUT / 'extended_scores.csv').set_index('Model')
ranking = pd.read_csv(OUT / 'ranking.csv')
march = ranking[ranking.Quarter.eq('2024-03-31')].set_index('Model')
means = pd.read_csv(OUT / 'mean_ranking.csv').set_index('Model')
season = pd.read_csv(OUT / 'seasonal_rules.csv')
hist = json.loads((DEST / 'training_histogram.json').read_text())
CYAN, TEAL, INK, GRAY = '#17BDE0', '#007C89', '#20272B', '#69757B'
plt.rcParams.update({'font.family':'DejaVu Sans', 'text.color':INK, 'text.parse_math':False})
pages = []
def text(f,x,y,s,size=20,color=INK,weight='normal',ha='left'):
    f.text(x,y,s,size=size,color=color,weight=weight,ha=ha,va='top',linespacing=1.15)
def page(title,note):
    f=plt.figure(figsize=(14,8),facecolor='white')
    pages.append(f)
    text(f,.055,.95,'SIMPLE MODELS FIGHT BACK',11,TEAL,'bold')
    text(f,.055,.885,title,34 if len(pages)==1 else 30,INK,'bold')
    text(f,.055,.105,note,11)
    text(f,.945,.035,f'{len(pages):02} / 06  •  Reused 2024 evaluation',9,GRAY,ha='right')
    return f
def score(model,metric): return scores.loc[model,metric]

predictions = pd.read_csv(OUT / 'predictions.csv')
actual = 100 * predictions['growth'].to_numpy()
mlp = 100 * predictions['MLP'].to_numpy()
for model in ['Zero growth','MLP']:
    error=actual-100*predictions[model].to_numpy()
    assert np.isclose(np.mean(abs(error)),score(model,'MAE (pp)'),atol=1e-6)
    assert np.isclose(np.sqrt(np.mean(error**2)),score(model,'RMSE (pp)'),atol=1e-6)
# One shared display window and the identical population in both panels.
lo,hi=np.quantile(actual,[.01,.99]);visible=(actual>=lo)&(actual<=hi)&(mlp>=lo)&(mlp<=hi)
shown=int(visible.sum()); omitted=len(actual)-shown
f=page('NEURAL NETS CAN FIT COMPLEX CURVES.\nMINE LOST TO A FLAT LINE.',
       f'Actual predictions. Both axes: {lo:.1f}% to {hi:.1f}% (actual-growth central 98%); {omitted:,} rows outside either axis omitted in both panels.\nAll {len(actual):,} rows scored. 214,425 training examples. Original four-model comparison; historical results.')
text(f,.055,.695,'Predicting next-quarter deposit growth from quarterly bank reports.',16)
for xpos,name,values in [(.12,'ZERO MODEL',np.zeros(len(actual))),(.58,'NEURAL NET',mlp)]:
    ax=f.add_axes([xpos,.245,.32,.34]);ax.scatter(actual[visible],values[visible],s=5,alpha=.18,color=TEAL,rasterized=True)
    ax.plot([lo,hi],[lo,hi],ls='--',color=GRAY,lw=1.2,label='Perfect prediction: y = x')
    ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Actual growth (%)',ylabel='Predicted growth (%)')
    ax.set_aspect('equal',adjustable='box');ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=10)
    ax.set_title(name,fontsize=15,fontweight='bold');ax.legend(fontsize=8,loc='upper left')
    text(f,xpos+.16,.19,f"{'Zero' if name=='ZERO MODEL' else 'MLP'} MAE {score('Zero growth' if name=='ZERO MODEL' else 'MLP','MAE (pp)'):.3f} pp",18,TEAL,'bold',ha='center')

f=page('WHY WAS ZERO SO HARD TO BEAT?',
       'Central 98% of training log-growth targets. Full tails stayed in the experiment.\nThis concentration motivates the benchmark; it does not by itself explain the later score.')
ax=f.add_axes([.12,.25,.78,.43]);edges=np.array(hist['edges'])
ax.bar(edges[:-1],hist['counts'],width=np.diff(edges),align='edge',color='#C8CED1',edgecolor='white');ax.axvline(0,color=CYAN,lw=4)
ax.annotate('A lot of the target mass lives near here.',xy=(0,34000),xytext=(2,40000),arrowprops={'arrowstyle':'->','color':TEAL},color=TEAL,fontsize=14,fontweight='bold')
ax.set_ylim(0,47000);ax.set_xlabel('100 × log(next deposits / current deposits)');ax.set_ylabel('Training examples');ax.spines[['top','right']].set_visible(False)

f=page('ZERO WON MAE.\nTHE NET WON RMSE.',
       'Original four models; same evaluation rows. Errors are in ordinary-growth percentage points.\nLower is better. Small historical differences do not establish future superiority.')
for j,metric in enumerate(['MAE (pp)','RMSE (pp)']):
    d=scores.loc[['Zero growth','MLP','Ridge','Persistence']].sort_values(metric)
    ax=f.add_axes([.18+j*.46,.31,.29,.34]);ax.scatter(d[metric],range(4),color=[TEAL,GRAY,GRAY,GRAY],s=80)
    ax.set_yticks(range(4),['Zero' if x=='Zero growth' else x for x in d.index]);ax.invert_yaxis();ax.set_xlim(0,6 if j==0 else 11)
    ax.set_title(metric.replace(' (pp)','')+' ↓',fontsize=19,fontweight='bold');ax.set_xlabel('Percentage points');ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.12)
    for i,v in enumerate(d[metric]):ax.text(v+.12,i,f'{v:.3f}',va='center',fontsize=12,fontweight='bold' if i==0 else 'normal')
text(f,.5,.205,'Different scoring metric. Different winner.',25,INK,'bold',ha='center')

f=page('ZERO CAN FORECAST.\nIT CAN’T RANK.',
       'March → June 2024. 4,536 eligible banks; 454 hypothetical review slots.\nFound = later in that quarter’s lowest-growth 10%, which can include positive growth.')
text(f,.055,.70,'Every bank → 0% predicted growth → complete tie',18)
n=int(march.loc['Ridge','Banks']);k=int(march.loc['Ridge','Selected'])
prevalence=float(march.loc['Ridge','Random expectation'])
values=[100*prevalence,100*march.loc['Ridge','Precision'],100*march.loc['MLP','Precision']]
ax=f.add_axes([.23,.265,.60,.34]);ax.barh(range(3),values,color=[GRAY,CYAN,TEAL],height=.48)
ax.set_yticks(range(3),['Random selection','Ridge','MLP']);ax.invert_yaxis();ax.set_xlim(0,35);ax.set_xticks([0,10,20,30],['0%','10%','20%','30%']);ax.set_xlabel('Precision @ 10% review capacity');ax.spines[['top','right']].set_visible(False)
for i,(v,count) in enumerate(zip(values,[f'{k*prevalence:.1f} expected','112 found','109 found'])):ax.text(v+.7,i,f'{v:.1f}%  |  {count}',va='center',size=13,weight='bold')

f=page('I THREW OUT 4 INPUTS.\nKEPT ONLY DEPOSIT SIZE.',
       'Equal-quarter average across three reused 2024 quarters. Size-only challenger added after examining 2024.\nThis comparison does not decompose the five-feature model’s causes.')
a_val=100*means.loc['Size-only Ridge','Precision'];b_val=100*means.loc['Ridge','Precision']
ax=f.add_axes([.12,.34,.77,.26]);ax.plot([a_val,b_val],[0,0],color=GRAY,lw=3);ax.scatter([a_val,b_val],[0,0],color=[GRAY,CYAN],s=160,zorder=3)
ax.axvline(10,ls='--',color=GRAY);ax.text(10,.53,'Chance ≈ 10%',ha='center',size=12)
ax.text(a_val-.35,.2,f'SIZE-ONLY RIDGE\n{a_val:.1f}%',ha='right',size=16,weight='bold');ax.text(b_val+.35,.2,f'5-FEATURE RIDGE\n{b_val:.1f}%',ha='left',size=16,weight='bold')
ax.set(xlim=(0,30),ylim=(-.4,.8),yticks=[],xlabel='Precision @ 10% review capacity');ax.set_xticks([0,10,20,30],['0%','10%','20%','30%']);ax.spines[['top','right','left']].set_visible(False)
text(f,.5,.225,f'Same model family. Four features removed. {b_val-a_val:.1f} percentage-point difference.',19,INK,'bold',ha='center')

f=page('A CALENDAR RULE BEAT ZERO\nON HISTORICAL MAE.',
       'Seasonal median: predict each calendar quarter’s training-period median growth. Developed after examining 2024.\nHistorical comparison, not fresh confirmation. Analysis notebook built with wm-notecards.')
text(f,.07,.69,'How often did deposits fall next quarter?',16)
ax=f.add_axes([.10,.32,.43,.30]);values=season['Decline share'].to_numpy()*100
ax.bar(season.Quarter,values,color=[CYAN,CYAN,GRAY,GRAY],width=.6);ax.set_ylim(0,60);ax.set_yticks([0,20,40,60],['0%','20%','40%','60%']);ax.set_xticks([1,2,3,4],['Q1','Q2','Q3','Q4']);ax.set_xlabel('Predictor quarter · training years');ax.spines[['top','right']].set_visible(False)
for q,v in zip(season.Quarter,values):ax.text(q,v+2,f'{v:.1f}%',ha='center',size=13,weight='bold')
ax=f.add_axes([.70,.36,.21,.22]);v=[score('Seasonal median','MAE (pp)'),score('Zero growth','MAE (pp)')]
ax.scatter(v,[0,1],s=90,color=[CYAN,GRAY]);ax.set_yticks([0,1],['Seasonal median','Zero']);ax.set_ylim(1.6,-.6);ax.set_xlim(0,4.3);ax.set_title('MAE ↓',size=18,weight='bold');ax.set_xlabel('Percentage points');ax.spines[['top','right']].set_visible(False)
for i,value in enumerate(v):ax.text(value+.1,i,f'{value:.3f}',va='center',size=13,weight='bold')
text(f,.5,.205,'OKAY. THAT GETS A FRESH TEST.',27,INK,'bold',ha='center')

assert len(pages)==6
assert (n,k)==(4536,454)
assert round(k*march.loc['Ridge','Precision']) == 112
assert round(k*march.loc['MLP','Precision']) == 109
assert sum(hist['counts'])==210135
assert round(b_val-a_val,1)==3.1
with PdfPages(DEST/'simple_models_fight_back_reviewed.pdf') as pdf:
    for i,f in enumerate(pages,1):
        pdf.savefig(f);f.savefig(DEST/f'{i:02}.png',dpi=150);plt.close(f)
sources=[OUT/name for name in ['extended_scores.csv','mean_ranking.csv','seasonal_rules.csv','ranking.csv','predictions.csv']]+[DEST/'training_histogram.json']
(DEST/'manifest.json').write_text(json.dumps({'pages':6,'historical_evaluation':'2024 Q1–Q3 predictors; reused','source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2))
print(DEST/'simple_models_fight_back_reviewed.pdf')
