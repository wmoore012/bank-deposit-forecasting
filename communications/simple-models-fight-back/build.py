"""Build the verified carousel from repository results. Run: .venv/bin/python communications/simple-models-fight-back/build.py"""
from pathlib import Path
import json, hashlib, textwrap, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[2]
DEST=Path(__file__).resolve().parent
OUT=ROOT/'growth_outputs/masterclass'
C='#17BDE0'; T='#0B6F75'; B='#3F6294'; A='#B76B1E'; G='#73777B'; INK='#2C3034'
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':INK,'font.size':14,'text.parse_math':False})
scores=pd.read_csv(OUT/'extended_scores.csv').set_index('Model')
lift=pd.read_csv(OUT/'mean_ranking.csv').set_index('Model')
season=pd.read_csv(OUT/'seasonal_rules.csv')
ranking=pd.read_csv(OUT/'ranking.csv'); march=ranking[ranking.Quarter.eq('2024-03-31')].set_index('Model')
foundation={k:json.loads((ROOT/f'growth_outputs/{k}_predictions.json').read_text()) for k in ['chronos_zero_shot','chronos_finetuned','timesfm_zero_shot','timesfm_finetuned']}
# Freeze the display histogram as a documented export input; all model rows stay unchanged.
hist=json.loads((DEST/'training_histogram.json').read_text())
pages=[]
def page(n,title,sub,note):
 f=plt.figure(figsize=(14,8),facecolor='white')
 f.text(.055,.94,f'SIMPLE MODELS FIGHT BACK  /  {n:02}',color=T,size=11,weight='bold')
 f.text(.055,.875,textwrap.fill(title,62),size=27,weight='bold',va='top')
 f.text(.055,.735,textwrap.fill(sub,115),size=13,va='top')
 f.text(.055,.12,textwrap.fill(note,135),size=11,va='top',linespacing=1.4)
 f.text(.945,.035,f'{n:02} / 08 • Reused 2024 evaluation',ha='right',size=9,color=G)
 pages.append(f);return f
def axis(f,rect):
 a=f.add_axes(rect);a.spines[['top','right']].set_visible(False);a.grid(axis='x',alpha=.15);a.set_axisbelow(True);return a
f=page(1,'I trained a neural network on 214,425 bank-quarter examples.','Among the original four forecasts, predicting zero growth had the lowest mean absolute error.','MAE measures the average absolute miss in percentage points. All scores use the same 13,532 bank-quarter examples in three already-examined 2024 quarters. Follow-up baselines appear later.')
f.text(.5,.52,'PREDICT 0%',ha='center',size=65,weight='bold',color=T)
f.text(.5,.40,'for every bank.',ha='center',size=27)
f.text(.5,.25,'Zero: 3.601 pp MAE   |   MLP: 3.627   |   Ridge: 3.641',ha='center',size=20)
f=page(2,'Many training targets sit near zero.','That makes a constant forecast a useful benchmark. Its later performance still has to be measured.','This histogram shows the central 98% of training log-growth targets. The full tails remain in the experiment. The horizontal unit is 100 × log growth; it is not ordinary percentage-point growth.')
a=axis(f,[.12,.27,.79,.36]);edges=np.array(hist['edges']);a.bar(edges[:-1],hist['counts'],width=np.diff(edges),align='edge',color='#BEC3C5',edgecolor='white');a.axvline(0,color=T,lw=3,label='Zero forecast');a.set_xlabel('100 × log(D next / D today)');a.set_ylabel('Training examples');a.legend()
f=page(3,'The error measure changes the winner.','Original four models; lower is better. Zero leads MAE. The neural network leads RMSE.','MAE differences among zero, Ridge and MLP are only about 0.04 pp. RMSE gives large misses more weight. These rankings describe the historical sample; they do not establish future superiority.')
for j,metric in enumerate(['MAE (pp)','RMSE (pp)']):
 d=scores.loc[['Zero growth','MLP','Ridge','Persistence']].sort_values(metric);a=axis(f,[.18+j*.47,.28,.28,.34]);a.scatter(d[metric],range(4),c=[T,G,G,G],s=70);a.set_yticks(range(4),d.index);a.invert_yaxis();a.set_xlim(0,6 if j==0 else 11);a.set_xlabel(metric)
 for i,v in enumerate(d[metric]):a.text(v+.1,i,f'{v:.3f}',va='center',size=10)
f=page(4,'Zero growth gives every bank the same forecast.','Imagine ten banks and enough time to examine only one. The zero-growth forecast leaves all ten tied.','Zero growth remains a valid point-forecast baseline. Selecting among tied forecasts requires a separate rule. Random selection supplies the comparison for the review-list experiment.')
a=axis(f,[.16,.25,.34,.41]);a.scatter(np.zeros(10),range(10),color=T,s=55);a.set_yticks(range(10),[f'Bank {i+1:02}' for i in range(10)]);a.set_xticks([0],['0%']);a.set_xlim(-1,1);a.invert_yaxis();f.text(.59,.49,'ONE REVIEW SLOT.\nTEN IDENTICAL SCORES.',size=27,weight='bold',linespacing=1.6)
f=page(5,'Same 454 review slots. More intended cases found.','March 2024 reports → June outcomes. There were 4,536 eligible banks; 454 review slots were assumed.','A hit is a selected bank in the next quarter’s lowest-growth 10%, which can include positive growth. Grids show rounded percentages, not individual banks. Selection does not diagnose distress; no operational savings were measured.')
n=int(march.loc['Ridge','Banks']);k=int(march.loc['Ridge','Selected'])
for j,(name,h,col) in enumerate([('Random',k*k/n,G),('Ridge',int(march.loc['Ridge','Hits']),C),('Neural network',int(march.loc['MLP','Hits']),T)]):
 a=f.add_axes([.09+j*.31,.27,.24,.35]);count=round(100*h/k);a.scatter(np.tile(np.arange(10),10),np.repeat(np.arange(10)[::-1],10),c=[col]*count+['#DEDEDE']*(100-count),s=42);a.axis('off');a.set_title(name,size=18);a.text(.5,-.15,f'{h:.1f} expected' if j==0 else f'{h} hits',transform=a.transAxes,ha='center',size=20,weight='bold')
f=page(6,'Size alone already carries substantial ranking signal.','Lift compares each list’s hit rate with its quarter’s exact random-selection expectation.','Equal-quarter means over three reused quarters. Size-only Ridge is a post-hoc comparison. These results do not measure a causal contribution from size. The paired MLP–Ridge precision interval includes either ordering.')
a=axis(f,[.25,.26,.59,.38]);models=['Persistence','Size-only Ridge','Ridge','MLP'];v=lift.loc[models,'Lift'];a.scatter(v,range(4),color=[G,B,C,T],s=90);a.set_yticks(range(4),['Previous quarter','Size-only Ridge','Five-feature Ridge','Neural network']);a.invert_yaxis();a.axvline(1,color=G,ls='--');a.set_xlim(.9,2.5);a.set_xlabel('Lift versus random selection (1×)')
for i,x in enumerate(v):a.text(x+.04,i,f'{x:.2f}×',va='center',weight='bold')
f=page(7,'The calendar earns a place in the next frozen test.','Training-period decline rates differ by predictor quarter. A seasonal-median forecast uses only training data.','Seasonal median: 3.500 pp MAE; zero growth: 3.601 pp on reused 2024 data. This baseline was added after 2024 had been examined. Its fitting uses earlier rows; its development was post-hoc.')
a=axis(f,[.15,.28,.7,.35]);a.bar(season.Quarter,season['Decline share']*100,color=[C,C,G,G]);a.set_xticks([1,2,3,4],['Q1','Q2','Q3','Q4']);a.set_ylim(0,60);a.set_ylabel('Next-quarter declines (%)')
for q,v in zip(season.Quarter,season['Decline share']*100):a.text(q,v+2,f'{v:.1f}%',ha='center',weight='bold')
f=page(8,'These pretrained models did not beat the simple benchmarks.','Same evaluation rows. The small neural network had lower RMSE than either foundation model in this experiment.','As-is means frozen pretrained weights. Adaptation updated only the forecast head for 256 sampled steps, with earlier validation selecting the checkpoint. Pretraining overlap is unknown. This limited test does not establish what other checkpoints or training budgets could achieve.')
labels=['MLP','Zero growth','Chronos as-is','Chronos head adapted','TimesFM as-is','TimesFM head adapted'];values=[scores.loc['MLP','RMSE (pp)'],scores.loc['Zero growth','RMSE (pp)']]+[foundation[k]['RMSE (pp)'] for k in ['chronos_zero_shot','chronos_finetuned','timesfm_zero_shot','timesfm_finetuned']]
a=axis(f,[.28,.26,.55,.40]);a.scatter(values,range(6),color=[T,G,B,B,A,A],s=65);a.set_xscale('log');a.set_xticks([5,10,100,300],['5','10','100','300']);a.minorticks_off();a.set_xlim(5,420);a.set_yticks(range(6),labels);a.invert_yaxis();a.set_xlabel('RMSE (pp), logarithmic axis — lower is better')
for i,v in enumerate(values):a.text(v*1.07,i,f'{v:.3f}',va='center',size=11)
with PdfPages(DEST/'simple_models_fight_back_reviewed.pdf') as pdf:
 for i,f in enumerate(pages,1):pdf.savefig(f);f.savefig(DEST/f'{i:02}.png',dpi=120);plt.close(f)
sources=[OUT/n for n in ['extended_scores.csv','mean_ranking.csv','seasonal_rules.csv','ranking.csv']]+[ROOT/f'growth_outputs/{k}_predictions.json' for k in foundation]+[DEST/'training_histogram.json']
(DEST/'manifest.json').write_text(json.dumps({'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'pages':8,'historical_evaluation':'2024 Q1–Q3 predictors; reused'},indent=2))
print(DEST/'simple_models_fight_back_reviewed.pdf')
