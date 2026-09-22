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
 n=len(pages)+1
 f=plt.figure(figsize=(14,8),facecolor='white')
 f.text(.055,.94,f'SIMPLE MODELS FIGHT BACK  /  {n:02}',color=T,size=11,weight='bold')
 f.text(.055,.875,textwrap.fill(title,52),size=27,weight='bold',va='top')
 f.text(.055,.735,textwrap.fill(sub,115),size=13,va='top')
 f.text(.055,.12,textwrap.fill(note,135),size=11,va='top',linespacing=1.4)
 f.text(.945,.035,f'{n:02} / 11 • Bank deposit forecasting',ha='right',size=9,color=G)
 pages.append(f);return f
def axis(f,rect):
 a=f.add_axes(rect);a.spines[['top','right']].set_visible(False);a.grid(axis='x',alpha=.15);a.set_axisbelow(True);return a
f=page(1,'I trained a neural network on 214,425 bank-quarter examples.','And the lowest average miss among the original four forecasts came from this:','MAE measures the average absolute miss in percentage points. All scores use the same 13,532 bank-quarter examples in three already-examined 2024 quarters. Follow-up baselines appear later.')
f.text(.5,.52,'PREDICT 0%',ha='center',size=65,weight='bold',color=T)
f.text(.5,.40,'for every bank.',ha='center',size=27)
f.text(.5,.25,'Zero: 3.601 pp MAE   |   MLP: 3.627   |   Ridge: 3.641',ha='center',size=20)
f=page(0,'Zero is a guess. The next report supplies the answer.',
 'The zero-growth rule always predicts: next quarter’s deposits will equal today’s deposits.',
 'Illustrative amounts. Deposit growth describes a balance change. Even an accurate forecast says nothing by itself about whether a bank is healthy.')
f.text(.08,.60,'THE FORECAST',color=T,size=15,weight='bold')
f.text(.08,.48,'$100M → $100M',size=34,weight='bold')
f.text(.08,.37,'Predicted growth: 0%',size=21)
f.text(.55,.60,'SUPPOSE THIS HAPPENS',color=B,size=15,weight='bold')
f.text(.55,.48,'$100M → $90M',size=34,weight='bold')
f.text(.55,.37,'Actual growth: −10%',size=21)
f.text(.5,.22,'Zero missed by 10 percentage points. Now another model has something to beat.',ha='center',size=17,weight='bold')
f=page(0,'Wait. Zero what? The units change the meaning.',
 'Growth is the change in deposits. Forecast error measures how far the prediction missed.',
 'Illustrative arithmetic, not observed banks. RMSE combines squared misses across many forecasts; it is not a bank balance or a health score.')
for x,value,balance in [(.08,'0%','$100M → $100M'),(.39,'+500%','$100M → $600M'),(.70,'+2,000%','$100M → $2.1B')]:
 f.text(x,.60,value,size=38,color=T,weight='bold');f.text(x,.52,balance,size=19);f.text(x,.45,'Deposit growth',size=13)
f.text(.08,.32,'0 pp RMSE: every forecast is exactly right.',size=20,weight='bold')
f.text(.08,.25,'500 or 2,000 pp RMSE: enormous forecast errors.',size=20,weight='bold')
f.text(.08,.19,'With one forecast, predicting +500% when actual growth is 0% gives 500 pp RMSE.',size=14)
f=page(2,'Wait. How can predicting zero win anything?','Look where the training examples pile up. Zero is worth testing. The later scores tell us whether it works.','This histogram shows the central 98% of training log-growth targets. The full tails remain in the experiment. The horizontal unit is 100 × log growth; it is not ordinary percentage-point growth.')
a=axis(f,[.12,.27,.79,.36]);edges=np.array(hist['edges']);a.bar(edges[:-1],hist['counts'],width=np.diff(edges),align='edge',color='#BEC3C5',edgecolor='white');a.axvline(0,color=T,lw=3,label='Zero forecast');a.set_xlabel('100 × log(D next / D today)');a.set_ylabel('Training examples');a.legend()
f=page(3,'Okay, so which forecast actually wins?','Average size of the miss: zero. Give the biggest misses more weight: the neural network. Same banks.','MAE differences among zero, Ridge and MLP are only about 0.04 pp. RMSE gives large misses more weight. These rankings describe the historical sample; they do not establish future superiority.')
for j,metric in enumerate(['MAE (pp)','RMSE (pp)']):
 d=scores.loc[['Zero growth','MLP','Ridge','Persistence']].sort_values(metric);a=axis(f,[.18+j*.47,.28,.28,.34]);a.scatter(d[metric],range(4),c=[T,G,G,G],s=70);a.set_yticks(range(4),d.index);a.invert_yaxis();a.set_xlim(0,6 if j==0 else 11);a.set_xlabel(metric)
 for i,v in enumerate(d[metric]):a.text(v+.1,i,f'{v:.3f}',va='center',size=10)
f=page(4,'Good. Now tell me which bank to examine first.','Bank A gets zero. Bank B gets zero. Every bank gets zero. Imagine you have time to examine just one.','Zero growth remains a valid point-forecast baseline. Selecting among tied forecasts requires a separate rule. Random selection supplies the comparison for the review-list experiment.')
a=axis(f,[.16,.25,.34,.41]);a.scatter(np.zeros(10),range(10),color=T,s=55);a.set_yticks(range(10),[f'Bank {i+1:02}' for i in range(10)]);a.set_xticks([0],['0%']);a.set_xlim(-1,1);a.invert_yaxis();f.text(.59,.49,'So... who goes first?\nEverybody is tied.',size=27,weight='bold',linespacing=1.6)
f=page(5,'Imagine 4,536 bank reports on your desk.','', 'March → June 2024. Review capacity is hypothetical. Lowest growth can still be positive. Each full grid represents 454 selections. Colored dots show rounded shares, not individual banks.')
f.text(.055,.735,"You have time to examine 454. Which reports get your attention?\nSo you want a list that finds banks whose deposits will grow least, or shrink most.\nThree months later, count how many selected banks landed in the lowest-growth 454.",size=15,va='top',linespacing=1.4)
n=int(march.loc['Ridge','Banks']);k=int(march.loc['Ridge','Selected'])
for j,(name,h,col) in enumerate([('Random',k*k/n,G),('Ridge',int(march.loc['Ridge','Hits']),C),('Neural network',int(march.loc['MLP','Hits']),T)]):
 a=f.add_axes([.09+j*.31,.25,.24,.29]);count=round(100*h/k);a.scatter(np.tile(np.arange(10),10),np.repeat(np.arange(10)[::-1],10),c=[col]*count+['#DEDEDE']*(100-count),s=42);a.axis('off');a.set_title(name,size=18);a.text(.5,-.15,f'{h:.2f} expected' if j==0 else f'{h} found',transform=a.transAxes,ha='center',size=20,weight='bold')
f=page(6,'Okay, but how much of this is just bank size?','', 'Equal-quarter averages across three reused 2024 quarters; select the lowest predicted-growth 10%. Size-only Ridge was added after examining 2024. Found = later in the lowest-growth 10%. Governance practice: test added complexity against a simple challenger.')
f.text(.055,.735,"Here, size means deposit dollars: a bigger deposit base holds more customer money.\nWho gets reviewed? Banks with the lowest predicted growth, rather than simply the biggest.\nSize-only Ridge finds about 19 per 100 selections; all five inputs find about 22.",size=15,va='top',linespacing=1.4)
a=axis(f,[.25,.26,.59,.29]);models=['Persistence','Size-only Ridge','Ridge','MLP'];v=100*lift.loc[models,'Precision'];a.scatter(v,range(4),color=[G,B,C,T],s=90);a.set_yticks(range(4),['Previous quarter','Deposit size only','All five inputs: Ridge','Neural network']);a.invert_yaxis();a.axvline(10,color=G,ls='--');a.set_xlim(0,27);a.set_xlabel('Banks in the lowest-growth group per 100 selections')
a.text(10,-.45,'Chance ≈ 10',ha='center',size=10,color=G)
for i,x in enumerate(v):a.text(x+.5,i,f'{x:.1f}',va='center',weight='bold')
f=page(7,'So let’s give a simple rule a calendar.','Meet the seasonal-median baseline. For each calendar quarter, predict its training-period median growth. This is a separate rule.','Seasonal median: 3.500 pp MAE; zero growth: 3.601 pp on reused 2024 data. This baseline was added after 2024 had been examined. Its fitting uses earlier rows; its development was post-hoc.')
a=axis(f,[.15,.36,.7,.27]);a.bar(season.Quarter,season['Decline share']*100,color=[C,C,G,G]);a.set_xticks([1,2,3,4],['Q1','Q2','Q3','Q4']);a.set_ylim(0,60);a.set_ylabel('Next-quarter declines (%)')
for q,v in zip(season.Quarter,season['Decline share']*100):a.text(q,v+2,f'{v:.1f}%',ha='center',weight='bold')
f.text(.5,.23,'And... the seasonal rule beat zero on MAE: 3.500 vs 3.601 pp.',ha='center',size=18,weight='bold')
f=page(8,'Can pretrained models beat the no-change forecast?','The no-change forecast scores 6.718 pp RMSE. These tested pretrained models score higher. Higher means larger errors.','As-is means frozen pretrained weights. Adaptation updated only the forecast head for 256 sampled steps, with earlier validation selecting the checkpoint. Pretraining overlap is unknown. This limited test does not establish what other checkpoints or training budgets could achieve.')
labels=['MLP','Zero growth','Chronos as-is','Chronos head adapted','TimesFM as-is','TimesFM head adapted'];values=[scores.loc['MLP','RMSE (pp)'],scores.loc['Zero growth','RMSE (pp)']]+[foundation[k]['RMSE (pp)'] for k in ['chronos_zero_shot','chronos_finetuned','timesfm_zero_shot','timesfm_finetuned']]
a=axis(f,[.28,.32,.55,.32]);a.scatter(values,range(6),color=[T,G,B,B,A,A],s=65);a.set_xscale('log');a.set_xticks([5,10,100,300],['5','10','100','300']);a.minorticks_off();a.set_xlim(5,420);a.set_yticks(range(6),labels);a.invert_yaxis();a.set_xlabel('RMSE (pp), logarithmic axis — lower is better')
for i,v in enumerate(values):a.text(v*1.07,i,f'{v:.3f}',va='center',size=11)
f=page(0,'Now run it before the answers arrive.',
 'Proposed shadow-mode trial: save forecasts each quarter while people keep making the decisions.',
 'Model-governance practices demonstrated here: benchmarking, coverage checks, outcome monitoring, and documented change decisions. Proposed workflow; no deployment or regulatory-compliance claim.')
steps=[('1  FREEZE + SAVE','Version the model, inputs and selection rule. Save the dated review list.'),('2  WAIT + MEASURE','When reports arrive, check MAE, RMSE, review precision and banks left unscored.'),('3  COMPARE + INVESTIGATE','Compare every challenger on the same rows. Check data gaps and institutional events.'),('4  DECIDE + RECORD','If performance slips, investigate first. Validate any replacement before adopting it.')]
for y,(title,body) in zip([.61,.49,.37,.25],steps):
 f.text(.07,y,title,size=17,color=T,weight='bold');f.text(.07,y-.045,body,size=14)

with PdfPages(DEST/'simple_models_fight_back_reviewed.pdf') as pdf:
 for i,f in enumerate(pages,1):pdf.savefig(f);f.savefig(DEST/f'{i:02}.png',dpi=120);plt.close(f)
sources=[OUT/n for n in ['extended_scores.csv','mean_ranking.csv','seasonal_rules.csv','ranking.csv']]+[ROOT/f'growth_outputs/{k}_predictions.json' for k in foundation]+[DEST/'training_histogram.json']
(DEST/'manifest.json').write_text(json.dumps({'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'pages':len(pages),'historical_evaluation':'2024 Q1–Q3 predictors; reused'},indent=2))
print(DEST/'simple_models_fight_back_reviewed.pdf')
