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
    text(f,.055,.885,title,30,INK,'bold')
    text(f,.055,.105,note,11)
    text(f,.945,.035,f'{len(pages):02} / 06  •  Reused 2024 evaluation',9,GRAY,ha='right')
    return f
def score(model,metric): return scores.loc[model,metric]

f=page('NEURAL NETS LEARN CURVES.\nMINE LOST TO A FLAT LINE.',
       'Average absolute forecast miss (MAE), percentage points. Lower is better. Original four-model comparison.\n214,425 training examples; 13,532 evaluation examples. The small gap does not establish a future winner.')
text(f,.055,.705,'I used quarterly bank reports to predict how deposits would change next quarter.',16)
a=f.add_axes([.08,.48,.35,.14]);x=np.linspace(0,1,100);a.plot(x,.3*np.sin(x*13)+.2*x,color=GRAY,lw=3);a.axis('off')
text(f,.255,.455,'NEURAL NET',14,GRAY,'bold',ha='center')
a=f.add_axes([.57,.48,.35,.14]);a.plot([0,1],[0,0],color=CYAN,lw=5);a.set_ylim(-1,1);a.axis('off')
text(f,.745,.455,'0% FOR EVERY BANK',14,TEAL,'bold',ha='center')
text(f,.255,.37,f"{score('MLP','MAE (pp)'):.3f}",44,INK,'bold',ha='center')
text(f,.745,.37,f"{score('Zero growth','MAE (pp)'):.3f}",54,TEAL,'bold',ha='center')
text(f,.5,.185,'Lines illustrate the two ideas; they are not plotted predictions.',11,GRAY,ha='center')

f=page('WHY WAS A FLAT LINE SO HARD TO BEAT?',
       'Central 98% of training log-growth targets. Full tails stayed in the experiment.\nTraining concentration motivates the baseline; the later evaluation establishes its score.')
text(f,.055,.765,'A LOT OF DEPOSIT CHANGES SIT NEAR ZERO.',23,INK,'bold')
a=f.add_axes([.12,.30,.78,.37]);edges=np.array(hist['edges']);a.bar(edges[:-1],hist['counts'],width=np.diff(edges),align='edge',color='#C8CED1',edgecolor='white');a.axvline(0,color=CYAN,lw=4)
a.annotate('0% growth',xy=(0,max(hist['counts'])),xytext=(1.3,max(hist['counts'])*1.10),color=TEAL,fontsize=16,fontweight='bold');a.set_ylim(0,max(hist['counts'])*1.25)
a.spines[['top','right']].set_visible(False);a.set_xlabel('100 × log(next deposits / current deposits)',fontsize=12);a.set_ylabel('Training examples',fontsize=12)
text(f,.5,.205,'BORING CAN WORK.',27,INK,'bold',ha='center')

f=page('ZERO WON THE AVERAGE MISS.\nTHE NET WON WHEN BIG MISSES HURT MORE.',
       'MAE on the left; RMSE on the right. Both measure ordinary-growth forecast errors in percentage points.\nSame evaluation rows, original four models. Lower RMSE does not mean every large miss improved.')
text(f,.08,.66,'AVERAGE MISS ↓',18,TEAL,'bold');text(f,.57,.66,'BIG MISSES HURT MORE ↓',18,TEAL,'bold')
text(f,.08,.56,f"0%   {score('Zero growth','MAE (pp)'):.3f}",40,TEAL,'bold')
text(f,.57,.56,f"Neural net   {score('MLP','RMSE (pp)'):.3f}",32,TEAL,'bold')
text(f,.08,.43,f"Neural net   {score('MLP','MAE (pp)'):.3f}",23)
text(f,.57,.43,f"0%   {score('Zero growth','RMSE (pp)'):.3f}",23)
text(f,.5,.235,'“WINNER” DEPENDS ON WHAT HURTS.',29,INK,'bold',ha='center')

f=page('THE FLAT LINE COULDN’T TELL ME WHERE TO LOOK.',
       'March → June 2024. Hypothetical 10% review capacity. Found = selected bank later in the lowest-growth 10%.\nThat group can include positive growth. A human still investigates the cause; no savings were measured.')
text(f,.07,.75,'BANK A   0%     BANK B   0%     BANK C   0%     BANK D   0%',19,GRAY)
text(f,.07,.665,'EVERYBODY IS TIED.',24,INK,'bold')
text(f,.07,.57,'4,536 BANKS.  454 REVIEW SLOTS.',27,INK,'bold')
n=int(march.loc['Ridge','Banks']);k=int(march.loc['Ridge','Selected'])
for xpos,label,val,suffix in [(.20,'RANDOM SELECTION',f'{k*k/n:.1f}','expected'),(.51,'RIDGE',str(int(march.loc['Ridge','Hits'])),'found'),(.81,'NEURAL NET',str(int(march.loc['MLP','Hits'])),'found')]:
    text(f,xpos,.465,label,15,TEAL,'bold',ha='center');text(f,xpos,.395,val,45,INK,'bold',ha='center');text(f,xpos,.29,suffix,16,GRAY,ha='center')
text(f,.5,.20,'NOW I HAVE A LIST.',29,INK,'bold',ha='center')

f=page('I THREW OUT 4 INPUTS.\nKEPT ONLY DEPOSIT SIZE.',
       'Lowest-growth banks found per 100 selected; equal-quarter averages over three reused 2024 quarters.\nSize-only Ridge was a later challenger. This is not a causal decomposition; the gap needs a fresh test.')
text(f,.055,.705,'Deposit size = customer deposit dollars. Both comparisons use Ridge.',16)
a_val=100*means.loc['Size-only Ridge','Precision'];b_val=100*means.loc['Ridge','Precision']
for xpos,label,value in [(.24,'1 INPUT',a_val),(.76,'5 INPUTS',b_val)]:
    text(f,xpos,.59,label,20,TEAL,'bold',ha='center');text(f,xpos,.505,f'{value:.1f}',64,INK,'bold',ha='center')
text(f,.5,.455,f'+{b_val-a_val:.1f}',28,TEAL,'bold',ha='center');text(f,.5,.365,'per 100 selected',13,GRAY,ha='center')
text(f,.5,.295,'Random selection ≈ 10',17,GRAY,ha='center')
text(f,.5,.205,'THAT’S IT?',31,INK,'bold',ha='center')

f=page('A CALENDAR RULE BEAT THE FLAT LINE.',
       'Seasonal median was developed after examining 2024. It predicts each calendar quarter’s training median.\nHistorical MAE in percentage points; lower is better. Analysis notebook built with wm-notecards.')
text(f,.055,.765,'How often did deposits fall next quarter? Training years:',16)
for xpos,(_,row) in zip([.15,.38,.61,.84],season.iterrows()):
    text(f,xpos,.695,f"Q{int(row['Quarter'])}   {100*row['Decline share']:.1f}%",22,TEAL,'bold',ha='center')
text(f,.15,.585,'SEASONAL RULE',15,TEAL,'bold');text(f,.60,.585,'FLAT LINE',15,GRAY,'bold')
text(f,.15,.535,f"{score('Seasonal median','MAE (pp)'):.3f}",43,TEAL,'bold');text(f,.60,.535,f"{score('Zero growth','MAE (pp)'):.3f}",43,INK,'bold')
text(f,.5,.405,'SERIOUSLY?',23,INK,'bold',ha='center')
text(f,.5,.32,'THE NEXT RESULT I CARE ABOUT\nIS THE ONE I DON’T KNOW YET.',26,INK,'bold',ha='center')
text(f,.5,.19,'Freeze these rules. Test the next unseen quarter.',16,ha='center')

assert len(pages)==6
assert (n,k)==(4536,454)
assert sum(hist['counts'])==210135
assert round(b_val-a_val,1)==3.1
with PdfPages(DEST/'simple_models_fight_back_reviewed.pdf') as pdf:
    for i,f in enumerate(pages,1):
        pdf.savefig(f);f.savefig(DEST/f'{i:02}.png',dpi=150);plt.close(f)
sources=[OUT/name for name in ['extended_scores.csv','mean_ranking.csv','seasonal_rules.csv','ranking.csv']]+[DEST/'training_histogram.json']
(DEST/'manifest.json').write_text(json.dumps({'pages':6,'historical_evaluation':'2024 Q1–Q3 predictors; reused','source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2))
print(DEST/'simple_models_fight_back_reviewed.pdf')
