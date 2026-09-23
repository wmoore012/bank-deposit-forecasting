"""Add the approved Study 2 introduction and measured payoff to the v5 PDF.

Run with Python containing reportlab and pypdf. Existing pages stay vector-based.
The v5 PDF remains the source; this writes the 13-page v6 edition.
"""
import csv
import math
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'simple_models_fight_back_data_science_story_v5.pdf'
OUTPUT = HERE / 'simple_models_fight_back_data_science_story_v6.pdf'
FONTS = Path('/System/Library/Fonts/Supplemental')
for name, file in [('Body', 'Arial.ttf'), ('Bold', 'Arial Bold.ttf'), ('Heavy', 'Arial Black.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONTS / file)))
W, H = 1344, 768
INK, CYAN, TEAL = '#2C3034', '#18BFE0', '#0B7279'
WARM, CARD, BORDER, MUTED = '#F4F2EC', '#FFFDF9', '#D8D5CD', '#74797D'


def text(c, x, y, value, size=20, font='Body', color=INK):
    c.setFillColor(color)
    c.setFont(font, size)
    c.drawString(x, H-y, value)


def base(c, number):
    c.setFillColor(WARM)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(CARD)
    c.setStrokeColor(BORDER)
    c.roundRect(33, 33, 1278, 702, 25, fill=1, stroke=1)
    text(c, 90, 88, f'SIMPLE MODELS FIGHT BACK · {number:02}', 13.5, 'Bold', TEAL)


def centered(c, y, value, size=20, font='Body', color=INK):
    width = pdfmetrics.stringWidth(value, font, size)
    text(c, (W-width)/2, y, value, size, font, color)


def line(c, x1, y1, x2, y2, color=TEAL, dash=None):
    c.setStrokeColor(color)
    c.setLineWidth(2)
    c.setDash(dash or [])
    c.line(x1, H-y1, x2, H-y2)
    c.setDash([])


rows = list(csv.DictReader((ROOT / 'experiments/review_history/outputs/outcome_capture.csv').open()))
pooled = {r['method']: r for r in rows if r['date'] == 'All three quarters'}
methods = [('Ridge', 'Ridge forecast'), ('MLP', 'Original neural network'),
           ('Autoencoder 42', 'Dense autoencoder'), ('Isolation Forest 42', 'Isolation Forest'), ('PCA', 'PCA'), ('Low equity/assets', 'Low equity/assets'),
           ('Largest deposits', 'Largest deposits first')]
assert all(int(pooled[m]['selected']) == 1351 and int(pooled[m]['eligible']) == 13490 for m, _ in methods)
assert [int(pooled[m]['captured']) for m, _ in methods] == [296, 293, 274, 237, 236, 170, 86]

new = BytesIO()
c = canvas.Canvas(new, pagesize=(W, H))
# Rebuild the size-only page with separate, clearly named scoring tasks.
base(c, 8)
text(c, 90, 142, 'I THREW OUT 4 INPUTS.', 36, 'Heavy')
text(c, 90, 190, 'KEPT DEPOSIT SIZE ALONE.', 36, 'Heavy', CYAN)
text(c, 90, 235, 'Do cash, loans, equity, and prior growth help choose which banks to review?', 23)
text(c, 90, 285, 'REVIEW-LIST TASK: FIND MORE LOW-GROWTH BANKS · HIGHER IS BETTER', 20, 'Bold', TEAL)
means = {r['Model']: float(r['Precision'])*100 for r in csv.DictReader((ROOT/'growth_outputs/masterclass/mean_ranking.csv').open())}
rank_rows = [('Five-input Ridge', means['Ridge']), ('Neural network', means['MLP']),
             ('Deposit-size Ridge', means['Size-only Ridge']), ('Random selection', 10.0)]
for i, (label, value) in enumerate(rank_rows):
    y = 330+i*48
    text(c, 100, y+5, label, 22, 'Bold' if i == 0 else 'Body')
    c.setFillColor(TEAL if i == 0 else '#9FB6BC')
    c.roundRect(395, H-y-8, value*28, 27, 5, fill=1, stroke=0)
    text(c, 409+value*28, y+5, f'{value:.1f}', 23, 'Bold')
text(c, 395, 522, 'Banks in next-quarter lowest-growth 10% per 100 selected', 18)
text(c, 90, 575, 'ZERO FORECAST: EVERY BANK TIED', 22, 'Bold', MUTED)
text(c, 90, 613, 'Zero cannot choose whom to review first. Random selection is the comparison above.', 21)
text(c, 90, 661, 'This page scores the review list. More low-growth banks found is better.', 24, 'Bold', TEAL)
text(c, 90, 713, 'Ranking: equal-quarter averages, three reused 2024 quarters. Size-only challenger added after examining outcomes. Original evaluation population.', 12, color=MUTED)
c.showPage()
base(c, 9)
text(c, 90, 143, "I KNOW WHAT YOU'RE THINKING.", 35, 'Heavy')
text(c, 90, 190, 'WHY NOT JUST USE ANOMALY DETECTION?', 35, 'Heavy', CYAN)
text(c, 90, 237, "Find the unusual banks. Put them on the review list. Isn't that what we're looking for?", 22)
text(c, 90, 280, "Well, let's try it.", 25, 'Bold')
for y, bullet in zip([319, 351, 383], [
    'Eight quarters of the same five financial inputs.',
    'Three methods: PCA, Isolation Forest, and a small autoencoder.',
    'Same banks to choose from. Same 10% review capacity.',
]):
    text(c, 90, y, '•  '+bullet, 21)
# Draw observed balances and a saved forecast, never invented plotting coordinates.
history = sorted([r for r in csv.DictReader((ROOT/'experiments/review_history/outputs/example_histories.csv').open())
                  if r['CERT'] == '27010'], key=lambda r: r['date'])
pred = next(r for r in csv.DictReader((ROOT/'growth_outputs/masterclass/predictions.csv').open())
            if r['CERT'] == '27010' and r['date'] == '2024-03-31')
forecast_balance = float(pred['DEPDOM'])*(1+float(pred['Ridge']))/1000
assert len(history) == 8
assert round(forecast_balance, 3) == 15.766
text(c, 90, 420, 'FORECAST: Young Americans Bank', 21, 'Bold', TEAL)
text(c, 90, 446, 'Domestic deposits (USD millions)', 16, color=MUTED)
x0, x1, top, bottom = 140, 490, 467, 555
fy = lambda v: bottom-(v-15)/6*(bottom-top)
for value in [15,18,21]:
    line(c, x0, fy(value), x1, fy(value), BORDER)
    text(c, 105, fy(value)+5, str(value), 13, color=MUTED)
points = [(x0+i*(x1-x0)/8, fy(float(r['DEPDOM'])/1000)) for i,r in enumerate(history)]
for start,end in zip(points,points[1:]):
    line(c,*start,*end)
for x,y in points:
    c.setFillColor(TEAL);c.circle(x,H-y,3,fill=1,stroke=0)
line(c,*points[-1],x1,fy(forecast_balance),CYAN,[4,3])
line(c,*points[-1],x1,points[-1][1],MUTED,[2,3])
text(c, 500, 513, 'Zero: 16.426', 14, color=MUTED)
text(c, 500, 545, f'Ridge: {forecast_balance:.3f}', 14, 'Bold', TEAL)
text(c, 140, 577, 'Jun 2022', 13, color=MUTED)
text(c, 386, 577, 'Mar 2024', 13, color=MUTED)
text(c, 485, 577, 'Jun forecast', 13, color=MUTED)
# Sorted empirical scores show exactly what determines the anomaly review list.
score_rows = [r for r in csv.DictReader((ROOT/'experiments/review_history/outputs/scores.csv').open())
              if r['date'] == '2024-03-31']
score_rows.sort(key=lambda r: (float(r['PCA']), -int(r['CERT'])))
assert len(score_rows) == 4548
k = math.ceil(.1*len(score_rows))
assert k == 455
text(c, 700, 420, 'ANOMALY: actual March 2024 PCA scores', 21, 'Bold', TEAL)
text(c, 700, 446, '4,548 banks · PCA reconstruction error (log scale)', 16, color=MUTED)
px0, px1 = 760, 1215
logs = [math.log10(float(r['PCA'])) for r in score_rows]
lo,hi = math.floor(min(logs)),math.ceil(max(logs))
py = lambda v: bottom-(v-lo)/(hi-lo)*(bottom-top)
for exponent in range(lo,hi+1,2):
    line(c,px0,py(exponent),px1,py(exponent),BORDER)
    text(c,704,py(exponent)+4,f'10^{exponent}',12,color=MUTED)
for i,value in enumerate(logs):
    x = px0+i/(len(logs)-1)*(px1-px0)
    c.setFillColor('#B16B24' if i >= len(logs)-k else '#9FB6BC')
    c.circle(x,H-py(value),1,fill=1,stroke=0)
cutoff = px0+(len(logs)-k)/(len(logs)-1)*(px1-px0)
line(c,cutoff,top,cutoff,bottom,'#B16B24',[3,3])
text(c,760,577,'Banks ordered by anomaly score',13,color=MUTED)
text(c,1092,463,'Top 455',14,'Bold','#B16B24')
text(c, 90, 611, 'Lowest forecast growth → review first.', 21, 'Bold', TEAL)
text(c, 700, 611, 'Highest reconstruction error → review first.', 21, 'Bold', TEAL)
text(c, 90, 647, 'Zero stays in the comparison: it predicts no change for every bank, so every rank is tied.', 21)
text(c, 90, 684, 'Same 10% workload. Which list finds more low-growth banks next quarter?', 25, 'Bold')
text(c, 90, 714, 'Sources: saved Study 1 predictions and Study 2 PCA scores. Historical forecast issued from March inputs; June outcome is not drawn.', 12, color=MUTED)
c.showPage()
base(c, 10)
text(c, 90, 143, 'RIDGE STILL BUILT THE BEST REVIEW LIST.', 34, 'Heavy')
text(c, 90, 190, 'MORE LOW-GROWTH BANKS. SAME WORKLOAD.', 34, 'Heavy', CYAN)
text(c, 90, 236, 'STILL THE REVIEW-LIST TASK: cases found per 100 selected. HIGHER IS BETTER.', 21, 'Bold', TEAL)
text(c, 90, 278, 'Out of every 100 selected, how many landed in the lowest-growth 10% next quarter?', 21, 'Bold')
text(c, 90, 307, 'All seven methods: 1,351 selections each across three 2024 quarters.', 18, color=MUTED)
# Rates answer the limited-review question directly; counts remain beside each bar.
for tick in [0, 5, 10, 15, 20, 25]:
    x = 400+tick*28
    c.setStrokeColor(BORDER)
    c.setLineWidth(0.5)
    c.line(x, H-330, x, H-550)
    text(c, x-5, 575, str(tick), 14, color=MUTED)
for i, (method, label) in enumerate(methods):
    y = 347+i*33
    value = int(pooled[method]['captured'])
    rate = 100*value/int(pooled[method]['selected'])
    text(c, 105, y+5, label, 20, 'Bold' if i == 0 else 'Body')
    c.setFillColor(TEAL if i == 0 else '#496B9D' if i == 1 else '#9FB6BC')
    c.roundRect(400, H-y-6, rate*28, 21, 5, fill=1, stroke=0)
    text(c, 413+rate*28, y+5, f'{rate:.1f}', 22, 'Bold')
    text(c, 1130, y+5, f'{value} / 1,351', 17, color=MUTED)
text(c, 105, 325, 'METHOD', 12, 'Bold', MUTED)
text(c, 1130, 325, 'EXACT COUNTS', 12, 'Bold', MUTED)
text(c, 90, 601, 'Zero forecast: all banks tied. No informative review order.', 19, 'Bold', MUTED)
text(c, 90, 629, 'SCORECARD SO FAR', 17, 'Bold', MUTED)
text(c, 90, 655, 'MAE: zero growth     |     RMSE: neural network     |     Review list: Ridge', 24, 'Bold', TEAL)
text(c, 90, 697, '13,490 complete histories with observed outcomes; lists recalculated at 10% within each quarter. Reused 2024 data, not a fresh test.', 12, color=MUTED)
text(c, 90, 716, 'Stochastic models: seed 42 shown. No other tested seed or version without deposit size exceeded Ridge. Anomalies do not establish distress.', 12, color=MUTED)
c.showPage()
# The shorter-window chart stacks the baseline MAE and its measured increase.
# These sum to each run's MAE; errors from separate models are never added.
window_rows = list(csv.DictReader((ROOT / 'experiments/training_window_sensitivity/scores.csv').open()))
window = {int(r['training_start']): float(r['MAE_pp']) for r in window_rows
          if r['model'] == 'MLP' and r['seed'] == '42.0'}
assert set(window) == {2013, 2020, 2021}
assert all(window[y] >= window[2013] for y in window)
base(c, 11)
text(c, 90, 143, 'BUT WAIT... IN THESE VOLATILE ECONOMIC TIMES,', 29, 'Heavy')
text(c, 90, 187, 'WHAT IF WE TRAIN ON ONLY THE RECENT YEARS?', 31, 'Heavy', CYAN)
text(c, 90, 232, 'Maybe older examples are less useful now. So I retrained starting in 2020, then 2021.', 22)
text(c, 90, 270, 'Same model settings. Same 2023 validation. Same 13,532 bank-quarters scored in 2024.', 20, 'Bold')
text(c, 105, 310, 'BACK TO FORECAST ERROR: NEURAL NETWORK MAE (pp) · LOWER IS BETTER', 19, 'Bold')
for x, color, label in [(265, TEAL, 'Full-window error'), (640, '#B16B24', 'Additional error with shorter window')]:
    c.setFillColor(color)
    c.rect(x, H-344, 17, 17, fill=1, stroke=0)
    text(c, x+26, 342, label, 17)
for tick in range(6):
    x = 385+tick*145
    line(c, x, 364, x, 540, BORDER)
    text(c, x-5, 566, str(tick), 15, color=MUTED)
for y, year in zip([390, 450, 510], [2013, 2020, 2021]):
    v = window[year]
    baseline = window[2013]
    c.setFillColor(TEAL)
    c.rect(385, H-y-13, baseline*145, 32, fill=1, stroke=0)
    c.setFillColor('#B16B24')
    c.rect(385+baseline*145, H-y-13, (v-baseline)*145, 32, fill=1, stroke=0)
    text(c, 403+v*145, y+10, f'{v:.3f}', 22, 'Bold')
    text(c, 175, y+10, f'{year}-Sep 2022', 22, 'Bold')
zero_mae = next(float(r['MAE (pp)']) for r in csv.DictReader((ROOT/'growth_outputs/masterclass/extended_scores.csv').open()) if r['Model']=='Zero growth')
zero_x = 385+zero_mae*145
line(c,zero_x,365,zero_x,533,'#496B9D',[3,3])
text(c, 890, 591, f'Zero MAE: {zero_mae:.3f} pp', 17, 'Bold', '#496B9D')
# Arrow points to the additional error for the 2020-start fit.
arrow_x = 385+(window[2013]+window[2020])/2*145
line(c, 1145, 413, arrow_x, 445, '#C23830')
line(c, arrow_x,445,arrow_x+13,433,'#C23830')
line(c, arrow_x,445,arrow_x+17,448,'#C23830')
text(c, 1030, 397, f"+{window[2020]-window[2013]:.3f} pp", 18, 'Bold', '#C23830')
centered(c, 617, 'AND... THE ERRORS GOT BIGGER.', 30, 'Heavy', '#B16B24')
centered(c, 650, 'Both shorter windows had higher MAE with all three seeds. Ridge had higher MAE too.', 21)
centered(c, 683, 'So what if we keep the history, but let the forecast know the quarter?', 23, 'Bold', TEAL)
text(c, 90, 713, 'Seed 42 shown; also checked 7 and 99. Reused 2024 outcomes. Shorter windows also have fewer training examples; no regime effect is isolated.', 12, color=MUTED)
c.showPage()
c.save()

reader = PdfReader(SOURCE)
assert len(reader.pages) == 11
added = PdfReader(new)
writer = PdfWriter()
for i, original in enumerate(reader.pages):
    if i == 7:
        for p in added.pages:
            writer.add_page(p)
        continue  # Four generated pages replace old pages 8-9 and add Study 2.
    if i == 8:
        continue
    if i >= 9:
        # Cover only the old page-number header, retaining all original artwork.
        overlay = BytesIO()
        header = canvas.Canvas(overlay, pagesize=(W, H))
        header.setFillColor(CARD)
        header.rect(85, H-93, 450, 22, fill=1, stroke=0)
        text(header, 90, 88, f'SIMPLE MODELS FIGHT BACK · {i+3:02}', 13.5, 'Bold', TEAL)
        header.save()
        original.merge_page(PdfReader(overlay).pages[0])
    writer.add_page(original)
writer.add_metadata({'/Title': 'Simple Models Fight Back', '/Subject': 'Deposit forecasts and unusual financial histories'})
with OUTPUT.open('wb') as f:
    writer.write(f)
assert len(PdfReader(OUTPUT).pages) == 13
print(OUTPUT)
