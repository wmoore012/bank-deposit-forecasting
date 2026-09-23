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


def crown(c, x, y, size=30):
    """Tilted vector crown: sharp at PDF scale, independent of emoji fonts."""
    c.saveState()
    c.translate(x, H-y)
    c.rotate(13)
    c.scale(size/30, size/30)
    c.setFillColor('#F5BF24')
    c.setStrokeColor('#513A06')
    c.setLineWidth(1.5)
    path=c.beginPath()
    path.moveTo(-14,-8)
    for px,py in [(-17,10),(-7,3),(0,15),(7,3),(17,10),(14,-8)]:
        path.lineTo(px,py)
    path.close()
    c.drawPath(path,fill=1,stroke=1)
    c.rect(-14,-12,28,5,fill=1,stroke=1)
    c.restoreState()


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
text(c, 90, 141, 'OKAY. WE CAN RANK THE BANKS.', 33, 'Heavy')
text(c, 90, 184, 'CAN DEPOSIT SIZE ALONE MATCH FIVE INPUTS?', 33, 'Heavy', CYAN)
text(c, 90, 239, 'No. Size alone found fewer low-growth banks than five-input Ridge.', 23, 'Bold')
text(c, 90, 285, 'BANKS FOUND IN THE LOWEST-GROWTH 10% · HIGHER IS BETTER', 20, 'Bold', TEAL)
means = {r['Model']: float(r['Precision'])*100 for r in csv.DictReader((ROOT/'growth_outputs/masterclass/mean_ranking.csv').open())}
rank_rows = [('Five-input Ridge', means['Ridge']), ('Neural network', means['MLP']),
             ('Deposit-size Ridge', means['Size-only Ridge']), ('Random selection', 10.0)]
for i, (label, value) in enumerate(rank_rows):
    y = 330+i*48
    text(c, 100, y+5, label, 22, 'Bold' if i == 0 else 'Body')
    c.setFillColor(TEAL if i == 0 else '#9FB6BC')
    c.roundRect(395, H-y-8, value*28, 27, 5, fill=1, stroke=0)
    text(c, 409+value*28, y+5, f'{value:.1f}', 23, 'Bold')
crown(c, 1110, 328)
text(c, 395, 522, 'Low-growth banks found per 100 banks reviewed', 18)
text(c, 90, 605, 'ZERO GIVES EVERY BANK THE SAME ANSWER.', 30, 'Heavy')
text(c, 90, 647, 'IT CANNOT TELL ME WHOM TO REVIEW FIRST.', 30, 'Heavy', TEAL)
text(c, 90, 713, 'Ranking: equal-quarter averages, three reused 2024 quarters. Size-only challenger added after examining outcomes. Original evaluation population.', 12, color=MUTED)
c.showPage()
base(c, 9)
text(c, 90, 143, "WHY NOT TRY ANOMALY DETECTION?", 35, 'Heavy')
text(c, 90, 190, 'HOW DOES PCA CHOOSE BANKS TO REVIEW?', 35, 'Heavy', CYAN)
text(c, 90, 237, 'We can review only 10% of banks.', 30, 'Bold')
text(c, 90, 279, "EXAMPLE · Let's try PCA.", 24, 'Bold', TEAL)
text(c, 90, 313, 'It learns to compress past bank histories', 24)
text(c, 90, 344, 'and rebuild the original numbers.', 24)
text(c, 90, 382, 'We review the banks with the biggest errors.', 24, 'Bold')
text(c, 800, 266, 'SCOPE', 20, 'Bold', TEAL)
text(c, 800, 296, 'Eight quarters · five financial inputs', 22)
text(c, 800, 335, 'ANOMALY MODELS TESTED', 20, 'Bold', TEAL)
text(c, 800, 365, 'PCA, Isolation Forest,', 22)
text(c, 800, 392, 'and a small autoencoder', 22)
# Sorted empirical scores show exactly what determines the anomaly review list.
score_rows = [r for r in csv.DictReader((ROOT/'experiments/review_history/outputs/scores.csv').open())
              if r['date'] == '2024-03-31']
score_rows.sort(key=lambda r: (float(r['PCA']), -int(r['CERT'])))
assert len(score_rows) == 4548
k = math.ceil(.1*len(score_rows))
assert k == 455
# One empirical ranking: highlight the review set without repeating the same curve.
text(c, 90, 430, 'RANK ALL 4,548 BANKS · REVIEW THE 455 HIGHEST ERRORS', 23, 'Bold', TEAL)
text(c, 90, 459, 'March 2024 · PCA reconstruction error (log scale)', 18, color=MUTED)
top, bottom = 482, 594
all_logs = [math.log10(float(r['PCA'])) for r in score_rows]
full_lo, full_hi = math.floor(min(all_logs)), math.ceil(max(all_logs))
full_y = lambda v: bottom-(v-full_lo)/(full_hi-full_lo)*(bottom-top)
for exponent in range(full_lo, full_hi+1):
    line(c,140,full_y(exponent),850,full_y(exponent),BORDER)
    text(c,91,full_y(exponent)+4,f'10^{exponent}',14,color=MUTED)
for i,value in enumerate(all_logs):
    c.setFillColor('#B3132B' if i >= len(score_rows)-k else '#9FB6BC')
    c.circle(140+i/(len(score_rows)-1)*710,H-full_y(value),1.4,fill=1,stroke=0)
cut_x = 140+(len(score_rows)-k)/(len(score_rows)-1)*710
line(c,cut_x,478,cut_x,598,'#B3132B',[3,3])
text(c,140,623,'Lowest error',18,color=MUTED)
text(c,646,623,'455 selected →',20,'Bold','#B3132B')
flag_y=full_y(all_logs[-1])
c.setStrokeColor('#B3132B');c.setLineWidth(3)
c.line(850,H-flag_y,850,H-flag_y+35)
c.setFillColor('#D51C37')
flag=c.beginPath();flag.moveTo(850,H-flag_y+35)
flag.lineTo(885,H-flag_y+27);flag.lineTo(850,H-flag_y+18);flag.close()
c.drawPath(flag,fill=1,stroke=0)
assert score_rows[-1]['CERT'] == '27330'
c.setFillColor('#FFF0F1');c.setStrokeColor('#B3132B');c.setLineWidth(1.5)
c.roundRect(945,H-614,305,142,10,fill=1,stroke=1)
text(c,963,502,'SILVERGATE',24,'Bold','#B3132B')
text(c,963,530,'The highest error.',22,'Bold')
text(c,963,559,'PCA rebuilt its history',21)
text(c,963,586,'least accurately.',21)
line(c,885,flag_y,945,543,'#B3132B')
text(c,90,671,'Will these unusual banks have low growth next quarter?',28,'Bold',TEAL)
text(c,90,711,'Source: saved March 2024 PCA scores. Unusual does not mean distressed.',14,color=MUTED)
c.showPage()
base(c, 10)
text(c, 90, 143, 'RIDGE RANKED THE REVIEW LIST BEST HERE.', 32, 'Heavy')
text(c, 90, 201, 'Low-growth banks found per 100 banks reviewed', 29, 'Bold', TEAL)
text(c, 90, 237, 'Target: next-quarter bottom 10% in deposit growth. Higher is better.', 22)
# A single scale and directly labeled bars carry the comparison.
for tick in [0, 5, 10, 15, 20, 25]:
    x = 435+tick*27
    c.setStrokeColor(BORDER)
    c.setLineWidth(0.5)
    c.line(x, H-266, x, H-533)
    text(c, x-6, 562, str(tick), 17, color=MUTED)
for i, (method, label) in enumerate(methods):
    y = 285+i*38
    value = int(pooled[method]['captured'])
    rate = 100*value/int(pooled[method]['selected'])
    text(c, 100, y+7, label, 24, 'Bold' if i == 0 else 'Body')
    c.setFillColor(TEAL if i == 0 else '#496B9D' if i == 1 else '#9FB6BC')
    c.roundRect(435, H-y-8, rate*27, 26, 5, fill=1, stroke=0)
    text(c, 449+rate*27, y+7, f'{rate:.1f}', 25, 'Bold')
crown(c, 1135, 286, 30)
c.setFillColor('#E9F5F5');c.setStrokeColor(BORDER)
c.roundRect(90,H-648,1160,61,12,fill=1,stroke=1)
text(c, 112, 626, 'PREDICT ZERO CANNOT RANK BANKS. SO WE CANNOT USE IT FOR THIS LIST.', 24, 'Bold', TEAL)
text(c, 90, 683, '1,351 bank-quarter reviews per method · 13,490 eligible rows · three reused 2024 quarters', 18)
text(c, 90, 712, 'Primary seed 42 shown. These historical results do not establish future performance.', 18, color=MUTED)
c.showPage()
# Training-window color is shared by each date label and its entire MAE bar.
window_rows = list(csv.DictReader((ROOT / 'experiments/training_window_sensitivity/scores.csv').open()))
window = {int(r['training_start']): float(r['MAE_pp']) for r in window_rows
          if r['model'] == 'MLP' and r['seed'] == '42.0'}
assert set(window) == {2013, 2020, 2021}
assert all(window[y] >= window[2013] for y in window)
base(c, 11)
text(c, 90, 143, 'BUT WAIT... IN THESE VOLATILE ECONOMIC TIMES,', 29, 'Heavy')
text(c, 90, 187, 'WHAT IF WE TRAIN ON ONLY THE RECENT YEARS?', 31, 'Heavy', INK)
text(c, 90, 232, 'Maybe older examples are less useful now. So I retrained starting in 2020, then 2021.', 22)
text(c, 90, 270, 'Model settings held fixed; validated in 2023; scored on 13,532 bank-quarters in 2024.', 20, 'Bold')
text(c, 105, 310, 'BACK TO FORECAST ERROR: NEURAL NETWORK MAE (pp) · LOWER IS BETTER', 19, 'Bold')
older, recent, latest = CYAN, '#F05A71', '#C91D46'
text(c,175,342,'OLDER HISTORY',18,'Bold',older)
text(c,433,342,'MORE RECENT HISTORY',18,'Bold',latest)
for tick in range(6):
    x = 385+tick*145
    line(c, x, 364, x, 540, BORDER)
    text(c, x-5, 566, str(tick), 15, color=MUTED)
for y, year in zip([390, 450, 510], [2013, 2020, 2021]):
    v = window[year]
    period_color = {2013: older, 2020: recent, 2021: latest}[year]
    c.setFillColor(period_color)
    c.rect(385, H-y-13, v*145, 32, fill=1, stroke=0)
    text(c, 403+v*145, y+10, f'{v:.3f}', 22, 'Bold')
    text(c, 175, y+10, f'{year}-Sep 2022', 22, 'Bold', period_color)
zero_mae = next(float(r['MAE (pp)']) for r in csv.DictReader((ROOT/'growth_outputs/masterclass/extended_scores.csv').open()) if r['Model']=='Zero growth')
zero_x = 385+zero_mae*145
line(c,zero_x,365,zero_x,533,INK,[3,3])
text(c, 895, 360, 'Zero', 15, 'Bold', INK)
# Arrow points to the additional error for the 2020-start fit.
arrow_x = 385+(window[2013]+window[2020])/2*145
# A white halo separates the red arrow from both teal and amber bars.
for color,width in [('#FFFFFF',10),(INK,5)]:
    c.setStrokeColor(color);c.setLineWidth(width)
    c.line(1145,H-413,arrow_x,H-445)
    c.line(arrow_x,H-445,arrow_x+16,H-431)
    c.line(arrow_x,H-445,arrow_x+20,H-450)
text(c, 1000, 366, '2020-start training:', 18, 'Bold', INK)
text(c, 1000, 389, f"+{window[2020]-window[2013]:.3f} pp average error", 17, 'Bold', INK)
text(c, 1000, 410, 'versus the full window', 17, 'Bold', INK)
centered(c, 612, 'ALL THAT TRAINING. ZERO STILL WON.', 34, 'Heavy', INK)
crown(c, 1175, 603, 37)
centered(c, 645, f'Zero MAE: {zero_mae:.3f} pp. Lower than every neural-network fit shown.', 22, 'Bold')
centered(c, 686, 'So what if we keep the history, but let the forecast know the quarter?', 23, 'Bold', INK)
text(c, 90, 715, 'Seed 42 shown. Both shorter windows increased MAE across seeds 42, 7, 99; Ridge too. Reused 2024 data; shorter windows also have fewer examples.', 12, color=MUTED)
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
    if i in (5, 6):
        revision = BytesIO()
        revised = canvas.Canvas(revision, pagesize=(W,H))
        revised.setFillColor(CARD)
        if i == 5:
            revised.rect(75,H-678,1190,113,fill=1,stroke=0)
            text(revised,90,591,'SMALL PERCENTAGE GAP. REAL DOLLAR SCALE.',27,'Heavy')
            text(revised,90,625,'On a $1 billion deposit base: 0.026 pp = $260,000; 0.289 pp = $2.89 million.',23,'Bold')
            text(revised,90,657,'MAE gap (zero vs MLP) · RMSE gap (zero vs MLP). Scale illustration, not measured savings.',19)
        else:
            revised.rect(80,H-190,1190,91,fill=1,stroke=0)
            text(revised,90,140,'PREDICTING 0% TIES EVERY BANK.',32,'Heavy')
            text(revised,90,180,'BUT, THE LEARNED MODELS TELL ME WHO TO REVIEW FIRST.',28,'Heavy',CYAN)
        revised.save()
        original.merge_page(PdfReader(revision).pages[0])
    winner_positions = {0: [(1185,458)], 5: [(561,272),(1168,272)], 6: [(1195,477)]}
    if i in winner_positions:
        marks=BytesIO()
        overlay_canvas=canvas.Canvas(marks,pagesize=(W,H))
        for cx,cy in winner_positions[i]:
            crown(overlay_canvas,cx,cy,27)
        overlay_canvas.save()
        original.merge_page(PdfReader(marks).pages[0])
    writer.add_page(original)
writer.add_metadata({'/Title': 'Simple Models Fight Back', '/Subject': 'Deposit forecasts and unusual financial histories'})
with OUTPUT.open('wb') as f:
    writer.write(f)
assert len(PdfReader(OUTPUT).pages) == 13
print(OUTPUT)
