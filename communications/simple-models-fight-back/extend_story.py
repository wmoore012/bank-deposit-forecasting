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
text(c, 90, 575, 'FORECAST-ERROR TASK: MAE · LOWER IS BETTER', 19, 'Bold', '#496B9D')
# A separate score list keeps the constant forecast visible without inventing a rank.
scores = {r['Model']: float(r['MAE (pp)']) for r in csv.DictReader((ROOT/'growth_outputs/masterclass/extended_scores.csv').open())}
for x, model, label in [(90,'Zero growth','Predict 0%'),(370,'MLP','Neural network'),(650,'Ridge','Five-input Ridge'),(950,'Size-only Ridge','Deposit-size Ridge')]:
    text(c, x, 613, label, 19, 'Bold' if model == 'Zero growth' else 'Body')
    text(c, x, 647, f"{scores[model]:.3f} pp", 24, 'Bold', TEAL if model == 'Zero growth' else MUTED)
text(c, 90, 682, 'Zero has the lowest MAE here. It gives every bank the same forecast, so it cannot order a review list.', 19)
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
    text(c, 98, y, '•  '+bullet, 21)
# Both mini-plots are explicitly schematic, rather than invented empirical results.
text(c, 105, 420, 'FORECAST: predict the next value', 21, 'Bold', TEAL)
text(c, 720, 420, 'ANOMALY: find unusual observations', 21, 'Bold', TEAL)
text(c, 105, 444, 'Schematic deposit history', 14, color=MUTED)
text(c, 720, 444, 'Schematic feature-space view', 14, color=MUTED)
line(c, 130, 548, 595, 548, BORDER)
line(c, 130, 460, 130, 548, BORDER)
points = [(145,492),(205,480),(265,497),(325,486),(385,509),(445,503)]
for start, end in zip(points, points[1:]):
    line(c, *start, *end)
for x,y in points:
    c.setFillColor(TEAL); c.circle(x,H-y,4,fill=1,stroke=0)
line(c,445,503,565,533,CYAN,[4,4])
c.setFillColor(CYAN);c.circle(565,H-533,5,fill=1,stroke=0)
text(c, 365, 476, 'Observed', 15, color=TEAL)
text(c, 473, 494, 'Forecast', 16, 'Bold', CYAN)
text(c, 132, 569, 'Earlier quarters', 14, color=MUTED)
text(c, 422, 569, 'Now', 14, color=MUTED)
text(c, 533, 569, 'Next quarter', 14, color=MUTED)
line(c, 750, 548, 1220, 548, BORDER)
line(c, 750, 460, 750, 548, BORDER)
for i in range(38):
    angle = i*2.39996
    radius = 10+math.sqrt(i)*8
    x = 905+math.cos(angle)*radius*1.6
    y = 498+math.sin(angle)*radius*.65
    c.setFillColor('#9FB6BC');c.circle(x,H-y,3.5,fill=1,stroke=0)
for x,y in [(1110,466),(1170,521),(1070,537)]:
    c.setFillColor('#B16B24');c.circle(x,H-y,5,fill=1,stroke=0)
text(c, 855, 572, 'Common pattern', 15, color=MUTED)
text(c, 1080, 448, 'Unusual', 16, 'Bold', '#B16B24')
text(c, 100, 603, 'Select the 10% with lowest predicted growth.', 20, 'Bold', TEAL)
text(c, 715, 603, 'Select the 10% with highest anomaly scores.', 20, 'Bold', TEAL)
text(c, 90, 641, 'Same workload: choose 10 of every 100 banks. Then check their next-quarter growth.', 22)
text(c, 90, 681, 'Would choosing unusual histories find more low-growth banks?', 25, 'Bold')
text(c, 90, 712, 'Illustrative mini-plots, not measured model outputs or a claim that the detectors use clustering. Forecast inputs include prior growth.', 12, color=MUTED)
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
text(c, 90, 620, 'SCORECARD SO FAR', 17, 'Bold', MUTED)
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
