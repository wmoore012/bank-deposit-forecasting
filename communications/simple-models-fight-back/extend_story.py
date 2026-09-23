"""Add the approved Study 2 introduction and measured payoff to the v5 PDF.

Run with Python containing reportlab and pypdf. Existing pages stay vector-based.
The v5 PDF remains the source; this writes the 13-page v6 edition.
"""
import csv
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


def box(c, x, y, width, height, lines, color=TEAL):
    c.setFillColor('#E9F5F5')
    c.setStrokeColor(BORDER)
    c.roundRect(x, H-y-height, width, height, 10, fill=1, stroke=1)
    for i, line in enumerate(lines):
        text(c, x+18, y+29+i*27, line, 19, 'Bold' if i == 0 else 'Body', color if i == 0 else INK)




rows = list(csv.DictReader((ROOT / 'experiments/review_history/outputs/outcome_capture.csv').open()))
pooled = {r['method']: r for r in rows if r['date'] == 'All three quarters'}
methods = [('Ridge', 'Ridge forecast'), ('MLP', 'Original neural network'),
           ('Autoencoder 42', 'Dense autoencoder'), ('Isolation Forest 42', 'Isolation Forest'), ('PCA', 'PCA')]
assert all(int(pooled[m]['selected']) == 1351 and int(pooled[m]['eligible']) == 13490 for m, _ in methods)
assert [int(pooled[m]['captured']) for m, _ in methods] == [296, 293, 274, 237, 236]

new = BytesIO()
c = canvas.Canvas(new, pagesize=(W, H))
base(c, 9)
text(c, 90, 143, "I KNOW WHAT YOU'RE THINKING.", 35, 'Heavy')
text(c, 90, 190, 'WHY NOT JUST USE ANOMALY DETECTION?', 35, 'Heavy', CYAN)
text(c, 90, 237, "Find the unusual banks. Put them on the review list. Isn't that what we're looking for?", 22)
text(c, 90, 280, "Well, let's try it.", 25, 'Bold')
for y, line in zip([319, 351, 383], [
    'Eight quarters of the same five financial inputs.',
    'Three methods: PCA, Isolation Forest, and a small autoencoder.',
    'Same banks to choose from. Same 10% review capacity.',
]):
    text(c, 98, y, '•  '+line, 21)
# Explain the input, ranking rule, and action without decorative report icons.
box(c, 90, 412, 550, 148, [
    'FORECAST: who will have lower deposit growth?',
    'Use the five current-quarter inputs.',
    'Predict next-quarter growth for each bank.',
    'Choose the 10% with the lowest predictions.',
])
box(c, 670, 412, 580, 148, [
    'ANOMALY: whose history looks most unusual?',
    'Compare eight-quarter financial histories',
    'with patterns learned from training banks.',
    'Choose the 10% with the highest anomaly scores.',
])
text(c, 90, 593, '10% is an assumed workload: choose 10 of every 100 banks for a closer look.', 22, 'Bold', TEAL)
text(c, 90, 628, 'Then check how many chosen banks had the lowest deposit growth next quarter.', 22)
text(c, 90, 671, 'Would choosing unusual histories find more of those banks?', 26, 'Bold')
text(c, 90, 710, 'Historical comparison on shared eligible banks. Forecast inputs include prior growth. No real analyst reviews were measured.', 12, color=MUTED)
c.showPage()
base(c, 10)
text(c, 90, 143, 'ANOMALY DETECTION DID NOT BEAT RIDGE', 34, 'Heavy')
text(c, 90, 190, 'AT FINDING LOW-GROWTH BANKS HERE.', 34, 'Heavy', CYAN)
text(c, 90, 236, 'I checked what happened to deposits next quarter. Every method had the same review capacity.', 21)
text(c, 90, 278, 'Out of every 100 selected, how many landed in the lowest-growth 10% next quarter?', 21, 'Bold')
text(c, 90, 307, 'Same 1,351 selections per method, pooled across three 2024 quarters. Higher is better.', 18, color=MUTED)
# Rates answer the limited-review question directly; counts remain beside each bar.
for tick in [0, 5, 10, 15, 20, 25]:
    x = 400+tick*28
    c.setStrokeColor(BORDER)
    c.setLineWidth(0.5)
    c.line(x, H-330, x, H-550)
    text(c, x-5, 575, str(tick), 14, color=MUTED)
for i, (method, label) in enumerate(methods):
    y = 350+i*45
    value = int(pooled[method]['captured'])
    rate = 100*value/int(pooled[method]['selected'])
    text(c, 105, y+5, label, 20, 'Bold' if i == 0 else 'Body')
    c.setFillColor(TEAL if i == 0 else '#496B9D' if i == 1 else '#9FB6BC')
    c.roundRect(400, H-y-6, rate*28, 24, 5, fill=1, stroke=0)
    text(c, 413+rate*28, y+5, f'{rate:.1f}', 22, 'Bold')
    text(c, 1130, y+5, f'{value} / 1,351', 17, color=MUTED)
text(c, 105, 325, 'METHOD', 12, 'Bold', MUTED)
text(c, 1130, 325, 'EXACT COUNTS', 12, 'Bold', MUTED)
text(c, 90, 622, 'Same review budget. Ridge found about 22 per 100 selected.', 26, 'Bold', TEAL)
text(c, 90, 657, 'The primary autoencoder found 20.3 per 100. PCA and Isolation Forest each found about 17.5.', 21)
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
text(c, 90, 143, 'BUT WAIT. IN VOLATILE ECONOMIC TIMES,', 33, 'Heavy')
text(c, 90, 187, 'WHAT IF WE TRAIN ON ONLY THE RECENT YEARS?', 31, 'Heavy', CYAN)
text(c, 90, 232, 'Maybe older examples are less useful now. So I retrained starting in 2020, then 2021.', 22)
text(c, 90, 270, 'Same model settings. Same 2023 validation. Same 13,532 bank-quarters scored in 2024.', 20, 'Bold')
text(c, 105, 310, 'NEURAL NETWORK MAE (PERCENTAGE POINTS) · LOWER IS BETTER', 19, 'Bold')
for x, color, label in [(265, TEAL, 'Full-window error'), (640, '#B16B24', 'Additional error with shorter window')]:
    c.setFillColor(color)
    c.rect(x, H-344, 17, 17, fill=1, stroke=0)
    text(c, x+26, 342, label, 17)
for tick in range(6):
    y = 550-tick*35
    c.setStrokeColor(BORDER)
    c.setLineWidth(0.5)
    c.line(205, H-y, 1120, H-y)
    text(c, 168, y+5, str(tick), 15, color=MUTED)
for x, year in zip([310, 630, 950], [2013, 2020, 2021]):
    v = window[year]
    baseline = window[2013]
    c.setFillColor(TEAL)
    c.rect(x, H-550, 130, baseline*35, fill=1, stroke=0)
    c.setFillColor('#B16B24')
    c.rect(x, H-550+baseline*35, 130, (v-baseline)*35, fill=1, stroke=0)
    text(c, x+26, 550-v*35-13, f'{v:.3f}', 22, 'Bold')
    text(c, x-5, 576, f'{year}-Sep 2022', 18, 'Bold')
text(c, 355, 617, 'AND... THE ERRORS GOT BIGGER.', 30, 'Heavy', '#B16B24')
text(c, 155, 650, 'Both shorter windows had higher MAE with all three seeds. Ridge had higher MAE too.', 21)
text(c, 180, 683, 'So what if we keep the history, but let the forecast know the quarter?', 23, 'Bold', TEAL)
text(c, 90, 713, 'Seed 42 shown; also checked 7 and 99. Reused 2024 outcomes. Shorter windows also have fewer training examples; no regime effect is isolated.', 12, color=MUTED)
c.showPage()
c.save()

reader = PdfReader(SOURCE)
assert len(reader.pages) == 11
added = PdfReader(new)
writer = PdfWriter()
for i, original in enumerate(reader.pages):
    if i == 8:
        for p in added.pages:
            writer.add_page(p)
        continue  # The third generated page replaces the old training-window page.
    if i >= 8:
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
