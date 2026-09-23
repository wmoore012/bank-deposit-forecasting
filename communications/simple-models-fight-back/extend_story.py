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


def report(c, x, y, label, active=True):
    c.setStrokeColor(TEAL if active else BORDER)
    c.setFillColor('#E9F5F5' if active else CARD)
    c.roundRect(x, H-y-52, 42, 52, 4, fill=1, stroke=1)
    c.setStrokeColor(TEAL if active else BORDER)
    for offset in [15, 24, 33]:
        c.line(x+8, H-y-offset, x+34, H-y-offset)
    text(c, x+5, y+70, label, 10, color=MUTED)


def review(c, x, y):
    for row in range(2):
        for col in range(5):
            c.setFillColor(CYAN if row == col == 0 else '#DADDDC')
            c.circle(x+col*17, H-y-row*17, 5, fill=1, stroke=0)
    text(c, x-4, y+49, 'Review 10%', 16, 'Bold', TEAL)


def arrow(c, x, y):
    c.setStrokeColor(MUTED)
    c.setLineWidth(2)
    c.line(x, H-y, x+33, H-y)
    c.line(x+25, H-y+5, x+33, H-y)
    c.line(x+25, H-y-5, x+33, H-y)


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
# Two equally sized routes make the information sets and capacity visible.
text(c, 95, 432, 'FORECAST', 16, 'Bold', TEAL)
text(c, 700, 432, 'ANOMALY DETECTION', 16, 'Bold', TEAL)
report(c, 105, 452, 'Now')
arrow(c, 170, 478)
text(c, 228, 474, 'Predicted next-quarter', 17, 'Bold')
text(c, 228, 500, 'deposit growth', 17, 'Bold')
arrow(c, 425, 478)
review(c, 494, 471)
for i in range(8):
    report(c, 700+i*45, 451, str(i+1))
text(c, 700, 549, 'Eight quarters → unusual-history score', 18, 'Bold')
arrow(c, 1076, 478)
review(c, 1138, 471)
text(c, 90, 598, 'But keep the original question in mind.', 20)
text(c, 90, 636, "We're looking for banks whose deposits will grow slowly or decline next quarter.", 23, 'Bold')
text(c, 90, 672, 'Would unusual histories help us find them?', 25, 'Bold', TEAL)
text(c, 90, 710, 'Shared population requires complete histories. The forecast includes prior-quarter growth; anomaly inputs span eight quarters.', 12, color=MUTED)
c.showPage()
base(c, 10)
text(c, 90, 143, 'THE ANOMALY MODELS CHANGED THE LIST.', 34, 'Heavy')
text(c, 90, 190, 'RIDGE STILL FOUND MORE LOW-GROWTH BANKS.', 32, 'Heavy', CYAN)
text(c, 90, 236, 'I checked what happened to deposits next quarter. Every method had the same review capacity.', 21)
text(c, 90, 278, 'Banks later in the lowest-growth 10%, out of 1,351 selections', 23, 'Bold')
text(c, 90, 306, 'Pooled bank-quarter selections across March, June, and September 2024. Higher is better.', 17, color=MUTED)
for i, (method, label) in enumerate(methods):
    y = 350+i*45
    value = int(pooled[method]['captured'])
    text(c, 105, y+5, label, 20, 'Bold' if i == 0 else 'Body')
    c.setFillColor(TEAL if i == 0 else '#496B9D' if i == 1 else '#9FB6BC')
    c.roundRect(400, H-y-6, value*2.1, 24, 5, fill=1, stroke=0)
    text(c, 414+value*2.1, y+5, str(value), 22, 'Bold')
text(c, 400, 576, '0', 13, color=MUTED)
text(c, 1015, 576, '300', 13, color=MUTED)
text(c, 90, 620, 'For this historical review task, I kept the Ridge forecast.', 27, 'Bold', TEAL)
text(c, 90, 655, 'Unusual histories gave us different priorities. They did not capture more low-growth cases here.', 20)
text(c, 90, 697, '13,490 complete histories with observed outcomes; lists recalculated at 10% within each quarter. Reused 2024 data, not a fresh test.', 12, color=MUTED)
text(c, 90, 716, 'Stochastic models: seed 42 shown. No other tested seed or version without deposit size exceeded Ridge. Anomalies do not establish distress.', 12, color=MUTED)
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
