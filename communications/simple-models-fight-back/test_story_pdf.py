import unittest
from pathlib import Path
from pypdf import PdfReader

PDF = Path(__file__).with_name('simple_models_fight_back_data_science_story_v6.pdf')

class StoryEvidenceTests(unittest.TestCase):
    def test_mini_plots_show_identified_observations_and_saved_scores(self):
        page = PdfReader(PDF).pages[8].extract_text()
        self.assertIn('1. RANK ALL 4,548 BANKS', page)
        self.assertIn('4,548 BANKS', page)
        self.assertIn('PCA reconstruction error', page)
        self.assertNotIn('Schematic', page)
        self.assertNotIn('Illustrative mini-plots', page)

    def test_review_cut_matches_saved_population(self):
        import csv
        import math
        root = PDF.parents[2]
        with (root / 'experiments/review_history/outputs/scores.csv').open() as source:
            rows = list(csv.DictReader(source))
        march = [r for r in rows if r['date'] == '2024-03-31']
        selected = math.ceil(len(march) * .1)
        page = PdfReader(PDF).pages[8].extract_text()
        self.assertIn(f'{selected} banks make the cut.', page)
        self.assertEqual(max(march, key=lambda r: float(r['PCA']))['CERT'], '27330')
        self.assertIn('SILVERGATE: THE HIGHEST ERROR', page)

    def test_review_page_has_one_scoring_direction(self):
        page = PdfReader(PDF).pages[7].extract_text()
        self.assertIn('HIGHER IS BETTER', page)
        self.assertNotIn('LOWER IS BETTER', page)
        self.assertIn('ZERO GIVES EVERY BANK THE SAME ANSWER.', page)
        self.assertIn('IT CANNOT TELL ME WHOM TO REVIEW FIRST.', page)

    def test_panels_share_title_and_unit_baselines(self):
        import pdfplumber
        with pdfplumber.open(PDF) as pdf:
            page = pdf.pages[8]
            left = page.search('1. RANK ALL 4,548 BANKS')[0]
            right = page.search('2. REVIEW THE TOP 10%')[0]
            self.assertAlmostEqual(left['top'], right['top'], delta=0.2)
            lead = page.search("WHY NOT TRY ANOMALY DETECTION?")[0]
            self.assertAlmostEqual(left['x0'], lead['x0'], delta=0.2)
            a = page.search('March 2024 · PCA')[0]
            b = page.search('455 banks ·')[0]
            self.assertAlmostEqual(a['top'], b['top'], delta=0.2)

    def test_zero_remains_on_forecast_error_chart(self):
        page = PdfReader(PDF).pages[10].extract_text()
        self.assertIn('Zero MAE: 3.601 pp', page)
        self.assertIn('+0.694 pp', page)

if __name__ == '__main__':
    unittest.main()
