import unittest
from pathlib import Path
from pypdf import PdfReader

PDF = Path(__file__).with_name('simple_models_fight_back_data_science_story_v6.pdf')

class StoryEvidenceTests(unittest.TestCase):
    def test_mini_plots_show_identified_observations_and_saved_scores(self):
        page = PdfReader(PDF).pages[8].extract_text()
        self.assertIn('Young Americans Bank', page)
        self.assertIn('4,548 banks', page)
        self.assertIn('PCA reconstruction error', page)
        self.assertNotIn('Schematic', page)
        self.assertNotIn('Illustrative mini-plots', page)

    def test_forecast_matches_saved_prediction(self):
        page = PdfReader(PDF).pages[8].extract_text()
        self.assertIn('Ridge: 15.766', page)
        self.assertIn('Zero: 16.426', page)

    def test_review_page_has_one_scoring_direction(self):
        page = PdfReader(PDF).pages[7].extract_text()
        self.assertIn('HIGHER IS BETTER', page)
        self.assertNotIn('LOWER IS BETTER', page)
        self.assertIn('EVERY BANK TIED', page)

    def test_panels_share_title_and_unit_baselines(self):
        import pdfplumber
        with pdfplumber.open(PDF) as pdf:
            page = pdf.pages[8]
            left = page.search('FORECAST: Young Americans Bank')[0]
            right = page.search('ANOMALY: actual March 2024 PCA scores')[0]
            self.assertAlmostEqual(left['top'], right['top'], delta=0.2)
            lead = page.search("I KNOW WHAT YOU'RE THINKING.")[0]
            self.assertAlmostEqual(left['x0'], lead['x0'], delta=0.2)
            a = page.search('Domestic deposits')[0]
            b = page.search('4,548 banks')[0]
            self.assertAlmostEqual(a['top'], b['top'], delta=0.2)

    def test_zero_remains_on_forecast_error_chart(self):
        page = PdfReader(PDF).pages[10].extract_text()
        self.assertIn('Zero MAE: 3.601 pp', page)
        self.assertIn('+0.694 pp', page)

if __name__ == '__main__':
    unittest.main()
