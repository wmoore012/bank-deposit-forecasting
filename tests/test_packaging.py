"""A shareable package contains runnable inputs and no internal working history."""
import unittest
from pathlib import Path
import package_project

ROOT = Path(__file__).resolve().parents[1]


class PackageTests(unittest.TestCase):
    def test_reviewed_file_selection(self):
        names = {p.relative_to(ROOT).as_posix() for p in package_project.selected_files(ROOT)}
        for required in ['deposit_experiment.py', 'notebook_lessons.py',
                         'experiments/training_window_sensitivity/run.py',
                         'experiments/training_window_sensitivity/scores.csv',
                         'notebooks/source/training_window_lesson.py',
                         'vendor/wm_notecards-0.1.0-py3-none-any.whl',
                         'data/fdic_financials_2013_2024.csv']:
            self.assertIn(required, names)
        for name in names:
            self.assertFalse(any(part in name for part in ['.internal', '.local-archive', '.webapp-tester',
                                                          'career/', 'COMMUNICATION_PLAN', 'CHART_REVIEW',
                                                          'scratch.ipynb', 'takeover.ipynb', '__pycache__']))

    def test_offline_plotly_keeps_inline_library(self):
        source = '<script>window.Plotly = bundled;</script><script type="module">import "https://cdn.plot.ly/plotly-4.1.1.min"</script><script src="https://cdn.plot.ly/plotly.min.js"></script>'
        cleaned = package_project.offline_html(source)
        self.assertEqual(cleaned, '<script>window.Plotly = bundled;</script>')
