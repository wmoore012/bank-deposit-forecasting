"""Publication fallbacks retain literal chart values and readable labels."""
import unittest
from export_static_notebooks import figures_in_html, readable_card_spacing


class StaticPreviewTests(unittest.TestCase):
    def test_chart_arguments_remain_literal(self):
        html = 'Plotly.newPlot("chart", [{"x":[1,2],"y":[3,4]}], {"title":{"text":"Values"}}, {});'
        chart_id, figure = next(figures_in_html(html))
        self.assertEqual(chart_id, 'chart')
        self.assertEqual(list(figure.data[0].y), [3, 4])
        self.assertEqual(figure.layout.title.text, 'Values')

    def test_labels_do_not_require_css_spacing(self):
        html = '<section class="wm-micro-rail"><span>Mean</span><span>12.26</span></section>'
        result = readable_card_spacing(html)
        self.assertIn('</span> <span>', result)
        self.assertEqual(readable_card_spacing(result), result)
