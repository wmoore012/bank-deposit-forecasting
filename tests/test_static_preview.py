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

    def test_preview_has_no_fixed_height_ancestors(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import nbformat
        from export_static_notebooks import add_static_fallbacks

        html = ('<div style="height:300px;overflow:hidden">'
                '<div id="chart"></div><script>'
                'Plotly.newPlot("chart", [{"x":[1],"y":[2]}], {}, {});'
                '</script></div>')
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
            'display(chart)', execution_count=1, outputs=[nbformat.v4.new_output(
                'display_data', data={'text/html': html})])])

        def render_images(*, fig, file, format, scale):
            self.assertEqual(scale, 2)
            for target in file:
                Path(target).write_bytes(b'publication image fixture')

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.ipynb'
            nbformat.write(notebook, path)
            with patch('export_static_notebooks.pio.write_images', render_images):
                add_static_fallbacks(path)
            output = nbformat.read(path, 4).cells[0].outputs[0].data['text/html']
            static_html = output.split('<script>', 1)[0]
            self.assertIn('width="900"', static_html)
            self.assertNotIn('overflow:hidden', static_html)
            self.assertNotIn('height:300px', static_html)
            self.assertIn('host.innerHTML=', output)
