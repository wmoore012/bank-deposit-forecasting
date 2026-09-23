"""Embed static Plotly fallbacks while preserving each interactive notebook output.

Optional publication tool: run with nbformat, Plotly 7.1.0, Kaleido 1.4.0 and
Chrome installed. Modeling and notebook execution do not require Kaleido.
"""

import argparse
import base64
import json
from html import escape
from pathlib import Path
import re
import tempfile
import nbformat
import plotly.graph_objects as go
import plotly.io as pio


PLOT_CALL = re.compile(r'Plotly\.newPlot\(\s*"([^"\s]+)"\s*,\s*')


def figures_in_html(html):
    """Read literal chart arguments from saved Plotly output; never execute HTML."""
    decoder = json.JSONDecoder()
    for match in PLOT_CALL.finditer(html):
        data, consumed = decoder.raw_decode(html[match.end() :])
        remaining = html[match.end() + consumed :].lstrip()
        if not remaining.startswith(","):
            raise ValueError("Expected a literal Plotly layout after chart data")
        layout, _ = decoder.raw_decode(remaining[1:].lstrip())
        yield match.group(1), go.Figure(data=data, layout=layout)


def readable_card_spacing(html):
    """Keep labels separate even when a viewer removes the card styles."""
    if not any(name in html for name in ("wm-data-chip-board", "wm-micro-rail", "wm-dtype")):
        return html
    # Only alter markup outside scripts/styles; chart data and CSS stay literal.
    parts = re.split(r"(<(?:script|style)\b[^>]*>.*?</(?:script|style)>)", html, flags=re.S)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"</span>(?!\s)", "</span> ", parts[i])
    return "".join(parts)


def add_static_fallbacks(path):
    notebook = nbformat.read(path, as_version=4)
    pending = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            html = output.get("data", {}).get("text/html", "")
            if not html:
                continue
            html = readable_card_spacing(html)
            output.data["text/html"] = html
            if 'data-static-version="3"' in html:
                continue
            if 'data-static-version="2"' in html:
                html, _ = json.JSONDecoder().raw_decode(html.split('host.innerHTML=', 1)[1])
            # Upgrade the first publication format without nesting its old image.
            html = re.sub(r'<img\b[^>]*data-static-fallback="true"[^>]*>', "", html)
            html = re.sub(r"<script>if\(window.Plotly\)\{var img=.*?</script>", "", html)
            figures = list(figures_in_html(html))
            if len(figures) > 1:
                raise ValueError("Expected one chart per notebook display output")
            for chart_id, figure in figures:
                figure.update_layout(width=900)
                figure.update_layout(title_y=0.96, title_pad_t=0)
                pending.append((output, chart_id, figure, html))
    with tempfile.TemporaryDirectory(prefix="notebook-charts-") as directory:
        images = [Path(directory) / f"{i}.png" for i in range(len(pending))]
        if images:
            pio.write_images(
                fig=[entry[2] for entry in pending], file=images, format="png", scale=2
            )
        for (output, chart_id, figure, html), image in zip(pending, images):
            encoded = base64.b64encode(image.read_bytes()).decode()
            host_id = chart_id + "-preview"
            title = re.sub(r"<[^>]+>", " ", figure.layout.title.text or "Chart")
            fallback = (
                f'<div id="{host_id}" data-static-version="3">'
                f'<img alt="{escape(title, quote=True)}" width="900" '
                f'src="data:image/png;base64,{encoded}" '
                'style="max-width:100%;height:auto"/></div>'
            )
            # The static image has no fixed-height chart ancestors. Trusted HTML
            # replaces it with the original interactive output; GitHub strips JS.
            payload = json.dumps(html).replace("</", "<\\/")
            interactive = (
                "<script>if(window.Plotly){"
                f'const host=document.getElementById("{host_id}");'
                f"host.innerHTML={payload};"
                'host.querySelectorAll("script").forEach(old=>{'
                'const script=document.createElement("script");'
                "script.textContent=old.textContent;old.replaceWith(script);});"
                "}</script>"
            )
            output.data["text/html"] = fallback + interactive
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(path.name, len(pending), "high-resolution chart previews embedded", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.notebooks:
        add_static_fallbacks(path)
