"""Embed static Plotly fallbacks while preserving each interactive notebook output.

Optional publication tool: run with nbformat, Plotly 7.1.0, Kaleido 1.4.0 and
Chrome installed. Modeling and notebook execution do not require Kaleido.
"""

import argparse
import base64
import json
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
            if html:
                html = readable_card_spacing(html)
                output.data["text/html"] = html
            if "data-static-fallback" in html:
                continue
            for chart_id, figure in figures_in_html(html):
                pending.append((output, chart_id, figure))
    if not pending:
        print(path.name, "already has fallbacks or contains no charts")
        nbformat.write(notebook, path)
        return
    with tempfile.TemporaryDirectory(prefix="notebook-charts-") as directory:
        images = [Path(directory) / f"{i}.png" for i in range(len(pending))]
        pio.write_images(fig=[entry[2] for entry in pending], file=images, format="png", scale=1)
        for (output, chart_id, figure), image in zip(pending, images):
            encoded = base64.b64encode(image.read_bytes()).decode()
            image_id = chart_id + "-static"
            fallback = (
                f'<img id="{image_id}" data-static-fallback="true" '
                f'alt="Static chart preview" src="data:image/png;base64,{encoded}" '
                'style="width:100%;height:auto;display:block"/>'
            )
            html = output.data["text/html"]
            opening = re.compile(r'(<div\b[^>]*\bid="' + re.escape(chart_id) + r'"[^>]*>)')
            html, replacements = opening.subn(
                lambda match: match.group(1) + fallback, html, count=1
            )
            if replacements != 1:
                raise ValueError(f"Cannot locate chart container {chart_id}")
            # GitHub strips scripts and retains the image. Trusted notebooks and
            # offline HTML hide it when their bundled interactive library exists.
            hide = (
                f'<script>if(window.Plotly){{var img=document.getElementById("{image_id}");'
                'if(img)img.style.display="none";}</script>'
            )
            output.data["text/html"] = html + hide
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(path.name, len(pending), "static chart fallbacks embedded", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.notebooks:
        add_static_fallbacks(path)
