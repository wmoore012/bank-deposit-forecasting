"""Build shareable notebook bundles from reviewed, explicit file selections."""
import argparse
import re
from pathlib import Path
import zipfile
import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parent
COMMON = [
    'README.md', 'VALIDATION.md', 'export_static_notebooks.py', 'review_history.py', 'build_review_history.py',
    'FDIC_Review_History.ipynb', 'experiments/review_history/run.py',
    'experiments/review_history/README.md', 'sources/failure_coverage/audit.py', 'sources/failure_coverage/README.md', 'MASTERCLASS_README.md', 'PROFESSOR_README.md',
    'pyproject.toml', 'uv.lock', 'deposit_experiment.py', 'freeze_forecasts.py', 'build_masterclass.py',
    'notebook_lessons.py', 'concrete_teaching.py', 'export_story.py',
    'benchmark_timesfm.py', 'benchmark_chronos.py', 'finetune_foundation.py',
    'prepare_finetune_data.py', 'build_professor_submission.py', 'execute_masterclass.py',
    'download_fdic.py', 'download_supplement.py', 'package_project.py', 'Assign1.ipynb',
    'communications/simple-models-fight-back/README.md',
    'communications/simple-models-fight-back/manifest.json',
    'communications/simple-models-fight-back/prospective_forecast_2026-09-22.json',
    'communications/simple-models-fight-back/ridge_review_queue_2024-03-31.csv',
    'communications/simple-models-fight-back/training_histogram.json',
    'communications/simple-models-fight-back/young_americans_deposits_2025_2026.json',
    'communications/simple-models-fight-back/young_americans_deposits_full_history.json',
    'communications/simple-models-fight-back/simple_models_fight_back_data_science_story_v5.html',
    'communications/simple-models-fight-back/simple_models_fight_back_data_science_story_v5.pdf',
]
# Patterns are intentionally narrow: authoring notes, archives and QA are not inputs.
PATTERNS = [
    'experiments/review_history/outputs/*.csv', 'experiments/review_history/outputs/*.json',
    'experiments/review_history/outputs/models/*.joblib', 'experiments/review_history/outputs/models/*.keras',
    'sources/failure_coverage/*.csv', 'sources/failure_coverage/*.json',
    'data/*.csv', 'vendor/*.whl', 'vendor/*.patch', 'sources/*.md',
    'sources/*.json', 'sources/*.csv', 'sources/*.yaml',
    'notebooks/source/*.py', 'tests/test_*.py', 'growth_outputs/*.csv',
    'growth_outputs/*.json', 'growth_outputs/frozen_forecasts/*.json',
    'growth_outputs/frozen_forecasts/*.csv', 'growth_outputs/frozen_forecasts/*.joblib',
    'growth_outputs/frozen_forecasts/*.keras', 'growth_outputs/masterclass/*.csv',
    'growth_outputs/masterclass/*.json', 'growth_outputs/submission/*.csv',
    'growth_outputs/submission/*.json', 'experiments/experiment_0/*.py',
    'experiments/experiment_0/*.ipynb', 'experiments/experiment_0/README.md',
    'experiments/experiment_0/outputs/*.csv', 'experiments/experiment_0/outputs/*.json',
    'experiments/training_window_sensitivity/run.py',
    'experiments/training_window_sensitivity/REPORT.md',
    'experiments/training_window_sensitivity/*.csv',
    'experiments/training_window_sensitivity/*.json',
]


def offline_html(html):
    """Keep embedded libraries and remove redundant network script loaders."""
    html = re.sub(r'<script[^>]+src=[\"\']https?://[^\"\'<>]+[\"\'][^>]*>\s*</script>', '', html)
    return re.sub(r'<script[^>]*>\s*import\s+[\"\']https://cdn\.plot\.ly/[^\"\']+[\"\'];?\s*</script>', '', html)


def selected_files(root):
    paths = {root / name for name in COMMON}
    for pattern in PATTERNS:
        matches = list(root.glob(pattern))
        if not matches:
            raise FileNotFoundError(f'Package pattern matched no files: {pattern}')
        paths.update(matches)
    for edition in ['Masterclass', 'Submission']:
        paths.add(root / f'FDIC_Deep_Learning_{edition}.ipynb')
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted(paths)


def build_packages(root, output):
    output.mkdir(parents=True, exist_ok=True)
    paths = selected_files(root)
    rendered = {}
    for filename in ['FDIC_Deep_Learning_Masterclass.ipynb', 'FDIC_Deep_Learning_Submission.ipynb', 'FDIC_Review_History.ipynb']:
        notebook = root / filename
        nb = nbformat.read(notebook, as_version=4)
        code = [c for c in nb.cells if c.cell_type == 'code']
        if any(c.execution_count is None or any(o.output_type == 'error' for o in c.outputs) for c in code):
            raise ValueError(f'Execute all cells successfully before packaging: {notebook.name}')
        body, _ = HTMLExporter().from_notebook_node(nb)
        body = offline_html(body)
        html = notebook.with_suffix('.html').name
        (output / html).write_text(body)
        rendered[html] = body
    for edition in ['Masterclass', 'Submission']:
        name = f'FDIC_Deep_Learning_{edition}'
        with zipfile.ZipFile(output / f'{name}.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, Path(name) / path.relative_to(root))
            for html, body in rendered.items():
                archive.writestr(str(Path(name) / html), body)
            assert not any(any(part in n for part in ('.webapp-tester', '.internal', '.local-archive', 'career/', '.venv')) for n in archive.namelist())
        print(name, 'HTML and ZIP built')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    build_packages(ROOT, args.output)
