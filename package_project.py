"""Export readable HTML and reproducible ZIPs from an explicit allowlist."""
from pathlib import Path
import re, zipfile
import nbformat
from nbconvert import HTMLExporter
ROOT=Path(__file__).resolve().parent
COMMON=['VALIDATION.md','README.md','MASTERCLASS_README.md','PROFESSOR_README.md','pyproject.toml','uv.lock',
        'build_masterclass.py','benchmark_timesfm.py','benchmark_chronos.py','finetune_foundation.py','prepare_finetune_data.py','build_professor_submission.py','execute_masterclass.py',
        'download_fdic.py','download_supplement.py','package_project.py','Assign1.ipynb']
notebooks=[ROOT/f'FDIC_Deep_Learning_{edition}.ipynb' for edition in ['Masterclass','Submission']]
for notebook in notebooks:
    body,_=HTMLExporter().from_notebook_node(nbformat.read(notebook,as_version=4))
    # Equations already render inside WM cards; Plotly is embedded locally.
    body=re.sub(r'<script[^>]+src="https?://[^"<>]+"[^>]*>\s*</script>','',body)
    notebook.with_suffix('.html').write_text(body)
paths=[ROOT/p for p in COMMON]+notebooks+[p.with_suffix('.html') for p in notebooks]
for folder in ['data','vendor','sources','experiments/experiment_0','notebooks/source','tests','growth_outputs']:
    paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and p.name!='.DS_Store' and 'charts' not in p.parts)
for edition in ['Masterclass','Submission']:
    name=f'FDIC_Deep_Learning_{edition}'
    with zipfile.ZipFile(ROOT/f'{name}.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,Path(name)/p.relative_to(ROOT))
        assert not any('.webapp-tester' in p or '.venv' in p for p in z.namelist())
    print(name,'HTML and ZIP built')
