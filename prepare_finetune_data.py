"""Export exact notebook eligibility and chronological splits for local fine-tuning."""
import contextlib, io
from pathlib import Path
import nbformat
nb=nbformat.read('FDIC_Deep_Learning_Masterclass.ipynb',as_version=4)
Path('.finetune').mkdir(exist_ok=True)
ns={}
with contextlib.redirect_stdout(io.StringIO()):
    for cell in nb.cells:
        if cell.cell_type=='code':
            exec(cell.source,ns)
            if 'train' in ns and 'valid' in ns and 'holdout' in ns:
                break
for name in ['train','valid','holdout']:
    ns[name][['CERT','date','DEPDOM','target_date','next_deposits','growth']].to_csv(f'.finetune/{name}.csv',index=False)
    print(name,len(ns[name]))
