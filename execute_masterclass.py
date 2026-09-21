"""Execute either generated notebook using this interpreter; remove CDN dependencies."""
from pathlib import Path
import re, sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
root=Path(__file__).resolve().parent
path=root/(sys.argv[1] if len(sys.argv)>1 else 'FDIC_Deep_Learning_Masterclass.ipynb')
nb=nbformat.read(path,as_version=4)
manager=KernelManager(kernel_name='python3')
manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
def progress(cell,cell_index,**kwargs):
    if cell.cell_type=='code':print(f'Running cell {cell_index+1}: {cell.source.splitlines()[0]}',flush=True)
client=NotebookClient(nb,km=manager,timeout=1800,resources={'metadata':{'path':str(root)}},on_cell_start=progress)
client.execute()
# The bootstrap output already embeds Plotly. Avoid external script requests in saved outputs.
for cell in nb.cells:
    for output in cell.get('outputs',[]):
        data=output.get('data',{})
        if 'text/html' in data:
            data['text/html']=re.sub(r'<script[^>]+src="https://cdn.plot.ly/[^"<>]+"[^>]*></script>','',data['text/html'])
nbformat.validate(nb);nbformat.write(nb,path)
print('Saved successfully executed notebook:',path,flush=True)
