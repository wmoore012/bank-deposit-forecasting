"""Public reporting-coverage evidence, kept separate from the modeling extract."""
import csv, json, time
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
FIELDS=['CERT','REPDTE','BKCLASS','CALLFORM','DEPUNA','DEPUNINS','ACTEVT']
ROOT=Path(__file__).resolve().parent

def fetch(q):
    url='https://api.fdic.gov/banks/financials?'+urlencode(dict(filters=f'REPDTE:{q}',fields=','.join(FIELDS),limit=10000))
    for attempt in range(3):
        try:
            data=json.load(urlopen(url,timeout=45));rows=[r['data'] for r in data['data']]
            assert len(rows)==data['meta']['total']
            assert all(str(r['REPDTE'])==q for r in rows)
            print(q,len(rows),flush=True);return rows
        except Exception:
            if attempt==2:raise
            time.sleep(2**attempt)
if __name__=='__main__':
    quarters=[f'{y}{m}' for y in range(2010,2025) for m in ['0331','0630','0930','1231']]
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows=[r for batch in pool.map(fetch,quarters) for r in batch]
    assert len({(r['CERT'],r['REPDTE']) for r in rows})==len(rows)
    p=ROOT/'data/fdic_reporting_2010_2024.csv'
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore');w.writeheader();w.writerows(rows)
