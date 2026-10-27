"""Read-only network availability probes; no credentials or paid API calls."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import requests

OUT=Path('experiments/portfolio_gp_macro_news_2026-10-18')
URLS={
 'fred_daily':'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2&cosd=2019-01-01&coed=2026-10-16',
 'alfred_vintage':'https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=CPIAUCSL&cosd=2019-01-01&coed=2019-06-30&vintage_date=2019-06-30',
 'fed_archive':'https://www.federalreserve.gov/monetarypolicy/fomchistorical2020.htm',
 'bls_archive':'https://www.bls.gov/news.release/archives/cpi_06122019.htm',
 'gdelt_bulk':'https://data.gdeltproject.org/events/20190201.export.CSV.zip',
 'gdelt_doc':'https://api.gdeltproject.org/api/v2/doc/doc?query=inflation&mode=artlist&format=json&startdatetime=20190601000000&enddatetime=20190602000000&maxrecords=5'
}
def probe(item):
 name,url=item
 try:
  r=requests.get(url,timeout=35)
  (OUT/'logs'/f'probe_{name}.bin').write_bytes(r.content)
  return dict(name=name,url=url,status=r.status_code,bytes=len(r.content),preview=r.text[:120] if name!='gdelt_bulk' else 'zip')
 except requests.RequestException as e: return dict(name=name,url=url,error=str(e))
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=6) as pool: results=list(pool.map(probe,URLS.items()))
 (OUT/'logs'/'source_probes.json').write_text(json.dumps(results,indent=2))
 print(json.dumps(results,indent=2))
