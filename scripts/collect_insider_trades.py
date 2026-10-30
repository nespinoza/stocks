"""Open-market insider purchases and sales from SEC's quarterly Form 345 data sets.

Source (free): https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets
One zip per quarter (~10 MB). Each quarter is downloaded, reduced to Form 4/4A
non-derivative trades with code P (open-market purchase) or S (open-market
sale), joined to the issuer and reporting owner, and appended to
data_store/insider/trades.csv.gz; the zip is then deleted.

The data sets carry the filing *date* only, not the acceptance time, so a
filing is treated as public at the close of the next session.

    SEC_USER_AGENT="<name> <email>" .conda-env/bin/python scripts/collect_insider_trades.py --first 2009q4
"""
import argparse
import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

from stock_api.news import sources

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ['accession_number', 'trans_date', 'code', 'shares', 'price', 'trans_acquired_disp_cd', 'shares_after',
           'direct_indirect_ownership', 'filing_date', 'document_type', 'issuer_cik', 'symbol', 'plan_10b5_1',
           'owner_cik', 'owners', 'relationship', 'title', 'quarter']
INDEX_URL = 'https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets'


def quarter_links(session):
    page = session.get(INDEX_URL, headers=sources.sec_headers(), timeout=60).text
    links = re.findall(r'href="([^"]*/(\d{4}q\d)_form345\.zip)"', page)
    return {quarter: 'https://www.sec.gov'+path if path.startswith('/') else path for path, quarter in links}


def read(archive, name, columns):
    with archive.open(name) as handle:
        return pd.read_csv(handle, sep='\t', usecols=columns, dtype=str, quoting=3, on_bad_lines='skip')


def reduce_quarter(blob):
    archive = zipfile.ZipFile(io.BytesIO(blob))
    trades = read(archive, 'NONDERIV_TRANS.tsv', ['ACCESSION_NUMBER', 'TRANS_DATE', 'TRANS_CODE', 'TRANS_SHARES',
                                                  'TRANS_PRICEPERSHARE', 'TRANS_ACQUIRED_DISP_CD',
                                                  'SHRS_OWND_FOLWNG_TRANS', 'DIRECT_INDIRECT_OWNERSHIP'])
    trades = trades[trades.TRANS_CODE.isin(['P', 'S'])]
    columns = ['ACCESSION_NUMBER', 'FILING_DATE', 'DOCUMENT_TYPE', 'ISSUERCIK', 'ISSUERTRADINGSYMBOL']
    submission = read(archive, 'SUBMISSION.tsv', None)
    submission = submission[[c for c in columns+['AFF10B5ONE'] if c in submission.columns]]
    submission = submission[submission.DOCUMENT_TYPE.isin(['4', '4/A'])]
    owners = read(archive, 'REPORTINGOWNER.tsv', ['ACCESSION_NUMBER', 'RPTOWNERCIK', 'RPTOWNER_RELATIONSHIP',
                                                  'RPTOWNER_TITLE'])
    # Joint filings list several owners; keep one row per filing with the union of roles.
    owners = owners.groupby('ACCESSION_NUMBER').agg(
        owner_cik=('RPTOWNERCIK', 'first'), owners=('RPTOWNERCIK', 'nunique'),
        relationship=('RPTOWNER_RELATIONSHIP', lambda s: ','.join(sorted({r for v in s.dropna() for r in v.split(',')}))),
        title=('RPTOWNER_TITLE', 'first'))
    out = trades.merge(submission, on='ACCESSION_NUMBER').join(owners, on='ACCESSION_NUMBER')
    out = out.rename(columns=str.lower).rename(columns={'issuercik': 'issuer_cik', 'issuertradingsymbol': 'symbol',
                                                       'trans_code': 'code', 'trans_shares': 'shares',
                                                       'trans_pricepershare': 'price', 'shrs_ownd_folwng_trans':
                                                       'shares_after', 'aff10b5one': 'plan_10b5_1'})
    for column in ('filing_date', 'trans_date'):
        out[column] = pd.to_datetime(out[column], format='%d-%b-%Y', errors='coerce').dt.date
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', default='2009q4')
    parser.add_argument('--last', default='')
    parser.add_argument('--out', default=str(ROOT/'data_store/insider/trades.csv.gz'))
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    session, throttle = requests.Session(), sources.Throttle(1.)
    links = quarter_links(session)
    done = set(pd.read_csv(out, usecols=['quarter']).quarter) if out.exists() else set()
    wanted = sorted(q for q in links if q >= args.first and (not args.last or q <= args.last) and q not in done)
    print(f'{len(links)} quarters published, {len(done)} already collected, fetching {len(wanted)}')
    for quarter in wanted:
        throttle.wait()
        response = session.get(links[quarter], headers=sources.sec_headers(), timeout=600)
        response.raise_for_status()
        # Older quarters lack some columns (e.g. AFF10B5ONE, from 2023); a fixed order keeps appends aligned.
        frame = reduce_quarter(response.content).assign(quarter=quarter).reindex(columns=COLUMNS)
        frame.to_csv(out, mode='a', header=not out.exists(), index=False, compression='gzip')
        print(f"{quarter}: {len(frame)} trades ({(frame.code == 'P').sum()} purchases)", flush=True)


if __name__ == '__main__':
    main()
