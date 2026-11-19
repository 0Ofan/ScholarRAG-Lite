"""Audit qrel UID coverage against same-snapshot CORD-19 metadata."""
import csv
import io
import json
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'data/evaluation'
qrels={}
for line in (EVAL/'trec_covid_complete_qrels.txt').read_text().splitlines():
    q,it,uid,grade=line.split();qrels[(q,uid)]=int(grade)
all_ids={uid for q,uid in qrels}
positive_ids={uid for (q,uid),g in qrels.items() if g>0}
metadata_ids=set();parse_ids=set();doi_ids=set();duplicate_rows=0
url='https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-07-16/metadata.csv'
with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (ScholarRAG Lite research)'}),timeout=120) as response:
    for row in csv.DictReader(io.TextIOWrapper(response,encoding='utf-8-sig',newline='')):
        uid=row.get('cord_uid','')
        if uid not in all_ids: continue
        if uid in metadata_ids: duplicate_rows+=1
        metadata_ids.add(uid)
        if row.get('pdf_json_files') or row.get('pmc_json_files'):parse_ids.add(uid)
        if row.get('doi'):doi_ids.add(uid)
sample_ids={p.stem for p in (ROOT/'data/raw/cord19_2020-07-16/phase25').glob('*.json')}
stats={'official_judged_unique_ids':len(all_ids),'metadata_mapped_judged_ids':len(metadata_ids),
       'metadata_mapped_positive_ids':len(metadata_ids&positive_ids),'positive_unique_ids':len(positive_ids),
       'parse_available_judged_ids':len(parse_ids),'parse_available_positive_ids':len(parse_ids&positive_ids),
       'doi_available_judged_ids':len(doi_ids),'duplicate_metadata_rows_for_judged_ids':duplicate_rows,
       'sample_fulltext_positive_ids':len(sample_ids&positive_ids),
       'judged_metadata_missing_ids':sorted(all_ids-metadata_ids),
       'positive_metadata_missing_ids':sorted(positive_ids-metadata_ids),
       'positive_parse_missing_ids':len(positive_ids-parse_ids)}
(EVAL/'phase25_id_audit.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in stats.items() if not isinstance(v,list)},indent=2))
