"""Audit every positive qrel UID against original CORD-19 metadata rows."""
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'data/evaluation'
META=ROOT/'data/raw/cord19_2020-07-16/phase26/metadata.csv'
positive=set();judged=set()
for line in (EVAL/'trec_covid_complete_qrels.txt').read_text().splitlines():
    q,iteration,uid,grade=line.split();judged.add(uid)
    if int(grade)>0:positive.add(uid)
rows=defaultdict(list);path_to_uids=defaultdict(set)
with META.open(encoding='utf-8-sig',newline='') as f:
    for row in csv.DictReader(f):
        uid=row.get('cord_uid','')
        if uid not in positive:continue
        rows[uid].append(row)
        for col in ('pmc_json_files','pdf_json_files'):
            for path in (row.get(col) or '').split(';'):
                path=path.strip()
                if path:path_to_uids[path].add(uid)
uids={}
for uid in sorted(positive):
    rr=rows.get(uid,[])
    pmc=sorted({p.strip() for r in rr for p in (r.get('pmc_json_files') or '').split(';') if p.strip()})
    pdf=sorted({p.strip() for r in rr for p in (r.get('pdf_json_files') or '').split(';') if p.strip()})
    uids[uid]={'pmc':pmc,'pdf':pdf,'doi':next((r['doi'] for r in rr if r.get('doi')),None),
               'title':next((r['title'] for r in rr if r.get('title')),None),
               'abstract':next((r['abstract'] for r in rr if r.get('abstract')),None),
               'license':next((r['license'] for r in rr if r.get('license')),None),
               'metadata_rows':len(rr)}
old_map=json.loads((EVAL/'phase25_positive_metadata.json').read_text(encoding='utf8'))['mappings']
old_uids={v['doc_id'] for v in old_map.values()}
anomalies=sorted(uid for uid,rec in uids.items() if (rec['pmc'] or rec['pdf']) and uid not in old_uids)
classes=Counter('both' if x['pmc'] and x['pdf'] else 'pmc_only' if x['pmc'] else 'pdf_only' if x['pdf'] else 'none' for x in uids.values())
summary={'positive_unique':len(positive),'metadata_matched':sum(bool(rows.get(uid)) for uid in positive),
         'classes_mutually_exclusive':dict(classes),'parse_path_unique_docs':sum(bool(x['pmc'] or x['pdf']) for x in uids.values()),
         'shared_paths':{path:sorted(ids) for path,ids in path_to_uids.items() if len(ids)>1},
         'anomaly_ids':anomalies,'anomaly_original_rows':{uid:rows[uid] for uid in anomalies},
         'path_to_uids':{path:sorted(ids) for path,ids in path_to_uids.items()}}
(EVAL/'phase26_metadata_audit.json').write_text(json.dumps({k:v for k,v in summary.items() if k!='path_to_uids'},ensure_ascii=False,indent=2),encoding='utf8')
(EVAL/'phase26_positive_records.json').write_text(json.dumps(uids,ensure_ascii=False),encoding='utf8')
(EVAL/'phase26_path_to_uids.json').write_text(json.dumps(summary['path_to_uids'],ensure_ascii=False),encoding='utf8')
print(json.dumps({k:v for k,v in summary.items() if k not in ('path_to_uids','anomaly_original_rows','shared_paths')},indent=2))
print('shared_path_count',len(summary['shared_paths']))
