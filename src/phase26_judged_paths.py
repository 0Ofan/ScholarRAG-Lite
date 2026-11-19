"""Build a many-to-many metadata path index for every judged TREC document."""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'data/evaluation'
META=ROOT/'data/raw/cord19_2020-07-16/phase26/metadata.csv'
judged=set()
for line in (EVAL/'trec_covid_complete_qrels.txt').read_text().splitlines():judged.add(line.split()[2])
records=defaultdict(lambda:{'pmc':set(),'pdf':set(),'doi':set(),'title':set(),'abstract':set(),'license':set(),'rows':0})
paths=defaultdict(set)
with META.open(encoding='utf-8-sig',newline='') as f:
    for row in csv.DictReader(f):
        uid=row.get('cord_uid','')
        if uid not in judged:continue
        record=records[uid];record['rows']+=1
        for field in ('doi','title','abstract','license'):
            if row.get(field):record[field].add(row[field])
        for kind,column in (('pmc','pmc_json_files'),('pdf','pdf_json_files')):
            for path in (row.get(column) or '').split(';'):
                path=path.strip()
                if not path:continue
                if not path.startswith('document_parses/'+kind+'_json/') or '..' in Path(path).parts:
                    raise ValueError(f'unexpected parse path: {path}')
                record[kind].add(path);paths[path].add(uid)
normalized={uid:{k:(sorted(v) if isinstance(v,set) else v) for k,v in rec.items()} for uid,rec in records.items()}
(EVAL/'phase26_judged_records.json').write_text(json.dumps(normalized,ensure_ascii=False),encoding='utf8')
(EVAL/'phase26_judged_path_index.json').write_text(json.dumps({p:sorted(v) for p,v in paths.items()}),encoding='utf8')
print(json.dumps({'judged_ids':len(judged),'metadata_ids':len(records),'path_ids':sum(bool(x['pmc'] or x['pdf']) for x in normalized.values()),
                  'unique_paths':len(paths),'shared_paths':sum(len(v)>1 for v in paths.values())},indent=2))
