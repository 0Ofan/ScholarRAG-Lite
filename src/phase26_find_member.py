"""Look up an archived parse by cord_uid or exact member path."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'data/evaluation'
parser=argparse.ArgumentParser()
parser.add_argument('identifier',help='cord_uid or document_parses/... JSON member path')
args=parser.parse_args()
records=json.loads((EVAL/'phase26_judged_records.json').read_text(encoding='utf8'))
index=json.loads((EVAL/'phase26_judged_path_index.json').read_text(encoding='utf8'))
paths=json.loads((EVAL/'phase26_archive_paths.json').read_text(encoding='utf8'))
if args.identifier in records:
    r=records[args.identifier]
    selected=r['pmc']+r['pdf']
else:selected=[args.identifier]
for path in selected:
    if path not in index:raise SystemExit(f'not indexed: {path}')
    raw=ROOT/'data/raw/cord19_2020-07-16/phase26/parses'/path
    print(json.dumps({'archive_member':path,'cord_uids':index[path],
                      'archive_status':paths[path]['status'],'paper_id':paths[path].get('paper_id'),
                      'local_file_exists':raw.is_file(),'local_path':str(raw)},ensure_ascii=False))
