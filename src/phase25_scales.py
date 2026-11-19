"""Freeze nested pilot/medium/large ID proposals with fixed positive labels."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'data/evaluation'
manifest=json.loads((EVAL/'phase25_eval_manifest.json').read_text(encoding='utf8'))
topics={t['topic_id'] for t in manifest['topics']}
pilot=set(manifest['doc_ids'])
old={json.loads(x)['doc_id'] for x in (ROOT/'data/processed/cord19_sample.jsonl').read_text(encoding='utf8').splitlines()}
new={p.stem for p in (ROOT/'data/raw/cord19_2020-07-16/phase25').glob('*.json')}
positive=set()
for line in (EVAL/'trec_covid_complete_qrels.txt').read_text().splitlines():
    q,it,uid,grade=line.split()
    if q in topics and int(grade)>0:positive.add(uid)
extra=sorted((old|new)-pilot-positive)
assert len(pilot)==58 and len(extra)>=142
scales={name:sorted(pilot|set(extra[:count-58])) for name,count in [('pilot',58),('medium',100),('large',200)]}
assert set(scales['pilot'])<=set(scales['medium'])<=set(scales['large'])
assert (set(scales['large'])-pilot).isdisjoint(positive)
output={'source_snapshot':'2020-07-16','topics':sorted(topics,key=int),
        'rule':'pilot frozen 58 IDs; add lexicographically smallest available full-text IDs with no positive qrel for any selected topic',
        'scales':scales,'available_fulltext_unique':len(old|new),'eligible_extra':len(extra)}
(EVAL/'phase25_scale_manifest.json').write_text(json.dumps(output,indent=2),encoding='utf8')
print(json.dumps({'available':len(old|new),'eligible_extra':len(extra),'counts':{k:len(v) for k,v in scales.items()}},indent=2))
