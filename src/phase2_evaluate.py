"""Report official-qrel coverage; score only a defensible multi-positive sample."""
import json
from collections import Counter, defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1]
docs = [json.loads(x) for x in (root/'data/processed/cord19_sample.jsonl').read_text(encoding='utf8').splitlines()]
ids = {d['doc_id'] for d in docs if d['doc_id']}
qrels = defaultdict(dict)
for line in (root/'data/evaluation/trec_covid_complete_qrels.txt').read_text().splitlines():
    topic, iteration, uid, grade = line.split()
    qrels[topic][uid] = int(grade)
per_query = {}
for topic, judgments in sorted(qrels.items(),key=lambda x:int(x[0])):
    rel = {uid for uid,g in judgments.items() if g>0}
    per_query[topic] = {'official_relevant':len(rel),'sample_relevant':len(rel & ids),
                        'sample_judged':len(set(judgments)&ids)}
summary = {'sample_unique_docs':len(ids),'official_judged_unique_docs':len({uid for j in qrels.values() for uid in j}),
           'matched_judged_docs':len(ids & {uid for j in qrels.values() for uid in j}),
           'official_relevant_pairs':sum(sum(g>0 for g in j.values()) for j in qrels.values()),
           'matched_relevant_pairs':sum(x['sample_relevant'] for x in per_query.values()),
           'queries_with_sample_positive':sum(x['sample_relevant']>0 for x in per_query.values()),
           'per_query':per_query,
           'quality_metrics':'not_computed: sample has insufficient relevance coverage'}
path=root/'data/evaluation/sample_coverage.json'
path.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps({k:v for k,v in summary.items() if k!='per_query'},indent=2))
