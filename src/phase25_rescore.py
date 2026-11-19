"""Rescore saved document rankings without regenerating embeddings."""
import json
import statistics
from pathlib import Path
from evaluation import score_topic

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/phase25'
manifest=json.loads((ROOT/'data/evaluation/phase25_eval_manifest.json').read_text(encoding='utf8'))
qrels={t['topic_id']:{} for t in manifest['topics']}
for line in (ROOT/'data/evaluation/trec_covid_complete_qrels.txt').read_text().splitlines():
    q,it,uid,grade=line.split()
    if q in qrels and int(grade)>=0:qrels[q][uid]=int(grade)
corpus=set(manifest['doc_ids'])
result=json.loads((OUT/'comparison.json').read_text(encoding='utf8'))
for name in ('bge','minilm'):
    rankings=json.loads((OUT/f'{name}_rankings.json').read_text(encoding='utf8'))
    per={q:score_topic(rank,qrels[q],corpus) for q,rank in rankings.items()}
    valid=[v for v in per.values() if v]
    result[name]['per_query']=per
    result[name]['mean_metrics']={key:statistics.mean(v[key] for v in valid) for key in ('ndcg@10','recall@10','recall@100','mrr')}
    result[name]['mean_unjudged_top10']=statistics.mean(v['unjudged_in_top10'] for v in valid)
(OUT/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps({name:result[name]['mean_metrics'] for name in ('bge','minilm')},indent=2))
