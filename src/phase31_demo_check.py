"""Exercise three real TREC-COVID queries through the CLI retriever API."""
import json
from pathlib import Path
from .cli import EXAMPLES
from .retriever import PilotRetriever

ROOT=Path(__file__).resolve().parents[1]
topics={x['topic_id']:x['query'] for x in map(json.loads,(ROOT/'data/evaluation/topics.jsonl').read_text(encoding='utf8').splitlines())}
assert EXAMPLES==[topics[q] for q in ('1','2','31')]
retriever=PilotRetriever();rows=[]
for q in EXAMPLES:
    docs,perf=retriever.retrieve(q,3)
    assert len(docs)==len({d['doc_id'] for d in docs})==3
    rows.append({'query':q,'top3':docs,'performance':perf})
    print(f'{q}: '+', '.join(d['doc_id'] for d in docs),flush=True)
out={'version':'phase31-cli-demo-v1','query_source':'TREC-COVID topics 1, 2, 31',
     'queries':rows,'all_top3_distinct':True,'score_type':'normalized vector inner product'}
path=ROOT/'experiments/results/phase31_pilot/cli_demo_results.json'
path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
