"""Cross-check frozen files, full audit rankings, and persisted FAISS mapping."""
import json
from collections import Counter
from pathlib import Path
import faiss
import numpy as np
from evaluation import score_topic

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'data/evaluation';OUT=ROOT/'experiments/results/phase27'
load=lambda p:json.loads(p.read_text(encoding='utf8'))
stats=load(EVAL/'phase27_statistics.json')
manifest=load(EVAL/'phase27_final_candidate_manifest.json')
audit=load(OUT/'top10_audit.json')
config=load(ROOT/'experiments/configs/phase27_experiment_config.json')
doc_ids=[d['doc_id'] for d in manifest['documents']]
assert len(doc_ids)==len(set(doc_ids))==stats['phase27_final_canonical_docs']==audit['documents']
assert len(config['query_ids'])==len(audit['per_query'])==50
doc_nums=np.load(OUT/'full_corpus_doc_numbers.npy',mmap_mode='r')
idx=faiss.read_index(str(OUT/'full_corpus_bge_fixed.faiss'),faiss.IO_FLAG_MMAP)
assert idx.ntotal==len(doc_nums)==audit['chunks']
assert int(doc_nums.max())<len(doc_ids)
official=Counter()
for line in (EVAL/'trec_covid_complete_qrels.txt').read_text().splitlines():
    q,_,uid,g=line.split()
    if int(g)>0:official[q]+=1
for q,x in audit['per_query'].items():
    assert x['official_positive']==official[q]==stats['per_query'][q]['official_positive']
    assert x['available_positive']==stats['per_query'][q]['final_canonical_positive']
    hits=x['top100_results'];ids=[h['doc_id'] for h in hits]
    assert len(ids)==len(set(ids))==100
    assert [h['rank'] for h in hits]==list(range(1,101))
    for h in hits:
        assert doc_ids[int(doc_nums[h['chunk_index']])]==h['doc_id']
        assert h['official_grade']==manifest['effective_qrels'][q].get(h['doc_id'])
    for n in (10,20,100):
        labels=[h['official_grade'] for h in hits[:n]]
        expected={'positive':sum(g is not None and g>0 for g in labels),
                  'grade_0':sum(g==0 for g in labels),
                  'unjudged':sum(g is None for g in labels)}
        assert all(x['top'][str(n)][k]==v for k,v in expected.items())
        assert sum(expected.values())==n
    metric=score_topic(ids,manifest['effective_qrels'][q],doc_ids)
    assert all(abs(metric[k]-x['metrics'][k])<1e-10 for k in ('ndcg@10','recall@10','recall@100','mrr'))
assert sum(x['available_positive'] for x in audit['per_query'].values())==stats['phase27_positive_pairs']
print(json.dumps({'documents':len(doc_ids),'chunks':idx.ntotal,'queries':len(audit['per_query']),
                  'positive_pairs':stats['phase27_positive_pairs'],'faiss_reload':'passed',
                  'rank_and_grade_checks':'passed'},indent=2))
