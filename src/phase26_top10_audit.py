"""One-model exploratory check of unjudged top-10 rate after pool expansion."""
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from phase25_compare import fixed_chunks
from evaluation import document_ranking

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'data/evaluation';OUT=ROOT/'experiments/results/phase26';OUT.mkdir(parents=True,exist_ok=True)
status=json.loads((EVAL/'phase26_doc_status.json').read_text(encoding='utf8'))
old_manifest=json.loads((EVAL/'phase25_eval_manifest.json').read_text(encoding='utf8'))
topics=old_manifest['topics'];topic_ids={t['topic_id'] for t in topics}
base_ids=set(old_manifest['doc_ids'])
qrels=defaultdict(dict)
for line in (EVAL/'trec_covid_complete_qrels.txt').read_text().splitlines():
    q,it,uid,g=line.split()
    if q in topic_ids and int(g)>=0:qrels[q][uid]=int(g)
alias=set(json.loads((EVAL/'phase26_candidate_manifest.json').read_text(encoding='utf8'))['excluded_ambiguous_doc_ids'])
added=set()
for q in sorted(topic_ids,key=int):
    for grade_predicate in (lambda g:g>0,lambda g:g==0):
        candidates=sorted(uid for uid,g in qrels[q].items() if grade_predicate(g) and uid not in base_ids
                          and uid not in alias and status[uid]['full_text_status']=='valid_body')
        added.update(candidates[:10])
old={json.loads(line)['doc_id']:json.loads(line) for line in (ROOT/'data/processed/phase25_eval_docs.jsonl').read_text(encoding='utf8').splitlines()}
docs=[]
for uid in sorted(base_ids|added):
    if uid in old:
        d=old[uid];sections=[s['text'] for s in d['sections'] if s.get('text')]
    else:
        entry=status[uid]
        raw=ROOT/'data/raw/cord19_2020-07-16/phase26/parses'/entry['source_path']
        obj=json.loads(raw.read_text(encoding='utf8'))
        sections=[s.get('text','') for s in obj.get('body_text',[]) if s.get('text')]
    docs.append((uid,sections))
model=SentenceTransformer('BAAI/bge-small-en-v1.5',cache_folder=str(ROOT/'models'),device='cuda' if torch.cuda.is_available() else 'cpu')
chunks=[]
for uid,sections in docs:
    for text in sections:
        for piece in fixed_chunks(text):
            queue=[piece]
            while queue:
                x=queue.pop(0)
                if len(model.tokenizer(x,add_special_tokens=True)['input_ids'])>240:
                    mid=len(x)//2;queue[:0]=[x[:mid],x[mid:]]
                elif x.strip():chunks.append({'doc_id':uid,'text':x})
start=time.perf_counter()
vectors=model.encode([c['text'] for c in chunks],batch_size=32,normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True).astype('float32')
if torch.cuda.is_available():torch.cuda.synchronize()
seconds=time.perf_counter()-start
index=faiss.IndexFlatIP(vectors.shape[1]);index.add(vectors)
faiss.write_index(index,str(OUT/'bge_8topic_expanded.faiss'))
per={}
for topic in topics:
    q=topic['topic_id']
    prompt='Represent this sentence for searching relevant passages: '+topic['query']
    vec=model.encode([prompt],normalize_embeddings=True,convert_to_numpy=True).astype('float32')
    scores,positions=index.search(vec,index.ntotal)
    ranking=document_ranking([{'doc_id':chunks[int(i)]['doc_id'],'score':float(s)} for s,i in zip(scores[0],positions[0]) if i>=0])
    top=ranking[:10]
    per[q]={'top10':top,'judged':sum(uid in qrels[q] for uid in top),'unjudged':sum(uid not in qrels[q] for uid in top),
            'corpus_judged':sum(uid in qrels[q] for uid,_ in docs),'corpus_positive':sum(qrels[q].get(uid,0)>0 for uid,_ in docs)}
result={'model':'BAAI/bge-small-en-v1.5','device':'cuda' if torch.cuda.is_available() else 'cpu',
        'baseline_documents':len(base_ids),'expanded_documents':len(docs),'added_documents':len(added),
        'expanded_chunks':len(chunks),'embedding_seconds':seconds,'mean_unjudged_top10':statistics.mean(v['unjudged'] for v in per.values()),
        'prior_phase25_mean_unjudged_top10':json.loads((ROOT/'experiments/results/phase25/comparison.json').read_text(encoding='utf8'))['bge']['mean_unjudged_top10'],
        'per_query':per,'added_ids':sorted(added),'selection_rule':'same 8 topics and prior 58 docs; add first 10 valid positive and 10 valid grade-0 qrel IDs per topic, deduplicated'}
(OUT/'top10_coverage_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k not in ('per_query','added_ids')},indent=2))
