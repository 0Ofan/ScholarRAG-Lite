"""Full-corpus, 50-topic BGE audit. No labels are inferred for unjudged hits."""
import json
import os
import statistics
import time
from array import array
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from phase27_chunking import make_chunks
from evaluation import score_topic

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'data/evaluation';OUT=ROOT/'experiments/results/phase27'
prepared=json.loads((EVAL/'phase27_prepared_manifest.json').read_text(encoding='utf8'))
status=json.loads((EVAL/'phase26_doc_status.json').read_text(encoding='utf8'))
ids=prepared['canonical_doc_ids'];qrels=prepared['effective_qrels']
topics={x['topic_id']:x for x in map(json.loads,(EVAL/'topics.jsonl').read_text(encoding='utf8').splitlines())}
assert len(ids)==25594 and len(topics)==50
snapshot=ROOT/'models/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
device='cuda' if torch.cuda.is_available() else 'cpu'
model=SentenceTransformer(str(snapshot),device=device,local_files_only=True)
torch.set_num_threads(min(8,os.cpu_count() or 1))
index=faiss.IndexFlatIP(384);doc_numbers=array('I');pending=[]
stats=Counter();start=time.perf_counter();embedding_seconds=0.0

def flush():
    global embedding_seconds
    if not pending:return
    t=time.perf_counter()
    vectors=model.encode(pending,batch_size=64,normalize_embeddings=True,show_progress_bar=False,
                         convert_to_numpy=True).astype('float32')
    if device=='cuda':torch.cuda.synchronize()
    embedding_seconds+=time.perf_counter()-t
    index.add(vectors);pending.clear()

for doc_no,uid in enumerate(ids):
    d=status[uid];path=ROOT/'data/raw/cord19_2020-07-16/phase26/parses'/d['source_path']
    obj=json.loads(path.read_text(encoding='utf8'))
    sections=[{'section':'Title','text':d.get('title','')},
              {'section':'Abstract','text':' '.join(x.get('text','') for x in obj.get('abstract',[]))}]
    sections.extend(obj.get('body_text',[]))
    chunks,clean=make_chunks(uid,sections,model.tokenizer,None,'fixed',d['source_path'])
    for c in chunks:
        pending.append(c['text']);doc_numbers.append(doc_no)
        if len(pending)>=1024:flush()
    stats['docs']+=1;stats['chunks']+=len(chunks)
    stats['short_sections']+=clean['short_sections'];stats['merged_short_sections']+=clean['merged_short_sections']
    stats['removed_control_chars']+=clean['removed_control_chars'];stats['fallback_splits']+=clean['fallback_splits']
    if (doc_no+1)%500==0:
        progress={'documents':doc_no+1,'chunks':len(doc_numbers),'elapsed_seconds':time.perf_counter()-start,
                  'embedding_seconds':embedding_seconds,'device':device}
        (OUT/'full_audit_progress.json').write_text(json.dumps(progress,indent=2),encoding='utf8')
        print(f"indexed {doc_no+1}/{len(ids)} docs; {len(doc_numbers)} chunks; {progress['elapsed_seconds']:.1f}s",flush=True)
flush()
assert index.ntotal==len(doc_numbers)
build_seconds=time.perf_counter()-start
print(f'index complete: {index.ntotal} chunks in {build_seconds:.1f}s',flush=True)
faiss.write_index(index,str(OUT/'full_corpus_bge_fixed.faiss'))
np.save(OUT/'full_corpus_doc_numbers.npy',np.asarray(doc_numbers,dtype=np.uint32))
queries=[topics[str(q)]['query'] for q in range(1,51)]
prompts=['Represent this sentence for searching relevant passages: '+q for q in queries]
qvec=model.encode(prompts,batch_size=32,normalize_embeddings=True,show_progress_bar=False,
                  convert_to_numpy=True).astype('float32')
per={};search_start=time.perf_counter()
for qi in range(50):
    topic_id=str(qi+1);k=min(4000,index.ntotal)
    while True:
        scores,positions=index.search(qvec[qi:qi+1],k)
        ranked=[];seen=set();hit_details=[]
        for score,pos in zip(scores[0],positions[0]):
            if pos<0:continue
            uid=ids[doc_numbers[int(pos)]]
            if uid in seen:continue
            seen.add(uid);ranked.append(uid)
            hit_details.append({'rank':len(ranked),'doc_id':uid,'score':float(score),
                                'chunk_index':int(pos),'raw_source_path':status[uid]['source_path'],
                                'official_grade':qrels[topic_id].get(uid)})
            if len(ranked)>=100:break
        if len(ranked)>=100 or k==index.ntotal:break
        k=min(k*2,index.ntotal)
    assert len(ranked)>=100
    metric=score_topic(ranked,qrels[topic_id],ids)
    assert metric is not None
    tops={}
    for n in (10,20,100):
        sample=ranked[:n];labels=[qrels[topic_id].get(uid) for uid in sample]
        tops[str(n)]={'positive':sum(g is not None and g>=1 for g in labels),
                      'grade_0':sum(g==0 for g in labels),
                      'unjudged':sum(g is None for g in labels),
                      'unjudged_rate':sum(g is None for g in labels)/n}
    per[topic_id]={'query':queries[qi],
        'official_positive':sum(int(g)>0 for line in (EVAL/'trec_covid_complete_qrels.txt').read_text().splitlines()
                                 if line.split()[0]==topic_id for g in [line.split()[3]]),
        'available_positive':sum(g>0 for g in qrels[topic_id].values()),
        'metrics':metric,'top':tops,'top100_results':hit_details,'searched_chunks':k}
    print(f"topic {topic_id}/50: unjudged@10={tops['10']['unjudged']} @100={tops['100']['unjudged']}",flush=True)
search_seconds=time.perf_counter()-search_start
average={key:statistics.mean(per[q]['metrics'][key] for q in per) for key in ('ndcg@10','recall@10','recall@100','mrr')}
unjudged={str(n):{'mean':statistics.mean(per[q]['top'][str(n)]['unjudged_rate'] for q in per),
                  'count_distribution':dict(sorted(Counter(per[q]['top'][str(n)]['unjudged'] for q in per).items()))}
          for n in (10,20,100)}
result={'version':'phase27-full-audit-v1','corpus_manifest':'phase27_prepared_manifest.json',
        'snapshot':'CORD-19 2020-07-16','model':'BAAI/bge-small-en-v1.5',
        'query_instruction':'Represent this sentence for searching relevant passages: ',
        'chunking':'phase27 fixed target 180, overlap 20, max 240','faiss':'IndexFlatIP, normalized vectors',
        'device':device,'documents':len(ids),'chunks':index.ntotal,'index_build_seconds':build_seconds,
        'embedding_seconds':embedding_seconds,'search_seconds':search_seconds,
        'index_size_bytes':(OUT/'full_corpus_bge_fixed.faiss').stat().st_size,
        'source_summary':dict(stats),'average_metrics':average,'unjudged':unjudged,
        'queries_without_available_positive':[q for q in per if per[q]['available_positive']==0],
        'per_query':per}
(OUT/'top10_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'documents':len(ids),'chunks':index.ntotal,'build_seconds':build_seconds,
                  'search_seconds':search_seconds,'average_metrics':average,'unjudged':unjudged},indent=2),flush=True)
