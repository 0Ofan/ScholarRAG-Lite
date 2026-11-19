"""Fair CPU retrieval comparison on a frozen, enriched CORD-19 sample."""
import json
import os
import statistics
import time
from pathlib import Path
import faiss
import numpy as np
import psutil
import torch
from sentence_transformers import SentenceTransformer
from evaluation import document_ranking,score_topic

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/phase25';OUT.mkdir(parents=True,exist_ok=True)
DOCS=[json.loads(x) for x in (ROOT/'data/processed/phase25_eval_docs.jsonl').read_text(encoding='utf8').splitlines()]
MANIFEST=json.loads((ROOT/'data/evaluation/phase25_eval_manifest.json').read_text(encoding='utf8'))
QRELS={t['topic_id']:{} for t in MANIFEST['topics']}
for line in (ROOT/'data/evaluation/trec_covid_complete_qrels.txt').read_text().splitlines():
    q,it,uid,grade=line.split()
    if q in QRELS and int(grade)>=0:QRELS[q][uid]=int(grade)
PROCESS=psutil.Process()

def fixed_chunks(text,size=600,overlap=60):
    if not text.strip():return []
    step=size-overlap
    return [text[i:i+size] for i in range(0,len(text),step) if text[i:i+size].strip()]

def build_chunks(tokenizers):
    records=[];split_count=0
    for d in DOCS:
        for sec_no,section in enumerate(d['sections']):
            for piece_no,piece in enumerate(fixed_chunks(section['text'])):
                # Both model tokenizers must stay below the stricter 256-token MiniLM limit.
                queue=[piece]
                while queue:
                    x=queue.pop(0)
                    if any(len(tok(x,add_special_tokens=True)['input_ids'])>240 for tok in tokenizers):
                        mid=len(x)//2
                        queue[:0]=[x[:mid],x[mid:]]
                        split_count+=1
                    elif x.strip():
                        records.append({'chunk_id':f"{d['doc_id']}:{sec_no}:{piece_no}:{len(records)}",'doc_id':d['doc_id'],
                                        'title':d['title'],'text':x})
    return records,split_count

def main():
    models={name:SentenceTransformer(model_id,cache_folder=str(ROOT/'models'),device='cpu') for name,model_id in
            [('bge','BAAI/bge-small-en-v1.5'),('minilm','sentence-transformers/all-MiniLM-L6-v2')]}
    records,split_count=build_chunks([m.tokenizer for m in models.values()])
    corpus={d['doc_id'] for d in DOCS}
    (OUT/'chunk_mapping.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf8')
    result={'documents':len(DOCS),'chunks':len(records),'topics':len(MANIFEST['topics']),'token_splits':split_count,
            'chunk_mean_chars':statistics.mean(len(r['text']) for r in records),'device':'cpu','torch':torch.__version__,
            'cuda_available':torch.cuda.is_available(),'batch_size':16}
    for name,model in models.items():
        # Warmup and three identical short-batch throughput runs.
        sample=[r['text'] for r in records[:120]]
        model.encode(sample[:8],batch_size=16,normalize_embeddings=True,show_progress_bar=False)
        repeats=[];ram=[];cpu=[]
        for _ in range(3):
            PROCESS.cpu_percent(None);start=time.perf_counter()
            model.encode(sample,batch_size=16,normalize_embeddings=True,show_progress_bar=False)
            repeats.append(time.perf_counter()-start);ram.append(PROCESS.memory_info().rss);cpu.append(PROCESS.cpu_percent(None))
        start=time.perf_counter()
        vectors=model.encode([r['text'] for r in records],batch_size=16,normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True).astype('float32')
        full_seconds=time.perf_counter()-start
        index=faiss.IndexFlatIP(vectors.shape[1]);index.add(vectors)
        index_path=OUT/f'{name}.faiss';faiss.write_index(index,str(index_path))
        per_query={};rankings={}
        for topic in MANIFEST['topics']:
            q=topic['query']
            if name=='bge':q='Represent this sentence for searching relevant passages: '+q
            vec=model.encode([q],normalize_embeddings=True,convert_to_numpy=True).astype('float32')
            scores,indices=index.search(vec,index.ntotal)
            hits=[{'doc_id':records[int(i)]['doc_id'],'score':float(s)} for s,i in zip(scores[0],indices[0]) if i>=0]
            ranking=document_ranking(hits)
            rankings[topic['topic_id']]=ranking
            per_query[topic['topic_id']]=score_topic(ranking,QRELS[topic['topic_id']],corpus)
        (OUT/f'{name}_rankings.json').write_text(json.dumps(rankings,indent=2),encoding='utf8')
        valid=[v for v in per_query.values() if v is not None]
        result[name]={'model':model._first_module().auto_model.name_or_path if hasattr(model._first_module(),'auto_model') else name,
                      'dimension':vectors.shape[1],'vector_bytes':vectors.nbytes,'index_bytes':index_path.stat().st_size,
                      'index_reload_ntotal':faiss.read_index(str(index_path)).ntotal,
                      'full_encode_seconds':full_seconds,'full_throughput':len(records)/full_seconds,
                      'short_repeat_seconds':repeats,'short_median_texts_per_second':120/statistics.median(repeats),
                      'short_range_seconds':[min(repeats),max(repeats)],'rss_bytes_after_repeats':ram,'cpu_percent_repeats':cpu,
                      'per_query':per_query,'mean_metrics':{k:statistics.mean(v[k] for v in valid) for k in ('ndcg@10','recall@10','recall@100','mrr')},
                      'mean_unjudged_top10':statistics.mean(v['unjudged_in_top10'] for v in valid)}
        del vectors,index
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps({k:({x:y for x,y in v.items() if x!='per_query'} if isinstance(v,dict) else v) for k,v in result.items()},indent=2,ensure_ascii=False))

if __name__=='__main__':main()
