"""Bounded embedding, chunking and FAISS smoke test; no quality claims."""
import json
import os
import re
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'data/processed/cord19_sample.jsonl'
OUT = ROOT / 'experiments/results/phase2'
OUT.mkdir(parents=True, exist_ok=True)

def chunks(text, mode, target=1000, maximum=1400):
    if mode == 'fixed':
        return [text[i:i+target] for i in range(0, len(text), target) if text[i:i+target].strip()]
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z(])', text)
    result, pending = [], ''
    for sentence in sentences:
        if len(sentence) > maximum:
            if pending: result.append(pending); pending = ''
            result.extend(sentence[i:i+maximum] for i in range(0,len(sentence),maximum))
        elif len(pending)+len(sentence)+1 > target:
            if pending: result.append(pending)
            pending = sentence
        else: pending = (pending+' '+sentence).strip()
    if pending: result.append(pending)
    return [x for x in result if x.strip()]

def main():
    docs = [json.loads(line) for line in DOCS.read_text(encoding='utf8').splitlines()]
    # Keep only papers with official IDs. Deduplicate PDF/PMC parses by cord_uid.
    docs = list({d['doc_id']: d for d in docs if d['doc_id']}.values())[:12]
    query = 'How does SARS-CoV-2 spread between people?'
    results = {'device': 'cpu', 'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'documents':len(docs)}
    for mode in ('fixed','sentence'):
        records = []
        for d in docs:
            for i, txt in enumerate(chunks(d['full_text'], mode)):
                records.append({'chunk_id': f"{d['doc_id']}:{mode}:{i}", 'doc_id':d['doc_id'], 'title':d['title'], 'text':txt})
        lens = [len(x['text']) for x in records]
        results[mode] = {'chunks':len(records), 'mean_chars':sum(lens)/len(lens), 'min_chars':min(lens), 'max_chars':max(lens)}
        if mode != 'fixed': continue
        # Bound runtime and downloaded model resources for this compatibility check.
        by_doc = defaultdict(list)
        for record in records:
            if len(by_doc[record['doc_id']]) < 10:
                by_doc[record['doc_id']].append(record)
        records = [record for group in by_doc.values() for record in group]
        for name, model_id in [('bge','BAAI/bge-small-en-v1.5'),('minilm','sentence-transformers/all-MiniLM-L6-v2')]:
            t=time.perf_counter(); model=SentenceTransformer(model_id,cache_folder=str(ROOT/'models'),device='cpu'); load=time.perf_counter()-t
            t=time.perf_counter(); vectors=model.encode([x['text'] for x in records],batch_size=8,normalize_embeddings=True,convert_to_numpy=True); elapsed=time.perf_counter()-t
            vectors=np.asarray(vectors,dtype='float32'); dim=vectors.shape[1]
            index=faiss.IndexFlatIP(dim); index.add(vectors)
            path=OUT/f'{name}.faiss'; faiss.write_index(index,str(path))
            (OUT/f'{name}_mapping.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in records)+'\n',encoding='utf8')
            prompt=('Represent this sentence for searching relevant passages: '+query) if name=='bge' else query
            q=model.encode([prompt],normalize_embeddings=True,convert_to_numpy=True).astype('float32')
            scores, positions=index.search(q,10)
            reopened=faiss.read_index(str(path))
            results[name]={'model':model_id,'dimension':dim,'load_seconds':load,'embedding_seconds':elapsed,'texts_per_second':len(records)/elapsed,
                           'batch_size':8,'index_bytes':path.stat().st_size,'index_reload_ntotal':reopened.ntotal,
                           'top5':[{'doc_id':records[int(i)]['doc_id'],'title':records[int(i)]['title'],'score':float(s)} for s,i in zip(scores[0][:5],positions[0][:5])],
                           'vector_norm_range':[float(np.linalg.norm(vectors,axis=1).min()),float(np.linalg.norm(vectors,axis=1).max())]}
            del model
    (OUT/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(results,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
