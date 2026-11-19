"""Reusable document-level Pilot retriever over an exact FAISS chunk index."""
import json
import time
from pathlib import Path
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT=Path(__file__).resolve().parents[1]
PROMPT='Represent this sentence for searching relevant passages: '

class PilotRetriever:
    def __init__(self,load_model=True,variant='phase31',run_id=None):
        variants={'phase31':('phase31_pilot','phase31_pilot_chunks.jsonl'),
                  'phase32':('phase32_sentence_aware','phase32_sentence_aware_chunks.jsonl'),
                  'phase34':('phase34_ablation','phase34_ablation_chunks.jsonl')}
        if run_id is not None:
            allowed={'phase33_medium_fixed','phase33_medium_sentence',
                     'phase33_large_fixed','phase33_large_sentence'}
            if run_id not in allowed:raise ValueError(f'unknown run ID: {run_id}')
            index_name,chunk_name=run_id,f'{run_id}_chunks.jsonl'
        else:
            if variant not in variants:raise ValueError(f'unknown retriever variant: {variant}')
            index_name,chunk_name=variants[variant]
        self.proc=ROOT/'data/processed';self.index_dir=ROOT/'data/indexes'/index_name
        self.metadata=json.loads((self.index_dir/'index_metadata.json').read_text(encoding='utf8'))
        self.chunk_ids=json.loads((self.index_dir/'chunk_ids.json').read_text(encoding='utf8'))
        with (self.proc/chunk_name).open(encoding='utf8') as stream:
            self.chunks=[json.loads(line) for line in stream]
        self.index=faiss.read_index(str(self.index_dir/'faiss.index'))
        assert len(self.chunks)==len(self.chunk_ids)==self.index.ntotal
        assert all(c['chunk_id']==cid for c,cid in zip(self.chunks,self.chunk_ids))
        self.model=None
        if load_model:
            snapshot=ROOT/'models/models--BAAI--bge-small-en-v1.5/snapshots'/self.metadata['model_snapshot']
            device='cuda' if torch.cuda.is_available() else 'cpu'
            self.model=SentenceTransformer(str(snapshot),device=device,local_files_only=True)
            self.model.eval()

    def encode_queries(self,queries):
        if self.model is None:raise RuntimeError('model not loaded')
        t=time.perf_counter()
        vec=self.model.encode([PROMPT+q for q in queries],batch_size=32,
                   normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True).astype('float32')
        if torch.cuda.is_available():torch.cuda.synchronize()
        return vec,time.perf_counter()-t

    def retrieve_vector(self,vector,top_k=10):
        if not 1<=top_k<=2000:raise ValueError('top_k must be 1..2000')
        v=np.ascontiguousarray(np.asarray(vector,dtype='float32').reshape(1,384))
        search_k=min(self.index.ntotal,max(1000,top_k*20))
        start=time.perf_counter()
        while True:
            scores,positions=self.index.search(v,search_k)
            docs=[];seen=set()
            for score,pos in zip(scores[0],positions[0]):
                if pos<0:continue
                chunk=self.chunks[int(pos)];uid=chunk['doc_id']
                if uid in seen:continue
                seen.add(uid)
                docs.append({'rank':len(docs)+1,'doc_id':uid,'original_uid':chunk['original_uid'],
                             'title':chunk['title'],'doi':chunk['doi'],'score':float(score),
                             'score_type':'normalized vector inner product',
                             'best_chunk_id':chunk['chunk_id'],'best_chunk_text':chunk['text'],
                             'section_title':chunk['section_title'],'source_path':chunk['source_path'],
                             'source_spans':chunk['source_spans'],'chunk_index':int(pos)})
                if len(docs)>=top_k:break
            if len(docs)>=top_k or search_k==self.index.ntotal:break
            search_k=min(search_k*2,self.index.ntotal)
        if len(docs)<top_k:raise RuntimeError(f'only {len(docs)} distinct documents found')
        return docs,{'retrieval_seconds':time.perf_counter()-start,'searched_chunks':search_k}

    def retrieve(self,query,top_k=10):
        vectors,query_seconds=self.encode_queries([query])
        docs,perf=self.retrieve_vector(vectors[0],top_k)
        perf['query_encoding_seconds']=query_seconds
        return docs,perf
