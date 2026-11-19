"""Verify and consolidate Pilot measurements without changing frozen inputs."""
import hashlib
import json
from pathlib import Path
import faiss
import numpy as np

ROOT=Path(__file__).resolve().parents[1];PROC=ROOT/'data/processed'
RESULT=ROOT/'experiments/results/phase31_pilot';INDEX=ROOT/'data/indexes/phase31_pilot'
load=lambda p:json.loads(p.read_text(encoding='utf8'))
manifest=load(PROC/'phase31_pilot_manifest.json');chunking=load(RESULT/'chunking_stats.json')
embedding=load(RESULT/'embedding_metrics.json');index_meta=load(INDEX/'index_metadata.json')
metrics=load(RESULT/'metrics.json');pre=load(RESULT/'preprocess_repeats.json')
performance_path=RESULT/'performance_metrics.json';performance=load(performance_path)
ids=load(PROC/'phase31_pilot_chunk_ids.json');index_ids=load(INDEX/'chunk_ids.json')
vectors=np.load(PROC/'phase31_pilot_embeddings.npy',mmap_mode='r')
index=faiss.read_index(str(INDEX/'faiss.index'))
assert manifest['document_count']==2000 and len(set(d['doc_id'] for d in manifest['documents']))==2000
assert len(ids)==len(set(ids))==len(index_ids)==index.ntotal==len(vectors)==chunking['total_chunks']==embedding['chunk_count']
assert ids==index_ids and vectors.shape[1]==384
assert index_meta['index_reload_verified'] and len(metrics['per_query'])==50
assert all(r['chunk_id_sha256']==pre['runs'][0]['chunk_id_sha256'] for r in pre['runs'])
assert all(r['max_absolute_difference_from_first']<1e-5 for r in embedding['repeats'])
assert len(performance['retrieval_latency_repeats'])==3 and len(performance['query_encoding_repeats'])==3
performance['preprocessing_repeats']=pre['runs']
performance['preprocessing_stage_seconds']=pre['stage_seconds']
performance['index_build_seconds']=index_meta['index_build_seconds']
performance['index_write_seconds']=index_meta['index_write_seconds']
performance['index_total_seconds']=index_meta['index_total_seconds']
performance_path.write_text(json.dumps(performance,ensure_ascii=False,indent=2),encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'version':'phase31-validation-v1','manifest_documents':2000,'chunks':len(ids),'embedding_shape':list(vectors.shape),
        'index_vectors':index.ntotal,'queries':len(metrics['per_query']),
        'preprocessing_repeats':len(pre['runs']),'embedding_repeats':len(embedding['repeats']),
        'query_and_retrieval_repeats':len(performance['retrieval_latency_repeats']),
        'chunk_id_sha256':pre['runs'][0]['chunk_id_sha256'],
        'manifest_sha256':sha(PROC/'phase31_pilot_manifest.json'),
        'index_chunk_ids_sha256':sha(INDEX/'chunk_ids.json'),
        'all_checks_passed':True}
(RESULT/'validation_summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
