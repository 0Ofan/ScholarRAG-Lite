"""Cross-check Improved artifacts and preserve a baseline immutability proof."""
import hashlib
import json
from pathlib import Path
import faiss
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROC=ROOT/'data/processed';NEW=ROOT/'experiments/results/phase32_improved'
INDEX=ROOT/'data/indexes/phase32_sentence_aware';load=lambda p:json.loads(p.read_text(encoding='utf8'))
manifest=load(PROC/'phase31_pilot_manifest.json')
chunking=load(NEW/'chunking_stats.json');embedding=load(NEW/'embedding_metrics.json')
index_meta=load(INDEX/'index_metadata.json');metrics=load(NEW/'metrics.json')
compare=load(NEW/'comparison.json');pre=load(NEW/'preprocess_repeats.json')
perf_path=NEW/'performance_metrics.json';perf=load(perf_path)
ids=load(PROC/'phase32_sentence_aware_chunk_ids.json')
index_ids=load(INDEX/'chunk_ids.json')
vectors=np.load(PROC/'phase32_sentence_aware_embeddings.npy',mmap_mode='r')
index=faiss.read_index(str(INDEX/'faiss.index'))
assert manifest['document_count']==metrics['candidate_documents']==compare['document_count']==2000
assert len(ids)==len(set(ids))==len(index_ids)==len(vectors)==index.ntotal==chunking['total_chunks']
assert ids==index_ids and vectors.shape==(len(ids),384)
assert index_meta['index_reload_verified'] and len(metrics['per_query'])==50
assert all(x['chunk_id_sha256']==pre['runs'][0]['chunk_id_sha256'] for x in pre['runs'])
assert all(x['max_absolute_difference_from_first']<1e-5 for x in embedding['repeats'])
assert len(perf['retrieval_latency_repeats'])==len(perf['query_encoding_repeats'])==3
assert chunking['over_240_chunks']==0 and chunking['source_mapping_success_rate']==1
assert len(chunking['missing_char_examples'])==chunking['missing_nonspace_chars']
assert all(x['char']=='\uf0b7' for x in chunking['missing_char_examples'])
baseline=load(NEW/'phase31_baseline_hashes.json')['files']
for rel,old in baseline.items():
    p=ROOT/rel
    assert p.stat().st_size==old['bytes']
    with p.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==old['sha256']
perf['preprocessing_repeats']=pre['runs'];perf['preprocessing_stage_seconds']=pre['stage_seconds']
perf['index_write_seconds']=index_meta['index_write_seconds']
perf['index_total_seconds']=index_meta['index_total_seconds']
perf_path.write_text(json.dumps(perf,ensure_ascii=False,indent=2),encoding='utf8')
result={'version':'phase32-validation-v1','baseline_files_unchanged':len(baseline),
        'pilot_documents':2000,'chunks':len(ids),'embedding_shape':list(vectors.shape),
        'index_vectors':index.ntotal,'queries':len(metrics['per_query']),
        'chunking_repeats':len(pre['runs']),'embedding_repeats':len(embedding['repeats']),
        'retrieval_repeats':len(perf['retrieval_latency_repeats']),
        'source_mapping_rate':chunking['source_mapping_success_rate'],
        'skipped_layout_bullets':chunking['missing_nonspace_chars'],'all_checks_passed':True}
(NEW/'validation_summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
