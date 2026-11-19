"""Build and reload the exact normalized FAISS Pilot index."""
import argparse
import datetime as dt
import hashlib
import json
import time
from pathlib import Path
import faiss
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--variant',choices=('phase31','phase32','phase34'),default='phase31')
parser.add_argument('--run-id',choices=('phase33_medium_fixed','phase33_medium_sentence',
                                        'phase33_large_fixed','phase33_large_sentence'))
args=parser.parse_args();variant=args.variant
names={'phase31':('phase31_pilot','phase31_pilot_chunk_ids.json','phase31_pilot_embeddings.npy','phase31_pilot'),
       'phase32':('phase32_sentence_aware','phase32_sentence_aware_chunk_ids.json',
                  'phase32_sentence_aware_embeddings.npy','phase32_improved'),
       'phase34':('phase34_ablation','phase34_ablation_chunk_ids.json',
                  'phase34_ablation_embeddings.npy','phase34_ablation')}
if args.run_id:
    run_id=args.run_id
    scale=run_id.split('_')[1];method=run_id.split('_')[2]
    index_name=run_id;id_name=f'{run_id}_chunk_ids.json'
    embedding_name=f'{run_id}_embeddings.npy';result_name=f'phase33/{run_id}'
else:
    index_name,id_name,embedding_name,result_name=names[variant]
PROC=ROOT/'data/processed';INDEX=ROOT/'data/indexes'/index_name;INDEX.mkdir(parents=True,exist_ok=True)
RESULT=ROOT/'experiments/results'/result_name;RESULT.mkdir(parents=True,exist_ok=True)
manifest_path=PROC/(f'phase33_{scale}_manifest.json' if args.run_id else 'phase31_pilot_manifest.json')
manifest=json.loads(manifest_path.read_text(encoding='utf8'))
config=json.loads((ROOT/'experiments/configs/phase27_experiment_config.json').read_text(encoding='utf8'))
ids=json.loads((PROC/id_name).read_text(encoding='utf8'))
vectors=np.load(PROC/embedding_name,mmap_mode='r')
assert vectors.shape==(len(ids),384) and vectors.dtype==np.float32
assert len(ids)==len(set(ids))
start=time.perf_counter();index=faiss.IndexFlatIP(384)
for pos in range(0,len(vectors),16384):index.add(np.ascontiguousarray(vectors[pos:pos+16384]))
build_seconds=time.perf_counter()-start
assert index.ntotal==len(ids)
path=INDEX/'faiss.index';write_start=time.perf_counter();faiss.write_index(index,str(path))
write_seconds=time.perf_counter()-write_start
(INDEX/'chunk_ids.json').write_text(json.dumps(ids,ensure_ascii=False),encoding='utf8')
loaded=faiss.read_index(str(path))
assert loaded.ntotal==len(ids) and np.allclose(loaded.reconstruct(0),vectors[0],atol=1e-6)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
meta={'version':'phase33-index-v1' if args.run_id else f'{variant}-index-v1',
      'dataset_version':manifest['source_snapshot'],
      'manifest_version':manifest['version'],'manifest_sha256':sha(manifest_path),
      'embedding_model':'BAAI/bge-small-en-v1.5','model_snapshot':config['model_snapshots']['bge'],
      'embedding_dimension':384,
      'chunking_method':method if args.run_id else ('fixed' if variant=='phase31' else
                           'sentence-aware-overlap20' if variant=='phase34' else 'sentence-aware'),
      'chunk_size_tokens':180,'target_tokens':180,
      'maximum_tokens':220 if (args.run_id and method=='sentence') or
                                      (not args.run_id and variant in ('phase32','phase34')) else 240,
      'overlap_tokens':20 if (args.run_id and method=='fixed') or
                                      (not args.run_id and variant in ('phase31','phase34')) else 0,
      'max_chunk_tokens':240,'normalization':'L2 unit vectors; cosine equivalent',
      'faiss_index_type':'IndexFlatIP','chunk_count':len(ids),'document_count':manifest['document_count'],
      'creation_time_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
      'index_size_bytes':path.stat().st_size,'chunk_id_map_size_bytes':(INDEX/'chunk_ids.json').stat().st_size,
      'index_build_seconds':build_seconds,'index_write_seconds':write_seconds,
      'index_total_seconds':build_seconds+write_seconds,'index_reload_verified':True,
      'chunk_id_map_sha256':sha(INDEX/'chunk_ids.json')}
(INDEX/'index_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
(RESULT/'index_metrics.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
print(json.dumps({'chunks':len(ids),'index_size_bytes':meta['index_size_bytes'],
                  'build_seconds':build_seconds,'write_seconds':write_seconds,'reloaded':True},indent=2))
