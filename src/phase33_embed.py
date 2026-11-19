"""Resumable three-run BGE embedding for Medium and Large experiments."""
import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
import numpy as np
import psutil
import torch
from sentence_transformers import SentenceTransformer
from embedding_core import encode_block

ROOT=Path(__file__).resolve().parents[1]
P=argparse.ArgumentParser();P.add_argument('--scale',choices=('medium','large'),required=True)
P.add_argument('--method',choices=('fixed','sentence'),required=True)
a=P.parse_args();run=f'phase33_{a.scale}_{a.method}'
PROC=ROOT/'data/processed';OUT=ROOT/'experiments/results/phase33'/run;OUT.mkdir(parents=True,exist_ok=True)
chunk_path=PROC/f'{run}_chunks.jsonl';vector_path=PROC/f'{run}_embeddings.npy'
id_path=PROC/f'{run}_chunk_ids.json';progress_path=OUT/'embedding_progress.json'
chunk_stats=json.loads((OUT/'chunking_stats.json').read_text(encoding='utf8'))
config=json.loads((ROOT/'experiments/configs/phase27_experiment_config.json').read_text(encoding='utf8'))
expected=chunk_stats['total_chunks'];assert expected>0

# Bind each vector row to an explicit stable chunk ID before GPU work begins.
if not id_path.exists():
    ids=[]
    with chunk_path.open(encoding='utf8') as stream:
        for line in stream:ids.append(json.loads(line)['chunk_id'])
    assert len(ids)==len(set(ids))==expected
    temp=id_path.with_suffix('.json.tmp');temp.write_text(json.dumps(ids,ensure_ascii=False),encoding='utf8')
    temp.replace(id_path)
else:
    ids=json.loads(id_path.read_text(encoding='utf8'))
    assert len(ids)==len(set(ids))==expected

assert torch.cuda.is_available(),'frozen GPU configuration unavailable'
torch.set_num_threads(min(8,os.cpu_count() or 1))
snapshot=ROOT/'models/models--BAAI--bge-small-en-v1.5/snapshots'/config['model_snapshots']['bge']
t=time.perf_counter();model=SentenceTransformer(str(snapshot),device='cuda',local_files_only=True)
model_load=time.perf_counter()-t;model.eval()
proc=psutil.Process();proc.cpu_percent(None)
state={'version':'phase33-embedding-progress-v1','run_id':run,
       'chunk_content_sha256':chunk_stats['chunk_content_sha256'],
       'chunk_count':expected,'repeat':1,'next_row':0,'active_seconds':0.0,
       'batch_size':64,'max_abs_diff':0.0,'cpu_peak_percent':0.0,
       'rss_peak_bytes':0,'gpu_peak_allocated_bytes':0,'gpu_peak_reserved_bytes':0,
       'oom_fallback_batch_sizes':[],'oom_errors':[],'runs':[]}
if progress_path.exists():
    saved=json.loads(progress_path.read_text(encoding='utf8'))
    assert saved['run_id']==run and saved['chunk_content_sha256']==state['chunk_content_sha256']
    assert saved['chunk_count']==expected
    state=saved

def save():
    temp=progress_path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
    temp.replace(progress_path)

def batches(skip):
    row=0;texts=[]
    with chunk_path.open(encoding='utf8') as stream:
        for line in stream:
            if row<skip:row+=1;continue
            texts.append(json.loads(line)['text']);row+=1
            if len(texts)==1024:
                yield row-len(texts),texts
                texts=[]
        if texts:yield row-len(texts),texts
    assert row==expected

while state['repeat']<=3:
    rep=state['repeat']
    if rep==1:
        if state['next_row']==0:
            output=np.lib.format.open_memmap(vector_path,mode='w+',dtype='float32',shape=(expected,384))
        else:
            output=np.load(vector_path,mmap_mode='r+')
    else:output=np.load(vector_path,mmap_mode='r')
    assert output.shape==(expected,384)
    torch.cuda.reset_peak_memory_stats()
    start_row=state['next_row']
    for pos,texts in batches(start_row):
        t=time.perf_counter()
        vectors,batch_size,fallbacks,errors=encode_block(model,texts,state['batch_size'])
        if rep==1:
            output[pos:pos+len(texts)]=vectors
            output.flush()
        else:
            diff=float(np.max(np.abs(np.asarray(output[pos:pos+len(texts)])-vectors)))
            state['max_abs_diff']=max(state['max_abs_diff'],diff)
        state['active_seconds']+=time.perf_counter()-t
        state['batch_size']=batch_size
        state['oom_fallback_batch_sizes'].extend(fallbacks)
        state['oom_errors'].extend(errors)
        state['cpu_peak_percent']=max(state['cpu_peak_percent'],proc.cpu_percent(None))
        state['rss_peak_bytes']=max(state['rss_peak_bytes'],proc.memory_info().rss)
        state['gpu_peak_allocated_bytes']=max(state['gpu_peak_allocated_bytes'],torch.cuda.max_memory_allocated())
        state['gpu_peak_reserved_bytes']=max(state['gpu_peak_reserved_bytes'],torch.cuda.max_memory_reserved())
        state['next_row']=pos+len(texts)
        save()
        if state['next_row']%20480<1024 or state['next_row']==expected:
            print(f'{run} embedding repeat {rep}/3: {state["next_row"]}/{expected}',flush=True)
    assert state['next_row']==expected
    row={'repeat':rep,'active_seconds':state['active_seconds'],
         'throughput_chunks_per_second':expected/state['active_seconds'],
         'batch_size_final':state['batch_size'],
         'oom_fallback_batch_sizes':state['oom_fallback_batch_sizes'],
         'oom_errors':state['oom_errors'],'max_absolute_difference_from_first':state['max_abs_diff'],
         'cpu_process_percent_max':state['cpu_peak_percent'],
         'process_rss_peak_bytes':state['rss_peak_bytes'],
         'gpu_peak_allocated_bytes':state['gpu_peak_allocated_bytes'],
         'gpu_peak_reserved_bytes':state['gpu_peak_reserved_bytes']}
    state['runs'].append(row)
    print(f'{run} embedding repeat {rep}/3 complete: {state["active_seconds"]:.1f}s',flush=True)
    state.update({'repeat':rep+1,'next_row':0,'active_seconds':0.0,'batch_size':64,
                  'max_abs_diff':0.0,'cpu_peak_percent':0.0,'rss_peak_bytes':0,
                  'gpu_peak_allocated_bytes':0,'gpu_peak_reserved_bytes':0,
                  'oom_fallback_batch_sizes':[],'oom_errors':[]})
    save();del output

vectors=np.load(vector_path,mmap_mode='r')
minimum=math.inf;maximum=-math.inf;nonfinite=0;zero=0
for pos in range(0,expected,4096):
    block=np.asarray(vectors[pos:pos+4096]);norm=np.linalg.norm(block,axis=1)
    nonfinite+=int((~np.isfinite(block)).sum());zero+=int((norm==0).sum())
    minimum=min(minimum,float(norm.min()));maximum=max(maximum,float(norm.max()))
assert nonfinite==zero==0 and .999<=minimum<=maximum<=1.001
seconds=[x['active_seconds'] for x in state['runs']]
result={'version':'phase33-embedding-v1','run_id':run,'model':'BAAI/bge-small-en-v1.5',
        'snapshot':config['model_snapshots']['bge'],'device':'cuda',
        'gpu_name':torch.cuda.get_device_name(0),'gpu_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
        'chunk_count':expected,'dimension':384,'dtype':'float32','normalization':'L2 unit norm',
        'embedding_shape':[expected,384],'embedding_file_bytes':vector_path.stat().st_size,
        'chunk_id_file_bytes':id_path.stat().st_size,'model_load_seconds':model_load,
        'repeats':state['runs'],'seconds_median':statistics.median(seconds),
        'seconds_min':min(seconds),'seconds_max':max(seconds),
        'nonfinite_elements':nonfinite,'zero_vectors':zero,'norm_min':minimum,'norm_max':maximum,
        'measurement_note':'active encoding and vector-write time; excludes idle time between resumed sessions'}
(OUT/'embedding_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'run':run,'shape':result['embedding_shape'],'median_seconds':result['seconds_median'],
                  'norm_min':minimum,'norm_max':maximum},indent=2),flush=True)
