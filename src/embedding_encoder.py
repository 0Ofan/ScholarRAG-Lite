"""Encode frozen Pilot chunks with the local BGE model; measure three runs."""
import argparse
import json
import math
import os
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from sentence_transformers import SentenceTransformer
from embedding_core import encode_block

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--variant',choices=('phase31','phase32','phase34'),default='phase31')
variant=parser.parse_args().variant
names={'phase31':('phase31_pilot','phase31_pilot_chunks.jsonl','phase31_pilot_chunk_ids.json',
                  'phase31_pilot_embeddings.npy'),
       'phase32':('phase32_improved','phase32_sentence_aware_chunks.jsonl',
                  'phase32_sentence_aware_chunk_ids.json','phase32_sentence_aware_embeddings.npy'),
       'phase34':('phase34_ablation','phase34_ablation_chunks.jsonl',
                  'phase34_ablation_chunk_ids.json','phase34_ablation_embeddings.npy')}
result_name,chunk_name,id_name,embedding_name=names[variant]
PROC=ROOT/'data/processed';RESULT=ROOT/'experiments/results'/result_name
RESULT.mkdir(parents=True,exist_ok=True)
config=json.loads((ROOT/'experiments/configs/phase27_experiment_config.json').read_text(encoding='utf8'))
snapshot=ROOT/'models/models--BAAI--bge-small-en-v1.5/snapshots'/config['model_snapshots']['bge']
chunk_file=PROC/chunk_name
texts=[];chunk_ids=[]
t=time.perf_counter()
for line in chunk_file.open(encoding='utf8'):
    row=json.loads(line);texts.append(row['text']);chunk_ids.append(row['chunk_id'])
load_seconds=time.perf_counter()-t
assert len(texts)==len(chunk_ids)==len(set(chunk_ids))>0
(PROC/id_name).write_text(json.dumps(chunk_ids,ensure_ascii=False),encoding='utf8')
assert torch.cuda.is_available(), 'Pilot experiment requires the validated CUDA environment'
torch.set_num_threads(min(8,os.cpu_count() or 1))
t=time.perf_counter();model=SentenceTransformer(str(snapshot),device='cuda',local_files_only=True)
model_load_seconds=time.perf_counter()-t
model.eval()
process=psutil.Process();vm_total=psutil.virtual_memory().total

def encode_run(repeat,baseline=None):
    batch_size=64;batch_count=0;fallbacks=[];errors=[];max_diff=0.0
    cpu_samples=[];rss_samples=[];system_cpu=[];stop=threading.Event()
    def sample():
        process.cpu_percent(None);psutil.cpu_percent(None)
        while not stop.wait(.5):
            cpu_samples.append(process.cpu_percent(None))
            rss_samples.append(process.memory_info().rss)
            system_cpu.append(psutil.cpu_percent(None))
    sampler=threading.Thread(target=sample,daemon=True);sampler.start()
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    output=baseline
    if repeat==1:output=np.lib.format.open_memmap(PROC/embedding_name,mode='w+',dtype='float32',shape=(len(texts),384))
    try:
        for pos in range(0,len(texts),1024):
            block=texts[pos:pos+1024]
            vectors,batch_size,reduced,failed=encode_block(model,block,batch_size)
            fallbacks.extend(reduced);errors.extend(failed)
            assert vectors.shape==(len(block),384)
            if repeat==1:output[pos:pos+len(block)]=vectors
            else:max_diff=max(max_diff,float(np.max(np.abs(output[pos:pos+len(block)]-vectors))))
            batch_count+=1
            if batch_count%20==0:print(f'embedding run {repeat}/3: {min(pos+1024,len(texts))}/{len(texts)}',flush=True)
        if repeat==1:output.flush()
    finally:
        elapsed=time.perf_counter()-start;stop.set();sampler.join()
    return {'repeat':repeat,'seconds':elapsed,'throughput_chunks_per_second':len(texts)/elapsed,
            'batch_size_final':batch_size,'oom_fallback_batch_sizes':fallbacks,'oom_errors':errors,
            'cpu_process_percent_max':max(cpu_samples,default=0),
            'cpu_process_percent_mean':statistics.mean(cpu_samples) if cpu_samples else None,
            'cpu_system_percent_mean':statistics.mean(system_cpu) if system_cpu else None,
            'process_rss_peak_bytes':max(rss_samples+[process.memory_info().rss]),
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'max_absolute_difference_from_first':max_diff}

runs=[];baseline=None
for rep in (1,2,3):
    row=encode_run(rep,baseline);runs.append(row)
    if rep==1:baseline=np.load(PROC/embedding_name,mmap_mode='r')
    (RESULT/'embedding_progress.json').write_text(json.dumps({'completed_repeats':runs},indent=2),encoding='utf8')
    print(f"embedding repeat {rep}/3 complete: {row['seconds']:.1f}s",flush=True)
vectors=np.load(PROC/embedding_name,mmap_mode='r')
norm_min=math.inf;norm_max=-math.inf;nonfinite=0;zero=0
for pos in range(0,len(vectors),4096):
    block=np.asarray(vectors[pos:pos+4096]);norms=np.linalg.norm(block,axis=1)
    nonfinite+=int((~np.isfinite(block)).sum());zero+=int((norms==0).sum())
    norm_min=min(norm_min,float(norms.min()));norm_max=max(norm_max,float(norms.max()))
assert nonfinite==0 and zero==0 and .999<=norm_min<=norm_max<=1.001
elapsed=[x['seconds'] for x in runs]
result={'version':f'{variant}-embedding-v1','model':'BAAI/bge-small-en-v1.5',
        'snapshot':config['model_snapshots']['bge'],'device':'cuda',
        'gpu_name':torch.cuda.get_device_name(0),'gpu_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
        'process_ram_total_bytes':vm_total,'chunk_count':len(texts),'dimension':384,
        'dtype':'float32','normalization':'L2 unit norm','embedding_shape':list(vectors.shape),
        'embedding_file_bytes':(PROC/embedding_name).stat().st_size,
        'chunk_id_file_bytes':(PROC/id_name).stat().st_size,
        'chunk_text_load_seconds':load_seconds,'model_load_seconds':model_load_seconds,
        'repeats':runs,'seconds_median':statistics.median(elapsed),'seconds_min':min(elapsed),
        'seconds_max':max(elapsed),'nonfinite_elements':nonfinite,'zero_vectors':zero,
        'norm_min':norm_min,'norm_max':norm_max}
(RESULT/'embedding_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({k:result[k] for k in ('chunk_count','embedding_shape','seconds_median','seconds_min','seconds_max',
                                        'norm_min','norm_max','nonfinite_elements','zero_vectors')},indent=2),flush=True)
