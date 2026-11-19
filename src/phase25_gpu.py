"""Repeat identical short embedding batches on CPU and CUDA."""
import json
import statistics
import time
from pathlib import Path
import psutil
import torch
from sentence_transformers import SentenceTransformer

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/phase25';OUT.mkdir(parents=True,exist_ok=True)
sample=[json.loads(x)['text'] for x in (OUT/'chunk_mapping.jsonl').read_text(encoding='utf8').splitlines()[:120]]
process=psutil.Process()
result={'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'texts':len(sample),'batch_size':16,'models':{}}
for name,model_id in [('bge','BAAI/bge-small-en-v1.5'),('minilm','sentence-transformers/all-MiniLM-L6-v2')]:
    result['models'][name]={}
    for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
        start=time.perf_counter()
        model=SentenceTransformer(model_id,cache_folder=str(ROOT/'models'),device=device)
        load=time.perf_counter()-start
        model.encode(sample[:8],batch_size=16,normalize_embeddings=True,show_progress_bar=False)
        if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        repeats=[];rss=[];cpu=[]
        for _ in range(3):
            process.cpu_percent(None);start=time.perf_counter()
            model.encode(sample,batch_size=16,normalize_embeddings=True,show_progress_bar=False)
            if device=='cuda':torch.cuda.synchronize()
            repeats.append(time.perf_counter()-start)
            rss.append(process.memory_info().rss);cpu.append(process.cpu_percent(None))
        result['models'][name][device]={'load_seconds':load,'repeat_seconds':repeats,
            'median_seconds':statistics.median(repeats),'median_texts_per_second':len(sample)/statistics.median(repeats),
            'range_seconds':[min(repeats),max(repeats)],'rss_bytes_after_repeats':rss,
            'cpu_percent_repeats':cpu,
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
            'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else None}
        del model
        if device=='cuda':torch.cuda.empty_cache()
(OUT/'gpu_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
