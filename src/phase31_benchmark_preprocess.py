"""Repeat loading and Fixed-length preprocessing twice without altering artifacts."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
import psutil
from transformers import AutoTokenizer
from phase27_chunking import make_chunks

ROOT=Path(__file__).resolve().parents[1];PROC=ROOT/'data/processed'
parser=argparse.ArgumentParser()
parser.add_argument('--variant',choices=('phase31','phase32','phase34'),default='phase31')
variant=parser.parse_args().variant
RESULT=ROOT/'experiments/results'/('phase31_pilot' if variant=='phase31' else
                                   'phase32_improved' if variant=='phase32' else 'phase34_ablation')
chunk_name='phase31_pilot_chunks.jsonl' if variant=='phase31' else (
           'phase32_sentence_aware_chunks.jsonl' if variant=='phase32' else 'phase34_ablation_chunks.jsonl')
strategy='fixed' if variant=='phase31' else 'sentence'
m=json.loads((PROC/'phase31_pilot_manifest.json').read_text(encoding='utf8'))
c=json.loads((ROOT/'experiments/configs/phase27_experiment_config.json').read_text(encoding='utf8'))
snapshot=ROOT/'models/models--BAAI--bge-small-en-v1.5/snapshots'/c['model_snapshots']['bge']
tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
mini=None
if variant=='phase34':
    mini_snapshot=ROOT/'models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots'/c['model_snapshots']['minilm']
    mini=AutoTokenizer.from_pretrained(str(mini_snapshot),local_files_only=True)
first=json.loads((RESULT/'chunking_stats.json').read_text(encoding='utf8'))
id_hash=hashlib.sha256()
with (PROC/chunk_name).open(encoding='utf8') as stream:
    for line in stream:
        row=json.loads(line)
        id_hash.update((row['chunk_id']+'\n').encode())
        if variant=='phase34':id_hash.update((row['text']+'\n').encode())
expected_hash=id_hash.hexdigest();runs=[]
runs.append({'repeat':1,'document_loading_seconds':first['timing']['document_loading_seconds'],
             'section_assembly_seconds':first['timing']['section_assembly_seconds'],
             'common_cleaning_and_chunking_seconds':first['timing']['common_cleaning_and_chunking_seconds'],
             'end_to_end_seconds':first['total_seconds'],'chunk_count':first['total_chunks'],
             'chunk_id_sha256':expected_hash,'includes_jsonl_write':True})
process=psutil.Process()
for rep in (2,3):
    t0=time.perf_counter();load=0;assembly=0;chunking=0;count=0;rss=0;digest=hashlib.sha256()
    for i,doc in enumerate(m['documents'],1):
        t=time.perf_counter()
        path=ROOT/'data/raw/cord19_2020-07-16/phase26/parses'/doc['source_path']
        obj=json.loads(path.read_text(encoding='utf8'))
        load+=time.perf_counter()-t
        t=time.perf_counter()
        sections=[{'section':'Title','text':doc['title']},
                  {'section':'Abstract','text':' '.join(a.get('text','') for a in obj.get('abstract',[]))}]
        sections.extend(obj.get('body_text',[]));assembly+=time.perf_counter()-t
        t=time.perf_counter();chunks,_=make_chunks(doc['doc_id'],sections,tok,mini,strategy,doc['source_path'],
                                                   overlap_tokens=20 if variant=='phase34' else 0)
        chunking+=time.perf_counter()-t;count+=len(chunks)
        for x in chunks:
            digest.update((x['chunk_id']+'\n').encode())
            if variant=='phase34':digest.update((x['text']+'\n').encode())
        rss=max(rss,process.memory_info().rss)
        if i%500==0:print(f'preprocess repeat {rep}/3: {i}/2000',flush=True)
    assert count==first['total_chunks'] and digest.hexdigest()==expected_hash
    runs.append({'repeat':rep,'document_loading_seconds':load,'section_assembly_seconds':assembly,
                 'common_cleaning_and_chunking_seconds':chunking,'end_to_end_seconds':time.perf_counter()-t0,
                 'chunk_count':count,'chunk_id_sha256':digest.hexdigest(),
                 'process_rss_peak_bytes':rss,'includes_jsonl_write':False})
def group(k):
    x=[r[k] for r in runs]
    return {'median':statistics.median(x),'min':min(x),'max':max(x)}
result={'version':f'{variant}-preprocess-benchmark-v1','strategy':strategy,'runs':runs,
        'stage_seconds':{k:group(k) for k in ('document_loading_seconds','section_assembly_seconds',
                          'common_cleaning_and_chunking_seconds','end_to_end_seconds')},
        'chunk_id_reproducibility':'exact ID and text match across 3 runs' if variant=='phase34'
                                   else 'exact match across 3 runs',
        'note':'first run includes JSONL write; repeats 2-3 do not; use stage times for comparable computation'}
(RESULT/'preprocess_repeats.json').write_text(json.dumps(result,indent=2),encoding='utf8')
performance_path=RESULT/'performance_metrics.json'
if performance_path.is_file():
    performance=json.loads(performance_path.read_text(encoding='utf8'))
    performance['preprocessing_repeats']=result['runs']
    performance['preprocessing_stage_seconds']=result['stage_seconds']
    performance_path.write_text(json.dumps(performance,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result['stage_seconds'],indent=2),flush=True)
