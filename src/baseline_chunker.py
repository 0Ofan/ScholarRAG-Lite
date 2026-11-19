"""Build traceable Fixed-length Pilot chunks with the frozen Phase 2.7 rules."""
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer
from phase27_chunking import make_chunks

ROOT=Path(__file__).resolve().parents[1]
PROC=ROOT/'data/processed';RESULT=ROOT/'experiments/results/phase31_pilot'
manifest=json.loads((PROC/'phase31_pilot_manifest.json').read_text(encoding='utf8'))
config=json.loads((ROOT/'experiments/configs/phase27_experiment_config.json').read_text(encoding='utf8'))
snapshot=ROOT/'models/models--BAAI--bge-small-en-v1.5/snapshots'/config['model_snapshots']['bge']
tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
assert config['chunking']['fixed']=={'target_tokens':180,'overlap_tokens':20,'maximum_tokens':240,'minimum_tail_tokens':40}

def dist(values):
    v=sorted(values)
    return {'count':len(v),'mean':statistics.mean(v),'median':statistics.median(v),
            'p95':v[int(.95*(len(v)-1))],'min':v[0],'max':v[-1]}

chunk_path=PROC/'phase31_pilot_chunks.jsonl';tokens=[];doc_counts=[];seen=set()
phase=Counter();clean_totals=Counter();bad_mapping=0;over_limit=0
start=time.perf_counter()
with chunk_path.open('w',encoding='utf8',newline='\n') as out:
    for j,doc in enumerate(manifest['documents'],1):
        t=time.perf_counter()
        path=ROOT/'data/raw/cord19_2020-07-16/phase26/parses'/doc['source_path']
        obj=json.loads(path.read_text(encoding='utf8'))
        phase['document_loading_seconds']+=time.perf_counter()-t
        t=time.perf_counter()
        sections=[{'section':'Title','text':doc['title']},
                  {'section':'Abstract','text':' '.join(a.get('text','') for a in obj.get('abstract',[]))}]
        sections.extend(obj.get('body_text',[]))
        phase['section_assembly_seconds']+=time.perf_counter()-t
        t=time.perf_counter()
        chunks,clean=make_chunks(doc['doc_id'],sections,tokenizer,None,'fixed',doc['source_path'])
        phase['common_cleaning_and_chunking_seconds']+=time.perf_counter()-t
        clean_totals.update(clean);doc_counts.append(len(chunks))
        for c in chunks:
            spans=c['source_spans'];first=spans[0]
            valid=bool(spans and all(0<=p['section_index']<len(sections) and
                    0<=p['raw_char_start']<p['raw_char_end']<=len(sections[p['section_index']]['text']) for p in spans))
            bad_mapping+=not valid
            if c['chunk_id'] in seen:raise AssertionError('duplicate chunk ID')
            seen.add(c['chunk_id'])
            over_limit+=c['tokens_bge']>240
            row={'chunk_id':c['chunk_id'],'doc_id':doc['doc_id'],'original_uid':doc['original_uid'],
                 'aliases':doc['aliases'],'title':doc['title'],'doi':doc['doi'],
                 'section_title':' | '.join(h['heading'] or '' for h in c['headings']),
                 'section_index':first['section_index'],
                 'section_kind':'title' if first['section_index']==0 else 'abstract' if first['section_index']==1 else 'body',
                 'text':c['text'],'token_count':c['tokens_bge'],'source_path':c['source'],
                 'source_start':first['raw_char_start'],'source_end':first['raw_char_end'],
                 'source_spans':spans}
            out.write(json.dumps(row,ensure_ascii=False)+'\n');tokens.append(c['tokens_bge'])
        if j%250==0:print(f'chunked {j}/2000 docs; {len(tokens)} chunks',flush=True)
assert len(seen)==len(tokens) and not over_limit and not bad_mapping
result={'version':'phase31-chunk-stats-v1','method':'phase27 fixed 180/20/max240',
        'document_count':len(doc_counts),'total_chunks':len(tokens),'token_lengths':dist(tokens),
        'chunks_per_document':dist(doc_counts),'short_chunks_under_40':sum(x<40 for x in tokens),
        'over_240_chunks':over_limit,'source_mapping_success_rate':1-bad_mapping/len(tokens),
        'common_preprocessing':dict(clean_totals),'timing':dict(phase),
        'total_seconds':time.perf_counter()-start,'chunk_file_bytes':chunk_path.stat().st_size}
(RESULT/'chunking_stats.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'chunks':len(tokens),'tokens':result['token_lengths'],
                  'seconds':result['total_seconds']},indent=2),flush=True)
