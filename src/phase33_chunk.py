"""Resumable chunk generation and repeatable preprocessing benchmarks."""
import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path
import psutil
from transformers import AutoTokenizer
from phase27_chunking import make_chunks

ROOT=Path(__file__).resolve().parents[1]
P=argparse.ArgumentParser();P.add_argument('--scale',choices=('medium','large'),required=True)
P.add_argument('--method',choices=('fixed','sentence'),required=True)
P.add_argument('--mode',choices=('build','benchmark'),default='build')
a=P.parse_args();run=f'phase33_{a.scale}_{a.method}'
PROC=ROOT/'data/processed';OUT=ROOT/'experiments/results/phase33'/run;OUT.mkdir(parents=True,exist_ok=True)
PARTS=PROC/'phase33_parts'/run;PARTS.mkdir(parents=True,exist_ok=True)
manifest_path=PROC/f'phase33_{a.scale}_manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf8'))
config=json.loads((ROOT/'experiments/configs/phase27_experiment_config.json').read_text(encoding='utf8'))
assert manifest['document_count']==len(manifest['documents'])==(6000 if a.scale=='medium' else 12000)
snapshot=ROOT/'models/models--BAAI--bge-small-en-v1.5/snapshots'/config['model_snapshots']['bge']
tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
chunk_path=PROC/f'{run}_chunks.jsonl';process=psutil.Process()

def make_doc(doc):
    t=time.perf_counter()
    source=ROOT/'data/raw/cord19_2020-07-16/phase26/parses'/doc['source_path']
    obj=json.loads(source.read_text(encoding='utf8'))
    loading=time.perf_counter()-t
    t=time.perf_counter()
    sections=[{'section':'Title','text':doc['title']},
              {'section':'Abstract','text':' '.join(x.get('text','') for x in obj.get('abstract',[]))}]
    sections.extend(obj.get('body_text',[]))
    assembly=time.perf_counter()-t
    t=time.perf_counter()
    chunks,clean=make_chunks(doc['doc_id'],sections,tok,None,a.method,doc['source_path'])
    chunking=time.perf_counter()-t
    assert len({x['chunk_id'] for x in chunks})==len(chunks)
    assert all(x['tokens_bge']<=240 and x['source_spans'] for x in chunks)
    return chunks,clean,loading,assembly,chunking

def record(doc,chunk):
    first=chunk['source_spans'][0]
    return {'chunk_id':chunk['chunk_id'],'doc_id':doc['doc_id'],'original_uid':doc['original_uid'],
            'aliases':doc['aliases'],'title':doc['title'],'doi':doc['doi'],
            'section_title':' | '.join(h['heading'] or '' for h in chunk['headings']),
            'section_index':first['section_index'],
            'section_kind':'title' if first['section_index']==0 else 'abstract' if first['section_index']==1 else 'body',
            'text':chunk['text'],'token_count':chunk['tokens_bge'],'source_path':doc['source_path'],
            'source_start':first['raw_char_start'],'source_end':first['raw_char_end'],
            'source_spans':chunk['source_spans']}

def digest_update(h,chunk):
    h.update(chunk['chunk_id'].encode());h.update(b'\0');h.update(chunk['text'].encode('utf8'));h.update(b'\n')

def run_part(part_index,docs):
    path=PARTS/f'part_{part_index:04d}.jsonl';meta=PARTS/f'part_{part_index:04d}.json'
    if path.exists() and meta.exists():
        saved=json.loads(meta.read_text(encoding='utf8'))
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if saved.get('manifest_sha256')==manifest_sha and saved.get('file_sha256')==actual and saved.get('doc_ids')==[d['doc_id'] for d in docs]:
            return saved,True
    temp=path.with_suffix('.jsonl.tmp');times=Counter();clean_total=Counter();lengths=[];per_doc=[]
    rss_peak=0;start=time.perf_counter()
    with temp.open('w',encoding='utf8',newline='\n') as stream:
        for doc in docs:
            chunks,clean,loading,assembly,chunking=make_doc(doc)
            times['document_loading_seconds']+=loading
            times['section_assembly_seconds']+=assembly
            times['common_cleaning_and_chunking_seconds']+=chunking
            clean_total.update(clean);per_doc.append(len(chunks))
            for c in chunks:
                stream.write(json.dumps(record(doc,c),ensure_ascii=False)+'\n')
                lengths.append(c['tokens_bge'])
            rss_peak=max(rss_peak,process.memory_info().rss)
    temp.replace(path)
    with path.open('rb') as stream:file_sha=hashlib.file_digest(stream,'sha256').hexdigest()
    saved={'version':'phase33-chunk-part-v1','run_id':run,'part_index':part_index,
           'manifest_sha256':manifest_sha,'file_sha256':file_sha,
           'doc_ids':[d['doc_id'] for d in docs],'chunk_count':len(lengths),
           'token_lengths':lengths,'per_doc_chunks':per_doc,'common_preprocessing':dict(clean_total),
           'timing':dict(times),'part_wall_seconds':time.perf_counter()-start,
           'process_rss_peak_bytes':rss_peak,'file_bytes':path.stat().st_size}
    meta.write_text(json.dumps(saved,ensure_ascii=False),encoding='utf8')
    return saved,False

def distribution(values):
    values=sorted(values)
    return {'count':len(values),'mean':statistics.mean(values),'median':statistics.median(values),
            'p95':values[int(.95*(len(values)-1))],'min':values[0],'max':values[-1]}

if a.mode=='build':
    parts=[];new=0
    for i,start in enumerate(range(0,len(manifest['documents']),250)):
        entry,reused=run_part(i,manifest['documents'][start:start+250]);parts.append(entry)
        new+=not reused
        print(f'{run} part {i+1}/{(len(manifest["documents"])+249)//250}: '
              f'{entry["chunk_count"]} chunks, {"cached" if reused else "built"}',flush=True)
    # The complete JSONL is built atomically from checksum-verified parts.
    combined=chunk_path.with_suffix('.jsonl.tmp');digest=hashlib.sha256();count=0
    with combined.open('wb') as output:
        for i,entry in enumerate(parts):
            with (PARTS/f'part_{i:04d}.jsonl').open('rb') as stream:
                while block:=stream.read(4*1024*1024):output.write(block)
    combined.replace(chunk_path)
    with chunk_path.open(encoding='utf8') as stream:
        for line in stream:
            row=json.loads(line);digest_update(digest,row);count+=1
    lengths=[x for part in parts for x in part['token_lengths']]
    per_doc=[x for part in parts for x in part['per_doc_chunks']]
    assert count==len(lengths)==sum(x['chunk_count'] for x in parts)
    assert len(per_doc)==len(manifest['documents']) and max(lengths)<=240
    times=Counter();clean=Counter()
    for part in parts:
        times.update(part['timing']);clean.update(part['common_preprocessing'])
    stats={'version':'phase33-chunk-stats-v1','run_id':run,'scale':a.scale,'method':a.method,
           'manifest_sha256':manifest_sha,'document_count':len(per_doc),'total_chunks':count,
           'token_lengths':distribution(lengths),'chunks_per_document':distribution(per_doc),
           'short_chunks_under_40':sum(x<40 for x in lengths),'over_240_chunks':sum(x>240 for x in lengths),
           'source_mapping_success_rate':1.0,'common_preprocessing':dict(clean),'timing':dict(times),
           'chunking_active_wall_seconds':sum(x['part_wall_seconds'] for x in parts),
           'chunk_file_bytes':chunk_path.stat().st_size,
           'process_rss_peak_bytes':max(x['process_rss_peak_bytes'] for x in parts),
           'parts_total':len(parts),'parts_built_this_invocation':new,
           'chunk_content_sha256':digest.hexdigest()}
    (OUT/'chunking_stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'run':run,'docs':len(per_doc),'chunks':count,
                      'chunking_seconds':times['common_cleaning_and_chunking_seconds'],
                      'parts_built':new,'content_sha256':digest.hexdigest()},indent=2),flush=True)
else:
    expected=json.loads((OUT/'chunking_stats.json').read_text(encoding='utf8'))
    digest=hashlib.sha256();times=Counter();count=0;rss=0;start=time.perf_counter()
    for j,doc in enumerate(manifest['documents'],1):
        chunks,_,loading,assembly,chunking=make_doc(doc)
        times['document_loading_seconds']+=loading
        times['section_assembly_seconds']+=assembly
        times['common_cleaning_and_chunking_seconds']+=chunking
        for c in chunks:digest_update(digest,c)
        count+=len(chunks);rss=max(rss,process.memory_info().rss)
        if j%1000==0:print(f'{run} benchmark {j}/{len(manifest["documents"])}',flush=True)
    assert count==expected['total_chunks'] and digest.hexdigest()==expected['chunk_content_sha256']
    path=OUT/'chunking_repeats.json'
    data=json.loads(path.read_text(encoding='utf8')) if path.exists() else {'version':'phase33-chunk-repeats-v1','runs':[]}
    if not data['runs']:
        data['runs'].append({'repeat':1,'timing':expected['timing'],
          'active_wall_seconds':expected['chunking_active_wall_seconds'],
          'chunk_count':expected['total_chunks'],'content_sha256':expected['chunk_content_sha256'],
          'includes_jsonl_write':True})
    rep=len(data['runs'])+1
    assert rep<=3
    data['runs'].append({'repeat':rep,'timing':dict(times),'active_wall_seconds':time.perf_counter()-start,
                        'chunk_count':count,'content_sha256':digest.hexdigest(),
                        'process_rss_peak_bytes':rss,'includes_jsonl_write':False})
    if len(data['runs'])==3:
        keys=('document_loading_seconds','section_assembly_seconds','common_cleaning_and_chunking_seconds')
        data['stage_seconds']={k:{'median':statistics.median(x['timing'][k] for x in data['runs']),
                                  'min':min(x['timing'][k] for x in data['runs']),
                                  'max':max(x['timing'][k] for x in data['runs'])} for k in keys}
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'run':run,'repeat':rep,'chunks':count,'chunking_seconds':times['common_cleaning_and_chunking_seconds']},indent=2),flush=True)
