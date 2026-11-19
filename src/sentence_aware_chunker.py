"""Build sentence-aware Pilot chunks, optionally with overlap for Phase 3.4."""
import argparse
import json
import statistics
import time
from collections import Counter,defaultdict
from pathlib import Path
from transformers import AutoTokenizer
from phase27_chunking import make_chunks

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--overlap-tokens',type=int,default=0)
overlap_tokens=parser.parse_args().overlap_tokens
if not 0<=overlap_tokens<180:raise ValueError('overlap_tokens must be in 0..179')
PROC=ROOT/'data/processed'
RESULT=ROOT/'experiments/results'/('phase32_improved' if overlap_tokens==0 else 'phase34_ablation')
RESULT.mkdir(parents=True,exist_ok=True)
manifest=json.loads((PROC/'phase31_pilot_manifest.json').read_text(encoding='utf8'))
config=json.loads((ROOT/'experiments/configs/phase27_experiment_config.json').read_text(encoding='utf8'))
assert len(manifest['documents'])==2000
assert config['chunking']['sentence']=={'target_tokens':180,'soft_maximum_tokens':220,
                                         'hard_maximum_tokens':240,'minimum_tail_tokens':40}
models=ROOT/'models'
bge=AutoTokenizer.from_pretrained(str(models/'models--BAAI--bge-small-en-v1.5/snapshots'/config['model_snapshots']['bge']),local_files_only=True)
mini=AutoTokenizer.from_pretrained(str(models/'models--sentence-transformers--all-MiniLM-L6-v2/snapshots'/config['model_snapshots']['minilm']),local_files_only=True)
def dist(values):
    v=sorted(values)
    return {'count':len(v),'mean':statistics.mean(v),'median':statistics.median(v),
            'p95':v[int(.95*(len(v)-1))],'min':v[0],'max':v[-1]}

def missing_nonspace(sections,chunks):
    """Count non-whitespace source characters absent from all source spans."""
    intervals=defaultdict(list)
    for c in chunks:
        for p in c['source_spans']:
            intervals[p['section_index']].append((p['raw_char_start'],p['raw_char_end']))
    missing=0;total=0;examples=[]
    for sec_no,section in enumerate(sections):
        raw=section.get('text','')
        marks=bytearray(len(raw))
        for a,b in intervals.get(sec_no,[]):marks[a:b]=b'\x01'*(b-a)
        for pos,ch in enumerate(raw):
            if ch.isspace() or ch=='\x00' or (ord(ch)<32 and not ch.isspace()):continue
            total+=1
            if not marks[pos]:
                missing+=1
                if len(examples)<10:examples.append({'section_index':sec_no,'char_offset':pos,
                    'char':ch,'context':raw[max(0,pos-24):pos+25]})
    return total,missing,examples

path=PROC/('phase32_sentence_aware_chunks.jsonl' if overlap_tokens==0 else 'phase34_ablation_chunks.jsonl')
if overlap_tokens and path.exists():raise FileExistsError(f'Phase 3.4 chunks already exist: {path}')
write_path=path if overlap_tokens==0 else path.with_suffix('.jsonl.tmp')
ids=set();bge_tokens=[];mini_tokens=[];counts=[];clean_totals=Counter()
timing=Counter();missing_chars=0;source_chars=0;mapping_fail=0;missing_examples=[];start=time.perf_counter()
actual_overlaps=[];overlap_rules=Counter()
with write_path.open('w',encoding='utf8',newline='\n') as out:
    for j,doc in enumerate(manifest['documents'],1):
        t=time.perf_counter()
        source=ROOT/'data/raw/cord19_2020-07-16/phase26/parses'/doc['source_path']
        obj=json.loads(source.read_text(encoding='utf8'))
        timing['document_loading_seconds']+=time.perf_counter()-t
        t=time.perf_counter()
        sections=[{'section':'Title','text':doc['title']},
                  {'section':'Abstract','text':' '.join(a.get('text','') for a in obj.get('abstract',[]))}]
        sections.extend(obj.get('body_text',[]))
        timing['section_assembly_seconds']+=time.perf_counter()-t
        t=time.perf_counter()
        chunks,clean=make_chunks(doc['doc_id'],sections,bge,mini if overlap_tokens else None,
                                 'sentence',doc['source_path'],overlap_tokens=overlap_tokens)
        timing['common_cleaning_and_chunking_seconds']+=time.perf_counter()-t
        t=time.perf_counter()
        if not overlap_tokens:
            for c in chunks:
                c['tokens_minilm']=len(mini(c['text'],add_special_tokens=True)['input_ids'])
        timing['minilm_validation_seconds']+=time.perf_counter()-t
        clean_totals.update(clean);counts.append(len(chunks))
        t=time.perf_counter();total,missing,examples=missing_nonspace(sections,chunks)
        timing['coverage_check_seconds']+=time.perf_counter()-t
        source_chars+=total;missing_chars+=missing
        if examples and len(missing_examples)<20:
            missing_examples.extend({'doc_id':doc['doc_id'],**x} for x in examples[:20-len(missing_examples)])
        for c in chunks:
            spans=c['source_spans'];first=spans[0]
            valid=bool(spans and all(0<=p['section_index']<len(sections) and
                 0<=p['raw_char_start']<p['raw_char_end']<=len(sections[p['section_index']]['text']) for p in spans))
            mapping_fail+=not valid
            if c['chunk_id'] in ids:raise AssertionError('duplicate chunk ID')
            ids.add(c['chunk_id'])
            row={'chunk_id':c['chunk_id'],'doc_id':doc['doc_id'],'original_uid':doc['original_uid'],
                 'aliases':doc['aliases'],'title':doc['title'],'doi':doc['doi'],
                 'section_title':' | '.join(h['heading'] or '' for h in c['headings']),
                 'section_index':first['section_index'],
                 'section_kind':'title' if first['section_index']==0 else 'abstract' if first['section_index']==1 else 'body',
                 'text':c['text'],'token_count':c['tokens_bge'],'minilm_token_count':c['tokens_minilm'],
                 'source_path':c['source'],'source_start':first['raw_char_start'],
                 'source_end':first['raw_char_end'],'source_spans':spans}
            if overlap_tokens:
                row.update({k:c[k] for k in ('overlap_tokens_requested','overlap_tokens_actual',
                    'overlap_rule','overlap_whole_sentences','overlap_source_spans','base_source_spans')})
                actual_overlaps.append(c['overlap_tokens_actual']);overlap_rules[c['overlap_rule']]+=1
            out.write(json.dumps(row,ensure_ascii=False)+'\n')
            bge_tokens.append(c['tokens_bge']);mini_tokens.append(c['tokens_minilm'])
        if j%250==0:print(f'sentence chunked {j}/2000 docs; {len(ids)} chunks; missing nonspace chars {missing_chars}',flush=True)
assert len(ids)==len(bge_tokens)>0 and mapping_fail==0
assert max(bge_tokens)<=240 and max(mini_tokens)<=240
if overlap_tokens:write_path.replace(path)
result={'version':'phase32-chunk-stats-v1' if overlap_tokens==0 else 'phase34-chunk-stats-v1',
        'method':'phase27 sentence-aware target180/soft220/hard240' +
                 (f'/overlap{overlap_tokens}' if overlap_tokens else ''),
        'document_count':len(counts),'total_chunks':len(ids),'bge_token_lengths':dist(bge_tokens),
        'minilm_token_lengths':dist(mini_tokens),'chunks_per_document':dist(counts),
        'short_chunks_under_40':sum(x<40 for x in bge_tokens),
        'bge_over_220':sum(x>220 for x in bge_tokens),
        'minilm_over_220':sum(x>220 for x in mini_tokens),
        'over_240_chunks':sum(max(a,b)>240 for a,b in zip(bge_tokens,mini_tokens)),
        'source_mapping_success_rate':1-mapping_fail/len(ids),
        'source_nonspace_chars':source_chars,'missing_nonspace_chars':missing_chars,
        'missing_char_examples':missing_examples,
        'nonspace_coverage_rate':1-missing_chars/source_chars,
        'common_preprocessing':dict(clean_totals),'timing':dict(timing),
        'total_seconds':time.perf_counter()-start,'chunk_file_bytes':path.stat().st_size}
if overlap_tokens:
    result['overlap']={'requested_tokens':overlap_tokens,'actual_token_lengths':dist(actual_overlaps),
                       'rule_counts':dict(overlap_rules),
                       'chunks_with_overlap':sum(x>0 for x in actual_overlaps),
                       'chunks_without_overlap':sum(x==0 for x in actual_overlaps)}
(RESULT/'chunking_stats.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'documents':len(counts),'chunks':len(ids),'bge_max':max(bge_tokens),
                  'mini_max':max(mini_tokens),'missing_nonspace_chars':missing_chars,
                  'long_sentence_splits':clean_totals['long_sentence_splits'],
                  'seconds':result['total_seconds']},indent=2),flush=True)
