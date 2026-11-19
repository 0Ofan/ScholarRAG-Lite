"""Validate both frozen chunk algorithms on the earlier deterministic 300-paper sample."""
import hashlib
import json
import statistics
from pathlib import Path
from transformers import AutoTokenizer
from phase27_chunking import make_chunks

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'data/evaluation'
OUT=ROOT/'experiments/results/phase27'
status=json.loads((EVAL/'phase26_doc_status.json').read_text(encoding='utf8'))
positive={line.split()[2] for line in (EVAL/'trec_covid_complete_qrels.txt').read_text().splitlines() if int(line.split()[3])>0}
valid={u for u,d in status.items() if d['full_text_status']=='valid_body'}
stable=lambda xs,n:sorted(xs,key=lambda u:hashlib.sha256(u.encode()).hexdigest())[:n]
chosen=stable(valid&positive,200)+stable(valid-positive,100)
models=ROOT/'models'
bge=AutoTokenizer.from_pretrained(str(models/'models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'),local_files_only=True)
mini=AutoTokenizer.from_pretrained(str(models/'models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41'),local_files_only=True)

def summary(values):
    values=sorted(values)
    return {'count':len(values),'mean':statistics.mean(values),'median':statistics.median(values),
            'p95':values[int(.95*(len(values)-1))],'min':values[0],'max':values[-1]}

data={s:{'bge':[],'minilm':[],'chunks':0,'short_chunks_under_40':0,'over_240':0,
         'source_mapping_valid':0,'source_mapping_total':0,'input_sections':0,'short_sections':0,
         'merged_short_sections':0,'removed_control_chars':0,'long_sentence_splits':0,'tail_merges':0,
         'fallback_splits':0,'per_document_chunks':[]} for s in ('fixed','sentence')}
for j,uid in enumerate(chosen,1):
    d=status[uid]
    path=ROOT/'data/raw/cord19_2020-07-16/phase26/parses'/d['source_path']
    obj=json.loads(path.read_text(encoding='utf8'))
    # Synthetic negative section indices retain title/abstract provenance.
    sections=[{'section':'Title','text':d.get('title','')},
              {'section':'Abstract','text':' '.join(x.get('text','') for x in obj.get('abstract',[]))}]
    sections.extend(obj.get('body_text',[]))
    for strategy in data:
        chunks,stats=make_chunks(uid,sections,bge,mini,strategy,d['source_path'])
        x=data[strategy];x['chunks']+=len(chunks);x['per_document_chunks'].append(len(chunks))
        for key in ('input_sections','short_sections','merged_short_sections','removed_control_chars',
                    'long_sentence_splits','tail_merges','fallback_splits'):
            x[key]+=stats[key]
        for chunk in chunks:
            x['bge'].append(chunk['tokens_bge']);x['minilm'].append(chunk['tokens_minilm'])
            x['short_chunks_under_40']+=chunk['tokens_bge']<40
            x['over_240']+=max(chunk['tokens_bge'],chunk['tokens_minilm'])>240
            x['source_mapping_total']+=1
            x['source_mapping_valid']+=bool(chunk['source_spans'] and all(0<=p['section_index']<len(sections) and
                   0<=p['raw_char_start']<p['raw_char_end']<=len(sections[p['section_index']]['text']) for p in chunk['source_spans']))
    if j%50==0:print(f'validated {j}/300',flush=True)
for x in data.values():
    for key in ('bge','minilm','per_document_chunks'):x[key]=summary(x[key])
    x['source_mapping_rate']=x['source_mapping_valid']/x['source_mapping_total']
result={'version':'phase27-chunk-validation-v1','sample_rule':'same SHA256 UID order as Phase 2.6: 200 valid positive + 100 valid other',
        'sample_documents':len(chosen),'strategies':data,'token_limit':240,
        'common_cleaning':'normalize whitespace, remove control characters, retain title/abstract/body and source spans'}
(OUT/'chunk_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
