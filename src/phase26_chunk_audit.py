"""Audit token lengths and a traceable fixed-token chunk proposal on new text."""
import hashlib
import json
import re
import statistics
from pathlib import Path
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'data/evaluation'
status=json.loads((EVAL/'phase26_doc_status.json').read_text(encoding='utf8'))
positive=set()
for line in (EVAL/'trec_covid_complete_qrels.txt').read_text().splitlines():
    q,it,uid,g=line.split()
    if int(g)>0:positive.add(uid)
valid={uid for uid,d in status.items() if d['full_text_status']=='valid_body'}
def stable(ids,n):return sorted(ids,key=lambda u:hashlib.sha256(u.encode()).hexdigest())[:n]
chosen=stable(valid&positive,200)+stable(valid-positive,100)
tok=AutoTokenizer.from_pretrained(str(ROOT/'models/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'),local_files_only=True)
doc_tokens=[];sections=[];sentence_lengths=[];chunk_tokens=[];old_tails=[];examples=[]
single_section=0;short_sections=0;merged_tails=0;raw_short_tails=0;long_sentences=0
for uid in chosen:
    d=status[uid]
    path=ROOT/'data/raw/cord19_2020-07-16/phase26/parses'/d['source_path']
    obj=json.loads(path.read_text(encoding='utf8'))
    body=[x for x in obj.get('body_text',[]) if x.get('text')]
    if len(body)==1:single_section+=1
    sections.append(len(body));total=0
    for sec_no,entry in enumerate(body):
        text=entry['text']
        encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True)
        offsets=encoded['offset_mapping'];n=len(offsets);total+=n
        # Historical 600-char tail diagnostic, for comparison only.
        rem=len(text)%540
        if 0<rem<40:old_tails.append(rem)
        if n<40:short_sections+=1
        spans=[];start=0
        while start<n:
            end=min(start+180,n)
            spans.append([start,end])
            if end==n:break
            start=end-20
        if len(spans)>1 and spans[-1][1]-spans[-1][0]<40:
            prior=spans[-2]
            if n-prior[0]<=238:
                prior[1]=n;spans.pop();merged_tails+=1
            else:raw_short_tails+=1
        for st,en in spans:
            count=en-st+2
            chunk_tokens.append(count)
            if len(examples)<20:
                cs,ce=offsets[st][0],offsets[en-1][1]
                examples.append({'doc_id':uid,'source':d['source_path'],'section_index':sec_no,
                                 'section_heading':entry.get('section',''),'char_start':cs,'char_end':ce,
                                 'tokens_including_special':count,'text':text[cs:ce]})
        starts=[0]+[m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z(])',text)]
        ends=starts[1:]+[len(text)]
        for cs,ce in zip(starts,ends):
            sentence=text[cs:ce]
            count=len(tok(sentence,add_special_tokens=False)['input_ids'])
            sentence_lengths.append(count)
            if count>220:long_sentences+=1
    doc_tokens.append(total)
def dist(values):
    values=sorted(values)
    return {'count':len(values),'min':values[0],'median':statistics.median(values),
            'p95':values[int(.95*(len(values)-1))],'max':values[-1],'mean':statistics.mean(values)}
result={'sample_rule':'SHA256(cord_uid) order: 200 valid positive and 100 valid other judged docs',
        'sample_documents':len(chosen),'document_body_tokens':dist(doc_tokens),'body_entries_per_document':dist(sections),
        'sentence_tokens':dist(sentence_lengths),'sentences_over_220_tokens':long_sentences,
        'fixed_token_chunks':dist(chunk_tokens),'fixed_tail_merges':merged_tails,
        'unmerged_short_tails':raw_short_tails,'short_sections_under_40_tokens':short_sections,
        'single_body_entry_documents':single_section,'historical_short_char_tail_count':len(old_tails),
        'all_fixed_chunks_at_most_240_tokens':max(chunk_tokens)<=240}
(EVAL/'phase26_chunk_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
(EVAL/'phase26_chunk_examples.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in examples),encoding='utf8')
print(json.dumps(result,indent=2))
