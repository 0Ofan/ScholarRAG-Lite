"""Shared cleaning and two traceable token-bounded chunking strategies."""
from __future__ import annotations
import hashlib
import re

ABBREVIATIONS={'dr.','mr.','mrs.','prof.','fig.','figs.','eq.','e.g.','i.e.','et al.','vs.','no.'}

def clean_with_positions(raw):
    """Collapse whitespace/control noise; map each cleaned char to raw offset."""
    chars=[];positions=[];pending=None;removed=0
    for i,ch in enumerate(raw):
        if ch=='\x00' or (ord(ch)<32 and not ch.isspace()):
            removed+=1;continue
        if ch.isspace():
            if chars and pending is None:pending=i
            continue
        if pending is not None:
            chars.append(' ');positions.append(pending);pending=None
        chars.append(ch);positions.append(i)
    return ''.join(chars),positions,removed

def common_blocks(body_text,tokenizer,short_section_tokens=40):
    """Merge short sections with a neighbor before either strategy runs."""
    sections=[];removed=0
    for i,entry in enumerate(body_text):
        raw=entry.get('text','') if isinstance(entry,dict) else ''
        text,pos,n=clean_with_positions(raw);removed+=n
        if not text:continue
        count=len(tokenizer(text,add_special_tokens=False)['input_ids'])
        sections.append({'text':text,'positions':[(i,p) for p in pos],
                         'section_index':i,'heading':entry.get('section',''),'tokens':count})
    groups=[];short=0;merged=0
    for sec in sections:
        if sec['tokens']<short_section_tokens:
            short+=1
            if groups:
                groups[-1].append(sec);merged+=1
            else:groups.append([sec])
        elif groups and sum(s['tokens'] for s in groups[-1])<short_section_tokens:
            groups[-1].append(sec);merged+=1
        else:groups.append([sec])
    blocks=[]
    for group in groups:
        text='\n'.join(s['text'] for s in group)
        positions=[]
        for i,s in enumerate(group):
            if i:positions.append(None)
            positions.extend(s['positions'])
        blocks.append({'text':text,'positions':positions,'headings':[{'section_index':s['section_index'],'heading':s['heading']} for s in group]})
    return blocks,{'input_sections':len(body_text),'nonempty_sections':len(sections),
                   'short_sections':short,'merged_short_sections':merged,'removed_control_chars':removed}

def source_spans(positions,start,end):
    spans=[]
    for item in positions[start:end]:
        if item is None:continue
        section,raw=item
        if spans and spans[-1]['section_index']==section and raw<=spans[-1]['raw_char_end']+1:
            spans[-1]['raw_char_end']=raw+1
        else:spans.append({'section_index':section,'raw_char_start':raw,'raw_char_end':raw+1})
    return spans

def fixed_ranges(text,tokenizer,target=180,overlap=20,min_tail=40):
    offsets=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
    n=len(offsets)
    if not n:return [],0
    spans=[];start=0
    while start<n:
        end=min(start+target,n);spans.append([start,end])
        if end==n:break
        start=end-overlap
    merged=0
    if len(spans)>1 and spans[-1][1]-spans[-1][0]<min_tail and n-spans[-2][0]<=238:
        spans[-2][1]=n;spans.pop();merged=1
    return [(offsets[a][0],offsets[b-1][1]) for a,b in spans],merged

def sentence_char_ranges(text):
    cuts=[0]
    # Include closing quotes/brackets after punctuation; decimals have no
    # intervening whitespace and therefore cannot create a boundary here.
    boundary=r'[.!?][)\]"\'”’]*[ \t]+(?=[A-Z0-9(“"\'‘])|\n+'
    for m in re.finditer(boundary,text):
        if text[m.start()]!='\n':
            before=text[max(0,m.start()-20):m.start()+1].lower().strip()
            if any(before.endswith(x) for x in ABBREVIATIONS):continue
            if re.search(r'(?:\b[a-z]\.){2,}$',before):continue
        cuts.append(m.end())
    cuts.append(len(text))
    return [(a,b) for a,b in zip(cuts,cuts[1:]) if text[a:b].strip()]

def sentence_ranges(text,tokenizer,target=180,maximum=220,min_tail=40):
    units=[];long_splits=0
    for a,b in sentence_char_ranges(text):
        segment=text[a:b]
        offsets=tokenizer(segment,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
        if not offsets:continue
        if len(offsets)>maximum:
            long_splits+=1
            for start in range(0,len(offsets),target):
                stop=min(start+target,len(offsets))
                units.append((a+offsets[start][0],a+offsets[stop-1][1],stop-start))
        else:units.append((a+offsets[0][0],a+offsets[-1][1],len(offsets)))
    ranges=[];current=[];count=0
    for unit in units:
        if current and count+unit[2]>target:
            ranges.append((current[0][0],current[-1][1],count))
            current=[];count=0
        current.append(unit);count+=unit[2]
        if count>=maximum:
            ranges.append((current[0][0],current[-1][1],count));current=[];count=0
    if current:ranges.append((current[0][0],current[-1][1],count))
    merged=0
    if len(ranges)>1 and ranges[-1][2]<min_tail:
        prev=ranges[-2];last=ranges[-1]
        joined=text[prev[0]:last[1]]
        if len(tokenizer(joined,add_special_tokens=False)['input_ids'])<=maximum:
            ranges[-2]=(prev[0],last[1],prev[2]+last[2]);ranges.pop();merged=1
    return [(a,b) for a,b,_ in ranges],{'long_sentence_splits':long_splits,'merged_tails':merged}

def sentence_overlap_ranges(text,base_ranges,tokenizer,overlap_tokens=20,maximum=220,
                            other_tokenizer=None):
    """Prefix each later base range with a bounded suffix of the previous one.

    Prefer whole trailing sentences. If the last sentence is too long, use its
    final tokens. The base ranges are never shortened, so source coverage is
    exactly the same as in the non-overlapping sentence strategy.
    """
    if overlap_tokens<0:raise ValueError('overlap_tokens must be nonnegative')
    if not overlap_tokens:return [(a,b,{'base_start':a,'base_end':b,'actual_tokens':0,
                                       'rule':'none','whole_sentences':0}) for a,b in base_ranges]
    result=[]
    for index,(a,b) in enumerate(base_ranges):
        meta={'base_start':a,'base_end':b,'actual_tokens':0,'rule':'none',
              'whole_sentences':0}
        if index:
            previous_a,previous_b=base_ranges[index-1]
            base_count=len(tokenizer(text[a:b],add_special_tokens=True)['input_ids'])
            # An existing base chunk may exceed the soft maximum by a small
            # tokenizer-boundary effect; never remove its source content.
            cap=max(maximum,base_count)
            def fits(start):
                actual=len(tokenizer(text[start:a],add_special_tokens=False)['input_ids'])
                return (0<actual<=overlap_tokens and
                        len(tokenizer(text[start:b],add_special_tokens=True)['input_ids'])<=cap and
                        (other_tokenizer is None or
                         len(other_tokenizer(text[start:b],add_special_tokens=True)['input_ids'])<=240))
            selected=None;sentence_count=0
            previous_text=text[previous_a:previous_b]
            for local_a,_ in reversed(sentence_char_ranges(previous_text)):
                candidate=previous_a+local_a
                if fits(candidate):selected=candidate;sentence_count+=1
                else:break
            if selected is not None:
                meta['rule']='whole_sentence';meta['whole_sentences']=sentence_count
            else:
                offsets=tokenizer(text[previous_a:a],add_special_tokens=False,
                                  return_offsets_mapping=True)['offset_mapping']
                for count in range(min(overlap_tokens,len(offsets)),0,-1):
                    candidate=previous_a+offsets[-count][0]
                    if fits(candidate):selected=candidate;break
                if selected is not None:meta['rule']='token_tail'
            if selected is not None:
                meta['actual_tokens']=len(tokenizer(text[selected:a],add_special_tokens=False)['input_ids'])
                a=selected
            else:meta['rule']='no_capacity'
        result.append((a,b,meta))
    return result

def make_chunks(doc_id,body_text,tokenizer,other_tokenizer,strategy,source,overlap_tokens=0):
    if overlap_tokens<0:raise ValueError('overlap_tokens must be nonnegative')
    if overlap_tokens and strategy!='sentence':raise ValueError('overlap is only configurable for sentence-aware chunks')
    blocks,clean_stats=common_blocks(body_text,tokenizer)
    result=[];stats={**clean_stats,'tail_merges':0,'long_sentence_splits':0,'fallback_splits':0}
    if overlap_tokens:
        stats.update({'requested_overlap_tokens':overlap_tokens,'actual_overlap_tokens':0,
                      'whole_sentence_overlap_chunks':0,'token_tail_overlap_chunks':0,
                      'no_capacity_overlap_chunks':0})
    for block_index,block in enumerate(blocks):
        if strategy=='fixed':ranges,merged=fixed_ranges(block['text'],tokenizer);stats['tail_merges']+=merged
        elif strategy=='sentence':
            ranges,detail=sentence_ranges(block['text'],tokenizer)
            stats['tail_merges']+=detail['merged_tails'];stats['long_sentence_splits']+=detail['long_sentence_splits']
        else:raise ValueError(strategy)
        if overlap_tokens:
            queue=sentence_overlap_ranges(block['text'],ranges,tokenizer,overlap_tokens,
                                          other_tokenizer=other_tokenizer)
        else:queue=[(a,b,None) for a,b in ranges]
        while queue:
            a,b,overlap=queue.pop(0);piece=block['text'][a:b]
            n1=len(tokenizer(piece,add_special_tokens=True)['input_ids'])
            n2=len(other_tokenizer(piece,add_special_tokens=True)['input_ids']) if other_tokenizer is not None else n1
            if max(n1,n2)>240:
                if overlap_tokens:raise ValueError('overlap chunk exceeds hard 240-token limit')
                offsets=tokenizer(piece,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
                mid=len(offsets)//2
                if mid<=0:raise ValueError('cannot split overlong token chunk')
                cut=a+offsets[mid][0]
                queue[:0]=[(a,cut,None),(cut,b,None)];stats['fallback_splits']+=1;continue
            if not piece.strip():continue
            spans=source_spans(block['positions'],a,b)
            if not spans:continue
            chunk={'chunk_id':f'{doc_id}:{strategy if not overlap_tokens else "sentence_overlap"+str(overlap_tokens)}:{block_index}:{len(result)}',
                           'doc_id':doc_id,'source':source,'text':piece,'tokens_bge':n1,'tokens_minilm':n2,
                           'headings':block['headings'],'source_spans':spans}
            if overlap_tokens:
                chunk.update({'overlap_tokens_requested':overlap_tokens,
                              'overlap_tokens_actual':overlap['actual_tokens'],
                              'overlap_rule':overlap['rule'],
                              'overlap_whole_sentences':overlap['whole_sentences'],
                              'overlap_source_spans':source_spans(block['positions'],a,overlap['base_start']),
                              'base_source_spans':source_spans(block['positions'],overlap['base_start'],b)})
                stats['actual_overlap_tokens']+=overlap['actual_tokens']
                if overlap['rule']=='whole_sentence':stats['whole_sentence_overlap_chunks']+=1
                elif overlap['rule']=='token_tail':stats['token_tail_overlap_chunks']+=1
                elif overlap['rule']=='no_capacity':stats['no_capacity_overlap_chunks']+=1
            result.append(chunk)
    return result,stats
