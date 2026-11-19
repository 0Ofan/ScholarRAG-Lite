"""Freeze a small, qrels-enriched evaluation sample without changing labels."""
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'data/evaluation'
RAW=ROOT/'data/raw/cord19_2020-07-16/phase25'
PROCESSED=ROOT/'data/processed'

def main():
    stats=json.loads((EVAL/'phase25_collection_stats.json').read_text(encoding='utf8'))
    mappings=json.loads((EVAL/'phase25_positive_metadata.json').read_text(encoding='utf8'))['mappings']
    by_uid={m['doc_id']:m for m in mappings.values()}
    topics={t['topic_id']:t for t in map(json.loads,(EVAL/'topics.jsonl').read_text(encoding='utf8').splitlines())}
    selected_topics=[q for q,s in sorted(stats['per_query'].items(),key=lambda kv:(-kv[1]['collected_fulltext'],int(kv[0]))) if s['collected_fulltext']>=5][:8]
    chosen=set()
    for q in selected_topics:
        chosen.update(stats['per_query'][q]['collected_positive_ids'][:5])
    old=[json.loads(x) for x in (PROCESSED/'cord19_sample.jsonl').read_text(encoding='utf8').splitlines()]
    old_by_uid={d['doc_id']:d for d in old if d['doc_id']}
    all_qrels={}
    for line in (EVAL/'trec_covid_complete_qrels.txt').read_text().splitlines():
        q,iteration,uid,grade=line.split()
        if q in selected_topics: all_qrels[(q,uid)]=int(grade)
    distractors=[uid for uid in sorted(old_by_uid) if uid not in chosen and all(all_qrels.get((q,uid),-1)<=0 for q in selected_topics)][:20]
    chosen.update(distractors)
    docs=[]
    for uid in sorted(chosen):
        path=RAW/(uid+'.json')
        if path.exists():
            obj=json.loads(path.read_text(encoding='utf8'))
            meta=by_uid[uid]
            sections=[{'heading':x.get('section') or '', 'text':x.get('text') or ''} for x in obj.get('body_text',[]) if x.get('text')]
            d={'doc_id':uid,'title':meta.get('title') or obj.get('metadata',{}).get('title'),
               'abstract':meta.get('abstract'),'doi':meta.get('doi'),'source':meta['path'],
               'sections':sections,'full_text':'\n'.join(x['text'] for x in sections)}
        else:
            original=old_by_uid[uid]
            d={k:original.get(k) for k in ('doc_id','title','abstract','sections','full_text','source')}
            d['doi']=original.get('metadata',{}).get('doi')
        if d['full_text']: docs.append(d)
    out=PROCESSED/'phase25_eval_docs.jsonl'
    out.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in docs),encoding='utf8')
    doc_ids={d['doc_id'] for d in docs}
    manifest={'snapshot':'2020-07-16','selection_rule':'top 8 topics by collected positive count; first 5 lexicographic positive IDs each; 20 lexicographic old-sample distractors with no positive label for chosen topics',
              'topics':[{'topic_id':q,'query':topics[q]['query'],'question':topics[q]['question'],
                         'official_positive':stats['per_query'][q]['official_positive'],
                         'collected_fulltext_positive':stats['per_query'][q]['collected_fulltext'],
                         'corpus_positive':sum(g>0 and uid in doc_ids for (topic,uid),g in all_qrels.items() if topic==q)} for q in selected_topics],
              'doc_ids':sorted(doc_ids),'documents':len(docs),'positive_source_docs':len(doc_ids-set(distractors)),
              'distractor_docs':len(doc_ids&set(distractors))}
    (EVAL/'phase25_eval_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='doc_ids'},indent=2,ensure_ascii=False))

if __name__=='__main__':main()
