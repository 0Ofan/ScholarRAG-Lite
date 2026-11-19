"""Classify full-text quality and report every TREC topic's coverage."""
import json
import statistics
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'data/evaluation'
records=json.loads((EVAL/'phase26_judged_records.json').read_text(encoding='utf8'))
archive=json.loads((EVAL/'phase26_archive_paths.json').read_text(encoding='utf8'))
valid_status={'valid_body'};readable={'valid_body','short_body','empty_body'}
docs={}
for uid,meta in records.items():
    paths=meta['pmc']+meta['pdf']
    states=[archive.get(path,{'status':'missing_archive_member'})['status'] for path in paths]
    chosen=next((p for p in paths if archive.get(p,{}).get('status')=='valid_body'),None)
    if chosen:status='valid_body'
    elif not paths:status='no_parse_path'
    elif any(s=='short_body' for s in states):status='short_body'
    elif any(s=='empty_body' for s in states):status='empty_body'
    elif any(s=='read_or_parse_error' for s in states):status='read_or_parse_error'
    elif any(s=='oversize' for s in states):status='oversize'
    else:status='missing_archive_member'
    if not chosen:chosen=next((p for p in paths if archive.get(p,{}).get('status') in readable),None)
    info=archive.get(chosen,{}) if chosen else {}
    docs[uid]={'doc_id':uid,'cord_uid':uid,'doi':meta['doi'][0] if meta['doi'] else None,
               'title':meta['title'][0] if meta['title'] else None,'license':meta['license'][0] if meta['license'] else None,
               'source_path':chosen,'source_version':'CORD-19 2020-07-16','full_text_status':status,
               'body_chars':info.get('body_chars',0),'body_entries':info.get('body_entries',0),
               'paper_id':info.get('paper_id'),'sha256':info.get('sha256'),
               'has_abstract':bool(meta['abstract']) or bool(info.get('abstract_entries')),
               'path_count':len(paths),'has_pmc':bool(meta['pmc']),'has_pdf':bool(meta['pdf']),
               'path_found':any(s!='missing_archive_member' for s in states),
               'path_readable':any(s in readable for s in states)}
qrels=defaultdict(dict)
for line in (EVAL/'trec_covid_complete_qrels.txt').read_text().splitlines():
    q,it,uid,g=line.split()
    if int(g)>=0:qrels[q][uid]=int(g)
topics={x['topic_id']:x for x in map(json.loads,(EVAL/'topics.jsonl').read_text(encoding='utf8').splitlines())}
valid_ids={uid for uid,x in docs.items() if x['full_text_status']=='valid_body'}
positive_ids={uid for q in qrels.values() for uid,g in q.items() if g>0}
per_query={}
for q,judged in sorted(qrels.items(),key=lambda kv:int(kv[0])):
    positives={uid for uid,g in judged.items() if g>0}
    with_path={uid for uid in positives if docs[uid]['path_count']}
    found={uid for uid in positives if docs[uid]['path_found']}
    readable_ids={uid for uid in positives if docs[uid]['path_readable']}
    valid=positives&valid_ids
    judged_in=valid_ids&set(judged)
    per_query[q]={'topic_id':q,'query_text':topics[q]['query'],'official_positive':len(positives),
                  'positive_with_path':len(with_path),'positive_path_found':len(found),
                  'positive_file_readable':len(readable_ids),'positive_valid_fulltext':len(valid),
                  'positive_valid_coverage':len(valid)/len(positives) if positives else None,
                  'valid_positive_ids':sorted(valid),'judged_valid_docs':len(judged_in),
                  'judged_negative_valid_docs':sum(judged[uid]==0 for uid in judged_in),
                  'unjudged_valid_docs':len(valid_ids)-len(judged_in),
                  'qrels_coverage_in_valid_corpus':len(judged_in)/len(valid_ids) if valid_ids else None}
positive_docs={uid:docs[uid] for uid in positive_ids}
chars=[d['body_chars'] for d in positive_docs.values() if d['full_text_status']=='valid_body']
entries=[d['body_entries'] for d in positive_docs.values() if d['full_text_status']=='valid_body']
summary={'official_positive_unique':len(positive_ids),'positive_status_counts':dict(Counter(d['full_text_status'] for d in positive_docs.values())),
         'positive_parse_path':sum(d['path_count']>0 for d in positive_docs.values()),
         'positive_file_found':sum(d['path_found'] for d in positive_docs.values()),
         'positive_file_readable':sum(d['path_readable'] for d in positive_docs.values()),
         'positive_valid_body':sum(d['full_text_status']=='valid_body' for d in positive_docs.values()),
         'judged_valid_body_unique':len(valid_ids),'judged_valid_status_counts':dict(Counter(d['full_text_status'] for d in docs.values())),
         'positive_missing_title':sum(not d['title'] for d in positive_docs.values() if d['full_text_status']=='valid_body'),
         'positive_missing_abstract':sum(not d['has_abstract'] for d in positive_docs.values() if d['full_text_status']=='valid_body'),
         'positive_body_chars_mean':statistics.mean(chars) if chars else None,
         'positive_body_chars_median':statistics.median(chars) if chars else None,
         'positive_body_chars_p95':sorted(chars)[int(.95*(len(chars)-1))] if chars else None,
         'positive_body_entries_median':statistics.median(entries) if entries else None,
         'positive_single_body_entry':sum(x<=1 for x in entries),
         'positive_valid_unique_raw_sha256':len({d['sha256'] for d in positive_docs.values() if d['full_text_status']=='valid_body'}),
         'positive_selected_formats':dict(Counter('pmc_json' if '/pmc_json/' in d['source_path'] else 'pdf_json' for d in positive_docs.values() if d['full_text_status']=='valid_body')),
         'queries_with_1':sum(x['positive_valid_fulltext']>=1 for x in per_query.values()),
         'queries_with_5':sum(x['positive_valid_fulltext']>=5 for x in per_query.values()),
         'queries_with_10':sum(x['positive_valid_fulltext']>=10 for x in per_query.values()),
         'queries_with_30':sum(x['positive_valid_fulltext']>=30 for x in per_query.values()),
         'per_query':per_query}
(EVAL/'phase26_doc_status.json').write_text(json.dumps(docs,ensure_ascii=False),encoding='utf8')
(EVAL/'phase26_coverage.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in summary.items() if k!='per_query'},indent=2,ensure_ascii=False))
