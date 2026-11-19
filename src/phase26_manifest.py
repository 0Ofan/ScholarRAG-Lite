"""Freeze a candidate evaluation corpus with explicit alias exclusions."""
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'data/evaluation'
docs=json.loads((EVAL/'phase26_doc_status.json').read_text(encoding='utf8'))
coverage=json.loads((EVAL/'phase26_coverage.json').read_text(encoding='utf8'))
paths=json.loads((EVAL/'phase26_judged_path_index.json').read_text(encoding='utf8'))
aliases=sorted({uid for ids in paths.values() if len(ids)>1 for uid in ids})
valid={uid for uid,d in docs.items() if d['full_text_status']=='valid_body'}-set(aliases)
qrels=defaultdict(dict)
for line in (EVAL/'trec_covid_complete_qrels.txt').read_text().splitlines():
    q,it,uid,g=line.split()
    if int(g)>=0:qrels[q][uid]=int(g)
doc_qrels=defaultdict(dict)
for q,items in qrels.items():
    for uid,g in items.items():
        if uid in valid:doc_qrels[uid][q]=g
documents=[]
for uid in sorted(valid):
    d=docs[uid]
    documents.append({'doc_id':uid,'cord_uid':uid,'doi':d['doi'],'title':d['title'],
                      'source':d['source_path'],'source_version':d['source_version'],
                      'full_text_status':d['full_text_status'],'license':d['license'],
                      'body_chars':d['body_chars'],'body_entries':d['body_entries'],
                      'sha256':d['sha256'],'query_relevance':doc_qrels[uid]})
queries=[]
for q in sorted(qrels,key=int):
    stats=coverage['per_query'][q]
    available={uid:g for uid,g in qrels[q].items() if uid in valid}
    pos={uid:g for uid,g in available.items() if g>0}
    queries.append({'topic_id':q,'query_text':stats['query_text'],'official_qrels_source':'trec_covid_complete_qrels.txt',
                    'official_positive':stats['official_positive'],'available_positive_ids':pos,
                    'judged_candidate_ids':available,'available_judged':len(available),
                    'available_positive':len(pos),'unjudged_candidate_docs':len(valid)-len(available),
                    'status':'exploratory_candidate','reason':'top-10 unjudged rate on expanded corpus not yet established; pooled qrels incomplete'})
manifest={'snapshot':'CORD-19 2020-07-16','qrels':'TREC-COVID Complete',
          'document_rule':'all qrel-judged UIDs with readable body >=1000 chars; exclude all UIDs sharing a parse path',
          'topic_rule':'retain all 50 official topics; no outcome-based exclusion',
          'formal_query_ids':[],'exploratory_query_ids':[q['topic_id'] for q in queries],
          'excluded_query_ids':[],'excluded_ambiguous_doc_ids':aliases,
          'documents':documents,'queries':queries}
(EVAL/'phase26_candidate_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf8')
print(json.dumps({'documents':len(documents),'queries':len(queries),'excluded_alias_uids':aliases,
                  'positive_pairs':sum(q['available_positive'] for q in queries),'formal_queries':0},indent=2))
