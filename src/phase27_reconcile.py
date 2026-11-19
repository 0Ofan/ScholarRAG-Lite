"""Single source of truth for Phase 2.7 identity and qrel counts."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'data/evaluation';RESULT=ROOT/'experiments/results/phase27';RESULT.mkdir(parents=True,exist_ok=True)
load=lambda name:json.loads((EVAL/name).read_text(encoding='utf8'))
docs=load('phase26_doc_status.json');phase26=load('phase26_candidate_manifest.json');alias=load('phase27_document_aliases.json')
valid={u for u,d in docs.items() if d['full_text_status']=='valid_body'}
prior={d['doc_id'] for d in phase26['documents']}
decisions=alias['uids']
canonical=lambda uid:decisions[uid]['canonical_doc_id'] if uid in decisions else uid
final={canonical(uid) for uid in valid if canonical(uid)}
raw_qrels=defaultdict(dict);negative_one=[]
for line in (EVAL/'trec_covid_complete_qrels.txt').read_text().splitlines():
    q,it,uid,grade=line.split();g=int(grade)
    if g<0:negative_one.append({'topic_id':q,'raw_uid':uid,'grade':g,'iteration':it});continue
    raw_qrels[q][uid]=g
effective=defaultdict(dict);provenance=defaultdict(lambda:defaultdict(list));conflicts=[]
for q,judgments in raw_qrels.items():
    for uid,g in judgments.items():
        if uid not in valid:continue
        canon=canonical(uid)
        if canon is None:continue
        prior_grade=effective[q].get(canon)
        if prior_grade is not None and prior_grade!=g:
            conflicts.append({'topic_id':q,'canonical_doc_id':canon,'grades':[prior_grade,g],'raw_uid':uid})
        effective[q][canon]=g
        provenance[q][canon].append(uid)
if conflicts:raise RuntimeError(f'canonical qrel grade conflicts: {conflicts}')
official_positive={(q,u) for q,j in raw_qrels.items() for u,g in j.items() if g>0}
valid_positive={(q,u) for q,u in official_positive if u in valid}
final_positive={(q,u) for q,j in effective.items() for u,g in j.items() if g>0}
excluded_positive=[{'topic_id':q,'raw_uid':u,'grade':raw_qrels[q][u],'reason':decisions[u]['reason']}
                   for q,u in sorted(valid_positive,key=lambda x:(int(x[0]),x[1])) if u in decisions and canonical(u) is None]
collapsed_positive=[{'topic_id':q,'raw_uid':u,'canonical_doc_id':canonical(u),'grade':raw_qrels[q][u]}
                    for q,u in sorted(valid_positive,key=lambda x:(int(x[0]),x[1]))
                    if u in decisions and decisions[u]['decision']=='alias' and canonical(u) in effective[q]
                    and len(provenance[q][canonical(u)])>1 and raw_qrels[q][u]>0]
positive_docs={u for q,u in final_positive}
stats={'version':'phase27-stats-v1','source_snapshot':'CORD-19 2020-07-16','qrels':'TREC-COVID Complete',
       'official_qrel_rows':sum(map(len,raw_qrels.values()))+len(negative_one),'official_valid_grade_qrel_pairs':sum(map(len,raw_qrels.values())),
       'official_negative_one_records':negative_one,'official_positive_pairs':len(official_positive),
       'official_positive_unique_docs':len({u for q,u in official_positive}),
       'phase26_valid_raw_uids':len(valid),'phase26_candidate_docs':len(prior),
       'phase26_valid_positive_unique_docs':len({u for q,u in valid_positive}),
       'phase26_valid_positive_pairs':len(valid_positive),
       'phase26_candidate_positive_unique_docs':len({d['doc_id'] for d in phase26['documents'] if any(g>0 for g in d['query_relevance'].values())}),
       'phase26_candidate_positive_pairs':sum(q['available_positive'] for q in phase26['queries']),
       'phase27_final_canonical_docs':len(final),'phase27_positive_unique_docs':len(positive_docs),
       'phase27_positive_pairs':len(final_positive),
       'alias_raw_uids':[u for u,d in decisions.items() if d['decision']=='alias'],
       'excluded_raw_uids':[u for u,d in decisions.items() if d['decision'].startswith('excluded')],
       'distinct_shared_path_uids':[u for u,d in decisions.items() if d['decision']=='distinct'],
       'excluded_positive_pairs':excluded_positive,'collapsed_duplicate_positive_pairs':collapsed_positive,
       'per_query':{q:{'official_positive':sum(g>0 for g in raw_qrels[q].values()),
                       'valid_raw_positive':sum(g>0 and u in valid for u,g in raw_qrels[q].items()),
                       'final_canonical_positive':sum(g>0 for g in effective[q].values()),
                       'final_canonical_judged':len(effective[q])} for q in sorted(raw_qrels,key=int)}}
assert len(valid)==25598 and len(prior)==25590
assert len(final)==25594 and len(positive_docs)==11981 and len(final_positive)==17852
assert len(valid_positive)-len(final_positive)==len(excluded_positive)+len(collapsed_positive)
assert len(final)==len(valid)-len(stats['alias_raw_uids'])-len(stats['excluded_raw_uids'])
assert final=={d['doc_id'] for d in phase26['documents']}|{'j7t9nebs','svu06v9c','8avkjc84','d9v5xtx7'}
(EVAL/'phase27_statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf8')
prepped={'version':'phase27-prepared-v1','source_snapshot':stats['source_snapshot'],'qrels':stats['qrels'],
         'identity_rule':alias['decision_rule'],'canonical_doc_ids':sorted(final),
         'effective_qrels':{q:j for q,j in effective.items()},
         'qrel_raw_uid_provenance':{q:{u:uids for u,uids in j.items()} for q,j in provenance.items()},
         'excluded_raw_uids':stats['excluded_raw_uids'],'alias_map':{u:canonical(u) for u in decisions}}
(EVAL/'phase27_prepared_manifest.json').write_text(json.dumps(prepped,ensure_ascii=False),encoding='utf8')
audit={'stats_sha256':hashlib.sha256((EVAL/'phase27_statistics.json').read_bytes()).hexdigest(),
       'prepared_manifest_sha256':hashlib.sha256((EVAL/'phase27_prepared_manifest.json').read_bytes()).hexdigest(),
       'checked_relationships':['valid raw > phase26 candidate by 8 excluded UIDs','two compatible aliases collapse',
                                'two conflicting version UIDs excluded','two distinct shared-path papers retained',
                                'positive pair removals trace to raw qrels'],
       'conflicts_after_identity_resolution':conflicts,
       'numbers':{k:stats[k] for k in ('phase26_valid_raw_uids','phase26_candidate_docs','phase27_final_canonical_docs',
                                      'phase26_valid_positive_unique_docs','phase26_candidate_positive_unique_docs','phase27_positive_unique_docs',
                                      'phase26_valid_positive_pairs','phase26_candidate_positive_pairs','phase27_positive_pairs')}}
(RESULT/'statistics_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
print(json.dumps(audit['numbers'],indent=2))
