"""Freeze the Phase 3.1 Pilot from Phase 2.7 without resampling."""
import hashlib
import json
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'data/evaluation';PROC=ROOT/'data/processed';RESULT=ROOT/'experiments/results/phase31_pilot'
RESULT.mkdir(parents=True,exist_ok=True)
load=lambda p:json.loads(p.read_text(encoding='utf8'))
config=load(ROOT/'experiments/configs/phase27_experiment_config.json')
full=load(EVAL/'phase27_final_candidate_manifest.json')
aliases=load(EVAL/'phase27_document_aliases.json')['uids']
status=load(EVAL/'phase26_doc_status.json')
selection=config['scales']['pilot']['doc_ids']
assert config['version']=='phase27-experiment-v1' and full['version']=='phase27-final-v1'
assert len(selection)==len(set(selection))==2000 and config['anchor_count']==956
full_docs={d['doc_id']:d for d in full['documents']}
assert set(selection)<=set(full_docs)
alias_reverse={}
for uid,info in aliases.items():
    if info['decision']=='alias':alias_reverse.setdefault(info['canonical_doc_id'],[]).append(uid)
docs=[];bad=[]
t=time.perf_counter()
for uid in selection:
    d=full_docs[uid]
    raw=ROOT/'data/raw/cord19_2020-07-16/phase26/parses'/d['source_path']
    if not raw.is_file() or status[uid]['full_text_status']!='valid_body' or status[uid]['body_chars']<1000:
        bad.append(uid);continue
    docs.append({'doc_id':uid,'original_uid':d['raw_uid'],'aliases':sorted(alias_reverse.get(uid,[])),
                 'title':status[uid]['title'],'doi':d['doi'],'source_path':d['source_path'],
                 'source_sha256':d['source_sha256'],'body_chars':d['body_chars'],
                 'source_version':full['source_snapshot']})
assert not bad and len(docs)==2000
positive_any={uid for q in full['effective_qrels'].values() for uid,g in q.items() if g>0}
qrels={q:{uid:g for uid,g in judged.items() if uid in selection} for q,judged in full['effective_qrels'].items()}
counts={q:sum(g>0 for g in judged.values()) for q,judged in qrels.items()}
assert counts==config['scales']['pilot']['positive_per_query']
anchors={uid for uid in selection if uid in positive_any}
assert len(anchors)==956
manifest={'version':'phase31-pilot-v1','parent_manifest':'phase27-final-v1',
          'source_snapshot':full['source_snapshot'],'qrels_source':'data/evaluation/trec_covid_complete_qrels.txt',
          'selection_rule':config['positive_anchor_rule']+'; '+config['distractor_rule'],
          'document_count':len(docs),'positive_anchor_count':len(anchors),
          'positive_query_doc_pairs':sum(counts.values()),'positive_per_query':counts,
          'invalid_or_missing_fulltext_count':len(bad),'documents':docs,
          'effective_qrels':qrels,'query_ids':config['query_ids']}
path=PROC/'phase31_pilot_manifest.json'
path.write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf8')
(RESULT/'manifest_build.json').write_text(json.dumps({'seconds':time.perf_counter()-t,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'documents':len(docs),
            'positive_anchors':len(anchors),'positive_pairs':sum(counts.values())},indent=2),encoding='utf8')
print(json.dumps({'documents':len(docs),'anchors':len(anchors),'positive_pairs':sum(counts.values()),
                  'invalid':len(bad)},indent=2))
