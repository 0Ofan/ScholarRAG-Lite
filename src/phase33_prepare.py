"""Freeze Medium/Large manifests from the Phase 2.7 nested selection."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'data/evaluation';PROC=ROOT/'data/processed'
CONFIG=ROOT/'experiments/configs';RESULT=ROOT/'experiments/results/phase33';RESULT.mkdir(parents=True,exist_ok=True)
load=lambda p:json.loads(p.read_text(encoding='utf8'))
frozen=load(CONFIG/'phase27_experiment_config.json')
full=load(EVAL/'phase27_final_candidate_manifest.json')
pilot=load(PROC/'phase31_pilot_manifest.json')
aliases=load(EVAL/'phase27_document_aliases.json')['uids']
status=load(EVAL/'phase26_doc_status.json')
assert frozen['version']=='phase27-experiment-v1' and full['version']=='phase27-final-v1'
assert len(pilot['documents'])==2000 and frozen['anchor_count']==pilot['positive_anchor_count']==956
pilot_ids={d['doc_id'] for d in pilot['documents']}
docs={d['doc_id']:d for d in full['documents']}
reverse={}
for uid,x in aliases.items():
    if x['decision']=='alias':reverse.setdefault(x['canonical_doc_id'],[]).append(uid)
positive_all={uid for q in full['effective_qrels'].values() for uid,g in q.items() if g>0}
produced={}
for scale,size in (('medium',6000),('large',12000)):
    ids=frozen['scales'][scale]['doc_ids']
    selected=set(ids)
    assert len(ids)==len(selected)==size and pilot_ids<=selected<=set(docs)
    source=[];bad=[]
    for uid in ids:
        d=docs[uid];record=status[uid]
        raw=ROOT/'data/raw/cord19_2020-07-16/phase26/parses'/d['source_path']
        if not raw.is_file() or record['full_text_status']!='valid_body' or record['body_chars']<1000:
            bad.append(uid);continue
        source.append({'doc_id':uid,'original_uid':d['raw_uid'],'aliases':sorted(reverse.get(uid,[])),
                       'title':record['title'],'doi':d['doi'],'source_path':d['source_path'],
                       'source_sha256':d['source_sha256'],'body_chars':d['body_chars'],
                       'source_version':full['source_snapshot']})
    assert not bad and len(source)==size
    qrels={q:{uid:g for uid,g in rows.items() if uid in selected} for q,rows in full['effective_qrels'].items()}
    positive={q:sum(g>0 for g in rows.values()) for q,rows in qrels.items()}
    assert positive==pilot['positive_per_query']==frozen['scales'][scale]['positive_per_query']
    anchors=sum(uid in positive_all for uid in ids)
    assert anchors==956
    manifest={'version':f'phase33-{scale}-v1','parent_manifest':full['version'],
              'selection_config':frozen['version'],'source_snapshot':full['source_snapshot'],
              'qrels_source':'data/evaluation/trec_covid_complete_qrels.txt',
              'selection_rule':frozen['positive_anchor_rule']+'; '+frozen['distractor_rule'],
              'document_count':size,'positive_anchor_count':anchors,
              'positive_query_doc_pairs':sum(positive.values()),'positive_per_query':positive,
              'invalid_or_missing_fulltext_count':len(bad),'documents':source,
              'effective_qrels':qrels,'query_ids':frozen['query_ids']}
    path=PROC/f'phase33_{scale}_manifest.json'
    path.write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf8')
    produced[scale]={'document_count':size,'positive_anchor_count':anchors,
                     'positive_pairs':sum(positive.values()),'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'path':path.relative_to(ROOT).as_posix()}
run_config={'version':'phase33-runs-v1','parent_config':frozen['version'],
            'model_snapshot':frozen['model_snapshots']['bge'],'query_ids':frozen['query_ids'],
            'quality_metrics':frozen['metrics'],'faiss':'IndexFlatIP',
            'runs':[{'run_id':f'phase33_{scale}_{method}','scale':scale,'method':method,
                     'documents':size,'manifest':produced[scale]['path'],
                     'target_tokens':180,'overlap_tokens':20 if method=='fixed' else 0,
                     'soft_maximum_tokens':220 if method=='sentence' else None,
                     'hard_maximum_tokens':240}
                    for scale,size in (('medium',6000),('large',12000))
                    for method in ('fixed','sentence')]}
(CONFIG/'phase33_runs.json').write_text(json.dumps(run_config,ensure_ascii=False,indent=2),encoding='utf8')
(RESULT/'manifest_audit.json').write_text(json.dumps(produced,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(produced,indent=2))
