"""Freeze canonical corpus and nested, fixed-positive scale subsets."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];EVAL=ROOT/'data/evaluation'
CONFIG=ROOT/'experiments/configs';CONFIG.mkdir(parents=True,exist_ok=True)
stats=json.loads((EVAL/'phase27_statistics.json').read_text(encoding='utf8'))
prepared=json.loads((EVAL/'phase27_prepared_manifest.json').read_text(encoding='utf8'))
status=json.loads((EVAL/'phase26_doc_status.json').read_text(encoding='utf8'))
all_ids=set(prepared['canonical_doc_ids']);qrels=prepared['effective_qrels']
assert len(all_ids)==stats['phase27_final_canonical_docs']
key=lambda uid:hashlib.sha256(('phase27-v1:'+uid).encode()).hexdigest()
anchors=set()
for q in sorted(qrels,key=int):
    candidates=sorted((uid for uid,g in qrels[q].items() if g>0),key=key)
    assert len(candidates)>=25
    anchors.update(candidates[:25])
positive_any={uid for judgments in qrels.values() for uid,g in judgments.items() if g>0}
distractors=sorted(all_ids-positive_any,key=key)
assert len(anchors)<=2000 and len(distractors)>=12000-len(anchors)
scales={}
for name,size in (('pilot',2000),('medium',6000),('large',12000)):
    selected=set(anchors)|set(distractors[:size-len(anchors)])
    assert len(selected)==size
    scales[name]={'document_count':size,'doc_ids':sorted(selected),
                  'positive_per_query':{q:sum(uid in selected and g>0 for uid,g in qrels[q].items()) for q in qrels},
                  'judged_per_query':{q:sum(uid in selected for uid in qrels[q]) for q in qrels}}
assert set(scales['pilot']['doc_ids'])<set(scales['medium']['doc_ids'])<set(scales['large']['doc_ids'])
for q in qrels:
    assert scales['pilot']['positive_per_query'][q]==scales['medium']['positive_per_query'][q]==scales['large']['positive_per_query'][q]
documents=[{'doc_id':uid,'raw_uid':uid,'source_path':status[uid]['source_path'],
            'source_sha256':status[uid]['sha256'],'doi':status[uid]['doi'],'body_chars':status[uid]['body_chars']}
           for uid in sorted(all_ids)]
manifest={'version':'phase27-final-v1','source_snapshot':'CORD-19 2020-07-16',
          'official_qrels':'TREC-COVID Complete, unchanged on disk',
          'selection_rule':'valid judged full text >=1000 body chars; two exact duplicate alias merges; two conflicting-version UIDs excluded; two distinct papers retained',
          'identity_decisions':'phase27_document_aliases.json',
          'statistics':'phase27_statistics.json','document_count':len(documents),
          'positive_unique_docs':stats['phase27_positive_unique_docs'],
          'positive_query_doc_pairs':stats['phase27_positive_pairs'],
          'query_ids':[str(q) for q in range(1,51)],'documents':documents,
          'effective_qrels':qrels,'qrel_raw_uid_provenance':prepared['qrel_raw_uid_provenance']}
(EVAL/'phase27_final_candidate_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf8')
config={'version':'phase27-experiment-v1','status':'protocol frozen; formal scale runs require Phase 3 approval',
        'source_snapshot':'CORD-19 2020-07-16','corpus_manifest':'data/evaluation/phase27_final_candidate_manifest.json',
        'identity_map':'data/evaluation/phase27_document_aliases.json',
        'qrels':'data/evaluation/trec_covid_complete_qrels.txt','query_ids':[str(q) for q in range(1,51)],
        'query_text':'topics.jsonl query field, unchanged','positive_anchor_rule':'first 25 canonical positive IDs per topic by SHA256(phase27-v1:UID); union shared by all scales',
        'distractor_rule':'all canonical docs with no positive qrel in any topic, ordered SHA256(phase27-v1:UID); take prefix',
        'anchor_count':len(anchors),'distractor_pool_count':len(distractors),
        'scales':scales,'chunking':{'common':'title, abstract, body; whitespace normalization; source spans; merge short sections <40 BGE tokens',
                    'fixed':{'target_tokens':180,'overlap_tokens':20,'maximum_tokens':240,'minimum_tail_tokens':40},
                    'sentence':{'target_tokens':180,'soft_maximum_tokens':220,'hard_maximum_tokens':240,'minimum_tail_tokens':40}},
        'embedding_models':['BAAI/bge-small-en-v1.5','sentence-transformers/all-MiniLM-L6-v2'],
        'model_snapshots':{'bge':'5c38ec7c405ec4b44b94cc5a9bb96e735b38267a',
                           'minilm':'1110a243fdf4706b3f48f1d95db1a4f5529b4d41'},
        'primary_model':'BAAI/bge-small-en-v1.5','similarity':'normalized cosine via FAISS IndexFlatIP',
        'metrics':['nDCG@10','Recall@10','Recall@100','MRR'],
        'quality_repetitions':1,'performance_repetitions':3,
        'limitation':'nested scale sets fix 25 selected positives per query; results are conditional and not full TREC-COVID recall'}
(CONFIG/'phase27_experiment_config.json').write_text(json.dumps(config,ensure_ascii=False),encoding='utf8')
print(json.dumps({'manifest_docs':len(documents),'anchors':len(anchors),'distractors':len(distractors),
                  'scale_docs':{k:v['document_count'] for k,v in scales.items()},
                  'positive_per_query_range':[min(scales['pilot']['positive_per_query'].values()),
                                               max(scales['pilot']['positive_per_query'].values())]},indent=2))
