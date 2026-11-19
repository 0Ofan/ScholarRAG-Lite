import json
import unittest
from pathlib import Path

from transformers import AutoTokenizer
from src.phase27_chunking import make_chunks
from src.evaluation import document_ranking, score_topic

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'data/evaluation'

class Phase27Tests(unittest.TestCase):
 def test_aliases_and_canonical_qrels(self):
    aliases=json.loads((EVAL/'phase27_document_aliases.json').read_text(encoding='utf8'))['uids']
    manifest=json.loads((EVAL/'phase27_final_candidate_manifest.json').read_text(encoding='utf8'))
    ids={d['doc_id'] for d in manifest['documents']}
    assert len(ids)==25594
    assert aliases['dtnuet6c']['canonical_doc_id']=='j7t9nebs'
    assert aliases['6rpt47gm']['canonical_doc_id']=='svu06v9c'
    assert aliases['g7qioapl']['canonical_doc_id'] is None
    assert aliases['xa6kwguo']['canonical_doc_id'] is None
    assert {'8avkjc84','d9v5xtx7'}<=ids
    assert manifest['effective_qrels']['35']['j7t9nebs']==2
    assert 'dtnuet6c' not in ids

 def test_chunking_handles_long_unpunctuated_section_and_retains_origin(self):
    path=ROOT/'models/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
    tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
    raw=' '.join(['coronavirus transmission']*600)
    body=[{'section':'Long','text':raw},{'section':'Short','text':'Brief but useful detail.'}]
    for strategy in ('fixed','sentence'):
        chunks,stats=make_chunks('paper',body,tok,tok,strategy,'example.json')
        assert len(chunks)>2
        assert all(0<c['tokens_bge']<=240 and c['source_spans'] for c in chunks)
        assert any(span['section_index']==1 for c in chunks for span in c['source_spans'])
        assert stats['merged_short_sections']>=1
    assert make_chunks('paper',body,tok,tok,'sentence','example.json')[1]['long_sentence_splits']>=1

 def test_document_metrics_do_not_duplicate_chunks_or_aliases(self):
    hits=[{'doc_id':'a','score':.9},{'doc_id':'a','score':.8},{'doc_id':'b','score':.7}]
    rank=document_ranking(hits)
    assert rank==['a','b']
    metrics=score_topic(rank,{'a':2,'b':0},['a','b'])
    assert metrics['recall@10']==1
    assert metrics['ndcg@10']==1

 def test_frozen_scales_have_same_known_positive_denominator(self):
    config=json.loads((ROOT/'experiments/configs/phase27_experiment_config.json').read_text(encoding='utf8'))
    scopes=config['scales']
    assert set(scopes['pilot']['doc_ids'])<set(scopes['medium']['doc_ids'])<set(scopes['large']['doc_ids'])
    assert [len(scopes[x]['doc_ids']) for x in ('pilot','medium','large')]==[2000,6000,12000]
    for q in map(str,range(1,51)):
        counts=[scopes[x]['positive_per_query'][q] for x in ('pilot','medium','large')]
        assert counts[0]>=25 and counts[0]==counts[1]==counts[2]
