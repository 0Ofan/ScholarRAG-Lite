"""Incremental checks for frozen baseline and isolated evidence selection."""
import json
import unittest
from pathlib import Path

from src.evidence_selector import baseline, optimized, score_chunk, source_role
from src.phase42_offline import load_inputs, point_checks

ROOT=Path(__file__).resolve().parents[1]


class EvidenceSelectionTests(unittest.TestCase):
    def test_baseline_exact_frozen_top_three(self):
        questions, history=load_inputs()
        for q in questions:
            old=history[q['question_id']]
            now=baseline(old['retrieval_top10'],q)
            self.assertEqual([e['chunk_id'] for e in now], [e['chunk_id'] for e in old['evidence']])
            self.assertEqual([e['text'] for e in now], [e['text'] for e in old['evidence']])
            self.assertEqual([e['doc_id'] for e in now], [e['doc_id'] for e in old['evidence']])

    def test_role_requires_explicit_named_study(self):
        named={'category':'single_paper','question':"According to 'A Named Trial of Masks', what changed?"}
        self.assertEqual(source_role(named,{'title':'A Named Trial of Masks'}),'Primary Evidence')
        self.assertEqual(source_role(named,{'title':'Unrelated Review'}),'Unclassified')
        generic={'category':'single_paper','question':'What did the SARS-CoV-2 study find?'}
        self.assertEqual(source_role(generic,{'title':'SARS-CoV-2 vaccines'}),'Unclassified')

    def test_optimized_preserves_paper_order_and_exact_text(self):
        q={'category':'single_paper','question':'What did the named trial find?'}
        docs=[];pool={}
        for i in range(1,4):
            uid=f'd{i}'
            text=f'The baseline evidence for trial {i}.'
            docs.append({'doc_id':uid,'title':f'Trial {i}','best_chunk_id':f'{uid}:fixed:0:0',
                         'best_chunk_text':text,'score':.8,'source_path':'source',
                         'source_spans':[{'section_index':0,'raw_char_start':0,'raw_char_end':len(text)}]})
            pool[uid]=[{'doc_id':uid,'title':f'Trial {i}','chunk_id':f'{uid}:fixed:0:0',
                        'text':text,'source_path':'source','source_spans':docs[-1]['source_spans'],
                        'section_title':'Abstract','section_index':0,'token_count':15}]
        result=optimized(docs,q,pool)
        self.assertEqual([x['doc_id'] for x in result],['d1','d2','d3'])
        self.assertEqual([x['text'] for x in result],[d['best_chunk_text'] for d in docs])
        self.assertEqual(len({x['chunk_id'] for x in result}),3)
        self.assertEqual([x['evidence_id'] for x in result],['E1','E2','E3'])

    def test_lexical_leads_never_claim_semantic_coverage(self):
        item={'chunk_id':'d1:fixed:0:0','doc_id':'d1','text':'COVID-19 model result.'}
        check=point_checks(['The result is clinically valid.'],[item],[item])[0]
        self.assertEqual(check['baseline_semantic_coverage'],'Unverified')
        self.assertEqual(check['optimized_semantic_coverage'],'Unverified')
        self.assertGreater(score_chunk('What result?',{'text':'The result was significant.',
                            'section_title':'Results'}), 0)

    def test_saved_offline_artifacts_keep_history_isolated(self):
        data=json.loads((ROOT/'experiments/results/phase42/evidence_comparison.json').read_text(encoding='utf8'))
        self.assertEqual(len(data['comparison']),20)
        for q in data['comparison']:
            self.assertEqual(len(q['baseline']),3)
            self.assertEqual(len(q['optimized']),3)
            self.assertEqual([e['doc_id'] for e in q['baseline']],[e['doc_id'] for e in q['optimized']])
            self.assertEqual(len(q['point_checks']),len(q['expected_answer_points']))
            self.assertTrue(all(x['baseline_semantic_coverage']=='Unverified' and
                                x['optimized_semantic_coverage']=='Unverified' for x in q['point_checks']))


if __name__=='__main__':
    unittest.main()
