"""Incremental tests for fixed Top-3 preserving RRF and exact RAG reuse."""
import inspect
import json
import unittest

from src.evaluation import score_topic
from src.phase44_rrf import MANIFEST, fuse
from src.phase46_protected_rrf import OUT, protected_rrf


class ProtectedRRFTests(unittest.TestCase):
    def test_protected_top3_and_constrained_order(self):
        bge=[f'b{i:03}' for i in range(100)]
        bm25=[f'b{i:03}' for i in range(20,100)]+[f'c{i:03}' for i in range(20)]
        result=protected_rrf(bge,bm25)
        self.assertEqual(len(result),100)
        self.assertEqual(len({x['doc_id'] for x in result}),100)
        self.assertTrue(set(bge[:3])<={x['doc_id'] for x in result[:10]})
        self.assertEqual(sum(x['protected'] for x in result),3)
        self.assertEqual([x['rank'] for x in result],list(range(1,101)))
        self.assertEqual(result[:10],sorted(result[:10],key=lambda x:(-x['rrf_score'],x['doc_id'])))
        self.assertEqual(result[10:],sorted(result[10:],key=lambda x:(-x['rrf_score'],x['doc_id'])))
        original={x['doc_id']:x for x in fuse(bge,bm25,limit=200)}
        for row in result:
            before=original[row['doc_id']]
            self.assertEqual(row['rrf_score'],before['rrf_score'])
            self.assertEqual((row['bge_rank'],row['bm25_rank']),
                             (before['bge_rank'],before['bm25_rank']))

    def test_pure_rule_and_tie_break(self):
        self.assertNotIn('qrel',inspect.getsource(protected_rrf).lower())
        bge=['z']+[f'x{i:03}' for i in range(1,100)]
        bm25=['a']+[f'y{i:03}' for i in range(1,100)]
        rows=protected_rrf(bge,bm25)
        self.assertTrue({'z','x001','x002'}<={x['doc_id'] for x in rows[:10]})
        tied=[x for x in rows[:10] if x['rrf_score']==1/61]
        self.assertEqual([x['doc_id'] for x in tied],sorted(x['doc_id'] for x in tied))

    def test_fifty_saved_queries_and_frozen_evaluation(self):
        manifest=json.loads(MANIFEST.read_text(encoding='utf8'))
        corpus={x['doc_id'] for x in manifest['documents']}
        records=[json.loads(line) for line in (OUT/'protected_results.jsonl').read_text(encoding='utf8').splitlines()]
        self.assertEqual(len(records),50)
        for record in records:
            qid=record['topic_id']
            expected=protected_rrf(record['input_bge_top100'],record['input_bm25_top100'])
            actual=record['protected_top100']
            self.assertEqual([x['doc_id'] for x in actual],[x['doc_id'] for x in expected])
            self.assertEqual([x['rrf_score'] for x in actual],[x['rrf_score'] for x in expected])
            self.assertTrue(set(record['input_bge_top100'][:3])<=
                            {x['doc_id'] for x in actual[:10]})
            self.assertTrue({x['doc_id'] for x in actual}<=corpus)
            self.assertEqual(sum(x['protected'] for x in actual),3)
            metrics=score_topic([x['doc_id'] for x in actual],
                                manifest['effective_qrels'][qid],corpus)
            self.assertEqual(metrics['known_relevant_in_corpus'],manifest['positive_per_query'][qid])
            for item in actual:
                grade=manifest['effective_qrels'][qid].get(item['doc_id'])
                self.assertEqual(item['judgment'],grade)
                self.assertEqual(item['judgment_status'],'unjudged' if grade is None else
                                 'positive' if grade>=1 else 'grade_0')

    def test_reuse_gate_and_complementarity(self):
        gate=json.loads((OUT/'rag_gate.json').read_text(encoding='utf8'))['conditions']
        self.assertTrue(all(gate.values()))
        comp=json.loads((OUT/'complementarity.json').read_text(encoding='utf8'))
        self.assertEqual(comp['method_totals']['standard']['top100'],
                         comp['method_totals']['protected']['top100'])
        rag=json.loads((OUT/'rag_validation.json').read_text(encoding='utf8'))
        self.assertEqual(rag['question_count'],20)
        self.assertEqual(rag['reused_real_api_answers'],40)
        self.assertEqual(rag['new_api_calls'],0)
        self.assertEqual(len(rag['rows']),40)
        self.assertTrue(all(x['real_api'] and not x['new_api_call'] for x in rag['rows']))


if __name__=='__main__':unittest.main()
