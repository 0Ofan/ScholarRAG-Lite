"""Incremental tests for deterministic Top-100 RRF over frozen paper IDs."""
import json
import unittest

from src.evaluation import score_topic
from src.phase44_rrf import MANIFEST, OUT, RRF_K, fuse


class RRFTests(unittest.TestCase):
    def test_two_sided_and_single_sided_formula(self):
        rows=fuse(['d1','d2'],['d2','d3'],limit=3)
        by_id={x['doc_id']:x for x in rows}
        self.assertAlmostEqual(by_id['d2']['rrf_score'],1/(60+2)+1/(60+1))
        self.assertAlmostEqual(by_id['d1']['rrf_score'],1/(60+1))
        self.assertAlmostEqual(by_id['d3']['rrf_score'],1/(60+2))
        self.assertIsNone(by_id['d1']['bm25_rank'])
        self.assertIsNone(by_id['d3']['bge_rank'])
        self.assertEqual(rows[0]['doc_id'],'d2')

    def test_tie_break_and_top100(self):
        self.assertEqual([x['doc_id'] for x in fuse(['z'],['a'])],['a','z'])
        left=[f'a{i:03}' for i in range(100)]
        right=[f'b{i:03}' for i in range(100)]
        result=fuse(left,right)
        self.assertEqual(len(result),100)
        self.assertEqual(len({r['doc_id'] for r in result}),100)

    def test_saved_results_reproduce_and_match_frozen_corpus(self):
        manifest=json.loads(MANIFEST.read_text(encoding='utf8'))
        corpus={d['doc_id'] for d in manifest['documents']}
        saved=[json.loads(x) for x in (OUT/'hybrid_results.jsonl').read_text(encoding='utf8').splitlines()]
        self.assertEqual(len(saved),50)
        for item in saved:
            expected=fuse(item['input_bge_top100'],item['input_bm25_top100'])
            actual=item['hybrid_top100']
            self.assertEqual([x['doc_id'] for x in actual],[x['doc_id'] for x in expected])
            self.assertEqual([x['rrf_score'] for x in actual],[x['rrf_score'] for x in expected])
            self.assertTrue({x['doc_id'] for x in actual}<=corpus)
        q=saved[0]
        metrics=score_topic([x['doc_id'] for x in q['hybrid_top100']],
                            manifest['effective_qrels'][q['topic_id']],corpus)
        self.assertIsNotNone(metrics['ndcg@10'])
        self.assertEqual(metrics['known_relevant_in_corpus'],manifest['positive_per_query'][q['topic_id']])

    def test_positive_partition_and_unjudged(self):
        data=json.loads((OUT/'complementarity.json').read_text(encoding='utf8'))
        self.assertEqual(data['known_positive_pairs'],1560)
        for name in ('input_categories','hybrid_top100_retained','hybrid_top100_lost'):
            self.assertEqual(set(data['categories_total'][name]),{'bge_only','bm25_only','both','neither'})
        for category,n in data['categories_total']['input_categories'].items():
            self.assertEqual(n, data['categories_total']['hybrid_top100_retained'][category]
                               +data['categories_total']['hybrid_top100_lost'][category])
        self.assertEqual(sum(data['categories_total']['input_categories'].values()),1560)
        metrics=json.loads((OUT/'metrics.json').read_text(encoding='utf8'))
        self.assertEqual(metrics['rrf_k'],RRF_K)
        self.assertEqual(metrics['query_count'],50)
        self.assertGreater(metrics['quality']['hybrid']['top10_unjudged'],0)


if __name__=='__main__':
    unittest.main()
