"""Focused tests for the standalone paper-level BM25 baseline."""
import json
import math
import unittest

from src.evaluation import score_topic
from src.phase43_bm25 import BM25Index, MANIFEST, OUT
from src.phase43_evaluate import complement


class BM25Tests(unittest.TestCase):
    def test_default_formula_and_technical_tokens(self):
        index=BM25Index.from_texts(['SARS-CoV-2 mRNA-1273 15-minute assay',
                                    'SARS-CoV-2 unrelated',
                                    'influenza clinical'], ['a','b','c'])
        self.assertIn('sars-cov-2',index.vocabulary)
        self.assertIn('mrna-1273',index.vocabulary)
        self.assertIn('15-minute',index.vocabulary)
        hits=index.search('mRNA-1273 assay',3)
        self.assertEqual(hits[0]['doc_id'],'a')
        self.assertGreater(hits[0]['score'],0)
        self.assertEqual(hits[1]['score'],0)
        self.assertEqual([h['doc_id'] for h in hits[1:]],['b','c'])
        tf=1;dl=4;avg=index.avgdl;df=1;n=3
        expected=math.log1p((n-df+.5)/(df+.5))*tf*2.5/(tf+1.5*(.25+.75*dl/avg))
        self.assertAlmostEqual(index.score('assay')[0],expected)

    def test_reload_and_frozen_id_mapping(self):
        index=BM25Index.load(OUT)
        manifest=json.loads(MANIFEST.read_text(encoding='utf8'))
        self.assertEqual(index.doc_ids,[d['doc_id'] for d in manifest['documents']])
        stored=[json.loads(line) for line in (OUT/'retrieval_results.jsonl').read_text(encoding='utf8').splitlines()]
        self.assertEqual(len(stored),50)
        for item in (stored[0],stored[21],stored[33],stored[-1]):
            hits=index.search(item['query'],100)
            self.assertEqual([h['doc_id'] for h in hits],[h['doc_id'] for h in item['top100']])
            self.assertEqual(len({h['doc_id'] for h in hits}),100)

    def test_frozen_metric_and_unjudged_handling(self):
        corpus=['a','b','c']
        judgments={'a':2,'b':0}
        result=score_topic(['c','a','b'],judgments,corpus)
        self.assertEqual(result['unjudged_in_top10'],1)
        self.assertEqual(result['known_relevant_in_corpus'],1)
        self.assertEqual(result['recall@10'],1)
        self.assertEqual(result['mrr'],.5)
        saved=json.loads((OUT/'metrics.json').read_text(encoding='utf8'))
        self.assertEqual(saved['query_count'],50)
        self.assertEqual(saved['known_positive_pairs'],1560)

    def test_positive_complement_does_not_count_unknown_as_negative(self):
        data=complement(['a','x'],['b','x'],{'a','b','c'})
        self.assertEqual({k:data[k] for k in ('bm25_only','bge_only','both','neither')},
                         {'bm25_only':1,'bge_only':1,'both':0,'neither':1})
        report=json.loads((OUT/'comparison.json').read_text(encoding='utf8'))
        for k in ('top10','top100'):
            counts=report['known_positive_complement_totals'][k]
            self.assertEqual(sum(counts.values()),1560)


if __name__=='__main__':
    unittest.main()
