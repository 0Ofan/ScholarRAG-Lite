import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evaluation import document_ranking,score_topic

class EvaluationTests(unittest.TestCase):
    def test_deduplicate_chunks_by_first_rank(self):
        hits=[{'doc_id':'a','score':.9},{'doc_id':'a','score':.8},{'doc_id':'b','score':.7}]
        self.assertEqual(document_ranking(hits),['a','b'])
    def test_all_retrieved_and_grades(self):
        q={'a':2,'b':1,'c':0}
        r=score_topic(['a','b','c'],q,q)
        self.assertAlmostEqual(r['ndcg@10'],1)
        self.assertEqual(r['recall@10'],1)
        self.assertEqual(r['recall@100'],1)
        self.assertEqual(r['mrr'],1)
    def test_relevant_late(self):
        q={str(i):0 for i in range(100)};q['a']=2
        rank=[str(i) for i in range(99)]+['a']
        r=score_topic(rank,q,q)
        self.assertEqual(r['recall@10'],0)
        self.assertEqual(r['recall@100'],1)
        self.assertAlmostEqual(r['mrr'],1/100)
    def test_none_retrieved(self):
        q={'a':2,'b':0}
        r=score_topic(['b'],q,q)
        self.assertEqual(r['recall@10'],0)
        self.assertEqual(r['mrr'],0)
    def test_no_valid_positive(self):
        self.assertIsNone(score_topic(['a'],{'a':0},{'a'}))
        self.assertIsNone(score_topic(['a'],{'a':2},{'b'}))
    def test_unjudged_is_tracked(self):
        r=score_topic(['x','a'],{'a':2},{'x','a'})
        self.assertEqual(r['unjudged_in_top10'],1)
        self.assertEqual(r['mrr'],.5)
    def test_negative_one_has_zero_gain(self):
        r=score_topic(['x','a'],{'x':-1,'a':2},{'x','a'})
        self.assertEqual(r['mrr'],.5)
        self.assertGreaterEqual(r['ndcg@10'],0)

if __name__=='__main__':unittest.main()
