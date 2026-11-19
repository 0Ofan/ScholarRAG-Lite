import json
import unittest
from pathlib import Path

from src.evidence_metadata import source_role
from src.phase410_audit import classify

ROOT=Path(__file__).resolve().parents[1]


class Phase410Tests(unittest.TestCase):
    def test_citation_letter_is_not_review(self):
        for qid in ('Q01','Q16'):
            row=json.loads((ROOT/f'experiments/results/phase49/{qid}.json').read_text(encoding='utf8'))
            self.assertEqual(source_role(row['evidence'][2]),'Unclassified')

    def test_independent_known_contradictions(self):
        q15=json.loads((ROOT/'experiments/results/phase45/A_Q15.json').read_text(encoding='utf8'))
        status,_=classify('法国36例患者的研究是单臂',q15['evidence'][2]['text'],True)
        self.assertEqual(status,'Contradicted')
        q16=json.loads((ROOT/'experiments/results/phase45/A_Q16.json').read_text(encoding='utf8'))
        source=' '.join(e['text'] for e in q16['evidence'])
        self.assertEqual(classify('未提供表面类型',source,True)[0],'Contradicted')
        self.assertEqual(classify('未提供消毒剂实际使用浓度/接触时间',source,True)[0],'Contradicted')

    def test_same_evidence_in_online_pairs(self):
        for qid in ('Q07','Q15','Q16'):
            row=json.loads((ROOT/f'experiments/results/phase410/online_{qid}.json').read_text(encoding='utf8'))
            self.assertEqual(set(row['results']),{'V1','V2'})
            self.assertTrue(all(row['results'][v]['status']=='success' for v in ('V1','V2')))
            self.assertEqual(len(row['evidence']),3)


if __name__=='__main__':unittest.main()
