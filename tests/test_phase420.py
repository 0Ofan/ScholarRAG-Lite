"""Targeted tests for conservative offline verification."""
import json
import unittest
from pathlib import Path

from src.phase420_verifier import categories, check_claim, extract_claims, question_points, source_role

ROOT = Path(__file__).resolve().parents[1]


def saved(qid, k):
    return json.loads((ROOT / f'experiments/results/phase418/raw_answers/{qid}_K{k}.json').read_text(encoding='utf-8'))


class Phase420Tests(unittest.TestCase):
    def test_split_and_sentence_end_citation(self):
        ev = [dict(evidence_id='E1', doc_id='d1', chunk_id='d1:1', title='T', text='text')]
        claims = extract_claims('第一项成立，但第二项不同[E1]。', ev)
        self.assertEqual(len(claims), 2)
        self.assertTrue(all(c['evidence_ids'] == ['E1'] for c in claims))
        self.assertEqual(claims[0]['parent_claim_id'], claims[1]['parent_claim_id'])

    def test_numeric_pair_kept_together(self):
        ev = [dict(evidence_id='E1', doc_id='d1', chunk_id='d1:1', title='T', text='text')]
        claims = extract_claims('24名儿童中60%住院[E1]。', ev)
        self.assertEqual(len(claims), 1)
        self.assertIn('24名儿童中60%住院', claims[0]['claim_text'])

    def test_q07_denominator_and_adult_scope(self):
        data = saved('Q07', 8)
        results = [(c, check_claim(c, data['evidence'], data['question']))
                   for c in extract_claims(data['answer'], data['evidence'])]
        self.assertTrue(any(r['status'] == 'Contradicted' and 'Numerator used as denominator' in r['issues']
                            for _, r in results))
        self.assertTrue(any('Adult-source scope requires review' in r['issues'] for _, r in results))

    def test_correct_numeric_and_irrelevant_evidence_not_false_positive(self):
        item = dict(evidence_id='E1', doc_id='d', chunk_id='c', title='Children',
                    text='Twenty‐four (60%) children were hospitalized.')
        base = dict(evidence_ids=['E1'], status='Unverified', issue_type=None)
        correct = check_claim(dict(base, claim_text='24名儿童（60%）住院[E1]'), [item])
        self.assertEqual(correct['status'], 'Unverified')
        self.assertNotIn('Numerator used as denominator', correct['issues'])
        other = check_claim(dict(base, claim_text='5年后疫苗仍有效[E1]'), [item])
        self.assertEqual(other['status'], 'Unverified')

    def test_source_role_q01(self):
        data = saved('Q01', 8)
        secondary = [e for e in data['evidence'] if 'et al.' in e['text']]
        self.assertTrue(secondary)
        self.assertTrue(all(source_role(e) == 'Secondary Citation' for e in secondary))
        self.assertTrue(all(check_claim(c, data['evidence'], data['question'])['status'] == 'Unverified'
                            for c in extract_claims(data['answer'], data['evidence'])))
        self.assertEqual(source_role(dict(text='Our results show an observed effect.')), 'Direct Primary Source')
        self.assertEqual(source_role(dict(text='A result was observed.')), 'Unknown Attribution')

    def test_citation_mapping_and_conditions(self):
        item = dict(evidence_id='E1', doc_id='d', chunk_id='c', title='T', text='At 20 °C, an effect was observed.')
        base = dict(status='Unverified', issue_type=None)
        missing = check_claim(dict(base, claim_text='事实[E9]', evidence_ids=['E9']), [item])
        self.assertIn('Citation Mapping Error', missing['issues'])
        broad = check_claim(dict(base, claim_text='所有条件均有效[E1]', evidence_ids=['E1']), [item])
        self.assertIn('Broad claim with limited experimental conditions; review required', broad['issues'])
        self.assertEqual(broad['status'], 'Unverified')

    def test_q11_separate_points(self):
        data = saved('Q11', 8)
        points = question_points('Q11', data['answer'], data['evidence'], [])
        self.assertEqual(points[-2]['evidence_status'], 'Directly Supported')
        self.assertEqual(points[-1]['evidence_status'], 'Unverified')

    def test_q20_insufficient_five_year_evidence(self):
        data = saved('Q20', 8)
        points = question_points('Q20', data['answer'], data['evidence'], [])
        self.assertEqual(points[-1]['evidence_status'], 'Unverified')
        self.assertTrue(all(check_claim(c, data['evidence'], data['question'])['status'] != 'Directly Supported'
                            for c in extract_claims(data['answer'], data['evidence'])))

    def test_categories_and_60_answers(self):
        self.assertIn('Numerical', categories('24名儿童中60%住院'))
        for q in range(1, 21):
            for k in (3, 5, 8):
                data = saved(f'Q{q:02d}', k)
                self.assertTrue(extract_claims(data['answer'], data['evidence']))


if __name__ == '__main__':
    unittest.main()
