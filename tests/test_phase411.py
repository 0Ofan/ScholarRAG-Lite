import unittest

from src.evidence_verifier import verify
from src.phase411_validate import _saved, _example, cases


class Phase411Tests(unittest.TestCase):
    def test_curated_narrow_cases(self):
        for case in cases():
            with self.subTest(case=case['id']):
                result = verify(case['answer'], case['evidence'])
                issues = {c['issue_type'] for c in result['claims'] if c['issue_type']}
                if case['expected_issue']:
                    self.assertIn(case['expected_issue'], issues)
                else:
                    self.assertFalse(issues)

    def test_referenced_mapping_and_clause_split(self):
        evidence = _example('The first study measured 40 children.')
        evidence.append(dict(evidence[0], evidence_id='E2', doc_id='second-doc',
                             chunk_id='second-doc:1', text='The second study measured 20 adults.'))
        result = verify('第一项研究有40名儿童[E1]；第二项研究有20名成人[E2]。', evidence)
        self.assertEqual(len(result['claims']), 2)
        self.assertEqual(result['claims'][0]['evidence_ids'], ['E1'])
        self.assertEqual(result['claims'][1]['evidence_ids'], ['E2'])
        self.assertEqual(result['claims'][1]['document_ids'], ['second-doc'])
        self.assertEqual(result['claims'][0]['original_evidence'], [evidence[0]['text']])

    def test_ambiguous_historical_cases_stay_unverified(self):
        for qid in ('Q11', 'Q13', 'Q17'):
            with self.subTest(qid=qid):
                answer, evidence = _saved(qid, 'V2')
                result = verify(answer, evidence)
                self.assertTrue(any(c['status'] == 'Unverified' for c in result['claims']))
                self.assertFalse(result['critical'])

    def test_missing_citation_does_not_gain_verified_status(self):
        result = verify('该研究样本为40人。', _example('The study included 40 patients.'))
        self.assertTrue(all(c['status'] != 'Verified' for c in result['claims']))


if __name__ == '__main__':
    unittest.main()
