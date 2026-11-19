import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/phase419'


class Phase419Tests(unittest.TestCase):
    def test_claims_keep_source_and_audit_status(self):
        with (OUT / 'claim_level_audit.csv').open(encoding='utf-8-sig', newline='') as source:
            rows = list(csv.DictReader(source))
        self.assertGreater(len(rows), 60)
        self.assertEqual({x['status'] for x in rows}, {'Unverified', 'Contradicted'})
        for row in rows:
            self.assertTrue(row['claim_text'])
            self.assertEqual(len(json.loads(row['citation_ids'])), len(json.loads(row['document_ids'])))
            self.assertEqual(len(json.loads(row['document_ids'])), len(json.loads(row['chunk_ids'])))
            self.assertEqual(len(json.loads(row['chunk_ids'])), len(json.loads(row['original_evidence'])))
        errors = [x for x in rows if x['status'] == 'Contradicted']
        self.assertEqual([(x['question_id'], x['k']) for x in errors], [('Q07', '8')])

    def test_all_paired_questions_and_no_invented_macro_scores(self):
        with (OUT / 'per_question_comparison.csv').open(encoding='utf-8-sig', newline='') as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 60)
        self.assertEqual(len({x['question_id'] for x in rows}), 20)
        aggregate = json.loads((OUT / 'aggregate_quality_metrics.json').read_text(encoding='utf-8'))
        self.assertEqual(aggregate['answers'], 60)
        self.assertEqual(aggregate['claim_units'], sum(aggregate['claim_status_counts'].values()))
        self.assertIn('not_computed', aggregate['quality_scores'])


if __name__ == '__main__':
    unittest.main()
