import json
import unittest
from pathlib import Path

from src.deepseek_client import is_complete_generation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/phase418'


class Phase418Tests(unittest.TestCase):
    def test_offline_counts_mapping_and_historical_baseline(self):
        records = json.loads((OUT / 'offline_records.json').read_text(encoding='utf-8'))
        self.assertEqual(len(records), 20)
        for qid, row in records.items():
            top8 = row['groups']['8']['evidence']
            historical = json.loads((ROOT / f'experiments/results/phase37/{qid}.json').read_text(encoding='utf-8'))
            self.assertEqual([e['chunk_id'] for e in row['groups']['3']['evidence']],
                             [e['chunk_id'] for e in historical['evidence']])
            for k in (3, 5, 8):
                group = row['groups'][str(k)]
                self.assertEqual(group['retrieved_candidates'], 10)
                self.assertEqual(group['context_evidence'], k)
                self.assertEqual(len(group['evidence']), k)
                self.assertEqual([e['chunk_id'] for e in group['evidence']],
                                 [e['chunk_id'] for e in top8[:k]])
                self.assertEqual(len({e['doc_id'] for e in group['evidence']}), k)
                self.assertTrue(all(c['status'] == 'valid' for c in group['source_checks']))

    def test_real_results_have_usage_and_no_repeated_requests(self):
        paths = list((OUT / 'raw_answers').glob('*.json'))
        self.assertEqual(len(paths), 60)
        for path in paths:
            row = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(row['status'], 'complete')
            self.assertTrue(is_complete_generation(row['answer'], row['generation']))
            self.assertEqual(len(row['evidence']), row['k'])
            self.assertEqual(row['citation_check']['invalid'], [])
            self.assertEqual(row['estimated_cost']['status'], 'computed_from_usage_and_published_price')

    def test_aggregate_is_composed_from_real_records(self):
        aggregate = json.loads((OUT / 'aggregate_metrics.json').read_text(encoding='utf-8'))
        self.assertEqual(aggregate['total_real_requests'], 60)
        self.assertTrue(aggregate['all_three_groups_same_questions'])
        self.assertEqual([aggregate['groups'][str(k)]['complete'] for k in (3, 5, 8)], [20, 20, 20])
        self.assertEqual(sum(aggregate['groups'][str(k)]['output_tokens']['total'] for k in (3, 5, 8)),
                         sum(json.loads(p.read_text(encoding='utf-8'))['generation']['usage']['output_tokens']
                             for p in (OUT / 'raw_answers').glob('*.json')))


if __name__ == '__main__':
    unittest.main()
