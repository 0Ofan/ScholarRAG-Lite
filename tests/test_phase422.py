"""Offline checks only; no evaluator API calls."""
import csv
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from src.phase422_prepare import SEED, build, validate_judgement

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/phase422'


def load(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def rows(name):
    with (OUT / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def mock_response(sample):
    return {'sample_id': sample['sample_id'],
            'point_evaluations': [{'point_id': p['point_id'], 'coverage': 'Unverified',
                                   'evidence_support': 'Unverified', 'answer_text': '',
                                   'omission': False, 'factual_error': False,
                                   'reason': 'not enough semantic basis', 'confidence': 'Low',
                                   'evidence_quotes': []} for p in sample['points']],
            'claim_evaluations': [{'claim_id': c['claim_id'], 'support': 'Unverified',
                                   'reason': 'not enough semantic basis', 'confidence': 'Low',
                                   'evidence_quotes': []} for c in sample['claims']],
            'scores': {key: 'Unverified' for key in ('Completeness','Faithfulness','Relevance','Citation Support')},
            'citation_validity': 'Unverified', 'source_attribution': 'Unverified',
            'abstention_quality': 'Unverified'}


class Phase422Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load('blind_evaluation_dataset.json')['samples']
        cls.mapping = load('anonymization_mapping.json')['mapping']

    def test_60_samples_randomized_and_reproducible(self):
        again, mapping = build()
        self.assertEqual(again, self.dataset)
        self.assertEqual(mapping, self.mapping)
        self.assertEqual(len(again), 60)
        self.assertEqual(load('anonymization_mapping.json')['seed'], SEED)

    def test_mapping_isolated_from_judge_payload(self):
        text = (OUT / 'blind_evaluation_dataset.json').read_text(encoding='utf-8')
        self.assertNotIn('Top-3', text)
        self.assertNotIn('Top-5', text)
        self.assertNotIn('Top-8', text)
        self.assertNotIn('evidence_k', text)
        self.assertNotIn('phase420_status', text)
        self.assertEqual(Counter(m['evidence_k'] for m in self.mapping), {3:20,5:20,8:20})

    def test_input_complete_and_evidence_corresponds(self):
        by_id = {s['sample_id']: s for s in self.dataset}
        for m in self.mapping:
            s = by_id[m['sample_id']]
            raw = (ROOT / f"experiments/results/phase418/raw_answers/{m['question_id']}_K{m['evidence_k']}.json").read_bytes()
            original = json.loads(raw)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), m['source_sha256'])
            self.assertEqual(s['answer'], original['answer'])
            self.assertEqual(len(s['evidence']), m['evidence_k'])
            self.assertTrue(s['points'] and s['claims'])
            for e, prior in zip(s['evidence'], original['evidence']):
                self.assertEqual((e['document_id'],e['chunk_id'],e['text']), (prior['doc_id'],prior['chunk_id'],prior['text']))

    def test_point_and_claim_ids_and_source_map(self):
        for s in self.dataset:
            pids = {p['point_id'] for p in s['points']}
            eids = {e['evidence_id'] for e in s['evidence']}
            self.assertEqual(len(pids), len(s['points']))
            for ref in s['point_evidence_candidates']:
                self.assertIn(ref['point_id'], pids)
                self.assertIn(ref['evidence_id'], eids)
            for claim in s['claims']:
                self.assertTrue(set(claim['citation_ids']) <= eids)

    def test_schema_and_unverified_retained(self):
        sample = self.dataset[0]
        self.assertEqual(validate_judgement(sample, mock_response(sample))['scores']['Completeness'], 'Unverified')

    def test_invalid_score_rejected(self):
        sample = self.dataset[0]
        value = mock_response(sample)
        value['scores']['Faithfulness'] = 6
        with self.assertRaises(Exception):
            validate_judgement(sample, value)

    def test_invented_quote_rejected(self):
        sample = self.dataset[0]
        value = mock_response(sample)
        value['claim_evaluations'][0]['support'] = 'Directly Supported'
        value['claim_evaluations'][0]['evidence_quotes'] = [{'evidence_id': sample['evidence'][0]['evidence_id'], 'quote': 'invented finding'}]
        with self.assertRaises(ValueError):
            validate_judgement(sample, value)

    def test_six_pilot_samples_and_known_cases(self):
        pilot = load('pilot_evaluation_inputs.json')
        self.assertEqual(len(pilot['samples']), 6)
        self.assertEqual({s['sample_id'][:3] for s in pilot['samples']}, {'Q07','Q11'})
        self.assertTrue(any('24名儿童中60%' in s['answer'] for s in pilot['samples'] if s['sample_id'].startswith('Q07')))
        self.assertTrue(any('血液中的抗体' in s['answer'] for s in pilot['samples'] if s['sample_id'].startswith('Q11')))
        self.assertEqual(load('pilot_manifest.json')['status'], 'prepared; no judge API calls')

    def test_other_case_inputs_present(self):
        ids = {m['question_id'] for m in self.mapping}
        self.assertTrue({'Q01','Q20','Q02','Q19'} <= ids)
        q20 = [s for s in self.dataset if s['sample_id'].startswith('Q20')]
        self.assertEqual(len(q20), 3)
        self.assertTrue(all('无法确定' in s['answer'] or '不能确立' in s['answer'] for s in q20))
        q01_full = next(s for s in self.dataset if s['sample_id'] == next(
            m['sample_id'] for m in self.mapping if m['question_id'] == 'Q01' and m['evidence_k'] == 8))
        roles = {e['evidence_id']: e['source_role_hint'] for e in q01_full['evidence']}
        self.assertEqual(roles['E4'], 'Secondary Citation')
        self.assertEqual(roles['E7'], 'Secondary Citation')

    def test_same_rubric_and_neutral_conditional_points(self):
        prompt = (ROOT / 'experiments/configs/phase422/judge_prompt.md').read_text(encoding='utf-8')
        self.assertIn('每个 Point', prompt)
        self.assertIn('每个 Claim', prompt)
        for qid, pid in (('Q11', 'Q11-P03'), ('Q13', 'Q13-P02')):
            points = [[p['point_text'] for p in s['points'] if p['point_id'] == pid][0]
                      for s in self.dataset if s['sample_id'].startswith(qid)]
            self.assertEqual(len(set(points)), 1)
            self.assertNotIn('Top-', points[0])

    def test_no_fake_results_or_usage(self):
        self.assertEqual(len(rows('answer_level_evaluation.csv')), 60)
        self.assertTrue(all(r['judge_status'] == 'Not Run' and r['completeness'] == 'Unverified'
                            for r in rows('answer_level_evaluation.csv')))
        self.assertEqual(len(rows('judge_usage.csv')), 0)
        self.assertEqual(load('judge_consistency.json')['status'], 'Not Run')
        self.assertEqual(load('aggregate_metrics.json')['judge_completed'], 0)


if __name__ == '__main__':
    unittest.main()
