"""Incremental checks for the offline structured answer benchmark."""
import csv
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from src.phase421_benchmark import CHECKED_REFERENCES, reference_role

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/phase421'


def j(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def c(name):
    with (OUT / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


class Phase421Tests(unittest.TestCase):
    def test_frozen_questions_and_points(self):
        questions = j('question_benchmark.json')
        frozen = json.loads((ROOT / 'experiments/configs/phase36_questions.json').read_text(encoding='utf-8'))['questions']
        self.assertEqual(len(questions), len(frozen), 20)
        for q, original in zip(questions, frozen):
            self.assertEqual(q['original_question'], original['question'])
            self.assertEqual([p['point_text'] for p in q['points']], original['expected_answer_points'])
            self.assertEqual(len({p['point_id'] for p in q['points']}), len(q['points']))

    def test_reference_source_mapping_and_role(self):
        refs = j('evidence_reference_map.json')
        offline = json.loads((ROOT / 'experiments/results/phase418/offline_records.json').read_text(encoding='utf-8'))
        for row in refs:
            e = next(x for x in offline[row['question_id']]['groups']['8']['evidence'] if x['evidence_id'] == row['evidence_id'])
            self.assertEqual((row['document_id'], row['chunk_id'], row['paper_title'], row['evidence_text']),
                             (e['doc_id'], e['chunk_id'], e['title'], e['text']))
            self.assertEqual(row['source_role'], reference_role(e))
        self.assertEqual(len(refs), 64)

    def test_reference_status_does_not_follow_answers(self):
        refs = j('evidence_reference_map.json')
        self.assertEqual(sum(r['support_status'] != 'Unverified' for r in refs), len([x for v in CHECKED_REFERENCES.values() for x in v]))
        self.assertEqual(len([r for r in refs if r['question_id'] == 'Q01' and r['support_status'] == 'Indirect Evidence']), 2)

    def test_claims_and_citations(self):
        claims = c('claim_evaluation.csv')
        self.assertEqual(len(claims), 366)
        for row in claims:
            ids = json.loads(row['citation_ids'])
            docs = json.loads(row['document_ids'])
            chunks = json.loads(row['chunk_ids'])
            self.assertEqual(len(ids), len(docs))
            self.assertEqual(len(ids), len(chunks))
            self.assertNotEqual(row['semantic_support_scope'], 'correct')

    def test_answer_60_and_hashes(self):
        rows = c('answer_evaluation.csv')
        self.assertEqual(len(rows), 60)
        hashes = json.loads((ROOT / 'experiments/results/phase419/aggregate_quality_metrics.json').read_text(encoding='utf-8'))['original_answer_sha256']
        for row in rows:
            qid, k = row['question_id'], int(row['k'])
            raw = (ROOT / f'experiments/results/phase418/raw_answers/{qid}_K{k}.json').read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row['answer_file_sha256'])
            self.assertEqual(row['answer_file_sha256'], hashes[f'{qid}_K{k}'])
            self.assertEqual(row['original_answer'], json.loads(raw)['answer'])
            self.assertEqual(json.loads(row['original_evidence']), json.loads(raw)['evidence'])

    def test_same_point_standard_across_groups(self):
        rows = c('point_evaluation.csv')
        self.assertEqual(len(rows), 180)
        for qid in {r['question_id'] for r in rows}:
            by_k = {k: [r['point_id'] for r in rows if r['question_id'] == qid and int(r['k']) == k] for k in (3, 5, 8)}
            self.assertEqual(by_k[3], by_k[5])
            self.assertEqual(by_k[5], by_k[8])

    def test_unverified_and_citation_validity(self):
        rows = c('answer_evaluation.csv')
        self.assertTrue(all(r['completeness_score'] == 'Unverified' and r['faithfulness_score'] == 'Unverified' for r in rows))
        self.assertTrue(all(r['citation_validity'] == 'Valid' and r['citation_semantic_support'] == 'Unverified' for r in rows))

    def test_q07_and_q01(self):
        rows = c('answer_evaluation.csv')
        q07 = next(r for r in rows if r['question_id'] == 'Q07' and r['k'] == '8')
        self.assertEqual(q07['confirmed_contradictions'], '1')
        self.assertIn('Adult-source scope requires review', q07['review_issues'])
        q01 = [r for r in j('evidence_reference_map.json') if r['question_id'] == 'Q01' and r['evidence_id'] in ('E4','E7')]
        self.assertTrue(all(r['source_role'] == 'Secondary Citation' for r in q01))

    def test_q11_q20_and_other_cases(self):
        points = c('point_evaluation.csv')
        q11 = {int(r['k']): r for r in points if r['point_id'] == 'Q11-P02'}
        self.assertEqual(q11[8]['answer_observation'], 'Partially Covered')
        self.assertEqual(q11[3]['answer_observation'], 'Unverified')
        answers = c('answer_evaluation.csv')
        self.assertEqual({r['abstention_quality'] for r in answers if r['question_id'] == 'Q20'}, {'Appropriate Abstention'})
        self.assertEqual({r['abstention_quality'] for r in answers if r['question_id'] == 'Q19'}, {'Appropriate Abstention'})
        self.assertEqual({r['k'] for r in answers if r['question_id'] == 'Q02'}, {'3','5','8'})

    def test_pairwise_scope_and_metrics(self):
        pairs = c('pairwise_comparison.csv')
        self.assertEqual(len(pairs), 60)
        self.assertEqual(Counter(r['classification'] for r in pairs), Counter({'Unverified': 58, 'Mixed': 2}))
        self.assertTrue(all(r['classification'] != 'Unchanged' for r in pairs))
        metrics = j('evaluation_metrics.json')
        self.assertEqual(metrics['claim_status_counts'], {'Unverified': 365, 'Contradicted': 1})
        self.assertIn('Unverified', metrics['answer_scores'])


if __name__ == '__main__':
    unittest.main()
