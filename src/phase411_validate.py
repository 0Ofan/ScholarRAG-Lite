"""Phase 4.11 offline verifier regression; no retrieval or API requests."""
import json
import statistics
import time
from pathlib import Path

from .evidence_metadata import enrich
from .evidence_verifier import verify


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/phase411'


def _saved(qid, version):
    if version == 'V1':
        path = ROOT / f'experiments/results/phase45/A_{qid}.json'
    else:
        path = ROOT / f'experiments/results/phase49/{qid}.json'
    row = json.loads(path.read_text(encoding='utf8'))
    return row['answer'], enrich(row['evidence'])


def _example(text):
    return enrich([{'evidence_id': 'E1', 'citation': '[E1]', 'doc_id': 'synthetic-doc',
                    'chunk_id': 'synthetic-doc:1', 'title': 'Synthetic regression evidence',
                    'text': text}])


def cases():
    # Binary labels cover only explicit, narrow claims. Unverified is not a negative judgment.
    for qid, issue in [('Q07', 'Missing Denominator'),
                       ('Q08', 'Study Design Overgeneralization'),
                       ('Q15', 'Study Design'), ('Q16', 'False Absence')]:
        answer, evidence = _saved(qid, 'V1')
        yield {'id': f'{qid}_V1', 'expected_issue': issue, 'answer': answer, 'evidence': evidence,
               'reference': 'Phase 4.10 independently documented known issue'}
    numeric = _example('In the same pediatric cohort, 8 (20%) children were asymptomatic, '
                       '4 (10%) had pneumonia, and 24 (60%) were hospitalized.')
    yield {'id': 'correct_denominator', 'expected_issue': None,
           'answer': '该研究总体40人，8人（20%）无症状，通常症状较轻[E1]。', 'evidence': numeric,
           'reference': 'Synthetic explicit matching denominator'}
    yield {'id': 'wrong_denominator', 'expected_issue': 'Population Mismatch',
           'answer': '该研究总体100人，8人（20%）无症状，通常症状较轻[E1]。', 'evidence': numeric,
           'reference': 'Synthetic conflicting denominator'}
    yield {'id': 'same_design', 'expected_issue': None,
           'answer': '两项试验均为随机双盲III期[E1]。',
           'evidence': _example('Two trials were conducted. Both trials were randomized, double-blind phase III.'),
           'reference': 'Synthetic explicit common study design'}
    yield {'id': 'different_design', 'expected_issue': 'Study Design Overgeneralization',
           'answer': '两项试验均为随机双盲III期[E1]。',
           'evidence': _example('Two trials were launched. Only one was a randomized, double-blind phase III trial.'),
           'reference': 'Synthetic one-of-two design conflict'}
    yield {'id': 'genuinely_absent', 'expected_issue': None,
           'answer': '该研究未提供表面类型[E1]。',
           'evidence': _example('The report describes coronavirus persistence but does not specify surfaces.'),
           'reference': 'Synthetic source without surface type'}
    yield {'id': 'false_absence', 'expected_issue': 'False Absence',
           'answer': '该研究未提供表面类型[E1]。',
           'evidence': _example('Virus persistence was measured on glass, metal and plastic surfaces.'),
           'reference': 'Synthetic source with explicit surface types'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases():
        result = verify(case['answer'], case['evidence'])
        issues = sorted({c['issue_type'] for c in result['claims'] if c['issue_type']})
        predicted = bool(issues)
        expected = case['expected_issue'] is not None
        rows.append({'case_id': case['id'], 'expected_issue': case['expected_issue'],
                     'predicted_issues': issues, 'expected_alert': expected,
                     'predicted_alert': predicted, 'reference': case['reference'],
                     'claim_count': len(result['claims'])})
    tp = sum(r['expected_alert'] and r['predicted_alert'] for r in rows)
    fp = sum(not r['expected_alert'] and r['predicted_alert'] for r in rows)
    fn = sum(r['expected_alert'] and not r['predicted_alert'] for r in rows)
    tn = sum(not r['expected_alert'] and not r['predicted_alert'] for r in rows)
    metrics = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
               'precision': tp / (tp + fp) if tp + fp else None,
               'recall': tp / (tp + fn) if tp + fn else None,
               'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}
    review = []
    timings = []
    for qid in [f'Q{i:02d}' for i in range(1, 21)]:
        for version in ('V1', 'V2'):
            answer, evidence = _saved(qid, version)
            result = verify(answer, evidence)
            review.append({'question_id': qid, 'version': version,
                           'issues': [{'claim_id': c['claim_id'], 'issue_type': c['issue_type'],
                                      'status': c['status'], 'evidence_ids': c['evidence_ids'],
                                      'document_ids': c['document_ids'], 'reason': c['reason']}
                                     for c in result['claims'] if c['issue_type']],
                           'unverified_claims': sum(c['status'] == 'Unverified' for c in result['claims']),
                           'total_claims': len(result['claims'])})
            for _ in range(10):
                start = time.perf_counter()
                verify(answer, evidence)
                timings.append(time.perf_counter() - start)
    ordered = sorted(timings)
    perf = {'measurement': 'Offline saved 20 V1 + 20 V2 answers, 10 repeats each; no API or retrieval',
            'n': len(timings), 'mean_seconds': statistics.mean(timings),
            'median_seconds': statistics.median(timings),
            'p95_seconds': ordered[int(.95 * (len(ordered) - 1))],
            'historic_phase410_v2_verification_mean_seconds': 0.0006667333339767841,
            'historic_comparison_limit': 'Different answer set, invocation and timing context; not paired E2E.'}
    (OUT / 'verifier_regression.json').write_text(json.dumps({'cases': rows, 'matrix': metrics,
        'scope': '10 curated, narrow reference cases; not population accuracy. Unverified excluded as semantic judgments.'},
        ensure_ascii=False, indent=2), encoding='utf8')
    (OUT / 'saved_answer_review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding='utf8')
    (OUT / 'performance_regression.json').write_text(json.dumps(perf, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'matrix': metrics, 'performance': perf}, ensure_ascii=False))


if __name__ == '__main__':
    main()
