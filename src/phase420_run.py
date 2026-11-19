"""Offline comparison against frozen Phase 4.18 answers and Phase 4.19 labels."""
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from src.phase420_verifier import check_claim, extract_claims, question_points

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/phase420'


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError('Phase 4.20 results already exist; refusing to overwrite')
    old = list(csv.DictReader((ROOT / 'experiments/results/phase419/claim_level_audit.csv').open(encoding='utf-8-sig', newline='')))
    old_map = {(r['question_id'], int(r['k']), r['claim_id']): r for r in old}
    frozen = {q['question_id']: q for q in json.loads(
        (ROOT / 'experiments/configs/phase36_questions.json').read_text(encoding='utf-8'))['questions']}
    prior_hashes = json.loads((ROOT / 'experiments/results/phase419/aggregate_quality_metrics.json').read_text(encoding='utf-8'))['original_answer_sha256']
    claims, per_question, errors, cases = [], [], [], {}
    for qid, question in frozen.items():
        cases[qid] = {}
        for k in (3, 5, 8):
            path = ROOT / f'experiments/results/phase418/raw_answers/{qid}_K{k}.json'
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != prior_hashes[f'{qid}_K{k}']:
                raise ValueError(f'{qid} K{k}: saved answer differs from Phase 4.19 input')
            answer = json.loads(raw)
            if answer['status'] != 'complete' or answer['finish_reason'] != 'stop' or answer['question'] != question['question']:
                raise ValueError(f'{qid} K{k}: incompatible answer')
            evidence = answer['evidence']
            started = time.perf_counter()
            local = extract_claims(answer['answer'], evidence)
            groups = Counter()
            issue_counts = Counter()
            for claim in local:
                prior = old_map[(qid, k, claim['parent_claim_id'])]
                result = check_claim(claim, evidence, question['question'])
                mapping = {e['evidence_id']: e for e in evidence}
                cited = [mapping[eid] for eid in claim['evidence_ids'] if eid in mapping]
                groups[result['status']] += 1
                issue_counts.update(result['issues'])
                row = {'question_id': qid, 'k': k, 'claim_id': claim['claim_id'],
                       'parent_claim_id': claim['parent_claim_id'], 'original_claim_text': prior['claim_text'],
                       'claim_text': claim['claim_text'], 'citation_scope': claim['citation_scope'],
                       'citation_ids': json.dumps(claim['evidence_ids']),
                       'document_ids': json.dumps([e['doc_id'] for e in cited]),
                       'paper_titles': json.dumps([e['title'] for e in cited], ensure_ascii=False),
                       'chunk_ids': json.dumps([e['chunk_id'] for e in cited]),
                       'original_evidence': json.dumps([e['text'] for e in cited], ensure_ascii=False),
                       'categories': json.dumps(result['categories']),
                       'source_roles': json.dumps(result['source_roles']),
                       'phase419_status': prior['status'], 'phase420_status': result['status'],
                       'phase420_reason': result['reason'], 'review_issues': json.dumps(result['issues']),
                       'review_provenance': prior['review_provenance']}
                claims.append(row)
                if result['status'] == 'Contradicted' or result['issues']:
                    errors.append({'question_id': qid, 'k': k, 'claim_id': claim['claim_id'],
                                   'classification': 'confirmed contradiction' if result['status'] == 'Contradicted' else 'review hint; not confirmed error',
                                   'issues': json.dumps(result['issues']), 'citation_ids': row['citation_ids'],
                                   'document_ids': row['document_ids'], 'claim_text': claim['claim_text'],
                                   'original_evidence': row['original_evidence']})
            points = question_points(qid, answer['answer'], evidence, question['expected_answer_points'])
            per_question.append({'question_id': qid, 'k': k, 'old_claim_count': sum(
                r['question_id'] == qid and int(r['k']) == k for r in old),
                'new_claim_count': len(local), 'status_counts': json.dumps(dict(groups)),
                'issue_counts': json.dumps(dict(issue_counts)),
                'expected_point_checks': json.dumps(points, ensure_ascii=False),
                'verification_seconds': round(time.perf_counter() - started, 6)})
            if qid in ('Q01', 'Q07', 'Q11', 'Q20'):
                cases[qid][f'K{k}'] = {'statuses': dict(groups), 'issues': dict(issue_counts),
                                      'point_checks': points}
    counts = Counter(r['phase420_status'] for r in claims)
    old_counts = Counter(r['status'] for r in old)
    by_category = defaultdict(Counter)
    role_counts = Counter()
    for r in claims:
        for category in json.loads(r['categories']):
            by_category[category][r['phase420_status']] += 1
        role_counts.update(json.loads(r['source_roles']).values())
    result = {'inputs': {'answers': len(per_question), 'questions': len(frozen),
                         'source': 'Phase 4.18 saved answers; SHA-256 matched Phase 4.19'},
              'phase419': {'claim_units': len(old), 'statuses': dict(old_counts),
                           'online_local_verifier_statuses': dict(Counter(r['local_verifier_status'] for r in old))},
              'phase420': {'claim_units': len(claims), 'statuses': dict(counts),
                           'category_statuses': {key: dict(value) for key, value in by_category.items()},
                           'cited_evidence_role_occurrences': dict(role_counts),
                           'review_hints': dict(Counter(issue for r in claims for issue in json.loads(r['review_issues'])))},
              'known_confirmed_error': {'case': 'Q07 K8 E5 numerator/denominator',
                                        'phase419_local_detected': False,
                                        'phase420_detected': any(r['question_id'] == 'Q07' and r['k'] == 8 and r['phase420_status'] == 'Contradicted' for r in claims)},
              'overall_accuracy': 'not computed; insufficient independent semantic labels'}
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / 'claim_level_comparison.csv', claims)
    write_csv(OUT / 'per_question_verification.csv', per_question)
    write_csv(OUT / 'verification_errors.csv', errors)
    write_json(OUT / 'verifier_comparison.json', result)
    write_json(OUT / 'regression_cases.json', cases)
    print(json.dumps({'answers': len(per_question), 'old_claims': len(old), 'new_claims': len(claims), 'statuses': dict(counts), 'error_rows': len(errors)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
