"""Prepare reproducible, group-blinded judge inputs; makes no API requests."""
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator

from src.phase421_benchmark import reference_role

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/phase422'
SEED = 4222026


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(path, columns, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def validate_judgement(sample, value):
    """Reject missing units, invented IDs, invalid scores and unsupported quotes."""
    schema = read_json(ROOT / 'experiments/configs/phase422/judge_schema.json')
    Draft202012Validator(schema).validate(value)
    expected_points = {p['point_id'] for p in sample['points']}
    expected_claims = {c['claim_id'] for c in sample['claims']}
    evidence = {e['evidence_id']: e['text'] for e in sample['evidence']}
    if value.get('sample_id') != sample['sample_id']:
        raise ValueError('sample ID mismatch')
    if {p.get('point_id') for p in value.get('point_evaluations', [])} != expected_points:
        raise ValueError('point coverage mismatch')
    if {c.get('claim_id') for c in value.get('claim_evaluations', [])} != expected_claims:
        raise ValueError('claim coverage mismatch')
    if len(value['point_evaluations']) != len(expected_points) or len(value['claim_evaluations']) != len(expected_claims):
        raise ValueError('duplicate point or claim')
    for item in value['point_evaluations'] + value['claim_evaluations']:
        if not isinstance(item.get('reason'), str) or not item['reason'].strip():
            raise ValueError('missing reason')
        if item.get('confidence') not in ('High', 'Medium', 'Low'):
            raise ValueError('invalid confidence')
        if 'point_id' in item:
            if item.get('answer_text') and item['answer_text'] not in sample['answer']:
                raise ValueError('point answer text not found in original answer')
            if item.get('coverage') not in ('Covered', 'Partially Covered', 'Not Covered', 'Unverified'):
                raise ValueError('invalid point coverage')
            if item.get('evidence_support') not in ('Direct Evidence', 'Partial Evidence', 'Indirect Evidence', 'Insufficient Evidence', 'Unverified'):
                raise ValueError('invalid point support')
        else:
            if item.get('support') not in ('Directly Supported', 'Partially Supported', 'Contradicted', 'Unsupported', 'Unverified'):
                raise ValueError('invalid claim support')
        for quote in item.get('evidence_quotes', []):
            eid = quote.get('evidence_id')
            if eid not in evidence or not quote.get('quote') or quote['quote'] not in evidence[eid]:
                raise ValueError('quote not found in cited evidence')
        if (item.get('support') in ('Directly Supported', 'Partially Supported', 'Contradicted')
                or item.get('evidence_support') in ('Direct Evidence', 'Partial Evidence', 'Indirect Evidence')) and not item.get('evidence_quotes'):
            raise ValueError('support decision lacks source quote')
    scores = value.get('scores', {})
    if set(scores) != {'Completeness', 'Faithfulness', 'Relevance', 'Citation Support'}:
        raise ValueError('score dimensions mismatch')
    for score in scores.values():
        if score != 'Unverified' and (type(score) is not int or not 1 <= score <= 5):
            raise ValueError('invalid score')
    if value.get('citation_validity') not in ('Valid', 'Invalid', 'Unverified'):
        raise ValueError('invalid citation validity')
    if value.get('source_attribution') not in ('Correct', 'Incorrect', 'Unverified'):
        raise ValueError('invalid source attribution')
    if value.get('abstention_quality') not in ('Appropriate Abstention', 'Unnecessary Abstention',
                                               'Partial Abstention', 'Unsupported Answer', 'Unverified'):
        raise ValueError('invalid abstention quality')
    return value


def build():
    questions = read_json(ROOT / 'experiments/results/phase421/question_benchmark.json')
    refs = read_json(ROOT / 'experiments/results/phase421/evidence_reference_map.json')
    ref_by_question = defaultdict(list)
    for row in refs:
        ref_by_question[row['question_id']].append(row)
    claims = read_csv(ROOT / 'experiments/results/phase421/claim_evaluation.csv')
    claim_by_answer = defaultdict(list)
    for row in claims:
        claim_by_answer[(row['question_id'], int(row['k']))].append(row)
    hashes = read_json(ROOT / 'experiments/results/phase419/aggregate_quality_metrics.json')['original_answer_sha256']
    samples, mappings = [], []
    for question in questions:
        qid = question['question_id']
        groups = [3, 5, 8]
        random.Random(f'{SEED}:{qid}').shuffle(groups)
        for label, k in zip('ABC', groups):
            sample_id = f'{qid}-{label}'
            path = ROOT / f'experiments/results/phase418/raw_answers/{qid}_K{k}.json'
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != hashes[f'{qid}_K{k}']:
                raise ValueError(f'{sample_id}: source SHA-256 mismatch')
            saved = json.loads(raw)
            evidence = saved['evidence']
            evidence_ids = {e['evidence_id'] for e in evidence}
            if saved['question'] != question['original_question'] or saved['status'] != 'complete':
                raise ValueError(f'{sample_id}: answer mismatch')
            sample = {
                'sample_id': sample_id,
                'question': question['original_question'],
                'points': [{'point_id': p['point_id'],
                            'point_text': ('Qualify detailed assay comparison according to the supplied evidence.'
                                           if p['point_id'] == 'Q11-P03' else
                                           'Assess whether the supplied snippets contain substantive physical-distancing evidence.'
                                           if p['point_id'] == 'Q13-P02' else p['point_text'])}
                           for p in question['points']],
                'answer': saved['answer'],
                'evidence': [{'evidence_id': e['evidence_id'], 'document_id': e['doc_id'],
                              'paper_title': e['title'], 'chunk_id': e['chunk_id'],
                              'text': e['text'], 'source_role_hint': reference_role(e)}
                             for e in evidence],
                'point_evidence_candidates': [
                    {'point_id': r['point_id'], 'evidence_id': r['evidence_id'],
                     'document_id': r['document_id'], 'chunk_id': r['chunk_id']}
                    for r in ref_by_question[qid] if r['evidence_id'] in evidence_ids],
                'claims': [{'claim_id': c['claim_id'], 'claim_text': c['claim_text'],
                            'citation_ids': json.loads(c['citation_ids']),
                            'document_ids': json.loads(c['document_ids']),
                            'chunk_ids': json.loads(c['chunk_ids'])}
                           for c in claim_by_answer[(qid, k)]],
            }
            if not sample['claims']:
                raise ValueError(f'{sample_id}: missing claims')
            samples.append(sample)
            mappings.append({'sample_id': sample_id, 'question_id': qid, 'evidence_k': k,
                             'source_sha256': hashes[f'{qid}_K{k}']})
    random.Random(SEED).shuffle(samples)
    return samples, mappings


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError('Phase 4.22 results already exist; refusing to overwrite')
    samples, mapping = build()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'blind_evaluation_dataset.json', {'version': 'phase422-blind-v1', 'samples': samples})
    write_json(OUT / 'anonymization_mapping.json', {'seed': SEED, 'mapping': mapping,
        'note': 'Keep this file outside judge payloads. Evidence count itself can reveal likely group.'})
    pilot_ids = [m['sample_id'] for m in mapping if m['question_id'] in ('Q07', 'Q11')]
    write_json(OUT / 'pilot_manifest.json', {'sample_ids': pilot_ids, 'status': 'prepared; no judge API calls',
        'known_guardrails': ['Q07 source says Twenty-four (60%) children were hospitalized; count is numerator',
                             'Q07 adult-unit evidence cannot silently become pediatric outcome evidence',
                             'Q11 antibody detection target is in the later evidence; best timing is not established']})
    write_json(OUT / 'pilot_evaluation_inputs.json', {'status': 'prepared; not judged',
        'samples': [s for s in samples if s['sample_id'] in pilot_ids]})
    write_csv(OUT / 'point_level_evaluation.csv', ['sample_id','point_id','judge_status','coverage','evidence_support','reason'],
              [{'sample_id': s['sample_id'], 'point_id': p['point_id'], 'judge_status': 'Not Run',
                'coverage': 'Unverified', 'evidence_support': 'Unverified', 'reason': 'independent judge unavailable'}
               for s in samples for p in s['points']])
    write_csv(OUT / 'claim_level_evaluation.csv', ['sample_id','claim_id','judge_status','support','reason'],
              [{'sample_id': s['sample_id'], 'claim_id': c['claim_id'], 'judge_status': 'Not Run',
                'support': 'Unverified', 'reason': 'independent judge unavailable'}
               for s in samples for c in s['claims']])
    write_csv(OUT / 'answer_level_evaluation.csv', ['sample_id','judge_status','completeness','faithfulness','relevance','citation_support','abstention_quality'],
              [{'sample_id': s['sample_id'], 'judge_status': 'Not Run', 'completeness': 'Unverified',
                'faithfulness': 'Unverified', 'relevance': 'Unverified', 'citation_support': 'Unverified',
                'abstention_quality': 'Unverified'} for s in samples])
    write_csv(OUT / 'pairwise_comparison.csv', ['question_id','comparison','judge_status','classification','reason'],
              [{'question_id': qid, 'comparison': name, 'judge_status': 'Not Run',
                'classification': 'Unverified', 'reason': 'no independent judge results'}
               for qid in sorted({m['question_id'] for m in mapping})
               for name in ('A vs B','A vs C','B vs C')])
    write_csv(OUT / 'judge_usage.csv', ['sample_id','request_status','model','input_tokens','output_tokens','api_seconds','estimated_cost_cny'], [])
    write_json(OUT / 'judge_consistency.json', {'status': 'Not Run', 'repeated_samples': 0,
        'reason': 'no independent judge configured; no evaluator requests'})
    historic = read_json(ROOT / 'experiments/results/phase418/aggregate_metrics.json')['groups']
    write_json(OUT / 'aggregate_metrics.json', {
        'prepared_samples': len(samples), 'prepared_pilot_samples': len(pilot_ids),
        'judge_completed': 0, 'judge_model': None, 'judge_version': None,
        'judge_scores': 'Unverified', 'judge_usage': 'Not Run',
        'original_generation_by_k': {k: {metric: historic[k][metric] for metric in
            ('api_completion_seconds','input_tokens','output_tokens','estimated_cost_cny')} for k in ('3','5','8')},
        'cost_note': 'original generation cost estimate, not judge cost or provider bill'})
    print(json.dumps({'samples': len(samples), 'pilot': len(pilot_ids),
                      'judge_completed': 0, 'independent_model': None}, ensure_ascii=False))


if __name__ == '__main__':
    main()
