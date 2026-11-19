"""Aggregate frozen A/B and new C under one automated rubric, without new API calls."""
import csv
import json
import statistics
from collections import Counter

from src.phase38_evaluate import load_triplet
from src.phase38_run import OUT, load_inputs, write


def describe(numbers):
    if not numbers:
        return {'n': 0, 'mean': None, 'median': None, 'p90': None, 'sum': None}
    sorted_values = sorted(numbers)
    pos = .9 * (len(sorted_values)-1)
    low = int(pos)
    p90 = sorted_values[low] + (sorted_values[min(low+1, len(sorted_values)-1)]-sorted_values[low])*(pos-low)
    return {'n': len(numbers), 'mean': statistics.mean(numbers), 'median': statistics.median(numbers),
            'p90': p90, 'sum': sum(numbers)}


def main():
    questions, manifest, _ = load_inputs()
    checks = json.loads((OUT/'deterministic_checks.json').read_text(encoding='utf8'))
    if len(checks['results']) != 60:
        raise SystemExit('Missing deterministic checks')
    rows = []
    complete = []
    for question in questions:
        qid = question['question_id']
        records, evidence = load_triplet(question)
        evaluation = json.loads((OUT/f'eval_{qid}.json').read_text(encoding='utf8'))
        if evaluation['status'] != 'success':
            raise SystemExit(f'Evaluator incomplete: {qid}')
        if set(evaluation['evaluations_by_version']) != set('ABC'):
            raise SystemExit(f'Evaluator missing response: {qid}')
        per_question = {'question_id': qid, 'question': question['question'],
                        'category': question['category'], 'expected_answer_points': question['expected_answer_points'],
                        'evidence_availability': question['evidence_availability'],
                        'evidence': evidence, 'versions': {}}
        for version in 'ABC':
            record = records[version]
            judged = evaluation['evaluations_by_version'][version]
            rule = next(x['check'] for x in checks['results']
                        if x['question_id'] == qid and x['version'] == version)
            usage = record['generation']['usage']
            timing = record['generation']
            row = {'question_id': qid, 'category': question['category'], 'version': version,
                   'status': record['status'], 'Correctness': judged['scores']['Correctness']['score'],
                   'Completeness': judged['scores']['Completeness']['score'],
                   'Relevance': judged['scores']['Relevance']['score'],
                   'Faithfulness': judged['scores']['Faithfulness']['score'],
                   'citation_support': judged['citation_support']['class'],
                   'source_attribution': judged['source_attribution']['class'],
                   'overgeneralization': judged['overgeneralization']['class'],
                   'abstention': judged['abstention']['class'],
                   'claim_direct': sum(c['support']=='Direct Support' for c in judged['claims']),
                   'claim_reasonable': sum(c['support']=='Reasonable Inference' for c in judged['claims']),
                   'claim_partial': sum(c['support']=='Partial Support' for c in judged['claims']),
                   'claim_unsupported': sum(c['support']=='Unsupported' for c in judged['claims']),
                   'invalid_citations': len(rule['citation']['invalid']),
                   'uncited_factual_candidates': len(rule['citation']['uncited_factual_candidates']),
                   'attribution_mismatch_candidates': len(rule['citation']['explicit_attribution_mismatch_sentences']),
                   'qualifier_loss_candidates': len(rule['qualifier_loss_candidates']),
                   'numeric_warning_candidates': len(rule['numeric_claims_without_same_number_in_cited_text']),
                   'api_ttft_seconds': timing['api_ttft_seconds'],
                   'api_total_seconds': timing['api_total_seconds'],
                   'input_tokens': usage['input_tokens'] if usage else None,
                   'output_tokens': usage['output_tokens'] if usage else None,
                   'cost_estimate_cny': record['cost'].get('estimated_cny'),
                   'answer': record['answer'],
                   'citation_support_reason': judged['citation_support']['reason'],
                   'source_attribution_reason': judged['source_attribution']['reason'],
                   'abstention_reason': judged['abstention']['reason']}
            rows.append(row)
            per_question['versions'][version] = {'answer': record['answer'], 'automated_semantic_evaluation': judged,
                                                  'deterministic_checks': rule,
                                                  'generation': record['generation'], 'cost': record['cost']}
        complete.append(per_question)
    write(OUT/'comparison.json', {'version': 'phase38-comparison-v1', 'label': 'Automated Evaluation',
                                  'input_manifest': manifest, 'questions': complete})
    with (OUT/'per_question_comparison.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate = {'version': 'phase38-aggregate-v1', 'label': 'Automated Evaluation',
                 'question_count': len(questions), 'real_C_successes': sum(x['status']=='success' for x in rows if x['version']=='C'),
                 'evaluator_successes': len(questions), 'generator': {}, 'evaluator': {},
                 'limitations': ['Same-service DeepSeek judge, not independent human review',
                                 '20 frozen questions support exploratory comparisons only',
                                 'Deterministic warnings do not prove semantic support']}
    for version in 'ABC':
        selected = [x for x in rows if x['version']==version]
        aggregate['generator'][version] = {
            'quality': {name: describe([r[name] for r in selected]) for name in
                        ('Correctness','Completeness','Relevance','Faithfulness')},
            'citation_support_counts': dict(Counter(r['citation_support'] for r in selected)),
            'source_attribution_counts': dict(Counter(r['source_attribution'] for r in selected)),
            'overgeneralization_counts': dict(Counter(r['overgeneralization'] for r in selected)),
            'abstention_counts': dict(Counter(r['abstention'] for r in selected)),
            'claim_support_counts': {key: sum(r[key] for r in selected) for key in
                ('claim_direct','claim_reasonable','claim_partial','claim_unsupported')},
            'performance': {name: describe([r[name] for r in selected if r[name] is not None]) for name in
                ('api_ttft_seconds','api_total_seconds','input_tokens','output_tokens','cost_estimate_cny')},
            'structural_warning_counts': {name: sum(r[name] for r in selected) for name in
                ('invalid_citations','uncited_factual_candidates','attribution_mismatch_candidates',
                 'qualifier_loss_candidates','numeric_warning_candidates')}}
    evaluations = [json.loads((OUT/f"eval_{q['question_id']}.json").read_text(encoding='utf8')) for q in questions]
    aggregate['evaluator'] = {'model': 'deepseek-flash', 'calls': len(evaluations),
        'same_service_as_generator': True, 'prompt_sha256': evaluations[0]['evaluator_prompt_sha256'],
        'usage_and_cost': {name: describe([e['usage'][name] if name != 'cost_estimate_cny' else e['cost'].get('estimated_cny')
            for e in evaluations if e.get('usage') and
            (name != 'cost_estimate_cny' or e['cost'].get('estimated_cny') is not None)])
            for name in ('input_tokens','output_tokens','cost_estimate_cny')},
        'elapsed_seconds': describe([e['elapsed_seconds'] for e in evaluations])}
    write(OUT/'aggregate.json', aggregate)
    print('Aggregated', len(rows), 'A/B/C rows across', len(questions), 'questions.')


if __name__ == '__main__':
    main()
