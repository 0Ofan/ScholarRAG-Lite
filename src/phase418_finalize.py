"""Aggregate saved Phase 4.18 responses without further API calls."""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/phase418'


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    offline = json.loads((OUT / 'offline_records.json').read_text(encoding='utf-8'))
    rows, usage_rows = [], []
    for qid, question in offline.items():
        for k in (3, 5, 8):
            record = json.loads((OUT / 'raw_answers' / f'{qid}_K{k}.json').read_text(encoding='utf-8'))
            usage = (record.get('generation') or {}).get('usage') or {}
            citation = record.get('citation_check') or {}
            cost = record.get('estimated_cost') or {}
            rows.append({'question_id': qid, 'category': question['category'], 'k': k,
                         'status': record['status'], 'finish_reason': record.get('finish_reason'),
                         'retrieved_candidates': 10, 'selected_evidence': k, 'context_evidence': k,
                         'input_tokens': usage.get('input_tokens'),
                         'output_tokens': usage.get('output_tokens'),
                         'api_completion_seconds': record.get('api_completion_seconds'),
                         'estimated_cost_cny': cost.get('estimated_cny'),
                         'valid_citation_count': len(citation.get('valid', [])),
                         'invalid_citation_count': len(citation.get('invalid', [])),
                         'uncited_factual_candidate_count': len(citation.get('uncited_factual_candidates', [])),
                         'completeness': 'Unverified', 'faithfulness': 'Unverified',
                         'relevance': 'Unverified', 'citation_support': 'Unverified',
                         'evidence_coverage': 'Unverified'})
            usage_rows.append({'question_id': qid, 'k': k, 'status': record['status'],
                               'finish_reason': record.get('finish_reason'),
                               'input_tokens': usage.get('input_tokens'),
                               'output_tokens': usage.get('output_tokens'),
                               'prompt_cache_hit_tokens': usage.get('prompt_cache_hit_tokens'),
                               'prompt_cache_miss_tokens': usage.get('prompt_cache_miss_tokens'),
                               'api_completion_seconds': record.get('api_completion_seconds'),
                               'estimated_cost_cny': cost.get('estimated_cny'),
                               'cost_status': cost.get('status')})
    write_csv('per_question_results.csv', rows)
    write_csv('api_usage.csv', usage_rows)
    groups = {}
    for k in (3, 5, 8):
        subset = [r for r in rows if r['k'] == k]
        def stats(field):
            values = [r[field] for r in subset if isinstance(r[field], (int, float))]
            return {'mean': statistics.mean(values), 'median': statistics.median(values),
                    'stdev': statistics.stdev(values), 'total': sum(values)} if values else None
        groups[str(k)] = {'questions': len(subset), 'complete': sum(r['status'] == 'complete' for r in subset),
                          'finish_reasons': sorted(set(str(r['finish_reason']) for r in subset)),
                          'input_tokens': stats('input_tokens'), 'output_tokens': stats('output_tokens'),
                          'api_completion_seconds': stats('api_completion_seconds'),
                          'estimated_cost_cny': stats('estimated_cost_cny'),
                          'invalid_citations': sum(r['invalid_citation_count'] for r in subset),
                          'uncited_factual_candidates': sum(r['uncited_factual_candidate_count'] for r in subset),
                          'quality_scores': 'Unverified: no independent semantic review or extra evaluator API calls'}
    result = {'version': 'phase418-aggregate-v1', 'total_real_requests': len(rows),
              'all_three_groups_same_questions': len({r['question_id'] for r in rows}) == 20,
              'groups': groups, 'citation_scope': 'structural only; valid IDs do not prove semantic support',
              'cost_scope': 'estimated from actual usage and published price; not provider bill'}
    (OUT / 'aggregate_metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('aggregated', len(rows), 'real answers')


if __name__ == '__main__':
    main()
