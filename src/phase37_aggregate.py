"""Validate paired Phase 3.6/3.7 records and export auditable comparisons."""
import csv
import hashlib
import json
import statistics as stats
from collections import Counter
from pathlib import Path

from src.phase37_prepare import digest

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'experiments/results/phase36'
OUT = ROOT/'experiments/results/phase37'


def describe(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {'n': 0, 'mean': None, 'median': None, 'p90': None}
    p = (len(values)-1)*.9
    i = int(p)
    return {'n': len(values), 'mean': stats.mean(values), 'median': stats.median(values),
            'p90': values[i] + (values[min(i+1, len(values)-1)]-values[i])*(p-i)}


def main():
    manifest = json.loads((OUT/'inputs_manifest.json').read_text(encoding='utf8'))
    entries = {x['question_id']: x for x in manifest['entries']}
    questions = json.loads((ROOT/'experiments/configs/phase36_questions.json').read_text(encoding='utf8'))['questions']
    old_review = {x['question_id']: x for x in json.loads(
        (OLD/'reviewer_scores.json').read_text(encoding='utf8'))['reviews']}
    new_review = {x['question_id']: x for x in json.loads(
        (OUT/'reviewer_scores.json').read_text(encoding='utf8'))['reviews']}
    assert len(questions) == len(entries) == len(old_review) == len(new_review) == 20
    rows = []
    for q in questions:
        qid = q['question_id']
        old_bytes = (OLD/f'{qid}.json').read_bytes()
        assert hashlib.sha256(old_bytes).hexdigest() == entries[qid]['phase36_record_sha256']
        old = json.loads(old_bytes)
        new = json.loads((OUT/f'{qid}.json').read_text(encoding='utf8'))
        assert old['question'] == new['question'] == q['question']
        assert old['status'] == 'success' and new['status'] == 'success' and new['real_api']
        assert digest(old['evidence']) == entries[qid]['evidence_sha256']
        assert [e['doc_id'] for e in old['evidence']] == [e['doc_id'] for e in new['evidence']]
        assert [e['chunk_id'] for e in old['evidence']] == [e['chunk_id'] for e in new['evidence']]
        assert [e['title'] for e in old['evidence']] == [e['title'] for e in new['evidence']]
        assert [e['text'] for e in old['evidence']] == [e['text'] for e in new['evidence']]
        assert all(x['status'] == 'valid' for x in new['source_checks'])
        assert new['generation']['response_model'] == old['generation']['response_model'] == 'deepseek-flash'
        assert new['generation']['thinking_requested_disabled'] and not new['generation']['reasoning_content_seen']
        review = new_review[qid]
        old_score = old_review[qid]
        assert all(1 <= v <= 5 for v in review['scores'].values())
        row = {'question_id': qid, 'category': q['category'], 'question': q['question'],
               'old_answer': old['answer'], 'new_answer': new['answer'],
               'evidence_document_ids': '|'.join(x['doc_id'] for x in old['evidence']),
               'old_citation_support': old_score['citation_support'],
               'new_citation_support': review['citation_support'],
               'new_invalid_citations': '|'.join(new['citation_check']['invalid']),
               'new_uncited_fact_candidates': len(new['citation_check']['uncited_factual_candidates']),
               'new_multi_source_sentence_candidates': len(new['citation_check']['multi_source_sentences_for_review']),
               'new_nonterminal_citation_sentences': len(new['citation_check']['nonterminal_citation_sentences']),
               'new_explicit_source_mismatches': len(new['citation_check']['explicit_attribution_mismatch_sentences']),
               'source_attribution_error_reviewed': review['source_attribution_error'],
               'overgeneralization_reviewed': review['overgeneralization'],
               'abstention_reviewed': review['abstention'] or '',
               'review_note': review['review_note'],
               'old_api_ttft_seconds': old['generation']['api_ttft_seconds'],
               'new_api_ttft_seconds': new['generation']['api_ttft_seconds'],
               'old_api_total_seconds': old['generation']['api_total_seconds'],
               'new_api_total_seconds': new['generation']['api_total_seconds'],
               'new_evidence_preparation_seconds': new['evidence_preparation_seconds_measured'],
               'old_input_tokens': old['generation']['usage']['input_tokens'],
               'new_input_tokens': new['generation']['usage']['input_tokens'],
               'old_output_tokens': old['generation']['usage']['output_tokens'],
               'new_output_tokens': new['generation']['usage']['output_tokens'],
               'old_cost_estimate_cny': old['cost'].get('estimated_cny'),
               'new_cost_estimate_cny': new['cost'].get('estimated_cny')}
        for name in ('correctness', 'completeness', 'relevance', 'faithfulness'):
            row[f'old_{name}'] = old_score['scores'][name]
            row[f'new_{name}'] = review['scores'][name]
        rows.append(row)
    with (OUT/'per_question_comparison.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = ['api_ttft_seconds', 'api_total_seconds', 'input_tokens', 'output_tokens',
            'cost_estimate_cny', 'correctness', 'completeness', 'relevance', 'faithfulness']
    aggregate = {'version': 'phase37-paired-aggregate-v1', 'question_count': 20,
                 'real_api_successes': 20,
                 'same_questions_evidence_order_and_text_verified': True,
                 'reviewer': 'Codex assistant; non-blinded evidence review, not independent human evaluation',
                 'metrics': {name: {
                     'old': describe([r['old_'+name] for r in rows]),
                     'new': describe([r['new_'+name] for r in rows]),
                     'mean_difference_new_minus_old': stats.mean(r['new_'+name]-r['old_'+name] for r in rows),
                     } for name in keys},
                 'citation_support_old': dict(Counter(r['old_citation_support'] for r in rows)),
                 'citation_support_new': dict(Counter(r['new_citation_support'] for r in rows)),
                 'old_partial_to_new_supported': [r['question_id'] for r in rows
                                                  if r['old_citation_support'] == 'partial' and r['new_citation_support'] == 'supported'],
                 'old_supported_to_new_partial': [r['question_id'] for r in rows
                                                  if r['old_citation_support'] == 'supported' and r['new_citation_support'] == 'partial'],
                 'new_reviewed_source_attribution_errors': [r['question_id'] for r in rows if r['source_attribution_error_reviewed']],
                 'new_reviewed_overgeneralizations': [r['question_id'] for r in rows if r['overgeneralization_reviewed']],
                 'new_invalid_citation_question_ids': [r['question_id'] for r in rows if r['new_invalid_citations']],
                 'new_uncited_fact_candidate_question_count': sum(r['new_uncited_fact_candidates'] > 0 for r in rows),
                 'new_multi_source_sentence_candidate_question_count': sum(r['new_multi_source_sentence_candidates'] > 0 for r in rows),
                 'new_nonterminal_citation_question_count': sum(r['new_nonterminal_citation_sentences'] > 0 for r in rows),
                 'abstention_Q19_Q20_correct': all(new_review[q]['abstention'] == 'correct' for q in ('Q19', 'Q20')),
                 'token_totals': {name: sum(r[name] for r in rows) for name in
                     ('old_input_tokens', 'new_input_tokens', 'old_output_tokens', 'new_output_tokens')},
                 'cost_total_estimate_cny': {side: round(sum(r[f'{side}_cost_estimate_cny'] for r in rows), 8)
                                              for side in ('old', 'new')}}
    (OUT/'aggregate.json').write_text(json.dumps(aggregate, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
