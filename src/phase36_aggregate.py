"""Validate real records and export reproducible Phase 3.6 summaries."""
import csv
import hashlib
import json
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/phase36'
CONFIG = ROOT / 'experiments/configs/phase36_questions.json'


def percentile(values, percent):
    values = sorted(values)
    p = (len(values)-1)*percent/100
    low = int(p)
    return values[low] + (values[min(low+1, len(values)-1)]-values[low])*(p-low)


def describe(values):
    values = [v for v in values if v is not None]
    return {'n': len(values), 'mean': stats.mean(values) if values else None,
            'median': stats.median(values) if values else None,
            'p90': percentile(values, 90) if values else None,
            'min': min(values) if values else None,
            'max': max(values) if values else None}


def main():
    raw = CONFIG.read_bytes()
    digest = hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()
    lock = json.loads((ROOT/'experiments/configs/phase36_question_lock.json').read_text(encoding='utf8'))
    assert digest == lock['sha256']
    questions = json.loads(raw)['questions']
    review = json.loads((OUT/'reviewer_scores.json').read_text(encoding='utf8'))
    reviews = {r['question_id']: r for r in review['reviews']}
    assert len(questions) == len(reviews) == 20
    rows = []
    categories = defaultdict(list)
    for q in questions:
        qid = q['question_id']
        item = json.loads((OUT/f'{qid}.json').read_text(encoding='utf8'))
        judgement = reviews[qid]
        assert item['question_set_sha256'] == digest
        assert item['status'] == 'success' and item['real_api']
        assert item['generation']['thinking_requested_disabled']
        assert not item['generation']['reasoning_content_seen']
        assert len(item['evidence']) == 3
        assert [e['doc_id'] for e in item['evidence']] == [d['doc_id'] for d in item['retrieval_top10'][:3]]
        valid_ids = {e['citation'] for e in item['evidence']}
        assert set(item['citation_check']['valid']) <= valid_ids
        assert not item['citation_check']['invalid']
        assert all(1 <= s <= 5 for s in judgement['scores'].values())
        t = item['timing']
        u = item['generation']['usage']
        c = item['cost']
        row = {'question_id': qid, 'category': q['category'],
               'evidence_availability': q['evidence_availability'],
               'status': item['status'], 'question': q['question'], 'answer': item['answer'],
               'retrieved_document_ids': '|'.join(e['doc_id'] for e in item['evidence']),
               'actual_citations': '|'.join(item['citation_check']['valid']),
               'citation_validity': 'valid', 'citation_support': judgement['citation_support'],
               'abstention': judgement['abstention'] or '',
               'retrieval_only_usefulness': judgement['retrieval_only_usefulness'],
               'rag_usefulness': judgement['rag_usefulness'],
               **judgement['scores'],
               'query_embedding_seconds': t['query_embedding_seconds'],
               'faiss_retrieval_seconds': t['retrieval_seconds'],
               'evidence_preparation_seconds': t['evidence_seconds'],
               'retrieval_total_seconds': t['retrieval_total_seconds'],
               'api_ttft_seconds': item['generation']['api_ttft_seconds'],
               'llm_generation_seconds': item['generation']['api_total_seconds'],
               'end_to_end_first_text_seconds': t['submit_to_first_text_seconds'],
               'end_to_end_complete_seconds': t['submit_to_complete_seconds'],
               'input_tokens': u['input_tokens'], 'output_tokens': u['output_tokens'],
               'cache_hit_tokens': u['prompt_cache_hit_tokens'],
               'cache_miss_tokens': u['prompt_cache_miss_tokens'],
               'cost_estimate_cny': c.get('estimated_cny'),
               'pricing_period': c.get('pricing_period'),
               'review_note': judgement['review_note']}
        rows.append(row)
        categories[q['category']].append(row)
    with (OUT/'per_question.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = ['query_embedding_seconds', 'faiss_retrieval_seconds', 'evidence_preparation_seconds',
               'retrieval_total_seconds', 'api_ttft_seconds', 'llm_generation_seconds',
               'end_to_end_first_text_seconds', 'end_to_end_complete_seconds',
               'input_tokens', 'output_tokens', 'cache_hit_tokens', 'cache_miss_tokens',
               'cost_estimate_cny', 'correctness', 'completeness', 'relevance', 'faithfulness',
               'retrieval_only_usefulness', 'rag_usefulness']
    aggregate = {'version': 'phase36-aggregate-v1', 'question_set_sha256': digest,
                 'question_count': len(rows), 'successful_real_api_calls': len(rows),
                 'cold_load_seconds': json.loads((OUT/'run_metadata.json').read_text(encoding='utf8'))['cold_load_seconds'],
                 'pricing_source': 'https://api-docs.deepseek.com/zh-cn/quick_start/pricing/',
                 'metrics': {name: describe([r[name] for r in rows]) for name in metrics},
                 'category_metrics': {category: {
                     'n': len(group),
                     'correctness': describe([r['correctness'] for r in group]),
                     'completeness': describe([r['completeness'] for r in group]),
                     'faithfulness': describe([r['faithfulness'] for r in group]),
                     'rag_usefulness': describe([r['rag_usefulness'] for r in group]),
                     } for category, group in categories.items()},
                 'citation_validity': Counter(r['citation_validity'] for r in rows),
                 'citation_support': Counter(r['citation_support'] for r in rows),
                 'abstention': Counter(r['abstention'] for r in rows if r['abstention']),
                 'rag_usefulness_wins': sum(r['rag_usefulness'] > r['retrieval_only_usefulness'] for r in rows),
                 'rag_usefulness_ties': sum(r['rag_usefulness'] == r['retrieval_only_usefulness'] for r in rows),
                 'rag_usefulness_losses': sum(r['rag_usefulness'] < r['retrieval_only_usefulness'] for r in rows),
                 'cost_estimate_total_cny': round(sum(r['cost_estimate_cny'] for r in rows), 8),
                 'actual_token_totals': {'input': sum(r['input_tokens'] for r in rows),
                                         'output': sum(r['output_tokens'] for r in rows),
                                         'cache_hit': sum(r['cache_hit_tokens'] for r in rows),
                                         'cache_miss': sum(r['cache_miss_tokens'] for r in rows)}}
    (OUT/'aggregate.json').write_text(json.dumps(aggregate, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
