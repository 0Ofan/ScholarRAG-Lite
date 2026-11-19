"""One-pass real Phase 3.6 RAG evaluation with per-question checkpoints."""
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.deepseek_client import DeepSeekClient, DeepSeekRequestError, MissingAPIKey
from src.rag_service import load_retriever, retrieve_evidence, validate_citations

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'experiments/configs/phase36_questions.json'
LOCK = ROOT / 'experiments/configs/phase36_question_lock.json'
OUT = ROOT / 'experiments/results/phase36'
BEIJING = timezone(timedelta(hours=8))
PRICE_URL = 'https://api-docs.deepseek.com/zh-cn/quick_start/pricing/'


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def cost_estimate(usage, at):
    """Compute published RMB token charge; unavailable if cache split is absent."""
    if not usage:
        return {'status': 'unavailable', 'reason': 'API did not return Token Usage'}
    hit = usage.get('prompt_cache_hit_tokens')
    miss = usage.get('prompt_cache_miss_tokens')
    output = usage.get('output_tokens')
    if None in (hit, miss, output):
        return {'status': 'unavailable', 'reason': 'API did not return input cache hit/miss split'}
    if hit + miss != usage.get('input_tokens'):
        return {'status': 'unavailable', 'reason': 'cache hit/miss split does not sum to input tokens'}
    local = at.astimezone(BEIJING)
    minute = local.hour * 60 + local.minute
    peak = local.weekday() < 5 and (540 <= minute < 720 or 840 <= minute < 1080)
    rates = {'hit': .04 if peak else .02, 'miss': 2.0 if peak else 1.0,
             'output': 8.0 if peak else 4.0}
    return {'status': 'computed_from_usage_and_published_price',
            'estimated_cny': round((hit*rates['hit'] + miss*rates['miss'] + output*rates['output'])/1_000_000, 8),
            'pricing_period': 'peak' if peak else 'off_peak',
            'rates_cny_per_million_tokens': rates,
            'pricing_time_beijing': local.isoformat(),
            'caveat': 'Published schedule and actual token usage; not a verified provider bill. Public-holiday exception must be checked.',
            'source': PRICE_URL}


def main():
    raw = CONFIG.read_bytes()
    lock = json.loads(LOCK.read_text(encoding='utf8'))
    # Text written on Windows may use CRLF; the lock hashes canonical LF JSON.
    digest = hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()
    if digest != lock['sha256']:
        raise SystemExit('Frozen question-set hash mismatch; refusing API requests.')
    questions = json.loads(raw)['questions']
    if len(questions) != 20 or len({q['question_id'] for q in questions}) != 20:
        raise SystemExit('Expected exactly 20 unique frozen questions.')
    OUT.mkdir(parents=True, exist_ok=True)
    if all((OUT/f"{q['question_id']}.json").exists() for q in questions):
        print('All 20 question records already exist; no API request or output overwrite.', flush=True)
        return
    retriever, cold = load_retriever('phase33_large_fixed')
    metadata_path = OUT/'run_metadata.json'
    if not metadata_path.exists():
        json_write(metadata_path, {'question_set_sha256': digest, 'run_id': 'phase33_large_fixed',
                                   'model': 'deepseek-flash', 'thinking': 'disabled', 'streaming': True,
                                   'temperature': .3, 'evidence_count': 3,
                                   'cold_load_seconds': cold,
                                   'cold_load_excluded_from_per_query_latency': True,
                                   'started_at_utc': datetime.now(timezone.utc).isoformat(),
                                   'pricing_source': PRICE_URL})
    client = DeepSeekClient()
    if client.config.get('temperature') != .3:
        raise SystemExit('Temperature differs from frozen Phase 3.6 protocol.')
    for question in questions:
        qid = question['question_id']
        path = OUT / f'{qid}.json'
        if path.exists():
            print(qid, 'already recorded; skipped', flush=True)
            continue
        start = time.perf_counter()
        at = datetime.now(timezone.utc)
        result = {'question_id': qid, 'question_set_sha256': digest,
                  'question': question['question'], 'category': question['category'],
                  'started_at_utc': at.isoformat(), 'real_api': True,
                  'retrieval_run_id': 'phase33_large_fixed'}
        try:
            evidence, top10, timing = retrieve_evidence(retriever, question['question'], 3)
            result.update(evidence=evidence, retrieval_top10=top10[:10],
                          retrieval_only={'format': 'titles_and_original_evidence',
                                          'evidence': evidence}, timing=timing,
                          status='retrieval_complete_api_pending')
            json_write(path, result)  # Checkpoint before a paid, non-retried request.
            if not evidence:
                raise RuntimeError('No evidence returned')
            chunks = []
            first = None
            meta = None
            for kind, payload in client.stream(question['question'], evidence):
                if kind == 'text':
                    if first is None and payload.strip():
                        first = time.perf_counter() - start
                    chunks.append(payload)
                elif kind == 'done':
                    meta = payload
            answer = ''.join(chunks)
            end = time.perf_counter()
            result.update(status='success' if answer.strip() else 'empty_answer', answer=answer,
                          generation=meta, citation_check=validate_citations(answer, evidence),
                          timing={**timing, 'submit_to_first_text_seconds': first,
                                  'submit_to_complete_seconds': end-start},
                          cost=cost_estimate(meta['usage'] if meta else None, at))
        except (DeepSeekRequestError, MissingAPIKey, RuntimeError) as exc:
            result.update(status='error', error_type=type(exc).__name__,
                          error_message=str(exc) if isinstance(exc, (DeepSeekRequestError, MissingAPIKey)) else '检索证据不可用',
                          timing={**result.get('timing', {}),
                                  'submit_to_complete_seconds': time.perf_counter()-start})
        json_write(path, result)
        print(qid, result['status'],
              round(result['timing']['submit_to_complete_seconds'], 3), 's', flush=True)


if __name__ == '__main__':
    main()
