"""Two-request live check of the existing V1/V2 streaming path."""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for name, folder in {'HF_HOME': 'huggingface', 'TORCH_HOME': 'torch',
                     'PIP_CACHE_DIR': 'pip'}.items():
    os.environ[name] = str(ROOT / '.cache' / folder)

from src.deepseek_client import DeepSeekClient, DeepSeekRequestError, is_complete_generation
from src.evidence_metadata import enrich
from src.evidence_verifier import verify
from src.rag_service import load_retriever, retrieve_evidence, prepare_grounded_evidence
from src.citation_grounding import verify_evidence_source, validate_grounded_answer

OUT = ROOT / 'experiments/results/phase417'
RESULT = OUT / 'live_validation.json'


def save(record):
    RESULT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if RESULT.exists():
        raise FileExistsError('Phase 4.17 真实请求记录已存在；不会重复调用 API。')
    questions = json.loads((ROOT / 'experiments/configs/phase36_questions.json').read_text(encoding='utf-8'))['questions']
    question = next(q['question'] for q in questions if q['question_id'] == 'Q01')
    load_start = time.perf_counter()
    retriever, _ = load_retriever('phase33_large_fixed')
    load_seconds = time.perf_counter() - load_start
    raw, docs, retrieval_timing = retrieve_evidence(retriever, question, 3)
    evidence = prepare_grounded_evidence(raw)
    source_map = {e['chunk_id']: retriever.chunks[d['chunk_index']] for e, d in zip(evidence, docs)}
    source_checks = verify_evidence_source(evidence, source_map)
    if any(c['status'] != 'valid' for c in source_checks):
        raise ValueError('原始 Chunk 来源映射失败')
    client = DeepSeekClient()
    prompt = (ROOT / 'prompts/scholar_rag_v2.txt').read_text(encoding='utf-8').strip()
    record = {'question_id': 'Q01', 'question': question, 'index_load_seconds': load_seconds,
              'retrieval_timing': retrieval_timing, 'evidence': evidence,
              'source_checks': source_checks, 'config': {'model': client.config['model'],
              'temperature': client.config['temperature'], 'max_tokens': client.config['max_tokens'],
              'thinking': 'disabled', 'stream': True}, 'results': {},
              'browser_interaction_verified': False}
    save(record)
    for mode in ('V1', 'V2'):
        selected = evidence if mode == 'V1' else enrich(evidence)
        item = {'mode': mode, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'status': 'request_started', 'answer': '', 'finish_reason': None}
        record['results'][mode] = item
        save(record)
        start = time.perf_counter()
        parts, metadata = [], None
        try:
            for kind, payload in client.stream(question, selected,
                                               system_prompt=prompt if mode == 'V2' else None):
                if kind == 'text':
                    parts.append(payload)
                elif kind == 'done':
                    metadata = payload
            answer = ''.join(parts)
            complete = is_complete_generation(answer, metadata)
            item.update(status='complete' if complete else 'incomplete', answer=answer,
                        streamed_text_received=bool(answer.strip()),
                        finish_reason=(metadata or {}).get('finish_reason'),
                        generation=metadata, api_wall_seconds=time.perf_counter() - start)
            if complete:
                item['citation_check'] = validate_grounded_answer(answer, selected, source_checks)
                if mode == 'V2':
                    item['verification'] = verify(answer, selected)
        except DeepSeekRequestError as exc:
            item.update(status='interrupted' if ''.join(parts).strip() else 'api_error',
                        answer=''.join(parts), streamed_text_received=bool(''.join(parts).strip()),
                        error_type=type(exc).__name__, error_message=str(exc),
                        api_wall_seconds=time.perf_counter() - start)
        save(record)
        print(mode, item['status'], item.get('finish_reason'), flush=True)
        if item['status'] in ('api_error', 'interrupted'):
            break


if __name__ == '__main__':
    main()
