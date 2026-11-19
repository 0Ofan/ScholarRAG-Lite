"""Checkpointed v2 generation on the exact saved Phase 4.5 A evidence."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.answer_refinement import refine_once
from src.deepseek_client import DeepSeekClient, DeepSeekRequestError
from src.evidence_metadata import enrich
from src.evidence_verifier import verify
from src.phase36_run import cost_estimate
from src.phase37_prepare import digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/phase49'
P45 = ROOT / 'experiments/results/phase45'
PROMPT = (ROOT / 'prompts/scholar_rag_v2.txt').read_text(encoding='utf8').strip()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    entries = json.loads((P45 / 'inputs.json').read_text(encoding='utf8'))['entries']
    client = DeepSeekClient()
    assert (client.config['model'], client.config['temperature'], client.config['max_tokens']) == ('deepseek-flash', .3, 512)
    for entry in entries:
        qid = entry['question_id']
        path = OUT / f'{qid}.json'
        previous = None
        if path.exists():
            previous = json.loads(path.read_text(encoding='utf8'))
            if not (previous.get('status') == 'error' and 'APIConnectionError' in previous.get('error_message', '')
                    and previous.get('retry_count', 0) == 0):
                print(qid, 'checkpoint exists', flush=True)
                continue
        v1 = json.loads((P45 / f'A_{qid}.json').read_text(encoding='utf8'))
        if not (v1['status'] == 'success' and v1['real_api'] and v1['question'] == entry['question']
                and v1['evidence_sha256'] == entry['A']['evidence_sha256']
                and digest(v1['evidence']) == v1['evidence_sha256']
                and v1['model'] == 'deepseek-flash' and v1['temperature'] == .3
                and v1['max_tokens'] == 512 and v1['thinking'] == 'disabled'):
            raise ValueError(f'{qid}: V1 comparison input differs')
        evidence = enrich(v1['evidence'])
        started = datetime.now(timezone.utc)
        wall = time.perf_counter()
        record = {'version': 'phase49-v2-real-v1', 'question_id': qid, 'question': entry['question'],
                  'v1_record': f'experiments/results/phase45/A_{qid}.json',
                  'v1_evidence_sha256': v1['evidence_sha256'], 'evidence': evidence,
                  'prompt_path': 'prompts/scholar_rag_v2.txt', 'model': 'deepseek-flash',
                  'temperature': .3, 'max_tokens': 512, 'thinking': 'disabled',
                  'started_at_utc': started.isoformat(), 'real_api': True, 'status': 'pending'}
        if previous:
            record['retry_count'] = 1
            record['previous_failed_attempt'] = {key: previous.get(key) for key in
                                                 ('started_at_utc', 'error_type', 'error_message', 'wall_seconds')}
        write(path, record)
        try:
            parts, generation = [], None
            for kind, payload in client.stream(entry['question'], evidence, system_prompt=PROMPT):
                if kind == 'text':
                    parts.append(payload)
                else:
                    generation = payload
            answer = ''.join(parts)
            check = verify(answer, evidence)
            revision = refine_once(client, entry['question'], answer, evidence, check, system_prompt=PROMPT) if check['critical'] else None
            if revision:
                answer, check = revision['answer'], revision['verification']
            record.update(status='success' if answer.strip() else 'empty_answer', answer=answer,
                          original_answer=''.join(parts), generation=generation,
                          verification=check, revision=revision,
                          wall_seconds=time.perf_counter() - wall,
                          cost=cost_estimate(generation['usage'], started))
        except DeepSeekRequestError as exc:
            record.update(status='error', error_type=type(exc).__name__,
                          error_message=str(exc), wall_seconds=time.perf_counter() - wall)
        write(path, record)
        print(qid, record['status'], round(record['wall_seconds'], 3), flush=True)


if __name__ == '__main__':
    main()
