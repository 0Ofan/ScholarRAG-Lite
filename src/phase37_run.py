"""One new real request per frozen Phase 3.6 question; never rerun retrieval."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.citation_grounding import validate_grounded_answer
from src.deepseek_client import DeepSeekClient, DeepSeekRequestError, MissingAPIKey
from src.phase36_run import cost_estimate
from src.phase37_prepare import digest
from src.rag_service import prepare_grounded_evidence

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'experiments/results/phase36'
OUT = ROOT/'experiments/results/phase37'
CONFIG = ROOT/'experiments/configs/phase36_questions.json'


def save(path, value):
    with path.open('w', encoding='utf8') as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write('\n')


def main():
    questions = json.loads(CONFIG.read_text(encoding='utf8'))['questions']
    manifest = json.loads((OUT/'inputs_manifest.json').read_text(encoding='utf8'))
    entries = {e['question_id']: e for e in manifest['entries']}
    qhash = hashlib.sha256(CONFIG.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    if qhash != manifest['question_set_sha256'] or len(questions) != 20 or len(entries) != 20:
        raise SystemExit('Frozen Phase 3.7 inputs do not match the Phase 3.6 question set.')
    if all((OUT/f"{q['question_id']}.json").exists() for q in questions):
        print('All Phase 3.7 question records exist; no API requests repeated.', flush=True)
        return
    client = DeepSeekClient()
    if client.config['model'] != 'deepseek-flash' or client.config['temperature'] != .3 or client.config['max_tokens'] != 512:
        raise SystemExit('Model, temperature or max output tokens differ from frozen comparison protocol.')
    for q in questions:
        qid = q['question_id']
        path = OUT/f'{qid}.json'
        if path.exists():
            print(qid, 'already recorded; skipped', flush=True)
            continue
        start = time.perf_counter()
        old_path = OLD/f'{qid}.json'
        old_bytes = old_path.read_bytes()
        old = json.loads(old_bytes)
        entry = entries[qid]
        if hashlib.sha256(old_bytes).hexdigest() != entry['phase36_record_sha256']:
            raise SystemExit(f'Historical Phase 3.6 result changed: {qid}')
        if q['question'] != old['question'] or digest(old['evidence']) != entry['evidence_sha256']:
            raise SystemExit(f'Frozen question or evidence differs: {qid}')
        prepared = prepare_grounded_evidence(old['evidence'])
        if digest(prepared) != entry['grounded_evidence_sha256']:
            raise SystemExit(f'Evidence context differs from Phase 3.7 manifest: {qid}')
        prep_seconds = time.perf_counter()-start
        at = datetime.now(timezone.utc)
        result = {'version': 'phase37-real-v1', 'question_id': qid, 'question': q['question'],
                  'category': q['category'], 'question_set_sha256': qhash,
                  'phase36_record_sha256': entry['phase36_record_sha256'],
                  'original_evidence_sha256': entry['evidence_sha256'],
                  'grounded_evidence_sha256': entry['grounded_evidence_sha256'],
                  'evidence': prepared, 'source_checks': entry['source_checks'],
                  'model': 'deepseek-flash', 'temperature': .3, 'max_tokens': 512,
                  'thinking': 'disabled', 'streaming': True,
                  'started_at_utc': at.isoformat(), 'real_api': True,
                  'historical_retrieval_seconds': old['timing']['retrieval_total_seconds'],
                  'evidence_preparation_seconds_measured': prep_seconds,
                  'status': 'evidence_frozen_api_pending'}
        save(path, result)  # Checkpoint before a paid request; no automatic retry.
        try:
            fragments = []
            first = None
            meta = None
            for kind, payload in client.stream(q['question'], prepared):
                if kind == 'text':
                    if first is None and payload.strip():
                        first = time.perf_counter()-start
                    fragments.append(payload)
                elif kind == 'done':
                    meta = payload
            answer = ''.join(fragments)
            result.update(status='success' if answer.strip() else 'empty_answer',
                          answer=answer, generation=meta,
                          citation_check=validate_grounded_answer(answer, prepared, entry['source_checks']),
                          measured_seconds={'evidence_preparation': prep_seconds,
                                            'prompt_to_first_text': first,
                                            'prompt_to_complete': time.perf_counter()-start},
                          cost=cost_estimate(meta['usage'] if meta else None, at))
        except (DeepSeekRequestError, MissingAPIKey) as exc:
            result.update(status='error', error_type=type(exc).__name__,
                          error_message=str(exc),
                          measured_seconds={'evidence_preparation': prep_seconds,
                                            'prompt_to_complete': time.perf_counter()-start})
        save(path, result)
        print(qid, result['status'],
              round(result['measured_seconds']['prompt_to_complete'], 3), 's', flush=True)


if __name__ == '__main__':
    main()
