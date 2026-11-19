"""One real C-group request per frozen question, with exact A/B input checks."""
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
A = ROOT/'experiments/results/phase36'
B = ROOT/'experiments/results/phase37'
OUT = ROOT/'experiments/results/phase38'
PROMPT = OUT/'candidate_prompt.txt'
CONFIG = ROOT/'experiments/configs/phase36_questions.json'


def write(path, obj):
    with path.open('w', encoding='utf8') as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)
        file.write('\n')


def load_inputs():
    questions = json.loads(CONFIG.read_text(encoding='utf8'))['questions']
    qhash = hashlib.sha256(CONFIG.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    locked = json.loads((ROOT/'experiments/configs/phase36_question_lock.json').read_text(encoding='utf8'))
    if qhash != locked['sha256'] or len(questions) != 20:
        raise SystemExit('Frozen question set changed or incomplete.')
    phase37 = json.loads((B/'inputs_manifest.json').read_text(encoding='utf8'))
    source_entries = {x['question_id']: x for x in phase37['entries']}
    prompt_bytes = PROMPT.read_bytes()
    manifest = {'version': 'phase38-inputs-v1', 'question_set_sha256': qhash,
                'candidate_prompt_sha256': hashlib.sha256(prompt_bytes).hexdigest(),
                'candidate_prompt_path': 'experiments/results/phase38/candidate_prompt.txt',
                'run_config': {'model': 'deepseek-flash', 'thinking': 'disabled',
                               'stream': True, 'temperature': .3, 'max_tokens': 512,
                               'evidence_count': 3}, 'entries': []}
    for q in questions:
        qid = q['question_id']
        a_bytes = (A/f'{qid}.json').read_bytes()
        b_bytes = (B/f'{qid}.json').read_bytes()
        a = json.loads(a_bytes)
        b = json.loads(b_bytes)
        evidence = prepare_grounded_evidence(a['evidence'])
        if a['question'] != b['question'] or a['question'] != q['question']:
            raise SystemExit(f'Question mismatch: {qid}')
        if digest(evidence) != digest(b['evidence']):
            raise SystemExit(f'Evidence differs between A and B: {qid}')
        if hashlib.sha256(a_bytes).hexdigest() != source_entries[qid]['phase36_record_sha256']:
            raise SystemExit(f'Historical A record changed: {qid}')
        if any(x['status'] != 'valid' for x in b['source_checks']):
            raise SystemExit(f'Historical evidence source failed: {qid}')
        manifest['entries'].append({'question_id': qid,
            'A_record_sha256': hashlib.sha256(a_bytes).hexdigest(),
            'B_record_sha256': hashlib.sha256(b_bytes).hexdigest(),
            'evidence_sha256': digest(evidence)})
    lock_path = OUT/'inputs_manifest.json'
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding='utf8')) != manifest:
            raise SystemExit('Existing Phase 3.8 input manifest differs; refusing overwrite.')
    else:
        with lock_path.open('x', encoding='utf8') as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
            file.write('\n')
    return questions, manifest, PROMPT.read_text(encoding='utf8').strip()


def main():
    questions, manifest, prompt = load_inputs()
    entries = {x['question_id']: x for x in manifest['entries']}
    if all((OUT/f"C_{q['question_id']}.json").exists() for q in questions):
        print('All 20 C records exist; no API request repeated.', flush=True)
        return
    client = DeepSeekClient()
    if (client.config['model'], client.config['temperature'], client.config['max_tokens']) != ('deepseek-flash', .3, 512):
        raise SystemExit('C generation config differs from frozen protocol.')
    for q in questions:
        qid = q['question_id']
        path = OUT/f'C_{qid}.json'
        if path.exists():
            print(qid, 'already recorded; skipped', flush=True)
            continue
        a = json.loads((A/f'{qid}.json').read_text(encoding='utf8'))
        b = json.loads((B/f'{qid}.json').read_text(encoding='utf8'))
        start = time.perf_counter()
        evidence = prepare_grounded_evidence(a['evidence'])
        at = datetime.now(timezone.utc)
        record = {'version': 'phase38-C-real-v1', 'question_id': qid, 'question': q['question'],
                  'category': q['category'], 'real_api': True,
                  'A_record_sha256': entries[qid]['A_record_sha256'],
                  'B_record_sha256': entries[qid]['B_record_sha256'],
                  'evidence_sha256': entries[qid]['evidence_sha256'],
                  'candidate_prompt_sha256': manifest['candidate_prompt_sha256'],
                  'evidence': evidence, 'source_checks': b['source_checks'],
                  'started_at_utc': at.isoformat(), 'status': 'input_frozen_api_pending'}
        write(path, record)
        try:
            parts = []
            first = None
            metadata = None
            for kind, payload in client.stream(q['question'], evidence, system_prompt=prompt):
                if kind == 'text':
                    if first is None and payload.strip():
                        first = time.perf_counter()-start
                    parts.append(payload)
                elif kind == 'done':
                    metadata = payload
            answer = ''.join(parts)
            record.update(status='success' if answer.strip() else 'empty_answer',
                          answer=answer, generation=metadata,
                          citation_check=validate_grounded_answer(answer, evidence, b['source_checks']),
                          timing={'input_prepare_to_first_text_seconds': first,
                                  'input_prepare_to_complete_seconds': time.perf_counter()-start},
                          cost=cost_estimate(metadata['usage'] if metadata else None, at))
        except (DeepSeekRequestError, MissingAPIKey) as exc:
            record.update(status='error', error_type=type(exc).__name__, error_message=str(exc),
                          timing={'input_prepare_to_complete_seconds': time.perf_counter()-start})
        write(path, record)
        print(qid, record['status'], round(record['timing']['input_prepare_to_complete_seconds'], 3), 's', flush=True)


if __name__ == '__main__':
    main()
