"""Freeze exact Phase 3.6 retrieval evidence and validate it against source chunks."""
import hashlib
import json
from pathlib import Path

from src.citation_grounding import verify_evidence_source
from src.rag_service import prepare_grounded_evidence

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'experiments/results/phase36'
OUT = ROOT/'experiments/results/phase37'
QUESTIONS = ROOT/'experiments/configs/phase36_questions.json'
CHUNKS = ROOT/'data/processed/phase33_large_fixed_chunks.jsonl'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf8')).hexdigest()


def main():
    raw = QUESTIONS.read_bytes()
    question_hash = hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()
    lock = json.loads((ROOT/'experiments/configs/phase36_question_lock.json').read_text(encoding='utf8'))
    if question_hash != lock['sha256']:
        raise SystemExit('Frozen Phase 3.6 questions changed.')
    questions = json.loads(raw)['questions']
    if len(questions) != 20:
        raise SystemExit('Expected 20 frozen questions.')
    originals = {}
    target_chunks = set()
    for question in questions:
        qid = question['question_id']
        path = OLD/f'{qid}.json'
        raw_record = path.read_bytes()
        record = json.loads(raw_record)
        if record['status'] != 'success' or record['question'] != question['question'] or len(record['evidence']) != 3:
            raise SystemExit(f'Invalid frozen question/evidence: {qid}')
        originals[qid] = {'record': record, 'file_sha256': hashlib.sha256(raw_record).hexdigest()}
        target_chunks.update(item['chunk_id'] for item in record['evidence'])
    source_map = {}
    prefix = '{"chunk_id": "'
    with CHUNKS.open(encoding='utf8') as stream:
        for line in stream:
            if not line.startswith(prefix):
                raise SystemExit('Unexpected source chunk format')
            cid = line[len(prefix):].split('"', 1)[0]
            if cid in target_chunks:
                source_map[cid] = json.loads(line)
                if len(source_map) == len(target_chunks):
                    break
    if len(source_map) != len(target_chunks):
        raise SystemExit('Some frozen evidence chunks are absent from the source corpus.')
    entries = []
    for question in questions:
        qid = question['question_id']
        old = originals[qid]
        evidence = prepare_grounded_evidence(old['record']['evidence'])
        checks = verify_evidence_source(evidence, source_map)
        if any(row['status'] != 'valid' for row in checks):
            raise SystemExit(f'Frozen evidence differs from source metadata/text: {qid}')
        entries.append({'question_id': qid, 'phase36_record_sha256': old['file_sha256'],
                        'evidence_sha256': digest(old['record']['evidence']),
                        'grounded_evidence_sha256': digest(evidence),
                        'source_checks': checks})
    manifest = {'version': 'phase37-inputs-v1', 'question_set_sha256': question_hash,
                'source_chunk_file': 'data/processed/phase33_large_fixed_chunks.jsonl',
                'retrieval_run_id': 'phase33_large_fixed',
                'evidence_count_per_question': 3,
                'source_text_modified': False,
                'question_count': len(entries), 'entries': entries}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'inputs_manifest.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf8')) != manifest:
            raise SystemExit('Existing Phase 3.7 input manifest differs; refusing overwrite.')
        print('Validated existing Phase 3.7 manifest:', len(entries), 'questions')
    else:
        with path.open('x', encoding='utf8') as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
            file.write('\n')
        print('Frozen and source-validated:', len(entries), 'questions /', len(source_map), 'source chunks')


if __name__ == '__main__':
    main()
