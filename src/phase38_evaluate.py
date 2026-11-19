"""Blinded per-question automated semantic evaluation; checkpoints prevent repeat calls."""
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.citation_grounding import _sentences
from src.phase36_run import cost_estimate
from src.phase38_rules import check_answer, normalize_for_blind_evaluation
from src.phase38_run import A, B, CONFIG, OUT, load_inputs, write

ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = OUT/'evaluator_prompt.txt'
SUPPORT = {'Direct Support', 'Reasonable Inference', 'Partial Support', 'Unsupported'}
CLASSES = {'citation_support': SUPPORT,
           'source_attribution': {'Correct', 'Incorrect', 'Unverified'},
           'overgeneralization': {'Yes', 'No', 'Unverified'},
           'abstention': {'Appropriate', 'Over-abstention', 'Failure to Abstain'}}
METRICS = ('Correctness', 'Completeness', 'Relevance', 'Faithfulness')


def load_triplet(question):
    qid = question['question_id']
    records = {'A': json.loads((A/f'{qid}.json').read_text(encoding='utf8')),
               'B': json.loads((B/f'{qid}.json').read_text(encoding='utf8')),
               'C': json.loads((OUT/f'C_{qid}.json').read_text(encoding='utf8'))}
    if any(r.get('status') != 'success' or not r.get('real_api') for r in records.values()):
        raise ValueError(f'{qid}: all A/B/C real answers required')
    evidence = records['B']['evidence']
    if evidence != records['C']['evidence']:
        raise ValueError(f'{qid}: B/C evidence differs')
    return records, evidence


def blinded_payload(question, records, evidence):
    qid = question['question_id']
    versions = list('ABC')
    random.Random(qid).shuffle(versions)
    mapping = {f'R{i}': version for i, version in enumerate(versions, 1)}
    answers = {}
    for label, version in mapping.items():
        answer = normalize_for_blind_evaluation(records[version]['answer'], version)
        answers[label] = {'answer': answer, 'sentences': [
            {'sentence_index': i, 'text': sentence} for i, sentence in enumerate(_sentences(answer), 1)]}
    payload = {'question_id': qid, 'question': question['question'],
               'expected_answer_points': question['expected_answer_points'],
               'evaluation_notes': question['evaluation_notes'],
               'evidence_availability': question['evidence_availability'],
               'evidence_records': [{k: e[k] for k in ('evidence_id', 'doc_id', 'title', 'chunk_id', 'text')}
                                    for e in evidence], 'answers': answers}
    return mapping, payload


def validate_evaluation(value, labels, evidence_ids, sentence_counts):
    if set(value.get('responses', {})) != set(labels):
        raise ValueError('Evaluator response labels mismatch')
    for label, item in value['responses'].items():
        if set(item.get('scores', {})) != set(METRICS):
            raise ValueError(f'{label}: score fields mismatch')
        for name in METRICS:
            field = item['scores'][name]
            if type(field.get('score')) is not int or not 1 <= field['score'] <= 5:
                raise ValueError(f'{label}: invalid {name} score')
            _reason_refs(field, evidence_ids)
        for name, allowed in CLASSES.items():
            field = item.get(name, {})
            if field.get('class') not in allowed:
                raise ValueError(f'{label}: invalid {name}')
            _reason_refs(field, evidence_ids)
        claims = item.get('claims', [])
        indices = [x.get('sentence_index') for x in claims]
        if indices != list(range(1, sentence_counts[label] + 1)):
            raise ValueError(f'{label}: claim sentence coverage mismatch')
        for claim in claims:
            if claim.get('support') not in SUPPORT:
                raise ValueError(f'{label}: invalid claim support')
            _reason_refs(claim, evidence_ids)
    return value


def _reason_refs(field, evidence_ids):
    if not isinstance(field.get('reason'), str) or not field['reason'].strip():
        raise ValueError('Missing evaluation reason')
    refs = field.get('evidence_ids')
    if not isinstance(refs, list) or any(ref not in evidence_ids for ref in refs):
        raise ValueError('Invalid evaluation evidence IDs')


def save_rule_checks(questions):
    path = OUT/'deterministic_checks.json'
    if path.exists():
        return
    results = []
    for question in questions:
        records, evidence = load_triplet(question)
        for version in 'ABC':
            answer = normalize_for_blind_evaluation(records[version]['answer'], version)
            results.append({'question_id': question['question_id'], 'version': version,
                            'check': check_answer(answer, evidence, records['B']['source_checks'], question['question'])})
    write(path, {'version': 'phase38-rules-v1', 'label': 'Automated Evaluation', 'results': results})


def main():
    questions, _manifest, _prompt = load_inputs()
    save_rule_checks(questions)
    load_dotenv(ROOT/'.env', override=False)
    import os
    key = os.getenv('DEEPSEEK_API_KEY', '').strip()
    if not key:
        raise SystemExit('Evaluator API key missing; no semantic scores generated.')
    prompt_bytes = PROMPT_PATH.read_bytes()
    prompt_hash = hashlib.sha256(prompt_bytes).hexdigest()
    prompt = prompt_bytes.decode('utf8')
    client = OpenAI(api_key=key, base_url='https://api.deepseek.com', timeout=60, max_retries=0)
    for question in questions:
        qid = question['question_id']
        path = OUT/f'eval_{qid}.json'
        if path.exists():
            print(qid, 'evaluation checkpoint exists; skipped', flush=True)
            continue
        records, evidence = load_triplet(question)
        mapping, payload = blinded_payload(question, records, evidence)
        at = datetime.now(timezone.utc)
        record = {'version': 'phase38-automated-semantic-v1', 'label': 'Automated Evaluation',
                  'question_id': qid, 'model': 'deepseek-flash', 'thinking': 'disabled',
                  'response_format': 'json_object', 'temperature': 0,
                  'evaluator_prompt_sha256': prompt_hash, 'started_at_utc': at.isoformat(),
                  'blinded_response_mapping': mapping, 'input': payload,
                  'status': 'api_pending'}
        write(path, record)
        started = time.perf_counter()
        try:
            result = client.chat.completions.create(
                model='deepseek-flash', temperature=0, max_tokens=4096,
                response_format={'type': 'json_object'},
                extra_body={'thinking': {'type': 'disabled'}},
                messages=[{'role': 'system', 'content': prompt},
                          {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}])
            text = result.choices[0].message.content or ''
            usage_obj = result.usage
            usage = {'input_tokens': usage_obj.prompt_tokens, 'output_tokens': usage_obj.completion_tokens,
                     'total_tokens': usage_obj.total_tokens,
                     'prompt_cache_hit_tokens': getattr(usage_obj, 'prompt_cache_hit_tokens', None),
                     'prompt_cache_miss_tokens': getattr(usage_obj, 'prompt_cache_miss_tokens', None)} if usage_obj else None
            record.update(raw_response=text, finish_reason=result.choices[0].finish_reason,
                          response_model=result.model, usage=usage, elapsed_seconds=time.perf_counter()-started,
                          cost=cost_estimate(usage, at))
            parsed = json.loads(text)
            validate_evaluation(parsed, mapping, {e['evidence_id'] for e in evidence},
                                {label: len(x['sentences']) for label, x in payload['answers'].items()})
            record.update(status='success', evaluations_by_version={
                mapping[label]: parsed['responses'][label] for label in mapping})
        except Exception as exc:
            record.update(status='error', error_type=type(exc).__name__,
                          error_message='Evaluator request or structured response failed; see recorded status.',
                          elapsed_seconds=time.perf_counter()-started)
        write(path, record)
        print(qid, record['status'], round(record['elapsed_seconds'], 2), 's', flush=True)


if __name__ == '__main__':
    main()
