"""One paired automated evaluation per frozen question; same historical rubric."""
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.citation_grounding import _sentences
from src.phase36_run import cost_estimate
from src.phase38_evaluate import validate_evaluation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/phase49'
P45 = ROOT / 'experiments/results/phase45'
PROMPT = (ROOT / 'experiments/results/phase38/evaluator_prompt.txt').read_text(encoding='utf8').replace(
    '三条原始证据、三个匿名回答', '每个匿名回答各自对应的三条原始证据、两个匿名回答')


def write(path, row):
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def main():
    questions = json.loads((ROOT / 'experiments/configs/phase36_questions.json').read_text(encoding='utf8'))['questions']
    load_dotenv(ROOT / '.env', override=False)
    key = os.getenv('DEEPSEEK_API_KEY', '').strip()
    if not key:
        raise RuntimeError('缺少 DeepSeek API Key')
    client = OpenAI(api_key=key, base_url='https://api.deepseek.com', timeout=60, max_retries=0)
    for q in questions:
        qid = q['question_id']
        path = OUT / f'eval_{qid}.json'
        previous = None
        if path.exists():
            previous = json.loads(path.read_text(encoding='utf8'))
            if not (previous.get('status') == 'error' and previous.get('error_type') == 'JSONDecodeError'
                    and previous.get('retry_count', 0) == 0):
                print(qid, 'evaluation checkpoint exists', flush=True)
                continue
        old = json.loads((P45 / f'A_{qid}.json').read_text(encoding='utf8'))
        new = json.loads((OUT / f'{qid}.json').read_text(encoding='utf8'))
        if old['status'] != 'success' or new['status'] != 'success':
            continue
        answers = {}
        for label, record in (('R1', old), ('R2', new)):
            answers[label] = {'answer': record['answer'],
                              'sentences': [{'sentence_index': i, 'text': s} for i, s in enumerate(_sentences(record['answer']), 1)],
                              'evidence_records': [{k: e[k] for k in ('evidence_id', 'doc_id', 'title', 'chunk_id', 'text')}
                                                   for e in record['evidence']]}
        payload = {'question_id': qid, 'question': q['question'],
                   'expected_answer_points': q['expected_answer_points'],
                   'evaluation_notes': q['evaluation_notes'],
                   'evidence_availability': q['evidence_availability'], 'answers': answers}
        at = datetime.now(timezone.utc)
        row = {'version': 'phase49-automated-v1', 'label': 'Automated Evaluation', 'question_id': qid,
               'prompt_sha256': hashlib.sha256(PROMPT.encode('utf8')).hexdigest(),
               'source_rubric': 'experiments/results/phase38/evaluator_prompt.txt',
               'input': payload, 'status': 'pending', 'started_at_utc': at.isoformat()}
        if previous:
            row['retry_count'] = 1
            row['previous_error_type'] = previous['error_type']
        write(path, row)
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(model='deepseek-flash', temperature=0, max_tokens=4096,
                response_format={'type':'json_object'}, extra_body={'thinking':{'type':'disabled'}},
                messages=[{'role':'system','content':PROMPT},
                          {'role':'user','content':json.dumps(payload,ensure_ascii=False)}])
            raw = response.choices[0].message.content or ''
            row['raw_response'] = raw
            parsed = json.loads(raw)
            validate_evaluation(parsed, {'R1':'V1','R2':'V2'}, {'E1','E2','E3'},
                                {label:len(item['sentences']) for label,item in answers.items()})
            u = response.usage
            usage = {'input_tokens':u.prompt_tokens,'output_tokens':u.completion_tokens,
                     'total_tokens':u.total_tokens,
                     'prompt_cache_hit_tokens':getattr(u,'prompt_cache_hit_tokens',None),
                     'prompt_cache_miss_tokens':getattr(u,'prompt_cache_miss_tokens',None)} if u else None
            row.update(status='success', raw_response=raw, usage=usage, cost=cost_estimate(usage,at),
                       evaluations_by_version={'V1':parsed['responses']['R1'],'V2':parsed['responses']['R2']})
        except Exception as exc:
            row.update(status='error', error_type=type(exc).__name__,
                       error_message='评价请求或结构化响应失败；未重试。')
        row['elapsed_seconds'] = time.perf_counter() - started
        write(path,row)
        print(qid,row['status'],round(row['elapsed_seconds'],2),flush=True)

if __name__ == '__main__':
    main()
