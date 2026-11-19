"""Paired automated evaluation using the Phase 3.8 rubric and each answer's own evidence."""
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
from src.phase38_evaluate import validate_evaluation

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/phase45'
RUBRIC=ROOT/'experiments/results/phase38/evaluator_prompt.txt'
QUESTIONS=ROOT/'experiments/configs/phase36_questions.json'


def write(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def recover_saved_format():
    """Repair one trailing brace only, retaining the original model response."""
    for path in sorted(OUT.glob('eval_Q*.json')):
        row=json.loads(path.read_text(encoding='utf8'))
        if row.get('status')!='error' or row.get('error_type')!='JSONDecodeError':
            continue
        raw=row.get('raw_response','')
        options=[('removed_one_trailing_brace',raw[:-1]) if raw.endswith('}') else None,
                 ('added_one_trailing_brace',raw+'}')]
        for option in options:
            if option is None:continue
            repair,text=option
            try:
                parsed=json.loads(text)
                payload=row['input'];mapping=row['blinded_response_mapping']
                for item in parsed.get('responses',{}).values():
                    scores=item.get('scores',{})
                    misplaced={'claims','citation_support','source_attribution','overgeneralization','abstention'}
                    if misplaced<=scores.keys() and not misplaced & item.keys():
                        for key in misplaced:item[key]=scores.pop(key)
                        repair += '_and_misnested_fields'
                validate_evaluation(parsed,mapping,{'E1','E2','E3'},
                    {label:len(item['sentences']) for label,item in payload['answers'].items()})
            except (ValueError,KeyError,TypeError):
                continue
            row.update(status='success',json_format_repair=repair,
                       evaluations_by_version={version:parsed['responses'][label]
                                               for label,version in mapping.items()})
            row.pop('error_type',None);row.pop('error_message',None)
            write(path,row)
            print(path.name,'recovered by',repair,flush=True)
            break


def payload_for(q,records):
    qid=q['question_id'];versions=list('AB');random.Random(qid).shuffle(versions)
    mapping={f'R{i}':v for i,v in enumerate(versions,1)}
    answers={}
    for label,version in mapping.items():
        record=records[version]
        answer=record['answer']
        answers[label]={'answer':answer,
            'sentences':[{'sentence_index':i,'text':text}
                         for i,text in enumerate(_sentences(answer),1)],
            'evidence_records':[{key:e[key] for key in ('evidence_id','doc_id','title','chunk_id','text')}
                                for e in record['evidence']]}
    return mapping,{'question_id':qid,'question':q['question'],
                    'expected_answer_points':q['expected_answer_points'],
                    'evaluation_notes':q['evaluation_notes'],
                    'evidence_availability':q['evidence_availability'],
                    'answers':answers}


def main():
    recover_saved_format()
    questions=json.loads(QUESTIONS.read_text(encoding='utf8'))['questions']
    rubric=RUBRIC.read_text(encoding='utf8')
    # Only the input layout differs: paired answers use different frozen evidence.
    prompt=rubric.replace('三条原始证据、三个匿名回答','每个匿名回答各自对应的三条原始证据、两个匿名回答')
    if prompt==rubric:
        raise ValueError('Phase 3.8 evaluator rubric layout changed')
    prompt_hash=hashlib.sha256(prompt.encode('utf8')).hexdigest()
    pending=[q for q in questions if not (OUT/f"eval_{q['question_id']}.json").exists()]
    if not pending:
        print('All evaluation checkpoints exist; no calls repeated.',flush=True);return
    load_dotenv(ROOT/'.env',override=False)
    import os
    key=os.getenv('DEEPSEEK_API_KEY','').strip()
    if not key:raise SystemExit('Evaluator API key missing.')
    client=OpenAI(api_key=key,base_url='https://api.deepseek.com',timeout=60,max_retries=0)
    for q in pending:
        qid=q['question_id'];path=OUT/f'eval_{qid}.json'
        records={v:json.loads((OUT/f'{v}_{qid}.json').read_text(encoding='utf8')) for v in 'AB'}
        if any(r.get('status')!='success' or not r.get('real_api') for r in records.values()):
            print(qid,'generation incomplete; evaluation skipped',flush=True);continue
        mapping,payload=payload_for(q,records)
        at=datetime.now(timezone.utc)
        row={'version':'phase45-automated-v1','label':'Automated Evaluation',
             'question_id':qid,'model':'deepseek-flash','thinking':'disabled',
             'temperature':0,'evaluator_prompt_sha256':prompt_hash,
             'source_rubric':'experiments/results/phase38/evaluator_prompt.txt',
             'blinded_response_mapping':mapping,'input':payload,
             'started_at_utc':at.isoformat(),'status':'api_pending'}
        write(path,row)
        start=time.perf_counter()
        try:
            result=client.chat.completions.create(model='deepseek-flash',temperature=0,max_tokens=4096,
                response_format={'type':'json_object'},extra_body={'thinking':{'type':'disabled'}},
                messages=[{'role':'system','content':prompt},
                          {'role':'user','content':json.dumps(payload,ensure_ascii=False)}])
            body=result.choices[0].message.content or ''
            usage=result.usage
            tokens={'input_tokens':usage.prompt_tokens,'output_tokens':usage.completion_tokens,
                    'total_tokens':usage.total_tokens,
                    'prompt_cache_hit_tokens':getattr(usage,'prompt_cache_hit_tokens',None),
                    'prompt_cache_miss_tokens':getattr(usage,'prompt_cache_miss_tokens',None)} if usage else None
            row.update(raw_response=body,usage=tokens,response_model=result.model,
                       finish_reason=result.choices[0].finish_reason,
                       elapsed_seconds=time.perf_counter()-start,cost=cost_estimate(tokens,at))
            parsed=json.loads(body)
            validate_evaluation(parsed,mapping,{'E1','E2','E3'},
                {label:len(item['sentences']) for label,item in payload['answers'].items()})
            row.update(status='success',evaluations_by_version={
                version:parsed['responses'][label] for label,version in mapping.items()})
        except Exception as exc:
            row.update(status='error',error_type=type(exc).__name__,
                       error_message='Evaluator request or structured response failed; see recorded status.',
                       elapsed_seconds=time.perf_counter()-start)
        write(path,row)
        print(qid,row['status'],round(row['elapsed_seconds'],2),'s',flush=True)


if __name__=='__main__':main()
