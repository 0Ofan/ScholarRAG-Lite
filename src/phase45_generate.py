"""One real paired DeepSeek request per frozen Phase 4.5 evidence set."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.citation_grounding import validate_grounded_answer
from src.deepseek_client import DeepSeekClient, DeepSeekRequestError, MissingAPIKey
from src.phase36_run import cost_estimate
from src.phase37_prepare import digest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/phase45'


def write(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def one(client, entry, group):
    qid=entry['question_id'];path=OUT/f'{group}_{qid}.json'
    previous=None
    if path.exists():
        previous=json.loads(path.read_text(encoding='utf8'))
        if not (previous.get('status')=='error' and
                'APIConnectionError' in previous.get('error_message','') and
                previous.get('sandbox_retry_count',0)==0):
            print(group,qid,'checkpoint exists; skipped',flush=True)
            return
    evidence=entry[group]['evidence']
    if len(evidence)!=3 or digest(evidence)!=entry[group]['evidence_sha256']:
        raise ValueError(f'{qid} {group}: frozen evidence mismatch')
    started=time.perf_counter();at=datetime.now(timezone.utc)
    record={'version':'phase45-real-v1','group':group,'question_id':qid,
            'question':entry['question'],'retrieval_topic_id':entry['retrieval_topic_id'],
            'retrieval_query':entry['retrieval_query'],'evidence':evidence,
            'evidence_sha256':entry[group]['evidence_sha256'],
            'model':'deepseek-flash','thinking':'disabled','temperature':.3,
            'max_tokens':512,'streaming':True,'started_at_utc':at.isoformat(),
            'real_api':True,'status':'api_pending'}
    if previous:
        record['sandbox_retry_count']=1
        record['previous_failed_attempt']={key:previous.get(key) for key in
            ('started_at_utc','status','error_type','error_message','generation_wall_seconds')}
    write(path,record)
    try:
        parts=[];meta=None
        for kind,payload in client.stream(entry['question'],evidence):
            if kind=='text':parts.append(payload)
            elif kind=='done':meta=payload
        answer=''.join(parts)
        record.update(status='success' if answer.strip() else 'empty_answer',answer=answer,
                      generation=meta,citation_check=validate_grounded_answer(answer,evidence),
                      generation_wall_seconds=time.perf_counter()-started,
                      cost=cost_estimate(meta['usage'] if meta else None,at))
    except (DeepSeekRequestError,MissingAPIKey) as exc:
        record.update(status='error',error_type=type(exc).__name__,
                      error_message=str(exc),generation_wall_seconds=time.perf_counter()-started)
    write(path,record)
    print(group,qid,record['status'],round(record['generation_wall_seconds'],3),'s',flush=True)


def main():
    entries=json.loads((OUT/'inputs.json').read_text(encoding='utf8'))['entries']
    pending=[]
    for e in entries:
        for group in 'AB':
            path=OUT/f"{group}_{e['question_id']}.json"
            if not path.exists():
                pending.append((group,e))
            else:
                old=json.loads(path.read_text(encoding='utf8'))
                if (old.get('status')=='error' and 'APIConnectionError' in old.get('error_message','')
                        and old.get('sandbox_retry_count',0)==0):
                    pending.append((group,e))
    if not pending:
        print('All paired generation records exist; no calls repeated.',flush=True)
        return
    client=DeepSeekClient()
    if (client.config['model'],client.config['temperature'],client.config['max_tokens'])!=('deepseek-flash',.3,512):
        raise SystemExit('Generation settings differ from frozen protocol.')
    for group,entry in pending:
        one(client,entry,group)


if __name__=='__main__':main()
