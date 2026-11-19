"""Three paired live queries, with one retrieval/evidence set per question."""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for name, folder in {'HF_HOME':'huggingface','TORCH_HOME':'torch','PIP_CACHE_DIR':'pip'}.items():
    os.environ[name] = str(ROOT / '.cache' / folder)

from src.deepseek_client import DeepSeekClient, DeepSeekRequestError
from src.evidence_metadata import enrich
from src.evidence_verifier import verify
from src.phase36_run import cost_estimate
from src.rag_service import load_retriever, retrieve_evidence, prepare_grounded_evidence

OUT = ROOT / 'experiments/results/phase410'
PROMPT = (ROOT / 'prompts/scholar_rag_v2.txt').read_text(encoding='utf8').strip()
IDS = ('Q07','Q15','Q16')


def write(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    questions = {q['question_id']:q for q in json.loads((ROOT/'experiments/configs/phase36_questions.json').read_text(encoding='utf8'))['questions']}
    load_start = time.perf_counter()
    retriever, load_seconds = load_retriever('phase33_large_fixed')
    client = DeepSeekClient()
    assert (client.config['model'], client.config['temperature'],client.config['max_tokens']) == ('deepseek-flash',.3,512)
    for index,qid in enumerate(IDS):
        path = OUT/f'online_{qid}.json'
        row = json.loads(path.read_text(encoding='utf8')) if path.exists() else {
            'question_id':qid,'question':questions[qid]['question'],'index_load_seconds':load_seconds if index==0 else None,
            'cold_start_note':'Index/model loaded once before first query; query timing below excludes load.',
            'results':{}}
        if 'evidence' not in row:
            started=time.perf_counter()
            raw,docs,timing=retrieve_evidence(retriever,row['question'],3)
            row['evidence']=prepare_grounded_evidence(raw)
            row['retrieval_timing']=timing
            row['retrieval_wall_seconds']=time.perf_counter()-started
            row['evidence_doc_ids']=[e['doc_id'] for e in row['evidence']]
            write(path,row)
        order=('V1','V2') if index%2==0 else ('V2','V1')
        for version in order:
            if version in row['results']:
                continue
            evidence=enrich(row['evidence']) if version=='V2' else row['evidence']
            prep=time.perf_counter()
            prompt=PROMPT if version=='V2' else None
            prompt_seconds=time.perf_counter()-prep
            at=datetime.now(timezone.utc)
            wall=time.perf_counter()
            item={'version':version,'status':'pending','started_at_utc':at.isoformat(),
                  'prompt_preparation_seconds':prompt_seconds,'request_order':list(order).index(version)+1}
            row['results'][version]=item;write(path,row)
            try:
                parts=[];meta=None
                for kind,payload in client.stream(row['question'],evidence,system_prompt=prompt):
                    if kind=='text':parts.append(payload)
                    else:meta=payload
                answer=''.join(parts)
                check=verify(answer,evidence) if version=='V2' else None
                item.update(status='success',answer=answer,generation=meta,
                            verification=check,refinement_seconds=0,refinement_calls=0,
                            api_cost=cost_estimate(meta['usage'],at),
                            total_e2e_warm_seconds=row['retrieval_wall_seconds']+time.perf_counter()-wall)
            except DeepSeekRequestError as exc:
                item.update(status='error',error_type=type(exc).__name__,error_message=str(exc),
                            total_e2e_warm_seconds=None)
            write(path,row)
            print(qid,version,item['status'],flush=True)
            if item['status']!='success':
                return

if __name__=='__main__':main()
