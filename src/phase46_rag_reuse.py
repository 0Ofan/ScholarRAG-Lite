"""Validate exact Phase 4.5 real-answer reuse when protected Top-3 evidence is unchanged."""
import csv
import hashlib
import json
from pathlib import Path

from src.phase37_prepare import digest
from src.phase45_prepare import evidence_from_bge_doc
from src.rag_service import prepare_grounded_evidence

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/phase46'
OLD=ROOT/'experiments/results/phase45'


def read_lines(path):
    return {x['topic_id']:x for x in map(json.loads,path.read_text(encoding='utf8').splitlines())}


def main():
    gate=json.loads((OUT/'rag_gate.json').read_text(encoding='utf8'))['conditions']
    if not gate['run_real_rag']:raise SystemExit('Offline gate not satisfied; no RAG validation allowed.')
    old_inputs=json.loads((OLD/'inputs.json').read_text(encoding='utf8'))
    if old_inputs['prompt']!='src/deepseek_client.py DeepSeekClient.stream default system prompt (Phase 3.7)':
        raise ValueError('Historical prompt provenance differs')
    bge=read_lines(ROOT/'experiments/results/phase33/phase33_large_fixed/retrieval_results.jsonl')
    protected=read_lines(OUT/'protected_results.jsonl')
    standard=read_lines(ROOT/'experiments/results/phase44/hybrid_results.jsonl')
    questions=json.loads((ROOT/'experiments/configs/phase36_questions.json').read_text(encoding='utf8'))['questions']
    old_scores={}
    with (OLD/'per_question_comparison.csv').open(encoding='utf-8-sig',newline='') as file:
        for row in csv.DictReader(file):old_scores[(row['question_id'],row['group'])]=row
    rows=[]
    for q,entry in zip(questions,old_inputs['entries']):
        qid=q['question_id'];topic=entry['retrieval_topic_id']
        if entry['question']!=q['question'] or topic!=str(q['source_topic_ids'][0]):
            raise ValueError(f'Question/topic mismatch: {qid}')
        dense=bge[topic]['top100'];by_id={x['doc_id']:x for x in dense}
        new_top3=protected[topic]['protected_top100'][:3]
        old_top3=standard[topic]['hybrid_top100'][:3]
        if [x['doc_id'] for x in new_top3]!=[x['doc_id'] for x in old_top3]:
            raise ValueError(f'Protected Top-3 changed: {qid}; new API protocol required')
        current_a=prepare_grounded_evidence([evidence_from_bge_doc(doc,i)
                                             for i,doc in enumerate(dense[:3],1)])
        current_b=prepare_grounded_evidence([evidence_from_bge_doc(by_id[x['doc_id']],i,
                     x['rank'],x['bm25_rank'],x['rrf_score']) for i,x in enumerate(new_top3,1)])
        # Phase 4.5 evidence includes RRF metadata. Compare exact prompt-facing fields,
        # and separately confirm the preserved metadata has not changed.
        for group,current in (('A',current_a),('B',current_b)):
            old=entry[group]['evidence']
            prompt_fields=('citation','evidence_id','doc_id','title','chunk_id','text')
            if [{k:x[k] for k in prompt_fields} for x in current]!=[
                {k:x[k] for k in prompt_fields} for x in old]:
                raise ValueError(f'Prompt-facing evidence differs: {qid}/{group}')
            record_path=OLD/f'{group}_{qid}.json'
            raw=record_path.read_bytes();record=json.loads(raw)
            if (record.get('status')!='success' or not record.get('real_api') or
                record['question']!=q['question'] or record['retrieval_query']!=entry['retrieval_query'] or
                record['evidence']!=old or record['evidence_sha256']!=digest(old) or
                (record['model'],record['thinking'],record['temperature'],record['max_tokens'],record['streaming'])!=
                ('deepseek-flash','disabled',.3,512,True)):
                raise ValueError(f'Historical real answer is not reusable: {qid}/{group}')
            score=old_scores[(qid,group)]
            rows.append({'question_id':qid,'topic_id':topic,'group':group,
                 'source_record':str(record_path.relative_to(ROOT)),
                 'source_record_sha256':hashlib.sha256(raw).hexdigest(),
                 'question':q['question'],'retrieval_query':entry['retrieval_query'],
                 'evidence_doc_ids':[x['doc_id'] for x in old],
                 'evidence_chunk_ids':[x['chunk_id'] for x in old],
                 'prompt_facing_evidence_sha256':digest([{k:x[k] for k in prompt_fields} for x in old]),
                 'answer':record['answer'],'real_api':True,'new_api_call':False,
                 'automated_evaluation':{name:int(score[name]) for name in
                                         ('Correctness','Completeness','Relevance','Faithfulness')},
                 'citation_support':score['citation_support'],
                 'source_attribution':score['source_attribution'],
                 'api_ttft_seconds_historical':record['generation']['api_ttft_seconds'],
                 'api_total_seconds_historical':record['generation']['api_total_seconds'],
                 'usage_historical':record['generation']['usage'],
                 'cost_historical':record['cost']})
    result={'version':'phase46-rag-reuse-v1','label':'Automated Evaluation',
            'question_count':20,'comparison_rows':40,'reused_real_api_answers':40,
            'new_api_calls':0,'protected_top3_same_as_standard_topics':20,
            'prompt_provenance':'Phase 4.5 used DeepSeekClient.stream default Phase 3.7 prompt with no override; no prompt module changed in Phase 4.6',
            'historical_evaluator':'experiments/results/phase45/eval_Q01.json through eval_Q20.json',
            'historical_aggregate':'experiments/results/phase45/aggregate.json',
            'inference':'Protected Top-10 membership changed, but the evidence selected by the current Top-3 RAG workflow did not change for any of the 20 questions.',
            'rows':rows}
    (OUT/'rag_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('RAG comparison reused',len(rows),'exact real API records; new calls 0',flush=True)


if __name__=='__main__':main()
