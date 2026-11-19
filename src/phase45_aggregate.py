"""Summarize paired real generation and automated ratings without new API requests."""
import csv
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from src.phase38_aggregate import describe
from src.phase38_rules import check_answer
from src.phase45_prepare import BGE, HYBRID, evidence_from_bge_doc, lines
from src.rag_service import prepare_grounded_evidence

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/phase45'
QUESTIONS=ROOT/'experiments/configs/phase36_questions.json'
METRICS=('Correctness','Completeness','Relevance','Faithfulness')


def write(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def main():
    questions=json.loads(QUESTIONS.read_text(encoding='utf8'))['questions']
    inputs={x['question_id']:x for x in json.loads((OUT/'inputs.json').read_text(encoding='utf8'))['entries']}
    rows=[];details=[]
    for q in questions:
        qid=q['question_id'];entry=inputs[qid]
        result={'question_id':qid,'question':q['question'],
                'retrieval_topic_id':entry['retrieval_topic_id'],
                'expected_answer_points':q['expected_answer_points'],'groups':{}}
        evaluation_path=OUT/f'eval_{qid}.json'
        evaluation=json.loads(evaluation_path.read_text(encoding='utf8')) if evaluation_path.exists() else {}
        for group in 'AB':
            record=json.loads((OUT/f'{group}_{qid}.json').read_text(encoding='utf8'))
            judged=evaluation.get('evaluations_by_version',{}).get(group)
            check=check_answer(record.get('answer',''),record['evidence'],None,q['question']) if record['status']=='success' else None
            generation=record.get('generation') or {};usage=generation.get('usage') or {}
            scores={m:judged['scores'][m]['score'] if judged else None for m in METRICS}
            row={'question_id':qid,'topic_id':entry['retrieval_topic_id'],
                 'category':q['category'],'group':group,'status':record['status'],
                 'evaluation_status':evaluation.get('status','unavailable'),**scores,
                 'citation_support':judged['citation_support']['class'] if judged else None,
                 'source_attribution':judged['source_attribution']['class'] if judged else None,
                 'overgeneralization':judged['overgeneralization']['class'] if judged else None,
                 'abstention':judged['abstention']['class'] if judged else None,
                 'invalid_citations':len(check['citation']['invalid']) if check else None,
                 'api_ttft_seconds':generation.get('api_ttft_seconds'),
                 'api_total_seconds':generation.get('api_total_seconds'),
                 'generation_wall_seconds':record.get('generation_wall_seconds'),
                 'input_tokens':usage.get('input_tokens'),'output_tokens':usage.get('output_tokens'),
                 'cost_estimate_cny':record.get('cost',{}).get('estimated_cny'),
                 'evidence_doc_ids':'|'.join(e['doc_id'] for e in record['evidence']),
                 'evidence_chunk_ids':'|'.join(e['chunk_id'] for e in record['evidence']),
                 'answer':record.get('answer'),
                 'citation_support_reason':judged['citation_support']['reason'] if judged else None,
                 'source_attribution_reason':judged['source_attribution']['reason'] if judged else None}
            rows.append(row)
            result['groups'][group]={'record_file':f'{group}_{qid}.json',
                'evaluation':judged,'deterministic_check':check,
                'evidence_doc_ids':[e['doc_id'] for e in record['evidence']],
                'evidence_chunk_ids':[e['chunk_id'] for e in record['evidence']],
                'answer':record.get('answer')}
        details.append(result)
    with (OUT/'per_question_comparison.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(OUT/'comparison.json',{'version':'phase45-comparison-v1','label':'Automated Evaluation',
                                 'questions':details})
    group_summary={}
    for group in 'AB':
        selected=[r for r in rows if r['group']==group]
        group_summary[group]={'successes':sum(r['status']=='success' for r in selected),
            'evaluated':sum(r['evaluation_status']=='success' for r in selected),
            'quality':{m:describe([r[m] for r in selected if r[m] is not None]) for m in METRICS},
            'citation_support':dict(Counter(r['citation_support'] for r in selected if r['citation_support'])),
            'source_attribution':dict(Counter(r['source_attribution'] for r in selected if r['source_attribution'])),
            'overgeneralization':dict(Counter(r['overgeneralization'] for r in selected if r['overgeneralization'])),
            'abstention':dict(Counter(r['abstention'] for r in selected if r['abstention'])),
            'performance':{m:describe([r[m] for r in selected if r[m] is not None]) for m in
                ('api_ttft_seconds','api_total_seconds','generation_wall_seconds','input_tokens',
                 'output_tokens','cost_estimate_cny')},
            'invalid_citations':sum(r['invalid_citations'] or 0 for r in selected)}
    paired={m:{'B_higher':sum(a[m] is not None and b[m] is not None and b[m]>a[m]
                             for a,b in zip(rows[::2],rows[1::2])),
               'B_lower':sum(a[m] is not None and b[m] is not None and b[m]<a[m]
                            for a,b in zip(rows[::2],rows[1::2])),
               'tie':sum(a[m] is not None and b[m] is not None and b[m]==a[m]
                         for a,b in zip(rows[::2],rows[1::2]))} for m in METRICS}
    changed=sum(a['evidence_chunk_ids']!=b['evidence_chunk_ids'] for a,b in zip(rows[::2],rows[1::2]))
    evaluator=[json.loads((OUT/f"eval_{q['question_id']}.json").read_text(encoding='utf8'))
               for q in questions if (OUT/f"eval_{q['question_id']}.json").exists()]
    write(OUT/'aggregate.json',{'version':'phase45-aggregate-v1','label':'Automated Evaluation',
        'question_count':len(questions),'evidence_changed_questions':changed,
        'groups':group_summary,'paired_score_changes':paired,
        'evaluator':{'attempts':len(evaluator),
                     'successes':sum(x.get('status')=='success' for x in evaluator),
                     'same_service_as_generator':True,
                     'cost_estimate_cny':sum(x.get('cost',{}).get('estimated_cny') or 0 for x in evaluator)},
        'limitations':['Same-service automated evaluation, not independent human review',
                       'Source topic is first frozen source_topic_ids entry',
                       'Offline cached retrieval cannot establish live end-to-end latency',
                       'Unjudged qrels remain unknown']})
    bge=lines(BGE);hybrid=lines(HYBRID);selection=[]
    for entry in inputs.values():
        topic=entry['retrieval_topic_id'];a=bge[topic]['top100']
        by_id={x['doc_id']:x for x in a}
        for group in 'AB':
            start=time.perf_counter()
            if group=='A':
                prepared=prepare_grounded_evidence([evidence_from_bge_doc(doc,i)
                    for i,doc in enumerate(a[:3],1)])
            else:
                prepared=prepare_grounded_evidence([evidence_from_bge_doc(by_id[x['doc_id']],i,
                    x['rank'],x['bm25_rank'],x['rrf_score'])
                    for i,x in enumerate(hybrid[topic]['hybrid_top100'][:3],1)])
            elapsed=time.perf_counter()-start
            if prepared!=entry[group]['evidence']:raise ValueError('Selection replay differs')
            selection.append({'question_id':entry['question_id'],'group':group,'seconds':elapsed})
    bge_perf=json.loads((ROOT/'experiments/results/phase33/phase33_large_fixed/performance_metrics.json').read_text(encoding='utf8'))
    bm25_perf=json.loads((ROOT/'experiments/results/phase43/performance.json').read_text(encoding='utf8'))
    rrf_perf=json.loads((ROOT/'experiments/results/phase44/performance.json').read_text(encoding='utf8'))
    write(OUT/'performance.json',{'version':'phase45-performance-v1',
        'selection_replay':{'per_question_group':selection,
            'A':describe([x['seconds'] for x in selection if x['group']=='A']),
            'B':describe([x['seconds'] for x in selection if x['group']=='B']),
            'note':'Replayed from in-memory frozen rankings and chunks; excludes reading files, model/index load and API context serialization.'},
        'historical_retrieval_reference':{
            'bge_query_encoding_50_query_batch_warm_repeat_seconds':
                [x['seconds'] for x in bge_perf['query_encoding_repeats'][1:]],
            'bge_top100_retrieval_median_seconds':bge_perf['retrieval_median_seconds_across_repeats'],
            'bm25_query_mean_seconds':bm25_perf['bm25_query_seconds']['mean'],
            'rrf_fusion_mean_seconds':rrf_perf['rrf_query_seconds']['mean']},
        'current_real_api':{group:group_summary[group]['performance'] for group in 'AB'},
        'live_end_to_end_latency_seconds':None,
        'note':'Saved Phase 3.3/4.3/4.4 rankings were reused. No live BGE/BM25 searches were run, so historical stage timings and new API timings cannot be combined into a measured end-to-end latency.'})
    print('Aggregated',len(rows),'A/B rows; evaluated',
          sum(x.get('status')=='success' for x in evaluator),'questions',flush=True)


if __name__=='__main__':main()
