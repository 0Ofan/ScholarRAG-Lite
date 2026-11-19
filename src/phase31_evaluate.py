"""Evaluate all 50 frozen queries on the Pilot document-level baseline."""
import argparse
import csv
import json
import statistics
import time
from collections import Counter
from pathlib import Path
import numpy as np
from retriever import PilotRetriever
from evaluation import score_topic

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--variant',choices=('phase31','phase32','phase34'),default='phase31')
parser.add_argument('--run-id',choices=('phase33_medium_fixed','phase33_medium_sentence',
                                        'phase33_large_fixed','phase33_large_sentence'))
args=parser.parse_args();variant=args.variant
PROC=ROOT/'data/processed'
if args.run_id:
    run_id=args.run_id;scale=run_id.split('_')[1];method=run_id.split('_')[2]
    RESULT=ROOT/'experiments/results/phase33'/run_id
    manifest_path=PROC/f'phase33_{scale}_manifest.json'
else:
    RESULT=ROOT/'experiments/results'/('phase31_pilot' if variant=='phase31' else
                                       'phase32_improved' if variant=='phase32' else 'phase34_ablation')
    manifest_path=PROC/'phase31_pilot_manifest.json'
RESULT.mkdir(parents=True,exist_ok=True)
manifest=json.loads(manifest_path.read_text(encoding='utf8'))
topics={x['topic_id']:x for x in map(json.loads,(ROOT/'data/evaluation/topics.jsonl').read_text(encoding='utf8').splitlines())}
assert len(topics)==50 and manifest['query_ids']==[str(x) for x in range(1,51)]
queries=[topics[str(i)]['query'] for i in range(1,51)]
retriever=PilotRetriever(variant=variant,run_id=args.run_id);corpus_ids=[d['doc_id'] for d in manifest['documents']]
query_runs=[];query_vectors=None
for rep in (1,2,3):
    vec,sec=retriever.encode_queries(queries)
    diff=0.0 if query_vectors is None else float(np.max(np.abs(query_vectors-vec)))
    if query_vectors is None:query_vectors=vec
    query_runs.append({'repeat':rep,'seconds':sec,'max_absolute_difference_from_first':diff})
    print(f'query encoding repeat {rep}/3: {sec:.3f}s',flush=True)
per={};latency_runs=[];repeat_ids={}
with (RESULT/'retrieval_results.jsonl').open('w',encoding='utf8',newline='\n') as out:
    for rep in (1,2,3):
        latencies=[]
        for i,q in enumerate(map(str,range(1,51))):
            docs,perf=retriever.retrieve_vector(query_vectors[i],100)
            latencies.append(perf['retrieval_seconds'])
            ids=[x['doc_id'] for x in docs]
            assert len(ids)==len(set(ids))==100
            if rep>1:
                assert ids==repeat_ids[q]
                continue
            repeat_ids[q]=ids
            judged=manifest['effective_qrels'][q]
            metrics=score_topic(ids,judged,corpus_ids)
            assert metrics is not None and metrics['known_relevant_in_corpus']==manifest['positive_per_query'][q]
            top={}
            for n in (10,20,100):
                grades=[judged.get(uid) for uid in ids[:n]]
                top[str(n)]={'positive':sum(g is not None and g>0 for g in grades),
                             'grade_0':sum(g==0 for g in grades),
                             'unjudged':sum(g is None for g in grades),
                             'unjudged_rate':sum(g is None for g in grades)/n}
            per[q]={'query':queries[i],'available_positive':manifest['positive_per_query'][q],
                    'official_positive':sum(1 for line in (ROOT/'data/evaluation/trec_covid_complete_qrels.txt').read_text().splitlines()
                                             if line.split()[0]==q and int(line.split()[3])>0),
                    'metrics':metrics,'top':top,'retrieval_seconds':perf['retrieval_seconds'],
                    'searched_chunks':perf['searched_chunks']}
            for d in docs:d['official_grade']=judged.get(d['doc_id'])
            out.write(json.dumps({'topic_id':q,'query':queries[i],'top100':docs},ensure_ascii=False)+'\n')
        latency_runs.append({'repeat':rep,'mean_seconds':statistics.mean(latencies),
                             'median_seconds':statistics.median(latencies),
                             'min_seconds':min(latencies),'max_seconds':max(latencies),
                             'per_query_seconds':latencies})
        print(f'retrieval repeat {rep}/3: median {latency_runs[-1]["median_seconds"]:.4f}s',flush=True)
average={k:statistics.mean(per[q]['metrics'][k] for q in per) for k in ('ndcg@10','recall@10','recall@100','mrr')}
metrics={'version':'phase33-metrics-v1' if args.run_id else f'{variant}-pilot-metrics-v1',
         'run_id':args.run_id,'model':'BAAI/bge-small-en-v1.5',
         'method':('fixed 180/20/max240' if method=='fixed' else 'sentence-aware target180/soft220/max240')
                  if args.run_id else ('fixed 180/20/max240' if variant=='phase31' else
                                       'sentence-aware target180/soft220/max240/overlap20' if variant=='phase34' else
                                       'sentence-aware target180/soft220/max240'),
         'candidate_documents':len(corpus_ids),'query_count':len(per),'quality_run_count':1,
         'recall_denominator':'known positive documents in current frozen corpus; not full TREC-COVID recall'
                              if args.run_id else 'known positive documents in frozen Pilot corpus; not full TREC-COVID recall',
         'official_qrels_unchanged':True,'unjudged_treatment':'unknown; zero gain only in conventional qrel score',
         'macro_average':average,'per_query':per}
(RESULT/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf8')
with (RESULT/'per_query_metrics.csv').open('w',encoding='utf8',newline='') as csv_file:
    w=csv.writer(csv_file);w.writerow(['topic_id','query','official_positive','pilot_positive','ndcg@10',
                  'recall@10','recall@100','mrr','top10_unjudged','top20_unjudged','top100_unjudged'])
    for q in map(str,range(1,51)):
        x=per[q];m=x['metrics'];w.writerow([q,x['query'],x['official_positive'],x['available_positive'],
                m['ndcg@10'],m['recall@10'],m['recall@100'],m['mrr'],
                x['top']['10']['unjudged'],x['top']['20']['unjudged'],x['top']['100']['unjudged']])
unjudged={'version':'phase33-unjudged-v1' if args.run_id else f'{variant}-unjudged-v1',
          'per_query':{q:{n:per[q]['top'][n] for n in ('10','20','100')} for q in per},
          'average_rate':{n:statistics.mean(per[q]['top'][n]['unjudged_rate'] for q in per) for n in ('10','20','100')},
          'top10_count_distribution':dict(sorted(Counter(per[q]['top']['10']['unjudged'] for q in per).items())),
          'high_unjudged_top10_topics':[q for q in per if per[q]['top']['10']['unjudged']>=5]}
(RESULT/'unjudged_audit.json').write_text(json.dumps(unjudged,ensure_ascii=False,indent=2),encoding='utf8')
chunk_stats=json.loads((RESULT/'chunking_stats.json').read_text(encoding='utf8'))
embedding=json.loads((RESULT/'embedding_metrics.json').read_text(encoding='utf8'))
index=json.loads((RESULT/'index_metrics.json').read_text(encoding='utf8'))
seconds=[x['median_seconds'] for x in latency_runs]
perf={'version':'phase33-performance-v1' if args.run_id else f'{variant}-performance-v1',
      'manifest_build':json.loads((ROOT/'experiments/results/phase33/manifest_audit.json').read_text(encoding='utf8'))[scale]
                        if args.run_id else json.loads((ROOT/'experiments/results/phase31_pilot/manifest_build.json').read_text(encoding='utf8')),
      'document_loading_seconds':chunk_stats['timing']['document_loading_seconds'],
      'section_assembly_seconds':chunk_stats['timing']['section_assembly_seconds'],
      'common_preprocessing_and_chunking_seconds':chunk_stats['timing']['common_cleaning_and_chunking_seconds'],
      'chunking_total_seconds':chunk_stats.get('total_seconds',chunk_stats.get('chunking_active_wall_seconds')),
      'embedding_seconds':{k:embedding[k] for k in ('seconds_median','seconds_min','seconds_max')},
      'embedding_repeats':embedding['repeats'],'index_build_seconds':index['index_build_seconds'],
      'query_encoding_repeats':query_runs,'retrieval_latency_repeats':latency_runs,
      'retrieval_median_seconds_across_repeats':statistics.median(seconds),
      'ram_peak_bytes':max(x['process_rss_peak_bytes'] for x in embedding['repeats']),
      'vram_peak_reserved_bytes':max(x['gpu_peak_reserved_bytes'] for x in embedding['repeats']),
      'vram_peak_allocated_bytes':max(x['gpu_peak_allocated_bytes'] for x in embedding['repeats']),
      'cpu_process_peak_percent':max(x['cpu_process_percent_max'] for x in embedding['repeats']),
      'index_size_bytes':index['index_size_bytes'],'embedding_file_bytes':embedding['embedding_file_bytes'],
      'chunk_count':embedding['chunk_count']}
(RESULT/'performance_metrics.json').write_text(json.dumps(perf,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'macro_average':average,'unjudged':unjudged['average_rate'],
                  'queries':len(per),'documents':len(corpus_ids)},indent=2),flush=True)
