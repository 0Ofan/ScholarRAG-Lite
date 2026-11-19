"""Frozen Top-100 paper-level RRF fusion and same-protocol evaluation."""
import csv
import json
import statistics
import time
from pathlib import Path

import psutil

from src.evaluation import score_topic

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/phase44'
BGE=ROOT/'experiments/results/phase33/phase33_large_fixed'
BM25=ROOT/'experiments/results/phase43'
MANIFEST=ROOT/'data/processed/phase33_large_manifest.json'
TOPICS=ROOT/'data/evaluation/topics.jsonl'
RRF_K=60


def write(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def fuse(bge_ids, bm25_ids, k=RRF_K, limit=100):
    """Use only one-based ranks; absent side contributes zero."""
    ranks={}
    for label, ids in (('bge_rank',bge_ids),('bm25_rank',bm25_ids)):
        for rank, doc_id in enumerate(ids,1):
            ranks.setdefault(doc_id,{'doc_id':doc_id,'bge_rank':None,'bm25_rank':None})[label]=rank
    results=[]
    for item in ranks.values():
        item['rrf_score']=sum(1/(k+rank) for rank in (item['bge_rank'],item['bm25_rank']) if rank is not None)
        results.append(item)
    results.sort(key=lambda x:(-x['rrf_score'],x['doc_id']))
    return [{**item,'rank':rank} for rank,item in enumerate(results[:limit],1)]


def stat(values):
    return {'n':len(values),'mean':statistics.mean(values),'median':statistics.median(values),
            'min':min(values),'max':max(values),'sum':sum(values)}


def load_lines(path):
    return {x['topic_id']:x for x in map(json.loads,path.read_text(encoding='utf8').splitlines())}


def main():
    manifest=json.loads(MANIFEST.read_text(encoding='utf8'))
    corpus={d['doc_id'] for d in manifest['documents']}
    titles={d['doc_id']:d['title'] for d in manifest['documents']}
    topics=load_lines(TOPICS)
    bge=load_lines(BGE/'retrieval_results.jsonl')
    bm25=load_lines(BM25/'retrieval_results.jsonl')
    old_bge=json.loads((BGE/'metrics.json').read_text(encoding='utf8'))
    old_bm25=json.loads((BM25/'metrics.json').read_text(encoding='utf8'))
    old_bge_perf=json.loads((BGE/'performance_metrics.json').read_text(encoding='utf8'))
    old_bm25_perf=json.loads((BM25/'performance.json').read_text(encoding='utf8'))
    old_bm25_per={x['topic_id']:x for x in json.loads((BM25/'comparison.json').read_text(encoding='utf8'))['per_query']}
    if not (set(topics)==set(bge)==set(bm25)==set(manifest['effective_qrels'])=={str(i) for i in range(1,51)}):
        raise ValueError('Frozen query identifiers differ')
    process=psutil.Process();rss_before=process.memory_info().rss;rss_observed=rss_before
    rows=[];complements=[];output=[];durations=[];rank_differences=[];shared_top10=0
    for qid in map(str,range(1,51)):
        a=bge[qid];b=bm25[qid];query=topics[qid]['query']
        a_ids=[x['doc_id'] for x in a['top100']];b_ids=[x['doc_id'] for x in b['top100']]
        if a['query']!=query or b['query']!=query or len(set(a_ids))!=len(set(b_ids))!=100:
            raise ValueError(f'Input ranking mismatch: {qid}')
        if not set(a_ids+b_ids)<=corpus:
            raise ValueError(f'Unknown paper ID: {qid}')
        started=time.perf_counter()
        fused=fuse(a_ids,b_ids)
        elapsed=time.perf_counter()-started
        durations.append(elapsed);rss_observed=max(rss_observed,process.memory_info().rss)
        ids=[x['doc_id'] for x in fused]
        bge_ranks={uid:rank for rank,uid in enumerate(a_ids,1)}
        bm25_ranks={uid:rank for rank,uid in enumerate(b_ids,1)}
        rank_differences.extend(abs(bge_ranks[uid]-bm25_ranks[uid]) for uid in bge_ranks.keys()&bm25_ranks.keys())
        shared_top10+=sum(item['bge_rank'] is not None and item['bm25_rank'] is not None for item in fused[:10])
        qrels=manifest['effective_qrels'][qid]
        metrics=score_topic(ids,qrels,corpus)
        if len(ids)!=100 or len(set(ids))!=100 or metrics['known_relevant_in_corpus']!=manifest['positive_per_query'][qid]:
            raise ValueError(f'Hybrid Top-100 or denominator mismatch: {qid}')
        known={uid for uid,grade in qrels.items() if grade>=1 and uid in corpus}
        A=set(a_ids)&known;B=set(b_ids)&known;H=set(ids)&known;H10=set(ids[:10])&known
        groups={'bge_only':A-B,'bm25_only':B-A,'both':A&B,'neither':known-(A|B)}
        comp={'topic_id':qid,'known_positive':len(known),
              'input_categories':{name:sorted(group) for name,group in groups.items()},
              'hybrid_top100_retained':{name:sorted(H&group) for name,group in groups.items()},
              'hybrid_top100_lost':{name:sorted(group-H) for name,group in groups.items()},
              'hybrid_top10_lifted':{name:sorted(H10&group) for name,group in groups.items()},
              'hybrid_positive_ids':sorted(H),
              'new_vs_bge':sorted(H-A),'lost_vs_bge':sorted(A-H),
              'new_vs_bm25':sorted(H-B),'lost_vs_bm25':sorted(B-H)}
        complements.append(comp)
        historical_a=old_bge['per_query'][qid]['metrics']
        historical_b=old_bm25_per[qid]
        row={'topic_id':qid,'query':query,
             'bge_ndcg@10':historical_a['ndcg@10'],'bm25_ndcg@10':historical_b['ndcg@10'],
             'hybrid_ndcg@10':metrics['ndcg@10'],
             'bge_recall@10':historical_a['recall@10'],'bm25_recall@10':historical_b['recall@10'],
             'hybrid_recall@10':metrics['recall@10'],
             'bge_recall@100':historical_a['recall@100'],'bm25_recall@100':historical_b['recall@100'],
             'hybrid_recall@100':metrics['recall@100'],
             'bge_mrr':historical_a['mrr'],'bm25_mrr':historical_b['mrr'],'hybrid_mrr':metrics['mrr'],
             'bge_top10_unjudged':old_bge['per_query'][qid]['top']['10']['unjudged_rate'],
             'bm25_top10_unjudged':historical_b['top10_unjudged_rate'],
             'hybrid_top10_unjudged':metrics['unjudged_in_top10']/10,
             'hybrid_positive_top10':len(H10),'hybrid_positive_top100':len(H),
             'hybrid_new_vs_bge_top100':len(H-A),'hybrid_lost_vs_bge_top100':len(A-H),
             'hybrid_new_vs_bm25_top100':len(H-B),'hybrid_lost_vs_bm25_top100':len(B-H),
             'rrf_seconds':elapsed}
        rows.append(row)
        output.append({'topic_id':qid,'query':query,'input_bge_top100':a_ids,
                       'input_bm25_top100':b_ids,'hybrid_top100':[
            {**item,'title':titles[item['doc_id']],
             'judgment':qrels.get(item['doc_id']),
             'judgment_status':'unjudged' if item['doc_id'] not in qrels else
             'positive' if qrels[item['doc_id']]>=1 else 'grade_0'} for item in fused]})
    with (OUT/'hybrid_results.jsonl').open('w',encoding='utf8',newline='\n') as file:
        for item in output:file.write(json.dumps(item,ensure_ascii=False)+'\n')
    with (OUT/'per_query_metrics.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    quality={method:{metric:statistics.mean(r[f'{method}_{metric}'] for r in rows)
                     for metric in ('ndcg@10','recall@10','recall@100','mrr','top10_unjudged')}
             for method in ('bge','bm25','hybrid')}
    if any(abs(quality['bm25'][key]-old_bm25['macro_average'][key if key!='top10_unjudged' else 'top10_unjudged_rate'])>1e-10
           for key in quality['bm25']):
        raise ValueError('BM25 historical macro differs')
    totals={name:{category:sum(len(q[name][category]) for q in complements)
                  for category in ('bge_only','bm25_only','both','neither')}
            for name in ('input_categories','hybrid_top100_retained','hybrid_top100_lost','hybrid_top10_lifted')}
    gains={name:sum(len(q[name]) for q in complements)
           for name in ('new_vs_bge','lost_vs_bge','new_vs_bm25','lost_vs_bm25')}
    wins={old:{'hybrid_higher':sum(r['hybrid_ndcg@10']>r[f'{old}_ndcg@10'] for r in rows),
               'hybrid_lower':sum(r['hybrid_ndcg@10']<r[f'{old}_ndcg@10'] for r in rows),
               'tie':sum(r['hybrid_ndcg@10']==r[f'{old}_ndcg@10'] for r in rows)}
          for old in ('bge','bm25')}
    write(OUT/'metrics.json',{'version':'phase44-metrics-v1','query_count':50,'document_count':12000,
          'rrf_k':RRF_K,'quality':quality,'ndcg_query_wins':wins,
          'recall_denominator':'known positive documents in frozen current Large corpus; not full TREC-COVID recall',
          'unjudged_treatment':'unknown; zero gain only in conventional qrel score'})
    write(OUT/'complementarity.json',{'version':'phase44-complement-v1','known_positive_pairs':1560,
          'categories_total':totals,'hybrid_vs_inputs':gains,
          'rank_conflict':{'shared_input_paper_occurrences':len(rank_differences),
                           'mean_absolute_rank_difference':statistics.mean(rank_differences),
                           'median_absolute_rank_difference':statistics.median(rank_differences),
                           'rank_difference_over_20':sum(x>20 for x in rank_differences),
                           'hybrid_top10_from_both_inputs':shared_top10,
                           'hybrid_top10_total':500},
          'per_query':complements})
    rss_after=process.memory_info().rss
    rss_observed=max(rss_observed,rss_after)
    write(OUT/'performance.json',{'version':'phase44-performance-v1','rrf_query_seconds':stat(durations),
          'rrf_output_file_bytes':(OUT/'hybrid_results.jsonl').stat().st_size,
          'rss_before_fusion_bytes':rss_before,'rss_after_fusion_bytes':rss_after,
          'rss_max_observed_bytes':rss_observed,
          'rss_observed_increase_bytes':rss_observed-rss_before,
          'bge_historical_top100_retrieval_median_seconds':old_bge_perf['retrieval_median_seconds_across_repeats'],
          'bm25_historical_top100_query_mean_seconds':old_bm25_perf['bm25_query_seconds']['mean'],
          'note':'RRF timing excludes index load, query encoding and both original searches; RSS sampled at query boundaries, not a continuous peak.'})
    print('Hybrid completed',quality,'fusion seconds',sum(durations),flush=True)


if __name__=='__main__':
    main()
