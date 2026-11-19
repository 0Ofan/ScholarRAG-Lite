"""Evaluate the independent BM25 paper ranking against frozen TREC-COVID qrels."""
import csv
import json
import statistics
import time
from pathlib import Path

import psutil

from src.evaluation import score_topic
from src.phase43_bm25 import BM25Index, MANIFEST, OUT, ROOT

TOPICS = ROOT/'data/evaluation/topics.jsonl'
BGE = ROOT/'experiments/results/phase33/phase33_large_fixed'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def summary(values):
    return {'mean':statistics.mean(values), 'median':statistics.median(values),
            'min':min(values), 'max':max(values), 'sum':sum(values)}


def complement(bm_ids, bge_ids, known):
    a = set(bm_ids)&known
    b = set(bge_ids)&known
    return {'bm25_only':len(a-b), 'bge_only':len(b-a), 'both':len(a&b),
            'neither':len(known-(a|b)), 'bm25_positive_ids':sorted(a),
            'bge_positive_ids':sorted(b), 'bm25_only_ids':sorted(a-b),
            'bge_only_ids':sorted(b-a)}


def main():
    manifest=json.loads(MANIFEST.read_text(encoding='utf8'))
    docs=manifest['documents']
    corpus={d['doc_id'] for d in docs}
    titles={d['doc_id']:d['title'] for d in docs}
    topics={x['topic_id']:x for x in map(json.loads, TOPICS.read_text(encoding='utf8').splitlines())}
    history={x['topic_id']:x for x in map(json.loads, (BGE/'retrieval_results.jsonl').read_text(encoding='utf8').splitlines())}
    bge_metrics=json.loads((BGE/'metrics.json').read_text(encoding='utf8'))
    if not (set(topics)==set(history)==set(manifest['effective_qrels'])):
        raise ValueError('Frozen topics, qrels and BGE results differ')
    start_load=time.perf_counter()
    index=BM25Index.load(OUT)
    load_seconds=time.perf_counter()-start_load
    if set(index.doc_ids)!=corpus or len(index.doc_ids)!=12000:
        raise ValueError('BM25 and BGE Large paper IDs differ')
    process=psutil.Process()
    rss_peak=process.memory_info().rss
    per=[]; result_lines=[]
    for qid in map(str, range(1,51)):
        query=topics[qid]['query']
        started=time.perf_counter()
        hits=index.search(query,100)
        seconds=time.perf_counter()-started
        rss_peak=max(rss_peak,process.memory_info().rss)
        ids=[h['doc_id'] for h in hits]
        bge_ids=[h['doc_id'] for h in history[qid]['top100']]
        if len(set(ids))!=100 or len(set(bge_ids))!=100 or query!=history[qid]['query']:
            raise ValueError(f'Invalid top100 alignment: {qid}')
        judgments=manifest['effective_qrels'][qid]
        known={uid for uid,grade in judgments.items() if grade>=1 and uid in corpus}
        metrics=score_topic(ids,judgments,corpus)
        if metrics['known_relevant_in_corpus']!=manifest['positive_per_query'][qid]:
            raise ValueError(f'Recall denominator changed: {qid}')
        bge_q=bge_metrics['per_query'][qid]['metrics']
        overlap10=len(set(ids[:10])&set(bge_ids[:10]))
        overlap100=len(set(ids)&set(bge_ids))
        comp10=complement(ids[:10],bge_ids[:10],known)
        comp100=complement(ids,bge_ids,known)
        per.append({'topic_id':qid,'query':query,'known_positive':len(known),
            'ndcg@10':metrics['ndcg@10'],'recall@10':metrics['recall@10'],
            'recall@100':metrics['recall@100'],'mrr':metrics['mrr'],
            'top10_unjudged_rate':metrics['unjudged_in_top10']/10,
            'bge_ndcg@10':bge_q['ndcg@10'],'bge_recall@10':bge_q['recall@10'],
            'bge_recall@100':bge_q['recall@100'],'bge_mrr':bge_q['mrr'],
            'bge_top10_unjudged_rate':bge_metrics['per_query'][qid]['top']['10']['unjudged_rate'],
            'top10_overlap_count':overlap10,'top10_overlap_rate':overlap10/10,
            'bm25_only_top10_count':10-overlap10,'bge_only_top10_count':10-overlap10,
            'top100_overlap_count':overlap100,'query_seconds':seconds,
            'positive_top10':comp10,'positive_top100':comp100})
        result_lines.append({'topic_id':qid,'query':query,'top100':[
            {**h,'title':titles[h['doc_id']],
             'judgment':judgments.get(h['doc_id']),
             'judgment_status':'unjudged' if h['doc_id'] not in judgments else
                 'positive' if judgments[h['doc_id']]>=1 else 'grade_0'} for h in hits]})
        print(qid,'nDCG',round(metrics['ndcg@10'],4),'s',round(seconds,3),flush=True)
    with (OUT/'retrieval_results.jsonl').open('w',encoding='utf8',newline='\n') as file:
        for x in result_lines:file.write(json.dumps(x,ensure_ascii=False)+'\n')
    columns=[key for key in per[0] if key not in ('positive_top10','positive_top100')]
    columns+=['positive_top10_bm25_only','positive_top10_bge_only','positive_top10_both','positive_top10_neither',
              'positive_top100_bm25_only','positive_top100_bge_only','positive_top100_both','positive_top100_neither']
    with (OUT/'per_query_metrics.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=columns);writer.writeheader()
        for row in per:
            flat={key:row[key] for key in columns if key in row}
            for k in (10,100):
                for name in ('bm25_only','bge_only','both','neither'):
                    flat[f'positive_top{k}_{name}']=row[f'positive_top{k}'][name]
            writer.writerow(flat)
    quality={name:statistics.mean(row[name] for row in per)
             for name in ('ndcg@10','recall@10','recall@100','mrr','top10_unjudged_rate')}
    bge_quality={name:statistics.mean(row['bge_'+name] for row in per)
                 for name in ('ndcg@10','recall@10','recall@100','mrr','top10_unjudged_rate')}
    complement_totals={f'top{k}':{name:sum(row[f'positive_top{k}'][name] for row in per)
                              for name in ('bm25_only','bge_only','both','neither')}
                       for k in (10,100)}
    overlap={'top10':{'mean_count':statistics.mean(r['top10_overlap_count'] for r in per),
                      'mean_rate':statistics.mean(r['top10_overlap_rate'] for r in per),
                      'total_bm25_unique':sum(r['bm25_only_top10_count'] for r in per),
                      'total_bge_unique':sum(r['bge_only_top10_count'] for r in per)},
             'top100':{'mean_count':statistics.mean(r['top100_overlap_count'] for r in per)}}
    win={'bm25_higher':sum(r['ndcg@10']>r['bge_ndcg@10'] for r in per),
         'bge_higher':sum(r['ndcg@10']<r['bge_ndcg@10'] for r in per),
         'tie':sum(r['ndcg@10']==r['bge_ndcg@10'] for r in per)}
    write(OUT/'metrics.json',{'version':'phase43-metrics-v1','run':'bm25-large-paper-level',
        'query_count':50,'document_count':12000,'known_positive_pairs':sum(r['known_positive'] for r in per),
        'recall_denominator':'known positives within current frozen Large corpus, not full TREC-COVID recall',
        'unjudged_treatment':'unknown; zero gain only in conventional qrel score',
        'macro_average':quality,'bge_historical_macro_average':bge_quality})
    write(OUT/'comparison.json',{'version':'phase43-complement-v1','query_count':50,
          'quality':quality,'bge_historical_quality':bge_quality,
          'ndcg_wins':win,'overlap':overlap,'known_positive_complement_totals':complement_totals,
          'per_query':per})
    times=[r['query_seconds'] for r in per]
    write(OUT/'performance.json',{'version':'phase43-performance-v1',
        'bm25_index_load_seconds':load_seconds,
        'bm25_query_seconds':summary(times),
        'bm25_evaluation_process_peak_rss_bytes':rss_peak,
        'bge_historical_embedding_seconds_median':1356.55453859939,
        'bge_historical_faiss_index_build_seconds':.8252590000047348,
        'bge_historical_retrieval_top100_seconds_median':.016999550018226728,
        'warning':'BM25 paper-level query time and BGE chunk-level FAISS time cover different work; BGE query embedding excluded from historical FAISS time.'})
    print('complete: 50 queries',quality,flush=True)


if __name__=='__main__':
    main()
