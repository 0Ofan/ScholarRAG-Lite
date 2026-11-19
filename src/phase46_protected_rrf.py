"""BGE Top-3 preserving RRF over frozen paper-level Top-100 rankings."""
import csv
import json
import statistics
import time
from pathlib import Path

import psutil

from src.evaluation import score_topic
from src.phase44_rrf import BGE, BM25, MANIFEST, ROOT, fuse, load_lines

OUT=ROOT/'experiments/results/phase46'
STANDARD=ROOT/'experiments/results/phase44'
PROTECT=3
DEPTH=100
CASES=('16','22','31','34','41','2','7','20')
METRICS=('ndcg@10','recall@10','recall@100','mrr','top10_unjudged')


def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def protected_rrf(bge_ids,bm25_ids,protect=PROTECT,depth=DEPTH):
    """Constrain Top-10 membership, then order within Top-10 and the tail by RRF."""
    if len(bge_ids)<protect or depth<10 or protect>10:
        raise ValueError('Invalid fixed protection budget')
    all_rows=fuse(bge_ids,bm25_ids,limit=len(set(bge_ids)|set(bm25_ids)))
    protected=set(bge_ids[:protect])
    top10=[x for x in all_rows if x['doc_id'] in protected]
    if len(top10)>10:raise ValueError('Protected candidates exceed Top-10')
    for item in all_rows:
        if len(top10)==10:break
        if item['doc_id'] not in protected:top10.append(item)
    top10.sort(key=lambda x:(-x['rrf_score'],x['doc_id']))
    selected={x['doc_id'] for x in top10}
    tail=[x for x in all_rows if x['doc_id'] not in selected][:depth-10]
    original_ranks={row['doc_id']:rank for rank,row in enumerate(all_rows,1)}
    return [{**x,'rank':i,'protected':x['doc_id'] in protected,
             'standard_rrf_rank':original_ranks[x['doc_id']]}
            for i,x in enumerate(top10+tail,1)]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    bge=load_lines(BGE/'retrieval_results.jsonl')
    bm25=load_lines(BM25/'retrieval_results.jsonl')
    standard=load_lines(STANDARD/'hybrid_results.jsonl')
    manifest=json.loads(MANIFEST.read_text(encoding='utf8'))
    corpus={x['doc_id'] for x in manifest['documents']}
    titles={x['doc_id']:x['title'] for x in manifest['documents']}
    old_standard=json.loads((STANDARD/'metrics.json').read_text(encoding='utf8'))
    old_rows={x['topic_id']:x for x in csv.DictReader(
        (STANDARD/'per_query_metrics.csv').open(encoding='utf-8-sig',newline=''))}
    process=psutil.Process();rss_before=process.memory_info().rss;rss_max=rss_before
    durations=[];rows=[];outputs=[];complements=[];cases=[]
    same_top3=0;same_top100_sets=0;changed_top10_sets=0
    for qid in map(str,range(1,51)):
        a=bge[qid];b=bm25[qid];s=standard[qid]
        a_ids=[x['doc_id'] for x in a['top100']]
        b_ids=[x['doc_id'] for x in b['top100']]
        s_ids=[x['doc_id'] for x in s['hybrid_top100']]
        if (a_ids!=s['input_bge_top100'] or b_ids!=s['input_bm25_top100'] or
            a['query']!=b['query'] or len(set(a_ids))!=100 or len(set(b_ids))!=100 or
            not set(a_ids+b_ids)<=corpus):
            raise ValueError(f'Frozen input mismatch: Topic {qid}')
        start=time.perf_counter()
        c=protected_rrf(a_ids,b_ids)
        durations.append(time.perf_counter()-start)
        rss_max=max(rss_max,process.memory_info().rss)
        c_ids=[x['doc_id'] for x in c]
        same_top3+=c_ids[:3]==s_ids[:3]
        same_top100_sets+=set(c_ids)==set(s_ids)
        changed_top10_sets+=set(c_ids[:10])!=set(s_ids[:10])
        if len(c_ids)!=100 or len(set(c_ids))!=100 or not set(a_ids[:3])<=set(c_ids[:10]):
            raise ValueError(f'Protected Top-10/100 invalid: Topic {qid}')
        qrels=manifest['effective_qrels'][qid]
        metrics=score_topic(c_ids,qrels,corpus)
        known={uid for uid,grade in qrels.items() if grade>=1 and uid in corpus}
        A=set(a_ids)&known;B=set(b_ids)&known;S=set(s_ids)&known;C=set(c_ids)&known
        groups={'bge_only':A-B,'bm25_only':B-A,'both':A&B,'neither':known-(A|B)}
        by_method={'bge':a_ids,'standard':s_ids,'protected':c_ids}
        comp={'topic_id':qid,'known_positive':len(known),
              'protected_bge_top3_ids':a_ids[:3],
              'protected_bge_top3_known_positive':sorted(set(a_ids[:3])&known),
              'input_categories':{k:sorted(v) for k,v in groups.items()},
              'methods':{m:{'top10_known_positive':sorted(set(ids[:10])&known),
                             'top100_known_positive':sorted(set(ids[:100])&known),
                             'top10_by_input_category':{k:sorted(set(ids[:10])&v) for k,v in groups.items()},
                             'top100_by_input_category':{k:sorted(set(ids[:100])&v) for k,v in groups.items()}}
                         for m,ids in by_method.items()}}
        complements.append(comp)
        history=old_rows[qid]
        row={'topic_id':qid,'query':a['query']}
        for metric in METRICS:
            row[f'bge_{metric}']=float(history[f'bge_{metric}'])
            row[f'standard_{metric}']=float(history[f'hybrid_{metric}'])
            row[f'protected_{metric}']=metrics['unjudged_in_top10']/10 if metric=='top10_unjudged' else metrics[metric]
        row.update(protected_top3_known_positive=len(set(a_ids[:3])&known),
                   standard_top10_known_positive=len(set(s_ids[:10])&known),
                   protected_top10_known_positive=len(set(c_ids[:10])&known),
                   standard_top100_known_positive=len(S),protected_top100_known_positive=len(C),
                   standard_bm25_only_top100=len(S&groups['bm25_only']),
                   protected_bm25_only_top100=len(C&groups['bm25_only']),
                   protection_seconds=durations[-1])
        rows.append(row)
        outputs.append({'topic_id':qid,'query':a['query'],'input_bge_top100':a_ids,
                        'input_bm25_top100':b_ids,'protected_top100':[
            {**x,'title':titles[x['doc_id']],'judgment':qrels.get(x['doc_id']),
             'judgment_status':'unjudged' if x['doc_id'] not in qrels else
             'positive' if qrels[x['doc_id']]>=1 else 'grade_0'} for x in c]})
        if qid in CASES:
            cases.append({'topic_id':qid,'query':a['query'],
                          'metrics':{m:{key:row[f'{m}_{key}'] for key in METRICS}
                                     for m in ('bge','standard','protected')},
                          'bge_top3':[{ 'doc_id':uid,'title':titles[uid],
                                        'judgment':qrels.get(uid),
                                        'standard_rank':next((x['rank'] for x in s['hybrid_top100']
                                                              if x['doc_id']==uid),None),
                                        'protected_rank':c_ids.index(uid)+1} for uid in a_ids[:3]],
                          'standard_top10':[{'doc_id':x['doc_id'],'title':x['title'],
                                             'judgment':x['judgment'],'bge_rank':x['bge_rank'],
                                             'bm25_rank':x['bm25_rank'],'rrf_score':x['rrf_score']}
                                            for x in s['hybrid_top100'][:10]],
                          'protected_top10':outputs[-1]['protected_top100'][:10],
                          'known_positive_coverage':{
                              'input_total':len(known),'input_union_top100':len(A|B),
                              'standard_top100':len(S),'protected_top100':len(C)}})
    with (OUT/'protected_results.jsonl').open('w',encoding='utf8',newline='\n') as file:
        for x in outputs:file.write(json.dumps(x,ensure_ascii=False)+'\n')
    with (OUT/'per_query_metrics.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    quality={m:{key:statistics.mean(row[f'{m}_{key}'] for row in rows) for key in METRICS}
             for m in ('bge','standard','protected')}
    if any(abs(quality['bge'][key]-old_standard['quality']['bge'][key])>1e-10 or
           abs(quality['standard'][key]-old_standard['quality']['hybrid'][key])>1e-10
           for key in METRICS):
        raise ValueError('Historical metric mismatch')
    save(OUT/'metrics.json',{'version':'phase46-metrics-v1','query_count':50,
         'corpus_documents':len(corpus),'quality':quality,
         'ranking_effects':{'same_top3_order_topics':same_top3,
                            'same_top100_set_topics':same_top100_sets,
                            'changed_top10_set_topics':changed_top10_sets},
         'ndcg_wins_vs_standard':{
             'higher':sum(r['protected_ndcg@10']>r['standard_ndcg@10'] for r in rows),
             'lower':sum(r['protected_ndcg@10']<r['standard_ndcg@10'] for r in rows),
             'tie':sum(r['protected_ndcg@10']==r['standard_ndcg@10'] for r in rows)},
         'recall_definition':'known positive papers within frozen 12,000-paper corpus'})
    totals={m:{depth:{category:sum(len(q['methods'][m][f'{depth}_by_input_category'][category])
                             for q in complements)
                       for category in ('bge_only','bm25_only','both','neither')}
               for depth in ('top10','top100')}
            for m in ('bge','standard','protected')}
    save(OUT/'complementarity.json',{'version':'phase46-complement-v1',
         'input_categories':{category:sum(len(q['input_categories'][category]) for q in complements)
                             for category in ('bge_only','bm25_only','both','neither')},
         'protected_bge_top3_known_positive':sum(len(q['protected_bge_top3_known_positive'])
                                                for q in complements),
         'method_totals':totals,'per_query':complements})
    save(OUT/'case_analysis.json',{'version':'phase46-cases-v1','cases':cases})
    rss_after=process.memory_info().rss
    old_perf=json.loads((STANDARD/'performance.json').read_text(encoding='utf8'))
    save(OUT/'performance.json',{'version':'phase46-performance-v1',
         'queries':50,'total_seconds':sum(durations),'mean_seconds':statistics.mean(durations),
         'median_seconds':statistics.median(durations),
         'standard_rrf_historical_mean_seconds':old_perf['rrf_query_seconds']['mean'],
         'output_bytes':(OUT/'protected_results.jsonl').stat().st_size,
         'rss_before_bytes':rss_before,'rss_after_bytes':rss_after,
         'rss_max_sampled_bytes':max(rss_max,rss_after),
         'note':'Offline Python fusion only; historical RRF and this run are separate measurements. Excludes both searches, model/index loads and RAG.'})
    cases_by_id={c['topic_id']:c for c in cases}
    gate={'ndcg_not_below_standard':quality['protected']['ndcg@10']>=quality['standard']['ndcg@10'],
          'recall100_not_below_standard':quality['protected']['recall@100']>=quality['standard']['recall@100'],
          'topics_16_31_41_all_improve':all(cases_by_id[q]['metrics']['protected']['ndcg@10']>
                                           cases_by_id[q]['metrics']['standard']['ndcg@10']
                                           for q in ('16','31','41')),
          'bm25_only_top100_not_lost':totals['protected']['top100']['bm25_only']>=
                                     totals['standard']['top100']['bm25_only']}
    gate['run_real_rag']=all(gate.values())
    save(OUT/'rag_gate.json',{'version':'phase46-rag-gate-v1','conditions':gate,
         'decision':'run small real RAG' if gate['run_real_rag'] else 'skip DeepSeek API'})
    print('Protected RRF:',quality,'gate:',gate,flush=True)


if __name__=='__main__':main()
