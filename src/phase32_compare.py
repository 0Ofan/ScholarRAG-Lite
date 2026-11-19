"""Paired Pilot comparison; bootstrap intervals are exploratory only."""
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/results/phase31_pilot';NEW=ROOT/'experiments/results/phase32_improved'
PROC=ROOT/'data/processed';load=lambda p:json.loads(p.read_text(encoding='utf8'))
manifest=load(PROC/'phase31_pilot_manifest.json')
b=load(BASE/'metrics.json');n=load(NEW/'metrics.json')
bc=load(BASE/'chunking_stats.json');nc=load(NEW/'chunking_stats.json')
be=load(BASE/'embedding_metrics.json');ne=load(NEW/'embedding_metrics.json')
bi=load(BASE/'index_metrics.json');ni=load(NEW/'index_metrics.json')
bp=load(BASE/'performance_metrics.json');npf=load(NEW/'performance_metrics.json')
assert b['query_count']==n['query_count']==50 and b['candidate_documents']==n['candidate_documents']==2000
for q in map(str,range(1,51)):
    assert b['per_query'][q]['query']==n['per_query'][q]['query']
    assert b['per_query'][q]['available_positive']==n['per_query'][q]['available_positive']==manifest['positive_per_query'][q]

def counts(path):
    result=Counter()
    with path.open(encoding='utf8') as stream:
        for line in stream:result[json.loads(line)['doc_id']]+=1
    return result
old_counts=counts(PROC/'phase31_pilot_chunks.jsonl')
new_counts=counts(PROC/'phase32_sentence_aware_chunks.jsonl')
ids=[d['doc_id'] for d in manifest['documents']]
assert len(old_counts)==len(new_counts)==len(ids)==2000
assert set(old_counts)==set(new_counts)==set(ids)
assert sum(old_counts.values())==bc['total_chunks'] and sum(new_counts.values())==nc['total_chunks']
(NEW/'document_chunk_counts.json').write_text(json.dumps({'version':'phase32-doc-chunks-v1',
    'document_ids':ids,'fixed':[old_counts[x] for x in ids],
    'sentence_aware':[new_counts[x] for x in ids]},ensure_ascii=False),encoding='utf8')

keys=('ndcg@10','recall@10','recall@100','mrr')
per={};rng=np.random.default_rng(3201)
for q in map(str,range(1,51)):
    old=b['per_query'][q];new=n['per_query'][q]
    row={'query':old['query'],'pilot_positive':old['available_positive'],
         'baseline':{k:old['metrics'][k] for k in keys},
         'sentence_aware':{k:new['metrics'][k] for k in keys},
         'delta':{k:new['metrics'][k]-old['metrics'][k] for k in keys},
         'baseline_unjudged':{str(k):old['top'][str(k)]['unjudged_rate'] for k in (10,20,100)},
         'sentence_unjudged':{str(k):new['top'][str(k)]['unjudged_rate'] for k in (10,20,100)},
         'unjudged_delta':{str(k):new['top'][str(k)]['unjudged_rate']-old['top'][str(k)]['unjudged_rate'] for k in (10,20,100)}}
    per[q]=row
bootstrap={};direction={}
for key in keys:
    diffs=np.asarray([per[str(q)]['delta'][key] for q in range(1,51)],dtype='float64')
    draws=rng.integers(0,50,size=(10000,50))
    means=diffs[draws].mean(axis=1)
    bootstrap[key]={'method':'paired query bootstrap, 10000 draws, seed 3201; exploratory',
                    'mean_delta':float(diffs.mean()),
                    'percentile_95_interval':[float(x) for x in np.quantile(means,[.025,.975])]}
    direction[key]={'improved':int((diffs>1e-12).sum()),
                    'declined':int((diffs<-1e-12).sum()),
                    'unchanged':int((abs(diffs)<=1e-12).sum()),
                    'improved_queries':[str(i+1) for i,x in enumerate(diffs) if x>1e-12],
                    'declined_queries':[str(i+1) for i,x in enumerate(diffs) if x<-1e-12]}

def per_doc_analysis(counter,path):
    lengths=np.asarray([counter[x] for x in ids],dtype='float64')
    body=np.asarray([d['body_chars'] for d in manifest['documents']],dtype='float64')
    with path.open(encoding='utf8') as stream:
        top100=[json.loads(line)['top100'] for line in stream]
    top_means=[statistics.mean(counter[x['doc_id']] for x in rows) for rows in top100]
    return {'mean_chunks':float(lengths.mean()),'median_chunks':float(np.median(lengths)),
            'p95_chunks':float(np.quantile(lengths,.95)),'max_chunks':int(lengths.max()),
            'pearson_body_chars_vs_chunks':float(np.corrcoef(body,lengths)[0,1]),
            'top100_mean_chunks_macro':statistics.mean(top_means),
            'top100_chunk_count_enrichment':statistics.mean(top_means)/float(lengths.mean())}
data={'version':'phase32-comparison-v1','pilot_manifest_version':manifest['version'],
      'query_count':50,'document_count':2000,'method':'same BGE, query encoding, qrels, scoring and retriever; different chunking only',
      'macro':{k:{'baseline':b['macro_average'][k],'sentence_aware':n['macro_average'][k],
                  'delta':n['macro_average'][k]-b['macro_average'][k]} for k in keys},
      'unjudged':{str(k):{'baseline':statistics.mean(per[q]['baseline_unjudged'][str(k)] for q in per),
                            'sentence_aware':statistics.mean(per[q]['sentence_unjudged'][str(k)] for q in per),
                            'delta':statistics.mean(per[q]['unjudged_delta'][str(k)] for q in per)} for k in (10,20,100)},
      'chunks':{'baseline':bc['total_chunks'],'sentence_aware':nc['total_chunks'],
                'delta':nc['total_chunks']-bc['total_chunks']},
      'embedding_seconds_median':{'baseline':be['seconds_median'],'sentence_aware':ne['seconds_median'],
                                  'delta':ne['seconds_median']-be['seconds_median']},
      'embedding_throughput_median':{'baseline':statistics.median(x['throughput_chunks_per_second'] for x in be['repeats']),
                                      'sentence_aware':statistics.median(x['throughput_chunks_per_second'] for x in ne['repeats'])},
      'index_bytes':{'baseline':bi['index_size_bytes'],'sentence_aware':ni['index_size_bytes'],
                     'delta':ni['index_size_bytes']-bi['index_size_bytes']},
      'index_build_seconds':{'baseline':bi['index_build_seconds'],'sentence_aware':ni['index_build_seconds']},
      'index_write_seconds':{'baseline':bi['index_write_seconds'],'sentence_aware':ni['index_write_seconds']},
      'retrieval_median_seconds':{'baseline':bp['retrieval_median_seconds_across_repeats'],
                                  'sentence_aware':npf['retrieval_median_seconds_across_repeats']},
      'ram_peak_bytes':{'baseline':bp['ram_peak_bytes'],'sentence_aware':npf['ram_peak_bytes']},
      'vram_peak_reserved_bytes':{'baseline':bp['vram_peak_reserved_bytes'],'sentence_aware':npf['vram_peak_reserved_bytes']},
      'per_doc_chunks':{'baseline':per_doc_analysis(old_counts,BASE/'retrieval_results.jsonl'),
                        'sentence_aware':per_doc_analysis(new_counts,NEW/'retrieval_results.jsonl')},
      'directions':direction,'bootstrap_exploratory':bootstrap,'per_query':per}
(NEW/'comparison.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
with (NEW/'per_query_comparison.csv').open('w',encoding='utf8',newline='') as f:
    writer=csv.writer(f);writer.writerow(['topic_id','query','pilot_positive']+
        [f'baseline_{x}' for x in keys]+[f'sentence_{x}' for x in keys]+[f'delta_{x}' for x in keys]+
        ['baseline_unjudged@10','sentence_unjudged@10','delta_unjudged@10'])
    for q in map(str,range(1,51)):
        x=per[q];writer.writerow([q,x['query'],x['pilot_positive']]+
          [x['baseline'][k] for k in keys]+[x['sentence_aware'][k] for k in keys]+
          [x['delta'][k] for k in keys]+[x['baseline_unjudged']['10'],x['sentence_unjudged']['10'],x['unjudged_delta']['10']])
print(json.dumps({'macro':data['macro'],'chunks':data['chunks'],'unjudged_top10':data['unjudged']['10']},indent=2))
