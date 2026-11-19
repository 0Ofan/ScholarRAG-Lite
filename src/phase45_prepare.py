"""Freeze Phase 4.5 ranking audit and paired evidence from saved Phase 4.4 results."""
import hashlib
import json
import time
from pathlib import Path

from src.phase37_prepare import digest
from src.rag_service import prepare_grounded_evidence

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results/phase45'
QUESTIONS = ROOT/'experiments/configs/phase36_questions.json'
BGE = ROOT/'experiments/results/phase33/phase33_large_fixed/retrieval_results.jsonl'
BM25 = ROOT/'experiments/results/phase43/retrieval_results.jsonl'
HYBRID = ROOT/'experiments/results/phase44/hybrid_results.jsonl'
HISTORICAL_A = ROOT/'experiments/results/phase37'


def lines(path):
    return {row['topic_id']: row for row in map(json.loads, path.read_text(encoding='utf8').splitlines())}


def save(path, obj, *, refresh=False):
    if path.exists():
        if json.loads(path.read_text(encoding='utf8')) != obj:
            if not refresh:
                raise ValueError(f'Existing frozen Phase 4.5 file differs: {path.name}')
            path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        return
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def evidence_from_bge_doc(doc, paper_rank, rrf_rank=None, bm25_rank=None, rrf_score=None):
    return {'title': doc['title'], 'doc_id': doc['doc_id'],
            'chunk_id': doc['best_chunk_id'], 'text': doc['best_chunk_text'],
            'score': doc['score'], 'source_path': doc['source_path'],
            'source_spans': doc['source_spans'], 'paper_rank': paper_rank,
            'bge_paper_rank': doc['rank'], 'bm25_paper_rank': bm25_rank,
            'rrf_rank': rrf_rank, 'rrf_score': rrf_score,
            'chunk_selection': 'saved BGE highest-similarity chunk within this paper'}


def top10(rows, bge_by_id, bm25_by_id, hybrid_by_id):
    answer=[]
    for row in rows[:10]:
        doc_id=row['doc_id']
        bge=bge_by_id.get(doc_id)
        bm25=bm25_by_id.get(doc_id)
        hybrid=hybrid_by_id.get(doc_id)
        answer.append({'doc_id':doc_id,'paper_title':row.get('title') or bge['title'],
                       'bge_rank':bge['rank'] if bge else None,
                       'bm25_rank':bm25['rank'] if bm25 else None,
                       'hybrid_rank':hybrid['rank'] if hybrid else None,
                       'rrf_score':hybrid['rrf_score'] if hybrid else None,
                       'qrel_status':hybrid['judgment_status'] if hybrid else
                       (bm25['judgment_status'] if bm25 else
                        'unjudged' if row.get('official_grade') is None else
                        'positive' if row['official_grade'] >= 1 else 'grade_0'),
                       'qrel_grade':hybrid['judgment'] if hybrid else
                       (bm25['judgment'] if bm25 else row.get('official_grade'))})
    return answer


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    questions=json.loads(QUESTIONS.read_text(encoding='utf8'))['questions']
    bge=lines(BGE); bm25=lines(BM25); hybrid=lines(HYBRID)
    metrics={row['topic_id']:row for row in __import__('csv').DictReader(
        (ROOT/'experiments/results/phase44/per_query_metrics.csv').open(encoding='utf-8-sig'))}
    successes=sorted(metrics, key=lambda q: float(metrics[q]['hybrid_ndcg@10'])-
                     max(float(metrics[q]['bge_ndcg@10']),float(metrics[q]['bm25_ndcg@10'])), reverse=True)
    successes=[qid for qid in successes if float(metrics[qid]['hybrid_ndcg@10'])>
               max(float(metrics[qid]['bge_ndcg@10']),float(metrics[qid]['bm25_ndcg@10']))][:3]
    cases=[]
    for qid in sorted(set(['16','22','31','34','41']+successes),key=int):
        a=bge[qid]['top100'];b=bm25[qid]['top100'];h=hybrid[qid]['hybrid_top100']
        by_a={x['doc_id']:x for x in a};by_b={x['doc_id']:x for x in b};by_h={x['doc_id']:x for x in h}
        cases.append({'topic_id':qid,'query':bge[qid]['query'],
                      'case_type':'failure' if qid in {'16','22','31','34','41'} else 'gain',
                      'ndcg@10':{method:float(metrics[qid][f'{method}_ndcg@10'])
                                 for method in ('bge','bm25','hybrid')},
                      'top10':{'bge':top10(a,by_a,by_b,by_h),
                               'bm25':top10(b,by_a,by_b,by_h),
                               'hybrid':top10(h,by_a,by_b,by_h)}})
    save(OUT/'ranking_cases.json',{'version':'phase45-ranking-v1','success_topics':successes,
                                   'cases':cases},refresh=True)
    entries=[]
    for question in questions:
        qid=question['question_id']
        topic=str(question['source_topic_ids'][0])
        a=bge[topic]['top100'];h=hybrid[topic]['hybrid_top100']
        a_by_id={x['doc_id']:x for x in a}
        a_evidence=prepare_grounded_evidence([evidence_from_bge_doc(doc,i)
                                              for i,doc in enumerate(a[:3],1)])
        h_evidence=prepare_grounded_evidence([evidence_from_bge_doc(a_by_id[item['doc_id']],i,
                                                item['rank'],item['bm25_rank'],item['rrf_score'])
                                              for i,item in enumerate(h[:3],1)])
        old=json.loads((HISTORICAL_A/f'{qid}.json').read_text(encoding='utf8'))
        old_keys=[(x['doc_id'],x['chunk_id'],x['text']) for x in old['evidence']]
        current_keys=[(x['doc_id'],x['chunk_id'],x['text']) for x in a_evidence]
        # The old call must also have used the same retrieval query. Phase 3.7 used the natural question.
        exact_reuse=(old['question']==question['question'] and old_keys==current_keys and
                     old.get('status')=='success' and old.get('model')=='deepseek-flash' and
                     old.get('temperature')==.3 and old.get('thinking')=='disabled')
        entries.append({'question_id':qid,'question':question['question'],
                        'source_topic_ids':question['source_topic_ids'],
                        'retrieval_topic_id':topic,'retrieval_query':bge[topic]['query'],
                        'topic_mapping_rule':'first frozen source_topic_ids entry',
                        'historical_phase37_reuse_eligible':exact_reuse,
                        'A':{'method':'bge','evidence':a_evidence,'evidence_sha256':digest(a_evidence)},
                        'B':{'method':'hybrid','evidence':h_evidence,'evidence_sha256':digest(h_evidence)}})
    raw=QUESTIONS.read_bytes().replace(b'\r\n',b'\n')
    save(OUT/'inputs.json',{'version':'phase45-inputs-v1','question_set_sha256':hashlib.sha256(raw).hexdigest(),
         'retrieval_inputs':{'bge':str(BGE.relative_to(ROOT)),'bm25':str(BM25.relative_to(ROOT)),
                             'hybrid':str(HYBRID.relative_to(ROOT))},
         'prompt':'src/deepseek_client.py DeepSeekClient.stream default system prompt (Phase 3.7)',
         'model':'deepseek-flash','thinking':'disabled','temperature':.3,'max_tokens':512,
         'evidence_budget':3,'entries':entries})
    print('Prepared',len(entries),'questions;',sum(x['historical_phase37_reuse_eligible'] for x in entries),
          'old A responses eligible; success topics',successes,flush=True)


if __name__=='__main__':
    main()
