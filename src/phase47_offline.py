"""Frozen 20-question A/B/C evidence comparison over existing BGE chunk vectors."""
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.evidence_selector import BACKGROUND, LIMIT, RESULT, terms
from src.phase47_chunk_selection import global_top3, mmr_top3, rank_chunks
from src.rag_service import prepare_grounded_evidence
from src.retriever import PROMPT

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/phase47'
QUESTIONS=ROOT/'experiments/configs/phase36_questions.json'
HISTORY=ROOT/'experiments/results/phase36'
GROUNDED=ROOT/'experiments/results/phase37'
CHUNKS=ROOT/'data/processed/phase33_large_fixed_chunks.jsonl'
EMBEDDINGS=ROOT/'data/processed/phase33_large_fixed_embeddings.npy'
INDEX_META=ROOT/'data/indexes/phase33_large_fixed/index_metadata.json'


def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def source_rows(needed):
    by_doc={uid:[] for uid in needed}
    start=time.perf_counter()
    with CHUNKS.open(encoding='utf8') as file:
        for position,line in enumerate(file):
            uid=line[14:].split(':',1)[0]
            if uid in by_doc:
                row=json.loads(line);row['_position']=position
                by_doc[uid].append(row)
    if any(not rows for rows in by_doc.values()):
        raise ValueError('A frozen Top-10 paper lacks source chunks')
    return by_doc,time.perf_counter()-start


def evidence_from_ranked(selected,by_position):
    original=[]
    for row in selected:
        chunk=by_position[row['chunk_index']]
        original.append({'title':chunk['title'],'doc_id':chunk['doc_id'],
            'chunk_id':chunk['chunk_id'],'text':chunk['text'],'score':row['score'],
            'paper_rank':row['paper_rank'],'chunk_index':row['chunk_index'],
            'token_count':chunk['token_count'],'source_path':chunk['source_path'],
            'source_spans':chunk['source_spans'],'source_role':'Unclassified'})
    return prepare_grounded_evidence(original)


def flags(evidence):
    return {'result_cue':bool(RESULT.search(evidence['text'])),
            'method_cue':any(word in evidence['text'].casefold() for word in
                ('method','design','randomized','retrospective','assay','experiment')),
            'limitation_cue':bool(LIMIT.search(evidence['text'])),
            'background_cue':bool(BACKGROUND.search(evidence['text'])),
            'low_information_candidate':len(evidence['text'].split())<35,
            'semantic_status':'Unverified'}


def pairwise_similarity(evidence,by_cid,embeddings):
    positions=[by_cid[x['chunk_id']]['_position'] for x in evidence]
    vectors=np.asarray(embeddings[positions],dtype=np.float32)
    return {'mean':statistics.mean(float(vectors[i]@vectors[j])
                                   for i,j in ((0,1),(0,2),(1,2))),
            'max':max(float(vectors[i]@vectors[j])
                      for i,j in ((0,1),(0,2),(1,2)))}


def point_leads(points,groups):
    checks=[]
    for point in points:
        vocab=terms(point)
        checks.append({'expected_answer_point':point,
            'groups':{group:{'lexical_review_leads':[
                {'evidence_id':item['evidence_id'],'chunk_id':item['chunk_id'],
                 'shared_terms':sorted(vocab&terms(item['text']))}
                for item in evidence if vocab&terms(item['text'])],
                'semantic_coverage':'Unverified'} for group,evidence in groups.items()},
            'note':'Term overlap only identifies text to review; it is not evidence of semantic coverage.'})
    return checks


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    raw=QUESTIONS.read_bytes()
    lock=json.loads((ROOT/'experiments/configs/phase36_question_lock.json').read_text(encoding='utf8'))
    if hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()!=lock['sha256']:
        raise ValueError('Frozen question set differs')
    questions=json.loads(raw)['questions']
    history={q['question_id']:json.loads((HISTORY/f"{q['question_id']}.json").read_text(encoding='utf8'))
             for q in questions}
    grounded={q['question_id']:json.loads((GROUNDED/f"{q['question_id']}.json").read_text(encoding='utf8'))
              for q in questions}
    needed={d['doc_id'] for old in history.values() for d in old['retrieval_top10']}
    by_doc,scan_seconds=source_rows(needed)
    by_position={row['_position']:row for chunks in by_doc.values() for row in chunks}
    by_cid={row['chunk_id']:row for row in by_position.values()}
    embeddings=np.load(EMBEDDINGS,mmap_mode='r')
    metadata=json.loads(INDEX_META.read_text(encoding='utf8'))
    if embeddings.shape!=(metadata['chunk_count'],metadata['embedding_dimension']):
        raise ValueError('Frozen embedding matrix differs from index metadata')
    snapshot=ROOT/'models/models--BAAI--bge-small-en-v1.5/snapshots'/metadata['model_snapshot']
    model_start=time.perf_counter()
    model=SentenceTransformer(str(snapshot),device='cuda' if torch.cuda.is_available() else 'cpu',
                              local_files_only=True)
    model.eval();model_load_seconds=time.perf_counter()-model_start
    encode_start=time.perf_counter()
    vectors=model.encode([PROMPT+q['question'] for q in questions],batch_size=32,
                         normalize_embeddings=True,show_progress_bar=False,
                         convert_to_numpy=True).astype('float32')
    if torch.cuda.is_available():torch.cuda.synchronize()
    query_encode_seconds=time.perf_counter()-encode_start
    rows=[];details=[];candidate_records=[]
    for q,vector in zip(questions,vectors):
        qid=q['question_id'];old=history[qid];docs=old['retrieval_top10']
        paper_ranks={d['doc_id']:d['rank'] for d in docs}
        positions=[row['_position'] for d in docs for row in by_doc[d['doc_id']]]
        a_start=time.perf_counter()
        a=[{**item,'paper_rank':i,'token_count':by_cid[item['chunk_id']]['token_count'],
            'source_role':'Unclassified'}
           for i,item in enumerate(grounded[qid]['evidence'],1)]
        a_seconds=time.perf_counter()-a_start
        if ([x['chunk_id'] for x in a]!=[x['best_chunk_id'] for x in docs[:3]] or
            [x['doc_id'] for x in a]!=[x['doc_id'] for x in docs[:3]]):
            raise ValueError(f'Baseline differs from original retriever: {qid}')
        b_start=time.perf_counter()
        ranked=rank_chunks(vector,positions,embeddings,by_position,paper_ranks)
        b=evidence_from_ranked(global_top3(ranked),by_position)
        b_seconds=time.perf_counter()-b_start
        c_start=time.perf_counter()
        c=evidence_from_ranked(mmr_top3(ranked,embeddings),by_position)
        c_seconds=time.perf_counter()-c_start
        for group,evidence in (('A',a),('B',b),('C',c)):
            if len(evidence)!=3 or len({x['chunk_id'] for x in evidence})!=3:
                raise ValueError(f'{qid}/{group}: evidence budget or duplicate chunk')
            for item in evidence:
                source=by_cid[item['chunk_id']]
                if any(item[k]!=source[k] for k in ('doc_id','title','text','source_path','source_spans')):
                    raise ValueError(f'{qid}/{group}: source mismatch')
        # Query encoding must reproduce saved best-chunk FAISS scores to numerical tolerance.
        for doc in docs[:3]:
            scored=next(x['score'] for x in ranked if x['chunk_id']==doc['best_chunk_id'])
            if abs(scored-doc['score'])>2e-4:
                raise ValueError(f'{qid}: query/vector scores differ from frozen BGE')
        groups={'A':a,'B':b,'C':c}
        timings={'A':a_seconds,'B':b_seconds,'C':c_seconds}
        row={'question_id':qid,'category':q['category'],'candidate_papers':len(docs),
             'candidate_chunks':len(ranked)}
        for label,evidence in groups.items():
            similarity=pairwise_similarity(evidence,by_cid,embeddings)
            row.update({f'{label}_chunk_ids':'|'.join(x['chunk_id'] for x in evidence),
                        f'{label}_doc_ids':'|'.join(x['doc_id'] for x in evidence),
                        f'{label}_distinct_papers':len({x['doc_id'] for x in evidence}),
                        f'{label}_tokens':sum(by_cid[x['chunk_id']]['token_count'] for x in evidence),
                        f'{label}_selection_seconds':timings[label],
                        f'{label}_mean_pairwise_similarity':similarity['mean'],
                        f'{label}_max_pairwise_similarity':similarity['max'],
                        f'{label}_background_cue_count':sum(flags(x)['background_cue'] for x in evidence),
                        f'{label}_result_cue_count':sum(flags(x)['result_cue'] for x in evidence),
                        f'{label}_low_information_candidates':sum(flags(x)['low_information_candidate'] for x in evidence),
                        f'{label}_semantic_coverage':'Unverified'})
        rows.append(row)
        details.append({'question_id':qid,'question':q['question'],
                        'expected_answer_points':q['expected_answer_points'],
                        'evidence_availability':q['evidence_availability'],
                        'top10_papers':[{ 'rank':d['rank'],'doc_id':d['doc_id'],'title':d['title']}
                                        for d in docs],
                        'evidence':groups,'flags':{g:[flags(x) for x in evidence]
                                                  for g,evidence in groups.items()},
                        'point_checks':point_leads(q['expected_answer_points'],groups),
                        'measurements':row})
        candidate_records.append({'question_id':qid,'candidate_count':len(ranked),
                                  'ranked_chunks':ranked})
    save(OUT/'evidence_comparison.json',{'version':'phase47-evidence-v1',
        'question_set_sha256':lock['sha256'],'questions':details,
        'semantic_review_status':'Unverified unless separately reviewed against original chunks'})
    with (OUT/'candidate_rankings.jsonl').open('w',encoding='utf8',newline='\n') as file:
        for item in candidate_records:file.write(json.dumps(item,ensure_ascii=False)+'\n')
    with (OUT/'per_question_evidence.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(OUT/'performance.json',{'version':'phase47-performance-v1',
        'source_chunk_scan_seconds':scan_seconds,'unique_candidate_papers':len(needed),
        'model_cold_load_seconds':model_load_seconds,
        'new_query_encoding_20_questions_seconds':query_encode_seconds,
        'corpus_embeddings_regenerated':False,'faiss_search_rerun':False,
        'candidate_chunks':{'min':min(r['candidate_chunks'] for r in rows),
                            'mean':statistics.mean(r['candidate_chunks'] for r in rows),
                            'max':max(r['candidate_chunks'] for r in rows)},
        'groups':{g:{'selection_seconds_mean':statistics.mean(r[f'{g}_selection_seconds'] for r in rows),
                     'selection_seconds_total':sum(r[f'{g}_selection_seconds'] for r in rows),
                     'bge_token_count_total':sum(r[f'{g}_tokens'] for r in rows),
                     'distinct_papers_mean':statistics.mean(r[f'{g}_distinct_papers'] for r in rows)}
                  for g in 'ABC'},
        'note':'Selection timings exclude initial source scan, model load and query encoding; B includes scoring all candidate chunks; C uses cached B scores and computes MMR.'})
    print('Offline A/B/C completed:',len(rows),'questions;',len(needed),'candidate papers;',
          'B changed',sum(r['A_chunk_ids']!=r['B_chunk_ids'] for r in rows),
          'C changed',sum(r['A_chunk_ids']!=r['C_chunk_ids'] for r in rows),flush=True)


if __name__=='__main__':main()
