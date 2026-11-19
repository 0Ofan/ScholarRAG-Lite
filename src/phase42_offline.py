"""One-pass offline comparison using frozen Top-10 records and existing chunks."""
import csv
import hashlib
import json
import time
from pathlib import Path

from src.evidence_selector import BACKGROUND, CHUNKS, LIMIT, RESULT, baseline, optimized, terms

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results/phase42'
QUESTIONS = ROOT/'experiments/configs/phase36_questions.json'
HISTORY = ROOT/'experiments/results/phase36'


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def load_inputs():
    raw = QUESTIONS.read_bytes()
    lock = json.loads((ROOT/'experiments/configs/phase36_question_lock.json').read_text(encoding='utf8'))
    if hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() != lock['sha256']:
        raise ValueError('Frozen questions changed')
    questions = json.loads(raw)['questions']
    history = {q['question_id']:json.loads((HISTORY/f"{q['question_id']}.json").read_text(encoding='utf8'))
               for q in questions}
    if len(questions) != 20 or any(len(history[q['question_id']]['retrieval_top10']) != 10 for q in questions):
        raise ValueError('Expected 20 questions with frozen Top-10')
    return questions, history


def collect_candidates(history):
    ids = {doc['doc_id'] for record in history.values() for doc in record['retrieval_top10']}
    candidates = {uid:[] for uid in ids}
    start = time.perf_counter()
    with CHUNKS.open(encoding='utf8') as file:
        for line in file:
            uid = line[14:].split(':', 1)[0]
            if uid in candidates:
                candidates[uid].append(json.loads(line))
    if any(not rows for rows in candidates.values()):
        raise ValueError('A frozen Top-10 paper has no source chunks')
    return candidates, time.perf_counter()-start


def item_flags(item):
    text = item['text']
    return {'result_cue': bool(RESULT.search(text)), 'limitation_cue': bool(LIMIT.search(text)),
            'background_cue': bool(BACKGROUND.search(text)),
            'low_information_candidate': len(text.split()) < 35,
            'semantic_status': 'Unverified'}


def point_checks(points, baseline_evidence, optimized_evidence):
    """Trace lexical leads to source chunks; do not infer semantic coverage."""
    checks=[]
    for point in points:
        vocab=terms(point)
        def leads(evidence):
            return [{'chunk_id':item['chunk_id'], 'doc_id':item['doc_id'],
                     'shared_term_stems':sorted(vocab & terms(item['text']))}
                    for item in evidence if vocab & terms(item['text'])]
        checks.append({'expected_answer_point':point, 'baseline_lexical_leads':leads(baseline_evidence),
                       'optimized_lexical_leads':leads(optimized_evidence),
                       'baseline_semantic_coverage':'Unverified',
                       'optimized_semantic_coverage':'Unverified',
                       'reason':'Term overlap is a review lead, not proof that the original text supports the point.'})
    return checks


def main():
    questions, history = load_inputs()
    candidates, scan_seconds = collect_candidates(history)
    by_cid = {row['chunk_id']:row for rows in candidates.values() for row in rows}
    comparison = []
    csv_rows = []
    for question in questions:
        qid = question['question_id']
        old = history[qid]
        docs = old['retrieval_top10']
        start = time.perf_counter()
        a = baseline(docs, question)
        baseline_seconds = time.perf_counter()-start
        if any(x['chunk_id'] != y['chunk_id'] or x['text'] != y['text'] or x['doc_id'] != y['doc_id']
               for x,y in zip(a, old['evidence'])):
            raise ValueError(f'{qid}: baseline differs from frozen Phase 3.6')
        start = time.perf_counter()
        b = optimized(docs, question, candidates)
        optimized_seconds = time.perf_counter()-start
        if [x['doc_id'] for x in a] != [x['doc_id'] for x in b]:
            raise ValueError(f'{qid}: paper order changed')
        if len({x['chunk_id'] for x in b}) != 3:
            raise ValueError(f'{qid}: repeated optimized chunk')
        for item in b:
            source = by_cid[item['chunk_id']]
            if any(item[k] != source[k] for k in ('doc_id','title','text','source_path','source_spans')):
                raise ValueError(f'{qid}: altered source chunk')
        changed = [i+1 for i,(x,y) in enumerate(zip(a,b)) if x['chunk_id'] != y['chunk_id']]
        row = {'question_id':qid,'category':question['category'],
               'evidence_availability':question['evidence_availability'],
               'candidate_papers':len(docs), 'selected_papers':3,
               'baseline_chunk_ids':'|'.join(x['chunk_id'] for x in a),
               'optimized_chunk_ids':'|'.join(x['chunk_id'] for x in b),
               'changed_positions':'|'.join(map(str,changed)),
               'baseline_primary_count':sum(x['source_role']=='Primary Evidence' for x in a),
               'optimized_primary_count':sum(x['source_role']=='Primary Evidence' for x in b),
               'baseline_result_cue_count':sum(item_flags(x)['result_cue'] for x in a),
               'optimized_result_cue_count':sum(item_flags(x)['result_cue'] for x in b),
               'baseline_background_cue_count':sum(item_flags(x)['background_cue'] for x in a),
               'optimized_background_cue_count':sum(item_flags(x)['background_cue'] for x in b),
               'baseline_low_information_candidates':sum(item_flags(x)['low_information_candidate'] for x in a),
               'optimized_low_information_candidates':sum(item_flags(x)['low_information_candidate'] for x in b),
               'baseline_duplicate_chunks':3-len({x['chunk_id'] for x in a}),
               'optimized_duplicate_chunks':3-len({x['chunk_id'] for x in b}),
               'baseline_bge_tokens':sum(by_cid[x['chunk_id']]['token_count'] for x in a),
               'optimized_bge_tokens':sum(by_cid[x['chunk_id']]['token_count'] for x in b),
               'baseline_selection_seconds':baseline_seconds,
               'optimized_selection_seconds':optimized_seconds,
               'expected_point_coverage': 'Unverified',
               'research_population_conditions': 'Unverified',
               'cross_paper_attribution': 'Unverified'}
        csv_rows.append(row)
        comparison.append({'question_id':qid,'question':question['question'],
                           'expected_answer_points':question['expected_answer_points'],
                           'point_checks':point_checks(question['expected_answer_points'], a, b),
                           'top10_doc_ids':[d['doc_id'] for d in docs],
                           'baseline':a,'optimized':b,'baseline_flags':[item_flags(x) for x in a],
                           'optimized_flags':[item_flags(x) for x in b],
                           'measurements':row})
    save(OUT/'evidence_comparison.json', {'version':'phase42-offline-v1',
         'question_set_sha256':hashlib.sha256(QUESTIONS.read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
         'source_chunk_file':'data/processed/phase33_large_fixed_chunks.jsonl',
         'candidate_paper_count':len(candidates),'candidate_chunk_scan_seconds':scan_seconds,
         'comparison':comparison})
    with (OUT/'per_question_evidence.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(csv_rows[0]))
        writer.writeheader();writer.writerows(csv_rows)
    save(OUT/'coverage_stats.json', {'version':'phase42-offline-stats-v1','question_count':len(csv_rows),
         'changed_questions':sum(bool(r['changed_positions']) for r in csv_rows),
         'changed_chunks':sum(len(r['changed_positions'].split('|')) for r in csv_rows if r['changed_positions']),
         'result_cue_baseline':sum(r['baseline_result_cue_count'] for r in csv_rows),
         'result_cue_optimized':sum(r['optimized_result_cue_count'] for r in csv_rows),
         'background_cue_baseline':sum(r['baseline_background_cue_count'] for r in csv_rows),
         'background_cue_optimized':sum(r['optimized_background_cue_count'] for r in csv_rows),
         'low_information_baseline':sum(r['baseline_low_information_candidates'] for r in csv_rows),
         'low_information_optimized':sum(r['optimized_low_information_candidates'] for r in csv_rows),
         'bge_tokens_baseline':sum(r['baseline_bge_tokens'] for r in csv_rows),
         'bge_tokens_optimized':sum(r['optimized_bge_tokens'] for r in csv_rows),
         'baseline_selection_seconds_mean':sum(r['baseline_selection_seconds'] for r in csv_rows)/len(csv_rows),
         'optimized_selection_seconds_mean':sum(r['optimized_selection_seconds'] for r in csv_rows)/len(csv_rows),
         'semantic_expected_point_coverage':'Unverified',
         'gate':'pending_traceable_source_review',
         'warning':'Cue counts and BGE token counts do not establish answer-point coverage or LLM input tokens.'})
    print('offline complete:',len(csv_rows),'questions;',sum(bool(r['changed_positions']) for r in csv_rows),'changed')


if __name__=='__main__':
    main()
