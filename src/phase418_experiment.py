"""Isolated evidence-count experiment over the frozen Large Fixed retriever."""
import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for name, folder in {'HF_HOME': 'huggingface', 'TORCH_HOME': 'torch',
                     'PIP_CACHE_DIR': 'pip'}.items():
    os.environ[name] = str(ROOT / '.cache' / folder)

from src.rag_service import load_retriever, prepare_grounded_evidence
from src.citation_grounding import verify_evidence_source, validate_grounded_answer
from src.deepseek_client import DeepSeekClient, DeepSeekRequestError, is_complete_generation
from src.phase36_run import cost_estimate

OUT = ROOT / 'experiments/results/phase418'
CONFIG = ROOT / 'experiments/configs/phase418/experiment_config.json'
KS = (3, 5, 8)
PILOT = ('Q01', 'Q04', 'Q07', 'Q13', 'Q20')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def select(docs, k):
    raw = [{'citation': f'[{i}]', 'title': d['title'], 'doc_id': d['doc_id'],
            'chunk_id': d['best_chunk_id'], 'text': d['best_chunk_text'],
            'score': d['score'], 'paper_rank': d['rank'],
            'source_path': d['source_path'], 'source_spans': d['source_spans']}
           for i, d in enumerate(docs[:k], 1)]
    return prepare_grounded_evidence(raw)


def offline():
    if (OUT / 'offline_records.json').exists():
        raise FileExistsError('离线结果已存在；不会覆盖。')
    questions = json.loads((ROOT / 'experiments/configs/phase36_questions.json').read_text(encoding='utf-8'))['questions']
    retriever, cold = load_retriever('phase33_large_fixed')
    records, rows = {}, []
    for q in questions:
        qid, question = q['question_id'], q['question']
        vectors, embed_seconds = retriever.encode_queries([question])
        docs, perf = retriever.retrieve_vector(vectors[0], 10)
        source = {d['best_chunk_id']: retriever.chunks[d['chunk_index']] for d in docs}
        groups = {}
        for k in KS:
            evidence = select(docs, k)
            checks = verify_evidence_source(evidence, source)
            groups[str(k)] = {'retrieved_candidates': len(docs),
                              'selected_evidence': len(evidence), 'context_evidence': len(evidence),
                              'evidence': evidence, 'source_checks': checks,
                              'distinct_documents': len({e['doc_id'] for e in evidence}),
                              'distinct_chunks': len({e['chunk_id'] for e in evidence}),
                              'semantic_coverage': 'Unverified'}
            rows.append({'question_id': qid, 'category': q['category'], 'k': k,
                         'retrieved_candidates': len(docs), 'selected_evidence': len(evidence),
                         'context_evidence': len(evidence),
                         'document_ids': json.dumps([e['doc_id'] for e in evidence]),
                         'titles': json.dumps([e['title'] for e in evidence], ensure_ascii=False),
                         'chunk_ids': json.dumps([e['chunk_id'] for e in evidence]),
                         'scores': json.dumps([e['score'] for e in evidence]),
                         'ranks': json.dumps([e['paper_rank'] for e in evidence]),
                         'duplicate_chunks': k - groups[str(k)]['distinct_chunks'],
                         'semantic_coverage': 'Unverified'})
        records[qid] = {'question': question, 'category': q['category'],
                        'expected_answer_points': q['expected_answer_points'],
                        'retrieval': {'query_embedding_seconds': embed_seconds, **perf},
                        'groups': groups}
    checks_ok = all(len(r['groups'][str(k)]['evidence']) == k and
                    all(c['status'] == 'valid' for c in r['groups'][str(k)]['source_checks']) and
                    r['groups'][str(k)]['distinct_documents'] == k and
                    r['groups'][str(k)]['distinct_chunks'] == k and
                    [e['chunk_id'] for e in r['groups'][str(k)]['evidence']] ==
                    [e['chunk_id'] for e in r['groups']['8']['evidence'][:k]]
                    for r in records.values() for k in KS)
    write(OUT / 'offline_records.json', records)
    with (OUT / 'offline_evidence_comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write(OUT / 'offline_gate.json', {'questions': len(records), 'groups': len(rows),
          'mapping_count_and_nesting_passed': checks_ok, 'semantic_coverage': 'Unverified',
          'pilot_api_allowed': checks_ok, 'cold_load_seconds': cold})
    print('offline', len(records), len(rows), 'gate', checks_ok, flush=True)


def api(batch):
    gate = json.loads((OUT / 'offline_gate.json').read_text(encoding='utf-8'))
    if not gate['pilot_api_allowed']:
        raise RuntimeError('离线门槛未通过')
    records = json.loads((OUT / 'offline_records.json').read_text(encoding='utf-8'))
    client = DeepSeekClient()
    targets = PILOT if batch == 'pilot' else tuple(q for q in records if q not in PILOT)
    for qid in targets:
        for k in KS:
            path = OUT / 'raw_answers' / f'{qid}_K{k}.json'
            if path.exists():
                continue
            evidence = records[qid]['groups'][str(k)]['evidence']
            item = {'question_id': qid, 'k': k, 'question': records[qid]['question'],
                    'evidence': evidence, 'status': 'request_started',
                    'started_at_utc': datetime.now(timezone.utc).isoformat()}
            write(path, item)
            parts, meta = [], None
            started = time.perf_counter()
            try:
                for kind, payload in client.stream(item['question'], evidence):
                    if kind == 'text': parts.append(payload)
                    elif kind == 'done': meta = payload
                answer = ''.join(parts)
                item.update(answer=answer, generation=meta,
                            status='complete' if is_complete_generation(answer, meta) else 'incomplete',
                            finish_reason=(meta or {}).get('finish_reason'),
                            api_completion_seconds=time.perf_counter() - started,
                            citation_check=validate_grounded_answer(answer, evidence) if answer else None,
                            estimated_cost=cost_estimate((meta or {}).get('usage'),
                                datetime.fromisoformat(item['started_at_utc'])))
            except DeepSeekRequestError as exc:
                item.update(answer=''.join(parts), status='api_error', error_type=type(exc).__name__,
                            error_message=str(exc), api_completion_seconds=time.perf_counter() - started)
            write(path, item)
            print(qid, k, item['status'], flush=True)
            if item['status'] != 'complete':
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('offline', 'pilot', 'remaining'))
    args = parser.parse_args()
    offline() if args.action == 'offline' else api(args.action)
