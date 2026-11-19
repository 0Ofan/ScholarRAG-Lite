"""Evidence selection and citation validation around the frozen retriever."""
import re
import os
import time

from .retriever import PilotRetriever

VARIANTS = {'Fixed-length (Large)': 'phase33_large_fixed',
            'Sentence-aware (Large)': 'phase33_large_sentence'}


def load_retriever(run_id):
    if run_id not in VARIANTS.values():
        raise ValueError('未知检索方法')
    start = time.perf_counter()
    if os.getenv('SCHOLARRAG_CLOUD') == '1':
        from .cloud_retriever import CloudRetriever
        retriever = CloudRetriever(run_id=run_id)
    else:
        retriever = PilotRetriever(run_id=run_id)
    return retriever, time.perf_counter() - start


def retrieve_evidence(retriever, question, count=3):
    if not 1 <= count <= 5:
        raise ValueError('证据数量必须为 1–5')
    submitted = time.perf_counter()
    vectors, embed_seconds = retriever.encode_queries([question])
    docs, retrieval_perf = retriever.retrieve_vector(vectors[0], max(10, count))
    evidence_start = time.perf_counter()
    evidence = []
    for i, doc in enumerate(docs[:count], 1):
        evidence.append({
            'citation': f'[{i}]',
            'title': doc['title'],
            'doc_id': doc['doc_id'],
            'chunk_id': doc['best_chunk_id'],
            'text': doc['best_chunk_text'],
            'score': doc['score'],
            'source_path': doc['source_path'],
            'source_spans': doc['source_spans'],
        })
    evidence_seconds = time.perf_counter() - evidence_start
    return evidence, docs, {
        'query_embedding_seconds': embed_seconds,
        'retrieval_seconds': retrieval_perf['retrieval_seconds'],
        'evidence_seconds': evidence_seconds,
        'retrieval_total_seconds': time.perf_counter() - submitted,
    }


def validate_citations(answer, evidence):
    valid = {item['citation'] for item in evidence}
    used = set(re.findall(r'\[\d+\]', answer))
    return {'valid': sorted(used & valid), 'invalid': sorted(used - valid),
            'uncited': sorted(valid - used)}


def prepare_grounded_evidence(evidence):
    """Assign new evidence IDs without changing frozen retrieval text or source order."""
    prepared = []
    for position, item in enumerate(evidence, 1):
        for field in ('doc_id', 'title', 'chunk_id', 'text'):
            if not item.get(field):
                raise ValueError(f'证据缺少 {field}')
        row = dict(item)
        row['evidence_id'] = f'E{position}'
        row['citation'] = f'[E{position}]'
        prepared.append(row)
    if len({item['chunk_id'] for item in prepared}) != len(prepared):
        raise ValueError('同一证据集合出现重复 Chunk ID')
    return prepared
