"""One-question local CPU smoke check for both unchanged 12,000-paper variants."""
import json
from pathlib import Path
import threading
import time

import psutil
import app  # Include actual Streamlit, chart and page imports in the RSS measurement.

from src.cloud_retriever import CloudRetriever, SharedResources

ROOT = Path(__file__).resolve().parents[1]


def main():
    question = 'What do studies report about COVID-19 symptoms and outcomes in children, and where is evidence limited?'
    process = psutil.Process()
    samples = []
    stop = threading.Event()
    def sample():
        while not stop.wait(0.05):
            samples.append(process.memory_info().rss)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    shared = SharedResources()
    rows = []
    try:
        for run in ('phase33_large_fixed', 'phase33_large_sentence', 'phase33_large_fixed'):
            start = time.perf_counter()
            retriever = CloudRetriever(run, resources=shared)
            docs, performance = retriever.retrieve(question, 10)
            for doc in docs:
                original = retriever.chunks[doc['chunk_index']]
                assert (doc['doc_id'], doc['best_chunk_id'], doc['best_chunk_text']) == (
                    original['doc_id'], original['chunk_id'], original['text'])
            assert retriever.metadata['document_count'] == 12000
            row = {'run_id': run, 'chunks': len(retriever.chunks), 'papers': 12000,
                   'top10_doc_ids': [doc['doc_id'] for doc in docs],
                   'top3_chunk_ids': [doc['best_chunk_id'] for doc in docs[:3]],
                   'seconds': time.perf_counter() - start, 'performance': performance,
                   'rss_bytes': process.memory_info().rss}
            history_path = ROOT / 'experiments/results/phase36/Q07.json'
            if run.endswith('fixed') and history_path.exists():
                history = json.loads(history_path.read_text(encoding='utf-8'))
                assert history['question'] == question
                expected = history['retrieval_top10']
                row['historical_q07_top10_match'] = [doc['doc_id'] for doc in docs] == [doc['doc_id'] for doc in expected]
                row['historical_q07_top3_chunks_match'] = [doc['best_chunk_id'] for doc in docs[:3]] == [doc['best_chunk_id'] for doc in expected[:3]]
                assert row['historical_q07_top10_match'] and row['historical_q07_top3_chunks_match']
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        stop.set()
        monitor.join()
    result = {'platform': 'Windows local CPU with actual app imports, existing CUDA-capable PyTorch wheel; not Community Cloud',
              'real_api_requests': 0, 'rss_peak_bytes': max(samples), 'model_loads': shared.model_loads,
              'index_loads': shared.index_loads, 'runs': rows}
    output = ROOT / 'experiments/results/cloud_deployment'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'cpu_smoke.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Peak RSS: {result["rss_peak_bytes"] / 1024**3:.3f} GiB; model loads: {shared.model_loads}')


if __name__ == '__main__':
    main()
