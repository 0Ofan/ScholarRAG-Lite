"""Focused checks for the frozen Phase 3.3 scale experiments."""
import json
import unittest
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data/processed'
RESULT = ROOT / 'experiments/results/phase33'
INDEX = ROOT / 'data/indexes'


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


class Phase33ScaleTests(unittest.TestCase):
    def test_nested_manifests_and_frozen_positives(self):
        pilot = load(PROC / 'phase31_pilot_manifest.json')
        medium = load(PROC / 'phase33_medium_manifest.json')
        large = load(PROC / 'phase33_large_manifest.json')
        ids = [{d['doc_id'] for d in m['documents']} for m in (pilot, medium, large)]
        self.assertEqual([len(x) for x in ids], [2000, 6000, 12000])
        self.assertTrue(ids[0] <= ids[1] <= ids[2])
        for manifest in (medium, large):
            self.assertEqual(manifest['positive_anchor_count'], 956)
            self.assertEqual(manifest['positive_query_doc_pairs'], 1560)
            self.assertEqual(manifest['positive_per_query'], pilot['positive_per_query'])
            self.assertEqual(manifest['query_ids'], [str(i) for i in range(1, 51)])

    def test_four_artifacts_and_50_query_evaluations(self):
        for scale, size in (('medium', 6000), ('large', 12000)):
            manifest = load(PROC / f'phase33_{scale}_manifest.json')
            corpus = {d['doc_id'] for d in manifest['documents']}
            for method in ('fixed', 'sentence'):
                with self.subTest(scale=scale, method=method):
                    run = f'phase33_{scale}_{method}'
                    out = RESULT / run
                    chunks = load(out / 'chunking_stats.json')
                    embedding = load(out / 'embedding_metrics.json')
                    index_stats = load(out / 'index_metrics.json')
                    metrics = load(out / 'metrics.json')
                    ids = load(PROC / f'{run}_chunk_ids.json')
                    vectors = np.load(PROC / f'{run}_embeddings.npy', mmap_mode='r')
                    self.assertEqual(chunks['document_count'], size)
                    self.assertEqual(len(ids), len(set(ids)))
                    self.assertEqual(len(ids), chunks['total_chunks'])
                    self.assertEqual(vectors.shape, (len(ids), 384))
                    self.assertEqual(index_stats['chunk_count'], len(ids))
                    self.assertTrue(index_stats['index_reload_verified'])
                    index = faiss.read_index(str(INDEX / run / 'faiss.index'))
                    self.assertEqual(index.ntotal, len(ids))
                    self.assertTrue(np.allclose(index.reconstruct(0), vectors[0], atol=1e-6))
                    self.assertEqual(metrics['query_count'], 50)
                    self.assertEqual(len(metrics['per_query']), 50)
                    rows = [json.loads(line) for line in (out / 'retrieval_results.jsonl').read_text(encoding='utf8').splitlines()]
                    self.assertEqual(len(rows), 50)
                    for row in rows:
                        docs = row['top100']
                        self.assertEqual(len(docs), 100)
                        self.assertEqual(len({d['doc_id'] for d in docs}), 100)
                        self.assertTrue({d['doc_id'] for d in docs} <= corpus)
                    del index, vectors


if __name__ == '__main__':
    unittest.main()
