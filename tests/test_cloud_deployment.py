import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

import faiss
import numpy as np

from src.cloud_retriever import CloudRetriever, DiskChunks, SharedResources
from src.retriever import PilotRetriever
from src.runtime_download import ensure_runtime, extract_verified


ROOT = Path(__file__).resolve().parents[1]


class CloudRetrievalTests(unittest.TestCase):
    def setUp(self):
        cache = ROOT / '.cache/cloud-tests'
        cache.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.root = Path(self.temp.name)
        self.rows = [{'doc_id': f'd{i // 2}', 'chunk_id': f'c{i}', 'original_uid': f'u{i // 2}',
                      'title': f'Paper {i // 2}', 'doi': '10/test', 'text': f'原文 {i}',
                      'section_title': 'Results', 'source_path': 'original.json',
                      'source_spans': [{'start': i}]} for i in range(12)]
        self.vectors = np.random.default_rng(42).normal(size=(12, 384)).astype('float32')
        faiss.normalize_L2(self.vectors)
        self.index = faiss.IndexFlatIP(384)
        self.index.add(self.vectors)
        for run in ('phase33_large_fixed', 'phase33_large_sentence'):
            path = self.root / 'data/indexes' / run
            path.mkdir(parents=True)
            faiss.write_index(self.index, str(path / 'faiss.index'))
            (path / 'index_metadata.json').write_text(json.dumps({'chunk_count': 12, 'model_snapshot': 'test'}))
            processed = self.root / 'data/processed'
            processed.mkdir(exist_ok=True)
            (processed / f'{run}_chunks.jsonl').write_text(''.join(
                json.dumps(row, ensure_ascii=False) + '\n' for row in self.rows), encoding='utf-8')
        self.resources = SharedResources()

    def tearDown(self):
        self.temp.cleanup()

    def test_disk_chunks_preserve_unicode_fields_and_order(self):
        chunks = DiskChunks(self.root / 'data/processed/phase33_large_fixed_chunks.jsonl')
        self.assertEqual(list(chunks), self.rows)
        self.assertEqual(chunks[-1], self.rows[-1])
        self.assertEqual(chunks[1:4], self.rows[1:4])
        with self.assertRaises(IndexError):
            _ = chunks[12]

    def test_frozen_search_order_scores_and_source_mapping_match(self):
        original = PilotRetriever.__new__(PilotRetriever)
        original.index, original.chunks = self.index, self.rows
        cloud = CloudRetriever('phase33_large_fixed', self.root, self.resources)
        expected, _ = original.retrieve_vector(self.vectors[4], 5)
        actual, _ = cloud.retrieve_vector(self.vectors[4], 5)
        self.assertEqual(actual, expected)
        for row in actual:
            self.assertEqual(cloud.chunks[row['chunk_index']]['chunk_id'], row['best_chunk_id'])
            self.assertEqual(cloud.chunks[row['chunk_index']]['text'], row['best_chunk_text'])

    def test_single_index_cache_and_reproducible_method_switch(self):
        a = CloudRetriever('phase33_large_fixed', self.root, self.resources)
        b = CloudRetriever('phase33_large_sentence', self.root, self.resources)
        first, _ = a.retrieve_vector(self.vectors[2], 3)
        a.retrieve_vector(self.vectors[2], 3)
        self.assertEqual(self.resources.index_loads, 1)
        b.retrieve_vector(self.vectors[2], 3)
        again, _ = a.retrieve_vector(self.vectors[2], 3)
        self.assertEqual(first, again)
        self.assertEqual(self.resources.index_loads, 3)

    @patch('src.cloud_retriever.SentenceTransformer')
    def test_cpu_model_loaded_once_and_batch_encoding_preserved(self, factory):
        factory.return_value.encode.return_value = np.ones((2, 384), dtype='float32')
        a = CloudRetriever('phase33_large_fixed', self.root, self.resources)
        b = CloudRetriever('phase33_large_sentence', self.root, self.resources)
        a.encode_queries(['first', 'second'])
        b.encode_queries(['first', 'second'])
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(factory.call_args.kwargs['device'], 'cpu')
        self.assertEqual(self.resources.model_loads, 1)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        cache = ROOT / '.cache/cloud-tests'
        cache.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.root = Path(self.temp.name)
        self.contents = b'unchanged original evidence'
        self.name = 'data/processed/chunks.jsonl'
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            item = tarfile.TarInfo(self.name)
            item.size = len(self.contents)
            archive.addfile(item, io.BytesIO(self.contents))
        self.archive = stream.getvalue()
        digest = lambda value: hashlib.sha256(value).hexdigest()
        self.manifest = {'archive_sha256': digest(self.archive), 'base_url': 'https://example.test',
                         'parts': [{'name': 'part01', 'bytes': len(self.archive), 'sha256': digest(self.archive)}],
                         'files': [{'path': self.name, 'bytes': len(self.contents), 'sha256': digest(self.contents)}]}
        (self.root / 'deployment').mkdir()
        (self.root / 'deployment/cloud_manifest.json').write_text(json.dumps(self.manifest))

    def tearDown(self):
        self.temp.cleanup()

    @patch('src.runtime_download.urlopen')
    def test_download_verify_extract_and_warm_start_without_network(self, open_url):
        open_url.return_value = io.BytesIO(self.archive)
        ensure_runtime(self.root)
        self.assertEqual((self.root / self.name).read_bytes(), self.contents)
        ensure_runtime(self.root)
        self.assertEqual(open_url.call_count, 1)

    @patch('src.runtime_download.urlopen')
    def test_bad_download_never_used(self, open_url):
        open_url.return_value = io.BytesIO(b'corrupted download')
        with self.assertRaises(ValueError):
            ensure_runtime(self.root)
        self.assertFalse((self.root / self.name).exists())
        self.assertFalse((self.root / '.cache/cloud-runtime/ready.json').exists())

    def test_existing_different_file_is_not_overwritten(self):
        path = self.root / self.name
        path.parent.mkdir(parents=True)
        path.write_bytes(b'user-owned content')
        archive = self.root / 'test.tar.gz'
        archive.write_bytes(self.archive)
        with self.assertRaises(ValueError):
            extract_verified(archive, self.root, self.manifest['files'])
        self.assertEqual(path.read_bytes(), b'user-owned content')

    def test_path_traversal_rejected(self):
        archive = self.root / 'bad.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            item = tarfile.TarInfo('../escaped')
            item.size = 1
            tar.addfile(item, io.BytesIO(b'x'))
        with self.assertRaises(ValueError):
            extract_verified(archive, self.root, [{'path': '../escaped', 'bytes': 1, 'sha256': ''}])


if __name__ == '__main__':
    unittest.main()
