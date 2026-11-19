import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from deployment.package_runtime import ROOT, RUNS, build, runtime_files, verify


class RuntimePackageTests(unittest.TestCase):
    def setUp(self):
        cache = ROOT / '.cache/deployment-tests'
        cache.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.root = Path(self.temp.name)
        for run in RUNS:
            folder = self.root / 'data/indexes' / run
            folder.mkdir(parents=True)
            (folder / 'index_metadata.json').write_text(json.dumps({'model_snapshot': 'test'}))
            (folder / 'faiss.index').write_bytes(b'test-index')
            (folder / 'chunk_ids.json').write_text('["chunk"]')
            chunks = self.root / 'data/processed' / f'{run}_chunks.jsonl'
            chunks.parent.mkdir(exist_ok=True)
            chunks.write_text('{"text":"original evidence"}\n')
        self.model = self.root / 'models/models--BAAI--bge-small-en-v1.5/snapshots/test/model.safetensors'
        self.model.parent.mkdir(parents=True)
        self.model.write_bytes(b'test-weights')
        (self.root / '.env').write_text('DO_NOT_EXPORT=secret-marker')

    def tearDown(self):
        self.temp.cleanup()

    def test_allowlist_keeps_model_and_both_variants_without_secrets(self):
        files = runtime_files(self.root)
        self.assertEqual(len(files), 9)
        self.assertIn(self.model, files)
        self.assertNotIn(self.root / '.env', files)

    def test_export_preserves_bytes_and_can_be_verified(self):
        out = self.root / 'export'
        build(self.root, out)
        with tarfile.open(out / 'scholarrag_runtime.tar.gz') as archive:
            self.assertEqual(len(archive.getmembers()), 9)
            for member in archive.getmembers():
                self.assertTrue(member.isfile())
                self.assertEqual(archive.extractfile(member).read(), (self.root / member.name).read_bytes())
        verify(self.root, out / 'runtime_manifest.json')
        self.model.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            verify(self.root, out / 'runtime_manifest.json')

    def test_export_does_not_overwrite(self):
        out = self.root / 'export'
        build(self.root, out)
        with self.assertRaises(FileExistsError):
            build(self.root, out)

    def test_missing_weights_fails(self):
        self.model.unlink()
        with self.assertRaises(FileNotFoundError):
            runtime_files(self.root)


if __name__ == '__main__':
    unittest.main()
