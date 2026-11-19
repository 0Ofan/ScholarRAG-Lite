"""Verify the anonymous pilot package without calling a judge."""
import hashlib
import json
import unittest
import zipfile
from collections import Counter
from pathlib import Path

from src.phase422_pilot_export import ARCHIVE, EXPORT, FILES, FORBIDDEN, build, sha, verify_export

ROOT = Path(__file__).resolve().parents[1]


class PilotExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((EXPORT / FILES[0]).read_text(encoding='utf-8'))
        cls.samples = cls.data['samples']
        cls.manifest = json.loads((EXPORT / FILES[-1]).read_text(encoding='utf-8'))

    def test_exact_six_anonymous_samples(self):
        self.assertEqual(len(self.samples), 6)
        self.assertEqual(Counter(s['question_id'] for s in self.samples), {'Q07':3, 'Q11':3})
        self.assertEqual({s['anonymous_sample_id'] for s in self.samples},
                         {f'{q}-{label}' for q in ('Q07','Q11') for label in 'ABC'})

    def test_source_answer_evidence_and_hash_integrity(self):
        self.assertEqual(self.samples, build())
        source_hashes = self.manifest['source_file_sha256']
        sources = {'blind_dataset': ROOT / 'experiments/results/phase422/blind_evaluation_dataset.json',
                   'pilot_inputs': ROOT / 'experiments/results/phase422/pilot_evaluation_inputs.json',
                   'question_benchmark': ROOT / 'experiments/results/phase421/question_benchmark.json',
                   'evidence_reference_map': ROOT / 'experiments/results/phase421/evidence_reference_map.json',
                   'judge_prompt': ROOT / 'experiments/configs/phase422/judge_prompt.md',
                   'judge_schema': ROOT / 'experiments/configs/phase422/judge_schema.json'}
        self.assertEqual({key: sha(path) for key, path in sources.items()}, source_hashes)

    def test_evidence_and_reference_fields_complete(self):
        for sample in self.samples:
            ids = {e['evidence_id'] for e in sample['actual_evidence_context']}
            self.assertTrue(sample['expected_answer_points'] and sample['claims_to_evaluate'])
            for e in sample['actual_evidence_context']:
                self.assertEqual(e['citation_id'], f"[{e['evidence_id']}]")
                self.assertTrue(all(e[key] for key in ('document_id','paper_title','chunk_id','original_evidence_text','source_role')))
            for ref in sample['evidence_reference_map']:
                self.assertIn(ref['evidence_id'], ids)
                self.assertTrue(ref['support_status'] and ref['basis'] and ref['original_evidence_text'])
            for claim in sample['claims_to_evaluate']:
                self.assertTrue(set(claim['citation_ids']) <= ids)

    def test_points_unchanged_from_existing_blind_input(self):
        source = {s['sample_id']: s for s in json.loads((ROOT / 'experiments/results/phase422/pilot_evaluation_inputs.json').read_text(encoding='utf-8'))['samples']}
        for sample in self.samples:
            self.assertEqual(sample['expected_answer_points'], source[sample['anonymous_sample_id']]['points'])

    def test_shared_instructions_and_schema(self):
        instructions = (EXPORT / FILES[1]).read_text(encoding='utf-8')
        for word in ('Completeness','Faithfulness','Relevance','Citation Support','Evidence Coverage','Abstention Quality'):
            self.assertIn(word, instructions)
        original = (ROOT / 'experiments/configs/phase422/judge_prompt.md').read_text(encoding='utf-8')
        self.assertTrue(instructions.startswith(original))
        self.assertEqual((EXPORT / FILES[2]).read_bytes(),
                         (ROOT / 'experiments/configs/phase422/judge_schema.json').read_bytes())

    def test_no_group_mapping_or_historical_assessment_leak(self):
        for name in FILES:
            text = (EXPORT / name).read_text(encoding='utf-8')
            for token in FORBIDDEN:
                self.assertNotIn(token, text)
        self.assertNotIn('mapping', [Path(n).stem for n in FILES])
        self.assertNotIn('group', self.data)

    def test_manifest_zip_and_checksums(self):
        result = verify_export(self.samples)
        self.assertTrue(result['zip_extract_and_crc_passed'])
        self.assertEqual(self.manifest['sample_count'], 6)
        for name, digest in self.manifest['exported_file_sha256'].items():
            self.assertEqual(sha(EXPORT / name), digest)
        with zipfile.ZipFile(ARCHIVE) as archive:
            self.assertEqual(sorted(archive.namelist()), sorted(FILES))
            self.assertIsNone(archive.testzip())
            for name in FILES:
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), sha(EXPORT / name))


if __name__ == '__main__':
    unittest.main()
