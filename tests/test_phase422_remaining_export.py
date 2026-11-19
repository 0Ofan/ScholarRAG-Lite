"""Incremental validation of the 54-answer anonymous export."""
import csv
import json
import unittest
import zipfile
from collections import Counter

from src.phase422_remaining_export import (ARCHIVE, BATCHES, EXPORT, FILES,
                                            PENDING, build, load, sha, validate_export)


class RemainingExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = load(EXPORT / 'remaining_54_answers.json')['samples']

    def test_exact_54_and_fixed_batches(self):
        self.assertEqual(len(self.samples), 54)
        expected = {qid for pair in BATCHES for qid in pair}
        self.assertEqual(Counter(s['question_id'] for s in self.samples), dict.fromkeys(expected, 3))
        self.assertFalse({'Q07','Q11'} & expected)
        for number, pair in enumerate(BATCHES, 1):
            batch = load(EXPORT / f'batch_{number:02d}.json')
            self.assertEqual(batch['question_ids'], list(pair))
            self.assertEqual(len(batch['samples']), 6)
            self.assertEqual(Counter(s['question_id'] for s in batch['samples']), dict.fromkeys(pair, 3))

    def test_answers_evidence_and_hashes_match_existing_sources(self):
        self.assertEqual(self.samples, build())
        self.assertTrue(load(EXPORT / 'export_manifest.json')['original_answer_and_evidence_verified'])

    def test_ids_claims_and_citation_offsets(self):
        for sample in self.samples:
            qid = sample['question_id']
            self.assertEqual(sample['anonymous_sample_id'][:3], qid)
            ids = {e['evidence_id'] for e in sample['actual_evidence_context']}
            self.assertTrue(sample['expected_answer_points'] and sample['claims_to_evaluate'])
            for claim in sample['claims_to_evaluate']:
                self.assertTrue(set(claim['citation_ids']) <= ids)
            for citation in sample['original_citation_positions']:
                self.assertIn(citation['evidence_id'], ids)
                self.assertEqual(sample['original_answer'][citation['char_start']:citation['char_end_exclusive']], citation['citation_id'])
            for ref in sample['evidence_reference_map']:
                self.assertIn(ref['evidence_id'], ids)
                self.assertTrue(ref['support_status'] and ref['basis'])

    def test_unchanged_rubric_schema_and_blank_templates(self):
        from src.phase422_remaining_export import RESULTS
        self.assertEqual((EXPORT / 'evaluation_instructions.md').read_bytes(),
                         (RESULTS / 'pilot_export/pilot_evaluation_instructions.md').read_bytes())
        self.assertEqual((EXPORT / 'evaluation_schema.json').read_bytes(),
                         (RESULTS / 'pilot_export/pilot_json_schema.json').read_bytes())
        self.assertEqual(load(EXPORT / 'answer_evaluation_template.json')['status'], 'template_only')
        for name, value in (('claim_evaluation_template.csv', 'support'), ('point_evaluation_template.csv', 'coverage')):
            with (EXPORT / name).open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertTrue(all(not r[value] for r in rows))

    def test_anonymity_and_pilot_pending(self):
        from src.phase422_pilot_export import FORBIDDEN
        for name in FILES:
            data = (EXPORT / name).read_text(encoding='utf-8')
            self.assertTrue(all(token not in data for token in FORBIDDEN))
        pending = load(PENDING)
        self.assertEqual(pending['scores_imported'], 0)
        self.assertIn('pending', pending['status'])

    def test_manifest_zip_integrity(self):
        result = validate_export(self.samples)
        self.assertTrue(result['hashes_passed'] and result['zip_passed'])
        manifest = load(EXPORT / 'export_manifest.json')
        self.assertEqual(manifest['sample_count'], 54)
        for name, digest in manifest['exported_file_sha256'].items():
            self.assertEqual(sha(EXPORT / name), digest)
        with zipfile.ZipFile(ARCHIVE) as archive:
            self.assertEqual(sorted(archive.namelist()), sorted(FILES))
            self.assertIsNone(archive.testzip())


if __name__ == '__main__':
    unittest.main()
