"""Consistency checks for the frozen Phase 2.6 audit outputs."""
import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/'data/evaluation'/name).read_text(encoding='utf8'))

class Phase26AuditTests(unittest.TestCase):
    def test_positive_class_counts(self):
        audit=read('phase26_metadata_audit.json')
        self.assertEqual(sum(audit['classes_mutually_exclusive'].values()),audit['positive_unique'])
        self.assertEqual(audit['parse_path_unique_docs'],12066)
    def test_archive_paths_and_checksum(self):
        archive=read('phase26_archive_summary.json')
        self.assertEqual(archive['missing_target_paths'],0)
        self.assertEqual(archive['target_paths'],archive['found_target_paths'])
        self.assertEqual(archive['read_or_parse_errors'],0)
        self.assertTrue(read('phase26_gzip_integrity.json')['gzip_crc_verified'])
    def test_manifest_and_coverage(self):
        coverage=read('phase26_coverage.json')
        manifest=read('phase26_candidate_manifest.json')
        aliases=set(manifest['excluded_ambiguous_doc_ids'])
        ids={d['doc_id'] for d in manifest['documents']}
        self.assertEqual(len(coverage['per_query']),50)
        self.assertEqual(len(manifest['queries']),50)
        self.assertTrue(ids.isdisjoint(aliases))
        self.assertEqual(coverage['positive_valid_body'],11984)
        self.assertTrue(all(q['available_positive']>0 for q in manifest['queries']))
    def test_shared_path_preserves_both_ids(self):
        paths=read('phase26_judged_path_index.json')
        key='document_parses/pdf_json/ad2b326221b48e904660968e5d10c4cb2882c5e0.json'
        self.assertEqual(set(paths[key]),{'g7qioapl','xa6kwguo'})

if __name__=='__main__':unittest.main()
