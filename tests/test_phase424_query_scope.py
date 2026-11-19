"""Focused tests for the conservative scope checks and frozen review decisions."""

import json
import hashlib
import unittest
from pathlib import Path

from src.phase424.pilot_scope_refinement import prepare_revisions
from src.phase424.query_scope import validate_query_scope
from src.phase424.query_decomposition import SYSTEM


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/results/phase424/query_decomposition"
OUT = ROOT / "experiments/results/phase424/query_scope_refinement"


class QueryScopeTests(unittest.TestCase):
    def test_prompt_explicitly_limits_expansion_and_repetition(self):
        for phrase in ("explicitly requested", "long-term results", "mechanisms", "mediators",
                       "population", "timeframe", "not near-duplicates"):
            self.assertIn(phrase, SYSTEM)

    def test_historical_six_question_flags_are_hints(self):
        rows = json.loads((SOURCE / "generated_subqueries.json").read_text(encoding="utf8"))
        observed = {}
        for row in rows:
            result = validate_query_scope(row["original_query"], row["subqueries"])
            observed[row["question_id"]] = result
            self.assertFalse(result["semantic_scope_confirmed"])
        self.assertEqual({qid for qid, x in observed.items() if x["flagged_subqueries"]},
                         {"Q07", "Q09", "Q13"})
        self.assertIn("new_timeframe", observed["Q07"]["reasons"])
        self.assertIn("new_mechanism", observed["Q09"]["reasons"])
        self.assertIn("possible_redundancy", observed["Q13"]["reasons"])
        for qid in ("Q07", "Q09", "Q13"):
            self.assertEqual(observed[qid]["flagged_subqueries"][0]["index_one_based"], 4)
            self.assertEqual(observed[qid]["scope_status"], "needs_review")

    def test_population_outcome_and_unrelated_goal_are_only_risks(self):
        cases = [
            ("COVID-19 outcomes in adults", ["pediatric COVID-19 outcomes"], "new_population"),
            ("COVID-19 severity in adults", ["COVID-19 myocarditis in adults"], "new_outcome"),
            ("face mask effects on respiratory transmission", ["spike protein crystal structure"], "weak_anchor_overlap"),
        ]
        for original, subs, risk in cases:
            with self.subTest(risk=risk):
                result = validate_query_scope(original, subs)
                self.assertIn(risk, result["reasons"])
                self.assertEqual(result["flagged_subqueries"][0]["assessment"],
                                 "rule_hint_needs_semantic_review")

    def test_exact_duplicate_and_blank_are_structural_issues(self):
        result = validate_query_scope("PCR diagnosis", ["PCR diagnosis", "  "])
        self.assertIn("exact_duplicate", result["reasons"])
        self.assertIn("empty_subquery", result["reasons"])
        self.assertEqual(result["scope_status"], "needs_review")

    def test_decisions_preserve_original_and_retained_text(self):
        before = json.loads((SOURCE / "generated_subqueries.json").read_text(encoding="utf8"))
        decisions = json.loads((OUT / "scope_decisions.json").read_text(encoding="utf8"))
        after = prepare_revisions(before, decisions)
        self.assertEqual(len(after), 6)
        counts = {row["question_id"]: len(row["after_subqueries"]) for row in after}
        self.assertEqual(counts, {"Q04": 4, "Q07": 3, "Q09": 3, "Q11": 3, "Q13": 3, "Q16": 4})
        for old, new in zip(before, after):
            self.assertEqual(old["original_query"], new["original_query"])
            self.assertEqual(new["before_subqueries"], old["subqueries"])
            self.assertTrue(set(new["after_subqueries"]) <= set(old["subqueries"]))
            self.assertEqual(new["after_scope_check"]["flagged_subqueries"], [])

    def test_saved_replay_is_complete_and_historical_inputs_unchanged(self):
        old = json.loads((SOURCE / "retrieval_comparison.json").read_text(encoding="utf8"))["questions"]
        new = json.loads((OUT / "retrieval_comparison.json").read_text(encoding="utf8"))["questions"]
        reruns = json.loads((OUT / "revised_retrieval.json").read_text(encoding="utf8"))
        self.assertEqual(len(new), len(reruns), 6)
        for before, after, replay in zip(old, new, reruns):
            self.assertEqual(before["question_id"], after["question_id"])
            self.assertTrue(after["baseline_unchanged"])
            self.assertEqual(after["before_top10_doc_ids"], before["decomposed_top10_doc_ids"])
            self.assertEqual(after["after_top10_doc_ids"],
                             [paper["doc_id"] for paper in replay["merged_top10"]])
            self.assertEqual(len(replay["merged_top100"]), 100)
            for paper in replay["merged_top100"]:
                self.assertEqual(paper["doc_id"], paper["best_chunk"]["doc_id"])
                self.assertIn(paper["best_chunk"]["chunk_id"],
                              {item["chunk"]["chunk_id"] for item in paper["candidate_chunks"]})
        old_hashes = {item["path"]: item["sha256"] for item in json.loads(
            (SOURCE / "input_hashes.json").read_text(encoding="utf8"))["files"]}
        new_hashes = {item["path"]: item["sha256"] for item in json.loads(
            (OUT / "input_hashes.json").read_text(encoding="utf8"))["files"]}
        for path, digest in old_hashes.items():
            self.assertEqual(new_hashes[path], digest)
        for path in (SOURCE / "generated_subqueries.json", OUT / "scope_decisions.json"):
            self.assertEqual(new_hashes[str(path.relative_to(ROOT)).replace("\\", "/")],
                             hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
