"""Incremental tests for the independent Phase 4.24.2 pilot path."""

import json
import unittest
from types import SimpleNamespace
from pathlib import Path

from src.phase424.multi_query_retrieval import retrieve_branches
from src.phase424.query_decomposition import decompose, parse_subqueries
from src.phase424.result_merger import merge_branches


def obj(**values):
    return SimpleNamespace(**values)


class FakeCompletion:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeRetriever:
    def __init__(self):
        self.queries = None
        self.chunks = [{"chunk_id": "A:1", "doc_id": "A", "section_kind": "body"},
                       {"chunk_id": "B:1", "doc_id": "B", "section_kind": "abstract"}]

    def encode_queries(self, queries):
        self.queries = queries
        return range(len(queries)), 0.01

    def retrieve_vector(self, vector, depth):
        assert depth == 2
        docs = []
        for rank, index in enumerate(([0, 1] if vector == 0 else [1, 0]), 1):
            chunk = self.chunks[index]
            docs.append({"doc_id": chunk["doc_id"], "original_uid": chunk["doc_id"],
                         "best_chunk_id": chunk["chunk_id"], "best_chunk_text": f"Original {chunk['doc_id']}",
                         "chunk_index": index, "title": chunk["doc_id"], "doi": "", "score": 1 / rank,
                         "rank": rank, "source_path": "paper.json", "source_spans": [], "section_title": "Results"})
        return docs, {"retrieval_seconds": 0.02}


class QueryDecompositionTests(unittest.TestCase):
    def test_parse_rejects_bad_json_empty_duplicate_and_bounds(self):
        for content in ('x', '{}', '{"subqueries": ["one"]}',
                        '{"subqueries": ["one", "", "two"]}',
                        '{"subqueries": ["one", "ONE"]}',
                        '{"subqueries": ["original", "other"]}',
                        json.dumps({"subqueries": ["a", "b", "c", "d", "e"]})):
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_subqueries(content, "original")
        self.assertEqual(parse_subqueries('{"subqueries": [" first ", "second"]}', "original"),
                         ["first", "second"])

    def test_real_client_shape_and_usage_without_calling_api(self):
        response = obj(choices=[obj(finish_reason="stop", message=obj(content='{"subqueries": ["a", "b"]}'))],
                       usage=obj(prompt_tokens=80, completion_tokens=20,
                                 prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=80))
        completion = FakeCompletion(response)
        client = obj(config={"model": "deepseek-flash", "temperature": 0.3},
                     client=obj(chat=obj(completions=completion)))
        result = decompose(" Original question? ", client)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["original_query"], "Original question?")
        self.assertEqual(result["subqueries"], ["a", "b"])
        self.assertEqual(result["usage"]["input_tokens"], 80)
        self.assertEqual(completion.calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(completion.calls[0]["extra_body"], {"thinking": {"type": "disabled"}})

    def test_incomplete_and_api_failure_are_not_success(self):
        for finish, content in (("length", '{"subqueries": ["a", "b"]}'),
                                ("stop", " "), ("stop", "not JSON")):
            response = obj(choices=[obj(finish_reason=finish, message=obj(content=content))], usage=None)
            client = obj(config={"model": "deepseek-flash"},
                         client=obj(chat=obj(completions=FakeCompletion(response))))
            self.assertNotEqual(decompose("question", client)["status"], "ok")
        client = obj(config={"model": "deepseek-flash"},
                     client=obj(chat=obj(completions=FakeCompletion(error=TimeoutError("secret")))))
        failed = decompose("question", client)
        self.assertEqual(failed["status"], "api_error")
        self.assertEqual(failed["error"], "TimeoutError")
        self.assertNotIn("secret", str(failed))

    def test_batch_retrieval_and_original_text_mapping(self):
        retriever = FakeRetriever()
        result = retrieve_branches(retriever, "original", ["aspect 1", "aspect 2"], depth=2)
        self.assertEqual(retriever.queries, ["original", "aspect 1", "aspect 2"])
        self.assertEqual(len(result["branches"]), 3)
        self.assertEqual(result["branches"][0]["candidates"][0]["chunk_text"], "Original A")
        self.assertEqual(result["branches"][0]["candidates"][0]["section_kind"], "body")
        self.assertEqual(result["faiss_total_seconds"], 0.06)
        with self.assertRaises(ValueError):
            retrieve_branches(retriever, "", ["a", "b"], depth=2)
        with self.assertRaises(ValueError):
            retrieve_branches(retriever, "original", ["a"], depth=2)

    def test_rrf_dedup_stable_sort_and_original_chunk_preference(self):
        branches = retrieve_branches(FakeRetriever(), "original", ["aspect 1", "aspect 2"], depth=2)["branches"]
        merged, seconds = merge_branches(branches, limit=2)
        self.assertGreaterEqual(seconds, 0)
        self.assertEqual([row["doc_id"] for row in merged], ["B", "A"])
        self.assertAlmostEqual(merged[0]["rrf_score"], 1 / 62 + 2 / 61)
        self.assertEqual(len(merged[0]["candidate_chunks"]), 1)
        self.assertEqual(merged[0]["best_chunk"]["chunk_text"], "Original B")
        self.assertEqual(merged[0]["branch_ranks"], {"q0": 2, "q1": 1, "q2": 1})

    def test_rrf_tie_uses_original_rank_then_document_id(self):
        branches = [{"query_id": "q0", "candidates": [
            {"doc_id": "z", "chunk_id": "z:1", "branch_rank": 1, "chunk_text": "z"},
            {"doc_id": "a", "chunk_id": "a:1", "branch_rank": 2, "chunk_text": "a"}]},
                    {"query_id": "q1", "candidates": [
            {"doc_id": "a", "chunk_id": "a:1", "branch_rank": 1, "chunk_text": "a"},
            {"doc_id": "z", "chunk_id": "z:1", "branch_rank": 2, "chunk_text": "z"}]}]
        merged, _ = merge_branches(branches)
        self.assertEqual([x["doc_id"] for x in merged], ["z", "a"])
        self.assertEqual(len(merged[0]["candidate_chunks"]), 1)
        branches[0]["candidates"].append(branches[0]["candidates"][0])
        with self.assertRaises(ValueError):
            merge_branches(branches)


class SavedPilotResultsTests(unittest.TestCase):
    def test_saved_six_question_outputs_and_source_mapping(self):
        root = Path(__file__).resolve().parents[1]
        output = root / "experiments/results/phase424/query_decomposition"
        generated = json.loads((output / "generated_subqueries.json").read_text(encoding="utf8"))
        baseline = json.loads((output / "baseline_retrieval.json").read_text(encoding="utf8"))
        branches = json.loads((output / "decomposed_retrieval.json").read_text(encoding="utf8"))
        merged = json.loads((output / "merged_candidates.json").read_text(encoding="utf8"))
        comparison = json.loads((output / "retrieval_comparison.json").read_text(encoding="utf8"))["questions"]
        self.assertEqual([len(x) for x in (generated, baseline, branches, merged, comparison)], [6] * 5)
        for response, a, multi, b, review in zip(generated, baseline, branches, merged, comparison):
            self.assertEqual(response["status"], "ok")
            self.assertEqual(response["question_id"], a["question_id"])
            self.assertTrue(a["historical_top10_exact_match"])
            self.assertEqual(len(a["top100"]), len(b["top100"]), 100)
            self.assertEqual(len(b["top10"]), 10)
            self.assertEqual(b["top10"], b["top100"][:10])
            self.assertEqual(len(multi["branches"]), len(response["subqueries"]) + 1)
            self.assertEqual(review["baseline_top10_doc_ids"], [x["doc_id"] for x in a["top100"][:10]])
            for paper in b["top100"]:
                self.assertEqual(paper["doc_id"], paper["best_chunk"]["doc_id"])
                self.assertIn(paper["best_chunk"]["chunk_id"],
                              {item["chunk"]["chunk_id"] for item in paper["candidate_chunks"]})
                for entry in paper["candidate_chunks"]:
                    self.assertEqual(entry["chunk"]["doc_id"], paper["doc_id"])


if __name__ == "__main__":
    unittest.main()
