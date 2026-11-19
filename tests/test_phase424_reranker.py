"""Focused Phase 4.24.3 mapping, sorting and resource-path tests."""

import json
import math
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from src.phase424.reranker import load_reranker, make_evidence, rerank


class FakeModel:
    device = "cpu"

    def __init__(self, scores):
        self.scores = scores
        self.inputs = None

    def predict(self, pairs, **kwargs):
        self.inputs = pairs
        return self.scores


def candidate(doc_id, rank, text=None):
    text = text if text is not None else f"Original text of {doc_id}"
    return {"doc_id": doc_id, "original_uid": doc_id, "chunk_id": f"{doc_id}:fixed:1:1",
            "chunk_index": rank - 1, "chunk_text": text, "title": f"Paper {doc_id}",
            "doi": f"10.1/{doc_id}", "bge_score": 1 / rank, "branch_rank": rank,
            "source_path": f"{doc_id}.json", "source_spans": [{"section_index": rank}],
            "section_title": "Results", "section_kind": "body"}


def chunk(row):
    return {"doc_id": row["doc_id"], "chunk_id": row["chunk_id"], "title": row["title"],
            "text": row["chunk_text"], "doi": row["doi"], "source_path": row["source_path"],
            "source_spans": row["source_spans"], "section_title": row["section_title"]}


class RerankerTests(unittest.TestCase):
    def test_ten_candidates_sort_and_tie_break(self):
        rows = [candidate(f"d{i}", i) for i in range(1, 11)]
        scores = [0.5, 0.5] + [i / 10 for i in range(3, 11)]
        model = FakeModel(scores)
        ranked, seconds = rerank("original question", rows, model=model)
        self.assertEqual(len(ranked), 10)
        self.assertGreaterEqual(seconds, 0)
        self.assertEqual(ranked[0]["doc_id"], "d10")
        self.assertLess([x["doc_id"] for x in ranked].index("d1"),
                        [x["doc_id"] for x in ranked].index("d2"))
        self.assertEqual([x["reranker_rank"] for x in ranked], list(range(1, 11)))
        self.assertEqual(model.inputs[0], ("original question", "Original text of d1"))
        self.assertEqual(rows[0]["chunk_text"], "Original text of d1")

    def test_empty_and_invalid_inputs_do_not_load_model(self):
        self.assertEqual(rerank("question", [])[0], [])
        with self.assertRaises(ValueError):
            rerank("  ", [candidate("d1", 1)], model=FakeModel([1]))
        with self.assertRaises(ValueError):
            rerank("question", [candidate("d1", 1, "")], model=FakeModel([1]))
        with self.assertRaises(ValueError):
            rerank("question", [candidate("d1", 1)] * 2, model=FakeModel([1, 1]))
        with self.assertRaises(ValueError):
            rerank("question", [candidate(f"d{i}", i) for i in range(1, 12)], model=FakeModel([1] * 11))

    def test_long_input_passed_intact_to_model_and_invalid_score_rejected(self):
        long_text = "coronavirus surface evidence " * 300
        model = FakeModel([0.2])
        ranked, _ = rerank("surface survival", [candidate("d1", 1, long_text)], model=model)
        self.assertEqual(model.inputs[0][1], long_text)
        self.assertEqual(ranked[0]["chunk_text"], long_text)
        with self.assertRaises(ValueError):
            rerank("question", [candidate("d1", 1)], model=FakeModel([float("nan")]))

    def test_reordered_evidence_uses_ids_not_positions(self):
        rows = [candidate("d1", 1), candidate("d2", 2), candidate("d3", 3)]
        originals = [chunk(row) for row in rows]
        ranked, _ = rerank("question", rows, model=FakeModel([0.1, 0.9, 0.5]))
        evidence, checks = make_evidence(ranked[:3], rows, originals)
        self.assertEqual([(x["citation"], x["doc_id"], x["chunk_id"]) for x in evidence],
                         [("[E1]", "d2", "d2:fixed:1:1"),
                          ("[E2]", "d3", "d3:fixed:1:1"),
                          ("[E3]", "d1", "d1:fixed:1:1")])
        self.assertEqual([x["status"] for x in checks], ["valid"] * 3)
        self.assertEqual(evidence[0]["doi"], "10.1/d2")
        self.assertEqual(evidence[0]["source_path"], "d2.json")
        self.assertEqual(evidence[0]["source_spans"], [{"section_index": 2}])
        self.assertEqual(evidence[0]["text"], "Original text of d2")
        originals[1]["source_path"] = "wrong.json"
        with self.assertRaises(ValueError):
            make_evidence(ranked[:3], rows, originals)

    def test_model_cache_and_device_errors(self):
        load_reranker.cache_clear()
        fake = object()
        with patch("src.phase424.reranker.CrossEncoder", return_value=fake) as constructor:
            self.assertIs(load_reranker("cpu"), fake)
            self.assertIs(load_reranker("cpu"), fake)
            self.assertEqual(constructor.call_count, 1)
            self.assertEqual(constructor.call_args.kwargs["local_files_only"], True)
        load_reranker.cache_clear()
        with patch("src.phase424.reranker.torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                load_reranker("cuda")
        with self.assertRaises(ValueError):
            load_reranker("mps")

    def test_saved_six_question_baseline_and_fixed_pool(self):
        root = Path(__file__).resolve().parents[1]
        folder = root / "experiments/results/phase424/reranking"
        baseline = json.loads((folder / "baseline_candidates.json").read_text(encoding="utf8"))["questions"]
        reranked = json.loads((folder / "reranked_candidates.json").read_text(encoding="utf8"))["questions"]
        scores = json.loads((folder / "retrieval_comparison.json").read_text(encoding="utf8"))["questions"]
        evidence = json.loads((folder / "evidence_comparison.json").read_text(encoding="utf8"))["questions"]
        self.assertEqual([len(x) for x in (baseline, reranked, scores, evidence)], [6] * 4)
        for a, c, metric, pair in zip(baseline, reranked, scores, evidence):
            self.assertEqual(a["question_id"], c["question_id"])
            self.assertTrue(a["historical_top10_exact_match"])
            self.assertTrue(metric["candidate_set_identical"])
            self.assertEqual({x["doc_id"] for x in a["top10"]},
                             {x["doc_id"] for x in c["ranked_top10"]})
            self.assertEqual([len(a["top10"]), len(c["ranked_top10"])], [10, 10])
            self.assertEqual([x["reranker_rank"] for x in c["ranked_top10"]], list(range(1, 11)))
            for group in (pair["baseline_top3"], pair["reranked_top3"]):
                self.assertEqual([x["citation"] for x in group], ["[E1]", "[E2]", "[E3]"])
                self.assertEqual(len({x["doc_id"] for x in group}), 3)
            self.assertTrue(all(x["status"] == "valid" for x in pair["reranked_source_checks"]))
            for topic in metric["baseline_metrics_by_source_topic"]:
                old = metric["baseline_metrics_by_source_topic"][topic]
                new = metric["reranked_metrics_by_source_topic"][topic]
                self.assertEqual(old["recall@10"], new["recall@10"])
                self.assertEqual(old["recall@100"], new["recall@100"])

    def test_z_real_cuda_long_input_and_model_limit(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        root = Path(__file__).resolve().parents[1]
        historical = json.loads((root / "experiments/results/phase36/Q07.json").read_text(encoding="utf8"))
        text = historical["retrieval_top10"][0]["best_chunk_text"] * 8
        model = load_reranker("cuda")
        self.assertEqual(model.max_seq_length, 512)
        self.assertGreater(len(model.tokenizer.tokenize(text)), 512)
        ranked, elapsed = rerank(historical["question"], [candidate("long", 1, text)], model=model)
        self.assertEqual(ranked[0]["chunk_text"], text)
        self.assertTrue(math.isfinite(ranked[0]["reranker_score"]))
        self.assertGreater(elapsed, 0)


if __name__ == "__main__":
    unittest.main()
