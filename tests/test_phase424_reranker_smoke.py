"""Check the Phase 4.24.1 smoke record against frozen source evidence."""

import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments/results/phase424/reranker_smoke_test/smoke_results.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class RerankerSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_model_files_match_downloaded_revision_record(self):
        self.assertEqual(self.result["status"], "passed")
        self.assertEqual(self.result["revision"], "2cfc18c9415c912f9d8155881c133215df768a70")
        self.assertEqual(digest(ROOT / "experiments/configs/phase424/reranker_smoke.json"),
                         self.result["config_sha256"])
        folder = ROOT / self.result["model_dir"]
        for entry in self.result["model_files"]:
            path = folder / entry["name"]
            self.assertEqual(path.stat().st_size, entry["bytes"])
            self.assertEqual(digest(path), entry["sha256"])
        self.assertTrue((folder / "model.safetensors").is_file())

    def test_real_query_chunk_sources_and_ids(self):
        self.assertEqual(len(self.result["inputs"]), 10)
        for source in self.result["input_sources"]:
            self.assertEqual(digest(ROOT / source["path"]), source["sha256"])
        originals = {qid: json.loads((ROOT / f"experiments/results/phase36/{qid}.json").read_text(encoding="utf-8"))
                     for qid in ("Q07", "Q11")}
        for index, item in enumerate(self.result["inputs"]):
            source = originals[item["question_id"]]
            hit = source["retrieval_top10"][index % 5]
            self.assertEqual(item["question"], source["question"])
            self.assertEqual(item["doc_id"], hit["doc_id"])
            self.assertEqual(item["chunk_id"], hit["best_chunk_id"])
            self.assertEqual(item["chunk_sha256"], hashlib.sha256(hit["best_chunk_text"].encode("utf-8")).hexdigest())

    def test_cuda_scores_ranking_and_edge_cases(self):
        self.assertTrue(self.result["environment"]["cuda_available"])
        self.assertGreater(self.result["vram_peak_allocated_bytes"], 0)
        self.assertEqual({row["batch_size"] for row in self.result["batches"]}, {1, 2, 4})
        self.assertEqual(len({tuple(row["ranking"]) for row in self.result["batches"]}), 1)
        ids = {row["chunk_id"] for row in self.result["inputs"]}
        for row in self.result["batches"]:
            self.assertEqual(len(row["scores"]), 10)
            self.assertEqual(set(row["ranking"]), ids)
            self.assertTrue(all(math.isfinite(score) for score in row["scores"]))
            self.assertTrue(all(seconds > 0 for seconds in row["run_seconds"]))
        edge = self.result["edge_cases"]
        self.assertTrue(edge["empty_chunk"]["finite"])
        self.assertTrue(edge["long_chunk"]["finite"])
        self.assertGreater(edge["long_chunk"]["raw_input_tokens"], edge["long_chunk"]["model_max_length"])
        self.assertTrue(edge["invalid_pair"]["rejected"])


if __name__ == "__main__":
    unittest.main()
