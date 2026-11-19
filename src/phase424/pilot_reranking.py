"""Offline six-question reranking and a 20-question resource continuity probe."""

import hashlib
import json
import statistics
import time
from pathlib import Path

import psutil
import torch

from src.evidence_metadata import enrich
from src.phase424.pilot_retrieval import judged_metrics, qrels_by_topic
from src.phase424.reranker import load_reranker, make_evidence, rerank
from src.retriever import PilotRetriever


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/results/phase424/reranking"
SOURCE = ROOT / "experiments/results/phase424/query_decomposition/baseline_retrieval.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def memory():
    free, total = torch.cuda.mem_get_info()
    return {"allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "device_free_bytes": free, "device_total_bytes": total,
            "process_rss_bytes": psutil.Process().memory_info().rss}


def history_candidates(history, chunks):
    rows = []
    for hit in history["retrieval_top10"]:
        original = chunks[hit["chunk_index"]]
        if original["chunk_id"] != hit["best_chunk_id"] or original["doc_id"] != hit["doc_id"]:
            raise ValueError("historical Document/Chunk mapping mismatch")
        rows.append({"doc_id": hit["doc_id"], "original_uid": hit["original_uid"],
                     "chunk_id": hit["best_chunk_id"], "chunk_index": hit["chunk_index"],
                     "chunk_text": hit["best_chunk_text"], "title": hit["title"], "doi": hit["doi"],
                     "bge_score": hit["score"], "branch_rank": hit["rank"],
                     "source_path": hit["source_path"], "source_spans": hit["source_spans"],
                     "section_title": hit["section_title"], "section_kind": original["section_kind"]})
    return rows


def main():
    config = read(OUT / "reranker_config.json")
    baseline = {row["question_id"]: row for row in read(SOURCE)}
    expected = config["pilot_question_ids"]
    if list(baseline) != expected or not all(baseline[qid]["historical_top10_exact_match"] for qid in expected):
        raise ValueError("A group differs from saved Phase 3.6 baseline; do not score benefits")
    output_files = ("baseline_candidates.json", "reranked_candidates.json", "retrieval_comparison.json",
                    "evidence_comparison.json", "latency_metrics.json", "gpu_metrics.json")
    if any((OUT / name).exists() for name in output_files):
        raise FileExistsError("Phase 4.24.3 result already exists")
    frozen = {row["question_id"]: row for row in read(ROOT / "experiments/configs/phase36_questions.json")["questions"]}
    source_hash = sha256(SOURCE)
    original_hashes = {qid: sha256(ROOT / f"experiments/results/phase36/{qid}.json") for qid in expected}
    model_hash = sha256(ROOT / "models/bge-reranker-base/model.safetensors")
    smoke = read(ROOT / "experiments/results/phase424/reranker_smoke_test/smoke_results.json")
    if model_hash != next(row["sha256"] for row in smoke["model_files"] if row["name"] == "model.safetensors"):
        raise ValueError("reranker weights changed since Phase 4.24.1")

    if config["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required by the frozen Phase 4.24.3 config")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    retriever = PilotRetriever(run_id=config["retrieval_run_id"])
    bge_load_seconds = time.perf_counter() - start
    after_bge = memory()
    start = time.perf_counter()
    model = load_reranker("cuda")
    torch.cuda.synchronize()
    reranker_load_seconds = time.perf_counter() - start
    after_both = memory()
    if load_reranker("cuda") is not model:
        raise RuntimeError("reranker cache did not reuse model instance")
    corpus = {row["doc_id"] for row in retriever.chunks}
    qrels = qrels_by_topic()
    a_rows, c_rows, metric_rows, evidence_rows, timings = [], [], [], [], []
    for qid in expected:
        question = frozen[qid]["question"]
        records = baseline[qid]["top100"]
        top10 = records[:10]
        history = read(ROOT / f"experiments/results/phase36/{qid}.json")
        for now, old in zip(top10, history["retrieval_top10"]):
            if (now["doc_id"], now["chunk_id"], now["chunk_text"]) != (
                    old["doc_id"], old["best_chunk_id"], old["best_chunk_text"]):
                raise ValueError(f"A group historical mismatch: {qid}")
        ranked, seconds = rerank(question, top10, model=model, batch_size=config["batch_size"])
        if {row["doc_id"] for row in ranked} != {row["doc_id"] for row in top10}:
            raise ValueError("reranker changed candidate pool")
        a_evidence, a_checks = make_evidence(top10[:3], top10, retriever.chunks)
        c_evidence, c_checks = make_evidence(ranked[:3], top10, retriever.chunks)
        a_evidence, c_evidence = enrich(a_evidence), enrich(c_evidence)
        a_ids = [row["doc_id"] for row in records]
        c_ids = [row["doc_id"] for row in ranked] + a_ids[10:]
        a_metrics = judged_metrics(a_ids, frozen[qid]["source_topic_ids"], qrels, corpus)
        c_metrics = judged_metrics(c_ids, frozen[qid]["source_topic_ids"], qrels, corpus)
        for topic in a_metrics:
            if a_metrics[topic] and c_metrics[topic]:
                if a_metrics[topic]["recall@10"] != c_metrics[topic]["recall@10"] or \
                   a_metrics[topic]["recall@100"] != c_metrics[topic]["recall@100"]:
                    raise ValueError("fixed-pool reranking changed recall")
        a_rows.append({"question_id": qid, "original_query": question, "top10": top10,
                       "top100_doc_ids": a_ids, "historical_top10_exact_match": True})
        c_rows.append({"question_id": qid, "original_query": question, "ranked_top10": ranked,
                       "top100_doc_ids_after_top10_reorder": c_ids})
        a_chunks = {row["chunk_id"] for row in a_evidence}
        c_chunks = {row["chunk_id"] for row in c_evidence}
        evidence_rows.append({"question_id": qid, "original_query": question,
                              "baseline_top3": a_evidence, "reranked_top3": c_evidence,
                              "baseline_source_checks": a_checks, "reranked_source_checks": c_checks,
                              "entered_chunk_ids": sorted(c_chunks - a_chunks),
                              "exited_chunk_ids": sorted(a_chunks - c_chunks)})
        metric_rows.append({"question_id": qid, "source_topic_ids": frozen[qid]["source_topic_ids"],
                            "qrel_applicability": "source-topic proxy; not direct question labels",
                            "baseline_metrics_by_source_topic": a_metrics,
                            "reranked_metrics_by_source_topic": c_metrics,
                            "baseline_top10_doc_ids": a_ids[:10], "reranked_top10_doc_ids": c_ids[:10],
                            "candidate_set_identical": True,
                            "top3_paper_overlap": len(set(a_ids[:3]) & set(c_ids[:3]))})
        timings.append({"question_id": qid, "warm_rerank_seconds": seconds,
                        "gpu_after_query": memory()})
        print(qid, "top3 overlap", metric_rows[-1]["top3_paper_overlap"], flush=True)

    # A resource-only probe: no new retrieval or answer generation for these 20 frozen questions.
    continuous = []
    for number in range(1, 21):
        qid = f"Q{number:02d}"
        history = read(ROOT / f"experiments/results/phase36/{qid}.json")
        rows = history_candidates(history, retriever.chunks)
        _, seconds = rerank(history["question"], rows, model=model, batch_size=config["batch_size"])
        continuous.append({"question_id": qid, "rerank_seconds": seconds,
                           "gpu_allocated_bytes_after": torch.cuda.memory_allocated(),
                           "gpu_reserved_bytes_after": torch.cuda.memory_reserved()})
    torch.cuda.synchronize()
    after_continuous = memory()
    write("baseline_candidates.json", {"source_path": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
                                        "source_sha256": source_hash, "phase36_file_sha256": original_hashes,
                                        "questions": a_rows})
    write("reranked_candidates.json", {"questions": c_rows})
    write("retrieval_comparison.json", {"note": "Source-topic qrels are proxies; only Top-10 order changes, not the Top-100 set.",
                                        "questions": metric_rows})
    write("evidence_comparison.json", {"note": "Source-role labels are heuristic; semantic support needs review.",
                                       "questions": evidence_rows})
    write("latency_metrics.json", {"bge_index_cold_load_seconds": bge_load_seconds,
                                   "reranker_cold_load_seconds_with_bge_resident": reranker_load_seconds,
                                   "pilot_queries": timings,
                                   "pilot_mean_warm_rerank_seconds": statistics.mean(x["warm_rerank_seconds"] for x in timings),
                                   "continuous_20": continuous,
                                   "continuous_mean_seconds": statistics.mean(x["rerank_seconds"] for x in continuous)})
    write("gpu_metrics.json", {"device": torch.cuda.get_device_name(0),
                               "after_bge": after_bge, "after_bge_and_reranker": after_both,
                               "after_continuous_20": after_continuous,
                               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                               "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                               "reranker_cache": {"loads": load_reranker.cache_info().misses,
                                                  "hits": load_reranker.cache_info().hits}})


if __name__ == "__main__":
    main()
