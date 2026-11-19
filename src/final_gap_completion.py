"""Derive proposal-gap metrics strictly from frozen ScholarRAG result files."""

import hashlib
import json
import random
import statistics
from pathlib import Path

from src.evaluation import score_topic


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/results/final_gap_completion"
QRELS = "data/evaluation/trec_covid_complete_qrels.txt"
RUNS = (
    ("pilot_fixed", "Pilot", "Fixed-length", "phase31_pilot", "phase31_pilot_manifest.json", "preprocess_repeats.json"),
    ("pilot_sentence", "Pilot", "Sentence-aware", "phase32_improved", "phase31_pilot_manifest.json", "preprocess_repeats.json"),
    ("medium_fixed", "Medium", "Fixed-length", "phase33/phase33_medium_fixed", "phase33_medium_manifest.json", "chunking_repeats.json"),
    ("medium_sentence", "Medium", "Sentence-aware", "phase33/phase33_medium_sentence", "phase33_medium_manifest.json", "chunking_repeats.json"),
    ("large_fixed", "Large", "Fixed-length", "phase33/phase33_large_fixed", "phase33_large_manifest.json", "chunking_repeats.json"),
    ("large_sentence", "Large", "Sentence-aware", "phase33/phase33_large_sentence", "phase33_large_manifest.json", "chunking_repeats.json"),
)


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def digest(relative):
    with (ROOT / relative).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def close(actual, expected):
    return abs(actual - expected) < 1e-10


def p95(values):
    ordered = sorted(values)
    point = (len(ordered) - 1) * 0.95
    low = int(point)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (point - low)


def summary(values):
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "p95": p95(values), "min": min(values), "max": max(values)}


def run_metrics(key, scale, method, folder, manifest_name, repeats_name, sample):
    base = f"experiments/results/{folder}"
    manifest_path = f"data/processed/{manifest_name}"
    paths = {name: f"{base}/{name}.json" for name in
             ("metrics", "performance_metrics", "chunking_stats", "index_metrics")}
    paths["ranking"] = f"{base}/retrieval_results.jsonl"
    paths["manifest"] = manifest_path
    paths["preprocessing_repeats"] = f"{base}/{repeats_name}"
    sources = {relative: digest(relative) for relative in paths.values()}
    manifest, prior = read(manifest_path), read(paths["metrics"])
    perf, chunks, index, repeats = [read(paths[name]) for name in
                                   ("performance_metrics", "chunking_stats", "index_metrics", "preprocessing_repeats")]
    corpus = {doc["doc_id"] for doc in manifest["documents"]}
    assert len(corpus) == manifest["document_count"] == prior["candidate_documents"] == chunks["document_count"] == index["document_count"]
    assert chunks["total_chunks"] == index["chunk_count"] == perf["chunk_count"]
    assert manifest["query_ids"] == [str(i) for i in range(1, 51)]
    assert prior["query_count"] == 50
    qrels = manifest["effective_qrels"]
    by_query = {}
    with (ROOT / paths["ranking"]).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            topic = str(row["topic_id"])
            top100 = row["top100"]
            ids = [hit["doc_id"] for hit in top100]
            assert len(ids) == len(set(ids)) == 100 and topic not in by_query
            assert [hit["rank"] for hit in top100] == list(range(1, 101))
            assert all(hit["official_grade"] == qrels[topic].get(hit["doc_id"]) for hit in top100)
            relevant = {uid for uid, grade in qrels[topic].items() if grade >= 1 and uid in corpus}
            assert len(relevant) == prior["per_query"][topic]["available_positive"] == manifest["positive_per_query"][topic]
            recall5 = sum(uid in relevant for uid in ids[:5]) / len(relevant)
            assessed = score_topic(ids, qrels[topic], corpus)
            assert assessed is not None
            for metric in ("ndcg@10", "recall@10", "recall@100", "mrr"):
                assert close(assessed[metric], prior["per_query"][topic]["metrics"][metric])
            assert recall5 <= assessed["recall@10"] + 1e-12 <= assessed["recall@100"] + 1e-12
            by_query[topic] = {
                "topic_id": topic, "query": row["query"], "known_positive_in_corpus": len(relevant),
                "positive_in_top5": sum(uid in relevant for uid in ids[:5]),
                "recall@5": recall5, "recall@10": assessed["recall@10"],
                "recall@100": assessed["recall@100"], "ndcg@10": assessed["ndcg@10"],
                "ranking_source": paths["ranking"], "qrels_source": manifest_path + "#effective_qrels/" + topic,
            }
    assert set(by_query) == set(manifest["query_ids"])
    macro = {metric: statistics.mean(row[metric] for row in by_query.values())
             for metric in ("recall@5", "recall@10", "recall@100", "ndcg@10")}
    for metric in ("recall@10", "recall@100", "ndcg@10"):
        assert close(macro[metric], prior["macro_average"][metric])
    selected = sorted(sample.sample(list(by_query), 5), key=int)
    # Reopen the original ranking for a separate, sampled direct-count check.
    with (ROOT / paths["ranking"]).open(encoding="utf-8") as stream:
        sampled_rankings = {str(row["topic_id"]): row["top100"][:5]
                            for row in map(json.loads, stream)
                            if str(row["topic_id"]) in selected}
    for topic in selected:
        relevant = {uid for uid, grade in qrels[topic].items() if grade >= 1 and uid in corpus}
        direct = sum(hit["doc_id"] in relevant for hit in sampled_rankings[topic]) / len(relevant)
        assert close(by_query[topic]["recall@5"], direct)

    times = [record.get("timing", record) for record in repeats["runs"]]
    load_seconds = statistics.median(record["document_loading_seconds"] for record in times)
    chunk_seconds = statistics.median(record["common_cleaning_and_chunking_seconds"] for record in times)
    embedding_seconds = perf["embedding_seconds"]["seconds_median"]
    assert all(record["chunk_count"] == chunks["total_chunks"] for record in repeats["runs"])
    assert close(index["index_size_bytes"], perf["index_size_bytes"])
    latency_repeats = perf["retrieval_latency_repeats"]
    assert len(latency_repeats) == 3 and all(len(record["per_query_seconds"]) == 50 for record in latency_repeats)
    per_query_latency = [statistics.median(record["per_query_seconds"][position] for record in latency_repeats)
                         for position in range(50)]
    faiss_latency = summary(per_query_latency)
    batch_encoding = statistics.median(record["seconds"] for record in perf["query_encoding_repeats"])
    out = {
        "scale": scale, "method": method, "documents": len(corpus), "chunks": chunks["total_chunks"],
        "model": prior["model"], "query_count": 50, "quality_metrics_macro": macro,
        "per_query": {topic: by_query[topic] for topic in manifest["query_ids"]},
        "preprocessing_time_seconds": {"document_reading_median": load_seconds,
                                       "common_cleaning_and_chunking_median": chunk_seconds,
                                       "median_of_three_repeats": True,
                                       "scope": "separate timed phases; chunking includes common cleaning"},
        "indexing_time_seconds": {"build": index["index_build_seconds"], "write": index["index_write_seconds"],
                                  "build_plus_write": index["index_total_seconds"]},
        "documents_per_second": {
            "document_reading_only": len(corpus) / load_seconds,
            "common_cleaning_and_chunking": len(corpus) / chunk_seconds,
            "embedding_and_vector_write_only": len(corpus) / embedding_seconds,
            "complete_offline_pipeline": None,
            "complete_offline_pipeline_status": "Not Available: no single elapsed time from read through index persisted"},
        "embedding_time_seconds_median": embedding_seconds,
        "chunk_embedding_throughput_per_second": chunks["total_chunks"] / embedding_seconds,
        "query_latency_seconds": {
            "faiss_and_document_aggregation_warm": faiss_latency,
            "measurement": "per-topic median across three 50-topic retrieval repeats; excludes query encoding",
            "bge_batch_50_encoding_median": batch_encoding,
            "bge_single_query_encoding": None,
            "bge_single_query_encoding_status": "Not Available in scale benchmark: only 50-query batch timing",
            "end_to_end_retrieval": None,
            "end_to_end_retrieval_status": "Not Available as a directly measured stage for this run"},
        "queries_per_second": {"sequential_faiss_only": 1 / faiss_latency["mean"],
                               "concurrent": None, "concurrent_status": "Not Available: no concurrency benchmark"},
        "memory": {"embedding_process_rss_peak_bytes": perf["ram_peak_bytes"],
                   "embedding_process_cuda_reserved_peak_bytes": perf["vram_peak_reserved_bytes"]},
        "index_storage_bytes": index["index_size_bytes"],
        "sampled_topic_ids_recalculated": selected,
        "source_sha256": sources,
    }
    return out


def main():
    assert not (OUTPUT / "final_metrics_summary.json").exists(), "Do not overwrite existing metrics"
    source_qrels_hash = digest(QRELS)
    rng = random.Random(20261117)
    results = {key: run_metrics(key, scale, method, folder, manifest, repeats, rng)
               for key, scale, method, folder, manifest, repeats in RUNS}
    for scale in ("Pilot", "Medium", "Large"):
        pair = [row for row in results.values() if row["scale"] == scale]
        assert len(pair) == 2
        assert pair[0]["documents"] == pair[1]["documents"]
        assert [x["query"] for x in pair[0]["per_query"].values()] == [x["query"] for x in pair[1]["per_query"].values()]
        assert all(pair[0]["per_query"][q]["known_positive_in_corpus"] ==
                   pair[1]["per_query"][q]["known_positive_in_corpus"] for q in pair[0]["per_query"])
    phase36_path = "experiments/results/phase36/aggregate.json"
    rag = read(phase36_path)
    assert rag["successful_real_api_calls"] == 20
    result = {
        "version": "final-gap-completion-v1", "run_count": 6,
        "definitions": {
            "relevance": "effective qrels grade >= 1 after frozen Document ID normalization",
            "recall_denominator": "known positive documents within the current frozen corpus",
            "unjudged": "unknown, not negative; conventional qrel nDCG awards zero gain",
            "recall@5": "count of unique known positive document IDs in ranks 1..5 / corpus known positive count",
            "documents_per_second": "completed unique papers / separately measured stage seconds",
            "queries_per_second": "one sequential query / mean warm FAISS+paper aggregation seconds; no concurrency",
            "latency_p95": "linear interpolation at (n-1)*0.95 after sorting",
        },
        "official_qrels_source": {"path": QRELS, "sha256": source_qrels_hash},
        "runs": results,
        "historical_v1_rag_large_fixed": {
            "source": phase36_path, "source_sha256": digest(phase36_path),
            "real_api_questions": 20, "cold_load_seconds": rag["cold_load_seconds"],
            "query_embedding_latency_seconds": rag["metrics"]["query_embedding_seconds"],
            "faiss_retrieval_latency_seconds": rag["metrics"]["faiss_retrieval_seconds"],
            "end_to_end_retrieval_latency_seconds": rag["metrics"]["retrieval_total_seconds"],
            "sequential_effective_retrieval_queries_per_second": 1 / rag["metrics"]["retrieval_total_seconds"]["mean"],
            "llm_generation_latency_seconds": rag["metrics"]["llm_generation_seconds"],
            "end_to_end_answer_latency_seconds": rag["metrics"]["end_to_end_complete_seconds"],
            "actual_token_usage_totals": rag["actual_token_totals"],
            "actual_token_usage_per_query": {"input": rag["metrics"]["input_tokens"],
                                             "output": rag["metrics"]["output_tokens"]},
            "historical_api_cost_estimate_total_cny": rag["cost_estimate_total_cny"],
            "historical_pricing_source": rag["pricing_source"],
            "note": "20 historical real calls; cost is a historical price-based estimate, not a bill or new API call"},
        "validation": {"historical_recall_ndcg_cross_checks": "300 topic/run comparisons for each stored metric passed",
                       "sample_seed": 20261117,
                       "sampled_topics_per_run": 5,
                       "source_files_exist_and_sha256_recorded": True,
                       "query_count_and_rank_uniqueness": "50 topics and 100 distinct ranked IDs per run",
                       "recall_monotonicity": "Recall@5 <= Recall@10 <= Recall@100 across all 300 topic/run pairs",
                       "same_scale_query_and_positive_denominator_checks": "passed"},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "final_metrics_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: {"recall@5": row["quality_metrics_macro"]["recall@5"],
                            "documents_per_second_embedding": row["documents_per_second"]["embedding_and_vector_write_only"]}
                      for key, row in results.items()}, indent=2))


if __name__ == "__main__":
    main()
