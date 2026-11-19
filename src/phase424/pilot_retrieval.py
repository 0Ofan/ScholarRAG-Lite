"""Six-question offline BGE versus decomposed BGE retrieval pilot."""

import json
from pathlib import Path

from src.evaluation import score_topic
from src.phase424.multi_query_retrieval import retrieve_branches
from src.phase424.result_merger import merge_branches
from src.retriever import PilotRetriever


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/results/phase424/query_decomposition"


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def qrels_by_topic():
    result = {}
    with (ROOT / "data/evaluation/trec_covid_complete_qrels.txt").open(encoding="utf-8") as stream:
        for line in stream:
            topic, _, uid, grade = line.split()
            result.setdefault(int(topic), {})[uid] = int(grade)
    return result


def judged_metrics(ranking, source_topic_ids, qrels, corpus):
    metrics = {}
    for topic in source_topic_ids:
        judgments = qrels.get(topic, {})
        scored = score_topic(ranking, judgments, corpus)
        metrics[str(topic)] = None if scored is None else {
            **scored,
            "precision@3_known_positive": sum(judgments.get(uid, -1) >= 1 for uid in ranking[:3]) / 3,
            "top3_unjudged": sum(uid not in judgments for uid in ranking[:3]),
        }
    return metrics


def main():
    generated = json.loads((OUT / "generated_subqueries.json").read_text(encoding="utf-8"))
    if len(generated) != 6 or any(row["status"] != "ok" for row in generated):
        raise ValueError("six successful frozen decomposition responses required")
    frozen = {row["question_id"]: row for row in json.loads((ROOT / "experiments/configs/phase36_questions.json")
               .read_text(encoding="utf-8"))["questions"]}
    outputs = [OUT / name for name in ("baseline_retrieval.json", "decomposed_retrieval.json",
                                      "merged_candidates.json", "retrieval_comparison.json")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("pilot retrieval outputs already exist")
    retriever = PilotRetriever(run_id="phase33_large_fixed")
    corpus = {chunk["doc_id"] for chunk in retriever.chunks}
    qrels = qrels_by_topic()
    baseline_rows, branch_rows, merged_rows, comparisons = [], [], [], []
    for row in generated:
        qid = row["question_id"]
        question = frozen[qid]
        if row["original_query"] != question["question"]:
            raise ValueError("frozen question mismatch")
        retrieved = retrieve_branches(retriever, row["original_query"], row["subqueries"], depth=100)
        branches = retrieved["branches"]
        merged, merge_seconds = merge_branches(branches, limit=100)
        a = branches[0]["candidates"]
        b = merged
        baseline_ids = [item["doc_id"] for item in a]
        merged_ids = [item["doc_id"] for item in b]
        historical = json.loads((ROOT / f"experiments/results/phase36/{qid}.json").read_text(encoding="utf-8"))
        historical_match = all(old["doc_id"] == now["doc_id"] and
                               old["best_chunk_id"] == now["chunk_id"] and
                               old["best_chunk_text"] == now["chunk_text"]
                               for old, now in zip(historical["retrieval_top10"], a[:10]))
        metrics_a = judged_metrics(baseline_ids, question["source_topic_ids"], qrels, corpus)
        metrics_b = judged_metrics(merged_ids, question["source_topic_ids"], qrels, corpus)
        top10_a, top10_b = set(baseline_ids[:10]), set(merged_ids[:10])
        baseline_rows.append({"question_id": qid, "top100": a, "historical_top10_exact_match": historical_match})
        branch_rows.append({"question_id": qid, "branches": branches,
                            "query_embedding_seconds": retrieved["query_encoding_seconds"],
                            "faiss_total_seconds": retrieved["faiss_total_seconds"]})
        merged_rows.append({"question_id": qid, "top10": b[:10], "top100": b,
                            "merge_seconds": merge_seconds})
        comparisons.append({
            "question_id": qid, "source_topic_ids": question["source_topic_ids"],
            "qrel_applicability": "single-source-topic proxy; wording differs from TREC topic" if
                len(question["source_topic_ids"]) == 1 else
                "multiple source topics; no unique qrels for this natural-language question",
            "baseline_metrics_by_source_topic": metrics_a,
            "decomposed_metrics_by_source_topic": metrics_b,
            "baseline_top10_doc_ids": baseline_ids[:10],
            "decomposed_top10_doc_ids": merged_ids[:10],
            "top10_overlap": len(top10_a & top10_b),
            "added_top10_doc_ids": sorted(top10_b - top10_a),
            "removed_top10_doc_ids": sorted(top10_a - top10_b),
            "historical_top10_exact_match": historical_match,
            "timings": {
                "batch_query_embedding_seconds": retrieved["query_encoding_seconds"],
                "baseline_faiss_seconds": branches[0]["retrieval_seconds"],
                "subquery_faiss_seconds": sum(x["retrieval_seconds"] for x in branches[1:]),
                "rrf_merge_seconds": merge_seconds,
                "decomposed_local_total_seconds": retrieved["query_encoding_seconds"] +
                    retrieved["faiss_total_seconds"] + merge_seconds,
                "decomposition_api_seconds": row["latency_seconds"],
            },
        })
        print(qid, "top10 overlap", len(top10_a & top10_b), "history", historical_match, flush=True)
    save("baseline_retrieval.json", baseline_rows)
    save("decomposed_retrieval.json", branch_rows)
    save("merged_candidates.json", merged_rows)
    save("retrieval_comparison.json", {"note": "TREC qrels are source-topic proxies, not direct labels for rewritten questions.",
                                        "questions": comparisons})


if __name__ == "__main__":
    main()
