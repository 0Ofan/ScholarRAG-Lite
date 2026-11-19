"""Offline scope-review replay of the six frozen decomposition pilot questions."""

import hashlib
import json
from pathlib import Path

from src.phase424.multi_query_retrieval import retrieve_branches
from src.phase424.pilot_retrieval import judged_metrics, qrels_by_topic
from src.phase424.query_scope import validate_query_scope
from src.phase424.result_merger import merge_branches
from src.retriever import PilotRetriever


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/results/phase424/query_decomposition"
OUT = ROOT / "experiments/results/phase424/query_scope_refinement"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_revisions(originals, decisions):
    by_id = {item["question_id"]: item for item in decisions["decisions"]}
    if len(originals) != 6 or len(by_id) != 6:
        raise ValueError("expected six original questions and six decisions")
    rows = []
    for original in originals:
        qid = original["question_id"]
        if original["status"] != "ok":
            raise ValueError(f"original decomposition failed: {qid}")
        checks = validate_query_scope(original["original_query"], original["subqueries"])
        exclude = set(by_id[qid]["exclude_subquery_indices_one_based"])
        flagged = {item["index_one_based"] for item in checks["flagged_subqueries"]}
        if not exclude <= flagged:
            raise ValueError(f"decision excludes an unflagged query: {qid}")
        revised = [query for index, query in enumerate(original["subqueries"], 1) if index not in exclude]
        if not 2 <= len(revised) <= 4:
            raise ValueError(f"subquery count outside 2–4: {qid}")
        rows.append({"question_id": qid, "original_query": original["original_query"],
                     "before_subqueries": original["subqueries"], "after_subqueries": revised,
                     "excluded_indices_one_based": sorted(exclude),
                     "decision_reason": by_id[qid]["reason"],
                     "before_scope_check": checks,
                     "after_scope_check": validate_query_scope(original["original_query"], revised)})
    return rows


def main():
    originals = read(SOURCE / "generated_subqueries.json")
    decisions = read(OUT / "scope_decisions.json")
    revised = prepare_revisions(originals, decisions)
    old_comparison = {row["question_id"]: row for row in read(SOURCE / "retrieval_comparison.json")["questions"]}
    old_merged = {row["question_id"]: row for row in read(SOURCE / "merged_candidates.json")}
    old_baseline = {row["question_id"]: row for row in read(SOURCE / "baseline_retrieval.json")}
    frozen = {row["question_id"]: row for row in read(ROOT / "experiments/configs/phase36_questions.json")["questions"]}
    output_names = ("scope_validation.json", "revised_subqueries.json", "revised_retrieval.json",
                    "retrieval_comparison.json", "input_hashes.json")
    if any((OUT / name).exists() for name in output_names):
        raise FileExistsError("refinement outputs already exist")
    retriever = PilotRetriever(run_id="phase33_large_fixed")
    corpus = {chunk["doc_id"] for chunk in retriever.chunks}
    qrels = qrels_by_topic()
    reruns, comparisons = [], []
    for item in revised:
        qid = item["question_id"]
        if item["original_query"] != frozen[qid]["question"]:
            raise ValueError("frozen question mismatch")
        retrieved = retrieve_branches(retriever, item["original_query"], item["after_subqueries"], depth=100)
        branches = retrieved["branches"]
        merged, merge_seconds = merge_branches(branches, k=60, limit=100)
        old_a = old_baseline[qid]["top100"]
        new_a = branches[0]["candidates"]
        if [(x["doc_id"], x["chunk_id"], x["chunk_text"]) for x in old_a] != [
                (x["doc_id"], x["chunk_id"], x["chunk_text"]) for x in new_a]:
            raise ValueError(f"baseline changed: {qid}")
        previous = old_merged[qid]["top100"]
        old_ids = [x["doc_id"] for x in previous]
        new_ids = [x["doc_id"] for x in merged]
        before_top10, after_top10 = set(old_ids[:10]), set(new_ids[:10])
        metrics = judged_metrics(new_ids, frozen[qid]["source_topic_ids"], qrels, corpus)
        reruns.append({"question_id": qid, "branches": branches, "merged_top100": merged,
                       "merged_top10": merged[:10], "query_embedding_seconds": retrieved["query_encoding_seconds"],
                       "faiss_total_seconds": retrieved["faiss_total_seconds"], "rrf_merge_seconds": merge_seconds})
        comparisons.append({
            "question_id": qid,
            "source_topic_ids": frozen[qid]["source_topic_ids"],
            "qrel_applicability": old_comparison[qid]["qrel_applicability"],
            "original_baseline_metrics_by_source_topic": old_comparison[qid]["baseline_metrics_by_source_topic"],
            "before_decomposition_metrics_by_source_topic": old_comparison[qid]["decomposed_metrics_by_source_topic"],
            "after_refinement_metrics_by_source_topic": metrics,
            "before_top10_doc_ids": old_ids[:10], "after_top10_doc_ids": new_ids[:10],
            "top10_overlap": len(before_top10 & after_top10),
            "top100_overlap": len(set(old_ids) & set(new_ids)),
            "added_top10_doc_ids": sorted(after_top10 - before_top10),
            "removed_top10_doc_ids": sorted(before_top10 - after_top10),
            "baseline_unchanged": True,
            "timings": {"query_embedding_seconds": retrieved["query_encoding_seconds"],
                        "faiss_seconds": retrieved["faiss_total_seconds"], "merge_seconds": merge_seconds},
        })
        print(qid, "Top-10 overlap", len(before_top10 & after_top10), flush=True)
    write("scope_validation.json", {"note": "Rules produce hints, not semantic proof.",
                                    "questions": [{"question_id": x["question_id"],
                                                   "before": x["before_scope_check"],
                                                   "after": x["after_scope_check"]} for x in revised]})
    write("revised_subqueries.json", revised)
    write("revised_retrieval.json", reruns)
    write("retrieval_comparison.json", {"note": "TREC qrels are source-topic proxies, not direct labels for these natural-language questions.",
                                        "questions": comparisons})
    paths = [SOURCE / "generated_subqueries.json", OUT / "scope_decisions.json",
             ROOT / "experiments/configs/phase36_questions.json",
             ROOT / "data/evaluation/trec_covid_complete_qrels.txt",
             ROOT / "data/processed/phase33_large_fixed_chunks.jsonl",
             ROOT / "data/indexes/phase33_large_fixed/faiss.index"]
    write("input_hashes.json", {"files": [{"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                                            "bytes": path.stat().st_size, "sha256": digest(path)} for path in paths]})


if __name__ == "__main__":
    main()
