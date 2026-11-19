"""Paper-level RRF using ranks, with original-query best chunks preferred."""

import time


def merge_branches(branches, k=60, limit=10):
    start = time.perf_counter()
    papers = {}
    for branch in branches:
        seen = set()
        for row in branch["candidates"]:
            uid = row["doc_id"]
            if uid in seen:
                raise ValueError("duplicate document within a branch")
            seen.add(uid)
            item = papers.setdefault(uid, {"doc_id": uid, "rrf_score": 0.0,
                                           "ranks": {}, "candidate_chunks": {}})
            item["ranks"][branch["query_id"]] = row["branch_rank"]
            item["rrf_score"] += 1 / (k + row["branch_rank"])
            entry = item["candidate_chunks"].setdefault(row["chunk_id"],
                        {"chunk": row, "seen_in_query_ids": []})
            entry["seen_in_query_ids"].append(branch["query_id"])
    ranked = sorted(papers.values(), key=lambda item: (-item["rrf_score"],
                    item["ranks"].get("q0", float("inf")), item["doc_id"]))
    output = []
    for rank, item in enumerate(ranked[:limit], 1):
        entries = list(item["candidate_chunks"].values())
        preferred = min(entries, key=lambda e: (
            0 if "q0" in e["seen_in_query_ids"] else 1,
            min(item["ranks"][qid] for qid in e["seen_in_query_ids"]),
            e["chunk"]["chunk_id"]))
        output.append({"rank": rank, "doc_id": item["doc_id"], "rrf_score": item["rrf_score"],
                       "branch_ranks": item["ranks"], "best_chunk": preferred["chunk"],
                       "candidate_chunks": entries})
    return output, time.perf_counter() - start
