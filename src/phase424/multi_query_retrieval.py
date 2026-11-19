"""Reuse the frozen BGE encoder and FAISS retriever for multiple queries."""

import time


def retrieve_branches(retriever, original_query, subqueries, depth=10):
    if not original_query.strip() or not 2 <= len(subqueries) <= 4:
        raise ValueError("original query and 2–4 subqueries required")
    queries = [original_query, *subqueries]
    vectors, encoding_seconds = retriever.encode_queries(queries)
    branches = []
    for index, (query, vector) in enumerate(zip(queries, vectors)):
        docs, perf = retriever.retrieve_vector(vector, depth)
        rows = []
        for doc in docs:
            chunk = retriever.chunks[doc["chunk_index"]]
            if chunk["chunk_id"] != doc["best_chunk_id"] or chunk["doc_id"] != doc["doc_id"]:
                raise ValueError("Document/Chunk mapping mismatch")
            rows.append({
                "doc_id": doc["doc_id"], "original_uid": doc["original_uid"],
                "chunk_id": doc["best_chunk_id"], "chunk_index": doc["chunk_index"],
                "chunk_text": doc["best_chunk_text"], "title": doc["title"], "doi": doc["doi"],
                "bge_score": doc["score"], "branch_rank": doc["rank"],
                "query_id": f"q{index}", "query_type": "original" if index == 0 else "subquery",
                "origin_query": query, "source_path": doc["source_path"],
                "source_spans": doc["source_spans"], "section_title": doc["section_title"],
                "section_kind": chunk["section_kind"],
            })
        branches.append({"query_id": f"q{index}", "query_type": "original" if index == 0 else "subquery",
                         "query": query, "retrieval_seconds": perf["retrieval_seconds"], "candidates": rows})
    return {"branches": branches, "query_encoding_seconds": encoding_seconds,
            "faiss_total_seconds": sum(branch["retrieval_seconds"] for branch in branches)}
