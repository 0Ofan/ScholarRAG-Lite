"""Local-only Cross-Encoder reranking of the original BGE paper best chunks."""

import math
import time
from functools import lru_cache
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

from src.citation_grounding import verify_evidence_source
from src.rag_service import prepare_grounded_evidence


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models/bge-reranker-base"


@lru_cache(maxsize=2)
def load_reranker(device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the local reranker")
    if device not in ("cuda", "cpu"):
        raise ValueError("device must be cuda or cpu")
    if not (MODEL_DIR / "model.safetensors").is_file():
        raise FileNotFoundError(f"Local reranker weights missing: {MODEL_DIR}")
    return CrossEncoder(str(MODEL_DIR), device=device, local_files_only=True, max_length=512)


def rerank(query, candidates, *, model=None, batch_size=4):
    """Return ranked original chunks and measured inference time; never alter text."""
    if not query or not query.strip():
        raise ValueError("query is empty")
    if not candidates:
        return [], 0.0
    if len(candidates) > 10:
        raise ValueError("at most 10 paper best chunks are allowed")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len({row["doc_id"] for row in candidates}) != len(candidates) or \
       len({row["chunk_id"] for row in candidates}) != len(candidates):
        raise ValueError("candidate Document/Chunk IDs must be unique")
    if any(not row["chunk_text"].strip() for row in candidates):
        raise ValueError("empty chunk text cannot be reranked")
    model = model if model is not None else load_reranker()
    pairs = [(query, row["chunk_text"]) for row in candidates]
    if torch.cuda.is_available() and str(model.device).startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    if torch.cuda.is_available() and str(model.device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if len(scores) != len(candidates) or any(not math.isfinite(float(score)) for score in scores):
        raise ValueError("reranker returned invalid scores")
    rows = [{"chunk_id": row["chunk_id"], "doc_id": row["doc_id"],
             "original_rank": row["branch_rank"], "original_bge_score": row["bge_score"],
             "reranker_score": float(score), "chunk_text": row["chunk_text"],
             "title": row["title"], "doi": row["doi"], "original_uid": row["original_uid"],
             "source_path": row["source_path"], "source_spans": row["source_spans"],
             "section_title": row["section_title"], "section_kind": row["section_kind"],
             "chunk_index": row["chunk_index"]}
            for row, score in zip(candidates, scores)]
    rows.sort(key=lambda row: (-row["reranker_score"], row["original_rank"], row["doc_id"]))
    for index, row in enumerate(rows, 1):
        row["reranker_rank"] = index
    return rows, elapsed


def make_evidence(selected, original_candidates, original_chunks):
    """Map re-ordered candidates by IDs, then assign citations and check source."""
    if len(selected) > 3:
        raise ValueError("final evidence budget is three chunks")
    original = {(row["doc_id"], row["chunk_id"]): row for row in original_candidates}
    evidence = []
    source_map = {}
    for row in selected:
        key = (row["doc_id"], row["chunk_id"])
        source = original.get(key)
        if source is None:
            raise ValueError("selected Document/Chunk pair absent from original candidates")
        chunk = original_chunks[source["chunk_index"]]
        for field, value in (("doc_id", source["doc_id"]), ("chunk_id", source["chunk_id"]),
                             ("title", source["title"]), ("text", source["chunk_text"]),
                             ("doi", source["doi"]), ("source_path", source["source_path"]),
                             ("source_spans", source["source_spans"]),
                             ("section_title", source["section_title"])):
            if chunk[field] != value:
                raise ValueError(f"original Chunk {field} mismatch")
        source_map[source["chunk_id"]] = chunk
        evidence.append({"doc_id": source["doc_id"], "chunk_id": source["chunk_id"],
                         "title": source["title"], "doi": source["doi"], "text": source["chunk_text"],
                         "score": source["bge_score"], "source_path": source["source_path"],
                         "source_spans": source["source_spans"], "section_title": source["section_title"],
                         "section_kind": source["section_kind"],
                         "paper_rank": source["branch_rank"]})
    prepared = prepare_grounded_evidence(evidence)
    checks = verify_evidence_source(prepared, source_map)
    if any(check["status"] != "valid" for check in checks):
        raise ValueError("Evidence source mapping invalid")
    return prepared, checks
