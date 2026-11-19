"""Phase 4.24.1 local-only smoke test for the official BGE reranker."""

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
from sentence_transformers import CrossEncoder


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "experiments/configs/phase424/reranker_smoke.json").read_text(encoding="utf-8"))
MODEL = ROOT / CONFIG["model_dir"]
OUTPUT = ROOT / "experiments" / "results" / "phase424" / "reranker_smoke_test"
REVISION = CONFIG["revision"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def real_pairs():
    rows = []
    sources = []
    for question_id in CONFIG["question_ids"]:
        source = ROOT / "experiments" / "results" / "phase36" / f"{question_id}.json"
        record = json.loads(source.read_text(encoding="utf-8"))
        sources.append({"path": str(source.relative_to(ROOT)), "sha256": sha256(source)})
        for hit in record["retrieval_top10"][:CONFIG["hits_per_question"]]:
            text = hit["best_chunk_text"]
            rows.append({
                "question_id": question_id,
                "question": record["question"],
                "doc_id": hit["doc_id"],
                "chunk_id": hit["best_chunk_id"],
                "chunk_text": text,
                "chunk_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            })
    assert len(rows) == len(CONFIG["question_ids"]) * CONFIG["hits_per_question"] == 10
    return rows, sources


def predict(model, inputs, batch_size):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    scores = model.predict(inputs, batch_size=batch_size, show_progress_bar=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return [float(score) for score in scores], time.perf_counter() - start


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, sources = real_pairs()
    files = [path for path in sorted(MODEL.iterdir()) if path.is_file()]
    if not files or not any(path.name.endswith(".safetensors") for path in files):
        raise FileNotFoundError("Official reranker weights are missing")

    result = {
        "status": "running",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_repo": CONFIG["model_repo"],
        "revision": REVISION,
        "config_sha256": sha256(ROOT / "experiments/configs/phase424/reranker_smoke.json"),
        "model_dir": str(MODEL.relative_to(ROOT)),
        "model_files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files],
        "input_sources": sources,
        "inputs": [{key: value for key, value in row.items() if key != "chunk_text"} for row in rows],
        "environment": {
            "python": __import__("sys").version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        },
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; GPU smoke test cannot run")
        torch.cuda.reset_peak_memory_stats()
        before_cpu = psutil.Process().memory_info().rss
        start = time.perf_counter()
        model = CrossEncoder(str(MODEL), device="cuda", local_files_only=True,
                             max_length=CONFIG["max_length"])
        torch.cuda.synchronize()
        result["cold_load_seconds"] = time.perf_counter() - start
        result["cpu_rss_before_bytes"] = before_cpu
        result["cpu_rss_after_load_bytes"] = psutil.Process().memory_info().rss
        result["vram_after_load_bytes"] = torch.cuda.memory_allocated()

        pairs = [(row["question"], row["chunk_text"]) for row in rows]
        single, single_seconds = predict(model, pairs[:1], 1)
        result["single_input"] = {"score": single[0], "seconds": single_seconds}
        result["batches"] = []
        for batch_size in CONFIG["batch_sizes"]:
            timings = []
            scores = None
            for _ in range(CONFIG["repeat_count"]):
                scores, elapsed = predict(model, pairs, batch_size)
                timings.append(elapsed)
            result["batches"].append({
                "batch_size": batch_size,
                "pair_count": len(pairs),
                "run_seconds": timings,
                "mean_total_seconds": sum(timings) / len(timings),
                "mean_seconds_per_pair": sum(timings) / (len(timings) * len(pairs)),
                "scores": scores,
                "ranking": [rows[i]["chunk_id"] for i in sorted(range(len(rows)), key=lambda i: (-scores[i], rows[i]["chunk_id"]))],
            })

        empty_scores, empty_seconds = predict(model, [(rows[0]["question"], "")], 1)
        long_text = rows[0]["chunk_text"] * 8
        raw_tokens = len(model.tokenizer(rows[0]["question"], long_text, truncation=False)["input_ids"])
        long_scores, long_seconds = predict(model, [(rows[0]["question"], long_text)], 1)
        result["edge_cases"] = {
            "empty_chunk": {"score": empty_scores[0], "seconds": empty_seconds, "finite": math.isfinite(empty_scores[0])},
            "long_chunk": {"raw_input_tokens": raw_tokens, "model_max_length": model.max_seq_length,
                           "score": long_scores[0], "seconds": long_seconds, "finite": math.isfinite(long_scores[0])},
        }
        try:
            predict(model, [(rows[0]["question"],)], 1)
        except Exception as exc:
            result["edge_cases"]["invalid_pair"] = {"rejected": True, "exception_type": type(exc).__name__}
        else:
            result["edge_cases"]["invalid_pair"] = {"rejected": False}

        result["vram_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["vram_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        result["cpu_rss_peak_observed_bytes"] = psutil.Process().memory_info().rss
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        raise
    finally:
        (OUTPUT / "smoke_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "output": str(OUTPUT / "smoke_results.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
