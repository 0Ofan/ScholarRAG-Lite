"""Run at most one paid decomposition request for each approved pilot question."""

import json
from pathlib import Path

from src.deepseek_client import DeepSeekClient
from src.phase424.query_decomposition import decompose


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/results/phase424/query_decomposition"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def peak_cost(usage):
    if not usage or usage.get("input_tokens") is None or usage.get("output_tokens") is None:
        return None
    return (2 * usage["input_tokens"] + 8 * usage["output_tokens"]) / 1_000_000


def main():
    budget = json.loads((OUT / "budget_estimate.json").read_text(encoding="utf-8"))
    questions = budget["questions"]
    if len(questions) != 6:
        raise ValueError("pilot must contain exactly six frozen questions")
    frozen = {q["question_id"]: q for q in json.loads(
        (ROOT / "experiments/configs/phase36_questions.json").read_text(encoding="utf-8"))["questions"]}
    for row in questions:
        if frozen[row["question_id"]]["question"] != row["question"]:
            raise ValueError("pilot question differs from frozen input")
    save(OUT / "pilot_queries.json", questions)
    path = OUT / "generated_subqueries.json"
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    completed = {row["question_id"] for row in records}
    client = DeepSeekClient()
    for row in questions:
        if row["question_id"] in completed:
            continue
        spent = sum(peak_cost(item.get("usage")) or 0 for item in records)
        if spent + (2 * 400 + 8 * 200) / 1_000_000 > budget["estimated_cost_cny"]["proposed_approval_ceiling"]:
            raise RuntimeError("estimated peak budget ceiling reached")
        result = decompose(row["question"], client)
        records.append({"question_id": row["question_id"], **result,
                        "estimated_peak_cost_cny": peak_cost(result["usage"])})
        save(path, records)
        print(row["question_id"], result["status"], "finish_reason=", result["finish_reason"], flush=True)
        if result["status"] != "ok":
            break
    save(OUT / "api_usage.json", {
        "pricing_basis": "DeepSeek official peak CNY rates checked 2026-11-17; input assumed cache miss",
        "requests_recorded": len(records),
        "input_tokens": sum((row.get("usage") or {}).get("input_tokens") or 0 for row in records),
        "output_tokens": sum((row.get("usage") or {}).get("output_tokens") or 0 for row in records),
        "estimated_peak_cost_cny": sum(row["estimated_peak_cost_cny"] or 0 for row in records),
        "unknown_usage_count": sum(row.get("usage") is None for row in records),
    })


if __name__ == "__main__":
    main()
