"""One paid DeepSeek request to turn a complex question into retrieval queries."""

import json
import time

from src.deepseek_client import DeepSeekClient


SYSTEM = (
    "Return only a JSON object like {\"subqueries\": [\"query 1\", \"query 2\"]}. "
    "Write 2 to 4 distinct English academic retrieval queries for the user's question. "
    "Each subquery must target a research dimension explicitly requested in the original question. "
    "Split complex dimensions without expanding the research scope. Keep the same population, age range, "
    "timeframe and comparison conditions. Do not add disease outcomes, long-term results, mechanisms, "
    "mediators or study populations unless the original question explicitly asks for them. "
    "Do not turn background knowledge into a new retrieval objective. Make subqueries complementary, "
    "not near-duplicates. Do not assume findings or answer the question. The original query is searched separately."
)


def parse_subqueries(content, original):
    data = json.loads(content)
    values = data.get("subqueries") if isinstance(data, dict) else None
    if not isinstance(values, list) or not 2 <= len(values) <= 4:
        raise ValueError("expected 2–4 subqueries")
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise ValueError("empty or non-text subquery")
    cleaned = [item.strip() for item in values]
    normalized = [item.casefold() for item in cleaned]
    if len(set(normalized)) != len(normalized) or original.strip().casefold() in normalized:
        raise ValueError("duplicate query")
    return cleaned


def decompose(question, client=None):
    original = question.strip()
    if not original:
        raise ValueError("original query is empty")
    result = {"original_query": original, "subqueries": [], "generation_model": None,
              "finish_reason": None, "usage": None, "latency_seconds": None,
              "status": "api_error", "error": None, "raw_json_content": None}
    start = time.perf_counter()
    try:
        client = client or DeepSeekClient()
        result["generation_model"] = client.config["model"]
        response = client.client.chat.completions.create(
            model=client.config["model"],
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": f"Original research question: {original}"}],
            response_format={"type": "json_object"},
            max_tokens=200,
            temperature=client.config.get("temperature", 0.3),
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )
        choice = response.choices[0] if response.choices else None
        result["finish_reason"] = getattr(choice, "finish_reason", None)
        usage = getattr(response, "usage", None)
        if usage is not None:
            result["usage"] = {
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
                "prompt_cache_hit_tokens": getattr(usage, "prompt_cache_hit_tokens", None),
                "prompt_cache_miss_tokens": getattr(usage, "prompt_cache_miss_tokens", None),
            }
        content = getattr(getattr(choice, "message", None), "content", None)
        result["raw_json_content"] = content
        if result["finish_reason"] != "stop":
            result["status"] = "incomplete"
            result["error"] = "finish_reason is not stop"
        elif not content or not content.strip():
            result["status"] = "invalid_json"
            result["error"] = "empty JSON content"
        else:
            try:
                result["subqueries"] = parse_subqueries(content, original)
                result["status"] = "ok"
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                result["status"] = "invalid_json"
                result["error"] = type(exc).__name__
    except Exception as exc:
        # Never persist provider response bodies, headers, or credentials.
        result["error"] = type(exc).__name__
    result["latency_seconds"] = time.perf_counter() - start
    return result
