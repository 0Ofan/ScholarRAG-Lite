"""One real, bounded end-to-end API check; never save the key or answer text."""
import json
import argparse
import time
from pathlib import Path

from src.deepseek_client import DeepSeekClient, DeepSeekRequestError, MissingAPIKey
from src.rag_service import load_retriever, retrieve_evidence, validate_citations

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/phase35'
OUT.mkdir(parents=True, exist_ok=True)
QUESTION = 'What evidence links COVID-19 infection to respiratory symptoms?'
parser = argparse.ArgumentParser()
parser.add_argument('--warm', action='store_true')
args = parser.parse_args()
start = time.perf_counter()
result = {'version': 'phase35-real-smoke-v1', 'real_api': True,
          'query': QUESTION, 'run_id': 'phase33_large_fixed',
          'measurement': 'warm' if args.warm else 'cold'}
try:
    retriever, cold = load_retriever('phase33_large_fixed')
    if args.warm:
        start = time.perf_counter()
    evidence, _, retrieval = retrieve_evidence(retriever, QUESTION, 3)
    result.update(cold_load_seconds=cold, **retrieval)
    client = DeepSeekClient()
    first = None
    fragments = []
    for kind, payload in client.stream(QUESTION, evidence, max_tokens=160):
        if kind == 'text':
            if first is None:
                first = time.perf_counter() - start
            fragments.append(payload)
        else:
            result.update(payload)
    answer = ''.join(fragments)
    result['answer_characters'] = len(answer)
    result['citation_check'] = validate_citations(answer, evidence)
    result['submit_to_first_text_seconds'] = first
    result['submit_to_complete_seconds'] = time.perf_counter() - start
    result['status'] = 'success' if answer else 'empty_response'
except (MissingAPIKey, DeepSeekRequestError) as exc:
    result['status'] = 'error'
    result['error'] = str(exc)
    result['submit_to_complete_seconds'] = time.perf_counter() - start
(OUT / ('warm_api_smoke_with_temperature.json' if args.warm else 'real_api_smoke.json')).write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps({k: result.get(k) for k in ('status', 'error', 'cold_load_seconds',
    'query_embedding_seconds', 'retrieval_seconds', 'evidence_seconds',
    'api_ttft_seconds', 'api_total_seconds', 'submit_to_first_text_seconds',
    'submit_to_complete_seconds', 'usage', 'thinking_requested_disabled',
    'reasoning_content_seen', 'citation_check')}, ensure_ascii=False, indent=2))
