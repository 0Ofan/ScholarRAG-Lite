"""Focused checks for the frozen Phase 3.6 run and published-price arithmetic."""
import hashlib
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from src.deepseek_client import DeepSeekClient
from src.phase36_run import cost_estimate, main as run_main
from src.rag_service import validate_citations

ROOT = Path(__file__).resolve().parents[1]


class Phase36Tests(unittest.TestCase):
    def test_completed_run_does_not_reload_or_repeat_api(self):
        with patch('src.phase36_run.load_retriever') as load:
            run_main()
            load.assert_not_called()

    def test_stream_returns_actual_cache_usage_for_cost(self):
        sdk = Mock()
        sdk.chat.completions.create.return_value = [
            NS(model='deepseek-flash', choices=[NS(delta=NS(content='回答[1]', reasoning_content=None),
                                                 finish_reason='stop')], usage=None),
            NS(model='deepseek-flash', choices=[], usage=NS(prompt_tokens=600, completion_tokens=100,
                total_tokens=700, prompt_cache_hit_tokens=100, prompt_cache_miss_tokens=500)),
        ]
        client = DeepSeekClient(client=sdk, config={'api_key': 'test-only',
            'base_url': 'https://api.deepseek.com', 'model': 'deepseek-flash',
            'temperature': .3, 'max_tokens': 512, 'timeout': 30})
        events = list(client.stream('question', [{'citation': '[1]', 'title': 'Paper',
            'doc_id': 'd1', 'chunk_id': 'c1', 'text': 'evidence'}]))
        self.assertEqual(events[-1][1]['usage']['prompt_cache_hit_tokens'], 100)
        self.assertEqual(events[-1][1]['usage']['prompt_cache_miss_tokens'], 500)
        self.assertTrue(sdk.chat.completions.create.call_args.kwargs['stream'])

    def test_peak_price_uses_actual_cache_split(self):
        usage = {'input_tokens': 600, 'output_tokens': 100,
                 'prompt_cache_hit_tokens': 100, 'prompt_cache_miss_tokens': 500}
        at = datetime(2026, 11, 11, 1, 16, tzinfo=timezone.utc)
        result = cost_estimate(usage, at)
        self.assertEqual(result['pricing_period'], 'peak')
        self.assertAlmostEqual(result['estimated_cny'], (100*.04+500*2+100*8)/1_000_000)

    def test_missing_cache_usage_is_not_priced_as_real(self):
        result = cost_estimate({'input_tokens': 600, 'output_tokens': 100},
                               datetime(2026, 11, 11, 1, tzinfo=timezone.utc))
        self.assertEqual(result['status'], 'unavailable')

    def test_frozen_questions_and_real_outputs(self):
        raw = (ROOT/'experiments/configs/phase36_questions.json').read_bytes()
        lock = json.loads((ROOT/'experiments/configs/phase36_question_lock.json').read_text(encoding='utf8'))
        digest = hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()
        self.assertEqual(digest, lock['sha256'])
        questions = json.loads(raw)['questions']
        self.assertEqual(len(questions), 20)
        self.assertEqual(len({q['question_id'] for q in questions}), 20)
        for question in questions:
            with self.subTest(question=question['question_id']):
                record = json.loads((ROOT/'experiments/results/phase36'/
                                     f"{question['question_id']}.json").read_text(encoding='utf8'))
                self.assertEqual(record['status'], 'success')
                self.assertTrue(record['real_api'])
                self.assertEqual(record['question_set_sha256'], digest)
                self.assertEqual(len(record['evidence']), 3)
                self.assertEqual([e['doc_id'] for e in record['evidence']],
                                 [d['doc_id'] for d in record['retrieval_top10'][:3]])
                self.assertEqual(validate_citations(record['answer'], record['evidence']),
                                 record['citation_check'])
                self.assertEqual(record['generation']['response_model'], 'deepseek-flash')


if __name__ == '__main__':
    unittest.main()
