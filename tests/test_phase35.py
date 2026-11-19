"""Focused Phase 3.5 tests using a fake DeepSeek stream."""
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

from src.deepseek_client import DeepSeekClient, DeepSeekRequestError, is_complete_generation
from src.rag_service import retrieve_evidence, validate_citations


class Phase35Tests(unittest.TestCase):
    def test_completion_requires_stop_and_nonempty_text(self):
        for reason in ('length', 'content_filter', 'aborted', 'tool_calls',
                       'insufficient_system_resource', 'unexpected', None):
            with self.subTest(reason=reason):
                self.assertFalse(is_complete_generation('partial', {'finish_reason': reason, 'usage': {'input_tokens': 1}}))
        self.assertFalse(is_complete_generation('', {'finish_reason': 'stop'}))
        self.assertFalse(is_complete_generation('  ', {'finish_reason': 'stop'}))
        self.assertFalse(is_complete_generation('partial', None))
        self.assertTrue(is_complete_generation('answer', {'finish_reason': 'stop'}))

    def test_sdk_stream_preserves_terminal_reason_and_usage(self):
        evidence = [{'citation': '[E1]', 'title': 'Paper', 'doc_id': 'd1',
                     'chunk_id': 'c1', 'text': 'Evidence'}]
        for reason in ('stop', 'length', 'content_filter', 'aborted', None):
            with self.subTest(reason=reason):
                sdk = Mock()
                sdk.chat.completions.create.return_value = [
                    NS(model='deepseek-flash', usage=None,
                       choices=[NS(delta=NS(content='partial', reasoning_content=None), finish_reason=None)]),
                    NS(model='deepseek-flash', usage=NS(prompt_tokens=5, completion_tokens=1, total_tokens=6),
                       choices=[NS(delta=NS(content=None, reasoning_content=None), finish_reason=reason)]),
                ]
                service = DeepSeekClient(client=sdk, config={'api_key': 'fake', 'model': 'deepseek-flash',
                                                               'max_tokens': 512, 'timeout': 30})
                events = list(service.stream('question', evidence))
                metadata = events[-1][1]
                self.assertEqual(metadata['finish_reason'], reason)
                self.assertEqual(metadata['usage']['input_tokens'], 5)
                self.assertEqual(is_complete_generation('partial', metadata), reason == 'stop')

    def test_sdk_stream_read_error_after_partial_text(self):
        sdk = Mock()
        def broken_stream():
            yield NS(model='deepseek-flash', usage=None,
                     choices=[NS(delta=NS(content='partial', reasoning_content=None), finish_reason=None)])
            raise OSError('network lost')
        sdk.chat.completions.create.return_value = broken_stream()
        service = DeepSeekClient(client=sdk, config={'api_key': 'fake', 'model': 'deepseek-flash',
                                                       'max_tokens': 512, 'timeout': 30})
        events = []
        with self.assertRaises(DeepSeekRequestError):
            for event in service.stream('question', [{'citation': '[E1]', 'title': 'Paper',
                'doc_id': 'd1', 'chunk_id': 'c1', 'text': 'Evidence'}]):
                events.append(event)
        self.assertEqual(events, [('text', 'partial')])

    def test_three_streamlit_pages_render(self):
        from streamlit.testing.v1 import AppTest
        app_path = Path(__file__).resolve().parents[1] / 'app.py'
        app = AppTest.from_file(str(app_path), default_timeout=120).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.sidebar.button), 3)
        app.sidebar.button[1].click().run()
        self.assertFalse(app.exception)
        app.text_input[0].set_value('coronavirus respiratory symptoms').run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(item.label == 'Shared papers' for item in app.metric))
        app.sidebar.button[2].click().run()
        self.assertFalse(app.exception)

    def test_streaming_non_thinking_usage_and_content(self):
        client = Mock()
        client.chat.completions.create.return_value = [
            NS(model='deepseek-flash', usage=None,
               choices=[NS(delta=NS(content=' ', reasoning_content=None), finish_reason=None)]),
            NS(model='deepseek-flash', usage=None,
               choices=[NS(delta=NS(content='证据显示', reasoning_content=None), finish_reason=None)]),
            NS(model='deepseek-flash', usage=NS(prompt_tokens=45, completion_tokens=12, total_tokens=57),
               choices=[NS(delta=NS(content=None, reasoning_content=None), finish_reason='stop')]),
        ]
        service = DeepSeekClient(client=client, config={
            'api_key': 'fake', 'base_url': 'https://api.deepseek.com',
            'model': 'deepseek-flash', 'max_tokens': 512, 'timeout': 30,
        })
        evidence = [{'citation': '[1]', 'title': 'Paper', 'doc_id': 'd1',
                     'chunk_id': 'c1', 'text': 'Academic evidence.'}]
        events = list(service.stream('question', evidence))
        self.assertEqual(events[0], ('text', ' '))
        self.assertEqual(events[1], ('text', '证据显示'))
        self.assertEqual(events[-1][1]['usage']['input_tokens'], 45)
        self.assertFalse(events[-1][1]['reasoning_content_seen'])
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs['model'], 'deepseek-flash')
        self.assertEqual(kwargs['extra_body'], {'thinking': {'type': 'disabled'}})
        self.assertTrue(kwargs['stream'])
        self.assertTrue(kwargs['stream_options']['include_usage'])
        self.assertEqual(kwargs['temperature'], .3)

    def test_api_error_does_not_expose_secret(self):
        client = Mock()
        client.chat.completions.create.side_effect = RuntimeError('secret-value')
        service = DeepSeekClient(client=client, config={
            'api_key': 'secret-value', 'base_url': 'https://api.deepseek.com',
            'model': 'deepseek-flash', 'max_tokens': 512, 'timeout': 30,
        })
        evidence = [{'citation': '[1]', 'title': 'Paper', 'doc_id': 'd1',
                     'chunk_id': 'c1', 'text': 'Evidence'}]
        with self.assertRaises(DeepSeekRequestError) as caught:
            list(service.stream('question', evidence))
        self.assertNotIn('secret-value', str(caught.exception))

    def test_retrieval_evidence_and_citation_validation(self):
        docs = [{'doc_id': str(i), 'title': f'Paper {i}', 'best_chunk_id': f'chunk{i}',
                 'best_chunk_text': f'Evidence {i}', 'score': .8,
                 'source_path': f'{i}.json', 'source_spans': []} for i in range(10)]
        retriever = Mock()
        retriever.encode_queries.return_value = ([[0.0] * 384], .02)
        retriever.retrieve_vector.return_value = (docs, {'retrieval_seconds': .04})
        evidence, returned, timing = retrieve_evidence(retriever, 'question', 3)
        self.assertEqual([e['citation'] for e in evidence], ['[1]', '[2]', '[3]'])
        self.assertEqual(len(returned), 10)
        self.assertEqual(timing['query_embedding_seconds'], .02)
        self.assertEqual(validate_citations('Answer [1] [9]', evidence)['invalid'], ['[9]'])


if __name__ == '__main__':
    unittest.main()
