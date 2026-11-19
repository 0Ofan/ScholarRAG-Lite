"""Focused UI flow tests with fake Streamlit and API; no browser or real requests."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import app
from src.deepseek_client import DeepSeekRequestError


class Slot:
    def __init__(self):
        self.markdowns = []
        self.cleared = False

    def markdown(self, value):
        self.markdowns.append(value)

    def info(self, value):
        pass

    def empty(self):
        self.cleared = True


class Tab:
    def __init__(self, is_open):
        self.open = is_open

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class FakeStreamlit:
    def __init__(self):
        self.session_state = {}
        self.question = ''
        self.variant = next(iter(app.VARIANTS))
        self.use_v2 = False
        self.submitted = False
        self.messages = []
        self.slots = []
        self.expanders = []
        self.mode_defaults = []
        self.sidebar = self
        self.nav_click = None
        self.compare_click = False
        self.tables = []
        self.metrics = []
        self.charts = []
        self.active_tab = 'Retrieval quality'
        self.tab_options = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def form(self, *_):
        return self

    def container(self, *_args, **_kwargs):
        return self

    def spinner(self, *_):
        return self

    def expander(self, label, **kwargs):
        self.expanders.append((label, kwargs.get('expanded')))
        return self

    def empty(self):
        slot = Slot()
        self.slots.append(slot)
        return slot

    def columns(self, count, **_kwargs):
        return [self] * (count if isinstance(count, int) else len(count))

    def text_input(self, *_args, **_kwargs):
        return self.question

    def text_area(self, *_args, **_kwargs):
        return self.question

    def selectbox(self, label, options, **_kwargs):
        return self.variant if label == 'Retrieval method' else options[0]

    def select_slider(self, *_args, **_kwargs):
        return 3

    def slider(self, *_args, **_kwargs):
        return 5

    def tabs(self, names, **_kwargs):
        self.tab_options.append((list(names), _kwargs))
        return [Tab(True) for _ in names]

    def dataframe(self, value, **_kwargs):
        self.tables.append(value)

    def line_chart(self, *_args, **_kwargs):
        pass

    def bar_chart(self, *_args, **_kwargs):
        pass

    def altair_chart(self, chart, **_kwargs):
        self.charts.append(chart)

    def vega_lite_chart(self, *args, spec=None, **_kwargs):
        self.charts.append(spec)

    def set_page_config(self, **_kwargs):
        pass

    def segmented_control(self, *_args, **kwargs):
        self.mode_defaults.append(kwargs['default'])
        return 'V2 — Experimental' if self.use_v2 else 'V1 — Stable Default'

    def divider(self):
        pass

    def button(self, _label, **kwargs):
        if kwargs.get('on_click') and kwargs['args'][0] == self.nav_click:
            kwargs['on_click'](*kwargs['args'])
            self.nav_click = None
            return True
        return self.compare_click

    def form_submit_button(self, *_args, **_kwargs):
        return self.submitted

    def metric(self, label, value, **_kwargs):
        self.metrics.append((label, value))

    def __getattr__(self, name):
        if name in {'title', 'write', 'subheader', 'caption', 'info', 'error',
                    'warning', 'markdown', 'text'}:
            return lambda value, **_kwargs: self.messages.append((name, str(value)))
        raise AttributeError(name)


class FakeClient:
    calls = []
    fail = False
    finish_reason = 'stop'
    empty = False
    omit_done = False

    def stream(self, question, evidence, *, system_prompt=None):
        self.calls.append({'question': question, 'prompt': system_prompt,
                           'evidence_ids': [e['evidence_id'] for e in evidence]})
        yield 'text', '' if self.empty else f'论文证据回答 {evidence[0]["doc_id"]} [E1]。'
        if self.fail:
            raise DeepSeekRequestError('DeepSeek 请求失败（TimeoutError）。')
        if self.omit_done:
            return
        yield 'done', {'api_ttft_seconds': .1, 'api_total_seconds': .2,
                       'usage': {'input_tokens': 10, 'output_tokens': 8},
                       'reasoning_content_seen': False, 'finish_reason': self.finish_reason}


class Phase412Tests(unittest.TestCase):
    def setUp(self):
        self.ui = FakeStreamlit()
        self.client = FakeClient()
        self.patches = [patch.object(app, 'st', self.ui),
                        patch.object(app, 'cached_retriever', side_effect=self.retriever),
                        patch.object(app, 'retrieve_evidence', side_effect=self.retrieve),
                        patch.object(app, 'DeepSeekClient', return_value=self.client)]
        self.active_mocks = [item.start() for item in self.patches]
        self.addCleanup(lambda: [item.stop() for item in reversed(self.patches)])

    def retriever(self, *_):
        doc = self.ui.question.replace(' ', '_')
        return SimpleNamespace(chunks=[{'doc_id': doc, 'chunk_id': doc + ':1',
                                        'title': 'Paper ' + doc, 'text': 'Study evidence for ' + doc}])

    def retrieve(self, retriever, *_):
        chunk = retriever.chunks[0]
        evidence = [dict(chunk, score=.9, source_path='synthetic.json', source_spans=[])]
        return evidence, [{'chunk_index': 0}], {'query_embedding_seconds': .01,
              'retrieval_seconds': .02, 'evidence_seconds': .001, 'retrieval_total_seconds': .031}

    def test_default_v1_then_v2_two_questions_and_mode_isolation(self):
        self.ui.question = 'first question'
        self.ui.submitted = True
        app.search_page()
        self.assertEqual(self.ui.messages[0], ('title', 'Research Assistant'))
        first = self.ui.session_state['last_answer']
        self.assertFalse(first['use_v2'])
        self.assertEqual(first['status'], 'complete')
        self.assertIsNone(self.client.calls[0]['prompt'])
        self.assertEqual(self.ui.mode_defaults[0], 'V1 — Stable Default')
        self.ui.question = 'second question'
        self.ui.use_v2 = True
        app.search_page()
        second = self.ui.session_state['last_answer']
        self.assertTrue(second['use_v2'])
        self.assertIn('你是学术证据问答助手', self.client.calls[1]['prompt'])
        self.assertEqual(len(self.ui.session_state['answer_history']), 2)
        self.assertIs(self.ui.session_state['answer_history'][0], first)
        self.assertNotEqual(first['evidence'][0]['doc_id'], second['evidence'][0]['doc_id'])
        self.assertEqual(second['evidence'][0]['evidence_id'], 'E1')
        self.assertIsNotNone(second['verification'])
        self.assertIn(('Verification Details', False), self.ui.expanders)
        self.ui.submitted = False
        self.ui.use_v2 = False
        self.ui.messages.clear()
        app.search_page()
        self.assertFalse(any('上次提交：' in text for _, text in self.ui.messages))

    def test_stream_interruption_keeps_partial_answer_and_retry_recovers(self):
        self.ui.question = 'first question'
        self.ui.submitted = True
        self.client.fail = True
        app.search_page()
        first = self.ui.session_state['last_answer']
        self.assertEqual(first['status'], 'interrupted')
        self.assertIn('DeepSeek 请求失败', first['error'])
        self.assertNotIn('▌', self.ui.slots[-2].markdowns[-1])
        self.client.fail = False
        self.ui.question = 'second question'
        app.search_page()
        self.assertEqual(self.ui.session_state['last_answer']['status'], 'complete')
        self.assertEqual(len(self.ui.session_state['answer_history']), 2)

    def test_nonstop_and_empty_streams_keep_history_without_verification(self):
        self.ui.submitted = True
        self.ui.use_v2 = True
        for reason in ('length', 'content_filter', 'aborted', 'unexpected', None, 'stop'):
            with self.subTest(reason=reason):
                self.ui.question = f'question {reason}'
                self.client.finish_reason = reason
                self.client.empty = reason == 'stop'
                app.search_page()
                record = self.ui.session_state['last_answer']
                self.assertEqual(record['status'], 'interrupted')
                self.assertIsNone(record['verification'])
                self.assertIs(record, self.ui.session_state['answer_history'][-1])
                self.assertTrue(any('未完整生成' in message for kind, message in self.ui.messages if kind == 'warning'))
        self.ui.question = 'no done'
        self.client.empty = False
        self.client.omit_done = True
        app.search_page()
        self.assertEqual(self.ui.session_state['last_answer']['status'], 'interrupted')

    def test_v1_length_keeps_partial_answer(self):
        self.ui.question = 'v1 truncated'
        self.ui.submitted = True
        self.client.finish_reason = 'length'
        app.search_page()
        record = self.ui.session_state['last_answer']
        self.assertFalse(record['use_v2'])
        self.assertEqual(record['status'], 'interrupted')
        self.assertIn('论文证据回答', record['answer'])
        self.assertEqual(record['generation']['finish_reason'], 'length')
        self.assertIsNone(record['verification'])

    def test_verifier_failure_preserves_answer(self):
        self.ui.question = 'test question'
        self.ui.use_v2 = True
        self.ui.submitted = True
        with patch.object(app, 'verify', side_effect=RuntimeError('internal detail')):
            app.search_page()
        record = self.ui.session_state['last_answer']
        self.assertEqual(record['status'], 'complete')
        self.assertIn('论文证据回答', record['answer'])
        self.assertIsNone(record['verification'])
        self.assertIn('本地证据核查失败', record['warning'])
        self.assertNotIn('internal detail', record['warning'])

    def test_refinement_failure_keeps_initial_answer(self):
        self.ui.question = 'test question'
        self.ui.use_v2 = True
        self.ui.submitted = True
        check = {'critical': True, 'seconds': .001, 'claims': []}
        with patch.object(app, 'verify', return_value=check), patch.object(
                app, 'refine_once', side_effect=DeepSeekRequestError('secret response')):
            app.search_page()
        record = self.ui.session_state['last_answer']
        self.assertIn('论文证据回答', record['answer'])
        self.assertIs(record['verification'], check)
        self.assertIn('定向修正失败', record['warning'])
        self.assertNotIn('secret response', record['warning'])

    def test_sidebar_routes_to_all_three_pages_and_dashboard_data_loads(self):
        app.render_app()
        self.assertEqual(self.ui.session_state['active_page'], 'Search & Answer')
        self.ui.nav_click = 'Method Comparison'
        app.render_app()
        self.assertEqual(self.ui.session_state['active_page'], 'Method Comparison')
        self.ui.nav_click = 'Experiment Dashboard'
        app.render_app()
        self.assertEqual(self.ui.session_state['active_page'], 'Experiment Dashboard')
        self.assertEqual(self.active_mocks[1].call_count, 0)
        self.assertEqual(self.active_mocks[3].call_count, 0)
        self.assertEqual(len(self.ui.tables), 3)
        self.assertEqual(len(self.ui.tables[0]), 6)
        self.assertTrue(any(label == 'Large · Fixed nDCG@10' for label, _ in self.ui.metrics))
        self.assertNotIn('on_change', self.ui.tab_options[-1][1])
        spec = self.ui.charts[0]
        self.assertEqual(spec['encoding']['x']['sort'], ['A', 'B', 'C'])
        self.assertEqual(spec['encoding']['x']['axis']['labelAngle'], 0)
        self.assertEqual(spec['encoding']['x']['axis']['labelFontSize'], 16)
        self.assertEqual(spec['encoding']['x']['axis']['labelPadding'], 12)
        history = [{'question': 'Previous question', 'use_v2': False,
                    'variant': 'Fixed-length (Large)', 'answer': 'Previous answer [E1]',
                    'evidence': [{'citation': '[E1]', 'doc_id': 'prior', 'title': 'Paper'}]}]
        self.ui.session_state['answer_history'] = history
        for _ in range(3):
            for page in ('Search & Answer', 'Method Comparison', 'Experiment Dashboard'):
                self.ui.nav_click = page
                app.render_app()
        self.assertIs(self.ui.session_state['answer_history'], history)
        self.assertEqual(self.active_mocks[1].call_count, 0)
        self.assertEqual(self.active_mocks[3].call_count, 0)

    def test_dashboard_cache_reuses_data_and_invalidates_on_file_change(self):
        app.dashboard_data.clear()
        phase33 = app.ROOT / 'experiments/results/phase33/comparison.json'
        phase34 = app.ROOT / 'experiments/results/phase34_ablation/comparison.json'
        sig33 = (phase33.stat().st_mtime_ns, phase33.stat().st_size)
        sig34 = (phase34.stat().st_mtime_ns, phase34.stat().st_size)
        reads = []
        original = Path.read_text

        def tracked(path, *args, **kwargs):
            reads.append(path)
            return original(path, *args, **kwargs)

        with patch.object(Path, 'read_text', tracked):
            first = app.dashboard_data(sig33, sig34)
            second = app.dashboard_data(sig33, sig34)
            refreshed = app.dashboard_data((sig33[0] + 1, sig33[1]), sig34)
        self.assertEqual(first, second)
        self.assertEqual(first, refreshed)
        self.assertEqual(len(reads), 4)
        app.dashboard_data.clear()

    def test_long_history_keeps_all_records_but_renders_one(self):
        history = [{'question': f'Question {i}', 'use_v2': False,
                    'variant': 'Fixed-length (Large)', 'answer': f'Answer {i} [E1]',
                    'evidence': [{'citation': '[E1]', 'doc_id': f'd{i}', 'title': 'Paper'}]}
                   for i in range(50)]
        app.show_history(history)
        self.assertEqual(len(history), 50)
        self.assertLessEqual(len(self.ui.messages), 8)
        self.assertTrue(any('Question 49' in text for _, text in self.ui.messages))
        self.assertFalse(any('Question 0' in text for _, text in self.ui.messages))

    def test_comparison_keeps_both_saved_retrieval_methods(self):
        self.ui.question = 'coronavirus surface persistence'
        self.ui.compare_click = True
        calls = []

        def compare_retrieval(retriever, *_):
            calls.append(retriever.run_id)
            ids = ['d1', 'd2', 'd3', 'd4', 'd5'] if retriever.run_id.endswith('fixed') else [
                'd3', 'd4', 'd5', 'd6', 'd7']
            docs = [{'rank': i, 'doc_id': doc, 'title': f'Paper {doc}', 'score': .8,
                     'best_chunk_id': f'{doc}:1', 'best_chunk_text': f'Evidence {doc}'}
                    for i, doc in enumerate(ids, 1)]
            return [], docs, {'retrieval_total_seconds': .03}

        with patch.object(app, 'cached_retriever', side_effect=lambda run_id: SimpleNamespace(run_id=run_id)), \
             patch.object(app, 'retrieve_evidence', side_effect=compare_retrieval):
            app.comparison_page()
        self.assertEqual(calls, list(app.VARIANTS.values()))
        self.assertIn(('Shared papers', 3), self.ui.metrics)
        self.assertIn(('Fixed-only papers', 2), self.ui.metrics)
        self.assertIn(('Sentence-aware-only papers', 2), self.ui.metrics)
        self.assertEqual(len(self.ui.expanders), 10)


if __name__ == '__main__':
    unittest.main()
