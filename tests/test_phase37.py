"""Focused structural and prompt checks for sentence-level grounding."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from src.citation_grounding import validate_grounded_answer, verify_evidence_source
from src.deepseek_client import DeepSeekClient
from src.rag_service import prepare_grounded_evidence

ROOT = Path(__file__).resolve().parents[1]


class Phase37Tests(unittest.TestCase):
    def setUp(self):
        self.original = [
            {'citation': '[1]', 'doc_id': 'doc00001', 'chunk_id': 'c1',
             'title': 'A pediatric review of COVID-19', 'text': 'Children usually had mild symptoms.', 'score': .9},
            {'citation': '[2]', 'doc_id': 'doc00002', 'chunk_id': 'c2',
             'title': 'Adult care in a children hospital', 'text': 'Adults were treated by pediatric teams.', 'score': .8},
            {'citation': '[3]', 'doc_id': 'doc00003', 'chunk_id': 'c3',
             'title': 'T cell markers in severe COVID-19', 'text': 'Ki-67 and PD-1 were measured.', 'score': .7},
        ]
        self.grounded = prepare_grounded_evidence(self.original)

    def test_evidence_units_keep_source_and_exact_text(self):
        self.assertEqual([e['citation'] for e in self.grounded], ['[E1]', '[E2]', '[E3]'])
        self.assertEqual([e['text'] for e in self.grounded], [e['text'] for e in self.original])
        self.assertEqual([e['doc_id'] for e in self.grounded], [e['doc_id'] for e in self.original])
        self.assertEqual(self.original[0]['citation'], '[1]')  # frozen input untouched
        source_map = {e['chunk_id']: e for e in self.original}
        self.assertTrue(all(row['status'] == 'valid' for row in
                            verify_evidence_source(self.grounded, source_map)))
        changed = [dict(e) for e in self.grounded]
        changed[1]['title'] = self.original[0]['title']
        checks = verify_evidence_source(changed, source_map)
        self.assertEqual(checks[1]['status'], 'invalid')
        self.assertIn('title', checks[1]['mismatched_fields'])

    def test_prompt_preserves_independent_records_and_streaming(self):
        sdk = Mock()
        sdk.chat.completions.create.return_value = [
            NS(model='deepseek-flash', usage=None,
               choices=[NS(delta=NS(content='儿童通常轻症 [E1]。', reasoning_content=None), finish_reason='stop')]),
            NS(model='deepseek-flash', usage=NS(prompt_tokens=50, completion_tokens=15,
               total_tokens=65, prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=50), choices=[]),
        ]
        client = DeepSeekClient(client=sdk, config={'api_key': 'test-only',
            'base_url': 'https://api.deepseek.com', 'model': 'deepseek-flash',
            'temperature': .3, 'max_tokens': 512, 'timeout': 30})
        events = list(client.stream('What happens in children?', self.grounded))
        kwargs = sdk.chat.completions.create.call_args.kwargs
        self.assertTrue(kwargs['stream'])
        self.assertEqual(kwargs['max_tokens'], 512)
        self.assertEqual(kwargs['temperature'], .3)
        self.assertEqual(kwargs['extra_body'], {'thinking': {'type': 'disabled'}})
        self.assertIn('每个具体研究事实句末', kwargs['messages'][0]['content'])
        records = [json.loads(line) for line in kwargs['messages'][1]['content'].splitlines()
                   if line.startswith('{')]
        self.assertEqual(len(records), 3)
        self.assertEqual(records[1]['Document ID'], 'doc00002')
        self.assertEqual(records[1]['Paper Title'], self.original[1]['title'])
        self.assertEqual(records[1]['Original Text'], self.original[1]['text'])
        self.assertEqual(events[-1][1]['usage']['input_tokens'], 50)

    def test_sentence_level_and_invalid_reference_detection(self):
        answer = '儿童通常轻症 [E1]。成人研究发现成人接受儿科团队照护 [E2]。研究发现异常 [E9]。'
        check = validate_grounded_answer(answer, self.grounded)
        self.assertEqual(check['invalid'], ['E9'])
        self.assertEqual(check['nonterminal_citation_sentences'], [])
        self.assertEqual(check['uncited_factual_candidates'], [])
        self.assertEqual({m['doc_id'] for m in check['citation_mapping']}, {'doc00001', 'doc00002'})
        uncited = validate_grounded_answer('研究发现儿童通常轻症。', self.grounded)
        self.assertEqual(uncited['uncited_factual_candidates'], [1])

    def test_historical_failure_patterns_remain_unverified(self):
        # Q04: another review's limitations falsely assigned to target study.
        # Q07: adult cohort used for pediatric outcomes.
        # Q13: specific trial generalized to all settings.
        # Q14: a list of measured markers treated as observed findings.
        examples = [
            '该目标研究存在另一综述所述局限 [E2]。',
            '儿童结局研究存在成人病房的抽样偏差 [E2]。',
            '所有场景下口罩效果完全相同 [E1]。',
            'Ki-67 表达显著升高 [E3]。',
        ]
        for answer in examples:
            with self.subTest(answer=answer):
                check = validate_grounded_answer(answer, self.grounded)
                self.assertTrue(check['unverified_semantic_support'])
                self.assertTrue(any(s['semantic_support'] == 'unverified' for s in check['sentences']))
        mismatch = validate_grounded_answer('doc00001 的结论 [E2]。', self.grounded)
        self.assertEqual(mismatch['explicit_attribution_mismatch_sentences'], [1])
        self.assertEqual(mismatch['overall_status'], 'invalid')
        bad_source = verify_evidence_source([dict(self.grounded[0], title='Wrong title')],
                                            {self.original[0]['chunk_id']: self.original[0]})
        bad_check = validate_grounded_answer('研究发现儿童轻症 [E1]。', [self.grounded[0]], bad_source)
        self.assertEqual(bad_check['invalid'], ['E1'])

    def test_frozen_question_set_and_evidence_manifest_unchanged(self):
        manifest = json.loads((ROOT/'experiments/results/phase37/inputs_manifest.json').read_text(encoding='utf8'))
        self.assertEqual(manifest['question_count'], 20)
        self.assertTrue(all(len(e['source_checks']) == 3 and
                            all(c['status'] == 'valid' for c in e['source_checks'])
                            for e in manifest['entries']))

    def test_completed_run_never_repeats_api(self):
        from src.phase37_run import main as run_main
        with patch('src.phase37_run.DeepSeekClient') as client:
            run_main()
            client.assert_not_called()

    def test_search_page_uses_grounded_ids_without_live_api(self):
        from streamlit.testing.v1 import AppTest
        meta = {'api_ttft_seconds': .01, 'api_total_seconds': .02,
                'usage': {'input_tokens': 10, 'output_tokens': 5},
                'reasoning_content_seen': False, 'finish_reason': 'stop'}
        with patch('src.deepseek_client.DeepSeekClient.__init__', return_value=None), \
             patch('src.deepseek_client.DeepSeekClient.stream',
                   return_value=iter([('text', '检索证据表明存在相关研究 [E1]。'), ('done', meta)])):
            app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=120).run()
            app.text_input[0].set_value('COVID-19 respiratory symptoms')
            app.button[0].click()
            app.run()
        self.assertFalse(app.exception)
        state = app.session_state['last_answer']
        self.assertEqual([e['citation'] for e in state['evidence']], ['[E1]', '[E2]', '[E3]'])
        self.assertEqual(state['answer'], '检索证据表明存在相关研究 [E1]。')


if __name__ == '__main__':
    unittest.main()
