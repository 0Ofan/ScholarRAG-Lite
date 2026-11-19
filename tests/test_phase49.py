import json
import unittest
from pathlib import Path

from src.evidence_metadata import enrich, source_role
from src.evidence_verifier import verify
from src.answer_refinement import refine_once
from src.deepseek_client import DeepSeekClient

ROOT = Path(__file__).resolve().parents[1]


class Phase49Tests(unittest.TestCase):
    def saved(self, qid):
        return json.loads((ROOT / f'experiments/results/phase45/A_{qid}.json').read_text(encoding='utf8'))

    def test_metadata_preserves_evidence(self):
        source = self.saved('Q15')['evidence']
        enriched = enrich(source)
        self.assertEqual([e['text'] for e in source], [e['text'] for e in enriched])
        self.assertEqual([e['evidence_id'] for e in source], [e['evidence_id'] for e in enriched])
        self.assertEqual([e['doc_id'] for e in source], [e['doc_id'] for e in enriched])
        self.assertTrue(all(e['source_role'] in {'Primary Study Result', 'Background / Cited Study',
                                                 'Study Plan', 'Review / Summary', 'Unclassified'} for e in enriched))

    def test_known_contradictions(self):
        q15 = self.saved('Q15')
        v15 = verify(q15['answer'], enrich(q15['evidence']))
        self.assertTrue(v15['critical'])
        self.assertTrue(any(c['issue_type'] == 'Study Design' for c in v15['claims']))
        q16 = self.saved('Q16')
        v16 = verify(q16['answer'], enrich(q16['evidence']))
        self.assertTrue(any(c['issue_type'] == 'False Absence' for c in v16['claims']))

    def test_no_keyword_only_success_claim(self):
        sample = [{'evidence_id':'E1','citation':'[E1]','doc_id':'d','title':'t','chunk_id':'d:1',
                   'text':'The study included 36 patients.'}]
        result = verify('该研究有36名患者[E1]。', enrich(sample))
        self.assertEqual(result['claims'][0]['status'], 'Unverified')

    def test_refinement_only_for_critical(self):
        class NoClient:
            def stream(self, *args, **kwargs):
                raise AssertionError('unexpected API call')
        evidence = enrich(self.saved('Q01')['evidence'])
        result = verify('该论文报告了表面存活结果[E1]。', evidence)
        revised = refine_once(NoClient(), 'question', '该论文报告了表面存活结果[E1]。', evidence,
                              result, system_prompt='prompt')
        self.assertFalse(revised['revised'])

    def test_v1_prompt_copy_matches_runtime(self):
        class Completions:
            messages = None
            def create(self, **kwargs):
                self.messages = kwargs['messages']
                return []
        class SDK:
            def __init__(self):
                self.chat = type('Chat', (), {'completions': Completions()})()
        sdk = SDK()
        client = DeepSeekClient(client=sdk, config={'api_key':'test-only', 'model':'deepseek-flash',
                                                    'max_tokens':512, 'temperature':.3})
        list(client.stream('test', enrich(self.saved('Q01')['evidence'])))
        actual = sdk.chat.completions.messages[0]['content']
        frozen = (ROOT/'prompts/scholar_rag_v1_phase37.txt').read_text(encoding='utf8').strip()
        self.assertEqual(actual, frozen)

    def test_one_refinement_for_critical_claim(self):
        record = self.saved('Q15')
        evidence = enrich(record['evidence'])
        initial = verify(record['answer'], evidence)
        class FakeClient:
            calls = 0
            def stream(self, *args, **kwargs):
                self.calls += 1
                yield 'text', '法国研究为非随机设计，且设有未治疗对照 [E3]。'
                yield 'done', {'usage': None, 'finish_reason': 'stop'}
        fake = FakeClient()
        result = refine_once(fake, record['question'], record['answer'], evidence,
                             initial, system_prompt='v2')
        self.assertEqual(fake.calls, 1)
        self.assertTrue(result['revised'])
        self.assertFalse(result['verification']['critical'])

    def test_incomplete_refinement_keeps_original(self):
        record = self.saved('Q15')
        evidence = enrich(record['evidence'])
        initial = verify(record['answer'], evidence)
        for reason in ('length', 'content_filter', 'aborted', 'unexpected', None):
            with self.subTest(reason=reason):
                class FakeClient:
                    def stream(self, *args, **kwargs):
                        yield 'text', '法国研究为非随机设计，且设有未治疗对照 [E3]。'
                        yield 'done', {'finish_reason': reason, 'usage': {'input_tokens': 3}}
                result = refine_once(FakeClient(), record['question'], record['answer'], evidence,
                                     initial, system_prompt='v2')
                self.assertFalse(result['revised'])
                self.assertEqual(result['answer'], record['answer'])
                self.assertIs(result['verification'], initial)


if __name__ == '__main__':
    unittest.main()
