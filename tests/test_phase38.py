"""Incremental checks for frozen Phase 3.8 candidate generation inputs."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

from src.deepseek_client import DeepSeekClient
from src.phase38_run import load_inputs
from src.phase38_evaluate import blinded_payload, load_triplet, validate_evaluation, SUPPORT
from src.phase38_rules import check_answer, normalize_for_blind_evaluation
from src.phase38_aggregate import describe

ROOT = Path(__file__).resolve().parents[1]


class Phase38Tests(unittest.TestCase):
    def test_candidate_inputs_are_same_twenty_evidence_sets(self):
        questions, manifest, prompt = load_inputs()
        self.assertEqual(len(questions), len(manifest['entries']))
        self.assertEqual(len(questions), 20)
        self.assertEqual(manifest['run_config']['max_tokens'], 512)
        self.assertIn('合理推知', prompt)
        self.assertIn('研究设计', prompt)

    def test_candidate_prompt_only_changes_system_message(self):
        sdk = Mock()
        sdk.chat.completions.create.return_value = [
            NS(model='deepseek-flash', choices=[NS(delta=NS(content='试验旨在评估效果 [E1]。',
                                                          reasoning_content=None), finish_reason='stop')], usage=None),
            NS(model='deepseek-flash', choices=[], usage=NS(prompt_tokens=150,
                completion_tokens=20, total_tokens=170,
                prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=150)),
        ]
        client = DeepSeekClient(client=sdk, config={'api_key': 'test-only',
            'base_url': 'https://api.deepseek.com', 'model': 'deepseek-flash',
            'temperature': .3, 'max_tokens': 512, 'timeout': 30})
        prompt = (ROOT/'experiments/results/phase38/candidate_prompt.txt').read_text(encoding='utf8').strip()
        evidence = [{'evidence_id': 'E1', 'citation': '[E1]', 'doc_id': 'd1',
                     'chunk_id': 'c1', 'title': 'Trial', 'text': 'The trial evaluated treatment.'}]
        events = list(client.stream('What was the purpose?', evidence, system_prompt=prompt))
        args = sdk.chat.completions.create.call_args.kwargs
        self.assertEqual(args['messages'][0]['content'], prompt)
        self.assertEqual(args['model'], 'deepseek-flash')
        self.assertEqual(args['max_tokens'], 512)
        self.assertEqual(args['temperature'], .3)
        self.assertTrue(args['stream'])
        self.assertEqual(args['extra_body'], {'thinking': {'type': 'disabled'}})
        self.assertEqual(json.loads(args['messages'][1]['content'].splitlines()[-1])['Original Text'],
                         evidence[0]['text'])
        self.assertEqual(events[-1][1]['usage']['output_tokens'], 20)

    def test_blinded_triplet_and_source_binding(self):
        question = load_inputs()[0][1]
        records, evidence = load_triplet(question)
        mapping, payload = blinded_payload(question, records, evidence)
        self.assertEqual(set(mapping.values()), {'A', 'B', 'C'})
        self.assertEqual(len(evidence), 3)
        self.assertNotIn('A', payload['answers'])
        self.assertNotIn('B', payload['answers'])
        self.assertNotIn('C', payload['answers'])
        self.assertEqual(evidence, records['C']['evidence'])
        self.assertEqual(payload['evidence_records'][0]['doc_id'], evidence[0]['doc_id'])
        self.assertEqual(payload['evidence_records'][0]['text'], evidence[0]['text'])
        self.assertIn('[E1]', normalize_for_blind_evaluation(records['A']['answer'], 'A'))

    def test_rule_warnings_do_not_claim_semantic_truth(self):
        evidence = [{'evidence_id':'E1','citation':'[E1]','doc_id':'d1','title':'Trial',
                     'chunk_id':'c1','text':'In a hospital, 30% improved.'}]
        sources = [{'evidence_id':'E1','status':'valid'}]
        result = check_answer('The study showed 50% improved [E1].', evidence, sources)
        self.assertEqual(result['semantic_support_status'], 'Unverified')
        self.assertEqual(result['numeric_claims_without_same_number_in_cited_text'][0]['number'], '50%')
        self.assertEqual(result['citation']['citation_mapping'][0]['doc_id'], 'd1')

    def test_structured_evaluator_covers_all_claims_and_four_support_classes(self):
        def response(support):
            metric = {'score':4,'reason':'Compared against E1.','evidence_ids':['E1']}
            cls = lambda value: {'class':value,'reason':'Checked E1.','evidence_ids':['E1']}
            return {'scores':{k:dict(metric) for k in ('Correctness','Completeness','Relevance','Faithfulness')},
                    'claims':[{'sentence_index':1,'support':support,'reason':'Checked E1.',
                               'evidence_ids':['E1']}],
                    'citation_support':cls(support), 'source_attribution':cls('Correct'),
                    'overgeneralization':cls('No'), 'abstention':cls('Appropriate')}
        for support in SUPPORT:
            value = {'responses':{'R1':response(support)}}
            self.assertIs(validate_evaluation(value, {'R1':'A'}, {'E1'}, {'R1':1}), value)
        value['responses']['R1']['claims'][0]['support'] = 'Fully supported'
        with self.assertRaises(ValueError):
            validate_evaluation(value, {'R1':'A'}, {'E1'}, {'R1':1})

    def test_all_real_results_and_automated_evaluations_are_auditable(self):
        questions, manifest, _ = load_inputs()
        entries = {x['question_id']:x for x in manifest['entries']}
        for question in questions:
            qid = question['question_id']
            records, evidence = load_triplet(question)
            self.assertEqual(records['C']['evidence_sha256'], entries[qid]['evidence_sha256'])
            self.assertEqual(records['C']['generation']['response_model'], 'deepseek-flash')
            self.assertFalse(records['C']['generation']['reasoning_content_seen'])
            self.assertEqual(records['C']['generation']['finish_reason'], 'stop')
            self.assertIsNotNone(records['C']['generation']['usage']['input_tokens'])
            self.assertEqual(len(evidence), 3)
            judged = json.loads((ROOT/f'experiments/results/phase38/eval_{qid}.json').read_text(encoding='utf8'))
            self.assertEqual(judged['status'], 'success')
            self.assertEqual(judged['label'], 'Automated Evaluation')
            self.assertEqual(set(judged['evaluations_by_version']), {'A','B','C'})
            self.assertEqual(set(judged['blinded_response_mapping'].values()), {'A','B','C'})
            validate_evaluation({'responses': {label: judged['evaluations_by_version'][version]
                for label, version in judged['blinded_response_mapping'].items()}},
                judged['blinded_response_mapping'], {e['evidence_id'] for e in evidence},
                {label:len(payload['sentences']) for label,payload in judged['input']['answers'].items()})

    def test_aggregate_percentile(self):
        self.assertEqual(describe([1,2,3,4,5])['p90'], 4.6)


if __name__ == '__main__':
    unittest.main()
