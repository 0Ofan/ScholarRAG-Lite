"""Incremental checks for frozen Phase 4.5 paired evidence and results."""
import json
import unittest
from pathlib import Path

from src.phase37_prepare import digest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/phase45'
BGE=ROOT/'experiments/results/phase33/phase33_large_fixed/retrieval_results.jsonl'
CHUNKS=ROOT/'data/processed/phase33_large_fixed_chunks.jsonl'


class Phase45Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs=json.loads((OUT/'inputs.json').read_text(encoding='utf8'))['entries']
        cls.bge={x['topic_id']:x for x in map(json.loads,BGE.read_text(encoding='utf8').splitlines())}

    def test_paired_evidence_from_original_bge_chunks(self):
        self.assertEqual(len(self.inputs),20)
        for entry in self.inputs:
            topic=entry['retrieval_topic_id'];a_docs=self.bge[topic]['top100']
            self.assertEqual(entry['retrieval_query'],self.bge[topic]['query'])
            self.assertFalse(entry['historical_phase37_reuse_eligible'])
            by_id={x['doc_id']:x for x in a_docs}
            for group in 'AB':
                evidence=entry[group]['evidence']
                self.assertEqual(len(evidence),3)
                self.assertEqual([x['evidence_id'] for x in evidence],['E1','E2','E3'])
                self.assertEqual(digest(evidence),entry[group]['evidence_sha256'])
                for i,item in enumerate(evidence,1):
                    original=by_id[item['doc_id']]
                    self.assertEqual(item['paper_rank'],i)
                    self.assertEqual(item['chunk_id'],original['best_chunk_id'])
                    self.assertEqual(item['text'],original['best_chunk_text'])
                    self.assertEqual(item['score'],original['score'])
                    self.assertEqual(item['bge_paper_rank'],original['rank'])
                    if group=='B':self.assertEqual(item['rrf_rank'],i)
                    else:self.assertIsNone(item['rrf_rank'])

    def test_selected_chunk_text_matches_source_file(self):
        wanted={x['chunk_id']:x for entry in self.inputs for group in 'AB'
                for x in entry[group]['evidence']}
        found=set()
        with CHUNKS.open(encoding='utf8') as file:
            for line in file:
                prefix='{"chunk_id": "'
                if not line.startswith(prefix):continue
                chunk_id=line[len(prefix):].split('"',1)[0]
                if chunk_id not in wanted:continue
                source=json.loads(line);item=wanted[chunk_id]
                self.assertEqual((source['doc_id'],source['title'],source['text']),
                                 (item['doc_id'],item['title'],item['text']))
                found.add(chunk_id)
                if len(found)==len(wanted):break
        self.assertEqual(found,set(wanted))

    def test_real_calls_and_automated_evaluations(self):
        for entry in self.inputs:
            qid=entry['question_id']
            for group in 'AB':
                row=json.loads((OUT/f'{group}_{qid}.json').read_text(encoding='utf8'))
                self.assertEqual(row['status'],'success')
                self.assertTrue(row['real_api'])
                self.assertEqual(row['evidence_sha256'],entry[group]['evidence_sha256'])
                self.assertTrue(row['answer'].strip())
                self.assertEqual(len(row['evidence']),3)
                self.assertIsNotNone(row['generation']['usage']['input_tokens'])
                self.assertEqual(row['citation_check']['invalid'],[])
            evaluation=json.loads((OUT/f'eval_{qid}.json').read_text(encoding='utf8'))
            self.assertEqual(evaluation['status'],'success')
            self.assertEqual(set(evaluation['evaluations_by_version']),{'A','B'})
            for item in evaluation['evaluations_by_version'].values():
                self.assertEqual(set(item['scores']),{'Correctness','Completeness','Relevance','Faithfulness'})

    def test_ranking_cases_and_aggregate(self):
        data=json.loads((OUT/'ranking_cases.json').read_text(encoding='utf8'))
        manifest=json.loads((ROOT/'data/processed/phase33_large_manifest.json').read_text(encoding='utf8'))
        self.assertTrue({'16','22','31','34','41'}<=
                        {x['topic_id'] for x in data['cases']})
        self.assertEqual(len(data['success_topics']),3)
        for case in data['cases']:
            for method in ('bge','bm25','hybrid'):
                self.assertEqual(len(case['top10'][method]),10)
                self.assertTrue(all(x['doc_id'] and x['paper_title'] for x in case['top10'][method]))
                qrels=manifest['effective_qrels'][case['topic_id']]
                for item in case['top10'][method]:
                    grade=qrels.get(item['doc_id'])
                    self.assertEqual(item['qrel_grade'],grade)
                    self.assertEqual(item['qrel_status'],'unjudged' if grade is None else
                                     'positive' if grade>=1 else 'grade_0')
        summary=json.loads((OUT/'aggregate.json').read_text(encoding='utf8'))
        self.assertEqual(summary['evaluator']['successes'],20)
        self.assertEqual(summary['groups']['A']['successes'],20)
        self.assertEqual(summary['groups']['B']['successes'],20)


if __name__=='__main__':unittest.main()
