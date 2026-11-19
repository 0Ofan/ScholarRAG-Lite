"""Incremental checks of global chunk ranking, MMR and frozen source mapping."""
import json
import unittest
from pathlib import Path

import numpy as np

from src.phase47_chunk_selection import global_top3, mmr_top3, rank_chunks

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/phase47'
CHUNKS=ROOT/'data/processed/phase33_large_fixed_chunks.jsonl'


class ChunkSelectionTests(unittest.TestCase):
    def test_global_top3_can_use_same_paper_and_is_deterministic(self):
        vectors=np.asarray([[1,0],[.9,.1],[.8,.2],[.7,.3]],dtype=np.float32)
        chunks={i:{'chunk_id':f'c{i}','doc_id':'paper1' if i<3 else 'paper2'}
                for i in range(4)}
        paper_ranks={'paper1':1,'paper2':2}
        ranked=rank_chunks(np.asarray([1,0],dtype=np.float32),list(range(4)),vectors,chunks,paper_ranks)
        self.assertEqual([x['chunk_id'] for x in global_top3(ranked)],['c0','c1','c2'])
        self.assertEqual(ranked,rank_chunks(np.asarray([1,0],dtype=np.float32),
                                            list(range(4)),vectors,chunks,paper_ranks))

    def test_mmr_first_best_and_diversity_with_fixed_lambda(self):
        vector=np.asarray([[1,0],[1,0],[0,1]],dtype=np.float32)
        ranked=[{'chunk_id':f'c{i}','chunk_index':i,'score':1.0}
                for i in range(3)]
        selected=mmr_top3(ranked,vector,lam=.8,relevance_floor=.1)
        self.assertEqual(selected[0]['chunk_id'],ranked[0]['chunk_id'])
        self.assertEqual(selected[1]['chunk_id'],'c2')
        self.assertEqual(len(selected),3)
        self.assertEqual(len({x['chunk_id'] for x in selected}),3)
        self.assertEqual([x['chunk_id'] for x in selected],
                         [x['chunk_id'] for x in mmr_top3(ranked,vector,lam=.8,relevance_floor=.1)])

    def test_twenty_questions_budget_and_original_chunk_text(self):
        data=json.loads((OUT/'evidence_comparison.json').read_text(encoding='utf8'))
        self.assertEqual(len(data['questions']),20)
        candidates={x['question_id']:x for x in map(json.loads,
            (OUT/'candidate_rankings.jsonl').read_text(encoding='utf8').splitlines())}
        self.assertEqual(len(candidates),20)
        original={}
        paper_sets={}
        for q in data['questions']:
            qid=q['question_id'];ranking=candidates[qid]['ranked_chunks']
            paper_sets[qid]={x['doc_id'] for x in q['top10_papers']}
            self.assertEqual(len(ranking),q['measurements']['candidate_chunks'])
            self.assertEqual(len(q['top10_papers']),10)
            self.assertTrue({x['doc_id'] for x in ranking}<=
                            {x['doc_id'] for x in q['top10_papers']})
            self.assertEqual(q['evidence']['B'][0]['chunk_id'],ranking[0]['chunk_id'])
            for group in 'ABC':
                evidence=q['evidence'][group]
                self.assertEqual(len(evidence),3)
                self.assertEqual([x['evidence_id'] for x in evidence],['E1','E2','E3'])
                self.assertEqual(len({x['chunk_id'] for x in evidence}),3)
                self.assertTrue(all(x['source_role']=='Unclassified' for x in evidence))
                original.update({x['chunk_id']:x for x in evidence})
        found=set()
        source_counts={uid:0 for ids in paper_sets.values() for uid in ids}
        with CHUNKS.open(encoding='utf8') as file:
            for line in file:
                uid=line[14:].split('"',1)[0]
                if uid.split(':',1)[0] in source_counts:
                    source_counts[uid.split(':',1)[0]]+=1
                if uid not in original:continue
                source=json.loads(line);item=original[uid]
                self.assertEqual((item['doc_id'],item['title'],item['text'],item['source_path'],item['source_spans']),
                                 (source['doc_id'],source['title'],source['text'],source['source_path'],source['source_spans']))
                found.add(uid)
        self.assertEqual(found,set(original))
        for qid,ids in paper_sets.items():
            self.assertEqual(candidates[qid]['candidate_count'],
                             sum(source_counts[uid] for uid in ids))

    def test_review_status_gate_and_references(self):
        review=json.loads((OUT/'source_review.json').read_text(encoding='utf8'))
        stats=json.loads((OUT/'coverage_stats.json').read_text(encoding='utf8'))
        self.assertEqual(len(review['point_results']),60)
        self.assertEqual(stats['question_count'],20)
        self.assertFalse(stats['real_api_gate']['run_real_api'])
        self.assertEqual(sum(stats['counts_by_group']['A'].values()),60)
        for item in review['point_results']:
            for group in 'ABC':
                check=item['groups'][group]
                self.assertEqual(len(check['evidence_ids']),len(check['chunk_ids']))
                if check['status'] in ('Direct','Partial'):
                    self.assertTrue(check['chunk_ids'])
                if check['status']=='Unverified':
                    self.assertEqual(check['chunk_ids'],[])


if __name__=='__main__':unittest.main()
