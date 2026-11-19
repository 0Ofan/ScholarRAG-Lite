"""Artifact and behavior checks for the frozen Pilot baseline."""
import json
import unittest
from pathlib import Path
import faiss
import numpy as np
from src.retriever import PilotRetriever
from src.evaluation import score_topic

ROOT=Path(__file__).resolve().parents[1]
PROC=ROOT/'data/processed';INDEX=ROOT/'data/indexes/phase31_pilot';RESULT=ROOT/'experiments/results/phase31_pilot'
load=lambda p:json.loads(p.read_text(encoding='utf8'))

class Phase31PilotTests(unittest.TestCase):
    def test_manifest_matches_frozen_selection(self):
        m=load(PROC/'phase31_pilot_manifest.json')
        c=load(ROOT/'experiments/configs/phase27_experiment_config.json')
        ids=[d['doc_id'] for d in m['documents']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(len(ids),2000)
        self.assertEqual(set(ids),set(c['scales']['pilot']['doc_ids']))
        self.assertEqual(m['positive_anchor_count'],956)
        self.assertEqual(m['invalid_or_missing_fulltext_count'],0)
        self.assertEqual(m['positive_query_doc_pairs'],sum(m['positive_per_query'].values()))

    def test_chunks_have_unique_traceable_ids_and_bound(self):
        m=load(PROC/'phase31_pilot_manifest.json');docs={d['doc_id'] for d in m['documents']}
        ids=set();count=0
        with (PROC/'phase31_pilot_chunks.jsonl').open(encoding='utf8') as stream:
            for line in stream:
                c=json.loads(line);count+=1
                self.assertNotIn(c['chunk_id'],ids);ids.add(c['chunk_id'])
                self.assertIn(c['doc_id'],docs)
                self.assertTrue(0<c['token_count']<=240)
                self.assertTrue(c['source_spans'])
                self.assertEqual(c['source_start'],c['source_spans'][0]['raw_char_start'])
                self.assertEqual(c['source_end'],c['source_spans'][0]['raw_char_end'])
        self.assertEqual(count,load(RESULT/'chunking_stats.json')['total_chunks'])

    def test_embeddings_index_and_mapping_reload(self):
        ids=load(PROC/'phase31_pilot_chunk_ids.json')
        indexed=load(INDEX/'chunk_ids.json')
        vectors=np.load(PROC/'phase31_pilot_embeddings.npy',mmap_mode='r')
        index=faiss.read_index(str(INDEX/'faiss.index'))
        self.assertEqual(ids,indexed)
        self.assertEqual(vectors.shape,(len(ids),384))
        self.assertEqual(index.ntotal,len(ids))
        self.assertTrue(np.all(np.isfinite(vectors)))
        self.assertTrue(np.allclose(np.linalg.norm(vectors,axis=1),1,atol=.001))
        self.assertTrue(np.allclose(index.reconstruct(0),vectors[0],atol=1e-6))

    def test_retriever_dedup_and_reproducibility(self):
        retriever=PilotRetriever(load_model=False)
        vec=np.load(PROC/'phase31_pilot_embeddings.npy',mmap_mode='r')[0]
        first,_=retriever.retrieve_vector(vec,10)
        second,_=retriever.retrieve_vector(vec,10)
        self.assertEqual([x['doc_id'] for x in first],[x['doc_id'] for x in second])
        self.assertEqual(len({x['doc_id'] for x in first}),10)
        self.assertTrue(all(x['best_chunk_id']==retriever.chunk_ids[x['chunk_index']] for x in first))

    def test_all_50_queries_and_qrel_metrics(self):
        m=load(PROC/'phase31_pilot_manifest.json');metrics=load(RESULT/'metrics.json')
        with (RESULT/'retrieval_results.jsonl').open(encoding='utf8') as stream:
            rows=[json.loads(line) for line in stream]
        self.assertEqual(len(rows),50)
        self.assertEqual(set(metrics['per_query']),set(m['query_ids']))
        for row in rows:
            q=row['topic_id'];hits=row['top100'];ids=[x['doc_id'] for x in hits]
            self.assertEqual(len(ids),len(set(ids)))
            self.assertEqual(len(ids),100)
            self.assertTrue(all(h['official_grade']==m['effective_qrels'][q].get(h['doc_id']) for h in hits))
            expected=score_topic(ids,m['effective_qrels'][q],[d['doc_id'] for d in m['documents']])
            for k in ('ndcg@10','recall@10','recall@100','mrr'):
                self.assertAlmostEqual(expected[k],metrics['per_query'][q]['metrics'][k])
