"""End-to-end checks for the sentence-aware Pilot and frozen baseline."""
import hashlib
import json
import unittest
from pathlib import Path
import faiss
import numpy as np
from transformers import AutoTokenizer
from src.phase27_chunking import make_chunks,sentence_char_ranges
from src.retriever import PilotRetriever
from src.evaluation import score_topic

ROOT=Path(__file__).resolve().parents[1]
PROC=ROOT/'data/processed';NEW=ROOT/'experiments/results/phase32_improved'
INDEX=ROOT/'data/indexes/phase32_sentence_aware';load=lambda p:json.loads(p.read_text(encoding='utf8'))

class Phase32ImprovedTests(unittest.TestCase):
    def test_sentence_boundaries_handle_abbreviations_decimals_and_quotes(self):
        text='Dr. Smith measured 3.14. "It worked." The result was good! Next step? Yes.'
        pieces=[text[a:b].strip() for a,b in sentence_char_ranges(text)]
        self.assertEqual(len(pieces),5)
        self.assertTrue(pieces[0].startswith('Dr. Smith measured 3.14.'))

    def test_baseline_artifacts_are_unchanged(self):
        expected=load(NEW/'phase31_baseline_hashes.json')['files']
        for rel,x in expected.items():
            p=ROOT/rel
            self.assertEqual(p.stat().st_size,x['bytes'])
            with p.open('rb') as stream:
                self.assertEqual(hashlib.file_digest(stream,'sha256').hexdigest(),x['sha256'])

    def test_chunks_are_unique_bounded_and_same_pilot(self):
        pilot=load(PROC/'phase31_pilot_manifest.json');ids={d['doc_id'] for d in pilot['documents']}
        stats=load(NEW/'chunking_stats.json');seen=set();count=0;docs=set()
        with (PROC/'phase32_sentence_aware_chunks.jsonl').open(encoding='utf8') as stream:
            for line in stream:
                c=json.loads(line);count+=1;docs.add(c['doc_id'])
                self.assertNotIn(c['chunk_id'],seen);seen.add(c['chunk_id'])
                self.assertLessEqual(c['token_count'],240)
                self.assertLessEqual(c['minilm_token_count'],240)
                self.assertTrue(c['source_spans'])
        self.assertEqual(count,stats['total_chunks'])
        self.assertEqual(docs,ids)
        self.assertEqual(stats['source_mapping_success_rate'],1)
        self.assertEqual({x['char'] for x in stats['missing_char_examples']},{'\uf0b7'})

    def test_selected_chunks_reproduce_same_text_and_id(self):
        pilot=load(PROC/'phase31_pilot_manifest.json')
        selected={pilot['documents'][i]['doc_id'] for i in (0,999,1999)}
        observed={uid:[] for uid in selected}
        with (PROC/'phase32_sentence_aware_chunks.jsonl').open(encoding='utf8') as stream:
            for line in stream:
                c=json.loads(line)
                if c['doc_id'] in selected:observed[c['doc_id']].append((c['chunk_id'],c['text']))
        cfg=load(ROOT/'experiments/configs/phase27_experiment_config.json')
        snapshot=ROOT/'models/models--BAAI--bge-small-en-v1.5/snapshots'/cfg['model_snapshots']['bge']
        tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
        for doc in pilot['documents']:
            if doc['doc_id'] not in selected:continue
            path=ROOT/'data/raw/cord19_2020-07-16/phase26/parses'/doc['source_path']
            raw=load(path)
            sections=[{'section':'Title','text':doc['title']},
                      {'section':'Abstract','text':' '.join(a.get('text','') for a in raw.get('abstract',[]))}]
            sections.extend(raw.get('body_text',[]))
            chunks,_=make_chunks(doc['doc_id'],sections,tok,None,'sentence',doc['source_path'])
            self.assertEqual(observed[doc['doc_id']],[(c['chunk_id'],c['text']) for c in chunks])

    def test_embeddings_and_index_reload_mapping(self):
        chunk_ids=load(PROC/'phase32_sentence_aware_chunk_ids.json')
        index_ids=load(INDEX/'chunk_ids.json')
        vectors=np.load(PROC/'phase32_sentence_aware_embeddings.npy',mmap_mode='r')
        index=faiss.read_index(str(INDEX/'faiss.index'))
        self.assertEqual(chunk_ids,index_ids)
        self.assertEqual(vectors.shape,(len(chunk_ids),384))
        self.assertEqual(index.ntotal,len(chunk_ids))
        self.assertTrue(np.all(np.isfinite(vectors)))
        self.assertTrue(np.allclose(np.linalg.norm(vectors,axis=1),1,atol=.001))
        self.assertTrue(np.allclose(index.reconstruct(0),vectors[0],atol=1e-6))

    def test_retriever_and_50_query_metrics(self):
        retriever=PilotRetriever(load_model=False,variant='phase32')
        vec=np.load(PROC/'phase32_sentence_aware_embeddings.npy',mmap_mode='r')[0]
        docs,_=retriever.retrieve_vector(vec,10)
        self.assertEqual(len({d['doc_id'] for d in docs}),10)
        metrics=load(NEW/'metrics.json');pilot=load(PROC/'phase31_pilot_manifest.json')
        corpus=[d['doc_id'] for d in pilot['documents']]
        with (NEW/'retrieval_results.jsonl').open(encoding='utf8') as stream:
            rows=[json.loads(line) for line in stream]
        self.assertEqual(len(rows),50)
        for row in rows:
            q=row['topic_id'];top=row['top100'];rank=[d['doc_id'] for d in top]
            self.assertEqual(len(rank),len(set(rank)))
            self.assertEqual(len(rank),100)
            self.assertTrue(all(d['official_grade']==pilot['effective_qrels'][q].get(d['doc_id']) for d in top))
            actual=score_topic(rank,pilot['effective_qrels'][q],corpus)
            for k in ('ndcg@10','recall@10','recall@100','mrr'):
                self.assertAlmostEqual(actual[k],metrics['per_query'][q]['metrics'][k])

    def test_comparison_uses_same_queries(self):
        comp=load(NEW/'comparison.json')
        self.assertEqual(len(comp['per_query']),50)
        self.assertEqual(comp['document_count'],2000)
        self.assertEqual(comp['chunks']['baseline'],88849)
        for q in comp['per_query']:
            self.assertEqual(comp['per_query'][q]['pilot_positive'],load(PROC/'phase31_pilot_manifest.json')['positive_per_query'][q])
