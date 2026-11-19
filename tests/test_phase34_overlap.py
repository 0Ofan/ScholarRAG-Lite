"""Incremental overlap checks without rebuilding A or B artifacts."""
import json
import unittest
from pathlib import Path

import faiss
import numpy as np

from transformers import AutoTokenizer
from src.phase27_chunking import make_chunks

ROOT=Path(__file__).resolve().parents[1]


class Phase34OverlapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config=json.loads((ROOT/'experiments/configs/phase27_experiment_config.json').read_text(encoding='utf8'))
        snapshot=ROOT/'models/models--BAAI--bge-small-en-v1.5/snapshots'/config['model_snapshots']['bge']
        cls.tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)

    def chunks(self,text,overlap=0):
        sections=[{'section':'Body','text':text}]
        return make_chunks('test-doc',sections,self.tokenizer,None,'sentence','synthetic.json',
                           overlap_tokens=overlap)[0]

    def test_default_is_explicit_zero(self):
        text='First sentence is here. Second sentence follows. '*100
        sections=[{'section':'Body','text':text}]
        old=make_chunks('test-doc',sections,self.tokenizer,None,'sentence','synthetic.json')
        zero=make_chunks('test-doc',sections,self.tokenizer,None,'sentence','synthetic.json',overlap_tokens=0)
        self.assertEqual(old,zero)

    def test_whole_sentence_overlap_preserves_base_chunks(self):
        text='Alpha beta gamma delta epsilon. '*120
        base=self.chunks(text)
        changed=self.chunks(text,20)
        self.assertGreater(len(base),2)
        self.assertEqual(len(base),len(changed))
        self.assertEqual(changed,self.chunks(text,20))
        self.assertEqual(changed[0]['overlap_tokens_actual'],0)
        self.assertTrue(any(c['overlap_rule']=='whole_sentence' for c in changed[1:]))
        for b,c in zip(base,changed):
            self.assertTrue(c['text'].endswith(b['text']))
            self.assertLessEqual(c['overlap_tokens_actual'],20)
            self.assertLessEqual(c['tokens_bge'],240)
            self.assertTrue(c['base_source_spans'])

    def test_long_sentence_uses_token_tail(self):
        text=' '.join(['longsentence']*420)+'.'
        base=self.chunks(text)
        changed=self.chunks(text,20)
        self.assertEqual(len(base),len(changed))
        self.assertTrue(any(c['overlap_rule']=='token_tail' for c in changed[1:]))
        for b,c in zip(base,changed):
            self.assertTrue(c['text'].endswith(b['text']))
            self.assertLessEqual(c['overlap_tokens_actual'],20)
            self.assertLessEqual(c['tokens_bge'],240)

    def test_new_corpus_preserves_every_base_chunk(self):
        old=ROOT/'data/processed/phase32_sentence_aware_chunks.jsonl'
        new=ROOT/'data/processed/phase34_ablation_chunks.jsonl'
        if not new.exists():self.skipTest('C chunks have not been built yet')
        count=0;with_overlap=0
        with old.open(encoding='utf8') as baseline,new.open(encoding='utf8') as changed:
            for old_line,new_line in zip(baseline,changed,strict=True):
                b=json.loads(old_line);c=json.loads(new_line);count+=1
                self.assertEqual(c['doc_id'],b['doc_id'])
                self.assertTrue(c['text'].endswith(b['text']))
                self.assertEqual(c['base_source_spans'],b['source_spans'])
                self.assertLessEqual(c['overlap_tokens_actual'],20)
                self.assertLessEqual(c['token_count'],240)
                self.assertLessEqual(c['minilm_token_count'],240)
                self.assertTrue(c['source_spans'])
                if count%307==0:
                    prefix=c['text'][:-len(b['text'])]
                    self.assertEqual(len(self.tokenizer(prefix,add_special_tokens=False)['input_ids']),
                                     c['overlap_tokens_actual'])
                with_overlap+=c['overlap_tokens_actual']>0
        stats=json.loads((ROOT/'experiments/results/phase34_ablation/chunking_stats.json').read_text(encoding='utf8'))
        old_stats=json.loads((ROOT/'experiments/results/phase32_improved/chunking_stats.json').read_text(encoding='utf8'))
        self.assertEqual(count,stats['total_chunks'])
        self.assertEqual(count,old_stats['total_chunks'])
        self.assertEqual(with_overlap,stats['overlap']['chunks_with_overlap'])
        self.assertEqual(stats['missing_nonspace_chars'],old_stats['missing_nonspace_chars'])

    def test_new_embedding_index_and_50_queries(self):
        result=ROOT/'experiments/results/phase34_ablation'
        if not (result/'metrics.json').exists():self.skipTest('C retrieval has not been evaluated yet')
        ids=json.loads((ROOT/'data/processed/phase34_ablation_chunk_ids.json').read_text(encoding='utf8'))
        vectors=np.load(ROOT/'data/processed/phase34_ablation_embeddings.npy',mmap_mode='r')
        index=faiss.read_index(str(ROOT/'data/indexes/phase34_ablation/faiss.index'))
        metrics=json.loads((result/'metrics.json').read_text(encoding='utf8'))
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(vectors.shape,(len(ids),384))
        self.assertEqual(index.ntotal,len(ids))
        self.assertTrue(np.allclose(index.reconstruct(0),vectors[0],atol=1e-6))
        self.assertEqual(metrics['query_count'],50)
        self.assertEqual(len(metrics['per_query']),50)
        with (result/'retrieval_results.jsonl').open(encoding='utf8') as stream:
            rows=[json.loads(line) for line in stream]
        self.assertEqual(len(rows),50)
        self.assertTrue(all(len({d['doc_id'] for d in r['top100']})==100 for r in rows))

    def test_three_group_comparison_keeps_paired_query_differences(self):
        path=ROOT/'experiments/results/phase34_ablation/comparison.json'
        if not path.exists():self.skipTest('comparison has not been generated yet')
        comparison=json.loads(path.read_text(encoding='utf8'))
        self.assertEqual(set(comparison['groups']),{'A','B','C'})
        self.assertEqual(comparison['query_count'],50)
        self.assertEqual(len(comparison['per_query']),50)
        self.assertEqual(comparison['positive_anchor_count'],956)
        for row in comparison['per_query'].values():
            for metric in ('ndcg@10','recall@10','recall@100','mrr','top10_unjudged_rate'):
                self.assertAlmostEqual(row['C_minus_B'][metric],row['C'][metric]-row['B'][metric])
                self.assertAlmostEqual(row['C_minus_A'][metric],row['C'][metric]-row['A'][metric])


if __name__=='__main__':unittest.main()
