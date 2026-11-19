"""Paper-level sparse BM25 baseline using existing project dependencies."""
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results/phase43'
MANIFEST = ROOT/'data/processed/phase33_large_manifest.json'
PARSES = ROOT/'data/raw/cord19_2020-07-16/phase26/parses'
PATTERN = r"(?u)\b[a-z0-9]+(?:[-'][a-z0-9]+)*\b"
K1 = 1.5
B = .75


def vectorizer(vocabulary=None):
    return CountVectorizer(lowercase=True, token_pattern=PATTERN, vocabulary=vocabulary,
                           dtype=np.int32)


def paper_texts(documents):
    for doc in documents:
        source = json.loads((PARSES/doc['source_path']).read_text(encoding='utf8'))
        sections = [doc['title']]
        sections += [part.get('text', '') for part in source.get('abstract', [])]
        sections += [part.get('text', '') for part in source.get('body_text', [])]
        yield ' '.join(' '.join(part.split()) for part in sections if part)


class BM25Index:
    def __init__(self, matrix, vocabulary, doc_ids, doc_lengths, df):
        self.matrix = matrix.tocsc()
        self.vocabulary = vocabulary
        self.doc_ids = doc_ids
        self.doc_lengths = np.asarray(doc_lengths, dtype=np.float64)
        self.df = np.asarray(df)
        self.n = len(doc_ids)
        self.avgdl = float(self.doc_lengths.mean())
        self.tokenize = vectorizer(vocabulary).build_tokenizer()
        self.length_norm = K1*(1-B+B*self.doc_lengths/self.avgdl)

    @classmethod
    def from_texts(cls, texts, doc_ids):
        cv = vectorizer()
        counts = cv.fit_transform(texts)
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        df = counts.getnnz(axis=0)
        return cls(counts, cv.vocabulary_, doc_ids, lengths, df)

    def score(self, query):
        scores = np.zeros(self.n, dtype=np.float64)
        for term in self.tokenize(query.lower()):
            col = self.vocabulary.get(term)
            if col is None:
                continue
            start, end = self.matrix.indptr[col:col+2]
            ids = self.matrix.indices[start:end]
            tf = self.matrix.data[start:end].astype(np.float64)
            idf = math.log1p((self.n-self.df[col]+.5)/(self.df[col]+.5))
            scores[ids] += idf * tf * (K1+1)/(tf+self.length_norm[ids])
        return scores

    def search(self, query, k=100):
        scores = self.score(query)
        positions = np.lexsort((self.doc_ids, -scores))[:k]
        return [{'doc_id':self.doc_ids[int(i)], 'score':float(scores[i]),
                 'rank':rank} for rank,i in enumerate(positions, 1)]

    def save(self, folder):
        sparse.save_npz(folder/'term_counts.npz', self.matrix, compressed=True)
        (folder/'vocabulary.json').write_text(json.dumps(self.vocabulary, ensure_ascii=False), encoding='utf8')
        (folder/'document_ids.json').write_text(json.dumps(self.doc_ids), encoding='utf8')
        np.save(folder/'document_lengths.npy', self.doc_lengths)
        np.save(folder/'document_frequency.npy', self.df)

    @classmethod
    def load(cls, folder):
        matrix = sparse.load_npz(folder/'term_counts.npz')
        vocabulary = json.loads((folder/'vocabulary.json').read_text(encoding='utf8'))
        ids = json.loads((folder/'document_ids.json').read_text(encoding='utf8'))
        lengths = np.load(folder/'document_lengths.npy')
        df = np.load(folder/'document_frequency.npy')
        if matrix.shape != (len(ids), len(vocabulary)):
            raise ValueError('BM25 index dimensions do not match identifiers')
        return cls(matrix, vocabulary, ids, lengths, df)


def build():
    manifest = json.loads(MANIFEST.read_text(encoding='utf8'))
    docs = manifest['documents']
    ids = [d['doc_id'] for d in docs]
    if len(ids) != 12000 or len(set(ids)) != 12000:
        raise ValueError('Expected frozen Large corpus of 12,000 unique papers')
    proc = psutil.Process()
    samples = [proc.memory_info().rss]
    stop = threading.Event()
    def sample():
        while not stop.wait(.25):
            samples.append(proc.memory_info().rss)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        index = BM25Index.from_texts(paper_texts(docs), ids)
        vectorize_seconds = time.perf_counter()-start
        index.save(OUT)
        save_seconds = time.perf_counter()-start-vectorize_seconds
    finally:
        stop.set(); thread.join()
    files = ['term_counts.npz','vocabulary.json','document_ids.json',
             'document_lengths.npy','document_frequency.npy']
    data = {'version':'phase43-build-v1','document_count':len(ids),
            'vocabulary_size':len(index.vocabulary),'nonzero_term_document_pairs':index.matrix.nnz,
            'mean_document_terms':index.avgdl,'k1':K1,'b':B,
            'vectorize_seconds':vectorize_seconds,'save_seconds':save_seconds,
            'build_total_seconds':time.perf_counter()-start,
            'process_peak_rss_bytes':max(samples+[proc.memory_info().rss]),
            'index_files':{name:(OUT/name).stat().st_size for name in files}}
    (OUT/'build_metrics.json').write_text(json.dumps(data, indent=2)+'\n', encoding='utf8')
    print('BM25 built:',len(ids),'papers,',len(index.vocabulary),'terms,',round(data['build_total_seconds'],2),'s',flush=True)


if __name__ == '__main__':
    build()
