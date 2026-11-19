"""Memory-bounded access to the unchanged Large FAISS indexes and JSONL chunks."""
from array import array
from collections.abc import Sequence
import gc
import json
from pathlib import Path
from threading import RLock
import time

import faiss
from sentence_transformers import SentenceTransformer

from .retriever import PilotRetriever, PROMPT


class DiskChunks(Sequence):
    """Read original JSON records on demand, using only a compact offset array."""
    def __init__(self, path):
        self.path = Path(path)
        offsets = array('Q')
        with self.path.open('rb') as stream:
            position = 0
            for line in stream:
                offsets.append(position)
                position += len(line)
        self.offsets = offsets

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        # A request-local handle avoids cross-session file-position races.
        with self.path.open('rb') as stream:
            stream.seek(self.offsets[index])
            return json.loads(stream.readline())


class SharedResources:
    """Keep one CPU encoder and at most one original IndexFlatIP in memory."""
    def __init__(self):
        self.lock = RLock()
        self.index = None
        self.index_path = None
        self.model = None
        self.model_path = None
        self.index_loads = 0
        self.model_loads = 0

    def get_index(self, path):
        path = Path(path)
        if self.index_path != path:
            self.index = None
            self.index_path = None
            gc.collect()
            self.index = faiss.read_index(str(path))
            self.index_path = path
            self.index_loads += 1
        return self.index

    def get_model(self, path):
        path = Path(path)
        if self.model is None:
            self.model = SentenceTransformer(str(path), device='cpu', local_files_only=True)
            self.model.eval()
            self.model_path = path
            self.model_loads += 1
        elif self.model_path != path:
            raise ValueError('Cloud variants must share the same frozen BGE model snapshot')
        return self.model


RESOURCES = SharedResources()


class CloudRetriever(PilotRetriever):
    """Reuse frozen search, max-chunk aggregation and output fields without copying text."""
    def __init__(self, run_id, root=None, resources=None):
        from .rag_service import VARIANTS
        if run_id not in VARIANTS.values():
            raise ValueError('Unknown Large retrieval variant')
        root = Path(root) if root else Path(__file__).resolve().parents[1]
        self.resources = resources if resources is not None else RESOURCES
        self.proc = root / 'data/processed'
        self.index_dir = root / 'data/indexes' / run_id
        self.metadata = json.loads((self.index_dir / 'index_metadata.json').read_text(encoding='utf-8'))
        self.chunks = DiskChunks(self.proc / f'{run_id}_chunks.jsonl')
        if len(self.chunks) != self.metadata['chunk_count']:
            raise ValueError('Chunk count differs from frozen index metadata')
        self.snapshot = root / 'models/models--BAAI--bge-small-en-v1.5/snapshots' / self.metadata['model_snapshot']

    @property
    def index(self):
        return self.resources.get_index(self.index_dir / 'faiss.index')

    def encode_queries(self, queries):
        with self.resources.lock:
            start = time.perf_counter()
            model = self.resources.get_model(self.snapshot)
            vectors = model.encode([PROMPT + query for query in queries], batch_size=32,
                                   normalize_embeddings=True, show_progress_bar=False,
                                   convert_to_numpy=True).astype('float32')
            return vectors, time.perf_counter() - start

    def retrieve_vector(self, vector, top_k=10):
        with self.resources.lock:
            # Includes an index switch in cloud timing when another method was used.
            start = time.perf_counter()
            index = self.index
            if index.ntotal != len(self.chunks):
                raise ValueError('Index and original chunks have different lengths')
            docs, performance = super().retrieve_vector(vector, top_k)
            performance['retrieval_seconds'] = time.perf_counter() - start
            return docs, performance
