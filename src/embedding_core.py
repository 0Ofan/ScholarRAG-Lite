"""Shared BGE chunk encoding settings for Pilot and scale experiments."""
import numpy as np
import torch

def encode_block(model,texts,batch_size=64):
    """Return normalized float32 vectors, reducing only the GPU batch on OOM."""
    fallbacks=[];errors=[]
    while True:
        try:
            vectors=model.encode(texts,batch_size=batch_size,normalize_embeddings=True,
                                 show_progress_bar=False,convert_to_numpy=True).astype('float32')
            if torch.cuda.is_available():torch.cuda.synchronize()
            assert vectors.ndim==2 and vectors.shape==(len(texts),384)
            return np.ascontiguousarray(vectors),batch_size,fallbacks,errors
        except torch.cuda.OutOfMemoryError as exc:
            errors.append(str(exc)[:300]);torch.cuda.empty_cache()
            if batch_size<=8:raise
            batch_size//=2;fallbacks.append(batch_size)
