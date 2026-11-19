"""Global BGE chunk ranking and fixed-lambda diversity selection."""
import numpy as np


def rank_chunks(query_vector, positions, embeddings, chunks, paper_ranks):
    """Score every existing chunk of the frozen Top-10 papers by BGE inner product."""
    matrix=np.asarray(embeddings[positions],dtype=np.float32)
    scores=matrix @ np.asarray(query_vector,dtype=np.float32)
    rows=[{'chunk_index':int(pos),'chunk_id':chunks[pos]['chunk_id'],
           'doc_id':chunks[pos]['doc_id'],'paper_rank':paper_ranks[chunks[pos]['doc_id']],
           'score':float(score)} for pos,score in zip(positions,scores)]
    rows.sort(key=lambda r:(-r['score'],r['chunk_id']))
    return rows


def global_top3(ranked):
    return ranked[:3]


def mmr_top3(ranked, embeddings, lam=.8, relevance_floor=.1):
    """First take best BGE hit; then trade relevance for vector diversity."""
    if not ranked:return []
    eligible=[row for row in ranked if row['score']>=ranked[0]['score']-relevance_floor]
    if len(eligible)<3:eligible=ranked[:3]
    selected=[eligible[0]]
    while len(selected)<min(3,len(eligible)):
        best=None;best_key=None
        for row in eligible:
            if row in selected:continue
            vector=np.asarray(embeddings[row['chunk_index']],dtype=np.float32)
            redundancy=max(float(vector @ np.asarray(embeddings[x['chunk_index']],dtype=np.float32))
                           for x in selected)
            mmr=lam*row['score']-(1-lam)*redundancy
            key=(mmr,row['score'],row['chunk_id'])
            if best_key is None or key[0]>best_key[0] or (key[0]==best_key[0] and
                    (key[1]>best_key[1] or (key[1]==best_key[1] and key[2]<best_key[2]))):
                best=row;best_key=key
        selected.append(best)
    return selected
