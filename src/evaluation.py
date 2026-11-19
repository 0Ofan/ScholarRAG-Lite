"""Document-level qrel scoring for a fixed corpus and fixed topic set."""
from __future__ import annotations
import math

def document_ranking(chunk_hits):
    """Sort scored chunk hits, retain the first occurrence of each paper."""
    ordered=sorted(enumerate(chunk_hits),key=lambda p:(-p[1]['score'],p[0]))
    seen=set(); result=[]
    for _,hit in ordered:
        uid=hit['doc_id']
        if not uid or uid in seen: continue
        seen.add(uid); result.append(uid)
    return result

def score_topic(ranked_docs, judgments, corpus_ids, threshold=1):
    """Return metrics; unjudged IDs are not labeled negative.

    Conventional qrel scoring awards no gain to unjudged retrieved IDs. The
    caller should separately report judgment coverage and pooled-label bias.
    """
    corpus=set(corpus_ids)
    relevant={uid for uid,g in judgments.items() if g>=threshold and uid in corpus}
    if not relevant: return None
    ranking=[]; seen=set()
    for uid in ranked_docs:
        if uid in corpus and uid not in seen:
            ranking.append(uid);seen.add(uid)
    def gain(g): return 2**max(g,0)-1
    def dcg(ids,k): return sum(gain(judgments.get(uid,0))/math.log2(i+2) for i,uid in enumerate(ids[:k]))
    ideal=sorted((g for uid,g in judgments.items() if uid in corpus and g>=0),reverse=True)
    idcg=sum(gain(g)/math.log2(i+2) for i,g in enumerate(ideal[:10]))
    first=next((i+1 for i,uid in enumerate(ranking) if uid in relevant),None)
    return {'ndcg@10':dcg(ranking,10)/idcg if idcg else None,
            'recall@10':len(set(ranking[:10])&relevant)/len(relevant),
            'recall@100':len(set(ranking[:100])&relevant)/len(relevant),
            'mrr':1/first if first else 0.0,
            'known_relevant_in_corpus':len(relevant),
            'judged_in_top10':sum(judgments.get(uid,-1)>=0 for uid in ranking[:10]),
            'unjudged_in_top10':sum(judgments.get(uid,-1)<0 for uid in ranking[:10])}
