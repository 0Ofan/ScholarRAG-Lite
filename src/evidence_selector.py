"""Small query-aware within-paper chunk selector over an existing Top-10 ranking."""
import re
from pathlib import Path

from src.rag_service import prepare_grounded_evidence

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT/'data/processed/phase33_large_fixed_chunks.jsonl'
STOP = set('a an and are as at be by can could do does for from how in into is it its of on or paper report study studies that the their these this to was were what which who with without about according patients covid coronavirus sars cov 19'.split())
WORD = re.compile(r'[a-zA-Z][a-zA-Z0-9-]*')
RESULT = re.compile(r'\b(?:we found|we observed|results? show|results? demonstrate|significantly|our findings|in conclusion|we conclude|was associated|were associated)\b', re.I)
LIMIT = re.compile(r'\b(?:limitation|limited by|limitations of|caution|cannot conclude)\b', re.I)
BACKGROUND = re.compile(r'\b(?:previous studies|other studies|has been reported|et al\.|cited in|review article)\b', re.I)


def terms(text):
    words = {w.lower() for w in WORD.findall(text)} - STOP
    return {w[:5] if len(w) > 5 else w for w in words if len(w) > 2}


def source_role(question, paper):
    """Only label an explicitly named paper; otherwise leave provenance role unknown."""
    if question.get('category') != 'single_paper':
        return 'Unclassified'
    text = question['question'].casefold()
    title = paper['title'].casefold()
    quoted = re.findall(r"['‘’\"]([^'‘’\"]{20,})['‘’\"]", text)
    if quoted and any(name in title for name in quoted):
        return 'Primary Evidence'
    # Acronym in a named study, e.g. CoDEX; require it in paper title.
    acronyms = re.findall(r'\b[A-Z][A-Za-z]*[A-Z][A-Za-z]*\b', question['question'])
    generic = {'sars', 'covid', 'ards', 'pcr', 'ct', 'igm', 'igg'}
    if any(word.casefold() in title for word in acronyms
           if len(word) >= 4 and word.casefold() not in generic):
        return 'Primary Evidence'
    return 'Unclassified'


def baseline(top10, question):
    """Exact old top-three paper and highest-score chunk, with offline-only roles."""
    rows = []
    for i, doc in enumerate(top10[:3], 1):
        rows.append({'citation': f'[{i}]', 'title': doc['title'], 'doc_id': doc['doc_id'],
                     'chunk_id': doc['best_chunk_id'], 'text': doc['best_chunk_text'],
                     'score': doc['score'], 'source_path': doc['source_path'],
                     'source_spans': doc['source_spans'], 'source_role': source_role(question, doc)})
    roles = [r['source_role'] for r in rows]
    if 'Primary Evidence' in roles:
        for row in rows:
            if row['source_role'] == 'Unclassified':
                row['source_role'] = 'Supporting Context'
    return prepare_grounded_evidence(rows)


def score_chunk(question, chunk, is_best=False):
    q = terms(question)
    c = terms(chunk['text'])
    overlap = len(q & c) / max(1, len(q))
    body = chunk['text']
    section = chunk.get('section_title', '').casefold()
    asks_limits = bool(re.search(r'\blimit|caution|uncertain|drawback', question, re.I))
    asks_results = bool(re.search(r'\bfind|result|effect|outcome|compar|impact|report|evidence', question, re.I))
    useful_section = bool(re.search(r'abstract|result|finding|conclusion|discussion|limitation', section))
    value = 3*overlap + .35*bool(RESULT.search(body)) + .25*useful_section
    if asks_results:
        value += .25*bool(RESULT.search(body))
    if asks_limits:
        value += .35*bool(LIMIT.search(body))
    if len(c) < 12:
        value -= .5
    if BACKGROUND.search(body):
        value -= .25
    if is_best:
        value += .2  # Keep the frozen best chunk unless evidence cues clearly help.
    return round(value, 6)


def optimized(top10, question, candidates):
    """Preserve top-three papers; pick one question-oriented source chunk per paper."""
    base = baseline(top10, question)
    rows = []
    for i, (doc, original) in enumerate(zip(top10[:3], base), 1):
        pool = candidates[doc['doc_id']]
        best = max(pool, key=lambda row: (score_chunk(question['question'], row,
                         row['chunk_id'] == doc['best_chunk_id']),
                         row['chunk_id'] == doc['best_chunk_id'], -row['section_index']))
        selected = dict(original)
        selected.update(chunk_id=best['chunk_id'], text=best['text'],
                        source_path=best['source_path'], source_spans=best['source_spans'],
                        section_title=best['section_title'], token_count=best['token_count'],
                        selection_score=score_chunk(question['question'], best,
                                                    best['chunk_id']==doc['best_chunk_id']))
        # The retrieval score is only valid for its original top chunk.
        selected['score'] = doc['score'] if best['chunk_id']==doc['best_chunk_id'] else None
        rows.append(selected)
    return rows
