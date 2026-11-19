"""Add conservative source-role hints without changing retrieved text."""
import re


def source_role(item):
    text = item['text']
    low = text.lower()
    if re.search(r'\b(we (will|plan|aim|propose)|this (study|trial) (will|aims)|protocol for|results will)\b', low):
        return 'Study Plan'
    if re.search(r'\b(in (our|this) (study|trial|cohort)|we (found|observed|measured|enrolled|randomized))\b', low):
        return 'Primary Study Result'
    if re.search(r'\b(reviewed in|according to|et al\.|previous (studies|findings)|a recent (study|correspondence))\b', low):
        return 'Background / Cited Study'
    if re.search(r'\b(systematic review|we reviewed|this review)\b', low):
        return 'Review / Summary'
    return 'Unclassified'


def enrich(evidence):
    return [{**item, 'retrieval_rank': item.get('paper_rank', i),
             'original_similarity_score': item.get('score'),
             'source_role': source_role(item)} for i, item in enumerate(evidence, 1)]
