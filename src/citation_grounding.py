"""Structural citation checks; semantic support always requires source review."""
import re

REF = re.compile(r'\[(E\d+|\d+)\]')
FACT_HINT = re.compile(r'研究|试验|论文|患者|病例|样本|报告|显示|发现|结果|显著|风险|传播|疗效|敏感性|死亡|口罩|效果|保护|有效|无效|高于|低于|差异|study|trial|patients?|results?|showed|reported|\d', re.I)


def verify_evidence_source(evidence, source_map):
    """Compare each supplied unit to immutable chunk records before model use."""
    checks = []
    for item in evidence:
        original = source_map.get(item['chunk_id'])
        fields = ('doc_id', 'title', 'chunk_id', 'text')
        mismatches = [field for field in fields if original is None or item.get(field) != original.get(field)]
        checks.append({'evidence_id': item['evidence_id'], 'citation': item['citation'],
                       'doc_id': item['doc_id'], 'chunk_id': item['chunk_id'],
                       'status': 'valid' if not mismatches else 'invalid',
                       'mismatched_fields': mismatches})
    return checks


def _sentences(answer):
    pieces = re.split(r'(?<=[。！？!?])\s*|(?<=[.])\s+|\n+', answer)
    return [piece.strip() for piece in pieces if piece.strip()]


def validate_grounded_answer(answer, evidence, source_checks=None):
    """Verify provenance/format and flag claims for manual semantic review."""
    known = {item['evidence_id']: item for item in evidence}
    source_status = {row['evidence_id']: row['status'] for row in source_checks or []}
    observed = {match.group(1) for match in REF.finditer(answer)}
    invalid = sorted(observed - known.keys())
    valid = sorted(eid for eid in observed & known.keys()
                   if source_status.get(eid) != 'invalid')
    invalid.extend(eid for eid in observed & known.keys()
                   if source_status.get(eid) == 'invalid')
    mapped = []
    for eid in valid:
        item = known[eid]
        status = source_status.get(eid, 'unverified')
        mapped.append({'evidence_id': eid, 'citation': f'[{eid}]',
                       'doc_id': item['doc_id'], 'title': item['title'],
                       'chunk_id': item['chunk_id'], 'source_status': status,
                       'semantic_support': 'unverified'})
    invalid = sorted(set(invalid))
    checks = []
    uncited_facts = []
    multi_source_review = []
    position_issues = []
    attribution_issues = []
    for position, sentence in enumerate(_sentences(answer), 1):
        refs = list(dict.fromkeys(REF.findall(sentence)))
        plain = REF.sub('', sentence).strip()
        factual_candidate = bool(FACT_HINT.search(plain))
        at_end = not refs or bool(re.search(r'(?:\[(?:E\d+|\d+)\]\s*)+[。！？.!?]?\s*$', sentence))
        named_without_own_citation = [eid for eid, item in known.items()
            if ((item['doc_id'] in plain) or
                (len(item['title']) >= 20 and item['title'].casefold() in plain.casefold()))
            and eid not in refs]
        record = {'sentence_index': position, 'text': sentence, 'citation_ids': refs,
                  'factual_candidate': factual_candidate,
                  'citation_at_sentence_end': at_end,
                  'named_source_without_matching_citation': named_without_own_citation,
                  'semantic_support': 'unverified' if factual_candidate else 'not_applicable'}
        checks.append(record)
        if factual_candidate and not refs:
            uncited_facts.append(position)
        if len(refs) > 1:
            multi_source_review.append(position)
        if not at_end:
            position_issues.append(position)
        if named_without_own_citation:
            attribution_issues.append(position)
    return {'valid': valid, 'invalid': invalid,
            'overall_status': 'invalid' if invalid or attribution_issues else
                              'unverified' if any(s['factual_candidate'] for s in checks) else 'valid',
            'unverified_semantic_support': valid,
            'unused_evidence': sorted(known.keys() - observed),
            'citation_mapping': mapped,
            'sentences': checks,
            'uncited_factual_candidates': uncited_facts,
            'multi_source_sentences_for_review': multi_source_review,
            'nonterminal_citation_sentences': position_issues,
            'explicit_attribution_mismatch_sentences': attribution_issues,
            'automatic_check_limit': '结构和原始来源可核验；句子是否被片段支持、是否错误跨论文归属须人工复核。'}
