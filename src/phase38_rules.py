"""Version-neutral deterministic checks; semantic support remains unverified here."""
import re

from src.citation_grounding import REF, _sentences, validate_grounded_answer

NUMBER = re.compile(r'(?<![A-Za-z0-9])\d+(?:\.\d+)?%?')
COMPARATIVE = re.compile(r'显著|相当|更高|更低|增加|降低|有效|无效|可造成|优于|差异|comparable|higher|lower|significantly|effective', re.I)
SCOPE = {
    'medical_setting': re.compile(r'healthcare|hospital|medical setting|ICU|医院|医疗|医护|重症监护', re.I),
    'influenza': re.compile(r'influenza|flu|流感', re.I),
    'age_or_population': re.compile(r'children|pediatric|adult|elderly|儿童|成人|老年|岁', re.I),
    'study_design': re.compile(r'randomi[sz]ed|trial|review|cohort|case report|随机|试验|综述|队列|病例报告', re.I),
}


def normalize_for_blind_evaluation(answer, version):
    """Map A's legacy [1] references to the same evidence IDs as B/C."""
    if version == 'A':
        return re.sub(r'\[(\d+)\]', lambda m: f'[E{m.group(1)}]', answer)
    return answer


def check_answer(answer, evidence, source_checks, question=''):
    base = validate_grounded_answer(answer, evidence, source_checks)
    by_id = {item['evidence_id']: item for item in evidence}
    numeric_candidates = []
    scope_candidates = []
    for index, sentence in enumerate(_sentences(answer), 1):
        ids = [eid for eid in REF.findall(sentence) if eid in by_id]
        plain = REF.sub('', sentence)
        cited_text = ' '.join(by_id[eid]['text'] for eid in ids)
        for number in NUMBER.findall(plain):
            if number not in cited_text and number not in question:
                numeric_candidates.append({'sentence_index': index, 'number': number,
                                           'evidence_ids': ids, 'status': 'Unverified'})
        if ids and COMPARATIVE.search(plain):
            scope = [name for name, pattern in SCOPE.items()
                     if pattern.search(cited_text) and not pattern.search(plain)]
            if scope:
                scope_candidates.append({'sentence_index': index,
                                         'possible_missing_scope': scope,
                                         'evidence_ids': ids, 'status': 'Unverified'})
    return {'version': 'phase38-rules-v1', 'citation': base,
            'numeric_claims_without_same_number_in_cited_text': numeric_candidates,
            'qualifier_loss_candidates': scope_candidates,
            'semantic_support_status': 'Unverified',
            'note': 'These checks are deterministic warnings, not proof of semantic support or error.'}
