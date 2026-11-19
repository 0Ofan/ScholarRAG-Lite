"""Conservative checks for a few consequential contradictions; not semantic proof."""
import re
import time

from .citation_grounding import validate_grounded_answer


def _denominator_hint(answer, evidence):
    """Only warn when two independent count/percent pairs imply the same cohort size."""
    if not re.search(r'\d+(?:\.\d+)?\s*[%％]', answer) or not re.search(r'通常|普遍|总体|整体', answer):
        return None
    for item in evidence:
        pairs = re.findall(r'\b(\d+)\s*\(\s*(\d+(?:\.\d+)?)\s*%\s*\)', item['text'])
        sizes = []
        for count, percent in pairs:
            if float(percent) > 0:
                size = int(count) * 100 / float(percent)
                if size >= 10 and abs(size - round(size)) < .01:
                    sizes.append(round(size))
        # A chunk may contain several cohorts. Require two agreeing pairs and
        # leave the inferred denominator as a review hint, not a verified fact.
        repeated = {n for n in sizes if sizes.count(n) >= 2}
        if len(repeated) == 1:
            size = repeated.pop()
            declared = re.search(r'(?:总体|总计|共|样本(?:量|数)?)\s*(\d+)\s*(?:名|人|例|个|位)|(?:n|N)\s*=\s*(\d+)', answer)
            if declared:
                stated = int(declared.group(1) or declared.group(2))
                if stated == size:
                    return None
                return item['evidence_id'], size, stated
            return item['evidence_id'], size, None
    return None


def _design_overgeneralization(claim, source):
    if not (re.search(r'两项|两个|多项|全部|均为|都为|both|all', claim, re.I)
            and re.search(r'随机|双盲|III期|Ⅲ期|phase\s*(?:iii|3)', claim, re.I)):
        return False
    low = source.lower()
    if not re.search(r'\b(two|multiple|several)\s+(?:clinical\s+)?trials?\b', low):
        return False
    # Each claimed attribute must occur in an explicit common-design statement.
    shared = re.search(r'\b(?:both|all)\s+(?:of\s+the\s+)?trials?\s+(?:were|are)\s+([^.]*)', low)
    if shared:
        claimed = [(r'随机|random', r'randomiz'), (r'双盲|double.blind', r'double.blind'),
                   (r'III期|Ⅲ期|phase\s*(?:iii|3)', r'phase\s*(?:iii|3)')]
        if all(not re.search(answer_term, claim, re.I) or re.search(source_term, shared.group(1), re.I)
               for answer_term, source_term in claimed):
            return False
    return True


def verify(answer, evidence):
    started = time.perf_counter()
    mapping = {e['evidence_id']: e for e in evidence}
    structural = validate_grounded_answer(answer, evidence)
    checks = []
    denominator = _denominator_hint(answer, evidence)
    for sentence in structural['sentences']:
      for part_index, part in enumerate(re.split(r'[；;]', sentence['text']), 1):
        text = part.strip()
        if not text:
            continue
        local_ids = re.findall(r'\[(E\d+)\]', text)
        ids = list(dict.fromkeys(local_ids or sentence['citation_ids']))
        cited = [mapping[i] for i in ids if i in mapping]
        # An uncited absence statement must be checked against all supplied evidence.
        scope = cited or list(mapping.values())
        source = ' '.join(e['text'] for e in scope).lower()
        # Numeric co-occurrence is only a review hint; it cannot establish subject or study linkage.
        numeric_hints = []
        for number in re.findall(r'(?<![A-Za-z0-9])\d+(?:\.\d+)?(?=\s*(?:%|％|人|例|天|日|分钟|小时|mg\b|ml\b|μmol\b|patients?\b|subjects?\b|minutes?\b|days?\b))', text, re.I):
            numeric_hints.append({'value': number,
                                  'same_number_in_reviewed_evidence': bool(re.search(r'(?<!\d)' + re.escape(number) + r'(?!\d)', source)),
                                  'relationship_status': 'Unverified'})
        issue = None
        if re.search(r'单臂|single.arm', text, re.I) and re.search(r'untreated controls?|control group|对照组', source):
            issue = ('Contradicted', 'Study Design', '回答称单臂，但对应证据明示对照组。')
        if not issue and _design_overgeneralization(text, source):
            issue = ('Potential Issue', 'Study Design Overgeneralization',
                     '回答把多项试验统一赋予研究设计属性；所引片段没有分别说明每项试验均满足该属性。')
        absence = bool(re.search(r'未(提供|说明|报告|列出)|没有(提供|给出|说明)|无.{0,6}(信息|数据)', text))
        if absence:
            categories = [
                (r'表面类型', r'\b(metal|glass|plastic)\b', 'Surface Type'),
                (r'浓度', r'\d+(?:\.\d+)?\s*[%％]', 'Concentration'),
                (r'接触时间|暴露时间', r'\b(?:within\s*)?\d+\s*(?:min|minute|hour|day)s?\b', 'Exposure Duration'),
                (r'样本数|样本数量', r'\b(?:\d+\s*patients|\d+\s*subjects|\d+\s*volunteers)\b', 'Sample Size'),
            ]
            for statement, pattern, label in categories:
                if re.search(statement, text) and re.search(pattern, source, re.I):
                    issue = ('Potential Issue', 'False Absence', f'回答称缺少 {label}，证据中有相关原文；需核对是否为同一研究条件。')
                    break
        if any(i not in mapping for i in ids):
            issue = ('Contradicted', 'Citation Mapping', '引用编号不在本次证据中。')
        if (not issue and denominator and denominator[0] in ids
                and re.search(r'通常|普遍|总体|整体', text)):
            if denominator[2] is None:
                issue = ('Potential Issue', 'Missing Denominator',
                         f'同一证据中至少两组人数/百分比一致指向约 {denominator[1]} 人；回答概括该人群却未写总体规模。原片段未明示总 N，需核对全文及子集。')
            else:
                issue = ('Potential Issue', 'Population Mismatch',
                         f'回答称总体 {denominator[2]} 人，但同一证据中至少两组人数/百分比指向约 {denominator[1]} 人；需核对全文及子集。')
        checks.append({'claim_id': f"S{sentence['sentence_index']:02d}-C{part_index:02d}", 'claim_text': text,
                       'evidence_ids': ids, 'document_ids': [e['doc_id'] for e in scope],
                       'citation_scope': 'clause' if local_ids else 'sentence_fallback',
                       'source_roles': [e.get('source_role', 'Unclassified') for e in scope],
                       'original_evidence': [e['text'] for e in scope],
                       'numeric_hints': numeric_hints,
                       'status': issue[0] if issue else 'Unverified',
                       'issue_type': issue[1] if issue else None,
                       'reason': issue[2] if issue else '轻量规则无法证明语义支持；请结合原文复核。'})
    return {'claims': checks, 'structural': structural,
            'seconds': time.perf_counter() - started,
            'critical': any(c['status'] == 'Contradicted' for c in checks)}
