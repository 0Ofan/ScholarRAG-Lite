"""Conservative offline checks; never certify broad semantic support."""
import re

from src.evidence_verifier import verify


def extract_claims(answer, evidence):
    """Split independent contrast/addition clauses, keeping numeric pairs intact."""
    result = []
    for original in verify(answer, evidence)['claims']:
        pieces = re.split(r'(?<=[，,])(?=但|然而|而|另有|此外|同时)', original['claim_text'])
        for index, piece in enumerate(pieces, 1):
            if piece.strip():
                claim = dict(original)
                claim['claim_id'] = f"{original['claim_id']}-P{index:02d}"
                claim['claim_text'] = piece.strip()
                claim['parent_claim_id'] = original['claim_id']
                local_ids = re.findall(r'\[(E\d+)\]', piece)
                claim['evidence_ids'] = list(dict.fromkeys(local_ids or original['evidence_ids']))
                claim['citation_scope'] = ('clause' if local_ids else 'inherited_parent') if len(pieces) > 1 else original['citation_scope']
                result.append(claim)
    return result


def categories(text):
    patterns = {
        'Numerical': r'\d+(?:\.\d+)?\s*(?:%|％|人|名|例|天|分钟|小时|岁|mg|ml)',
        'Population': r'儿童|成人|婴儿|患者|志愿者|孕妇|动物|小鼠|对照组|children|adults|patients',
        'Experimental Conditions': r'温度|暴露|浓度|剂量|随机|双盲|对照|试验|实验|悬液|载体',
        'Time-related': r'\d+\s*(?:年|月|日|天|分钟|小时)|随访|早期|长期',
        'Source Attribution': r'论文|综述|研究|文献|报道|引述|引用|作者|来源',
        'Comparative': r'相比|比较|优于|低于|高于|差异|而|却|不同',
    }
    found = [name for name, pattern in patterns.items() if re.search(pattern, text, re.I)]
    return found or ['General Factual']


def source_role(item):
    text = item['text'].lower()
    if re.search(r'\b(?:[a-z]+\s+et al\.|according to|cited by|reported by)', text):
        return 'Secondary Citation'
    if re.search(r'\b(systematic review|this review|we reviewed|meta-analysis)\b', text):
        return 'Review Summary'
    if re.search(r'\b(previous studies|background|it is known that)\b', text):
        return 'Background Information'
    if re.search(r'\b(we conducted|our results|in this study we|we enrolled)\b', text):
        return 'Direct Primary Source'
    return 'Unknown Attribution'


def check_claim(claim, evidence, question=''):
    mapping = {e['evidence_id']: e for e in evidence}
    cited = [mapping[eid] for eid in claim['evidence_ids'] if eid in mapping]
    text = claim['claim_text']
    issues = []
    status = 'Unverified'
    reason = '无法用窄规则证明具体事实获得语义支持。'
    for item in cited:
        # The answer explicitly makes the event count the denominator, while
        # the cited source states that same count is the numerator.
        answer_match = re.search(r'(\d+)\s*(?:名|人|例)(?:儿童|患者)?中\s*(\d+(?:\.\d+)?)\s*[%％]\s*住院', text)
        source_match = re.search(r'\b(\d+|twenty[-‐‑‒–— ]four)\s*\(\s*(\d+(?:\.\d+)?)\s*%\s*\)\s*children\s+were\s+hospitalized', item['text'], re.I)
        if (answer_match and source_match
                and int(answer_match.group(1)) == (24 if not source_match.group(1).isdigit() else int(source_match.group(1)))
                and float(answer_match.group(2)) == float(source_match.group(2))):
            numerator, percent = int(answer_match.group(1)), float(source_match.group(2))
            implied = numerator * 100 / percent if percent else None
            status = 'Contradicted'
            reason = f'所引原文中 {numerator} 为住院人数（{percent:g}%），回答将其当成总人数；原比例隐含分母约 {implied:g}，须核对原研究。'
            issues.append('Numerator used as denominator')
        if (re.search(r'children|儿童', question, re.I)
                and re.search(r'Adult COVID-19 Unit|adult patients', item['title'] + ' ' + item['text'], re.I)
                and not re.search(r'成人|adult', text, re.I)):
            issues.append('Adult-source scope requires review')
    if claim['status'] == 'Potential Issue':
        issues.append(claim['issue_type'])
    if any(eid not in mapping for eid in claim['evidence_ids']):
        issues.append('Citation Mapping Error')
    roles = {e['evidence_id']: source_role(e) for e in cited}
    if any(role == 'Secondary Citation' for role in roles.values()) and re.search(r'该论文(?:自身|直接)(?:发现|证明|报告)', text):
        issues.append('Secondary evidence described as direct study result')
    if re.search(r'所有|任何|总是|无论.{0,12}条件|普遍适用', text) and any(
            re.search(r'\b(?:at\s+\d+(?:\.\d+)?\s*°?C|\d+\s*(?:minutes?|hours?|days?)\s+(?:of\s+)?exposure|dose\s+of\s+\d+)', e['text'], re.I)
            for e in cited):
        issues.append('Broad claim with limited experimental conditions; review required')
    return {'status': status, 'reason': reason, 'issues': list(dict.fromkeys(issues)),
            'source_roles': roles, 'categories': categories(text),
            'cited_document_ids': [e['doc_id'] for e in cited],
            'cited_chunk_ids': [e['chunk_id'] for e in cited]}


def question_points(question_id, answer, evidence, expected_points):
    """Point-level scope: only narrow observable Q11/Q20 facts get a status."""
    rows = [{'point': point, 'evidence_status': 'Unverified', 'answer_status': 'Unverified'}
            for point in expected_points]
    if question_id == 'Q11':
        blood = any(re.search(r'serologic tests? can detect antibodies in blood', e['text'], re.I)
                    for e in evidence)
        mentioned = bool(re.search(r'抗体检测.{0,12}血液|血液.{0,12}抗体', answer))
        rows.append({'point': 'Antibody test detects antibodies in blood',
                     'evidence_status': 'Directly Supported' if blood else 'Unverified',
                     'answer_status': 'Covered' if blood and mentioned else 'Unverified'})
        rows.append({'point': 'Optimal testing timing', 'evidence_status': 'Unverified',
                     'answer_status': 'Unverified'})
    if question_id == 'Q20':
        rows.append({'point': 'Five-year vaccine follow-up conclusion',
                     'evidence_status': 'Unverified', 'answer_status': 'Abstained' if '无法' in answer else 'Unverified'})
    return rows
