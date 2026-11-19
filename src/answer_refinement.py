"""One optional targeted revision for a locally detected contradiction."""
from .evidence_verifier import verify
from .deepseek_client import is_complete_generation


def refine_once(client, question, answer, evidence, check, *, system_prompt):
    critical = [c for c in check['claims'] if c['status'] == 'Contradicted']
    if not critical:
        return {'answer': answer, 'verification': check, 'revised': False, 'generation': None}
    details = '\n'.join(f"{c['claim_text']} — {c['reason']}" for c in critical)
    instruction = (f"原问题：{question}\n原回答：{answer}\n已发现的明确矛盾：\n{details}\n"
                   '请只修正上述矛盾，保留其余有依据内容；仅依据下列原始证据作答。')
    parts, generation = [], None
    for kind, payload in client.stream(instruction, evidence, system_prompt=system_prompt):
        if kind == 'text':
            parts.append(payload)
        else:
            generation = payload
    candidate = ''.join(parts).strip()
    if not is_complete_generation(candidate, generation):
        reason = (generation or {}).get('finish_reason') or '未返回结束原因'
        return {'answer': answer, 'verification': check, 'revised': False,
                'generation': generation, 'warning': f'定向修正未完整生成（{reason}），保留原回答。'}
    new_check = verify(candidate, evidence) if candidate else check
    if candidate and not new_check['critical']:
        return {'answer': candidate, 'verification': new_check, 'revised': True, 'generation': generation}
    return {'answer': answer, 'verification': check, 'revised': False, 'generation': generation,
            'warning': '定向修正未消除明确矛盾，保留原回答。'}
