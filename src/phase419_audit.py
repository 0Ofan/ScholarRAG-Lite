"""Offline provenance and conservative claim audit of saved Phase 4.18 answers."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P418 = ROOT / 'experiments/results/phase418'
OUT = ROOT / 'experiments/results/phase419'

from src.citation_grounding import validate_grounded_answer
from src.evidence_verifier import verify


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def csv_write(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError('Phase 4.19 审计结果已存在；不会覆盖。')
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = {q['question_id']: q for q in json.loads(
        (ROOT / 'experiments/configs/phase36_questions.json').read_text(encoding='utf-8'))['questions']}
    offline = json.loads((P418 / 'offline_records.json').read_text(encoding='utf-8'))
    claims, per_answer, hashes = [], [], {}
    for qid, question in frozen.items():
        for k in (3, 5, 8):
            path = P418 / 'raw_answers' / f'{qid}_K{k}.json'
            raw = path.read_bytes()
            row = json.loads(raw)
            hashes[f'{qid}_K{k}'] = hashlib.sha256(raw).hexdigest()
            evidence = row['evidence']
            original = offline[qid]['groups'][str(k)]['evidence']
            if row['question'] != question['question'] or evidence != original or len(evidence) != k:
                raise ValueError(f'{qid} K{k}: question/evidence mismatch')
            if row['status'] != 'complete' or row['finish_reason'] != 'stop':
                raise ValueError(f'{qid} K{k}: not a completed response')
            source_checks = offline[qid]['groups'][str(k)]['source_checks']
            if any(item['status'] != 'valid' for item in source_checks):
                raise ValueError(f'{qid} K{k}: original source mapping failed')
            citation = validate_grounded_answer(row['answer'], evidence, source_checks)
            local = verify(row['answer'], evidence)
            mapping = {e['evidence_id']: e for e in evidence}
            for item in local['claims']:
                ids = item['evidence_ids']
                sources = [mapping[eid] for eid in ids if eid in mapping]
                label, reason = 'Unverified', '引用与原始来源可核验；当前规则不能证明语义支持。'
                provenance = 'not_semantically_reviewed'
                if qid == 'Q07' and k == 8 and '24名儿童中60%住院' in item['claim_text']:
                    label = 'Contradicted'
                    reason = 'E5 原文为 24 人（60%）住院：24 是住院人数，不是 60% 的分母。'
                    provenance = 'Codex_assisted_original_text_review'
                claims.append({'question_id': qid, 'k': k, 'claim_id': item['claim_id'],
                               'claim_text': item['claim_text'],
                               'citation_ids': json.dumps(ids, ensure_ascii=False),
                               'document_ids': json.dumps([e['doc_id'] for e in sources]),
                               'paper_titles': json.dumps([e['title'] for e in sources], ensure_ascii=False),
                               'chunk_ids': json.dumps([e['chunk_id'] for e in sources]),
                               'original_evidence': json.dumps([e['text'] for e in sources], ensure_ascii=False),
                               'status': label, 'reason': reason,
                               'local_verifier_status': item['status'],
                               'local_verifier_reason': item['reason'],
                               'review_provenance': provenance})
            per_answer.append({'question_id': qid, 'k': k, 'category': question['category'],
                               'expected_points': json.dumps(question['expected_answer_points'], ensure_ascii=False),
                               'claims': len(local['claims']), 'valid_citation_ids': len(citation['valid']),
                               'invalid_citation_ids': len(citation['invalid']),
                               'source_mapping_valid': True,
                               'completeness': 'Unverified', 'faithfulness': 'Unverified',
                               'relevance': 'Unverified', 'citation_support': 'Unverified'})
    csv_write(OUT / 'claim_level_audit.csv', claims)
    pairs = []
    for qid in frozen:
        for a, b in ((3, 5), (3, 8), (5, 8)):
            status = 'Unverified'
            note = '未完成对两组全部事实句及冻结要点的可靠语义复核。'
            if qid == 'Q07' and b == 8:
                status, note = ('Mixed', 'K8 增加儿童分组资料，但新出现 E5 分母误读及 E2 成人病区局限归属风险。')
            elif qid == 'Q20':
                status, note = ('Unchanged', '三组均未提供五年随访证据，回答均拒绝确立五年安全性和有效性。')
            elif qid == 'Q11' and a == 3 and b == 8:
                note = 'K8 E6 补充血液抗体检测对象；完整问答质量仍未全部核实。'
            pairs.append({'question_id': qid, 'category': frozen[qid]['category'],
                          'comparison': f'Top-{a} vs Top-{b}', 'classification': status,
                          'supplemented_points': note if '补充' in note or '增加' in note else '',
                          'remaining_omissions_or_new_errors': note,
                          'source_attribution_review': 'partially_reviewed' if qid in ('Q01','Q07','Q11','Q20') else 'Unverified',
                          'irrelevant_information_review': 'Unverified'})
    csv_write(OUT / 'per_question_comparison.csv', pairs)
    counts = Counter(c['status'] for c in claims)
    pair_counts = {name: dict(Counter(p['classification'] for p in pairs if p['comparison'] == name))
                   for name in ('Top-3 vs Top-5', 'Top-3 vs Top-8', 'Top-5 vs Top-8')}
    group_counts = {str(k): dict(Counter(c['status'] for c in claims if c['k'] == k))
                    for k in (3, 5, 8)}
    quality = {'answers': len(per_answer), 'questions': len(frozen), 'claim_units': len(claims),
               'claim_status_counts': dict(counts), 'claim_status_by_group': group_counts,
               'pair_classifications': pair_counts,
               'quality_scores': 'not_computed: semantic review coverage insufficient',
               'original_answer_sha256': hashes}
    dump(OUT / 'aggregate_quality_metrics.json', quality)
    dump(OUT / 'evidence_support_summary.json', {'citation_id_and_chunk_mapping_valid_for': 60,
          'semantic_support_note': 'Valid ID and mapped Chunk do not establish that a claim is supported.',
          'reviewed_example': 'Q07 K8 denominator contradiction is based on source text E5.',
          'local_verifier_hints': dict(Counter(c['local_verifier_status'] for c in claims))})
    dump(OUT / 'audit_cases.json', json.loads((P418 / 'case_review.json').read_text(encoding='utf-8')))
    print('audited answers', len(per_answer), 'claim units', len(claims), 'labels', dict(counts))


if __name__ == '__main__':
    main()
