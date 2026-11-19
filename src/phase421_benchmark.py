"""Conservative, answer-independent benchmark over saved Phase 4.18 data."""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.phase420_verifier import source_role

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/phase421'

# Only these point/evidence relationships were checked against saved source text.
# Entries are keyed by frozen point, never selected from generated answers.
CHECKED_REFERENCES = {
    'Q01-P01': [('E1', 'Partial Evidence', '综述片段说明了纳入表面持续与消毒剂灭活原始研究；没有在此给出所有具体结果。'),
                ('E4', 'Indirect Evidence', '另一论文转述目标综述的表面持续时间及消毒条件。'),
                ('E7', 'Indirect Evidence', '另一论文引用目标综述，不能作为目标综述直接原文。')],
    'Q01-P02': [('E1', 'Partial Evidence', '片段明确列出 suspension/carrier tests，但不能凭此推断现实传播风险。')],
    'Q02-P01': [('E1', 'Direct Evidence', '目标研究原文写 IgG-IgM lateral flow assay，结果时间少于 15 分钟。')],
    'Q03-P01': [('E1', 'Direct Evidence', '目标论文片段明确写 pragmatic/prospective/randomized/multicenter/open-label 设计。')],
    'Q07-P01': [('E1', 'Direct Evidence', '系统综述片段直接写儿童证据有限、通常轻症及无症状报告。')],
    'Q07-P02': [('E3', 'Direct Evidence', '新生儿病例报告指出幼儿可能出现呼吸衰竭。')],
    'Q07-P03': [('E1', 'Partial Evidence', '系统综述仅有小型病例系列和病例报告，支持证据有限。'),
                ('E2', 'Insufficient Evidence', 'E2 是儿童医院中的成人病区研究，不能直接支持儿童结局的选择偏倚。')],
    'Q11-P01': [('E2', 'Direct Evidence', '片段明确写 RT-PCR 直接检测 SARS-CoV-2 RNA。')],
    'Q11-P02': [('E6', 'Partial Evidence', '片段明确写血清学检测血液中的抗体；未提供最佳检测时间。')],
    'Q19-P01': [('E1', 'Partial Evidence', '片段明确说免疫反应与保护水平及持续时间的关系尚不确定。')],
    'Q20-P01': [('E1', 'Insufficient Evidence', '早期 45 人安全性试验不能确立五年安全性或有效性。')],
    'Q20-P02': [('E1', 'Direct Evidence', '片段记载 mRNA-1273 的早期 45 人安全性试验。'),
                ('E3', 'Direct Evidence', '片段记载 mRNA-1273 I 期招募和安全性、反应原性、免疫原性目的。')],
    'Q20-P03': [('E1', 'Insufficient Evidence', '所给早期研究没有五年随访结局；合理拒绝长期确定性判断。')],
}


def reference_role(e):
    role = source_role(e)
    title = e['title'].lower()
    if role == 'Unknown Attribution' and 'systematic review' in title:
        return 'Review Summary'
    if role == 'Unknown Attribution' and 'case report' in title and 'our patient' in e['text'].lower():
        return 'Direct Primary Source'
    return role


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def point_observation(qid, point_id, answer, k):
    """Only explicit, previously reviewed cases; no general semantic matching."""
    if point_id == 'Q02-P01' and re.search(r'IgM', answer, re.I) and re.search(r'IgG', answer, re.I) and '15分钟' in answer:
        return 'Partially Covered (limited text check)', '回答明确出现 IgM、IgG 和 15 分钟；侧向流及研究归属仍需逐句复核。'
    if point_id == 'Q11-P02':
        if k == 8 and re.search(r'血液中的抗体', answer):
            return 'Partially Covered', 'K8 回答写出血液抗体对象；最佳检测时机仍缺证据。'
        return 'Unverified', '未确认检测对象与时机两部分均得到回答。'
    if point_id in ('Q19-P01', 'Q19-P03', 'Q20-P01', 'Q20-P03') and re.search(r'无法确定|不能确立|无法确立', answer):
        return 'Appropriate Abstention (limited review)', '已保存回答未给出缺乏证据的精确长期结论。'
    return 'Unverified', '跨语言要点覆盖与语义支持未完成可靠逐项核查。'


def abstention_case(qid, answer, k):
    if qid == 'Q20' and re.search(r'无法确定|不能确立|无法确立', answer) and re.search(r'五年', answer):
        return 'Appropriate Abstention', '现有 2020 年片段不能证明五年疫苗结局；Phase 4.19 个案原文核对。'
    if qid == 'Q19' and re.search(r'无法确定|不能确立', answer):
        return 'Appropriate Abstention', '所引片段明确表示保护持续时间未知；限于已审个案。'
    if qid == 'Q11' and re.search(r'无法从这些记录确定|最佳时机无法|最佳检测时间', answer):
        return 'Partial Abstention', '检测对象与最佳时机须分别判断；最佳时机缺直接片段。'
    return 'Unverified', '没有足够逐要点审阅依据判定拒答质量。'


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError('Phase 4.21 results already exist; refusing to overwrite')
    frozen = load_json(ROOT / 'experiments/configs/phase36_questions.json')['questions']
    offline = load_json(ROOT / 'experiments/results/phase418/offline_records.json')
    hashes = load_json(ROOT / 'experiments/results/phase419/aggregate_quality_metrics.json')['original_answer_sha256']
    historical = {r['question_id']: r for r in load_json(ROOT / 'experiments/results/phase36/reviewer_scores.json')['reviews']}
    prior_claims = list(csv.DictReader((ROOT / 'experiments/results/phase420/claim_level_comparison.csv').open(encoding='utf-8-sig', newline='')))
    claim_groups = defaultdict(list)
    for claim in prior_claims:
        claim_groups[(claim['question_id'], int(claim['k']))].append(claim)
    benchmark, reference_map, point_checks, answer_rows, claim_rows = [], [], [], [], []
    for q in frozen:
        qid = q['question_id']
        source_evidence = offline[qid]['groups']['8']['evidence']
        by_id = {e['evidence_id']: e for e in source_evidence}
        points = []
        for index, point in enumerate(q['expected_answer_points'], 1):
            pid = f'{qid}-P{index:02d}'
            annotations = CHECKED_REFERENCES.get(pid, [])
            # An unreviewed candidate is visible, but never treated as support.
            if not annotations:
                annotations = [('E1', 'Unverified', '仅为冻结检索的候选片段；未完成该要点的语义支持核查。')]
            statuses = []
            for eid, support, basis in annotations:
                e = by_id[eid]
                statuses.append(support)
                reference_map.append({'question_id': qid, 'point_id': pid, 'evidence_id': eid,
                                      'available_from_k': e['paper_rank'], 'document_id': e['doc_id'],
                                      'paper_title': e['title'], 'chunk_id': e['chunk_id'],
                                      'evidence_text': e['text'], 'source_role': reference_role(e),
                                      'support_status': support, 'basis': basis,
                                      'provenance': 'Codex-assisted source-text check' if support != 'Unverified' else 'unreviewed candidate'})
            points.append({'point_id': pid, 'point_text': point, 'evidence_reference_ids': [a[0] for a in annotations],
                           'best_checked_status': ('Direct Evidence' if 'Direct Evidence' in statuses else
                                                   'Partial Evidence' if 'Partial Evidence' in statuses else
                                                   'Indirect Evidence' if 'Indirect Evidence' in statuses else
                                                   'Insufficient Evidence' if 'Insufficient Evidence' in statuses else 'Unverified'),
                           'review_status': 'limited_source_check' if any(s != 'Unverified' for s in statuses) else 'Unverified'})
        benchmark.append({'question_id': qid, 'original_question': q['question'], 'category': q['category'],
                          'source_topic_ids': q['source_topic_ids'], 'relevant_document_ids': q['relevant_document_ids'],
                          'frozen_evidence_availability': q['evidence_availability'],
                          'frozen_evaluation_notes': q['evaluation_notes'], 'points': points,
                          'historical_phase36_review_note': historical[qid]['review_note'],
                          'historical_note_scope': 'Phase 3.6 Top-3 answer only; not ground truth for Phase 4.18'})
        for k in (3, 5, 8):
            path = ROOT / f'experiments/results/phase418/raw_answers/{qid}_K{k}.json'
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != hashes[f'{qid}_K{k}']:
                raise ValueError(f'{qid} K{k}: answer SHA-256 mismatch')
            saved = json.loads(raw)
            evidence = offline[qid]['groups'][str(k)]['evidence']
            if saved['question'] != q['question'] or saved['evidence'] != evidence or saved['status'] != 'complete':
                raise ValueError(f'{qid} K{k}: answer or evidence mismatch')
            if any(c['status'] != 'valid' for c in offline[qid]['groups'][str(k)]['source_checks']):
                raise ValueError(f'{qid} K{k}: source check failed')
            valid_ids = set(e['evidence_id'] for e in evidence)
            claims = claim_groups[(qid, k)]
            for claim in claims:
                ids = json.loads(claim['citation_ids'])
                if any(eid not in valid_ids for eid in ids):
                    raise ValueError(f'{qid} K{k}: invalid claim citation')
                claim_rows.append(dict(claim, experiment_group=f'Top-{k}',
                                       semantic_support_scope='narrow rule only; Unverified is not correct'))
            checks = []
            coverage = Counter()
            for point in points:
                status, basis = point_observation(qid, point['point_id'], saved['answer'], k)
                refs = [r for r in reference_map if r['point_id'] == point['point_id'] and r['available_from_k'] <= k]
                point_checks.append({'question_id': qid, 'k': k, 'point_id': point['point_id'],
                                     'point_text': point['point_text'], 'available_evidence_ids': json.dumps([r['evidence_id'] for r in refs]),
                                     'evidence_statuses': json.dumps([r['support_status'] for r in refs]),
                                     'answer_observation': status, 'basis': basis})
                checks.append({'point_id': point['point_id'], 'observation': status})
                statuses = [r['support_status'] for r in refs]
                strongest = next((s for s in ('Direct Evidence', 'Partial Evidence', 'Indirect Evidence',
                                              'Insufficient Evidence') if s in statuses), 'Unverified')
                coverage[strongest] += 1
            abstention, abstention_basis = abstention_case(qid, saved['answer'], k)
            issues = sorted({issue for c in claims for issue in json.loads(c['review_issues'])})
            label_counts = Counter(c['phase420_status'] for c in claims)
            usage = saved['generation']['usage']
            answer_rows.append({'question_id': qid, 'k': k, 'experiment_group': f'Top-{k}',
                                'original_question': q['question'], 'original_answer': saved['answer'],
                                'original_evidence': json.dumps(evidence, ensure_ascii=False),
                                'answer_file_sha256': hashes[f'{qid}_K{k}'],
                                'point_observations': json.dumps(checks, ensure_ascii=False),
                                'evidence_coverage_counts': json.dumps(dict(coverage)),
                                'claim_count': len(claims), 'claim_status_counts': json.dumps(dict(label_counts)),
                                'citation_validity': 'Valid' if not saved['citation_check']['invalid'] else 'Invalid',
                                'citation_source_mapping': 'Valid', 'citation_semantic_support': 'Unverified',
                                'source_attribution': 'Review Required' if any('source' in s.lower() for s in issues) else 'Unverified',
                                'completeness_score': 'Unverified', 'faithfulness_score': 'Unverified',
                                'relevance_score': 'Unverified', 'citation_support_score': 'Unverified',
                                'abstention_quality': abstention, 'abstention_basis': abstention_basis,
                                'confirmed_contradictions': label_counts['Contradicted'],
                                'review_issues': json.dumps(issues),
                                'api_completion_seconds': saved['api_completion_seconds'],
                                'input_tokens': usage['input_tokens'], 'output_tokens': usage['output_tokens'],
                                'estimated_cost_cny': saved['estimated_cost']['estimated_cny']})
    pairs = []
    for q in frozen:
        qid = q['question_id']
        for a, b in ((3, 5), (3, 8), (5, 8)):
            left = next(x for x in answer_rows if x['question_id'] == qid and x['k'] == a)
            right = next(x for x in answer_rows if x['question_id'] == qid and x['k'] == b)
            local = 'Q07 K8 adds a confirmed denominator error and adult-study scope hint' if qid == 'Q07' and b == 8 else (
                    'Q11 K8 adds cited blood-antibody target; optimal timing remains unverified' if qid == 'Q11' and b == 8 else (
                    'Both appropriately abstain from five-year conclusions' if qid == 'Q20' else 'No sufficiently reviewed quality comparison'))
            pairs.append({'question_id': qid, 'comparison': f'Top-{a} vs Top-{b}',
                          'classification': 'Mixed' if qid == 'Q07' and b == 8 else 'Unverified',
                          'classification_scope': 'limited reviewed facts only' if qid == 'Q07' and b == 8 else 'whole-answer quality unverified',
                          'local_finding': local, 'left_confirmed_contradictions': left['confirmed_contradictions'],
                          'right_confirmed_contradictions': right['confirmed_contradictions'],
                          'latency_difference_seconds': round(right['api_completion_seconds'] - left['api_completion_seconds'], 6),
                          'input_token_difference': right['input_tokens'] - left['input_tokens'],
                          'estimated_cost_difference_cny': round(right['estimated_cost_cny'] - left['estimated_cost_cny'], 8)})
    historical_cost = load_json(ROOT / 'experiments/results/phase418/aggregate_metrics.json')['groups']
    metrics = {'questions': len(benchmark), 'frozen_points': sum(len(q['points']) for q in benchmark),
               'reference_rows': len(reference_map), 'limited_checked_reference_rows': sum(r['support_status'] != 'Unverified' for r in reference_map),
               'answers': len(answer_rows), 'claim_units': len(claim_rows),
               'claim_status_counts': dict(Counter(c['phase420_status'] for c in claim_rows)),
               'answer_scores': 'Unverified: insufficient point and claim semantic review',
               'pair_classifications': dict(Counter(p['classification'] for p in pairs)),
               'evidence_point_coverage_by_group': {str(k): dict(sum((Counter(json.loads(r['evidence_coverage_counts']))
                   for r in answer_rows if r['k'] == k), Counter())) for k in (3, 5, 8)},
               'historic_api_group_metrics': {k: {key: historical_cost[k][key] for key in ('api_completion_seconds','input_tokens','output_tokens','estimated_cost_cny')} for k in ('3','5','8')},
               'cost_note': 'estimated from actual saved Usage and published price at Phase 4.18 run time; not provider bill'}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json('question_benchmark.json', benchmark)
    write_json('evidence_reference_map.json', reference_map)
    write_csv('point_evaluation.csv', point_checks)
    write_csv('claim_evaluation.csv', claim_rows)
    write_csv('answer_evaluation.csv', answer_rows)
    write_csv('pairwise_comparison.csv', pairs)
    write_json('evaluation_metrics.json', metrics)
    print(json.dumps({key: metrics[key] for key in ('questions','frozen_points','reference_rows','limited_checked_reference_rows','answers','claim_units','claim_status_counts','pair_classifications')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
