"""Export the existing 54 anonymous answer inputs in fixed six-sample batches."""
import hashlib
import json
import re
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.phase422_pilot_export import canonical_hash, check_clean, sha

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'experiments/results/phase422'
EXPORT = RESULTS / 'full_evaluation_export'
ARCHIVE = RESULTS / 'phase422_remaining_54_export.zip'
PENDING = RESULTS / 'pilot_evaluation_pending.json'
BATCHES = (
    ('Q01', 'Q02'), ('Q03', 'Q04'), ('Q05', 'Q06'),
    ('Q08', 'Q09'), ('Q10', 'Q12'), ('Q13', 'Q14'),
    ('Q15', 'Q16'), ('Q17', 'Q18'), ('Q19', 'Q20'),
)
BATCH_FILES = tuple(f'batch_{n:02d}.json' for n in range(1, 10))
OTHER_FILES = ('remaining_54_answers.json', 'evaluation_instructions.md',
               'evaluation_schema.json', 'README.md', 'answer_evaluation_template.json',
               'claim_evaluation_template.csv', 'point_evaluation_template.csv')
MANIFEST = 'export_manifest.json'
FILES = BATCH_FILES + OTHER_FILES + (MANIFEST,)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(path, header, rows):
    import csv
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def build():
    blind = load(RESULTS / 'blind_evaluation_dataset.json')['samples']
    mapping = {m['sample_id']: m for m in load(RESULTS / 'anonymization_mapping.json')['mapping']}
    references = load(ROOT / 'experiments/results/phase421/evidence_reference_map.json')
    questions = {q['question_id']: q for q in load(ROOT / 'experiments/results/phase421/question_benchmark.json')}
    original_hashes = load(ROOT / 'experiments/results/phase419/aggregate_quality_metrics.json')['original_answer_sha256']
    offline = load(ROOT / 'experiments/results/phase418/offline_records.json')
    expected_qids = {qid for pair in BATCHES for qid in pair}
    selected = [sample for sample in blind if sample['sample_id'][:3] in expected_qids]
    if len(selected) != 54 or Counter(s['sample_id'][:3] for s in selected) != {qid: 3 for qid in expected_qids}:
        raise ValueError('expected exactly 54 answers across 18 questions')
    records = []
    for sample in selected:
        sid = sample['sample_id']
        qid = sid[:3]
        source = mapping[sid]
        source_key = f"{qid}_K{source['evidence_k']}"
        path = ROOT / f'experiments/results/phase418/raw_answers/{source_key}.json'
        raw_bytes = path.read_bytes()
        original = json.loads(raw_bytes)
        if hashlib.sha256(raw_bytes).hexdigest() != source['source_sha256'] or source['source_sha256'] != original_hashes[source_key]:
            raise ValueError(f'{sid}: original file hash mismatch')
        if (source['question_id'] != qid or sample['question'] != questions[qid]['original_question']
                or sample['answer'] != original['answer'] or original['evidence'] != offline[qid]['groups'][str(source['evidence_k'])]['evidence']):
            raise ValueError(f'{sid}: question, answer or evidence differs from original')
        frozen_points = questions[qid]['points']
        if [p['point_id'] for p in sample['points']] != [p['point_id'] for p in frozen_points]:
            raise ValueError(f'{sid}: frozen point ID mismatch')
        for point, frozen in zip(sample['points'], frozen_points):
            if point['point_text'] != frozen['point_text'] and point['point_id'] != 'Q13-P02':
                raise ValueError(f'{sid}: unexpected point wording change')
        original_evidence = {e['evidence_id']: e for e in original['evidence']}
        context = []
        for item in sample['evidence']:
            e = original_evidence[item['evidence_id']]
            if (item['document_id'], item['paper_title'], item['chunk_id'], item['text']) != (
                    e['doc_id'], e['title'], e['chunk_id'], e['text']):
                raise ValueError(f'{sid}: evidence mapping mismatch')
            context.append({'evidence_id': item['evidence_id'], 'citation_id': f"[{item['evidence_id']}]",
                            'document_id': item['document_id'], 'paper_title': item['paper_title'],
                            'chunk_id': item['chunk_id'], 'original_evidence_text': item['text'],
                            'source_role': item['source_role_hint']})
        ids = set(original_evidence)
        ref_rows = []
        for ref in references:
            if ref['question_id'] != qid or ref['evidence_id'] not in ids:
                continue
            e = original_evidence[ref['evidence_id']]
            if (ref['document_id'], ref['chunk_id'], ref['evidence_text']) != (e['doc_id'], e['chunk_id'], e['text']):
                raise ValueError(f'{sid}: reference map differs from original evidence')
            ref_rows.append({'point_id': ref['point_id'], 'evidence_id': ref['evidence_id'],
                             'citation_id': f"[{ref['evidence_id']}]", 'document_id': ref['document_id'],
                             'paper_title': ref['paper_title'], 'chunk_id': ref['chunk_id'],
                             'original_evidence_text': ref['evidence_text'], 'source_role': ref['source_role'],
                             'support_status': ref['support_status'], 'basis': ref['basis']})
        claims = sample['claims']
        if not claims or any(not set(c['citation_ids']) <= ids for c in claims):
            raise ValueError(f'{sid}: claim/citation mismatch')
        occurrences = [{'citation_id': m.group(0), 'evidence_id': m.group(1),
                        'char_start': m.start(), 'char_end_exclusive': m.end()}
                       for m in re.finditer(r'\[(E\d+)\]', sample['answer'])]
        if any(o['evidence_id'] not in ids for o in occurrences):
            raise ValueError(f'{sid}: invalid citation in answer text')
        records.append({'question_id': qid, 'anonymous_sample_id': sid,
                        'original_question': sample['question'],
                        'expected_answer_points': sample['points'], 'original_answer': sample['answer'],
                        'actual_evidence_context': context, 'evidence_reference_map': ref_rows,
                        'claims_to_evaluate': claims, 'original_citation_positions': occurrences})
    return records


def validate_export(expected):
    combined = load(EXPORT / 'remaining_54_answers.json')['samples']
    if combined != expected or len(combined) != 54:
        raise ValueError('combined export mismatch')
    ids = {x['anonymous_sample_id'] for x in combined}
    if len(ids) != 54 or any(f'{qid}-{letter}' not in ids for pair in BATCHES for qid in pair for letter in 'ABC'):
        raise ValueError('anonymous sample IDs incomplete')
    for number, pair in enumerate(BATCHES, 1):
        batch = load(EXPORT / f'batch_{number:02d}.json')
        members = batch['samples']
        if len(members) != 6 or Counter(m['question_id'] for m in members) != dict.fromkeys(pair, 3):
            raise ValueError(f'batch {number:02d}: wrong question balance')
        if members != [m for m in combined if m['question_id'] in pair]:
            raise ValueError(f'batch {number:02d}: content mismatch')
    manifest = load(EXPORT / MANIFEST)
    if manifest['sample_count'] != 54 or manifest['batch_count'] != 9 or manifest['file_count'] != len(FILES):
        raise ValueError('manifest count mismatch')
    for name, digest in manifest['exported_file_sha256'].items():
        if sha(EXPORT / name) != digest:
            raise ValueError(f'{name}: SHA-256 mismatch')
    for name in FILES:
        check_clean((EXPORT / name).read_text(encoding='utf-8'))
    with zipfile.ZipFile(ARCHIVE) as archive:
        if sorted(archive.namelist()) != sorted(FILES) or archive.testzip() is not None:
            raise ValueError('ZIP member list or CRC failed')
        for name in FILES:
            if archive.read(name) != (EXPORT / name).read_bytes():
                raise ValueError(f'{name}: ZIP extract mismatch')
            check_clean(archive.read(name).decode('utf-8'))
    return {'samples': 54, 'questions': 18, 'batches': 9, 'files': len(FILES),
            'hashes_passed': True, 'zip_passed': True, 'anonymity_scan_passed': True,
            'zip_sha256': sha(ARCHIVE)}


def main():
    if EXPORT.exists() or ARCHIVE.exists() or PENDING.exists():
        raise FileExistsError('remaining export or pending record already exists; refusing to overwrite')
    records = build()
    EXPORT.mkdir(parents=True)
    for number, pair in enumerate(BATCHES, 1):
        write_json(EXPORT / f'batch_{number:02d}.json', {'format': 'phase422-blind-batch-v1',
            'batch_id': f'{number:02d}', 'question_ids': list(pair),
            'samples': [r for r in records if r['question_id'] in pair]})
    write_json(EXPORT / 'remaining_54_answers.json', {'format': 'phase422-blind-remaining-v1', 'samples': records})
    (EXPORT / 'evaluation_instructions.md').write_bytes(
        (RESULTS / 'pilot_export/pilot_evaluation_instructions.md').read_bytes())
    (EXPORT / 'evaluation_schema.json').write_bytes(
        (RESULTS / 'pilot_export/pilot_json_schema.json').read_bytes())
    readme = ('# 匿名回答评价包\n\n'
              '九个 `batch_XX.json` 文件各含两道问题、每题 A/B/C 三条匿名回答。'
              '`remaining_54_answers.json` 是同一批 54 条数据的汇总。每条记录保留'
              '原始问题、盲化要点、回答、实际 Evidence、来源角色、要点候选映射、Claim 及'
              '回答中引用的零起始字符位置（结束位置不含该字符）。\n\n'
              '全部样本使用相同 `evaluation_instructions.md` 和 `evaluation_schema.json`。'
              '结果应按 `anonymous_sample_id` 回传；模板文件只是字段预留，空白处不是评分。'
              '在现有 Schema 中，证据不足使用 `evidence_support=Insufficient Evidence` 表示；'
              '要点覆盖字段保持原有枚举，不新增值。\n\n'
              '本包没有匿名映射或历史评价。Evidence 条数仍可能使配置被推测，因此只称组名盲化。'
              'Q13-P02 沿用离线准备时已有的条件性盲化措辞，本次导出未再次改写。'
              '证据映射中的已有支持状态与依据仅是核查线索，评价者应独立查阅原文。\n')
    (EXPORT / 'README.md').write_text(readme, encoding='utf-8')
    write_json(EXPORT / 'answer_evaluation_template.json', {'format': 'phase422-answer-results-template-v1',
        'status': 'template_only', 'records': [{'sample_id': r['anonymous_sample_id'], 'evaluation': None}
                                               for r in records]})
    write_csv(EXPORT / 'claim_evaluation_template.csv',
              ('sample_id','claim_id','support','reason','confidence','evidence_quotes'),
              [{'sample_id': r['anonymous_sample_id'], 'claim_id': c['claim_id'],
                'support': '', 'reason': '', 'confidence': '', 'evidence_quotes': ''}
               for r in records for c in r['claims_to_evaluate']])
    write_csv(EXPORT / 'point_evaluation_template.csv',
              ('sample_id','point_id','coverage','evidence_support','answer_text','omission',
               'factual_error','reason','confidence','evidence_quotes'),
              [{'sample_id': r['anonymous_sample_id'], 'point_id': p['point_id'],
                'coverage': '', 'evidence_support': '', 'answer_text': '', 'omission': '',
                'factual_error': '', 'reason': '', 'confidence': '', 'evidence_quotes': ''}
               for r in records for p in r['expected_answer_points']])
    source_paths = {
        'blind_dataset': RESULTS / 'blind_evaluation_dataset.json',
        'question_benchmark': ROOT / 'experiments/results/phase421/question_benchmark.json',
        'evidence_reference_map': ROOT / 'experiments/results/phase421/evidence_reference_map.json',
        'evaluation_instructions': RESULTS / 'pilot_export/pilot_evaluation_instructions.md',
        'evaluation_schema': RESULTS / 'pilot_export/pilot_json_schema.json',
    }
    write_json(EXPORT / MANIFEST, {'format': 'phase422-remaining-export-v1',
        'exported_at_beijing': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'sample_count': 54, 'question_count': 18, 'batch_count': 9, 'file_count': len(FILES),
        'source_file_sha256': {key: sha(path) for key, path in source_paths.items()},
        'exported_file_sha256': {name: sha(EXPORT / name) for name in FILES if name != MANIFEST},
        'anonymous_answer_evidence_content_sha256': {
            r['anonymous_sample_id']: canonical_hash({'answer': r['original_answer'],
                'evidence': r['actual_evidence_context']}) for r in records},
        'original_answer_and_evidence_verified': True,
        'note': 'No experiment-group mapping or historical evaluator result is exported.'})
    write_json(PENDING, {'status': 'external_pilot_reported; structured_results_pending_import',
        'anonymous_sample_ids': [f'{qid}-{label}' for qid in ('Q07','Q11') for label in 'ABC'],
        'scores_imported': 0,
        'note': 'User reports six external pilot evaluations. No Phase 4.22 machine-readable pilot scores were found in the project; none were fabricated.'})
    with zipfile.ZipFile(ARCHIVE, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(EXPORT / name, arcname=name)
    print(json.dumps(validate_export(records), ensure_ascii=False))


if __name__ == '__main__':
    main()
