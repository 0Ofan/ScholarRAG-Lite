"""Export six already-blinded pilot inputs; never invoke an evaluator."""
import hashlib
import json
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'experiments/results/phase422'
EXPORT = RESULTS / 'pilot_export'
ARCHIVE = RESULTS / 'phase422_pilot_export.zip'
FILES = ('pilot_Q07_Q11.json', 'pilot_evaluation_instructions.md',
         'pilot_json_schema.json', 'README.md', 'export_manifest.json')
FORBIDDEN = ('Top-3', 'Top-5', 'Top-8', 'anonymization_mapping',
             'phase419_status', 'phase420_status', 'evidence_k', '_K3', '_K5', '_K8')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def check_clean(data):
    for token in FORBIDDEN:
        if token in data:
            raise ValueError(f'forbidden export token: {token}')


def build():
    pilot = load(RESULTS / 'pilot_evaluation_inputs.json')['samples']
    blind = {s['sample_id']: s for s in load(RESULTS / 'blind_evaluation_dataset.json')['samples']}
    mapping = {m['sample_id']: m for m in load(RESULTS / 'anonymization_mapping.json')['mapping']}
    refs = load(RESULTS / '../phase421/evidence_reference_map.json')
    benchmark = {q['question_id']: q for q in load(RESULTS / '../phase421/question_benchmark.json')}
    expected = {f'{qid}-{label}' for qid in ('Q07', 'Q11') for label in 'ABC'}
    if len(pilot) != 6 or {s['sample_id'] for s in pilot} != expected:
        raise ValueError('pilot must contain exactly six Q07/Q11 A/B/C samples')
    exported = []
    for sample in pilot:
        sid = sample['sample_id']
        qid = sid[:3]
        if sample != blind[sid]:
            raise ValueError(f'{sid}: prepared sample differs from blind source')
        source = mapping[sid]
        raw_path = ROOT / f"experiments/results/phase418/raw_answers/{qid}_K{source['evidence_k']}.json"
        raw_bytes = raw_path.read_bytes()
        raw = json.loads(raw_bytes)
        if hashlib.sha256(raw_bytes).hexdigest() != source['source_sha256']:
            raise ValueError(f'{sid}: original answer hash mismatch')
        offline = load(ROOT / 'experiments/results/phase418/offline_records.json')
        original_evidence = offline[qid]['groups'][str(source['evidence_k'])]['evidence']
        if raw['answer'] != sample['answer'] or raw['evidence'] != original_evidence:
            raise ValueError(f'{sid}: original answer or evidence changed')
        if raw['question'] != sample['question'] or source['question_id'] != qid:
            raise ValueError(f'{sid}: question mismatch')
        if [p['point_id'] for p in sample['points']] != [p['point_id'] for p in benchmark[qid]['points']]:
            raise ValueError(f'{sid}: frozen point IDs changed')
        if qid == 'Q07' and [p['point_text'] for p in sample['points']] != [p['point_text'] for p in benchmark[qid]['points']]:
            raise ValueError(f'{sid}: Q07 point text changed')
        if qid == 'Q11' and any(p['point_text'] != b['point_text'] for p, b in zip(sample['points'], benchmark[qid]['points'])
                               if p['point_id'] != 'Q11-P03'):
            raise ValueError(f'{sid}: Q11 point text changed outside prior blind adjustment')
        eids = {e['evidence_id'] for e in sample['evidence']}
        by_original = {e['evidence_id']: e for e in raw['evidence']}
        context = []
        for item in sample['evidence']:
            original = by_original[item['evidence_id']]
            if (item['document_id'], item['chunk_id'], item['paper_title'], item['text']) != (
                    original['doc_id'], original['chunk_id'], original['title'], original['text']):
                raise ValueError(f'{sid}: evidence field mismatch')
            context.append({'evidence_id': item['evidence_id'], 'citation_id': f"[{item['evidence_id']}]",
                            'document_id': item['document_id'], 'paper_title': item['paper_title'],
                            'chunk_id': item['chunk_id'], 'original_evidence_text': item['text'],
                            'source_role': item['source_role_hint']})
        map_rows = []
        for row in refs:
            if row['question_id'] != qid or row['evidence_id'] not in eids:
                continue
            original = by_original[row['evidence_id']]
            if (row['document_id'], row['chunk_id'], row['evidence_text']) != (
                    original['doc_id'], original['chunk_id'], original['text']):
                raise ValueError(f'{sid}: reference map mismatch')
            map_rows.append({'point_id': row['point_id'], 'evidence_id': row['evidence_id'],
                             'citation_id': f"[{row['evidence_id']}]", 'document_id': row['document_id'],
                             'paper_title': row['paper_title'], 'chunk_id': row['chunk_id'],
                             'original_evidence_text': row['evidence_text'],
                             'source_role': row['source_role'], 'support_status': row['support_status'],
                             'basis': row['basis']})
        for claim in sample['claims']:
            if not set(claim['citation_ids']) <= eids:
                raise ValueError(f'{sid}: claim has invalid citation')
        exported.append({'question_id': qid, 'anonymous_sample_id': sid,
                         'original_question': sample['question'],
                         'expected_answer_points': sample['points'],
                         'original_answer': sample['answer'],
                         'actual_evidence_context': context,
                         'evidence_reference_map': map_rows,
                         'claims_to_evaluate': sample['claims']})
    return exported


def verify_export(expected):
    payload = load(EXPORT / FILES[0])
    samples = payload['samples']
    if len(samples) != 6 or Counter(x['question_id'] for x in samples) != {'Q07': 3, 'Q11': 3}:
        raise ValueError('exported sample count or question balance wrong')
    if samples != expected:
        raise ValueError('exported data differs from validated in-memory records')
    for sample in samples:
        ids = {e['evidence_id'] for e in sample['actual_evidence_context']}
        if not ids or any(e['citation_id'] != f"[{e['evidence_id']}]" or not e['chunk_id']
                          for e in sample['actual_evidence_context']):
            raise ValueError('evidence citation/chunk incomplete')
        if any(r['evidence_id'] not in ids for r in sample['evidence_reference_map']):
            raise ValueError('reference map uses absent evidence')
    manifest = load(EXPORT / FILES[-1])
    if manifest['sample_count'] != 6 or manifest['file_count'] != 5:
        raise ValueError('manifest count wrong')
    for name, digest in manifest['exported_file_sha256'].items():
        if sha(EXPORT / name) != digest:
            raise ValueError(f'{name}: SHA-256 mismatch')
    for name in FILES:
        check_clean((EXPORT / name).read_text(encoding='utf-8'))
    with zipfile.ZipFile(ARCHIVE) as archive:
        if sorted(archive.namelist()) != sorted(FILES) or archive.testzip() is not None:
            raise ValueError('ZIP file list or CRC failed')
        for name in FILES:
            if archive.read(name) != (EXPORT / name).read_bytes():
                raise ValueError(f'{name}: ZIP extraction mismatch')
            check_clean(archive.read(name).decode('utf-8'))
    return {'samples': 6, 'questions': {'Q07': 3, 'Q11': 3}, 'files': 5,
            'export_file_hashes_passed': True, 'zip_extract_and_crc_passed': True,
            'group_name_leak_scan_passed': True, 'zip_sha256': sha(ARCHIVE)}


def main():
    if EXPORT.exists() or ARCHIVE.exists():
        raise FileExistsError('pilot export already exists; refusing to overwrite')
    samples = build()
    EXPORT.mkdir(parents=True)
    write = lambda path, data: path.write_text(data, encoding='utf-8')
    write(EXPORT / FILES[0], json.dumps({'format': 'phase422-pilot-v1', 'samples': samples},
                                       ensure_ascii=False, indent=2) + '\n')
    original_prompt = (ROOT / 'experiments/configs/phase422/judge_prompt.md').read_text(encoding='utf-8')
    rubric = load(ROOT / 'experiments/configs/phase421/benchmark_config.json')['rubric']
    instructions = original_prompt + '\n## 统一评分标准\n\n'
    for key in ('completeness', 'faithfulness', 'relevance', 'citation_support'):
        instructions += f'- {key.title()}: {rubric[key]}\n'
    instructions += ('- Evidence Coverage: 分开判断 Direct Evidence、Partial Evidence、Indirect Evidence、'
                     'Insufficient Evidence、Unverified；数量不能替代支持程度。\n'
                     '- Abstention Quality: ' + ', '.join(rubric['abstention_quality']) +
                     '。应区分有据部分的回答与缺证部分的拒答。\n')
    write(EXPORT / FILES[1], instructions)
    (EXPORT / FILES[2]).write_bytes((ROOT / 'experiments/configs/phase422/judge_schema.json').read_bytes())
    readme = ('# 匿名试点评价包\n\n'
              '本目录有六条 Q07/Q11 匿名回答，A/B/C 仅是随机样本标识。请逐条使用相同的'
              '`pilot_evaluation_instructions.md` 和 `pilot_json_schema.json` 评价。'
              '证据条数仍可能使配置可被推测，因此这里只称组名盲化。\n\n'
              '- `pilot_Q07_Q11.json`：问题、要点、原始回答、实际 Evidence、来源角色、'
              '要点证据映射与待核查 Claim。映射中的支持状态是预先有限原文核对资料，'
              '评价者仍须独立核对，不得把候选或既有状态当成答案。\n'
              '- `pilot_evaluation_instructions.md`：统一评价指令和评分标准。\n'
              '- `pilot_json_schema.json`：预期评价结果结构。\n'
              '- `export_manifest.json`：文件 SHA-256、来源和导出时间。\n\n'
              '本包不含匿名映射表，也没有模型评价结果。要点文字沿用已准备的盲化样本；'
              '其中一条要点在离线准备阶段曾为隐藏实验组字样而作中性调整，本次导出没有再次改写。\n')
    write(EXPORT / FILES[3], readme)
    source_paths = {
        'blind_dataset': RESULTS / 'blind_evaluation_dataset.json',
        'pilot_inputs': RESULTS / 'pilot_evaluation_inputs.json',
        'question_benchmark': ROOT / 'experiments/results/phase421/question_benchmark.json',
        'evidence_reference_map': ROOT / 'experiments/results/phase421/evidence_reference_map.json',
        'judge_prompt': ROOT / 'experiments/configs/phase422/judge_prompt.md',
        'judge_schema': ROOT / 'experiments/configs/phase422/judge_schema.json',
    }
    manifest = {
        'format': 'phase422-pilot-export-v1',
        'exported_at_beijing': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'sample_count': 6, 'file_count': 5,
        'anonymous_sample_ids': [s['anonymous_sample_id'] for s in samples],
        'source_file_sha256': {key: sha(path) for key, path in source_paths.items()},
        'exported_file_sha256': {name: sha(EXPORT / name) for name in FILES[:-1]},
        'source_answer_evidence_verified': True,
        'original_answer_and_evidence_content_sha256': {
            s['anonymous_sample_id']: canonical_hash({'answer': s['original_answer'],
                 'evidence': s['actual_evidence_context']}) for s in samples},
        'note': 'Content hashes identify anonymous records only; no experiment-group mapping is exported.',
    }
    write(EXPORT / FILES[-1], json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    with zipfile.ZipFile(ARCHIVE, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(EXPORT / name, arcname=name)
    print(json.dumps(verify_export(samples), ensure_ascii=False))


if __name__ == '__main__':
    main()
