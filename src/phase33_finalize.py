"""Write the single Chinese Phase 3.3 report from measured artifacts."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/phase33'


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def gib(value):
    return value / 1024**3


def pct(value):
    return f'{100 * value:.2f}%'


runs = []
for scale, size, method, folder in (
    ('Pilot', 2000, 'Fixed-length', 'phase31_pilot'),
    ('Pilot', 2000, 'Sentence-aware', 'phase32_improved'),
    ('Medium', 6000, 'Fixed-length', 'phase33/phase33_medium_fixed'),
    ('Medium', 6000, 'Sentence-aware', 'phase33/phase33_medium_sentence'),
    ('Large', 12000, 'Fixed-length', 'phase33/phase33_large_fixed'),
    ('Large', 12000, 'Sentence-aware', 'phase33/phase33_large_sentence'),
):
    base = ROOT / 'experiments/results' / folder
    chunk = load(base / 'chunking_stats.json')
    embedding = load(base / 'embedding_metrics.json')
    index = load(base / 'index_metrics.json')
    metrics = load(base / 'metrics.json')
    perf = load(base / 'performance_metrics.json')
    unjudged = load(base / 'unjudged_audit.json')
    if scale == 'Pilot':
        chunk_seconds = perf['preprocessing_stage_seconds']['common_cleaning_and_chunking_seconds']['median']
    else:
        repeats = load(base / 'chunking_repeats.json')
        assert len(repeats['runs']) == 3
        chunk_seconds = repeats['stage_seconds']['common_cleaning_and_chunking_seconds']['median']
    assert len(metrics['per_query']) == 50 and metrics['query_count'] == 50
    row = {
        'scale': scale, 'documents': size, 'method': method, 'artifact_dir': str(base.relative_to(ROOT)),
        'chunks': chunk['total_chunks'],
        'mean_tokens': chunk.get('token_lengths', chunk.get('bge_token_lengths'))['mean'],
        'chunking_seconds_median': chunk_seconds, 'embedding_seconds_median': embedding['seconds_median'],
        'embedding_throughput': chunk['total_chunks'] / embedding['seconds_median'],
        'index_bytes': index['index_size_bytes'],
        'retrieval_latency_seconds_median': perf['retrieval_median_seconds_across_repeats'],
        'ram_peak_bytes': perf['ram_peak_bytes'], 'vram_reserved_peak_bytes': perf['vram_peak_reserved_bytes'],
        'vram_allocated_peak_bytes': perf['vram_peak_allocated_bytes'],
        'ndcg@10': metrics['macro_average']['ndcg@10'],
        'recall@10': metrics['macro_average']['recall@10'],
        'recall@100': metrics['macro_average']['recall@100'],
        'mrr': metrics['macro_average']['mrr'],
        'top10_unjudged_rate': unjudged['average_rate']['10'],
        'per_query': metrics['per_query'],
    }
    runs.append(row)

# Machine-readable full comparison retains the per-topic scores and timings.
(OUT / 'comparison.json').write_text(json.dumps({'version': 'phase33-comparison-v1', 'runs': runs},
                                              ensure_ascii=False, indent=2), encoding='utf8')
with (OUT / 'per_query_comparison.csv').open('w', encoding='utf8', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['topic_id', 'scale', 'documents', 'method', 'ndcg@10', 'recall@10',
                     'recall@100', 'mrr', 'top10_unjudged_rate', 'available_positive'])
    for run in runs:
        for topic in map(str, range(1, 51)):
            q = run['per_query'][topic]
            writer.writerow([topic, run['scale'], run['documents'], run['method'],
                             *(q['metrics'][k] for k in ('ndcg@10', 'recall@10', 'recall@100', 'mrr')),
                             q['top']['10']['unjudged_rate'], q['available_positive']])


def chart(path, title, key, unit, multiplier=1):
    width, height = 760, 440
    left, right, top, bottom = 90, 25, 75, 70
    ymax = max(r[key] * multiplier for r in runs) * 1.12
    x = {2000: 175, 6000: 405, 12000: 635}
    color = {'Fixed-length': '#2764a5', 'Sentence-aware': '#d47a2d'}
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="21">{title}</text>',
             f'<text x="20" y="52" font-family="Arial" font-size="12">{unit}</text>']
    def y(value):
        return height - bottom - value / ymax * (height - top - bottom)
    for i in range(5):
        value = ymax * i / 4
        yy = y(value)
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#ddd"/>')
        lines.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.2f}</text>')
    for size, label in ((2000, 'Pilot 2k'), (6000, 'Medium 6k'), (12000, 'Large 12k')):
        lines.append(f'<text x="{x[size]}" y="{height-32}" text-anchor="middle" font-family="Arial" font-size="13">{label}</text>')
    for method in ('Fixed-length', 'Sentence-aware'):
        seq = sorted((r for r in runs if r['method'] == method), key=lambda r: r['documents'])
        pts = ' '.join(f'{x[r["documents"]]},{y(r[key]*multiplier):.1f}' for r in seq)
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{color[method]}" stroke-width="3"/>')
        for r in seq:
            xx, yy, value = x[r['documents']], y(r[key] * multiplier), r[key] * multiplier
            lines.append(f'<circle cx="{xx}" cy="{yy:.1f}" r="5" fill="{color[method]}"/>')
            lines.append(f'<text x="{xx}" y="{yy-12:.1f}" text-anchor="middle" font-family="Arial" font-size="12" fill="{color[method]}">{value:.2f}</text>')
    lines += [f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#555"/>',
              f'<rect x="215" y="{height-20}" width="15" height="4" fill="{color["Fixed-length"]}"/>',
              f'<text x="237" y="{height-14}" font-family="Arial" font-size="12">Fixed-length</text>',
              f'<rect x="425" y="{height-20}" width="15" height="4" fill="{color["Sentence-aware"]}"/>',
              f'<text x="447" y="{height-14}" font-family="Arial" font-size="12">Sentence-aware</text>', '</svg>']
    path.write_text('\n'.join(lines), encoding='utf8')


chart(OUT / 'ndcg_by_scale.svg', 'nDCG@10 by corpus size', 'ndcg@10', 'nDCG@10')
chart(OUT / 'embedding_time_by_scale.svg', 'Embedding time by corpus size',
      'embedding_seconds_median', 'minutes (median of 3 runs)', 1 / 60)

quality_rows = []
resource_rows = []
for r in runs:
    quality_rows.append(f"| {r['scale']} | {r['method']} | {r['ndcg@10']:.4f} | {r['recall@10']:.4f} | "
                        f"{r['recall@100']:.4f} | {r['mrr']:.4f} | {pct(r['top10_unjudged_rate'])} |")
    resource_rows.append(f"| {r['scale']} | {r['method']} | {r['chunks']:,} | {r['mean_tokens']:.1f} | "
                         f"{r['chunking_seconds_median']:.1f} | {r['embedding_seconds_median']:.1f} | "
                         f"{r['embedding_throughput']:.1f} | {gib(r['index_bytes']):.3f} | "
                         f"{r['retrieval_latency_seconds_median']*1000:.2f} | "
                         f"{gib(r['ram_peak_bytes']):.2f} | {gib(r['vram_reserved_peak_bytes']):.2f} |")

by_key = {(r['scale'], r['method']): r for r in runs}
differences = []
for scale in ('Pilot', 'Medium', 'Large'):
    fixed, sentence = by_key[scale, 'Fixed-length'], by_key[scale, 'Sentence-aware']
    differences.append((scale, sentence['ndcg@10'] - fixed['ndcg@10'],
                        sentence['recall@100'] - fixed['recall@100'],
                        sentence['embedding_seconds_median'] / fixed['embedding_seconds_median'] - 1))
difference_lines = [f'- {scale}：Sentence-aware 相对 Fixed-length 的 nDCG@10 差值 {d:+.4f}，'
                    f'Recall@100 差值 {rec:+.4f}，Embedding 中位耗时变化 {100*cost:+.1f}%。'
                    for scale, d, rec, cost in differences]
topic_lines = []
for scale in ('Pilot', 'Medium', 'Large'):
    fixed, sentence = by_key[scale, 'Fixed-length'], by_key[scale, 'Sentence-aware']
    deltas = [sentence['per_query'][str(i)]['metrics']['ndcg@10'] -
              fixed['per_query'][str(i)]['metrics']['ndcg@10'] for i in range(1, 51)]
    topic_lines.append(f'- {scale}：50 个查询中 Sentence-aware 的 nDCG@10 较高 {sum(x > 1e-12 for x in deltas)} 个，'
                       f'较低 {sum(x < -1e-12 for x in deltas)} 个，持平 {sum(abs(x) <= 1e-12 for x in deltas)} 个。')
growth_lines = []
for method in ('Fixed-length', 'Sentence-aware'):
    pilot, large = by_key['Pilot', method], by_key['Large', method]
    growth_lines.append(f'- {method}：Pilot 到 Large 的论文数增长 6 倍；chunk 数增长 '
                        f'{large["chunks"]/pilot["chunks"]:.2f} 倍、Embedding 中位耗时增长 '
                        f'{large["embedding_seconds_median"]/pilot["embedding_seconds_median"]:.2f} 倍、'
                        f'FAISS 文件增长 {large["index_bytes"]/pilot["index_bytes"]:.2f} 倍，'
                        f'Top-100 检索延迟增长 {large["retrieval_latency_seconds_median"]/pilot["retrieval_latency_seconds_median"]:.2f} 倍。')
large_gain = differences[-1][1]
advantage = ('在 Large 上出现 nDCG@10 优势' if large_gain > 0 else
             '在 Large 上未出现 nDCG@10 优势' if large_gain < 0 else '在 Large 上 nDCG@10 持平')
large_sentence_chunk_runs = load(OUT / 'phase33_large_sentence/chunking_repeats.json')['runs']
large_sentence_chunk_times = [r['timing']['common_cleaning_and_chunking_seconds']
                              for r in large_sentence_chunk_runs]
chunk_time_note = '、'.join(f'{x:.1f}' for x in large_sentence_chunk_times)

report = f'''# ScholarRAG Lite — Phase 3.3 总结

## 完成情况与协议

四组新增实验（Medium/Large × Fixed-length/Sentence-aware）均已完成切分、BGE-small 编码、FAISS IndexFlatIP 建索引及 50 个 TREC-COVID 查询的论文级评价。复用 Phase 2.7 冻结的 956 篇正例锚点、1,560 个正例查询－论文对、官方 qrels 和规范化 Document ID；Medium 6,000 篇与 Large 12,000 篇嵌套包含 Pilot 2,000 篇。Fixed-length 为 180/20，Sentence-aware 为目标 180/软上限 220/硬上限 240；均未改动。各规模共享同一组 50 个查询。

## 检索质量

| 规模 | 方法 | nDCG@10 | Recall@10 | Recall@100 | MRR | Top-10 未评判 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(quality_rows)}

![nDCG@10 随语料规模变化](experiments/results/phase33/ndcg_by_scale.svg)

Recall 的分母是**当前语料内的已知正例**，不能解释为完整 TREC-COVID Recall。质量指标对确定性结果只计算一次；逐查询结果保存在各组 `per_query_metrics.csv` 和统一的 `experiments/results/phase33/per_query_comparison.csv`。

## 计算与资源

| 规模 | 方法 | Chunk 数 | 平均 Token | 切分秒数 | Embedding 秒数 | 吞吐量 chunk/s | FAISS GiB | 检索延迟 ms | RAM GiB | VRAM GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(resource_rows)}

![Embedding 耗时随语料规模变化](experiments/results/phase33/embedding_time_by_scale.svg)

切分、Embedding 和检索延迟均取三次计时的中位数；RAM 是 Embedding 进程峰值，VRAM 是该进程 CUDA reserved 峰值。Embedding 时间为编码及向量写入的有效耗时，不含模型加载和断点间空闲时间。FAISS 大小为写入磁盘的索引文件大小。检索延迟是论文级 Top-100 检索，不含查询编码；索引构建、查询编码的原始计时和全部重复值见各组机器可读结果。不同规模的系统状态会影响耗时，跨组时间差只能作为本机观察值。

Large + Sentence-aware 的三次有效切分耗时为 {chunk_time_note} 秒，存在明显波动；表中使用中位数，原始重复值仍保留。

## 发现与解释

{chr(10).join(difference_lines)}

{chr(10).join(topic_lines)}

{chr(10).join(growth_lines)}

Sentence-aware {advantage}。三种规模下，它的 nDCG@10 与 Recall@100 均低于 Fixed-length，同时 chunk 和索引更多、Embedding 中位耗时更长；按当前冻结指标，没有证据表明额外计算成本换来了检索质量收益。Large 的 Sentence-aware Top-10 未评判比例比 Fixed-length 低 1 个百分点，但不足以改变上述已评判指标的方向。固定正例锚点使新增论文主要成为干扰项，因此规模变化反映在既定正例集合上增加候选文档后的检索行为，不能外推到正例覆盖率同步增长的真实大规模语料。未评判文档在传统 qrels 得分中按零增益处理，但其真实相关性未知；Top-10 未评判比例越高，分数解释越受限制。

## 限制与下一阶段条件

没有修改冻结评价规则、模型或切分参数，也没有接入 LLM API 或 Streamlit。四组索引和 50 查询结果可供后续集成复用；从工程准备程度看，可进入 LLM + Streamlit 集成阶段，但本阶段的检索质量结论仍受固定正例锚点、未评判文档和单机性能波动限制。进入下一阶段须由用户另行批准。

可复现配置：`experiments/configs/phase33_runs.json`；嵌套清单：`data/processed/phase33_medium_manifest.json`、`phase33_large_manifest.json`；完整机器可读汇总：`experiments/results/phase33/comparison.json`。新增分组各自保存在 `data/processed`、`data/indexes` 与 `experiments/results/phase33` 的独立路径，未覆盖 Pilot 产物。
'''
(ROOT / 'phase_3_3_summary.md').write_text(report, encoding='utf8')
print('Phase 3.3 comparison and Chinese summary written.')
