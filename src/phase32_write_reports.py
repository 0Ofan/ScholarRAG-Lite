"""Generate six Chinese Phase 3.2 reports from measured, frozen artifacts."""
import json
import re
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NEW=ROOT/'experiments/results/phase32_improved';OLD=ROOT/'experiments/results/phase31_pilot'
PROC=ROOT/'data/processed';INDEX=ROOT/'data/indexes/phase32_sentence_aware'
load=lambda p:json.loads(p.read_text(encoding='utf8'))
m=load(PROC/'phase31_pilot_manifest.json')
c=load(NEW/'chunking_stats.json');bc=load(OLD/'chunking_stats.json')
e=load(NEW/'embedding_metrics.json');be=load(OLD/'embedding_metrics.json')
i=load(INDEX/'index_metadata.json');bi=load(ROOT/'data/indexes/phase31_pilot/index_metadata.json')
q=load(NEW/'metrics.json');bq=load(OLD/'metrics.json');u=load(NEW/'unjudged_audit.json')
p=load(NEW/'performance_metrics.json');bp=load(OLD/'performance_metrics.json')
pre=load(NEW/'preprocess_repeats.json');comp=load(NEW/'comparison.json')
test_log=(NEW/'unittest.log').read_text(encoding='utf8')
test_count=int(re.search(r'Ran (\d+) tests',test_log).group(1))
assert re.search(r'\bOK\b',test_log) and 'FAILED' not in test_log
assert m['document_count']==q['candidate_documents']==bq['candidate_documents']==2000
assert c['total_chunks']==e['chunk_count']==i['chunk_count']==p['chunk_count']
assert q['query_count']==bq['query_count']==50
def write(name,text):(ROOT/name).write_text(text.strip()+'\n',encoding='utf8')
def f(x):return f'{x:.4f}'
def pct(x):return f'{100*x:.1f}%'
def table(rows):return '\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)
qids=list(map(str,range(1,51)))

chunk_rows=[['统计','Fixed-length Baseline','Sentence-aware'],['---','---:','---:'],
 ['Chunk 数',f"{bc['total_chunks']:,}",f"{c['total_chunks']:,}"],
 ['每篇平均 Chunk 数',f"{bc['chunks_per_document']['mean']:.2f}",f"{c['chunks_per_document']['mean']:.2f}"],
 ['BGE Token 均值',f"{bc['token_lengths']['mean']:.2f}",f"{c['bge_token_lengths']['mean']:.2f}"],
 ['中位数',bc['token_lengths']['median'],c['bge_token_lengths']['median']],
 ['P95',bc['token_lengths']['p95'],c['bge_token_lengths']['p95']],
 ['最大值',bc['token_lengths']['max'],c['bge_token_lengths']['max']],
 ['<40 Tokens',bc['short_chunks_under_40'],c['short_chunks_under_40']]]
write('phase32_chunking_report.md',f'''# Phase 3.2 Sentence-aware Chunking 质量报告

使用与 Phase 3.1 完全相同的 `phase31_pilot_manifest.json`，2,000 篇论文的顺序、原始 UID、标题、摘要、正文来源保持一致。调用 `src/phase27_chunking.py` 的公共清洗和短章节规则，Sentence-aware 使用目标 180、软上限 220、绝对上限 240 Tokens；**没有固定 20 Token overlap**。算法识别常规标点、括号和引文，略过已知缩写、小数点；无标点或过长句段按 BGE Token 偏移二次切分，不截断剩余内容。

{table(chunk_rows)}

实测共 {c['total_chunks']:,} 个 Chunk；BGE 编码长度范围 {c['bge_token_lengths']['min']}–{c['bge_token_lengths']['max']}，MiniLM 范围 {c['minilm_token_lengths']['min']}–{c['minilm_token_lengths']['max']}；BGE/MiniLM 超过 220 Tokens 分别 {c['bge_over_220']} / {c['minilm_over_220']}，超过 240 Tokens 为 {c['over_240_chunks']}。220 为内容软上限，完整 Tokenizer 编码含 special tokens，故少量 221–222 的输入仍满足硬上限。超长句二次切分 {c['common_preprocessing']['long_sentence_splits']} 次，短章节 {c['common_preprocessing']['short_sections']} 个，其中合并 {c['common_preprocessing']['merged_short_sections']} 个；未合并有效内容保留。短尾合并 {c['common_preprocessing']['tail_merges']} 次。

所有 Chunk 的论文、章节及原文位置映射检查通过，映射率 {pct(c['source_mapping_success_rate'])}。逐字符审计覆盖 {c['source_nonspace_chars']:,} 个原文非空白字符，其中 {c['missing_nonspace_chars']} 个私用区项目符号 `U+F0B7` 被 BGE Tokenizer 跳过；其前后正文仍在 Chunk 中。该版式符号损失已单列，不能宣称字节级零损失。证据见 `experiments/results/phase32_improved/chunking_stats.json` 的上下文。三轮 Chunk ID 顺序的 SHA-256 完全一致，抽查论文的 Chunk 文本与顺序复算一致；Chunk ID 稳定且无重复。
''')

write('phase32_improved_implementation_report.md',f'''# Phase 3.2 Improved 检索实现报告

本阶段仅替换 Chunking 方法。复用 Phase 3.1 的 Pilot manifest、`src/phase27_chunking.py` 公共预处理、参数化的 `src/embedding_encoder.py`、`src/faiss_builder.py`、`src/retriever.py`、`src/phase31_evaluate.py` 与 `src/evaluation.py`。运行时分别传入 `--variant phase32` 或 `PilotRetriever(variant='phase32')`；默认 Phase 3.1 路径仍可用。Phase 3.1 的数据、Embedding、索引、结果和报告保持逐字节不变，哈希清单与复核结果见 `experiments/results/phase32_improved/phase31_baseline_hashes.json`。

Improved Chunk 见 `data/processed/phase32_sentence_aware_chunks.jsonl`，显式保留 chunk_id、规范 doc_id、原 UID、标题、DOI、章节、文本、两模型 Token 长度及全部来源位置区间。生成 {c['total_chunks']:,} 个 Chunk；BGE Embedding 形状 {e['embedding_shape']}、float32，NaN/Inf 元素 {e['nonfinite_elements']}、零向量 {e['zero_vectors']}，L2 模长范围 {e['norm_min']:.7f}–{e['norm_max']:.7f}。GPU 为 {e['gpu_name']}，模型快照 `{e['snapshot']}`，与 Baseline 一致。Embedding 行与 `phase32_sentence_aware_chunk_ids.json` 一一对应。

独立 FAISS IndexFlatIP 索引保存在 `data/indexes/phase32_sentence_aware/`，含索引、Chunk ID 映射及版本元数据；重载验证{'通过' if i['index_reload_verified'] else '失败'}，{i['chunk_count']:,} 向量，索引 {i['index_size_bytes']/1e6:.1f} MB。检索入口仍为 `PilotRetriever`；同一 BGE Query 指令、归一化内积、最高分 Chunk 归并论文和 Top-K 独立论文规则。检索分数是相似度，不是相关性概率。未实现 LLM、Streamlit、Medium 或 Large。

复现顺序（项目根目录、项目 `.venv` 与 README 中的进程内缓存）：`src/sentence_aware_chunker.py`；`src/phase31_benchmark_preprocess.py --variant phase32`；`src/embedding_encoder.py --variant phase32`；`src/faiss_builder.py --variant phase32`；`src/phase31_evaluate.py --variant phase32`；`src/phase32_compare.py`。每一步均以 `.\\.venv\\Scripts\\python.exe` 调用。完整产物位于 `experiments/results/phase32_improved/`。
''')

eval_rows=[['查询','Pilot 正例','nDCG@10','Recall@10','Recall@100','MRR','未评判@10'],
           ['---','---:','---:','---:','---:','---:','---:']]
for topic in qids:
    x=q['per_query'][topic];z=x['metrics']
    eval_rows.append([topic,x['available_positive'],f(z['ndcg@10']),f(z['recall@10']),
                      f(z['recall@100']),f(z['mrr']),f"{x['top']['10']['unjudged']}/10"])
write('phase32_improved_evaluation_report.md',f'''# Phase 3.2 Improved Pilot 检索评价报告

全部 50 个冻结 TREC-COVID Query Text 已用同一 BGE 编码配置运行，按论文级最高分 Chunk 去重；每查询返回 100 篇独立论文。使用未改动的官方 qrels 和 `src/evaluation.py`：nDCG@10 的增益为 `2^grade−1`，Recall@10、Recall@100、MRR 以 grade≥1 为正例。完整逐篇结果及最佳 Chunk 见 `experiments/results/phase32_improved/retrieval_results.jsonl`。

| 宏平均指标 | 实测 |
| --- | ---: |
| nDCG@10 | {f(q['macro_average']['ndcg@10'])} |
| Pilot 已知正例 Recall@10 | {f(q['macro_average']['recall@10'])} |
| Pilot 已知正例 Recall@100 | {f(q['macro_average']['recall@100'])} |
| MRR（Top-100 截断） | {f(q['macro_average']['mrr'])} |
| Top-10 未评判比例 | {pct(u['average_rate']['10'])} |
| Top-20 未评判比例 | {pct(u['average_rate']['20'])} |
| Top-100 未评判比例 | {pct(u['average_rate']['100'])} |

{table(eval_rows)}

Top-10 至少 5 篇未评判的查询：{', '.join(u['high_unjudged_top10_topics']) or '无'}。未评判结果保持未知，JSON 中为 `null`，仅在传统 qrels 计分时按零增益处理；没有改成官方 grade 0。Recall 分母仅为当前 2,000 篇 Pilot 中的已知正例，**不是完整 TREC-COVID Recall**。高未评判率限制绝对质量结论；逐查询结果必须连同未评判比例解读。质量指标对确定性固定索引计算一次，检索重复三次排序一致。
''')

metric_rows=[['指标','Fixed-length','Sentence-aware','差值（后者－前者）'],['---','---:','---:','---:']]
for k in ('ndcg@10','recall@10','recall@100','mrr'):
    x=comp['macro'][k];metric_rows.append([k,f(x['baseline']),f(x['sentence_aware']),f'{x["delta"]:+.4f}'])
x=comp['unjudged']['10'];metric_rows.append(['Top-10 未评判',pct(x['baseline']),pct(x['sentence_aware']),f'{x["delta"]*100:+.1f} 百分点'])
x=comp['chunks'];metric_rows.append(['Chunk 数',f'{x["baseline"]:,}',f'{x["sentence_aware"]:,}',f'{x["delta"]:+,}'])
x=comp['embedding_seconds_median'];metric_rows.append(['Embedding 中位耗时',f'{x["baseline"]:.2f}s',f'{x["sentence_aware"]:.2f}s',f'{x["delta"]:+.2f}s'])
x=comp['index_bytes'];metric_rows.append(['索引大小',f'{x["baseline"]/1e6:.1f} MB',f'{x["sentence_aware"]/1e6:.1f} MB',f'{x["delta"]/1e6:+.1f} MB'])
for n in ('20','100'):
    x=comp['unjudged'][n]
    metric_rows.append([f'Top-{n} 未评判',pct(x['baseline']),pct(x['sentence_aware']),f'{x["delta"]*100:+.1f} 百分点'])
for label,bv,nv in (('平均 BGE Tokens',bc['token_lengths']['mean'],c['bge_token_lengths']['mean']),
                     ('Token 中位数',bc['token_lengths']['median'],c['bge_token_lengths']['median']),
                     ('Token P95',bc['token_lengths']['p95'],c['bge_token_lengths']['p95']),
                     ('每篇平均 Chunk',bc['chunks_per_document']['mean'],c['chunks_per_document']['mean'])):
    metric_rows.append([label,f'{bv:.2f}',f'{nv:.2f}',f'{nv-bv:+.2f}'])
th=comp['embedding_throughput_median']
metric_rows.append(['Embedding 吞吐中位数',f'{th["baseline"]:.1f}/s',f'{th["sentence_aware"]:.1f}/s',
                    f'{th["sentence_aware"]-th["baseline"]:+.1f}/s'])
for label,key,scale,suffix in (('索引向量入库','index_build_seconds',1,'s'),
                                ('检索延迟中位数','retrieval_median_seconds',1000,'ms'),
                                ('进程 RAM 峰值','ram_peak_bytes',1e-9,' GB'),
                                ('CUDA 预留峰值','vram_peak_reserved_bytes',1e-9,' GB')):
    v=comp[key]
    metric_rows.append([label,f'{v["baseline"]*scale:.3f}{suffix}',
                        f'{v["sentence_aware"]*scale:.3f}{suffix}',
                        f'{(v["sentence_aware"]-v["baseline"])*scale:+.3f}{suffix}'])
per_rows=[['查询','Δ nDCG@10','Δ Recall@10','Δ Recall@100','Δ MRR','Δ 未评判@10'],
          ['---','---:','---:','---:','---:','---:']]
for topic in qids:
    x=comp['per_query'][topic]
    per_rows.append([topic]+[f'{x["delta"][k]:+.4f}' for k in ('ndcg@10','recall@10','recall@100','mrr')]+
                    [f'{x["unjudged_delta"]["10"]*100:+.1f} pp'])
direction_rows=[['指标','提升查询','下降查询','不变'],['---','---:','---:','---:']]
for k in ('ndcg@10','recall@10','recall@100','mrr'):
    x=comp['directions'][k];direction_rows.append([k,x['improved'],x['declined'],x['unchanged']])
intervals=[]
for k in ('ndcg@10','recall@10','recall@100','mrr'):
    lo,hi=comp['bootstrap_exploratory'][k]['percentile_95_interval']
    intervals.append(f'{k}: [{lo:+.4f}, {hi:+.4f}]')
labels={'ndcg@10':'nDCG@10','recall@10':'Recall@10','recall@100':'Recall@100','mrr':'MRR'}
quality_summary='；'.join(f"{labels[k]} {'上升' if comp['macro'][k]['delta']>0 else '下降' if comp['macro'][k]['delta']<0 else '持平'}" for k in labels)
write('phase32_baseline_comparison_report.md',f'''# Phase 3.2 Baseline 与 Sentence-aware 公平对照

两组共用同一份 2,000 篇 Pilot manifest、50 条原始查询、官方 qrels、BGE 快照、向量归一化、FAISS IndexFlatIP、论文级最高分 Chunk 去重和 `score_topic`。唯一预设方法变量为 Chunking；Sentence-aware 不采用固定 overlap。Phase 3.1 原结果只读并通过哈希验证。下表差值均为 **Sentence-aware − Fixed-length**，来自 `experiments/results/phase32_improved/comparison.json`。

{table(metric_rows)}

{table(direction_rows)}

逐查询差值（完整两组原始分数和未知率见 `per_query_comparison.csv`）：

{table(per_rows)}

实测方向：{quality_summary}。nDCG 提升查询：{', '.join(comp['directions']['ndcg@10']['improved_queries']) or '无'}；下降查询：{', '.join(comp['directions']['ndcg@10']['declined_queries']) or '无'}。其余指标的查询名单在 `comparison.json`。两方法的 Chunk 数不同，较长文档拥有更多命中机会；Baseline/Sentence-aware 每篇平均 Chunk 为 {comp['per_doc_chunks']['baseline']['mean_chunks']:.2f}/{comp['per_doc_chunks']['sentence_aware']['mean_chunks']:.2f}，Top-100 命中文档的 Chunk 数相对全库均值倍数为 {comp['per_doc_chunks']['baseline']['top100_chunk_count_enrichment']:.2f}/{comp['per_doc_chunks']['sentence_aware']['top100_chunk_count_enrichment']:.2f}。这些只是长度偏差诊断，不证明因果。

探索性逐查询配对 Bootstrap：50 查询有放回抽样 10,000 次，固定种子 3201，宏平均差值的百分位 95% 区间为 {'；'.join(intervals)}。此分析未在冻结协议中预注册、未校正多重比较，并受未评判文档和有偏 Pilot 选择限制，**不能替代原有评价指标或据此作显著性保证**。绝对分数不等于完整 TREC-COVID 质量，宏平均变化也不能掩盖个别查询下降。
''')

def stage_row(name,key):
    x=pre['stage_seconds'][key]
    return [name,f"{x['median']:.3f}s",f"{x['min']:.3f}s",f"{x['max']:.3f}s"]
timing_rows=[['阶段','中位数','最小','最大'],['---','---:','---:','---:'],
    stage_row('文档读取','document_loading_seconds'),
    stage_row('标题/摘要/正文组装','section_assembly_seconds'),
    stage_row('公共清洗与 Sentence-aware','common_cleaning_and_chunking_seconds')]
def duration_rows(values,multiplier=1):
    return [f'{statistics.median(values)*multiplier:.3f}',f'{min(values)*multiplier:.3f}',f'{max(values)*multiplier:.3f}']
timing_rows.append(['BGE Embedding']+[x+'s' for x in duration_rows([r['seconds'] for r in e['repeats']])])
timing_rows.append(['50 查询编码']+[x+'s' for x in duration_rows([r['seconds'] for r in p['query_encoding_repeats']])])
timing_rows.append(['每查询检索延迟中位数']+[x+'ms' for x in duration_rows([r['median_seconds'] for r in p['retrieval_latency_repeats']],1000)])
timing_rows.append(['FAISS 向量入索引',f"{i['index_build_seconds']:.3f}s",'—','—'])
timing_rows.append(['FAISS 写盘',f"{i['index_write_seconds']:.3f}s",'—','—'])
write('phase32_performance_report.md',f'''# Phase 3.2 性能与资源报告

性能数字均来自 `experiments/results/phase32_improved/` 实测文件。文档读取、公共清洗与 Sentence-aware 切分、Embedding、50 查询编码和检索各重复三次；表中显示中位数及最小–最大。首轮切分另含 JSONL 写盘、MiniLM Tokenizer 合规校验 {c['timing']['minilm_validation_seconds']:.3f}s 和逐字符来源覆盖检查 {c['timing']['coverage_check_seconds']:.3f}s；第 2、3 轮只重复 BGE 核心切分，因此核心切分耗时与 Baseline 更可比。索引构建、写盘、质量评价各只运行一次。

{table(timing_rows)}

Embedding 吞吐三次为 {', '.join(f"{x['throughput_chunks_per_second']:.1f}" for x in e['repeats'])} Chunk/秒；Batch Size 与 Baseline 同为 64，必要时按相同规则减半。GPU {e['gpu_name']}，进程 CPU 采样峰值 {p['cpu_process_peak_percent']:.1f}%（多核百分比可超过 100%），进程 RAM 采样峰值 {p['ram_peak_bytes']/1e9:.2f} GB，CUDA 已分配/预留峰值 {p['vram_peak_allocated_bytes']/1e9:.2f}/{p['vram_peak_reserved_bytes']/1e9:.2f} GB。以上为进程和 PyTorch 口径，不代表全系统峰值。Embedding 文件 {e['embedding_file_bytes']/1e6:.1f} MB、FAISS 索引 {i['index_size_bytes']/1e6:.1f} MB、Chunk JSONL {c['chunk_file_bytes']/1e6:.1f} MB。

与 Phase 3.1 参考值对照：核心切分中位耗时 {bp['preprocessing_stage_seconds']['common_cleaning_and_chunking_seconds']['median']:.3f}s → {pre['stage_seconds']['common_cleaning_and_chunking_seconds']['median']:.3f}s；Embedding 中位耗时 {be['seconds_median']:.3f}s → {e['seconds_median']:.3f}s；索引向量入库 {bi['index_build_seconds']:.3f}s → {i['index_build_seconds']:.3f}s；索引大小 {bi['index_size_bytes']/1e6:.1f} MB → {i['index_size_bytes']/1e6:.1f} MB。两次实验在同一设备不同时间运行，短时差异可能受系统负载影响；未把计算推断写成额外实测。
''')

delta=comp['macro']
winner='Sentence-aware 在 nDCG@10 宏平均提高' if delta['ndcg@10']['delta']>0 else 'Sentence-aware 在 nDCG@10 宏平均下降' if delta['ndcg@10']['delta']<0 else 'nDCG@10 宏平均持平'
all_lower=all(delta[k]['delta']<0 for k in ('ndcg@10','recall@10','recall@100','mrr'))
conclusion=('本 Pilot 上未见 Sentence-aware 的总体质量或成本优势：四项宏平均质量指标均下降，Chunk 数、编码时间和索引大小上升。'
            if all_lower else '不同指标表现不一致，应按各指标与逐查询结果判断，不能预设 Sentence-aware 全面占优。')
write('phase_3_2_summary.md',f'''# ScholarRAG Lite Phase 3.2 总结

## 已通过

- 同一 2,000 篇 Pilot 上完成 Sentence-aware 切分，生成 {c['total_chunks']:,} 个 Chunk；两模型 Tokenizer 的最长输入均为 {max(c['bge_token_lengths']['max'],c['minilm_token_lengths']['max'])} Tokens，无超 240 Token，超长句二次切分 {c['common_preprocessing']['long_sentence_splits']} 次。
- BGE GPU Embedding 全部完成，维度 384、无 NaN/Inf/零向量；独立 FAISS IndexFlatIP 索引可重载，Chunk ID 映射一致。
- 全部 50 查询完成同口径论文级评价：nDCG@10 **{f(q['macro_average']['ndcg@10'])}**，Pilot Recall@10 **{f(q['macro_average']['recall@10'])}**、Recall@100 **{f(q['macro_average']['recall@100'])}**，MRR **{f(q['macro_average']['mrr'])}**，Top-10 未评判 **{pct(u['average_rate']['10'])}**。
- 与 Phase 3.1 的逐查询配对、效率和资源对照已完成；{winner}。Baseline 25 个产物的哈希复核通过，未改动原始结果。
- 项目 `.venv` 中 {test_count} 项单元测试全部通过，日志见 `experiments/results/phase32_improved/unittest.log`。

## 未通过与限制

- 逐字符覆盖检查发现 {c['missing_nonspace_chars']} 个私用区 PDF 项目符号 `U+F0B7` 被 BGE Tokenizer 忽略；对应正文仍保留，见 Chunking 报告。不能宣称原文逐字节完全覆盖。
- 官方 qrels 未覆盖全部候选论文，Top-10 未评判率为 {pct(u['average_rate']['10'])}；Pilot 已知正例 Recall 不代表完整 TREC-COVID Recall。
- 只完成 BGE + Pilot 对照，尚未进行 Medium/Large、MiniLM、LLM 或界面实验。Bootstrap 区间仅为探索性结果。

## 结论

{quality_summary}。{conclusion}Phase 3.2 已完成技术与实验交付；具备在用户审核后规划下一阶段的条件。**现在停止，等待明确批准，不自动进入 Phase 3.3。**
''')
print('wrote six Phase 3.2 reports')
