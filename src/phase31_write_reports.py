"""Render four Chinese Phase 3.1 reports from measured artifacts."""
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PROC=ROOT/'data/processed';RESULT=ROOT/'experiments/results/phase31_pilot';INDEX=ROOT/'data/indexes/phase31_pilot'
load=lambda p:json.loads(p.read_text(encoding='utf8'))
m=load(PROC/'phase31_pilot_manifest.json');c=load(RESULT/'chunking_stats.json')
e=load(RESULT/'embedding_metrics.json');i=load(RESULT/'index_metrics.json')
q=load(RESULT/'metrics.json');u=load(RESULT/'unjudged_audit.json')
p=load(RESULT/'performance_metrics.json');pre=load(RESULT/'preprocess_repeats.json')
demo=load(RESULT/'cli_demo_results.json')
test_log=(RESULT/'unittest.log').read_text(encoding='utf8')
test_count=int(re.search(r'Ran (\d+) tests',test_log).group(1))
assert re.search(r'\bOK\b',test_log) and 'FAILED' not in test_log
assert m['document_count']==q['candidate_documents']==2000
assert c['total_chunks']==e['chunk_count']==i['chunk_count']==p['chunk_count']
assert q['query_count']==50 and len(demo['queries'])==3
def write(name,text):(ROOT/name).write_text(text.strip()+'\n',encoding='utf8')
def pct(x):return f'{100*x:.1f}%'
def f(x):return f'{x:.4f}'
def table(rows):return '\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)
cfg=load(ROOT/'experiments/configs/phase27_experiment_config.json')

write('phase31_baseline_implementation_report.md',f'''# Phase 3.1 Pilot Baseline 实现报告

数据与规则：CORD-19 2020-07-16、TREC-COVID Complete、`phase27-final-v1`、`phase27-experiment-v1`。`src/pilot_manifest.py` 直接读取冻结的 Pilot ID 列表；**{m['document_count']:,} 篇**规范化论文，{m['positive_anchor_count']} 篇共同正例锚点，{m['positive_query_doc_pairs']:,} 个正相关查询–论文对；缺失/正文异常 {m['invalid_or_missing_fulltext_count']} 篇。`data/processed/phase31_pilot_manifest.json` 保留原始 UID、别名、标题、DOI、解析路径和快照版本；官方 qrels 未改动。

模块链路：`pilot_manifest.py` 冻结文档 → `baseline_chunker.py` 读取 JSON、组装标题/摘要/正文并调用 Phase 2.7 的 `phase27_chunking.make_chunks` → `embedding_encoder.py` 调用本地 BGE → `faiss_builder.py` 建立/重载精确索引 → `retriever.py` 按最高分 Chunk 去重 → `phase31_evaluate.py` 调用既有 `evaluation.score_topic` → `cli.py` 提供交互入口。没有接入 LLM 或界面。

Fixed-length 参数：目标 180、重叠 20、最大 240 BGE Tokens；短章节按冻结规则与相邻章节合并，未合并部分保留。共 {c['total_chunks']:,} 个 Chunk，每篇平均 {c['chunks_per_document']['mean']:.2f} 个；Token 长度均值 {c['token_lengths']['mean']:.2f}、中位数 {c['token_lengths']['median']}、P95 {c['token_lengths']['p95']}、范围 {c['token_lengths']['min']}–{c['token_lengths']['max']}；<40 Tokens {c['short_chunks_under_40']}，>240 Tokens {c['over_240_chunks']}，来源映射率 {pct(c['source_mapping_success_rate'])}。Chunk ID 由规范 ID、方法、块号与稳定顺序组成；每条保留标题、DOI、原 UID、章节和原文位置区间。

CUDA 模型为本地 `BAAI/bge-small-en-v1.5` 快照 `{e['snapshot']}`，设备 {e['gpu_name']}。Embedding 形状 {e['embedding_shape']}、float32、L2 归一化，NaN/Inf 元素 {e['nonfinite_elements']}、零向量 {e['zero_vectors']}，模长范围 {e['norm_min']:.7f}–{e['norm_max']:.7f}。向量行序由 `phase31_pilot_chunk_ids.json` 显式绑定。FAISS 为归一化向量的 IndexFlatIP；保存索引、Chunk ID 映射及版本元数据于 `data/indexes/phase31_pilot/`，重载校验{'通过' if i['index_reload_verified'] else '失败'}。

CLI：设置 README 中的项目内缓存变量后，在项目根目录执行 `& .\\.venv\\Scripts\\python.exe -m src.cli --examples` 查看 3 条真实查询；执行 `& .\\.venv\\Scripts\\python.exe -m src.cli` 交互检索，或 `& .\\.venv\\Scripts\\python.exe -m src.cli --query 'coronavirus origin' --top-k 10` 单次检索。输出是论文及最佳 Chunk、归一化向量内积分数，不生成答案。三条示例均完成实际检索，结果见 `experiments/results/phase31_pilot/cli_demo_results.json`。
''')

rows=[['查询','Pilot 正例','nDCG@10','Recall@10','Recall@100','MRR','Top10 未评判'],
      ['---','---:','---:','---:','---:','---:','---:']]
for topic in map(str,range(1,51)):
    x=q['per_query'][topic];mtr=x['metrics']
    rows.append([topic,x['available_positive'],f(mtr['ndcg@10']),f(mtr['recall@10']),
                 f(mtr['recall@100']),f(mtr['mrr']),f"{x['top']['10']['unjudged']}/10"])
high=', '.join(u['high_unjudged_top10_topics']) or '无'
write('phase31_pilot_evaluation_report.md',f'''# Phase 3.1 Pilot 检索评价报告

`src/phase31_evaluate.py` 用冻结的 50 个 TREC-COVID query 字段检索每查询 Top-100 篇独立论文。原始逐篇结果及最高分 Chunk 见 `experiments/results/phase31_pilot/retrieval_results.jsonl`；逐查询表见 `per_query_metrics.csv`。全部 50 查询成功，每查询 Pilot 已知正例 {min(m['positive_per_query'].values())}–{max(m['positive_per_query'].values())} 篇。候选集按规范 Document ID 去重；官方 grade 0 和未评判（JSON `null`）严格分开。

| 宏平均指标 | 实测 |
| --- | ---: |
| nDCG@10 | {f(q['macro_average']['ndcg@10'])} |
| 当前 Pilot 已知正例 Recall@10 | {f(q['macro_average']['recall@10'])} |
| 当前 Pilot 已知正例 Recall@100 | {f(q['macro_average']['recall@100'])} |
| MRR（Top-100 截断） | {f(q['macro_average']['mrr'])} |
| Top-10 未评判比例 | {pct(u['average_rate']['10'])} |
| Top-20 未评判比例 | {pct(u['average_rate']['20'])} |
| Top-100 未评判比例 | {pct(u['average_rate']['100'])} |

{table(rows)}

Top-10 至少 5 篇未评判的查询：{high}。逐查询未评判数量及 Top-20/100 比例见 `unjudged_audit.json`。nDCG 使用官方多级等级和 `2^grade−1` 增益；Recall/MRR 以 grade≥1 为正例。未评判在 qrels 中保持未知，传统 qrels 分数只按惯例给予零增益，没有私自标注为 grade 0。Recall 分母仅为本 Pilot 中的已知正例；**不是完整 TREC-COVID Recall**。Pilot 固定选择了 956 篇正例锚点及哈希排序的无正标签文档，因此不能与 Phase 2.7 的完整 25,594 篇语料直接比较绝对分数。确定性质量指标仅对固定索引计算一次；相同向量的检索排序重复三次一致。
''')

er=e['repeats'];stage=pre['stage_seconds']
timing_rows=[['阶段','中位数','最小','最大','说明'],['---','---:','---:','---:','---']]
for key,label in (('document_loading_seconds','文档读取'),('section_assembly_seconds','标题/摘要/正文组装'),
                  ('common_cleaning_and_chunking_seconds','公共清洗与 Chunking')):
    z=stage[key];timing_rows.append([label,f"{z['median']:.3f}s",f"{z['min']:.3f}s",f"{z['max']:.3f}s",'3 次'])
timing_rows.append(['BGE Embedding',f"{e['seconds_median']:.3f}s",f"{e['seconds_min']:.3f}s",f"{e['seconds_max']:.3f}s",'3 次'])
qs=[x['seconds'] for x in p['query_encoding_repeats']]
timing_rows.append(['50 查询编码',f"{sorted(qs)[1]:.3f}s",f"{min(qs):.3f}s",f"{max(qs):.3f}s",'3 次'])
rs=[x['median_seconds'] for x in p['retrieval_latency_repeats']]
timing_rows.append(['每查询检索延迟中位数',f"{sorted(rs)[1]*1000:.2f}ms",f"{min(rs)*1000:.2f}ms",f"{max(rs)*1000:.2f}ms",'3 次 × 50 查询'])
timing_rows.append(['FAISS 向量入索引',f"{i['index_build_seconds']:.3f}s",'—','—','1 次'])
timing_rows.append(['FAISS 索引写盘',f"{i['index_write_seconds']:.3f}s",'—','—','1 次'])
write('phase31_performance_report.md',f'''# Phase 3.1 Pilot 性能与资源报告

全部数字来自 `experiments/results/phase31_pilot/` 内实测 JSON。文档加载、组装、公共清洗与切分重复 3 次，Chunk ID 的 SHA-256 三次完全一致；首轮含 JSONL 写盘，后两轮仅计算，因此阶段时间比端到端时间更可比。Embedding 重复 3 次，查询编码和 50 查询检索也各重复 3 次；索引构建与质量评估各运行 1 次。

{table(timing_rows)}

资源：Embedding 期间进程 CPU 峰值 {p['cpu_process_peak_percent']:.1f}%（多核心进程百分比可超过 100%）；进程 RAM 采样峰值 {p['ram_peak_bytes']/1e9:.2f} GB；CUDA 分配峰值 {p['vram_peak_allocated_bytes']/1e9:.2f} GB、预留峰值 {p['vram_peak_reserved_bytes']/1e9:.2f} GB，GPU 总显存 {e['gpu_total_memory_bytes']/1e9:.2f} GB。这些是本进程采样/框架统计，不是系统全部进程的峰值。FAISS 索引 {p['index_size_bytes']/1e6:.1f} MB，Embedding 文件 {p['embedding_file_bytes']/1e6:.1f} MB，Chunk JSONL {c['chunk_file_bytes']/1e6:.1f} MB，Chunk 数 {c['total_chunks']:,}。Embedding 三轮吞吐分别为 {', '.join(f"{x['throughput_chunks_per_second']:.1f}" for x in er)} Chunk/秒。未进行 Medium/Large、MiniLM 或 Improved RAG 测试。

运行命令（项目根目录 PowerShell，先设置 README 中的项目缓存环境）：`& .\\.venv\\Scripts\\python.exe src\\pilot_manifest.py`、`src\\baseline_chunker.py`、`src\\embedding_encoder.py`、`src\\faiss_builder.py`、`src\\phase31_evaluate.py`、`src\\phase31_benchmark_preprocess.py`。各命令均使用项目虚拟环境。环境既有的 `D:\\python\\python.exe` 路径警告不影响本次运行。
''')

write('phase_3_1_summary.md',f'''# ScholarRAG Lite Phase 3.1 总结

## 已通过

- 按 Phase 2.7 冻结规则生成恰好 {m['document_count']:,} 篇 Pilot，{m['positive_anchor_count']} 篇正例锚点，缺失/异常全文 {m['invalid_or_missing_fulltext_count']} 篇；未改数据集或官方 qrels。
- Fixed-length 生成 {c['total_chunks']:,} 个可追溯 Chunk；长度最大 {c['token_lengths']['max']} Tokens，无超 240 Token，来源映射率 {pct(c['source_mapping_success_rate'])}。
- {e['gpu_name']} 上完成 BGE GPU Embedding，形状 {e['embedding_shape']}，三次计时中位数 {e['seconds_median']:.1f} 秒；无 NaN、Inf 或零向量。
- FAISS IndexFlatIP 索引与 Chunk ID 映射成功保存、重载；命令行检索与 3 条真实查询演示成功，论文级 Top-K 去重。
- 全部 50 查询完成评价：nDCG@10 **{f(q['macro_average']['ndcg@10'])}**，Pilot 已知正例 Recall@10 **{f(q['macro_average']['recall@10'])}**、Recall@100 **{f(q['macro_average']['recall@100'])}**，MRR **{f(q['macro_average']['mrr'])}**；Top-10 平均未评判 {pct(u['average_rate']['10'])}。
- 三次预处理、Embedding、查询编码及检索计时已记录；项目 `.venv` 的 {test_count} 项单元测试全部通过，日志见 `experiments/results/phase31_pilot/unittest.log`；`phase31_finalize.py` 的跨文件一致性验证通过。

## 局限与后续条件

未评判命中仍会影响绝对质量判断，特别是查询 {high}。Pilot 的 Recall 只针对所选语料已知正例，不能称为完整 TREC-COVID Recall。当前只验证 BGE + Fixed-length + Pilot，尚无与 Sentence-aware 的正式方法对照。性能统计为当前机器单次阶段实测，不保证其他机器重现相同耗时。

**Phase 3.1 Baseline 已完整运行；具备申请 Phase 3.2 方法对照的技术条件。** 本阶段到此停止，等待用户审核和明确批准，不自动进入 Phase 3.2。
''')
print('wrote four Phase 3.1 reports')
