"""Render Phase 2.7 reports exclusively from frozen JSON and measured audit artifacts."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'data/evaluation';RESULT=ROOT/'experiments/results/phase27'
load=lambda path:json.loads(path.read_text(encoding='utf8'))
s=load(EVAL/'phase27_statistics.json');a=load(EVAL/'phase27_document_aliases.json')
c=load(RESULT/'chunk_validation.json');r=load(RESULT/'top10_audit.json')
config=load(ROOT/'experiments/configs/phase27_experiment_config.json')
assert r['documents']==s['phase27_final_canonical_docs']==25594
assert len(r['per_query'])==50 and all(len(v['top100_results'])==100 for v in r['per_query'].values())
def write(name,content):(ROOT/name).write_text(content.strip()+'\n',encoding='utf8')
def pct(x):return f'{x*100:.1f}%'
def f(x):return f'{x:.4f}'
def mdtable(rows):return '\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)

excluded=mdtable([['查询','原始 UID','等级','原因'],['---','---','---:','---']]+[
    [x['topic_id'],x['raw_uid'],x['grade'],x['reason']] for x in s['excluded_positive_pairs']])
write('phase27_data_reconciliation_report.md',f'''# Phase 2.7 数据统计核对

数据源：`data/evaluation/phase27_statistics.json`；复算脚本：`src/phase27_reconcile.py`；关系断言：`experiments/results/phase27/statistics_audit.json`。CORD-19 快照为 2020-07-16，qrels 为 TREC-COVID Complete 原始文件。

| 口径 | 文档/唯一正相关论文 | 正相关查询–论文对 |
| --- | ---: | ---: |
| 官方 qrels 全部 | — / {s['official_positive_unique_docs']:,} | {s['official_positive_pairs']:,} |
| 可读正文 ≥1,000 字符的原始 UID | {s['phase26_valid_raw_uids']:,} / {s['phase26_valid_positive_unique_docs']:,} | {s['phase26_valid_positive_pairs']:,} |
| Phase 2.6 候选 manifest：排除全部 8 个共享路径 UID | {s['phase26_candidate_docs']:,} / {s['phase26_candidate_positive_unique_docs']:,} | {s['phase26_candidate_positive_pairs']:,} |
| Phase 2.7 规范化候选集 | {s['phase27_final_canonical_docs']:,} / {s['phase27_positive_unique_docs']:,} | {s['phase27_positive_pairs']:,} |

25,598 来自 `phase26_doc_status.json` 的有效正文原始 UID；25,590 来自旧 `phase26_candidate_manifest.json`，差额恰为曾整体排除的 8 个 UID。11,984 与 11,980 相差的四个正例 UID 是 `dtnuet6c`、`j7t9nebs`、`g7qioapl`、`xa6kwguo`。前两者是同一论文的兼容别名，后两者是标签冲突的版本，详见身份报告。17,860 与 17,850 相差 10 个原始正相关判断对；Phase 2.7 回收其中兼容记录，规范化后为 17,852 对。官方共 {s['official_qrel_rows']:,} 行，其中 {len(s['official_negative_one_records'])} 条 grade=-1 保持未知且不计入正相关；有效等级判断对 {s['official_valid_grade_qrel_pairs']:,} 条。未改动官方 qrels。

规范化期间排除的七个正相关原始判断对：

{excluded}

另有 1 个重复正例判断对：查询 35 的 `dtnuet6c` → `j7t9nebs`，两者均 grade 2，只保留一个规范化查询–论文对。公式：17,860 − 7 − 1 = 17,852。所有正例均可映射到候选全文；每查询计数由 `effective_qrels` 复算并断言。历史报告中的数字各有明确口径，不能直接互换。
''')

identity_rows=[['共享 PDF 的 paper_id 前缀','UID','分类','处理'] ,['---','---','---','---']]
for group in a['groups']:
    members=group['members'];decisions=', '.join(f"{u}→{a['uids'][u]['canonical_doc_id'] or '排除'}" for u in members)
    identity_rows.append([group['shared_paper_id'][:12],', '.join(members),group['classification'],decisions])
details=[]
for uid in ('dtnuet6c','j7t9nebs','6rpt47gm','svu06v9c','g7qioapl','xa6kwguo','8avkjc84','d9v5xtx7'):
    x=a['uids'][uid]
    details.append(f"| `{uid}` | {', '.join(x['doi']) or '无'} | `{x['selected_path']}` | `{x['selected_sha256']}` | `{x['selected_paper_id']}` | {x['official_qrels']} | {x['decision']} |")
write('phase27_document_identity_report.md',f'''# Phase 2.7 论文身份与共享路径审计

证据及原始/规范化映射：`data/evaluation/phase27_document_aliases.json`；生成脚本：`src/phase27_aliases.py`。逐组交叉核对了 CORD-19 元数据、选用 JSON 原文的标题/paper_id/SHA-256、DOI、共享路径及官方 qrels。共享文件名本身不作为同一论文的充分证据。

{mdtable(identity_rows)}

| 原始 UID | DOI | 实际选用解析路径 | 原文 SHA-256 | paper_id | 官方查询:等级 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(details)}

`dtnuet6c` 与 `j7t9nebs` 的选用原文 SHA 和标题完全一致，查询 35 标签同为 2；以有 DOI 的后者为规范 ID。`6rpt47gm` 与 `svu06v9c` 同理，相关查询均为 grade 0。`g7qioapl` 与 `xa6kwguo` 标题及共享 PDF 相近，但实际选用原文不同，查询 1 分别为 grade 1 与 2；当前无法保证版本与标签对应关系可靠，故两者继续排除，原始冲突保留。`8avkjc84` 与 `d9v5xtx7` DOI、标题、选用原文均不同；共享的是错误的次要 PDF 映射，作为两篇论文保留。

评价的索引键、去重键和统计键统一使用规范化 Document ID；原始 UID 与选用路径、SHA 和 qrel 来源保存在映射及最终 manifest。遇到未解决的版本冲突不合并标签，也不改写官方 qrels。
''')

rows=[['查询','官方正例','候选正例','Top10 未评判','Top20 未评判','Top100 未评判','nDCG@10','Recall@10','Recall@100','MRR'],['---','---:','---:','---:','---:','---:','---:','---:','---:','---:']]
for q in map(str,range(1,51)):
    x=r['per_query'][q];t=x['top'];m=x['metrics']
    rows.append([q,x['official_positive'],x['available_positive'],
                 f"{t['10']['unjudged']}/10 ({pct(t['10']['unjudged_rate'])})",
                 f"{t['20']['unjudged']}/20 ({pct(t['20']['unjudged_rate'])})",
                 f"{t['100']['unjudged']}/100 ({pct(t['100']['unjudged_rate'])})",
                 f(m['ndcg@10']),f(m['recall@10']),f(m['recall@100']),f(m['mrr'])])
high=[q for q,x in r['per_query'].items() if x['top']['10']['unjudged']>=5]
dist10=', '.join(f'{k} 篇:{v} 查询' for k,v in r['unjudged']['10']['count_distribution'].items())
dist20=', '.join(f'{k} 篇:{v} 查询' for k,v in r['unjudged']['20']['count_distribution'].items())
write('phase27_unjudged_audit_report.md',f'''# Phase 2.7 完整候选语料未评判审计

运行：`src/phase27_full_audit.py`；逐查询 Top-100 明细、分数、规范 ID、原路径、官方等级见 `experiments/results/phase27/top10_audit.json`。BGE-small-en-v1.5、官方全部 50 个查询、规范化后 {r['documents']:,} 篇全文、Fixed-length 180/20 Chunk；查询使用 BGE 检索指令，向量归一化并用 FAISS IndexFlatIP 精确内积检索。运行设备 {r['device']}，构建 {r['chunks']:,} 个 Chunk，嵌入耗时 {r['embedding_seconds']:.1f} 秒，索引构建至写盘前 {r['index_build_seconds']:.1f} 秒，50 查询检索 {r['search_seconds']:.1f} 秒。论文按最高分 Chunk 去重，故每个 Top-K 均为 K 篇独立文档。

| 指标 | 50 查询宏平均 |
| --- | ---: |
| nDCG@10 | {f(r['average_metrics']['ndcg@10'])} |
| 当前语料已知正例 Recall@10 | {f(r['average_metrics']['recall@10'])} |
| 当前语料已知正例 Recall@100 | {f(r['average_metrics']['recall@100'])} |
| MRR | {f(r['average_metrics']['mrr'])} |
| Top-10 未评判比例 | {pct(r['unjudged']['10']['mean'])} |
| Top-20 未评判比例 | {pct(r['unjudged']['20']['mean'])} |
| Top-100 未评判比例 | {pct(r['unjudged']['100']['mean'])} |

{mdtable(rows)}

Top-10 未评判数量分布：{dist10}。Top-20 分布：{dist20}。Top-10 至少 5 篇未评判的查询：{', '.join(high) or '无'}。未评判在数据中保持 `null`，没有改成 grade 0，也没有自行补标签。标准 qrels 指标按惯例给未评判命中零增益，因而存在池化评判偏差；高未评判比例的查询应单独标记，不删除。所有 50 查询在候选语料中均有正例：{'是' if not r['queries_without_available_positive'] else '否'}。

nDCG 使用官方多级等级和 `2^grade−1` 增益；Recall/MRR 以 grade≥1 为正例。Recall 分母为当前语料中已知正例，不是官方完整集合。官方全文正例覆盖是 {s['phase27_positive_pairs']:,}/{s['official_positive_pairs']:,} 查询–论文对；两者不能混称为完整 TREC-COVID Recall。基于现有官方标签，不能给未知结果可靠的数值上下界；报告未评判比例作为敏感性证据。该语料可用于受限、同条件的检索方法对照，指标不能直接解释为全部 TREC-COVID 文献上的绝对质量。
''')

v=c['strategies'];vrows=[['方法','Chunk 数','BGE 均值/中位/P95/最大','MiniLM 最大','<40 Token','>240 Token','映射率','超长句二次切分'],['---','---:','---','---:','---:','---:','---:','---:']]
for name,label in (('fixed','Fixed-length'),('sentence','Sentence-aware')):
    x=v[name];b=x['bge'];vrows.append([label,f"{x['chunks']:,}",f"{b['mean']:.1f}/{b['median']}/{b['p95']}/{b['max']}",x['minilm']['max'],x['short_chunks_under_40'],x['over_240'],pct(x['source_mapping_rate']),x['long_sentence_splits']])
write('phase27_chunking_protocol.md',f'''# Phase 2.7 Chunking 与公共预处理协议

实现：`src/phase27_chunking.py`；在 Phase 2.6 相同 SHA-256 顺序选出的 200 篇正例与 100 篇其他全文上验证：`src/phase27_chunk_validate.py`、`experiments/results/phase27/chunk_validation.json`。两个方法读取同一 CORD-19 解析 JSON，按标题、摘要、正文顺序构建章节；保留原始 UID、选用 JSON 路径、章节编号、标题及原文字符区间。公共清洗仅合并空白并移除控制字符，不丢弃有效短章节。低于 40 BGE Token 的章节与邻近章节合并；样本中 {v['fixed']['short_sections']:,} 个短章节均得到合并。合并前有效章节数 {v['fixed']['input_sections']:,}，最终 Chunk 数如下。

{mdtable(vrows)}

固定窗口：BGE Tokenizer 不含 special tokens 的正文偏移按 180 Token 切分，相邻窗口重叠 20；末尾 <40 Token 时，如与前窗合并后 ≤238 内容 Token 则合并，否则保留。样本末尾合并 {v['fixed']['tail_merges']:,} 次。Sentence-aware：先识别句号/问号/感叹号和部分缩写，再按目标 180、软上限 220 Token 聚合；无标点及超长句按 Token 偏移二次切分，样本触发 {v['sentence']['long_sentence_splits']} 次；短尾在 ≤220 Token 时合并 {v['sentence']['tail_merges']} 次。最终对两种模型的完整编码长度分别检查，超过 240 则继续二分；样本回退切分次数为 {v['fixed']['fallback_splits']} / {v['sentence']['fallback_splits']}。

两种方案的最大输入均 ≤240；正文来源映射率均为 100%。Sentence-aware 仍有 {v['sentence']['short_chunks_under_40']} 个 <40 Token Chunk，保留其有效内容。固定参数可进入后续受控对照，完整语料的另一种切分尚未运行；本阶段没有正式三规模实验。异常公式与未识别缩写由 Token 上限兜底，不能保证所有句法边界完全正确。
''')

mean_chunks=v['fixed']['per_document_chunks']['mean'];mb_per_chunk=384*4/1_000_000
scale_rows=[['规模','论文数','估算 Chunk（按样本均值）','估算 IndexFlatIP 向量体积'],['---','---:','---:','---:']]
for name,label in (('pilot','Pilot'),('medium','Medium'),('large','Large')):
    count=config['scales'][name]['document_count'];scale_rows.append([label,f'{count:,}',f'{count*mean_chunks:,.0f}',f'{count*mean_chunks*mb_per_chunk:,.0f} MB'])
write('phase27_experiment_scale_plan.md',f'''# Phase 2.7 正式实验规模计划

冻结配置：`experiments/configs/phase27_experiment_config.json`；生成：`src/phase27_freeze.py`。三规模不再沿用 58/100/200，因为最终候选集已有 {s['phase27_final_canonical_docs']:,} 篇全文。计划采用：

{mdtable(scale_rows)}

选择规则：每查询按 SHA-256(`phase27-v1:`+规范 ID) 取前 25 篇已知正例，合并形成 {config['anchor_count']:,} 篇共同锚点；其余文档从在全部 50 查询都没有正标签的 {config['distractor_pool_count']:,} 篇中按同一哈希顺序取前缀。Pilot 2,000 ⊂ Medium 6,000 ⊂ Large 12,000。每查询在三规模中保留相同的 25–58 篇已知正例，全部 50 查询均有正例；但附加文档对个别查询仍可能是 grade 0 或未评判，须逐规模报告。规模间质量变化主要反映干扰文档增多，不能外推为完整语料上的召回趋势。

估算采用 300 篇样本 Fixed-length 每篇 {mean_chunks:.2f} 个 Chunk、384 维 float32 ×4 字节，未计 FAISS 元数据、文本、模型和框架内存；不是正式三规模实测。完整语料已在同设备运行 {r['chunks']:,} Chunk，索引文件 {r['index_size_bytes']/1_000_000:.1f} MB，BGE 嵌入 {r['embedding_seconds']:.1f} 秒（约 {r['chunks']/r['embedding_seconds']:.0f} Chunk/秒）。按样本 Chunk 数和这次 BGE 吞吐量粗算，BGE 的六个规模×切分条件单轮嵌入约 {2*sum(config['scales'][k]['document_count']*mean_chunks for k in ('pilot','medium','large'))/(r['chunks']/r['embedding_seconds'])/60:.0f} 分钟；三次重复仅嵌入约 {6*sum(config['scales'][k]['document_count']*mean_chunks for k in ('pilot','medium','large'))/(r['chunks']/r['embedding_seconds'])/60:.0f} 分钟。MiniLM GPU 速度、预处理、索引和失败重试时间尚未实测，不能据此承诺总工期。项目机器 RAM 约 32 GB、GPU RTX 5060 8 GB（历史环境审计）；同一设备上须逐规模测峰值 RAM/VRAM、耗时、索引体积，内存不足时记录失败并经后续审批调整。用户未给出新的明确截止日，不把估算写成实测。
''')

write('phase27_experimental_protocol.md',f'''# Phase 2.7 正式实验协议

版本 `phase27-experiment-v1`；机器可执行参数见 `experiments/configs/phase27_experiment_config.json`。Phase 3 获批后才实施。

研究设计：2 种 Chunking（Fixed-length / Sentence-aware）× 3 个嵌套规模（2,000 / 6,000 / 12,000）× 2 个 Embedding（BGE-small-en-v1.5 / all-MiniLM-L6-v2）。主问题是 Chunking 对检索质量、效率、资源及增长趋势的影响；Embedding 作为控制条件及补充交互分析。四个待检验假设 H1 句子切分可能改善语义质量、H2 固定切分可能更省处理时间、H3 两者随规模的资源增长可能不同、H4 两模型可能与切分方法交互，均不是既定结论。

固定条件：CORD-19 2020-07-16 快照、最终规范化 Document ID、同一组 50 条原始 query 字段、未修改的官方 qrels、公共预处理、模型对应的正确 Tokenizer、归一化向量、FAISS IndexFlatIP、论文级最高 Chunk 分数去重、Top-10/20/100、相同设备与批处理配置。BGE 查询使用官方检索指令；MiniLM 按其已验证编码配置。不得因结果调整查询或删困难查询。每个模型内比较两种 Chunking 时保持模型完全相同；跨模型对照明确模型差异。

质量：nDCG@10 采用等级增益 `2^grade−1`；Recall@10、Recall@100 和 MRR（Top-100 截断）采用 grade≥1；按查询宏平均并列出每查询值。Recall 分母为对应规模的已知正例；未评判结果单列数量/比例，保持未知。索引候选集以规范 ID 唯一计数，同论文多个 Chunk 不重复。零正例查询如出现则报告并从该指标宏平均中单独处理，不能静默记 0；当前冻结规模没有零正例查询。

效率：分别计时公共预处理、Chunking、Embedding、索引构建、查询编码与检索，输出 Embedding Throughput、Embedding Time、检索延迟分布及索引建立时间。资源：峰值 RAM、VRAM、FAISS 文件字节、Chunk 数；成本报告处理/推理耗时与索引空间随规模变化。正式性能测量同条件重复 3 次，报告中位数与最小–最大范围；确定性质量分数对同一固定索引只需计算 1 次并说明。记录硬件、软件版本、随机种子、开始/结束时间、失败和重试。全部数据必须来自真实运行。

解释限制：三规模只固定选择的 25 个每查询正例锚点，不能将其 Recall 称为完整 TREC-COVID Recall。全候选语料审计揭示的未评判率应随正式结果展示。若后续要求真实全语料质量结论，需要额外评判或独立评价设计，不能改写未知标签。
''')

ready=(r['unjudged']['10']['mean']<0.5)
write('phase_2_7_summary.md',f'''# ScholarRAG Lite Phase 2.7 总结

## 已通过

- 统一统计复算并解释旧报告的 25,598/25,590、11,984/11,980、17,860/17,850 差异；最终冻结 {s['phase27_final_canonical_docs']:,} 篇规范化文档、{s['phase27_positive_unique_docs']:,} 篇唯一正相关论文、{s['phase27_positive_pairs']:,} 个正相关查询–论文对。
- 四组共享路径逐组按 DOI、标题、实际原文、SHA、paper_id 与 qrels 判定：两组兼容别名合并，一组冲突版本排除，一组不同论文分别保留。
- 对全部 50 查询和完整候选语料完成 BGE 论文级 Top-10/20/100 审计；每条结果可追溯。
- 两种 Chunking 在既有 300 篇样本上经两个 Tokenizer 验证，均无超 240 Token Chunk，来源映射率 100%。
- 冻结 2,000/6,000/12,000 篇嵌套规模、变量、控制条件与指标；未开展 Phase 3 正式三规模对照。

## 风险与未通过项

- Top-10 平均未评判比例 {pct(r['unjudged']['10']['mean'])}，Top-20 为 {pct(r['unjudged']['20']['mean'])}，Top-100 为 {pct(r['unjudged']['100']['mean'])}；官方池化标签不足以独立确认未评判论文相关性。{'Top-10 未评判率超过 50%，绝对质量结论不可靠。' if not ready else '即使平均比例较低，单查询未评判偏差仍须在正式对照中报告。'}
- 冲突版本 `g7qioapl`/`xa6kwguo` 暂未解决，保持排除；不擅改官方标签。
- Sentence-aware 仅完成小样本参数验证，完整语料效率与三规模资源曲线尚待下一阶段实测。

## 结论

**可有条件进入 Phase 3 的同条件方法对照，但不宜声称完整 TREC-COVID 的绝对检索质量。** 必须保留全部 50 查询、逐查询未知标签比例和受限 Recall 口径，并记录实际效率与资源。Phase 2.7 交付后停止，等待用户审核和明确批准，不自动进入 Phase 3。

## 验证

项目 `.venv\\Scripts\\python.exe` 中运行 15 项 `unittest`，全部通过；`src/phase27_verify.py` 复核 50 查询 Top-100、17,852 个正相关对、官方等级与指标，并成功重载 1,242,117 向量的 FAISS 索引。运行依赖：PyTorch 2.11.0+cu128（CUDA 可用）、FAISS 1.15.1、Transformers 5.22.2、Sentence Transformers 6.2.0。启动时仍有既有的 `D:\\python\\python.exe` 路径警告及 Windows torch redirects 提示，均未阻断运行。
''')
print('wrote seven Phase 2.7 reports')
