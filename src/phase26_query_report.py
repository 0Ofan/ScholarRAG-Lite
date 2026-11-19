"""Generate the complete 50-topic coverage table from audited JSON."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'data/evaluation'
stats=json.loads((EVAL/'phase26_coverage.json').read_text(encoding='utf8'))
per=stats['per_query']
high=sum(x['positive_valid_coverage']>=.7 for x in per.values())
mid=sum(.5<=x['positive_valid_coverage']<.7 for x in per.values())
low=sum(x['positive_valid_coverage']<.5 for x in per.values())
lines=[
'# Phase 2.6 全部查询全文覆盖分析（2026-11-09）',
'',
'数据来源：[NIST TREC-COVID Complete](https://ir.nist.gov/covidSubmit/data.html) 的原始 qrels 与 [AllenAI CORD-19 2020-07-16](https://github.com/allenai/cord19) 官方全文归档。分母是每个查询的官方 grade ≥ 1 正相关论文数；分子是同版归档中正文至少 1,000 字符的唯一 `cord_uid` 数。它是**全文可用覆盖率**，不是模型 Recall。两个 `-1` 判断不计入有效标签。',
'',
f'共 50 个查询：高覆盖（≥70%）{high} 个，中覆盖（50%–<70%）{mid} 个，低覆盖（<50%）{low} 个，完全无全文 0 个。所有 50 个查询均至少有 30 篇可用正相关全文。逐查询原始 ID 列表和已评判候选数见 `data/evaluation/phase26_coverage.json`。',
'',
'| Topic | 查询文本 | 官方正例 | 有路径 | 有效正文 | 覆盖率 | 当前语料已评判 / 总文档 |',
'|---:|---|---:|---:|---:|---:|---:|'
]
total=stats['judged_valid_body_unique']
for q,x in sorted(per.items(),key=lambda kv:int(kv[0])):
    query=x['query_text'].replace('|','\\|')
    lines.append(f"| {q} | {query} | {x['official_positive']} | {x['positive_with_path']} | {x['positive_valid_fulltext']} | {x['positive_valid_coverage']:.1%} | {x['judged_valid_docs']} / {total} |")
lines += ['',
'查询层面：至少 1、5、10、30 篇可用正相关全文的查询数均为 **50/50**。有效正文的正相关查询–论文对合计 **17,860**，官方正相关对共 26,664；这两个是查询–论文对数，不能与唯一论文数混用。',
'',
'已评判覆盖仍有限：对全体 25,598 篇有有效正文的已评判候选论文，单个查询的 qrels 覆盖率约 **1.78%–6.25%**，因为一篇论文被其他查询评判不代表被当前查询评判。未评判文档保留“未知”状态，不当作负例。',
'',
'一项额外的**探索性**检查在原 8 查询、58 篇旧语料基础上，按固定 ID 规则加入 142 篇有有效全文的官方正例和 grade 0 论文，形成 200 篇、13,838 chunk 的语料，仅用 BGE 重新检索。top-10 平均未评判数由旧语料 **5.375/10** 降至新语料 **3.25/10**；同查询但语料组成改变，不可解释为模型改进，也不能外推到 25,590 篇候选全集。逐查询结果和新增 ID 在 `experiments/results/phase26/top10_coverage_audit.json`。',
'',
'正式评价查询集尚未宣布：新候选 manifest 保留全部 50 个查询为探索性候选，0 个因无全文被排除；是否进入正式对照还需固定语料规模、核查文章许可、复核重复论文并在目标规模重新测 top-10 未评判比例。所有查询及状态见 `data/evaluation/phase26_candidate_manifest.json`。',
'']
(ROOT/'query_coverage_analysis.md').write_text('\n'.join(lines),encoding='utf8')
print({'high':high,'middle':mid,'low':low,'queries':len(per)})
