"""Compare frozen A/B Pilot artifacts with the new C overlap run."""
import json
import statistics
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/phase34_ablation'
BASE=ROOT/'experiments/results'
MANIFEST=ROOT/'data/processed/phase31_pilot_manifest.json'
load=lambda p:json.loads(p.read_text(encoding='utf8'))
manifest=load(MANIFEST)
assert manifest['document_count']==2000 and manifest['positive_anchor_count']==956
groups={}
for key,label,folder in (
    ('A','Fixed-length 180/20','phase31_pilot'),
    ('B','Sentence-aware 180/220; no overlap','phase32_improved'),
    ('C','Sentence-aware 180/220 + overlap 20','phase34_ablation'),
):
    root=BASE/folder
    chunk=load(root/'chunking_stats.json')
    embedding=load(root/'embedding_metrics.json')
    index=load(root/'index_metrics.json')
    metrics=load(root/'metrics.json')
    performance=load(root/'performance_metrics.json')
    unjudged=load(root/'unjudged_audit.json')
    preprocess=load(root/'preprocess_repeats.json')
    assert metrics['query_count']==50 and len(metrics['per_query'])==50
    assert metrics['candidate_documents']==2000 and embedding['chunk_count']==chunk['total_chunks']
    token_dist=chunk.get('bge_token_lengths',chunk.get('token_lengths'))
    groups[key]={'label':label,'artifact_dir':root.relative_to(ROOT).as_posix(),
        'quality':{**metrics['macro_average'],'top10_unjudged_rate':unjudged['average_rate']['10']},
        'performance':{'chunks':chunk['total_chunks'],'mean_bge_tokens':token_dist['mean'],
            'chunking_seconds_median':preprocess['stage_seconds']['common_cleaning_and_chunking_seconds']['median'],
            'chunking_seconds_repeats':[x['common_cleaning_and_chunking_seconds'] for x in preprocess['runs']],
            'embedding_seconds_median':embedding['seconds_median'],
            'embedding_seconds_repeats':[x.get('seconds',x.get('active_seconds')) for x in embedding['repeats']],
            'index_size_bytes':index['index_size_bytes'],
            'query_latency_seconds_median':performance['retrieval_median_seconds_across_repeats'],
            'query_latency_seconds_repeats':[x['median_seconds'] for x in performance['retrieval_latency_repeats']],
            'ram_peak_bytes':performance['ram_peak_bytes'],
            'vram_reserved_peak_bytes':performance['vram_peak_reserved_bytes']},
        'per_query':metrics['per_query']}

keys=('ndcg@10','recall@10','recall@100','mrr')
per={}
for topic in map(str,range(1,51)):
    a,b,c=(groups[k]['per_query'][topic] for k in 'ABC')
    assert a['query']==b['query']==c['query']
    assert a['available_positive']==b['available_positive']==c['available_positive']==manifest['positive_per_query'][topic]
    per[topic]={'query':a['query'],'known_relevant_in_pilot':a['available_positive'],
        'A':{**{k:a['metrics'][k] for k in keys},'top10_unjudged_rate':a['top']['10']['unjudged_rate']},
        'B':{**{k:b['metrics'][k] for k in keys},'top10_unjudged_rate':b['top']['10']['unjudged_rate']},
        'C':{**{k:c['metrics'][k] for k in keys},'top10_unjudged_rate':c['top']['10']['unjudged_rate']}}
    for base in 'BA':
        per[topic][f'C_minus_{base}']={k:per[topic]['C'][k]-per[topic][base][k]
                                      for k in (*keys,'top10_unjudged_rate')}

directions={}
for base in 'BA':
    directions[f'C_vs_{base}']={}
    for key in keys:
        diffs=[per[str(i)][f'C_minus_{base}'][key] for i in range(1,51)]
        directions[f'C_vs_{base}'][key]={
            'improved':sum(x>1e-12 for x in diffs),
            'declined':sum(x< -1e-12 for x in diffs),
            'unchanged':sum(abs(x)<=1e-12 for x in diffs)}

rng=np.random.default_rng(3401)
diffs=np.asarray([per[str(i)]['C_minus_B']['ndcg@10'] for i in range(1,51)],dtype='float64')
draws=rng.integers(0,50,size=(10000,50))
means=diffs[draws].mean(axis=1)
bootstrap={'method':'paired query bootstrap; 10000 draws; seed 3401; percentile 95% interval; exploratory',
           'metric':'ndcg@10','comparison':'C_minus_B',
           'mean_delta':float(diffs.mean()),
           'percentile_95_interval':[float(x) for x in np.quantile(means,[.025,.975])]}

overlap=load(OUT/'chunking_stats.json')['overlap']
chunk_stats_c=load(OUT/'chunking_stats.json')
data={'version':'phase34-comparison-v1','pilot_manifest_version':manifest['version'],
      'document_count':2000,'query_count':50,'positive_anchor_count':956,
      'positive_query_document_pairs':sum(manifest['positive_per_query'].values()),
      'model':'BAAI/bge-small-en-v1.5','index':'FAISS IndexFlatIP',
      'recall_denominator':'known positive papers in current Pilot corpus; not full TREC-COVID recall',
      'groups':groups,'overlap':overlap,'directions':directions,
      'paired_bootstrap_exploratory':bootstrap,'per_query':per}
(OUT/'comparison.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')

def gib(x):return x/1024**3
def percent(x):return f'{x*100:.1f}%'
quality=[];cost=[]
for name in 'ABC':
    row=groups[name];q=row['quality'];p=row['performance']
    quality.append(f'| {name} | {row["label"]} | {q["ndcg@10"]:.4f} | {q["recall@10"]:.4f} | '
                   f'{q["recall@100"]:.4f} | {q["mrr"]:.4f} | {percent(q["top10_unjudged_rate"])} |')
    cost.append(f'| {name} | {p["chunks"]:,} | {p["mean_bge_tokens"]:.1f} | '
                f'{p["chunking_seconds_median"]:.1f} | {p["embedding_seconds_median"]:.1f} | '
                f'{p["index_size_bytes"]/1024**2:.1f} | {p["query_latency_seconds_median"]*1000:.2f} | '
                f'{gib(p["ram_peak_bytes"]):.2f} | {gib(p["vram_reserved_peak_bytes"]):.2f} |')

delta_b=groups['C']['quality']['ndcg@10']-groups['B']['quality']['ndcg@10']
delta_a=groups['C']['quality']['ndcg@10']-groups['A']['quality']['ndcg@10']
delta_recall10=groups['C']['quality']['recall@10']-groups['B']['quality']['recall@10']
delta_recall100=groups['C']['quality']['recall@100']-groups['B']['quality']['recall@100']
delta_mrr=groups['C']['quality']['mrr']-groups['B']['quality']['mrr']
delta_unjudged=groups['C']['quality']['top10_unjudged_rate']-groups['B']['quality']['top10_unjudged_rate']
ci=bootstrap['percentile_95_interval']
dir_b=directions['C_vs_B']['ndcg@10']
added_tokens=groups['C']['performance']['mean_bge_tokens']-groups['B']['performance']['mean_bge_tokens']
added_chunking=groups['C']['performance']['chunking_seconds_median']-groups['B']['performance']['chunking_seconds_median']
chunking_pct=100*added_chunking/groups['B']['performance']['chunking_seconds_median']
added_embed=groups['C']['performance']['embedding_seconds_median']-groups['B']['performance']['embedding_seconds_median']
embedding_pct=100*added_embed/groups['B']['performance']['embedding_seconds_median']
added_index=groups['C']['performance']['index_size_bytes']-groups['B']['performance']['index_size_bytes']
latency_delta_ms=(groups['C']['performance']['query_latency_seconds_median']-
                  groups['B']['performance']['query_latency_seconds_median'])*1000
overlap_coverage=overlap['chunks_with_overlap']/groups['C']['performance']['chunks']
conditional_overlap=(overlap['actual_token_lengths']['mean']/overlap_coverage if overlap_coverage else 0)
if delta_b<=0:
    scale_advice='暂不建议把 C 扩大到 Medium/Large：Pilot 的 nDCG@10 差值接近零且探索性区间跨零，Recall@100 与 MRR 的小幅增益尚不足以证明可抵消本次更高的切分与编码成本。'
elif ci[0]<=0:
    scale_advice='C 的 Pilot 平均 nDCG@10 有所提高，但探索性区间跨过零；暂不建议投入 Medium/Large 全量实验。'
else:
    scale_advice='C 的 Pilot nDCG@10 提高，探索性区间未跨零；可以考虑下一阶段再批准 Medium/Large 验证，但本阶段不自动扩展。'
quality_conclusion='提升' if delta_b>0 else '下降' if delta_b<0 else '持平'
report=f'''# ScholarRAG Lite — Phase 3.4 Chunking 消融总结

## 新增方法与完成情况

只新增 C：在 Phase 3.2 Sentence-aware（目标 180、软上限 220、硬上限 240 Tokens）上请求 20 Token Overlap。先保留 B 的正文范围，再为后续 chunk 优先加入前一 chunk 的完整句子尾部；完整句子超出预算时，按分词器 offset 从尾部确定性选取 Token。保留完整原正文，不跨既有清洗块拼接。逐 chunk 保存请求值、实际重叠 Token 数、规则、重叠和正文来源范围。默认 Overlap 为零，A/B 的历史数据、索引与结果没有重建。

C 的 2,000 篇 Pilot 切分、BGE-small GPU Embedding、FAISS IndexFlatIP 索引、论文级 Top-100 检索及 50 查询评价均已完成。三组使用相同 Pilot manifest、官方 qrels、规范化 Document ID、模型和查询。C 的 {overlap['chunks_with_overlap']:,}/{groups['C']['performance']['chunks']:,} 个 chunk（{percent(overlap_coverage)}）实际带重叠；全体 chunk 的实际重叠均值为 {overlap['actual_token_lengths']['mean']:.2f} Tokens，带重叠 chunk 的均值为 {conditional_overlap:.2f}，最大为 {overlap['actual_token_lengths']['max']}。其余 chunk 多位于切分块起点，或因软上限无可用空间；实现保留各块边界以维持来源映射与 B 的正文组合。C 的 BGE 实际最大长度为 {chunk_stats_c['bge_token_lengths']['max']} Tokens，超过软上限 220 的有 {chunk_stats_c['bge_over_220']} 个，数量与 B 相同；这是原有正文组合与特殊 Token 计数的边界效应，仍低于绝对上限 240。

## 三组检索质量

| 组 | 方法 | nDCG@10 | Recall@10 | Recall@100 | MRR | Top-10 未评判 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(quality)}

C−B 的 nDCG@10 宏平均差值为 {delta_b:+.4f}，50 查询配对 Bootstrap 的探索性 95% 百分位区间为 [{ci[0]:+.4f}, {ci[1]:+.4f}]（10,000 次、有放回抽样、固定 seed 3401）。同一比较下 Recall@10 差值 {delta_recall10:+.4f}、Recall@100 差值 {delta_recall100:+.4f}、MRR 差值 {delta_mrr:+.4f}、Top-10 未评判比例差值 {delta_unjudged*100:+.1f} 个百分点。逐查询 nDCG@10：C 相比 B 提升 {dir_b['improved']} 个、下降 {dir_b['declined']} 个、持平 {dir_b['unchanged']} 个。C−A 的 nDCG@10 差值为 {delta_a:+.4f}；A 与 C 的切分策略本身不同，不能把这项差值全部归因于 Overlap。

Recall 分母为**当前 Pilot 语料中的已知正例**，不是完整 TREC-COVID Recall。未评判文档在传统 qrels 计分中按零增益处理，但真实相关性未知。

## 计算效率与资源

| 组 | Chunk 数 | 平均 BGE Token | 切分中位秒 | Embedding 中位秒 | FAISS MiB | 检索延迟中位 ms | RAM 峰值 GiB | VRAM reserved 峰值 GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(cost)}

C 与 B 的 chunk 数和 FAISS 文件字节数相同；平均 Token 长度变化 {added_tokens:+.2f}，切分中位耗时变化 {added_chunking:+.1f} 秒（{chunking_pct:+.1f}%），Embedding 中位耗时变化 {added_embed:+.1f} 秒（{embedding_pct:+.1f}%），FAISS 文件变化 {added_index/1024**2:+.2f} MiB。检索延迟中位数差值 {latency_delta_ms:+.2f} ms，单机波动下不能把该差值直接归因于 Overlap。三组切分、Embedding 与检索延迟均使用三次计时的中位数；切分列取共同清洗与切分阶段时间，检索延迟为论文级 Top-100，不含查询编码。RAM/VRAM 为 Embedding 进程峰值。具体重复值与逐查询分数见 `experiments/results/phase34_ablation/comparison.json`。

## 结论与限制

C 相比 B 的 nDCG@10 {quality_conclusion} {abs(delta_b):.4f}，幅度很小且配对区间跨零；Recall@100 和 MRR 小幅提高，Recall@10 下降，质量证据方向不一。24 个查询的 nDCG@10 提升与近零的宏平均并存，说明下降查询的幅度抵消了部分提升。C 与 A 的 nDCG@10 相差 {delta_a:+.4f}；C 的 chunk 数、Embedding 耗时和索引空间也高于 A，未呈现一致的质量或成本优势。A 与 C 的比较只反映两套完整策略的相对表现。Overlap 只在既有清洗块内生效，因此全体 chunk 的平均实际重叠小于请求的 20 Tokens。固定 956 篇正例锚点、50 个查询、较高的未评判比例和单机性能波动限制结论外推；Bootstrap 为探索性描述，不替代独立数据验证。本次未进行参数搜索，也未针对测试查询调整规则。

{scale_advice} 工程上现有检索结果和索引可供 LLM + Streamlit 集成使用；集成须由用户另行批准，本阶段没有接入 API 或开发界面。

新增 chunk、向量、索引和逐查询结果分别保存在 `data/processed/phase34_ablation_*`、`data/indexes/phase34_ablation/`、`experiments/results/phase34_ablation/`。
'''
(ROOT/'phase_3_4_summary.md').write_text(report,encoding='utf8')
print(json.dumps({'ndcg_C_minus_B':delta_b,'bootstrap_95':ci,
                  'queries_C_vs_B':dir_b,'chunks_C':groups['C']['performance']['chunks']},indent=2))
