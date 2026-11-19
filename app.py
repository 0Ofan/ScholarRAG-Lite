"""Local ScholarRAG Lite Streamlit demonstration."""
import json
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.deepseek_client import DeepSeekClient, DeepSeekRequestError, MissingAPIKey, is_complete_generation
from src.rag_service import VARIANTS, load_retriever, retrieve_evidence, prepare_grounded_evidence
from src.citation_grounding import verify_evidence_source, validate_grounded_answer
from src.evidence_metadata import enrich
from src.evidence_verifier import verify
from src.answer_refinement import refine_once

ROOT = Path(__file__).resolve().parent

STYLES = """
<style>
:root { --navy:#172B4D; --blue:#2563EB; --canvas:#F4F6FA; --ink:#1E293B;
        --muted:#64748B; --line:#DCE3ED; }
[data-testid="stAppViewContainer"] { background:var(--canvas); color:var(--ink); }
[data-testid="stHeader"] { background:var(--canvas); }
[data-testid="stSidebar"] { background:#172B4D; }
[data-testid="stSidebar"] * { color:#E8EDF7; }
[data-testid="stSidebar"] .stButton button { width:100%; justify-content:flex-start;
  border:1px solid transparent; border-radius:6px; padding:.64rem .8rem;
  background:transparent; color:#E8EDF7; font-weight:500; }
[data-testid="stSidebar"] .stButton button:hover { background:#253E64; border-color:#486184; }
[data-testid="stSidebar"] .stButton button[kind="primary"] {
  background:#2563EB; color:white; border-color:#3B75EF; }
.block-container { max-width:1480px; padding-top:2rem; padding-bottom:3.5rem; }
h1,h2,h3 { color:var(--navy); letter-spacing:-.025em; }
h1 { font-weight:700; }
[data-testid="stForm"], [data-testid="stMetric"], [data-testid="stExpander"] {
  background:#FFFFFF; border:1px solid var(--line); border-radius:7px; }
[data-testid="stMetric"] { padding:.8rem 1rem; min-height:100px; }
[data-testid="stMetricLabel"] { color:var(--muted); }
[data-testid="stMetricValue"] { color:var(--navy); }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:6px; }
div.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button {
  background:var(--blue); color:#fff; border-radius:6px; font-weight:600; }
div.stButton > button, [data-testid="stFormSubmitButton"] button { min-height:2.6rem; }
textarea, input { background:#fff !important; }
.sr-eyebrow { color:var(--blue); font-size:.74rem; font-weight:700;
  letter-spacing:.12em; text-transform:uppercase; margin-bottom:.2rem; }
.sr-subtitle { color:var(--muted); font-size:1rem; margin-top:-.35rem; margin-bottom:1.4rem; }
.sr-brand { display:flex; gap:.7rem; align-items:center; padding:.35rem 0 1rem; }
.sr-brand-mark { background:#2563EB; color:white !important; width:2.2rem; height:2.2rem;
  display:flex; align-items:center; justify-content:center; border-radius:5px; font-weight:800; }
.sr-brand strong { color:white; display:block; line-height:1.2; font-size:1rem; }
.sr-brand small { color:#AFC0D8; font-size:.76rem; }
.sr-sidebar-label { color:#AFC0D8; font-size:.7rem; letter-spacing:.1em;
  text-transform:uppercase; margin:.5rem 0 .7rem; }
.sr-sidebar-meta { border-top:1px solid #405575; padding-top:1rem;
  margin-top:2rem; color:#B9C6D8; font-size:.78rem; line-height:1.8; }
.sr-sidebar-meta b { color:#F1F5FB; font-weight:500; }
.sr-badge { display:inline-block; padding:.22rem .55rem; border-radius:4px;
  font-size:.76rem; font-weight:600; letter-spacing:.01em; }
.sr-verified { background:#E9F7EF; color:#166534; }
.sr-potential { background:#FFF4E4; color:#92400E; }
.sr-contradicted { background:#FDECEC; color:#B91C1C; }
.sr-unverified { background:#EEF1F5; color:#475569; }
.sr-answer-note { color:var(--muted); font-size:.83rem; }
@media (max-width:850px) { .block-container { padding:1rem .8rem 2rem; }
  [data-testid="stMetric"] { min-height:82px; } }
</style>
"""


def page_header(title, subtitle, eyebrow='ScholarRAG Lite'):
    if eyebrow:
        st.markdown(f'<div class="sr-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="sr-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def status_badge(status):
    css = {'Verified': 'sr-verified', 'Potential Issue': 'sr-potential',
           'Contradicted': 'sr-contradicted', 'Unverified': 'sr-unverified'}
    st.markdown(f'<span class="sr-badge {css.get(status, "sr-unverified")}">{status}</span>',
                unsafe_allow_html=True)


@st.cache_resource(show_spinner='正在加载 BGE 与论文索引…')
def cached_retriever(run_id):
    return load_retriever(run_id)[0]


def show_evidence(evidence):
    for rank, item in enumerate(evidence, 1):
        with st.expander(f"{item['citation']}  {item['title']}", expanded=False):
            st.caption(f"RETRIEVAL RANK {rank}  ·  SIMILARITY {item['score']:.3f}")
            st.caption(f"Document ID: {item['doc_id']}  ·  Chunk ID: {item['chunk_id']}")
            if item.get('source_role'):
                st.caption(f"Source Role: {item['source_role']}")
            st.write(item['text'])


def show_timing(timing, generation=None):
    cols = st.columns(4)
    cols[0].metric('查询编码', f"{timing['query_embedding_seconds']:.3f} s")
    cols[1].metric('FAISS + 论文聚合', f"{timing['retrieval_seconds']:.3f} s")
    cols[2].metric('证据整理', f"{timing['evidence_seconds']:.3f} s")
    if generation:
        cols[3].metric('API 首字', f"{generation['api_ttft_seconds']:.3f} s" if generation['api_ttft_seconds'] is not None else '不可用')
        st.caption(f"API 完整生成：{generation['api_total_seconds']:.3f} s；"
                   f"提交至首字：{generation['submit_to_first_text_seconds']:.3f} s" if generation['submit_to_first_text_seconds'] is not None else
                   f"API 完整生成：{generation['api_total_seconds']:.3f} s；提交至首字：不可用")
        st.caption(f"提交至完成：{generation['submit_to_complete_seconds']:.3f} s；"
                   f"Input Tokens：{generation['usage']['input_tokens'] if generation['usage'] else '不可用'}；"
                   f"Output Tokens：{generation['usage']['output_tokens'] if generation['usage'] else '不可用'}")
    else:
        cols[3].metric('检索总耗时', f"{timing['retrieval_total_seconds']:.3f} s")


def show_history(history):
    if not history:
        return
    with st.expander(f'本次会话历史（{len(history)} 条）', expanded=False):
        selected = st.selectbox('选择历史记录', range(len(history) - 1, -1, -1),
                                format_func=lambda i: f"{history[i]['question']} · "
                                f"{'V2 Experimental' if history[i]['use_v2'] else 'V1'}")
        record = history[selected]
        mode = 'V2 Experimental' if record['use_v2'] else 'V1'
        st.markdown(f"**{record['question']}**")
        st.caption(f"{mode}  ·  {record['variant']}")
        if record['answer']:
            st.markdown(record['answer'])
        for item in record['evidence']:
            st.caption(f"{item['citation']} · {item['doc_id']} · {item['title']}")
        if record.get('error'):
            st.warning(record['error'])
        if record.get('warning'):
            st.warning(record['warning'])


def search_page():
    page_header('Research Assistant',
                'Search scientific literature and generate evidence-based answers.',
                eyebrow=None)
    with st.form('ask'):
        st.markdown('#### Ask a research question')
        question = st.text_area('Research question', height=110,
                                placeholder='例如：What do studies report about COVID-19 outcomes in children?')
        settings = st.columns(2, gap='medium')
        with settings[0]:
            variant = st.selectbox('Retrieval method', list(VARIANTS), index=0)
        with settings[1]:
            evidence_count = st.select_slider('Evidence chunks', options=[3, 4, 5], value=3)
        mode = st.segmented_control('Answer mode',
                                    ['V1 — Stable Default', 'V2 — Experimental'],
                                    default='V1 — Stable Default', required=True)
        use_v2 = mode == 'V2 — Experimental'
        submitted = st.form_submit_button('Search literature and answer', type='primary',
                                          use_container_width=True)
    st.caption('V2 adds source roles and local verification hints. V1 remains the default.')
    history = st.session_state.setdefault('answer_history', [])
    show_history(history)

    if submitted and question.strip():
        start = time.perf_counter()
        try:
            with st.spinner('Retrieving Papers · BGE + FAISS'):
                retriever = cached_retriever(VARIANTS[variant])
                load_seconds = time.perf_counter() - start
                raw_evidence, docs, timing = retrieve_evidence(retriever, question.strip(), evidence_count)
            with st.spinner('Selecting Evidence · paper-level Top-K'):
                evidence = prepare_grounded_evidence(raw_evidence)
                if use_v2:
                    evidence = enrich(evidence)
                source_map = {item['chunk_id']: retriever.chunks[doc['chunk_index']]
                              for item, doc in zip(evidence, docs)}
                source_checks = verify_evidence_source(evidence, source_map)
                if any(row['status'] != 'valid' for row in source_checks):
                    raise ValueError('检索片段与原始 Chunk 元数据不一致')
        except Exception as exc:
            st.error(f'检索失败（{type(exc).__name__}）。')
            return
        timing['cold_load_seconds'] = load_seconds
        record = {
            'question': question.strip(), 'variant': variant, 'use_v2': use_v2, 'evidence': evidence,
            'timing': timing, 'answer': '', 'generation': None, 'verification': None,
            'status': 'retrieved', 'error': None,
        }
        st.session_state['last_answer'] = record
        history.append(record)
        st.divider()
        st.caption(f"MODEL & INDEX LOAD / CACHE  {load_seconds:.2f} s  ·  query timing shown separately")
        show_timing(timing)
        if not evidence:
            st.info('当前索引未返回论文证据，无法据此生成回答。')
            return
        answer_col, evidence_col = st.columns([1.55, 1], gap='large')
        with answer_col:
            st.subheader('Evidence-based answer')
            st.caption('Generated by DeepSeek Flash · citations map to the papers on the right')
            placeholder = st.empty()
            progress = st.empty()
        with evidence_col:
            st.subheader('Retrieved papers')
            st.caption('Open a paper to inspect its original evidence chunk.')
            show_evidence(evidence)
        progress.info('正在连接 DeepSeek；收到文字后会实时显示。')
        fragments = []
        meta = None
        first_submit = None
        try:
            client = DeepSeekClient()
            prompt = (ROOT / 'prompts/scholar_rag_v2.txt').read_text(encoding='utf-8').strip() if use_v2 else None
            for kind, payload in client.stream(question.strip(), evidence, system_prompt=prompt):
                if kind == 'text':
                    if first_submit is None and payload.strip():
                        first_submit = time.perf_counter() - start
                    fragments.append(payload)
                    placeholder.markdown(''.join(fragments) + '▌')
                else:
                    meta = payload
        except MissingAPIKey as exc:
            record['error'] = str(exc)
            st.info(record['error'])
        except DeepSeekRequestError as exc:
            record['error'] = str(exc)
            st.error(record['error'])
        except Exception as exc:
            record['error'] = f'回答生成失败（{type(exc).__name__}）；论文检索结果仍可使用。'
            st.error(record['error'])
        finally:
            progress.empty()
            answer = ''.join(fragments)
            record['answer'] = answer
            placeholder.markdown(answer or '模型未返回有效文字。')
        if record['error']:
            record['status'] = 'interrupted' if answer else 'api_error'
            if answer:
                st.warning('以上仅为中断前的部分文字，不能作为完整回答。')
            return
        if not meta:
            record['status'] = 'interrupted'
            st.warning('流式响应未返回完成事件；当前文字不是已确认的完整回答。')
            return
        meta['submit_to_first_text_seconds'] = first_submit
        meta['submit_to_complete_seconds'] = time.perf_counter() - start
        record['generation'] = meta
        if not is_complete_generation(answer, meta):
            record['status'] = 'interrupted'
            reason = meta.get('finish_reason') or '未返回结束原因'
            st.warning(f'回答未完整生成（结束原因：{reason}）；已保留收到的文字，请核对原始证据。')
            return
        record['status'] = 'complete'
        verification = None
        revision = None
        if use_v2 and answer:
            with st.spinner('Verifying Evidence · local review hints'):
                try:
                    verification = verify(answer, evidence)
                except Exception as exc:
                    record['warning'] = f'本地证据核查失败（{type(exc).__name__}）；已保留原回答，事实仍需人工核对。'
                    st.warning(record['warning'])
            if verification and verification['critical']:
                try:
                    revision = refine_once(client, question.strip(), answer, evidence, verification, system_prompt=prompt)
                    answer, verification = revision['answer'], revision['verification']
                    record['answer'] = answer
                    placeholder.markdown(answer)
                except Exception as exc:
                    record['warning'] = f'定向修正失败（{type(exc).__name__}）；已保留原回答。'
                    st.warning(record['warning'])
        record['verification'] = verification
        record['revised'] = bool(revision and revision['revised'])
        show_timing(timing, meta)
        citations = validate_grounded_answer(answer, evidence, source_checks)
        if citations['invalid']:
            st.warning('回答中出现无效引用或来源映射错误：' + ', '.join(citations['invalid']))
        if citations['explicit_attribution_mismatch_sentences']:
            st.warning('回答显式提及的论文或 Document ID 与句末引用不匹配；请核对对应句子。')
        if citations['uncited_factual_candidates'] or citations['nonterminal_citation_sentences']:
            st.warning('自动检查发现可能缺少句子级引用或引用位置不当；请对照下方原文核查。')
        if citations['multi_source_sentences_for_review']:
            st.caption('含多篇论文引用的句子需要人工核对各自支持的结论。')
        st.caption('引用编号与来源可自动核验；引用是否真正支持句子仍需人工核对。')
        if meta['reasoning_content_seen']:
            st.warning('响应包含思考内容；非思考模式请求未按预期生效。')
        if meta['finish_reason'] == 'length':
            st.info('回答达到输出 Token 上限。')
        if verification:
            st.caption(f"本地验证 {verification['seconds']:.4f} s；仅用于提示，不能证明医学事实正确。")
            if revision:
                st.info('已进行一次定向修正。' if revision['revised'] else revision.get('warning', '修正未采用。'))
            if verification['critical']:
                st.warning('仍存在明确的证据矛盾，请核对原文。')
            with st.expander('Verification Details', expanded=False):
                statuses = {check['status'] for check in verification['claims']}
                overall = ('Contradicted' if 'Contradicted' in statuses else
                           'Potential Issue' if 'Potential Issue' in statuses else 'Unverified')
                st.write('Verification Status')
                status_badge(overall)
                st.caption('Unverified 表示未能自动判断，不是错误；启发式提示也不是医学事实认证。')
                for check in verification['claims']:
                    st.divider()
                    st.caption(f"{check['claim_id']} · {check['issue_type'] or '语义未验证'}")
                    status_badge(check['status'])
                    st.write(check['claim_text'])
                    st.caption(check['reason'])
                    for eid, doc, role, original in zip(check['evidence_ids'] or [e['evidence_id'] for e in evidence],
                                                          check['document_ids'], check['source_roles'], check['original_evidence']):
                        st.caption(f'{eid} · {doc} · {role}')
                        st.text(original)
    elif (not submitted and 'last_answer' in st.session_state
          and st.session_state['last_answer']['variant'] == variant
          and st.session_state['last_answer'].get('use_v2', False) == use_v2
          and st.session_state['last_answer']['question'] == question.strip()):
        last = st.session_state['last_answer']
        st.caption(f"上次提交：{'V2 Experimental' if last['use_v2'] else 'V1'} · {last['question']}")
        st.subheader('Previous answer')
        if last['answer']:
            st.markdown(last['answer'])
            citations = validate_grounded_answer(last['answer'], last['evidence'])
            if citations['invalid']:
                st.warning('回答中的无效引用：' + ', '.join(citations['invalid']))
            if last.get('verification'):
                st.caption('V2 本地验证已完成；上方会话历史保留该回答。')
        if last.get('error'):
            st.warning(last['error'])
        st.subheader('Paper evidence')
        show_evidence(last['evidence'])
        show_timing(last['timing'], last['generation'])


def comparison_page():
    page_header('Retrieval Method Comparison',
                'Compare retrieval strategies and evaluate their evidence quality.')
    with st.container(border=True):
        question = st.text_input('Research question', key='compare_question',
                                 placeholder='Enter the same question for both retrieval methods')
        top_k = st.slider('Papers to display', 3, 10, 5)
        compare = st.button('Compare retrieval methods', type='primary',
                            disabled=not question.strip())
    if not compare:
        return
    outputs = {}
    for label, run_id in VARIANTS.items():
        retriever = cached_retriever(run_id)
        evidence, docs, timing = retrieve_evidence(retriever, question.strip(), min(top_k, 5))
        outputs[label] = {'docs': docs[:top_k], 'timing': timing}
    names = list(outputs)
    left = {d['doc_id'] for d in outputs[names[0]]['docs']}
    right = {d['doc_id'] for d in outputs[names[1]]['docs']}
    shared = left & right
    kpis = st.columns(3, gap='medium')
    kpis[0].metric('Shared papers', len(shared))
    kpis[1].metric('Fixed-only papers', len(left - right))
    kpis[2].metric('Sentence-aware-only papers', len(right - left))
    st.caption('Scores are normalized BGE vector similarities; each method keeps its original paper ranking.')
    for column, label in zip(st.columns(2), names):
        with column:
            st.subheader(label)
            st.caption(f"RETRIEVAL TIME  {outputs[label]['timing']['retrieval_total_seconds']:.3f} s")
            for doc in outputs[label]['docs']:
                relation = 'SHARED' if doc['doc_id'] in shared else 'UNIQUE'
                with st.expander(f"{doc['rank']:02d}  {doc['title']}"):
                    st.caption(f"{relation}  ·  SIMILARITY {doc['score']:.3f}")
                    st.caption(f"Document ID: {doc['doc_id']} · Chunk ID: {doc['best_chunk_id']}")
                    st.write(doc['best_chunk_text'])


@st.cache_data(ttl=3600, show_spinner=False)
def dashboard_data(phase33_version, phase34_version):
    """Cache frozen experiment displays; file signatures invalidate on update."""
    phase33 = json.loads((ROOT/'experiments/results/phase33/comparison.json').read_text(encoding='utf8'))
    phase34 = json.loads((ROOT/'experiments/results/phase34_ablation/comparison.json').read_text(encoding='utf8'))
    rows = []
    for run in phase33['runs']:
        rows.append({'规模': run['scale'], '论文数': run['documents'], '方法': run['method'],
                     'nDCG@10': run['ndcg@10'], 'Recall@10': run['recall@10'],
                     'Recall@100': run['recall@100'], 'MRR': run['mrr'],
                     'Chunk数': run['chunks'], 'Embedding秒': run['embedding_seconds_median'],
                     'FAISS MiB': run['index_bytes']/1024**2,
                     '检索毫秒': run['retrieval_latency_seconds_median']*1000})
    ablation = []
    for key, group in phase34['groups'].items():
        q, p = group['quality'], group['performance']
        ablation.append({'组': key, '方法': group['label'], 'nDCG@10': q['ndcg@10'],
                         'Recall@10': q['recall@10'], 'Recall@100': q['recall@100'],
                         'MRR': q['mrr'], 'Chunk数': p['chunks'],
                         'Embedding秒': p['embedding_seconds_median'],
                         'FAISS MiB': p['index_size_bytes']/1024**2,
                         '检索毫秒': p['query_latency_seconds_median']*1000})
    ablation_chart = (alt.Chart(pd.DataFrame(ablation))
                      .mark_bar(color='#2563EB')
                      .encode(
                          x=alt.X('组:N', sort=['A', 'B', 'C'], title=None,
                                  scale=alt.Scale(paddingInner=0.55, paddingOuter=0.25),
                                  axis=alt.Axis(labelAngle=0, labelFontSize=16, labelPadding=12,
                                                labelColor='#1E293B', tickSize=0)),
                          y=alt.Y('nDCG@10:Q', title='nDCG@10', scale=alt.Scale(zero=True)),
                          tooltip=[alt.Tooltip('组:N', title='Group'),
                                   alt.Tooltip('方法:N', title='Method'),
                                   alt.Tooltip('nDCG@10:Q', format='.3f')])
                      .properties(height=300)).to_dict()
    return rows, ablation, ablation_chart


def dashboard_page():
    page_header('Experiment Dashboard',
                'Explore retrieval performance, ablation studies and experimental results.')
    phase33_path = ROOT/'experiments/results/phase33/comparison.json'
    phase34_path = ROOT/'experiments/results/phase34_ablation/comparison.json'
    phase33_stat, phase34_stat = phase33_path.stat(), phase34_path.stat()
    rows, ablation, ablation_chart = dashboard_data(
        (phase33_stat.st_mtime_ns, phase33_stat.st_size),
        (phase34_stat.st_mtime_ns, phase34_stat.st_size))
    frame = pd.DataFrame(rows)
    large = {row['方法']: row for row in rows if row['规模'] == 'Large'}
    kpis = st.columns(3, gap='medium')
    kpis[0].metric('Large corpus · papers', f"{large['Fixed-length']['论文数']:,}")
    kpis[1].metric('Large · Fixed nDCG@10', f"{large['Fixed-length']['nDCG@10']:.3f}")
    kpis[2].metric('Large · Sentence-aware nDCG@10', f"{large['Sentence-aware']['nDCG@10']:.3f}")
    st.caption('KPI values come directly from the saved Phase 3.3 comparison; no new experiment was run.')
    quality_tab, resource_tab, ablation_tab = st.tabs(
        ['Retrieval quality', 'Scale & resources', 'Chunking ablation'])
    with quality_tab:
        st.subheader('Pilot / Medium / Large · retrieval quality')
        st.dataframe(frame[['规模', '论文数', '方法', 'nDCG@10',
                            'Recall@10', 'Recall@100', 'MRR']],
                     hide_index=True, width='stretch')
        metric = st.selectbox('Quality metric', ['nDCG@10', 'Recall@10', 'Recall@100', 'MRR'])
        st.markdown(f'#### {metric} by corpus size')
        chart = frame.pivot(index='论文数', columns='方法', values=metric).sort_index()
        st.line_chart(chart, x_label='Papers', y_label=metric)
    with resource_tab:
        st.subheader('Indexing and retrieval cost')
        st.dataframe(frame[['规模', '论文数', '方法', 'Chunk数',
                            'Embedding秒', 'FAISS MiB', '检索毫秒']],
                     hide_index=True, width='stretch')
        resource_metric = st.selectbox('Resource metric',
                                       ['Chunk数', 'Embedding秒', 'FAISS MiB', '检索毫秒'])
        st.markdown(f'#### {resource_metric} by corpus size')
        resource_chart = frame.pivot(
            index='论文数', columns='方法', values=resource_metric).sort_index()
        st.line_chart(resource_chart, x_label='Papers', y_label=resource_metric)
    with ablation_tab:
        st.subheader('Phase 3.4 · A / B / C')
        st.dataframe(ablation, hide_index=True, width='stretch')
        st.markdown('#### nDCG@10 by chunking method')
        st.vega_lite_chart(spec=ablation_chart, width='stretch')
    st.info('Recall 仅以当前语料中的已知正例为分母；正例锚点固定，且大量论文未被 qrels 评判。规模变化不能代表完整 TREC-COVID Recall。')


def render_app():
    st.set_page_config(page_title='ScholarRAG Lite', layout='wide')
    st.markdown(STYLES, unsafe_allow_html=True)
    st.sidebar.markdown('''<div class="sr-brand"><div class="sr-brand-mark">S</div>
      <div><strong>ScholarRAG Lite</strong><small>Research Assistant</small></div></div>
      <div class="sr-sidebar-label">Workspace</div>''', unsafe_allow_html=True)
    pages = [('⌕  Search & Answer', 'Search & Answer'),
             ('⇄  Method Comparison', 'Method Comparison'),
             ('▥  Experiment Dashboard', 'Experiment Dashboard')]
    st.session_state.setdefault('active_page', pages[0][1])

    def select_page(name):
        st.session_state['active_page'] = name

    for label, name in pages:
        st.sidebar.button(label, key=f'nav_{name.replace(" ", "_")}',
                          type='primary' if st.session_state['active_page'] == name else 'secondary',
                          use_container_width=True, on_click=select_page, args=(name,))
    st.sidebar.markdown('''<div class="sr-sidebar-meta">
      Project: <b>ScholarRAG Lite</b><br>
      Model: <b>DeepSeek Flash</b><br>
      Retrieval: <b>BGE + FAISS</b><br>
      Version: <b>V1 Default</b></div>''', unsafe_allow_html=True)
    page = st.session_state['active_page']
    if page == 'Search & Answer':
        search_page()
    elif page == 'Method Comparison':
        comparison_page()
    else:
        dashboard_page()


if __name__ == '__main__':
    render_app()
