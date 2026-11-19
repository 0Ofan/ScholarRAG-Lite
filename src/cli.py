"""Interactive command-line retrieval demo; no LLM generation."""
import argparse
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('HF_HUB_CACHE',str(ROOT/'.cache/huggingface/hub'))
os.environ.setdefault('TORCH_HOME',str(ROOT/'.cache/torch'))

EXAMPLES=['coronavirus origin','coronavirus response to weather changes',
          'difference between coronavirus and flu']

def display(docs,perf):
    print(f"查询编码 {perf['query_encoding_seconds']:.3f}s；检索 {perf['retrieval_seconds']:.3f}s")
    for d in docs:
        print(f"\n{d['rank']}. {d['title']}\n   ID: {d['doc_id']}  DOI: {d['doi'] or '无'}")
        print(f"   归一化向量内积: {d['score']:.4f}  章节: {d['section_title'] or '未标注'}")
        print(f"   最佳 Chunk: {d['best_chunk_id']}\n   {d['best_chunk_text'][:600]}")

def main():
    p=argparse.ArgumentParser(description='ScholarRAG Lite Phase 3.1 Pilot Baseline retrieval')
    p.add_argument('--query',help='Run one query and exit')
    p.add_argument('--top-k',type=int,default=10)
    p.add_argument('--variant',choices=('phase31','phase32'),default='phase31',
                   help='Choose Fixed-length baseline or Sentence-aware Pilot index')
    p.add_argument('--examples',action='store_true',help='Show three sample queries')
    args=p.parse_args()
    if args.examples:
        for q in EXAMPLES:print('-',q)
        return
    from .retriever import PilotRetriever
    retriever=PilotRetriever(variant=args.variant)
    if args.query:
        docs,perf=retriever.retrieve(args.query,args.top_k);display(docs,perf);return
    print(f'ScholarRAG Lite Pilot {args.variant}。输入 exit 退出。示例查询：')
    for q in EXAMPLES:print('-',q)
    while True:
        try:q=input('\n学术问题> ').strip()
        except (EOFError,KeyboardInterrupt):break
        if q.lower() in ('exit','quit'):break
        if not q:continue
        try:
            k=input('Top-K [10]> ').strip()
            docs,perf=retriever.retrieve(q,int(k) if k else 10)
            display(docs,perf)
        except (ValueError,RuntimeError) as exc:print('错误：',exc)

if __name__=='__main__':main()
