"""Bounded, qrels-guided CORD-19 2020-07-16 full-text collection."""
import csv
import io
import json
import os
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/cord19_2020-07-16/phase25'
EVAL = ROOT / 'data/evaluation'
BASE = 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-07-16/'
MAX_COMPRESSED = 350_000_000
MAX_DOCS = 500

def urlopen(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (ScholarRAG Lite research)'}),timeout=120)

def main():
    RAW.mkdir(parents=True,exist_ok=True)
    qrels=defaultdict(dict)
    for line in (EVAL/'trec_covid_complete_qrels.txt').read_text().splitlines():
        q,iteration,uid,grade=line.split(); qrels[q][uid]=int(grade)
    positives={uid for judged in qrels.values() for uid,g in judged.items() if g>0}
    mappings={}; no_parse=[]; metadata_matches=0
    with urlopen(BASE+'metadata.csv') as response:
        reader=csv.DictReader(io.TextIOWrapper(response,encoding='utf-8-sig',newline=''))
        for row in reader:
            uid=row.get('cord_uid','')
            if uid not in positives: continue
            metadata_matches+=1
            paths=[p.strip() for p in (row.get('pmc_json_files','')+';'+row.get('pdf_json_files','')).split(';') if p.strip()]
            if not paths: no_parse.append(uid); continue
            # Prefer PMC parse when both exist.
            path=paths[0]
            if path in mappings and mappings[path]['doc_id']!=uid:
                raise ValueError('共享解析路径不能映射为单个 cord_uid；请使用 phase26_judged_paths.py 的多对多索引')
            mappings[path]={'doc_id':uid,'doi':row.get('doi'),'title':row.get('title'),'abstract':row.get('abstract'),
                            'license':row.get('license'),'publish_time':row.get('publish_time'),'path':path}
    (EVAL/'phase25_positive_metadata.json').write_text(json.dumps({'mappings':mappings,'no_parse':no_parse},ensure_ascii=False),encoding='utf8')
    selected={}; bytes_read=0; inspected=0; stop='archive_end'
    with urlopen(BASE+'document_parses.tar.gz') as response:
        class Meter:
            def read(self,n=-1):
                nonlocal bytes_read
                if bytes_read >= MAX_COMPRESSED: return b''
                data=response.read(min(n,MAX_COMPRESSED-bytes_read) if n>=0 else MAX_COMPRESSED-bytes_read)
                bytes_read+=len(data)
                return data
        try:
            with tarfile.open(fileobj=Meter(),mode='r|gz') as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith('.json'): continue
                    inspected+=1
                    meta=mappings.get(member.name)
                    if meta is None or meta['doc_id'] in selected: continue
                    if member.size > 5_000_000: continue
                    blob=tar.extractfile(member).read()
                    try: obj=json.loads(blob)
                    except ValueError: continue
                    body=obj.get('body_text') or []
                    if not body: continue
                    path=RAW/(meta['doc_id']+'.json')
                    if not path.exists(): path.write_bytes(blob)
                    selected[meta['doc_id']]=meta
                    if len(selected)>=MAX_DOCS: stop='doc_limit'; break
        except (EOFError,tarfile.ReadError) as exc:
            stop='byte_limit' if bytes_read>=MAX_COMPRESSED else 'archive_error:'+str(exc)
    query_stats={}
    for q,judged in sorted(qrels.items(),key=lambda kv:int(kv[0])):
        rel={uid for uid,g in judged.items() if g>0}
        mapped=rel & {m['doc_id'] for m in mappings.values()}
        full=rel & set(selected)
        query_stats[q]={'official_positive':len(rel),'metadata_with_parse':len(mapped),'collected_fulltext':len(full),
                        'collected_positive_ids':sorted(full)}
    summary={'snapshot':'2020-07-16','official_topics':len(qrels),'qrels':sum(map(len,qrels.values())),
             'positive_unique_docs':len(positives),'positive_pairs':sum(sum(g>0 for g in j.values()) for j in qrels.values()),
             'positive_metadata_rows':metadata_matches,'positive_with_parse':len({v['doc_id'] for v in mappings.values()}),
             'positive_without_parse':len(set(no_parse)-{v['doc_id'] for v in mappings.values()}),'collected_fulltext_unique':len(selected),
             'compressed_bytes_streamed':bytes_read,'archive_json_inspected':inspected,'stop_reason':stop,
             'queries_with_fulltext_positive':sum(x['collected_fulltext']>0 for x in query_stats.values()),
             'queries_with_5_fulltext_positives':sum(x['collected_fulltext']>=5 for x in query_stats.values()),
             'per_query':query_stats}
    (EVAL/'phase25_collection_stats.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps({k:v for k,v in summary.items() if k!='per_query'},indent=2))

if __name__=='__main__': main()
