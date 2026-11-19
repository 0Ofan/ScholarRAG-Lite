"""Fetch a bounded CORD-19 2020-07-16 sample and TREC-COVID labels."""
import csv
import hashlib
import io
import json
import re
import statistics
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/cord19_2020-07-16'
EVAL = ROOT / 'data/evaluation'
PROC = ROOT / 'data/processed'
BASE = 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-07-16/'
UA = {'User-Agent': 'Mozilla/5.0 (ScholarRAG-Lite research sample)'}

def open_url(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90)

def fetch_small(url, path):
    if not path.exists():
        with open_url(url) as response, path.open('wb') as out:
            while data := response.read(65536):
                out.write(data)
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def main(limit=150):
    for p in (RAW, EVAL, PROC): p.mkdir(parents=True, exist_ok=True)
    qpath = EVAL / 'trec_covid_complete_qrels.txt'
    tpath = EVAL / 'trec_covid_topics.xml'
    sources = {
        'qrels': fetch_small('https://ir.nist.gov/covidSubmit/data/qrels-covid_d5_j0.5-5.txt', qpath),
        'topics': fetch_small('https://ir.nist.gov/covidSubmit/data/topics-rnd5.xml', tpath),
    }
    labels = {}
    for line in qpath.read_text().splitlines():
        topic, iteration, uid, rel = line.split()
        labels[(topic, uid)] = int(rel)
    topics = []
    for node in ET.parse(tpath).getroot():
        topics.append({'topic_id': node.attrib.get('number') or node.findtext('number'),
                       'query': (node.findtext('query') or '').strip(),
                       'question': (node.findtext('question') or '').strip()})
    with (EVAL / 'topics.jsonl').open('w', encoding='utf8') as out:
        for item in topics: out.write(json.dumps(item, ensure_ascii=False) + '\n')
    sample = {}
    bytes_read = 0
    with open_url(BASE + 'document_parses.tar.gz') as response:
        class Meter:
            def read(self, size=-1):
                nonlocal bytes_read
                b = response.read(size)
                bytes_read += len(b)
                return b
        with tarfile.open(fileobj=Meter(), mode='r|gz') as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith('.json'): continue
                if member.size > 5_000_000: continue
                blob = tar.extractfile(member).read()
                try: obj = json.loads(blob)
                except (ValueError, UnicodeDecodeError): continue
                body = obj.get('body_text') or []
                if not body: continue
                key = Path(member.name).name
                (RAW / key).write_bytes(blob)
                sample[key] = {'archive_path': member.name, 'paper_id': obj.get('paper_id'),
                               'format': 'pmc_json' if 'pmc_json' in member.name else 'pdf_json'}
                if len(sample) >= limit: break
    # Official metadata is streamed, never retained in full. Join by parse path or SHA.
    joined = {}
    with open_url(BASE + 'metadata.csv') as response:
        text = io.TextIOWrapper(response, encoding='utf-8-sig', newline='')
        for row in csv.DictReader(text):
            paths = (row.get('pdf_json_files') or '') + ';' + (row.get('pmc_json_files') or '')
            for path in paths.split(';'):
                key = Path(path).name
                if key in sample and key not in joined:
                    joined[key] = row
            if len(joined) == len(sample): break
    records = []
    for key, info in sample.items():
        obj = json.loads((RAW / key).read_text(encoding='utf8'))
        row = joined.get(key, {})
        uid = row.get('cord_uid')
        body = obj.get('body_text') or []
        sections = [{'section_id': str(i), 'heading': x.get('section') or '', 'text': x.get('text') or '', 'order': i}
                    for i, x in enumerate(body) if x.get('text')]
        records.append({'doc_id': uid, 'paper_id': obj.get('paper_id'), 'title': row.get('title') or obj.get('metadata', {}).get('title'),
                        'abstract': row.get('abstract') or ' '.join(x.get('text','') for x in obj.get('abstract', [])),
                        'sections': sections, 'full_text': '\n'.join(x['text'] for x in sections),
                        'metadata': {'doi': row.get('doi'), 'publish_time': row.get('publish_time'), 'license': row.get('license'), 'sha': row.get('sha')},
                        'source': info['archive_path'], 'format': info['format']})
    with (PROC / 'cord19_sample.jsonl').open('w', encoding='utf8') as out:
        for record in records: out.write(json.dumps(record, ensure_ascii=False) + '\n')
    ids = {r['doc_id'] for r in records if r['doc_id']}
    relevant = {(topic, uid) for (topic, uid), grade in labels.items() if grade > 0}
    lengths = [len(r['full_text']) for r in records]
    stats = {'downloaded': len(sample), 'parsed': len(records), 'joined': len(ids), 'raw_compressed_bytes_streamed': bytes_read,
             'full_text': sum(bool(x) for x in lengths), 'empty_text': sum(not x for x in lengths),
             'missing_title': sum(not r['title'] for r in records), 'missing_metadata': sum(not r['metadata']['doi'] for r in records),
             'duplicate_doc_ids': len(records)-len(ids), 'mean_chars': statistics.mean(lengths),
             'median_chars': statistics.median(lengths), 'p95_chars': sorted(lengths)[int(.95*(len(lengths)-1))],
             'formats': dict(Counter(r['format'] for r in records)), 'qrels': len(labels), 'topics': len(topics),
             'qrel_doc_ids': len({uid for _,uid in labels}), 'sample_qrel_doc_matches': len(ids & {uid for _,uid in labels}),
             'sample_relevant_doc_matches': len(ids & {uid for _,uid in relevant}),
             'sample_relevant_pairs': sum(uid in ids for _,uid in relevant),
             'query_relevant_counts': dict(Counter(topic for topic,uid in relevant if uid in ids)),
             'files': sources, 'source_snapshot': '2020-07-16'}
    (EVAL / 'sample_stats.json').write_text(json.dumps(stats, indent=2), encoding='utf8')
    print(json.dumps(stats, indent=2))

if __name__ == '__main__': main()
