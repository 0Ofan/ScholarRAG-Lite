"""Read-only evidence preview for freezing Phase 3.6 questions before API calls."""
import json
from pathlib import Path

from src.rag_service import load_retriever, retrieve_evidence

ROOT = Path(__file__).resolve().parents[1]
questions = json.loads((ROOT / 'experiments/configs/phase36_questions.json').read_text(encoding='utf8'))['questions']
retriever, load_seconds = load_retriever('phase33_large_fixed')
preview = []
for item in questions:
    evidence, _, timings = retrieve_evidence(retriever, item['question'], 3)
    preview.append({'question_id': item['question_id'], 'question': item['question'],
                    'evidence': evidence, 'timings': timings})
    print(item['question_id'], item['category'], item['question'], flush=True)
    for e in evidence:
        print(' ', e['citation'], e['doc_id'], e['title'][:90], flush=True)
        print('   ', e['text'][:340].replace('\n', ' ').encode('gbk', 'replace').decode('gbk'), flush=True)
out = ROOT / 'experiments/results/phase36'
out.mkdir(parents=True, exist_ok=True)
(out / 'evidence_preview.json').write_text(json.dumps({'load_seconds': load_seconds,
                                                        'items': preview}, ensure_ascii=False, indent=2), encoding='utf8')
