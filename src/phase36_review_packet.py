"""Print exact saved answers and evidence for independent Phase 3.6 review."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(errors='replace')
questions = json.loads((ROOT / 'experiments/configs/phase36_questions.json').read_text(encoding='utf8'))['questions']
first = int(sys.argv[1]) if len(sys.argv) > 1 else 1
last = int(sys.argv[2]) if len(sys.argv) > 2 else 20
for q in questions[first-1:last]:
    result = json.loads((ROOT / 'experiments/results/phase36' / f"{q['question_id']}.json").read_text(encoding='utf8'))
    print('\n' + '='*50)
    print(q['question_id'], q['category'], q['question'])
    print('Expected:', ' | '.join(q['expected_answer_points']))
    print('Availability:', q['evidence_availability'], q['evaluation_notes'])
    for e in result['evidence']:
        print('\n', e['citation'], e['doc_id'], e['title'])
        print(e['text'])
    print('\nANSWER:', result.get('answer', ''))
    print('CITATIONS:', result.get('citation_check'))
