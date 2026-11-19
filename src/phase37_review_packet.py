"""Print frozen evidence and paired answers for manual semantic review."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
questions = json.loads((ROOT/'experiments/configs/phase36_questions.json').read_text(encoding='utf8'))['questions']
old_reviews = {r['question_id']: r for r in json.loads(
    (ROOT/'experiments/results/phase36/reviewer_scores.json').read_text(encoding='utf8'))['reviews']}
first = int(sys.argv[1]) if len(sys.argv) > 1 else 1
last = int(sys.argv[2]) if len(sys.argv) > 2 else 20
sys.stdout.reconfigure(errors='replace')
for q in questions[first-1:last]:
    qid = q['question_id']
    old = json.loads((ROOT/'experiments/results/phase36'/f'{qid}.json').read_text(encoding='utf8'))
    new = json.loads((ROOT/'experiments/results/phase37'/f'{qid}.json').read_text(encoding='utf8'))
    print('\n' + '='*60)
    print(qid, q['category'], q['question'])
    print('EXPECTED:', ' | '.join(q['expected_answer_points']))
    print('OLD REVIEW:', old_reviews[qid]['review_note'])
    for item in new['evidence']:
        print(item['citation'], item['doc_id'], item['title'], '::', item['text'])
    print('OLD:', old['answer'])
    print('NEW:', new.get('answer'))
    c = new.get('citation_check', {})
    print('STRUCTURE:', 'invalid', c.get('invalid'), 'uncited', c.get('uncited_factual_candidates'),
          'multi', c.get('multi_source_sentences_for_review'), 'nonterminal', c.get('nonterminal_citation_sentences'))
