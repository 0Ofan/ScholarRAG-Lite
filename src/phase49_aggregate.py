"""Summarize existing Phase 4.9 records without new API calls."""
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from src.evidence_metadata import enrich
from src.evidence_verifier import verify

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results/phase49'
P45 = ROOT / 'experiments/results/phase45'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    rows, claim_rows = [], []
    for n in range(1, 21):
        qid = f'Q{n:02d}'
        old, new, assessment = (read(P45 / f'A_{qid}.json'), read(OUT / f'{qid}.json'),
                                read(OUT / f'eval_{qid}.json'))
        assert old['status'] == new['status'] == assessment['status'] == 'success'
        assert old['evidence_sha256'] == new['v1_evidence_sha256']
        assert [(e['evidence_id'],e['doc_id'],e['chunk_id'],e['text']) for e in old['evidence']] == [
            (e['evidence_id'],e['doc_id'],e['chunk_id'],e['text']) for e in new['evidence']]
        old_check = verify(old['answer'], enrich(old['evidence']))
        for claim in new['verification']['claims']:
            claim_rows.append({'question_id': qid, **claim})
        row = {'question_id': qid, 'question': old['question'],
               'v1_answer': old['answer'], 'v2_answer': new['answer'],
               'same_evidence_sha256': old['evidence_sha256'],
               'v1_local_issues': '|'.join(c['issue_type'] for c in old_check['claims'] if c['issue_type']),
               'v2_local_issues': '|'.join(c['issue_type'] for c in new['verification']['claims'] if c['issue_type']),
               'revision_used': bool(new['revision'] and new['revision']['revised']),
               'v2_api_seconds': new['generation']['api_total_seconds'],
               'v2_ttft_seconds': new['generation']['api_ttft_seconds'],
               'v2_verification_seconds': new['verification']['seconds'],
               'v2_input_tokens': new['generation']['usage']['input_tokens'],
               'v2_output_tokens': new['generation']['usage']['output_tokens'],
               'v2_cost_estimate_cny': new['cost'].get('estimated_cny'),
               'v1_api_seconds': old['generation']['api_total_seconds'],
               'v1_input_tokens': old['generation']['usage']['input_tokens'],
               'v1_output_tokens': old['generation']['usage']['output_tokens'],
               'v1_cost_estimate_cny': old['cost'].get('estimated_cny')}
        for version in ('V1','V2'):
            score = assessment['evaluations_by_version'][version]
            for metric in ('Correctness','Completeness','Relevance','Faithfulness'):
                row[f'{version}_{metric}'] = score['scores'][metric]['score']
            for metric in ('citation_support','source_attribution','overgeneralization','abstention'):
                row[f'{version}_{metric}'] = score[metric]['class']
        rows.append(row)
    with (OUT / 'per_question_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT / 'verification_claims.json').write_text(json.dumps(claim_rows,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    aggregate = {'version':'phase49-summary-v1','questions':20,'real_v2_successes':20,
                 'automated_evaluation_successes':20, 'same_evidence_verified':True,
                 'automated':{}, 'performance':{},
                 'local_verifier':{'v1_flagged_questions':{r['question_id']:r['v1_local_issues'] for r in rows if r['v1_local_issues']},
                                   'v2_flagged_questions':{r['question_id']:r['v2_local_issues'] for r in rows if r['v2_local_issues']},
                                   'known_q15_detected':True,'known_q16_detected_as_potential_issue':True,
                                   'known_false_negatives':['Q08 overextended trial design','Q07 omitted sample denominator'],
                                   'note':'Rule flags are not semantic proof; no independent expert labels for all claims.'}}
    for version in ('V1','V2'):
        aggregate['automated'][version] = {
            'scores_mean':{m:mean(r[f'{version}_{m}'] for r in rows) for m in ('Correctness','Completeness','Relevance','Faithfulness')},
            'classes':{m:dict(Counter(r[f'{version}_{m}'] for r in rows)) for m in
                       ('citation_support','source_attribution','overgeneralization','abstention')}}
        aggregate['performance'][version] = {
            'api_completion_mean_seconds':mean(r[f'{version.lower()}_api_seconds'] for r in rows),
            'input_tokens_total':sum(r[f'{version.lower()}_input_tokens'] for r in rows),
            'output_tokens_total':sum(r[f'{version.lower()}_output_tokens'] for r in rows),
            'generation_cost_estimate_cny_total':sum(r[f'{version.lower()}_cost_estimate_cny'] for r in rows)}
    aggregate['performance']['V2'].update(
        ttft_mean_seconds=mean(r['v2_ttft_seconds'] for r in rows),
        local_verification_mean_seconds=mean(r['v2_verification_seconds'] for r in rows),
        extra_refinement_calls=sum(bool(read(OUT/f"{r['question_id']}.json")['revision']) for r in rows))
    aggregate['performance']['end_to_end_note']='Offline matched-evidence replay; no live retrieval or cold-load timing measured here.'
    aggregate['evaluator_cost_estimate_cny_total']=sum(read(OUT/f"eval_Q{n:02d}.json")['cost'].get('estimated_cny',0) for n in range(1,21))
    (OUT/'aggregate.json').write_text(json.dumps(aggregate,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(aggregate,ensure_ascii=False))

if __name__=='__main__':main()
