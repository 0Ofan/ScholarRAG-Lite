"""Aggregate the bounded Phase 4.10 audit and live acceptance evidence."""
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/phase410'
P49=ROOT/'experiments/results/phase49'

def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def stats(x):return {'n':len(x),'mean':mean(x),'median':median(x)}

def main():
    claim=read(OUT/'claim_totals.json')
    p49=read(P49/'aggregate.json')
    online=[read(OUT/f'online_{q}.json') for q in ('Q07','Q15','Q16')]
    perf={'question_ids':['Q07','Q15','Q16'],'cold_model_index_load_seconds':online[0]['index_load_seconds'],
          'measurement':'Live BGE query encoding + FAISS + same Top-3 evidence + real DeepSeek, alternating V1/V2',
          'groups':{}}
    for version in ('V1','V2'):
        items=[r['results'][version] for r in online]
        perf['groups'][version]={
            'failures':sum(x['status']!='success' for x in items),
            'query_embedding_seconds':stats([r['retrieval_timing']['query_embedding_seconds'] for r in online]),
            'faiss_retrieval_seconds':stats([r['retrieval_timing']['retrieval_seconds'] for r in online]),
            'evidence_selection_seconds':stats([r['retrieval_timing']['evidence_seconds'] for r in online]),
            'prompt_preparation_seconds':stats([x['prompt_preparation_seconds'] for x in items]),
            'api_ttft_seconds':stats([x['generation']['api_ttft_seconds'] for x in items]),
            'api_completion_seconds':stats([x['generation']['api_total_seconds'] for x in items]),
            'verification_seconds':stats([x['verification']['seconds'] if x['verification'] else 0 for x in items]),
            'refinement_seconds':stats([x['refinement_seconds'] for x in items]),
            'warm_e2e_seconds':stats([x['total_e2e_warm_seconds'] for x in items]),
            'input_tokens':sum(x['generation']['usage']['input_tokens'] for x in items),
            'output_tokens':sum(x['generation']['usage']['output_tokens'] for x in items),
            'api_cost_estimate_cny':sum(x['api_cost'].get('estimated_cny',0) for x in items),
            'extra_refinement_calls':sum(x['refinement_calls'] for x in items)}
    perf['cold_start_note']='Cold load measured once; first query embedding was also slower. The 3-query means are small-sample observations.'
    write(OUT/'online_performance.json',perf)
    roles=[]
    import csv
    with (OUT/'source_roles.csv').open(encoding='utf-8-sig',newline='') as f:roles=list(csv.DictReader(f))
    write(OUT/'source_role_summary.json',{'phase49_counts':dict(Counter(x['phase49_role'] for x in roles)),
        'unclassified_count':sum(x['phase49_role']=='Unclassified' for x in roles),
        'corrected_broad_review_rule':[(x['question_id'],x['evidence_id']) for x in roles if x['status'].startswith('Changed')],
        'note':'Rule-consistent is not semantically verified; two letters quoting a review are now Unclassified.'})
    write(OUT/'verifier_reliability.json',{
        'reference':'Independent narrow source checks plus Phase 4.8 known issues; not Phase 4.9 verifier output as truth',
        'known_positive_question_cases':['V1 Q15 control-group contradiction','V1 Q16 false absence',
                                         'V1 Q07 missing denominator','V1 Q08 trial-design overextension'],
        'detected': ['V1 Q15','V1 Q16'], 'known_missed':['V1 Q07','V1 Q08'],
        'true_positive_on_known_cases':2,'false_negative_on_known_cases':2,
        'false_positive':'unknown: complete negative reference set unavailable',
        'true_negative':'unknown: complete negative reference set unavailable',
        'precision_recall_f1':'not computed: incomplete binary reference labels and different issue types',
        'unverifiable_excluded_from_binary_metrics':True,
        'phase49_v2_alert_count':0,
        'warning':'No alert does not establish factual correctness.'})
    write(OUT/'streamlit_acceptance.json',{
        'server_started':True,'bind_address':'127.0.0.1','health_http_status':200,'server_stopped_after_test':True,
        'app_test_attempted':True,'app_test_status':'timed out without completing interactive run; interrupted',
        'v1_default_code_check':'checkbox default False in app.py',
        'v2_toggle_code_check':'checkbox controls prompt, metadata and verifier in app.py',
        'actual_browser_interaction_verified':False,
        'unverified_interactions':['V1 question submission','V2 question submission','citation/source-role rendering',
                                   'Verification Details expand/collapse','history after V1/V2 switch','dashboard navigation'],
        'note':'HTTP 200 proves service health only, not interactive UI acceptance.'})
    write(OUT/'cost_statistics.json',{'historical_phase49_20_question_generation':p49['performance'],
        'historical_phase49_evaluator_cost_estimate_cny':p49['evaluator_cost_estimate_cny_total'],
        'phase410_online_3_question_generation':{v:perf['groups'][v]['api_cost_estimate_cny'] for v in ('V1','V2')},
        'phase410_new_evaluator_calls':0,
        'note':'New online fees are separate from historical Phase 4.9 generation and evaluation; published-price estimates, not bills.'})
    decision={'default_version':'V1','v2_status':'Experimental',
              'reason':['Independent narrow automation confirms removal of known Q15/Q16 contradictions but marks most claims Unverifiable.',
                        'Verifier misses Q07/Q08 issues and lacks a complete labeled negative set for precision/F1.',
                        'Streamlit service health passed, interactive page behavior unverified.',
                        'Three-query live latency/cost are acceptable as observations but too small for a deployment guarantee.'],
              'claim_counts':claim['counts'],'paired_known_issue_change':claim['paired_issue_change'],
              'human_review_completed':False}
    write(OUT/'version_decision.json',decision)
    print(json.dumps({'decision':decision['default_version'],'online':perf['groups'],
                      'claims':claim['counts']},ensure_ascii=False))

if __name__=='__main__':main()
