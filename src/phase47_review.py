"""Traceable, conservative source-text review of selected frozen answer points."""
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/phase47'

# Each tuple is (status, evidence IDs). Direct/Partial apply only to the cited
# original chunks; all other semantic points remain Unverified, not negative.
U=('Unverified',[])
REVIEW={
 'Q01':[
  {'A':('Direct',['E1']),'B':('Direct',['E1','E2']),'C':('Direct',['E1','E2'])},
  {'A':('Partial',['E1']),'B':('Partial',['E1']),'C':('Partial',['E1'])},
  {'A':U,'B':U,'C':U}],
 'Q02':[
  {'A':('Direct',['E1']),'B':('Direct',['E1']),'C':('Direct',['E1'])},
  {'A':('Partial',['E1','E2']),'B':('Partial',['E1']),'C':('Partial',['E1'])},
  {'A':('Partial',['E3']),'B':('Partial',['E3']),'C':('Partial',['E2'])}],
 'Q03':[
  {'A':('Partial',['E1']),'B':('Partial',['E1','E2']),'C':('Partial',['E1','E2'])},
  {'A':U,'B':U,'C':U},
  {'A':('Partial',['E1']),'B':('Direct',['E3']),'C':('Direct',['E3'])}],
 'Q04':[
  {'A':('Direct',['E1']),'B':('Direct',['E1','E2']),'C':('Direct',['E1','E3'])},
  {'A':('Direct',['E1']),'B':('Direct',['E1']),'C':('Direct',['E1'])},
  {'A':('Partial',['E2']),'B':('Direct',['E3']),'C':('Direct',['E2'])}],
 'Q05':[
  {'A':('Direct',['E2','E3']),'B':('Direct',['E2','E3']),'C':('Direct',['E2'])},
  {'A':('Direct',['E1','E2','E3']),'B':('Direct',['E1','E2','E3']),'C':('Partial',['E1','E2','E3'])},
  {'A':('Direct',['E3']),'B':('Direct',['E3']),'C':U}],
 'Q06':[
  {'A':('Direct',['E3']),'B':('Direct',['E3']),'C':('Direct',['E3'])},
  {'A':('Direct',['E2','E3']),'B':('Direct',['E2','E3']),'C':('Direct',['E2','E3'])},
  {'A':U,'B':U,'C':U}],
 'Q07':[
  {'A':('Direct',['E1']),'B':('Direct',['E1']),'C':('Direct',['E1'])},
  {'A':('Direct',['E3']),'B':('Partial',['E1']),'C':('Partial',['E1'])},
  {'A':('Partial',['E1']),'B':('Direct',['E2']),'C':('Partial',['E1'])}],
 'Q08':[
  {'A':('Partial',['E1','E3']),'B':('Partial',['E1','E3']),'C':('Partial',['E1','E2'])},
  {'A':('Direct',['E1','E2','E3']),'B':('Direct',['E1','E2','E3']),'C':('Direct',['E1','E2','E3'])},
  {'A':('Partial',['E1','E2']),'B':('Partial',['E1','E2']),'C':('Partial',['E1','E3'])}],
 'Q09':[
  {'A':('Direct',['E2','E3']),'B':('Direct',['E2','E3']),'C':('Direct',['E2','E3'])},
  {'A':U,'B':U,'C':U},
  {'A':('Partial',['E3']),'B':('Partial',['E3']),'C':('Partial',['E3'])}],
 'Q10':[
  {'A':('Direct',['E1','E2']),'B':('Direct',['E1','E2']),'C':('Direct',['E1','E2'])},
  {'A':('Direct',['E2']),'B':('Direct',['E2']),'C':('Direct',['E2'])},
  {'A':('Direct',['E1']),'B':('Direct',['E1']),'C':('Direct',['E1'])}],
 'Q11':[
  {'A':('Direct',['E2','E3']),'B':('Direct',['E2','E3']),'C':('Direct',['E3'])},
  {'A':('Missing',[]),'B':('Missing',[]),'C':('Partial',['E2'])},
  {'A':U,'B':U,'C':U}],
 'Q12':[
  {'A':('Partial',['E2']),'B':U,'C':U},
  {'A':('Partial',['E1','E2']),'B':('Direct',['E3']),'C':('Direct',['E3'])},
  {'A':('Partial',['E2']),'B':U,'C':U}],
 'Q13':[
  {'A':('Direct',['E1','E2']),'B':('Direct',['E1','E2']),'C':('Direct',['E1','E2'])},
  {'A':U,'B':U,'C':U},
  {'A':U,'B':U,'C':U}],
 'Q14':[
  {'A':('Direct',['E1','E2']),'B':('Direct',['E1','E2']),'C':('Direct',['E1','E2'])},
  {'A':('Direct',['E3']),'B':('Direct',['E3']),'C':('Missing',[])},
  {'A':U,'B':U,'C':U}],
 'Q15':[
  {'A':('Direct',['E1','E2']),'B':('Direct',['E1','E2']),'C':('Direct',['E1','E2','E3'])},
  {'A':('Direct',['E1','E2']),'B':('Direct',['E1','E2']),'C':('Direct',['E1','E2','E3'])},
  {'A':('Direct',['E1']),'B':('Direct',['E1']),'C':('Direct',['E1','E3'])}],
 'Q16':[
  {'A':('Partial',['E1','E3']),'B':('Partial',['E1','E3']),'C':('Partial',['E1','E2'])},
  {'A':('Partial',['E2','E3']),'B':('Partial',['E2','E3']),'C':('Partial',['E2','E3'])},
  {'A':U,'B':U,'C':U}],
 'Q17':[
  {'A':('Direct',['E3']),'B':('Direct',['E3']),'C':('Direct',['E3'])},
  {'A':('Direct',['E1','E3']),'B':('Direct',['E1','E3']),'C':('Direct',['E1','E3'])},
  {'A':('Direct',['E2','E3']),'B':('Direct',['E2','E3']),'C':('Direct',['E2','E3'])}],
 'Q18':[
  {'A':('Direct',['E2','E3']),'B':('Direct',['E2','E3']),'C':('Direct',['E2','E3'])},
  {'A':('Partial',['E2','E3']),'B':('Partial',['E2','E3']),'C':('Partial',['E2','E3'])},
  {'A':U,'B':U,'C':U}],
 'Q19':[
  {'A':('Direct',['E1','E2']),'B':('Direct',['E1','E2']),'C':('Direct',['E1','E3'])},
  {'A':('Direct',['E1']),'B':('Direct',['E1']),'C':('Direct',['E1'])},
  {'A':U,'B':U,'C':U}],
 'Q20':[
  {'A':('Partial',['E1','E3']),'B':('Partial',['E1','E3']),'C':('Partial',['E1','E3'])},
  {'A':('Direct',['E1','E3']),'B':('Direct',['E1','E3']),'C':('Direct',['E1','E3'])},
  {'A':U,'B':U,'C':U}],
}

NOTES={
 'Q01':'E1 states suspension/carrier review methods; E2 in B/C states the review aim. None directly quantifies everyday transmission risk.',
 'Q02':'E1 is the named rapid IgM/IgG study; A/C retain a separate waiting-period caveat, while B adds the named study population (397 positive, 128 negative samples).',
 'Q03':'The selected CoDEX title/protocol establishes trial intent; B/C E3 states results would be published after follow-up, not an efficacy finding.',
 'Q04':'B/C use three chunks of the named comparative study, including its own limited-diagnostic-efficacy and initial-CT limitations. A E2 describes a different review, not the named study.',
 'Q05':'A/B include a presymptomatic cluster report and a modeling assumption. C replaces the observed cluster with another modeling-context chunk.',
 'Q06':'The severe-disease immune and cytotoxic findings are identical across A/B/C; causal interpretation is not established by these snippets.',
 'Q07':'A E3 is a neonatal respiratory-failure case. B/C omit it; B adds review ascertainment bias, while C adds a financial-burden paper unrelated to pediatric clinical outcomes.',
 'Q08':'The same three chunks appear in all groups; C changes order only. Pooled benefit and early-trial context require study-specific attribution.',
 'Q09':'All groups retain the meta-analysis association; C replaces a diabetes-context review with another comorbidity discussion. Causality/confounding remains only partly described.',
 'Q10':'All groups retain the direct structural analysis; C substitutes a broader spike/ACE2 context paper for another review.',
 'Q11':'A/B E2 directly names SARS-CoV-2 RNA. C E2 adds IgM/IgG immune-status context but does not establish a complete timing comparison.',
 'Q12':'B/C add the target comparative study result but remove the narrative review discussing diagnostic overlap. CT-alone limitations remain uncertain in the selected excerpts.',
 'Q13':'All groups include mask discussion but no substantive distancing comparison. C E3 is a planned Hajj trial analysis, not observed effectiveness results.',
 'Q14':'A/B retain the cytotoxic T-cell study; C substitutes a methotrexate/immune-stabilization discussion and loses the specific cytotoxic evidence.',
 'Q15':'All groups retain trial/review context; C adds a retrospective rheumatic-disease study explicitly cautioning about hydroxychloroquine exposure.',
 'Q16':'B/C include target-review surface-inactivation findings; A retains a surrogate coronavirus experiment. None directly measures everyday infection risk.',
 'Q17':'The same serology evaluation chunks remain in all groups, including pre-pandemic controls and days-since-onset sensitivity.',
 'Q18':'The same growth-rate model remains; C adds stochastic-model limitation context in place of a population-balance excerpt.',
 'Q19':'All groups retain uncertainty about protection duration; C changes evidence order only.',
 'Q20':'All groups preserve the same early mRNA-1273 45-volunteer trial evidence; no selected chunk provides five-year outcomes.'}


def main():
    data=json.loads((OUT/'evidence_comparison.json').read_text(encoding='utf8'))
    rows=[];counts={g:Counter() for g in 'ABC'}
    for question in data['questions']:
        qid=question['question_id']
        if qid in REVIEW and len(REVIEW[qid])!=len(question['expected_answer_points']):
            raise ValueError(f'Review point count differs: {qid}')
        for index,point in enumerate(question['expected_answer_points'],1):
            checks=REVIEW.get(qid,[{g:U for g in 'ABC'}]*len(question['expected_answer_points']))[index-1]
            item={'question_id':qid,'point_index':index,'expected_answer_point':point,
                  'source_text_review_note':NOTES.get(qid,'Semantic source review not completed; no coverage credit.'),
                  'groups':{}}
            for group in 'ABC':
                status,ids=checks[group]
                evidence={e['evidence_id']:e for e in question['evidence'][group]}
                if any(eid not in evidence for eid in ids):
                    raise ValueError(f'{qid}/{group}: review cites missing Evidence ID')
                if status in ('Direct','Partial') and not ids:
                    raise ValueError(f'{qid}/{group}: support must cite original chunk')
                item['groups'][group]={'status':status,'evidence_ids':ids,
                    'chunk_ids':[evidence[eid]['chunk_id'] for eid in ids],
                    'doc_ids':[evidence[eid]['doc_id'] for eid in ids]}
                counts[group][status]+=1
            rows.append(item)
    status={'version':'phase47-source-review-v1',
        'review_type':'Codex source-text inspection, not independent human evaluation',
        'rule':'Direct means the whole point is stated in cited chunks; Partial means only some components are stated; Missing is confirmed absent from selected excerpts; Unverified receives no coverage credit.',
        'reviewed_question_ids':sorted(REVIEW),'unreviewed_question_ids':[
            q['question_id'] for q in data['questions'] if q['question_id'] not in REVIEW],
        'point_results':rows}
    (OUT/'source_review.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    gate={'clear_coverage_improvement':False,
          'critical_Q01_Q11_Q20_evidence_preserved':True,
          'no_obviously_off_topic_or_nonresult_evidence':False,
          'citation_mapping_valid':True,
          'three_chunk_budget':True,
          'run_real_api':False}
    stats={'version':'phase47-coverage-v1','question_count':20,
        'expected_points_total':len(rows),'source_reviewed_questions':len(REVIEW),
        'counts_by_group':{g:dict(counts[g]) for g in 'ABC'},
        'semantic_coverage_claim':'Partial source-text review only; Unverified points are excluded from coverage success.',
        'key_positive_examples':['Q04 B/C target-study design and limitations',
                                 'Q11 C IgM/IgG immune-status context',
                                 'Q16 B/C target-review inactivation findings'],
        'key_negative_examples':['Q07 B/C omit the neonatal respiratory-failure case',
                                 'Q07 C includes a pediatric financial-burden paper',
                                 'Q13 C includes trial analysis plan rather than results'],
        'real_api_gate':gate,
        'gate_reason':'Source review shows useful gains and consequential losses; coverage improvement across all 20 questions is not established, and C includes off-topic/nonresult evidence. No DeepSeek requests.'}
    (OUT/'coverage_stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Reviewed',len(REVIEW),'questions;',len(rows),'point comparisons; real API gate',gate['run_real_api'])


if __name__=='__main__':main()
