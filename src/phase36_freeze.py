"""Freeze the reviewed question set and its evidence-based expectations once."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'experiments/configs/phase36_questions.json'
lock = ROOT / 'experiments/configs/phase36_question_lock.json'
if lock.exists():
    raise SystemExit('Question set already frozen; refusing to change it.')
data = json.loads(path.read_text(encoding='utf8'))
preview = json.loads((ROOT / 'experiments/results/phase36/evidence_preview.json').read_text(encoding='utf8'))
assert len(data['questions']) == len(preview['items']) == 20

# Points are deliberately limited to what can be checked against the July 2020 corpus.
# A partial evidence label is retained even when other corpus papers could answer more.
annotations = {
 'Q01': (['Review covers coronavirus survival on inanimate surfaces and inactivation by biocides.', 'Study uses published suspension and carrier experiments; distinguish laboratory findings from actual transmission risk.', 'Do not claim a single universal survival time or disinfectant efficacy.'], 'partial', 'Top chunk of the named review is methodological, so exact durations may be unavailable.'),
 'Q02': (['The named study evaluates a rapid lateral-flow IgM/IgG antibody assay, with a stated turnaround under 15 minutes.', 'It measures antibodies rather than viral RNA.', 'Interpretation depends on antibody development and study-population validation; do not infer it rules out early infection.'], 'adequate', 'The target study is retrieved first; detailed diagnostic accuracy requires checking its reported numbers.'),
 'Q03': (['CoDEX is a prospective randomized multicenter open-label controlled superiority trial.', 'The design targeted moderate or severe COVID-19-associated ARDS in intensive care.', 'A study-design paper alone does not establish dexamethasone efficacy.'], 'adequate', 'Design and rationale are available; outcome claims would need a results paper.'),
 'Q04': (['Study compares CT imaging and clinical manifestations of early COVID-19 versus influenza pneumonia.', 'It reports significant differences among measured imaging and clinical features.', 'Features alone do not establish definitive diagnosis; design/sample limits should be stated when supported.'], 'partial', 'Retrieved best chunk summarizes feature counts but omits several individual feature details.'),
 'Q05': (['Distinguish asymptomatic from presymptomatic transmission.', 'Evidence includes observational reports and modeling assumptions about symptom-free infectiousness.', 'Quantitative transmission share remains uncertain in these three snippets.'], 'adequate', 'Three sources cover workplace transmission, modeling, and presymptomatic case investigation.'),
 'Q06': (['Severe disease has heterogeneous immune perturbations.', 'Some studies report altered T-cell/cytotoxic responses or innate/adaptive immune patterns.', 'Observational biomarker evidence does not by itself prove causality.'], 'adequate', 'The first retrieved snippet is mostly methodological; sources two and three are more directly relevant.'),
 'Q07': (['A systematic review reports generally mild or asymptomatic pediatric cases in early evidence.', 'Severe respiratory failure can occur in young children, including case reports.', 'Small case series and selection bias limit generalization.'], 'adequate', 'One of three retrieved papers is about adult care in a children’s hospital; avoid treating it as pediatric outcome evidence.'),
 'Q08': (['Describe the reported remdesivir benefits or null/uncertain findings as study-specific rather than settled fact.', 'Distinguish trial evidence from pooled reviews and benefit-risk assessments.', 'Acknowledge small/heterogeneous early evidence and remaining uncertainty.'], 'adequate', 'Retrieved snippets include a systematic review and assessment of trials; avoid a definitive clinical recommendation.'),
 'Q09': (['Hypertension and diabetes are reported as associated with greater severity in observational or pooled evidence.', 'Association is not causation.', 'Age, comorbidities, sampling, and adjustment may confound a simple causal claim.'], 'adequate', 'Top result is population-burden context, while the second directly reports associations.'),
 'Q10': (['Spike receptor-binding domain engages human ACE2.', 'Structural modeling/crystallography identifies interaction residues or contact patterns.', 'Proposed drug-target implications are hypotheses rather than proven treatments.'], 'adequate', 'One direct structural analysis and two review/engineering snippets.'),
 'Q11': (['RT-PCR detects SARS-CoV-2 RNA and is used for active-infection diagnosis.', 'Antibody assays detect host immune response, which has a different timing and purpose.', 'The retrieved Top-3 snippets offer little direct antibody evidence, so a detailed assay comparison should be qualified.'], 'partial', 'All three Top-3 snippets emphasize RT-PCR or imaging; this is a deliberate evidence-selection stress case.'),
 'Q12': (['Early COVID-19 and influenza pneumonia can overlap clinically and radiologically.', 'A comparative study assessed CT and clinical features and found some significant differences.', 'CT alone should not be portrayed as definitive because of overlap and study limitations.'], 'adequate', 'Comparative study and narrative review are retrieved.'),
 'Q13': (['Retrieved sources discuss masks for respiratory-virus prevention.', 'Physical-distancing evidence is not substantively present in the Top-3 snippets.', 'Do not infer a numerical or causal superiority of one intervention from this evidence.'], 'partial', 'Deliberate single-sided retrieval for a comparison question.'),
 'Q14': (['Severe versus mild cases show immune or coagulation differences in reported studies.', 'Cytotoxic T-cell activation is one investigated difference.', 'Different markers and study designs should not be conflated into a universal mechanism.'], 'partial', 'One retrieved study is coagulation rather than immune-cell profiling.'),
 'Q15': (['Early hydroxychloroquine evidence includes controlled trials and retrospective or review evidence.', 'Open-label, small-sample, surrogate endpoints or observational designs limit efficacy inference.', 'Do not state proven clinical benefit from weak evidence.'], 'adequate', 'Top results include a trial design/title and review of published studies.'),
 'Q16': (['Persistence/inactivation studies use laboratory suspension or carrier tests and surrogate coronaviruses.', 'Disinfectant effectiveness depends on agent, conditions and contact time.', 'Laboratory surface findings do not directly measure everyday infection risk.'], 'adequate', 'Review, dentistry context and surrogate-virus experiment are retrieved.'),
 'Q17': (['Serology evaluations compare assays using known positive and pre-pandemic negative samples.', 'Sensitivity varies with days since symptom onset and assay.', 'False negatives/positives and sample selection limit individual interpretation.'], 'adequate', 'Evaluation and comparative assay snippets include sensitivity/specificity methods.'),
 'Q18': (['Models estimate quarantine effects from case-growth or compartment/population dynamics.', 'Estimates depend on case ascertainment, parameter assumptions and behavior.', 'Model projections are not direct causal measurements of quarantine.'], 'adequate', 'Three retrieved studies discuss quarantine and model limitations.'),
 'Q19': (['Exact duration of protective post-infection immunity cannot be established from papers available by July 2020.', 'Antibody or immune response is not identical to proven durable protection.', 'Explicitly abstain from a precise duration.'], 'insufficient', 'One source explicitly says duration and correlates of protection are uncertain.'),
 'Q20': (['Five-year safety and efficacy cannot be established by July 2020 research.', 'The retrieved papers discuss early vaccine development or Phase I testing.', 'Explicitly refuse to claim long-term clinical outcomes.'], 'insufficient', 'Temporal impossibility is intentional; early-stage evidence is available.'),
}
manifest = json.loads((ROOT / 'data/processed/phase33_large_manifest.json').read_text(encoding='utf8'))
exclude_as_question_evidence = {'Q07': {'lyqbquxf'}, 'Q19': {'l29ooa1h'}}
for item, found in zip(data['questions'], preview['items']):
    assert item['question_id'] == found['question_id']
    points, availability, notes = annotations[item['question_id']]
    item['expected_answer_points'] = points
    item['relevant_document_ids'] = [e['doc_id'] for e in found['evidence']
                                     if e['doc_id'] not in exclude_as_question_evidence.get(item['question_id'], set())]
    item['source_topic_qrels_for_retrieved_ids'] = {
        str(topic): {e['doc_id']: manifest['effective_qrels'][str(topic)].get(e['doc_id'])
                     for e in found['evidence']}
        for topic in item['source_topic_ids']
    }
    item['evidence_availability'] = availability
    item['evaluation_notes'] = notes
data['status'] = 'frozen before DeepSeek Phase 3.6 calls'
data['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
body = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
path.write_text(body, encoding='utf8')
lock.write_text(json.dumps({'version': 'phase36-lock-v1', 'sha256': hashlib.sha256(body.encode('utf8')).hexdigest(),
                            'path': 'experiments/configs/phase36_questions.json'}, indent=2) + '\n', encoding='utf8')
print('Frozen', len(data['questions']), 'questions; SHA256', hashlib.sha256(body.encode('utf8')).hexdigest())
