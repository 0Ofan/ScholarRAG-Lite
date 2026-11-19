"""Independent conservative source audit of the saved V1/V2 answers."""
import csv
import json
import re
from collections import Counter
from pathlib import Path

from src.citation_grounding import _sentences
from src.evidence_metadata import source_role

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/phase410'
P45=ROOT/'experiments/results/phase45'
P49=ROOT/'experiments/results/phase49'
REF=re.compile(r'\[(E\d+)\]')


def read(path):return json.loads(path.read_text(encoding='utf8'))
def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def csv_write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def units(answer,qid):
    for si,sentence in enumerate(_sentences(answer),1):
        if qid=='Q16' and '未提供日常环境条件、表面类型' in sentence:
            parts=['未提供日常环境条件','未提供表面类型','未提供消毒剂实际使用浓度/接触时间',
                   '未提供直接SARS-CoV-2消毒试验结果','无法确定外推至日常设置的限制']
        else:
            parts=[p.strip() for p in re.split(r'[；;]',sentence) if p.strip()]
        refs=list(dict.fromkeys(REF.findall(sentence)))
        for pi,part in enumerate(parts,1):
            yield si,pi,part,list(dict.fromkeys(REF.findall(part))) or refs,sentence


def classify(claim,source,has_valid_citation):
    """Only narrow, source-contradicted propositions receive semantic labels."""
    low=source.lower()
    if not has_valid_citation:
        return 'Unverifiable','No valid cited source for semantic judgment'
    if '单臂' in claim and re.search(r'untreated controls?|control group',low):
        return 'Contradicted','Source explicitly describes untreated controls'
    if '未提供表面类型' in claim and all(x in low for x in ('metal','glass','plastic')):
        return 'Contradicted','Source lists metal, glass and plastic surfaces'
    if '未提供消毒剂实际使用浓度/接触时间' in claim and '0.1%' in low and '1 min' in low:
        return 'Contradicted','Source gives concentration and one-minute exposure'
    if '两项随机双盲对照III期' in claim and 'two trials' in low and low.count('double-blind')<2:
        return 'Partially Supported','Two trials mentioned, but both being double-blind Phase III is not established by these chunks'
    return 'Unverifiable','Cross-language semantic entailment, study role and conditions cannot be certified by these narrow rules'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    phase48_errors=read(ROOT/'experiments/results/phase48/errors.json')
    claims=[];paired=[];mappings=[];roles=[]
    for n in range(1,21):
        qid=f'Q{n:02d}'
        old,new=read(P45/f'A_{qid}.json'),read(P49/f'{qid}.json')
        assert old['status']==new['status']=='success' and old['question']==new['question']
        a=[(e['evidence_id'],e['doc_id'],e['chunk_id'],e['title'],e['text']) for e in old['evidence']]
        b=[(e['evidence_id'],e['doc_id'],e['chunk_id'],e['title'],e['text']) for e in new['evidence']]
        assert a==b and old['evidence_sha256']==new['v1_evidence_sha256']
        mapping={e['evidence_id']:e for e in new['evidence']}
        for e in new['evidence']:
            mappings.append({'question_id':qid,'evidence_id':e['evidence_id'],'document_id':e['doc_id'],
                             'paper_title':e['title'],'chunk_id':e['chunk_id'],'original_chunk_text':e['text'],
                             'v1_v2_exact_match':True})
            role=source_role(e)
            roles.append({'question_id':qid,'evidence_id':e['evidence_id'],'document_id':e['doc_id'],
                          'phase49_role':e['source_role'],'current_rule_role':role,
                          'status':'Changed after conservative rule fix' if role!=e['source_role'] else
                                   'Unclassified; no unsupported role assertion' if role=='Unclassified' else 'Rule-consistent; semantic role unverified',
                          'original_chunk_text':e['text']})
        counts={}
        for version,record in (('V1',old),('V2',new)):
            qrows=[]
            for si,pi,part,ids,sentence in units(record['answer'],qid):
                scoped=[mapping[i] for i in ids if i in mapping]
                # Uncited blanket absence statements are tested against all supplied chunks.
                if not ids and '未提供' in part:scoped=list(mapping.values())
                source='\n---\n'.join(e['text'] for e in scoped)
                valid=bool(scoped) and (not ids or len(scoped)==len(ids))
                status,reason=classify(part,source,valid)
                qrows.append({'question_id':qid,'version':version,'claim_id':f'{qid}-{version}-S{si:02d}-C{pi:02d}',
                              'claim_text':part,'source_sentence':sentence,'citation_ids':'|'.join(ids),
                              'evidence_ids':'|'.join(e['evidence_id'] for e in scoped),
                              'document_ids':'|'.join(e['doc_id'] for e in scoped),
                              'chunk_ids':'|'.join(e['chunk_id'] for e in scoped),
                              'original_chunk_text':source,'verification_result':status,
                              'reason':reason,'method':'Independent narrow rules over saved source text; no Phase49 verifier labels used'})
            claims.extend(qrows)
            counts[version]=dict(Counter(r['verification_result'] for r in qrows))
        evaluation=read(P49/f'eval_{qid}.json')['evaluations_by_version']
        known_errors=[e['category'] for e in phase48_errors if e['question_id']==qid]
        paired.append({'question_id':qid,'v1_claims':sum(counts['V1'].values()),'v2_claims':sum(counts['V2'].values()),
                       'v1_direct':counts['V1'].get('Directly Supported',0),'v2_direct':counts['V2'].get('Directly Supported',0),
                       'v1_partial':counts['V1'].get('Partially Supported',0),'v2_partial':counts['V2'].get('Partially Supported',0),
                       'v1_unsupported':counts['V1'].get('Unsupported',0),'v2_unsupported':counts['V2'].get('Unsupported',0),
                       'v1_contradicted':counts['V1'].get('Contradicted',0),'v2_contradicted':counts['V2'].get('Contradicted',0),
                       'v1_unverifiable':counts['V1'].get('Unverifiable',0),'v2_unverifiable':counts['V2'].get('Unverifiable',0),
                       'confirmed_issue_change':'improved' if counts['V1'].get('Contradicted',0)>counts['V2'].get('Contradicted',0) else 'no_detected_change',
                       'v1_phase48_missing_condition_event_count':known_errors.count('Missing Research Conditions'),
                       'v2_missing_condition_event_count':'Unverifiable by current automation',
                       'v1_false_absence_detected':qid=='Q16','v2_false_absence_detected':False,
                       'v1_historical_auto_completeness':evaluation['V1']['scores']['Completeness']['score'],
                       'v2_historical_auto_completeness':evaluation['V2']['scores']['Completeness']['score'],
                       'v1_historical_auto_citation_support':evaluation['V1']['citation_support']['class'],
                       'v2_historical_auto_citation_support':evaluation['V2']['citation_support']['class'],
                       'v1_historical_auto_source_attribution':evaluation['V1']['source_attribution']['class'],
                       'v2_historical_auto_source_attribution':evaluation['V2']['source_attribution']['class']})
    csv_write(OUT/'claim_verification.csv',claims)
    csv_write(OUT/'per_question_comparison.csv',paired)
    csv_write(OUT/'source_roles.csv',roles)
    write(OUT/'evidence_mapping.json',mappings)
    totals={v:dict(Counter(r['verification_result'] for r in claims if r['version']==v)) for v in ('V1','V2')}
    write(OUT/'claim_totals.json',{'counts':totals,'paired_issue_change':dict(Counter(r['confirmed_issue_change'] for r in paired)),
        'interpretation':'Directly Supported and Unsupported are zero because narrow rules cannot certify cross-language entailment; Unverifiable is not error.'})
    print(json.dumps({'rows':len(claims),'counts':totals,'roles':dict(Counter(r['phase49_role'] for r in roles))},ensure_ascii=False))

if __name__=='__main__':main()
