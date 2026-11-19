"""Evidence-backed identity decisions for the four shared archive paths."""
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'data/evaluation'
load=lambda name:json.loads((EVAL/name).read_text(encoding='utf8'))
paths=load('phase26_judged_path_index.json')
records=load('phase26_judged_records.json')
status=load('phase26_doc_status.json')
archive=load('phase26_archive_paths.json')
qrels=defaultdict(dict)
for line in (EVAL/'trec_covid_complete_qrels.txt').read_text().splitlines():
    q,it,uid,grade=line.split()
    if int(grade)>=0:qrels[uid][q]=int(grade)

decisions={
 'dtnuet6c':('j7t9nebs','alias','same selected JSON SHA-256 and title; canonical UID has DOI'),
 'j7t9nebs':('j7t9nebs','canonical','same selected JSON SHA-256 and title; canonical UID has DOI'),
 '6rpt47gm':('svu06v9c','alias','same selected JSON SHA-256 and title; canonical UID has DOI'),
 'svu06v9c':('svu06v9c','canonical','same selected JSON SHA-256 and title; canonical UID has DOI'),
 'g7qioapl':(None,'excluded_conflicting_versions','same title and shared PDF, but selected content differs and topic 1 grades conflict (1 vs 2)'),
 'xa6kwguo':(None,'excluded_conflicting_versions','same title and shared PDF, but selected content differs and topic 1 grades conflict (1 vs 2)'),
 '8avkjc84':('8avkjc84','distinct','different DOI/title and valid selected PMC parse; shared secondary PDF belongs to another paper'),
 'd9v5xtx7':('d9v5xtx7','distinct','different DOI/title and valid selected PDF parse; shared secondary PDF belongs to this paper'),
}
groups=[];uids={}
for path,ids in sorted(paths.items()):
    if len(ids)<=1:continue
    members=[]
    for uid in ids:
        canonical,decision,reason=decisions[uid]
        rec=records[uid];chosen=status[uid]['source_path'];selected=archive[chosen]
        item={'raw_uid':uid,'canonical_doc_id':canonical,'decision':decision,'reason':reason,
              'doi':rec['doi'],'titles':rec['title'],'pmc_paths':rec['pmc'],'pdf_paths':rec['pdf'],
              'selected_path':chosen,'selected_sha256':status[uid]['sha256'],
              'selected_paper_id':selected.get('paper_id'),'selected_json_title':selected.get('json_title'),
              'shared_path':path,'shared_path_sha256':archive[path].get('sha256'),
              'shared_path_paper_id':archive[path].get('paper_id'),'official_qrels':qrels[uid]}
        uids[uid]=item;members.append(item)
    conflicts=[]
    for topic in sorted({q for member in members for q in member['official_qrels']},key=int):
        grades={member['raw_uid']:member['official_qrels'][topic] for member in members if topic in member['official_qrels']}
        if len(set(grades.values()))>1:conflicts.append({'topic_id':topic,'grades':grades})
    groups.append({'shared_path':path,'shared_json_title':archive[path].get('json_title'),
                   'shared_paper_id':archive[path].get('paper_id'),'members':ids,
                   'classification':('same_paper_duplicate' if set(ids) in ({'dtnuet6c','j7t9nebs'},{'6rpt47gm','svu06v9c'})
                                     else 'same_work_different_versions_conflict' if set(ids)=={'g7qioapl','xa6kwguo'}
                                     else 'different_papers_bad_secondary_path'),
                   'label_conflicts':conflicts})
out={'version':'phase27-identity-v1','source_snapshot':'CORD-19 2020-07-16','qrels':'TREC-COVID Complete',
     'decision_rule':'exact selected SHA+title duplicates merge to DOI UID if labels compatible; conflicting versions excluded; distinct DOI/title/selected parse retained',
     'groups':groups,'uids':uids}
(EVAL/'phase27_document_aliases.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'groups':[(x['classification'],x['members'],x['label_conflicts']) for x in groups],
                  'aliases':sum(x['decision']=='alias' for x in uids.values()),
                  'excluded':sum(x['decision'].startswith('excluded') for x in uids.values())},ensure_ascii=False,indent=2))
