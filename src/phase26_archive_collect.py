"""Scan complete CORD-19 TAR once; extract only qrel-judged parse files."""
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'data/evaluation'
ARCHIVE=ROOT/'data/raw/cord19_2020-07-16/phase26/document_parses.tar.gz'
DEST=ROOT/'data/raw/cord19_2020-07-16/phase26/parses'
DEST.mkdir(parents=True,exist_ok=True)
targets=json.loads((EVAL/'phase26_judged_path_index.json').read_text(encoding='utf8'))
status={};seen=set();member_count=0;error_count=0;total_extracted=0
with tarfile.open(ARCHIVE,mode='r|gz') as tar:
    for member in tar:
        if not member.isfile():continue
        member_count+=1
        path=member.name.removeprefix('./')
        if path not in targets:continue
        seen.add(path)
        record={'archive_member':member.name,'member_size':member.size,'cord_uids':targets[path]}
        if member.size>30_000_000:
            record['status']='oversize';status[path]=record;continue
        try:
            blob=tar.extractfile(member).read()
            if len(blob)!=member.size:raise IOError('short member read')
            obj=json.loads(blob)
            body=obj.get('body_text') or []
            texts=[part.get('text','') for part in body if isinstance(part,dict) and part.get('text')]
            char_count=sum(len(x) for x in texts)
            record.update({'paper_id':obj.get('paper_id'),'body_chars':char_count,'body_entries':len(texts),
                           'abstract_entries':len(obj.get('abstract') or []),'json_title':(obj.get('metadata') or {}).get('title'),
                           'sha256':hashlib.sha256(blob).hexdigest()})
            if not texts:record['status']='empty_body'
            elif char_count<1000:record['status']='short_body'
            else:record['status']='valid_body'
            # Preserve the exact original JSON; use metadata path under a new phase26 root.
            target=DEST/path
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest()!=record['sha256']:
                    raise RuntimeError(f'preexisting raw file differs: {target}')
            else:target.write_bytes(blob)
            total_extracted+=len(blob)
        except (ValueError,UnicodeError,IOError,RuntimeError) as exc:
            record['status']='read_or_parse_error';record['error']=str(exc);error_count+=1
        status[path]=record
        if len(seen)%5000==0:print('matched_paths',len(seen),'archive_members',member_count,flush=True)
for path in targets.keys()-seen:status[path]={'status':'missing_archive_member','cord_uids':targets[path]}
counts=Counter(r['status'] for r in status.values())
summary={'archive_file_bytes':ARCHIVE.stat().st_size,'archive_members_scanned':member_count,
         'target_paths':len(targets),'found_target_paths':len(seen),'missing_target_paths':len(targets)-len(seen),
         'extracted_bytes':total_extracted,'status_counts':dict(counts),'read_or_parse_errors':error_count}
(EVAL/'phase26_archive_paths.json').write_text(json.dumps(status,ensure_ascii=False),encoding='utf8')
(EVAL/'phase26_archive_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps(summary,indent=2),flush=True)
