"""Download two official 2020-07-16 CORD-19 components with local resume."""
import hashlib
import json
import os
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/raw/cord19_2020-07-16/phase26'
DEST.mkdir(parents=True,exist_ok=True)
BASE='https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-07-16/'
FILES={'metadata.csv':269_219_095,'document_parses.tar.gz':2_333_034_478}

def get(name,expected):
    target=DEST/name
    partial=DEST/(name+'.part')
    if target.exists():
        if target.stat().st_size!=expected: raise RuntimeError(f'existing file has wrong size: {target}')
        print(name,'already present',flush=True)
    else:
        offset=partial.stat().st_size if partial.exists() else 0
        if offset>expected:raise RuntimeError('partial file exceeds HEAD size')
        if offset<expected:
            headers={'User-Agent':'Mozilla/5.0 (ScholarRAG Lite research)'}
            if offset:headers['Range']=f'bytes={offset}-'
            request=urllib.request.Request(BASE+name,headers=headers)
            with urllib.request.urlopen(request,timeout=120) as response:
                if offset and response.status!=206: raise RuntimeError('server did not honor resume range')
                with partial.open('ab' if offset else 'wb') as out:
                    while data:=response.read(1024*1024):
                        out.write(data)
                        if out.tell()//(250*1024*1024)>(out.tell()-len(data))//(250*1024*1024):
                            print(name,'bytes',out.tell(),flush=True)
        if partial.stat().st_size!=expected:raise RuntimeError(f'incomplete {name}: {partial.stat().st_size}/{expected}')
        partial.replace(target)
    h=hashlib.sha256()
    with target.open('rb') as f:
        while data:=f.read(4*1024*1024):h.update(data)
    return {'path':str(target.relative_to(ROOT)),'bytes':target.stat().st_size,'sha256':h.hexdigest(),'url':BASE+name}

def main():
    result={name:get(name,size) for name,size in FILES.items()}
    (ROOT/'data/evaluation/phase26_download_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
