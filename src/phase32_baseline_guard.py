"""Record and verify that frozen Phase 3.1 artifacts remain byte-identical."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/phase32_improved';OUT.mkdir(parents=True,exist_ok=True)
PATH=OUT/'phase31_baseline_hashes.json'
files=[]
for pattern in ('data/processed/phase31_*','data/indexes/phase31_pilot/*','experiments/results/phase31_pilot/*'):
    files.extend(p for p in ROOT.glob(pattern) if p.is_file())
for name in ('phase31_baseline_implementation_report.md','phase31_pilot_evaluation_report.md',
             'phase31_performance_report.md','phase_3_1_summary.md'):
    files.append(ROOT/name)
files=sorted(set(files))
def inventory():
    result={}
    for p in files:
        with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        result[p.relative_to(ROOT).as_posix()]={'bytes':p.stat().st_size,'sha256':digest}
    return result
parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('record','verify'))
args=parser.parse_args()
if args.mode=='record':
    assert not PATH.exists(),'baseline guard already recorded'
    data={'version':'phase32-baseline-guard-v1','files':inventory()}
    PATH.write_text(json.dumps(data,indent=2),encoding='utf8')
    print(f"recorded {len(data['files'])} frozen baseline files")
else:
    expected=json.loads(PATH.read_text(encoding='utf8'))['files']
    current=inventory()
    assert current==expected,{'missing':sorted(set(expected)-set(current)),
                              'new':sorted(set(current)-set(expected)),
                              'changed':[x for x in expected.keys()&current.keys() if expected[x]!=current[x]]}
    print(f'verified {len(expected)} frozen baseline files unchanged')
