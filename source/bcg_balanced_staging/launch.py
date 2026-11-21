#!/usr/bin/env python3
"""Submit only after the synthetic/all-family and real-extraction checks pass."""
import hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
from balanced_utils import ROOT,read_json,fingerprint
proof=read_json(ROOT/'data/derived/bcg_balanced_smoke/PREFLIGHT_PASS.json')
if proof['status']!='PASS' or proof['source']!=fingerprint():raise ValueError('Preflight checks must pass with the current scientific source')
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');run=ROOT/'data/derived/bcg_balanced_runs'/stamp;run.mkdir(parents=True)
checksums={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
 for folder in ('bcg_balanced_staging','bcg_full_training','bcg_jitter_staging') for p in sorted((ROOT/folder).iterdir())
 if p.is_file() and p.suffix in ('.py','.json','.md','.slurm')}
(run/'source_checksums.json').write_text(json.dumps(checksums,indent=2)+'\n');(ROOT/'logs').mkdir(exist_ok=True);jobs={}
def submit(name,dependencies=()):
 args=['sbatch','--parsable']
 if dependencies:args+=['--dependency=afterok:'+':'.join(dependencies)]
 args+=[str(ROOT/'bcg_balanced_staging'/(name+'.slurm')),str(run)]
 job=subprocess.check_output(args,cwd=ROOT,text=True).strip().split(';')[0]
 if not job.isdigit():raise ValueError('Unexpected submission result')
 jobs[name]=job
 (run/'launch.json').write_text(json.dumps({'run':str(run),'jobs':jobs,'submitted_utc':stamp,'scientific_source':fingerprint(),'preflight':proof},indent=2)+'\n')
 return job
prep=submit('prep');classic=submit('classic',[prep]);deep=submit('deep',[prep]);controls=submit('controls',[prep]);submit('finalize',[classic,deep,controls])
print((run/'launch.json').read_text())
