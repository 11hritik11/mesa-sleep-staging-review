#!/usr/bin/env python3
"""Freeze sources and submit the preparation, fitting and report dependency chain."""
import hashlib
import json
import subprocess
from datetime import datetime,timezone
from pathlib import Path
root=Path(__file__).resolve().parents[1]
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
run=root/'data/derived/bcg_full_training_runs'/stamp
run.mkdir(parents=True)
checksums={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
           for folder in ('bcg_full_training','bcg_jitter_staging') for p in sorted((root/folder).iterdir())
           if p.is_file() and p.suffix in ('.py','.json','.md','.slurm')}
(run/'source_checksums.json').write_text(json.dumps(checksums,indent=2)+'\n')
(root/'logs').mkdir(exist_ok=True)
jobs={}
def submit(name,dependencies=()):
    args=['sbatch','--parsable']
    if dependencies:args+=['--dependency=afterok:'+':'.join(dependencies)]
    args += [str(root/'bcg_full_training'/(name+'.slurm')),str(run)]
    job=subprocess.check_output(args,cwd=root,text=True).strip().split(';')[0]
    if not job.isdigit():raise ValueError('Unexpected submission result '+job)
    jobs[name]=job
    (run/'launch.json').write_text(json.dumps({'run':str(run),'jobs':jobs,'submitted_utc':stamp},indent=2)+'\n')
    return job
prep=submit('prep');trees=submit('trees',[prep]);temporal=submit('temporal',[prep]);submit('finalize',[trees,temporal])
print((run/'launch.json').read_text())
