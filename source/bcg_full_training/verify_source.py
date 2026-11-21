"""Refuse scheduled work if sources changed after submission."""
import hashlib
import json
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for name,expected in json.loads(Path(sys.argv[1]).read_text()).items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
        raise SystemExit('Scheduled source changed: '+name)
print('Frozen source checks passed',flush=True)
