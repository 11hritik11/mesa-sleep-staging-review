"""Prove each readable copy has the same Python syntax tree as the archive."""

import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
records = json.loads((root / "readable/equivalence.json").read_text())
for name, record in records.items():
    original = root / "source" / name
    readable = root / "readable" / name
    for path, field in ((original, "original_sha256"), (readable, "readable_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != record[field]:
            raise ValueError("Changed readable/original file: " + str(path))
    if ast.dump(ast.parse(original.read_text())) != ast.dump(
        ast.parse(readable.read_text())
    ):
        raise ValueError("Changed Python syntax tree: " + name)
print(f"PASS: {len(records)} readable files have identical Python syntax trees")
