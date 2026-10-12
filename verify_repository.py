"""Verify published bytes only; does not execute research experiments."""
from pathlib import Path
import hashlib, json

root = Path(__file__).resolve().parent
manifest = json.loads((root / "REPOSITORY_FILES_SHA256.json").read_text())
errors = []
for entry in manifest["files"]:
    path = root / entry["path"]
    if not path.is_file():
        errors.append(f"missing: {entry['path']}")
    elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        errors.append(f"hash mismatch: {entry['path']}")
if errors:
    raise SystemExit("\n".join(errors))
print(f"PASS: {len(manifest['files'])} published files match SHA256")
