from pathlib import Path
import hashlib, json
root = Path(__file__).resolve().parent
items = json.loads((root / 'BUNDLE_SHA256.json').read_text())
for name, meta in items.items():
    p = root / name
    assert p.is_file(), name
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    assert p.stat().st_size == meta['bytes'] and h.hexdigest() == meta['sha256'], name
print(json.dumps({'files_checked':len(items), 'passed':True}))
