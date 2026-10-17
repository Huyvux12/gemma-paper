"""Portable locations and integrity checks for the Gemma-only research bundle."""
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / 'vendor/stage11'))
from common import read, sha, verify, write



PRIVATE_KEYS = ('original_core_manifest_sha256', 'original_timing_manifest_sha256',
                'records_sha256', 'embedding_sha256', 'native_inputs_zip_sha256',
                'native_inputs_manifest_sha256')
_PRIVATE_PINS = None


def private_pins():
    global _PRIVATE_PINS
    if _PRIVATE_PINS is not None:return _PRIVATE_PINS
    explicit=os.environ.get('GEMMA_PRIVATE_CONFIG')
    candidates=[Path(explicit)] if explicit else [CODE_ROOT/'gemma_private_runtime.json',Path.cwd()/'gemma_private_runtime.json']
    if explicit and not Path(explicit).is_file():raise FileNotFoundError('GEMMA_PRIVATE_CONFIG does not point to a file')
    mount=Path('/kaggle/input')
    if not explicit and mount.exists():candidates+=sorted(mount.rglob('gemma_private_runtime.json'))
    paths=list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    if len(paths)>1:raise ValueError('Multiple private runtime configs; set GEMMA_PRIVATE_CONFIG explicitly')
    if not paths:return {}
    value=read(paths[0]);value=value.get('pins',value)
    if any(k not in value for k in PRIVATE_KEYS):raise ValueError('Private config lacks required data fingerprints')
    for key in PRIVATE_KEYS:
        digest=value[key]
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid private fingerprint: '+key)
    _PRIVATE_PINS={key:value[key] for key in PRIVATE_KEYS}
    return _PRIVATE_PINS


def require_private_config():
    if not private_pins():
        raise FileNotFoundError('Attach gemma_private_runtime.json as private Kaggle input, or set GEMMA_PRIVATE_CONFIG. See README_VI.md. No data fingerprints are embedded in the public notebook.')
    return private_pins()


def verify_code():
    pins = read(CODE_ROOT / 'PINNED.json')
    for name, digest in pins['vendor_sha256'].items():
        if sha(CODE_ROOT / name) != digest:
            raise ValueError('Frozen scientific source changed: ' + name)
    pins.update(private_pins())
    return pins


def find_bundle(explicit=None):
    candidates = []
    for path in [explicit, os.environ.get('GEMMA_BUNDLE_ROOT'), CODE_ROOT, Path.cwd(), Path.cwd().parent]:
        if path:
            root = Path(path).resolve()
            if (root / 'ASSET_MANIFEST_SHA256.json').is_file():
                return root
    mount = Path('/kaggle/input')
    if mount.exists():
        candidates = sorted({p.parent for p in mount.rglob('ASSET_MANIFEST_SHA256.json')})
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError('Set GEMMA_BUNDLE_ROOT to this bundle, or attach its assets as a private Kaggle dataset')


def verify_assets(root=None):
    root = find_bundle(root)
    verify_code()
    manifest = read(root / 'ASSET_MANIFEST_SHA256.json')
    for name, digest in manifest['files'].items():
        target = (root / name).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file() or sha(target) != digest:
            raise ValueError('Missing or corrupt scientific asset: ' + name)
    return root


def load_data(root=None):
    import numpy as np
    root = find_bundle(root)
    require_private_config()
    core = root / 'assets/core'
    pins = verify_code()
    if sha(core / 'records.json') != pins['records_sha256']:
        raise ValueError('Record bytes or split changed')
    records = read(core / 'records.json')
    metadata = read(core / 'gemma/embedding_metadata.json')
    path = core / 'gemma/embeddings.npz'
    if sha(path) != pins['embedding_sha256']:
        raise ValueError('Original native cache changed')
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {'image', 'text'}:
            raise ValueError('Unexpected cache arrays')
        cache = {key: z[key] for key in z.files}
    for key, names in [('image', metadata['images']), ('text', metadata['texts'])]:
        x = cache[key]
        if x.dtype != np.float32 or x.shape != (len(names), 768) or not np.isfinite(x).all():
            raise ValueError('Gemma cache shape/dtype/coverage differs')
        if np.max(np.abs(np.linalg.norm(x, axis=1)-1)) > 1e-5:
            raise ValueError('Cache is not unit-normalized')
    cache.update(image_index={n:i for i,n in enumerate(metadata['images'])},
                 text_index={n:i for i,n in enumerate(metadata['texts'])})
    return core, records, cache


def load_head(core, phase='mean', device='cpu'):
    import torch
    from learning import BoundedAdapter
    selection = read(Path(core) / 'gemma/selection.json')[phase]['selection']
    if selection['fallback_base']:
        return None
    checkpoint = torch.load(Path(core) / 'gemma' / (phase + '_seed11.pt'), map_location='cpu', weights_only=True)
    config = checkpoint['config']
    head = BoundedAdapter(checkpoint['dimension'], config['rank'], config['radius'])
    head.load_state_dict(checkpoint['state_dict'], strict=True)
    return head.to(device).eval()


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination):
                raise ValueError('Archive path escaped destination')
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Symlinks are not allowed in input archives')
        z.extractall(destination)


def native_inputs(explicit=None, extraction=None):
    require_private_config()
    pins = verify_code()
    locations = [Path(explicit)] if explicit else [CODE_ROOT / 'data', Path('/kaggle/input')]
    roots, archives = set(), set()
    for location in locations:
        if location.is_file():
            archives.add(location.resolve())
        elif location.exists():
            if (location / 'native_snapshot_receipt.json').is_file():roots.add(location.resolve())
            roots.update(p.parent.resolve() for p in location.rglob('native_snapshot_receipt.json'))
            archives.update(p.resolve() for p in location.rglob('native_inputs.zip'))
    if not roots:
        if len(archives) != 1:raise ValueError('Attach exactly one native_inputs.zip or expanded native snapshot')
        archive = next(iter(archives))
        if sha(archive) != pins['native_inputs_zip_sha256']:raise ValueError('Native archive changed')
        if extraction is None:raise ValueError('An extraction destination is required')
        safe_extract(archive, extraction)
        roots = {p.parent.resolve() for p in Path(extraction).rglob('native_snapshot_receipt.json')}
    if len(roots) != 1:raise ValueError('Ambiguous native data snapshots')
    root = next(iter(roots))
    if sha(root / 'manifest_sha256.json') != pins['native_inputs_manifest_sha256']:
        raise ValueError('Native input snapshot identity differs')
    verify(root)
    if sha(root / 'records.json') != pins['records_sha256']:raise ValueError('Native records changed')
    return root
