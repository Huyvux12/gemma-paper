"""Locate and validate the private FULL inputs without importing ML libraries."""
import json
import shutil
import stat
import zipfile
from pathlib import Path

FULL_REL = 'full_recovered/embeddinggemma2_full_validation/full_validation'
SUGAR_KINDS = ('gemma', 'clip', 'clip_large', 'siglip', 'specs')
HUMAN_KINDS = (*SUGAR_KINDS, 'siglip224')


def required_inputs():
    paths = [
        f'{FULL_REL}/manifest.json',
        'stage5/frozen_manifest.json', 'stage5/frozen_protocol.json',
        'stage5/primary_reference.json',
        'stage6/uploaded/frozen_manifest.json',
        'stage6/uploaded/frozen_protocol.json',
        'stage6/uploaded/primary_reference.json',
        'stage6/uploaded/primary_siglip224_results.json',
        'stage6/uploaded/runs/siglip224/embeddings.npz',
        'stage7/human_manifest.json', 'stage7/human_results.json',
        'stage7/human_scores.csv',
        'research/embeddinggemma2_clipscore_replacement/provenance/expert_author_annotations.json',
    ]
    for root, kinds in ((f'{FULL_REL}/runs', SUGAR_KINDS),
                        ('stage5/runs', SUGAR_KINDS),
                        ('stage7/runs', tuple(k + '-human' for k in HUMAN_KINDS))):
        for kind in kinds:
            paths.extend(f'{root}/{kind}/{name}' for name in ('metadata.json', 'embeddings.npz'))
    return paths


def validate_inputs(baseline):
    baseline = Path(baseline)
    missing = [p for p in required_inputs() if not (baseline / p).is_file()]
    if missing:
        raise FileNotFoundError(
            'Private FULL handoff inputs are required; the public repo is source-only. '
            'Attach baseline.zip or its extracted tree. Missing files:\n' + '\n'.join(missing))
    sugar = json.loads((baseline / FULL_REL / 'manifest.json').read_text())['records']
    human = json.loads((baseline / 'stage7/human_manifest.json').read_text())['records']
    frozen = json.loads((baseline / 'stage5/frozen_manifest.json').read_text())['records']
    counts = {
        'sugar_full_pairs': len(sugar),
        'sugar_full_images': len({r['filename'] for r in sugar}),
        'sugar_development_images': len({r['filename'] for r in sugar if r['previously_seen']}),
        'sugar_unseen_pairs': sum(not r['previously_seen'] for r in sugar),
        'sugar_unseen_images': len({r['filename'] for r in sugar if not r['previously_seen']}),
        'sugar_frozen_pairs': len(frozen),
        'human_full_captions': len(human),
        'human_full_images': len({r['filename'] for r in human}),
        'human_full_ratings': sum(len(r['ratings']) for r in human),
        'human_unseen_captions': sum(not r['previously_seen'] for r in human),
        'human_unseen_images': len({r['filename'] for r in human if not r['previously_seen']}),
        'human_unseen_ratings': sum(len(r['ratings']) for r in human if not r['previously_seen']),
    }
    expected = dict(zip(counts, (7511, 1560, 555, 3949, 1005, 560,
                                5664, 1000, 16992, 5112, 900, 15336)))
    if counts != expected:
        raise ValueError(f'Historical cohort counts differ: actual={counts}; expected={expected}')
    sugar_ids = {r['id'] for r in sugar if not r['previously_seen']}
    if not {r['id'] for r in frozen} <= sugar_ids:
        raise ValueError('SugarCrepe560 must be a subset of historical unseen3949')
    return counts


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            path = Path(member.filename)
            target = (destination / path).resolve()
            if (path.is_absolute() or '..' in path.parts or '\\' in member.filename
                    or not target.is_relative_to(destination)
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError(f'Unsafe ZIP entry: {member.filename}')
        bundle.extractall(destination)


def restore_baseline(inputs, destination):
    """Support Kaggle auto-extraction and ZIPs with or without a baseline/ prefix."""
    inputs, destination = Path(inputs), Path(destination)
    if destination.exists():
        raise FileExistsError(f'Refusing to reuse a previous run directory: {destination}')
    archives = sorted(inputs.rglob('baseline.zip'))
    roots = sorted({p.parent.parent for p in inputs.rglob('human_manifest.json')
                    if p.parent.name == 'stage7'})
    if len(archives) > 1 or (not archives and len(roots) > 1):
        raise ValueError('Multiple FULL baselines attached; attach exactly one input dataset')
    if archives:
        safe_extract(archives[0], destination)
        candidates = sorted({p.parent.parent for p in destination.rglob('human_manifest.json')
                             if p.parent.name == 'stage7'})
        if len(candidates) != 1:
            raise ValueError('baseline.zip must contain exactly one stage7/human_manifest.json')
        baseline = candidates[0]
    elif roots:
        shutil.copytree(roots[0], destination)
        baseline = destination
    else:
        raise FileNotFoundError('No private FULL baseline found under /kaggle/input; see RUN_ON_KAGGLE.md')
    validate_inputs(baseline)
    return baseline
