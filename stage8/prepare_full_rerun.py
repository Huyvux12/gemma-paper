"""Build a self-contained private GPU script. Does not call Kaggle or run ML."""
import argparse
import base64
import hashlib
import io
import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = 'dungdq210/embeddinggemma-stage8-20261009-cc802fcc'
DEFAULT_SLUG = 'embeddinggemma-full-rerun-20261012'
STAGE8_SOURCE = (
    'common.py', 'train_adapter.py', 'evaluate_adapter.py', 'metrics.py',
    'reference_metrics.py', 'pacscore_gpu.py', 'combine_tables.py',
    'uncertainty.py', 'compare_pac.py', 'benchmark_gpu.py', 'verify.py',
    'rerun_inputs.py', 'full_rerun.py', 'kaggle_bootstrap.py',
    'protocol.json', 'requirements_kaggle.txt',
)


def source_files():
    return sorted([*ROOT.joinpath('baseline').rglob('*.py'),
                   *(ROOT / 'stage8' / name for name in STAGE8_SOURCE)])


def build(output, username='dungdq210', dataset_sources=None, slug=DEFAULT_SLUG):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', username) or not re.fullmatch(r'[a-z0-9-]+', slug):
        raise ValueError('Invalid Kaggle username or kernel slug')
    dataset_sources = dataset_sources or [DEFAULT_DATASET]
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', value) for value in dataset_sources):
        raise ValueError('Use owner/dataset-slug for every dataset source')
    files = source_files()
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            # Fixed timestamps make regeneration byte-identical.
            info = zipfile.ZipInfo(str(path.relative_to(ROOT)), (2026, 10, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
        archive.writestr(zipfile.ZipInfo('source_sha256.json', (2026, 10, 12, 0, 0, 0)),
                         json.dumps(manifest, indent=2) + '\n')
    payload = buffer.getvalue()
    encoded = base64.b64encode(payload).decode('ascii')
    wrapped = '\n'.join(encoded[i:i + 100] for i in range(0, len(encoded), 100))
    code = '''"""Generated full Stage8 rerun; source-only payload, no data or credentials.
Regenerate with stage8/prepare_full_rerun.py. Execute on Kaggle GPU only.
"""
import base64
import hashlib
import io
import json
import runpy
import sys
import tempfile
import zipfile
from pathlib import Path

PAYLOAD_SHA256 = %r
PAYLOAD = """%s"""


def main():
    payload = base64.b64decode(PAYLOAD)
    if hashlib.sha256(payload).hexdigest() != PAYLOAD_SHA256:
        raise ValueError('Embedded source payload hash mismatch')
    # Scratch stays outside /kaggle/working: downloaded weights/data are not saved as output.
    scratch = Path(tempfile.mkdtemp(prefix='gemma-full-rerun-'))
    source = scratch / 'source'
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            path = Path(name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe embedded source path')
        archive.extractall(source)
    manifest = json.loads((source / 'source_sha256.json').read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Embedded source hash mismatch: ' + name)
    sys.path.insert(0, str(source / 'stage8'))
    runpy.run_path(str(source / 'stage8/kaggle_bootstrap.py'), run_name='__main__')


if __name__ == '__main__':
    main()
''' % (hashlib.sha256(payload).hexdigest(), wrapped)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'run_full_rerun.py').write_text(code)
    metadata = {'id': f'{username}/{slug}', 'title': slug.replace('-', ' '),
                'code_file': 'run_full_rerun.py', 'language': 'python', 'kernel_type': 'script',
                'is_private': True, 'enable_gpu': True, 'enable_internet': True,
                'dataset_sources': dataset_sources, 'competition_sources': [],
                'kernel_sources': [], 'model_sources': []}
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (output / 'source_sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--username', default='dungdq210')
    parser.add_argument('--dataset-source', action='append', help='Attached private FULL dataset ref; repeat if needed')
    parser.add_argument('--slug', default=DEFAULT_SLUG)
    parser.add_argument('--output', default=str(ROOT / 'stage8/kaggle_full_rerun'))
    parser.add_argument('--baseline-zip', help='Optionally stage a private dataset locally; never upload it')
    parser.add_argument('--dataset-dir', default='/tmp/gemma-full-rerun-inputs')
    args = parser.parse_args()
    refs = args.dataset_source
    if args.baseline_zip:
        if not refs or len(refs) != 1 or not refs[0].startswith(args.username + '/'):
            parser.error('--baseline-zip requires exactly one --dataset-source owned by --username')
        # Validate the archive path before building; the Kaggle bootstrap checks the FULL contents.
        source = Path(args.baseline_zip).resolve()
        if not source.is_file() or not zipfile.is_zipfile(source):
            parser.error('--baseline-zip must point to the original FULL ZIP')
        directory = Path(args.dataset_dir).resolve()
        if directory.is_relative_to(ROOT):
            parser.error('Keep private dataset staging outside the public repository')
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / 'baseline.zip'
        if target.exists():
            parser.error(f'{target} already exists; use a fresh --dataset-dir')
        shutil.copyfile(source, target)
        (directory / 'dataset-metadata.json').write_text(json.dumps({
            'id': refs[0], 'title': refs[0].split('/')[1].replace('-', ' '),
            'licenses': [{'name': 'other'}]}, indent=2) + '\n')
    metadata = build(args.output, args.username, refs, args.slug)
    print(json.dumps({'kernel': metadata['id'], 'path': args.output,
                      'dataset_sources': metadata['dataset_sources'], 'uploaded': False}, indent=2))


if __name__ == '__main__':
    main()
