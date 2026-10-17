"""Rebuild real public inputs on Kaggle CPU, before any model scoring."""
import concurrent.futures
import hashlib
import io
import random
import urllib.request
from pathlib import Path

from common import read, seal, sha, write

CATEGORIES = ['add_att', 'add_obj', 'replace_att', 'replace_obj', 'replace_rel', 'swap_att', 'swap_obj']


def download(url, path, expected=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        part = path.with_suffix(path.suffix + '.part')
        with urllib.request.urlopen(url, timeout=180) as source, part.open('wb') as dest:
            for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
                dest.write(block)
        part.replace(path)
    if expected and sha(path) != expected:
        raise ValueError('Source digest mismatch: ' + path.name)
    return path


def materialize(blob, images):
    from PIL import Image
    with Image.open(io.BytesIO(blob)) as image:
        image = image.convert('RGB')
        identity = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    path = images / (identity + '.jpg')
    if not path.exists():
        # Preserve original bytes: .jpg is an opaque filename, PIL uses the header.
        path.write_bytes(blob)
    return identity


def collect_human(repo, revision, prefixes, scratch, images):
    import pyarrow.parquet as pq
    import json
    with urllib.request.urlopen(f'https://huggingface.co/api/datasets/{repo}/revision/{revision}?blobs=true', timeout=60) as r:
        info = json.load(r)
    if info['sha'] != revision:
        raise ValueError('Dataset revision mismatch')
    sources, rows = {}, []
    entries = sorted(info['siblings'], key=lambda e: e['rfilename'])
    for prefix, limit in prefixes:
        selected = [e for e in entries if e['rfilename'].startswith(prefix) and e['rfilename'].endswith('.parquet')]
        if limit:
            selected = selected[:limit]
        expected_count = 2 if repo.endswith('Flickr8k-HumanEval') else (11 if prefix == 'data/test-' else 3)
        if len(selected) != expected_count:
            raise ValueError(f'Unexpected shard count {repo} {prefix}: {len(selected)}')
        for entry in selected:
            name = entry['rfilename']
            local = scratch / repo.split('/')[-1] / Path(name).name
            download(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}', local,
                     entry.get('lfs', {}).get('sha256'))
            sources[name] = {'sha256': sha(local), 'bytes': local.stat().st_size}
            print('verified shard', repo, name, flush=True)
            parquet = pq.ParquetFile(local)
            cols = set(parquet.schema_arrow.names)
            wanted = [c for c in ['imgid', 'img', 'cand', 'human_score', 'refs', 'source_row', 'id'] if c in cols]
            if not {'img', 'cand', 'human_score', 'refs'}.issubset(cols):
                raise ValueError('Unsupported annotation schema: ' + str(cols))
            for batch in parquet.iter_batches(batch_size=64, columns=wanted):
                for row in batch.to_pylist():
                    if row['cand'] in row['refs']:
                        continue
                    image = row['img']
                    blob = image['bytes'] if isinstance(image, dict) else image
                    identity = materialize(blob, images)
                    rows.append({'image': identity, 'caption': row['cand'], 'rating': float(row['human_score']),
                                 'original_name': str(row.get('imgid', identity)), 'refs': row['refs'],
                                 'official_split': 'test' if 'test-' in name else 'train'})
    groups = {}
    for row in rows:
        key = (row['official_split'], row['image'], row['caption'])
        if key not in groups:
            groups[key] = {k: v for k, v in row.items() if k != 'rating'}
            groups[key]['ratings'] = []
        if groups[key]['refs'] != row['refs']:
            raise ValueError('Inconsistent references for one candidate')
        groups[key]['ratings'].append(row['rating'])
    return list(groups.values()), {'repo': repo, 'revision': revision, 'files': sources}


def assign_splits(sugar, flickr, polaris, protocol):
    """Never place one decoded image in different partitions, even across corpora."""
    cfg = protocol['data']
    rng = random.Random(protocol['seed'])
    official_test = {r['image'] for r in polaris if r['official_split'] == 'test'}
    original_names = sorted({r['original_name'] for r in flickr})
    if len(original_names) != 1000:
        raise ValueError('Expected all 1000 Flickr images')
    development = set(random.Random(cfg['flickr_prior_seed']).sample(original_names, 100))
    flickr_test = {r['image'] for r in flickr if r['original_name'] not in development}
    # Reserve ALL official Polaris TEST and reconstructed Flickr900 test before choosing any training image.
    reserved = official_test | flickr_test
    fdev = sorted({r['image'] for r in flickr if r['original_name'] in development} - reserved)
    rng.shuffle(fdev)
    ftrain = set(fdev[:cfg['flickr_train_images']])
    fval = set(fdev[cfg['flickr_train_images']:])
    occupied = {r['image'] for r in flickr}
    snames = sorted({r['image'] for r in sugar} - reserved - occupied)
    rng.shuffle(snames)
    nt, nv = cfg['sugar_train_images'], cfg['sugar_validation_images']
    if len(snames) <= nt + nv:
        raise ValueError('Insufficient disjoint SugarCrepe images')
    strain, sval = set(snames[:nt]), set(snames[nt:nt + nv])
    # Extra Sugar media overlapping an external test remains test; never training.
    stest = {r['image'] for r in sugar} - strain - sval - ftrain - fval
    ptraining = sorted({r['image'] for r in polaris if r['official_split'] == 'train'}
                       - official_test - occupied - {r['image'] for r in sugar})
    rng.shuffle(ptraining)
    pt, pv = cfg['polaris_train_images'], cfg['polaris_validation_images']
    if len(ptraining) < pt + pv or len(fval) < 5:
        raise ValueError('Insufficient disjoint human development data')
    ptrain, pval = set(ptraining[:pt]), set(ptraining[pt:pt + pv])
    result = []
    for domain, records, train, val, test in [
        ('sugar', sugar, strain, sval, stest), ('flickr', flickr, ftrain, fval, flickr_test),
        ('polaris', polaris, ptrain, pval, official_test)]:
        for row in records:
            image = row['image']
            split = 'train' if image in train else 'validation' if image in val else 'test' if image in test else None
            if split is None:
                continue
            if domain == 'polaris' and ((split == 'test') != (row['official_split'] == 'test')):
                continue
            result.append(dict(row, domain=domain, split=split))
    media = {split: {r['image'] for r in result if r['split'] == split} for split in ['train', 'validation', 'test']}
    overlap = {a + '_' + b: len(media[a] & media[b]) for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]}
    if any(overlap.values()):
        raise ValueError('Global media leakage: ' + str(overlap))
    for idx, row in enumerate(result):
        row['id'] = f'{row["domain"]}:{idx}'
    counts = {domain: {split: {'records': sum(r['domain'] == domain and r['split'] == split for r in result),
                               'images': len({r['image'] for r in result if r['domain'] == domain and r['split'] == split})}
                       for split in media} for domain in ['sugar', 'flickr', 'polaris']}
    return result, {'counts': counts, 'global_overlap_images': overlap,
                    'historical_sugar_split_recovered': False, 'flickr_development_after_exclusions': len(fdev)}


def main(output, scratch):
    output, scratch = Path(output), Path(scratch)
    output.mkdir(parents=True, exist_ok=True)
    images = output / 'images'
    images.mkdir(exist_ok=True)
    protocol = read(Path(__file__).with_name('protocol.json'))
    write(output / 'protocol.json', protocol)
    write(output / 'status.json', {'complete': False, 'phase': 'public_input_download'})
    sugar, sources = [], {}
    for category in CATEGORIES:
        path = download(f'https://raw.githubusercontent.com/RAIVNLab/sugar-crepe/{protocol["data"]["sugar_commit"]}/data/{category}.json', scratch / f'{category}.json')
        sources[category] = sha(path)
        for key, row in read(path).items():
            sugar.append(dict(row, source_id=category + ':' + key, category=category))
    names = sorted({r['filename'] for r in sugar})
    def get_image(name):
        local = download('https://s3.us-east-1.amazonaws.com/images.cocodataset.org/val2017/' + name, scratch / 'coco' / name)
        return name, materialize(local.read_bytes(), images)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        image_ids = dict(pool.map(get_image, names))
    for row in sugar:
        row['image'] = image_ids[row['filename']]
    print('SugarCrepe public media verified', len(image_ids), flush=True)
    flickr, freceipt = collect_human(protocol['data']['flickr_repo'], protocol['data']['flickr_revision'], [('expert/test-', 2)], scratch, images)
    polaris, preceipt = collect_human(protocol['data']['polaris_repo'], protocol['data']['polaris_revision'], [('data/test-', None), ('data/train-', 3)], scratch, images)
    records, audit = assign_splits(sugar, flickr, polaris, protocol)
    write(output / 'records.json', records)
    write(output / 'split_audit.json', audit)
    write(output / 'source_receipts.json', {'sugar_commit': protocol['data']['sugar_commit'], 'sugar_sha256': sources,
                                           'flickr': freceipt, 'polaris': preceipt})
    write(output / 'status.json', {'complete': True, 'phase': 'verified_public_inputs', 'records': len(records),
                                 'records_sha256': sha(output / 'records.json'), 'protocol_sha256': sha(output / 'protocol.json')})
    seal(output)
    print('INPUTS COMPLETE', audit, flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--scratch', required=True)
    a = p.parse_args()
    main(a.output, a.scratch)
