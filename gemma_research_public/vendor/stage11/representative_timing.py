"""Supplemental native cost measurement on128 unique, label-blind test images.

This does not fit adapters or select models. Original fixed-panel timings remain
separate. GPU cost is bounded to two rounds, batch1/8, three declared variants.
"""
import gc
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import read, seal, sha, write
from learning import BoundedAdapter
from native import Encoder


def select_panel(records, seed=20261015):
    rng = random.Random(seed)
    chosen, used = [], set()
    for domain, count in [('sugar', 43), ('flickr', 43), ('polaris', 42)]:
        groups = {}
        for row in sorted(records, key=lambda r: r['id']):
            if row['domain'] == domain and row['split'] == 'test' and row['image'] not in used:
                groups.setdefault(row['image'], []).append(row)
        if len(groups) < count:
            raise ValueError('Insufficient distinct test images for timing panel')
        for image in rng.sample(sorted(groups), count):
            chosen.append(rng.choice(groups[image]))
            used.add(image)
    rng.shuffle(chosen)
    return chosen


def verify_file(root, name, manifest):
    if name not in manifest or sha(root / name) != manifest[name]:
        raise ValueError('Source manifest mismatch: ' + name)


def load_heads(core, kind, device):
    heads = {}
    selections = read(core / kind / 'selection.json')
    for phase in ['mean', 'uncertainty']:
        if selections[phase]['selection']['fallback_base']:
            heads[phase + '_single'] = None
            continue
        checkpoint = torch.load(core / kind / (phase + '_seed11.pt'), weights_only=True, map_location='cpu')
        head = BoundedAdapter(checkpoint['dimension'], checkpoint['config']['rank'], checkpoint['config']['radius'])
        head.load_state_dict(checkpoint['state_dict'])
        heads[phase + '_single'] = head.to(device).eval()
    return heads


def compare_receipts(current, original, common_device=False):
    required = ['weights', 'dtype'] if common_device else ['weights', 'dtype', 'gpu', 'torch_version', 'cuda_version']
    if any(current[key] != original[key] for key in required):
        raise ValueError('Timing must retain original weights/precision and declared hardware policy')


def main(data, core, output, expected_data_sha, expected_core_sha, common_device=False):
    if not torch.cuda.is_available():
        raise RuntimeError('Native timing requires GPU')
    data, core, output = Path(data), Path(core), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if sha(data / 'manifest_sha256.json') != expected_data_sha or sha(core / 'manifest_sha256.json') != expected_core_sha:
        raise ValueError('Mounted source version changed')
    dm, cm = read(data / 'manifest_sha256.json')['files'], read(core / 'manifest_sha256.json')['files']
    for name in ['records.json', 'status.json']:
        verify_file(data, name, dm)
    for name in ['protocol.json', 'status.json', 'bootstrap_status.json', 'input_receipt.json']:
        verify_file(core, name, cm)
    if not read(core / 'status.json')['complete'] or not read(core / 'bootstrap_status.json')['complete']:
        raise ValueError('Core producer incomplete')
    if read(core / 'input_receipt.json')['records_sha256'] != sha(data / 'records.json'):
        raise ValueError('Core/data record identity mismatch')
    protocol = read(core / 'protocol.json')
    panel = select_panel(read(data / 'records.json'))
    panel_identity = [{'id': r['id'], 'image': r['image'], 'caption': r['caption'], 'domain': r['domain']} for r in panel]
    logical_sha = hashlib.sha256(json.dumps(panel_identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    write(output / 'timing_protocol.json', {'version': 'stage11-representative-timing-v1', 'seed': 20261015,
                                           'unique_images': 128, 'domain_counts': {'sugar': 43, 'flickr': 43, 'polaris': 42},
                                           'batch_sizes': [1, 8], 'rounds': 2, 'warmup_per_batch_variant': 5,
                                           'selection': 'uniform images within each domain, then uniform caption within image; globally unique pixel images; no rating or model scores used',
                                           'logical_panel_sha256': logical_sha, 'data_manifest_sha256': expected_data_sha,
                                           'core_manifest_sha256': expected_core_sha,
                                           'hardware_policy': 'all backbones remeasured on one local GPU; original quality devices may differ' if common_device else 'same GPU type and runtime as original core',
                                           'remeasurement_gpu': torch.cuda.get_device_name(0),
                                           'remeasurement_torch': torch.__version__,
                                           'remeasurement_cuda': torch.version.cuda,
                                           'remeasurement_platform': platform.platform(),
                                           'precision': protocol['precision'],
                                           'scope': '128-image stratified sample, two rounds; p95 is not all production traffic'})
    write(output / 'panel.json', panel_identity)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    images = []
    for row in panel:
        name = 'images/' + row['image'] + '.jpg'
        verify_file(data, name, dm)
        with Image.open(data / name) as image:
            images.append(image.convert('RGB'))
    all_results, raw = [], []
    kinds = read(core / 'status.json')['expected_backbones']
    write(output / 'status.json', {'complete': False, 'completed_backbones': []})
    for kind in kinds:
        folder = output / kind
        folder.mkdir(exist_ok=True)
        for name, digest in cm.items():
            if name.startswith(kind + '/') and (name.endswith('.pt') or name.endswith('/selection.json') or name.endswith('/model_receipt.json')):
                verify_file(core, name, cm)
        original = read(core / kind / 'model_receipt.json')
        override = torch.float32 if common_device and kind == 'gemma' and original['dtype'] == 'torch.float32' else None
        encoder = Encoder(kind, protocol['models'][kind], folder, dtype_override=override)
        compare_receipts(encoder.receipt, original, common_device)
        write(folder / 'hardware_comparison.json', {'common_device_remeasurement': common_device,
                                                   'quality_receipt': original, 'timing_receipt': encoder.receipt,
                                                   'weights_and_dtype_match': True,
                                                   'quality_recomputed': False})
        heads = {'base': None, **load_heads(core, kind, 'cuda')}
        variants = list(heads)
        @torch.inference_mode()
        def call(indices, head):
            u = encoder.images([images[i] for i in indices])
            v = encoder.texts([panel[i]['caption'] for i in indices])
            if head is not None:
                u, v = head(u), head(v)
            score = (u*v).sum(-1)
            if kind != 'siglip224':
                score = score.clamp_min(0)
            return score * protocol['models'][kind]['scale']
        for batch in [1, 8]:
            groups = [list(range(start, start + batch)) for start in range(0, len(panel), batch)]
            for name in variants:
                for _ in range(5):
                    call(groups[0], heads[name])
            measurements = {name: [] for name in variants}
            for round_index in range(2):
                for group_index, indices in enumerate(groups):
                    shift = (round_index + group_index) % len(variants)
                    order = variants[shift:] + variants[:shift]
                    for name in order:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        start = time.perf_counter()
                        score = call(indices, heads[name])
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - start) * 1000
                        if not torch.isfinite(score).all():
                            raise ValueError('Invalid supplemental native score')
                        entry = {'backbone': kind, 'variant': name, 'batch': batch, 'round': round_index,
                                 'group_index': group_index, 'ms_call': elapsed,
                                 'peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2}
                        measurements[name].append(entry)
                        raw.append(entry)
                    if group_index % 32 == 0:
                        print('TIMING', kind, 'batch', batch, 'round', round_index, 'group', group_index, '/', len(groups), flush=True)
            base_values = np.array([x['ms_call'] for x in measurements['base']])
            for name in variants:
                values = np.array([x['ms_call'] for x in measurements[name]])
                paired = values - base_values
                # Resample groups, preserving both rounds for each group.
                rng = np.random.default_rng(20261015)
                matrix = paired.reshape(2, len(groups))
                ids = rng.integers(len(groups), size=(2000, len(groups)))
                medians = np.median(matrix[:, ids].transpose(1, 0, 2).reshape(2000, -1), axis=1)
                ci = np.percentile(medians, [2.5, 97.5])
                all_results.append({'backbone': kind, 'variant': name, 'batch': batch,
                                    'p50_ms_call': float(np.median(values)), 'p95_ms_call': float(np.percentile(values, 95)),
                                    'p50_ms_pair': float(np.median(values) / batch), 'p95_ms_pair': float(np.percentile(values, 95) / batch),
                                    'effective_throughput_pairs_s': float(batch * 1000 / np.median(values)),
                                    'aggregate_throughput_pairs_s': float(batch * len(values) * 1000 / values.sum()),
                                    'peak_allocated_mib': max(x['peak_allocated_mib'] for x in measurements[name]),
                                    'paired_added_p50_ms_call': float(np.median(paired)), 'paired_added_ci95_ms_call': ci.tolist(),
                                    'paired_bootstrap_unit': 'input group, retaining both rounds',
                                    'timed_calls': len(values), 'unique_images': 128, 'dtype': encoder.receipt['dtype'],
                                    'gpu': encoder.receipt['gpu'], 'used_gpus': 1})
            write(output / 'timing_results.json', all_results)
            write(output / 'timing_samples.json', raw)
        encoder.unload()
        del heads
        gc.collect()
        torch.cuda.empty_cache()
        write(output / 'status.json', {'complete': False, 'completed_backbones': kinds[:kinds.index(kind) + 1]})
    write(output / 'status.json', {'complete': True, 'completed_backbones': kinds, 'unique_images': 128})
    seal(output)
