"""Recompute all saved Gemma metrics; optionally re-score saved heads on CPU."""
import argparse
import math
from pathlib import Path

import numpy as np
from bundle import find_bundle, load_data, load_head, read, verify_assets, write
from learning import partition_metrics, prediction_pack


def compare(actual, expected, path='metrics'):
    if isinstance(expected, dict):
        if set(actual) != set(expected):raise ValueError('Metric fields differ: '+path)
        for key in expected:compare(actual[key], expected[key], path+'/'+key)
    elif isinstance(expected, (int, float)):
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):raise ValueError('Metric differs: '+path)
    elif actual != expected:
        raise ValueError('Metric differs: '+path)


def replay(root=None, output=None, rescore=True):
    import torch
    torch.set_num_threads(2)
    root = verify_assets(root)
    core = root / 'assets/core'
    records = read(core / 'records.json')
    test = [r for r in records if r['split']=='test']
    folder = core / 'gemma'
    mapping = read(folder / 'prediction_mapping.json')
    if mapping['test_record_ids'] != [r['id'] for r in test]:raise ValueError('Test record mapping differs')
    if mapping['sugar_negative_ids'] != [r['id'] for r in test if r['domain']=='sugar']:raise ValueError('Negative mapping differs')
    expected = read(folder / 'metrics.json')
    actual = {}
    with np.load(folder / 'predictions.npz', allow_pickle=False) as z:
        for name in expected:
            actual[name] = partition_metrics(test, z[name+'_candidate'], z[name+'_negative'])
            compare(actual[name], expected[name], name)
    audit = {'complete': True, 'saved_variants_recomputed': len(actual), 'test_records': len(test),
             'saved_metrics_match': True, 'new_native_encoding': False, 'new_training': False}
    if rescore:
        core, records, cache = load_data(root)
        differences = {}
        with np.load(folder / 'predictions.npz', allow_pickle=False) as z:
            for name, phase in [('base',None),('mean_single','mean'),('uncertainty_single','uncertainty')]:
                head = load_head(core, phase) if phase else None
                candidate, negative = prediction_pack(cache, test, head, 'cpu', 2.5, True)
                delta = max(float(np.abs(candidate-z[name+'_candidate']).max()),float(np.abs(negative-z[name+'_negative']).max()))
                if delta > 2e-5:raise ValueError('CPU checkpoint replay discrepancy: '+name)
                differences[name] = delta
        audit['cpu_vs_saved_cuda_max_abs_score_difference'] = differences
        audit['cpu_rescore_scope'] = 'Same frozen cache/checkpoints; CPU kernels may round differently. Saved-prediction metrics are the reference.'
    if output:
        output = Path(output)
        output.mkdir(parents=True,exist_ok=True)
        write(output/'recomputed_metrics.json',actual);write(output/'replay_audit.json',audit)
    return actual, audit


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle');p.add_argument('--output',default='runs/replay');p.add_argument('--saved-only',action='store_true')
    a=p.parse_args();_,audit=replay(a.bundle,a.output,not a.saved_only);print(audit)
