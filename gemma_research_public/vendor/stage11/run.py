"""Verified inputs -> fresh encoding -> validation-only fitting -> test and timing."""
import gc
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import read, seal, sha, verify, write
from learning import (correlation, fit, partition_metrics, prediction_pack, select_config,
                      validation_vector)
from native import Encoder, benchmark, encode_cache


def bootstrap(rows, base, candidate, base_negative, candidate_negative, cfg):
    """Paired image-cluster bootstrap, retaining every vote within each caption."""
    result = {}
    rng = np.random.default_rng(cfg['seed'])
    for domain in ['sugar', 'flickr', 'polaris']:
        indices = [i for i, row in enumerate(rows) if row['domain'] == domain]
        data = [rows[i] for i in indices]
        a, b = base[indices], candidate[indices]
        groups = {}
        for i, row in enumerate(data):
            groups.setdefault(row['image'], []).append(i)
        blocks = [np.asarray(g, dtype=np.int64) for g in groups.values()]
        if domain == 'sugar':
            delta = (b > candidate_negative).astype(float) - (a > base_negative).astype(float)
            values = []
            for _ in range(cfg['bootstrap_replicates']):
                ids = np.concatenate([blocks[j] for j in rng.integers(len(blocks), size=len(blocks))])
                values.append(float(delta[ids].mean()))
            result['sugar_accuracy'] = {'delta': float(delta.mean()), 'ci95': np.percentile(values, [2.5, 97.5]).tolist(), 'unit': 'fraction'}
            continue
        means = np.array([np.mean(r['ratings']) for r in data])
        votes = [np.array(r['ratings']) for r in data]
        caption_delta, raw_delta = [], []
        for _ in range(cfg['bootstrap_replicates']):
            ids = np.concatenate([blocks[j] for j in rng.integers(len(blocks), size=len(blocks))])
            ca, cb = correlation(a[ids], means[ids]), correlation(b[ids], means[ids])
            vote_ids = np.repeat(ids, [len(votes[i]) for i in ids])
            y = np.concatenate([votes[i] for i in ids])
            ra, rb = correlation(a[vote_ids], y), correlation(b[vote_ids], y)
            if ca is not None and cb is not None:
                caption_delta.append(cb - ca)
            if ra is not None and rb is not None:
                raw_delta.append(rb - ra)
        if min(len(caption_delta), len(raw_delta)) < 0.95 * cfg['bootstrap_replicates']:
            raise ValueError('Too many undefined bootstrap correlations')
        full_ids = np.repeat(np.arange(len(data)), [len(v) for v in votes])
        y = np.concatenate(votes)
        for label, distribution, point in [
            ('caption_mean_tau_c', caption_delta, correlation(b, means) - correlation(a, means)),
            ('raw_vote_tau_c', raw_delta, correlation(b[full_ids], y) - correlation(a[full_ids], y))]:
            result[domain + '_' + label] = {'delta': point, 'ci95': np.percentile(distribution, [2.5, 97.5]).tolist(),
                                           'valid_replicates': len(distribution), 'unit': 'correlation'}
    return {'sampling_unit': 'image_cluster', 'replicates': cfg['bootstrap_replicates'], 'seed': cfg['seed'], 'metrics': result}


def cached_head_benchmark(cache, heads, rows):
    panel = rows[:8]
    image = torch.tensor(np.stack([cache['image'][cache['image_index'][r['image']]] for r in panel]), device='cuda')
    text = torch.tensor(np.stack([cache['text'][cache['text_index'][r['caption']]] for r in panel]), device='cuda')
    result = []
    for batch in [1, 8]:
        for name, head in {'base': None, **heads}.items():
            times = []
            with torch.inference_mode():
                for i in range(220):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    u, v = image[:batch], text[:batch]
                    if head is not None:
                        u, v = head(u), head(v)
                    value = (u * v).sum(-1)
                    torch.cuda.synchronize()
                    if i >= 20:
                        times.append((time.perf_counter() - start) * 1000)
            result.append({'variant': name, 'batch': batch, 'p50_ms_call': float(np.median(times)),
                           'p95_ms_call': float(np.percentile(times, 95)), 'scope': 'cached FP32 embedding head only'})
    return result


def run_backbone(kind, model_config, data_root, output, records, protocol):
    output.mkdir(parents=True, exist_ok=True)
    write(output / 'status.json', {'complete': False, 'phase': 'native_encoding'})
    encoder = Encoder(kind, model_config, output)
    cache = encode_cache(encoder, data_root, records, output)
    train = [r for r in records if r['split'] == 'train']
    validation = [r for r in records if r['split'] == 'validation']
    test = [r for r in records if r['split'] == 'test']
    hyper = protocol['learning']
    scale, clamp = model_config['scale'], kind != 'siglip224'
    base_val, base_val_neg = prediction_pack(cache, validation, scale=scale, clamp=clamp)
    baseline_validation = validation_vector(partition_metrics(validation, base_val, base_val_neg))
    selections, heads, traces = {}, {}, {}
    write(output / 'status.json', {'complete': False, 'phase': 'validation_training'})
    for phase in hyper['phases']:
        trials = []
        for index, config in enumerate(hyper['configs']):
            gains, epochs = [], []
            for seed in hyper['seeds']:
                head, history, epoch = fit(cache, train, validation, config, hyper, seed, phase, 'cuda',
                                           baseline_validation=baseline_validation, scale=scale, clamp=clamp)
                prediction, negative = prediction_pack(cache, validation, head, 'cuda', scale, clamp)
                gain = (validation_vector(partition_metrics(validation, prediction, negative)) - baseline_validation) / [0.01, 0.02, 0.02]
                gains.append(gain)
                epochs.append(epoch)
                write(output / 'training' / f'{phase}_config{index}_seed{seed}.json', {'config': config, 'seed': seed, 'phase': phase, 'history': history, 'best_epoch': epoch})
                del head
            trials.append({'config': config, 'mean_normalized_gain': np.mean(gains, axis=0).tolist(), 'best_epochs': epochs})
            print(kind, phase, index, 'validation gain', trials[-1]['mean_normalized_gain'], flush=True)
        selection = select_config(trials)
        selections[phase] = {'selection': selection, 'trials': trials, 'primary_seed': 11,
                             'protocol_sha256': sha(Path(__file__).with_name('protocol.json')),
                             'test_used_for_selection': False}
        write(output / 'selection.json', selections)
        if selection['fallback_base']:
            heads[phase + '_single'] = None
            continue
        for seed in hyper['seeds']:
            head, history, _ = fit(cache, train + validation, [], selection['config'], hyper, seed, phase, 'cuda',
                                   epochs=selection['epochs'], scale=scale, clamp=clamp)
            key = phase + f'_seed{seed}'
            heads[key] = head
            torch.save({'state_dict': head.cpu().state_dict(), 'dimension': cache['image'].shape[1],
                        'config': selection['config'], 'seed': seed, 'phase': phase,
                        'records_sha256': sha(data_root / 'records.json'), 'protocol': protocol['version']}, output / (key + '.pt'))
            head.to('cuda')
            if seed == 11:
                heads[phase + '_single'] = head
            write(output / 'training' / (key + '_refit.json'), {'epochs': selection['epochs'], 'history': history})
    # All selection decisions are persisted before the first test score is evaluated.
    write(output / 'status.json', {'complete': False, 'phase': 'test_evaluation', 'selection_sha256': sha(output / 'selection.json')})
    base, base_neg = prediction_pack(cache, test, scale=scale, clamp=clamp)
    predictions = {'base': (base, base_neg)}
    metrics = {'base': partition_metrics(test, base, base_neg)}
    for name, head in heads.items():
        p, n = prediction_pack(cache, test, head, 'cuda', scale, clamp)
        predictions[name] = (p, n)
        metrics[name] = partition_metrics(test, p, n)
    for phase in hyper['phases']:
        keys = [phase + f'_seed{seed}' for seed in hyper['seeds']]
        if all(k in predictions for k in keys):
            p = np.mean([predictions[k][0] for k in keys], axis=0)
            n = np.mean([predictions[k][1] for k in keys], axis=0)
            name = phase + '_ensemble3_diagnostic'
            predictions[name] = (p, n)
            metrics[name] = partition_metrics(test, p, n)
    if kind == 'gemma':
        config = {'rank': 32, 'radius': 0.03, 'preservation': 0.1}
        for ablation in ['sugar_only', 'human_only', 'no_preservation', 'full_control']:
            head, history, _ = fit(cache, train + validation, [], config, hyper, 11, 'mean', 'cuda',
                                   epochs=hyper['epochs'], ablation=ablation, scale=scale, clamp=clamp)
            p, n = prediction_pack(cache, test, head, 'cuda', scale, clamp)
            name = 'ablation_' + ablation
            predictions[name] = (p, n)
            metrics[name] = partition_metrics(test, p, n)
            write(output / 'training' / (name + '.json'), {'config': config, 'history': history, 'same_epochs': hyper['epochs']})
            torch.save(head.cpu().state_dict(), output / (name + '.pt'))
            del head
    write(output / 'metrics.json', metrics)
    np.savez_compressed(output / 'predictions.npz', **{key + suffix: v for key, pair in predictions.items() for suffix, v in zip(['_candidate', '_negative'], pair)})
    write(output / 'prediction_mapping.json', {'test_record_ids': [r['id'] for r in test],
                                              'sugar_negative_ids': [r['id'] for r in test if r['domain'] == 'sugar'],
                                              'records_sha256': sha(data_root / 'records.json')})
    comparisons = {}
    for phase in hyper['phases']:
        key = phase + '_single'
        comparisons[key + '_vs_base'] = bootstrap(test, base, predictions[key][0], base_neg, predictions[key][1], protocol['statistics'])
    if 'uncertainty_single' in predictions and 'mean_single' in predictions:
        a, an = predictions['mean_single']
        b, bn = predictions['uncertainty_single']
        comparisons['uncertainty_vs_mean'] = bootstrap(test, a, b, an, bn, protocol['statistics'])
    write(output / 'paired_bootstrap.json', comparisons)
    deployed = {phase + '_single': heads[phase + '_single'] for phase in hyper['phases']}
    gc.collect()
    torch.cuda.empty_cache()
    write(output / 'native_latency.json', benchmark(encoder, deployed, records, data_root, protocol['latency']))
    write(output / 'cached_head_latency.json', cached_head_benchmark(cache, deployed, test))
    write(output / 'parameter_counts.json', {name: sum(p.numel() for p in head.parameters()) if head else 0 for name, head in deployed.items()})
    write(output / 'status.json', {'complete': True, 'phase': 'complete', 'backbone': kind,
                                 'records_sha256': sha(data_root / 'records.json'), 'precision': str(encoder.dtype)})
    encoder.unload()
    del heads, deployed
    gc.collect()
    torch.cuda.empty_cache()
    seal(output)


def main(data_root, output, backbones):
    if not torch.cuda.is_available():
        raise RuntimeError('Fresh native evaluation requires a GPU')
    data_root, output = Path(data_root), Path(output)
    verify(data_root)
    if not read(data_root / 'status.json')['complete']:
        raise ValueError('Input producer incomplete')
    protocol = read(Path(__file__).with_name('protocol.json'))
    source_protocol = read(data_root / 'protocol.json')
    if source_protocol['data'] != protocol['data'] or source_protocol['seed'] != protocol['seed']:
        raise ValueError('Input split protocol changed')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    records = read(data_root / 'records.json')
    write(output / 'protocol.json', protocol)
    write(output / 'records.json', records)
    write(output / 'split_audit.json', read(data_root / 'split_audit.json'))
    write(output / 'input_receipt.json', {'producer_manifest_sha256': sha(data_root / 'manifest_sha256.json'),
                                         'records_sha256': sha(data_root / 'records.json'), 'verified': True})
    write(output / 'runtime.json', {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                                   'gpu': torch.cuda.get_device_name(0), 'visible_gpus': torch.cuda.device_count()})
    complete = []
    write(output / 'status.json', {'complete': False, 'expected_backbones': backbones, 'completed_backbones': complete})
    for kind in backbones:
        run_backbone(kind, protocol['models'][kind], data_root, output / kind, records, protocol)
        complete.append(kind)
        write(output / 'status.json', {'complete': False, 'expected_backbones': backbones, 'completed_backbones': complete})
    write(output / 'status.json', {'complete': True, 'expected_backbones': backbones, 'completed_backbones': complete,
                                 'scope': 'Stage11 public rebuild; deferred baselines in protocol remain unavailable'})
    seal(output)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--backbones', default='gemma,clip,siglip224')
    a = p.parse_args()
    main(Path(a.data), Path(a.output), a.backbones.split(','))
