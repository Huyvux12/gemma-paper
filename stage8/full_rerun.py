"""Full cached re-evaluation + GPU Stage8 retraining, PAC-S and encoder benchmark.

Metric definitions come from EmbeddingGemma2_Stage8_Report_VI.md and protocol.json.
This is an orchestration entrypoint; the historical adapter implementation is unchanged.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from rerun_inputs import FULL_REL, required_inputs, validate_inputs

PROTOCOL_SHA256 = 'cc802fcccba757e8fe2f0bb3f619b8d2c0fe1ea8fb60b363c3ac772eedbfb54b'
ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def digest(path):
    import hashlib
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Run:
    def __init__(self, output):
        self.output = Path(output)
        self.state = {'complete': False, 'completed_steps': [], 'active_step': None,
                      'protocol_sha256': digest(ROOT / 'protocol.json')}
        self.save()

    def save(self):
        write(self.output / 'pipeline_status.json', self.state)

    def step(self, name, command):
        self.state['active_step'] = name
        self.save()
        print(f'START {name}', flush=True)
        with (self.output / 'logs' / (name + '.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            self.state.update(failed_step=name, exit_code=result.returncode)
            self.save()
            raise RuntimeError(f'{name} failed (exit {result.returncode}); see logs/{name}.log')
        self.state['completed_steps'].append(name)
        self.state['active_step'] = None
        self.save()
        print(f'DONE {name}', flush=True)

    def script(self, name, script, *params):
        self.step(name, [sys.executable, str(script), *map(str, params)])


def check_selection(selection):
    selected = selection['selected']
    finalists = selection['finalists']
    expected_seeds = [11, 23, 37, 53, 71]
    if (selected['rank'] != 32 or selected['anchor_weight'] != 1.0
            or [r['seed'] for r in finalists] != expected_seeds
            or any(r['rank'] != 32 or r['anchor_weight'] != 1.0 for r in finalists)
            or not selection['backbone_frozen']
            or selection['selection_uses_historical_benchmark_or_human_scores']):
        raise ValueError('Validation did not reproduce rank32/anchor1.0/five seeds. '
                         'Saved all runs; stopping instead of silently changing the Stage8 recipe.')


def export_sugar(baseline, output):
    import numpy as np
    from common import load_gemma
    manifest, vectors = load_gemma(baseline)
    records = manifest['records']
    # Same float32 cached normalization/scoring as Stage8 adapter evaluation.
    original = np.stack([2.5 * np.maximum(np.einsum('ij,ij->i', vectors[0], text), 0)
                         for text in vectors[1:]], axis=1)
    frozen = {r['id'] for r in read(baseline / 'stage5/frozen_manifest.json')['records']}
    with np.load(output / 'adapter_evaluation/adapter_sugar_scores.npz', allow_pickle=False) as archive:
        if archive['record_ids'].tolist() != [r['id'] for r in records]:
            raise ValueError('Sugar score record order differs from manifest')
        predictions = {'gemma-original-literal768': original}
        selection = read(output / 'adapter_training/selection.json')
        for run, pairs in zip(selection['finalists'], archive['all_seed_pairs'], strict=True):
            predictions['gemma-adapter-seed' + str(run['seed'])] = pairs
        predictions['gemma-adapter-ensemble5'] = np.stack(
            [archive['ensemble_positive'], archive['ensemble_negative']], axis=1)
    unseen = np.array([not r['previously_seen'] for r in records])
    cohorts = {'full7511': np.ones(len(records), dtype=bool),
               'development3562': ~unseen, 'historical_unseen3949': unseen,
               'historical560': np.array([r['id'] in frozen for r in records])}
    rows, score_rows = [], []
    for method, pairs in predictions.items():
        correct = pairs[:, 0] > pairs[:, 1]  # Ties count as wrong, as before.
        for cohort, mask in cohorts.items():
            categories = ['all', *sorted({r['category'] for r, keep in zip(records, mask) if keep})]
            for category in categories:
                selected = mask if category == 'all' else mask & np.array([r['category'] == category for r in records])
                rows.append({'cohort': cohort, 'method': method, 'category': category,
                             'n_images': len({r['filename'] for r, keep in zip(records, selected) if keep}),
                             'n_pairs': int(selected.sum()), 'correct': int(correct[selected].sum()),
                             'accuracy': float(correct[selected].mean()),
                             'score_ties': int((pairs[selected, 0] == pairs[selected, 1]).sum())})
        for record, pair, good in zip(records, pairs, correct):
            score_rows.append({'id': record['id'], 'filename': record['filename'],
                               'category': record['category'], 'previously_seen': record['previously_seen'],
                               'method': method, 'positive_score': float(pair[0]),
                               'negative_score': float(pair[1]), 'correct': int(good)})
    write_csv(output / 'sugar_metrics.csv', rows)
    write_csv(output / 'sugar_pair_scores.csv', score_rows)
    write(output / 'sugar_metrics.json', {
        'scope': 'Historical observed cohorts; full/development are descriptive. '
                 '560 is a subset of unseen3949, not another independent test.',
        'original_scoring': 'Stage8 float32 vectors, 2.5*max(cosine,0)', 'rows': rows})
    return [r for r in rows if r['category'] == 'all']


def check_human_tables(output):
    cohorts = read(output / 'all_metrics.json')['cohorts']
    for cohort, counts in (('full', (1000, 5664, 16992)), ('unseen900', (900, 5112, 15336))):
        table = cohorts[cohort]
        if len(table) != 47 or not {'PAC-S-ViTB32', 'gemma-original-literal768',
                                   'gemma-adapter-ensemble5', 'BLEU1', 'BLEU2', 'BLEU3',
                                   'BLEU4', 'ROUGE_L', 'CIDEr', 'METEOR'} <= table.keys():
            raise ValueError(f'Incomplete expanded metric table for {cohort}')
        for method, metrics in table.items():
            if tuple(metrics[k] for k in ('n_images', 'n_captions', 'n_individual_ratings')) != counts:
                raise ValueError(f'Unexpected cohort counts: {cohort}/{method}')
    return cohorts


def compare_historical_human(old, new):
    differences = []
    for cohort in ('full', 'unseen'):
        prior = old['cohorts'][cohort]['metrics']
        current = new['cohorts'][cohort]['metrics']
        if set(prior) != set(current) or len(current) != 32:
            raise ValueError('Expected all 32 historical human configurations')
        differences.extend(abs(current[k]['kendall_tau_c'] - prior[k]['kendall_tau_c']) for k in prior)
    maximum = max(differences)
    result = {'n_cohort_method_checks': len(differences), 'max_abs_tau_c_difference': maximum,
              'passed': maximum < 1e-12}
    if not result['passed']:
        raise ValueError(f'Recomputed baseline disagrees with original FULL results: {result}')
    return result


def run(baseline, output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'pipeline_status.json').exists():
        raise FileExistsError('Use a fresh output directory; refusing to mix rerun artifacts')
    (output / 'logs').mkdir(exist_ok=True)
    execution = Run(output)
    try:
        counts = validate_inputs(baseline)
        if digest(ROOT / 'protocol.json') != PROTOCOL_SHA256:
            raise ValueError('Stage8 protocol changed; this rerun requires the historical recipe')
        if digest(baseline / 'stage5/frozen_manifest.json') != 'a2e9fa5fb7e9866631c0546e5a09a2f9d11044335267712591476572b6104efe':
            raise ValueError('SugarCrepe560 manifest hash differs from the historical protocol')
        if digest(baseline / 'stage7/human_manifest.json') != '0e0a96c3659b565e7675bb41fbcde161a4e39617b7a9fd337174aac0018b9fa3':
            raise ValueError('Flickr8kExpert manifest hash differs from the historical protocol')
        write(output / 'input_receipts.json', {'counts': counts,
              'sha256': {p: digest(baseline / p) for p in required_inputs()}})
        shutil.copyfile(ROOT / 'protocol.json', output / 'protocol.json')
        write(output / 'source_sha256.json', {str(p.relative_to(ROOT.parent)): digest(p)
              for p in sorted(ROOT.glob('*.py'))})
        old_human = read(baseline / 'stage7/human_results.json')
        import importlib.metadata
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('This entrypoint requires CUDA; no CPU fallback')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        write(output / 'runtime.json', {
            'python': sys.version, 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'versions': {k: importlib.metadata.version(k) for k in (
                'numpy', 'scipy', 'scikit-learn', 'pycocoevalcap', 'transformers',
                'sentence-transformers', 'pyarrow', 'torchvision', 'gdown')},
            'precision': 'float32; TF32 disabled; GPU0',
            'baseline_mode': 'Full re-evaluation of frozen cached encoder vectors; no fresh full encoder inference'})

        execution.script('stage8_verify', ROOT / 'verify.py', '--baseline', baseline)
        # Only data(), not download_required.py's main (which also downloads unused model weights).
        execution.step('expert_download', [sys.executable, '-c',
                       'import sys; sys.path.insert(0, sys.argv[1]); '
                       'from download_required import data; data()', str(baseline / 'stage7')])
        execution.script('expert_materialize', baseline / 'stage7/materialize_expert.py')

        artifacts = [
            ('baseline_sugar_full', baseline / FULL_REL / 'analyze.py', [],
             [('results.json', 'baseline_sugar_full.json'), ('pair_scores.csv', 'baseline_sugar_pair_scores.csv')]),
            ('baseline_sugar560', baseline / 'stage5/analyze_560.py', [],
             [('results_560.json', 'baseline_sugar560.json'), ('pair_scores_560.csv', 'baseline_sugar560_pair_scores.csv')]),
            ('baseline_sugar560_verify', baseline / 'stage5/verify.py', ['--scores-only'],
             [('independent_scores_verification.json', 'baseline_sugar560_verification.json')]),
            ('baseline_confirmation', baseline / 'stage6/analyze_confirmation.py', [],
             [('verified_confirmation_results.json', 'baseline_confirmation.json'),
              ('pair_scores_confirmation.csv', 'baseline_confirmation_pair_scores.csv')]),
            ('baseline_human', baseline / 'stage7/analyze_human.py', [],
             [('human_results.json', 'baseline_human.json'), ('human_scores.csv', 'baseline_human_scores.csv')]),
            ('baseline_human_verify', baseline / 'stage7/verify_completion.py', [],
             [('independent_verification.json', 'baseline_human_verification.json')]),
        ]
        for name, script, params, exports in artifacts:
            execution.script(name, script, *params)
            for original, destination in exports:
                shutil.copyfile(script.parent / original, output / destination)
        reproduction = compare_historical_human(old_human, read(output / 'baseline_human.json'))
        write(output / 'historical_human_comparison.json', reproduction)

        execution.script('reference_metrics', ROOT / 'reference_metrics.py',
                         '--baseline', baseline, '--output', output / 'references')
        execution.script('train_adapter', ROOT / 'train_adapter.py',
                         '--baseline', baseline, '--output', output / 'adapter_training', '--device', 'cuda')
        selection = read(output / 'adapter_training/selection.json')
        check_selection(selection)
        execution.script('evaluate_adapter', ROOT / 'evaluate_adapter.py', '--baseline', baseline,
                         '--trained', output / 'adapter_training', '--output', output / 'adapter_evaluation',
                         '--device', 'cuda')
        execution.script('expanded_metrics', ROOT / 'metrics.py', '--baseline', baseline,
                         '--extra-scores', output / 'adapter_evaluation/adapter_human_scores.json',
                         '--output', output / 'expanded_metrics.json')
        # PAC-S is mandatory for this run, independent of whether a benchmark is requested.
        execution.script('pacscore', ROOT / 'pacscore_gpu.py', '--baseline', baseline,
                         '--output', output / 'pacscore')
        execution.script('combine_tables', ROOT / 'combine_tables.py',
                         '--expanded', output / 'expanded_metrics.json',
                         '--references', output / 'references/reference_metrics.json',
                         '--pac', output / 'pacscore/pac_metrics.json', '--output', output / 'all_metrics.json')
        execution.script('adapter_uncertainty', ROOT / 'uncertainty.py', '--baseline', baseline,
                         '--evaluation', output / 'adapter_evaluation', '--output', output / 'adapter_uncertainty.json')
        execution.script('pac_comparison', ROOT / 'compare_pac.py', '--baseline', baseline,
                         '--pac-scores', output / 'pacscore/pac_scores.json', '--output', output / 'pac_comparison.json')
        execution.script('fresh_gpu_benchmark', ROOT / 'benchmark_gpu.py', '--baseline', baseline,
                         '--trained', output / 'adapter_training', '--output', output / 'fresh_gpu_benchmark.json')
        cohorts = check_human_tables(output)
        sugar = export_sugar(baseline, output)
        benchmark = read(output / 'fresh_gpu_benchmark.json')
        if not benchmark['completed']:
            raise ValueError('GPU benchmark incomplete')
        latency_rows = []
        for model, measurements in benchmark['measurements'].items():
            for measurement in measurements:
                latency_rows.append({'model': model, **{k: v for k, v in measurement.items()
                                                        if k != 'samples_wall_ms'}})
        write_csv(output / 'gpu_benchmark.csv', latency_rows)
        primary_methods = ('gemma-original-literal768', 'gemma-adapter-ensemble5', 'PAC-S-ViTB32')
        write(output / 'summary.json', {
            'complete': True, 'scope': read(ROOT / 'protocol.json')['scope'],
            'metric_definition_source': 'EmbeddingGemma2_Stage8_Report_VI.md: Metric mở rộng / Baseline dùng reference',
            'human_named_configurations': 47, 'human_table_rows': 94,
            'cohort_counts': counts, 'adapter_selection': selection['selected'],
            'historical_human_reproduction': reproduction, 'backbone_frozen': True,
            'sugar_original_and_adapter': sugar,
            'human_primary': {c: {k: table[k] for k in primary_methods} for c, table in cohorts.items()},
            'spice': 'Not run: optional in Stage8 protocol and absent from historical all_metrics.csv',
            'outputs': ['all_metrics.csv', 'sugar_metrics.csv', 'gpu_benchmark.csv',
                        'adapter_uncertainty.json', 'pac_comparison.json', 'logs/']})
        execution.state.update(complete=True, active_step=None)
        execution.save()
    except Exception as error:
        execution.state.update(complete=False, error=str(error))
        execution.save()
        (output / 'logs/failure.log').write_text(traceback.format_exc())
        raise
    finally:
        write(output / 'output_sha256.json', {str(p.relative_to(output)): digest(p)
              for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'output_sha256.json'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--output', default='/kaggle/working/stage8_full_rerun')
    parser.add_argument('--validate-inputs-only', action='store_true',
                        help='Check private inputs without importing torch, downloading, or training')
    args = parser.parse_args()
    baseline = Path(args.baseline).resolve()
    if args.validate_inputs_only:
        print(json.dumps(validate_inputs(baseline), indent=2))
        return
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    run(baseline, Path(args.output).resolve())


if __name__ == '__main__':
    main()
