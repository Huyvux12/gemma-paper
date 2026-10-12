"""Local stdlib/NumPy checks: no downloads, training, Kaggle CLI, or GPU work."""
import ast
import base64
import hashlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from full_rerun import Run, check_human_tables, check_selection, compare_historical_human, export_sugar
from prepare_full_rerun import ROOT, build
from rerun_inputs import FULL_REL, required_inputs, restore_baseline, safe_extract, validate_inputs


def fake_full(root):
    """Synthetic records only; no dataset artifacts from the real handoff."""
    for name in required_inputs():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    sugar = []
    for seen, images, pairs in ((True, 555, 3562), (False, 1005, 3949)):
        for i in range(pairs):
            sugar.append({'id': f'{seen}-{i}', 'filename': f'{seen}-{i % images}.jpg',
                          'previously_seen': seen, 'category': 'replace_obj'})
    human = []
    for seen, images, captions in ((True, 100, 552), (False, 900, 5112)):
        for i in range(captions):
            human.append({'id': f'{seen}-{i}', 'filename': f'{seen}-{i % images}.jpg',
                          'previously_seen': seen, 'ratings': [1, 2, 3]})
    for name, records in ((f'{FULL_REL}/manifest.json', sugar),
                          ('stage7/human_manifest.json', human),
                          ('stage5/frozen_manifest.json', [r for r in sugar if not r['previously_seen']][:560])):
        (root / name).write_text(json.dumps({'records': records}))


class InputTests(unittest.TestCase):
    def test_zip_and_auto_extracted_layouts_with_wrapped_roots(self):
        for zipped, prefix in ((True, ''), (True, 'baseline/'), (False, 'expanded/baseline/')):
            with self.subTest(zipped=zipped, prefix=prefix), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                source = folder / 'fixture'
                fake_full(source)
                inputs = folder / 'inputs'
                inputs.mkdir()
                if zipped:
                    with zipfile.ZipFile(inputs / 'baseline.zip', 'w') as archive:
                        for path in source.rglob('*'):
                            if path.is_file():
                                archive.write(path, prefix + str(path.relative_to(source)))
                else:
                    import shutil
                    shutil.copytree(source, inputs / prefix)
                baseline = restore_baseline(inputs, folder / 'restored')
                self.assertEqual(validate_inputs(baseline)['human_unseen_images'], 900)

    def test_missing_cache_and_wrong_cohort_fail_before_ml(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake_full(root)
            path = root / 'stage7/runs/gemma-human/embeddings.npz'
            path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'gemma-human/embeddings.npz'):
                validate_inputs(root)
            path.write_text('{}')
            (root / 'stage5/frozen_manifest.json').write_text('{"records": []}')
            with self.assertRaisesRegex(ValueError, 'cohort counts differ'):
                validate_inputs(root)

    def test_ambiguous_attachment_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake_full(root / 'inputs/one')
            fake_full(root / 'inputs/two')
            with self.assertRaisesRegex(ValueError, 'Multiple FULL'):
                restore_baseline(root / 'inputs', root / 'work')

    def test_archive_traversal_and_symlink_rejected_before_extraction(self):
        import stat
        for entry in ('../escaped', '/absolute', 'nested/../../escaped', 'nested\\escaped', 'symlink'):
            with self.subTest(entry=entry), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                info = zipfile.ZipInfo(entry)
                if entry == 'symlink':
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                with zipfile.ZipFile(root / 'input.zip', 'w') as archive:
                    archive.writestr('safe', 'would be extracted first')
                    archive.writestr(info, 'payload')
                with self.assertRaisesRegex(ValueError, 'Unsafe ZIP'):
                    safe_extract(root / 'input.zip', root / 'output')
                self.assertFalse((root / 'output/safe').exists())


class OrchestrationTests(unittest.TestCase):
    def test_sugar_export_keeps_full_unseen_subset_and_score_ties_separate(self):
        import numpy as np
        from unittest.mock import patch
        records = [
            {'id': 'dev', 'filename': 'dev.jpg', 'category': 'replace_obj', 'previously_seen': True},
            {'id': 'frozen', 'filename': 'one.jpg', 'category': 'replace_obj', 'previously_seen': False},
            {'id': 'extra', 'filename': 'one.jpg', 'category': 'replace_rel', 'previously_seen': False},
        ]
        # The frozen pair has two negative cosines: clipping creates a tie (wrong).
        image = np.array([[1., 0.]] * 3, dtype=np.float32)
        positive = np.array([[1., 0.], [-.2, 0.], [.4, 0.]], dtype=np.float32)
        negative = np.array([[0., 1.], [-.4, 0.], [.8, 0.]], dtype=np.float32)
        seeds = [11, 23, 37, 53, 71]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline, output = root / 'baseline', root / 'output'
            (baseline / 'stage5').mkdir(parents=True)
            (baseline / 'stage5/frozen_manifest.json').write_text('{"records": [{"id": "frozen"}]}')
            (output / 'adapter_training').mkdir(parents=True)
            (output / 'adapter_evaluation').mkdir()
            (output / 'adapter_training/selection.json').write_text(json.dumps({
                'finalists': [{'seed': s} for s in seeds]}))
            pairs = np.array([[1., 0.], [.3, .2], [.4, .4]], dtype=np.float32)
            np.savez(output / 'adapter_evaluation/adapter_sugar_scores.npz',
                     record_ids=np.array([r['id'] for r in records]),
                     all_seed_pairs=np.stack([pairs] * 5), ensemble_positive=pairs[:, 0],
                     ensemble_negative=pairs[:, 1])
            with patch('common.load_gemma', return_value=({'records': records}, [image, positive, negative])):
                summary = export_sugar(baseline, output)
            indexed = {(r['cohort'], r['method']): r for r in summary}
            original = indexed['historical560', 'gemma-original-literal768']
            self.assertEqual((original['correct'], original['score_ties']), (0, 1))
            full = indexed['full7511', 'gemma-original-literal768']
            self.assertEqual((full['n_images'], full['n_pairs'], full['correct']), (2, 3, 1))
            unseen = indexed['historical_unseen3949', 'gemma-adapter-ensemble5']
            self.assertEqual((unseen['n_images'], unseen['n_pairs'], unseen['correct']), (1, 2, 1))
            self.assertTrue((output / 'sugar_pair_scores.csv').is_file())

    def test_failed_step_retains_log_and_never_reports_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'logs').mkdir()
            execution = Run(root)
            with self.assertRaisesRegex(RuntimeError, 'pacscore failed'):
                execution.step('pacscore', [sys.executable, '-c', 'print("download failed"); raise SystemExit(7)'])
            status = json.loads((root / 'pipeline_status.json').read_text())
            self.assertFalse(status['complete'])
            self.assertEqual(status['failed_step'], 'pacscore')
            self.assertEqual(status['exit_code'], 7)
            self.assertIn('download failed', (root / 'logs/pacscore.log').read_text())

    def test_validation_selection_cannot_silently_change_recipe(self):
        selection = {'selected': {'rank': 32, 'anchor_weight': 1.0}, 'backbone_frozen': True,
                     'selection_uses_historical_benchmark_or_human_scores': False,
                     'finalists': [{'rank': 32, 'anchor_weight': 1.0, 'seed': s} for s in (11, 23, 37, 53, 71)]}
        check_selection(selection)
        selection['selected']['rank'] = 16
        with self.assertRaisesRegex(ValueError, 'stopping instead'):
            check_selection(selection)

    def test_incomplete_human_table_is_not_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'all_metrics.json').write_text(json.dumps({'cohorts': {'full': {}, 'unseen900': {}}}))
            with self.assertRaisesRegex(ValueError, 'Incomplete expanded'):
                check_human_tables(root)

    def test_historical_comparison_uses_original_results(self):
        old = {'cohorts': {c: {'metrics': {str(i): {'kendall_tau_c': 0.5} for i in range(32)}}
                           for c in ('full', 'unseen')}}
        new = json.loads(json.dumps(old))
        self.assertEqual(compare_historical_human(old, new)['n_cohort_method_checks'], 64)
        new['cohorts']['unseen']['metrics']['0']['kendall_tau_c'] = 0.4
        with self.assertRaisesRegex(ValueError, 'disagrees with original'):
            compare_historical_human(old, new)


class PackagingTests(unittest.TestCase):
    def test_kernel_embeds_exact_source_and_never_embeds_private_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metadata = build(root)
            self.assertTrue(metadata['is_private'] and metadata['enable_gpu'] and metadata['enable_internet'])
            script = (root / metadata['code_file']).read_text()
            ast.parse(script)
            assignments = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(script).body
                           if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
            payload = base64.b64decode(assignments['PAYLOAD'])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), assignments['PAYLOAD_SHA256'])
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                for name in archive.namelist():
                    self.assertTrue(name.endswith(('.py', '.json', '.txt')))
                    if name != 'source_sha256.json':
                        self.assertEqual(archive.read(name), (ROOT / name).read_bytes())
                    if name.endswith('.py'):
                        ast.parse(archive.read(name).decode())
                self.assertEqual(archive.read('stage8/train_adapter.py'), (ROOT / 'stage8/train_adapter.py').read_bytes())
                self.assertNotIn('stage8/kaggle_control.py', archive.namelist())
                self.assertNotIn('stage8/download_results.py', archive.namelist())
                self.assertNotIn('stage8/prepare_kaggle.py', archive.namelist())
            first = (root / metadata['code_file']).read_bytes()
            build(root)
            self.assertEqual(first, (root / metadata['code_file']).read_bytes())
            committed = ROOT / 'stage8/kaggle_full_rerun/run_full_rerun.py'
            self.assertEqual(first, committed.read_bytes(), 'Regenerate the committed kernel after changing source')


if __name__ == '__main__':
    unittest.main()
