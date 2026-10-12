"""Kaggle-only bootstrap; no Kaggle API or CLI calls."""
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from rerun_inputs import restore_baseline


def main():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    output = Path('/kaggle/working/stage8_full_rerun')
    output.mkdir(parents=True, exist_ok=True)
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    source_root = Path(__file__).resolve().parent.parent
    scratch = source_root.parent
    os.environ['HF_HOME'] = str(scratch / 'hf_cache')
    os.environ['TORCH_HOME'] = str(scratch / 'torch_cache')
    status = {'complete': False, 'active_step': 'restore_private_baseline'}

    def save():
        (output / 'bootstrap_status.json').write_text(json.dumps(status, indent=2) + '\n')

    def logged(name, command):
        status['active_step'] = name
        save()
        print(f'START {name}', flush=True)
        with (logs / (name + '.log')).open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f'{name} failed (exit {result.returncode}); see logs/{name}.log')

    save()
    try:
        baseline = restore_baseline('/kaggle/input', scratch / 'private_baseline')
        # Run source from this kernel, rather than trusting whatever source accompanied the caches.
        for path in sorted((source_root / 'baseline').rglob('*.py')):
            destination = baseline / path.relative_to(source_root / 'baseline')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        (output / 'embedded_source_sha256.json').write_bytes((source_root / 'source_sha256.json').read_bytes())

        # The resolver must fail on an incompatible dependency rather than replacing
        # Kaggle's CUDA stack with a different torch/torchvision wheel pair.
        constraints = scratch / 'cuda_stack_constraints.txt'
        pinned = []
        for package in ('torch', 'torchvision', 'torchaudio'):
            try:
                pinned.append(package + '==' + importlib.metadata.version(package))
            except importlib.metadata.PackageNotFoundError:
                if package != 'torchaudio':
                    raise RuntimeError(f'Kaggle CUDA environment is missing {package}')
        constraints.write_text('\n'.join(pinned) + '\n')
        (output / 'cuda_stack_constraints.txt').write_bytes(constraints.read_bytes())
        logged('install_dependencies', [sys.executable, '-m', 'pip', 'install',
                                       '-r', str(source_root / 'stage8/requirements_kaggle.txt'),
                                       '--constraint', str(constraints)])
        # Keep Kaggle's CUDA torch/torchvision pair. Never install CPU torch into the kernel.
        logged('environment_check', [sys.executable, '-c',
               'import shutil, torch, torchvision; '
               'assert torch.cuda.is_available(), "CUDA GPU required"; '
               'assert shutil.which("java"), "Java required for PTBTokenizer/METEOR"; '
               'print("torch", torch.__version__, "torchvision", torchvision.__version__, '
               '"GPU", torch.cuda.get_device_name(0))'])
        # Gemma may require gated-model access. A configured HF_TOKEN secret is read at runtime only.
        if not os.environ.get('HF_TOKEN'):
            try:
                from kaggle_secrets import UserSecretsClient
                os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
            except Exception:
                pass  # Public model downloads can work without a token; errors are logged by the evaluator.
        logged('full_rerun', [sys.executable, str(source_root / 'stage8/full_rerun.py'),
                             '--baseline', str(baseline), '--output', str(output)])
        status.update(complete=True, active_step=None)
    except Exception as error:
        status.update(complete=False, error=str(error))
        (logs / 'bootstrap_failure.log').write_text(traceback.format_exc())
        raise
    finally:
        save()
        # Include bootstrap logs/status written after the research pipeline's own final manifest.
        import hashlib
        hashes = {}
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.name != 'output_sha256.json':
                sha = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        sha.update(chunk)
                hashes[str(path.relative_to(output))] = sha.hexdigest()
        (output / 'output_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')


if __name__ == '__main__':
    main()
