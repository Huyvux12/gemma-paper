"""Kaggle GPU: verified pixels -> native training/test -> report -> download ZIP."""
import argparse
import os
import platform
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

for key,value in {'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8',
                  'HF_HUB_DISABLE_TELEMETRY':'1','HF_HUB_DISABLE_XET':'1','HF_HUB_DOWNLOAD_TIMEOUT':'60','HF_HUB_ETAG_TIMEOUT':'30'}.items():
    os.environ.setdefault(key,value)

from bundle import CODE_ROOT,native_inputs,read,sha,verify_code,write


def run(data,output,timing=True):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('This route needs a CUDA GPU. Use replay.py/refit.py for CPU.')
    output=Path(output)
    if output.exists():raise FileExistsError('Choose a new output directory; original results are never overwritten')
    output=output.resolve()
    if output.with_name(output.name+'_results.zip').exists():
        raise FileExistsError('Choose a new run name; results ZIP already exists')
    extraction=(Path('/kaggle/temp') if Path('/kaggle/temp').exists() else output.parent)/(output.name+'_inputs')
    pins=verify_code()
    output.mkdir(parents=True)
    from native import Encoder as OriginalEncoder
    class FP32Encoder(OriginalEncoder):
        def __init__(self,kind,config,folder,dtype_override=None):
            if kind!='gemma':raise ValueError('Gemma-only entrypoint')
            super().__init__(kind,config,folder,dtype_override=torch.float32)
            if self.receipt['weights']!=pins['gemma_weights']:raise ValueError('Author weight identity changed')
    import run as original_run
    original_run.Encoder=FP32Encoder
    from common import seal
    started=time.monotonic();state={'complete':False,'phase':'verify_inputs','started_utc':datetime.now(timezone.utc).isoformat()}
    def phase(name):
        state.update(phase=name,elapsed_seconds=time.monotonic()-started)
        write(output/'execution_status.json',state);print('GEMMA_PHASE',name,flush=True)
    phase('verify_inputs')
    try:
        data=native_inputs(data,extraction)
        write(output/'rerun_receipt.json',{'frozen_sources_verified':True,'entrypoint_sha256':sha(__file__),
              'native_input_manifest_sha256':sha(data/'manifest_sha256.json'),'precision':'Gemma FP32 fixed',
              'engineering_override':'Inject explicit FP32 Encoder without editing frozen source',
              'model':pins['gemma_model'],'original_primary_replaced':False,
              'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,
              'platform':platform.platform(),'new_native_encoding':True,'new_training':True})
        with zipfile.ZipFile(output/'execution_code.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for folder in ['src','vendor/stage11','report_context']:
                for path in sorted((CODE_ROOT/folder).glob('*')):
                    if path.is_file() and path.suffix in ['.py','.json','.csv']:
                        archive.write(path,path.relative_to(CODE_ROOT).as_posix())
            for name in ['PINNED.json','requirements_gpu.txt']:
                archive.write(CODE_ROOT/name,name)
        core=output/'core';core.mkdir()
        phase('native_encoding_validation_refit_test')
        write(core/'bootstrap_status.json',{'complete':False})
        original_run.main(data,core,['gemma'])
        write(core/'bootstrap_status.json',{'complete':True});seal(core)
        destination=None
        if timing:
            phase('timing128')
            import representative_timing
            representative_timing.Encoder=FP32Encoder
            destination=output/'timing128';destination.mkdir()
            write(destination/'bootstrap_status.json',{'complete':False})
            representative_timing.main(data,core,destination,sha(data/'manifest_sha256.json'),sha(core/'manifest_sha256.json'))
            write(destination/'bootstrap_status.json',{'complete':True});seal(destination)
        phase('audit_and_report')
        from report import build_report
        build_report(core,output/'report',destination,evidence_label='new_native_gpu_run')
        state.update(complete=True,report='report/REPORT_VI.html',finished_utc=datetime.now(timezone.utc).isoformat())
        phase('complete');seal(output)
        archive_path=output.with_name(output.name+'_results.zip')
        if archive_path.exists():raise FileExistsError('Results ZIP already exists')
        with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for path in sorted(output.rglob('*')):
                if path.is_file():archive.write(path,output.name+'/'+path.relative_to(output).as_posix())
        write(archive_path.with_suffix('.zip.sha256.json'),{'sha256':sha(archive_path),'bytes':archive_path.stat().st_size,
              'scope':'Complete new run folder: core embeddings/checkpoints/predictions, timing, reports, code; original input images/model weights excluded'})
        print('REPORT_READY',output/'report/REPORT_VI.html',flush=True)
        print('DOWNLOAD_ZIP',archive_path,flush=True)
    except Exception as exc:
        state.update(complete=False,failed_phase=state['phase'],error_type=type(exc).__name__,elapsed_seconds=time.monotonic()-started)
        write(output/'execution_status.json',state)
        raise
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data');p.add_argument('--output',required=True);p.add_argument('--skip-timing',action='store_true')
    a=p.parse_args();print(run(a.data,a.output,not a.skip_timing))
