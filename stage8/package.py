"""Package only an explicit artifact allowlist; never auth/config/environment files."""
import argparse,json,zipfile
from pathlib import Path
from common import digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline-zip',required=True);p.add_argument('--report',required=True);p.add_argument('--full',required=True);p.add_argument('--source',required=True);p.add_argument('--secret-check-file');a=p.parse_args()
    root=Path(__file__).parent.resolve();report=Path(a.report).resolve()
    files=[(f,'stage8/'+f.name) for pattern in ['*.py','*.json','*.txt','README_VI.md'] for f in root.glob(pattern)]
    files += [(f,'stage8/kaggle_kernel/'+f.name) for f in (root/'kaggle_kernel').glob('*') if f.is_file()]
    skip={'kaggle_v2','kaggle_v2_summary','kaggle_v2_logs'}
    files += [(f,'stage8/results/'+str(f.relative_to(root/'results'))) for f in (root/'results').rglob('*') if f.is_file() and '__pycache__' not in f.parts and not(skip&set(f.relative_to(root/'results').parts)) and f.suffix not in {'.pth','.bin','.safetensors','.parquet','.part'}]
    files.append((report,report.name));names=[n for _,n in files];assert len(names)==len(set(names))
    if a.secret_check_file:
        secret=Path(a.secret_check_file).read_text().strip().encode();assert len(secret)>20
        for f,n in files:assert secret not in f.read_bytes(),n
        # Inspect embedded source snapshots too; their compressed bytes alone are insufficient.
        for f,n in files:
            if f.suffix=='.zip':
                with zipfile.ZipFile(f) as z:assert all(secret not in z.read(item) for item in z.namelist())
    manifests={n:{'sha256':digest(f),'size_bytes':f.stat().st_size} for f,n in files}
    for destination,full in [(a.source,False),(a.full,True)]:
        with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for f,n in files:z.write(f,n)
            manifest=manifests.copy()
            if full:
                baseline=Path(a.baseline_zip);z.write(baseline,'baseline.zip',compress_type=zipfile.ZIP_STORED)
                manifest['baseline.zip']={'sha256':digest(baseline),'size_bytes':baseline.stat().st_size}
            z.writestr('STAGE8_SHA256.json',json.dumps(manifest,indent=2))
        print(json.dumps({'path':str(Path(destination).resolve()),'bytes':Path(destination).stat().st_size,'sha256':digest(destination)}))

if __name__=='__main__':main()
