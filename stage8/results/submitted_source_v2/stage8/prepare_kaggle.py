"""Build private dataset metadata and a self-contained notebook without credentials."""
import argparse,base64,hashlib,io,json,os,shutil,zipfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline-zip',required=True);p.add_argument('--username',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parent;ds=root/'kaggle_dataset';kd=root/'kaggle_kernel';ds.mkdir(exist_ok=True);kd.mkdir(exist_ok=True)
    protocol_sha=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()
    ref=f'{a.username}/embeddinggemma-stage8-20261009-{protocol_sha[:8]}'
    target=ds/'baseline.zip'
    if not target.exists():shutil.copyfile(Path(a.baseline_zip).resolve(),target)
    (ds/'dataset-metadata.json').write_text(json.dumps({'title':'EmbeddingGemma Stage8 private research inputs','id':ref,'licenses':[{'name':'other'}]},indent=2))
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(root.glob('*.py')):z.write(file,'stage8/'+file.name)
        z.write(root/'protocol.json','stage8/protocol.json')
    payload=base64.b64encode(buffer.getvalue()).decode()
    bootstrap=f'''import os,sys,subprocess,zipfile,io,base64,json,hashlib,shutil
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['HF_HUB_DISABLE_XET']='1'
work=Path('/kaggle/working')
archives=list(Path('/kaggle/input').rglob('baseline.zip'))
if archives:
    with zipfile.ZipFile(archives[0]) as z:
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
        z.extractall(work/'baseline')
else:
    manifests=[p for p in Path('/kaggle/input').rglob('human_manifest.json') if p.parent.name=='stage7']
    assert manifests, 'Private baseline dataset is attached but expected manifest is absent'
    shutil.copytree(manifests[0].parent.parent,work/'baseline',dirs_exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))) as z:z.extractall(work)
subprocess.run([sys.executable,'-m','pip','install','numpy==2.3.5','scipy==1.17.0','scikit-learn','pycocoevalcap==1.2','transformers==5.19.0','sentence-transformers==6.1.0','ftfy','regex','Pillow','pyarrow==25.0.1','gdown'],check=True)
import torch
assert torch.cuda.is_available(), 'CUDA required'
(work/'runtime.json').write_text(json.dumps({{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'python':sys.version}},indent=2))
subprocess.run([sys.executable,str(work/'stage8/verify.py'),'--baseline',str(work/'baseline')],check=True)
subprocess.run([sys.executable,str(work/'stage8/pipeline.py'),'--baseline',str(work/'baseline'),'--output',str(work/'stage8_results'),'--device','cuda','--gpu-benchmark'],check=True)
'''
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[{'cell_type':'markdown','id':'purpose','metadata':{},'source':['# Stage8: exploratory metrics and frozen-backbone adapters\nPrivate research run. Prior benchmarks have been observed; this run is not a fresh confirmatory test.']},{'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':bootstrap.splitlines(keepends=True)}]}
    (kd/'run_stage8.ipynb').write_text(json.dumps(notebook))
    metadata={'id':f'{a.username}/embeddinggemma-stage8-20261009','title':'EmbeddingGemma Stage8 20261009','code_file':'run_stage8.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'dataset_sources':[ref],'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (kd/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps({'dataset':ref,'kernel':metadata['id'],'embedded_source_bytes':len(buffer.getvalue()),'protocol_sha256':protocol_sha}))

if __name__=='__main__':main()
