"""Download only research outputs, with paginated metadata and concurrent transfers."""
import argparse,concurrent.futures,os,re
from pathlib import Path
import requests
from kaggle_control import auth_env
from common import write,digest,read

def main():
    p=argparse.ArgumentParser();p.add_argument('--credential-file');p.add_argument('--kernel',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    env,secret=auth_env(a.credential_file);os.environ.update({k:v for k,v in env.items() if k.startswith('KAGGLE_')})
    from kaggle import api
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    api.authenticate();owner,slug,_=api.parse_kernel_string(a.kernel);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    selected=[];token=None;log='';pages=0
    while True:
        with api.build_kaggle_client() as client:
            request=ApiListKernelSessionOutputRequest();request.user_name=owner;request.kernel_slug=slug
            api._set_paging(request,200,token)
            response=client.kernels.kernels_api_client.list_kernel_session_output(request)
        pages+=1;log=log or response.log or ''
        for item in response.files or []:
            n=item.file_name
            if n=='runtime.json' or (n.startswith('stage8_results/') and re.search(r'\.(json|csv|npz|log)$',n)):
                assert not Path(n).is_absolute() and '..' not in Path(n).parts
                selected.append((n,item.url))
        token=response.next_page_token
        if not token:break
    print('Selected',len(selected),'research outputs across',pages,'metadata pages',flush=True)
    priority={'fresh_gpu_benchmark.json','pac_metrics.json','pipeline_status.json','pacscore_status.json','runtime.json','selection.json','output_sha256.json'}
    selected.sort(key=lambda item:(Path(item[0]).name not in priority,item[0]))
    def transfer(item):
        name,url=item;dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True)
        with requests.get(url,stream=True,timeout=60) as r:
            r.raise_for_status()
            with dest.with_suffix(dest.suffix+'.part').open('wb') as f:
                for chunk in r.iter_content(1024*1024):f.write(chunk)
        dest.with_suffix(dest.suffix+'.part').replace(dest);return name
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        done=list(pool.map(transfer,selected))
    (out/(slug+'.log')).write_text(log.replace(secret,'[REDACTED]'))
    receipt=read(out/'stage8_results/output_sha256.json');verified=[]
    for name in done:
        if name.startswith('stage8_results/'):
            relative=name.removeprefix('stage8_results/')
            if relative in receipt:assert digest(out/name)==receipt[relative],name;verified.append(name)
    write(out/'download_verification.json',{'downloaded_files':len(done),'verified_against_remote_sha256':len(verified),'complete':True,'metadata_pages':pages,'large_encoder_weights_and_raw_images_downloaded':False})
    print('Downloaded and verified',len(verified),'files',flush=True)

if __name__=='__main__':main()
