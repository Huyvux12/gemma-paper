"""Credential-aware CLI wrapper. Secret never goes into notebook/data artifacts."""
import argparse,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
from common import write,read

def auth_env(path):
    env=os.environ.copy();secret=None
    if path:
        text=Path(path).read_text().strip()
        if text.startswith('{'):
            j=json.loads(text);env['KAGGLE_USERNAME']=j['username'];env['KAGGLE_KEY']=j['key'];secret=j['key']
        else:env['KAGGLE_API_TOKEN']=text;secret=text
    elif env.get('KAGGLE_API_TOKEN'):secret=env['KAGGLE_API_TOKEN']
    else:raise RuntimeError('Provide --credential-file or KAGGLE_API_TOKEN. No credential is bundled.')
    return env,secret

def cli(args,env,secret,timeout=900):
    executable=shutil.which('kaggle')
    if executable is None:
        sibling=Path(sys.executable).parent/'kaggle'
        executable=str(sibling) if sibling.exists() else None
    assert executable,'Install kaggle==2.2.4 first'
    r=subprocess.run([executable,*args],env=env,capture_output=True,text=True,timeout=timeout)
    output=(r.stdout+r.stderr).replace(secret,'[REDACTED]') if secret else r.stdout+r.stderr
    return r.returncode,output

def main():
    p=argparse.ArgumentParser();p.add_argument('--credential-file');p.add_argument('--state',default='kaggle_state.json');sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('probe');launch=sub.add_parser('launch');launch.add_argument('--username');launch.add_argument('--dataset-dir',required=True);launch.add_argument('--kernel-dir',required=True);launch.add_argument('--accelerator',default='NvidiaTeslaT4');launch.add_argument('--create-dataset',action='store_true',help='Create a new dataset exactly once; do not infer absence from HTTP403')
    status=sub.add_parser('status');status.add_argument('--kernel');fetch=sub.add_parser('fetch');fetch.add_argument('--kernel');fetch.add_argument('--output',required=True)
    a=p.parse_args();env,secret=auth_env(a.credential_file)
    if a.action=='probe':
        # Import after setting credentials; api.authenticate introspects the token and resolves username.
        os.environ.update({k:v for k,v in env.items() if k.startswith('KAGGLE_')})
        from kaggle import api
        api.authenticate();username=api.get_config_value(api.CONFIG_NAME_USER)
        code,output=cli(['quota','--format','json'],env,secret,60)
        if code:raise RuntimeError(output)
        state={'authenticated':True,'username':username,'quota':json.loads(output),'kernel_started':False}
        write(a.state,state);print(json.dumps(state,indent=2));return
    state=read(a.state) if Path(a.state).exists() else {}
    if a.action=='launch':
        name=a.username or state.get('username');assert name and re.fullmatch(r'[a-zA-Z0-9_-]+',name)
        ds=Path(a.dataset_dir);kd=Path(a.kernel_dir);meta=read(ds/'dataset-metadata.json');km=read(kd/'kernel-metadata.json')
        assert meta['id'].split('/')[0]==name and km['id'].split('/')[0]==name
        assert km.get('is_private') is True and km.get('enable_gpu') is True
        dsref=meta['id']
        if a.create_dataset:
            code,output=cli(['datasets','create','-p',str(ds),'-t'],env,secret,900)
            assert code==0,output
        else:
            code,output=cli(['datasets','status',dsref],env,secret,60)
        if code:
            # A non-404 failure must not be guessed into a create request.
            assert any(s in output.lower() for s in ['404','not found','does not exist']),output
            code,output=cli(['datasets','create','-p',str(ds),'-t'],env,secret,900)
            assert code==0,output
        print(output,flush=True)
        state.update({'dataset':dsref,'kernel':km['id'],'kernel_url':'https://www.kaggle.com/code/'+km['id'],'kernel_started':False})
        write(a.state,state)
        for attempt in range(30):
            code,output=cli(['datasets','status',dsref],env,secret,60)
            print(output.strip(),flush=True)
            if code==0 and 'ready' in output.lower():break
            if code:raise RuntimeError(output)
            time.sleep(10)
        else:raise RuntimeError('Dataset not ready after bounded status checks; no kernel launched.')
        code,output=cli(['kernels','push','-p',str(kd),'--accelerator',a.accelerator,'--timeout','7200'],env,secret,180)
        assert code==0,output;print(output,flush=True)
        state.update({'dataset':dsref,'kernel':km['id'],'kernel_url':'https://www.kaggle.com/code/'+km['id'],'kernel_started':True})
        write(a.state,state)
    elif a.action in ['status','fetch']:
        kernel=a.kernel or state.get('kernel');assert kernel
        cmd=['kernels','status',kernel] if a.action=='status' else ['kernels','output',kernel,'-p',a.output,'--force']
        code,output=cli(cmd,env,secret,900 if a.action=='fetch' else 60)
        print(output);assert code==0

if __name__=='__main__':main()
