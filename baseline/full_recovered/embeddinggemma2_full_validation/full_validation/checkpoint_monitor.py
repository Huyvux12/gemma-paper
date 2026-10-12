"""Durable checkpoint saver; persistent destination identity is an argument."""
import argparse,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def inference_done():
    return all((ROOT/'runs'/(k+s)/'metadata.json').exists() for k in ['gemma','clip','clip_large','specs','siglip'] for s in ['','-human'])

def save(a,version):
    subprocess.run(['python3',str(ROOT/'package.py'),'--checkpoint'],check=True)
    request={'uploads':[{'local_path':str(ROOT.parent/'embeddinggemma2_full_validation.zip'),'purpose':'replace_library_file','library_file_id':a.library_id,'expected_current_version':version,'version_reason':'Resumable full benchmark progress; no unfinished-method conclusions'}]}
    p=subprocess.run(['python3',a.helper],input=json.dumps(request),text=True,capture_output=True)
    with (ROOT.parent/'recovery/checkpoint_write_receipts.jsonl').open('a') as f:f.write(p.stdout+'\n'+p.stderr+'\n')
    if p.returncode:raise RuntimeError('Checkpoint save failed or uncertain; inspect receipt, do not blindly retry')
    objects=[json.loads(line) for line in p.stdout.splitlines() if line.startswith('{')]
    result=next(o['results'][0] for o in objects if 'results' in o)
    if result['status']!='succeeded':raise RuntimeError('Checkpoint save did not succeed')
    version=result['current_version_number']
    (ROOT.parent/'recovery/current_checkpoint_version.json').write_text(json.dumps({'version':version,'library_file_id':a.library_id,'saved_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}))
    print('CHECKPOINT SAVED version',version,flush=True)
    return version

def main():
    p=argparse.ArgumentParser();p.add_argument('--helper',required=True);p.add_argument('--library-id',required=True);p.add_argument('--expected-version',type=int,required=True);p.add_argument('--interval',type=int,default=300);a=p.parse_args()
    version=a.expected_version
    for _ in range(120):
        if inference_done():
            print('INFERENCE COMPLETE; stop checkpoint writes before final verification/package',flush=True);return
        version=save(a,version)
        for _ in range(a.interval//10):
            if inference_done():
                print('INFERENCE COMPLETE; stop checkpoint writes before final verification/package',flush=True);return
            log=ROOT/'finish.log'
            if log.exists() and 'FINAL ARTIFACTS READY' in log.read_text():
                print('FINAL READY; stop checkpoint writes for review',flush=True);return
            time.sleep(10)
    raise RuntimeError('Saver reached bounded duration; resume saved checkpoint')

if __name__=='__main__':main()
