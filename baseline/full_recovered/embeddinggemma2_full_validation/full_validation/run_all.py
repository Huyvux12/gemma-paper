"""Run all fixed methods; two CPU workers totaling eight threads."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def wait_for_media():
    for _ in range(900):
        p=ROOT/'coco_mirror_receipt.json'
        if p.exists() and json.loads(p.read_text()).get('verified'):return
        time.sleep(1)
    raise RuntimeError('Media preparation did not complete; inspect materialize_media.log')

def worker(kinds,threads):
    for kind in kinds:
        for dataset in ['sugar','human']:
            name=kind+('_human' if dataset=='human' else '')
            with (ROOT/(name+'.log')).open('a') as log:
                cmd=[sys.executable,str(ROOT/'run.py'),kind,'--dataset',dataset,'--dtype','float32','--threads',str(threads)]
                print('RUN',kind,dataset,'threads',threads,flush=True)
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
                print('DONE',kind,dataset,flush=True)

def main():
    if '--gemma-worker' in sys.argv:worker(['gemma'],6);return
    if '--baseline-worker' in sys.argv:worker(['clip','clip_large','specs'],2);return
    wait_for_media();print('MEDIA READY; beginning full float32 runs',flush=True)
    children=[subprocess.Popen([sys.executable,__file__,mode]) for mode in ['--gemma-worker','--baseline-worker']]
    codes=[p.wait() for p in children];assert codes==[0,0],codes
    print('ALL EIGHT INFERENCE RUNS COMPLETE',flush=True)
    for _ in range(21600):
        if (ROOT/'runs/siglip-human/metadata.json').exists():break
        time.sleep(1)
    else:raise RuntimeError('Secondary SigLIP2 worker has not completed')
    print('ALL TEN INFERENCE RUNS COMPLETE',flush=True)
    subprocess.run([sys.executable,str(ROOT/'analyze.py')],check=True,stdout=(ROOT/'analysis.log').open('w'))
    subprocess.run([sys.executable,str(ROOT/'analyze.py'),'--human'],check=True,stdout=(ROOT/'human_analysis.log').open('w'))
    print('ALL ANALYSES COMPLETE',flush=True)

if __name__=='__main__':main()
