"""Finish verification and evidence generation only after all runs/analyses."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    for _ in range(21600):
        done=all((ROOT/'runs'/(k+suffix)/'metadata.json').exists() for k in ['gemma','clip','clip_large','specs','siglip'] for suffix in ['','-human'])
        if done and (ROOT/'results.json').exists() and (ROOT/'human_results.json').exists():break
        time.sleep(1)
    else:raise RuntimeError('Full runs and analyses have not completed')
    with (ROOT/'verification.log').open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'verify.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
    for script in ['negative_control.py','make_report.py','gallery.py']:
        subprocess.run([sys.executable,str(ROOT/script)],check=True)
    # The general-purpose plotting runtime already has matplotlib installed.
    subprocess.run(['python3',str(ROOT/'figure.py')],check=True)
    subprocess.run([sys.executable,str(ROOT/'package.py')],check=True)
    print('FINAL ARTIFACTS READY FOR VISUAL REVIEW AND SAVING',flush=True)

if __name__=='__main__':main()
