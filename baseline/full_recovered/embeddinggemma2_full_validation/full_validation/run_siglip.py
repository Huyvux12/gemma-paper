"""Run the secondary baseline after the two-thread baseline queue finishes."""
import subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    for _ in range(21600):
        if (ROOT/'runs/specs-human/metadata.json').exists() and (ROOT/'models/siglip/download_receipt.json').exists():break
        time.sleep(1)
    else:raise RuntimeError('Baseline/download queue did not complete')
    for ds in ['sugar','human']:
        with (ROOT/('siglip'+('_human' if ds=='human' else '')+'.log')).open('a') as log:
            subprocess.run([sys.executable,str(ROOT/'run.py'),'siglip','--dataset',ds,'--dtype','float32','--threads','2'],check=True,stdout=log,stderr=subprocess.STDOUT)
    print('SIGLIP2 BOTH DATASETS COMPLETE',flush=True)

if __name__=='__main__':main()
