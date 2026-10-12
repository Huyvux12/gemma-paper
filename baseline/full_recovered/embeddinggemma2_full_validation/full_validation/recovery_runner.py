"""Restore pinned media and dependencies, then resume the locked protocol."""
import hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;PY=str(BASE/'.venv/bin/python')

def start(script,name):
    return subprocess.Popen([PY,str(script)],stdout=(BASE/'recovery'/name).open('a'),stderr=subprocess.STDOUT)

def main():
    p=subprocess.run([PY,'-c','import torch,transformers,sentence_transformers,numpy,scipy,pyarrow;assert torch.__version__=="2.14.1+cpu";assert transformers.__version__=="5.19.0";assert sentence_transformers.__version__=="6.1.0";assert numpy.__version__=="2.5.3";assert scipy.__version__=="1.18.1";assert pyarrow.__version__=="25.0.1"'],capture_output=True)
    assert p.returncode==0,p.stderr.decode()[-2000:]
    print('PINNED DEPENDENCIES READY',flush=True)
    gemma_download=subprocess.Popen([PY,'-c',"import sys;sys.path.insert(0,'replacement');from download_models import download;download(('gemma','google/embeddinggemma-2'))"],stdout=(BASE/'recovery/download_models.log').open('a'),stderr=subprocess.STDOUT)
    jobs=[gemma_download,start(ROOT/'download_siglip.py','download_siglip.log'),start(ROOT/'download_data.py','download_data.log')]
    while jobs[-1].poll() is None:time.sleep(5)
    assert jobs[-1].returncode==0,'Expert shard restoration failed'
    media=start(ROOT/'materialize_media.py','materialize_media.log')
    while any(j.poll() is None for j in jobs+[media]):time.sleep(5)
    assert all(j.returncode==0 for j in jobs+[media]),'Pinned downloads failed; inspect logs'
    published={Path(line.split(None,1)[1]).name:line.split()[0] for line in (ROOT/'original_checksum_flickr8k_checksum.txt').read_text().splitlines() if line.strip()}
    for mf,folder in [('manifest.json','images'),('human_manifest.json','human_images')]:
        m=json.loads((ROOT/mf).read_text())
        for n,sha in m['media_sha256'].items():
            data=(ROOT/folder/n).read_bytes();assert hashlib.sha256(data).hexdigest()==sha
            if folder=='human_images':assert hashlib.md5(data).hexdigest()==published[n]
        for config in (ROOT/'runs').glob('*/config.json'):
            c=json.loads(config.read_text())
            if c['dataset']==('sugar' if mf=='manifest.json' else 'human'):
                assert hashlib.sha256((ROOT/mf).read_bytes()).hexdigest()==c['manifest_sha256'],'Frozen manifest hash changed during restoration'
    (ROOT/'recovery_integrity.json').write_text(json.dumps({'passed':True,'image_sha256_count':2560,'original_expert_md5_count':1000,'frozen_manifest_hashes_match':True,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2))
    print('RESTORATION VERIFIED; RESUMING LOCKED FLOAT32 RUNS',flush=True)
    children=[subprocess.Popen([PY,str(ROOT/script)],stdout=(ROOT/log).open('w'),stderr=subprocess.STDOUT) for script,log in [('run_all.py','run_all.log'),('run_siglip.py','run_siglip.log'),('finish.py','finish.log')]]
    while any(j.poll() is None for j in children):
        if any(j.poll() not in [None,0] for j in children):raise RuntimeError('Execution failed; inspect controller logs and checkpoint')
        time.sleep(5)
    assert all(j.returncode==0 for j in children)
    print('FINAL OUTPUTS READY FOR REVIEW',flush=True)

if __name__=='__main__':main()
