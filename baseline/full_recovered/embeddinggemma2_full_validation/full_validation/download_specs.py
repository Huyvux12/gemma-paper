"""Download the trained SPECS checkpoint linked by the official repository."""
import hashlib,json,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    info=json.loads((ROOT/'specs_hf_metadata.json').read_text());entry=next(r for r in info['siblings'] if r['rfilename']=='spec.pt')
    path=ROOT/'models/spec.pt';path.parent.mkdir(exist_ok=True)
    url=f'https://huggingface.co/XiaoFu666/SPECS/resolve/{info["sha"]}/spec.pt'
    if not path.exists():
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url,timeout=90) as r,path.with_suffix('.part').open('wb') as f:
                    for chunk in iter(lambda:r.read(8*1024*1024),b''):f.write(chunk)
                path.with_suffix('.part').replace(path);break
            except Exception:
                if attempt==4:raise
                time.sleep(min(2**attempt,8))
    assert path.stat().st_size==entry['size']
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==entry['lfs']['sha256']
    receipt={'model':'XiaoFu666/SPECS','revision':info['sha'],'filename':'spec.pt','sha256':h.hexdigest(),
      'bytes':path.stat().st_size,'verified':True,'selection':'Official trained spec.pt, not openai_b32_longclip.pt'}
    (ROOT/'specs_checkpoint.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
