"""Bundle reproducible evidence and sampled media, excluding weights."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    assert json.loads((ROOT/'independent_verification.json').read_text())['passed']
    assert json.loads((ROOT/'adapter_integration_verification.json').read_text())['passed']
    assert json.loads((ROOT/'adapter_default_verification.json').read_text())['passed']
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if rel.parts[0]=='models' or '__pycache__' in rel.parts:continue
        if p.suffix=='.parquet' or p.name in ['partial.npz','package_manifest.json']:continue
        files.append(p)
    manifest={'files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
       'weights_excluded':True,'parquet_shards_excluded':True,'media_included':True,'all_verifications_passed':True}
    (ROOT/'package_manifest.json').write_text(json.dumps(manifest,indent=2));files.append(ROOT/'package_manifest.json')
    archive=ROOT.parent/'embeddinggemma2_clipscore_replacement.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,'embeddinggemma2_clipscore_replacement/'+p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(archive,archive.stat().st_size,'bytes;',len(files),'files')

if __name__=='__main__':main()
