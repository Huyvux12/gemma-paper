"""Export data fingerprints into a private input, never credentials or records."""
import argparse
import json
import re
from pathlib import Path

KEYS=('original_core_manifest_sha256','original_timing_manifest_sha256','records_sha256',
      'embedding_sha256','native_inputs_zip_sha256','native_inputs_manifest_sha256')


def main(source,output):
    source,output=Path(source),Path(output)
    if output.exists():raise FileExistsError('Refuse to overwrite an existing private config')
    value=json.loads(source.read_text(encoding='utf-8'))
    pins={key:value[key] for key in KEYS}
    if any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in pins.values()):
        raise ValueError('Expected verified data fingerprint fields')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'schema':'gemma-private-runtime-v1','pins':pins},indent=2)+'\n',encoding='utf-8')
    print('PRIVATE_RUNTIME_CONFIG_WRITTEN',output,'Do not commit or publish this input.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pins',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();main(args.pins,args.output)
