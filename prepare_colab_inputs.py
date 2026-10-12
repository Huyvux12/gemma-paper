"""Build private Colab input locally; never commit the generated payload."""
import argparse
import base64
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-directory', type=Path, required=True,
                        help='stage6/uploaded from the complete baseline handoff')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = {}
    for name in ('frozen_manifest.json', 'frozen_protocol.json',
                 'primary_reference.json', 'run_siglip224.py'):
        data = (args.input_directory / name).read_bytes()
        payload[name] = {'b64': base64.b64encode(data).decode(),
                         'sha256': hashlib.sha256(data).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload))
    print('Private Colab input written locally; do not commit or publish it.')


if __name__ == '__main__':
    main()
