"""Verify public source/report bytes and inspect embedded notebook archives."""
import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/'PUBLIC_FILES_SHA256.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file():raise ValueError('Missing/unsafe public artifact: '+name)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Public bytes changed: '+name)
    for notebook in (ROOT/'notebooks').glob('*.ipynb'):
        doc=json.loads(notebook.read_text(encoding='utf-8'));archives=[]
        for cell in doc['cells']:
            if cell['cell_type']!='code':continue
            if cell.get('outputs'):raise ValueError('Public notebook must not contain saved private outputs')
            for node in ast.walk(ast.parse(''.join(cell['source']))):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):archives.append(node.args[0].value)
        if len(archives)!=1:raise ValueError('Expected exactly one public code archive')
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(archives[0]))) as archive:
            for name in archive.namelist():
                if archive.read(name)!=(ROOT/name).read_bytes():raise ValueError('Stale notebook source: '+name)
            pins=json.loads(archive.read('PINNED.json'))
            if set(pins)!={'gemma_model','gemma_weights','vendor_sha256'}:raise ValueError('Private fingerprints embedded')
    print('PUBLIC_SOURCE_AND_NOTEBOOKS_VERIFIED',len(manifest['files']),'files')


if __name__=='__main__':main()
