"""Pinned encoders with explicit precision and fresh, checked normalized outputs."""
import gc
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from common import read, sha, write


def encoding_dtype(kind, native_bf16, dtype_override=None):
    if dtype_override is not None:
        if kind != 'gemma' or dtype_override != torch.float32:
            raise ValueError('Only an explicit Gemma FP32 override is supported')
        return dtype_override
    return (torch.bfloat16 if native_bf16 else torch.float32) if kind == 'gemma' else torch.float16


class Encoder:
    def __init__(self, kind, config, output, dtype_override=None):
        from huggingface_hub import HfApi, snapshot_download
        self.kind, self.config = kind, config
        # PyTorch defaults including_emulation=True: T4 can report True without
        # native BF16 tensor cores. Never use that default for hardware selection.
        native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        self.dtype = encoding_dtype(kind, native_bf16, dtype_override)
        write(Path(output) / 'loading_status.json', {'complete': False, 'phase': 'author_metadata', 'kind': kind})
        print('LOAD_METADATA', kind, flush=True)
        info = HfApi().model_info(config['repo'], revision=config['revision'], files_metadata=True)
        if info.sha != config['revision']:
            raise ValueError('Model revision mismatch')
        # Some pinned author releases (CLIP B/32) only provide pytorch_model.bin.
        # Prefer safetensors when present; verify the original LFS SHA either way.
        entries = [entry for entry in info.siblings if entry.rfilename.endswith('.safetensors')]
        if not entries:
            entries = [entry for entry in info.siblings if Path(entry.rfilename).name.startswith('pytorch_model') and entry.rfilename.endswith('.bin')]
        if not entries:
            raise ValueError('Pinned author release contains no supported model weights')
        write(Path(output) / 'loading_status.json', {'complete': False, 'phase': 'author_weights_download', 'kind': kind})
        print('DOWNLOAD_AUTHOR_WEIGHTS', kind, flush=True)
        path = Path(snapshot_download(config['repo'], revision=config['revision'], max_workers=2,
                                     allow_patterns=['*.json', '*.model', '*.jinja', 'merges.txt', 'vocab.txt'] + [entry.rfilename for entry in entries]))
        weights = []
        for entry in entries:
            local = path / entry.rfilename
            if local.is_file():
                digest = sha(local)
                expected = entry.lfs.sha256 if entry.lfs else None
                if not expected or digest != expected:
                    raise ValueError('Model weight digest mismatch')
                weights.append({'name': entry.rfilename, 'sha256': digest, 'bytes': local.stat().st_size})
        if not weights:
            raise ValueError('No verified model weights')
        write(Path(output) / 'loading_status.json', {'complete': False, 'phase': 'verified_weights_model_load', 'kind': kind})
        if kind == 'gemma':
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(str(path), device='cuda', config_kwargs={'audio_config': None},
                                            model_kwargs={'dtype': self.dtype, 'attn_implementation': 'sdpa'})
            self.processor = None
        else:
            from transformers import AutoModel, AutoProcessor
            self.model = AutoModel.from_pretrained(str(path), dtype=self.dtype).to('cuda')
            self.processor = AutoProcessor.from_pretrained(str(path))
        self.model.eval()
        self.model.requires_grad_(False)
        self.receipt = {'kind': kind, **config, 'dtype': str(self.dtype), 'weights': weights,
                        'gpu': torch.cuda.get_device_name(0), 'visible_gpus': torch.cuda.device_count(),
                        'used_gpus': 1, 'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
                        'native_bf16_supported': native_bf16,
                        'explicit_dtype_override': str(dtype_override) if dtype_override is not None else None,
                        'float32_cache': True, 'native_encoding': True, 'backbone_frozen': True}
        if kind != 'gemma':
            tokenizer=self.processor.tokenizer
            self.receipt['loaded_model_class']=type(self.model).__name__
            self.receipt['loaded_text_vocab_size']=self.model.config.text_config.vocab_size
            self.receipt['tokenizer_special_ids']={key:getattr(tokenizer,key,None) for key in ['bos_token_id','eos_token_id','pad_token_id']}
        write(Path(output) / 'model_receipt.json', self.receipt)
        write(Path(output) / 'loading_status.json', {'complete': True, 'kind': kind})

    @staticmethod
    def checked(value):
        if hasattr(value, 'pooler_output'):
            value = value.pooler_output
        value = value.float()
        if not torch.isfinite(value).all() or (value.norm(dim=-1) < 1e-10).any():
            raise ValueError('Invalid native embeddings; no substituted/cast cache permitted')
        return F.normalize(value, dim=-1)

    @torch.inference_mode()
    def images(self, images):
        if self.kind == 'gemma':
            value = self.model.encode([{'image': im} for im in images], batch_size=len(images),
                                      normalize_embeddings=False, convert_to_tensor=True, show_progress_bar=False)
        else:
            values = self.processor(images=images, return_tensors='pt').to('cuda')
            values['pixel_values'] = values['pixel_values'].to(self.dtype)
            value = self.model.get_image_features(**values)
        return self.checked(value)

    @torch.inference_mode()
    def texts(self, captions):
        captions = [self.config['prefix'] + text for text in captions]
        if self.kind == 'gemma':
            value = self.model.encode(captions, batch_size=len(captions), prompt='', normalize_embeddings=False,
                                      convert_to_tensor=True, show_progress_bar=False)
        else:
            options = {'padding': 'max_length', 'max_length': 64} if self.kind.startswith('siglip') else {'padding': True, 'max_length': 77}
            values = self.processor(text=captions, truncation=True, return_tensors='pt', **options).to('cuda')
            value = self.model.get_text_features(**values)
        return self.checked(value)

    def unload(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def encode_cache(encoder, root, records, output):
    names = sorted({r['image'] for r in records})
    texts = sorted({r['caption'] for r in records} | {r['negative_caption'] for r in records if r['domain'] == 'sugar'})
    metadata = {'images': names, 'texts': texts, 'record_ids': [r['id'] for r in records],
                'records_sha256': sha(root / 'records.json'), 'model': encoder.receipt}
    cache = {}
    for key, items, batch_size in [('image', names, 8), ('text', texts, 32)]:
        vectors = []
        for start in range(0, len(items), batch_size):
            selected = items[start:start + batch_size]
            if key == 'image':
                images = []
                for name in selected:
                    with Image.open(root / 'images' / (name + '.jpg')) as image:
                        images.append(image.convert('RGB'))
                value = encoder.images(images)
            else:
                value = encoder.texts(selected)
            vectors.append(value.cpu().numpy())
            if start % (batch_size * 32) == 0:
                write(output / 'encoding_status.json', {'complete': False, 'kind': key, 'completed': start + len(selected), 'total': len(items)})
                print(encoder.kind, key, start + len(selected), '/', len(items), flush=True)
        cache[key] = np.concatenate(vectors).astype(np.float32)
    np.savez_compressed(output / 'embeddings.npz', image=cache['image'], text=cache['text'])
    write(output / 'embedding_metadata.json', metadata)
    write(output / 'encoding_status.json', {'complete': True, 'images': len(names), 'texts': len(texts)})
    cache.update(image_index={name: i for i, name in enumerate(names)}, text_index={text: i for i, text in enumerate(texts)})
    return cache


@torch.inference_mode()
def benchmark(encoder, heads, records, root, config):
    panel = sorted([r for r in records if r['domain'] == 'flickr' and r['split'] == 'test'], key=lambda r: r['id'])[:8]
    images = []
    for row in panel:
        with Image.open(root / 'images' / (row['image'] + '.jpg')) as image:
            images.append(image.convert('RGB'))
    labels = {'base': None, **heads}
    rows = []
    for batch in config['batch_sizes']:
        measurements = {name: [] for name in labels}
        peak = {name: [] for name in labels}
        resident = torch.cuda.memory_allocated()
        def call(head):
            u, v = encoder.images(images[:batch]), encoder.texts([r['caption'] for r in panel[:batch]])
            if head is not None:
                u, v = head(u), head(v)
            value = (u * v).sum(-1)
            if encoder.kind != 'siglip224':
                value = value.clamp_min(0)
            return value * encoder.config['scale']
        names = list(labels)
        for index in range(config['warmup'] + config['repetitions']):
            order = names[index % len(names):] + names[:index % len(names)]
            for name in order:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                value = call(labels[name])
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                if not torch.isfinite(value).all():
                    raise ValueError('Nonfinite native score')
                if index >= config['warmup']:
                    measurements[name].append(elapsed)
                    peak[name].append(torch.cuda.max_memory_allocated() / 1024**2)
        for name in names:
            values = np.asarray(measurements[name])
            base = np.asarray(measurements['base'])
            rows.append({'variant': name, 'batch': batch, 'p50_ms_call': float(np.median(values)),
                         'p95_ms_call': float(np.percentile(values, 95)), 'p50_ms_pair': float(np.median(values) / batch),
                         'throughput_pairs_s': float(batch * 1000 / np.median(values)),
                         'peak_allocated_mib': float(max(peak[name])), 'resident_all_heads_mib': resident / 1024**2,
                         'paired_overhead_p50_ms': float(np.median(values - base)), 'samples_ms': values.tolist(),
                         'gpu': encoder.receipt['gpu'], 'dtype': encoder.receipt['dtype'], 'used_gpus': 1,
                         'scope': 'fixed 8-record predecoded panel; p95 of20 repetitions; resident memory includes all small fitted heads'})
    return {'panel_ids': [r['id'] for r in panel], 'rows': rows}
