# EmbeddingGemma 2 CLIPScore replacement — stage 4

Results are an exploratory extension of the previously observed 350-image SugarCrepe and 100-image Flickr8k samples. SigLIP2 is new inference; Gemma and CLIP vectors are reused and independently checked. Future 560-image manifest is frozen, **not run**.

Unzip the archive into one directory. Keep `stage4` and `research/embeddinggemma2_clipscore_replacement` as siblings. Encoder weights and the virtual environment are omitted. The archived image data and embeddings are included so score analysis needs no network or model download.

To recompute stored-vector diagnostics and comparisons:

```bash
python -m pip install numpy scipy
OPENBLAS_NUM_THREADS=1 python stage4/diagnostics.py
OPENBLAS_NUM_THREADS=1 python stage4/analyze_siglip2.py
python stage4/vector_codec.py research/embeddinggemma2_clipscore_replacement/runs/gemma/embeddings.npz stage4/gemma_256_int8.npz --dimension 256
python stage4/verify_stage4.py
python stage4/make_report.py
```

The independent verification uses scalar arithmetic and can take a few minutes on CPU. Windows: omit the `OPENBLAS_NUM_THREADS=1` prefix or set that environment variable separately.

To rerun the added SigLIP2 encoder:

```bash
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install transformers==4.57.6 sentencepiece safetensors pillow numpy scipy
.venv/bin/python stage4/download_siglip2.py
.venv/bin/python stage4/run_siglip2.py
.venv/bin/python stage4/run_siglip2.py --dataset human
.venv/bin/python stage4/analyze_siglip2.py
```

The model revision and official file metadata are frozen in `siglip2_source.json`; weight SHA-256 is checked in the downloader. Archived local metadata and dependency versions document the run. If a partial image cache exists, image inference resumes from it; remove only the corresponding `stage4/siglip2*/partial.npz` to force a fresh full image run. Timings from resumed runs are not full latency measurements.

Files:

- `REPORT_VI.md`: scope, results, statistics, literature and limitations.
- `protocol.json`: stage-4 extension settings, written before SigLIP2 inference.
- `diagnostics_results.json`, `compression_scores.csv`: all storage configurations and controls.
- `baseline_results.json`, `baseline_pair_scores.csv`: all declared SigLIP2 prompts and old baselines.
- `gemma_256_int8.npz`, `vector_codec.py`: real compressed vectors and codec. Image and text vectors must use the same dimension and normalization. Quantizing stored vectors does not quantize the encoder.
- `siglip2*/embeddings.npz`, `siglip2*/metadata.json`: new model inference vectors, model revision, token-length audit, timings and direct-logit validation.
- `independent_verification.json`: scalar score and decoded-vector checks.
- `next_560_manifest.json`, `next_protocol.json`: unrun confirmation study; do not count its sample size as results.

No SOTA claim, no proof of equivalent quality under compression, no guaranteed paper novelty, no claim that our held-out images were absent from model pretraining. Respect source dataset and checkpoint terms when redistributing or publishing.
