# Stage7 expert validation: COMPLETED

Completed Gemma human continuation and fresh SigLIP2-224 inference in this environment. All1000 images/5664 captions;900 unseen images/5112 captions separately. Protocol and native224/clipped sensitivity addendum were written before expanded Gemma caption scores.

## Offline reproduction

```sh
python -m pip install numpy scipy
OPENBLAS_NUM_THREADS=1 python stage7/analyze_human.py
python stage7/verify_completion.py
```

Requires only packaged vectors, not model weights or images. Analysis is secondary human evaluation, with individual-rating tau-c and image-cluster bootstrap. Full includes100 old development images. Package verification report already checks source image hashes/labels; offline verifier does not newly fetch original bytes.

## Repeat inference

Use Python3.12 with torch2.14.1+cpu and torchvision0.29.1+cpu from the official CPU wheel index; install transformers5.19.0, sentence-transformers6.1.0, pyarrow25.0.1, Pillow, numpy, scipy. Recorded dependency versions are in run metadata/environment.

1. Run stage7/download_required.py and stage7/download_siglip224.py (network required).
2. Run stage7/materialize_expert.py (verifies all image/source bytes and labels).
3. Run stage7/run_human.py gemma --threads 6 and siglip224 --threads 2 sequentially or following declared schedule.

Completed metadata makes inference runner exit; to force a fresh run, use a separate copy and move that copy's completed output directories first. Keep cache/provenance intact. prepare.py references earlier source archives and is retained as provenance, not required offline. Do not interpret continuation timing as controlled model latency.

See Stage7 report for actual results and limits. Model weights, raw images and parquet shards are deliberately excluded from the compact reproducibility archive.
