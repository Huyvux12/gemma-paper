# Replace CLIPScore's encoder with EmbeddingGemma 2

This experiment changes the image/text encoder, retains CLIPScore's scoring
rule, and tests caption ranking plus correlation with human judgments.
No model training, API key, or paid service is used. The report distinguishes
literal replacement from a task-prefix adaptation.

## Included evidence

- Frozen protocol and manifests, source/checkpoint revisions and checksums.
- Exact 350-image SugarCrepe test sample, disjoint from all 210 previous pairs.
- Random 100-image Flickr8k-Expert sample: all non-reference captions and
  original individual expert ratings (not 1,656 independent images).
- Real inference vectors for every encoder, per-caption CSV scores and logs.
- Original-code CLIP equivalence check and independent scalar score checks.
- Reusable `gemma_clipscore.py` adapter and all reproduction scripts.

The archive includes sampled media. Checkpoints and full source Parquet shards
are excluded. They are downloadable from the pinned sources. Refer to the
datasets' original usage terms; sampling does not change their provenance.

## Reproduce from saved vectors (fast)

```bash
pip install numpy scipy
python analyze.py
python analyze.py --human
python verify.py
```

## Repeat inference

Use Python 3.12. Exact environment versions are in `environment_freeze.txt`.
The local-path CLIP package entry is only needed for the official-code check;
install it from the OpenAI repository revision in `source_revisions.json`.

```bash
pip install torch transformers==5.19.0 sentence-transformers==6.1.0 pillow numpy scipy pyarrow
python download_models.py
python run.py gemma
python run.py clip
python run.py clip_large
python run.py gemma --dataset human
python run.py clip --dataset human
python run.py clip_large --dataset human
python add_similarity_prefix.py
python analyze.py
python analyze.py --human
python verify.py
```

Do not rerun `prepare.py` to silently redefine test data. The committed
manifests specify the evaluated inputs. `prepare.py` documents sample creation;
when run against the included source JSON, it recreates the same sample.

`prepare_human.py` documents sampling from the two original Expert Parquet
shards. Download URLs:

- https://huggingface.co/datasets/yuwd/Flickr8k-HumanEval/resolve/c30e1e67e2b37a42ecffaf9a88a05458ca05b973/expert/test-00000-of-00002.parquet
- https://huggingface.co/datasets/yuwd/Flickr8k-HumanEval/resolve/c30e1e67e2b37a42ecffaf9a88a05458ca05b973/expert/test-00001-of-00002.parquet

The sampled image bytes and labels are already in the archive. Their SHA-256
digests are recorded; `verify.py` checks them directly.

## Use the scorer on your own captions

Create `captions.json`, mapping image filenames to candidate captions:

```json
{"photo.jpg": "A person rides a bicycle beside a river."}
```

```bash
python gemma_clipscore.py captions.json images/ \
  --model-path models/gemma --dimension 768 --prompt CLIPLiteral \
  --save-per-instance scored.json
```

The adapter defaults to literal CLIPScore-prefix replacement with full 768d
vectors. For the separately evaluated task-prefix adaptation, pass
`--prompt FactChecking`. For smaller stored vectors, pass `--dimension 256`.
Truncation is followed by L2 normalization. The 256d option is a storage
choice, not a universal accuracy guarantee.

**A score is uncalibrated image–caption compatibility, not probability of
truth.** Compare methods by ranking accuracy or correlation, not by their mean
raw score; raw scales differ across encoders. No threshold trained for CLIP
should be reused for Gemma without separate validation.

## Interpretation limits

This is a feasibility study with two CLIP checkpoints, one Gemma checkpoint,
two sampled benchmarks, and no fine-tuning. It does not establish SOTA, a new
architecture, general audio improvements, real-time operation, or a universal
hallucination detector. The datasets predate the model's training cutoff, so
pretraining contamination cannot be excluded. Image-disjoint sampling refers
only to previous experiments in this project.
