# Stage 5 reproducibility

No new encoder inference was performed in this ChatGPT turn. Scores are computed from recovered complete SugarCrepe float32 inference vectors. SigLIP2-256 is supplemental; the originally declared SigLIP2-224 primary contrast remains pending. Human Gemma inference is incomplete.

## Offline 560 score reproduction

Unzip preserving the stage5 directory. Requires Python, numpy and scipy.

```sh
python stage5/analyze_560.py
python stage5/verify.py --scores-only
```

The second command independently recalculates all 31,360 scores and paired exact p-values. It saves `independent_scores_verification.json`. The packaged `independent_verification.json` additionally records the completed cross-archive 350-record bridge and disjointness audit. Running that broader bridge (`verify.py` without --scores-only), `prepare.py`, or `recover_vectors.py` requires the original stage4/replacement and full-validation archives in their original extracted paths. Those source archives are not duplicated here.

## Wider cohort

`full_sugar_results.json`, `full_sugar_pair_scores.csv`, `full_source_protocol.json`, and `full_source_analyze.py` preserve the wider 7,511-pair analysis. Recomputing that wider analysis requires `embeddinggemma2_full_validation.zip` version31, with its complete vector files; the subset560 vectors cannot recreate 7,511 scores. The 560 sample is contained in the 1,005-image unseen cohort, not an independent extra dataset.

## Missing original primary contrast

Open the self-contained `EmbeddingGemma2_560_SigLIP224_Colab.ipynb` in Colab, select GPU, Run all. The notebook is syntax/schema checked but NOT executed in this environment. It downloads exact image bytes and pinned SigLIP2-224 weights, verifies hashes, runs float32, then computes the fixed primary comparison using CLIPScore clipping as declared in stage4. It needs network access to Hugging Face/COCO; image or model failures stop rather than substitute data. New SigLIP224 inference uses saved verified Gemma scores; it does not regenerate Gemma. No runtime or primary result is claimed in this package.

Compression is embedding storage, not encoder quantization. Do not use cross-session cached timings for latency comparisons.
