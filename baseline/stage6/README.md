# Stage 6: Original SigLIP2-224 primary contrast completed

The uploaded user Colab inference is verified from vectors. The five-config 560-image analysis is reproducible offline:

```sh
python -m pip install numpy scipy
OPENBLAS_NUM_THREADS=1 python stage6/analyze_confirmation.py
```

Do not treat archived CPU timings and returned CUDA timings as a hardware-controlled comparison. This package does not include model weights or raw images. It verifies the byte-identical runner/input, vector arithmetic, exact p-values, bootstrap reproduction, and matching user-returned metrics. Image/model SHA provenance is the original runner's checks and recorded metadata.

Stage5 metadata/results are historical provenance; any pending-primary status in those old files is superseded by `stage6/verified_confirmation_results.json` and the Stage6 report. Human evaluation remains incomplete. The 560 sample is part of the larger unseen cohort, not independent additional data.
