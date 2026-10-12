# Chạy lại đầy đủ Stage8 trên Kaggle

Kernel: `stage8/kaggle_full_rerun/` (script tự chứa source), private, GPU bật,
internet bật. Mặc định dùng account/dataset trong context: `dungdq210`.
Các lệnh Kaggle dưới đây dành cho **agent của Huy**; quá trình chuẩn bị code không chạy chúng.

## Input riêng bắt buộc

Gắn **một** dataset chứa `baseline.zip` từ gói **FULL** bàn giao, hoặc cây đã giải nén.
Kaggle tự giải nén ZIP cũng được hỗ trợ. Repo public không chứa đủ input để chạy.

| Thành phần trong FULL baseline | Dùng để |
|---|---|
| `full_recovered/embeddinggemma2_full_validation/full_validation/manifest.json` và `runs/{gemma,clip,clip_large,siglip,specs}/{metadata.json,embeddings.npz}` | SugarCrepe full/development/historical unseen; train adapter |
| `stage5/frozen_manifest.json`, `frozen_protocol.json`, `primary_reference.json` và `runs/` của 5 encoder trên | Mẫu560, prefix/nén vector và đối chứng |
| `stage6/uploaded/` gồm input frozen, `primary_siglip224_results.json`, `runs/siglip224/embeddings.npz` | Tái lập xác nhận SigLIP224 |
| `stage7/human_manifest.json`, `human_results.json`, `human_scores.csv`, `runs/{gemma,clip,clip_large,siglip,specs,siglip224}-human/{metadata.json,embeddings.npz}` | 32 cấu hình baseline human, so sánh với kết quả gốc |
| `research/embeddinggemma2_clipscore_replacement/provenance/expert_author_annotations.json` | 5 reference gốc cho BLEU/ROUGE/CIDEr/METEOR |

Danh sách kiểm tra đầy đủ: `stage8/rerun_inputs.py`. Có thể kiểm tra gói đã giải nén,
không download/train/import torch:

```bash
python3 stage8/full_rerun.py --baseline /path/to/full/baseline --validate-inputs-only
```

Không cần checkpoint adapter Stage8 cũ: job train lại 20 runs theo protocol gốc
(rank16/32 × anchor0,1/1,0 × 5 seed; 30 epochs). Chọn bằng validation như cũ;
nếu không chọn lại rank32/anchor1,0, job dừng và giữ evidence, không tự thay recipe.
Backbone frozen. Đây là **chạy lại analysis từ toàn bộ encoder cache**, không phải
fresh inference toàn bộ Gemma/CLIP/SigLIP/SPECS. PAC-S và benchmark encoder chạy GPU mới.

Internet tải hai Flickr8kExpert parquet pinned revision/hash bằng downloader Stage7,
materialize/kiểm tra 1.000 ảnh; benchmark tải pinned Gemma/SigLIP224/CLIP B32;
PAC-S tải source tác giả revision `bfbc488e9f5cccd818e75c7cdfaa37e1c87f00eb`
và trained ViT-B/32 checkpoint từ Google Drive như `stage8/pacscore_gpu.py`.
Nếu model yêu cầu quyền HF, chấp nhận điều khoản bằng account phù hợp và bật
Kaggle Secret `HF_TOKEN` cho kernel; không ghi token vào source/dataset/metadata.
Giữ cặp CUDA torch/torchvision của Kaggle; dependency còn lại ở `requirements_kaggle.txt`.
Java phải có sẵn cho PTBTokenizer/METEOR; thiếu dependency sẽ báo lỗi.

## Agent chạy lệnh

Nếu có quyền đọc dataset private lịch sử, dùng trực tiếp:

```bash
cd /workspace/gemma-paper/repo
kaggle datasets list --mine --search embeddinggemma-stage8
kaggle datasets files dungdq210/embeddinggemma-stage8-20261009-cc802fcc
python3 stage8/prepare_full_rerun.py
kaggle kernels push -p stage8/kaggle_full_rerun --accelerator NvidiaTeslaT4
kaggle kernels status dungdq210/embeddinggemma-full-rerun-20261012
```

Nếu thiếu dataset hoặc dùng account khác, thay username và đường dẫn FULL thật:

```bash
KAGGLE_USER=dungdq210
kaggle datasets list --mine --search embeddinggemma-full-inputs
python3 stage8/prepare_full_rerun.py \
  --username "$KAGGLE_USER" \
  --dataset-source "$KAGGLE_USER/embeddinggemma-full-inputs-20261012" \
  --baseline-zip /path/to/full/baseline.zip \
  --dataset-dir /tmp/gemma-full-rerun-inputs
kaggle datasets create -p /tmp/gemma-full-rerun-inputs --keep-tabular
kaggle datasets files "$KAGGLE_USER/embeddinggemma-full-inputs-20261012"
kaggle kernels push -p stage8/kaggle_full_rerun --accelerator NvidiaTeslaT4
kaggle kernels status "$KAGGLE_USER/embeddinggemma-full-rerun-20261012"
```

`datasets create` mặc định private; không thêm `--public`. Nếu dataset đã tồn tại,
bỏ bước create và rebuild bằng `--username ... --dataset-source ...` **không** có
`--baseline-zip`. Builder chỉ ghi file cục bộ; không gọi CLI/API.

Lấy logs và kết quả (thay owner nếu khác):

```bash
kaggle kernels output dungdq210/embeddinggemma-full-rerun-20261012 \
  -p /tmp/gemma-rerun-logs --file-pattern '.*\.log$|.*status\.json$'
kaggle kernels output dungdq210/embeddinggemma-full-rerun-20261012 \
  -p /tmp/gemma-rerun-output
```

CLI không có subcommand `kernels logs`; dùng `kernels output` để lấy các file log
do kernel ghi, hoặc xem execution log trên trang kernel. Cú pháp theo
[Kaggle kernels docs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md) và
[dataset docs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md).

## Output và tiêu chí hoàn tất

Tất cả output nằm trong **`/kaggle/working/stage8_full_rerun/`**:

- `all_metrics.csv/.json`: 47 cấu hình × full1.000 và unseen900 = 94 hàng;
  Kendall τ-c/τ-b, Spearman/Pearson rating cá nhân, Spearman mean rating,
  pair accuracy micro/macro, tie-aware NDCG@5/top1, AUROC/AP (mean rating≥3).
- `references/`: BLEU1–4, ROUGE-L, CIDEr, METEOR; `pacscore/`: PAC-S chính thức,
  w=2,0, float32, scores/receipt. Định nghĩa lấy từ báo cáo Stage8 và protocol;
  SPICE vẫn là tùy chọn chưa chạy trong báo cáo, không thêm vào run này.
- `sugar_metrics.csv/.json`, `sugar_pair_scores.csv`: Gemma gốc, 5 seed, ensemble;
  full7.511, development3.562, historical unseen3.949, historical560 và từng category.
- `baseline_*.json/.csv`: tính lại baseline SugarCrepe560/full, xác nhận Stage6,
  prefix/nén/grounding và human Stage7 với CI/kiểm chứng gốc.
- `adapter_training/`, `adapter_evaluation/`, `adapter_uncertainty.json`,
  `pac_comparison.json`, `fresh_gpu_benchmark.json`, `gpu_benchmark.csv`.
- `summary.json`, `pipeline_status.json`, `bootstrap_status.json`, `runtime.json`,
  input/source/output SHA256 receipts và `logs/` (cả lỗi).

Chỉ coi run hoàn tất khi **cả hai status có `complete: true`**, `summary.json`
tồn tại và đủ bảng94 hàng. PAC-S là bắt buộc; lỗi không được bỏ qua.
Private output có checkpoint/record metadata, giữ ngoài Git public. Encoder cache,
raw ảnh và weights benchmark nằm ở scratch, không nhân bản vào output; source và
trained checkpoint PAC-S được giữ dưới `pacscore/` theo script gốc.
Historical tests đã được quan sát; chạy lại không tạo test độc lập mới.
