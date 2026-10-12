# EmbeddingGemma-2 cho đánh giá caption ảnh

Repository nghiên cứu thực nghiệm: thay encoder trong score ảnh–caption, kiểm chứng đối chứng, nén embedding và học residual adapter ở đầu ra. Source Stage1–8 đã giải nén thành cây thư mục; có context, notebook, báo cáo, bảng metric, prediction dạng bảng và log thực thi. Checkpoint/cache embedding và ảnh mẫu chưa được công khai: xem `REPOSITORY_LAYOUT.md`.

## Đọc trước

- [Context bàn giao](CONTEXT_HANDOFF_VI.md): mục tiêu, protocol, kết luận và hướng còn thiếu.
- [Báo cáo hiện tại — Stage8](EmbeddingGemma2_Stage8_Report_VI.md): adapter, metric mở rộng, PAC-S, CI và GPU.
- [Báo cáo Stage1–7](baseline/EmbeddingGemma2_Research_Report_VI.md): đối chứng encoder/prefix, human ranking và nén vector.
- [Hướng dẫn tái lập](START_HERE.md).
- [Bảng metric đầy đủ](stage8/results/all_metrics.csv), [biểu đồ trade-off](stage8/results/adapter_tradeoff.png).
- [Bố cục và kiểm chứng byte](REPOSITORY_LAYOUT.md).

## Kết quả chính đã đo

| Endpoint | Gemma gốc | Adapter ensemble |
|---|---:|---:|
| SugarCrepe 560 accuracy | 89,8214% | 91,25% |
| SugarCrepe 3.949 cặp accuracy | 91,5675% | 92,6564% |
| Flickr8kExpert 900 ảnh, Kendall τ-c | 0,544936 | 0,423398 |

Adapter tăng accuracy quan sát trên SugarCrepe nhưng giảm tương hợp với đánh giá con người. Backbone Gemma frozen; chỉ hai residual head được học. Chưa chứng minh cải tiến tổng thể, novelty thuật toán hay SOTA. Historical test đã được quan sát, không dùng chúng như một holdout mới. Đọc báo cáo để xem confidence interval và protocol trước khi diễn giải các chênh lệch.

## Chạy từ cây repository

Không cần giải nén thêm. Python 3.12; Java cần cho PTBTokenizer/METEOR.

```bash
git clone https://github.com/Huyvux12/gemma-paper.git
cd gemma-paper
python -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r stage8/requirements_cpu.txt
.venv/bin/python stage8/verify.py --baseline baseline
OPENBLAS_NUM_THREADS=1 .venv/bin/python stage8/pipeline.py --baseline baseline --output reproduced --device cpu
```

Trước lệnh verify/pipeline, chép cache `.npz` từ gói bàn giao đầy đủ vào đúng đường dẫn trong `baseline/` (xem danh sách `PUBLICATION_EXCLUSIONS.json`), hoặc chạy downloader/inference theo README baseline để tạo lại. Không cần tải backbone nếu đã khôi phục cache. CUDA benchmark/PAC-S có dependency riêng trong notebook. Trọng số encoder lớn và raw dataset đầy đủ không nằm trong bộ bàn giao; downloader, revision và receipt được giữ để tải lại.

## Source và bằng chứng

- `baseline/`: source, báo cáo, protocol và kết quả dạng bảng của Stage1–7.
- `stage8/*.py`: train/evaluate adapter, metric, bootstrap, GPU benchmark và Kaggle orchestration.
- `stage8/kaggle_kernel/run_stage8.ipynb`: notebook GPU đã gửi.
- `stage8/results/submitted_source_v1/`, `submitted_source_v2/`: source snapshot đã giải nén.
- `stage8/results/adapter_training/`: history của 20 lần train, split và selection; các checkpoint `.npz` chưa công khai.
- `stage8/results/kaggle_final/`: output GPU/PAC-S, runtime, log và kiểm chứng tải về.

Kết quả thực nghiệm chốt ngày 09/10/2026; context bàn giao ngày 11/10; bản cây GitHub ngày 12/10. Quota/trạng thái API trong log là snapshot lịch sử. Token không nằm trong repo; chỉ cấu hình credential riêng khi thực sự cần chạy Kaggle mới.

Các nguồn, model và dữ liệu bên thứ ba giữ điều khoản riêng. LICENSE của snapshot SPECS được giữ tại thư mục nguồn; repo không cấp lại một giấy phép chung cho tất cả thành phần.
