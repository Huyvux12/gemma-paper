# Gemma: public research code, notebook và kết quả tổng hợp

Đọc [file bàn giao hiện hành](../GEMMA_PAPER_HANDOFF_20261017_VI.md) trước khi tiếp tục nghiên cứu. Gói này công khai code/phương pháp, notebook tự chứa code, bảng aggregate và8 bộ hình. Ảnh/annotations, cache, checkpoint, per-record predictions/errors, raw logs và dataset fingerprints vẫn nằm trong input/output riêng; clone repo không tự cấp quyền truy cập các artifact đó.

## Kết quả đã thực sự chạy

Full package gốc đã chạy từ ảnh đến báo cáo/ZIP trên Kaggle Tesla T4 ngày17/10/2026, khoảng77 phút. Đã kiểm đủ15 variants/18537 test records, native cache5817 ảnh/20065 texts và864 timing calls. Cache/predictions/metrics tái lập snapshot cũ chính xác; đây là kiểm chứng recipe đã khóa, không phải gain mới trên fresh holdout.

- [Báo cáo đầy đủ dạng HTML](reports/REPORT_VI.html), [Markdown](reports/REPORT_VI.md).
- [Kết quả và trade-off](RESULTS_VI.md), [idea/method](IDEA_METHOD_VI.md).
- [Notebook Kaggle GPU](notebooks/02_Gemma_Kaggle_Full_Rerun.ipynb), [CPU replay/refit](notebooks/01_Gemma_CPU_Replay_Refit.ipynb).
- [Kaggle gốc private](https://www.kaggle.com/code/huyv12345/gemma-end-to-end-report-20261017): version1 có Output đã hoàn tất; version2 chỉ Quick Save sửa bảng Markdown, không chạy thêm GPU.

**Bản public tách private runtime config ra khỏi code nhúng.** Frozen scientific source giữ nguyên byte; loader chỉ đọc expected data fingerprints từ file input riêng rồi thực hiện các kiểm SHA cũ. Packaging public đã được kiểm code/notebook/config loading; chưa gửi một lượt GPU mới chỉ để xuất bản. Không gọi bản public là một thí nghiệm mới.

## Chạy trên Kaggle, không dùng GPU laptop

1. Import notebook GPU ở trên, chọn GPU T4 và bật Internet.
2. Attach native input snapshot riêng. Chủ tài khoản hiện tại có dataset private `huyv12345/gemma-standalone-native-20261017`; người khác cần dữ liệu/splits hợp lệ của mình và đúng runtime config tương ứng, hoặc khôi phục gói đầy đủ.
3. Attach thêm **`gemma_private_runtime.json`** dưới dạng private Kaggle input. File chứa sáu fingerprint expected, không chứa credential. Tạo file từ gói đầy đủ đã xác minh bằng Python thường:

```bash
python tools/export_private_runtime.py --pins /path/to/gemma_research_standalone/PINNED.json --output /path/outside/git/gemma_private_runtime.json
```

4. Bấm Run All. Trên Kaggle, install/run tự bật; notebook tìm cấu hình riêng trong `/kaggle/input`. Nếu có nhiều file cấu hình, đặt biến môi trường `GEMMA_PRIVATE_CONFIG` tới file chính xác trước bootstrap. Thiếu cấu hình thì fail sớm, không âm thầm bỏ kiểm SHA.
5. Nếu Hugging Face yêu cầu quyền model, chấp nhận điều khoản và dùng Kaggle Secret `HF_TOKEN`; không ghi token vào notebook.
6. Save Version → Save & Run All để Kaggle chạy khi laptop tắt. Output mới ở `/kaggle/working/gemma_e2e_<timestamp>/`, ZIP nằm cạnh folder. Có core/cache/heads/predictions/CI/training histories, timing128/raw calls, báo cáo HTML/Markdown và PNG/SVG.

Không đính kèm config riêng vào Git, kể cả nhúng base64. Không cần conda hoặc CUDA Torch trên laptop. `vendor/stage11/prepare_data.py` giữ source tạo dữ liệu từ nguồn public; việc rebuild dữ liệu phải tuân thủ source versions/split policy và giấy phép. Dữ liệu dựng theo protocol khác không được gọi là exact rerun của snapshot này.

## CPU và tạo lại báo cáo

CPU replay/refit cần private `assets/`, asset manifest và cấu hình fingerprint tương ứng; notebook public không nhúng những artifact này. Nếu dùng gói đầy đủ, trỏ `GEMMA_BUNDLE_ROOT` tới gói đó và `GEMMA_PRIVATE_CONFIG` tới cấu hình riêng. CPU refit trên runtime khác có thể có sai số số học; không lấy CPU/CUDA drift làm effect của loss.

Tạo báo cáo từ một producer hoàn tất, không train/encode lại:

```bash
python src/report.py --core /path/to/run/core --timing /path/to/run/timing128 --output /path/to/new_report
```

Lệnh kiểm producer manifests, mapping/metrics/raw timing và từ chối overwrite. Tên report output phải mới. Nếu không có timing, bỏ `--timing`; report ghi thiếu cost, không lấy số cũ điền vào.

## Cấu trúc và kiểm tra

```text
src/                   portable entrypoints; public loader + private config binding
vendor/stage11/        frozen scientific source and protocol
notebooks/             source embedded, no saved outputs/private fingerprints
tools/build_notebooks.py
tools/export_private_runtime.py
tools/verify_public.py
report_context/        historical aggregate controls/seven-backbone context
reports/               aggregate CSV, HTML/Markdown, PNG/SVG
PINNED.json            author model/version/weights and source hashes only
PUBLIC_FILES_SHA256.json
```

Chạy `python tools/verify_public.py` để kiểm public bytes/source/notebook payload. Manifest chỉ mô tả source/docs/aggregate reports công khai, không phải dataset manifest.

[Phạm vi QA trước khi publish](VERIFICATION_VI.md): CPU replay/rescore và GPU-disabled đã chạy bằng source nhúng; báo cáo được dựng lại từ producer đã hoàn tất. Không train/encode thêm trên laptop.

Không tuyên bố SOTA, fresh holdout, exact Stage9/10 recovery hoặc đã đủ chất lượng venue A. Ba corpus hiện tại đã được quan sát nhiều lần. Ưu tiên tiếp theo nằm ở file handoff: prior-art gap, matched-backbone controls và đánh giá generalization theo protocol mới, trước khi tốn GPU vào backbone khác.
