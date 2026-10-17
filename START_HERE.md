# Mở repository chuyển phiên

**Bắt đầu với bản hiện hành 17/10/2026:** [GEMMA_PAPER_HANDOFF_20261017_VI.md](GEMMA_PAPER_HANDOFF_20261017_VI.md). File này tổng hợp những gì đã làm, kết quả đã xác minh, giới hạn và thứ tự P0–P5 để nâng chất lượng paper. Xem [gemma_research_public](gemma_research_public/README_VI.md) để lấy notebook Kaggle và code mới. Lượt GPU đã COMPLETE; không chạy lại chỉ để kiểm trạng thái. Phần còn lại của trang này là hướng dẫn Stage1–8 lịch sử.

Repository chứa context, báo cáo Stage8, source, metric và output dạng bảng đã xác minh. Checkpoint, cache `.npz`, record dữ liệu, dataset manifest và raw log chưa công khai; chúng còn trong gói bàn giao đầy đủ. Xem `REPOSITORY_LAYOUT.md`. `baseline/` giữ toàn bộ gói Stage1–7 đã giải nén; source snapshot v1/v2 cũng đã giải nén thành thư mục. Không cần ZIP để đọc source/báo cáo. Tái lập train/evaluate cần khôi phục cache từ gói bàn giao đầy đủ hoặc tạo lại bằng inference.

## Đọc và phân tích trước, chưa cần train

1. Đọc `CONTEXT_HANDOFF_VI.md`.
2. Đọc `EmbeddingGemma2_Stage8_Report_VI.md`.
3. Mở `stage8/results/all_metrics.csv`.
4. Đọc kết quả GPU/PAC-S và CI trong báo cáo Stage8; raw JSON/status/metadata chưa thuộc cây public, còn trong gói đầy đủ riêng.
5. Mở `baseline/` để đọc báo cáo tổng hợp cũ và truy cập cache embedding.

`baseline/EmbeddingGemma2_Research_Report_VI.md` giải thích Stage1–7. `stage8/README_VI.md` hướng dẫn tái lập. `REPOSITORY_FILES_SHA256.json` xác minh source và báo cáo hiện tại. Manifest gói bàn giao gốc được giữ trong gói đầy đủ riêng, không thuộc cây public này. Xem `REPOSITORY_LAYOUT.md`.

## Tái lập CPU khi cần

Python3.12 là môi trường CPU đã dùng. Java cần cho PTBTokenizer/METEOR. Trọng số encoder lớn không được đính kèm; các embedding cache trong gói bàn giao đầy đủ đủ cho train/evaluate adapter trên CPU, nhưng chưa công khai trong repo này. Khôi phục chúng trước khi chạy các lệnh dưới đây.

```bash
python -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r stage8/requirements_cpu.txt
.venv/bin/python stage8/verify.py --baseline baseline
OPENBLAS_NUM_THREADS=1 .venv/bin/python stage8/pipeline.py --baseline baseline --output reproduced --device cpu
```

Trên Windows dùng `.venv\Scripts\python.exe`, và đặt `OPENBLAS_NUM_THREADS=1` theo shell Windows. Không bắt buộc tái train để đọc hoặc phân tích kết quả đã có.

Để tái benchmark CUDA/PAC-S, đọc notebook và dependency của phiên chạy, dùng `--device cuda --gpu-benchmark`. Notebook lưu source thực sự gửi v2; các tiện ích tổng hợp/CI/packaging ở source mới hơn được chạy cục bộ. Các snapshot giúp phân biệt phiên bản.

## Gửi sang phiên mới

Đưa link repository và `CONTEXT_HANDOFF_VI.md` cho phiên mới; dùng prompt ở cuối context. Mọi đường dẫn archive/source trong context lịch sử được ánh xạ sang thư mục ở `REPOSITORY_LAYOUT.md`. Các đường dẫn raw artifact trong context không nằm trong bản public. Đọc context không đồng nghĩa đã verify source; agent cần đối chiếu JSON/manifest trước các claim mới.

API token không có trong gói. Chỉ cần credential nếu phiên mới thực sự gọi Kaggle. Trạng thái COMPLETE/quota trong file là snapshot đã xác minh09/10/2026, không phải kiểm tra live11/10.

Notebook Colab trên repo là bản đọc private input riêng, giữ logic chạy. Xem REPOSITORY_LAYOUT.md và prepare_colab_inputs.py trước khi chạy.
