# Stage8 — metric mở rộng và Gemma với adapter được học

Gói đầy đủ chứa `baseline.zip` (toàn bộ kết quả/source Stage1–7) và `stage8/` (source, protocol, 20 checkpoint/runs, metric và trạng thái Kaggle). Token không được đóng gói.

## Tái lập trên CPU

Python 3.12; tạo môi trường riêng, cài PyTorch CPU từ kho chính thức và các dependency bên dưới. Java cần cho COCO PTBTokenizer/METEOR.

```bash
python -m zipfile -e baseline.zip baseline
python -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r stage8/requirements_cpu.txt
.venv/bin/python stage8/verify.py --baseline baseline
OPENBLAS_NUM_THREADS=1 .venv/bin/python stage8/pipeline.py --baseline baseline --output reproduced --device cpu
```

`pipeline.py` tái lập các metric tham chiếu, train 4 cấu hình × 5 seed và đánh giá adapter. Mọi lựa chọn cấu hình dùng validation của 111 ảnh; train 444 ảnh. Encoder giữ nguyên trọng số và dùng embedding đã xác minh. GPU encoder benchmark chạy riêng bằng `--device cuda --gpu-benchmark`; cần CUDA và các dependency trong notebook.

## Kaggle

Notebook `kaggle_kernel/run_stage8.ipynb` tự chứa source được gửi của phiên bản mới nhất. `results/submitted_source_v1.zip` và `submitted_source_v2.zip` lưu từng snapshot. Các tiện ích tổng hợp bảng, bootstrap và xây dựng báo cáo bổ sung được chạy cục bộ sau khi gửi v1 và được bổ sung dần; snapshot giúp phân biệt source thực tế mỗi phiên.

Dataset và notebook đều riêng tư. Để kiểm tra/tải output, cung cấp token qua biến môi trường `KAGGLE_API_TOKEN` hoặc file riêng `--credential-file`; tránh đưa vào notebook hoặc repository.

```bash
.venv/bin/python stage8/kaggle_control.py --state stage8/results/kaggle_connection.json status
.venv/bin/python stage8/kaggle_control.py --state stage8/results/kaggle_connection.json fetch --output downloaded
```

Tạo job mới: chạy `prepare_kaggle.py --baseline-zip baseline.zip --username USERNAME`, thay slug/ngày cho lần thử mới rồi `kaggle_control.py ... launch --dataset-dir ... --kernel-dir ... --create-dataset`. Cờ create chỉ dùng lần đầu; lỗi 403 không được diễn giải thành dataset chưa tồn tại.

## Diễn giải kết quả

Adapter residual ở đầu ra Gemma học trọng số mới; backbone vẫn frozen. Accuracy SugarCrepe tăng ở ensemble, nhưng các chỉ số khớp đánh giá con người giảm. Đây là kết quả thử nghiệm phương pháp và trade-off, chưa phải cải tiến tổng thể. Cả SugarCrepe và Flickr8k này đã được quan sát trước đây; cần một test độc lập mới trước khi đưa ra kết luận xác nhận.

`results/all_metrics.csv` chứa các metric cho toàn bộ 1.000 ảnh và nhóm lịch sử chưa dùng trong development (900 ảnh). Các tên method gồm cấu hình chiều/prompt/precision, không phải số lượng backbone khác nhau. `gemma-original-literal768` tái tính score gốc để kiểm tra, trùng baseline 768.

BLEU/ROUGE/CIDEr/METEOR nhận thêm 5 reference do tác giả cung cấp; so sánh với metric reference-free phải giữ sự khác biệt đầu vào này. AP là average precision ở ngưỡng mean rating ≥3. NDCG và top1 xử lý score ties. PAC-S và phép đo GPU có status riêng; chỉ kết quả chạy thành công mới được dùng trong báo cáo.

Trọng số/dữ liệu gốc tiếp tục chịu điều khoản của nguồn gốc; gói này không cấp lại giấy phép cho chúng.
