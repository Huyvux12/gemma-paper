# Phạm vi kiểm chứng bản public — 17/10/2026

Producer đầy đủ đã chạy Kaggle T4 trước khi export; xem handoff để phân biệt version1 COMPLETE và version2 Quick Save. Bản public không tự chứa dữ liệu, vì vậy cần private inputs để chạy lại.

Kiểm tra riêng cho lần publication này đã pass:

- Parse toàn bộ source Python, validate hai notebook; không có saved outputs.
- Giải mã cả hai ZIP source nhúng: mỗi member trùng byte với source/context/PINNED công khai. Không có sáu private fingerprint values, raw data, cache hoặc checkpoint trong payload.
- Frozen scientific source khớp byte với gói đầy đủ và source hashes đã khóa.
- Thiếu private runtime config thì báo lỗi; cấu hình riêng được đọc và ghép đúng với public model/source pins.
- Chạy notebook CPU trong thư mục độc lập bằng code nhúng, dùng private assets/config: tái tính saved metrics và rescore các heads trên cache. Tắt `RUN_REFIT` trong bản QA để không train lại; notebook phân phối vẫn giữ tùy chọn refit.
- Chạy hết nhánh GPU-disabled trong notebook GPU, cũng từ code nhúng và thư mục độc lập. Không cài CUDA stack hoặc encode/train backbone trên laptop.
- Tạo lại báo cáo bằng public `src/report.py` từ core/timing của producer đã hoàn tất. Sáu bảng `all_variants`, `main_metrics`, `paired_ci`, `data_splits`, `sugar_categories`, `timing128` trùng byte với bảng công khai.
- Scan public files và decoded payloads cho private data fingerprints và mẫu token xác thực; không phát hiện giá trị cần loại bỏ.

Không chạy thêm một phiên GPU cho thay đổi packaging; không gọi QA CPU/GPU-disabled là full public GPU rerun. Những thay đổi public tập trung vào loader private config và loại metadata dữ liệu khỏi gói công khai. Scientific source frozen giữ nguyên.

`PUBLIC_FILES_SHA256.json` và `tools/verify_public.py` kiểm byte của source/docs/aggregate reports cùng payload notebook sau clone. `verify_repository.py` ở root kiểm cây public toàn repo. `.gitattributes` giữ nguyên line endings để manifest không hỏng bởi Git đổi LF sang CRLF trên Windows.

Executed notebooks, private configs và receipt chi tiết của QA chỉ giữ local dưới `tmp/`; không đưa vào GitHub vì outputs có thể chứa metadata của input riêng.
