# Bộ bàn giao EmbeddingGemma 2 — 08/10/2026

Đọc `EmbeddingGemma2_Research_Report_VI.md` để xem kết quả cuối cùng. Bộ này gồm báo cáo, mã nguồn và bằng chứng phục vụ paper thực nghiệm; chưa phải bản thảo theo mẫu một venue.

## Hai gói bàn giao

- **FULL**: source, báo cáo, config, manifest/protocol, metadata, CSV/JSON và vector đã dùng. Chạy lại analysis offline không cần GPU hoặc trọng số model.
- **SOURCE**: cùng cây mã và tệp văn bản hỗ trợ, không có `.npz`/`.npy`. Dùng FULL để chạy analysis từ vector đã lưu.

Trọng số model, ảnh gốc, parquet nguồn và môi trường venv không được đóng gói. Các downloader và receipt giữ đúng revision/hash để tải lại. Mỗi model và dữ liệu có điều khoản riêng; snapshot SPECS giữ LICENSE nguồn. Gói tổng hợp không cấp lại một license chung cho các thành phần bên thứ ba.

## Trạng thái cuối cùng

Các báo cáo/README từng stage được giữ nguyên để lưu lịch sử. Trạng thái pending Sig224 trong Stage5 đã hoàn tất ở Stage6; human chưa hoàn chỉnh trong Stage5/6 đã hoàn tất ở Stage7. `stage6/verified_confirmation_results.json` và `stage7/human_results.json` là kết quả cuối cùng của hai endpoint.

560 ảnh là tập con của 1.005 ảnh SugarCrepe; không cộng thành hai test độc lập. Flickr8k full1.000 gồm 900 ảnh mới và 100 ảnh development. Unseen chỉ có nghĩa trong chuỗi nghiên cứu; không loại trừ contamination từ pretraining.

## Chạy analysis offline

Python3.12 được dùng. `requirements-analysis.txt` pin đúng NumPy/SciPy của vòng xác minh cuối. Chạy các lệnh từ thư mục gốc có README này:

```sh
python -m venv .venv
# Linux/macOS
. .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements-analysis.txt
python verify_bundle.py
python stage6/analyze_confirmation.py
python stage7/analyze_human.py
python stage7/verify_completion.py
```

`verify_bundle.py` kiểm tra hash gói FULL; `verify_source_bundle.py` kiểm tra gói SOURCE. Hai verifier chỉ kiểm tra byte bàn giao, không tải lại ảnh/model. Đặt `OPENBLAS_NUM_THREADS=1` trên Linux để hạn chế threading khi chạy analysis. Human analysis có bootstrap2.000 và within-image5.000 nên cần thời gian CPU.

Analysis ghi lại JSON/CSV; dùng một bản copy nếu muốn giữ gói phát hành nguyên trạng. Khác version thư viện có thể thay đổi chữ số cuối hoặc serialization. Chạy hash verifier trước khi sửa/chạy lại.

Các lệnh bổ sung từ vector trong FULL:

```sh
python stage5/analyze_560.py
python stage5/verify.py --scores-only
python full_recovered/embeddinggemma2_full_validation/full_validation/analyze.py
```

Có thể dùng `python stage5/verify.py` để chạy thêm đối chiếu pilot350 với nguồn full/subset. Lệnh `full_validation/analyze.py` mặc định chấm SugarCrepe. **Không dùng `analyze.py --human` của archive cũ**: Gemma human cũ chưa hoàn chỉnh. Dùng Stage7 cho human.

## Lặp lại human inference

Cài một cặp Torch/torchvision phù hợp CPU hoặc CUDA của máy trước. Metadata giữ môi trường đã chạy; `requirements-inference-reference.txt` ghi dependency tham chiếu và không tự chọn CUDA.

```sh
python -m pip install -r requirements-inference-reference.txt
python stage7/download_required.py
python stage7/download_siglip224.py
python stage7/materialize_expert.py
python stage7/run_human.py gemma --threads 6
python stage7/run_human.py siglip224 --threads 2
```

Runner sẽ thoát khi thấy metadata hoàn chỉnh. Để chạy fresh, giải nén một bản copy và chuyển các thư mục `stage7/runs/gemma-human` và `stage7/runs/siglip224-human` sang chỗ khác trước khi chạy. Downloader/materialize vẫn cần thiết vì raw input không đóng gói. Gemma có cơ chế cache bridge; không có cache thì chạy fresh.

Trong SOURCE, metadata cũng được giữ để trace mặc dù không có embedding arrays: chuyển thư mục output cần chạy tương tự. Bốn baseline human khác dùng nguồn full-validation gốc. Muốn chạy cả sáu fresh cần tải checkpoint theo README gốc và ánh xạ output vào layout Stage7. FULL đã có đủ sáu encoder nên analysis không cần inference.

## Lặp lại SugarCrepe inference

Notebook `EmbeddingGemma2_560_SigLIP224_Colab.ipynb` nhúng đầu vào và tải pinned Sig224 cho đúng mẫu560. Người dùng đã chạy và kết quả được xác minh trong `stage6/uploaded/`; không cần chạy lại để đọc phép chính.

Nguồn full SugarCrepe ở `full_recovered/embeddinggemma2_full_validation/`. `replacement/download_models.py` tải Gemma/CLIP; `full_validation/download_specs.py` và `download_siglip.py` tải đối chứng. README gốc hướng dẫn restore media và chạy encoder. Các `prepare.py`, `recover_vectors.py`, `make_report.py`, `package.py` phục vụ chuẩn bị/bàn giao lịch sử; một số cần archive hoặc đường dẫn phiên cũ. Chúng không phải entrypoint portable cho analysis. Các lệnh offline ở trên là entrypoint được kiểm tra cho bộ bàn giao này.

## Tìm nhanh mã

| Việc | Source |
|---|---|
| Score CLIP / adapter Gemma | `research/embeddinggemma2_clipscore_replacement/gemma_clipscore.py`, `score_adapter.py` |
| Encode năm baseline gốc | `full_recovered/embeddinggemma2_full_validation/full_validation/encoders.py`, `run.py` |
| Codec vector | `stage4/vector_codec.py`; `stage5/analyze_560.py`; `stage7/analyze_human.py` |
| Phép chính560, p/CI, Holm | `stage6/analyze_confirmation.py` |
| Human, image clusters, weighted tau-c | `stage7/analyze_human.py` |
| Scalar / shape / CSV verification | `stage5/verify.py`; `stage6/analyze_confirmation.py`; `stage7/verify_completion.py` |

CSV/JSON nằm cạnh script analysis. Protocol, revision, hash manifest và metadata từng run đều được giữ. Các trạng thái và timing lịch sử không phải claim tốc độ; 11,82 lần là mức giảm payload vector.
