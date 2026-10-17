# Bản source và báo cáo công khai

## Cập nhật 17/10/2026

[Bàn giao hiện hành](GEMMA_PAPER_HANDOFF_20261017_VI.md) ghi đầy đủ tiến độ và hướng nâng paper. `gemma_research_public/` bổ sung portable source, frozen Stage11 source/protocol, hai notebook, tài liệu method/kết quả, bảng kết quả tổng hợp, báo cáo HTML/Markdown và tám biểu đồ PNG/SVG. Những bảng này không chứa caption, image ID, predictions theo record hoặc dataset fingerprints. Notebook nhúng source và aggregate context; fingerprint dữ liệu được đọc từ input riêng khi người dùng chạy. Xem README trong gói để chuẩn bị input Kaggle private.

Không công khai dữ liệu, private runtime config, dataset manifest, cache, checkpoint, ZIP kết quả hoặc raw log. `PUBLIC_FILES_SHA256.json` trong gói mới và manifest ở root chỉ xác minh byte của artifact công khai. GPU producer cũ đã chạy hoàn tất; public packaging được kiểm riêng, không phải một lượt GPU mới.

## Cây lịch sử đã công khai 12/10/2026

Toàn bộ 113 file source Python từ gói bàn giao, mọi báo cáo Markdown và notebook Kaggle đã được giữ. Notebook Colab có bản public tương đương về logic, chuyển payload dữ liệu nhúng sang input riêng. Cây public thêm README, hướng dẫn GitHub và verifier byte. Có bảng metric tổng hợp, protocol cấu hình Stage8 và hai định dạng hình trade-off. Kết quả thống kê, GPU/PAC-S và CI được trình bày trong báo cáo Markdown.

| Nguồn trong gói bàn giao | Cây public |
|---|---|
| `baseline.zip` | Source/báo cáo trong `baseline/` |
| `stage8/results/submitted_source_v1.zip` | `stage8/results/submitted_source_v1/` |
| `stage8/results/submitted_source_v2.zip` | `stage8/results/submitted_source_v2/` |

Các source Python và snapshot v1/v2 giữ nguyên byte. Colab public không nhúng record/manifest/caption/hash; dùng `prepare_colab_inputs.py` để tạo input riêng từ `baseline/stage6/uploaded/` của gói đầy đủ. Builder đã được đối chiếu byte/hash với toàn bộ payload gốc; notebook không có output lưu sẵn. `CONTEXT_HANDOFF_VI.md` và báo cáo giữ nguyên nội dung lịch sử; khi nhắc checkpoint/cache/manifest/raw log, đó là artifact trong gói bàn giao đầy đủ riêng. `START_HERE.md` và README root đã cập nhật cho cây GitHub. Code chuẩn bị/package có thể tạo ZIP làm input Kaggle hoặc output bàn giao khi chạy; source được đăng dưới dạng file, không phải archive.

## Artifact chưa công khai và tái lập

Kiểm duyệt tự động chặn checkpoint và một dataset manifest/checksum: yêu cầu hiện tại cho phép context, báo cáo và code, chưa xác nhận công khai model/data artifact hay metadata của chúng trên repo public. Bản public vì vậy không đăng checkpoint, embedding cache, record/annotation dữ liệu, dataset manifest/checksum, prediction theo record, raw log và raw JSON chứa dataset/research metadata. History/selection/benchmark JSON nguyên bản cũng chưa công khai; số liệu chính của chúng có trong báo cáo. Chúng còn trong gói bàn giao đầy đủ của người dùng.

Để chạy train/evaluate từ cache, khôi phục các thư mục dữ liệu/cache/checkpoint trong gói đầy đủ vào layout `baseline/` và `stage8/`. Hoặc dùng downloader/prepare/inference theo README baseline để tạo lại dữ liệu và embedding từ nguồn gốc, tuân thủ điều khoản model/dataset. Verifier gói FULL trong source cần gói đầy đủ, sẽ báo thiếu file trên cây public này. Đọc source/báo cáo/metric tổng hợp không cần khôi phục artifact.

## Kiểm chứng bản public

`REPOSITORY_FILES_SHA256.json` chỉ hash source, tài liệu và kết quả tổng hợp được công khai, không hash các artifact bị chặn. Chạy `python verify_repository.py` từ root để kiểm tra byte. Syntax source Python và cấu trúc hai notebook được kiểm tra; lần xuất bản GitHub không chạy thêm thí nghiệm và không thay đổi kết luận hiện tại.

Các báo cáo/README theo stage được giữ lịch sử, bao gồm trạng thái trung gian đã được hoàn tất ở stage sau. Đọc báo cáo Stage8 và context trước; không suy diễn pending trong một README cũ là trạng thái hiện tại.

## Notebook Colab public

Chạy trên máy có gói đầy đủ:

```bash
python prepare_colab_inputs.py --input-directory /path/to/full/baseline/stage6/uploaded --output /path/to/private/gemma560_private_inputs.json
```

Upload file input riêng vào `/content/gemma560_private_inputs.json` trên Colab, rồi chạy notebook. Có thể đặt biến `GEMMA_560_INPUTS` trỏ tới file khác. Không commit/upload payload này vào repo public. Các cell cài dependency, chạy SigLIP224, kiểm tra hash input và tạo output giữ cùng logic.
