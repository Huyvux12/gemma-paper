# EmbeddingGemma 2 thay encoder CLIPScore — vòng kiểm chứng 4

Ngày 08/10/2026. Kết quả mới: inference thật SigLIP2 Base trên 350 ảnh SugarCrepe và 100 ảnh Flickr8k-Expert; chẩn đoán phụ thuộc ảnh và 24 cấu hình lưu vector Gemma. Không fine-tune.

## Kết luận

Hướng thay encoder vẫn có tín hiệu tốt khi thêm baseline mới: Gemma giữ prefix `A photo depicts ` đạt 90,0%, SigLIP2 Base nguyên bản đạt 80,57%; với cùng prefix CLIP, SigLIP2 đạt 81,14%. Lợi thế quan sát được không chỉ là vượt CLIP cũ. Tuy nhiên Gemma FactChecking chưa chứng minh vượt SigLIP2. Mọi so sánh vòng này dùng lại mẫu đã quan sát ở vòng trước, nên là mở rộng thăm dò, chưa phải xác nhận trên test mới.

Đóng góp đang hướng đến là **nghiên cứu thực nghiệm có kiểm soát về độ bền trước lỗi chi tiết, sự phụ thuộc vào ảnh, độ nhạy prefix và đánh đổi dung lượng vector trong CLIPScore**. Không nhận công thức cosine, MRL hoặc lượng tử hóa int8 là thuật toán mới; chưa bảo đảm đủ tính mới để nộp paper.

## 1. Phần nào mới và phần nào được tái sử dụng?

- Tái sử dụng vector float32 Gemma, CLIP B/32, CLIP L/14 đã chạy; kiểm tra checksum gói gốc 532 tệp và tính lại kết quả gốc.
- Chạy encoder SigLIP2 Base mới trên toàn bộ hai mẫu cũ: 350 ảnh/700 caption SugarCrepe; 100 ảnh/552 caption Flickr8k, 1.656 đánh giá cá nhân.
- Khóa cấu hình SigLIP2 trước inference: raw caption mặc định, prefix CLIP là phân tích phụ; không chọn prefix theo kết quả.
- Mở rộng chẩn đoán bằng mọi ảnh sai và prototype ảnh trung bình; chấm lại vector 128/256/512/768 chiều, float32/float16/int8.
- Đã tạo tập **560 ảnh chưa dùng trong các thí nghiệm của chúng ta**, 80 mỗi loại lỗi, và giao thức xác nhận. **Chưa tải ảnh và chưa chạy inference tập này.** Không cộng 560 vào số mẫu đã kiểm chứng.

## 2. Accuracy và đánh giá của chuyên gia

| Encoder / prefix | Đúng | SugarCrepe accuracy | Flickr8k tau-c | Xếp hạng caption cùng ảnh |
|---|---:|---:|---:|---:|
| CLIP ViT-B/32 | 255/350 | 72.86% | 0.5470 | 82.63% |
| CLIP ViT-L/14 | 248/350 | 70.86% | 0.5413 | 81.92% |
| SigLIP2 Base, caption nguyên bản | 282/350 | 80.57% | 0.5389 | 80.61% |
| SigLIP2 Base, prefix CLIP | 284/350 | 81.14% | 0.5463 | 83.54% |
| Gemma, giữ prefix CLIP | 315/350 | 90.00% | 0.5638 | 83.03% |
| Gemma, FactChecking | 296/350 | 84.57% | 0.5656 | 84.04% |

SugarCrepe: hòa tính sai; mỗi nhóm 50 ảnh; mẫu cân bằng 350 ảnh khác nhau, không phải toàn bộ 7.511 cặp benchmark. Flickr8k: tau-c theo đánh giá cá nhân, không gộp điểm chuyên gia thành 1.656 ảnh độc lập; 990 cặp caption cùng ảnh có điểm trung bình khác nhau.

### So sánh với SigLIP2 mới

| So sánh | Delta accuracy, điểm % | CI 95%, điểm % | McNemar p | Holm p |
|---|---:|---|---:|---:|
| gemma-CLIPLiteral_vs_siglip2-raw | +9.43 | [+4.86, +14.00] | 0.000176309 | 0.000352618 |
| gemma-FactChecking_vs_siglip2-raw | +4.00 | [-1.14, +9.14] | 0.155656 | 0.155656 |

Đối chiếu cùng prefix CLIP: Gemma − SigLIP2 = **+8.86 điểm %**, CI 95% [+4.86, +13.14]; McNemar p=8.83756e-05, phân tích phụ chưa chỉnh nhiều so sánh.

Các CI dùng 10.000 bootstrap ghép cặp theo ảnh, percentile 95%; Holm chỉ điều chỉnh hai p của so sánh raw đã khai báo, không điều chỉnh đồng thời CI. Đây là bằng chứng thăm dò trên mẫu cũ, không gọi là xác nhận đăng ký trước hay SOTA.

### Chi tiết Flickr8k

| So sánh | Delta tau-c | CI 95% tau-c | Delta xếp hạng, điểm % | CI 95% xếp hạng |
|---|---:|---|---:|---|
| gemma-CLIPLiteral_vs_siglip2-raw | +0.0249 | [+0.0020, +0.0506] | +2.42 | [-0.74, +5.80] |
| gemma-CLIPLiteral_vs_siglip2-CLIPLiteral | +0.0174 | [-0.0070, +0.0428] | -0.51 | [-3.53, +2.48] |
| gemma-FactChecking_vs_siglip2-raw | +0.0267 | [+0.0027, +0.0516] | +3.43 | [+0.30, +6.70] |
| gemma-FactChecking_vs_siglip2-CLIPLiteral | +0.0193 | [-0.0054, +0.0434] | +0.51 | [-2.32, +3.38] |

Bootstrap ghép cặp theo 100 ảnh, 2.000 lần. CI phụ chưa chỉnh nhiều so sánh. Gemma nhỉnh hơn SigLIP2 raw về tau-c trong lượt này, nhưng **CI khi so với SigLIP2 cùng prefix CLIP chứa 0**. Vì vậy chưa kết luận cải thiện chung đánh giá caption theo chuyên gia; chưa chứng minh tương đương khi CI chứa 0. Kết quả với CLIP cũ ở vòng trước vẫn có CI chứa 0.

## 3. Model có thực sự dùng ảnh?

Giữ nguyên từng cặp caption, thay ảnh đúng bằng **từng ảnh trong 349 ảnh còn lại**: 122.150 phép ghép sai/encoder-prefix. Số này là số phép tính phụ thuộc nhau, không phải 122.150 mẫu độc lập. Mỗi encoder còn được chấm với cùng 2.000 hoán vị ảnh không có điểm cố định. Khoảng 2,5–97,5% của các hoán vị là khoảng kết quả dưới ghép sai, không phải CI accuracy.

| Cấu hình | Ghép ảnh đúng | Ghép mọi ảnh sai, raw cosine | Prototype ảnh khác, leave-one-out |
|---|---:|---:|---:|
| Gemma, giữ prefix CLIP | 90.00% | 47.70% | 48.00% |
| Gemma, FactChecking | 84.57% | 49.12% | 46.86% |
| CLIP ViT-B/32 | 72.86% | 52.64% | 54.00% |
| CLIP ViT-L/14 | 70.86% | 52.73% | 54.57% |
| SigLIP2 Base, caption nguyên bản | 80.57% | 49.46% | Không chạy |

Gemma prefix CLIP giảm từ 90,0% xuống 47,70% khi ảnh bị đổi; prototype đạt 48,0%. Kết quả ủng hộ sự phụ thuộc vào ảnh cụ thể trong mẫu này. Không đồng nghĩa chứng minh model hiểu mọi quan hệ, không loại trừ ghi nhớ dữ liệu tiền huấn luyện, và prototype không phải baseline language model hoàn chỉnh.

**Lưu ý về công thức chấm:** với ảnh sai của SigLIP2 raw, công thức `max(cosine,0)` làm cả hai caption cùng điểm 0 ở 67,74% phép ghép. Vì hòa tính sai, accuracy ghép sai sau clipping chỉ là 16,36%; raw cosine là 49,46%. Không dùng con số 16,36% để nói SigLIP2 ít thiên lệch hay phụ thuộc ảnh mạnh hơn Gemma. Bảng trên dùng raw cosine để đọc đối chứng ảnh sai; mọi accuracy benchmark chính vẫn giữ công thức CLIPScore.

## 4. Nén vector: lợi ích và mất mát

Giữ prefix CLIP. Int8 đối xứng theo từng vector: cắt chiều → normalize → scale=max(abs)/127 → làm tròn → lưu int8 cùng scale float32 → giải lượng tử và normalize khi chấm. Đây là nén vector sau inference; không lượng tử hóa trọng số encoder, không đo tốc độ tính tích vô hướng int8.

| Chiều / định dạng | Byte/vector thô | Accuracy | Delta vs 768-f32, điểm % | CI 95% delta | Flickr8k tau-c |
|---|---:|---:|---:|---|---:|
| 768 / float32 | 3072 | 90.00% | +0.00 | [+0.00, +0.00] | 0.5638 |
| 768 / float16 | 1536 | 90.00% | +0.00 | [+0.00, +0.00] | 0.5639 |
| 768 / int8 | 772 | 89.71% | -0.29 | [-0.86, +0.00] | 0.5638 |
| 512 / float32 | 2048 | 88.86% | -1.14 | [-3.14, +0.57] | 0.5566 |
| 256 / float32 | 1024 | 87.43% | -2.57 | [-4.86, -0.29] | 0.5543 |
| 256 / int8 | 260 | 87.71% | -2.29 | [-4.57, +0.00] | 0.5540 |
| 128 / int8 | 132 | 81.14% | -8.86 | [-12.86, -4.86] | 0.5051 |

256-int8 cần 260 byte thay vì 3.072 byte, giảm khoảng **11,8 lần** payload vector; tính cả scale nhưng chưa tính metadata, header tệp, chỉ mục hay chi phí search. Tệp `gemma_256_int8.npz` có payload 1.001.000 byte và chứa cả các prefix đã có; dung lượng tệp có header lớn hơn payload. Mã giải lượng tử đọc tệp thật được kiểm tra lại.

768-int8 mất 1 dự đoán đúng trong 350 ảnh; 256-int8 mất ròng 8 dự đoán đúng nhưng có 18 quyết định khác so với 768-f32. Không gọi hai cấu hình tương đương hoặc non-inferior chỉ vì CI chứa 0. Int8 không đảm bảo tăng accuracy; kết quả tăng nhẹ so 256-f32 là nhiễu thứ tự điểm ở mẫu này, không phải lợi ích đã xác nhận.

FactChecking được báo cáo đủ trong `diagnostics_results.json`; không dùng kết quả 256d FactChecking gần 768d để suy ra nén mọi prefix đều không mất chất lượng.

## 5. Phân tích nhóm lỗi

| Nhóm lỗi, 50 ảnh/nhóm | CLIP L/14 | SigLIP2 raw | SigLIP2 prefix CLIP | Gemma prefix CLIP |
|---|---:|---:|---:|---:|
| add_att | 74.00% | 82.00% | 84.00% | 90.00% |
| add_obj | 80.00% | 86.00% | 88.00% | 94.00% |
| replace_att | 74.00% | 84.00% | 86.00% | 96.00% |
| replace_obj | 90.00% | 96.00% | 96.00% | 98.00% |
| replace_rel | 68.00% | 68.00% | 72.00% | 84.00% |
| swap_att | 60.00% | 78.00% | 70.00% | 86.00% |
| swap_obj | 50.00% | 70.00% | 72.00% | 82.00% |

Từng nhóm chỉ có 50 cặp, nên bảng mang tính mô tả. Hai nhóm quan hệ và hoán đổi vai trò vẫn là nơi còn nhiều lỗi; không bỏ các nhóm khó khỏi kết quả tổng.

## 6. Công bằng, chi phí và kiểm chứng

SigLIP2 Base checkpoint cố định `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`; trọng số SHA-256 đối chiếu metadata Hugging Face. Tổng tham số được nạp: 375.187.970, so với Gemma text+vision ở vòng trước 438.760.448. Dùng native preprocessing mỗi checkpoint, float32, CPU 6 threads, batch ảnh 4; không tối ưu riêng một model. Đây là đối chiếu cấu hình sử dụng thực tế, không phải cùng kiến trúc hoặc cùng token ảnh.

Không caption SugarCrepe nào bị cắt ở context 64 token của SigLIP2 (raw tối đa 54 token, prefix CLIP tối đa 57). Đối chiếu joint-forward calibrated logits với cosine từ encoder riêng, sai số nhỏ hơn 1e-3; sigmoid và scale/bias không đảo thứ tự caption của cùng ảnh. Log inference, revision, token audit và timing nằm trong metadata.

SigLIP2 mã hóa 350 ảnh khoảng 45,8 giây ở lượt này. **Không chia trực tiếp với timing Gemma của phiên trước để tuyên bố lợi thế tốc độ**, vì chưa chạy lại các encoder trong cùng phiên đo phần cứng/phần mềm. Chỉ giữ kết luận cũ: trong lượt CPU trước Gemma chậm hơn CLIP L/14 khoảng 7,7 lần. Nén vector không giảm thời gian encoder.

Kiểm tra độc lập tính score bằng số học scalar, đối chiếu quyết định trong CSV, giải lượng tử tệp int8 thật, kiểm checksum dữ liệu và protocol. Xem `independent_verification.json`, `original_package_verification.json` và log tính lại kết quả gốc.

## 7. Các nghiên cứu gần nhất và phần chưa mới

| Nghiên cứu | Đã làm gì liên quan | Hệ quả với đề tài này |
|---|---|---|
| CLIPScore, EMNLP 2021 | Dùng similarity ảnh–caption để đánh giá không cần tham chiếu | Giữ và ghi công công thức gốc; không đặt tên cosine như metric mới |
| PAC-S / PAC-S++ | Fine-tune không gian CLIP bằng positive augmentation | Baseline metric chuyên dụng cần bổ sung trước khi kết luận rộng |
| SPECS, 2025 | Đánh giá caption dài và specificity, huấn luyện objective mới | “Phạt chi tiết sai” tự nó đã có; thay encoder không thay thế việc đối chiếu SPECS |
| L-CLIPScore, 2025 | Nén/distill dual encoder để làm metric nhẹ | Nén metric không phải ý tưởng chưa từng có; ở đây mới đo nén vector, chưa nén encoder |
| RelCLIPScore / RelSIGLIPScore, 2025 | Metric quan hệ, dùng cả BLIP và SigLIP | Thay backbone và metric quan hệ đã có; không tuyên bố “đầu tiên dùng SigLIP thay CLIP” |
| EmbeddingGemma 2 model card | Có MRL 128/256/512/768 và tùy chỉnh prefix | MRL là tính năng model; giá trị đề tài là kiểm chứng tradeoff trong ứng dụng này |

Tra cứu tên model kết hợp CLIPScore chưa xác lập một paper trực tiếp trùng đúng cấu hình này, nhưng **không tìm thấy không chứng minh chưa ai làm**. Cần rà lại trước khi nộp; phân biệt bài thực nghiệm có kiểm soát với đề xuất thuật toán mới.

Tên làm việc: **Controlled Encoder Replacement for Reference-Free Caption Evaluation: Compositional Robustness, Prompt Sensitivity, and Storage Trade-offs**.

## 8. Giao thức xác nhận đã chuẩn bị

`next_560_manifest.json`: 560 ảnh khác tất cả ảnh dùng trong các vòng trước, 80 mỗi nhóm. Nhóm hiếm được lấy trước để tránh dùng hết ảnh cho nhóm khác. Thử 100 mỗi nhóm không đủ swap_obj còn lại sau loại trùng, nên chốt 80 mỗi nhóm trước inference. Đây là holdout đối với thí nghiệm của chúng ta, không chứng nhận sạch tiền huấn luyện.

`next_protocol.json`: một so sánh chính Gemma prefix CLIP 768-f32 vs SigLIP2 raw, McNemar exact, bootstrap paired theo ảnh; secondary giữ CLIP cũ, SigLIP2 cùng prefix và toàn bộ curve nén; không đổi prefix khi thấy kết quả. Tập này hiện **chưa chạy**, nên báo cáo hiện chỉ có 350 ảnh compositional và 100 ảnh human-rated đã chạy.

Để tiến đến paper hoàn chỉnh: chạy holdout đã khóa hoặc toàn bộ SugarCrepe với bootstrap theo ảnh nếu ảnh lặp; thêm benchmark khác nguồn như SugarCrepe++/Winoground và metric chuyên dụng PAC-S hoặc SPECS phù hợp ngữ cảnh; mở rộng đánh giá chuyên gia; đo latency/memory các encoder trong cùng phiên máy. Nếu lợi thế không lặp lại, thu hẹp kết luận hoặc dừng hướng paper thay encoder; không tiếp tục đổi prefix để tìm test thắng.

## Nguồn gốc và tài liệu

- CLIPScore: https://aclanthology.org/2021.emnlp-main.595/
- SugarCrepe repo: https://github.com/RAIVNLab/sugar-crepe
- EmbeddingGemma 2: https://huggingface.co/google/embeddinggemma-2
- SigLIP2 Base: https://huggingface.co/google/siglip2-base-patch16-224
- PAC-S: https://arxiv.org/abs/2303.12112 ; PAC-S++: https://arxiv.org/abs/2410.07336
- SPECS: https://arxiv.org/abs/2509.03897
- L-CLIPScore: https://arxiv.org/abs/2507.08710
- RelCLIPScore / RelSIGLIPScore: https://arxiv.org/abs/2509.01209
- Survey caption evaluation: https://arxiv.org/abs/2503.14604
