# Thay encoder của CLIPScore bằng EmbeddingGemma 2 — kiểm chứng thực nghiệm


Ngày: 08/10/2026. Inference thật trên CPU; không fine-tune; không dùng API chấm điểm.


## 1. Kết luận và hướng nên theo


Kết quả ủng hộ tiếp tục thay encoder của CLIPScore bằng EmbeddingGemma 2: cấu hình chính thắng cả baseline mạnh ViT-L/14 trên tập compositional mới. Trên Flickr8k-Expert, mức tương quan của Gemma cao hơn về point estimate nhưng các CI so sánh hiện còn chứa 0; chưa có bằng chứng đủ rõ về cải thiện đánh giá của con người, và cũng chưa chứng minh tương đương. Không gộp hai loại kết quả thành một kết luận thắng mọi bài toán.


Hướng cụ thể: **đánh giá caption không cần caption tham chiếu bằng cách thay encoder trong CLIPScore**, đồng thời đo độ bền trước lỗi chi tiết và chi phí lưu vector. Điểm nghiên cứu là kiểm chứng khả năng thay thế trong một phương pháp có sẵn; chưa tuyên bố công thức, kiến trúc hay thuật toán mới.


## 2. Đã rà những chỗ có thể thay


| Phương pháp có sẵn | Thành phần có thể thay | Quyết định trong lượt này |
|---|---|---|
| CLIPScore | CLIP image/text encoder | Đã viết adapter; đã chạy SugarCrepe và Flickr8k-Expert |
| CLAP cosine / CLAPScore-style | CLAP audio/text encoder | Chưa chọn làm hướng chính: kết quả vòng trước Gemma 70% < CLAP 81,5% trên 200 cặp mới |
| BERTopic | SentenceTransformer hoặc backend đa phương thức | Khả thi theo API có sẵn; chưa benchmark trong lượt này |
| RAG / semantic retrieval | Bộ tạo vector của retriever | Khả thi; chưa chứng minh lợi ích với benchmark RAG trong lượt này |
| SPECS / PAC-S | Backbone và quy trình học metric | Chỉ đổi backbone không tái lập phương pháp đã huấn luyện; chưa chạy checkpoint này |


CLIPScore được chọn vì có mã nguồn, quy tắc chấm đơn giản và dữ liệu đánh giá có sẵn; đồng thời vòng trước đã cho tín hiệu tốt trên ảnh. Giữ audio làm đối chứng âm, không suy diễn rằng model đa phương thức sẽ thắng mọi specialist.


## 3. Chính xác đã thay gì?


CLIPScore gốc: ảnh → CLIP image vector; caption → CLIP text vector; score = `2.5 × max(cosine, 0)`. Adapter thay hai vector bằng Gemma, giữ công thức. Không đưa caption, nhãn hay điểm của con người vào image encoder.


- **Thay nguyên bản:** giữ prefix `A photo depicts `, chỉ đổi encoder.
- **Cấu hình phù hợp Gemma:** prefix `FactChecking`, đã chọn từ vòng trước; đây là đổi encoder **và** prefix, không quy mọi cải thiện cho encoder.
- Prefix `None`, `SearchQuery`, `SentenceSimilarity`, các chiều 128/256/512 được báo cáo đầy đủ như phân tích phụ; không chọn theo test để thay giả thuyết chính. SentenceSimilarity được bổ sung khi rà lại vector vòng trước (92,14% trên 140 ảnh cũ), trước khi có bất kỳ score caption SugarCrepe mới hoặc Gemma Flickr8k mới; addendum có timestamp trong protocol.
- Gemma chỉ nạp text+vision; bỏ audio encoder. Cắt vector đầu rồi L2-normalize lại. Tất cả dùng float32.


## 4. Giao thức đã khóa trước khi chấm test


**SugarCrepe:** 350 cặp / 350 ảnh độc lập, 50 mỗi loại lỗi, seed 20261008. Không trùng bất cứ ảnh nào trong 210 cặp của hai vòng trước. Chọn đúng khi score caption đúng lớn hơn caption sai; hòa tính sai. Đây là mẫu cân bằng theo nhóm, không phải kết quả toàn bộ 7.511 cặp benchmark.


**Flickr8k-Expert:** 100 ảnh lấy ngẫu nhiên, toàn bộ 552 cặp ảnh–caption không nằm trong caption tham chiếu, 1656 điểm cá nhân của chuyên gia. Kendall tau-c theo quy trình CLIPScore; bootstrap ghép cặp theo ảnh 2.000 lần. Đây là benchmark phụ, không phải tái lập toàn bộ Flickr8k-Expert.


**Hai phép so sánh chính:** Gemma FactChecking 768d so B/32 và L/14; bootstrap ghép cặp theo ảnh 10.000 lần; kiểm định McNemar exact dạng binomial; Holm trên hai phép so sánh. CI 95% là percentile CI chưa điều chỉnh đồng thời. Không có tối ưu tham số trên test.


## 5. Kết quả mới: accuracy và tương quan với con người


| Hệ thống | Đúng / 350 | Accuracy | Kendall tau-c, 100 ảnh |
|---|---|---|---|
| CLIPScore ViT-B/32 | 255/350 | 72.86% | 0.5470 |
| CLIPScore ViT-L/14 | 248/350 | 70.86% | 0.5413 |
| Gemma 768d, prefix CLIP gốc | 315/350 | 90.00% | 0.5638 |
| Gemma 768d, FactChecking | 296/350 | 84.57% | 0.5656 |
| Gemma 256d, FactChecking | 299/350 | 85.43% | 0.5547 |
| Gemma 128d, FactChecking | 281/350 | 80.29% | 0.5074 |
| Gemma 768d, SentenceSimilarity (phụ) | 312/350 | 89.14% | 0.5594 |


Tau-c cao hơn thể hiện xếp hạng gần điểm của chuyên gia hơn; không phải phần trăm caption đúng. Điểm cosine tuyệt đối giữa các model không có cùng thang đo nên không dùng mean score để đánh giá model nào tốt hơn.


### So sánh chính trên SugarCrepe


| So sánh | Chênh lệch accuracy | CI 95%, điểm % | p Holm | Gemma đúng riêng / baseline đúng riêng |
|---|---|---|---|---|
| FactChecking768_vs_B32 | +11.71 điểm % | [+6.57, +17.14] | 3.11406e-05 | 68 / 27 |
| FactChecking768_vs_L14 | +13.71 điểm % | [+8.57, +18.86] | 7.76363e-07 | 69 / 21 |


### Thay nguyên bản và các cấu hình phụ


| So sánh phụ | Delta điểm % | CI 95% | p Holm trong 5 so sánh phụ |
|---|---|---|---|
| gemma-CLIPLiteral-768_vs_L14 | +19.14 | [14.29, 24.29] | 3.11234e-12 |
| gemma-None-768_vs_L14 | +18.29 | [13.14, 23.43] | 4.60218e-11 |
| gemma-SearchQuery-768_vs_L14 | +17.71 | [12.57, 22.86] | 9.7872e-11 |
| gemma-FactChecking-256_vs_L14 | +14.57 | [9.14, 20.0] | 5.53314e-07 |
| gemma-FactChecking-128_vs_L14 | +9.43 | [3.71, 14.86] | 0.00133183 |


Thay nguyên bản và nén FactChecking là phân tích phụ trong protocol. None và SearchQuery là phân tích thăm dò. Nhóm năm phép so sánh này được điều chỉnh Holm bổ sung sau khi có kết quả; không xem đây là nhóm giả thuyết chính đã đăng ký trước. Không thay hai kiểm định chính bằng cấu hình có kết quả cao nhất.



### So sánh phụ với chuyên gia


| So sánh | Delta tau-c | CI 95% theo ảnh |
|---|---|---|
| gemma-FactChecking-768_vs_clip-CLIPScore-512 | +0.0186 | [-0.0094, +0.0455] |
| gemma-FactChecking-768_vs_clip_large-CLIPScore-768 | +0.0243 | [-0.0000, +0.0480] |
| gemma-CLIPLiteral-768_vs_clip_large-CLIPScore-768 | +0.0225 | [-0.0025, +0.0465] |


Các kết quả tương quan là phân tích phụ. Không biến p-value hoặc CI theo hàng caption thành bằng chứng có 1.656 mẫu ảnh độc lập.


### Chẩn đoán: xếp hạng các caption của cùng một ảnh


Lấy mọi cặp caption cùng ảnh có điểm chuyên gia trung bình khác nhau: 990 cặp từ 98 ảnh. Phép này giảm ảnh hưởng của khác biệt mức cosine giữa các ảnh. Đây là phân tích phụ đã bổ sung trước khi có điểm caption Gemma mới.


| Hệ thống | Chọn đúng caption được chuyên gia chấm cao hơn |
|---|---|
| CLIPScore ViT-B/32 | 82.63% |
| CLIPScore ViT-L/14 | 81.92% |
| Gemma 768d, prefix CLIP gốc | 83.03% |
| Gemma 768d, FactChecking | 84.04% |
| Gemma 256d, FactChecking | 82.93% |
| Gemma 128d, FactChecking | 77.47% |
| Gemma 768d, SentenceSimilarity (phụ) | 83.43% |

| So sánh | Delta điểm % | CI 95% theo ảnh |
|---|---|---|
| gemma-FactChecking-768_vs_clip-CLIPScore-512 | +1.41 | [-2.39, 5.13] |
| gemma-FactChecking-768_vs_clip_large-CLIPScore-768 | +2.12 | [-0.61, 5.07] |
| gemma-CLIPLiteral-768_vs_clip_large-CLIPScore-768 | +1.11 | [-2.18, 4.46] |

## 6. Phân tích loại lỗi trên 350 ảnh mới


| Loại lỗi | CLIP B/32 | CLIP L/14 | Gemma 768d, CLIP prefix | Gemma 768d FC | Gemma 256d FC |
|---|---|---|---|---|---|
| add_att | 74.0% | 74.0% | 90.0% | 78.0% | 84.0% |
| add_obj | 78.0% | 80.0% | 94.0% | 90.0% | 92.0% |
| replace_att | 88.0% | 74.0% | 96.0% | 94.0% | 88.0% |
| replace_obj | 80.0% | 90.0% | 98.0% | 96.0% | 98.0% |
| replace_rel | 60.0% | 68.0% | 84.0% | 86.0% | 84.0% |
| swap_att | 64.0% | 60.0% | 86.0% | 82.0% | 82.0% |
| swap_obj | 66.0% | 50.0% | 82.0% | 66.0% | 70.0% |


`add`: thêm chi tiết không có; `replace`: thay chi tiết; `swap`: hoán đổi vai trò/thuộc tính; `obj`: vật thể; `att`: thuộc tính; `rel`: quan hệ. Mỗi ô chỉ có 50 cặp, nên phân tích từng nhóm mang tính mô tả.


## 7. Nén vector và chi phí thực đo


| Gemma FactChecking | Bytes / vector float32 | Accuracy | Delta so 768d | CI 95%, điểm % |
|---|---|---|---|---|
| 128d | 512 | 80.29% | -4.29 | [-8.29, -0.29] |
| 256d | 1024 | 85.43% | +0.86 | [-2.29, 3.71] |
| 512d | 2048 | 84.00% | -0.57 | [-2.57, 1.43] |
| 768d | 3072 | 84.57% | 0 | — |


256d giảm dung lượng vector thô 3 lần so 768d và 2 lần so CLIP B/32 512d. Không giảm chi phí chạy encoder. Không gọi cấu hình 256d “tương đương” chỉ vì point accuracy gần nhau; chưa khóa giao thức non-inferiority.


| Encoder | Tham số được nạp | Encode 350 ảnh (s) | Peak RSS tiến trình (MiB) |
|---|---|---|---|
| gemma | 438,760,448 | 1469.6 | 3161.4 |
| clip | 151,277,313 | 22.8 | 1179.4 |
| clip_large | 427,616,513 | 190.4 | 2243.3 |


Inference chính chạy tuần tự, CPU 6 threads, batch 4. Các thời gian mô tả máy và pipeline này, gồm preprocess và checkpoint vector sau mỗi batch; chưa phải benchmark latency tối ưu, không chứng minh chạy nhanh trên điện thoại hoặc GPU của người dùng.


## 8. Kiểm chứng


- Checkpoint cố định revision, trọng số lớn đối chiếu SHA-256 với metadata chính thức.
- Source SugarCrepe khớp byte với repo chính thức hiện tại. Nhãn chuyên gia của mọi caption được chọn khớp dữ liệu trong repo tác giả EXPERT.
- Cùng checkpoint B/32 được ánh xạ vào code CLIP gốc, load strict: kiểm tra 30 caption / 6 ảnh, sai số score tối đa 4.17e-07.
- Không có caption bị CLIP cắt ở giới hạn 77 token: SugarCrepe tối đa 55 token, Flickr8k tối đa 39 token (đã gồm prefix và token đặc biệt); xem `token_length_audit.json`.
- Tái tính tất cả điểm bằng phép tính scalar float64 độc lập; kiểm tra toàn bộ accuracy/tau-c/within-image ranking và SHA-256 media. Xem `independent_verification.json`.


## 9. Có đủ để viết paper chưa?


**Đủ cho báo cáo khả thi và quyết định hướng; chưa đủ để khẳng định một đóng góp xuất bản mạnh.** Đổi encoder và đặt tên metric mới tự nó chưa phải tính mới. CLIPScore, PAC-S/SPECS và các nghiên cứu robustness đã nghiên cứu caption evaluation; MIEB đã bao phủ compositionality. Cần định vị vào một vấn đề và một kết luận cụ thể có thể tái lập.


Tên nghiên cứu tạm: **Replacing CLIP in Reference-Free Caption Evaluation: Robustness and Compression Trade-offs with EmbeddingGemma 2**.


Bước nâng thành nghiên cứu hoàn chỉnh: chạy full SugarCrepe và full Flickr8k-Expert; thêm một bộ có caption sinh thực tế / domain khác; chạy checkpoint caption-metric đã huấn luyện như SPECS hoặc PAC-S; khóa tiêu chí chất lượng và ngưỡng chỉ trên validation riêng. Nếu cần cải tiến phương pháp, thêm cơ chế đã có như claim decomposition hoặc calibration rồi làm ablation; chưa xem những cơ chế chưa chạy là kết quả.


Giới hạn: dữ liệu có trước cutoff huấn luyện nên chưa loại được contamination; chỉ hai CLIP checkpoint và một Gemma; native preprocessing và độ dài context khác nhau; chưa test tiếng Việt, caption dài, video, khả năng phát hiện mọi hallucination, latency real-time hoặc tương đương SOTA. Test ảnh mới độc lập với thí nghiệm cũ, không có nghĩa là độc lập với pretraining.


## 10. Mã dùng ngay và tài liệu gốc


Xem `README.md`, `gemma_clipscore.py`, `score_adapter.py`. Archive có toàn bộ mẫu media, vector và logs; không đóng gói trọng số model.


- Model: https://huggingface.co/google/embeddinggemma-2
- Hướng dẫn: https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers
- CLIPScore: https://arxiv.org/abs/2104.08718 ; https://github.com/jmhessel/clipscore
- SugarCrepe: https://arxiv.org/abs/2306.14610 ; https://github.com/RAIVNLab/sugar-crepe
- Human evaluation: https://huggingface.co/datasets/yuwd/Flickr8k-HumanEval ; https://github.com/hjkim811/EXPERT
- Robustness trước đây: https://arxiv.org/abs/2305.14998
- SPECS: https://arxiv.org/abs/2509.03897 ; https://github.com/mbzuai-nlp/SPECS
- BERTopic embeddings: https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html
- MIEB task list: https://docs.mteb.org/overview/available_benchmarks/



`results.json`, `human_results.json` lưu tất cả prefix/chiều đã chạy, bao gồm kết quả không được chọn làm cấu hình chính. Kết quả vòng trước nằm trong `prior_stage2_report.md`, không được gộp vào test mới.