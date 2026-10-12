# Báo cáo tổng hợp: EmbeddingGemma 2 trong đánh giá caption ảnh

Ngày chốt: 08/10/2026. Phiên bản bàn giao sau khi hoàn tất SugarCrepe và Flickr8k-Expert.

## 1. Tóm tắt và trạng thái cuối cùng

Nghiên cứu kiểm tra khả năng dùng EmbeddingGemma 2 làm encoder trong một metric tương đồng ảnh–caption không cần caption tham chiếu, giữ công thức CLIPScore cho cấu hình Gemma. Không fine-tune, không học threshold từ nhãn test, không đề xuất hay kiểm chứng một thuật toán huấn luyện mới.

Trên mẫu SugarCrepe xác nhận đã khóa gồm 560 ảnh, Gemma literal 768 chiều float32 đạt 503/560 = 89,82%, so với SigLIP2-224 raw 426/560 = 76,07%. Chênh lệch +13,75 điểm phần trăm, CI95% [10,18; 17,50], McNemar exact p = 2,56 × 10^-13. Trên Flickr8k-Expert với 900 ảnh chưa dùng trong phát triển, Kendall tau-c là 0,5449 so với 0,5243 của SigLIP2-224 native raw; chênh lệch +0,0206, CI95% [0,0118; 0,0290].

Nén vector Gemma xuống 256 chiều int8, kèm một scale float32 cho mỗi vector, giảm payload từ 3.072 xuống 260 byte/vector, tương đương 11,82 lần. Chất lượng có đánh đổi: trên Flickr8k-Expert tau-c giảm khoảng 0,0125. Chưa chứng minh nén vector làm encoder nhanh hơn.

Các đánh giá cơ bản hiện đã hoàn tất. Có đủ chất liệu để viết một paper thực nghiệm trong phạm vi dưới đây. Bản này là báo cáo nghiên cứu và bàn giao bằng chứng, chưa phải bản thảo theo mẫu của một tạp chí/hội nghị; không khẳng định tính mới tuyệt đối hay khả năng được chấp nhận.

**Trạng thái cuối cùng ưu tiên:** SigLIP2-224 chính đã hoàn tất ở Stage6; Gemma human evaluation đã hoàn tất ở Stage7. Những dòng “pending”, “chưa có human correlation”, hoặc đề xuất tiếp tục trong các báo cáo lịch sử phản ánh thời điểm cũ, không phải trạng thái hiện tại. Không chỉnh sửa ngược các protocol hay báo cáo cũ.

## 2. Câu hỏi nghiên cứu và phạm vi

1. Khi giữ cấu hình đã chốt, Gemma phân biệt caption đúng với caption sai về thành phần ảnh tốt đến đâu so với CLIP và SigLIP2?
2. Điểm Gemma tương quan với đánh giá chuyên gia đến đâu, và kết quả có khác giữa tương quan toàn cục với xếp hạng caption trong cùng ảnh không?
3. Giảm chiều và lượng tử hóa vector sau encoder làm thay đổi chất lượng và dung lượng lưu trữ thế nào?

Phạm vi là caption ảnh ở hai benchmark hiện có. Chưa đánh giá audio/video, hệ thống lọc hallucination thời gian thực, threshold phát hiện lỗi, caption dài tổng quát, hay mọi miền dữ liệu. “Chưa dùng phát triển/unseen” chỉ có nghĩa trong chuỗi thí nghiệm này; không đảm bảo ngoài dữ liệu pretraining của các encoder.

Đóng góp có thể trình bày trong paper là một đánh giá có kiểm soát của encoder đa phương thức, đối chứng prefix và scoring, phân tích khác biệt giữa các endpoint, và đánh đổi chất lượng–dung lượng vector. Giữ cách mô tả này như định hướng đóng góp thực nghiệm; không gọi việc đổi tên metric là bằng chứng novelty.

## 3. Dữ liệu, lựa chọn mẫu và chống nhầm cỡ mẫu

| Bộ đánh giá | Quy mô | Vai trò |
|---|---:|---|
| SugarCrepe xác nhận | 560 ảnh / 560 cặp positive–negative | Phép chính đã khóa; 80 ảnh cho mỗi nhóm trong 7 nhóm |
| SugarCrepe cohort rộng chưa dùng phát triển | 1.005 ảnh / 3.949 cặp | Đánh giá bổ sung, nhiều cặp có thể cùng ảnh |
| SugarCrepe đầy đủ | 1.560 ảnh / 7.511 cặp | Mô tả; có dữ liệu đã dùng phát triển |
| Flickr8k-Expert chưa dùng phát triển | 900 ảnh / 5.112 caption / 15.336 điểm cá nhân | Human evaluation phụ, cohort quan trọng để diễn giải |
| Flickr8k-Expert đầy đủ | 1.000 ảnh / 5.664 caption / 16.992 điểm cá nhân | Mô tả; gồm 100 ảnh cũ / 552 caption |

**560 ảnh là tập con của 1.005 ảnh SugarCrepe. Không cộng hai cohort này thành hai test độc lập.** Tương tự, không cộng Flickr8k full1.000 và unseen900. Full dataset và unseen cohort được báo cáo tách biệt.

Manifest SugarCrepe560 SHA-256: `a2e9fa5fb7e9866631c0546e5a09a2f9d11044335267712591476572b6104efe`.

Manifest Flickr8k đầy đủ SHA-256: `0e0a96c3659b565e7675bb41fbcde161a4e39617b7a9fd337174aac0018b9fa3`.

Flickr8k-HumanEval dùng revision `c30e1e67e2b37a42ecffaf9a88a05458ca05b973`. Hai shard nguồn, nhãn, caption, thứ tự record và hash toàn bộ 1.000 ảnh đã được kiểm tra. Candidate trùng caption tham chiếu bị loại theo quy trình nguồn. Điểm chuyên gia là nhãn gốc, không được tạo bởi LLM.

## 4. Encoder, preprocessing và công thức

| Encoder | Revision đã sử dụng |
|---|---|
| google/embeddinggemma-2 | `914f7f89142e33e77833254d9c9b90c3cef7303b` |
| CLIP ViT-B/32 | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| CLIP ViT-L/14 | `32bd64288804d66eefd0ccbe215aa642df71cc41` |
| google/siglip2-base-patch16-224 | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` |
| google/siglip2-base-patch16-256 | `3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab` |
| SPECS, trọng số trained spec.pt | `40311a17859757cad7046b945ff80fa46ef53919` |

Gemma nạp text và vision float32, không nạp audio, native image budget280. Dùng prefix chính `A photo depicts `; FactChecking là cấu hình thay thế đã có từ trước, không được chọn lại từ kết quả mở rộng. Mỗi model dùng preprocessing native của checkpoint. Các checkpoint khác nhau về kiến trúc, context và kích thước, vì vậy đây là so sánh hệ thống encoder, không phải thí nghiệm cố định mọi thành phần kiến trúc.

Với ảnh và caption được L2-normalize, Gemma/CLIP dùng:

`score(image, caption) = 2.5 × max(cosine(image, caption), 0)`.

SugarCrepe chính với SigLIP2-224 cũng giữ công thức trên theo protocol đã khóa. Cosine nguyên và clipping cho cùng quyết định trên 560 ảnh đúng, không có hòa điểm. Human evaluation dùng cosine nguyên cho SigLIP2-224/256, đồng thời giữ kết quả Sig224 clipped làm sensitivity check vì clipping có thể tạo hòa điểm ở tương quan toàn cục. SPECS dùng `(cosine + 1) / 2` với trọng số trained chính thức, không dùng LongCLIP base thay thế.

Không so raw mean score giữa các thang điểm khác nhau. Cosine không được diễn giải thành xác suất caption đúng. SugarCrepe tính đúng khi positive_score > negative_score; hòa tính sai.

## 5. Kết quả SugarCrepe

### 5.1. Phép xác nhận 560 ảnh

| Cấu hình | Đúng / 560 | Accuracy |
|---|---:|---:|
| Gemma literal768 float32 | 503 | 89,82% |
| SigLIP2-224 raw | 426 | 76,07% |
| SigLIP2-224 cùng prefix literal | 434 | 77,50% |
| CLIP B/32 | 429 | 76,61% |
| CLIP L/14 | 416 | 74,29% |
| SigLIP2-256 native, bổ sung | 421 | 75,18% |
| SPECS trained, bổ sung | 356 | 63,57% |

Gemma vs Sig224 raw có 97 ảnh Gemma đúng/Sig sai, 20 ảnh Gemma sai/Sig đúng, 406 ảnh cả hai đúng và 37 ảnh cả hai sai. Exact McNemar dùng 117 bất đồng. Paired bootstrap 10.000 lần theo ảnh, seed20261009, cho chênh lệch +13,75 điểm phần trăm, CI95% [10,18; 17,50].

| Gemma literal768 − baseline phụ | Δ điểm phần trăm | CI95% | McNemar p, Holm family3 |
|---|---:|---|---:|
| Sig224 matched-prefix | +12,32 | [8,57; 16,25] | 2,91e-10 |
| CLIP B/32 | +13,21 | [9,46; 16,96] | 3,97e-11 |
| CLIP L/14 | +15,54 | [11,61; 19,64] | 7,36e-13 |

CI trong bảng là CI riêng từng phép, không hiệu chỉnh đồng thời. Phép chính là family riêng theo protocol. Đối chứng matched-prefix cho thấy lợi thế quan sát không được giải thích hoàn toàn bởi việc chỉ thêm prefix cho Gemma.

| Loại lỗi, 80 ảnh/nhóm | Gemma literal | Sig224 raw | Sig224 literal |
|---|---:|---:|---:|
| add_att | 93,75% | 80,00% | 76,25% |
| add_obj | 98,75% | 87,50% | 86,25% |
| replace_att | 93,75% | 76,25% | 72,50% |
| replace_obj | 97,50% | 93,75% | 97,50% |
| replace_rel | 81,25% | 63,75% | 71,25% |
| swap_att | 91,25% | 71,25% | 70,00% |
| swap_obj | 72,50% | 60,00% | 68,75% |

Theo nhóm là phân tích mô tả, không phải 7 phép kiểm định xác nhận riêng. Swap_obj và replace_rel vẫn là giới hạn đáng chú ý của Gemma.

### 5.2. Cohort rộng hơn, 3.949 cặp / 1.005 ảnh

| Cấu hình | Accuracy |
|---|---:|
| Gemma literal768 f32 | 91,57% |
| Gemma literal256 f32 | 90,17% |
| CLIP B/32 | 77,77% |
| CLIP L/14 | 77,79% |
| SigLIP2-256 native | 82,07% |
| SPECS trained | 61,51% |

Gemma−CLIP L/14: +13,78 điểm phần trăm, CI95% [12,35; 15,25]. Gemma−Sig256: +9,50, CI95% [8,14; 10,87]. Bootstrap 10.000 lần theo image-cluster; sign permutation theo ảnh 19.999 lần. Full7.511 và từng nhóm/dimension có số chi tiết trong JSON/CSV nguồn.

Micro-average cohort rộng dùng phân bố nhóm gốc; mẫu560 cân bằng nhóm. Không diễn giải khác biệt giữa 91,57% và 89,82% là tăng hiệu quả nhờ tăng dữ liệu. Sig256 là checkpoint khác Sig224, không gộp hai baseline.

## 6. Flickr8k-Expert: tương quan và xếp hạng

Kendall tau-c dùng điểm cá nhân: lặp lại score caption cho từng rating của 3 chuyên gia. Resample ảnh với replacement, giữ toàn bộ caption/rating thuộc ảnh; dùng cùng draws cho mọi model. Bootstrap 2.000 lần, seed202610084. Không resample từng caption như mẫu độc lập.

| Cấu hình | Tau-c, 900 ảnh | CI95% | Tau-c full1.000, mô tả |
|---|---:|---|---:|
| Gemma literal768 f32 | 0,5449 | [0,5239; 0,5662] | 0,5468 |
| Gemma literal256 f32 | 0,5324 | [0,5112; 0,5535] | 0,5346 |
| Gemma literal256 int8 | 0,5325 | [0,5113; 0,5536] | 0,5346 |
| Gemma FactChecking768 f32 | 0,5475 | [0,5261; 0,5681] | 0,5493 |
| CLIP B/32 | 0,5112 | [0,4891; 0,5323] | 0,5148 |
| CLIP L/14 | 0,5282 | [0,5063; 0,5491] | 0,5296 |
| SPECS trained | 0,3039 | [0,2831; 0,3262] | 0,3037 |
| SigLIP2-256 native raw | 0,5271 | [0,5059; 0,5483] | 0,5291 |
| SigLIP2-224 native raw | 0,5243 | [0,5028; 0,5458] | 0,5263 |
| SigLIP2-224 native literal | 0,5317 | [0,5098; 0,5535] | 0,5336 |
| SigLIP2-224 raw clipped | 0,5194 | [0,4978; 0,5412] | 0,5215 |
| SigLIP2-224 literal clipped | 0,5278 | [0,5056; 0,5496] | 0,5298 |

Tau-c không phải accuracy hay phần trăm caption đúng. FactChecking hơi cao hơn trên endpoint này nhưng không được dùng để thay cấu hình chính sau khi xem kết quả.

| Gemma literal768 − baseline | Δtau-c | CI95% riêng từng phép | CI Bonferroni family6, nominal |
|---|---:|---|---|
| CLIP B/32 | +0,0338 | [0,0252; 0,0431] | [0,0225; 0,0461] |
| CLIP L/14 | +0,0167 | [0,0078; 0,0254] | [0,0044; 0,0293] |
| SPECS trained | +0,2410 | [0,2243; 0,2591] | [0,2164; 0,2663] |
| SigLIP2-256 raw | +0,0178 | [0,0089; 0,0266] | [0,0061; 0,0293] |
| SigLIP2-224 raw | +0,0206 | [0,0118; 0,0290] | [0,0094; 0,0319] |
| SigLIP2-224 literal | +0,0132 | [0,0046; 0,0212] | [0,0021; 0,0245] |

Human evaluation là phân tích phụ. Bonferroni dùng percentile ở mức nominal1−0,05/6; có tính xấp xỉ bootstrap, không phải bảo đảm familywise chính xác. Không tạo p-value từ tỷ lệ bootstrap thắng.

### Xếp hạng caption trong cùng ảnh

Chỉ ở endpoint này mới dùng average human rating để xác định caption tốt hơn. Loại cặp có average rating bằng nhau; hòa score model tính sai. Cohort900 có 9.459 cặp từ 872 ảnh. Bootstrap 5.000 lần theo ảnh, seed202610085.

| Cấu hình | Tỷ lệ xếp hạng đúng |
|---|---:|
| Gemma literal768 f32 | 80,95% |
| Gemma literal256 f32 | 80,02% |
| Gemma literal256 int8 | 80,03% |
| Gemma FactChecking768 f32 | 81,12% |
| CLIP B/32 | 79,19% |
| CLIP L/14 | 80,32% |
| SPECS trained | 65,49% |
| SigLIP2-256 raw | 79,90% |
| SigLIP2-224 raw | 79,60% |
| SigLIP2-224 literal | 80,77% |

Gemma literal768−Sig224 literal là +0,18 điểm phần trăm, CI95% [−0,91; 1,23]. **Chưa thấy lợi thế rõ ở xếp hạng trong cùng ảnh so với matched-prefix Sig224.** CI chứa0 không chứng minh tương đương. Tương quan toàn cục và within-image ranking cần được diễn giải riêng.

## 7. Giảm chiều và nén vector

Truncate xuống 128/256/512/768 chiều rồi L2-normalize. Float16 là giảm precision vector sau encoder. Int8 đối xứng theo từng vector: scale=maxabs/127 lưu float32; round/clip vào [−127;127]; reconstruct và normalize trước chấm. Encoder vẫn float32.

| Cấu hình Gemma literal | Accuracy560 | Byte/vector payload | Tau-c900 |
|---|---:|---:|---:|
| 768 f32 | 89,82% | 3.072 | 0,5449 |
| 512 f32 | 88,21% | 2.048 | 0,5428 |
| 256 f32 | 87,68% | 1.024 | 0,5324 |
| 256 f16 | 87,68% | 512 | 0,5324 |
| 256 int8 + scale f32 | 88,21% | 260 | 0,5325 |
| 128 f32 | 83,04% | 512 | 0,4680 |

768f32 → 256int8 giảm payload11,82 lần, không tính metadata, index hay overhead file. Trên560, delta −1,61 điểm phần trăm, CI95% [−3,22; 0]; chưa có bằng chứng tương đương. Int8 cao hơn256f32 trên mẫu này do đổi quyết định sau làm tròn, không phải cải thiện tổng quát.

Trên cohort SugarCrepe rộng,256f32 giảm1,39 điểm phần trăm so với768f32; one-sided95% lower−1,9545, vượt rất sát ngưỡng−2 điểm phần trăm đã nêu trong protocol nguồn. Kết quả này chỉ áp dụng cho256f32/cohort đó, không áp dụng cho int8 hoặc human evaluation.

Trên human900,256f32−768f32 có delta tau-c−0,0126, CI95% [−0,0157;−0,0096]; one-sided95% lower−0,0151. Không đạt tolerance0,01 thăm dò trong protocol nguồn.256int8−768f32 có delta−0,0125, CI95% [−0,0155;−0,0095]. Toàn bộ24 cấu hình Gemma gồm2 prefix ×4 dimensions ×3 formats nằm trong CSV/JSON.

## 8. Grounding, precision, truncation và kiểm chứng

Đối chứng ảnh sai ghép mỗi cặp caption với559 ảnh còn lại, tạo313.040 phép ghép/cấu hình. Tỷ lệ cosine ưu tiên positive gốc: Gemma literal49,53%, Sig224 raw48,87%, Sig224 literal51,55%. Đây là caption preference dưới ảnh không liên quan, không phải accuracy caption cho ảnh sai. Các ghép lặp lại không là313.040 mẫu độc lập. Sig224 raw clipped có66,39% ghép sai hòa0; không diễn giải clipping tạo hòa thành grounding tốt.

Audit Sig224560: raw tối đa27 token, literal30, context64, không có truncation. Audit human Sig224: raw35/literal38, context64, không có truncation. Gemma human literal39/FactChecking43; metadata được sửa thông tin context sentinel sang giới hạn text_config262144 sau run, không thay vector; file correction được lưu.

- Stage5 kiểm tra140 hash nguồn, bridge350 mẫu cũ khớp vector/decision;31.360 scalar scores sai số tối đa3,55e-15.
- Stage6 xác minh file Colab đầu vào và runner khớp từng byte;5.600 scalar scores sai số tối đa2,66e-15; p exact được đối chiếu bằng tổng tổ hợp nhị thức.
- Stage7: hoàn tất1000/5664 cho cả6 encoder;45.312 native float32 scalar scores sai số tối đa2,89e-15. Weighted tau-c khớp SciPy tại point estimates và bootstrap reference draws;64 cặp cohort-method tính lại từ CSV khớp JSON.
- Gemma human reuse940 ảnh đã kiểm tra, inference60 ảnh mới; bridge16 ảnh và32 caption/prefix đạt trước reuse. Sig224 human chạy mới toàn bộ1000 ảnh và972 chuỗi caption unique cho mỗi prefix, sau đó ánh xạ về5664 records.

Gói này giữ báo cáo xác minh gốc và cung cấp kiểm tra hash toàn bộ payload. SHA của model/ảnh là provenance đã kiểm tra khi tải hoặc ở runner; offline hash verifier chỉ kiểm tra byte đang được bàn giao, không tải lại model/ảnh để tái xác nhận.

## 9. Giới hạn và cách viết kết luận paper

Có thể kết luận: ở cấu hình đã khóa và hai benchmark này, Gemma cải thiện compositional discrimination và tương quan toàn cục với expert ratings; matched-prefix vẫn giữ gain trên endpoint chính; giảm chiều/quantize tạo đường đánh đổi chất lượng–dung lượng rõ ràng.

Không kết luận thay thế mọi metric caption, SOTA chung, phát hiện mọi hallucination, hiệu quả realtime, hay không mất chất lượng khi nén. SPECS hướng tới caption dài; các caption ngắn hiện tại không bao phủ đầy đủ phạm vi đó. Chưa có PAC-S trong baseline đã chạy. Chưa loại trừ contamination từ pretraining.

Không so tốc độ từ các phiên khác nhau. SugarCrepe Sig224 do người dùng chạy CUDA, vector Gemma/CLIP từCPU; human Gemma continuation/cache, Sig224 fresh, CPU6+2threads cùng lịch khác nhau. Timing chỉ phục vụ trace/debug. Chưa có benchmark latency/peak memory trên phần cứng đồng nhất.

Nghiên cứu độ bền của caption metric đã có prior work; novelty của cách framing cần được lập luận trong related work. Không cần thay đổi model để viết một paper thực nghiệm có câu hỏi rõ, nhưng chấp nhận phụ thuộc venue/reviewer. Theo quyết định hiện tại, dừng mở rộng thí nghiệm và dùng dữ liệu hiện có để viết bài với các giới hạn này.

## 10. Tệp bằng chứng và tái lập

Đọc `README_VI.md` trước. Gói FULL gồm source gốc, protocol/manifest, metadata, CSV/JSON, vector hoàn chỉnh và các báo cáo từng stage. Gói SOURCE chỉ gồm mã và tệp văn bản hỗ trợ, không có embedding arrays. Model weights và raw ảnh/parquet không nằm trong cả hai gói; downloader pinned và receipts hỗ trợ tải lại.

| Thành phần | Tệp chính |
|---|---|
| Phép chính560 và kiểm chứng Sig224 | `stage6/analyze_confirmation.py`, `stage6/verified_confirmation_results.json` |
| Subset560 / nén vector / đối chứng ảnh sai | `stage5/analyze_560.py`, `stage5/verify.py`, `stage5/results_560.json` |
| SugarCrepe rộng và vector5encoder | `full_recovered/embeddinggemma2_full_validation/full_validation/analyze.py`, `runs/` |
| Human final6encoder,32cấu hình | `stage7/analyze_human.py`, `stage7/human_results.json`, `stage7/human_scores.csv` |
| Human verifier độc lập | `stage7/verify_completion.py`, `stage7/independent_verification.json` |
| Human inference / download | `stage7/run_human.py`, `download_required.py`, `download_siglip224.py`, `materialize_expert.py` |
| Runner224 trongColab | `EmbeddingGemma2_560_SigLIP224_Colab.ipynb`, `stage6/uploaded/run_siglip224.py` |
| Pilot và encoder adapters gốc | `research/embeddinggemma2_clipscore_replacement/`, `stage4/`, `full_recovered/.../replacement/` |
| SPECS third-party snapshot | `full_recovered/.../full_validation/sources/SPECS/` gồmLICENSE vàtokenizer |

Chạy offline từ thư mục gốc gói FULL:

```sh
python -m pip install -r requirements-analysis.txt
python verify_bundle.py
OPENBLAS_NUM_THREADS=1 python stage6/analyze_confirmation.py
OPENBLAS_NUM_THREADS=1 python stage7/analyze_human.py
python stage7/verify_completion.py
```

Các script analysis ghi lại JSON/CSV. Giữ một bản gói nguyên trạng trước khi chạy; nếu đổi version thư viện, chữ số cuối hoặc serialization có thể thay đổi. Hash verifier dùng để kiểm tra bản phát hành nguyên trạng. Lệnh chạy cohort rộng/560 và inference có trongREADME.

## 11. Nguồn liên quan

- Model: https://huggingface.co/google/embeddinggemma-2
- Google hướng dẫn multimodal embeddings: https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers
- CLIPScore: https://arxiv.org/abs/2104.08718
- Robustness của reference-free caption metrics: https://arxiv.org/abs/2305.14998
- SPECS, metric hướng tới caption dài: https://arxiv.org/abs/2509.03897
- PAC-S, đối chứng liên quan chưa chạy: https://arxiv.org/abs/2303.12112
- Flickr8k-HumanEval: https://huggingface.co/datasets/yuwd/Flickr8k-HumanEval
- SugarCrepe: https://github.com/RAIVNLab/sugar-crepe

Các bảng số của báo cáo lấy từ kết quả thực nghiệm kèm theo, không lấy từ các paper trên. Bản tổng hợp này không tuyên bố đã rà soát exhaustive toàn bộ literature.
