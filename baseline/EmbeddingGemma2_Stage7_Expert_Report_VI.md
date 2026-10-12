# EmbeddingGemma 2 thay encoder CLIPScore — Stage7: Expert evaluation

Ngày08/10/2026. **Đã hoàn tất inference phần còn thiếu ngay trong môi trường này**, không cần người dùng chạy Colab cho vòng này. Kết quả human evaluation là phân tích phụ đã nêu từ protocol nguồn; được tách khỏi giả thuyết SugarCrepe224 chính đã đạt ở stage6.

## Kết luận

Trên cohort900 ảnh mới, Gemma literal768 có tương quan chuyên gia cao hơn SigLIP2-224 native raw theo CI95% ghép cặp. CI nominal Bonferroni family6 cũng nằm trên0 cho phép đối chiếu này.

Gemma literal768 tau-c **0.5449**, SigLIP2-224 raw **0.5243**; Δtau-c **+0.0206**, CI95% **[0.0118; 0.0290]**.

Tau-c đo mức model xếp hạng caption giống các đánh giá chuyên gia; càng cao càng tốt. Đây không phải phần trăm caption đúng. Kết quả này không phủ định/biến đổi accuracy compositional89,82% của stage6; hai endpoint đo hai khía cạnh khác nhau.

## Dataset và cohort

- Flickr8k-Expert đầy đủ: **1.000 ảnh, 5.664 cặp ảnh-caption, 16.992 điểm cá nhân**. Mỗi caption có3 đánh giá.
- Cohort chưa dùng ở các lượt trước: **900 ảnh, 5.112 caption, 15.336 điểm cá nhân**; đây là cohort chính để đọc kết quả tiếp nối.
- 100 ảnh/552 caption cũ chỉ góp vào bảng full mô tả. Không cộng full và900 thành cỡ mẫu khác.
- Hai shard pinned revision `c30e1e67e2b37a42ecffaf9a88a05458ca05b973` của `yuwd/Flickr8k-HumanEval` đã khớp SHA-256 nguồn. Tái dựng toàn bộ caption, thứ tự và rating khớp từng record manifest; hash ảnh của toàn bộ1.000 ảnh cũng khớp.
- Manifest SHA-256 `0e0a96c3659b565e7675bb41fbcde161a4e39617b7a9fd337174aac0018b9fa3`. “Ảnh mới” nghĩa là chưa dùng trong chuỗi thí nghiệm, không đảm bảo ngoài dữ liệu pretraining.

## Cấu hình và phần đã chạy

Gemma giữ prefix CLIP literal `A photo depicts ` và FactChecking đã có từ trước, không chọn prefix theo điểm expert mở rộng. Text+vision float32, audio encoder không nạp; checkpoint `914f7f89142e33e77833254d9c9b90c3cef7303b`, native image budget280. Model và tokenizer lớn đều khớp receipt SHA-256.

- Gemma tiếp tục từ936 vector ảnh archive và prior100 đã xác minh, tổng seed **940** ảnh; mới mã hóa **60** ảnh trong lượt hoàn tất. Hai bộ caption được hoàn tất; reuse string-identical prior captions theo metadata, phần còn lại được inference mới.
- Bridge16 ảnh deterministic +32 caption/prefix khớp cache trước reuse; minimum cosine gần1, max error ảnh dưới6×10^-7 và caption dưới1,3×10^-7. Không relabel cache chưa hoàn chỉnh.
- SigLIP2-224 được inference mới đầy đủ1.000 ảnh và cả raw/literal caption, revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`, weight SHA đã khớp. Token audit không có over-context.
- CLIP B/32, L/14, SPECS trained `spec.pt`, SigLIP2-256 được giữ từ checkpoint full-human float32 hoàn chỉnh, cùng manifest/caption/ảnh. Không dùng SPECS LongCLIP base làm đại diện.
- Không fine-tune, không hiệu chỉnh score bằng human labels.

## Tương quan với điểm chuyên gia

| Cấu hình | Tau-c, 900 ảnh mới | CI95%, 900 ảnh mới | Tau-c, full1.000 |
|---|---:|---|---:|
| Gemma literal768 f32 | 0.5449 | [0.5239; 0.5662] | 0.5468 |
| Gemma literal256 f32 | 0.5324 | [0.5112; 0.5535] | 0.5346 |
| Gemma literal256 int8 | 0.5325 | [0.5113; 0.5536] | 0.5346 |
| Gemma FactChecking768 f32 | 0.5475 | [0.5261; 0.5681] | 0.5493 |
| CLIP B/32 | 0.5112 | [0.4891; 0.5323] | 0.5148 |
| CLIP L/14 | 0.5282 | [0.5063; 0.5491] | 0.5296 |
| SPECS trained spec.pt | 0.3039 | [0.2831; 0.3262] | 0.3037 |
| SigLIP2-256 native raw | 0.5271 | [0.5059; 0.5483] | 0.5291 |
| SigLIP2-224 native raw | 0.5243 | [0.5028; 0.5458] | 0.5263 |
| SigLIP2-224 native literal | 0.5317 | [0.5098; 0.5535] | 0.5336 |
| SigLIP2-224 raw, CLIPScore clipping | 0.5194 | [0.4978; 0.5412] | 0.5215 |
| SigLIP2-224 literal, CLIPScore clipping | 0.5278 | [0.5056; 0.5496] | 0.5298 |

Gemma/CLIP dùng `2.5×max(cosine,0)`. SPECS dùng affine cosine native `(cosine+1)/2`; SigLIP2-256 và224 native dùng cosine nguyên. Đã khóa addendum scoring224 trước khi có Gemma caption score mở rộng; đồng thời báo cáo cả224 CLIPScore-clipped sensitivity, tránh che ảnh hưởng tie ở tương quan giữa nhiều ảnh. Cosine có cùng thứ tự với logit native qua một scale dương và bias cố định; không dùng sigmoid probability làm xác suất caption đúng.

Kendall tau-c dùng **điểm cá nhân**, lặp lại score model cho từng chuyên gia, cùng quy trình nguồn. Spearman được giữ trong JSON bổ sung. Không dùng average rating thay endpoint tau-c chính.

## So sánh ghép cặp theo ảnh

| Gemma literal768 − baseline | Δtau-c | CI95% riêng từng phép | CI Bonferroni family6 (nominal) |
|---|---:|---|---|
| CLIP B/32 | +0.0338 | [0.0252; 0.0431] | [0.0225; 0.0461] |
| CLIP L/14 | +0.0167 | [0.0078; 0.0254] | [0.0044; 0.0293] |
| SPECS trained spec.pt | +0.2410 | [0.2243; 0.2591] | [0.2164; 0.2663] |
| SigLIP2-256 native raw | +0.0178 | [0.0089; 0.0266] | [0.0061; 0.0293] |
| SigLIP2-224 native raw | +0.0206 | [0.0118; 0.0290] | [0.0094; 0.0319] |
| SigLIP2-224 native literal | +0.0132 | [0.0046; 0.0212] | [0.0021; 0.0245] |

Image-cluster paired bootstrap2.000 lần: sample ảnh với replacement, giữ toàn bộ caption và điểm chuyên gia thuộc ảnh đó; cùng draws cho mọi model. CI riêng95% là các kết quả human phụ. Cột Bonferroni dùng percentile ở mức nominal1−0,05/6 cho sáu contrast literal768 với baseline, có tính xấp xỉ bootstrap; không báo p-value giả từ tỷ lệ bootstrap hoặc chọn phép thuận lợi nhất.

## Xếp hạng caption trong cùng ảnh

| Cấu hình | Xếp hạng đúng trong ảnh, 900 ảnh mới |
|---|---:|
| Gemma literal768 f32 | 80.95% |
| Gemma literal256 f32 | 80.02% |
| Gemma literal256 int8 | 80.03% |
| Gemma FactChecking768 f32 | 81.12% |
| CLIP B/32 | 79.19% |
| CLIP L/14 | 80.32% |
| SPECS trained spec.pt | 65.49% |
| SigLIP2-256 native raw | 79.90% |
| SigLIP2-224 native raw | 79.60% |
| SigLIP2-224 native literal | 80.77% |

Trên cohort900 có **9459** cặp caption với average rating khác nhau, từ **872** ảnh. Với endpoint phụ này mới dùng average rating để chọn caption tốt hơn; score hòa tính sai. CI contrast được bootstrap theo ảnh5.000 lần và nằm trong JSON. Các cặp cùng ảnh không là mẫu độc lập. Gemma literal768−SigLIP2-224 matched-prefix trong ảnh: Δ **+0.18 điểm %**, CI95% **[-0.91; 1.23]**. Cần đọc endpoint này riêng với tau-c toàn cục; không quy global correlation thành thắng rõ mọi kiểu ranking.

## Curve lưu vector trên human evaluation

### CLIPLiteral

| Chiều | Tau-c f32 | Tau-c f16 | Tau-c int8 (+scale f32) |
|---|---:|---:|---:|
| 128 | 0.4680 | 0.4680 | 0.4678 |
| 256 | 0.5324 | 0.5324 | 0.5325 |
| 512 | 0.5428 | 0.5428 | 0.5429 |
| 768 | 0.5449 | 0.5449 | 0.5450 |

### FactChecking

| Chiều | Tau-c f32 | Tau-c f16 | Tau-c int8 (+scale f32) |
|---|---:|---:|---:|
| 128 | 0.4704 | 0.4704 | 0.4705 |
| 256 | 0.5326 | 0.5326 | 0.5326 |
| 512 | 0.5452 | 0.5451 | 0.5451 |
| 768 | 0.5475 | 0.5475 | 0.5474 |


Gemma literal256f32−768f32: Δtau-c **-0.0126**, CI95% **[-0.0157; -0.0096]**; one-sided95% lower **-0.0151**. Protocol nguồn chỉ nêu tau tolerance0,01 theo hướng thăm dò; trạng thái theo ngưỡng nguồn: **False**. Không gọi đây là bằng chứng tương đương metric chung.

256int8−768f32: Δtau-c **-0.0125**, CI95% **[-0.0155; -0.0095]**. Int8 theo từng vector sau truncate và normalize, scale maxabs/127 float32, reconstruct rồi normalize. Payload260 byte/vector so với768f32 3.072 byte, giảm11,82 lần; không tính metadata/index. Curve là nén vector, không lượng tử hóa hoặc tăng tốc encoder.

## Kiểm chứng độc lập

Đã kiểm tra **45312** điểm native float32 bằng scalar fsum độc lập; sai số tối đa **2.89e-15**. Thuật toán tau-c theo histogram có trọng số dùng cho bootstrap khớp SciPy tại mọi point estimate và4 bootstrap draws/cohort trên mọi cấu hình được kiểm tra. Sau khi ghi CSV, tính lại tau-c tất cả **64** cặp cohort-method bằng SciPy đều khớp JSON. Shape/dtype/norm/alignment của cả6 encoder đã đạt.

Không so tốc độ Gemma với SigLIP224 từ run này: Gemma continuation/cache, Sig224 chạy đầy đủ; hai process CPU6+2 threads cùng chạy và warmup khác nhau. Metadata giữ timing phục vụ debug, không phục vụ claim latency.

## Trạng thái hướng nghiên cứu

Stage6 đã xác nhận gain compositional Gemma literal768 so với SigLIP2-224 raw: +13,75 điểm %, CI95%[10,18;17,50]. Stage7 bổ sung endpoint human thực tế; phải đọc theo CI ở trên, không suy từ SugarCrepe rằng tương quan human chắc chắn tốt hơn.

Phần kiểm chứng cơ bản trên SugarCrepe và Flickr8k-Expert hiện đã hoàn tất. Để phát triển thành paper, còn cần đóng góp được phân biệt với thay backbone và đánh giá ngoài domain này (caption sinh thực tế/hallucination hoặc bộ test độc lập khác). Nếu thêm cơ chế hoặc chọn threshold từ các dữ liệu đã xem, phải tách validation và test mới. Chưa tuyên bố novelty hay đủ điều kiện nhận của venue chỉ từ các bảng hiện tại.

## Tái lập

`embeddinggemma2_stage7_expert_validation.zip` có toàn bộ vector expert6 encoder, source manifest/protocol, code scoring/inference/download, CSV/JSON kết quả, receipt và logs. Không kèm weights1,5GB mỗi model hoặc raw image/shard; downloader lấy đúng pinned revision và kiểm tra hash nếu muốn chạy inference lại.

```sh
python -m pip install numpy scipy
OPENBLAS_NUM_THREADS=1 python stage7/analyze_human.py
python stage7/verify_completion.py
```

Hai lệnh trên chấm/kiểm tra offline từ vector hoàn chỉnh, không tải model. Mã infer có bridge và checkpoint resume. Nhãn expert dùng nguồn https://huggingface.co/datasets/yuwd/Flickr8k-HumanEval ; cấu hình Gemma theo https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers .
