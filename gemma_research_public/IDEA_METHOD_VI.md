# Idea và method: Gemma + bounded shared residual adapter

## Câu hỏi nghiên cứu

EmbeddingGemma 2 có thể làm metric **reference-free image–caption compatibility** tốt hơn bằng cách chỉ học một head nhỏ trên embeddings đã frozen không? Cần cân bằng hai mục tiêu: nhận ra caption sai quan hệ/thành phần trên SugarCrepe, đồng thời phù hợp đánh giá con người trên Flickr8k-HumanEval và Polaris.

Đây là triển khai/thử nghiệm hiện có, chưa có literature review đủ để tuyên bố phương pháp mới hoàn toàn. Không fine-tune toàn backbone và không chèn LoRA vào Transformer. Reference captions chỉ là metadata/filter, không đi vào score.

## Backbone và score

- Model: `google/embeddinggemma-2`, revision `914f7f89142e33e77833254d9c9b90c3cef7303b`.
- Weight SHA256: `197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`.
- Embedding image/text768 chiều, L2-normalized, cache FP32.
- Text nhận prefix chính xác `A photo depicts `, có khoảng trắng cuối.
- Native inference FP32, thử nghiệm gốc Tesla T4; không dùng Gemma FP16.

Với `u = normalize(E_image(I))`, `v = normalize(E_text(prefix + c))`, score base là:

```math
s(I,c) = 2.5 max(0, uᵀv).
```

Score là scaled similarity, không phải xác suất caption đúng đã calibration.

## Adapter chung hai modality

```math
r(x) = W_up tanh(W_down normalize(x))
b(x) = r(x) / max(1, ||r(x)||₂)
A(x) = normalize(normalize(x) + ρ b(x)).
```

Hai linear không bias; `W_up` khởi tạo0 nên ban đầu trùng base. Cùng một A áp dụng cho image và text; score adapted thay u/v bằng A(u)/A(v). Radius chặn correction **trước** normalization cuối, không phải một cam kết exact về cosine/Euclidean displacement cuối cùng.

Primary được chọn: rank32, radius0,03, preservation0,1. Có `2 ×768 ×32 =49.152` tham số, khoảng192KiB trọng số FP32. Đây là head rất nhỏ; chi phí encode backbone vẫn chiếm ưu thế.

Source thực: `vendor/stage11/learning.py`, class `BoundedAdapter`, các hàm `pair_table`, `fit`, `select_config`. Không có công thức thay thế ẩn trong notebook.

## Ranking loss và preservation

Sugar cung cấp image/positive-caption/negative-caption. Với human corpora, tạo cặp caption cùng ảnh có mean rating khác nhau; caption có mean cao hơn là positive. Giữ mọi individual vote để đánh giá raw-vote correlation.

```math
L_rank = sum_i w_i softplus((m - cos(Au,Av⁺) + cos(Au,Av⁻))/T) / sum_i w_i
L_pres = mean(||Au-u||² + ||Av⁺-v⁺||² + ||Av⁻-v⁻||²)
         + mean((cos(Au,Av⁺)-cos(u,v⁺))² + (cos(Au,Av⁻)-cos(u,v⁻))²)
L = mean_over_domains(L_rank + λ L_pres).
```

`T=0,05`; margin Sugar0,02, human0. Học trên cosine chưa clamp/scale. Ba domain có budget loss bằng nhau; số step/epoch từ Sugar pair budget, batch128. AdamW lr0,001, weight decay0,0001, backbone frozen và head FP32.

Mean phase: `w=1`. Uncertainty phase cho human pairs:

```math
w = |μ⁺-μ⁻| / (|μ⁺-μ⁻| + SE⁺ + SE⁻ + 10⁻⁶),  SE = sample_std/sqrt(n).
```

Sugar vẫn weight1. Chuẩn hóa weights trong mỗi sampled batch; không dùng uncertainty như feature đầu vào inference. Singleton được gán SE0 là heuristic, không có nghĩa annotator chắc chắn đúng. Mean/uncertainty là hai head khác nhau sau fitting, inference không cần ratings.

## Dữ liệu và chống leakage

Records22642; test18537 candidate/pairs. Test Sugar4703 pairs; Flickr5112 captions/900 ảnh/15336 votes; Polaris8722 captions/2737 ảnh/26098 votes. Cần5817 ảnh và20065 text strings để encode toàn bộ train/validation/test.

Split theo decoded RGB pixel-hash disjoint **toàn cục giữa ba corpus**. Giữ Polaris official TEST, dựng lại Flickr100 development bằng seed nguồn, dựng Sugar split mới; loại candidate trùng chính xác một reference trước scoring. Không phát hiện mọi crop/resize/reencode hoặc pretraining overlap. Đây không phải historical Sugar3949/560.

Dataset revisions, split policy và toàn bộ hyperparameters ở `vendor/stage11/protocol.json`; split counts/audit ở `assets/core/split_audit.json`. Không dùng test records trong `pair_table`/fitting.

## Chọn config, epoch và primary

Grid5 configs ×3 seeds11/23/37 cho mỗi phase, tối đa15 epochs. Mỗi seed chọn epoch maximize weakest normalized validation gain. Vector validation dùng Sugar accuracy, Flickr **caption-mean tau-c**, Polaris **caption-mean tau-c**; chia gain cho[0,01;0,02;0,02].

Config eligible khi mean validation gain qua ba seed không âm ở cả ba miền. Maximize min gain, tie-break rank/radius nhỏ hơn; không config đạt thì fallback base. Sau chọn config, lấy median best epoch và refit train+validation. Mean seed11 là primary đã khóa; ensemble3 chỉ diagnostic. Không đổi primary bằng điểm test.

Gemma mean: rank32/radius0,03/preservation0,1,14 epochs. Secondary uncertainty cùng config nhưng11 epochs. Vì số epoch khác nhau, so hai nhánh gốc chưa cô lập hoàn toàn loss weighting.

Control sau test: giữ config mean và14 epochs cho cả hai loss, seeds11/23/37/53/71. Cùng initialization, supervision order và sampled batch IDs trong từng cặp; cache giữ nguyên. CPU Torch2.14.1+cpu, chỉ so hai CPU controls với nhau. Bằng chứng hiện tại không ủng hộ thay mean bằng uncertainty.

## Đánh giá, CI và timing

Sugar strict positive>negative; tie tính fail. Human: global raw-vote Kendall tau-c, caption-mean tau-c/Spearman và within-image ranking là các mục tiêu khác nhau. CI gốc2000 paired image-cluster resamples, giữ captions/votes trong ảnh; control có CI ghép seed×image. Chưa hiệu chỉnh multiplicity và chưa fresh holdout.

Timing128 ảnh chọn label-blind:43 Sugar/43 Flickr/42 Polaris, hai rounds, batch1/8, warmup5, rotate variant order, CUDA synchronize/TF32 off. Gồm preprocess/encode/normalize/score; PIL đã decode, loại disk IO/model download. Giữ timing fixed8 và timing128 riêng. p95 chỉ mô tả sample này; cached-head timing không thay cho end-to-end native latency.

GPU notebook chạy lại đầy đủ Gemma gồm validation search, test, CI và bốn ablations fixed rank32/radius0,03/preservation0,1/15epochs: sugar_only, human_only, no_preservation, full_control. CPU notebook chỉ replay và paired fixed-config refit; không gán nó là tái tạo exact CUDA selection.
