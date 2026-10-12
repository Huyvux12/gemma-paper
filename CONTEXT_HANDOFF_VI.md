# Context bàn giao — EmbeddingGemma-2 làm metric đánh giá caption ảnh

Chốt context: 11/10/2026. Kết quả thực nghiệm hiện tại chốt ngày 09/10/2026; lần tổng hợp này không chạy thêm thí nghiệm.

## 1. Mục tiêu của người dùng

Người dùng muốn nghiên cứu để viết paper có đóng góp học thuật. Hướng ban đầu: thay encoder CLIP/SigLIP đang dùng trong metric ảnh–caption bằng EmbeddingGemma-2, kiểm chứng tự hành, thay vì chỉ nghiên cứu tính năng model. Sau baseline, người dùng yêu cầu thêm metric và một cách thích nghi model có học trọng số, đã cấp Kaggle API để chạy GPU. Người dùng muốn code, dữ liệu và kết quả minh bạch; ưu tiên thực thi, tránh dừng ở kế hoạch hoặc hỏi xác nhận lặp lại.

Phạm vi đã kiểm chứng: ảnh–caption, compositional discrimination và tương hợp với đánh giá con người. Chưa kiểm chứng audio/video, decision model, Gemma+Qwen, RAG, hệ thống lọc hallucination thời gian thực hoặc threshold xác suất.

## 2. Đọc gì trước

1. `START_HERE.md`: cấu trúc gói và cách mở dữ liệu.
2. File này: quyết định, trạng thái và giới hạn kết luận.
3. `EmbeddingGemma2_Stage8_Report_VI.md`: báo cáo cập nhật, gồm adapter, PAC-S, GPU và CI.
4. Giải nén `baseline.zip`, đọc `baseline/EmbeddingGemma2_Research_Report_VI.md`: báo cáo tổng hợp Stage1–7; đặc biệt đối chứng prefix, human ranking và nén vector.
5. `stage8/results/all_metrics.csv` / `.json`: bảng đầy đủ.
6. `stage8/protocol.json`, `stage8/results/adapter_training/selection.json`, `stage8/results/kaggle_final/stage8_results/`: protocol và bằng chứng thực thi.

Báo cáo cũ trong baseline giữ nguyên lịch sử. Dòng pending/chưa có human evaluation ở báo cáo trung gian không đại diện cho trạng thái cuối. Stage8 đã hoàn tất GPU và PAC-S.

## 3. Cấu hình Gemma gốc

- Model: `google/embeddinggemma-2`.
- Revision: `914f7f89142e33e77833254d9c9b90c3cef7303b`.
- Cấu hình chính: 768 chiều, encoder float32, prefix caption `A photo depicts `.
- Chuẩn hóa L2, score = `2.5 * max(cosine(image_embedding, caption_embedding), 0)`.
- Nạp text/vision, `audio_config=None`, native image budget 280.
- FactChecking là cấu hình thay thế đã có; không đổi cấu hình chính vì thấy test tốt hơn.
- Cosine/score không phải xác suất caption đúng.

## 4. Dữ liệu và chống nhầm cohort

| Bộ | Quy mô | Cách diễn giải |
|---|---:|---|
| SugarCrepe đầy đủ | 1.560 ảnh / 7.511 cặp | Gồm dữ liệu development đã xem |
| Development lịch sử | 555 ảnh / 3.562 cặp | Nguồn train/validation của adapter |
| SugarCrepe historical unseen | 1.005 ảnh / 3.949 cặp | Chưa dùng development ở các stage đầu; nay đã xem kết quả |
| SugarCrepe 560 | 560 ảnh / 560 cặp, 80 ảnh × 7 loại lỗi | Tập con của 1.005 ảnh, không phải test độc lập thứ hai |
| Flickr8kExpert full | 1.000 ảnh / 5.664 candidate / 16.992 rating | Gồm 100 ảnh development cũ |
| Flickr8kExpert historical unseen | 900 ảnh / 5.112 candidate / 15.336 rating | Nhãn/kết quả hiện đã được quan sát |

Không cộng 560 với 1.005 ảnh, hoặc full1.000 với unseen900. Historical unseen không đảm bảo không trùng pretraining của encoder.

Manifest SugarCrepe560 SHA256: `a2e9fa5fb7e9866631c0546e5a09a2f9d11044335267712591476572b6104efe`.
Manifest human đầy đủ SHA256: `0e0a96c3659b565e7675bb41fbcde161a4e39617b7a9fd337174aac0018b9fa3`.
Flickr8k-HumanEval source revision: `c30e1e67e2b37a42ecffaf9a88a05458ca05b973`.

## 5. Bằng chứng Stage1–7: Gemma gốc so với baseline

| SugarCrepe 560 | Accuracy |
|---|---:|
| Gemma literal768 f32 | 89,8214% = 503/560 |
| SigLIP2-224 raw | 76,0714% = 426/560 |
| SigLIP2-224 cùng prefix | 77,50% = 434/560 |
| CLIP B/32 | 76,61% |
| CLIP L/14 | 74,29% |

Gemma − Sig224 raw: +13,75 pp; paired image bootstrap CI95% [10,18; 17,50]; exact McNemar p≈2,56e−13. So với matched-prefix Sig224: +12,32 pp, CI95% [8,57; 16,25]. Đây là protocol baseline ban đầu; không chuyển mức xác nhận này sang claim cho adapter Stage8.

Trên human900, Gemma τ-c=0,544936; Sig224 raw≈0,524310, matched-prefix≈0,531725, CLIP B≈0,511155, CLIP L≈0,528237. Báo cáo Stage1–7 có CI ghép cặp và điều chỉnh family cho các so sánh tương quan. Riêng within-image ranking: Gemma80,95% vs Sig224 matched-prefix80,77%; Δ≈+0,18 pp, CI95% [−0,91; 1,23], chưa thấy ưu thế rõ ở endpoint này.

Nén vector 768f32 → 256int8 + một scale f32 giảm payload 3.072 → 260 byte/vector, tức 11,82×. Human τ-c giảm khoảng0,0125; không chứng minh tương đương chất lượng hoặc encoder nhanh hơn. Nén sau encoder và fine-tune/quantize encoder là các việc khác nhau.

## 6. Stage8: adapter đã học thật, encoder giữ cố định

Hai head riêng cho ảnh và text: `normalize(x + U(D(x)))`, residual rank thấp, U khởi tạo zero. Có cập nhật gradient lên trọng số adapter. Backbone Gemma frozen; chưa LoRA hoặc fine-tune backbone.

- Split development theo ảnh, seed20261009: 444 train /111 validation; 2.816/746 cặp.
- Tách ảnh và SHA256 byte ảnh giữa train/validation/historical Sugar test; hash ảnh train/validation không trùng expert.
- Training/chọn checkpoint chỉ forward embedding development.
- Grid: ranks16/32 × anchor0,1/1,0 × seeds11/23/37/53/71 =20 runs mỗi môi trường; 30 epochs.
- AdamW lr0,001, weight decay0,0001, batch128, clip gradient1.
- Loss: softplus((s_neg−s_pos+0,02)/0,05) + anchor·mean(1−cosine(embedding mới, embedding gốc)).
- Mỗi run chọn epoch có validation image-macro accuracy cao nhất, hòa chọn sớm nhất.
- Chọn rank/anchor bằng mean validation của 5 seed, hòa ưu tiên rank nhỏ rồi anchor mạnh.
- Chọn rank32/anchor1,0; validation mean0,951076. Mỗi adapter98.304 tham số.
- Báo cáo cả5 seed và ensemble trung bình score; không chọn seed theo test.
- CPU và GPU chọn cùng cấu hình/epoch; human score khác nhiều nhất4,29e−6. Không claim checkpoint byte-identical.

## 7. Trade-off của adapter — kết luận quan trọng nhất

| Endpoint | Gemma gốc | Ensemble adapter |
|---|---:|---:|
| SugarCrepe560 accuracy | 89,8214% | 91,2500% =511/560 |
| SugarCrepe3949 accuracy | 91,5675% | 92,6564% |
| Human900 Kendall τ-c | 0,544936 | 0,423398 |
| Human within-image pair accuracy | 80,9494% | 73,2847% |
| Human NDCG@5 | 0,891922 | 0,845250 |

Exploratory paired image-bootstrap:
- Δ accuracy560 +1,4286 pp, CI95% [−1,6071; 4,2857], 5.000 replicates.
- Δ accuracy3949 +1,0889 pp, CI95% [0; 2,1653], 5.000 replicates.
- Δ human τ-c −0,121538, CI95% [−0,136434; −0,105794], 1.000 replicates.

Có mức tăng điểm ước lượng trên bài phân biệt caption sai chi tiết, nhưng human alignment giảm mạnh. Chưa chứng minh cải tiến tổng thể. Độ chuyên biệt hóa theo loss SugarCrepe là giả thuyết giải thích; chưa có ablation nhân quả. Ví dụ lỗi màu/chủ thể chỉ là cách giải thích nhiệm vụ, không phải bằng chứng model đã xử lý đúng từng ví dụ tự đặt.

## 8. Metric mở rộng và PAC-S

`all_metrics.csv`: 47 tên cấu hình ×2 cohort, mỗi hàng có11 thống kê và số lượng mẫu. Không phải47 backbone hoặc47 metric độc lập. Có một score Gemma gốc tái tính trùng baseline768 để kiểm tra.

11 thống kê: Kendall τ-c/τ-b, Spearman và Pearson trên rating cá nhân, Spearman trên mean rating, pair accuracy micro/macro trong ảnh, tie-aware NDCG@5, tie-aware top1 human-best, AUROC và average precision với positive mean rating≥3.

Kendall cá nhân lặp score theo ba rating gốc. Pair loại cặp hòa mean rating, model hòa tính sai. NDCG gain=2^(mean rating−1)−1, lấy kỳ vọng khi score hòa. AP là average precision; score chưa calibration thành xác suất. Cần giữ số lượng nhóm có thể đánh giá đi kèm.

Đã chạy BLEU1–4, ROUGE-L, CIDEr, METEOR qua pycocoevalcap1.2/COCO PTBTokenizer; nhận5 reference gốc của tác giả. CIDEr dùng corpus candidate-record, reference lặp theo các candidate cùng ảnh. SPICE chưa chạy.

PAC-S chính thức đã chạy thành công: repo `aimagelab/pacscore`, commit `bfbc488e9f5cccd818e75c7cdfaa37e1c87f00eb`, trained checkpoint ViT-B/32, w=2,0; không thay bằng PAC-S++. Human900 τ-c=0,539870, pair accuracy81,1819%. Δ τ-c(PAC−Gemma)=−0,005066, CI95% [−0,012834; 0,003265]; chưa chứng minh bên nào có ưu thế human alignment. SHA checkpoint là receipt sau download; overlap fine-tuning/pretraining chưa được audit độc lập.

## 9. GPU và Kaggle — đã hoàn tất, không phải đang chờ

Job riêng tư: https://www.kaggle.com/code/dungdq210/embeddinggemma-stage8-20261009
Dataset: `dungdq210/embeddinggemma-stage8-20261009-cc802fcc`.
Account: `dungdq210`. Version2 COMPLETE đã xác minh ngày09/10. Version1 lỗi vì Kaggle tự giải nén ZIP; version2 xử lý cả ZIP và cây thư mục. Không cần chạy lại chỉ để lấy các kết quả đã có.

GPU Tesla T4, Torch2.11.0+cu128, float32, TF32 tắt; GPU0. Cùng8 input expert thật; native image budget/preprocessing khác nhau; warmup3, đo15 lần. Bao gồm decode, preprocessing, encoder ảnh+text, scoring/head; loại thời gian download/load. P95, samples và peak allocated/reserved đều có trong JSON.

| Model | Batch | P50 ms/batch | Cặp/s | Peak allocated MiB |
|---|---:|---:|---:|---:|
| Gemma gốc | 1 | 482,861 | 2,07 | 1842,3 |
| Adapter ensemble | 1 | 491,520 | 2,03 | 1844,2 |
| SigLIP2-224 | 1 | 45,645 | 21,91 | 1471,7 |
| CLIP B/32 | 1 | 74,806 | 13,37 | 613,1 |
| Gemma gốc | 8 | 3590,253 | 2,23 | 2779,3 |
| Adapter ensemble | 8 | 3569,635 | 2,24 | 2781,2 |
| SigLIP2-224 | 8 | 138,259 | 57,86 | 1524,0 |
| CLIP B/32 | 8 | 75,521 | 105,93 | 636,9 |

Gemma chậm hơn Sig224 khoảng10,6× ở batch1 trong cấu hình đã đo. Chi phí tăng adapter batch1 khoảng1,8% là điểm ước lượng, không phải kiểm định overhead. Batch8 adapter thấp hơn nhẹ do biến thiên phép đo; không claim adapter làm encoder nhanh hơn. Không suy rộng latency sang mọi GPU/precision/serving stack.

Đã tải94 file nghiên cứu,92 file khớp SHA256 manifest remote; runtime và manifest tải riêng. Quota30hGPU/20hTPU trong state là snapshot trước job09/10, không phải số dư hiện tại.

## 10. Mã và chứng cứ cần giữ

- `stage8/train_adapter.py`, `evaluate_adapter.py`: mô hình/loss/train/chọn/evaluate.
- `stage8/metrics.py`, `reference_metrics.py`, `pacscore_gpu.py`, `combine_tables.py`: các evaluator.
- `stage8/uncertainty.py`, `compare_pac.py`: CI image-bootstrap.
- `stage8/benchmark_gpu.py`: đo encoder mới trên GPU; không dùng timing cache.
- `stage8/verify.py`: identity, gradient, checkpoint roundtrip, tie handling, tách ảnh/hash, input ZIP/extracted.
- `stage8/prepare_kaggle.py`, `kaggle_control.py`, `download_results.py`, `pipeline.py`: notebook/API/quy trình.
- `stage8/kaggle_kernel/run_stage8.ipynb`: notebook tự chứa source snapshot.
- `stage8/results/submitted_source_v1.zip` và `v2.zip`: source thực sự gửi mỗi lần.
- `stage8/results/adapter_training/`: split, selection,20 checkpoints/history.
- `stage8/results/adapter_evaluation/`: prediction/scoring theo record ID.
- `stage8/results/all_metrics.json` /`.csv`: kết quả tổng hợp.
- `stage8/results/adapter_uncertainty.json`, `pac_comparison.json`, `cpu_gpu_comparison.json`.
- `stage8/results/kaggle_final/`: output đã tải, log, download_verification, runtime.
- `stage8/results/kaggle_final/stage8_results/`: train GPU, metric, latency, PAC-S, pipeline_status, output_sha256.
- `stage8/results/adapter_tradeoff.png` /`.pdf`: hình khoa học có CI.
- `baseline.zip`: toàn bộ gói Stage1–7 gồm source, manifest, cache embedding, score, report và receipts; cần giải nén để tái lập CPU.
- `STAGE8_SHA256.json`: manifest gói Stage8 được giữ nguyên.

32 endpoint τ-c baseline cũ được tái lập sai số<1e−12. Đây không phải claim mọi kết quả CPU/GPU bitwise giống nhau.

## 11. Kết luận về paper và việc còn thiếu

Đã có bằng chứng cho paper thực nghiệm về thay encoder, prefix/endpoint, nén vector và đánh đổi adaptation. Chưa có bằng chứng cho một phương pháp mới vượt trội tổng thể hoặc bảo đảm đủ novelty/được venue chấp nhận. Nhận định trước đây “đủ để viết paper” chỉ nói có số liệu để viết, không đồng nghĩa contribution đã đủ mạnh so literature.

Các bước chưa làm:
1. Đối chiếu literature/novelty và định vị contribution đầy đủ.
2. Thiết kế adaptation giữ human alignment hoặc đa nhiệm; đây mới là hướng đề xuất, chưa triển khai/chưa có kết quả.
3. Chọn validation thích hợp cho mục tiêu mới, khóa phương pháp rồi đánh giá test độc lập mới, audit overlap. Không gọi các historical test đang dùng là holdout mới.
4. Ablation giải thích mất human alignment, nếu theo hướng phương pháp adaptation.
5. Viết bản thảo theo venue cụ thể; chưa có bản manuscript hoàn chỉnh.

Không chọn seed/hyperparameter/bộ test theo score tốt nhất. Không coi CI chứa0 là chứng minh tương đương. Giữ kết quả âm và lịch sử protocol.

## 12. Trạng thái chuyển phiên và thông tin xác thực

Scratch cũ có thể đã được bảo trì/xóa; dùng gói bàn giao làm nguồn có thể khôi phục. Không giả định môi trường Python, GPU hay file token của phiên trước còn tồn tại.

Không chuyển API token vào context, notebook, report hoặc archive. Các file state chỉ là metadata, không đủ để đăng nhập. Phiên mới phân tích kết quả hoặc tái lập CPU không cần Kaggle credential. Nếu thực sự cần gọi API/tạo job mới, người dùng phải cấu hình credential riêng hoặc cung cấp lại bằng cách phù hợp; kiểm tra auth/quota trước khi chạy. Không in token ra log.

## 13. Prompt mở phiên mới

> Tôi tiếp tục nghiên cứu EmbeddingGemma-2 làm metric đánh giá caption ảnh. Hãy đọc CONTEXT_HANDOFF_VI.md, START_HERE.md và báo cáo Stage8 trước; sau đó mở gói source/results, đối chiếu các con số với JSON/CSV. Hiện baseline và Stage8 GPU/PAC-S đã hoàn tất: adapter tăng SugarCrepe nhưng giảm human alignment, chưa chứng minh cải tiến tổng thể; historical test đã được xem. Hãy xác định hướng tiếp theo để có contribution học thuật thực chất, rà soát literature và thiết kế kiểm chứng công bằng. Ưu tiên tận dụng bằng chứng đã có, không chạy lại vô ích; không claim novelty hoặc SOTA khi chưa đủ chứng cứ. Nếu cần thí nghiệm mới, giữ kết quả âm, chọn trên validation và khóa test độc lập mới. Token không có trong gói.
