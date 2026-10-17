# Gemma: bàn giao nghiên cứu và kế hoạch nâng chất lượng paper

**Cập nhật17/10/2026. Đọc file này trước khi tiếp tục.** Đây là bản tổng hợp hiện hành, thay cho trạng thái pending/RUNNING trong các handoff lịch sử. Người dùng muốn phát triển thành paper; chưa có cơ sở khẳng định đủ sức cạnh tranh venue A hoặc SOTA.

## 1. Mục tiêu và trạng thái thật

Nghiên cứu metric reference-free cho image–caption compatibility: backbone EmbeddingGemma2 frozen, chỉ học adapter nhỏ để cải thiện compositional discrimination và tương quan với đánh giá con người. Cần báo cả chất lượng lẫn latency/throughput/VRAM, giữ kết quả âm và kiểm khả năng tổng quát hóa.

Đã hoàn tất Stage11, phần mở rộng Stage12 bảy backbone, control CPU năm seed, và một lượt notebook Gemma chạy mới từ ảnh gốc đến báo cáo/ZIP trên Kaggle T4. Lượt mới tái lập cache/predictions/metrics của snapshot gốc chính xác; **đó là bằng chứng tái lập, không phải gain nghiên cứu mới trên holdout mới**.

Chưa viết xong manuscript để nộp. Chưa literature review đủ để tuyên bố novelty; còn thiếu baseline chuyên dụng và đánh giá ngoài các corpus đã quan sát. Không phục hồi được exact source/checkpoint Stage9/10; không gọi triển khai hiện tại là exact recovery của các stage đó.

## 2. Đọc code và kết quả ở đâu

Repo: https://github.com/Huyvux12/gemma-paper . Workspace Windows hiện tại: `E:\Dowload\project\AI project\100PRR\gemma0multi\gemma-paper`.

Gói đầy đủ local: `gemma_research_standalone/`. Bản GitHub gọn: `gemma_research_public/`, gồm source, notebook, tài liệu, số liệu tổng hợp và biểu đồ; không chứa ảnh/cache/checkpoint/predictions theo record hoặc cấu hình fingerprint dữ liệu riêng.

| Nội dung | Gói local đầy đủ |
|---|---|
| Idea và công thức | `gemma_research_standalone/IDEA_METHOD_VI.md` |
| Kết quả và trade-off | `gemma_research_standalone/RESULTS_VI.md` |
| Báo cáo GPU mới đầy đủ | `gemma_research_standalone/reports/latest_gpu_run_formatted/REPORT_VI.html` |
| Tóm tắt/audit rerun | `gemma_research_standalone/reports/gpu_rerun_audit/BAO_CAO_KET_THUC_VI.md` |
| Notebook chạy GPU | `gemma_research_standalone/notebooks/02_Gemma_Kaggle_Full_Rerun.ipynb` |
| Notebook CPU replay/refit | `gemma_research_standalone/notebooks/01_Gemma_CPU_Replay_Refit.ipynb` |
| Entrypoint GPU và report | `gemma_research_standalone/src/run_native.py`, `src/report.py` |
| Công thức/selection frozen | `gemma_research_standalone/vendor/stage11/learning.py`, `run.py`, `protocol.json` |
| Kết quả bảy backbone | `stage12/FINDINGS_VI.md`, `stage12/results/` |
| Phiên/audit đã hoàn tất | `gemma_research_standalone/verification/` |

Trong bản public, đọc `gemma_research_public/README_VI.md`, `IDEA_METHOD_VI.md`, `RESULTS_VI.md`, `reports/REPORT_VI.html` và `notebooks/`. Notebook public dùng cấu hình fingerprint từ input riêng; xem hướng dẫn public, không giả clone GitHub đã có toàn bộ dữ liệu.

## 3. Phiên Kaggle đã chạy xong — không submit lại

- Tài khoản hiện hành: `huyv12345`; không còn dùng `dungdq210`.
- Notebook: https://www.kaggle.com/code/huyv12345/gemma-end-to-end-report-20261017 . **Version1 COMPLETE**, session359208713.
- Version2 là **Quick Save `no_run=True`**, chỉ sửa dòng phân cách bảng Markdown; không chạy thêm GPU. Chọn version1 để xem Output đã chạy. Notebook local hiện tương ứng bản sửa.
- Dataset private: https://www.kaggle.com/datasets/huyv12345/gemma-standalone-native-20261017 . Gồm native inputs của recipe đã khóa.
- GPU thực: Tesla T4, Torch2.11.0+cu128/CUDA12.8. Pipeline75,48 phút; cả notebook khoảng77 phút; hoàn tất khoảng11:26 giờ Việt Nam17/10/2026.
- Không train/encode model trên laptop. Laptop chỉ làm kiểm chứng nhẹ và xuất tài liệu.
- Watcher đã kết thúc sau khi tải và kiểm kết quả. `verification/kaggle_execution.json` có `status=COMPLETE`, `full_pipeline_verified=true`.

ZIP gốc đã tải local: `gemma_research_standalone/runs/kaggle_verified/gemma_e2e_20261017_030922_673908_results.zip`,82.742.911 bytes. Nó giữ code snapshot, native cache mới, heads, predictions, CI, training/validation histories, raw timing và báo cáo. Ảnh input và pretrained weights không nằm trong ZIP kết quả.

Đã kiểm manifest của run/core/timing/report, tái tính độc lập15 variants trên18537 test records và864 timing calls, kiểm cache5817 ảnh/20065 texts, cùng source thực thi. Cache mới trùng byte snapshot gốc; max absolute prediction drift=0, metric drift=0; config/epoch được chọn giữ nguyên. Receipt: `verification/gpu_independent_audit.json`, `gpu_final_qa.json`, `DELIVERY_CHECKS.json`.

GPU version1 có HTML/biểu đồ/số liệu đầy đủ nhưng bảng Markdown có dòng trắng thừa. Bản `latest_gpu_run_formatted` được dựng lại từ cùng core/timing, chỉ sửa trình bày, không train/encode lại; provenance ở `verification/formatted_report_derivation.json`. Không sửa ZIP producer để che khác biệt.

## 4. Phương pháp đang có

Backbone `google/embeddinggemma-2`, revision `914f7f89142e33e77833254d9c9b90c3cef7303b`, frozen FP32; embedding768 chiều, L2-normalized. Prefix text chính xác `A photo depicts ` có khoảng trắng cuối. Không dùng Gemma FP16.

```text
x = normalize(x)
r(x) = W_up tanh(W_down x)
b(x) = r(x) / max(1, ||r(x)||2)
A(x) = normalize(x + radius * b(x))
score(I,c) = 2.5 * max(0, cosine(A(u_image), A(v_text)))
```

Adapter chung cho cả image/text, hai linear không bias, W_up khởi tạo0. Bound nằm trước normalization cuối; không tự diễn giải radius thành exact angular/Euclidean displacement sau normalization. Score không phải xác suất đã calibration.

Pairwise softplus ranking loss, temperature0.05; Sugar margin0.02, human margin0. Mean dùng mean rating để xếp caption pairs trong cùng ảnh. Uncertainty dùng weight `gap/(gap+SE_pos+SE_neg+1e-6)`; singleton SE=0 là heuristic. Preservation giữ cả embedding và cosine score. Equal domain loss budgets, AdamW lr0.001/wd0.0001, batch128, tối đa15 epochs.

Grid5 configs ×3 seeds11/23/37 cho mỗi phase. Chọn config/epoch **chỉ trên validation**: gain Sugar accuracy/Flickr caption-mean tau-c/Polaris caption-mean tau-c, chuẩn hóa[0.01,0.02,0.02]; yêu cầu mean gain không âm ở cả ba miền, maximize weakest gain, fallback base nếu không có config hợp lệ. Refit train+validation bằng median epoch. Mean seed11 primary đã khóa; ensemble3 diagnostic, không chọn lại bằng test.

Mean chọn rank32/radius0.03/preservation0.1/14 epochs; uncertainty cùng config/11 epochs. Head49152 params,192KiB FP32. So mean/uncertainty gốc chưa cô lập weighting vì khác epoch.

## 5. Dữ liệu và kết quả đã xác minh

22642 records;18537 test records. Test Sugar4703 pairs; Flickr5112 captions/900 ảnh/15336 votes; Polaris8722 captions/2737 ảnh/26098 votes. Native input cần5817 ảnh và20065 text strings.

Split theo decoded RGB pixel hash, disjoint toàn cục train/validation/test. Polaris official TEST được giữ; Flickr development được dựng từ seed nguồn; **Sugar split được dựng mới**; loại candidate trùng đúng một supplied reference trước scoring. Không phát hiện mọi crop/resize/reencode hoặc pretraining overlap. Không coi các metric này là cùng cohort với mọi bảng published.

| Variant | Sugar accuracy % | Flickr raw tau-c | Polaris raw tau-c |
|---|---:|---:|---:|
| Base |91.62237|0.544936|0.556743|
| Mean primary |93.53604|0.557082|0.558193|
| Uncertainty secondary |93.49351|0.558029|0.558149|

| Mean trừ base | Delta | Paired image-cluster CI95 |
|---|---:|---:|
| Sugar |+1.91367pp|[+1.38061,+2.42969]pp|
| Flickr raw tau-c |+0.012146|[+0.009446,+0.014905]|
| Polaris raw tau-c |+0.001450|[−0.000628,+0.003424]|

CI2000 resamples, giữ captions/votes trong mỗi ảnh; exploratory, chưa chỉnh multiplicity. Single-seed CI không bao gồm toàn bộ training variability. CI chứa0 không chứng minh tương đương.

Mean sửa124 Sugar pairs và tạo34 lỗi mới, net+90. Category swap_att giảm5/435 pairs. Flickr within-image ranking tăng0.67660pp; Polaris giảm0.08579pp dù global raw correlation có point gain. Không che các đánh đổi này.

Control CPU năm seed11/23/37/53/71 đã hoàn tất trước lượt GPU mới: cùng mean config, cùng14 epochs, paired initialization/batches. Uncertainty-minus-mean trung bình: Sugar−0.102063pp, CI seed×image[−0.328247,+0.117292]pp; Flickr raw−0.000561, CI[−0.001077,−0.000067]; Polaris raw−0.000839, CI[−0.001410,−0.000246]. Raw Flickr/Polaris giảm5/5 seeds. **Không đổi primary sang uncertainty.** Chỉ so CPU với CPU; đây là control lịch sử, không phải chạy mới cùng notebook GPU.

Timing mới của mean trên128 ảnh/864 calls: B1 p50/p95527.903/534.859ms; B8 effective2.10439 pairs/s; peak allocated2811.7MiB. Gồm preprocessing/backbone/normalize/head/score, loại disk IO/model download; ảnh đã decode. Peak không phải toàn bộ VRAM/reserved. Cached-head latency không thay native latency.

Stage12 đã mở rộng đủ7 backbone,81 variants/aliases,21 nhánh chính,42 timing rows/6048 calls. Gemma cao nhất ba point metrics trong7 mean recipes đã đo, **chưa phải SOTA**. Ngữ cảnh lịch sử: SigLIP256 mean89.98512%/.528995/.555353, B1 p5039.895ms; CLIP L14 mean giảm Polaris raw, CI dưới0; SPECS thấp trên cohort hiện tại; PAC-S và SigLIP256 có một số gain CI chứa0. Gemma FP32, phần lớn recipe khác FP16, session khác nhau; chưa có paired cross-model test, không quy khác biệt riêng cho architecture. Timing Stage12 cũ của Gemma522.618ms khác số đo mới527.903ms; giữ nhãn session/date.

## 6. Đánh giá khả năng nộp paper

Triển khai/tái lập mạnh; thực nghiệm khá; novelty và tổng quát hóa còn là điểm nghẽn. Không gán paper “rank A” chỉ vì notebook chạy được hay đạt điểm cao nhất trong7 recipes. CORE/ICORE xếp hạng venue, không cấp điểm cho từng paper. Định hướng hiện tại là workshop phù hợp hoặc cân nhắc venue B sau chỉnh sửa; A là mục tiêu cần bổ sung bằng chứng, không phải cam kết nhận bài.

CLIP-Adapter đã có bottleneck residual feature adaptation; phải phân biệt đóng góp bounded/shared/preservation và validation guard với prior art, không chỉ thay backbone. ARR cũng đánh giá soundness, novelty/impact và reproducibility riêng. Đọc nguồn:

- CLIP-Adapter: https://arxiv.org/abs/2110.04544
- ARR review criteria: https://aclrollingreview.org/reviewform
- Polos: https://github.com/YuigaWada/Polos
- FLEUR: https://aclanthology.org/2024.acl-long.205/
- VQAScore: https://github.com/linzhiqiu/t2v_metrics
- Phân tích liên quan mới: https://aclanthology.org/2026.findings-acl.1948/
- Venue ranking phải kiểm lại đúng edition: https://portal.core.edu.au/conf-ranks/

## 7. Việc tiếp theo, theo thứ tự ưu tiên

### P0 — Chốt đóng góp và protocol trước khi chạy thêm

Viết bảng related work phân biệt: nhiệm vụ, backbone, supervision, reference-free/reference-aware, kiểu adapter/bound/preservation, benchmark và chi phí. Kiểm paper mới2025–2026 trước khi khẳng định novelty. Đề xuất câu hỏi: **Khi nào adaptation tăng compositional discrimination nhưng giảm alignment với human judgments, và bounded shared adaptation kiểm soát đánh đổi đó đến đâu?**

Viết protocol/version mới: hypothesis, splits, hyperparameter budget, primary metrics/contrasts, seed list, thống kê, GPU budget, stop criteria. Đóng băng trước inference/training mới. Không đổi claim theo điểm test đang có; các corpus hiện tại đã được quan sát nhiều lần.

### P1 — Baseline cùng backbone để phân biệt đóng góp

Ưu tiên dùng cache hiện có, không encode Gemma lại chỉ để train head:

1. Frozen base.
2. Shared linear head.
3. Shared MLP/residual adapter không bound, cùng rank/params.
4. Bounded shared adapter không preservation.
5. Bounded shared adapter có preservation (phương pháp hiện tại).
6. Separate image/text adapters, budget tham số tương đương; ví dụ hai rank16 adapters so với shared rank32.

Giữ cùng training data, domain weighting, optimizer, epoch/search budget và seeds; source/config khác nhau phải ghi rõ. Controls cùng kiến trúc có thể paired init/batches; kiến trúc khác không được giả là cùng initialization. Các ablation cũ fixed15epochs khác primary14epochs nên chưa thay thế toàn bộ control này. Đánh giá paired seed×image CI cho contrasts đã định trước; giữ mọi kết quả âm. Không chỉ so phương pháp với frozen backbone.

### P2 — Chứng minh tổng quát hóa thật

Thiết kế leave-one-dataset-out: train trên hai corpus, chỉ chọn hyperparameters bằng validation của hai corpus đó, test corpus còn lại tuyệt đối không tham gia selection. Guard ba miền hiện tại cần thiết kế lại cho setting này; không dùng validation của miền held-out để chọn config.

Thêm ít nhất một corpus/benchmark chưa dùng để phát triển method, đúng định nghĩa metric và official protocol. Nếu train hiện tại đã trùng official test thì phải retrain loại phần trùng, không chỉ đổi tên split hoặc chạy lại cùng head. Audit exact/near duplicate và giải thích giới hạn pretraining overlap; không cam kết đã loại được pretraining contamination khi chưa kiểm.

### P3 — Đối thủ chuyên dụng và so sánh công bằng

Ưu tiên Polos/FLEUR/VQAScore và các baseline cập nhật qua literature review; chọn subset theo scope/compute. Dùng implementation/checkpoint/prompt tác giả, pin version. VQAScore có legacy release để tái lập paper và bản mới hơn; không trộn chúng rồi gọi cùng baseline. Label rõ reference-aware/reference-free và supervision để không biến khác thiết lập thành claim thắng tuyệt đối.

Chạy cùng cohort, preprocessing và metric definitions; báo paired cross-model CI. Chỉ đặt cạnh số published khi protocol/split thực sự tương đương. Không thêm GPT/Claude theo mặc định; lịch sử task loại chúng khỏi danh sách benchmark. Model cần VRAM vượt T4 phải lập kế hoạch trước, không giảm precision/đổi prompt tùy ý rồi vẫn gắn nhãn baseline chuẩn.

### P4 — Cost và cơ chế

Tách bảng best/native recipe khỏi control cùng hardware/runtime/precision. Không dùng Gemma FP16. Panel timing label-blind, interleave, warmup/synchronize, raw calls và paired timing CI như hiện tại. Khi so head mới, cố giữ inference batch/panel và model receipt thống nhất.

Phân tích angle/score drift, correction norms, preservation vs compositional gain, category failures và within-image/global-correlation disagreement. Nếu muốn claim bound sau normalization thì phải tự suy ra và kiểm chứng; công thức hiện tại chỉ bound residual trước normalization. Không bắt buộc thêm mô-đun phức tạp chỉ để “trông mới”.

### P5 — Manuscript và artifact

Sau P0–P3 mới chốt framing/venue. Viết: vấn đề, prior work gap, method, protocol/leakage audit, main results, controlled ablations, generalization, uncertainty, cost, limitations và reproducibility. Phân biệt rerun chính xác với nghiên cứu mới. Chuẩn bị artifact truy cập được theo quy định venue, ẩn danh khi cần và kiểm giấy phép dữ liệu/model/code. GitHub public hiện chỉ có code/tài liệu/aggregate results; private data artifacts không tự thành public vì có link repo.

Không hứa rank A, acceptance rate hoặc deadline khi chưa xác minh venue/CFP hiện hành. Không tối ưu trực tiếp theo reviewer guess hoặc test đã xem.

## 8. Môi trường, quyền và quy tắc thao tác

- Dùng Kaggle GPU cho phần nặng, laptop không đủ tài nguyên để train/encode model.
- Không cần conda. Local CLI đang dùng uv/Python3.13; Python mặc định máy là3.10. Repo có `.venv` CLI; dùng đúng interpreter khi gọi Kaggle. Không tự cài CUDA Torch vào laptop.
- Credential Kaggle nằm trong cấu hình tài khoản Windows, ngoài repo. Không đọc/in/copy token vào notebook, log, Git hoặc handoff. HF token nếu cần dùng Kaggle Secrets.
- Giữ notebook/dataset private trừ khi người dùng yêu cầu khác. Public code/docs không đồng nghĩa public ảnh, labels, per-record predictions, cache, checkpoint hoặc dataset fingerprints.
- Không sửa source frozen hoặc bằng chứng gốc để làm đẹp số liệu. Hướng nghiên cứu mới đặt folder/protocol/version riêng.
- Lượt GPU hiện hành đã COMPLETE. Không resubmit watcher/core/timing chỉ để kiểm trạng thái. Chỉ dùng GPU mới khi đã chốt experiment mới cần thiết; giới hạn tài nguyên rõ ràng.
- Workspace gốc có staging Stage11 của user. Không `git add .`, `reset --hard`, clean hoặc trộn staging đó vào commit khác. Publication mới dùng checkout riêng và allowlist code/docs/aggregate plots. Không sửa `kaggle-cli/`.

## 9. Tiêu chí kết thúc giai đoạn nâng paper

Phải trả lời được: đóng góp khác prior art ở đâu; gain còn khi so cùng backbone/budget không; có generalize ngoài corpus đã dùng phát triển không; đối thủ chuyên dụng nào đã so công bằng; gain nào CI chưa rõ/đánh đổi âm; phương pháp có đáng chi phí encoder không. Nếu bằng chứng không ủng hộ method claim, có thể viết paper phân tích/negative results với framing đúng, không ép thành SOTA.

## 10. Prompt chuyển sang phiên mới

> Đọc GEMMA_PAPER_HANDOFF_20261017_VI.md trước. Gemma GPU end-to-end version1 đã COMPLETE; version2 chỉ Quick Save sửa Markdown, không chạy GPU. Hãy kiểm receipts hiện hành, đọc method/report, lập novelty/baseline gap table và protocol mới cho P1/P2 trước khi dùng GPU. Ưu tiên head controls trên cache; không encode lại toàn bộ Gemma hoặc chọn lại primary bằng test đã quan sát. Chưa có cơ sở khẳng định đủ rank A/SOTA. Giữ source/evidence gốc, staging của user và private credentials/data artifacts; báo cáo cả kết quả âm, CI và cost. Dùng gemma_research_public cho cây GitHub và gemma_research_standalone cho gói đầy đủ local.

## 11. Publication và kiểm tra khi chuyển phiên

Lần publication này dùng worktree `tmp/github-paper-handoff-20261017`, branch `publish/gemma-paper-handoff-20261017`, base commit `a07bbb4`; đích là `origin/main` tại repo đã nêu. Commit mới chỉ chứa handoff, code/notebook/docs và aggregate reports/plots. Xem commit mới nhất trên GitHub để xác nhận mã commit đã push. Không đưa toàn bộ `gemma_research_standalone/` hoặc staging Stage11 gốc lên repo public.

QA riêng cho packaging public đã pass: parse source, validate/giải mã hai notebook, kiểm frozen vendor bytes, thiếu private config phải fail, đọc private config hợp lệ, CPU replay/rescore từ embedded source trong thư mục độc lập, GPU-disabled path, và dựng lại báo cáo từ producer hoàn tất. Sáu bảng aggregate chính trùng byte với bản đã kiểm. Không refit head hoặc chạy lại GPU trong QA packaging. Phạm vi chi tiết: `gemma_research_public/VERIFICATION_VI.md`.

`gemma_research_public/tools/export_private_runtime.py` tạo `gemma_private_runtime.json` từ local full `PINNED.json`. File này phải ở ngoài Git và attach thành input Kaggle private nếu chạy notebook public. Dataset native hiện có vẫn dùng được; bản public không nhúng sáu fingerprint dữ liệu. Full local notebook đã hoàn tất không cần bước tách cấu hình này.

Chạy `python verify_repository.py` và `python gemma_research_public/tools/verify_public.py` từ checkout public để kiểm artifact bytes; đây không phải training/evaluation mới. `.gitattributes` giữ line endings để hash không hỏng trên Windows.

Local workspace gốc được giữ nguyên HEAD/index; vì push từ worktree riêng nên branch `main` ở workspace gốc có thể đứng sau remote. Không `git pull`/reset trực tiếp lên staging đang có. Đọc/tiếp tục publication trong worktree; đồng bộ workspace gốc sau khi người dùng xử lý thay đổi riêng. Staging Stage11 có 25 file và không thuộc commit publication này.
