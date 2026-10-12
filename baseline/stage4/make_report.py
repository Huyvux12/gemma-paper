import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
diag=json.loads((ROOT/'diagnostics_results.json').read_text())
base=json.loads((ROOT/'baseline_results.json').read_text())
fmt=lambda x:f'{100*x:.2f}%'
ci=lambda x:f'[{x[0]:+.2f}, {x[1]:+.2f}]'
names={'clip-CLIPScore':'CLIP ViT-B/32','clip_large-CLIPScore':'CLIP ViT-L/14','siglip2-raw':'SigLIP2 Base, caption nguyên bản','siglip2-CLIPLiteral':'SigLIP2 Base, prefix CLIP','gemma-CLIPLiteral':'Gemma, giữ prefix CLIP','gemma-FactChecking':'Gemma, FactChecking'}
rows=[]
for k in ['clip-CLIPScore','clip_large-CLIPScore','siglip2-raw','siglip2-CLIPLiteral','gemma-CLIPLiteral','gemma-FactChecking']:
    m=base['sugar']['metrics'][k]
    rows.append(f"| {names[k]} | {m['correct']}/350 | {fmt(m['accuracy'])} | {base['human']['tau_c'][k]:.4f} | {fmt(base['human']['within_image_accuracy'][k])} |")
compression=[]
for dim,form in [(768,'float32'),(768,'float16'),(768,'int8'),(512,'float32'),(256,'float32'),(256,'int8'),(128,'int8')]:
    k=f'CLIPLiteral-{dim}-{form}';v=diag['compression'][k];h=diag['human_compression'][k]
    compression.append(f"| {dim} / {form} | {v['raw_bytes_per_vector']} | {fmt(v['accuracy'])} | {v['vs_768f32']['delta_pp']:+.2f} | {ci(v['vs_768f32']['ci95_pp'])} | {h['tau_c']:.4f} |")
ground=[]
for k,v in diag['grounding'].items():ground.append(f"| {names[k]} | {fmt(v['matched_accuracy'])} | {fmt(v['raw_cosine_wrong_image_accuracy'])} | {fmt(v['leave_one_out_prototype_accuracy'])} |")
ground.append(f"| SigLIP2 Base, caption nguyên bản | {fmt(base['sugar']['metrics']['siglip2-raw']['accuracy'])} | {fmt(base['sugar']['grounding']['raw']['raw_cosine_wrong_image_accuracy'])} | Không chạy |")
cats=[]
for c in sorted(base['sugar']['metrics']['gemma-CLIPLiteral']['category_accuracy']):
    cats.append('| '+c+' | '+' | '.join(fmt(base['sugar']['metrics'][k]['category_accuracy'][c]) for k in ['clip_large-CLIPScore','siglip2-raw','siglip2-CLIPLiteral','gemma-CLIPLiteral'])+' |')
human=[]
for k,v in base['human']['comparisons'].items():
    human.append(f"| {k} | {v['delta_tau_c']:+.4f} | [{v['ci95_tau_c'][0]:+.4f}, {v['ci95_tau_c'][1]:+.4f}] | {v['delta_within_pp']:+.2f} | {ci(v['ci95_within_pp'])} |")

text='''# EmbeddingGemma 2 thay encoder CLIPScore — vòng kiểm chứng 4

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
'''+ '\n'.join(rows)+'''

SugarCrepe: hòa tính sai; mỗi nhóm 50 ảnh; mẫu cân bằng 350 ảnh khác nhau, không phải toàn bộ 7.511 cặp benchmark. Flickr8k: tau-c theo đánh giá cá nhân, không gộp điểm chuyên gia thành 1.656 ảnh độc lập; 990 cặp caption cùng ảnh có điểm trung bình khác nhau.

### So sánh với SigLIP2 mới

| So sánh | Delta accuracy, điểm % | CI 95%, điểm % | McNemar p | Holm p |
|---|---:|---|---:|---:|
'''
for key,v in base['sugar']['comparisons'].items():
    text+=f"| {key} | {v['delta_pp']:+.2f} | {ci(v['ci95_pp'])} | {v['p_exact']:.6g} | {v['p_holm']:.6g} |\n"
v=base['sugar']['matched_prompt_secondary']
text+=f"\nĐối chiếu cùng prefix CLIP: Gemma − SigLIP2 = **{v['delta_pp']:+.2f} điểm %**, CI 95% {ci(v['ci95_pp'])}; McNemar p={v['p_exact']:.6g}, phân tích phụ chưa chỉnh nhiều so sánh.\n"
text+='''
Các CI dùng 10.000 bootstrap ghép cặp theo ảnh, percentile 95%; Holm chỉ điều chỉnh hai p của so sánh raw đã khai báo, không điều chỉnh đồng thời CI. Đây là bằng chứng thăm dò trên mẫu cũ, không gọi là xác nhận đăng ký trước hay SOTA.

### Chi tiết Flickr8k

| So sánh | Delta tau-c | CI 95% tau-c | Delta xếp hạng, điểm % | CI 95% xếp hạng |
|---|---:|---|---:|---|
'''+ '\n'.join(human)+'''

Bootstrap ghép cặp theo 100 ảnh, 2.000 lần. CI phụ chưa chỉnh nhiều so sánh. Gemma nhỉnh hơn SigLIP2 raw về tau-c trong lượt này, nhưng **CI khi so với SigLIP2 cùng prefix CLIP chứa 0**. Vì vậy chưa kết luận cải thiện chung đánh giá caption theo chuyên gia; chưa chứng minh tương đương khi CI chứa 0. Kết quả với CLIP cũ ở vòng trước vẫn có CI chứa 0.

## 3. Model có thực sự dùng ảnh?

Giữ nguyên từng cặp caption, thay ảnh đúng bằng **từng ảnh trong 349 ảnh còn lại**: 122.150 phép ghép sai/encoder-prefix. Số này là số phép tính phụ thuộc nhau, không phải 122.150 mẫu độc lập. Mỗi encoder còn được chấm với cùng 2.000 hoán vị ảnh không có điểm cố định. Khoảng 2,5–97,5% của các hoán vị là khoảng kết quả dưới ghép sai, không phải CI accuracy.

| Cấu hình | Ghép ảnh đúng | Ghép mọi ảnh sai, raw cosine | Prototype ảnh khác, leave-one-out |
|---|---:|---:|---:|
'''+ '\n'.join(ground)+'''

Gemma prefix CLIP giảm từ 90,0% xuống 47,70% khi ảnh bị đổi; prototype đạt 48,0%. Kết quả ủng hộ sự phụ thuộc vào ảnh cụ thể trong mẫu này. Không đồng nghĩa chứng minh model hiểu mọi quan hệ, không loại trừ ghi nhớ dữ liệu tiền huấn luyện, và prototype không phải baseline language model hoàn chỉnh.

**Lưu ý về công thức chấm:** với ảnh sai của SigLIP2 raw, công thức `max(cosine,0)` làm cả hai caption cùng điểm 0 ở 67,74% phép ghép. Vì hòa tính sai, accuracy ghép sai sau clipping chỉ là 16,36%; raw cosine là 49,46%. Không dùng con số 16,36% để nói SigLIP2 ít thiên lệch hay phụ thuộc ảnh mạnh hơn Gemma. Bảng trên dùng raw cosine để đọc đối chứng ảnh sai; mọi accuracy benchmark chính vẫn giữ công thức CLIPScore.

## 4. Nén vector: lợi ích và mất mát

Giữ prefix CLIP. Int8 đối xứng theo từng vector: cắt chiều → normalize → scale=max(abs)/127 → làm tròn → lưu int8 cùng scale float32 → giải lượng tử và normalize khi chấm. Đây là nén vector sau inference; không lượng tử hóa trọng số encoder, không đo tốc độ tính tích vô hướng int8.

| Chiều / định dạng | Byte/vector thô | Accuracy | Delta vs 768-f32, điểm % | CI 95% delta | Flickr8k tau-c |
|---|---:|---:|---:|---|---:|
'''+ '\n'.join(compression)+'''

256-int8 cần 260 byte thay vì 3.072 byte, giảm khoảng **11,8 lần** payload vector; tính cả scale nhưng chưa tính metadata, header tệp, chỉ mục hay chi phí search. Tệp `gemma_256_int8.npz` có payload 1.001.000 byte và chứa cả các prefix đã có; dung lượng tệp có header lớn hơn payload. Mã giải lượng tử đọc tệp thật được kiểm tra lại.

768-int8 mất 1 dự đoán đúng trong 350 ảnh; 256-int8 mất ròng 8 dự đoán đúng nhưng có 18 quyết định khác so với 768-f32. Không gọi hai cấu hình tương đương hoặc non-inferior chỉ vì CI chứa 0. Int8 không đảm bảo tăng accuracy; kết quả tăng nhẹ so 256-f32 là nhiễu thứ tự điểm ở mẫu này, không phải lợi ích đã xác nhận.

FactChecking được báo cáo đủ trong `diagnostics_results.json`; không dùng kết quả 256d FactChecking gần 768d để suy ra nén mọi prefix đều không mất chất lượng.

## 5. Phân tích nhóm lỗi

| Nhóm lỗi, 50 ảnh/nhóm | CLIP L/14 | SigLIP2 raw | SigLIP2 prefix CLIP | Gemma prefix CLIP |
|---|---:|---:|---:|---:|
'''+ '\n'.join(cats)+'''

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
'''
(ROOT/'REPORT_VI.md').write_text(text)
(ROOT.parent/'EmbeddingGemma2_Stage4_Report_VI.md').write_text(text)
print('Report generated',len(text),'characters')
