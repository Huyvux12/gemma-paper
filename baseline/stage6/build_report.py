import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
j=json.loads((ROOT/'verified_confirmation_results.json').read_text());assert j['verification']['passed']
labels={'gemma-CLIPLiteral':'Gemma CLIPLiteral, 768 f32','siglip224-raw':'SigLIP2-224, caption nguyên bản','siglip224-CLIPLiteral':'SigLIP2-224, cùng prefix CLIP','clip-B32':'CLIP ViT-B/32','clip-L14':'CLIP ViT-L/14'}
metrics=['| Cấu hình | Đúng / 560 | Accuracy |','|---|---:|---:|']
for k,label in labels.items():
 m=j['metrics'][k];metrics.append(f"| {label} | {m['correct']} | {m['accuracy']*100:.2f}% |")
secondary=['| Gemma − baseline phụ | Chênh lệch, điểm % | CI 95%, điểm % | McNemar p, Holm 3 |','|---|---:|---|---:|']
for k,m in j['secondary'].items():secondary.append(f"| {labels[k]} | +{m['delta_pp']:.2f} | [{m['ci95_pp'][0]:.2f}; {m['ci95_pp'][1]:.2f}] | {m['p_holm_family3']:.3g} |")
category=['| Nhóm lỗi, 80 ảnh/nhóm | Gemma literal | SigLIP2-224 raw | SigLIP2-224 literal |','|---|---:|---:|---:|']
for c in sorted(j['metrics']['gemma-CLIPLiteral']['category_accuracy']):
 category.append('| '+c+' | '+' | '.join(f"{100*j['metrics'][k]['category_accuracy'][c]:.2f}%" for k in ['gemma-CLIPLiteral','siglip224-raw','siglip224-CLIPLiteral'])+' |')
report='''# EmbeddingGemma 2 thay encoder CLIPScore — Stage 6

Ngày 08/10/2026. Kiểm chứng file Colab người dùng gửi: `siglip224_confirmation_results.zip`. Báo cáo này cập nhật trạng thái phép so sánh chính còn chờ ở stage 5.

## Kết luận

**Phép so sánh chính đã hoàn tất và đạt điều kiện tiếp tục nghiên cứu.** Trên đúng 560 ảnh đã khóa, Gemma prefix `A photo depicts `, 768 chiều float32 đạt **503/560 = 89,82%**; SigLIP2 Base patch16-224 dùng caption nguyên bản đạt **426/560 = 76,07%**. Chênh lệch **+13,75 điểm %**, paired bootstrap CI95% **[10,18; 17,50]**, McNemar exact **p = 2,56 × 10^-13**.

Đây là bằng chứng về khả năng phân biệt caption đúng/sai trên SugarCrepe. Chưa đủ để khẳng định thay thế chung CLIPScore trên mọi chất lượng caption, hoặc rằng chỉ đổi backbone đã là đóng góp mới đủ cho paper.

## Kiểm chứng file trả về

- ZIP hợp lệ; manifest, protocol, reference Gemma và runner **khớp nguyên vẹn từng byte** với các file đã gửi. Không thực thi mã Python trong file upload để xác minh; dùng script kiểm tra riêng.
- Manifest đúng SHA-256 `a2e9fa5fb7e9866631c0546e5a09a2f9d11044335267712591476572b6104efe`, 560 ảnh duy nhất, mỗi nhóm 80 ảnh. Đây là ảnh chưa dùng để phát triển cấu hình trong chuỗi thí nghiệm; không khẳng định ngoài dữ liệu pretraining.
- SigLIP2-224 có đủ vector ảnh và bốn bộ vector caption raw/literal × positive/negative, mỗi bộ 560 × 768, float32, hữu hạn, norm gần 1. Thứ tự ảnh được suy ra từ manifest đã khóa theo đúng runner, thứ tự caption giữ nguyên manifest.
- Tính lại điểm Gemma từ vector lưu ở stage 5 và đối chiếu reference. Tính lại SigLIP2-224 từ vector upload; số đúng, thắng/thua ghép cặp, p và CI đều khớp JSON Colab.
- Kiểm tra scalar độc lập **5.600 điểm số** của 5 cấu hình; sai số tối đa **2,66 × 10^-15**, không đổi bất kỳ quyết định caption nào. p McNemar được đối chiếu thêm bằng tổng tổ hợp nhị thức exact.
- Metadata Colab báo revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`, weight SHA-256 `612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b`. Runner gốc kiểm tra hash model và từng ảnh trước inference. Model weights và ảnh gốc không nằm trong ZIP; lượt này kiểm tra vector/metadata cùng mã runner, không tải lại các byte đó.

SigLIP2-224 được người dùng inference trên CUDA float32; Gemma và CLIP là vector CPU float32 đã kiểm chứng ở các lượt trước. **Không có encoder inference mới trong lượt xử lý này.**

## Kết quả mẫu 560

'''+ '\n'.join(metrics)+'''

Gemma và các baseline ở bảng này giữ công thức `2.5 × max(cosine, 0)` như stage 4. Dùng preprocessing native của từng checkpoint; prefix Gemma không thay đổi sau khi thấy kết quả mẫu 560. Với SigLIP2-224, bỏ clipping để dùng cosine nguyên cũng cho **426/560** với raw và **434/560** với literal; không có hòa điểm và không có quyết định bị clipping đảo trên ảnh đúng.

Đối chiếu primary có **97** ảnh Gemma đúng/SigLIP2 raw sai, **20** ảnh Gemma sai/SigLIP2 raw đúng, **406** ảnh cả hai đúng và **37** ảnh cả hai sai. McNemar chỉ dùng 117 cặp bất đồng. CI là bootstrap 10.000 lần ghép cặp theo ảnh, seed 20261009; một ảnh/một cặp trong mẫu này.

## Nhóm so sánh phụ đã khóa

'''+ '\n'.join(secondary)+'''

Đã đủ cả ba phép phụ: matched-prefix SigLIP2-224, CLIP B/32 và CLIP L/14; hiệu chỉnh Holm cùng một family 3. CI trong bảng là CI riêng từng phép, chưa hiệu chỉnh đồng thời. So sánh chính là một phép riêng theo protocol gốc.

Khi cùng dùng prefix `A photo depicts `, Gemma vẫn hơn SigLIP2-224 **12,32 điểm %**, CI95% **[8,57; 16,25]**, p Holm **2,91 × 10^-10**. Kết quả này cho thấy lợi thế quan sát không được giải thích hoàn toàn bằng việc chỉ thêm prefix CLIP cho Gemma.

## Các nhóm lỗi

'''+ '\n'.join(category)+'''

Các tỷ lệ theo nhóm dùng để mô tả, không có kiểm định từng nhóm đã hiệu chỉnh. Gemma vẫn yếu hơn trên swap_obj (72,50%) và replace_rel (81,25%) so với nhóm đơn giản. Không dùng mẫu này để chọn lại prefix hoặc thêm thành phần mới rồi gọi cùng mẫu là test xác nhận.

## Đối chứng ảnh sai và truncation

Ghép mỗi cặp với 559 ảnh còn lại tạo 313.040 phép ghép/cấu hình. SigLIP2-224 raw ưu tiên caption positive gốc theo cosine nguyên **48,87%**, literal **51,55%**; Gemma literal ở stage 5 là **49,53%**. Đây là ưu tiên caption gốc khi ảnh không liên quan, không phải accuracy của caption với ảnh sai. Các ghép lặp lại không là 313.040 mẫu độc lập.

Với SigLIP2 raw, ép clipping dẫn tới **66,39%** ghép sai hòa 0; không dùng tỷ lệ accuracy sau clipping thấp để tuyên bố grounding mạnh hơn. Trên ảnh đúng, clipping và cosine nguyên cho cùng các quyết định như đã kiểm tra ở trên.

Audit từ Colab: raw tối đa 27 token; literal tối đa 30; context64; cả 4 bộ caption đều có 0 trường hợp vượt context. Không có evidence truncation trong run này.

## Hiệu năng và dung lượng

Metadata báo khoảng **24,76 giây** cho phần encode ảnh còn thiếu + caption + ghi checkpoint trong run CUDA này; không gồm tải dữ liệu/model và load model, có cơ chế resume. ZIP không ghi tên GPU. Không so thời gian này với Gemma chạy CPU ở phiên khác để kết luận tốc độ.

Kết quả nén vector của stage 5 giữ nguyên: Gemma literal256 int8 **88,21%**, payload260 byte/vector gồm scale f32, so với 768f32 3.072 byte (**giảm11,82 lần**). Chênh lệch −1,61 điểm %, CI95% [−3,22;0]; đây là tradeoff mô tả, chưa chứng minh tương đương. Nén vector không giảm thời gian encoder.

## Ý nghĩa cho hướng paper và bước tiếp theo

Điều kiện go/no-go gốc là dừng/thu hẹp hướng nếu không có gain so với SigLIP2-224. Phép chính hiện có gain rõ, và phép matched-prefix cũng giữ gain: có cơ sở tiếp tục hướng **compositional discrimination + tradeoff lưu vector**.

Các kết quả cohort rộng hơn ở stage 5 vẫn bổ sung: 3.949 cặp/1.005 ảnh chưa dùng phát triển; Gemma literal768 đạt91,57%, SigLIP2-256 đạt82,07%. **560 ảnh là tập con của 1.005 ảnh**, không phải test độc lập để cộng cỡ mẫu. SigLIP2-256 là checkpoint khác, không gộp với SigLIP2-224.

Phần còn thiếu quan trọng là human correlation: Gemma Flickr8k-Expert mở rộng ở archive nguồn mới có 936/1.000 vector ảnh và chưa có vector caption hoàn chỉnh. Chưa có kết quả mở rộng để báo cáo. Cần hoàn tất 1.000 ảnh/5.664 caption, đánh giá Kendall tau-c và bootstrap ghép theo ảnh, đặc biệt cohort 900 ảnh mới. Mẫu expert100 ảnh ở stage4 vẫn thăm dò.

Để phát triển thành paper, còn phải làm rõ đóng góp vượt thay backbone: ví dụ cơ chế chấm có căn cứ, hoặc phân tích hệ thống chất lượng–dung lượng–grounding; mọi cơ chế học/chọn mới phải đánh giá trên dữ liệu test tách khỏi mẫu đã quan sát. Bằng chứng hiện tại xác nhận tín hiệu của hướng đã chọn, chưa xác nhận novelty hoặc khả năng chấp nhận của một venue.

## Tái lập

`embeddinggemma2_stage6_siglip224_verified.zip` có file Colab gốc đã trích, các vector Gemma/CLIP tham chiếu, mã kiểm chứng, CSV điểm, JSON kết quả và báo cáo. Từ thư mục giải nén:

```sh
python -m pip install numpy scipy
OPENBLAS_NUM_THREADS=1 python stage6/analyze_confirmation.py
```

Lệnh chỉ chấm và kiểm tra vector offline, không tải/chạy model. `stage6/uploaded/run_siglip224.py` là bản runner Colab gốc để đối chiếu, không được thực thi bởi script xác minh. `package_sha256.json` lưu hash các file kèm theo.
'''
reportpath=BASE/'EmbeddingGemma2_Stage6_SigLIP224_Verified_VI.md';reportpath.write_text(report)
(ROOT/'README.md').write_text('''# Stage 6: Original SigLIP2-224 primary contrast completed

The uploaded user Colab inference is verified from vectors. The five-config 560-image analysis is reproducible offline:

```sh
python -m pip install numpy scipy
OPENBLAS_NUM_THREADS=1 python stage6/analyze_confirmation.py
```

Do not treat archived CPU timings and returned CUDA timings as a hardware-controlled comparison. This package does not include model weights or raw images. It verifies the byte-identical runner/input, vector arithmetic, exact p-values, bootstrap reproduction, and matching user-returned metrics. Image/model SHA provenance is the original runner's checks and recorded metadata.

Stage5 metadata/results are historical provenance; any pending-primary status in those old files is superseded by `stage6/verified_confirmation_results.json` and the Stage6 report. Human evaluation remains incomplete. The 560 sample is part of the larger unseen cohort, not independent additional data.
''')
files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
files += [reportpath]
files += [BASE/'stage5'/name for name in ['frozen_manifest.json','frozen_protocol.json','primary_reference.json','run_siglip224.py','results_560.json','independent_verification.json','full_sugar_results.json','full_source_protocol.json']]
for kind in ['gemma','clip','clip_large']:
 files += [BASE/'stage5/runs'/kind/name for name in ['embeddings.npz','metadata.json']]
hashes={p.relative_to(BASE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
hashpath=ROOT/'package_sha256.json';hashpath.write_text(json.dumps(hashes,indent=2));files.append(hashpath)
archive=BASE/'embeddinggemma2_stage6_siglip224_verified.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(files):z.write(p,p.relative_to(BASE).as_posix())
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for name,sha in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==sha
print(json.dumps({'report_bytes':reportpath.stat().st_size,'zip_bytes':archive.stat().st_size,'files':len(files),'all_hashes_match':True}))
