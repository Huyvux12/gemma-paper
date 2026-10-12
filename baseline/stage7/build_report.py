import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
j=json.loads((ROOT/'human_results.json').read_text());v=json.loads((ROOT/'independent_verification.json').read_text());assert v['passed'];u=j['cohorts']['unseen'];f=j['cohorts']['full']
G='gemma-CLIPLiteral-768-float32';G256='gemma-CLIPLiteral-256-float32';FC='gemma-FactChecking-768-float32';B='clip-CLIPScore';L='clip_large-CLIPScore';S='specs-NativeSPECS';D='siglip-SigLIP2';R='siglip224-raw';M='siglip224-CLIPLiteral'
labels={G:'Gemma literal768 f32',G256:'Gemma literal256 f32','gemma-CLIPLiteral-256-int8':'Gemma literal256 int8',FC:'Gemma FactChecking768 f32',B:'CLIP B/32',L:'CLIP L/14',S:'SPECS trained spec.pt',D:'SigLIP2-256 native raw',R:'SigLIP2-224 native raw',M:'SigLIP2-224 native literal',R+'-clipped':'SigLIP2-224 raw, CLIPScore clipping',M+'-clipped':'SigLIP2-224 literal, CLIPScore clipping'}
metrics=['| Cấu hình | Tau-c, 900 ảnh mới | CI95%, 900 ảnh mới | Tau-c, full1.000 |','|---|---:|---|---:|']
for k,label in labels.items():
 m=u['metrics'][k];ci=m.get('ci95');cis=f"[{ci[0]:.4f}; {ci[1]:.4f}]" if ci else 'Mô tả';metrics.append(f"| {label} | {m['kendall_tau_c']:.4f} | {cis} | {f['metrics'][k]['kendall_tau_c']:.4f} |")
comparisons=['| Gemma literal768 − baseline | Δtau-c | CI95% riêng từng phép | CI Bonferroni family6 (nominal) |','|---|---:|---|---|']
for k in [B,L,S,D,R,M]:
 c=u['comparisons'][G+'_vs_'+k];ci=c['ci95'];joint=c['bonferroni_family6_ci'];comparisons.append(f"| {labels[k]} | {c['delta_tau_c']:+.4f} | [{ci[0]:.4f}; {ci[1]:.4f}] | [{joint[0]:.4f}; {joint[1]:.4f}] |")
key=G+'_vs_'+R;c=u['comparisons'][key];ci=c['ci95'];joint=c['bonferroni_family6_ci']
if ci[0]>0:conclusion='Trên cohort900 ảnh mới, Gemma literal768 có tương quan chuyên gia cao hơn SigLIP2-224 native raw theo CI95% ghép cặp.'
elif ci[1]<0:conclusion='Trên cohort900 ảnh mới, Gemma literal768 có tương quan chuyên gia thấp hơn SigLIP2-224 native raw theo CI95% ghép cặp. Cần thu hẹp claim về ưu thế compositional, không nói Gemma thay thế chung caption metric.'
else:conclusion='Trên cohort900 ảnh mới, CI95% chênh lệch Gemma literal768 và SigLIP2-224 native raw chứa0. Chưa có bằng chứng rõ rằng Gemma cải thiện tương quan chuyên gia so với baseline này; không suy ra tương đương từ CI chứa0.'
if joint[0]>0:conclusion+=' CI nominal Bonferroni family6 cũng nằm trên0 cho phép đối chiếu này.'
elif joint[1]<0:conclusion+=' CI nominal Bonferroni family6 cũng nằm dưới0 cho phép đối chiếu này.'
else:conclusion+=' CI nominal Bonferroni family6 chứa0 cho phép đối chiếu này.'
within_comparison=u['within_image_ranking']['comparisons'][G+'_vs_'+M]
within=['| Cấu hình | Xếp hạng đúng trong ảnh, 900 ảnh mới |','|---|---:|']
for k in [G,G256,'gemma-CLIPLiteral-256-int8',FC,B,L,S,D,R,M]:within.append(f"| {labels[k]} | {100*u['within_image_ranking']['accuracy'][k]:.2f}% |")
curve=[]
for prefix in ['CLIPLiteral','FactChecking']:
 curve+=['### '+prefix,'','| Chiều | Tau-c f32 | Tau-c f16 | Tau-c int8 (+scale f32) |','|---|---:|---:|---:|']
 for d in [128,256,512,768]:curve.append('| '+str(d)+' | '+' | '.join(f"{u['metrics'][f'gemma-{prefix}-{d}-{fmt}']['kendall_tau_c']:.4f}" for fmt in ['float32','float16','int8'])+' |')
 curve+=['']
gem=j['metadata']['gemma'];sig=j['metadata']['siglip224'];compression=u['comparisons'][G256+'_vs_'+G];int8=u['comparisons']['gemma-CLIPLiteral-256-int8_vs_'+G]
report=f'''# EmbeddingGemma 2 thay encoder CLIPScore — Stage7: Expert evaluation

Ngày08/10/2026. **Đã hoàn tất inference phần còn thiếu ngay trong môi trường này**, không cần người dùng chạy Colab cho vòng này. Kết quả human evaluation là phân tích phụ đã nêu từ protocol nguồn; được tách khỏi giả thuyết SugarCrepe224 chính đã đạt ở stage6.

## Kết luận

{conclusion}

Gemma literal768 tau-c **{u['metrics'][G]['kendall_tau_c']:.4f}**, SigLIP2-224 raw **{u['metrics'][R]['kendall_tau_c']:.4f}**; Δtau-c **{c['delta_tau_c']:+.4f}**, CI95% **[{ci[0]:.4f}; {ci[1]:.4f}]**.

Tau-c đo mức model xếp hạng caption giống các đánh giá chuyên gia; càng cao càng tốt. Đây không phải phần trăm caption đúng. Kết quả này không phủ định/biến đổi accuracy compositional89,82% của stage6; hai endpoint đo hai khía cạnh khác nhau.

## Dataset và cohort

- Flickr8k-Expert đầy đủ: **1.000 ảnh, 5.664 cặp ảnh-caption, 16.992 điểm cá nhân**. Mỗi caption có3 đánh giá.
- Cohort chưa dùng ở các lượt trước: **900 ảnh, 5.112 caption, 15.336 điểm cá nhân**; đây là cohort chính để đọc kết quả tiếp nối.
- 100 ảnh/552 caption cũ chỉ góp vào bảng full mô tả. Không cộng full và900 thành cỡ mẫu khác.
- Hai shard pinned revision `c30e1e67e2b37a42ecffaf9a88a05458ca05b973` của `yuwd/Flickr8k-HumanEval` đã khớp SHA-256 nguồn. Tái dựng toàn bộ caption, thứ tự và rating khớp từng record manifest; hash ảnh của toàn bộ1.000 ảnh cũng khớp.
- Manifest SHA-256 `{v['manifest_sha256']}`. “Ảnh mới” nghĩa là chưa dùng trong chuỗi thí nghiệm, không đảm bảo ngoài dữ liệu pretraining.

## Cấu hình và phần đã chạy

Gemma giữ prefix CLIP literal `A photo depicts ` và FactChecking đã có từ trước, không chọn prefix theo điểm expert mở rộng. Text+vision float32, audio encoder không nạp; checkpoint `914f7f89142e33e77833254d9c9b90c3cef7303b`, native image budget280. Model và tokenizer lớn đều khớp receipt SHA-256.

- Gemma tiếp tục từ936 vector ảnh archive và prior100 đã xác minh, tổng seed **{gem['seeded_image_count']}** ảnh; mới mã hóa **{gem['images_new_this_invocation']}** ảnh trong lượt hoàn tất. Hai bộ caption được hoàn tất; reuse string-identical prior captions theo metadata, phần còn lại được inference mới.
- Bridge16 ảnh deterministic +32 caption/prefix khớp cache trước reuse; minimum cosine gần1, max error ảnh dưới6×10^-7 và caption dưới1,3×10^-7. Không relabel cache chưa hoàn chỉnh.
- SigLIP2-224 được inference mới đầy đủ1.000 ảnh và cả raw/literal caption, revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`, weight SHA đã khớp. Token audit không có over-context.
- CLIP B/32, L/14, SPECS trained `spec.pt`, SigLIP2-256 được giữ từ checkpoint full-human float32 hoàn chỉnh, cùng manifest/caption/ảnh. Không dùng SPECS LongCLIP base làm đại diện.
- Không fine-tune, không hiệu chỉnh score bằng human labels.

## Tương quan với điểm chuyên gia

'''+ '\n'.join(metrics)+'''

Gemma/CLIP dùng `2.5×max(cosine,0)`. SPECS dùng affine cosine native `(cosine+1)/2`; SigLIP2-256 và224 native dùng cosine nguyên. Đã khóa addendum scoring224 trước khi có Gemma caption score mở rộng; đồng thời báo cáo cả224 CLIPScore-clipped sensitivity, tránh che ảnh hưởng tie ở tương quan giữa nhiều ảnh. Cosine có cùng thứ tự với logit native qua một scale dương và bias cố định; không dùng sigmoid probability làm xác suất caption đúng.

Kendall tau-c dùng **điểm cá nhân**, lặp lại score model cho từng chuyên gia, cùng quy trình nguồn. Spearman được giữ trong JSON bổ sung. Không dùng average rating thay endpoint tau-c chính.

## So sánh ghép cặp theo ảnh

'''+ '\n'.join(comparisons)+'''

Image-cluster paired bootstrap2.000 lần: sample ảnh với replacement, giữ toàn bộ caption và điểm chuyên gia thuộc ảnh đó; cùng draws cho mọi model. CI riêng95% là các kết quả human phụ. Cột Bonferroni dùng percentile ở mức nominal1−0,05/6 cho sáu contrast literal768 với baseline, có tính xấp xỉ bootstrap; không báo p-value giả từ tỷ lệ bootstrap hoặc chọn phép thuận lợi nhất.

## Xếp hạng caption trong cùng ảnh

'''+ '\n'.join(within)+f'''

Trên cohort900 có **{u['within_image_ranking']['n_unequal_human_caption_pairs']}** cặp caption với average rating khác nhau, từ **{u['within_image_ranking']['n_images_with_unequal_pairs']}** ảnh. Với endpoint phụ này mới dùng average rating để chọn caption tốt hơn; score hòa tính sai. CI contrast được bootstrap theo ảnh5.000 lần và nằm trong JSON. Các cặp cùng ảnh không là mẫu độc lập. Gemma literal768−SigLIP2-224 matched-prefix trong ảnh: Δ **{within_comparison['delta_pp']:+.2f} điểm %**, CI95% **[{within_comparison['ci95_pp'][0]:.2f}; {within_comparison['ci95_pp'][1]:.2f}]**. Cần đọc endpoint này riêng với tau-c toàn cục; không quy global correlation thành thắng rõ mọi kiểu ranking.

## Curve lưu vector trên human evaluation

'''+ '\n'.join(curve)+f'''

Gemma literal256f32−768f32: Δtau-c **{compression['delta_tau_c']:+.4f}**, CI95% **[{compression['ci95'][0]:.4f}; {compression['ci95'][1]:.4f}]**; one-sided95% lower **{compression['one_sided_95_lower']:+.4f}**. Protocol nguồn chỉ nêu tau tolerance0,01 theo hướng thăm dò; trạng thái theo ngưỡng nguồn: **{compression['source_exploratory_tolerance_passed']}**. Không gọi đây là bằng chứng tương đương metric chung.

256int8−768f32: Δtau-c **{int8['delta_tau_c']:+.4f}**, CI95% **[{int8['ci95'][0]:.4f}; {int8['ci95'][1]:.4f}]**. Int8 theo từng vector sau truncate và normalize, scale maxabs/127 float32, reconstruct rồi normalize. Payload260 byte/vector so với768f32 3.072 byte, giảm11,82 lần; không tính metadata/index. Curve là nén vector, không lượng tử hóa hoặc tăng tốc encoder.

## Kiểm chứng độc lập

Đã kiểm tra **{v['score_scalar_verification']['scalar_scores_checked']}** điểm native float32 bằng scalar fsum độc lập; sai số tối đa **{v['score_scalar_verification']['max_abs_score_error']:.3g}**. Thuật toán tau-c theo histogram có trọng số dùng cho bootstrap khớp SciPy tại mọi point estimate và4 bootstrap draws/cohort trên mọi cấu hình được kiểm tra. Sau khi ghi CSV, tính lại tau-c tất cả **{v['csv_tau_cohort_method_pairs_checked']}** cặp cohort-method bằng SciPy đều khớp JSON. Shape/dtype/norm/alignment của cả6 encoder đã đạt.

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
'''
reportpath=BASE/'EmbeddingGemma2_Stage7_Expert_Report_VI.md';reportpath.write_text(report)
(ROOT/'README.md').write_text('''# Stage7 expert validation: COMPLETED

Completed Gemma human continuation and fresh SigLIP2-224 inference in this environment. All1000 images/5664 captions;900 unseen images/5112 captions separately. Protocol and native224/clipped sensitivity addendum were written before expanded Gemma caption scores.

## Offline reproduction

```sh
python -m pip install numpy scipy
OPENBLAS_NUM_THREADS=1 python stage7/analyze_human.py
python stage7/verify_completion.py
```

Requires only packaged vectors, not model weights or images. Analysis is secondary human evaluation, with individual-rating tau-c and image-cluster bootstrap. Full includes100 old development images. Package verification report already checks source image hashes/labels; offline verifier does not newly fetch original bytes.

## Repeat inference

Use Python3.12 with torch2.14.1+cpu and torchvision0.29.1+cpu from the official CPU wheel index; install transformers5.19.0, sentence-transformers6.1.0, pyarrow25.0.1, Pillow, numpy, scipy. Recorded dependency versions are in run metadata/environment.

1. Run stage7/download_required.py and stage7/download_siglip224.py (network required).
2. Run stage7/materialize_expert.py (verifies all image/source bytes and labels).
3. Run stage7/run_human.py gemma --threads 6 and siglip224 --threads 2 sequentially or following declared schedule.

Completed metadata makes inference runner exit; to force a fresh run, use a separate copy and move that copy's completed output directories first. Keep cache/provenance intact. prepare.py references earlier source archives and is retained as provenance, not required offline. Do not interpret continuation timing as controlled model latency.

See Stage7 report for actual results and limits. Model weights, raw images and parquet shards are deliberately excluded from the compact reproducibility archive.
''')
files=[]
for p in ROOT.rglob('*'):
 if not p.is_file() or '__pycache__' in p.parts:continue
 relative=p.relative_to(ROOT)
 if relative.parts[0] in ['data','images','models']:continue
 if p.name in ['package_sha256.json','partial.npz'] or p.name.endswith('_partial_text.npz'):continue
 files.append(p)
files.append(reportpath)
for folder in ['gemma','siglip224']:
 for p in (ROOT/'models'/folder).rglob('*'):
  if p.is_file() and p.suffix in ['.json','.jinja','.txt'] and p.stat().st_size<200000 and not any(part.startswith('.') for part in p.relative_to(ROOT/'models'/folder).parts):files.append(p)
sha={p.relative_to(BASE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)};hashpath=ROOT/'package_sha256.json';hashpath.write_text(json.dumps(sha,indent=2));files.append(hashpath)
archive=BASE/'embeddinggemma2_stage7_expert_validation.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(files):z.write(p,p.relative_to(BASE).as_posix())
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for name,digest in sha.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
print(json.dumps({'report':str(reportpath),'zip_bytes':archive.stat().st_size,'file_count':len(files),'all_sha256_verified':True,'unseen_raw_comparison':c,'unseen_gemma_tau':u['metrics'][G]['kendall_tau_c'],'unseen_siglip224_tau':u['metrics'][R]['kendall_tau_c']},indent=2))
