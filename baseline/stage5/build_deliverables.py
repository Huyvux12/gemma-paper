import ast,base64,hashlib,json,shutil,textwrap,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
SOURCE=BASE/'full_recovered/embeddinggemma2_full_validation/full_validation'
for source,dest in [('results.json','full_sugar_results.json'),('protocol.json','full_source_protocol.json'),('analyze.py','full_source_analyze.py'),('pair_scores.csv','full_sugar_pair_scores.csv')]:
    path=SOURCE/source
    if path.exists():shutil.copyfile(path,ROOT/dest)
r=json.loads((ROOT/'results_560.json').read_text());full=json.loads((ROOT/'full_sugar_results.json').read_text());v=json.loads((ROOT/'independent_verification.json').read_text())
assert v['passed'] and v['scalar_scores_checked']==31360
# Runnable notebook: only the missing SigLIP224 encoder needs to be downloaded/run.
payload={p.name:{'b64':base64.b64encode(p.read_bytes()).decode(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [ROOT/'frozen_manifest.json',ROOT/'frozen_protocol.json',ROOT/'primary_reference.json',ROOT/'run_siglip224.py']}
intro='''# Hoàn tất so sánh chính Gemma vs SigLIP2-224 trên 560 ảnh

Notebook này **chưa được chạy**. Chọn GPU trong Colab rồi Run all. Tải ảnh COCO và model pinned revision; cần mạng cho Hugging Face và COCO. Dữ liệu đầu vào và điểm Gemma đã được nhúng, không phụ thuộc file của phiên ChatGPT.

Gemma được giữ nguyên từ vector float32 đã kiểm chứng. Chỉ SigLIP2-224 được inference mới; không dùng SigLIP2-256 thay thế. 560 ảnh, caption, thứ tự và hash đều được cố định. Notebook xuất McNemar exact và paired bootstrap 10.000 lần của so sánh chính theo công thức CLIPScore đã dùng ở stage 4. Không có kết quả inference hay timing mới cho tới khi chạy thành công.
'''
install='''import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==5.19.0', 'huggingface_hub==1.33.0', 'numpy', 'scipy', 'pillow', 'safetensors', 'sentencepiece', 'accelerate'])
'''
restore="import base64, hashlib, json\nfrom pathlib import Path\nroot=Path('/content/embeddinggemma2_560') if Path('/content').exists() else Path.cwd()/'embeddinggemma2_560'\nroot.mkdir(parents=True, exist_ok=True)\npayload=json.loads("+repr(json.dumps(payload))+ ")\nfor name,item in payload.items():\n    data=base64.b64decode(item['b64'])\n    assert hashlib.sha256(data).hexdigest()==item['sha256']\n    (root/name).write_bytes(data)\nprint('Restored fixed 560 manifest and verified Gemma reference:', root)\n"
run="import torch\nprint('Torch', torch.__version__, 'GPU available:', torch.cuda.is_available())\nsubprocess.check_call([sys.executable, str(root/'run_siglip224.py')])\n"
finish="""import zipfile
result=json.loads((root/'primary_siglip224_results.json').read_text())
print(json.dumps(result, indent=2))
output=root.parent/'siglip224_confirmation_results.zip'
with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in ['primary_siglip224_results.json','frozen_manifest.json','frozen_protocol.json','primary_reference.json','run_siglip224.py','runs/siglip224/embeddings.npz']:
        archive.write(root/name, name)
try:
    from google.colab import files
    files.download(str(output))
except ImportError:
    print(output)
"""
def cell(code):return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'cells':[{'cell_type':'markdown','metadata':{},'source':intro.splitlines(True)},cell(install),cell(restore),cell(run),cell(finish)]}
for i,c in enumerate(nb['cells']):
    c['id']=f'confirmation-{i}'
    if c['cell_type']=='code':ast.parse(''.join(c['source']))
notebook=BASE/'EmbeddingGemma2_560_SigLIP224_Colab.ipynb';notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1))
try:
    import nbformat
    nbformat.validate(nb)
except ImportError:pass
names={'gemma-CLIPLiteral-768-float32':'Gemma, prefix CLIP, 768 f32','gemma-FactChecking-768-float32':'Gemma, FactChecking, 768 f32','clip-CLIPScore-512-float32':'CLIP ViT-B/32','clip_large-CLIPScore-768-float32':'CLIP ViT-L/14','siglip-SigLIP2-768-float32':'SigLIP2 Base patch16-256 (bổ sung)','specs-NativeSPECS-512-float32':'SPECS, trained spec.pt (bổ sung)'}
rows=['| Cấu hình | Đúng / 560 | Accuracy |','|---|---:|---:|']
for k,name in names.items():m=r['metrics'][k];rows.append(f"| {name} | {m['correct']} | {m['accuracy']*100:.2f}% |")
comp=['| Gemma 768 − baseline | Chênh lệch, điểm % | CI 95%, điểm % | McNemar exact |','|---|---:|---|---:|']
for k in list(names)[2:]:
    m=r['comparisons'][k];comp.append(f"| {names[k]} | +{m['delta_pp']:.2f} | [{m['ci95_pp'][0]:.2f}, {m['ci95_pp'][1]:.2f}] | {m['p_exact']:.3g} |")
unseen=full['cohorts']['unseen'];wide=['| Cấu hình | Accuracy, 3.949 cặp / 1.005 ảnh |','|---|---:|']
for k,label in [('gemma-CLIPLiteral-768','Gemma CLIPLiteral 768 f32'),('gemma-CLIPLiteral-256','Gemma CLIPLiteral 256 f32'),('clip-CLIPScore-512','CLIP B/32'),('clip_large-CLIPScore-768','CLIP L/14'),('siglip-SigLIP2-768','SigLIP2-256'),('specs-NativeSPECS-512','SPECS trained')]:
    wide.append(f"| {label} | {100*unseen['metrics'][k]['accuracy']:.2f}% |")
report='''# EmbeddingGemma 2 thay encoder CLIPScore — Stage 5

Ngày 08/10/2026. Báo cáo tiếp nối stage 4, chấm lại vector inference đã lưu trên đúng manifest 560 ảnh và toàn SugarCrepe. **Không có encoder inference mới trong lượt này.**

## Kết quả và trạng thái

Gemma giữ nguyên prefix `A photo depicts `, 768 chiều float32 đạt **89,82%** trên 560 ảnh. Mức tăng so với CLIP L/14 là **15,54 điểm %**, CI 95% **[11,61; 19,64]**. Điều này ủng hộ tiếp tục nghiên cứu khả năng phân biệt caption đúng/sai về thành phần ảnh.

**So sánh chính Gemma vs SigLIP2 patch16-224 vẫn chưa hoàn tất.** Mạng trả 403 khi tải Hugging Face và ảnh COCO. Archive có SigLIP2 patch16-256, đây là checkpoint khác; kết quả của nó là bổ sung và không thay thế phép so sánh chính đã khóa. Notebook Colab đã chuẩn bị để chạy riêng phần thiếu, chưa thực thi hoặc đo hiệu năng.

## Nguồn dữ liệu và kiểm chứng

- Manifest 560 ảnh được giữ nguyên từ stage 4: 80 ảnh mỗi nhóm, 7 nhóm; mỗi ảnh một cặp caption. SHA-256: `a2e9fa5fb7e9866631c0546e5a09a2f9d11044335267712591476572b6104efe`.
- Tất cả 560 ID, ảnh và hai caption khớp archive `embeddinggemma2_full_validation.zip`, bản 31. Không trùng ảnh đã dùng để phát triển cấu hình trong các lượt trước; “unseen” chỉ có nghĩa trong chuỗi thí nghiệm này, không chứng minh ngoài dữ liệu pretraining.
- Archive chứa đầy đủ vector SugarCrepe của 5 encoder. Đã kiểm tra hash 140 file theo bảng kiểm của archive, trích đúng 560 ảnh và chấm lại. Prefix không được chọn lại dựa vào điểm của 560 ảnh.
- Kiểm tra scalar độc lập cả **31.360 điểm** thuộc 28 cấu hình, sai số tối đa **3,55 × 10^-15**; các số đúng và p McNemar khớp. Bridge trên 350 mẫu cũ của Gemma, CLIP B/32 và L/14 khớp toàn bộ phần tử vector và quyết định. Đây là đối chiếu hai archive, không phải inference lặp lại.
- Không suy luận về tốc độ encoder từ timing khác phiên. Không có phép đo latency mới.

## Mẫu 560 ảnh đã khóa

'''+ '\n'.join(rows)+'''

Gemma và CLIP dùng `2.5 × max(cosine, 0)`, native preprocessing từng checkpoint. SigLIP2-256 trong archive được chấm cosine nguyên, SPECS dùng `(cosine + 1)/2` theo protocol nguồn. Caption hòa điểm tính là sai. So sánh SigLIP2-224 đang chờ sẽ giữ công thức CLIPScore đã dùng ở stage 4; bảng SigLIP2-256 không xác nhận protocol đó.

'''+ '\n'.join(comp)+'''

CI bootstrap ghép cặp theo ảnh, 10.000 lần; 560 ảnh là 560 đơn vị độc lập ở mức mẫu. CLIP B/32 và L/14 thuộc nhóm phụ đã khóa: giữ đủ 3 phép so sánh trong hiệu chỉnh Holm, kể cả phép matched-prefix SigLIP2-224 còn thiếu. p đã hiệu chỉnh lần lượt **3,97 × 10^-11** và **7,36 × 10^-13**. SigLIP2-256 và SPECS là so sánh bổ sung chưa hiệu chỉnh; không gọi là xác nhận giả thuyết chính.

SPECS dùng trọng số trained `spec.pt`, không dùng LongCLIP base làm đại diện. SugarCrepe caption ngắn không bao phủ chuyên môn caption dài của SPECS; bảng này không chứng minh ưu thế tổng quát trên mọi loại caption.

## Tập SugarCrepe rộng hơn

Archive có 7.511 cặp caption trên 1.560 ảnh. Cohort loại ảnh đã dùng ở các lượt trước còn **3.949 cặp / 1.005 ảnh**. Đã chạy lại script thống kê nguồn từ vector hoàn chỉnh, ghép bootstrap theo ảnh để tránh coi nhiều caption cùng ảnh là các mẫu độc lập.

'''+ '\n'.join(wide)+'''

Gemma − CLIP L/14 = **13,78 điểm %**, CI95% **[12,35; 15,25]**; Gemma − SigLIP2-256 = **9,50 điểm %**, CI95% **[8,14; 10,87]**. Image-cluster bootstrap 10.000 lần; sign permutation theo ảnh 19.999 lần. Với CLIP B/L, p permutation đã Holm = 0,0001; độ phân giải p permutation thô là 0,00005, không phải p=0.

**560 ảnh là tập con của 1.005 ảnh**, không phải một thí nghiệm độc lập thứ hai; không cộng hai cỡ mẫu. Tập đầy đủ gồm ảnh development chỉ để mô tả: Gemma literal768 đạt 91,84%; không dùng tập này thay cohort unseen. Tập 560 cân bằng nhóm; tập rộng hơn dùng micro-average và phân bố nhóm gốc nên accuracy không trực tiếp so sánh được như một mức cải thiện do tăng dữ liệu.

Protocol nguồn đã nêu margin 2 điểm % cho 256f32 so với 768f32. Trên cohort unseen, chênh lệch −1,39 điểm %, lower bound một phía95% là −1,9545 nên vượt ngưỡng −2 rất sát. Đây là kết quả riêng của protocol nguồn, không áp dụng cho int8, tập 560 hoặc human evaluation.

## Đổi dung lượng lưu vector

| Gemma CLIPLiteral | Accuracy 560 | Byte/vector (payload) |
|---|---:|---:|
| 768 f32 | 89,82% | 3.072 |
| 512 f32 | 88,21% | 2.048 |
| 256 f32 | 87,68% | 1.024 |
| 256 f16 | 87,68% | 512 |
| 256 int8 + scale f32 | 88,21% | 260 |
| 128 f32 | 83,04% | 512 |

Truncate chiều rồi normalize; int8 symmetric theo từng vector, scale = maxabs/127 dạng float32, reconstruct rồi normalize để chấm. 256 int8 giảm **11,82 lần** payload so với 768 f32, không tính metadata/index. Chênh lệch accuracy −1,61 điểm %, CI95% **[−3,22; 0]**. Protocol 560 chỉ cho phân tích mô tả; CI chứa 0 không chứng minh tương đương hay non-inferiority. Int8 đạt cao hơn 256f32 do đổi một số quyết định sau làm tròn, không phải bằng chứng gain tổng quát.

Đây là nén vector sau encoder, không lượng tử hóa trọng số hoặc tăng tốc encoder.

## Đối chứng ghép sai ảnh

Ghép mỗi cặp caption với 559 ảnh còn lại: 313.040 phép ghép mỗi cấu hình. Với Gemma literal768, tỷ lệ cosine ưu tiên caption positive gốc là **49,53%**, so với **89,82%** khi dùng đúng ảnh. Kết quả phù hợp với việc lợi thế cần thông tin từ ảnh. Các phép ghép dùng lại ảnh/caption nên không coi 313.040 là các mẫu độc lập để tính p.

Đây là ưu tiên caption positive gốc trên ảnh không liên quan, không phải “accuracy” của caption với ảnh sai. SigLIP2-256 có 68,55% hòa nếu ép công thức max(cos,0) lên ghép sai; dùng cosine nguyên để tránh diễn giải clipping thành khả năng grounding. SPECS dùng affine cosine native; không gán điểm accuracy 0 cho SPECS từ công thức CLIP không tương thích.

## Phần còn thiếu và kết luận nghiên cứu

Flickr8k-Expert mở rộng của Gemma mới có checkpoint ảnh 936/1.000, chưa có vector caption hoàn chỉnh. **Không có kết quả human correlation mở rộng mới.** Tập expert 100 ảnh của stage 4 vẫn là thăm dò.

Tiếp tục hướng compositional discrimination và tradeoff lưu vector là có cơ sở. Chưa kết luận thay thế chung CLIPScore cho mọi đánh giá caption, chưa khẳng định đóng góp đủ cho paper chỉ từ thay backbone. Cần hoàn tất SigLIP2-224 theo protocol gốc, rồi human correlation có bootstrap theo ảnh. Nếu phép chính không có gain, phải thu hẹp claim theo điều kiện go/no-go đã khóa.

## Mã và dữ liệu kèm theo

`embeddinggemma2_stage5_confirmation.zip` chứa vector subset 560, manifest/protocol, CSV điểm, JSON kết quả, log và mã chấm/kiểm tra; thống kê tập rộng hơn và protocol nguồn cũng được giữ. Notebook `EmbeddingGemma2_560_SigLIP224_Colab.ipynb` có đầu vào nhúng và runner model224, chưa thực thi.

Các revision nằm trong `results_560.json` và metadata từng run. Tham khảo checkpoint: https://huggingface.co/google/embeddinggemma-2 ; https://huggingface.co/google/siglip2-base-patch16-224 ; https://huggingface.co/google/siglip2-base-patch16-256 ; https://huggingface.co/XiaoFu666/SPECS .
'''
(BASE/'EmbeddingGemma2_Stage5_Report_VI.md').write_text(report)
(ROOT/'README.md').write_text('''# Stage 5 reproducibility

No new encoder inference was performed in this ChatGPT turn. Scores are computed from recovered complete SugarCrepe float32 inference vectors. SigLIP2-256 is supplemental; the originally declared SigLIP2-224 primary contrast remains pending. Human Gemma inference is incomplete.

## Offline 560 score reproduction

Unzip preserving the stage5 directory. Requires Python, numpy and scipy.

```sh
python stage5/analyze_560.py
python stage5/verify.py --scores-only
```

The second command independently recalculates all 31,360 scores and paired exact p-values. It saves `independent_scores_verification.json`. The packaged `independent_verification.json` additionally records the completed cross-archive 350-record bridge and disjointness audit. Running that broader bridge (`verify.py` without --scores-only), `prepare.py`, or `recover_vectors.py` requires the original stage4/replacement and full-validation archives in their original extracted paths. Those source archives are not duplicated here.

## Wider cohort

`full_sugar_results.json`, `full_sugar_pair_scores.csv`, `full_source_protocol.json`, and `full_source_analyze.py` preserve the wider 7,511-pair analysis. Recomputing that wider analysis requires `embeddinggemma2_full_validation.zip` version31, with its complete vector files; the subset560 vectors cannot recreate 7,511 scores. The 560 sample is contained in the 1,005-image unseen cohort, not an independent extra dataset.

## Missing original primary contrast

Open the self-contained `EmbeddingGemma2_560_SigLIP224_Colab.ipynb` in Colab, select GPU, Run all. The notebook is syntax/schema checked but NOT executed in this environment. It downloads exact image bytes and pinned SigLIP2-224 weights, verifies hashes, runs float32, then computes the fixed primary comparison using CLIPScore clipping as declared in stage4. It needs network access to Hugging Face/COCO; image or model failures stop rather than substitute data. New SigLIP224 inference uses saved verified Gemma scores; it does not regenerate Gemma. No runtime or primary result is claimed in this package.

Compression is embedding storage, not encoder quantization. Do not use cross-session cached timings for latency comparisons.
''')
print(json.dumps({'notebook_bytes':notebook.stat().st_size,'report_chars':len(report),'verification_passed':v['passed'],'full_csv_copied':(ROOT/'full_sugar_pair_scores.csv').exists()},indent=2))
