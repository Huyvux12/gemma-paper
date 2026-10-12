"""Generate the Vietnamese continuation report from verified measured outputs."""
import json
from pathlib import Path
from analyze import G,G256,FC,B,L,S,D
ROOT=Path(__file__).resolve().parent

def table(head,rows):return '| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)
def percent(v):return f'{100*v:.2f}%'
def interval(v):return f'[{v[0]:+.2f}, {v[1]:+.2f}]'

def main():
    s=json.loads((ROOT/'results.json').read_text());h=json.loads((ROOT/'human_results.json').read_text());p=json.loads((ROOT/'protocol.json').read_text());v=json.loads((ROOT/'independent_verification.json').read_text());assert v['passed']
    sf=s['cohorts']['full'];sn=s['cohorts']['unseen'];hf=h['cohorts']['full'];hn=h['cohorts']['unseen']
    comp=sn['comparisons'][G256+'_vs_'+G];primary=[sn['comparisons'][G+'_vs_'+b] for b in [B,L]]
    supported=all(x['delta_pp']>0 and x['p_holm_primary']<.05 for x in primary)
    verdict=('Tập ảnh chưa dùng trước đây xác nhận lợi ích của thay encoder nguyên bản so với cả CLIP B/32 và L/14.' if supported else
      'Tập ảnh chưa dùng trước đây chưa xác nhận lợi ích vượt cả hai CLIP baseline theo hai phép kiểm định đã khóa. Giữ toàn bộ kết quả, không chọn lại mẫu hoặc prefix.')
    human=hn['comparisons'][G+'_vs_'+L]
    hv=('CI so với L/14 trên 900 ảnh mới nằm hoàn toàn trên 0; đây là bằng chứng phụ về cải thiện tương quan chuyên gia.' if human['ci95'][0]>0 else
      ('CI so với L/14 trên 900 ảnh mới nằm hoàn toàn dưới 0; tương quan chuyên gia thấp hơn.' if human['ci95'][1]<0 else
       'CI so với L/14 trên 900 ảnh mới còn chứa 0; chưa có bằng chứng rõ về cải thiện hoặc tương đương với đánh giá của chuyên gia.'))
    cv=('256d đạt tiêu chí non-inferiority đã khóa: cận dưới một phía 95% lớn hơn −2 điểm %.' if comp['noninferiority_passed'] else
      '256d không đạt tiêu chí non-inferiority đã khóa: chưa thể nói mức giảm chất lượng dưới 2 điểm %. Không nới biên sau khi xem kết quả.')
    keys=[B,L,S,D,G,'gemma-CLIPLiteral-512',G256,'gemma-CLIPLiteral-128',FC,'gemma-FactChecking-512','gemma-FactChecking-256','gemma-FactChecking-128']
    labels=['CLIPScore B/32','CLIPScore L/14','SPECS, native','SigLIP 2 Base, cosine','Gemma literal 768d','Gemma literal 512d','Gemma literal 256d','Gemma literal 128d',
      'Gemma FactChecking 768d','Gemma FactChecking 512d','Gemma FactChecking 256d','Gemma FactChecking 128d'];label=dict(zip(keys,labels))
    mainrows=[[label[k],percent(sf['metrics'][k]['accuracy']),percent(sn['metrics'][k]['accuracy']),percent(sf['metrics'][k]['macro_accuracy']),
      f'{hf["metrics"][k]["kendall_tau_c"]:.4f}',f'{hn["metrics"][k]["kendall_tau_c"]:.4f}'] for k in keys]
    primaryrows=[[label[G]+' − '+label[b],f'{sn["comparisons"][G+"_vs_"+b]["delta_pp"]:+.2f}',interval(sn['comparisons'][G+'_vs_'+b]['ci95_pp']),
      f'{sn["comparisons"][G+"_vs_"+b]["p_holm_primary"]:.5g}'] for b in [B,L]]
    secondaryrows=[[label[a]+' − '+label[b],f'{sn["comparisons"][a+"_vs_"+b]["delta_pp"]:+.2f}',interval(sn['comparisons'][a+'_vs_'+b]['ci95_pp'])]
      for a,b in [(G,S),(G,D),(FC,L),(G256,L)]]
    humanrows=[[label[G]+' − '+label[b],f'{hn["comparisons"][G+"_vs_"+b]["delta_tau_c"]:+.4f}',
      '['+', '.join(f'{x:+.4f}' for x in hn['comparisons'][G+'_vs_'+b]['ci95'])+']'] for b in [B,L,S,D]]
    cats=sorted(sf['metrics'][G]['category_accuracy']);counts={c:sum(r['category']==c for r in json.loads((ROOT/'manifest.json').read_text())['records']) for c in cats}
    dims=[[str(d)+'d',str(d*4),f'{768/d:g}×',percent(sn['metrics'][f'gemma-CLIPLiteral-{d}']['accuracy']),
      f'{hn["metrics"][f"gemma-CLIPLiteral-{d}"]["kendall_tau_c"]:.4f}'] for d in [768,512,256,128]]
    audit=json.loads((ROOT/'token_length_audit.json').read_text());sa=json.loads((ROOT/'siglip_token_length_audit.json').read_text())
    control=json.loads((ROOT/'negative_control.json').read_text())
    lines=['# CLIPScore thay encoder — kiểm chứng mở rộng trên hai benchmark đầy đủ',
      'Ngày: 08/10/2026. Tiếp nối thử nghiệm 350 ảnh; inference thật trên CPU float32, không fine-tune, không API trả phí.',
      '## 1. Kết luận đã kiểm chứng',verdict,hv,cv,
      '**Hướng nghiên cứu:** đánh giá khả năng thay encoder trong CLIPScore, độ bền trước lỗi chi tiết, liên hệ với đánh giá chuyên gia, và chi phí nén vector. Không đặt tên metric mới để xem việc đổi backbone là thuật toán mới.',
      '## 2. Lượt này khác báo cáo trước như thế nào?',
      table(['Nội dung','Lượt trước','Lượt mở rộng'],[['SugarCrepe','350 cặp cân bằng, 350 ảnh','7.511 cặp, 1.560 ảnh; 3.949 cặp / 1.005 ảnh mới để xác nhận'],
        ['Flickr8k-Expert','100 ảnh, 552 caption','1.000 ảnh, 5.664 caption; 900 ảnh mới được báo cáo riêng'],
        ['Đối chứng','CLIP B/32 và L/14','Thêm SPECS đã huấn luyện và SigLIP 2 Base hiện đại'],
        ['Thống kê','Mỗi cặp là một ảnh riêng','Resample theo ảnh vì nhiều cặp cùng dùng một ảnh'],
        ['Nén vector','Chỉ mô tả point accuracy','Khóa biên non-inferiority 2 điểm % cho 256d trước kết quả mới']]),
      'Toàn bộ benchmark có chứa dữ liệu phát triển của các lượt trước. Vì vậy bảng full là kết quả mô tả toàn tập; **cohort chưa dùng trước đây** mới dùng để xác nhận lựa chọn đã đưa ra. Không gọi toàn bộ 7.511 cặp là test độc lập mới. Các bộ dữ liệu có trước mô hình; chưa loại được pretraining contamination.',
      '## 3. Phương pháp và đối chứng',
      'Gemma và hai CLIP checkpoint dùng công thức `2.5 × max(cosine, 0)`. Thay nguyên bản giữ `A photo depicts ` cho caption. FactChecking là cấu hình phụ, khác ở cả encoder lẫn prefix. Ảnh không nhận prefix. Mọi chiều cắt được L2-normalize lại.',
      'SPECS dùng **checkpoint đã huấn luyện `spec.pt`** từ repo tác giả, tokenizer 248 token, caption thô và score native `max((cosine+1)/2, 0)`. Không thay checkpoint nền LongCLIP rồi gọi đó là SPECS. SigLIP 2 Base dùng caption thô, padding native 64 token và cosine không chặn; thứ tự tương đương logits được scale của encoder. Không so sánh mean raw score giữa các phương pháp có thang khác nhau.',
      'Cấu hình Gemma đã chọn từ kết quả trước; không tối ưu lại trên cohort mới. Hai kiểm định chính là literal 768d so B/32 và L/14 trên 1.005 ảnh mới. SPECS và SigLIP 2 là đối chứng phụ. SigLIP 2 được thêm trước khi có score caption của Gemma trên tập đầy đủ, có ghi thời gian và lý do trong protocol.',
      'BF16 được kiểm tra trên dữ liệu cũ, nhưng đổi 1/32 lựa chọn của L/14. Do đó **mọi lượt đầy đủ đều dùng float32**. Chỉ tái sử dụng vector float32 cũ khi ảnh khớp SHA-256, caption khớp chuỗi, prefix và revision khớp. Không trộn vector BF16 vào kết quả chính.',
      '## 4. Bảng kết quả đầy đủ',
      table(['Phương pháp','Sugar full micro','Sugar ảnh mới micro','Sugar full macro 7 nhóm','Expert full tau-c','Expert ảnh mới tau-c'],mainrows),
      'Micro là tỉ lệ đúng trên mọi cặp; macro là trung bình đều bảy loại lỗi. Tập trước lấy 50 cặp mỗi nhóm, trong khi full có số lượng nhóm khác nhau. Không so trực tiếp mức tăng full với điểm của mẫu cân bằng rồi kết luận model tốt hơn theo thời gian. Tau-c là tương quan thứ hạng, không phải phần trăm caption đúng.',
      '## 5. Xác nhận trên ảnh mới và kiểm định',
      f'Có {sn["n_pairs"]:,} cặp từ {sn["n_images"]:,} ảnh chưa xuất hiện trong toàn bộ các lượt trước. Đúng khi score caption đúng lớn hơn caption sai; hòa tính sai. Bootstrap ghép cặp theo cụm ảnh 10.000 lần. Kiểm định hoán vị đổi dấu tổng chênh lệch trong mỗi ảnh 19.999 lần; p Monte Carlo có hiệu chỉnh +1. Holm trên đúng hai phép kiểm định chính. CI là CI 95% percentile từng so sánh, chưa điều chỉnh đồng thời.',
      table(['Kiểm định chính','Delta điểm %','CI 95%, điểm %','p Holm'],primaryrows),
      '### So sánh phụ trên cùng cohort',table(['So sánh','Delta điểm %','CI 95%, điểm %'],secondaryrows),
      'Các CI phụ là khoảng tin cậy từng so sánh; chưa kiểm soát đồng thời toàn bộ nhóm. Không dùng kết quả phụ để thay các giả thuyết chính hoặc chọn lại prefix. Thống kê theo ảnh giảm việc đánh giá quá chắc chắn khi cùng một ảnh xuất hiện trong nhiều loại lỗi.',
      '### Đối chứng ghép nhầm ảnh',
      'Giữ hai caption, nhưng ghép với ảnh khác trong cohort mới. Dùng 20 hoán vị dạng chu trình, không có ảnh nào ghép lại chính nó; tất cả vector đã có, không chạy thêm encoder. Tỉ lệ ưu tiên caption positive của ảnh gốc trong tình huống này **không phải accuracy** đối với ảnh ghép nhầm. Nó chẩn đoán ảnh hưởng của prior ngôn ngữ và mức nhạy với ảnh đầu vào; không tự chứng minh reasoning hay loại bỏ contamination.',
      table(['Phương pháp','Accuracy với ảnh đúng','Ưu tiên positive khi ghép nhầm, trung bình 20 lần','Chênh lệch điểm %'],[[label[k],percent(control['methods'][k]['actual_accuracy']),percent(control['methods'][k]['wrong_image_positive_preference_mean']),f'{control["methods"][k]["actual_minus_wrong_image_mean_pp"]:+.2f}'] for k in [B,L,S,D,G,FC]]),
      'Range giữa các hoán vị và mapping từng ảnh được lưu trong `negative_control.json` và `negative_control_maps.json`; range này không phải CI 95%. Đối chứng được khóa trước khi có score caption Gemma mới.',
      '## 6. Liên hệ với điểm chuyên gia',
      f'Tập full có {hf["n_images"]:,} ảnh, {hf["n_captions"]:,} caption và {hf["n_individual_ratings"]:,} rating cá nhân. Cohort mới có {hn["n_images"]:,} ảnh. Kendall tau-c dùng rating cá nhân theo evaluator CLIPScore; bootstrap ghép cặp theo ảnh 2.000 lần. Không coi mọi rating hoặc mọi caption là ảnh độc lập.',
      table(['Gemma literal 768d so với','Delta tau-c trên ảnh mới','CI 95%'],humanrows),
      'Chẩn đoán thêm: so từng cặp caption **cùng một ảnh** có điểm chuyên gia trung bình khác nhau. Cách này giảm ảnh hưởng của chênh lệch mức cosine giữa các ảnh. Chỉ dùng cặp có rating trung bình khác nhau; score hòa tính sai.',
      table(['Phương pháp','Xếp hạng đúng trong cùng ảnh mới'],[[label[k],percent(hn['within_image_ranking']['accuracy'][k])] for k in [B,L,S,D,G,G256,FC]]),
      f'Chẩn đoán này gồm {hn["within_image_ranking"]["n_pairs"]:,} cặp từ {hn["within_image_ranking"]["n_images"]:,} ảnh mới. CI các chênh lệch được lưu đầy đủ trong `human_results.json`. Đây là kết quả phụ, không thay thế kiểm định chính.',
      '## 7. Nén vector: chất lượng đổi lấy dung lượng',
      table(['Literal Gemma','Bytes/vector float32','Giảm dung lượng','Accuracy ảnh mới','Expert tau-c ảnh mới'],dims),
      f'256d − 768d: **{comp["delta_pp"]:+.2f} điểm %**, CI hai phía 95% {interval(comp["ci95_pp"])}. Cận dưới một phía 95% **{comp["one_sided_95_lower_pp"]:+.2f} điểm %**; biên cho phép đã khóa là **−2 điểm %**. {cv}',
      'Dung lượng ở bảng là vector thô, chưa gồm metadata hoặc index. Cắt 768 xuống 256 giảm ba lần lưu trữ vector; **không giảm thời gian chạy encoder ảnh**. Tương quan expert của vector nén là phân tích phụ; không xem một point estimate gần nhau là chứng minh tương đương.',
      '## 8. Độ bền theo loại lỗi trên toàn SugarCrepe',
      table(['Loại lỗi','Số cặp','B/32','L/14','SPECS','SigLIP 2','Gemma 768d','Gemma 256d'],[[c,counts[c]]+[percent(sf['metrics'][k]['category_accuracy'][c]) for k in [B,L,S,D,G,G256]] for c in cats]),
      '`add`: thêm chi tiết; `replace`: thay chi tiết; `swap`: hoán đổi vai trò hoặc thuộc tính; `obj`: vật thể; `att`: thuộc tính; `rel`: quan hệ. Bảng nhóm là mô tả, chưa kiểm định đồng thời bảy nhóm. Xem `error_gallery.html` để đọc các ví dụ bất đồng trên ảnh mới; gallery không dùng để ước lượng tần suất lỗi.',
      '## 9. Kiểm chứng và chi phí chạy',
      f'Toàn bộ **{v["human"]["original_clipscore_published_image_checksums_matched"]:,} ảnh Flickr8k-Expert** còn khớp MD5 ảnh gốc do tác giả CLIPScore công bố; không chỉ khớp nguồn parquet hiện tại. MD5 ở đây dùng để đối chiếu tính tái lập byte của bộ ảnh, không phải tuyên bố an toàn mật mã.',
      f'- Đối chiếu trọng số lớn với SHA-256 chính thức và giữ revision cụ thể.\n- Toàn bộ {v["sugar"]["media_hashes_verified"]+v["human"]["media_hashes_verified"]:,} ảnh được kiểm tra SHA-256. COCO mirror khớp byte với tất cả 350 ảnh cũ và các ảnh đã tải từ nguồn gốc.\n- Mọi rating được chọn khớp annotation trong repo EXPERT của tác giả.\n- Tái tính độc lập **{v["sugar"]["scalar_scores_recomputed"]+v["human"]["scalar_scores_recomputed"]:,} điểm** bằng arithmetic scalar float64; kiểm tra mọi accuracy/tau-c/xếp hạng cùng ảnh của cả hai cohort. Sai số tối đa {max(v["sugar"]["maximum_abs_error"],v["human"]["maximum_abs_error"]):.3g}.\n- CLIP không cắt caption: SugarCrepe tối đa {audit["sugar"]["max_tokens"]} token, Expert tối đa {audit["human"]["max_tokens"]}. SigLIP 2: {sa["sugar"]["truncated_at_64"]} / {sa["human"]["truncated_at_64"]} caption bị cắt ở 64 token.\n- Các kết quả tie được lưu trong `results.json`.',
      table(['Encoder, SugarCrepe','Tham số nạp','Threads','Ảnh mới encode','Ảnh tái dùng','Encode ảnh (giây)'],[[k,f'{m["parameters"]:,}',m['threads'],m['reuse']['images_encoded_this_invocation'],m['reuse']['images_reused_this_invocation'],f'{m["timings"]["media_seconds_this_invocation"]:.1f}'] for k,m in s['runtime'].items()]),
      'Gemma chạy 6 threads cùng lúc với chuỗi baseline 2 threads; SigLIP 2 đợi chuỗi đó xong. Thời gian ở bảng gồm pipeline/checkpoint và có reuse; không phải so sánh latency được kiểm soát. Không suy ra real-time, hiệu năng GPU hoặc điện thoại từ lượt CPU này.',
      '## 10. Định vị đóng góp và bước tiếp theo',
      'Tên nghiên cứu tạm: **Encoder Substitution in CLIPScore: Compositional Robustness, Human Alignment, and Vector Compression**.',
      'Phần có thể đóng góp là phân tích thực nghiệm có đối chứng: thay encoder giữ scoring rule; xác nhận trên ảnh chưa dùng; kiểm tra xem tăng compositional accuracy có đi cùng tăng human alignment; đo mất chất lượng khi nén với biên đã khóa. Đây là khả năng đóng góp, chưa phải tuyên bố tính mới hoặc bảo đảm chấp nhận paper. Model card đã có benchmark đa phương thức và nén vector; chỉ lặp lại các con số đó không đủ.',
      'Đã rà nguồn CLIPScore, SugarCrepe, SPECS, SigLIP 2 và model card Google. Tìm kiếm tên EmbeddingGemma 2 với CLIPScore/caption evaluation chưa cung cấp bằng chứng về một bản tái lập trực tiếp giống giao thức này; kết quả tìm kiếm trống không chứng minh chưa ai làm.',
      'Trước khi gửi paper: thêm một benchmark caption dài hoặc caption sinh từ VLM có nhãn chuyên gia, thêm domain hoặc ngôn ngữ khác; giữ một validation riêng nếu cần học ngưỡng/calibration; mô tả các lỗi phổ biến từ gallery và kiểm tra annotation mơ hồ. Không chọn benchmark hoặc split theo điểm thuận lợi. Chưa cần huấn luyện mô hình mới để thực hiện hướng này.',
      '## 11. Hạn chế và cách dùng mã',
      'Chỉ một Gemma checkpoint và một variant SigLIP 2, hai CLIP checkpoint và SPECS; native preprocessing/model size khác nhau nên không tách riêng tác động của kiến trúc khỏi dữ liệu huấn luyện. Bộ dữ liệu có trước model, chưa loại được contamination. Chưa test tiếng Việt, caption dài, audio/video hay universal hallucination detection. Không cộng các kết quả phụ thành tuyên bố thắng mọi bài toán hoặc SOTA.',
      'Adapter giữ prefix CLIP gốc làm mặc định. Điểm là tương thích ảnh–caption, không phải xác suất sự thật. Nếu muốn lọc tự động, cần hiệu chỉnh ngưỡng bằng validation riêng; không tái dùng ngưỡng CLIP cũ theo cùng giá trị số. Xem `README.md`, `gemma_clipscore.py`, CSV từng caption và vectors trong archive.',
      '## Tài liệu gốc',
      '- EmbeddingGemma 2: https://huggingface.co/google/embeddinggemma-2\n- CLIPScore: https://aclanthology.org/2021.emnlp-main.595/ ; https://github.com/jmhessel/clipscore\n- SugarCrepe: https://arxiv.org/abs/2306.14610 ; https://github.com/RAIVNLab/sugar-crepe\n- SPECS: https://arxiv.org/abs/2509.03897 ; https://github.com/mbzuai-nlp/SPECS ; https://huggingface.co/XiaoFu666/SPECS\n- SigLIP 2: https://arxiv.org/abs/2502.14786 ; https://huggingface.co/google/siglip2-base-patch16-256\n- Expert annotations: https://github.com/hjkim811/EXPERT ; https://huggingface.co/datasets/yuwd/Flickr8k-HumanEval\n- Các revision và mã băm cụ thể nằm trong protocol, checkpoint receipts và `source_revisions.json`.']
    text='\n\n'.join(lines)+'\n';(ROOT/'REPORT_VI.md').write_text(text);(ROOT.parent/'EmbeddingGemma2_Full_Validation_Report_VI.md').write_text(text)
    print('report written',flush=True)

if __name__=='__main__':main()
