"""Write the Vietnamese report from measured outputs, with honest scope."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)

def main():
    s=json.loads((ROOT/'results.json').read_text());h=json.loads((ROOT/'human_results.json').read_text());v=json.loads((ROOT/'independent_verification.json').read_text())
    names=['clip-CLIPScore-512','clip_large-CLIPScore-768','gemma-CLIPLiteral-768','gemma-FactChecking-768','gemma-FactChecking-256','gemma-FactChecking-128','gemma-SentenceSimilarity-768']
    labels=['CLIPScore ViT-B/32','CLIPScore ViT-L/14','Gemma 768d, prefix CLIP gốc','Gemma 768d, FactChecking','Gemma 256d, FactChecking','Gemma 128d, FactChecking','Gemma 768d, SentenceSimilarity (phụ)']
    p=s['primary']['FactChecking768_vs_L14'];positive=all(x['delta_pp']>0 and x['p_holm']<.05 for x in s['primary'].values())
    decision=('Kết quả ủng hộ tiếp tục thay encoder của CLIPScore bằng EmbeddingGemma 2: cấu hình chính thắng cả baseline mạnh ViT-L/14 trên tập compositional mới.' if positive else
      'Thay encoder đã thực hiện được, nhưng chưa có bằng chứng thống kê đủ mạnh để nói cấu hình chính tốt hơn CLIP ViT-L/14 trên tập compositional mới. Không chọn lại test hoặc tuning prefix để tìm kết quả đẹp.')
    decision+=' Trên Flickr8k-Expert, mức tương quan của Gemma cao hơn về point estimate nhưng các CI so sánh hiện còn chứa 0; chưa có bằng chứng đủ rõ về cải thiện đánh giá của con người, và cũng chưa chứng minh tương đương. Không gộp hai loại kết quả thành một kết luận thắng mọi bài toán.'
    lines=['# Thay encoder của CLIPScore bằng EmbeddingGemma 2 — kiểm chứng thực nghiệm',
      '\nNgày: 08/10/2026. Inference thật trên CPU; không fine-tune; không dùng API chấm điểm.\n',
      '## 1. Kết luận và hướng nên theo\n',decision,
      '\nHướng cụ thể: **đánh giá caption không cần caption tham chiếu bằng cách thay encoder trong CLIPScore**, đồng thời đo độ bền trước lỗi chi tiết và chi phí lưu vector. Điểm nghiên cứu là kiểm chứng khả năng thay thế trong một phương pháp có sẵn; chưa tuyên bố công thức, kiến trúc hay thuật toán mới.\n',
      '## 2. Đã rà những chỗ có thể thay\n',
      table(['Phương pháp có sẵn','Thành phần có thể thay','Quyết định trong lượt này'],[
        ['CLIPScore','CLIP image/text encoder','Đã viết adapter; đã chạy SugarCrepe và Flickr8k-Expert'],
        ['CLAP cosine / CLAPScore-style','CLAP audio/text encoder','Chưa chọn làm hướng chính: kết quả vòng trước Gemma 70% < CLAP 81,5% trên 200 cặp mới'],
        ['BERTopic','SentenceTransformer hoặc backend đa phương thức','Khả thi theo API có sẵn; chưa benchmark trong lượt này'],
        ['RAG / semantic retrieval','Bộ tạo vector của retriever','Khả thi; chưa chứng minh lợi ích với benchmark RAG trong lượt này'],
        ['SPECS / PAC-S','Backbone và quy trình học metric','Chỉ đổi backbone không tái lập phương pháp đã huấn luyện; chưa chạy checkpoint này']]),
      '\nCLIPScore được chọn vì có mã nguồn, quy tắc chấm đơn giản và dữ liệu đánh giá có sẵn; đồng thời vòng trước đã cho tín hiệu tốt trên ảnh. Giữ audio làm đối chứng âm, không suy diễn rằng model đa phương thức sẽ thắng mọi specialist.\n',
      '## 3. Chính xác đã thay gì?\n',
      'CLIPScore gốc: ảnh → CLIP image vector; caption → CLIP text vector; score = `2.5 × max(cosine, 0)`. Adapter thay hai vector bằng Gemma, giữ công thức. Không đưa caption, nhãn hay điểm của con người vào image encoder.\n',
      '- **Thay nguyên bản:** giữ prefix `A photo depicts `, chỉ đổi encoder.\n- **Cấu hình phù hợp Gemma:** prefix `FactChecking`, đã chọn từ vòng trước; đây là đổi encoder **và** prefix, không quy mọi cải thiện cho encoder.\n- Prefix `None`, `SearchQuery`, `SentenceSimilarity`, các chiều 128/256/512 được báo cáo đầy đủ như phân tích phụ; không chọn theo test để thay giả thuyết chính. SentenceSimilarity được bổ sung khi rà lại vector vòng trước (92,14% trên 140 ảnh cũ), trước khi có bất kỳ score caption SugarCrepe mới hoặc Gemma Flickr8k mới; addendum có timestamp trong protocol.\n- Gemma chỉ nạp text+vision; bỏ audio encoder. Cắt vector đầu rồi L2-normalize lại. Tất cả dùng float32.\n',
      '## 4. Giao thức đã khóa trước khi chấm test\n',
      '**SugarCrepe:** 350 cặp / 350 ảnh độc lập, 50 mỗi loại lỗi, seed 20261008. Không trùng bất cứ ảnh nào trong 210 cặp của hai vòng trước. Chọn đúng khi score caption đúng lớn hơn caption sai; hòa tính sai. Đây là mẫu cân bằng theo nhóm, không phải kết quả toàn bộ 7.511 cặp benchmark.\n',
      f'**Flickr8k-Expert:** 100 ảnh lấy ngẫu nhiên, toàn bộ {h["n_caption_pairs"]} cặp ảnh–caption không nằm trong caption tham chiếu, {h["n_individual_ratings"]} điểm cá nhân của chuyên gia. Kendall tau-c theo quy trình CLIPScore; bootstrap ghép cặp theo ảnh 2.000 lần. Đây là benchmark phụ, không phải tái lập toàn bộ Flickr8k-Expert.\n',
      '**Hai phép so sánh chính:** Gemma FactChecking 768d so B/32 và L/14; bootstrap ghép cặp theo ảnh 10.000 lần; kiểm định McNemar exact dạng binomial; Holm trên hai phép so sánh. CI 95% là percentile CI chưa điều chỉnh đồng thời. Không có tối ưu tham số trên test.\n',
      '## 5. Kết quả mới: accuracy và tương quan với con người\n',
      table(['Hệ thống','Đúng / 350','Accuracy','Kendall tau-c, 100 ảnh'],[[label,f'{s["metrics"][name]["correct"]}/350',f'{100*s["metrics"][name]["accuracy"]:.2f}%',f'{h["metrics"][name]["kendall_tau_c"]:.4f}'] for name,label in zip(names,labels)]),
      '\nTau-c cao hơn thể hiện xếp hạng gần điểm của chuyên gia hơn; không phải phần trăm caption đúng. Điểm cosine tuyệt đối giữa các model không có cùng thang đo nên không dùng mean score để đánh giá model nào tốt hơn.\n',
      '### So sánh chính trên SugarCrepe\n',
      table(['So sánh','Chênh lệch accuracy','CI 95%, điểm %','p Holm','Gemma đúng riêng / baseline đúng riêng'],[[name,f'{x["delta_pp"]:+.2f} điểm %',f'[{x["ci95_pp"][0]:+.2f}, {x["ci95_pp"][1]:+.2f}]',f'{x["p_holm"]:.6g}',f'{x["a_only_correct"]} / {x["b_only_correct"]}'] for name,x in s['primary'].items()]),
      '\n### Thay nguyên bản và các cấu hình phụ\n',
      table(['So sánh phụ','Delta điểm %','CI 95%','p Holm trong 5 so sánh phụ'],[[name,f'{x["delta_pp"]:+.2f}',str([round(v,2) for v in x['ci95_pp']]),f'{x["p_holm"]:.6g}'] for name,x in s['secondary_holm'].items()]),
      '\nThay nguyên bản và nén FactChecking là phân tích phụ trong protocol. None và SearchQuery là phân tích thăm dò. Nhóm năm phép so sánh này được điều chỉnh Holm bổ sung sau khi có kết quả; không xem đây là nhóm giả thuyết chính đã đăng ký trước. Không thay hai kiểm định chính bằng cấu hình có kết quả cao nhất.\n',
      '\n### So sánh phụ với chuyên gia\n',
      table(['So sánh','Delta tau-c','CI 95% theo ảnh'],[[name,f'{x["delta_tau_c"]:+.4f}',f'[{x["ci95"][0]:+.4f}, {x["ci95"][1]:+.4f}]'] for name,x in h['comparisons_exploratory'].items()]),
      '\nCác kết quả tương quan là phân tích phụ. Không biến p-value hoặc CI theo hàng caption thành bằng chứng có 1.656 mẫu ảnh độc lập.\n',
      '### Chẩn đoán: xếp hạng các caption của cùng một ảnh\n',
      f'Lấy mọi cặp caption cùng ảnh có điểm chuyên gia trung bình khác nhau: {h["within_image_ranking_exploratory"]["n_pairs"]} cặp từ {h["within_image_ranking_exploratory"]["n_images_with_unequal_rating_pairs"]} ảnh. Phép này giảm ảnh hưởng của khác biệt mức cosine giữa các ảnh. Đây là phân tích phụ đã bổ sung trước khi có điểm caption Gemma mới.\n',
      table(['Hệ thống','Chọn đúng caption được chuyên gia chấm cao hơn'],[[label,f'{100*h["within_image_ranking_exploratory"]["accuracy"][name]:.2f}%'] for name,label in zip(names,labels)]),
      table(['So sánh','Delta điểm %','CI 95% theo ảnh'],[[name,f'{x["delta_pp"]:+.2f}',str([round(v,2) for v in x['ci95_pp']])] for name,x in h['within_image_ranking_exploratory']['comparisons'].items()]),
      '## 6. Phân tích loại lỗi trên 350 ảnh mới\n',
      table(['Loại lỗi','CLIP B/32','CLIP L/14','Gemma 768d, CLIP prefix','Gemma 768d FC','Gemma 256d FC'],[[cat]+[f'{100*s["metrics"][name]["category_accuracy"][cat]:.1f}%' for name in [names[0],names[1],names[2],names[3],names[4]]] for cat in sorted(s['metrics'][names[0]]['category_accuracy'])]),
      '\n`add`: thêm chi tiết không có; `replace`: thay chi tiết; `swap`: hoán đổi vai trò/thuộc tính; `obj`: vật thể; `att`: thuộc tính; `rel`: quan hệ. Mỗi ô chỉ có 50 cặp, nên phân tích từng nhóm mang tính mô tả.\n',
      '## 7. Nén vector và chi phí thực đo\n',
      table(['Gemma FactChecking','Bytes / vector float32','Accuracy','Delta so 768d','CI 95%, điểm %'],[[str(dim)+'d',str(dim*4),f'{100*s["metrics"][f"gemma-FactChecking-{dim}"]["accuracy"]:.2f}%',f'{s["compression_descriptive"][str(dim)]["delta_pp"]:+.2f}',str([round(v,2) for v in s['compression_descriptive'][str(dim)]['ci95_pp']])] for dim in [128,256,512]]+[['768d','3072',f'{100*s["metrics"][names[3]]["accuracy"]:.2f}%','0','—']]),
      '\n256d giảm dung lượng vector thô 3 lần so 768d và 2 lần so CLIP B/32 512d. Không giảm chi phí chạy encoder. Không gọi cấu hình 256d “tương đương” chỉ vì point accuracy gần nhau; chưa khóa giao thức non-inferiority.\n',
      table(['Encoder','Tham số được nạp','Encode 350 ảnh (s)','Peak RSS tiến trình (MiB)'],[[kind,f'{x["parameters"]:,}',f'{x["timings"]["media_seconds"]:.1f}',f'{x["peak_rss_mib"]:.1f}'] for kind,x in s['runtime'].items()]),
      '\nInference chính chạy tuần tự, CPU 6 threads, batch 4. Các thời gian mô tả máy và pipeline này, gồm preprocess và checkpoint vector sau mỗi batch; chưa phải benchmark latency tối ưu, không chứng minh chạy nhanh trên điện thoại hoặc GPU của người dùng.\n',
      '## 8. Kiểm chứng\n',
      '- Checkpoint cố định revision, trọng số lớn đối chiếu SHA-256 với metadata chính thức.\n- Source SugarCrepe khớp byte với repo chính thức hiện tại. Nhãn chuyên gia của mọi caption được chọn khớp dữ liệu trong repo tác giả EXPERT.\n- Cùng checkpoint B/32 được ánh xạ vào code CLIP gốc, load strict: kiểm tra 30 caption / 6 ảnh, sai số score tối đa '+f'{v["official_clip"]["errors"]["max_score_abs_error"]:.3g}'+'.\n- Không có caption bị CLIP cắt ở giới hạn 77 token: SugarCrepe tối đa 55 token, Flickr8k tối đa 39 token (đã gồm prefix và token đặc biệt); xem `token_length_audit.json`.\n- Tái tính tất cả điểm bằng phép tính scalar float64 độc lập; kiểm tra toàn bộ accuracy/tau-c/within-image ranking và SHA-256 media. Xem `independent_verification.json`.\n',
      '## 9. Có đủ để viết paper chưa?\n',
      '**Đủ cho báo cáo khả thi và quyết định hướng; chưa đủ để khẳng định một đóng góp xuất bản mạnh.** Đổi encoder và đặt tên metric mới tự nó chưa phải tính mới. CLIPScore, PAC-S/SPECS và các nghiên cứu robustness đã nghiên cứu caption evaluation; MIEB đã bao phủ compositionality. Cần định vị vào một vấn đề và một kết luận cụ thể có thể tái lập.\n',
      'Tên nghiên cứu tạm: **Replacing CLIP in Reference-Free Caption Evaluation: Robustness and Compression Trade-offs with EmbeddingGemma 2**.\n',
      'Bước nâng thành nghiên cứu hoàn chỉnh: chạy full SugarCrepe và full Flickr8k-Expert; thêm một bộ có caption sinh thực tế / domain khác; chạy checkpoint caption-metric đã huấn luyện như SPECS hoặc PAC-S; khóa tiêu chí chất lượng và ngưỡng chỉ trên validation riêng. Nếu cần cải tiến phương pháp, thêm cơ chế đã có như claim decomposition hoặc calibration rồi làm ablation; chưa xem những cơ chế chưa chạy là kết quả.\n',
      'Giới hạn: dữ liệu có trước cutoff huấn luyện nên chưa loại được contamination; chỉ hai CLIP checkpoint và một Gemma; native preprocessing và độ dài context khác nhau; chưa test tiếng Việt, caption dài, video, khả năng phát hiện mọi hallucination, latency real-time hoặc tương đương SOTA. Test ảnh mới độc lập với thí nghiệm cũ, không có nghĩa là độc lập với pretraining.\n',
      '## 10. Mã dùng ngay và tài liệu gốc\n',
      'Xem `README.md`, `gemma_clipscore.py`, `score_adapter.py`. Archive có toàn bộ mẫu media, vector và logs; không đóng gói trọng số model.\n',
      '- Model: https://huggingface.co/google/embeddinggemma-2\n- Hướng dẫn: https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers\n- CLIPScore: https://arxiv.org/abs/2104.08718 ; https://github.com/jmhessel/clipscore\n- SugarCrepe: https://arxiv.org/abs/2306.14610 ; https://github.com/RAIVNLab/sugar-crepe\n- Human evaluation: https://huggingface.co/datasets/yuwd/Flickr8k-HumanEval ; https://github.com/hjkim811/EXPERT\n- Robustness trước đây: https://arxiv.org/abs/2305.14998\n- SPECS: https://arxiv.org/abs/2509.03897 ; https://github.com/mbzuai-nlp/SPECS\n- BERTopic embeddings: https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html\n- MIEB task list: https://docs.mteb.org/overview/available_benchmarks/\n',
      '\n`results.json`, `human_results.json` lưu tất cả prefix/chiều đã chạy, bao gồm kết quả không được chọn làm cấu hình chính. Kết quả vòng trước nằm trong `prior_stage2_report.md`, không được gộp vào test mới.']
    text='\n\n'.join(lines);(ROOT/'REPORT_VI.md').write_text(text)
    (ROOT.parent/'EmbeddingGemma2_Replacement_Report_VI.md').write_text(text)

if __name__=='__main__':main()
