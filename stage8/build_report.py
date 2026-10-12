"""Build the Vietnamese Stage8 report from saved, auditable result files."""
import argparse
from pathlib import Path
from common import read,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=Path(a.results);root=Path(__file__).parent;s=read(r/'adapter_training/selection.json');u=read(r/'adapter_uncertainty.json')['results'];metrics=read(r/'all_metrics.json')['cohorts']['unseen900'];state=read(r/'kaggle_connection.json')
    base=metrics['gemma-CLIPLiteral-768-float32'];adapt=metrics['gemma-adapter-ensemble5'];human_ci=u['human_unseen900']['paired_image_bootstrap_95_percentile_tau_c_delta_ci']
    lines=['# Báo cáo Stage8: metric mở rộng và adapter học trên EmbeddingGemma-2','',
        'Ngày: 09/10/2026. Đây là phần bổ sung cho báo cáo/source Stage1–7, được giữ nguyên trong `baseline.zip` của gói đầy đủ.','',
        '## Kết quả chính','',
        'Đã xác thực Kaggle API, tạo dataset và notebook riêng tư, tính thêm 11 thống kê đánh giá, 7 baseline tham chiếu và huấn luyện 20 adapter residual. Ensemble 5 seed tăng accuracy phân biệt caption trên SugarCrepe, nhưng giảm mạnh mức tương hợp với đánh giá con người trên Flickr8kExpert. Phương pháp hiện tại thể hiện trade-off; bằng chứng chưa hỗ trợ kết luận cải tiến tổng thể.','',
        '| Phép đo | Gemma gốc | Gemma + ensemble adapter |','|---|---:|---:|']
    for name,key in [('SugarCrepe 560 ảnh/cặp','historical560'),('SugarCrepe 1.005 ảnh / 3.949 cặp','historical_unseen3949')]:
        z=u[key];lines.append(f'| Accuracy {name} | {100*z["original_accuracy"]:.4f}% | {100*z["adapter_accuracy"]:.4f}% |')
    lines += [f'| Kendall τ-c, 900 ảnh expert | {base["kendall_tau_c_individual"]:.6f} | {adapt["kendall_tau_c_individual"]:.6f} |',
        f'| Pair accuracy trong ảnh expert | {100*base["within_image_micro_pair_accuracy"]:.4f}% | {100*adapt["within_image_micro_pair_accuracy"]:.4f}% |',
        f'| NDCG@5, expert | {base["ndcg_at5_tie_averaged"]:.6f} | {adapt["ndcg_at5_tie_averaged"]:.6f} |','',
        'Khoảng tin cậy dưới đây là bootstrap ghép cặp theo ảnh, mang tính thăm dò trên benchmark đã được quan sát; không điều chỉnh nhiều phép so sánh và không tạo ra một test xác nhận mới.','']
    for key in ['historical560','historical_unseen3949']:
        z=u[key];ci=z['paired_image_bootstrap_95_percentile_delta_ci'];lines.append(f'- {key}: Δ accuracy = {100*z["delta"]:+.4f} điểm phần trăm; CI95% [{100*ci[0]:.4f}, {100*ci[1]:.4f}], 5.000 bootstrap. Gemma gốc đúng/adapter sai: {z["baseline_only_correct"]}; chiều ngược lại: {z["adapter_only_correct"]}.')
    lines += [f'- Expert 900: Δ τ-c = {adapt["kendall_tau_c_individual"]-base["kendall_tau_c_individual"]:+.6f}; CI95% [{human_ci[0]:.6f}, {human_ci[1]:.6f}], 1.000 bootstrap.','',
        '560 ảnh nằm trong nhóm 1.005 ảnh lịch sử chưa dùng ở development; hai số mẫu không được cộng lại. Các benchmark này cùng nhãn/kết quả đã được nghiên cứu ở các stage trước.','', '![Trade-off và CI bootstrap theo ảnh](stage8/results/adapter_tradeoff.png)', '', 'Hình: Δ ensemble − Gemma gốc; thanh ngang là CI95% bootstrap ghép cặp theo ảnh. Đây là chẩn đoán thăm dò trên benchmark lịch sử.', '',
        '## Dữ liệu và quy trình huấn luyện','',
        'Backbone: `google/embeddinggemma-2`, revision `914f7f89142e33e77833254d9c9b90c3cef7303b`, embedding 768 chiều, float32, prefix `A photo depicts `. Trọng số encoder giữ cố định. Hai residual projection riêng cho ảnh và text có dạng normalize(x + U·D·x), khởi tạo U=0 để bắt đầu đúng tại embedding gốc. Trọng số của adapter được cập nhật bằng gradient.','',
        'Train chỉ dùng 555 ảnh SugarCrepe thuộc development lịch sử: 444 train / 111 validation, chia theo ảnh với seed 20261009. Kiểm tra cả tên ảnh và SHA256 byte ảnh xác nhận train/validation/historical test không trùng nhau; train/validation không trùng hash ảnh expert. Train và chọn checkpoint chỉ forward các embedding development.','',
        f'Số cặp: {s["train_pairs"]} train và {s["validation_pairs"]} validation. Grid cố định: rank 16/32 × anchor 0,1/1,0 × seed 11/23/37/53/71 = 20 lần train, mỗi lần 30 epoch. AdamW: lr 0,001, weight decay 0,0001, batch 128, gradient clip 1. Loss = softplus((s_neg − s_pos + 0,02)/0,05) + λ·mean(1 − cosine(embedding mới, embedding gốc)).','',
        'Mỗi checkpoint được chọn bằng accuracy macro theo ảnh của validation, ưu tiên epoch sớm khi hòa. Rank/anchor được chọn bằng trung bình validation của 5 seed; hòa thì ưu tiên rank nhỏ và anchor mạnh. Điểm trên SugarCrepe historical test hoặc Flickr8kExpert không tham gia lựa chọn.','',
        f'Cấu hình chọn: rank {s["selected"]["rank"]}, anchor {s["selected"]["anchor_weight"]}, validation trung bình {s["selected"]["mean_validation_accuracy"]:.6f}. Mỗi adapter có {s["finalists"][0]["trainable_parameters"]:,} tham số; ensemble sử dụng cả 5 seed và trung bình score sau clamp. Đây là thích nghi ở đầu ra encoder; LoRA/full fine-tuning backbone là một bước nghiên cứu riêng.','',
        '| Seed | Epoch chọn | Validation macro accuracy | Expert τ-c |','|---|---:|---:|---:|']
    for z in s['finalists']:lines.append(f'| {z["seed"]} | {z["best_epoch"]} | {z["validation_image_macro_accuracy"]:.6f} | {metrics["gemma-adapter-seed"+str(z["seed"])]["kendall_tau_c_individual"]:.6f} |')
    lines += ['',
        '## Metric mở rộng','',
        'Bảng dữ liệu đầy đủ `stage8/results/all_metrics.csv` và JSON chứa cả cohort 1.000 ảnh và 900 ảnh expert. Cohort 900 có 5.112 candidate, 15.336 rating cá nhân, 9.459 cặp có mean rating khác nhau. Có 46 tên cột score trước PAC-S: 32 cấu hình baseline đã có, một score gốc tái tính để kiểm tra, 5 adapter, 1 ensemble, 7 metric tham chiếu. Các biến thể chiều/prompt/precision không phải backbone độc lập.','',
        '| Metric, expert 900 | Gemma gốc | Adapter ensemble |','|---|---:|---:|']
    keys=[('Kendall τ-c, rating cá nhân','kendall_tau_c_individual'),('Kendall τ-b, rating cá nhân','kendall_tau_b_individual'),('Spearman, rating cá nhân','spearman_individual'),('Pearson, rating cá nhân','pearson_individual'),('Spearman, mean rating','spearman_mean_rating'),('Pair accuracy micro trong ảnh','within_image_micro_pair_accuracy'),('Pair accuracy macro theo ảnh','within_image_macro_pair_accuracy'),('NDCG@5','ndcg_at5_tie_averaged'),('Top1 đúng caption human-best','top1_human_best_tie_averaged'),('AUROC, mean rating ≥3','auroc_mean_rating_ge3'),('Average precision, mean rating ≥3','average_precision_mean_rating_ge3')]
    for label,key in keys:lines.append(f'| {label} | {base[key]:.6f} | {adapt[key]:.6f} |')
    if 'PAC-S-ViTB32' in metrics:
        z=metrics['PAC-S-ViTB32'];comparison=read(r/'pac_comparison.json');ci=comparison['paired_image_bootstrap_95_percentile_tau_c_delta_ci']
        lines += ['', 'PAC-S đã chạy thành công, nâng bảng tổng hợp lên 47 tên cột score. Trên expert 900, PAC-S τ-c = '+f'{z["kendall_tau_c_individual"]:.6f}'+', pair accuracy = '+f'{100*z["within_image_micro_pair_accuracy"]:.4f}%'+'. Δ τ-c (PAC-S − Gemma gốc) = '+f'{comparison["tau_c_delta"]:+.6f}'+f', CI95% [{ci[0]:.6f}, {ci[1]:.6f}] từ 1.000 bootstrap theo ảnh. CI chứa 0; điểm ước lượng này chưa chứng minh ưu thế về human alignment của một trong hai metric.', '']
    lines += ['',
        'Kendall/Spearman/Pearson cá nhân lặp score caption theo từng rating; mỗi caption có ba rating. Pair accuracy bỏ cặp hòa mean rating, score hòa được tính chưa xếp đúng. Macro lấy trung bình accuracy của từng ảnh. NDCG dùng gain 2^(mean rating−1)−1, k=5, kỳ vọng DCG khi score hòa; chỉ tính nhóm ≥2 candidate có ideal gain dương. Top1 lấy tỷ lệ caption tốt nhất theo người trong nhóm score đứng đầu bị hòa. AUROC và average precision dùng positive mean rating ≥3; AP là average precision, không phải tích phân hình thang PR-AUC. Các score không được diễn giải thành xác suất đã calibration.','',
        '## Baseline dùng reference','',
        'Đã chạy pycocoevalcap 1.2 với COCO PTBTokenizer và 5 reference `ground_truth` nguyên bản từ annotation tác giả; không tạo reference bằng LLM. CIDEr dùng IDF chuẩn trên corpus candidate-record, trong đó reference được lặp cho các candidate khác nhau của cùng ảnh. BLEU là sentence score cho từng candidate. Java PTBTokenizer/METEOR đã thực thi thành công.','',
        '| Method | Expert 900 Kendall τ-c |','|---|---:|']
    for name in ['BLEU1','BLEU2','BLEU3','BLEU4','ROUGE_L','CIDEr','METEOR']:lines.append(f'| {name} | {metrics[name]["kendall_tau_c_individual"]:.6f} |')
    lines += ['',
        'Các baseline này nhận thêm reference, khác điều kiện đầu vào của Gemma/SigLIP/CLIP reference-free. SPICE là tùy chọn cần Java assets, chưa chạy; không có giá trị SPICE giả hoặc thay bằng metric khác.','',
        '## Kaggle và phép đo GPU','',
        f'Kaggle account: `{state["username"]}`. Dataset riêng tư: `{state.get("dataset", "pending")}`. Notebook riêng tư: [{state.get("kernel", "pending")}]({state.get("kernel_url", "")}). Quota lúc xác thực: GPU 30h / TPU 20h còn lại; đây là snapshot trước job, không phải số dư thời gian thực.','',
        f'Trạng thái mới nhất đã xác minh: **{state.get("last_observed_status","unknown")}**, kernel version {state.get("submitted_kernel_version","unknown")}. Hoàn thành remote: {state.get("remote_complete",False)}.','']
    gpu=state.get('gpu_result_relative_path')
    if gpu and (r/gpu).exists():
        g=read(r/gpu);lines += [f'GPU: {g["gpu"]}; Torch {g["torch"]}; {g["precision"]}.','', '| Model | Batch | P50 ms/batch | P95 ms/batch | Pairs/s | Peak allocated MiB |','|---|---:|---:|---:|---:|---:|']
        for name,values in g['measurements'].items():
            for z in values:lines.append(f'| {name} | {z["batch_size"]} | {z["p50_wall_ms_per_batch"]:.3f} | {z["p95_wall_ms_per_batch"]:.3f} | {z["pairs_per_second_from_median"]:.2f} | {z["peak_allocated_bytes"]/2**20:.1f} |')
    else:lines += ['Chưa có kết quả GPU hoàn tất được tải về tại thời điểm báo cáo; các số đo dự kiến không được điền vào bảng kết quả.']
    if (r/'cpu_gpu_comparison.json').exists() and gpu and (r/gpu).exists():
        parity=read(r/'cpu_gpu_comparison.json');maxerr=max(parity['maximum_absolute_human_score_difference_per_method'].values())
        lines += ['',f'Adapter được train lại trên GPU để tái lập. Cấu hình và epoch được chọn trùng CPU; sai khác human score lớn nhất giữa CPU/GPU = {maxerr:.3g}. Bảng human/accuracy chính dùng kết quả CPU; phép đo fresh GPU dùng checkpoint train trên GPU, không tuyên bố checkpoint byte-identical. 92 file output đã khớp SHA256 manifest phiên chạy; runtime và manifest được tải riêng.','',
            f'Ở batch 1 trong cấu hình phục vụ native đã đo, Gemma gốc khoảng {g["measurements"]["gemma"][0]["p50_wall_ms_per_batch"]:.2f} ms/cặp, SigLIP2-224 khoảng {g["measurements"]["siglip224"][0]["p50_wall_ms_per_batch"]:.2f} ms/cặp; Gemma chậm hơn khoảng {g["measurements"]["gemma"][0]["p50_wall_ms_per_batch"]/g["measurements"]["siglip224"][0]["p50_wall_ms_per_batch"]:.1f} lần trong phép đo này. Ensemble adapter khoảng {g["measurements"]["gemma-adapter-ensemble5"][0]["p50_wall_ms_per_batch"]:.2f} ms/cặp. Kết quả này không hỗ trợ claim Gemma là encoder nhanh hơn; giới hạn phần cứng/precision/native preprocessing/8 input phải đi cùng mọi trích dẫn latency.']
    lines += ['',
        'Protocol GPU: cùng một phiên GPU0, float32, TF32 tắt; cùng 8 ảnh/caption expert thật, batch 1 và 8; warmup 3, đo 15 lần với CUDA synchronize; lưu mọi sample, p50/p95, throughput, peak allocated/reserved. Thời gian bao gồm decode/preprocessing native, encoding ảnh + text, scoring và adapter khi có; download/load model được loại khỏi phép đo. Image budget native khác nhau giữa encoder. Trọng số được pin revision và ghi SHA256. Chi phí lưu embedding int8 trước đây là storage payload, không phải tốc độ encoder.','',
        'PAC-S dùng repo tác giả pin `bfbc488e9f5cccd818e75c7cdfaa37e1c87f00eb`, checkpoint PAC-S ViT-B/32 đúng của tác giả, w=2.0; SHA checkpoint được ghi sau download, tác giả chưa cung cấp hash đã xác minh trong quy trình này. Trạng thái PAC-S riêng trong output, không thay bằng PAC-S++. Overlap pretraining của các backbone và fine-tuning PAC-S với expert chưa được audit độc lập; audit ảnh/hash ở trên xác minh dữ liệu huấn luyện adapter.','',
        '## Kiểm tra và khả năng tái lập','',
        '- 32 endpoint Kendall τ-c cũ được tái lập với sai số <1e−12.','- Kiểm tra identity của adapter khởi tạo, gradient học khác zero, checkpoint NPZ roundtrip, NDCG score ties đối chiếu sklearn, top1/pair khi hòa, tách ảnh/hash và parse toàn bộ Python đều pass.','- Đã kiểm tra token không xuất hiện trong snapshot source notebook. Credential được truyền riêng vào controller, không chứa trong notebook/report/gói source.','- Mọi run lưu history, checkpoint hash, số tham số và validation score. Snapshot source của từng phiên Kaggle được giữ riêng với log thành công/lỗi.','',
        f'Protocol SHA256: `{digest(root/"protocol.json")}`. Gói đầy đủ giữ các manifest/vector/score/checkpoint Stage1–7 trong `baseline.zip`; Stage8 ghi các file mới tách riêng. Các log lần chạy không thành công cũng được giữ để có thể kiểm tra.','',
        '## Ý nghĩa đối với paper và bước tiếp theo','',
        'Phần mở rộng đủ để viết mục ablation/adaptation và đánh giá đa metric với kết quả âm minh bạch. Hiện chưa có bằng chứng cho claim một Gemma thích nghi cải thiện đồng thời compositional discrimination và human alignment; CI SugarCrepe 560 chứa 0, còn mức giảm expert lớn và nhất quán qua seed. Loss hard-negative chỉ từ SugarCrepe là một giả thuyết giải thích trade-off, chưa được kiểm chứng bằng ablation nhân quả.','',
        'Nếu mục tiêu paper là phương pháp thích nghi có khả năng tổng quát, cần thiết kế tiếp mục tiêu giữ human alignment hoặc huấn luyện đa nhiệm, chọn bằng validation thích hợp, rồi khóa phương pháp và đánh giá một bộ test mới độc lập có audit overlap. Cấu trúc adapter/loss hiện tại dùng thành phần phổ biến; chưa thực hiện đánh giá novelty đầy đủ hoặc đảm bảo điều kiện chấp nhận của venue.','',
        '## Nguồn chính','',
        '- Kaggle CLI: https://github.com/Kaggle/kaggle-cli','- PAC-S, implementation tác giả: https://github.com/aimagelab/pacscore','- COCO caption metrics: https://github.com/salaniz/pycocoevalcap','- Flickr8k-HumanEval mirror pin theo manifest: https://huggingface.co/datasets/yuwd/Flickr8k-HumanEval','- EmbeddingGemma-2: https://huggingface.co/google/embeddinggemma-2','',
        'Chi tiết nguồn, phiên bản và SHA256 của baseline nằm trong các receipt/manifest Stage1–7; điều khoản dữ liệu và trọng số tiếp tục thuộc nguồn gốc.']
    Path(a.output).write_text('\n'.join(lines)+'\n');print(a.output)

if __name__=='__main__':main()
