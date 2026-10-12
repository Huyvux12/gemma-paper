# EmbeddingGemma 2 — mở rộng mẫu và so sánh CLIP/CLAP

Báo cáo vòng 2: inference thật, không huấn luyện thêm, không chọn mẫu dựa trên điểm. Mẫu mới không trùng media với pilot; toàn bộ baseline chạy trên cùng danh sách cặp.

## 1. Phạm vi và giao thức

- Ảnh: 210 cặp SugarCrepe tổng cộng, gồm 70 pilot và 140 confirmation mới (20 cặp/nhóm lỗi).
- Audio: 250 cặp BRACE-Hallucination/Clotho, gồm 50 pilot và 200 confirmation mới từ 40 clip.
- Mẫu confirmation là kết quả chính. Kết quả gộp chỉ mang tính mô tả vì pilot đã được dùng để phát hiện giả thuyết.
- Inference CPU, float32; checkpoint được cố định revision và kiểm tra SHA256. Thông số đầy đủ nằm trong metadata đi kèm.
- FactChecking là prefix chính của Gemma. Giữ lại SentenceSimilarity, SearchQuery và không prefix để phân tích đầy đủ.
- CLIPScore: CLIP ViT-B/32 từ OpenAI, prefix `A photo depicts `, score `2.5 × max(cosine, 0)`; tái dựng bằng Transformers, không khẳng định bitwise giống phần mềm gốc.
- CLAP: `laion/clap-htsat-unfused`, cosine audio–text. Clip được chia cửa sổ 10 giây không chồng lấn, gộp trung bình các embedding đã chuẩn hóa rồi chuẩn hóa lại để phủ toàn bộ bằng chứng. Kết quả chỉ dùng 10 giây đầu được báo cáo riêng.
- Gemma đọc toàn bộ audio mono 16 kHz; CLAP dùng mono 48 kHz. Không đưa caption vào media embedding.
- Bốn số chiều Gemma được cắt từ cùng vector 768d và chuẩn hóa lại. CLIP/CLAP giữ nguyên vector 512d gốc; không giả định hai baseline có Matryoshka.
- Tất cả điểm chọn đúng yêu cầu score đúng > score sai; bằng nhau tính sai.

## 2. Kết quả chính trên mẫu mới

| Hệ thống | Ảnh / 140 cặp | Audio / 200 cặp |
|---|---:|---:|
| Gemma FactChecking 128d | 76.4% | 66.0% |
| Gemma FactChecking 256d | 85.7% | 69.0% |
| Gemma FactChecking 512d | 85.7% | 71.0% |
| Gemma FactChecking 768d | 85.7% | 70.0% |
| CLIPScore / CLAP toàn clip | 67.9% | 81.5% |

### So sánh ghép cặp đã khai báo trước

Delta là accuracy hệ thống thứ nhất trừ hệ thống thứ hai. Bootstrap 5.000 lần theo media ID; kiểm định đổi dấu ghép cặp theo media ID 20.000 lần. Holm điều chỉnh trong nhóm bốn phép so sánh chính. CI percentile 95% bên dưới chưa điều chỉnh nhiều phép so sánh.

| So sánh | Delta (điểm %) | CI 95% (điểm %) | p Holm |
|---|---:|---:|---:|
| image: Gemma768 - clip | +17.9 | +8.5 đến +27.3 | 0.00120 |
| image: Gemma128 - Gemma768 | -9.3 | -15.6 đến -2.9 | 0.02070 |
| audio: Gemma768 - clap | -11.5 | -19.0 đến -3.5 | 0.02290 |
| audio: Gemma128 - Gemma768 | -4.0 | -11.0 đến +3.0 | 0.33313 |

## 3. Kiểm chứng dấu hiệu prefix của pilot

| Prefix audio | 128d | 256d | 512d | 768d |
|---|---:|---:|---:|---:|
| FactChecking | 66.0% | 69.0% | 71.0% | 70.0% |
| SentenceSimilarity | 55.5% | 61.0% | 63.5% | 63.0% |
| SearchQuery | 67.5% | 71.5% | 75.5% | 75.0% |
| None | 67.5% | 73.5% | 71.0% | 72.0% |

Đây là hai giả thuyết phụ được đặt ra từ pilot và kiểm chứng trên media mới. Phép tương tác so sánh mức giảm accuracy do nén của SentenceSimilarity với mức giảm của FactChecking trên cùng các cặp.

| So sánh phụ | Delta (điểm %) | CI 95% (điểm %) | p Holm (nhóm 2) |
|---|---:|---:|---:|
| audio: SentenceSimilarity128 - SentenceSimilarity768 | -7.5 | -14.5 đến -0.5 | 0.12609 |
| audio: compression effect SentenceSimilarity minus FactChecking | -3.5 | -11.5 đến +4.5 | 0.47713 |

## 4. Kết quả gộp, chỉ để mô tả

| Hệ thống | Ảnh / 210 cặp | Audio / 250 cặp |
|---|---:|---:|
| Gemma FactChecking 128d | 78.1% | 67.2% |
| Gemma FactChecking 256d | 85.2% | 70.4% |
| Gemma FactChecking 512d | 85.2% | 71.6% |
| Gemma FactChecking 768d | 85.7% | 70.8% |
| CLIPScore / CLAP toàn clip | 67.6% | 82.0% |

## 5. Phân tích nhóm lỗi ảnh trên mẫu mới

| Nhóm lỗi | Gemma 128d | Gemma 768d | CLIPScore |
|---|---:|---:|---:|
| add_att | 75.0% | 85.0% | 65.0% |
| add_obj | 85.0% | 100.0% | 80.0% |
| replace_att | 85.0% | 100.0% | 70.0% |
| replace_obj | 90.0% | 85.0% | 100.0% |
| replace_rel | 45.0% | 70.0% | 60.0% |
| swap_att | 85.0% | 85.0% | 55.0% |
| swap_obj | 70.0% | 75.0% | 45.0% |

## 6. CLAP: toàn clip và 10 giây đầu

| Tập | Toàn clip | 10 giây đầu |
|---|---:|---:|
| pilot | 84.0% | 88.0% |
| confirmation | 81.5% | 85.0% |
| all | 82.0% | 85.6% |

## 7. Điều gì có thể kết luận?

- image: Gemma768 - clip: quan sát chênh lệch +17.9 điểm %; p Holm < 0,05 trong mẫu mới và giao thức này.
- image: Gemma128 - Gemma768: quan sát chênh lệch -9.3 điểm %; p Holm < 0,05 trong mẫu mới và giao thức này.
- audio: Gemma768 - clap: quan sát chênh lệch -11.5 điểm %; p Holm < 0,05 trong mẫu mới và giao thức này.
- audio: Gemma128 - Gemma768: chưa có bằng chứng đủ rõ sau điều chỉnh Holm; không diễn giải thành hai hệ thống tương đương.
- Dấu hiệu tương tác prefix × số chiều chưa được xác nhận rõ sau điều chỉnh Holm trên mẫu mới.

## 8. Giới hạn

- Đây là lựa chọn caption đúng trong một cặp; không phải accuracy kiểm chứng mọi câu do VLM/Audio-LLM sinh tự do.
- Không chứng minh dữ liệu sạch khỏi pretraining. Media mới chỉ độc lập với pilot, vẫn thuộc cùng benchmark.
- Chỉ một checkpoint CLIP và một checkpoint CLAP; không đủ để tuyên bố SOTA. Các model có kích thước và dữ liệu huấn luyện khác nhau.
- 200 cặp audio mới chỉ tương ứng 40 clip độc lập; thống kê đã gom theo clip.
- Chưa hiệu chỉnh ngưỡng, huấn luyện adapter hoặc triển khai chính sách chọn số chiều thích ứng.
- Retrieval trong JSON chỉ là chẩn đoán trên corpus được lấy mẫu, chưa phải benchmark retrieval chuẩn. Không dùng nó để khẳng định factuality suy giảm trước retrieval.
- Nén vector không giảm chi phí encoder. Không so latency giữa các lượt chạy CPU đồng thời; timer có thể gồm thời gian chờ media.

## 9. Bước nghiên cứu tiếp theo

Hướng paper tạm thời: **nén embedding và khả năng phân biệt lỗi compositional theo modality**. Đây là hướng để kiểm chứng tiếp, chưa phải tuyên bố đóng góp mới hoặc kết quả đủ cho một paper.

1. Ưu tiên kiểm tra lại mức giảm accuracy ảnh ở 128d trên toàn bộ SugarCrepe hoặc một tập test mới đủ lớn. Giữ FactChecking cố định cho kiểm định chính; mọi lựa chọn prefix/chiều mới phải dùng validation riêng.
2. Thêm benchmark retrieval chuẩn và các baseline mạnh hơn, với cùng media và caption. Chỉ khi có retrieval chuẩn mới kiểm định được giả thuyết factuality giảm trước retrieval.
3. Giữ audio như phép kiểm tra khái quát hóa và báo cáo cả kết quả Gemma thua CLAP. Không diễn giải khác biệt giữa hai dataset thành tác động nhân quả của modality.
4. Tín hiệu tương tác prefix × số chiều chưa đủ rõ để làm đóng góp chính. Kết quả 256d và 768d bằng nhau về point accuracy ảnh trong mẫu này chỉ gợi ý một cấu hình để kiểm chứng tiếp, không chứng minh tương đương.

## 10. Tái lập và dữ liệu lưu

- `protocol.json`: kế hoạch kiểm định và cách chia dữ liệu.
- `image_manifest.json`, `audio_manifest.json`: ID, split, nhãn và checksum media.
- `runs/*/embeddings.npz`: embedding thật; `runs/*/metadata.json`: revision, cấu hình và phiên bản.
- `comparison_results.json`, `all_pair_scores.csv`: metrics và điểm từng cặp.
- `baseline_checkpoints.json`: checksum checkpoint CLIP/CLAP đã xác minh.
- `independent_verification.json`: tái tính từng score từ vector bằng float64, kiểm tra nhãn gốc và toàn bộ accuracy xuất ra.
- Xem README.md để chạy lại hoặc chỉ phân tích lại vector.

## Nguồn

- https://huggingface.co/google/embeddinggemma-2
- https://github.com/jmhessel/clipscore
- https://huggingface.co/openai/clip-vit-base-patch32
- https://huggingface.co/laion/clap-htsat-unfused
- https://github.com/RAIVNLab/sugar-crepe
- https://huggingface.co/datasets/gtysssp/audio_benchmarks
