# Gemma: kết quả public và giới hạn

GPU end-to-end gốc đã COMPLETE trên Kaggle T4 ngày17/10/2026. Tái tính độc lập15 variants/18537 test records: cache trùng byte snapshot gốc, max prediction/metric drift=0, selection/epoch giữ nguyên. Public packaging tách cấu hình riêng khỏi notebook, không chạy một thí nghiệm GPU mới.

| Variant | Sugar % | Flickr raw tau-c | Polaris raw tau-c |
|---|---:|---:|---:|
| Base |91.62237|0.544936|0.556743|
| Mean primary |93.53604|0.557082|0.558193|
| Uncertainty secondary |93.49351|0.558029|0.558149|

Mean-base: Sugar+1.91367pp, CI95[1.38061,2.42969]pp; Flickr raw+0.012146, CI[0.009446,0.014905]; Polaris raw+0.001450, CI[−0.000628,0.003424]. Paired image-cluster2000 resamples, exploratory/unadjusted. Polaris CI chứa0; không suy thành bằng nhau hay chắc chắn cải thiện.

Mean sửa124 Sugar pairs và tạo34 lỗi, net+90; swap_att giảm5/435. Flickr within-image ranking tăng0.67660pp, Polaris giảm0.08579pp. Global correlation và ranking trong ảnh không tương đương.

Mean primary rank32/radius0.03/preservation0.1/14epochs; uncertainty cùng config/11epochs. Head49152 params,192KiB FP32. Native timing mới: B1 p50/p95527.903/534.859ms; B8 effective2.10439 pairs/s; peak allocated2811.7MiB. Adapter nhỏ không làm encoder rẻ; cached-head không thay native inference cost.

Control CPU lịch sử năm seed giữ cùng config/14epochs/paired batches: uncertainty-minus-mean Flickr raw−0.000561, seed×image CI[−0.001077,−0.000067]; Polaris raw−0.000839, CI[−0.001410,−0.000246]. Cả hai raw metrics giảm5/5 seeds. Giữ mean primary, không chọn lại theo test.

Stage12 có7 backbone/81 variants/21 main recipes; Gemma cao nhất ba point metrics trong nhóm đã đo. Các model khác không được chạy lại trong notebook Gemma mới. Gemma FP32, các recipe khác phần lớn FP16 và session khác; chưa có paired cross-model significance test. Không gọi đây là SOTA hoặc superiority ở mọi điều kiện.

[Báo cáo và8 bộ hình](reports/REPORT_VI.html), [mọi metric](reports/all_variants.csv), [CI](reports/paired_ci.csv), [timing](reports/timing128.csv), [metric drift](reports/metric_drift.csv), [kế hoạch nâng paper](../GEMMA_PAPER_HANDOFF_20261017_VI.md). Báo cáo public chỉ có aggregate evidence; per-record data và fingerprints giữ riêng.
