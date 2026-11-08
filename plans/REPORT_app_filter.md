# Báo cáo lọc app trước vòng gọi tool – ZAC 2026 LifeMate

Cập nhật: 2026-11-08. Đi kèm `PLAN_app_filter.md` (thiết kế, thí nghiệm E1–E6) và `PLAN_FC_clef_qwen35.md`. Báo cáo này ghi kết quả thí nghiệm E1 (zero-shot) và phần fine-tune của E2 cho 3 model embedding và BM25.

## 1. Kết luận

- **Fine-tune là bắt buộc.** Zero-shot cả ba model embedding chỉ đạt 58–72% recall ở top-3 trên toàn train, ngang hoặc kém BM25. Sau fine-tune 3 epoch (2.971 ví dụ, 2–4 phút trên T4), val_ood ở k=4 tăng từ 66–76% lên 92–95%, dù app giữ lại không có mặt trong dữ liệu train.
- **Ba model ngang nhau trong mức nhiễu** (val_in chỉ 109 mẫu, sai số khoảng ±3–5 điểm). vietnamese-bi-encoder nhỉnh nhất trên val_ood (98,5% ở k=5) nhưng không ổn định trên val_in, chậm hơn và cần tách từ. e5-small là lựa chọn ít phụ thuộc nhất.
- **Không model nào đạt mục tiêu recall ≥ 98–99% khi giữ ≤ 4 app.** Để đạt recall khoảng 95–98% phải giữ 5,8–6,2 trên 9,2 app cài trung bình, nên lọc cứng chỉ giảm khoảng 35% số app, ít hơn mức ước lượng ban đầu (giữ khoảng 3 app).
- **Đề xuất: không loại hẳn app, dùng catalog hai tầng.** Tầng 1 là top-k theo embedding, giữ mô tả đầy đủ. Tầng 2 là phần còn lại, chỉ ghi tên tool. Model vẫn chọn được app bị xếp thấp nhầm. Chưa đo mức tiết kiệm token và tác động lên điểm `metrics/` của cách này.
- Thêm câu trả lời trước của assistant vào query và kết hợp BM25 (RRF) **không ổn định**: có chỗ tăng vài điểm, có chỗ giảm vài điểm. Không đưa vào thiết kế.

## 2. Thiết lập

- Dữ liệu: 1.545 mẫu train, chia như clef (`asr/agent/clef/splits.json`): train 1.158, val_in 128, val_ood 259. App giữ lại cho val_ood: alarm, ewallet, flights, loyalty, podcast_radio.
- Mỗi mẫu chỉ chọn trong `installed_apps` của chính nó (trung bình 9,2 app). Mẫu không gọi app nào bị bỏ khỏi recall (val_in còn 109, val_ood 259, train 1.006).
- **Chỉ số:** recall = tỉ lệ mẫu giữ đủ **mọi** app vàng. Top-k theo điểm, cộng luật: app đã gọi trong history, và app cung cấp id theo `x-ref` của app đã giữ. Cột sau dấu `/` là số app giữ trung bình (đã gồm luật). Kí hiệu `k4+r` trong bảng là top-4 cộng luật.
- **Tài liệu app** = `name_vi` + mô tả app + mô tả các tool. **Query** = `audio_text` yêu cầu cuối (transcript sạch, chưa có lỗi ASR); biến thể `+prev` thêm text của lượt assistant ngay trước.
- **Fine-tune:** InfoNCE nhiều dương (softmax trên các app đã cài của chính mẫu, scale 20), AdamW lr 2e-5, batch 32, 3 epoch. Dữ liệu = 1.006 lượt cuối cộng 1.965 lượt history có call của split train. Mọi app giữ lại bị loại khỏi ứng viên (kể cả làm ví dụ âm), và các lượt history gọi app giữ lại bị loại.
- **Chạy trên:** Kaggle, T4, seed 42. Model: `intfloat/multilingual-e5-small` (tiền tố `query:` / `passage:`), `bkai-foundation-models/vietnamese-bi-encoder` (tách từ bằng pyvi), `paraphrase-multilingual-MiniLM-L12-v2`. Số tham số đo thật: 117,7M / 135,0M / 117,7M, đều nằm trong phần còn dư khoảng 0,25B của giới hạn 6B.
- Độ trễ: CPU fp32, một câu hỏi mỗi lần (embedding tài liệu app tính sẵn), đo trên 50 câu.

## 3. Kết quả

### 3.1 Sau fine-tune, query là `audio_text` (recall / số app giữ)

| Model | Tập | k=3 | k=4 | k=5 |
|---|---|---|---|---|
| BM25 | val_in | 83,5% / 4,1 | 89,9% / 5,0 | 92,7% / 5,8 |
| BM25 | val_ood | 71,8% / 4,7 | 74,9% / 5,4 | 79,5% / 6,2 |
| e5-small | val_in | 83,5% / 4,0 | 92,7% / 4,9 | 95,4% / 5,8 |
| e5-small | val_ood | 84,9% / 4,6 | 93,4% / 5,4 | 95,4% / 6,2 |
| vietnamese-bi-encoder | val_in | 84,4% / 4,0 | 89,0% / 4,9 | 93,6% / 5,8 |
| vietnamese-bi-encoder | val_ood | 87,3% / 4,6 | 95,0% / 5,4 | **98,5%** / 6,2 |
| MiniLM-L12 | val_in | 78,9% / 4,0 | 89,9% / 5,0 | 95,4% / 5,8 |
| MiniLM-L12 | val_ood | 75,3% / 4,6 | 92,3% / 5,5 | 97,7% / 6,3 |

BM25 không có fine-tune; đây là kết quả BM25 chạy trên cùng tập.

### 3.2 Zero-shot trên val_ood, k=4 (so với sau fine-tune)

| Model | Zero-shot | Sau fine-tune |
|---|---|---|
| BM25 | 74,9% / 5,4 | – |
| e5-small | 66,0% / 5,3 | 93,4% / 5,4 |
| vietnamese-bi-encoder | 76,4% / 5,5 | 95,0% / 5,4 |
| MiniLM-L12 | 75,7% / 5,6 | 92,3% / 5,5 |

Trên val_in, zero-shot ở k=3 chỉ đạt 62–68% với cả ba model.

### 3.3 Biến thể (k=5, val_ood)

| Model | Query thường | `+prev` | ⊕ BM25 (RRF) | `+prev` ⊕ BM25 |
|---|---|---|---|---|
| e5-small | 95,4% | 96,5% | 96,1% | 93,4% |
| vietnamese-bi-encoder | 98,5% | 98,5% | 98,5% | 98,8% |
| MiniLM-L12 | 97,7% | 96,1% | 95,8% | 95,4% |

Chênh lệch giữa các biến thể nằm trong khoảng ±2–3 điểm và đổi chiều giữa các model, nên coi là nhiễu. Ở k=3 và k=4, BM25 kết hợp lúc tăng lúc giảm tới 5 điểm.

### 3.4 Độ trễ CPU (một câu hỏi)

| Model | ms / câu hỏi | Ghi chú |
|---|---|---|
| MiniLM-L12 | 36,1 | |
| e5-small | 40,0 | |
| vietnamese-bi-encoder | 68,4 | gồm tách từ bằng pyvi |

Đều thấp so với ngân sách 30 giây mỗi mẫu. Số ms đo trên CPU của Kaggle, chưa đo trên Xeon Gold 6442Y của máy chấm.

## 4. Hạn chế

- **Mẫu nhỏ:** val_in 109 mẫu, nên chênh dưới 3–5 điểm chưa có ý nghĩa.
- **Transcript sạch:** query là `audio_text`. Chưa đo với transcript ASR có lỗi, nơi tên riêng và từ khoá quan trọng có thể sai.
- **Chưa đo tác động lên điểm cuối:** chưa đưa bộ lọc vào vòng Clef, chưa biết mất bao nhiêu điểm `metrics/` khi lọc sót, cũng chưa biết số token và độ trễ tiết kiệm thật.
- **Chưa phản ánh app lạ của public test.** val_ood chỉ giữ lại 5 app đã thấy mô tả trong catalog train, và các app đó vẫn nằm trong `installed_apps` của nhiều mẫu train (xem phần val_ood trong `PLAN_data_augment_v3.md`). Với 11 app hoàn toàn mới ở public test, recall có thể thấp hơn.
- **Chọn k trên chính các tập này:** số liệu theo k là để so sánh, chưa phải ngưỡng đã được kiểm trên tập khác.
- **Sinh trên một seed, một lượt chạy.**
- Số tham số, license và ngày phát hành của ba model chưa kiểm theo model card chính thức (AGENT.md yêu cầu trước khi đưa vào Docker).

## 5. Bước tiếp theo

1. **Catalog hai tầng:** tầng 1 top-k (k khoảng 4–5) giữ mô tả đầy đủ, tầng 2 chỉ ghi tên tool. Đo số token trước/sau trên train và public_test, kèm độ dài prompt tệ nhất so với giới hạn 16.384.
2. **Thí nghiệm E3:** cho chính LLM (adapter Clef P2 hiện có) tự trả lời có/không cho từng app. Đây là đối chứng không tốn tham số, và có thể kết hợp với embedding.
3. **Đo với transcript ASR:** chạy lại E1–E2 với query là transcript của PhoWhisper v1, và với query có nhiễu.
4. **Fine-tune thêm với dữ liệu tổng hợp** (mô tả app viết lại, app giả, theo `PLAN_data_augment_v3.md`) rồi đo lại trên val_ood.
5. **Đưa vào Clef** (E5): train với catalog hai tầng, kèm app nhiễu, rồi so điểm `metrics/` trên val_in và val_ood với bản không lọc.

## 6. Tái lập

- Kernel: Kaggle `duongantbat/zaic-app-filter-eval` (version 2), dữ liệu: dataset riêng tư `duongantbat/zaic-app-filter-data` (`train.jsonl`, `train_tools_catalog.json`, `splits.json`; không có audio, không có `*_environment/`).
- Mã kernel, BM25 local và kết quả thô nằm ở `asr/agent/clef/app_filter/` (xem README ở đó): `kernel.py`, `kernel-metadata.json`, `bm25_baseline.py`, `results.json`, `summary.log`.
- Token Kaggle dùng cho lượt này đã được dán trong chat; cần thu hồi và tạo token mới.
