# Plan lọc app trước vòng gọi tool (FC)

Cập nhật: 2026-11-08. Đi kèm `PLAN_FC_clef_qwen35.md`, `PLAN_data_augment_v3.md`.

> **Cập nhật 2026-11-08:** E1 và phần fine-tune của E2 đã chạy xong, xem `REPORT_app_filter.md`. Tóm tắt: sau fine-tune, embedding đạt 92–95% (val_ood, k=4) và 95–98,5% (k=5) nhưng phải giữ 5,8–6,2 trên 9,2 app, chưa đạt mục tiêu ở mục 2. Đề xuất chuyển sang catalog hai tầng (top-k giữ mô tả đầy đủ, phần còn lại chỉ ghi tên tool) thay cho lọc cứng.

## 0. Vì sao lọc

- Mỗi sample truyền toàn bộ tool của app đã cài: trung bình 9,2 app, 32,5 tool (public test: 8,5 app, 30,7 tool). Phần catalog chiếm **khoảng 4.400 / 5.450 token (80%) của prefix**, và Q1 liệt kê lại toàn bộ danh sách ở mỗi bước.
- Phần lớn app không liên quan: **86% sample gọi 0–2 app** (76% gọi 1–2). Ví dụ `train_0001` cài 14 app / 49 tool nhưng chỉ cần 4 app / 11 tool.
- Lọc giúp giảm token khi train và khi suy luận, giảm nhầm sang app na ná (lỗi hay gặp: `bill_payment` thành `ewallet/banking`, `calendar` thành `weather`), và giảm thiên vị vị trí trong Q1.

## 1. Thiết kế

```
yêu cầu (transcript ASR) + lượt assistant trước
        │
        ▼
[bộ lọc] chấm điểm từng app ĐÃ CÀI  →  giữ top-k / trên ngưỡng
        ∪ app đã gọi trong history
        ∪ app cung cấp id cho app đã giữ (theo x-ref trong catalog, ví dụ sms → contacts)
        ∪ sys (ask_missing_info)
        │
        ▼
vòng Clef như cũ, catalog trong system prompt và Q1 chỉ gồm các app đã giữ
        + phương án "Cần app khác" trong Q1 → dựng lại với catalog đầy đủ (lối thoát)
```

- **Lọc theo app, không theo tool.** Giữ trọn tool của app được chọn, để không mất các tool tra cứu (`list_events`, `contacts.search`) mà người dùng không nói ra. Loại call này chiếm 17% số call.
- **Lọc một lần mỗi lượt**, trước bước 0, giữ nguyên qua các bước để prefix cache vẫn trúng.
- Chỉ chọn trong `installed_apps`. Gọi app chưa cài thì `ToolEnv` từ chối.

## 2. Baseline đã đo: BM25, không model

Tài liệu cho mỗi app = `name_vi` + mô tả app + mô tả các tool. Query = yêu cầu cuối + text assistant lượt trước. Script đo tạm, chưa đưa vào repo. Recall ở đây là tỉ lệ sample giữ được **đủ mọi** app vàng:

| top-k | val_in (128): recall / số app giữ | val_ood (259): recall / số app giữ |
|---|---|---|
| 3 + luật history/x-ref | 79,7% / 3,9 | 68,7% / 4,5 |
| 6 + luật history/x-ref | 94,5% / 6,4 | **83,0%** / 6,9 |

BM25 không đủ: phải giữ gần hết app (6–7 trên khoảng 9) mà vẫn sót 17% trên app lạ. Bài toán cần hiểu nghĩa: "mai mưa" → weather, "chạy bộ" → calendar, "trả tiền điện" → bill_payment. **Mục tiêu: recall ≥ 99% (val_in) và ≥ 98% (val_ood) khi giữ trung bình ≤ 4 app.**

## 3. Ràng buộc chọn model

- **Ngân sách tham số:** LLM 4,206B + PhoWhisper-large khoảng 1,55B ≈ 5,75B, nên còn dư **khoảng 0,25B** so với giới hạn 6B. Model lọc phải ≤ khoảng 0,2B. Tham số của nó tính vào tổng.
- Phát hành trước 26/10/2026, license cho phép phân phối trong Docker, chạy offline, cho kết quả tất định (fp32 trên CPU hoặc cố định trên GPU).
- Phải chịu được transcript ASR có lỗi, và chạy trong vài chục ms. CPU 16 lõi là đủ, nên GPU để dành cho ASR và LLM.

## 4. Ứng viên

| Ứng viên | Tham số | License | Ghi chú |
|---|---|---|---|
| **`intfloat/multilingual-e5-small`** | khoảng 118M (cần kiểm trong `config.json`) | MIT (thẻ HF) | Đa ngôn ngữ khoảng 100 thứ tiếng, vector 384 chiều, cần tiền tố `query:` / `passage:`. **Ứng viên chính**: nhỏ, không phải tách từ |
| `bkai-foundation-models/vietnamese-bi-encoder` | khoảng 135M (theo nguồn thứ ba) | Apache-2.0 | PhoBERT-base-v2, chuyên tiếng Việt, nhưng **input phải tách từ sẵn** (cần thêm bộ tách từ như pyvi trong Docker) và tối đa 256 token |
| `paraphrase-multilingual-MiniLM-L12-v2` | khoảng 118M | cần kiểm | Tiếng Việt 60,1 trên SEA-BED (e5-large: 66,4) |
| Qwen3-Embedding-0.6B, Harrier-OSS-0.6B, EmbeddingGemma-300M, bge-m3 | 0,3–0,6B | – | **Vượt ngân sách**, loại |
| **Chính LLM** (câu hỏi Clef có/không cho từng app) | 0 | – | Không thêm tham số. Thêm 1 request khoảng 1,5k token. Dùng chung loss có/không đang có. Là **đối chứng** cho embedding và có thể hợp với embedding |

Chưa xác minh ngày phát hành và số tham số chính xác của từng model theo nguồn chính thức (AGENT.md yêu cầu). Phải kiểm trước khi đưa vào Docker.

## 5. Model embedding: cách dùng và cách train

**Biểu diễn:**
- App (passage): `name_vi: mô tả app. Tool: <tên> – <mô tả>; …`. Tính sẵn cho catalog của split khi khởi động.
- Query: transcript yêu cầu cuối + text assistant ngay trước + transcript lượt user trước. Giữ phần cuối nếu quá dài.
- **Yêu cầu nhiều ý** ("nếu mai mưa thì dời…, nhắn…"): tách câu theo `,` `;` `rồi` `và` `thì`, tính điểm từng đoạn, lấy max cho mỗi app. Thử cả có và không tách.

**Chọn app giữ lại:** lấy top-k theo cosine, cộng mọi app có điểm ≥ ngưỡng, cộng các luật ở mục 1. Chọn k và ngưỡng trên val để đạt mục tiêu recall. Ưu tiên giữ thừa.

**Fine-tune** (contrastive, MultipleNegativesRankingLoss hoặc InfoNCE):
- Cặp dương: (yêu cầu → từng app vàng). Lấy từ lượt cuối và lượt history (lượt history dùng `audio_text` và call của lượt đó), chỉ split train của clef.
- **Âm khó:** các app đã cài khác trong **cùng sample**. Đây đúng là thứ phải phân biệt lúc test.
- **Để generalize sang app lạ:**
  - viết lại mô tả app và tool bằng LLM;
  - đổi tên app và tool;
  - thêm app tổng hợp của plan v3 (mục 2);
  - **app giữ lại của val_ood không bao giờ làm cặp dương**.
- **Nhiễu ASR:** thay một phần query bằng transcript ASR của chính PhoWhisper (plan v3 mục 4.3).
- Mẫu không gọi app nào (11%): không có cặp dương, nhưng vẫn dùng để chỉnh ngưỡng (giữ ít app là được).
- Quy mô nhỏ: khoảng 3–5k cặp, vài phút trên 1 GPU.

## 6. Thí nghiệm

| # | Thí nghiệm | Chỉ số |
|---|---|---|
| E0 | BM25 (đã làm, mục 2) | Mốc |
| E1 | e5-small, MiniLM, vietnamese-bi-encoder **zero-shot** | Recall theo số app giữ, trên val_in / val_ood. Đo với cả `audio_text` thật và transcript ASR v1 |
| E2 | Fine-tune ứng viên tốt nhất của E1 (mục 5) | Như E1 |
| E3 | LLM tự lọc: câu hỏi có/không cho từng app, zero-shot bằng adapter Clef P2 hiện có | Như E1 + độ trễ |
| E4 | Hợp E2 ∪ E3 | Recall có tăng đủ để bù thêm 1 request không |
| E5 | Đưa bộ lọc tốt nhất vào dữ liệu Clef: catalog đã lọc **+ 1–3 app nhiễu ngẫu nhiên** (đôi khi giữ catalog đầy đủ), thêm ví dụ "Cần app khác" khi app vàng bị lọc mất. Train (kết hợp định dạng ghép bước) | Điểm `metrics/` val_in / val_ood so với bản không lọc; số token/epoch; số lần đi vào lối thoát |
| E6 | Độ trễ end-to-end trên GPU gần A30 | Lọc + Clef ≤ ngân sách 30 s |

Tiêu chí giữ bộ lọc: E5 không giảm điểm val_ood (cho phép chênh trong mức nhiễu khoảng ±5), và giảm rõ token hoặc độ trễ. Nếu recall của embedding không đạt khoảng 98% trên val_ood thì dùng E3, hoặc nới k.

## 6b. Kế hoạch train bộ lọc (cập nhật 2026-11-08)

Dựa trên `REPORT_app_filter.md`. Lượt đã chạy dùng loss "tổng xác suất các dương", 3 epoch, lr 2e-5, batch 32, chưa có augment, chưa có nhiễu ASR, chỉ một lần giữ lại 5 app.

### Chọn k tạm thời

Tầng 1 lấy `k = min(số app cài, 5)` cộng luật history và `x-ref` (giữ trung bình khoảng 6 app). Lý do: ở val_ood sau fine-tune, k=3 đạt 75–87%, k=4 đạt 92–95%, k=5 đạt 95–98,5%, các app thêm vào sau 5 tốn khoảng 480 token catalog mà chỉ tăng vài điểm. Con số này chỉ dựa trên recall nên là tạm thời; chốt bằng E5 (mục 6). Số app cài từ 3 đến 18 (train: trung vị 9, p95 14; public: trung vị 8, p95 13), nên lợi ích lọc nằm chủ yếu ở nhóm cài ≥ 10 app.

### 1. Chuẩn bị dữ liệu

| Việc | Chi tiết |
|---|---|
| Bộ nhãn | Mỗi ví dụ: query, các app đã cài, app vàng. Nguồn: 1.006 lượt cuối và 1.965 lượt history có call (split train của clef) |
| Bỏ dương do luật tự thêm | App được luật `x-ref` thêm (ví dụ `contacts` khi có `sms`) không làm dương, vì câu nói không nhắc tới nó |
| Chia theo app | Chia app thành 3–4 nhóm (leave-app-out); mỗi lần giữ một nhóm ngoài train. Không dùng nhóm giữ lại làm dương **hay âm** |
| Tập hiệu chỉnh | Một fold riêng để chọn k, ngưỡng và epoch. Không dùng val_ood vừa chọn vừa báo cáo |
| Query nhiễu ASR | Đã có transcript PhoWhisper-large **zero-shot** cho 1.545 lượt cuối ở `data/results/asr/comparison/20261107-isolated/phowhisper-large-full-attempt1/` (chưa fine-tune). Các lượt user trong history (khoảng 3.000 audio) chưa có; cần chạy ASR nếu muốn nhiễu cho cả phần này. Nhớ ghi rõ model nào sinh transcript |
| Augment | Viết lại mô tả app/tool bằng LLM, đổi tên app, bỏ ngẫu nhiên một phần mô tả; app tổng hợp từ `PLAN_data_augment_v3.md` thêm sau |
| Luật | Không dùng public_test hay `*_environment/` (AGENT.md) |

### 2. Thí nghiệm (mỗi cấu hình 3 seed, đổi một yếu tố mỗi lượt)

| # | So sánh | Câu hỏi |
|---|---|---|
| T1 | Loss tổng xác suất dương (đang dùng) / loss từng dương / loss dương tệ nhất (margin) / BCE theo từng app | Có tăng recall mẫu nhiều app (cần ≥ 2 app) không |
| T2 | Dương đầy đủ / bỏ dương do luật tự thêm | Bớt nhãn khó đoán không |
| T3 | Bi-encoder / cross-encoder (cùng cỡ ~100–130M) | Cross-encoder (9 cặp mỗi mẫu) có đáng không |
| T4 | Có / không tách câu nhiều ý và lấy max theo đoạn | Giúp mẫu 3–4 app không |
| T5 | Có / không augment mô tả và nhiễu ASR | Chịu app lạ, lỗi ASR tốt hơn không |
| T6 | Chọn 1–2 model nền (e5-small, vietnamese-bi-encoder) | Chốt model |

Mốc giữ nguyên: lr 2e-5, 3 epoch, batch 32, InfoNCE trong tập app đã cài của mẫu. Chỉ dùng BCE (T1) nếu muốn k thích nghi theo ngưỡng xác suất.

### 3. Đánh giá và cổng chấp nhận

- Chỉ số: recall theo k, số app giữ, khoảng tin cậy Wilson; báo riêng theo nhóm số app cài (3–6, 7–10, 11+) và theo query sạch / nhiễu ASR.
- **Cổng 1 (chọn bản):** cận dưới khoảng tin cậy của recall ≥ 98% trên fold hiệu chỉnh ở k ≤ 5, và tụt ≤ 3 điểm khi chuyển sang query nhiễu ASR.
- **Cổng 2 (dùng trong hệ thống):** E5 — điểm `metrics/` val_ood không thấp hơn bản không lọc quá khoảng 1–2 điểm.
- Không qua cổng: chuyển sang catalog hai tầng (tầng 1 top-k mô tả đầy đủ, tầng 2 chỉ tên tool) và hạ yêu cầu recall tầng 1 xuống khoảng 90–95%.

### 4. Hạ tầng

- Kaggle T4 (kernel và dataset riêng tư như `asr/agent/clef/app_filter/`). Mỗi lượt 2–5 phút; bảng T1–T6 với 3 seed ước khoảng 2–3 giờ GPU, trong hạn mức 30 giờ/tuần/tài khoản.
- Bản cuối: train trên toàn bộ train (kể cả val_in), 3 seed, chọn bản ổn định, lưu trọng số fp32.
- Dùng vietnamese-bi-encoder thì phải thêm bộ tách từ (pyvi) vào Docker và đo lại độ trễ.

### 5. Trước khi đưa vào Docker

- Kiểm số tham số, license, ngày phát hành theo model card chính thức (AGENT.md).
- Chạy CPU fp32, hai lần cho cùng kết quả.
- Cộng tham số của model lọc vào tổng hệ thống (còn dư khoảng 0,25B).
- Chép mã, cấu hình, seed và kết quả vào repo cạnh `asr/agent/clef/app_filter/`.

### 6. Thứ tự và điểm dừng

1. Dựng dữ liệu, các fold, transcript nhiễu ASR.
2. T1–T2; dừng và giữ loss cũ nếu không tăng recall mẫu nhiều app.
3. T3–T6, chọn cấu hình.
4. Train bản cuối, đo theo cổng 1.
5. E5 trong vòng Clef (cổng 2), quyết định lọc cứng hay hai tầng.

## 7. Tích hợp vào Docker

- Model embedding chạy trên CPU fp32 (tất định), batch các app của sample: dưới khoảng 50 ms.
- Embedding catalog tính một lần khi container khởi động.
- Thư viện `transformers` thuần (mean pooling). Không cần `sentence-transformers` nếu muốn image gọn.
- Ghi số tham số thực của model lọc vào bảng tổng tham số của hệ thống.

## 8. Rủi ro

- **Recall trên app lạ:** đây là rủi ro lớn nhất. Bỏ sót app vàng gần như chắc chắn làm sai mẫu. Có ba lớp bảo vệ: luật (history, x-ref), giữ thừa app, và lối thoát "Cần app khác".
- **Lệch giữa train và test** nếu lúc train luôn dùng catalog lọc hoàn hảo. Bắt buộc dùng đầu ra thật của bộ lọc, kèm app nhiễu.
- **Ngân sách tham số** chỉ dư khoảng 0,25B. Nếu sau này đổi sang ASR lớn hơn thì phải bỏ bộ lọc embedding và dùng E3.
- `vietnamese-bi-encoder` cần bộ tách từ, nên thêm phụ thuộc và có thể sai khi transcript ASR có lỗi.
