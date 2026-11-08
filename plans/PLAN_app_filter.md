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
