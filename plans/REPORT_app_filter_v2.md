# Báo cáo lọc app v2: ngân sách tham số, model ứng viên, zero-shot trên train

Cập nhật: 2026-11-10. Nối tiếp `REPORT_app_filter.md` (2026-11-08, Kaggle: BM25 + 3 embedding, có fine-tune). Code:
`agent/clef/app_filter/filter.py` (BM25 ∥ embedding/reranker → hợp top-k + luật; CLI `eval` / `train`),
`app_keywords.json` (từ khoá đồng nghĩa 45 app, viết tay bản đầu). Thiết kế: `PLAN_v2_pipeline.md` §2.

## 1. Ngân sách tham số

| Thành phần | Tham số | Nguồn |
|---|---|---|
| Qwen3.5-4B bản cắt (bỏ vision + MTP) | 4.206B | `strip_qwen35.py`, `PLAN_FC_clef_qwen35.md` |
| PhoWhisper-large | 1.543B | kiến trúc = whisper-large (d_model 1280, 32+32 lớp); `openai/whisper-large-v2` safetensors 1,543,304,960 |
| **Tổng hiện tại** | **5.749B** | |
| **Còn lại dưới 6B** | **≤ 251M** | model lọc app phải ≤ ~240M để có dư cho tokenizer/head |

## 2. Model ứng viên (số tham số từ metadata safetensors trên HF, kiểm 2026-11-10)

### 2.1 Vừa ngân sách (≤ 250M)

| Model | Tham số | Loại | License | Ghi chú |
|---|---|---|---|---|
| `bkai-foundation-models/vietnamese-bi-encoder` | 135M | bi-encoder (PhoBERT) | Apache-2.0 | cần tách từ `pyvi`; đã fine-tune thử (REPORT cũ: tốt nhất val_ood) |
| `intfloat/multilingual-e5-small` | 118M | bi-encoder | MIT | tiền tố `query:`/`passage:` |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 118M | bi-encoder | Apache-2.0 | nhanh nhất trong nhóm transformer |
| `sentence-transformers/distiluse-base-multilingual-cased-v2` | 135M | bi-encoder | Apache-2.0 | |
| `dangvantuan/vietnamese-embedding` | 135M | bi-encoder (PhoBERT) | Apache-2.0 | cần `pyvi` |
| `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base` | 136M | bi-encoder (PhoBERT) | **không ghi** | cần `pyvi`; license phải hỏi trước khi dùng |
| `keepitreal/vietnamese-sbert` | ~135M (không có safetensors) | bi-encoder (PhoBERT) | **không ghi** | như trên |
| `minishlab/potion-multilingual-128M` | 128M | static embedding (model2vec) | MIT | không có transformer, ~0.1 ms/câu; ứng viên cho máy chấm yếu |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 118M | cross-encoder (reranker) | Apache-2.0 | chấm từng cặp (câu, app): 9 cặp/mẫu |

### 2.2 Vượt ngân sách (chỉ để tham chiếu)

| Model | Tham số | Vượt |
|---|---|---|
| `intfloat/multilingual-e5-base`, `paraphrase-multilingual-mpnet-base-v2`, `hiieu/halong_embedding`, `BAAI/bge-reranker-base`, `jinaai/jina-reranker-v2-base-multilingual` (CC-BY-NC) | 278M | +27M → tổng 6.03B |
| `google/embeddinggemma-300m` (license Gemma), `Alibaba-NLP/gte-multilingual-base`, `gte-multilingual-reranker-base` | 303–306M | +55M |
| `nomic-embed-text-v2-moe`, `LaBSE` | 471–475M | |
| `BAAI/bge-m3`, `AITeamVN/Vietnamese_Embedding`, `AITeamVN/Vietnamese_Reranker`, `namdp-ptit/ViRanker`, `multilingual-e5-large`, `Qwen3-Embedding-0.6B` | 560–596M | tổng ~6.3B |

bge-m3 được đo zero-shot để biết "trần" của model lớn, không dùng được trong bài nộp.

## 3. Thiết lập zero-shot

- Dữ liệu: 1545 mẫu train, chia theo `agent/clef/splits.json` (train 1158 / val_in 128 / val_ood 259; app giữ lại: alarm,
  ewallet, flights, loyalty, podcast_radio). Mẫu không gọi app nào bỏ khỏi recall (còn train 1006, val_in 109, val_ood 259).
- Ứng viên = `installed_apps` của chính mẫu (TB 9.2). **Recall** = tỉ lệ mẫu giữ đủ **mọi** app vàng; "app giữ" = số app
  trung bình sau lọc (tính trên mọi mẫu).
- Tài liệu app = name_vi + mô tả app + tên/mô tả tool + mô tả tham số + enum + **từ khoá đồng nghĩa**. Query = câu cuối +
  câu user cuối trong history. BM25 trên âm tiết có dấu + không dấu + bigram.
- Luật luôn áp: + app đã gọi trong history, + đóng bao `x-ref` (giữ `sms` → kéo `contacts`).
- Hợp: top-k₁ BM25 ∪ top-k₂ embedding (∪ ngưỡng tương đối nếu có). Transcript là `audio_text` sạch (chưa qua ASR).
- Máy: CPU local (embedding batch), số ms/câu đo khi mã hoá batch 1545 câu.

## 4. Kết quả

### 4.1 Zero-shot (recall % val_in / val_ood; app giữ TB val_ood ≈ 6.0–7.2; mẫu có app vàng: val_in 109, val_ood 259)

| Model (tham số) | Chỉ embedding top-5 | BM25 top-3 ∪ emb top-4 | BM25 top-4 ∪ emb top-5 | ms/câu CPU |
|---|---|---|---|---|
| bge-m3 (568M, vượt quota) | 87.2 / 89.2 | 90.8 / 90.3 | 95.4 / 96.5 | 112 |
| paraphrase-multilingual-MiniLM-L12 (118M) | 80.7 / 89.6 | 85.3 / 89.6 | 92.7 / 96.1 | 10 |
| multilingual-e5-small (118M) | 85.3 / 80.3 | 87.2 / 91.1 | 92.7 / 95.4 | 11 |
| distiluse-multilingual (135M) | 85.3 / 83.0 | 89.0 / 87.6 | 92.7 / 94.2 | 17 |
| potion-multilingual-128M (static, MIT) | 86.2 / 82.6 | 91.7 / 89.2 | 93.6 / 94.2 | ~0 |
| vietnamese-bi-encoder (135M) | 82.6 / 78.4 | 89.0 / 90.3 | **96.3** / 96.1 | 28 |
| dangvantuan/vietnamese-embedding (135M) | 82.6 / 80.3 | 87.2 / 88.4 | 90.8 / 95.4 | 26 |
| keepitreal/vietnamese-sbert (~135M, **license không ghi**) | 83.5 / 81.8 | 90.8 / 88.0 | 91.7 / 96.5 | 27 |
| mmarco cross-encoder (118M, reranker) | 78.9 / 70.3 | 83.5 / 87.6 | 91.7 / 92.7 | 476 (9 cặp/câu) |
| VoVanPhuc SimCSE-PhoBERT (136M) | không đo, xem §4.4 | | | |

Sai số chuẩn ≈ ±2.4 điểm (val_in) và ±1.4 điểm (val_ood): các model ≤ 135M **không phân biệt được** ở zero-shot.
BM25 + từ khoá (top-5: 90.8 / 93.4; top-8: 98.2 / 99.6) **không so sánh được với embedding**: từ khoá do người viết
khi đã biết cả 5 app held-out nên val_ood của BM25 lạc quan.

### 4.2 Sau fine-tune vietnamese-bi-encoder (3 epoch, 2971 ví dụ, bỏ app held-out khỏi tập ứng viên; chạy trên myserver)

| Cấu hình | val_in | val_ood | app giữ TB (val_in / val_ood) |
|---|---|---|---|
| chỉ embedding top-5 (zero-shot → fine-tune) | 82.6 → **92.7** | 78.4 → **86.9** | 5.7 / 6.1 |
| BM25 top-3 ∪ emb top-4 | 89.0 → 95.4 | 90.3 → 94.2 | 5.5 / 6.0 |
| BM25 top-4 ∪ emb top-5 | 96.3 → **97.2** | 96.1 → **98.1** | 6.5 / 6.9 |
| BM25 top-4 ∪ emb top-5 ∪ điểm ≥ 0.5 khoảng | 97.2 | 98.8 | 6.5 / 7.1 |
| BM25 top-3 ∪ emb top-4 ∪ điểm ≥ 0.5 khoảng | 95.4 | 97.7 | 5.6 / 6.2 |

Cột "train" không dùng được vì model đã học trên đó. Fine-tune nâng mạnh embedding một mình (+10 / +8.5 điểm) nhưng
**khi hợp với BM25 chỉ thêm ~1–2 điểm** (nằm trong sai số của val_in).

### 4.3 Kết luận

- **Không cấu hình nào đạt "recall ≥ 99% với ≤ 4.5 app".** Tốt nhất khoảng 98% ở ~6.9/9.2 app. Chọn model không giải quyết được:
  mẫu sót là câu có app **ngầm hiểu** ("mở họp hội phụ nữ" → chat_ott; "đặt bún bò, mỗi người một phần" → chat_ott;
  "theo dõi kênh đó" → reminders) và câu 4–5 app. Cần cách dùng khác: **catalog hai tầng** (app giữ có schema đầy đủ, các
  app còn lại chỉ ghi tên tool) thay vì lọc cứng.
- **Chọn: `bkai-foundation-models/vietnamese-bi-encoder` (135M, Apache-2.0) fine-tune, hợp BM25.** Tổng tham số
  5.749B + 0.135B = **5.884B < 6B** (dư 116M). Lý do: tốt nhất val_in và val_ood sau fine-tune trong nhóm vừa quota, đồng
  thời tốt nhất val_in ở zero-shot hybrid; chuyên tiếng Việt (PhoBERT) nên hợp với câu nói có lỗi ASR. Nhược điểm: cần `pyvi`
  trong Docker, 28 ms/câu CPU (không đáng kể so với ngân sách 30 s). MiniLM-L12 / e5-small chưa fine-tune trong pipeline này
  (lần fine-tune cũ trên Kaggle: ngang vietnamese-bi-encoder trong sai số) nên là phương án dự phòng nếu `pyvi` gây khó.
- `potion-multilingual-128M` đáng chú ý: zero-shot hybrid ngang vietnamese-bi-encoder (pooled 97.3% cả hai), 0 ms/câu,
  không cần torch/pyvi; chưa thử fine-tune.

### 4.4 Còn thiếu

- **mmarco cross-encoder** (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`): đã chạy lại xong (lần đầu bị dừng), kết quả ở
  §4.1: **kém nhất nhóm** (chỉ reranker top-5: 78.9 / 70.3; hợp BM25 top-4 ∪ top-5: 91.7 / 92.7) và chậm nhất (476 ms/câu
  CPU vì phải chấm 9 cặp câu–app). Loại khỏi danh sách ứng viên.
- **VoVanPhuc SimCSE-PhoBERT**: không đo, vì không có license trên HF và file bị chặn (proxy trả 401). Không đưa vào bài nộp
  khi chưa xác minh license.
- Chưa fine-tune MiniLM-L12 / e5-small / potion trong pipeline này (chỉ có số từ lần Kaggle cũ) → làm ở bước F3 của
  `PLAN_app_filter_finetune.md` nếu `pyvi` gây khó cho Docker.
- Chưa đo recall trên bản ASR (query có lỗi) và trên `val_alias`; chưa đo end-to-end điểm BTC với bộ lọc (probe A0 trong
  `PLAN_v2_pipeline.md`).
