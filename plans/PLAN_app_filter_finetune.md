# PLAN: fine-tune model lọc app (≤ 250M) cho pipeline Clef v2

Ngày: 2026-11-10. Đầu vào: `REPORT_app_filter_v2.md` (ngân sách, ứng viên, zero-shot), `REPORT_app_filter.md`
(fine-tune lần 1 trên Kaggle). Vị trí trong hệ thống: `PLAN_v2_pipeline.md` §2 (trụ A). Code: `agent/clef/app_filter/filter.py`.

## 0. Mục tiêu và ràng buộc

- **Recall ≥ 99%** app vàng trên val_in, val_ood (và val_alias khi có) với **số app giữ TB ≤ 4.5** (hiện zero-shot cần
  ~7/9.2 app để tới 99%). Recall là tỉ lệ mẫu giữ đủ mọi app vàng; sót một app = mẫu đó hỏng không cứu được.
- Model ≤ 250M tham số (tổng hệ thống < 6B), license cho phép dùng thương mại (Apache/MIT), chạy offline trong Docker,
  ≤ 100 ms/câu trên CPU.
- Không dùng gì từ `public_test/` ngoài catalog tool (chỉ để thiết kế tài liệu app, không đưa vào train).

## 1. Chọn model

**Đã chọn (2026-11-10): `bkai-foundation-models/vietnamese-bi-encoder` (135M), fine-tune + hợp BM25** — số liệu ở
`REPORT_app_filter_v2.md` §4. Tổng hệ thống 5.884B < 6B. Đã fine-tune thử 1 lần (3 epoch): BM25 top-4 ∪ emb top-5 đạt
val_in 97.2% / val_ood 98.1% với ~6.9/9.2 app. Mục tiêu §0 (99% với ≤ 4.5 app) **không đạt được bằng cách chọn model**;
cần catalog hai tầng (§4). Dự phòng: MiniLM-L12 / e5-small (không cần pyvi). Các bước F1–F6 dưới đây vẫn áp dụng cho
model đã chọn; ba ứng viên không cần fine-tune lại từ đầu nữa.

Ghi chú lựa chọn ban đầu: chọn theo zero-shot (REPORT v2 §4) **và** kết quả fine-tune lần 1: ứng viên chính `bkai-foundation-models/vietnamese-bi-encoder`
(135M, Apache; fine-tune lần 1 đạt 98.5% ở k=5 val_ood), đối chứng `intfloat/multilingual-e5-small` (118M, MIT, không cần pyvi)
và `paraphrase-multilingual-MiniLM-L12-v2` (118M, nhanh nhất). Fine-tune cả ba cùng recipe, chọn theo tiêu chí §0 trên val;
nếu ngang nhau (±2 điểm, val_in chỉ 109 mẫu) thì chọn model **không cần pyvi** và nhanh hơn.
`potion-multilingual-128M` (static) chỉ giữ nếu zero-shot của nó không thua transformer quá 5 điểm; nó không fine-tune theo
cách dưới (không có encoder), nhưng có thể huấn luyện lớp tuyến tính trên embedding tĩnh.

## 2. Dữ liệu

| Nguồn | Số ví dụ (ước) | Ghi chú |
|---|---|---|
| Lượt cuối split train có app vàng | ~1000 | query = câu cuối + câu user cuối history |
| Lượt history (user → assistant có tool_calls) | ~2000 | như fine-tune lần 1 |
| **Biến thể phiên âm** `synth_phonetic/train_phonetic.jsonl` (#ph1/#ph2) | ~900 + history | để chịu lỗi ASR: "ét em ét", "Da lô" |
| **Alias / iface** `synth/train_alias.jsonl`, `train_iface.jsonl` | ~1800 | tên app/tool đổi → tài liệu app đổi theo overlay, model học bám mô tả không bám tên |
| **B2 / test-like** khi Gemini sinh xong | ~3000 | app mới, tài liệu kèm từ khoá |
| **Bản ASR của audio train** (PhoWhisper trên train, hợp lệ) | ~1500 | query thật có lỗi ASR; label = app vàng của mẫu |

Luật tách: **5 app held-out (val_ood) bị loại khỏi mọi tập ứng viên lúc train** (như lần 1), để val_ood đo app lạ. Thêm
`val_alias` (128) để đo bám tên. Tài liệu app lúc train = tài liệu lúc infer (name_vi + mô tả + tool + tham số + từ khoá);
với mẫu alias dùng overlay của mẫu đó.

Tăng cường lúc train (mỗi epoch rút lại):
- Rụng từ khoá: bỏ ngẫu nhiên 30% từ khoá đồng nghĩa khỏi tài liệu app → không phụ thuộc từ khoá viết tay.
- Nhiễu ASR giả: bỏ dấu 20% âm tiết, thay 5% âm tiết bằng âm gần (s/x, ch/tr, d/gi/r, l/n), ghép/tách từ nước ngoài.
- Đổi tên app/tool trong tài liệu theo `alias_pool.json` với p = 0.5.

## 3. Loss và huấn luyện

- **InfoNCE nhiều dương** trên tập ứng viên của chính mẫu (như lần 1): `loss = logsumexp(sim[cand]) − logsumexp(sim[gold])`,
  scale 20, cosine. Đây đúng là phân phối lúc infer (chỉ chọn trong app đã cài).
- **Thêm âm tính khó**: với mỗi mẫu, thêm 2 app ngoài `installed_apps` gần nghĩa nhất theo BM25 (ví dụ banking cho câu ví)
  vào tập ứng viên lúc train → model phân biệt ví/ngân hàng, SMS/OTT, nhắc việc/báo thức.
- **Trọng số mẫu nhiều app**: mẫu có ≥ 3 app vàng nhân trọng số 2 (đây là nhóm sót nhiều nhất).
- AdamW lr 2e-5 (encoder), batch 32, 3–5 epoch, warmup 10%, cosine; seed 42 và 2 seed khác để biết nhiễu.
- Chạy trên GPU myserver (`/home/hungpv/miniconda3/envs/vllm`, A100): ~3 phút/epoch/model. Lệnh:
  `python filter.py train --base <model> --out <dir> [--epochs 4]` (cần thêm cờ cho §2–3: `--phonetic`, `--synth`,
  `--asr-queries`, `--hard-neg 2`, `--kw-dropout 0.3`, `--asr-noise`).

## 4. Luật chọn app lúc infer (fit trên train, kiểm trên val)

`keep = top-k_emb(emb) ∪ top-k_bm25(BM25) ∪ {app: emb ≥ r·(max−min)+min} ∪ history ∪ x-ref`. Quét k_emb ∈ {2,3,4},
k_bm25 ∈ {0,2,3}, r ∈ {0, 0.5, 0.6, 0.7}; chọn cấu hình **nhỏ nhất** đạt recall ≥ 99% trên train rồi báo val_in/val_ood/
val_alias. Có hai cách dùng sau lọc, quyết định ở PLAN_v2 §2.3–2.5 (probe A0):
- lọc cứng (chỉ app giữ + decoy vào prompt), hoặc
- hai tầng: app giữ có schema đầy đủ, app còn lại chỉ ghi tên (giữ recall 100% về khả năng gọi, bớt ~70% token).

## 5. Đánh giá và tiêu chí chấp nhận

| Chỉ số | Mục tiêu |
|---|---|
| Recall val_in / val_ood / val_alias | ≥ 99% / ≥ 98% / ≥ 98% |
| App giữ TB | ≤ 4.5 (lọc cứng) |
| Recall trên bản ASR của audio val | không thấp hơn bản text quá 1 điểm |
| ms/câu CPU (Xeon máy chấm) | ≤ 100 |
| Điểm BTC end-to-end với lọc (A0/A2 trong PLAN_v2) | không giảm so với không lọc; token/bước giảm ≥ 50% |

Sót còn lại được liệt kê theo app và theo kiểu câu (`eval_*.json → miss`) để bổ sung từ khoá hoặc dữ liệu.

## 6. Việc

| # | Việc | Phụ thuộc |
|---|---|---|
| F1 | Thêm vào `filter.py train`: dữ liệu phonetic/synth/ASR, âm tính khó, trọng số, tăng cường (§2–3); `eval` thêm `val_alias` và bản ASR | – |
| F2 | Tạo bản ASR của audio train bằng PhoWhisper (hợp lệ, là audio train) → `data/results/asr/train_phowhisper.json` | GPU myserver, ~20 phút |
| F3 | Fine-tune 3 ứng viên × 3 seed trên myserver; quét luật §4; bảng kết quả vào REPORT v2 §5 | F1, F2 |
| F4 | Chọn model; đóng gói (model + từ khoá + luật + k) thành `AppFilter.load(dir)`; đo ms/câu CPU | F3 |
| F5 | Đưa vào `build_data.py --apps filter` và `clef_agent.py` (PLAN_v2 A2); probe A0 | F4 |
| F6 | Khi có B2: train lại với app mới, đo `val_novel` | Gemini |
