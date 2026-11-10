# PLAN: fine-tune model lọc app (≤ 250M) cho pipeline Clef v2

Ngày: 2026-11-10. Đầu vào: `REPORT_app_filter_v2.md` (ngân sách, ứng viên, zero-shot), `REPORT_app_filter.md`
(fine-tune lần 1 trên Kaggle). Vị trí trong hệ thống: `PLAN_v2_pipeline.md` §2 (trụ A). Code: `agent/clef/app_filter/filter.py`.

## 0. Mục tiêu và ràng buộc

- Bộ lọc dùng để **xếp hạng app cho catalog hai tầng** (PLAN_v2 §2.3), không lọc cứng: tầng 1 (app giữ) có schema đầy đủ,
  tầng 2 chỉ tên tool. Mục tiêu: **recall tầng 1 ≥ 98%** trên val_in, val_ood, val_alias với **≤ 7/9.2 app**; mẫu sót tầng 1
  không hỏng, chỉ thiếu mô tả tham số ở system prompt (thẻ tool bù lại sau Q1). Recall = tỉ lệ mẫu giữ đủ mọi app vàng.
- Model ≤ 250M tham số (tổng hệ thống < 6B), license cho phép dùng thương mại (Apache/MIT), chạy offline trong Docker,
  ≤ 100 ms/câu trên CPU.
- Không dùng gì từ `public_test/` ngoài catalog tool (chỉ để thiết kế tài liệu app, không đưa vào train).

## 1. Chọn model

**Đã chọn (2026-11-10): `bkai-foundation-models/vietnamese-bi-encoder` (135M), fine-tune + hợp BM25** — số liệu ở
`REPORT_app_filter_v2.md` §4. Tổng hệ thống 5.884B < 6B. Đã fine-tune thử 1 lần (3 epoch): BM25 top-4 ∪ emb top-5 đạt
val_in 97.2% / val_ood 98.1% với ~6.9/9.2 app. Lọc cứng 99% với ≤ 4.5 app **không đạt được bằng cách chọn model** (đo 2026-11-10), nên §0 đặt theo catalog hai tầng. Dự phòng: MiniLM-L12 / e5-small (không cần pyvi). Các bước F1–F6 dưới đây vẫn áp dụng cho
model đã chọn; ba ứng viên không cần fine-tune lại từ đầu nữa.

Ghi chú lựa chọn ban đầu: chọn theo zero-shot (REPORT v2 §4) **và** kết quả fine-tune lần 1: ứng viên chính `bkai-foundation-models/vietnamese-bi-encoder`
(135M, Apache; fine-tune lần 1 đạt 98.5% ở k=5 val_ood), đối chứng `intfloat/multilingual-e5-small` (118M, MIT, không cần pyvi)
và `paraphrase-multilingual-MiniLM-L12-v2` (118M, nhanh nhất). Fine-tune cả ba cùng recipe, chọn theo tiêu chí §0 trên val;
nếu ngang nhau (±2 điểm, val_in chỉ 109 mẫu) thì chọn model **không cần pyvi** và nhanh hơn.
`potion-multilingual-128M` (static) chỉ giữ nếu zero-shot của nó không thua transformer quá 5 điểm; nó không fine-tune theo
cách dưới (không có encoder), nhưng có thể huấn luyện lớp tuyến tính trên embedding tĩnh.

## 2. Dữ liệu

### 2.0 Dùng gì trước tiên (thứ tự đưa vào train)

| Đợt | Nguồn | Số ví dụ (đếm 2026-11-10, split train, đã bỏ app held-out) | Vì sao |
|---|---|---|---|
| **1 — bắt buộc, có ngay** | Lượt cuối train có app vàng | **1006** | query thật + nhãn thật; đây là phân phối lúc infer |
| | Lượt history (user → assistant có tool_calls) | **1965** | gần gấp đôi dữ liệu, cùng kiểu câu; đã dùng ở lần fine-tune 1 (tổng 2971) |
| **2 — thêm ngay sau đợt 1** | Biến thể phiên âm `synth_phonetic/train_phonetic.jsonl` (#ph1/#ph2) | 898 (+ history của chúng) | query có "ét em ét", "Da lô", "Mô mô" → bộ lọc chịu lỗi ASR; nhãn giữ nguyên |
| | Alias `synth/train_alias.jsonl` + iface `synth/train_iface.jsonl` | 1408 + 382 | tên app/tool đổi → **tài liệu app lấy từ `catalog_overlay` của mẫu**; model học bám mô tả, không bám tên |
| **3 — khi có** | Bản ASR PhoWhisper của audio train (F2) | ~1500 | query thật có lỗi ASR, nhãn = app vàng của mẫu |
| | B2 / test-like của Gemini | ~3000 | app mới; cần kèm từ khoá app |
| | **Mô tả app viết lại** (Gemini, `synth/app_desc_paraphrase.json`): 2–3 bản/app cho app train + B2, giữ nghĩa, đổi diễn đạt và thứ tự | ~45×3 + 150×3 đoạn | đa dạng bên tài liệu (§3.2 #1); mỗi epoch rút ngẫu nhiên một bản |

Đợt 1 là bản đã chạy (fine-tune lần 2 hôm nay: val_in 97.2 / val_ood 98.1 ở hybrid top-4∪5). Đợt 2 chỉ cần thêm cờ
`--phonetic`/`--synth` vào `filter.py train` (F1), không cần dữ liệu mới. Val giữ nguyên: val_in (109 mẫu có app vàng),
val_ood (259), thêm `val_alias` (128) để đo bám tên.

### 2.1 Một ví dụ train cụ thể (`train_0011`, lượt cuối)

- **Query** (câu cuối; không có câu user trước): *"Nếu ví đủ thì trả điện và nhắn Bố là đã đóng, còn không thì trả ngân
  hàng và nhắc tôi tám giờ tối nay nạp tiền vào ví nhé."*
- **Ứng viên** = 17 app đã cài của mẫu: `bill_payment, ewallet, banking, contacts, sms, reminders, calendar, weather, alarm,
  fuel_price, smart_home, ride_hailing, gold_price, food_delivery, phone, music, tv` (lúc train, app held-out `ewallet`,
  `alarm` bị **loại khỏi ứng viên** → còn 15).
- **Dương (gold)** = app của `label.tool_calls`: `bill_payment, ewallet, reminders` (→ sau khi loại held-out còn
  `bill_payment, reminders`; `contacts`/`sms` của nhánh "nhắn Bố" không nằm trong gold vì nhánh đó không xảy ra — đây là
  nhiễu nhãn chấp nhận được, luật x-ref lúc infer vẫn kéo `contacts` theo `sms`).
- **Âm** = 13 app còn lại, trong đó **âm tính khó** là `banking` ("trả ngân hàng" có trong câu) và `sms` — đúng cặp dễ nhầm.
- **Tài liệu app** (bên "passage"), ví dụ `ewallet` (952 ký tự, một chuỗi): `Ví điện tử Ví điện tử Ví điện tử: nạp, chuyển,
  thanh toán QR, nạp điện thoại ví ví điện tử Momo ZaloPay ví Momo nạp tiền nạp điện thoại nạp card quét mã QR trả bằng ví
  chuyển ví số dư ví get balance Số dư ví transfer Chuyển tiền ví tới liên hệ hoặc số điện thoại. Bắt buộc một trong:
  contact_id hoặc phone; … pay qr Thanh toán QR của một quán/merchant bằng ví …` = name_vi ×2 + mô tả app + từ khoá
  + (tên tool, mô tả, tên và mô tả tham số, enum) cho từng tool. Với PhoBERT (vietnamese-bi-encoder) chuỗi này được tách
  từ bằng `pyvi` rồi cắt ở 256 token.
- **Một bản ghi train** = `(query, [15 app ứng viên], {bill_payment, reminders})`. Với mẫu phonetic, query thành *"… nhắn
  Bố …"* giữ nguyên nhưng các từ nước ngoài đổi phiên âm; với mẫu alias, tài liệu `ewallet` lấy từ overlay có tên khác.

### 2.2 Bảng nguồn đầy đủ

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

### 3.1 Cách fine-tune (đang có trong `filter.py train`)

1. **Mã hoá**: `q = normalize(encoder(query))`, `d_a = normalize(encoder(doc_a))` cho mọi app `a` xuất hiện trong batch
   (mỗi app mã hoá một lần/batch). Cả hai bên cùng một encoder (bi-encoder), gradient chảy vào cả query và tài liệu.
2. **Điểm**: `sim[a] = 20 · ⟨q, d_a⟩` (cosine × scale 20).
3. **Loss cho một bản ghi** (InfoNCE nhiều dương, softmax **chỉ trên ứng viên của chính mẫu**):
   `loss = logsumexp_{a ∈ cand} sim[a] − logsumexp_{a ∈ gold} sim[a]`
   = −log( Σ_gold exp(sim) / Σ_cand exp(sim) ). Với `train_0011`: đẩy `bill_payment`, `reminders` lên trên 13 app còn lại,
   đặc biệt phải vượt `banking`, `sms`. Không so với app ngoài tập cài, vì lúc infer cũng chỉ chọn trong app đã cài.
4. **Tối ưu**: AdamW lr 2e-5, weight decay 0.01, batch 32 bản ghi, warmup 10% rồi giảm tuyến tính, clip grad 1.0,
   3 epoch, seed 42. Trên A100: ~50 s/epoch cho 2971 ví dụ (lần chạy hôm nay: loss 0.40 → 0.08 → 0.03).
5. **Lưu**: `model.save(out)` dạng sentence-transformers + `finetune.json` (base, siêu tham số, app held-out đã loại).
   Lúc eval/infer `Embedder` đọc `finetune.json` để biết base → có tách từ pyvi hay thêm tiền tố e5 không.

Lệnh (trên myserver, env `vllm`, dữ liệu ở `/mnt/hungpv/tmp_appfilter`):
```
python filter.py train --data /mnt/hungpv/tmp_appfilter --splits splits.json \
    --base bkai-foundation-models/vietnamese-bi-encoder --out ft_vibi --epochs 3 --lr 2e-5 --bs 32 --scale 20
python filter.py eval  --data /mnt/hungpv/tmp_appfilter --splits splits.json --emb ./ft_vibi --device cuda \
    --k-bm25 0 3 4 --k-emb 3 4 5 --rel 0 0.5 --out eval_ft_vibi.json
```
Loss ở epoch 3 = 0.03 trên 2971 ví dụ là dấu hiệu **gần như thuộc tập train**; vì vậy không được đọc cột "train" của
eval, và các mục §2.0 đợt 2–3 + §3.2 (tăng cường, âm tính khó) có vai trò chống thuộc lòng.

### 3.1b Vì sao bản đang chạy chưa đủ generalize (chẩn đoán 2026-11-10)

- Bên tài liệu chỉ có **40 đoạn** (40 app train) lặp suốt 3 epoch, loss về 0.03 → model có thể nhớ 40 "điểm neo" thay vì
  học đọc mô tả. val_ood chỉ kiểm 5 app lạ (emb-only 78.4 → 86.9 sau fine-tune: có generalize, nhưng ít bằng chứng); test có
  11 app lạ với mô tả viết kiểu khác.
- Âm mỗi bản ghi chỉ là ~9–15 app đã cài, đa số khác xa nghĩa; cặp khó (ví/ngân hàng, SMS/OTT, nhắc việc/báo thức/lịch)
  chỉ gặp khi tình cờ cùng được cài.
- Chưa có phép thử bám tên (`val_alias`).

### 3.2 Bổ sung để generalize (thứ tự tác dụng; 2–4 làm ngay không cần dữ liệu mới)

| # | Việc | Tác dụng | Cần |
|---|---|---|---|
| 1 | **Đa dạng bên tài liệu**: B2 (~150 app) → ~200 đoạn; trước khi có B2: alias/iface với tài liệu từ overlay (1790 mẫu) + **2–3 bản viết lại mô tả** mỗi app (Gemini, `synth/app_desc_paraphrase.json`), mỗi epoch rút ngẫu nhiên một bản | chống nhớ đoạn, học đọc mô tả | Gemini (C1) |
| 2 | **Thêm âm**: (a) âm trong batch kiểu MNRL — mọi app của các bản ghi khác trong batch (~40 âm); (b) **2 âm tính khó ngoài tập cài** = app gần nghĩa nhất theo BM25/embedding (banking cho câu ví, chat_ott cho câu SMS, alarm/calendar cho câu nhắc). Loss = InfoNCE trên tập cài (hạng chính) + 0.5·(MNRL trong batch) + 0.5·(InfoNCE có âm khó) | phân biệt cặp dễ nhầm | code |
| 3 | **Chống thuộc lòng**: 1–2 epoch hoặc **dừng sớm theo val_ood emb-only** (không theo train loss); rụng 30% từ khoá; nhiễu ASR giả phía query (bỏ dấu 20% âm tiết, thay 5% âm gần: s/x, ch/tr, d/gi/r, l/n); tuỳ chọn đóng băng 6 lớp dưới PhoBERT | giữ khả năng đọc | code |
| 4 | **Đo đúng**: thêm `val_alias` (đổi tên app/tool) vào eval — recall rớt nhiều so với val_in = đang nhớ tên; **k-fold theo app**: luân phiên 5 app held-out khác (4 fold) để đo app lạ trên ~20 app thay vì 5 | biết thật sự generalize không | code |
| 5 | Giữ **BM25 hợp cùng** làm lưới an toàn: khớp từ khoá/mô tả app lạ bất kể embedder | — | đã có |

Tiêu chí sau khi làm 2–4 (chạy lại trên GPU myserver ~5 phút): val_ood emb-only > 86.9; val_alias không thấp hơn val_in
quá 3 điểm; recall k-fold trung bình báo cáo cùng độ lệch chuẩn giữa các fold.

### 3.3 Bổ sung so với bản đang chạy (chi tiết)

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
k_bm25 ∈ {0,2,3}, r ∈ {0, 0.5, 0.6, 0.7}; chọn cấu hình **nhỏ nhất** đạt recall ≥ 98% trên train rồi báo val_in/val_ood/
val_alias. Cách dùng sau lọc: **catalog hai tầng** (PLAN_v2 §2.3) — app giữ có schema đầy đủ, app còn lại chỉ ghi tên tool; giữ
100% khả năng gọi, bớt ~⅔ token. Probe A0 (PLAN_v2 §2.5) đo xem danh sách ngắn có tăng accuracy không.

## 5. Đánh giá và tiêu chí chấp nhận

| Chỉ số | Mục tiêu |
|---|---|
| Recall tầng 1 val_in / val_ood / val_alias | ≥ 98% / ≥ 98% / ≥ 97% |
| App giữ TB (tầng 1) | ≤ 7 |
| Recall trên bản ASR của audio val | không thấp hơn bản text quá 1 điểm |
| ms/câu CPU (Xeon máy chấm) | ≤ 100 |
| Điểm BTC end-to-end với lọc (A0/A2 trong PLAN_v2) | không giảm so với không lọc; token/bước giảm ≥ 50% |

Sót còn lại được liệt kê theo app và theo kiểu câu (`eval_*.json → miss`) để bổ sung từ khoá hoặc dữ liệu.

## 6. Việc

| # | Việc | Phụ thuộc |
|---|---|---|
| F1 | `filter.py train`: dữ liệu phonetic/synth (tài liệu từ overlay), **âm trong batch (MNRL) + 2 âm tính khó ngoài tập cài**, rụng từ khoá, nhiễu ASR giả, dừng sớm theo val_ood emb-only (§3.2 #2–3); `eval` thêm `val_alias` + **k-fold theo app** (§3.2 #4) | – |
| F1b | Chạy lại fine-tune vietnamese-bi-encoder với F1 trên myserver (~5 phút), so với bản hiện tại theo tiêu chí §3.2 | F1 |
| F2 | Tạo bản ASR của audio train bằng PhoWhisper (hợp lệ, là audio train) → `data/results/asr/train_phowhisper.json` | GPU myserver, ~20 phút |
| F3 | Fine-tune 3 ứng viên × 3 seed trên myserver; quét luật §4; bảng kết quả vào REPORT v2 §5 | F1, F2 |
| F4 | Chọn model; đóng gói (model + từ khoá + luật + k) thành `AppFilter.load(dir)`; đo ms/câu CPU | F3 |
| F5 | Đưa vào `build_data.py --apps filter` và `clef_agent.py` (PLAN_v2 A2); probe A0 | F4 |
| F6 | Khi có B2: train lại với app mới, đo `val_novel` | Gemini |
