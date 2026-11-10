# PLAN v2: pipeline Clef gọn — lọc app, prompt ngắn, câu hỏi độc lập kiểu Jev

Ngày: 2026-11-10. Thay thế phần "việc còn lại" của `PLAN_generalize_tools.md` (giữ file đó làm tham chiếu cho phân tích
lỗi §0–1, hiệu chỉnh §3.0, latency §4.2, hạ tầng §6.1). Dữ liệu tổng hợp vẫn theo `PLAN_synth_fc_data_gemini.md`.

## 0. Vì sao đổi

Mốc hiện tại (`fmtA_mid_e1`, Qwen3.5-4B LoRA, 1 epoch, 11k ví dụ): val_in 78.2 (với τ₁) / val_ood 59.4; latency
p50 12.6 s, mẫu ≥ 18 request quá 30 s. Ba vấn đề gốc, mỗi cái một trụ của plan này:

| Vấn đề | Số liệu | Trụ |
|---|---|---|
| Prompt dài, model 4B phải attention qua 7k token để chọn 1 trong ~35 tool; 84% tool trong prompt không bao giờ dùng | 6971 token/bước, 68% là JSON 43 tool; app cài TB 9.2, dùng TB 1.5 | **A. Lọc app + prompt gọn** |
| Mỗi bước 1 + N request tuần tự, câu sau phụ thuộc câu trước → latency ∝ số tham số | 9.7 request/mẫu, 0.92 s/request cố định | **B. Câu hỏi độc lập, song song (Jev)** |
| Ghi nhớ tool thay vì đọc schema; lỗi tự tin ở tool lạ (điền thừa tham số, chọn ví dù hết tiền) | in−ood 17 điểm; 49 lần điền thừa với P > 0.9 | **C. Dữ liệu v2** (PLAN_generalize_tools §1) |

## 1. Jev: học được gì, tái tạo được gì (từ docs.typesafe.ai + blog, đọc 2026-11-10)

| Jev công bố | Tái tạo được? | Cách làm trên Qwen3.5 |
|---|---|---|
| Mọi câu hỏi **đánh giá song song và độc lập** trên cùng state; thêm câu hỏi không làm hỏng câu khác | **Có** (về ngữ nghĩa) | Qwen3.5 có 3/4 lớp Gated DeltaNet (recurrent) → không chặn được bằng attention mask trong một chuỗi. Làm bằng **N chuỗi riêng chung prefix**: train mỗi (prefix, câu hỏi) là một ví dụ; infer N request song song, vLLM prefix-cache (đã bật, 0.31 hỗ trợ hybrid) |
| Confidence `(p_max − 1/n)/(1 − 1/n)`; Noul `|2p−1|`; **confidence của call = câu yếu nhất** | **Có**, đã làm công thức; thêm "call = min" | `clef_agent.py --log-probs`, thêm `call_conf = min(conf các câu)` |
| Cardinality lớn: "chấm từng phương án độc lập rồi chọn tường minh" (2 giai đoạn) | **Có** | Q0 lọc app (trụ A) chính là giai đoạn 1; Q1 trong tập đã lọc là giai đoạn 2 |
| Function calling: Choice `__tool__`; mỗi tham số một Noul `stated` + Choice giá trị; int/text/ngày **không hỏi**, dùng default | Giống Q1 + `present` + enum/ref hiện có; **khác**: mình phải sinh text/số vì bài cần giá trị thật | Giữ sinh text; thêm bộ chuẩn hoá số (trụ C) |
| Train: chỉ cái tên **RLCD**; kiến trúc "parallel sampler" | **Không** (không có loss, data, base, size) | Giữ LS-CE 0.1 + Brier (proper scoring rule, cùng mục tiêu "xác suất đúng bậc"); ECE đo được 0.02–0.06 |
| Mô tả phương án phải phân biệt nhau; hỏi về *ý* chứ không về *từ*; tách câu hỏi nhỏ | Đưa vào rule dữ liệu | PLAN_synth B2: mô tả nêu "khác tool gần nghĩa ở đâu" |

Kết luận: không có cách train "giống Jev hẳn" từ tài liệu công khai, nhưng **giao diện và cách đặt câu hỏi của Jev
thì tái tạo được**, và đó là phần quyết định latency + tính độc lập. Phần hiệu chỉnh mình đã có số đo tốt.

### 1.1 RLCD: học theo Laya (open-weights, code công khai — đọc 2026-11-10)

Laya (`convaiinnovations/laya`, Apache-2.0, GitHub `NandhaKishorM/laya`) là model quyết định 421M (ModernBERT-large
+ head 2 lớp) **không sinh text**, train bằng RLCD và công bố **toàn bộ recipe trong `laya/train.py`**. Đây là nguồn
duy nhất có code RLCD chạy được; Jev không công bố gì.

**Recipe RLCD của Laya (từ code, không phải suy đoán):**

| Thành phần | Laya | Áp cho Clef của mình |
|---|---|---|
| Mỗi câu hỏi = **một chuỗi riêng** `[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state`; mỗi phương án được chấm tại `[MASK]` của nó → softmax trên phương án của câu đó; N câu = N chuỗi trong **một batch** | Encoder hai chiều, batch | **Đúng cách mình định làm** cho Q2 độc lập (N chuỗi chung prefix). Khác: mình chấm bằng logit 1 token nhãn của LM head, Laya chấm bằng head riêng tại marker |
| Reward **strictly proper**: `log_score + 0.75·spherical` (+ RPS cho câu thứ bậc) | `proper_reward` | Thêm spherical; Brier của mình cũng proper, giữ |
| Khám phá: nhiễu Gauss **zero-mean trên logit** (chiếu về tổng 0 trên các phương án), σ **giảm tuyến tính 0.4 → 0.1** theo epoch; **4 mẫu**/câu | `rlcd_loss`, `sigma_at` | Cài y hệt trên logit nhãn `z` (K phương án) |
| Update: REINFORCE với **baseline trung bình nhóm** (GRPO-style), advantage chuẩn hoá (chia std), log-density Gauss của mẫu; **cộng soft-CE** trên logit gốc | `−(adv·logp).mean() + soft_ce` | `loss_choice = LS-CE + Brier + PG` (PG = phần RLCD; `--rlcd-sigma 0` = tắt) |
| Xáo thứ tự phương án **mỗi epoch** (trừ câu thứ bậc) | `draw_option_order` | Mình đã xáo lúc build (một lần); với Q2 độc lập có thể xáo lại mỗi epoch rẻ |
| Multi-turn: TD(λ=1) trên các prefix của hội thoại | episode | Không cần: mỗi bước của mình đã có nhãn trực tiếp |
| **Hiệu chỉnh hậu kỳ: temperature theo bucket** `(loại câu, số phương án)`: `choice:2 / 3-5 / 6-10 / 11+`, `noul:2`, fit NLL trên 10% held-out (≤ 400), clamp [0.5, 5]; ECE 0.466 → 0.081 | `calibrate.py` | Mình đo T toàn cục = 1 là tốt; **thử T theo bucket** vì enum val_ood (2 phương án) có ECE 0.37 trong khi Q1 (40 phương án) 0.05 |
| Siêu tham số notebook: effective batch 64, LR encoder 2.5e-5 / head 1e-4, cosine, 4 epoch, label smoothing 0 | `TrainConfig` | Giữ LR/LoRA hiện tại; chỉ thêm PG term |

**Đọc đúng bản chất của PG term** (để không kỳ vọng sai): logit bị detach, nhiễu cộng vào, gradient của
`−adv·logp` theo logit là `adv·ε/σ²` — tức là ước lượng kiểu evolution-strategy của **điểm proper kỳ vọng dưới nhiễu
logit**. Với model đã xuất xác suất, CE trên one-hot *đã là* log score; phần RLCD thêm vào là (a) spherical score,
(b) làm mượt theo nhiễu → phạt phân phối quá nhọn/quá tự tin. Kỳ vọng: **cải thiện hiệu chỉnh và độ bền với tool
lạ**, không chắc tăng accuracy. Mình đã có Brier (cũng phạt tự tin sai), nên phần tăng thêm có thể nhỏ → phải đo.

**Cài trong `train.py`:** `--rlcd-sigma 0.4 --rlcd-sigma-end 0.1 --rl-samples 4 --w-sph 0.75` (mặc định tắt). Chạy
như **ablation (iv)** trong §6 trên cùng data v2: CE+Brier vs CE+Brier+RLCD; so điểm BTC, ECE theo bucket, AUROC
đúng/sai, và đặc biệt lỗi "điền thừa với P > 0.9" ở val_ood (chỗ hiệu chỉnh đang kém nhất).

## 2. Trụ A — lọc app bằng truy hồi + prompt gọn

### 2.1 Số liệu nền (train, 1545 mẫu)
App cài 9.2 (3–18); app dùng ở lượt cuối 1.52 (0: 11%, 1: 43%, 2: 33%, ≥3: 13%); tool trong prompt 32.5, tool của app
dùng 5.4 (16%). Chỉ 13% mẫu có đủ app cần dùng trong history → **phải lọc từ câu nói**. 46% mẫu dùng ≥ 2 app → lọc
theo **app** (không theo tool), giữ **nhiều app** chứ không phải top-1.

### 2.2 Bộ lọc: BM25 ∥ embedding/reranker tiếng Việt, hợp (union) → ưu tiên recall

Tập ứng viên nhỏ (3–18 app của mẫu), nên chi phí không đáng kể; mục tiêu **recall ≥ 99% trên app gold** với số app
giữ trung bình ≤ 4.

1. **Tài liệu mỗi app** (`agent/clef/app_filter/app_docs.py`, dựng từ catalog, kể cả catalog overlay / app synth):
   tên app + tên tool + mô tả tool + mô tả tham số + **từ khoá đồng nghĩa tiếng Việt** (sinh một lần, có duyệt:
   `sms` → "nhắn tin, tin nhắn, SMS, ét em ét"; `ewallet` → "ví, ví điện tử, nạp tiền, quét mã"...). Từ khoá là
   chỗ bộ lọc hay thiếu nhất với app lạ; cho app test-like/B2 cũng phải có.
2. **BM25** trên âm tiết tiếng Việt (lowercase, bỏ dấu câu; thử cả bản không dấu để chịu lỗi ASR) → điểm mỗi app.
3. **Embedding/reranker tiếng Việt** (chọn 1–2 trong: `bkai-foundation-models/vietnamese-bi-encoder`, `BAAI/bge-m3`,
   reranker `AITeamVN/Vietnamese_Reranker` hoặc `namdp-ptit/ViRanker`); query = câu cuối + câu user cuối cùng trong
   history (history dài thì chỉ 2 lượt cuối). Chạy offline trong Docker, CPU/GPU đều được (~20–50 ms).
4. **Hợp**: giữ app nếu thuộc top-k₁ BM25 **hoặc** top-k₂ embedding **hoặc** điểm ≥ ngưỡng tương đối; **luôn giữ**
   app đã gọi trong history và app mà tool được giữ tham chiếu qua `x-ref` (đóng bao phụ thuộc: `ewallet.transfer`
   cần `contacts`; `sms.send` cần `contacts`). Chọn k₁, k₂, ngưỡng trên train để recall ≥ 99%, báo recall trên val_in,
   val_ood, val_alias (tên app đổi → chỉ còn mô tả + từ khoá để bám).
5. Trường hợp 0 call (chào hỏi, đổi ý): bộ lọc vẫn trả top-k; model quyết DỪNG như hiện nay.
6. **Dự phòng khi recall không đủ**: Q0 bằng chính model — Noul "app này có cần không?" cho mỗi app, song song một
   lượt trên prefix ngắn (câu nói + 1 dòng/app), giữ P > 0.1. Chỉ dùng nếu truy hồi < 99%; so sánh recall/latency.

### 2.3 Prompt sau lọc (đã cài `--sys compact --rules short`, PLAN_generalize_tools §2.0)
- System: quy tắc ngắn (~370 token) + tools compact của các app được giữ (~50 token/tool → 3–4 app ≈ 600 token).
- **Decoy bắt buộc**: lúc build train, ngoài app gold thêm 1–2 app (ưu tiên app gần nghĩa, hoặc app bộ lọc giữ dư)
  để model không học "app có mặt = phải gọi". Lúc infer, app dư của bộ lọc đóng vai decoy tự nhiên.
- Q1: chỉ tool của app được giữ (+ ask + DỪNG) ≈ 8–14 phương án; `q1_desc=short` mặc định, thử `none`.
- **Thẻ tool** sau Q1 (schema đầy đủ của tool đã chọn, trích từ bản compact) — PLAN_generalize_tools §2.1.
- Ước tổng: ví dụ `train_0126` 6971 → **~1700–2100 token/bước** (¼–⅓ hiện tại).

### 2.4 Bộ lọc nằm trong vòng train
Build train bằng **đầu ra thật của bộ lọc** (không phải oracle) để phân phối lúc train = lúc infer: giữ = filter(câu)
∪ gold ∪ decoy. Mẫu train mà bộ lọc sót app gold vẫn có gold trong prompt (và được đếm là miss của bộ lọc); ở infer
miss là mất mẫu, nên số miss trên val là con số phải theo dõi riêng.

### 2.5 Kiểm chứng rẻ trước khi xây (A0, ~$1, không train)
Chạy `fmtA_mid_e1` trên val với prompt **lọc oracle** (app gold + 2 decoy, vẫn format JSON cũ để model không lệch).
Nếu điểm tăng rõ so với 75.8/58.9 → danh sách dài đang làm hại, trụ A đáng ưu tiên cao; nếu không đổi → trụ A chỉ còn
giá trị latency/token, vẫn làm nhưng xếp sau trụ C.

## 3. Trụ B — câu hỏi độc lập, song song

- **Format v2**: mỗi câu Q2 chỉ thấy prefix + Q1 + đáp án Q1 + thẻ tool, **không thấy đáp án Q2 khác**. Build: mỗi
  (bước, câu Q2) là một ví dụ riêng; để chặn chi phí train, mỗi bước lấy ngẫu nhiên ≤ 2 câu Q2 (model chỉ cần học
  "trả lời một câu hỏi từ prefix"). Với prompt ~2k token sau trụ A, token train/epoch vẫn ≈ hiện tại.
- **Infer**: Q1 → một lượt N request song song (present/enum/ref/bool của tool đã chọn) → sinh text tự do tuần tự
  (2.3 lượt/mẫu, giữ nguyên) → call. Một bước = 2 lượt + số text tự do, thay vì 1 + N.
- **`stated` kiểu Jev cho tham số tự do tuỳ chọn**: giữ câu "có điền không?" như hiện nay (chính là Noul `stated`),
  nhưng song song với các câu khác.
- **Confidence call = min** các câu; dùng cho hậu kiểm (§5) và báo cáo mẫu không chắc.
- Ước latency (điều kiện §4.2 PLAN_generalize_tools): mẫu 4 bước ~17 request/9 lượt → ~9 lượt ≈ 13 s; cộng lọc app
  và prefix ngắn → mẫu dài nhất về dưới ~20 s qua mạng, nhiều dư cho A30 offline.
- B0 (thử trước, ~$2, không train): hỏi Q2 song song trên `fmtA_mid_e1` như "câu đầu tiên" → đo điểm + latency. Dù kết
  quả thế nào, v2 vẫn train Q2 độc lập (quyết định theo Jev); B0 chỉ cho biết lợi ích sớm.

## 4. Trụ C — dữ liệu v2 (tóm tắt, chi tiết ở PLAN_generalize_tools §1 và PLAN_synth)
- B2 mở rộng ~150 app / 600 tool / 3000 hội thoại, mỗi tool 4–8 mẫu, bảng phân bổ đa dạng; mô tả nêu quy ước giá trị
  và "khác tool gần nghĩa ở đâu"; `alias_pool.json`, `decoys.json`.
- Đổi tên động khi build (`--rename-p 0.5`), decoy (giờ là **app decoy** theo §2.3).
- Hỏi lại khó ~400; điều kiện 2 nhánh + `ok:false` ~300; số/giờ bằng chữ ~800 + bộ chuẩn hoá số (dòng gợi ý trong prompt).
- Giữ 5 app held-out trong lúc phát triển; bản nộp train trên toàn bộ.
- **Tài liệu app cho bộ lọc** (từ khoá đồng nghĩa) là một sản phẩm của Gemini cho mọi app (train, B1, B2).

## 5. Infer v2 (gộp)
Lọc app (§2.2) → prefix gọn → Q1 với `--ask-tau 0.2` (đã có) → Q2 song song → text tự do → hậu kiểm theo schema
(enum ∈ enum, id ∈ kết quả đã thấy, số khớp số trong câu kể cả ×1000) → ngắt an toàn ở ~24 s tính cả ASR (trả call
đã có). Ghi `probs`, `call_conf`, `apps_kept`, `filter_miss` (khi có gold) vào output.

## 6. Đánh giá (chuẩn BTC, PLAN_generalize_tools §4.1)
| Tập | Đo thêm |
|---|---|
| val_in / val_ood / val_alias / val_novel / val_testlike | điểm BTC (replay + 30 s), **recall bộ lọc**, số app giữ TB, token/bước, lượt/mẫu, latency |
| Ablation | (i) lọc oracle vs bộ lọc thật vs không lọc; (ii) `q1_desc` short vs none; (iii) Q2 độc lập vs tuần tự (B0); (iv) loss CE+Brier vs +RLCD (§1.1); (v) temperature toàn cục vs theo bucket |
| Tiêu chí v2 | val_in ≥ 76; val_ood ≥ 66; in−ood < 10; recall lọc ≥ 99%; p90 latency < 20 s (điều kiện Modal qua mạng) |

## 7. Việc, thứ tự, chi phí

| # | Việc | Phụ thuộc | GPU |
|---|---|---|---|
| H | Hạ tầng account Modal mới (PLAN_generalize_tools §6.1: secret, strip, volume, deploy) | chủ repo tạo `llm-api-key` | ~$1 |
| A0 | Probe lọc oracle trên `fmtA_mid_e1` (§2.5) | H | ~$1 |
| B0 | Probe Q2 song song trên `fmtA_mid_e1` (§3) | H | ~$2 |
| A1 | `app_filter/`: app_docs + từ khoá (bản đầu tự viết cho 46 app train), BM25, embedding/reranker, union, đóng bao x-ref; đo recall/k trên train & val; chọn cấu hình | – | CPU |
| A2 | `build_data.py --apps filter|oracle --decoy-apps 2`; `clef_agent.py` dùng bộ lọc; thẻ tool trong `fmt.py` | A1 | – |
| B1 | Format v2 Q2 độc lập trong `build_data.py`/`core.py`/`clef_agent.py` (song song) + `test_fmt.py` | – | – |
| C1 | Bộ chuẩn hoá số/giờ + augment; gửi Gemini: B2, alias pool, decoys, hỏi lại khó, điều kiện, **từ khoá app** | – | – |
| D1 | Build `fmtA_v2` (compact/short, lọc + decoy, Q2 độc lập, rename, data C) → audit | A2, B1, C1 | – |
| D2 | Train 1 epoch H100 (`--sched-epochs 2`); với prompt ~2k token ước ~2.5 h | D1 | ~$10 |
| D3 | Eval đủ bảng §6 + phân tích lỗi; quyết định bản nộp | D2 | ~$5 |
| E | Bản nộp: train toàn bộ, 5 run, Docker offline đo latency trên GPU giống BTC; W8A8 INT8 chỉ nếu còn sát 30 s | D3 | ~$15 |

Thứ tự thực tế: **C1 (gửi Gemini) và A1 ngay** (không cần GPU, Gemini chờ lâu nhất) → H → A0 + B0 (một buổi, ~$3) →
A2 + B1 → D1 → D2 → D3 → E. Nếu A0 cho thấy danh sách ngắn **không** tăng accuracy, trụ A vẫn giữ (token, latency)
nhưng recall yêu cầu nới xuống 98% để bộ lọc đơn giản hơn.

Rủi ro chính: (1) recall bộ lọc với app lạ/đổi tên — giảm bằng từ khoá đồng nghĩa và đóng bao x-ref, dự phòng Q0;
(2) Q2 độc lập kém Q2 tuần tự ở tham số phụ thuộc nhau (ví dụ `start`/`end`) — đo ở B0 và ablation (iii), nếu kém thì
chỉ các cặp phụ thuộc mới hỏi tuần tự; (3) dữ liệu B2 không đều — QA PLAN_synth §7.
