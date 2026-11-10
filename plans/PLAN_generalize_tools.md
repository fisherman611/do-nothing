# PLAN: generalize tool call — tool chưa gặp, đọc schema là làm được

> 2026-11-10: phần việc còn lại (§6.2) được thay bằng `PLAN_v2_pipeline.md` (lọc app bằng BM25 ∥ embedding, prompt gọn,
> câu hỏi độc lập kiểu Jev). File này giữ làm tham chiếu: phân tích lỗi, hiệu chỉnh §3.0, latency §4.2, hạ tầng §6.1.

Ngày: 2026-11-10. Xuất phát từ `REPORT_errors_fmtA_mid_e1.md`. Bổ sung cho `PLAN_synth_fc_data_gemini.md` (gọi tắt
PLAN_synth) và `PLAN_prompt_format_v2.md`; chỗ nào trùng thì tham chiếu, không viết lại.

## 0. Mục tiêu và hiện trạng

Mục tiêu: model gặp tool **chưa từng thấy trong train**, chỉ đọc schema (tên, mô tả, tham số, kiểu, enum, x-ref,
required) vẫn chọn đúng tool, điền đúng tham số, biết khi nào không điền và khi nào hỏi lại.

Hiện trạng (`fmtA_mid_e1`, Qwen3.5-4B LoRA Clef, 1 epoch, 11k ví dụ):

| | val_in (app đã thấy) | val_ood (5 app giữ lại) |
|---|---|---|
| Điểm | 75.8 | 58.9 |
| Lỗi đặc trưng của tool lạ | – | điền thừa tham số tuỳ chọn (−5.7), sai quy ước giá trị id/tên/mã (−5.4), kéo về tool quen cùng chức năng (ewallet→banking, −3), thiếu call đọc trước call ghi (−3) |

Khoảng cách 17 điểm là cái giá của việc **ghi nhớ**: 147 tool lặp lại trong 8.7k mẫu nên model thuộc tên và quy ước
từng tool thay vì đọc schema. Public test có 11 app (≈ 40 tool) không có trong train; private test có thể khác nữa.

Nguyên tắc xuyên suốt: **làm cho việc nhớ tên không có lợi** (mỗi tên/tool chỉ xuất hiện vài lần, tên đổi liên tục,
luôn có tool mồi gần nghĩa) và **đưa schema đến ngay chỗ ra quyết định** (không bắt model nhớ lại từ system prompt
cách đó 5–10k token).

## 1. Dữ liệu: nhiều tool, mỗi tool ít lần

### 1.1 Mở rộng Part B2 (Gemini; luật ở PLAN_synth §1, §3.2, §4, §7)

Quy mô mới (thay cho 15–25 app ở PLAN_synth §3.2): **~150 app, ~600 tool, ~3000 hội thoại**; ít nhất 100 app có mẫu
trong train, 15 app giữ lại hoàn toàn làm `val_novel`. Mỗi tool **4–8 mẫu**, không hơn, để không tool nào đủ nhiều cho
model học thuộc.

Yêu cầu đa dạng **cố ý** (Gemini ghi bảng phân bổ trước khi viết, chủ repo duyệt):

| Trục | Phân bổ mục tiêu |
|---|---|
| Kiểu đặt tên app/tool | snake_case tiếng Anh 35% · camelCase 25% · tiếng Việt không dấu 25% · trộn 15% (theo PLAN_synth §2.2.2) |
| Tên tham số | tiếng Anh 60% · Việt không dấu 30% · viết tắt có nghĩa 10% (`so_luong`, `qty`, `maDon`) |
| Kiểu tham số (mỗi app phải có ≥ 3 kiểu) | string tự do · enum · boolean · integer/number · list · x-ref (id lấy từ tool đọc) · ISO datetime · ISO date |
| Tham số tuỳ chọn | 40% tool có ≥ 1 tham số tuỳ chọn; trong các mẫu của tool đó, **≤ 30%** mẫu có điền tham số tuỳ chọn, và chỉ khi câu người dùng nói rõ |
| Quy ước giá trị | id (`xx_123`) 50% · mã chuẩn (IATA, mã tỉnh, mã kênh) 15% · tên tự do 35%; **quy ước phải được nói trong mô tả tham số** ("ID chương trình, lấy từ `listPrograms`", "mã IATA 3 chữ, ví dụ HAN") |
| Tool gần nghĩa trong cùng app | 50% app có ≥ 2 tool chỉ khác nhau ở mô tả (`datLich` vs `doiLich`; `send` vs `reply`; `create` vs `append`) |
| Chia thao tác | 70% nhiều tool nhỏ · 30% một tool với tham số `action` enum |
| Tool đọc → ghi | mỗi app ≥ 1 cặp: tool ghi lấy id từ tool đọc qua `x-ref`; 20% mẫu cần đọc trước mới có id |
| Ràng buộc | 20% tool có `x-one-of` (contact_id hoặc phone) |
| Kết quả tool | 10% mẫu tool trả `ok:false`/`noop` (hết chỗ, không tìm thấy) và lượt sau xử lý đúng |
| Hội thoại | phân phối như PLAN_synth §4.4; thêm **≥ 12% mẫu hỏi lại** kiểu khó (tự sửa giữa câu, quên thông tin, tham chiếu không có trong state) |

Những lĩnh vực **không được làm**: 5 app giữ lại (alarm, ewallet, flights, loyalty, podcast_radio) và lĩnh vực của
11 app chỉ có ở test (xem B1, PLAN_synth §3.1); B1 làm riêng, vẫn theo luật "tương tự, không chép".

Mô tả tham số phải **đủ để làm mà không cần ví dụ**: nói rõ định dạng, nguồn lấy giá trị, khi nào điền. Đây là chỗ
model sẽ đọc lúc gặp tool lạ. Mô tả kiểu "Tên" hoặc "Thời gian" bị QA đánh trượt.

### 1.2 Đổi tên động khi build (`build_data.py`, không cần Gemini)

Mở rộng alias A1 (PLAN_synth §2) thành biến đổi **lúc build**, áp cho mọi mẫu (gốc + synth) với xác suất `--rename-p`
(mặc định 0.5):

- Chọn 1–3 app trong `context` của mẫu, thay tên app, tên tool, tên tham số theo một bộ alias lấy ngẫu nhiên từ
  `synth/alias_pool.json` (Gemini sinh sẵn 8–10 bộ alias cho mỗi tool train theo luật §2.2; B2 thì 2–3 bộ). Đổi đồng
  bộ ở catalog, history (tool_calls, tool_results), label và `x-ref`.
- Viết lại mô tả theo một trong 2–3 phiên bản có sẵn trong pool (diễn đạt khác, cùng nghĩa).
- **Xáo thứ tự tool trong system prompt** (hiện theo thứ tự catalog) và thứ tự app.
- Không đổi giá trị người dùng nói (tên người, số), không đổi tool `sys.*`.
- Kết quả: cùng một mẫu gốc xuất hiện với tên khác nhau ở các epoch/biến thể, model không thể dựa vào tên.

Giữ `#al` tĩnh hiện có cho `val_alias`; phần động chỉ cho train. Ghi `aug.rename` vào ví dụ để audit.

### 1.3 Tool mồi (decoy) trong catalog của mẫu

Với mỗi mẫu, thêm 2–4 tool mồi vào `catalog_overlay` (và vào danh sách tools của system prompt):
- 1–2 tool **gần nghĩa khác app** (đặt lịch cắt tóc khi đang nói đặt lịch họp; `banking.transfer` khi mẫu dùng ví);
- 1–2 tool **cùng app, khác mô tả nhỏ** (`send` vs `reply`, `create` vs `append`);
- lấy từ `synth/decoys.json` (Gemini sinh: với mỗi tool train/B2, 3–5 tool mồi có tên nghĩa, mô tả khác đủ để phân
  biệt được bằng đọc). Không bao giờ trùng tên tool thật trong mẫu.

Mồi là âm tính khó của PLAN_synth §5.2 nhưng đưa thẳng vào dữ liệu Clef; không cần loss riêng.

### 1.4 Dữ liệu sửa lỗi đã đo (giữ từ REPORT, §4 G2/G3/G5)

- Hỏi lại khó ~400 mẫu (tự sửa câu, quên, mơ hồ) trên app train;
- Câu điều kiện 2 nhánh + kết quả `ok:false` ~300 mẫu;
- Số/giờ bằng chữ: augment cơ học ~800 biến thể (`synth/num_words.py`), kèm bộ chuẩn hoá số tiếng Việt dùng cả lúc
  build và infer (một dòng gợi ý `[Số trong câu: …]` trong prompt).

### 1.5 Phân chia train/val

Trong lúc phát triển: **giữ nguyên 5 app held-out** (val_ood là thước đo chính), dù alarm/ewallet/flights có trong
test. Bản nộp cuối mới quyết định gộp hay không (mục 5).

## 2. Format: schema đến đúng chỗ (`fmt.py`, `core.py`; style mới `v2`, giữ `v1` để so)

### 2.0 Rút ngắn prompt (đã cài 2026-11-10: `--sys compact --rules short`, test_fmt 2–5 OK)

Ví dụ `train_0126` bước 1 (`docs/EXAMPLE_prompt_train_0126_step1*.md`): 6971 token, trong đó **68% là JSON schema của
43 tool** qua chat template Qwen (~110 token/tool) + ~250 token hướng dẫn gọi hàm của template, 14% quy tắc LifeMate,
10% khối Q1 (lặp lại tên + mô tả tool đã có ở system prompt). Model 4B khó attention qua 7k token để lấy 1 tool.

| Tuỳ chọn | Làm gì | Token (ví dụ) |
|---|---|---|
| `--sys compact` | Tools dạng text gọn trong system prompt, nhóm `[app]`, mỗi tham số một dòng `tên?: kiểu/enum/id←x-ref — mô tả`; **giữ đủ mô tả tool + tham số, bắt buộc, x-one-of**; không truyền `tools` cho template (bỏ hướng dẫn gọi hàm) | tools 4707 → 2164 |
| `--rules short` | `A.SYSTEM_SHORT`: giữ các luật về dữ liệu (ID, tham số tuỳ chọn, điều kiện, ask, định dạng), bỏ phần dạy format vì model được train | 776 → ~370 |
| `--q1-desc short` / `none` | Q1 chỉ tên + 40 ký tự mô tả / chỉ tên (mô tả đầy đủ đã ở system prompt) | 669 → 626 / ~300 |

Trên val_in (375 ví dụ): token TB **6346 → 3775** (prefix 5701 → 3167), max 9486 → 6072, với `compact/short/q1 short`.
Thêm `q1_desc=none` ước còn ~3450 (≈ 54% hiện tại). Khoá format mới (`sys`, `rules`) đi theo `fmt_opts()` →
`*.meta.json` → `config.json` → `--fmt-from`, nên build/train/infer không lệch nhau.

**Quyết định cho v2:** `sys=compact`, `rules=short`; `q1_desc` giữ `short` làm mặc định, thử `none` như ablation
(ablation rẻ: cùng data, chỉ khác khối Q1). Đổi system prompt = phải train lại → gộp vào lần build `fmtA_v2`. Lợi ích
kèm theo: prefill lần đầu mỗi mẫu giảm ~45% (đáng kể trên A30), và "thẻ tool" ở 2.1 bên dưới không còn là lặp lại
JSON dài mà là trích đúng tool đã chọn từ bản compact.

1. **Thẻ tool trước Q2.** Sau `Đáp án: X` của Q1, chèn khối:
   ```
   [Tool] <tên> – <mô tả đầy đủ>
   - <param> (<kiểu>, bắt buộc|tuỳ chọn[, enum: a|b|c][, ID từ <tool>][, một trong: p1/p2]) – <mô tả tham số đầy đủ>
   ...
   ```
   Lấy từ schema của tool được chọn (cả overlay). ~150–300 token/bước. Không tính loss trên khối này.
2. **Q2 nêu ràng buộc trong câu hỏi**: "Có điền `label` (tuỳ chọn, chỉ khi người dùng nêu) không?", "`program` (ID, lấy
   từ `list_programs`) = ?". Hiện Q2 chỉ có mô tả cắt 120 ký tự.
3. **Q1 giữ mô tả đầy đủ hơn**: `Q1_DESC` 70 → 100 ký tự cho style `v2` (đánh đổi ~1k token/bước; đo lại latency sau).
4. Giữ nguyên: Q1 là 1 token, Q2 từng tham số, xáo phương án lúc build, LS-CE + Brier.

Mọi thay đổi format gộp vào **một** lần build `fmtA_v2` + một lần train để không tốn thêm lần train.

## 3. Infer cho tool lạ (`clef_agent.py`, `core.py`, `modal_llm_server.py`; không cần train lại, thử được trên model hiện tại)

### 3.0 Lấy xác suất và confidence cho mỗi lựa chọn (điều kiện tiên quyết của 3.1–3.2)

Hiện trạng: loss `choice` lúc train là softmax **chỉ trên token nhãn** (A/B/…, có/không) + label smoothing 0.1 +
Brier, nên model có phân phối trên nhãn; nhưng `VLLMOracle.choose()` chỉ đọc token argmax
(`allowed_token_ids`, `max_tokens=1`), không lấy `logprobs` → infer chưa có xác suất, chưa có confidence.

Sửa:
1. **Server vLLM**: thêm `--max-logprobs 64` (mặc định 20; Q1 có tới 52 phương án) và
   `--logprobs-mode processed_logprobs` để logprob tính **sau** khi chặn `allowed_token_ids` → softmax chỉ trên nhãn,
   đúng phân phối lúc train.
2. **Client**: request thêm `logprobs=len(labels)`; đọc `top_logprobs[0]`, lấy xác suất từng nhãn (`probabilities`,
   nhãn vắng mặt coi là 0, chuẩn hoá lại cho tổng = 1). Confidence theo
   [typesafe.ai](https://docs.typesafe.ai/confidence): `conf = (max p − 1/n) / (1 − 1/n)` (0 = ngang ngẫu nhiên,
   1 = chắc chắn; với có/không: `2·max p − 1`).
3. **Ghi vào output**: mỗi phần tử `decisions` thành `(kind, label, probabilities, confidence)`; `clef_agent.py` thêm
   `--log-probs`. `score_subsets.py`/`err_analysis.py` đọc được.
4. **Đo hiệu chỉnh trên val** (trước khi đặt ngưỡng): ECE theo loại (q1 / present / ref / enum / bool), phân phối
   confidence của quyết định đúng vs sai, AUROC "confidence dự báo đúng/sai". Nếu tách kém → thêm **temperature
   scaling** như Clef gốc: một tham số T trên logit nhãn, fit trên val_in (không dùng val_ood), lưu vào `config.json`
   của run.
5. Quy ước dùng confidence: ngưỡng ở 3.1–3.2 đặt trên **xác suất nhãn** (dễ giải thích), confidence dùng để
   (a) xếp hạng mẫu "không chắc" khi phân tích lỗi, (b) quyết định có hậu kiểm/chọn lại ở 3.4, (c) tắt bớt hậu kiểm
   với quyết định conf > 0.9 để tiết kiệm thời gian.

#### Kết quả 3.0 (2026-11-10, `fmtA_mid_e1`, teacher-forcing, `agent/clef/calib_eval.py`)

Mỗi câu hỏi hỏi với prefix gold (chất lượng từng quyết định, chưa tính lỗi dây chuyền). Server `serve_mid_lp` (app
`lifemate-llm-lp`); đã kiểm tra logprob chỉ trên nhãn được phép, tổng = 1. Dữ liệu `agent/clef/data/calib/`,
kết quả `agent/runs/clef/calib/mid_*.jsonl`, `mid_report.json`.

| | acc | ECE | AUROC đúng/sai | conf đúng / sai |
|---|---|---|---|---|
| val_in (717 câu) | 0.941 | 0.061 | 0.87 | 0.89 / 0.69 |
| val_ood (2031 câu) | 0.906 | 0.016 | 0.79 | 0.87 / 0.72 |

Quyết định:
- **T = 1** (không temperature scaling): T = 0.84 fit trên val_in giảm ECE val_in 0.061→0.021 nhưng tăng ECE val_ood
  0.016→0.038. Thiếu tự tin ở val_in là do label smoothing 0.1 (max p ≈ 0.9).
- **τ₁ = 0.2** (hỏi lại nếu P(ask) > 0.2): val_in bắt 15→17/24 ask, không thêm hỏi nhầm; val_ood 31→37/41, +1 hỏi nhầm.
  P(ask) trung bình 0.53–0.59 khi gold là ask, 0.01 khi không.
- **Bỏ τ₂**: argmax (0.5) đã tốt nhất ở cả hai tập; 49 lần điền thừa ở val_ood phần lớn rất tự tin (18 lần P > 0.9) →
  sửa bằng dữ liệu (§1.1 quy ước trong mô tả).
- **Bỏ xáo lúc infer (G8 đóng)**: xác suất TB theo vị trí khớp tần suất gold theo vị trí, argmax=A bằng tỉ lệ gold=A,
  bản xáo/không xáo acc như nhau → không có thiên vị vị trí.
- Enum val_ood acc 0.57: cả 23 lỗi là `bill_payment.pay.source` (chọn ví dù số dư không đủ, conf ~0.87) → lỗi so sánh
  số của câu điều kiện, sửa bằng dữ liệu §1.4.

### 3.1–3.5 Luật infer

1. **Ngưỡng hỏi lại**: ở Q1 nếu P(`sys.ask_missing_info`) > τ₁ thì hỏi lại dù không phải argmax. Gọi ghi sai = 0 điểm,
   hỏi lại sai chỉ mất một call. τ₁ chọn trên val_in (thử 0.15/0.25/0.35). Chỉ áp khi tool argmax là tool **ghi**.
2. **Ngưỡng điền tham số tuỳ chọn**: điền khi P("có") > τ₂ (thử 0.5/0.6/0.7).
3. **Xáo phương án lúc infer** như lúc build (hiện infer không xáo; A luôn là item đầu, 24/41 lựa chọn sai rơi vào A).
   Nếu điểm không giảm thì giữ, loại thiên vị vị trí. Có xác suất rồi thì đo thẳng **thiên vị vị trí**: P(nhãn) trung
   bình theo vị trí, so với tỉ lệ gold theo vị trí.
4. **Hậu kiểm theo schema** (code): enum phải thuộc enum; id phải có trong kết quả tool đã thấy; số phải khớp một số
   parse từ câu (kể cả ×1000); bool/kiểu đúng. Sai thì chọn lại trong các phương án hợp lệ theo xác suất (hoặc hỏi
   lại nếu là tham số bắt buộc).
5. Chuẩn hoá số tiếng Việt: dòng gợi ý `[Số trong câu: …]` (mục 1.4) phải có mặt y hệt lúc build và lúc infer.

Thứ tự: 3.0 trước (server + client + chạy lại val với xác suất, ~$2) → báo cáo hiệu chỉnh → chọn τ → 3.1–3.4.

## 4. Đánh giá (bổ sung PLAN_synth §6)

| Tập | Đo gì | Tiêu chí |
|---|---|---|
| `val_in` (128) | không hỏng app đã thấy | ≥ 74 (hiện 75.8; chấp nhận −2 do mồi/đổi tên) |
| `val_ood` (259, 5 app) | thước đo chính | ≥ 66 (hiện 58.9); khoảng cách in−ood < 10 (hiện 17) |
| `val_alias` (128) | phụ thuộc tên | ≥ val_in − 3; tỉ lệ đổi đáp án gốc↔alias < 10% |
| `val_novel` (15 app B2 giữ lại) | đọc schema app lạ | báo cáo; kỳ vọng ≈ val_ood |
| `val_testlike` (B1) | chỉ theo dõi | không dùng chọn checkpoint |
| bản ASR của audio val | chịu lỗi ASR | báo cáo chênh với text gold |

Ba cấu hình phải chạy để tách tác động: (a) model cũ + infer v2; (b) model v2 + infer cũ; (c) model v2 + infer v2.
Phân tích lỗi bằng `scratchpad/deep_err.py` (chuyển vào `agent/clef/err_analysis.py`).

### 4.1 Chấm bằng đúng bộ metrics của challenge (`metrics/`), đủ 3 điều kiện

Điểm 75.8 / 58.9 hiện tại là `python -m metrics score` **thiếu** hai điều kiện: luật 30 giây (`--timings`) và xác minh
bằng `ToolEnv` (`--replay`). Từ nay mọi con số báo cáo phải chạy đủ:

```
python -m metrics score --gold <gold_val.jsonl> --submission <run.jsonl> --catalog <catalog của split> \
    --timings <run.timings.jsonl> --replay --data-root data/train+public_test --split train --output <report.json>
```

- **`--timings`**: `clef_agent.py` ghi thêm `<out>.timings.jsonl`, mỗi dòng `{"id", "elapsed_seconds", "status": ok|error|timeout, "error"}`.
  `elapsed_seconds` = **toàn bộ** thời gian mẫu (ASR + agent), đo tuần tự (`--workers 1`) trên máy chạy nộp bài;
  eval nhanh nhiều luồng chỉ để chẩn đoán và phải ghi rõ "chưa áp 30 s".
- **`--replay`**: hash tính lại bằng `ToolEnv` là có thẩm quyền; call bị từ chối khi replay không được tính khớp.
  Báo thêm số mẫu `reported_hash ≠ replay_hash` (phải là 0, khác 0 là bug ở agent).
- **Gold**: `python -m metrics prepare-gold --labels <val_x.jsonl>` cho từng tập (val_in, val_ood, val_alias, val_novel…);
  `--catalog` đúng split (val_alias / val_novel dùng catalog đã ghép overlay).
- **Chạy lại 5 lần** theo quy tắc BTC khi so sánh model để nộp: 5 run độc lập (seed khác nhau nếu có ngẫu nhiên; với
  greedy thì khác nhau ở timing), `python -m metrics average-runs`, rồi `python -m metrics rank` để xếp hạng theo
  đúng tie-break (score → tool F1 → thời gian). Lúc phát triển cho phép 1 run, nhưng quyết định chọn checkpoint
  cuối phải dựa trên trung bình 5 run.
- `score_subsets.py` sửa để gọi đúng lệnh trên (thêm `--timings`, `--replay`) và in cả `score`, `tool_call_f1`,
  `tie_break_score`, số mẫu timeout/error; không tự tính điểm riêng.
- ASR: `python -m metrics asr` trên bản ASR của audio val để báo WER/CER đi kèm điểm tool-call.

### 4.2 Mốc latency (2026-11-10, `fmtA_mid_e1`, 1 worker, L40S Modal us-east gọi từ VN, chưa cộng ASR)

Chạy dừng ở 225/387 mẫu (`agent/runs/clef/fmtA_mid_e1_w1.*`), bỏ 19 mẫu nhiễu (chen CPU với một eval khác, cold
start) → 206 mẫu sạch: 128 val_in + 78 val_ood **đầu tiên theo thứ tự split** (không phải mẫu ngẫu nhiên).

| | n | TB | p50 | p90 | max | > 30 s | request TB |
|---|---|---|---|---|---|---|---|
| val_in | 128 | 9.7 s | 8.9 | 20.1 | 29.7 | 0 | 7.2 |
| val_ood | 78 | 20.6 s | 19.2 | 37.5 | 41.4 | 13 (16.7%) | 13.6 |
| 0–13 request | 156 | | | | 20.1 | 0 | |
| 14–17 request | 32 | 22.5 s | 22.1 | 25.9 | 27.2 | 0 | 15.4 |
| ≥ 18 request | 18 | 34.5 s | 37.3 | 40.1 | 41.4 | 13 (72%) | 23.2 |

- Hồi quy (R² 0.98): **latency ≈ 0.30 + 0.92 s × số request + 0.066 s × nghìn token prompt** (7.3k token/request
  → 1.43 s/request). Phần cố định mỗi request (mạng VN↔Mỹ, HTTP) chiếm ~2/3.
- Theo luật BTC trên đúng các mẫu này: val_in 76.0 → 76.0 (0 timeout); val_ood 73.4 → **63.4** (13 timeout, mất 10 điểm).
- Ngưỡng thực tế: **~18 request/mẫu** ở điều kiện này. Mẫu ≥ 18 request (~9% mẫu) gần như chắc chắn quá 30 s.
- Đây là cận trên (có mạng, GPU L40S). Số dùng để quyết định phải đo trong Docker offline trên GPU giống BTC; việc đó
  để sau (G7). Biện pháp rẻ nếu cần: ngắt an toàn ~24 s, Q2 song song (mục 1c), bỏ câu "có điền" khi schema đã quyết.

## 5. Bản nộp

Sau khi (c) đạt tiêu chí: train bản cuối trên **toàn bộ** (gồm 5 app held-out và `val_in`), cùng số bước/LR, không
có val. Kỹ năng đọc schema đã học từ ~250 app khác nên không mất khi thêm 5 app. Giữ `val_novel` + `val_testlike`
để kiểm tra lần cuối (các app này vẫn ngoài train).

## 6. Việc và thứ tự (cập nhật 2026-11-10)

### 6.0 Đã xong

| # | Việc | Kết quả |
|---|---|---|
| 0 | Chấm đúng chuẩn BTC (§4.1): `clef_agent.py` ghi `timings.jsonl`, `score_subsets.py` gọi `metrics score --timings --replay` | Replay: 0 mẫu lệch hash. Mốc `fmtA_mid_e1` (replay, chưa 30 s): val_in 75.8 / val_ood 58.9. Latency §4.2 |
| 1a | Xác suất + confidence (§3.0), `calib_eval.py`, server `mid_lp` | ECE 0.02–0.06, AUROC 0.79–0.87; T = 1; bỏ τ₂, bỏ xáo lúc infer |
| 1b | `--ask-tau 0.2` trong `clef_agent.py` | val_in 75.8 → **78.2**, val_ood 58.9 → 59.4 (6 mẫu sửa đúng / 1 sai). **Bật mặc định từ nay** |
| – | Model `fmtA_mid_e1` lên HF `fisherman611/qw4-678342` (gộp + `lora/` + `checkpoints/step600/` + `train/`) | Dự phòng khi mất volume Modal |

### 6.1 Hạ tầng: account Modal mới (profile `kaiserihb`, client 1.6.1, builder 2025.06)

Account `anduongtrinhhoai` đã tiêu ~$30 credit; volume `hf-cache` (model gốc + bản gộp) và `clef-data` (data + checkpoint)
**vẫn nằm ở account cũ**, account mới chưa có gì của dự án (chỉ có secret `wandb`). Trước khi train/eval phải dựng lại:

| # | Việc | Lệnh / nguồn | Ghi chú |
|---|---|---|---|
| H1 | Secret `llm-api-key` | `modal secret create llm-api-key LLM_API_KEY=<key>` (chủ repo chạy; bộ lọc chặn Claude) | dùng lại key trong `scratchpad/llm_key.env` để `eval_smoke.sh` khỏi đổi |
| H2 | Volume `hf-cache`: model gốc text-only | `modal run agent/strip_qwen35.py` (~15 phút, L40S) | hoặc tải `fisherman611/qw4-678342` nếu chỉ cần eval |
| H3 | Volume `clef-data`: data | `modal volume put clef-data agent/clef/data/fmtA_mid /fmtA_mid` (+ `fmtA_v2` khi có) | data đủ trên máy local + HF `za-data-5eb2dec0` |
| H4 | Checkpoint `fmtA_mid_e1` (nếu cần resume / eval lại) | tải `lora/`, `checkpoints/step600/` từ HF về volume | model gộp: `modal_merge.py` hoặc tải `model.safetensors` từ HF |
| H5 | Deploy lại `modal_train.py` (`CLEF_GPU=H100`) và `modal_llm_server.py` (`LLM_APP=...`) | endpoint đổi URL → cập nhật `eval_smoke.sh` | |

### 6.2 Còn lại

| # | Việc | Ai | Phụ thuộc | GPU |
|---|---|---|---|---|
| 1c | **Q2 song song không train lại**: sau Q1 gửi mọi câu Q2 cùng lúc như "câu đầu tiên"; đo điểm + latency trên val (mục 3, câu hỏi song song kiểu Clef). Nếu val_in giảm < 1 → dùng; không → format v2 cần Q2 độc lập | Claude | H1, H2 hoặc H4, H5 | ~$2 |
| 2 | Bộ chuẩn hoá số/giờ tiếng Việt + augment số bằng chữ (§1.4); dòng gợi ý `[Số trong câu: …]` dùng cả build lẫn infer | Claude | – | 0 |
| 3 | File giao việc Gemini: `alias_pool.json`, `decoys.json`, B2 mở rộng (§1.1–1.3), hỏi lại khó ~400, điều kiện/`ok:false` ~300; QA theo PLAN_synth §7 | Claude viết, Gemini làm, chủ repo duyệt bảng phân bổ + danh sách app | – | 0 |
| 4 | `build_data.py --rename-p --decoys`; `fmt.py` style `v2` (thẻ tool sau Q1, Q2 nêu ràng buộc, Q1 desc 100; Q2 độc lập nếu 1c thất bại); `test_fmt.py` | Claude | 1c (quyết định Q2 độc lập) | 0 |
| 4b | Infer: hậu kiểm theo schema (§3.4), ngắt an toàn ~24 s (trả call đã có), bỏ câu "có điền" khi schema đã quyết | Claude | – | 0 |
| 5 | Build `fmtA_v2` (gốc + ph + al + if + B2 + 2), audit 0 leak / spans ok; upload volume | Claude | 3, 4 | 0 |
| 6 | Train 1 epoch H100 (`--sched-epochs 2`), ~15k ví dụ, ~5 h | Claude | 5, H2, H3 | ~$18 |
| 7 | Eval (b) model v2 + infer cũ, (c) model v2 + infer v2, bản ASR của audio val; chấm đủ 5 tập theo §4.1; phân tích lỗi `deep_err.py` | Claude | 6 | ~$6 |
| 8 | Bản nộp (§5): train trên toàn bộ, 5 run → `average-runs` → `rank`; đo latency Docker offline trên GPU giống BTC; nếu sát 30 s mới xét W8A8 INT8 + calibration Clef + chọn lại τ | Claude | 7 | ~$20 |

Thứ tự thực tế: **3 và 2 ngay** (không tốn GPU, Gemini chờ lâu nhất) → H1–H5 (nửa buổi) → 1c → 4/4b → 5 → 6 → 7 → 8.
Việc chủ repo: H1; chuyển repo HF `za-data-5eb2dec0`/`zaic-data-5eb2dec0` (chứa `public_test`) sang **private**; thu hồi
token HF/W&B/Modal đã dán trong chat; dọn `/mnt` myserver (còn 3 GB).

Ước tác động: val_ood +7–10, val_alias tiệm cận val_in, val_in −1…−2. Rủi ro chính: B2 chất lượng không đều (QA
PLAN_synth §7 bắt buộc); đổi tên động làm data khó hơn, loss train cao hơn — là chủ ý, không phải lỗi.
