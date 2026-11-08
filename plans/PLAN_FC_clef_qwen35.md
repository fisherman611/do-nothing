# Plan model gọi tool kiểu Clef trên Qwen3.5-4B, serve bằng vLLM – ZAC 2026 LifeMate

Cập nhật: 2026-11-08 (bản 2: chuyển sang "Clef ở mức token" để serve bằng vLLM gốc). Đi kèm `PLAN_SFT_qwen35_text.md` (SFT sinh text thường: mốc so sánh + dữ liệu chung), `REPORT_zeroshot_LLM.md`, `../reports/decision_models_research.md` (nghiên cứu Clef / Laya).

## 0. Đã chốt

- Backbone **Qwen3.5-4B bản cắt** (`/hf/qwen35_4b_text`, 4.206B, bỏ visual + MTP), **tắt thinking**. Zero-shot không thinking trên train: 37.5 điểm, F1 50.8.
- Thiết kế **"Clef ở mức token"**: giữ ý tưởng của Clef (phương án là text, chấm mọi phương án trong một lượt forward, loss label smoothing + Brier, calibration nhiệt độ) nhưng **head là chính LM head của Qwen, giới hạn vào các token nhãn của phương án**. Không thêm module mới → **serve bằng vLLM gốc** (LoRA, prefix caching, tất định).
- Không thêm tham số: LLM 4.206B, cả hệ thống với PhoWhisper-large 5.75B < 6B.

## 1. So với Clef

| Clef | Ở đây |
|---|---|
| Backbone đóng băng + LoRA + schema head | Backbone đóng băng + LoRA; "head" = LM head chỉ đọc logit của các token nhãn |
| Phương án là text; chấm trong một lượt forward | Giống: khối câu hỏi liệt kê phương án, đọc logit ở vị trí kế tiếp (1 token) |
| `choice` (softmax), `noul` (có/không) | `choice`: softmax trên các nhãn `A, B, …`; `noul`: softmax trên ` có` / ` không` |
| Label-smoothed CE + Brier, nhiệt độ theo số phương án | Giống, tính trên softmax giới hạn vào các nhãn |
| `EvidenceRoutingLayer`: phương án không attend lẫn nhau → bớt thiên vị thứ tự | **Không có**: phương án nằm trong cùng chuỗi causal nên có thể bị thiên vị vị trí. Bù bằng **xáo thứ tự phương án khi train**, thứ tự cố định khi suy luận |
| Không sinh chuỗi tự do | Phần chuỗi tự do (nội dung, giờ, từ khoá) do chính model sinh như SFT thường |

Đã kiểm trên tokenizer Qwen3.5: 52 chữ cái `A–Z`, `a–z` và 562 cặp `AA–ZZ` đều là 1 token (đủ cho tối đa 65 phương án/bước); ` có` / ` không` là 1 token. Không dùng số làm nhãn vì Qwen tách từng chữ số.

## 2. Dữ liệu train (1545 mẫu)

| Quyết định | Số lượng | Làm bằng |
|---|---|---|
| **Bước tiếp theo** (tool nào / `sys.ask_missing_info` / dừng) | 5086 bước (3324 gọi tool, 217 hỏi lại, 1545 dừng); TB 33.5, tối đa 65 phương án | Q1 `choice` |
| Tham số **id `x-ref`** | 1278, **1277 (99.9%) có sẵn trong kết quả tool / history** | Q2 `choice` trên id ứng viên |
| Tham số **enum** | 481 | Q2 `choice` |
| **Có điền tham số tùy chọn không** | 1070 / 3229 lượt tham số tùy chọn được điền | Q2 `noul` mỗi tham số |
| Bool | (trong 363 số/bool) | Q2 `noul` |
| Text fuzzy 740, thời gian 592, số, chuỗi khác 1111 | ~2800 | Q3: model sinh |

Ba lỗi chính của zero-shot (dừng sớm / gọi thừa, thiếu `ask_missing_info`, thêm tham số tùy chọn) đều là quyết định rời rạc, nằm ở Q1 / Q2.

## 3. Định dạng một bước

Một bước của vòng agent là phần nối thêm vào lượt assistant, sau prompt chuẩn (system + catalog + history + user + call/kết quả trước đó trong lượt, render bằng chat template của Qwen với `enable_thinking=False`, đúng như `agent/agent.py`):

```
<prefix chuẩn>
[Q1] Bước tiếp theo?
A) calendar.list_events – Xem lịch
B) calendar.update_event – Sửa sự kiện
…
Y) sys.ask_missing_info – Hỏi lại người dùng
Z) DỪNG – Đã xong / không cần làm gì
Đáp án: B                                   ← 1 token, giới hạn trong {A…Z}
[Q2] calendar.update_event
event_id = ? A) evt_01 (chạy bộ, 08:00 29/11)  B) evt_02 (đá bóng, 14:00 29/11)
Đáp án: A                                   ← giới hạn trong nhãn của câu này
Điền location? Đáp án: không                ← giới hạn trong { có, không}
…
<tool_call><function=calendar.update_event>
<parameter=event_id>evt_01</parameter>      ← điền từ Q2
<parameter=start>2026-11-29T09:00:00+07:00</parameter>   ← Q3: model sinh
<parameter=end>2026-11-29T10:00:00+07:00</parameter>
</function></tool_call>
```

- Phương án Q1 = các tool của app đã cài + `sys.ask_missing_info` + `DỪNG`, kèm mô tả ngắn (catalog đầy đủ đã ở system prompt). Khoảng 15 token/phương án → ~500 token/bước.
- Id ứng viên Q2 = mọi giá trị dạng id trong `tool_results` của lượt hiện tại và history, cùng loại `x-ref` (tiền tố `c_`, `evt_`…), mỗi id kèm vài trường đi cùng (tên, tiêu đề, giờ) để model phân biệt.
- **Lịch sử các bước trước chỉ giữ tool call + kết quả, bỏ khối Q1/Q2**, để prompt giống prompt chuẩn và không phình context.
- Chọn `DỪNG`: kết thúc lượt. Chọn `sys.ask_missing_info`: model sinh câu hỏi (Q3), vòng lặp dừng.
- Nếu không còn gì để sinh ở Q3 (mọi tham số đều rời rạc), đóng call ngay, không gọi model.

## 4. Train

- **Ví dụ**: mỗi bước có nhãn thành một chuỗi `prefix chuẩn + Q1 + đáp án + Q2 + đáp án + tool call`. Bước dừng chỉ có Q1. Khoảng 5086 chuỗi; thêm 2981 lượt assistant có call trong history làm ví dụ phụ. Dùng đúng hàm dựng prompt của `agent/agent.py`.
- **Loss** (chỉ trên phần sau prefix):
  - Token đáp án Q1/Q2: cross-entropy **trên softmax giới hạn vào các nhãn của câu hỏi đó**, label smoothing 0.1, cộng Brier score trên cùng phân phối (như Clef).
  - Token tool call do model sinh (Q3): cross-entropy thường. Các token điền từ Q2 (id, enum) và khung XML không tính loss.
  - Trọng số ban đầu 1 : 1, chỉnh theo val.
- **Xáo thứ tự phương án** ở mỗi ví dụ (nhãn gán sau khi xáo) để model không học vị trí. Có thể thêm nhiễu catalog (đổi tên / bỏ bớt app không liên quan) để học đọc mô tả cho tool lạ.
- LoRA r=32 trên attention, linear-attention (DeltaNet) và MLP; LR 1e-4, 2–3 epoch, không packing. Các kiểm tra ở bước 0 của `PLAN_SFT_qwen35_text.md` vẫn áp dụng.
- **Calibration**: fit nhiệt độ theo số phương án trên val held-in. Ngưỡng xác suất chỉ dùng nếu có lợi trên val (cần logprob sau giới hạn, xem mục 5).
- **Chia val** như plan SFT: giữ lại ~8 app (proxy cho app lạ ở public test) + 10% held-in. Chọn checkpoint theo điểm `metrics/` của vòng agent đầy đủ.
- Khối lượng: ~5k chuỗi × ~4.5k token ≈ 23M token/epoch.

## 5. Suy luận trên vLLM

Mỗi bước gồm 2–3 request tới **một** server vLLM (LoRA), dùng chung prefix nên prefix cache làm phần lớn việc:

| Request | Prompt | Tham số |
|---|---|---|
| Q1 | prefix + khối Q1 + `Đáp án:` | `max_tokens=1`, `temperature=0`, `allowed_token_ids` = nhãn Q1 |
| Q2 | … + đáp án Q1 + khối Q2 | Structured output (regex / grammar) để mỗi chỗ đáp án chỉ nhận nhãn của câu đó; greedy |
| Q3 | … + đáp án Q2 + `<tool_call><function=t>` + tham số đã điền | Sinh thường, dừng ở `</function>`; chỉ khi còn tham số chuỗi tự do |

Sau đó `env.validate` → `env.call` như hiện tại. Vòng agent viết thành `agent/clef_agent.py`, cùng giao diện với `agent.py run_one` để dùng lại `run_zeroshot.sh` và `metrics/`.

**Phải kiểm trên vLLM 0.31 trước khi làm (bước 0):**
1. `allowed_token_ids` với `max_tokens=1` cho đúng argmax trong tập nhãn.
2. Structured output (regex / xgrammar) chạy với tokenizer Qwen3.5.
3. Logprob sau khi giới hạn (`logprobs_mode`): chỉ cần nếu dùng ngưỡng.
4. **Prefix caching với kiến trúc hybrid (Gated DeltaNet)**: xem log tỉ lệ cache hit. Nếu không cache được thì mỗi request phải prefill lại 3–6k token và cách này chậm đi nhiều. Đây là kiểm tra quan trọng nhất.
5. Serve LoRA cho `Qwen3_5ForCausalLM` (gồm LoRA trên lớp DeltaNet); nếu không được thì gộp LoRA vào trọng số.
6. Tất định: chạy hai lần cùng tập ra cùng `tool_calls`.

## 6. Code cần viết

| File | Việc |
|---|---|
| `agent/clef/questions.py` | Dựng khối Q1 / Q2 (phương án, nhãn, id ứng viên kèm ngữ cảnh), thứ tự cố định khi suy luận, xáo khi train |
| `agent/clef/build_data.py` | Chuỗi train theo bước từ nhãn + vị trí token đáp án và tập nhãn hợp lệ của từng câu hỏi |
| `agent/clef/train.py` | Trainer: loss giới hạn nhãn (LS-CE + Brier) + CE thường cho phần sinh; eval bằng vòng agent |
| `agent/clef_agent.py` | Vòng suy luận Q1 → Q2 → Q3 qua vLLM; cùng định dạng output với `agent.py` |
| Test | Prompt train = prompt suy luận từng token; nhãn là 1 token; chạy lại cho cùng kết quả |

## 7. Đánh giá và ablation

Cùng hai tập val, cùng `metrics/`:

| # | Hệ thống | Vai trò |
|---|---|---|
| 0 | Zero-shot không thinking | Sàn |
| 1 | SFT sinh text thường (hướng A của plan SFT) | **Mốc chính phải vượt** |
| 2 | Clef-token, chỉ Q1 (model sinh mọi tham số) | Lợi ích của việc chọn bước bằng phương án |
| 3 | Clef-token đầy đủ (Q1 + Q2 + Q3) | Thiết kế đề xuất |
| 4 | (tùy chọn) Q1 + sinh call có ràng buộc grammar (id ∈ ứng viên, enum ∈ giá trị) thay cho Q2 | Biến thể gọn hơn, ít request hơn |

Báo: điểm, F1 tie-break, state match; số ca của ba lỗi chính; Brier / ECE của Q1; độ trễ mỗi bước và mỗi mẫu trên GPU gần A30.

**Tiêu chí chọn**: dùng #2/#3/#4 khi vượt #1 ít nhất ~1–2 điểm trên **cả hai** tập val (nhất là app held-out) và vẫn vừa 30 s/mẫu. Không thì nộp #1 (cùng dữ liệu, cùng vLLM, nên chuyển qua lại rẻ).

## 8. Thứ tự việc

| # | Việc | Kết quả |
|---|---|---|
| 1 | Bước 0 vLLM (mục 5: 1–6) bằng model zero-shot, chưa cần train | Biết serve được, đặc biệt prefix cache |
| 2 | Bước 0 train (kernel DeltaNet, LoRA target, smoke 8k) + **chọn nơi chạy** | Biết train được |
| 3 | Chia val, chấm lại zero-shot trên val; thử #2 zero-shot (chỉ đổi prompt, chưa train) | Sàn cùng tập; xem định dạng Q1 có hợp với model gốc không |
| 4 | `questions.py`, `build_data.py` + test | Dữ liệu |
| 5 | Train SFT thường (#1) và Clef-token Q1 (#2) | **Điểm dừng sớm**: #2 không hơn #1 thì không làm Q2 |
| 6 | Thêm Q2 (#3), thử #4 | So sánh |
| 7 | Đo độ trễ trên GPU gần A30, calibration, ngưỡng | Chọn hệ thống nộp |
| 8 | Ghép ASR v2 end-to-end | Điểm thật |

Phương án dự phòng nếu Clef-token không đủ và còn thời gian: head cross-attention thật (`EvidenceRoutingLayer`) qua plugin model của vLLM chạy chế độ pooling, phải dùng instance thứ hai (tốn gấp đôi bộ nhớ trọng số, không chung KV cache). Chỉ thử sau khi #3 đã có số.

## 9b. Ngân sách context (luật: ≤ 16,384 token)

Đo bằng tokenizer Qwen3.5 và chat template thật (`agent.py`, `enable_thinking=False`):

| Phần | p50 | p99 | Max |
|---|---|---|---|
| Prompt mỗi bước, zero-shot trên train (gồm kết quả tool trong lượt, số của vLLM) | 5,463 | 8,773 | **9,748** |
| Prefix ban đầu, train | 5,100 | 8,364 | 8,873 |
| Prefix ban đầu, public_test (lượt user giả định ~25 token vì chưa có ASR) | 4,827 | 8,159 | 8,836 |
| Khối Q1, mô tả tool đầy đủ | ~500 | – | ~1,000 |
| Khối Q2 (tối đa 32 id ứng viên/bước trên train × ~20 token) | – | – | ~700 |
| Phần sinh (call / câu trả lời) | ~50 | – | ~400 |

- Trên phân phối train: tệ nhất ≈ 9.75k + 1k + 0.7k + 0.4k ≈ **11.9k < 16,384**, dư ~4.5k.
- Nhưng tệ nhất về lý thuyết thì vượt: prefix 8.9k + 8 bước × kết quả tool tối đa 777 token ≈ 15.1k, cộng khối câu hỏi ~1.7k → ~16.8k. Vì vậy **bắt buộc có chốt chặn trong vòng agent** (cả bản SFT thường):
  1. Đếm token trước mỗi request; server vLLM đặt `--max-model-len 16384`.
  2. Khi prompt > ~13k: khối Q1 chỉ ghi tên tool, bỏ mô tả (catalog đầy đủ đã ở system prompt), còn ~8 token/phương án.
  3. Kết quả tool dài (danh sách): giữ N phần tử đầu + số lượng tổng, như plan pipeline.
  4. Nếu vẫn không vừa: dừng lượt (không gửi request lỗi). Call đã chạy vẫn nằm trong `tool_calls`.
- Khối Q1/Q2 của các bước trước không giữ trong lịch sử, nên không cộng dồn.
- Public test có app lạ: đo lại khi có transcript ASR thật; kiểm lại trên toàn bộ public_test trước khi nộp.

## 9. Rủi ro

- **Prefix caching với hybrid DeltaNet** không chạy hoặc chạy kém → mỗi bước prefill nhiều lần, có thể vượt 30 s.
- **Thiên vị vị trí** của phương án (không có cross-attention riêng như Clef): xáo khi train, đo trên val bằng cách đảo thứ tự lúc suy luận.
- **Tool lạ ở public test**: model phải học từ mô tả, không thuộc tên; theo dõi val app held-out.
- **Context**: xem mục 9b; cần chốt chặn token trong vòng agent.
- **Chi phí train**: ~23M token/epoch × 2–3 epoch. Trên Modal A100-80GB ước vài giờ (~$10+), vượt credit còn lại (~$10); myserver 2×A100-40GB miễn phí nhưng cần dọn ổ. Kaggle T4 không hợp cho 4B với context 4–6k.

## 11. Tiến độ (2026-11-08)

Code đã viết (`agent/clef/`, xem `agent/clef/README.md`), chạy được trên CPU, **chưa train, chưa chạy với vLLM**:

| Phần | Trạng thái |
|---|---|
| `fmt.py`, `core.py`: định dạng Q1/Q2/tool call, một hàm `run_step` cho cả train và suy luận | Xong |
| `test_fmt.py` (300 mẫu, 955 ví dụ): nhãn đúng 1 token tại vị trí đọc; tool call khớp từng ký tự với chat template; tham số dựng lại = nhãn | **Qua** |
| `test_loop.py`: `clef_agent.run_one` + `ToolEnv` thật, oracle theo nhãn vàng, chấm `metrics/` | **100 điểm** / 200 mẫu, 0 lỗi |
| `splits.py`: 5 app giữ lại (ewallet, alarm, podcast_radio, loyalty, flights) → val_ood 259, val_in 128, train 1158 | Xong |
| `build_data.py`: train 7881 ví dụ (1158 mẫu + 1975 lượt history), val_in 375, val_ood 980; bỏ 13 bước (đáp án vàng ngoài phương án) | Xong |
| `train.py` (LoRA, loss giới hạn nhãn LS-CE + Brier, CE tham số tự do), `modal_train.py` | Viết xong, chưa chạy |
| `clef_agent.py` (vLLM), `check_vllm.py` (bước 0) | Viết xong, chưa chạy |
| `agent.py`: chốt chặn 16,384 token + cắt danh sách dài trong kết quả tool | Xong (train không bị ảnh hưởng: danh sách dài nhất 14 phần tử) |

Baseline zero-shot (`agent.py`) chấm lại trên đúng tập:

| | val_in (128) | val_ood (259) |
|---|---|---|
| Qwen3.5-4B cắt, không thinking | 39.1 (F1 53.2) | 27.2 (F1 41.7) |
| Qwen3.5-4B cắt, thinking | 46.0 (F1 56.8) | 35.7 (F1 48.5) |
| Qwen3-4B-2507 | 29.5 | 14.5 |

Khối lượng train thật: ~7.9k ví dụ × ~5.4–6.7k token ≈ **45–55M token/epoch** (gấp ~2 lần ước tính ban đầu, do Q1 liệt kê mọi tool kèm mô tả và có thêm lượt history). Trên một A100-80GB ước 4–5 giờ/epoch → 2 epoch ~$20+ trên Modal, vượt credit còn lại. Cách giảm: bỏ lượt history (−40%), Q1 rút mô tả, 1 epoch; hoặc chạy trên myserver (2×A100-40GB, miễn phí, cần dọn ổ).


### Kết quả đầu tiên sau train (2026-11-08)
SFT LoRA 1 epoch, kiểu prompt P2, 3510 ví dụ (1158 mẫu, không dùng lượt history), Modal A100-80GB 131 phút. Agent thật qua vLLM, `metrics/`:

| | val_in | val_ood (app không có trong train) |
|---|---|---|
| **Clef-token P2 sau SFT** | **78.4** | **65.0** |
| Zero-shot thường, không thinking / thinking | 39.1 / 46.0 | 27.2 / 35.7 |

Teacher-forcing trên val_in: Q1 92.5%, có/không 96.2%, id/enum 99.25%. 0 call sai cấu trúc. Chi tiết: `agent/runs/clef/RUNLOG.md`. Việc còn thiếu để kết luận về phương pháp: **SFT sinh text thường trên cùng dữ liệu** (mốc #1 ở mục 7) và đo độ trễ trên GPU gần A30.

## 10. Luật

- LLM < 5B, tổng < 6B, context ≤ 16,384, offline, tất định (greedy, thứ tự phương án cố định, seed cố định).
- Chỉ train trên train (+ dữ liệu tự sinh); không gán nhãn public/private test; không đọc `*_environment/`.
- Chỉ dùng `ToolEnv.validate/call/get_tool_calls/get_state_hash`; mọi call đã chạy đều nằm trong `tool_calls`.
