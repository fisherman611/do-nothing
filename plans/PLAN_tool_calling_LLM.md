# Plan LLM gọi tool (function calling) – ZAC 2026 LifeMate

Cập nhật: 2026-11-07. Đi kèm `PLAN_PhoWhisper_finetune.md` (ASR). Luật áp dụng: `AGENT.md`.

## 0. Bài toán nhìn từ dữ liệu

Mỗi mẫu: audio yêu cầu cuối → agent gọi một **chuỗi** tool trên `ToolEnv` → nộp `tool_calls` + `state_hash`. Chấm bằng cách replay `tool_calls` trên state gốc: **F1 trên call + so state cuối**. Hòa thì so F1 không cần khớp state, rồi so thời gian.

### 0.1 Đầu vào (có ở cả train và test)

| Trường | Nội dung | Số liệu train (1545 mẫu) |
|---|---|---|
| `audio` | Yêu cầu cuối, phải qua ASR | 1545 file, TB 7.2s |
| `context.current_time`, `timezone` | Mốc để hiểu "mai", "tối nay" | Mọi mẫu, múi +07:00 |
| `context.location`, `user_profile` | lat/lng, tên, địa chỉ nhà/công ty, thành phố | Dùng cho maps, ride_hailing, weather |
| `context.installed_apps` | App đã cài và tool được phép gọi | 3–18 app, **8–62 tool/mẫu** (TB 32) |
| `history` | Các lượt trước: user (audio, **cần ASR**) + assistant (text, `tool_calls`, `tool_results`) | 0–24 lượt; 66% mẫu có history; 2981/3491 lượt assistant có tool call |
| catalog của split | JSON Schema từng tool: `required`, `enum`, `x-one-of`, `x-ref` (id lấy từ app khác), `x-match: fuzzy` | train 46 app / 147 tool; **public 36 app, 11 app chưa từng có trong train** (410/634 mẫu public có cài app lạ) |
| `ToolEnv` lúc chạy | Kết quả thật của từng call (danh bạ, lịch, số dư…) | Chỉ có qua `env.call`, không được đọc `state.json` |

Kích thước prompt (tokenizer Qwen, catalog rút gọn còn name/description/parameters): **p50 3.2k token, p95 5.1k, max 6.3k**. Phần catalog chiếm 2.6k (max 5.1k), history 0.3k (max 2.7k). Kết quả tool trong lượt: p50 170 token, max 1.1k. Dưới 16k token theo luật, nhưng phải tính thời gian prefill trên A30.

### 0.2 Đầu ra (label)

| Đặc điểm | Số liệu |
|---|---|
| Số call / mẫu | 0: 7%, 1: 24%, 2: 28%, 3: 23%, 4: 10%, 5–6: 8% (TB 2.3) |
| Mẫu `[]` | 111: ngoài phạm vi (không có app), người dùng tự hủy ý ("à thôi khỏi"), khen/chào, điều kiện không thỏa |
| `sys.ask_missing_info` | 217 mẫu (14%). Phần lớn sau `contacts.search` trả **≥2 kết quả** (66/74 ca) → thiếu `contact_id`; hoặc thiếu `amount`, `time` |
| Mẫu read→write | Mẫu phổ biến: `rw` 258, `rww` 150, `rrw` 91, `rrrw` 50. **692/1545 mẫu (45%) có tham số chỉ lấy được từ kết quả tool trong cùng lượt** (id liên hệ, hóa đơn, sự kiện…) |
| Nguồn giá trị tham số (chuỗi) | 1275 từ kết quả tool trong lượt; 740 từ history/context (vd. "biên nhận vừa rồi"); 1769 model tự sinh (text, thời gian, số) |
| Loại tham số | 4565 tham số: 2066 thường, 1278 id `x-ref`, 740 free-text `x-match: fuzzy` (content, title, note…), 481 `enum` |
| Thời gian | 455 ISO-8601 có offset (`2026-11-29T09:00:00+07:00`), 112 `date` dạng `YYYY-MM-DD`, 20 `HH:MM`. Offset ngày so với `current_time`: hôm nay 112, mai 213, 2–13 ngày sau 102 |
| Điều kiện | "Nếu mai mưa thì… không thì thôi": phải gọi weather trước rồi quyết định tiếp hay dừng |
| Kết quả tool | 100% `ok: true` trong label; thất bại nghiệp vụ trả `done: false`, không phải lỗi schema |

Hệ quả thiết kế:

1. **Bắt buộc là agent nhiều bước** (ReAct): sinh call → `env.call` → đọc kết quả → sinh tiếp hoặc dừng. Không thể sinh một lần toàn bộ chuỗi vì 45% mẫu cần id từ kết quả.
2. **Mọi call read đều tính vào F1.** Gọi thừa read (vd. `get_status` khi không cần) làm giảm precision; gọi thiếu read mà đoán id thì state sai. Phải học đúng "lúc nào cần read".
3. **Phải tổng quát sang tool chưa thấy**: public có 34 tool mới, private chắc còn khác. Train không được dạy model "thuộc" 147 tool mà dạy đọc schema. Cần giữ một phần app làm held-out khi fine-tune để đo điều này.
4. **Thời gian phải tính đúng**: ISO đầy đủ offset, "mai 7 giờ sáng" = `current_time` + 1 ngày. Nên tính sẵn trong prompt: thứ, hôm nay, mai, ngày kia, dạng ISO, để model chỉ ghép.
5. **Free-text fuzzy** (nội dung tin nhắn, tiêu đề) được chấm gần đúng, nên không cần khớp từng chữ; ưu tiên ngắn, đủ ý, lấy số/giờ đúng.
6. ASR sai tên → `contacts.search` không thấy → model phải xử lý (ask_missing_info), nên chuỗi hành vi khi tool trả rỗng cũng cần có trong train.

## 1. Ràng buộc chọn model (AGENT.md)

- Tổng hệ thống < 6B: PhoWhisper-large 1.543B → **LLM ≤ 4.45B**. LLM < 5B, context ≤ 16,384. Qwen3.5-4B gốc (4.66B) phải cắt visual + mtp còn 4.206B (đã làm, `agent/strip_qwen35.py`).
- Phát hành trước **26/10/2026**, license cho phép dùng và phân phối trong Docker. Offline hoàn toàn.
- A30 24GB, **30 s/mẫu** gồm cả ASR history (tối đa 12 audio) + nhiều bước LLM. Giải mã xác định (greedy).

## 2. Ứng viên LLM 2B–4B

| Model | Tham số | License | Phát hành | Context | Tool calling | Tiếng Việt | Nhận xét |
|---|---|---|---|---|---|---|---|
| **Qwen3-4B-Instruct-2507** | 4.0B | Apache-2.0 | 08/2025 | 256k (native) | Có, BFCL-v3 61.9 (số báo từ trang bên thứ ba, cần đối chiếu model card) | Tốt (Qwen đa ngữ) | **Ứng viên chính.** Non-thinking, template tool call chuẩn Hermes, vLLM hỗ trợ sẵn, nhiều recipe fine-tune |
| **Qwen3.5-4B** | 4B (có vision encoder) | Apache-2.0 | ~đầu 2026 (chưa xác minh ngày chính xác) | dài | Có | Tốt | Mới hơn Qwen3; cần xác minh ngày phát hành và cách tắt phần vision khi đếm tham số |
| **Gemma 4 E4B-it** | ~4B hiệu dụng (tổng lớn hơn do embedding PLE) | Apache-2.0 | 04/2026 | 128k | Native | Tốt (140 ngôn ngữ) | Mạnh nhưng **phải kiểm tra tổng tham số thật** (E4B ≈ 8B tổng nếu tính embedding) → rủi ro vượt 6B; có audio input nhưng không thay được PhoWhisper tiếng Việt |
| Gemma 4 E2B-it | ~2B hiệu dụng | Apache-2.0 | 04/2026 | 128k | Native | Khá | Dự phòng nếu cần nhanh; tool-use kém E4B rõ (42% vs 25% trên benchmark của Google) |
| Qwen3-1.7B | 1.7B | Apache-2.0 | 04/2025 | 32k | Có | Khá | Dự phòng tốc độ; dự kiến kém 4B ở chuỗi nhiều bước |
| Sailor2-3B-Chat | 3B (Qwen2.5 cắt tỉa) | Apache-2.0 | 2025 | 128k | Không có template tool riêng, cần dạy | **Rất tốt** (pretrain 500B token ĐNÁ) | Đáng thử nếu hiểu tiếng Việt là nút thắt; tool calling phải fine-tune từ đầu |
| Phi-4-mini-instruct | 3.8B | MIT | 02/2025 | 128k | Có | Yếu hơn | Thường kém Qwen ở đa ngữ |
| xLAM-2-3b-fc-r, Hammer2.1-3b | 3B | CC-BY-NC | 2025 | – | Rất mạnh tool calling | Trung bình | **Loại**: license phi thương mại, rủi ro với luật phân phối |
| Ministral-3B | 3B | Mistral Research | – | – | Có | – | **Loại**: license nghiên cứu |

Chọn (cập nhật 2026-11-08 sau zero-shot): **Qwen3.5-4B bản cắt làm mặc định** (zero-shot 37.5 điểm không thinking, 45.2 có thinking); Qwen3-4B-Instruct-2507 dự phòng (25.5). **Gemma 4 loại**: E4B-it 7.996B, E2B-it 5.12B tổng tham số, vượt LLM < 5B. Kết quả: `REPORT_zeroshot_LLM.md`. Plan train: `PLAN_SFT_qwen35_text.md`.

## 3. Pipeline suy luận

```
audio history u1..uk ──PhoWhisper(+prompt)──▶ text u1..uk    (lời assistant có sẵn)
audio cuối           ──PhoWhisper(+prompt)──▶ text cuối
                                   │
system: luật + thời gian tính sẵn + user_profile + catalog của installed_apps (rút gọn)
messages: history (user text, assistant text + tool_calls + tool_results đã có)
user: text cuối
                                   ▼
        LLM sinh 1 tool call (hoặc "xong")  ──▶ env.call ──▶ kết quả đưa lại làm tool message ──▶ lặp
                                   ▼
        env.get_tool_calls(), env.get_state_hash() → submission.jsonl
```

Chi tiết:

- **Một `ToolEnv` mỗi mẫu**, không `validate` thử nhiều lần rồi gọi (validate không tính vào tool_calls nên được dùng, nhưng vẫn phải gọi đúng cuối cùng).
- **Mỗi bước sinh đúng một call** (không sinh song song), vì call sau thường phụ thuộc kết quả call trước. Tối đa 8 bước; dừng khi model trả text thay vì tool call.
- **Rút gọn catalog**: chỉ tool trong `installed_apps`; bỏ `returns` rỗng, `app`; giữ `description`, `required`, `enum`, `x-one-of`, `x-ref`. Thêm một dòng luật: id phải lấy từ kết quả tool; thiếu tham số bắt buộc → `sys.ask_missing_info`; không có app phù hợp → không gọi gì.
- **Thời gian**: thêm vào system `today=2026-11-28 (Thứ Bảy), tomorrow=2026-11-29, ...`, giờ hiện tại, offset `+07:00`. Giảm hẳn lỗi tính ngày của model nhỏ.
- **Kết quả tool dài** (danh sách 20 giao dịch): cắt còn N phần tử đầu + đếm, để prompt không phình.
- **Giải mã**: greedy, `max_new_tokens` 200/bước, parse JSON chặt; parse hỏng → thử sửa 1 lần, vẫn hỏng → dừng (không gọi call sai schema vì bị ghi vào tool_calls).
- **Serving**: vLLM trên A30, bf16, prefix caching (system + catalog giống nhau giữa các bước), 1 mẫu/lần hoặc batch nhỏ. ASR dùng CTranslate2/faster-whisper nếu kịp đóng gói, nếu không thì HF bf16 beam 1–2 cho history, beam 5 cho lượt cuối.
- **Ngân sách 30 s** (ước, cần đo): ASR history ≤ 12 audio ≈ 4–8 s; LLM prefill 3–6k token ≈ 0.5 s/bước, decode ~60 token/bước ≈ 1.5 s/bước, 2.3 bước TB → 5 s, tệ nhất 6 bước ≈ 12 s. Tổng TB ~12 s, tệ nhất ~22 s. Phải benchmark thật trên A30 (hoặc GPU tương đương trên Modal: A10G/L4 chậm hơn A30 nên lấy làm cận trên).

## 4. Đánh giá

- **Scorer local**: replay `tool_calls` dự đoán bằng `zaic_tools` trên train, so với `label.final_state` qua `state_hash` và F1 call (khớp tên + tham số; fuzzy cho `x-match`). Viết `asr/agent/score.py` mô phỏng đúng luật trong `utils.canon`.
- **Hai chế độ**: (a) dùng `audio_text` gốc → đo riêng LLM; (b) dùng transcript ASR → đo end-to-end; hiệu số = phần lỗi do ASR.
- **Chia held-out theo app**: giữ ~8 app (≈ 15% mẫu) không xuất hiện trong train LLM để đo tổng quát sang tool lạ, vì public có 11 app lạ.
- Báo theo nhóm: 0-call, ask_missing_info, 1 call, chuỗi ≥3, mẫu điều kiện (weather), mẫu tham chiếu history.

## 5. Train

> Plan chi tiết, cập nhật: `PLAN_SFT_qwen35_text.md`. Phần dưới là bản ban đầu.

### 5.1 Bước 0: zero-shot (chưa train) — làm trước

Chạy Qwen3-4B-Instruct-2507 zero-shot trên train với `audio_text` gốc. Số này quyết định có cần fine-tune hay chỉ cần sửa prompt. Dự kiến vấn đề: gọi thừa read, sai định dạng thời gian, không biết luật `ask_missing_info` khi có ≥2 liên hệ trùng, bỏ sót call thứ hai trong lệnh ghép ("tắt đèn rồi nhắn Hương").

### 5.2 Bước 1: SFT theo từng bước (LoRA)

- Dữ liệu: từ 1545 mẫu train, mỗi call trong label thành một ví dụ: messages (system + history + user + các call/kết quả trước đó trong lượt) → call tiếp theo; thêm ví dụ "dừng" sau call cuối. Khoảng **1545 + 3541 ≈ 5k ví dụ**, mỗi ví dụ 3–7k token. Loss chỉ trên phần assistant (tool call), không trên prompt.
- Lượt user trong history dùng `audio_text` gốc khi train, nhưng **trộn 30–50% ví dụ dùng transcript ASR** (từ model PhoWhisper đã fine-tune, chạy trên train) để model quen với lỗi ASR, nhất là tên riêng sai → ask_missing_info.
- LoRA r=16–32 trên attention + MLP, LR 1e-4, 2–3 epoch, bf16, context 8k, batch theo token. Trên Modal A100-80GB ≈ 1–2 giờ/lần. Giữ held-out theo app.
- Kiểm tra sau mỗi epoch bằng scorer mục 4 (không chỉ loss), vì loss token không phản ánh F1 call.

### 5.3 Bước 2: dữ liệu tổng hợp cho tool lạ (được phép theo AGENT.md)

- Luật cho phép tạo dữ liệu tổng hợp bằng model khác ở giai đoạn phát triển. Dùng LLM lớn (API) sinh hội thoại + chuỗi call cho **app mới tự nghĩ** (giống kiểu 11 app public: parking, stocks, grocery…) với schema tự viết, chạy qua simulator tự viết để có `tool_results` nhất quán. Mục tiêu: dạy model đọc schema chứ không thuộc tool. Nhắm 2–3k mẫu.
- Không được dùng nhãn hay kết quả public_test để train (luật). Pseudo-label public test bị cấm, kể cả cho ASR.

### 5.4 Bước 3 (nếu còn thời gian): DPO/RL trên lỗi của chính model

- Lấy các mẫu train mà model sau SFT sai, chọn cặp (đúng từ label, sai từ model) làm DPO; hoặc GRPO với reward = F1 call + khớp state từ scorer local. Chỉ làm khi SFT đã ổn định và có scorer tin cậy.

## 6. Thứ tự việc

| # | Việc | Kết quả cần có |
|---|---|---|
| 1 | Viết `asr/agent/`: dựng prompt, vòng lặp agent với `ToolEnv`, scorer local | Chạy được trên train với `audio_text` |
| 2 | Zero-shot Qwen3-4B-Instruct-2507 trên train (+ Qwen3.5-4B nếu xác minh được) | F1 / state match baseline, phân tích lỗi theo nhóm |
| 3 | Benchmark thời gian end-to-end 1 mẫu (ASR + LLM) trên GPU gần A30 | Biết còn bao nhiêu dư địa cho beam, bước, kích thước model |
| 4 | SFT LoRA bước 5.2, held-out theo app | F1 held-in vs held-out |
| 5 | Chạy public_test, nộp leaderboard | Điểm public |
| 6 | Dữ liệu tổng hợp tool lạ (5.3) nếu held-out kém | F1 held-out tăng |
| 7 | Đóng gói Docker offline (vLLM + PhoWhisper + zaic_tools), cố định seed | `result/submission.jsonl` đúng định dạng |

## 7. Nguồn

- Qwen3-4B-Instruct-2507: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 (BFCL-v3 61.9 trích từ https://dev.co/ai/llms/qwen3-4b-instruct-2507, cần đối chiếu model card)
- Qwen3.5-4B: https://huggingface.co/Qwen/Qwen3.5-4B
- Gemma 4 E2B/E4B: https://huggingface.co/google/gemma-4-E4B-it, https://huggingface.co/google/gemma-4-E2B-it, https://en.wikipedia.org/wiki/Gemma_(language_model)
- Sailor2: https://github.com/sail-sg/sailor2, https://arxiv.org/html/2502.12982v1
- BFCL: https://pypi.org/project/bfcl-eval/
