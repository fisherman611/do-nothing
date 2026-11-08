# Plan SFT Qwen3.5-4B (bản cắt) trên dữ liệu text của train – ZAC 2026 LifeMate

Cập nhật: 2026-11-08. Đi kèm `PLAN_tool_calling_LLM.md` (bài toán, pipeline) và `REPORT_zeroshot_LLM.md` (baseline).

## 0. Mục tiêu và mốc so sánh

- Model: Qwen3.5-4B bản cắt `/hf/qwen35_4b_text` (4.206B, `Qwen3_5ForCausalLM`), bf16.
- Dữ liệu: 1545 mẫu train, lượt user lấy từ `audio_text` gốc (transcript ASR để sau, mục 6).
- Baseline zero-shot trên toàn train: không thinking **37.5** điểm (F1 50.8), có thinking **45.2** (F1 55.9). Phải chấm lại trên đúng các tập val ở mục 3 thì mới so được với model đã train.
- Ba lỗi chính cần sửa: thêm tham số tùy chọn, thiếu `sys.ask_missing_info`, dừng sớm hoặc gọi thừa.
- Ràng buộc: 30 s/mẫu trên A30 (gồm ASR), giải mã tất định, không dùng nhãn public/private test, không đọc `*_environment/`.

## 1. Bước 0: kiểm tra trước khi train (có thể chặn plan)

| # | Kiểm tra | Nếu hỏng |
|---|---|---|
| 0.1 | **Ổ myserver**: `/mnt` đầy 100%, còn 8.6G; HF cache 33G. Một checkpoint bf16 ≈ 8.4G | Dọn cache, hoặc train và eval trên Modal (A100-80 $2.5/h, volume `hf-cache` đã có bản cắt) |
| 0.2 | **Train được kiến trúc hybrid**: cài `flash-linear-attention` + `causal-conv1d` (không có thì transformers chạy đường torch rất chậm). In `named_modules()` để chọn LoRA target cho cả lớp linear-attention lẫn full-attention và MLP | Thiếu kernel thì chỉ LoRA lớp full-attention + MLP, hoặc chạy trên Modal với image có kernel |
| 0.3 | **Không packing** mặc định: trạng thái linear-attention sẽ lẫn giữa các mẫu nếu kernel không nhận `cu_seqlens` | Chỉ bật packing khi đã xác nhận có `cu_seqlens` |
| 0.4 | **Smoke test**: một lượt forward + backward ở 8k token trên một A100-40GB, đo token/s và bộ nhớ | Giảm seq len, bật gradient checkpointing, LoRA nhỏ hơn |
| 0.5 | **Serve adapter**: vLLM 0.31 `--enable-lora` có chạy với `Qwen3_5ForCausalLM` (gồm LoRA trên lớp linear-attention) không | Gộp adapter vào bf16 trước khi serve (tốn thêm ~8.4G cho mỗi bản) |
| 0.6 | **Template nhất quán**: prefix của mỗi lượt assistant trong chuỗi train phải trùng **từng token** với prompt mà `run_one` dựng lúc suy luận (đặc biệt khối `<think>\n\n</think>` rỗng ở các lượt trước) | Không trùng thì bỏ cách "một chuỗi / mẫu", dùng "một ví dụ / bước" (mục 2.2) |

Cần bạn chọn: train ở myserver (phải dọn ổ) hay trên Modal.

## 2. Dựng dữ liệu SFT

### 2.1 Nguồn

- `label.tool_calls` + `label.tool_results` (kết quả vàng có sẵn trong label, không cần chạy env). Label **không có** câu trả lời cuối của assistant.
- `history`: 2981 lượt assistant có `tool_calls` (+ `tool_results` + `text`), dùng làm ví dụ phụ ở hướng C.

### 2.2 Cách dựng

- **Không viết lại prompt.** Import `SYSTEM`, `build_messages`, `tool_specs`, `_call_msg` từ `agent/agent.py`, render bằng đúng tokenizer và `chat_template_kwargs` lúc suy luận. Viết script `agent/build_sft.py` + một test so token (kiểm tra 0.6).
- Mỗi mẫu là một chuỗi: system + history + user cuối → call 1 → kết quả 1 → … → call cuối → kết quả cuối → **câu dừng**.
- **Loss** chỉ tính trên các lượt assistant (tool call và câu dừng). Mask system, user, tool result.
- **Câu dừng**: label không có, nên dùng một câu cố định ngắn (vd. "Đã xong."). Mẫu kết thúc bằng `sys.ask_missing_info` không cần câu dừng vì vòng lặp tự dừng ở đó. Mẫu nhãn rỗng (111 mẫu) chỉ có câu dừng.
- Một call mỗi lượt, đúng như vòng lặp agent.
- Hai cách dựng:
  - **Một chuỗi / mẫu** (mặc định cho không thinking): rẻ, ~1545 chuỗi. Chỉ dùng khi kiểm tra 0.6 qua.
  - **Một ví dụ / bước** (bắt buộc cho thinking, vì template bỏ phần suy luận ở lượt cũ): ~5k ví dụ, prefix lặp lại nên tốn ~3 lần token.
- Độ dài: prompt p50 3.2k, max 6.3k token; chuỗi đầy đủ ước 4–8k. Đặt `max_len` 12k, không cắt mẫu.

## 3. Chia dữ liệu và cách đánh giá

- **Val app-held-out**: giữ ~8 app (~15% mẫu), không xuất hiện trong train SFT. Đây là proxy cho 11 app lạ ở public test. Mẫu có cài app held-out nhưng không gọi nó vẫn được train; chỉ mẫu có **gọi** tool của app held-out mới vào val.
- **Val held-in**: 10% ngẫu nhiên trong phần còn lại, cố định seed.
- Chạy lại zero-shot (không thinking + thinking) trên đúng hai tập val để có baseline cùng tập (có sẵn kết quả từng mẫu, chỉ cần lọc theo id và chấm lại).
- Tiêu chí chọn checkpoint: **điểm `metrics/` của vòng lặp agent đầy đủ** (vLLM + `zaic_tools`), không dùng loss. Báo thêm F1 tie-break, state match, và số ca của ba lỗi chính.
- Sau khi chốt cấu hình: train lại trên toàn bộ 1545 mẫu cho bản nộp.

## 4. Các hướng

### A. SFT không thinking, LoRA (hướng chính)

- Lý do: độ trễ thấp nhất (52 token/bước), dễ kịp 30 s; ba lỗi chính là lỗi theo quy ước dữ liệu nên SFT sửa tốt.
- Cấu hình khởi đầu: LoRA r=32, alpha=64, dropout 0.05, target attention + linear-attention + MLP (theo 0.2); LR 1e-4 cosine, warmup 3%; 3 epoch; batch hiệu dụng ~16 chuỗi; bf16; gradient checkpointing.
- Thử: r 16/64, 2 vs 4 epoch, có/không câu dừng (thay bằng chuỗi rỗng).
- Mục tiêu: vượt baseline thinking zero-shot trên cả hai tập val.

### B. SFT có thinking bằng tự chưng cất (self-distillation)

- Dữ liệu: các bước `raw` trong `agent/runs/full/qwen35_4b_text_think.jsonl` của **465 mẫu điểm 1.0**. Bỏ mẫu thuộc tập val. Một ví dụ / bước, target = suy luận + call.
- Mở rộng: chạy thinking nhiều lần với seed/temperature khác trên mẫu sai, giữ trace nào cho điểm 1.0 (rejection sampling). Nếu cần, dùng teacher lớn hơn (luật cho phép dữ liệu tổng hợp lúc phát triển).
- **Giới hạn độ dài suy luận**: p50 136, p90 334, p99 1161 token/bước. Chỉ giữ trace ≤ ~300–400 token để kịp 30 s.
- Đã kiểm: trong 112 mẫu thinking không gọi call nào, chỉ 9 mẫu bị cắt ở bước cuối (35 bước chạm 6000 token). Phần lớn là model tự quyết định không gọi, không phải do hết token. Còn 155 bước không có `</think>` mà không chạm giới hạn, cần xem lại.
- So với A ở cùng tập val, tính cả token/bước.

### C. Tăng cường dữ liệu (dùng chung cho A và B)

1. **Lượt history làm ví dụ phụ**: cắt hội thoại tại mỗi lượt assistant có call trong history, coi lượt user ngay trước là "yêu cầu cuối". Thêm ~2981 ví dụ, rẻ. Lượt này đã có `text` trả lời thật.
2. **Nhiễu catalog** để dạy đọc schema thay vì thuộc tool: xáo thứ tự tool, bỏ bớt app không liên quan, thêm app nhiễu, đổi tên tool/tham số đồng loạt trong cả catalog lẫn target. Đo bằng val app-held-out.
3. **Nhiễu thời gian/thực thể**: dời `current_time`, đổi tên liên hệ/địa điểm và cập nhật nhất quán trong kết quả tool và target.
4. **App tổng hợp mới** (kiểu parking, stocks, grocery) do LLM lớn sinh + simulator tự viết: chỉ làm khi val app-held-out còn thấp sau 1–3.

### D. LoRA vs full fine-tune

- Full FT 4B trên 2×A100-40GB cần ZeRO-3 + gradient checkpointing (có thể kèm offload CPU, RAM myserver 452G). Chỉ thử khi A chững lại.
- Rủi ro: overfit với ~1.5k mẫu; LR thấp (1e-5), 1–2 epoch.

### E. Học tăng cường (sau cùng)

- DPO: cặp (đúng từ label, sai từ model SFT) tại bước sai đầu tiên.
- GRPO: reward = điểm `metrics/` (replay bằng `zaic_tools` trên train). Chỉ làm khi SFT ổn định.

## 5. Thứ tự việc

| # | Việc | Kết quả cần có |
|---|---|---|
| 1 | Bước 0 (0.1–0.6) | Biết train ở đâu, LoRA target, có serve adapter được không |
| 2 | `build_sft.py` + test token, chia val | File JSONL train/val, baseline zero-shot trên val |
| 3 | A: LoRA không thinking | Điểm val held-in/held-out, ba loại lỗi |
| 4 | C1 + C2 thêm vào A | Đo riêng tác dụng lên val held-out |
| 5 | B: thinking tự chưng cất | So với A theo điểm và token/bước |
| 6 | Benchmark độ trễ trên GPU gần A30 | Chọn thinking hay không |
| 7 | Train lại trên toàn bộ train, chạy public_test, nộp | Điểm leaderboard |
| 8 | D / E nếu còn thời gian | – |

## 6. Ngoài phạm vi plan này

- Lượt user lấy từ transcript ASR (30–50% ví dụ) để quen lỗi ASR: làm sau khi có PhoWhisper fine-tune chạy hết train.
- Quantize / tối ưu serving: làm sau; mọi số ở đây là bf16 để làm mốc.
- Kiểm tra tính tất định của vLLM (bản cắt và bản gốc lệch ~12.7% mẫu): làm song song, cần cho luật chạy lại.

## 7. Rủi ro

- **Overfit theo app**: 1545 mẫu, 46 app; nhìn val app-held-out, không chỉ held-in.
- **Lệch template train/serve**: chặn bằng test 0.6.
- **Kernel linear-attention**: không có thì train rất chậm (0.2).
- **Ổ đĩa myserver** (0.1).
