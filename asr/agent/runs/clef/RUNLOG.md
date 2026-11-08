# Nhật ký chạy Clef-token (agent/clef_agent.py)

## 2026-11-08 – zero-shot trên toàn bộ train (1545 mẫu)
- Model: Qwen3.5-4B bản cắt (`/hf/qwen35_4b_text` trên volume `hf-cache`), chưa train, không thinking. Server vLLM 0.31.0 trên Modal L40S (`serve_qwen35_4b_text`, app `lifemate-llm-zeroshot`), `--max-model-len 16384`, prefix caching bật.
- Lệnh: `python agent/clef_agent.py --data data/train+public_test --split train --url <vLLM> --out agent/runs/clef/zs_train.jsonl --workers 48`
- Output: `agent/runs/clef/zs_train.jsonl` (mỗi dòng: tool_calls, state_hash, decisions, n_requests, prompt_tokens_sum, anomaly, latency, error), log tiến độ `zs_train.log`.
- Bước 0 vLLM (`agent/clef/check_vllm.py`, prompt 7412 token, 51 phương án):
  - `allowed_token_ids` + `max_tokens=1`: trả đúng một nhãn (` B` → weather.get_forecast). OK.
  - Tất định: hai lần cùng request ra cùng output. OK.
  - Prefix cache với Qwen3.5 hybrid (Gated DeltaNet): trúng 14784/22254 token (66%); thời gian request 1/2/3 = 1.85 s / 1.23 s / 1.23 s. Cache có chạy.
  - Logprob top-5 tính trên toàn vocab (tổng 0.878, có `<|im_end|>` và ` [`), KHÔNG phải sau khi giới hạn nhãn → muốn dùng ngưỡng xác suất phải tự softmax lại trên nhãn từ logprobs.
- 42 mẫu đầu: 0 lỗi, 0 anomaly, 3 call bị validate từ chối, 7.4 request/mẫu; latency TB 37 s/mẫu ở 48 luồng đồng thời (không đại diện 30 s/mẫu trên A30).
- Kết quả (1545 mẫu, 0 lỗi, 0 anomaly, 266 call bị validate từ chối, 8.8 request/mẫu, 3.3 bước/mẫu): **19.71 điểm**, F1 34.99, state match 0.346 (khớp/gửi/vàng = 1236/3968/3541).
  - val_in 23.9, val_ood 11.5, phần train 21.1.
  - Theo số call vàng (điểm): 0 call 61.3 | 1 call 29.4 | 2 calls 20.7 | ≥3 calls 6.1. Zero-shot thường không thinking: 55.0 | 51.1 | 41.6 | 23.7; thinking: 72.1 | 55.4 | 45.2 | 34.7.
  - Từng mẫu: Clef-token zero-shot hơn zero-shot thường 152 mẫu, kém 537 mẫu. 153 mẫu không gọi tool nào.
  - Quyết định của model: Q1 chọn DỪNG 868 lần, chọn tool 4205, chọn `sys.ask_missing_info` 29 (nhãn vàng có 217 ca hỏi lại). Câu "điền tham số tùy chọn?" trả lời "có" 309 / 4473 lần (**7%**), trong khi nhãn vàng điền 1070 / 3229 lượt tham số tùy chọn (**33%**).
- So sánh cùng tập (agent.py, zero-shot thường): không thinking 37.5 (F1 50.8), thinking 45.2 (F1 55.9).

### Phân tích lỗi zero-shot Clef-token (`agent/clef/analyze.py`, khác biệt đầu tiên so với nhãn)
| Loại | Clef-token | Thường (không thinking) |
|---|---|---|
| Đúng hết | 194 | 375 |
| Sai tool | **598** | 363 |
| Thiếu tham số tùy chọn (tự do / enum) | **138 / 94** | 21 / 3 |
| Thừa tham số (tự do / enum) | 26 / 2 | **307 / 101** |
| Gọi thừa ở cuối | **102** | 12 |
| Không gọi gì | **85** | 14 |
| Không hỏi lại khi cần | 80 | 88 |
- Thiếu tùy chọn hay gặp: `smart_home.light_set.power` 46, `contacts.search.query` 41, `gold_price.get_price.type` 26 → câu "điền tham số X?" mâu thuẫn với luật system "KHÔNG tự thêm tham số tùy chọn", model trả "không" cả khi tham số là cốt lõi của yêu cầu.
- Bước đầu chọn đúng tool 903/1545 (58%). Thiên vị vị trí: lựa chọn rơi vào 10% đầu danh sách 1987 lần (vàng: 1459).
- Sai tool hay gặp: bill_payment.list_bills → ewallet/banking.get_balance (44), calendar.list_events → weather.get_forecast (24).
- Khối lượng: Clef 8.8 request, ~54.6k token prompt gửi (phần lớn trúng prefix cache), ~20 token sinh / mẫu; thường 3.1 request, 17.4k token prompt, 162 token sinh (thinking: 633).

## 2026-11-08 – thử 3 kiểu prompt zero-shot trên val_in (128 mẫu), `--style`
| Kiểu | Điểm | F1 | State | Request/mẫu | "có" cho câu hỏi tham số tùy chọn | Ghi chú lỗi (khác biệt đầu tiên) |
|---|---|---|---|---|---|---|
| v1 (bản đầu) | 23.9 | 35.0 | 0.406 | 8.8 | 7% | sai tool nhiều nhất |
| p1: câu hỏi tham số theo yêu cầu người dùng + "một trong" | 24.5 | 36.2 | 0.422 | 8.0 | 27% (80/297) | sai tool 40, gọi thừa 10 |
| p2: p1 + Q1 nhắc yêu cầu + tool đã gọi | **25.6** | **38.2** | 0.445 | 5.3 | 16% | sai tool 33, gọi thừa 0, nhưng không gọi gì 25 |
| p3: p2 + viết tên tool (structured output) thay nhãn | 18.3 | 25.4 | 0.359 | 3.0 | 20% | không gọi gì 42, hỏi lại thừa 31 |
| Zero-shot thường (agent.py), không thinking | 39.1 | 53.2 | 0.570 | 3.1 | – | – |
- Sửa prompt chỉ thêm 1–2 điểm (trong mức nhiễu của 128 mẫu); zero-shot Clef-token vẫn thấp hơn sinh tự do ~14 điểm. p3 làm model hay dừng / hỏi lại ngay bước đầu.
- Kết luận: định dạng câu hỏi tách rời lạ với model chưa train; lợi ích (nếu có) phải đến từ train. Nếu train, dùng p2 (giữ thông tin tiến độ cho chống gọi thừa / dừng sớm) hoặc p1.

## 2026-11-08 – smoke train Clef-token P2 trên Modal A100-80GB (`agent/clef/modal_train.py`)
- Image: vllm 0.31.0, transformers 5.17.0, peft 0.21.2, flash-linear-attention 0.6.0, torch 2.13.0. Dữ liệu `clef-data:/data/train_p2.jsonl` (3510 ví dụ, ~24M token), `val_in_p2.jsonl` (375).
- `--max-train 64 --epochs 1 --log-every 1` → 4 bước tối ưu (accum 16), xong 6.1 phút, 0 lỗi.
- flash-linear-attention: có. LoRA target: q/k/v/o_proj, gate/up/down_proj, và lớp Gated DeltaNet in_proj_qkv / in_proj_z / in_proj_a / in_proj_b / out_proj; 64.9M tham số train (1.52%).
- Tốc độ: bước 1 mất 244 s (biên dịch kernel Triton); các bước sau ~102k token / 35–40 s ≈ **2,800–2,900 token/s**. Bộ nhớ GPU tối đa 14.1 GB (dư nhiều trên 80 GB).
- Loss: train_choice 1.76 → 2.18, train_lm 0.29 → 0.02 (4 bước, chưa có ý nghĩa). EVAL 16 ví dụ: acc_q1 0.31, acc_yn 0.86, acc_opt 0.75.
- Ước tính lượt đủ (1 epoch, 24M token): ~2.3 giờ train + ~10 phút khởi động/eval ≈ 2.5 giờ ≈ **$6–7**.

## 2026-11-08 – train đủ Clef-token P2 (`clef-data:/runs/clef_p2_e1`, 1 epoch = 220 bước, accum 16, LR 1e-4)
| Bước | train_choice | train_lm | Ghi chú |
|---|---|---|---|
| 20 | 1.37 | 0.20 | 885 s |
| 40 | 1.04 | 0.16 | |
| 60 | 0.87 | 0.12 | |
| 80 | 0.90 | 0.14 | |
| 100 | 0.81 | 0.17 | ~2,820 token/s |
- **EVAL bước 110 (0.5 epoch), 375 ví dụ val_in, teacher-forcing (lịch sử vàng)**: val_choice 0.498, val_lm 0.120, **acc Q1 90.7%**, acc có/không 97.1%, acc chọn id/enum 97.0%. (Zero-shot: bước đầu đúng tool 58% trên train.)
| 120 | 0.85 | 0.08 | |
| 140 | 0.74 | 0.16 | |
| 160 | 0.83 | 0.12 | |
| 180 | 0.76 | 0.10 | |
| 200 | 0.80 | 0.12 | |
| 220 | 0.82 | 0.10 | 7669 s, 21.8M token, 2,841 token/s TB |
- **EVAL bước 220 (1 epoch)**, val_in teacher-forcing: val_choice 0.402, val_lm 0.108, **acc Q1 92.5%**, có/không 96.2%, id/enum 99.25%. Tổng thời gian job 131.5 phút. Adapter: `clef-data:/runs/clef_p2_e1/final` (+ `step110`, `step220`).

## 2026-11-08 – đánh giá agent thật: Clef-token P2 sau SFT 1 epoch
- Gộp adapter: `agent/clef/modal_merge.py` → `/hf/qwen35_4b_clef_p2` (4.2058B, như gốc; max |logit diff| adapter vs gộp 0.14, argmax bằng nhau). Serve: `serve_qwen35_4b_clef_p2` (vLLM 0.31, L40S).
- `clef_agent.py --style p2`, chấm `metrics/` trên đúng tập:

| | val_in (128) | val_ood (259, 5 app không có trong train) |
|---|---|---|
| **Clef-token P2, SFT 1 epoch** | **78.4** (F1 80.7, state 0.875) | **65.0** (F1 74.5, state 0.761) |
| Zero-shot thường, không thinking | 39.1 | 27.2 |
| Zero-shot thường, thinking | 46.0 | 35.7 |
| Zero-shot Clef-token P2 | 25.6 | – |

- 0 lỗi, 0 anomaly, 0 call bị validate từ chối. val_in: 7.0 request/mẫu, 2.83 bước; val_ood: 10.9 request/mẫu, 3.93 bước.
- Khác biệt đầu tiên (val_in): đúng hết 88/128; sai tool 15, sai text fuzzy 6, hỏi lại thừa 6, sai chuỗi 4. val_ood: đúng hết 116/259; sai tool 57, sai chuỗi 20, sai text fuzzy 15, sai enum 12, thừa / thiếu tham số tự do 11 / 11.
- App giữ lại (không có trong train): model gọi đúng tool vàng 300/323 lần (ewallet 85/91, alarm 91/96, flights 43/45, podcast_radio 40/43, loyalty 41/48).
- **Chưa có đối chứng SFT sinh text thường trên cùng dữ liệu**: chưa tách được bao nhiêu phần tăng là do SFT, bao nhiêu do định dạng Clef-token.
