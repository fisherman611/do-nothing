# Báo cáo zero-shot LLM gọi tool – ZAC 2026 LifeMate

Cập nhật: 2026-11-08. Đi kèm `PLAN_tool_calling_LLM.md`. Code: `agent/modal_llm_server.py` (vLLM trên Modal), `agent/agent.py` (vòng lặp agent), `agent/strip_qwen35.py` (cắt Qwen3.5-4B), `agent/run_zeroshot.sh`, scorer `metrics/`.

## 1. Kết luận

- Zero-shot tốt nhất: **Qwen3.5-4B bản cắt (chỉ phần text, 4.206B) có thinking: 45.2 điểm** (F1 tie-break 55.9, state match 0.660). Không thinking: 37.5 điểm.
- Cắt visual + mtp **không làm đổi kết quả** (37.5 vs 37.6 của bản gốc) và đưa tổng tham số về mức hợp lệ (PhoWhisper-large 1.543B + 4.206B = 5.75B < 6B).
- Thinking cộng khoảng +7.8 điểm nhưng sinh nhiều token hơn ~4 lần. **Chưa biết có kịp 30 s/mẫu trên A30 hay không**; cần benchmark riêng trước khi chọn.
- Đây là trần của LLM trên transcript sạch (`audio_text` gốc), chưa có lỗi ASR, và trên tập train (đã dùng để thiết kế prompt).

## 2. Thiết lập

- Dữ liệu: 1545 mẫu train; lượt user lấy từ `audio_text` gốc.
- vLLM 0.31.0 trên Modal (L40S), pin revision HF, `temperature=0`, `seed=0`, context 16,384.
- Agent: vòng ReAct tối đa 8 bước, mỗi bước một tool call, `env.validate` trước `env.call`, mỗi mẫu một `ToolEnv`. Prompt hệ thống tiếng Việt (không thêm tham số tùy chọn, làm hết các bước được yêu cầu, chỉ `sys.ask_missing_info` khi thật sự thiếu thông tin).
- Chế độ thinking (`enable_thinking=True`): `max_tokens=6000`, chỉ phần sau `</think>` được parse, phần suy luận không đưa vào lịch sử hội thoại. Không thinking: `max_tokens=400`.
- Chấm bằng package `metrics/` (cùng cách tính với README): hash khớp × F1 call, tham số fuzzy khớp khi token-F1 **> 0.6**. Lưu ý `agent/score.py` cũ dùng ngưỡng ≥ 0.5, chưa sửa; các số trong báo cáo này đều từ `metrics/`. Chạy với `hash_verification: declared_only`.
- Luật cuộc thi không có điều khoản nào về thinking/reasoning, nên bật thinking là hợp lệ; ràng buộc thực tế chỉ là 30 s/mẫu và tất định.

## 3. Kết quả

| Model | Tham số (text) | Điểm | F1 tie-break | State match | Call khớp / gửi / vàng |
|---|---|---|---|---|---|
| Sailor2-3B-Chat | 3.58B | 7.2 | 7.2 | 0.245 | 0 / 0 / 3541 (không có định dạng tool) |
| Qwen3-1.7B | 2.03B | 12.0 | 18.1 | 0.269 | 449 / 2504 / 3541 |
| Qwen3-4B-Instruct-2507 | 4.02B | 25.5 | 38.0 | 0.425 | 1173 / 3111 / 3541 |
| Qwen3.5-4B gốc, không thinking | 4.21B (+0.45B visual/mtp) | 37.6 | 50.7 | 0.574 | 1825 / 4006 / 3541 |
| Qwen3.5-4B cắt, không thinking | 4.206B | 37.5 | 50.8 | 0.569 | 1823 / 4004 / 3541 |
| **Qwen3.5-4B cắt, có thinking** | 4.206B | **45.2** | **55.9** | **0.660** | 1929 / 3659 / 3541 |

Sàn (không gọi gì) có state match 0.245, nên mọi model có state ≤ 0.27 gần như không làm được gì.

Theo số call vàng (điểm mẫu / F1 / state match):

| Nhóm (số mẫu) | Qwen3-4B-2507 | Qwen3.5-4B cắt, không thinking | Qwen3.5-4B cắt, thinking |
|---|---|---|---|
| 0 call (111) | 61.3 / 61.3 / 0.86 | 55.0 / 55.0 / 0.78 | 72.1 / 72.1 / 0.86 |
| 1 call (369) | 44.4 / 45.3 / 0.70 | 51.1 / 52.4 / 0.73 | 55.4 / 57.2 / 0.81 |
| 2 calls (428) | 23.7 / 36.5 / 0.44 | 41.6 / 52.1 / 0.63 | 45.2 / 56.0 / 0.67 |
| ≥3 calls (637) | 9.7 / 30.9 / 0.18 | 23.7 / 48.1 / 0.39 | 34.7 / 52.4 / 0.54 |

## 4. Thinking có tốt hơn hẳn không?

- **Có về độ chính xác.** Hơn 7.8 điểm; so từng mẫu: thinking tốt hơn ở 324 mẫu, tệ hơn ở 166. Số mẫu đạt điểm tuyệt đối tăng từ 381 lên 465. Lợi ích rõ nhất ở chuỗi dài (≥3 calls: 23.7 → 34.7) và ở mẫu không cần gọi gì (55 → 72).
- **Chi phí token:** 220 token/bước so với 52, 633 token/mẫu so với 163 (2.87 bước/mẫu so với 3.12).
- **Chưa có số độ trễ hợp lệ.** Độ trễ đo trên Modal lúc ~96 request đồng thời chung một GPU nên không dùng được cho giới hạn 30 s. Tại A30 (chậm hơn L40S), 633 token/mẫu cộng ASR có thể sát giới hạn; mẫu quá giờ bị 0 điểm. Thời gian trung bình còn là tiêu chí phụ khi hòa điểm.
- Một điểm cần kiểm tra: thinking có 112 mẫu gửi 0 call (không thinking: 75), có thể do suy luận bị cắt ở 6000 token hoặc không ra tool call sau `</think>`. Chưa phân tích.
- Hướng cứu: SFT để model gọi đúng ngay không cần suy luận dài, hoặc giới hạn ngân sách thinking.

## 5. Phân tích lỗi

Số liệu loại lỗi dưới đây đo bằng `score.py` cũ (ngưỡng 0.5) trên Qwen3.5-4B gốc không thinking, chỉ dùng định tính:

| Loại lỗi | Qwen3-4B-2507 (1238 mẫu chưa hoàn hảo) | Qwen3.5-4B (1141) |
|---|---|---|
| Dừng sớm | 362 | 139 |
| Đúng tool, sai tham số | 307 | 326 |
| Khác | 271 | 249 |
| Không hỏi khi cần (`ask_missing_info`) | 187 | 179 |
| Thừa call | 40 | 184 |
| Gọi tool khi nhãn rỗng | 43 | 52 |
| Không gọi gì | 28 | 12 |

- Qwen3-4B-2507 hay dừng sớm nên chuỗi dài rất yếu; Qwen3.5-4B làm chuỗi dài tốt hơn nhưng hay gọi thừa.
- Cả hai yếu ở `ask_missing_info` (quy ước riêng của dữ liệu: hỏi lại khi `contacts.search` trả ≥ 2 kết quả) và ở tham số: thêm tham số tùy chọn (`room`, `priority`, `date_from/date_to`, `location`, `label`) dù prompt cấm. Đây là lỗi theo quy ước dữ liệu, SFT dễ sửa hơn sửa bằng prompt.

## 6. Cắt Qwen3.5-4B

- Gốc 4.66B (`Qwen3_5ForConditionalGeneration`) = text 4.206B + visual 0.334B + mtp 0.121B (embedding gắn với lm_head).
- Bản cắt: giữ mọi tensor `model.language_model.*` đổi tên thành `model.*`, config lấy từ `text_config` với `architectures=["Qwen3_5ForCausalLM"]`, `mtp_num_hidden_layers=0`. Lưu ở `/hf/qwen35_4b_text` trên volume `hf-cache`.
- Kiểm tra (CPU, bf16): tải được không thiếu/thừa key, 4.205751296B tham số, logit token cuối trên một prompt text **giống hệt** bản gốc (max |diff| = 0.0, argmax bằng nhau), vLLM 0.31.0 hỗ trợ `Qwen3_5ForCausalLM`.
- Kết quả đầu-cuối trên 1545 mẫu: 37.5 vs 37.6. Chênh 0.2 điểm (~12.7% mẫu có tool_calls khác nhau) chưa được giải thích; nghi ngờ do tính không tất định của vLLM khi batch, chưa kiểm chứng. Điều này quan trọng vì luật chạy lại nhiều lần và lấy trung bình nếu kết quả lệch.

## 7. Ràng buộc tham số

| Tổ hợp | Tổng | Hợp lệ (< 6B) |
|---|---|---|
| PhoWhisper-large 1.543B + Qwen3.5-4B cắt 4.206B | 5.75B | Có |
| PhoWhisper-large + Qwen3.5-4B gốc 4.66B | 6.21B | Không |
| PhoWhisper-large + Qwen3-4B-2507 4.02B | 5.57B | Có |
| Gemma 4 E4B-it (7.996B) / E2B-it (5.12B) | – | Không (LLM phải < 5B) |
| Qwen3-ASR-1.7B (2.038B) | – | Không, kể cả khi cắt LLM |

PhoWhisper-medium (0.764B, WER zero-shot 28.3%) là phương án ASR dự phòng nếu phải dùng LLM chưa cắt.

## 8. Chi phí Modal

L40S $1.95/giờ; server tự tắt sau 300 s rảnh. Workspace metered $12.66 → $14.67, nên đợt cắt + hai lần chạy 1545 mẫu (cùng một server, chạy song song) tốn khoảng **$2**. Tính đến nay đều trừ vào credit (billed $0).

## 9. Hạn chế

- Transcript sạch, chưa có lỗi ASR (PhoWhisper tốt nhất WER 6.2%, lỗi tên riêng còn 17/267 token).
- Tập train đã dùng để thiết kế prompt; public test có 11 app và 34 tool chưa từng thấy ở train (410/634 mẫu liên quan) nên điểm thực có thể thấp hơn.
- Một lần chạy mỗi cấu hình, chưa đo độ lặp lại.
- Không có số độ trễ hợp lệ cho 30 s/mẫu.
- `hash_verification: declared_only` trong scorer cục bộ.

## 10. Việc tiếp theo

1. Benchmark độ trễ trên GPU gần A30 (một request mỗi lần, gồm ASR), cho cả thinking và không thinking.
2. Kiểm tra tính tất định: chạy lại một tập con của bản cắt và so tool_calls.
3. Phân tích 112 mẫu thinking không gọi call nào.
4. SFT LoRA từng bước trên Qwen3.5-4B cắt, 30–50% lượt user lấy từ transcript ASR, giữ vài app làm tập kiểm tra app lạ; Qwen3-4B-2507 là phương án dự phòng.
5. Dữ liệu tổng hợp cho tool chưa thấy; nhắm vào ba lỗi: thiếu `ask_missing_info`, thừa tham số tùy chọn, dừng sớm.
6. Sửa `score.py` sang ngưỡng > 0.6 hoặc dùng `metrics/` làm scorer duy nhất.
