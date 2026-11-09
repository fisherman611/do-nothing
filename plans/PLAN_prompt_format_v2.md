# Plan cải thiện prompt format Clef-token (v2)

Cập nhật: 2026-11-09. Đi kèm `PLAN_FC_clef_qwen35.md` (mục 7 ablation, mục 8 bước 5). Mọi thay đổi format phải sửa trong `agent/clef/fmt.py` + `core.py` để train và suy luận dùng chung, rồi build lại cả train lẫn val.

## 0. Hiện trạng (đo trên `agent/clef/data/train.jsonl`, style v1)

| Thứ | Số liệu | Nhận xét |
|---|---|---|
| Khối Q sau prefix | TB 2083 ký tự = ~9% chuỗi; prefix (system + JSON tools + history) ~22k ký tự | Khối Q không phải thứ làm chuỗi dài |
| Q1 | TB 33,5 phương án × mô tả 70 ký tự ≈ 80% khối Q | Lặp lại mô tả đã có trong JSON tools |
| Câu hỏi / bước | TB 1,93; median 1; max 6 (`smart_home.light_set`) | Bình thường |
| Câu "có điền tham số X không?" | 4860 câu, **60% trả lời "không"** | Phần phình chính; mỗi câu = 1 request khi suy luận (val_in: 7 request/mẫu) |
| Lỗi tham số tự do trên val_ood (run P2) | thừa 11 / thiếu 11 | Câu "có điền không?" chưa chứng minh được giá trị |
| Thiên vị vị trí Q1 | chưa đo | Clef có `EvidenceRoutingLayer`; ta chỉ xáo khi train |

## 1. Thay đổi đề xuất (theo thứ tự ưu tiên)

### 1.1 Cờ `--q2 all|discrete|none` (ablation chính)
- `all`: như hiện tại.
- `discrete`: chỉ hỏi Q2 cho tham số **ref / enum / bool**; tham số tự do tuỳ chọn không hỏi "có điền không?", phần sinh tự quyết điền hay bỏ (như SFT). Giữ câu "có điền không?" cho ref/enum tuỳ chọn vì vẫn cần biết có chọn hay không.
- `none`: chỉ Q1, model sinh toàn bộ tool call (ablation #2 của plan gốc).
- Kỳ vọng: `discrete` cắt phần lớn 4860 câu yes/no, giảm ~40% request khi suy luận, mất ít thông tin.
- Sửa: `core.run_step` (vòng `q2_plan`), `GoldOracle.gen` phải sinh được cả nhóm tham số tự do theo thứ tự `canonical_args` và dừng ở `</function>` (span lm bao cả phần "không điền gì thêm").

### 1.2 Q1 gọn: không lặp mô tả
- Cờ `--q1-desc full|short|none`: `short` = mô tả ≤ 40 ký tự; `none` = chỉ tên tool (như nhánh `short=True` hiện có).
- Lý do: JSON tools ở đầu prompt đã có mô tả đầy đủ. Cắt ~60% khối Q, và làm nhánh `SHORT_Q1_AT` (prompt > 13k token) không còn là format lạ.
- Rủi ro: zero-shot cho thấy model chưa train cần mô tả ở gần; sau train có thể không. Chỉ quyết định bằng ablation.

### 1.3 Mô tả tham số ở Q2 không cắt cụt
- `MAX_DESC` 70 → 120 cho Q2 (37 tham số đang bị cắt ngay giữa ví dụ định dạng, vd "…05:30:00+07:0…"). Q1 dùng giới hạn riêng (1.2).

### 1.4 Mẫu DỪNG từ history
- `build_data.history_items` giữ cả lượt assistant **không** gọi tool (510 lượt) thành mẫu phụ nhãn rỗng. Không đổi format, chỉ thêm ~510 ví dụ 1 bước (+6% token). Làm luôn, không cần ablation.

### 1.5 Đo và bù thiên vị vị trí (không đổi format)
- Script eval: chạy val_in hai lần với Q1 thứ tự catalog và thứ tự đảo ngược; báo tỉ lệ đổi đáp án. > 3% thì thêm xáo nhiều seed khi train (mỗi mẫu 2 bản xáo khác nhau) hoặc trung bình logit hai thứ tự khi suy luận (tốn gấp đôi request Q1).

### 1.6 Nhiệt độ hiệu chỉnh (chỉ khi dùng xác suất)
- Fit một nhiệt độ theo số phương án trên val_in (như Clef) nếu dùng P(Q1) để đặt ngưỡng hỏi lại / dừng. Không ảnh hưởng argmax, để sau.

## 1b. Đã làm (2026-11-09)

- `fmt.py`: `set_style(name, q2, q1_desc)`, `fmt_opts()`, `MAX_DESC` 70 → 120 (Q2), `Q1_DESC` full/short/none = 70/40/0, `canonical_args` theo chế độ (none: bắt buộc → tuỳ chọn), `TAIL_STOP`, `parse_tail`.
- `core.py`: `run_step` có 3 chế độ q2; tham số tuỳ chọn không hỏi thì model sinh gộp ở đuôi call (`oracle.gen_tail`), loss lm gồm cả `</function>` để học dừng. `GoldOracle.gen_tail` cho train.
- `build_data.py`: cờ `--q2`, `--q1-desc`; `history_items` giữ lượt không gọi tool (1.4); ghi `<out>.meta.json`.
- `train.py`: chép `fmt` từ meta vào `config.json`, chặn val khác format train.
- `clef_agent.py`: `--q2`, `--q1-desc`, `--fmt-from <config.json|meta.json>`; `VLLMOracle.gen_tail`.
- `agent.py::_coerce`: bool "True"/"False" → bool (cần cho q2=none).
- `test_fmt.py`: nhận 3 cờ; **2–5 OK ở cả 3 chế độ** (all/full, discrete/short, none/none; 400 mẫu).
- Data: `agent/clef/data/fmtA/` (q2=all, q1_desc=full) và `fmtB/` (q2=discrete, q1_desc=full), đều `--history` và có 1.4.
- Chưa làm: 1.5 (script đo thiên vị), 1.6, cấu hình C/D (chỉ cần build thêm, code đã có).

## 2. Ablation

Cùng data v1 + history (và 1.4), cùng seed, 1 epoch, `train.py --hub-kernels`, chấm bằng `clef_agent.py` + `metrics/` trên val_in và val_ood (không dùng public test).

| # | Cấu hình | Mục đích |
|---|---|---|
| A | `--q2 all --q1-desc full` (hiện tại, có 1.3 + 1.4) | baseline mới |
| B | `--q2 discrete --q1-desc full` | giá trị của câu "có điền không?" |
| C | `--q2 discrete --q1-desc none` | giá trị của mô tả trong Q1 |
| D | `--q2 none --q1-desc full` | điểm dừng sớm của plan gốc: Q2 có đáng không |

Chi phí: ~4,3 giờ / ~$11 mỗi run trên A100 (B–D rẻ hơn A một chút vì chuỗi ngắn hơn). Chạy A và B trước; C, D nếu B không thua A.

## 3. Tiêu chí chọn

- Chọn cấu hình **gọn nhất** mà điểm val_ood không thấp hơn A quá 1 điểm và val_in không giảm quá 1 điểm.
- Báo thêm: số request/mẫu, độ trễ/mẫu, lỗi thừa/thiếu tham số tự do, tỉ lệ đổi đáp án khi đảo thứ tự Q1 (1.5).
- Format được chọn ghi vào `config.json` của run (`style`, `q2`, `q1_desc`); `clef_agent.py` đọc từ đó thay vì cờ tay.

## 4. Rủi ro

- **Lệch train / suy luận** khi đổi format: kiểm bằng `test_fmt.py` (prompt train == prompt suy luận từng token) sau mỗi thay đổi; sửa luôn lỗi cwd của `test_loop.py`.
- **`discrete` làm tăng lỗi thừa/thiếu tham số tự do**: theo dõi riêng chỉ số này; nếu tăng rõ thì quay về `all` cho các tool có nhiều tham số tự do (cờ theo tool).
- **Mỗi thay đổi format đòi build lại** train (~3 phút CPU) và val, và data augment (alias/iface/phonetic) cũng phải build lại bằng cùng cờ.
