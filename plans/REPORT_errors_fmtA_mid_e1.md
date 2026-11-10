# Phân tích lỗi `fmtA_mid_e1` và đề xuất (2026-11-10)

Model: Qwen3.5-4B + LoRA Clef, format A (`q2=all`, `q1_desc=full`, style v1), train 1 epoch H100 trên `fmtA_mid`
(11000 ví dụ, lấy từ `fmtA_full`). Eval: `clef_agent.py` trên **text gold** (`audio_text`, chưa qua ASR), tool chạy
qua `zaic_tools`, chấm bằng `metrics/` (điểm = tool-F1 của mẫu nếu state cuối khớp gold, sai state = 0).

| | val_in (128) | val_ood (259) |
|---|---|---|
| Điểm | **75.8** (smoke A: 69.7) | **58.9** (smoke A: 59.4) |
| F1 | 80.1 | 71.0 |
| State khớp | 0.852 | 0.745 |
| Đúng hoàn toàn / state ok nhưng F1<1 / state sai | 88 / 21 / 19 | 96 / 97 / 66 |

Độ nhiễu: 1 mẫu = 0.78 điểm val_in, 0.39 điểm val_ood. Chênh dưới ~2–3 điểm coi như không phân biệt được.

Cách đo: mỗi call không khớp được ghép với call cùng tên bên kia, rồi tìm nguyên nhân ở mức tham số theo đúng luật
`_arguments_match` (thừa/thiếu khoá là sai; `x-match: fuzzy` cần token-F1 > 0.6; còn lại so bằng). Điểm mất của một
mẫu chia đều cho các nguyên nhân của nó. Script: `scratchpad/deep_err.py`, `loss_by_cat.py`, `err_analysis.py`.

---

## 0. Phát hiện quan trọng nhất: 3 app bị giữ lại để làm val_ood lại có trong test

`splits.json` giữ `alarm, ewallet, flights, loyalty, podcast_radio` ra khỏi train để đo tổng quát hoá. Theo catalog
tool của public test, **`alarm`, `ewallet`, `flights` có trong test**; `loyalty`, `podcast_radio` thì không.

Hệ quả:
- Model nộp bài hiện tại **chưa từng thấy** 3 app sẽ gặp ở test (val_ood có 90 mẫu alarm, 82 ewallet, 33 flights).
- Một phần lớn lỗi val_ood bên dưới (đánh dấu **[giữ lại]**) là do model không biết quy ước của các app này,
  ví dụ `alarm.create` gold gần như không bao giờ có `label` (1/78), `flights.search` dùng mã sân bay `HAN`/`VCA`,
  `loyalty.redeem.program` dùng id `prog_xxx`. Những lỗi này tự hết khi app có trong train.

→ Bản train để nộp phải có dữ liệu của alarm/ewallet/flights (xem giải pháp G1).

---

## 1. Điểm mất theo nguyên nhân

### val_ood (mất 41.1 điểm)

| Mất | Mẫu | Nguyên nhân | Ghi chú |
|---|---|---|---|
| 5.6 | 47 | Thừa call ghi | 22 lần `bill_payment.pay` (mục 2.2), nhầm ví ↔ ngân hàng (2.3) |
| 5.4 | 32 | Chuỗi chính xác sai | [giữ lại] `loyalty.redeem.program` 17, `flights.search.origin/destination` 12 (Hà Nội vs HAN); `contacts.search.query` chọn nhầm người |
| 4.1 | 34 | Thừa tham số tuỳ chọn (fuzzy) | [giữ lại] `alarm.create.label` 33 |
| 3.2 | 17 | Thiếu call ghi | Hệ quả của nhầm tool (ví ↔ ngân hàng), `alarm.create` bị bỏ |
| 3.0 | 31 | Thiếu call đọc | `ewallet.get_balance` 10 (gold kiểm tra số dư trước khi trả), `contacts.search` 7 |
| 3.0 | 26 | Chuỗi tự do lệch (token-F1 ≤ 0.6) | `sms.send.content` 9, `chat_ott.send_message.content` 6 |
| 2.9 | 25 | Thừa call đọc | `banking.list_beneficiaries` 6 (đi nhầm sang ngân hàng) |
| 1.9 | 6 | Số tiền sai bậc ×10 / ×1000 | "một trăm hai mươi nghìn" → 1200000 (mục 2.4) |
| 1.9 | 8 | Chọn nhầm id trong danh sách | contact, chuyến bay, tập podcast, booking |
| 1.8 | 10 | Sai giờ (dời/lùi tương đối) | Một lỗi kéo theo 3–4 call sai |
| 1.5 | 4 | Số sai giá trị (không phải lệch bậc) | Nhân giá × số lượng: 60000 vs 160000, 45000 vs 90000 |
| 1.6 + 0.9 | 16 | Thừa / thiếu tham số tuỳ chọn khác | `loyalty.get_points.program`, `calendar.create_event.end` |
| 1.0 | 5 | Gold hỏi lại, model không hỏi | |
| 0.7 | 3 | Số điện thoại | "không chín năm…" → `1950…`; code giữ nguyên chuỗi model sinh, nên đây là lỗi model |
| 0.8 + 0.7 | 7 | Enum/bool sai, sai ngày | |

### val_in (mất 24.2 điểm): các app đã thấy

| Mất | Mẫu | Nguyên nhân | Ghi chú |
|---|---|---|---|
| 4.0 | 9 | Thừa call ghi | Phần lớn là **tự làm thay vì hỏi lại** (mục 2.1) |
| 3.8 | 10 | Chuỗi tự do lệch | `chat_ott.send_message.content`, `notes/reminders.create.content`, `tasks.add.title` |
| 2.8 | 8 | Thiếu `sys.ask_missing_info` | Cùng nhóm với dòng trên |
| 2.4 + 1.6 | 15 | Thiếu / thừa call đọc | |
| 1.9 | 5 | Thiếu call ghi | |
| 1.2 + 1.0 | 4 | Số sai, chọn nhầm id | |

---

## 2. Phân tích các nhóm chính

### 2.1 Tự làm thay vì hỏi lại (ảo giác tham số)

Gold hỏi lại (`sys.ask_missing_info`) khi người dùng nói mơ hồ, tự sửa giữa câu, hoặc quên thông tin; model thì đoán
giá trị rồi thực thi luôn:

| Mẫu | Câu | Model | Gold |
|---|---|---|---|
| train_0961 | "…bật tự trả một cái thôi, cái hạn thứ bảy, à không, cái hạn chủ nhật." | `schedule_autopay(type=internet)` | hỏi `type` |
| train_1029 | "…À thôi, khỏi. Cái sao kê chị in ra rồi, xoá mail đó đi." | `archive(email_id=mail_676, action=delete)` | hỏi `email_id` |
| train_1139 | "…khoá tiếng gì nhỉ, tôi quên mất rồi." | `log_progress(course_id=course_481)` | hỏi `course_id` |
| train_0684 | "…hôm đó anh đi từ nhà ông bà ngoại cơ, xem đường đi." | `get_route(origin=25 Láng Hạ…)` (địa chỉ nhà) | hỏi `origin` |

Đây là app đã thấy, nên lỗi này sẽ gặp ở test. Gọi một call ghi sai làm hỏng state (0 điểm), nên tệ hơn là không làm
gì. Dữ liệu train có 540 mẫu ask trên 8696 (6%), ít kiểu mơ hồ khó như tự sửa câu hay "quên mất".

### 2.2 Câu điều kiện và tool trả lỗi (`bill_payment.pay` thừa, 22 lần)

> train_0011: "Nếu ví đủ thì trả điện và nhắn Bố là đã đóng, còn không thì trả ngân hàng và nhắc tôi tám giờ tối nạp ví."

Model gọi `get_balance()` nhưng vẫn trả bằng ví, bị từ chối vì ví không đủ, rồi trả lại bằng ngân hàng. Gold đọc số dư,
so với số tiền rồi đi thẳng nhánh ngân hàng. State cuối vẫn khớp (call bị từ chối không đổi state), nhưng call thừa làm
F1 giảm.

Hai nguyên nhân:
- Model **không so sánh số** (số dư với số tiền hoá đơn) trước khi quyết định nhánh.
- Trong toàn bộ dữ liệu train, kết quả tool **luôn là `ok: true`**. Model chưa từng thấy call bị từ chối, nên không học
  được rằng thử rồi sửa sẽ bị phạt.

Một phần là do [giữ lại] (`ewallet.get_balance` là app chưa thấy), nhưng kiểu "nếu A thì X còn không thì Y" có ở mọi app.

### 2.3 Nhầm app tương tự (ví ↔ ngân hàng)

> train_0145: "Dùng ví điện tử nạp… rồi chuyển cho Bố đúng bằng tiền nước tháng này."
> Model: `banking.get_balance`, `banking.list_beneficiaries`, `banking.transfer(...)`. Gold: `contacts.search(Bố)`, `ewallet.transfer(contact_id=c_06)`.

Model chưa thấy `ewallet.transfer` nên kéo về `banking.transfer` quen thuộc, dù câu nói rõ "ví điện tử". [giữ lại] với
ewallet, nhưng đây là **mô phỏng tốt nhất cho 11 app chỉ có ở test**: gặp tool mới gần giống tool cũ, model chọn tool cũ.

### 2.4 Số đọc bằng chữ

| Mẫu | Câu | Model | Gold |
|---|---|---|---|
| train_0500 | "một trăm hai mươi nghìn" | 1200000 | 120000 |
| train_0874 | "một trăm rưỡi" | 1500000 | 150000 |
| train_1260 | "ba trăm" (ngầm là nghìn) | 3000000 | 300000 |
| train_1265 | "hai trăm", "ba trăm" | 200, 300 | 200000, 300000 |
| train_0009 | "không chín năm không năm…" | 19505360367 | 09505360367 |
| train_1326 | vé trẻ em + 1 vé người lớn | 60000 | 160000 (phải cộng giá) |

Lỗi xảy ra cả với `banking.transfer` (app đã thấy). Ở test, đầu vào là bản ASR, còn số có thể ở dạng chữ hoặc số tuỳ ASR,
nên lỗi này nhiều khả năng còn nặng hơn.

### 2.5 Thời gian tương đối

train_0292, train_0381: "lùi/dời sang…" → model tính sai giờ mới (lệch 90–120 phút). Sai giờ ở
`calendar.update_event` kéo theo `alarm.create.time` và nội dung `sms.send` đều sai, nên một lỗi mất cả mẫu.

### 2.6 Chuỗi tự do lệch (nội dung tin nhắn, tiêu đề)

Luật chấm: token-F1 với gold > 0.6. Model diễn đạt lại thay vì giữ lời người dùng:
- "Chị đóng tiền mạng" so với "đã đóng tiền mạng"
- "Nghe xong tập podcast "Quy hoạch xanh…"" so với "Nghe xong tập Quy hoạch xanh…"

Câu càng ngắn càng dễ rớt dưới 0.6.

### 2.7 Thừa tham số tuỳ chọn

Thừa một khoá là call sai. Model trả lời "có" cho các tham số tuỳ chọn mà gold để trống: `alarm.create.label` 33 lần
[giữ lại], `door_lock.grant_guest_code.label`, `calendar.list_events.date_from/date_to`. Với tool lạ ở test, đây là lỗi
chắc chắn sẽ gặp.

### 2.8 Chọn nhầm item trong danh sách

Model chọn id từ ứng viên có thật trong kết quả tool (không bịa id), nhưng chọn nhầm item. Phương án A được chọn hơi
nhiều ở các mẫu sai (24/41 so với 188/367 toàn bộ). Lúc infer danh sách ứng viên không bị xáo (A là id xuất hiện trước),
nên chưa đủ để kết luận model thiên vị vị trí. Cần thử xáo lúc infer (mục G8).

---

## 3. Latency (luật 30 giây)

`metrics/scorer.py` cho 0 điểm nếu `elapsed_seconds > 30`; `score_subsets.py` chưa áp luật này. Đo lại 40 mẫu (1 worker,
L40S, Modal us-east, gọi từ VN):

- Theo hồi quy (R² 0.99): **latency ≈ 0.85 s × số request + 0.067 s × nghìn token prompt**. Trung bình 7.5k token/request,
  nên khoảng **1.35 s/request, trong đó ~0.85 s là phần cố định** (mạng VN↔Mỹ, HTTP, lập lịch, decode vài token).
- Mẫu ≥ 14 request: p50 23.7 s, 5/24 quá 30 s. Mẫu < 14 request: 0/15.

Trong Docker offline, phần mạng mất hẳn, nên nhiều khả năng chỉ còn khoảng 0.4–0.6 s/request, tức ~8 s cho mẫu 14 request.
Đây là suy luận, phải đo trên phần cứng giống BTC. Nếu phần cứng BTC yếu hơn L40S thì cần giảm số request (G7).

---

## 4. Đề xuất giải pháp (theo thứ tự ưu tiên)

| # | Giải pháp | Nhắm vào | Ước tác động | Chi phí |
|---|---|---|---|---|
| G1 | **Đưa alarm/ewallet/flights vào train bản nộp**; đổi val_ood thành loyalty + podcast_radio + bộ test-like (Part B1) + `val_alias` | Mục 0, toàn bộ [giữ lại] | Lớn nhất với test (3 app test chưa từng thấy) | Build lại data + train |
| G2 | **Hỏi lại thay vì đoán**: sinh thêm mẫu ask khó (tự sửa câu, "quên mất", tham chiếu mơ hồ, thông tin không có trong context); lúc infer, chọn `sys.ask_missing_info` khi xác suất của nó gần bằng tool tốt nhất | 2.1 | val_in +3–5 | Gemini + build lại; ngưỡng thì chỉ sửa code infer |
| G3 | **Chuẩn hoá số và giờ tiếng Việt** bằng bộ parse cố định: "một trăm hai mươi nghìn" → 120000, "không chín năm…" → 0950…, "8 giờ tối nay" / "lùi 1 tiếng" → ISO; thêm dòng gợi ý vào prompt (đi qua cả lúc build data). Kèm augment đổi số trong câu train sang dạng chữ | 2.4, 2.5 | val_ood +3–4, giảm rủi ro ASR | 1–2 ngày code + build lại |
| G4 | **Ngưỡng điền tham số tuỳ chọn**: chỉ điền tham số tuỳ chọn tự do khi P("có") > τ, chọn τ trên val_in | 2.7 | val_ood ~+2–3 (phần lớn do alarm) | Chỉ sửa infer, không train |
| G5 | **Dữ liệu điều kiện và lỗi tool**: mẫu "nếu… thì… không thì…" với số dư / giá đủ và không đủ (cả hai nhánh); thêm kết quả `ok: false` trong history để model biết call bị từ chối là có thật | 2.2 | val_ood +2–4 | Gemini + build lại |
| G6 | **Giữ lời người dùng trong text tự do**: thêm ví dụ train mà nội dung trích nguyên văn; cân nhắc trọng số loss cao hơn cho span `lm` ngắn | 2.6 | +1–3 | Nhỏ |
| G7 | **Latency**: đo trong Docker offline trên GPU tương đương; nếu vẫn sát 30 s thì bỏ câu "có điền không?" cho tham số bắt buộc, lấy logprob nhiều tham số tuỳ chọn trong một request, hỏi song song các Q2 độc lập | Mục 3 | Tránh mất ~5% mẫu | Đo trước, sửa sau |
| G8 | **Kiểm tra thiên vị vị trí**: chạy lại val với ứng viên `ref`/Q1 được xáo lúc infer, so điểm (PLAN_prompt_format_v2 mục 1.5) | 2.8 | Chẩn đoán | ~$1 |
| G9 | **Eval trên bản ASR** của audio val (train audio, hợp lệ) qua `clef_agent.py --asr`, để thấy lỗi số/tên sau ASR | Mọi mục | Chẩn đoán | Nhỏ |

**Chưa nên train epoch 2:** loss val vẫn giảm, nhưng các lỗi trên là do thiếu dữ liệu đúng loại (ask, điều kiện, số,
app test) hoặc do luật suy luận, không phải do train chưa đủ lâu.

### Thứ tự làm đề xuất
1. G4 + G8 (chỉ sửa infer, có kết quả trong ngày, rẻ).
2. G9 + G7-đo (biết rủi ro thật trước khi đầu tư dữ liệu).
3. G1 + G3 + G2/G5 gộp vào **một lần build data mới**, rồi train một lần.
