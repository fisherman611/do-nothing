# Một ví dụ train đầy đủ: `train_0126`, bước 1 (format A: v1 / q2=all / q1_desc=full)

Lấy nguyên văn từ `agent/clef/data/fmtA_mid/val_in.jsonl`. Toàn bộ chuỗi = **6971 token**. Phần model phải tính loss: các token `Đáp án: X` và giá trị tham số tự do; phần còn lại là prompt.

Câu người dùng: *Hoá đơn nào đến hạn trong vòng năm ngày tới thì trả bằng ngân hàng cho tôi, trừ hoá đơn trên 500 nghìn thì cứ để đó đã.*

## Tỉ lệ token theo phần

| Phần | Token | % |
|---|---|---|
| System prompt: phần chữ | 970 | 14% |
| System prompt: danh sách tools (JSON, 43 tool) | 4707 | 68% |
| Lượt user | 16 | 0% |
| Lượt assistant | 54 | 1% |
| Lượt user | 66 | 1% |
| Lượt assistant | 17 | 0% |
| Lượt user | 14 | 0% |
| Lượt assistant | 16 | 0% |
| Lượt user | 39 | 1% |
| Lượt assistant | 26 | 0% |
| Lượt user | 193 | 3% |
| Bước hiện tại: tiền tố assistant | 7 | 0% |
| Khối Q1 | 671 | 10% |
| Khối Q2 | 131 | 2% |
| Tool call (đầu ra) | 43 | 1% |

Lưu ý: system prompt + history + các bước trước (~6126 token) được vLLM **prefix-cache** nên chỉ prefill một lần cho cả mẫu; khối Q1/Q2 (~802 token) prefill lại mỗi bước.

## Danh sách tool trong system prompt

| # | Tool | Tham số | Token |
|---|---|---|---|
| 1 | `bill_payment.list_bills` | type | 80 |
| 2 | `bill_payment.pay` | **bill_id**, **source** | 105 |
| 3 | `bill_payment.schedule_autopay` | **type**, **enabled** | 72 |
| 4 | `banking.get_balance` | – | 46 |
| 5 | `banking.list_transactions` | account_id, limit | 107 |
| 6 | `banking.list_beneficiaries` | – | 49 |
| 7 | `banking.transfer` | **from_account_id**, **beneficiary_id**, **amount**, note | 195 |
| 8 | `banking.open_savings` | **from_account_id**, **amount**, **term_months** | 170 |
| 9 | `food_delivery.list_restaurants` | – | 60 |
| 10 | `food_delivery.get_menu` | **restaurant_id** | 82 |
| 11 | `food_delivery.order_history` | limit | 74 |
| 12 | `food_delivery.place_order` | **restaurant_id**, **items**, address, payment | 185 |
| 13 | `food_delivery.cancel_order` | **order_id** | 82 |
| 14 | `smart_home.get_status` | room, floor, type | 141 |
| 15 | `smart_home.light_set` | **device_id**, power, brightness, color | 142 |
| 16 | `smart_home.fan_set` | **device_id**, power, speed | 135 |
| 17 | `smart_home.climate_set` | **device_id**, **mode**, temperature | 148 |
| 18 | `smart_home.switch_set` | **device_id**, **power** | 111 |
| 19 | `smart_home.scene_run` | **scene_id** | 97 |
| 20 | `tv.guide` | channel, date | 97 |
| 21 | `tv.tune` | **channel** | 72 |
| 22 | `tv.control` | **action**, volume | 102 |
| 23 | `tv.record` | **program_id** | 81 |
| 24 | `medication.list` | – | 43 |
| 25 | `medication.add` | **name**, **times**, days | 103 |
| 26 | `medication.log_taken` | **schedule_id**, time | 138 |
| 27 | `medication.remove` | **schedule_id** | 81 |
| 28 | `alarm.list` | – | 44 |
| 29 | `alarm.create` | **time**, label, repeat | 145 |
| 30 | `alarm.update` | **alarm_id**, time, label, enabled | 166 |
| 31 | `alarm.delete` | **alarm_id** | 80 |
| 32 | `ride_hailing.get_quotes` | **pickup**, **dropoff**, vehicle | 113 |
| 33 | `ride_hailing.book` | **pickup**, **dropoff**, **vehicle**, schedule_time, payment | 277 |
| 34 | `ride_hailing.cancel` | **ride_id** | 85 |
| 35 | `ride_hailing.get_ride` | ride_id | 87 |
| 36 | `contacts.search` | query | 103 |
| 37 | `contacts.get` | **contact_id** | 77 |
| 38 | `contacts.update` | **contact_id**, phone, email | 132 |
| 39 | `music.list_tracks` | – | 47 |
| 40 | `music.play` | track_id, playlist_id, device | 143 |
| 41 | `music.control` | **action**, volume | 109 |
| 42 | `music.add_to_playlist` | **track_id**, **playlist_id** | 116 |
| 43 | `sys.ask_missing_info` | **tool_name**, **params** | 135 |

(tham số **đậm** = bắt buộc)

## Nguyên văn từng phần

### System prompt: phần chữ — 970 token

```text
<|im_start|>system
# Tools

You have access to the following functions:



If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>

Bạn là LifeMate, trợ lý ảo trên điện thoại của người dùng. Nhiệm vụ: thực hiện YÊU CẦU CUỐI CÙNG của người dùng bằng cách gọi các tool được cung cấp. Các lượt trước chỉ là ngữ cảnh, những gì đã làm ở đó KHÔNG làm lại.

Quy tắc:
- Mỗi lần gọi một tool, xem kết quả rồi mới quyết định bước tiếp theo. Khi đã xong, hoặc không cần/không thể gọi tool, trả lời ngắn gọn bằng tiếng Việt và KHÔNG gọi tool nữa.
- Chỉ gọi tool có trong danh sách. Không gọi tool đọc dữ liệu nếu không cần cho yêu cầu.
- ID (contact_id, event_id, bill_id, device_id, ...) phải lấy từ kết quả tool hoặc lịch sử hội thoại, không được tự đặt.
- Chỉ điền tham số bắt buộc và những tham số người dùng đã nói rõ. KHÔNG tự thêm tham số tùy chọn (location, label, repeat, ...) khi người dùng không nhắc tới.
- Một yêu cầu thường gồm nhiều bước (kiểm tra điều kiện, tra cứu ID, thực hiện, nhắn tin). Tiếp tục gọi tool cho tới khi hoàn thành MỌI việc được yêu cầu; chỉ trả lời bằng văn bản khi đã xong hết.
- sys.ask_missing_info CHỈ dùng khi thật sự không thể xác định một tham số bắt buộc (ví dụ tìm danh bạ ra nhiều người cùng tên, người dùng không nói số tiền hay giờ). Nếu thông tin đã có trong yêu cầu (giờ, số điện thoại, số tiền, tên) thì phải điền vào tool, không hỏi lại. Khi dùng, truyền tool_name và danh sách params còn thiếu rồi dừng.
- Không gọi tool nào nếu yêu cầu nằm ngoài khả năng của các app đã cài, nếu người dùng đổi ý hoặc huỷ yêu cầu, hoặc nếu chỉ là chào hỏi, cảm ơn.
- Yêu cầu có điều kiện ("nếu mai mưa thì ..."): kiểm tra điều kiện trước, chỉ làm tiếp khi điều kiện đúng.
- Thời điểm dùng ISO-8601 đầy đủ giây và múi giờ, ví dụ 2026-11-11T05:30:00+07:00; ngày dùng YYYY-MM-DD. Số tiền là số nguyên đồng (hai triệu tám = 2800000).
- Nội dung tin nhắn, ghi chú, tiêu đề: viết ngắn gọn, giữ đủ thông tin người dùng nói.

Bây giờ là 18:10 Thứ Năm, ngày 2026-11-19 (múi giờ Asia/Ho_Chi_Minh, offset +07:00).
Lịch: hôm nay: Thứ Năm 2026-11-19; ngày mai: Thứ Sáu 2026-11-20; ngày kia: Thứ Bảy 2026-11-21; +3 ngày: Chủ Nhật 2026-11-22; +4 ngày: Thứ Hai 2026-11-23; +5 ngày: Thứ Ba 2026-11-24; +6 ngày: Thứ Tư 2026-11-25; +7 ngày: Thứ Năm 2026-11-26.

Người dùng: Anh Nam. Nhà: 25 Láng Hạ, Đống Đa, Hà Nội. Nơi làm việc: Tòa nhà Kim Cương, Nam Từ Liêm, Hà Nội. Thành phố hiện tại: Hà Nội. Vị trí hiện tại: lat 21.0285, lng 105.8542.<|im_end|>
```

### System prompt: danh sách tools (JSON, 43 tool) — 4707 token

```text
<tools>
{"type": "function", "function": {"name": "bill_payment.list_bills", "description": "Hoá đơn chưa thanh toán", "parameters": {"type": "object", "properties": {"type": {"type": "string", "enum": ["electricity", "water", "internet", "tv", "tuition", "phone"]}}, "required": []}}}
{"type": "function", "function": {"name": "bill_payment.pay", "description": "Thanh toán hoá đơn", "parameters": {"type": "object", "properties": {"bill_id": {"type": "string", "description": "ID hoá đơn, lấy từ bill_payment.list_bills", "x-ref": "bill_payment.bills"}, "source": {"type": "string", "enum": ["bank", "ewallet"]}}, "required": ["bill_id", "source"]}}}
{"type": "function", "function": {"name": "bill_payment.schedule_autopay", "description": "Bật tự động thanh toán", "parameters": {"type": "object", "properties": {"type": {"type": "string"}, "enabled": {"type": "boolean"}}, "required": ["type", "enabled"]}}}
{"type": "function", "function": {"name": "banking.get_balance", "description": "Số dư các tài khoản", "parameters": {"type": "object", "properties": {}, "required": []}}}
{"type": "function", "function": {"name": "banking.list_transactions", "description": "Giao dịch gần đây", "parameters": {"type": "object", "properties": {"account_id": {"type": "string", "description": "ID tài khoản, lấy từ banking.get_balance", "x-ref": "banking.accounts"}, "limit": {"type": "integer", "description": "Số nguyên, không dấu chấm hay phẩy, ví dụ 2"}}, "required": []}}}
{"type": "function", "function": {"name": "banking.list_beneficiaries", "description": "Người thụ hưởng đã lưu", "parameters": {"type": "object", "properties": {}, "required": []}}}
{"type": "function", "function": {"name": "banking.transfer", "description": "Chuyển tiền cho người thụ hưởng", "parameters": {"type": "object", "properties": {"from_account_id": {"type": "string", "description": "ID tài khoản, lấy từ banking.get_balance", "x-ref": "banking.accounts"}, "beneficiary_id": {"type": "string", "description": "ID người thụ hưởng, lấy từ banking.list_beneficiaries", "x-ref": "banking.beneficiaries"}, "amount": {"type": "integer", "description": "Số nguyên, không dấu chấm, phẩy hay đơn vị, ví dụ 1200000", "exclusiveMinimum": 0}, "note": {"type": "string", "x-match": "fuzzy"}}, "required": ["from_account_id", "beneficiary_id", "amount"]}}}
{"type": "function", "function": {"name": "banking.open_savings", "description": "Gửi tiết kiệm", "parameters": {"type": "object", "properties": {"from_account_id": {"type": "string", "description": "ID tài khoản, lấy từ banking.get_balance", "x-ref": "banking.accounts"}, "amount": {"type": "integer", "description": "Số nguyên, không dấu chấm, phẩy hay đơn vị, ví dụ 1200000", "exclusiveMinimum": 0}, "term_months": {"type": "integer", "description": "Số nguyên, không dấu chấm hay phẩy, ví dụ 2", "exclusiveMinimum": 0}}, "required": ["from_account_id", "amount", "term_months"]}}}
{"type": "function", "function": {"name": "food_delivery.list_restaurants", "description": "Danh sách quán giao tận nơi quanh đây (tên, nhóm món, đánh giá)", "parameters": {"type": "object", "properties": {}, "required": []}}}
{"type": "function", "function": {"name": "food_delivery.get_menu", "description": "Thực đơn quán", "parameters": {"type": "object", "properties": {"restaurant_id": {"type": "string", "description": "ID quán, lấy từ food_delivery.list_restaurants", "x-ref": "food_delivery.restaurants"}}, "required": ["restaurant_id"]}}}
{"type": "function", "function": {"name": "food_delivery.order_history", "description": "Đơn đã đặt gần đây", "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "description": "Số nguyên, không dấu chấm hay phẩy, ví dụ 2"}}, "required": []}}}
{"type": "function", "function": {"name": "food_delivery.place_order", "description": "Đặt đơn", "parameters": {"type": "object", "properties": {"restaurant_id": {"type": "string", "description": "ID quán, lấy từ food_delivery.list_restaurants", "x-ref": "food_delivery.restaurants"}, "items": {"type": "array", "items": {"type": "object", "properties": {"item_id": {"type": "string"}, "quantity": {"type": "integer"}, "note": {"type": "string"}}, "required": ["item_id", "quantity"]}}, "address": {"type": "string", "x-match": "fuzzy"}, "payment": {"type": "string", "enum": ["cash", "ewallet", "bank"]}}, "required": ["restaurant_id", "items"]}}}
{"type": "function", "function": {"name": "food_delivery.cancel_order", "description": "Huỷ đơn đang chờ", "parameters": {"type": "object", "properties": {"order_id": {"type": "string", "description": "ID đơn, lấy từ food_delivery.order_history", "x-ref": "food_delivery.orders"}}, "required": ["order_id"]}}}
{"type": "function", "function": {"name": "smart_home.get_status", "description": "Lấy thiết bị và trạng thái; lọc theo phòng/tầng/loại; không lọc thì trả về cả danh sách ngữ cảnh (scenes) đã lưu", "parameters": {"type": "object", "properties": {"room": {"type": "string"}, "floor": {"type": "integer", "description": "Số nguyên, không dấu chấm hay phẩy, ví dụ 2"}, "type": {"type": "string", "enum": ["light", "fan", "climate", "curtain", "plug", "tv"]}}, "required": []}}}
{"type": "function", "function": {"name": "smart_home.light_set", "description": "Bật/tắt/chỉnh sáng đèn", "parameters": {"type": "object", "properties": {"device_id": {"type": "string", "description": "ID thiết bị, lấy từ smart_home.get_status", "x-ref": "smart_home.devices"}, "power": {"type": "string", "enum": ["on", "off"]}, "brightness": {"type": "integer", "description": "Số nguyên, không dấu chấm hay phẩy, ví dụ 2"}, "color": {"type": "string"}}, "required": ["device_id"]}}}
{"type": "function", "function": {"name": "smart_home.fan_set", "description": "Bật/tắt/chỉnh tốc độ quạt", "parameters": {"type": "object", "properties": {"device_id": {"type": "string", "description": "ID thiết bị, lấy từ smart_home.get_status", "x-ref": "smart_home.devices"}, "power": {"type": "string", "enum": ["on", "off"]}, "speed": {"type": "integer", "description": "Số nguyên, không dấu chấm hay phẩy, ví dụ 2"}}, "required": ["device_id"]}}}
{"type": "function", "function": {"name": "smart_home.climate_set", "description": "Đặt chế độ/nhiệt độ điều hoà", "parameters": {"type": "object", "properties": {"device_id": {"type": "string", "description": "ID thiết bị, lấy từ smart_home.get_status", "x-ref": "smart_home.devices"}, "mode": {"type": "string", "enum": ["cool", "heat", "dry", "fan", "off"]}, "temperature": {"type": "integer", "description": "Số nguyên, không dấu chấm hay phẩy, ví dụ 2"}}, "required": ["device_id", "mode"]}}}
{"type": "function", "function": {"name": "smart_home.switch_set", "description": "Bật/tắt ổ cắm, rèm, TV thông minh", "parameters": {"type": "object", "properties": {"device_id": {"type": "string", "description": "ID thiết bị, lấy từ smart_home.get_status", "x-ref": "smart_home.devices"}, "power": {"type": "string", "enum": ["on", "off"]}}, "required": ["device_id", "power"]}}}
{"type": "function", "function": {"name": "smart_home.scene_run", "description": "Chạy ngữ cảnh đã đặt sẵn (đi ngủ, ra khỏi nhà...)", "parameters": {"type": "object", "properties": {"scene_id": {"type": "string", "description": "ID ngữ cảnh, lấy từ smart_home.get_status (không lọc)", "x-ref": "smart_home.scenes"}}, "required": ["scene_id"]}}}
{"type": "function", "function": {"name": "tv.guide", "description": "Danh sách kênh đang có và lịch phát sóng (gọi trước khi chuyển kênh)", "parameters": {"type": "object", "properties": {"channel": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD, ví dụ 2026-11-29"}}, "required": []}}}
{"type": "function", "function": {"name": "tv.tune", "description": "Chuyển kênh (tự bật TV nếu đang tắt); kênh phải có trong channels của tv.guide", "parameters": {"type": "object", "properties": {"channel": {"type": "string"}}, "required": ["channel"]}}}
{"type": "function", "function": {"name": "tv.control", "description": "Bật/tắt/âm lượng TV", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["on", "off", "volume", "mute"]}, "volume": {"type": "integer", "description": "Số nguyên, không dấu chấm hay phẩy, ví dụ 2"}}, "required": ["action"]}}}
{"type": "function", "function": {"name": "tv.record", "description": "Đặt ghi hình chương trình", "parameters": {"type": "object", "properties": {"program_id": {"type": "string", "description": "ID chương trình, lấy từ tv.guide", "x-ref": "tv.programs"}}, "required": ["program_id"]}}}
{"type": "function", "function": {"name": "medication.list", "description": "Lịch thuốc", "parameters": {"type": "object", "properties": {}, "required": []}}}
{"type": "function", "function": {"name": "medication.add", "description": "Thêm nhắc thuốc", "parameters": {"type": "object", "properties": {"name": {"type": "string", "x-match": "fuzzy"}, "times": {"type": "array", "items": {"type": "string"}}, "days": {"type": "string", "enum": ["daily", "weekdays"]}}, "required": ["name", "times"]}}}
{"type": "function", "function": {"name": "medication.log_taken", "description": "Ghi đã uống", "parameters": {"type": "object", "properties": {"schedule_id": {"type": "string", "description": "ID lịch thuốc, lấy từ medication.list", "x-ref": "medication.schedules"}, "time": {"type": "string", "description": "ISO-8601 đầy đủ giây và offset múi giờ, ví dụ 2026-11-11T05:30:00+07:00"}}, "required": ["schedule_id"]}}}
{"type": "function", "function": {"name": "medication.remove", "description": "Xoá lịch thuốc", "parameters": {"type": "object", "properties": {"schedule_id": {"type": "string", "description": "ID lịch thuốc, lấy từ medication.list", "x-ref": "medication.schedules"}}, "required": ["schedule_id"]}}}
{"type": "function", "function": {"name": "alarm.list", "description": "Danh sách báo thức", "parameters": {"type": "object", "properties": {}, "required": []}}}
{"type": "function", "function": {"name": "alarm.create", "description": "Tạo báo thức", "parameters": {"type": "object", "properties": {"time": {"type": "string", "description": "ISO-8601 đầy đủ giây và offset múi giờ, ví dụ 2026-11-11T05:30:00+07:00"}, "label": {"type": "string", "x-match": "fuzzy"}, "repeat": {"type": "string", "enum": ["once", "daily", "weekdays", "weekends"]}}, "required": ["time"]}}}
{"type": "function", "function": {"name": "alarm.update", "description": "Sửa giờ/nhãn báo thức", "parameters": {"type": "object", "properties": {"alarm_id": {"type": "string", "description": "ID báo thức, lấy từ alarm.list", "x-ref": "alarm.alarms"}, "time": {"type": "string", "description": "ISO-8601 đầy đủ giây và offset múi giờ, ví dụ 2026-11-11T05:30:00+07:00"}, "label": {"type": "string", "x-match": "fuzzy"}, "enabled": {"type": "boolean"}}, "required": ["alarm_id"]}}}
{"type": "function", "function": {"name": "alarm.delete", "description": "Xoá báo thức", "parameters": {"type": "object", "properties": {"alarm_id": {"type": "string", "description": "ID báo thức, lấy từ alarm.list", "x-ref": "alarm.alarms"}}, "required": ["alarm_id"]}}}
{"type": "function", "function": {"name": "ride_hailing.get_quotes", "description": "Báo giá và thời gian đón", "parameters": {"type": "object", "properties": {"pickup": {"type": "string", "x-match": "fuzzy"}, "dropoff": {"type": "string", "x-match": "fuzzy"}, "vehicle": {"type": "string", "enum": ["bike", "car4", "car7"]}}, "required": ["pickup", "dropoff"]}}}
{"type": "function", "function": {"name": "ride_hailing.book", "description": "Đặt chuyến", "parameters": {"type": "object", "properties": {"pickup": {"type": "string", "x-match": "fuzzy", "description": "Điểm đón: địa chỉ hoặc tên địa điểm; 'nhà'/'công ty' → home_address/work_address của profile"}, "dropoff": {"type": "string", "x-match": "fuzzy", "description": "Điểm đến: địa chỉ hoặc tên địa điểm (lấy từ lịch/kết quả tìm kiếm khi người dùng chỉ việc đó)"}, "vehicle": {"type": "string", "enum": ["bike", "car4", "car7"]}, "schedule_time": {"type": "string", "description": "Giờ đón khi đặt trước, ISO-8601 đầy đủ giây và offset múi giờ, ví dụ 2026-11-11T05:30:00+07:00; bỏ trống khi đặt ngay (giờ bằng hoặc trước current_time cũng coi là đặt ngay)"}, "payment": {"type": "string", "enum": ["cash", "ewallet"]}}, "required": ["pickup", "dropoff", "vehicle"]}}}
{"type": "function", "function": {"name": "ride_hailing.cancel", "description": "Huỷ chuyến", "parameters": {"type": "object", "properties": {"ride_id": {"type": "string", "description": "ID chuyến xe, lấy từ ride_hailing.get_ride", "x-ref": "ride_hailing.rides"}}, "required": ["ride_id"]}}}
{"type": "function", "function": {"name": "ride_hailing.get_ride", "description": "Trạng thái chuyến hiện tại", "parameters": {"type": "object", "properties": {"ride_id": {"type": "string", "description": "ID chuyến xe, lấy từ ride_hailing.get_ride", "x-ref": "ride_hailing.rides"}}, "required": []}}}
{"type": "function", "function": {"name": "contacts.search", "description": "Tìm liên hệ theo tên, quan hệ hoặc số điện thoại; so chuỗi con, không phân biệt hoa thường và dấu; không truyền query thì trả cả danh bạ", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Tên, quan hệ (mẹ, vợ, sếp) hoặc số điện thoại"}}, "required": []}}}
{"type": "function", "function": {"name": "contacts.get", "description": "Chi tiết liên hệ", "parameters": {"type": "object", "properties": {"contact_id": {"type": "string", "description": "ID liên hệ, lấy từ contacts.search", "x-ref": "contacts.contacts"}}, "required": ["contact_id"]}}}
{"type": "function", "function": {"name": "contacts.update", "description": "Sửa số/email liên hệ", "parameters": {"type": "object", "properties": {"contact_id": {"type": "string", "description": "ID liên hệ, lấy từ contacts.search", "x-ref": "contacts.contacts"}, "phone": {"type": "string", "description": "Giữ nguyên chuỗi, ví dụ 09123456785"}, "email": {"type": "string"}}, "required": ["contact_id"], "x-one-of": [["phone"], ["email"]]}}}
{"type": "function", "function": {"name": "music.list_tracks", "description": "Thư viện bài hát và playlist", "parameters": {"type": "object", "properties": {}, "required": []}}}
{"type": "function", "function": {"name": "music.play", "description": "Phát một bài hoặc một playlist. Bắt buộc một trong: track_id hoặc playlist_id", "parameters": {"type": "object", "properties": {"track_id": {"type": "string", "description": "ID bài hát, lấy từ music.list_tracks", "x-ref": "music.tracks"}, "playlist_id": {"type": "string", "description": "ID playlist, lấy từ music.list_tracks", "x-ref": "music.playlists"}, "device": {"type": "string"}}, "required": [], "x-one-of": [["track_id"], ["playlist_id"]]}}}
{"type": "function", "function": {"name": "music.control", "description": "Tạm dừng/tiếp tục/chuyển bài/âm lượng", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["pause", "resume", "next", "previous", "volume"]}, "volume": {"type": "integer", "description": "Số nguyên, không dấu chấm hay phẩy, ví dụ 2"}}, "required": ["action"]}}}
{"type": "function", "function": {"name": "music.add_to_playlist", "description": "Thêm bài vào playlist", "parameters": {"type": "object", "properties": {"track_id": {"type": "string", "description": "ID bài hát, lấy từ music.list_tracks", "x-ref": "music.tracks"}, "playlist_id": {"type": "string", "description": "ID playlist, lấy từ music.list_tracks", "x-ref": "music.playlists"}}, "required": ["track_id", "playlist_id"]}}}
{"type": "function", "function": {"name": "sys.ask_missing_info", "description": "Báo các tham số bắt buộc còn thiếu của một tool", "parameters": {"type": "object", "properties": {"tool_name": {"type": "string", "description": "Tên tool bị thiếu tham số, dạng app_name.tool_name, ví dụ sms.send"}, "params": {"type": "array", "items": {"type": "string"}, "description": "Tên các tham số bắt buộc còn thiếu, đúng tên trong schema, ví dụ [\"contact_id\"]"}}, "required": ["tool_name", "params"]}}}
</tools>
```

### Lượt user — 16 token

```text
<|im_start|>user
Báo thức bảy giờ rưỡi sáng mai nhé.<|im_end|>
```

### Lượt assistant — 54 token

```text
<|im_start|>assistant
<tool_call>
<function=alarm.create>
<parameter=time>
2026-11-20T07:30:00+07:00
</parameter>
</function>
</tool_call><|im_end|>
```

### Lượt user — 66 token

```text
<|im_start|>user
<tool_response>
{"ok":true,"id":"alm_03","item":{"id":"alm_03","repeat":"once","enabled":true,"time":"2026-11-20T07:30:00+07:00"}}
</tool_response><|im_end|>
```

### Lượt assistant — 17 token

```text
<|im_start|>assistant
Đã đặt báo thức 7 rưỡi sáng mai.<|im_end|>
```

### Lượt user — 14 token

```text
<|im_start|>user
Cảm ơn nhé, chu đáo thật.<|im_end|>
```

### Lượt assistant — 16 token

```text
<|im_start|>assistant
Cảm ơn bạn, cần gì cứ nói nhé.<|im_end|>
```

### Lượt user — 39 token

```text
<|im_start|>user
Hoá đơn nào đến hạn trong vòng năm ngày tới thì trả bằng ngân hàng cho tôi, trừ hoá đơn trên 500 nghìn thì cứ để đó đã.<|im_end|>
```

### Lượt assistant — 26 token

```text
<|im_start|>assistant
<think>

</think>

<tool_call>
<function=bill_payment.list_bills>
</function>
</tool_call><|im_end|>
```

### Lượt user — 193 token

```text
<|im_start|>user
<tool_response>
{"ok":true,"bills":[{"id":"bill_01","type":"electricity","provider":"EVN","amount":780000,"due_date":"2026-11-21","paid":false},{"id":"bill_02","type":"water","provider":"Sawaco","amount":210000,"due_date":"2026-11-22","paid":false},{"id":"bill_03","type":"internet","provider":"Vitech Telecom","amount":220000,"due_date":"2026-11-20","paid":false},{"id":"bill_04","type":"tv","provider":"S+","amount":150000,"due_date":"2026-11-28","paid":false}]}
</tool_response><|im_end|>
```

### Bước hiện tại: tiền tố assistant — 7 token

```text
<|im_start|>assistant
<think>

</think>
```

### Khối Q1 — 671 token

```text
[Q1] Bước tiếp theo là gì?
A) bill_payment.list_bills – Hoá đơn chưa thanh toán
B) bill_payment.pay – Thanh toán hoá đơn
C) bill_payment.schedule_autopay – Bật tự động thanh toán
D) banking.get_balance – Số dư các tài khoản
E) banking.list_transactions – Giao dịch gần đây
F) banking.list_beneficiaries – Người thụ hưởng đã lưu
G) banking.transfer – Chuyển tiền cho người thụ hưởng
H) banking.open_savings – Gửi tiết kiệm
I) food_delivery.list_restaurants – Danh sách quán giao tận nơi quanh đây (tên, nhóm món, đánh giá)
J) food_delivery.get_menu – Thực đơn quán
K) food_delivery.order_history – Đơn đã đặt gần đây
L) food_delivery.place_order – Đặt đơn
M) food_delivery.cancel_order – Huỷ đơn đang chờ
N) smart_home.get_status – Lấy thiết bị và trạng thái; lọc theo phòng/tầng/loại; không lọc thì t…
O) smart_home.light_set – Bật/tắt/chỉnh sáng đèn
P) smart_home.fan_set – Bật/tắt/chỉnh tốc độ quạt
Q) smart_home.climate_set – Đặt chế độ/nhiệt độ điều hoà
R) smart_home.switch_set – Bật/tắt ổ cắm, rèm, TV thông minh
S) smart_home.scene_run – Chạy ngữ cảnh đã đặt sẵn (đi ngủ, ra khỏi nhà...)
T) tv.guide – Danh sách kênh đang có và lịch phát sóng (gọi trước khi chuyển kênh)
U) tv.tune – Chuyển kênh (tự bật TV nếu đang tắt); kênh phải có trong channels của…
V) tv.control – Bật/tắt/âm lượng TV
W) tv.record – Đặt ghi hình chương trình
X) medication.list – Lịch thuốc
Y) medication.add – Thêm nhắc thuốc
Z) medication.log_taken – Ghi đã uống
a) medication.remove – Xoá lịch thuốc
b) alarm.list – Danh sách báo thức
c) alarm.create – Tạo báo thức
d) alarm.update – Sửa giờ/nhãn báo thức
e) alarm.delete – Xoá báo thức
f) ride_hailing.get_quotes – Báo giá và thời gian đón
g) ride_hailing.book – Đặt chuyến
h) ride_hailing.cancel – Huỷ chuyến
i) ride_hailing.get_ride – Trạng thái chuyến hiện tại
j) contacts.search – Tìm liên hệ theo tên, quan hệ hoặc số điện thoại; so chuỗi con, không…
k) contacts.get – Chi tiết liên hệ
l) contacts.update – Sửa số/email liên hệ
m) music.list_tracks – Thư viện bài hát và playlist
n) music.play – Phát một bài hoặc một playlist. Bắt buộc một trong: track_id hoặc pla…
o) music.control – Tạm dừng/tiếp tục/chuyển bài/âm lượng
p) music.add_to_playlist – Thêm bài vào playlist
q) sys.ask_missing_info – Báo các tham số bắt buộc còn thiếu của một tool
r) DỪNG – Đã xong hoặc không cần gọi tool nào nữa
Đáp án: B
```

### Khối Q2 — 131 token

```text
[Q2] bill_payment.pay: bill_id = ?
A) bill_01 (electricity, EVN, 780000)
B) bill_02 (water, Sawaco, 210000)
C) bill_03 (internet, Vitech Telecom, 220000)
D) bill_04 (tv, S+, 150000)
Đáp án: B
[Q2] bill_payment.pay: source = ?
A) bank
B) ewallet
Đáp án: A
```

### Tool call (đầu ra) — 43 token

```text
<tool_call>
<function=bill_payment.pay>
<parameter=bill_id>
bill_02
</parameter>
<parameter=source>
bank
</parameter>
</function>
</tool_call>
```
