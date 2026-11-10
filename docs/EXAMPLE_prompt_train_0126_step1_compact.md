# Cùng ví dụ train_0126 bước 1, format gọn: sys=compact / rules=short / q1_desc=short

Nguyên văn từ `agent/clef/data/calib/val_in_compact.jsonl`. Toàn bộ chuỗi = **3956 token**. Loss tính trên token `Đáp án: X` và giá trị tham số tự do; phần còn lại là prompt.

Câu người dùng: *Hoá đơn nào đến hạn trong vòng năm ngày tới thì trả bằng ngân hàng cho tôi, trừ hoá đơn trên 500 nghìn thì cứ để đó đã.*

## Tỉ lệ token theo phần

| Phần | Token | % |
|---|---|---|
| System prompt: phần chữ | 544 | 14% |
| System prompt: tools (compact) | 2164 | 55% |
| Lượt user | 16 | 0% |
| Lượt assistant | 54 | 1% |
| Lượt user | 66 | 2% |
| Lượt assistant | 17 | 0% |
| Lượt user | 14 | 0% |
| Lượt assistant | 16 | 0% |
| Lượt user | 39 | 1% |
| Lượt assistant | 26 | 1% |
| Lượt user | 193 | 5% |
| Bước hiện tại: tiền tố assistant | 7 | 0% |
| Khối Q1 | 626 | 16% |
| Khối Q2 | 131 | 3% |
| Tool call (đầu ra) | 43 | 1% |

Prefix cache được (system + history + bước trước): ~3156 token, prefill một lần/mẫu. Khối Q1/Q2 (~757 token) prefill lại mỗi bước.

## Nguyên văn từng phần

### System prompt: phần chữ — 544 token

```text
<|im_start|>system
Bạn là LifeMate, trợ lý ảo trên điện thoại. Thực hiện YÊU CẦU CUỐI của người dùng bằng cách gọi tool; các lượt trước chỉ là ngữ cảnh, việc đã làm ở đó không làm lại.

Luật:
- ID (contact_id, bill_id, ...) lấy từ kết quả tool hoặc lịch sử, không tự đặt. Không gọi tool đọc nếu không cần.
- Chỉ điền tham số bắt buộc và tham số người dùng nói rõ; không tự thêm tham số tuỳ chọn.
- Yêu cầu có điều kiện: kiểm tra điều kiện trước, chỉ làm khi đúng. Làm hết MỌI việc được yêu cầu rồi mới dừng.
- sys.ask_missing_info chỉ khi một tham số bắt buộc không thể xác định (nhiều người trùng tên, thiếu số tiền/giờ...). Không gọi tool nếu yêu cầu ngoài khả năng app, người dùng đổi ý/huỷ, hoặc chỉ chào hỏi.
- Thời điểm: ISO-8601 đủ giây và múi giờ (2026-11-11T05:30:00+07:00); ngày: YYYY-MM-DD; tiền: số nguyên đồng (hai triệu tám = 2800000). Tin nhắn/ghi chú/tiêu đề: ngắn gọn, giữ đủ ý người dùng.

Bây giờ là 18:10 Thứ Năm, ngày 2026-11-19 (múi giờ Asia/Ho_Chi_Minh, offset +07:00).
Lịch: hôm nay: Thứ Năm 2026-11-19; ngày mai: Thứ Sáu 2026-11-20; ngày kia: Thứ Bảy 2026-11-21; +3 ngày: Chủ Nhật 2026-11-22; +4 ngày: Thứ Hai 2026-11-23; +5 ngày: Thứ Ba 2026-11-24; +6 ngày: Thứ Tư 2026-11-25; +7 ngày: Thứ Năm 2026-11-26.

Người dùng: Anh Nam. Nhà: 25 Láng Hạ, Đống Đa, Hà Nội. Nơi làm việc: Tòa nhà Kim Cương, Nam Từ Liêm, Hà Nội. Thành phố: Hà Nội. Vị trí: lat 21.0285, lng 105.8542.
```

### System prompt: tools (compact) — 2164 token

```text
# Tools (tên đầy đủ: app.tool)
[bill_payment]
- list_bills: Hoá đơn chưa thanh toán
    · type?: electricity|water|internet|tv|tuition|phone
- pay: Thanh toán hoá đơn
    · bill_id: id←bill_payment.bills — ID hoá đơn, lấy từ bill_payment.list_bills
    · source: bank|ewallet
- schedule_autopay: Bật tự động thanh toán
    · type: str
    · enabled: bool
[banking]
- get_balance: Số dư các tài khoản
- list_transactions: Giao dịch gần đây
    · account_id?: id←banking.accounts — ID tài khoản, lấy từ banking.get_balance
    · limit?: int — Số nguyên, không dấu chấm hay phẩy, ví dụ 2
- list_beneficiaries: Người thụ hưởng đã lưu
- transfer: Chuyển tiền cho người thụ hưởng
    · from_account_id: id←banking.accounts — ID tài khoản, lấy từ banking.get_balance
    · beneficiary_id: id←banking.beneficiaries — ID người thụ hưởng, lấy từ banking.list_beneficiaries
    · amount: int — Số nguyên, không dấu chấm, phẩy hay đơn vị, ví dụ 1200000
    · note?: str
- open_savings: Gửi tiết kiệm
    · from_account_id: id←banking.accounts — ID tài khoản, lấy từ banking.get_balance
    · amount: int — Số nguyên, không dấu chấm, phẩy hay đơn vị, ví dụ 1200000
    · term_months: int — Số nguyên, không dấu chấm hay phẩy, ví dụ 2
[food_delivery]
- list_restaurants: Danh sách quán giao tận nơi quanh đây (tên, nhóm món, đánh giá)
- get_menu: Thực đơn quán
    · restaurant_id: id←food_delivery.restaurants — ID quán, lấy từ food_delivery.list_restaurants
- order_history: Đơn đã đặt gần đây
    · limit?: int — Số nguyên, không dấu chấm hay phẩy, ví dụ 2
- place_order: Đặt đơn
    · restaurant_id: id←food_delivery.restaurants — ID quán, lấy từ food_delivery.list_restaurants
    · items: list[obj]
    · address?: str
    · payment?: cash|ewallet|bank
- cancel_order: Huỷ đơn đang chờ
    · order_id: id←food_delivery.orders — ID đơn, lấy từ food_delivery.order_history
[smart_home]
- get_status: Lấy thiết bị và trạng thái; lọc theo phòng/tầng/loại; không lọc thì trả về cả danh sách ngữ cảnh (scenes) đã lưu
    · room?: str
    · floor?: int — Số nguyên, không dấu chấm hay phẩy, ví dụ 2
    · type?: light|fan|climate|curtain|plug|tv
- light_set: Bật/tắt/chỉnh sáng đèn
    · device_id: id←smart_home.devices — ID thiết bị, lấy từ smart_home.get_status
    · power?: on|off
    · brightness?: int — Số nguyên, không dấu chấm hay phẩy, ví dụ 2
    · color?: str
- fan_set: Bật/tắt/chỉnh tốc độ quạt
    · device_id: id←smart_home.devices — ID thiết bị, lấy từ smart_home.get_status
    · power?: on|off
    · speed?: int — Số nguyên, không dấu chấm hay phẩy, ví dụ 2
- climate_set: Đặt chế độ/nhiệt độ điều hoà
    · device_id: id←smart_home.devices — ID thiết bị, lấy từ smart_home.get_status
    · mode: cool|heat|dry|fan|off
    · temperature?: int — Số nguyên, không dấu chấm hay phẩy, ví dụ 2
- switch_set: Bật/tắt ổ cắm, rèm, TV thông minh
    · device_id: id←smart_home.devices — ID thiết bị, lấy từ smart_home.get_status
    · power: on|off
- scene_run: Chạy ngữ cảnh đã đặt sẵn (đi ngủ, ra khỏi nhà...)
    · scene_id: id←smart_home.scenes — ID ngữ cảnh, lấy từ smart_home.get_status (không lọc)
[tv]
- guide: Danh sách kênh đang có và lịch phát sóng (gọi trước khi chuyển kênh)
    · channel?: str
    · date?: str — YYYY-MM-DD, ví dụ 2026-11-29
- tune: Chuyển kênh (tự bật TV nếu đang tắt); kênh phải có trong channels của tv.guide
    · channel: str
- control: Bật/tắt/âm lượng TV
    · action: on|off|volume|mute
    · volume?: int — Số nguyên, không dấu chấm hay phẩy, ví dụ 2
- record: Đặt ghi hình chương trình
    · program_id: id←tv.programs — ID chương trình, lấy từ tv.guide
[medication]
- list: Lịch thuốc
- add: Thêm nhắc thuốc
    · name: str
    · times: list[str]
    · days?: daily|weekdays
- log_taken: Ghi đã uống
    · schedule_id: id←medication.schedules — ID lịch thuốc, lấy từ medication.list
    · time?: str — ISO-8601 đầy đủ giây và offset múi giờ, ví dụ 2026-11-11T05:30:00+07:00
- remove: Xoá lịch thuốc
    · schedule_id: id←medication.schedules — ID lịch thuốc, lấy từ medication.list
[alarm]
- list: Danh sách báo thức
- create: Tạo báo thức
    · time: str — ISO-8601 đầy đủ giây và offset múi giờ, ví dụ 2026-11-11T05:30:00+07:00
    · label?: str
    · repeat?: once|daily|weekdays|weekends
- update: Sửa giờ/nhãn báo thức
    · alarm_id: id←alarm.alarms — ID báo thức, lấy từ alarm.list
    · time?: str — ISO-8601 đầy đủ giây và offset múi giờ, ví dụ 2026-11-11T05:30:00+07:00
    · label?: str
    · enabled?: bool
- delete: Xoá báo thức
    · alarm_id: id←alarm.alarms — ID báo thức, lấy từ alarm.list
[ride_hailing]
- get_quotes: Báo giá và thời gian đón
    · pickup: str
    · dropoff: str
    · vehicle?: bike|car4|car7
- book: Đặt chuyến
    · pickup: str — Điểm đón: địa chỉ hoặc tên địa điểm; 'nhà'/'công ty' → home_address/work_address của profile
    · dropoff: str — Điểm đến: địa chỉ hoặc tên địa điểm (lấy từ lịch/kết quả tìm kiếm khi người dùng chỉ việc đó)
    · vehicle: bike|car4|car7
    · schedule_time?: str — Giờ đón khi đặt trước, ISO-8601 đầy đủ giây và offset múi giờ, ví dụ 2026-11-11T05:30:00+07:00; bỏ trống khi đặt ngay (giờ bằng hoặc trước current_time cũng coi là đặt ngay)
    · payment?: cash|ewallet
- cancel: Huỷ chuyến
    · ride_id: id←ride_hailing.rides — ID chuyến xe, lấy từ ride_hailing.get_ride
- get_ride: Trạng thái chuyến hiện tại
    · ride_id?: id←ride_hailing.rides — ID chuyến xe, lấy từ ride_hailing.get_ride
[contacts]
- search: Tìm liên hệ theo tên, quan hệ hoặc số điện thoại; so chuỗi con, không phân biệt hoa thường và dấu; không truyền query thì trả cả danh bạ
    · query?: str — Tên, quan hệ (mẹ, vợ, sếp) hoặc số điện thoại
- get: Chi tiết liên hệ
    · contact_id: id←contacts.contacts — ID liên hệ, lấy từ contacts.search
- update: Sửa số/email liên hệ
    · contact_id: id←contacts.contacts — ID liên hệ, lấy từ contacts.search
    · phone?: str — Giữ nguyên chuỗi, ví dụ 09123456785
    · email?: str
    · một trong: phone / email
[music]
- list_tracks: Thư viện bài hát và playlist
- play: Phát một bài hoặc một playlist. Bắt buộc một trong: track_id hoặc playlist_id
    · track_id?: id←music.tracks — ID bài hát, lấy từ music.list_tracks
    · playlist_id?: id←music.playlists — ID playlist, lấy từ music.list_tracks
    · device?: str
    · một trong: track_id / playlist_id
- control: Tạm dừng/tiếp tục/chuyển bài/âm lượng
    · action: pause|resume|next|previous|volume
    · volume?: int — Số nguyên, không dấu chấm hay phẩy, ví dụ 2
- add_to_playlist: Thêm bài vào playlist
    · track_id: id←music.tracks — ID bài hát, lấy từ music.list_tracks
    · playlist_id: id←music.playlists — ID playlist, lấy từ music.list_tracks
[sys]
- ask_missing_info: Báo các tham số bắt buộc còn thiếu của một tool
    · tool_name: str — Tên tool bị thiếu tham số, dạng app_name.tool_name, ví dụ sms.send
    · params: list[str] — Tên các tham số bắt buộc còn thiếu, đúng tên trong schema, ví dụ ["contact_id"]<|im_end|>
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

### Khối Q1 — 626 token

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
I) food_delivery.list_restaurants – Danh sách quán giao tận nơi quanh đây (…
J) food_delivery.get_menu – Thực đơn quán
K) food_delivery.order_history – Đơn đã đặt gần đây
L) food_delivery.place_order – Đặt đơn
M) food_delivery.cancel_order – Huỷ đơn đang chờ
N) smart_home.get_status – Lấy thiết bị và trạng thái; lọc theo ph…
O) smart_home.light_set – Bật/tắt/chỉnh sáng đèn
P) smart_home.fan_set – Bật/tắt/chỉnh tốc độ quạt
Q) smart_home.climate_set – Đặt chế độ/nhiệt độ điều hoà
R) smart_home.switch_set – Bật/tắt ổ cắm, rèm, TV thông minh
S) smart_home.scene_run – Chạy ngữ cảnh đã đặt sẵn (đi ngủ, ra kh…
T) tv.guide – Danh sách kênh đang có và lịch phát són…
U) tv.tune – Chuyển kênh (tự bật TV nếu đang tắt); k…
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
j) contacts.search – Tìm liên hệ theo tên, quan hệ hoặc số đ…
k) contacts.get – Chi tiết liên hệ
l) contacts.update – Sửa số/email liên hệ
m) music.list_tracks – Thư viện bài hát và playlist
n) music.play – Phát một bài hoặc một playlist. Bắt buộ…
o) music.control – Tạm dừng/tiếp tục/chuyển bài/âm lượng
p) music.add_to_playlist – Thêm bài vào playlist
q) sys.ask_missing_info – Báo các tham số bắt buộc còn thiếu của…
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
