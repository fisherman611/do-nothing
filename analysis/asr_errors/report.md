# Phân tích lỗi ASR

Sinh bởi `analysis/asr_errors.py` từ prediction đã lưu trong `data/results/asr/`. Chỉ đọc `train.jsonl` (transcript + tên tool), không đọc `*_environment/`.

Quy đổi trước khi chấm (cả hai phía): bỏ khác biệt vị trí dấu thanh kiểu cũ/mới (hòa = hoà, hủy = huỷ);  số/giờ tiếng Việt → giá trị số ở nhãn; chữ số, `50k`, `9h35`, `420.000`, `50 nghìn`, SĐT cách nhóm → giá trị số ở prediction. Loại từ (tên riêng, thời gian, phủ định, từ mượn, động từ lệnh) là heuristic theo danh sách từ, không phải gán nhãn thủ công.

## whisper-large-v3

**1029 mẫu** (status ok, đã có nhãn). Chuẩn hóa số/giờ ở cả hai phía trước khi chấm.

| Chỉ số | Giá trị |
| --- | ---: |
| WER thô (chỉ lowercase + bỏ dấu câu) | 28.40% |
| WER sau quy đổi số/giờ | 24.57% |
| WER bỏ dấu thanh/dấu mũ (chỉ xét âm tiết) | 22.06% |
| Thay thế / xoá / chèn (từ) | 4810 / 509 / 333 |
| Trong thay thế: chỉ sai dấu thanh/dấu mũ | 518 (10.8%) |
| Câu đúng hoàn toàn | 52 (5.1%) |
| Câu WER ≥ 0,3 / ≥ 0,5 / ≥ 0,8 | 33.5% / 12.4% / 2.4% |
| Trung vị WER theo câu | 0.220 |

- Chênh WER thô và WER sau quy đổi số/giờ: **3.8 điểm %** là lỗi do *cách viết* (số chữ vs chữ số), không phải nghe sai.
- Chênh WER sau quy đổi và WER bỏ dấu: **2.5 điểm %** là lỗi chỉ ở dấu thanh/dấu mũ (đúng âm tiết nền).

### Lỗi theo loại từ tham chiếu

Tỷ lệ lỗi = (thay thế + xoá) / số từ loại đó trong nhãn. Chèn thừa không tính vào đây.

| Loại từ | Số từ nhãn | Sai | Tỷ lệ lỗi | Đóng góp vào tổng lỗi (S+D) |
| --- | ---: | ---: | ---: | ---: |
| từ nội dung khác | 13545 | 3040 | 22.4% | 57.2% |
| động từ lệnh | 2269 | 713 | 31.4% | 13.4% |
| trợ từ / từ đệm | 3106 | 661 | 21.3% | 12.4% |
| tên riêng | 1003 | 366 | 36.5% | 6.9% |
| từ chỉ thời gian | 1682 | 295 | 17.5% | 5.5% |
| số (số lượng, tiền, SĐT…) | 1022 | 178 | 17.4% | 3.3% |
| phủ định | 313 | 43 | 13.7% | 0.8% |
| từ mượn / tiếng Anh | 62 | 23 | 37.1% | 0.4% |

Chèn thừa: 333 từ, trong đó 8 là token số (sau quy đổi).

### Thực thể quan trọng cho tool call

- Số (sau quy đổi): giữ đúng **82.9%** (847/1022).
- Tên riêng (cụm viết hoa): khớp nguyên cụm **60.6%** (470/775); khớp ít nhất nửa số từ 66.2%.

### Nhóm lỗi cấp câu

| Nhóm | Số câu | Tỷ lệ |
| --- | ---: | ---: |
| sai gần hết (không rõ nguyên nhân) | 16 | 1.6% |
| ảo giác kiểu video/YouTube | 6 | 0.6% |
| cắt cụt / thiếu nội dung | 6 | 0.6% |
| thừa nội dung | 2 | 0.2% |
| lặp / vòng lặp sinh token | 1 | 0.1% |

### WER theo nhóm (WER corpus của nhóm; nhóm < 15 mẫu bị bỏ)

| Nhóm | Số mẫu | WER |
| --- | ---: | ---: |
| độ dài audio: <4s | 134 | 32.2% |
| độ dài audio: 4-6s | 264 | 28.4% |
| độ dài audio: 6-9s | 381 | 25.2% |
| độ dài audio: >=13s | 44 | 20.7% |
| độ dài audio: 9-13s | 206 | 20.5% |
| có số: không số | 412 | 27.5% |
| có số: có số | 617 | 23.1% |
| có tên riêng: có tên riêng | 515 | 24.7% |
| có tên riêng: không tên riêng | 514 | 24.4% |
| phương ngữ/khẩu ngữ (heuristic): có từ phương ngữ | 76 | 32.9% |
| phương ngữ/khẩu ngữ (heuristic): không | 953 | 23.9% |
| chuỗi nhiều tool: ≤1 tool call | 298 | 27.2% |
| chuỗi nhiều tool: ≥2 tool call | 731 | 23.9% |
| hỏi thiếu thông tin: có ask_missing_info | 135 | 27.5% |
| hỏi thiếu thông tin: không | 894 | 24.2% |

**Theo ứng dụng của tool call đầu tiên (10 nhóm tệ nhất, ≥15 mẫu):**

| App | Số mẫu | WER |
| --- | ---: | ---: |
| subscriptions | 21 | 33.7% |
| chat_ott | 49 | 28.9% |
| contacts | 72 | 27.7% |
| (không gọi tool) | 74 | 27.4% |
| smart_home | 91 | 26.9% |
| reminders | 34 | 26.6% |
| bike_rental | 19 | 26.5% |
| alarm | 26 | 26.4% |
| sys | 24 | 26.0% |
| home_energy | 16 | 25.8% |

**Tốt nhất (5 nhóm):**

| App | Số mẫu | WER |
| --- | ---: | ---: |
| bill_payment | 114 | 20.1% |
| home_security | 20 | 19.7% |
| weather | 55 | 19.3% |
| learning | 19 | 18.9% |
| maps | 15 | 17.3% |

### Cặp nhầm hay gặp (nhãn → model), top 25

| Nhãn | Model | Lần |
| --- | --- | ---: |
| minh | mình | 30 |
| nhắn | nhấn | 24 |
| zalo | lô | 22 |
| giùm | dùm | 20 |
| báo | bảo | 19 |
| nha | nhé | 18 |
| dời | giờ | 17 |
| đèn | điện | 16 |
| nhé | nha | 15 |
| họp | học | 13 |
| hạn | hẹn | 13 |
| tôi | thôi | 13 |
| anh | ăn | 12 |
| chị | chỉ | 11 |
| nhắc | nhất | 11 |
| giùm | dùng | 11 |
| #9 | chính | 10 |
| hông | không | 10 |
| ừ | ờ | 10 |
| sjc | sgc | 9 |
| rồi | và | 9 |
| trả | trẻ | 9 |
| ấy | đấy | 9 |
| giờ | cho | 8 |
| chị | trị | 8 |

### Cặp chỉ sai dấu thanh/dấu mũ, top 15

| Nhãn | Model | Lần |
| --- | --- | ---: |
| minh | mình | 30 |
| nhắn | nhấn | 24 |
| báo | bảo | 19 |
| chị | chỉ | 11 |
| nha | nhà | 8 |
| phòng | phong | 7 |
| đặt | đạt | 7 |
| chuyển | chuyện | 7 |
| tối | tôi | 6 |
| tối | tới | 6 |
| dũng | dụng | 6 |
| cô | có | 6 |
| báo | bao | 6 |
| hộ | hồ | 5 |
| chị | chi | 5 |

### 20 câu tệ nhất

- `train_0554` WER 1.17 [sai gần hết (không rõ nguyên nhân)] — tool: `bike_rental.reserve`
  - nhãn: Chiếc xanh số 24 đi em.
  - model: Chào tạm biệt, tôi là Tư Điên
- `train_0015` WER 1.00 [ảo giác kiểu video/YouTube, cắt cụt / thiếu nội dung] — tool: `bill_payment.list_bills`
  - nhãn: Chuyển tiền mạng tháng này cho chú Phạm Hùng nha, nếu tài khoản thanh toán không đủ thì lấy từ tiết kiệm dùm tôi hen.
  - model: Chào mừng quý vị đến với chương trình Pháp Hồng.
- `train_0017` WER 1.00 [cắt cụt / thiếu nội dung] — tool: `tv.guide`
  - nhãn: Ơi, chuyển sang kênh S+ Sport dùm em nha.
  - model: Học hỏi
- `train_0147` WER 1.00 [sai gần hết (không rõ nguyên nhân)] — tool: `chat_ott.list_chats`
  - nhãn: Dạ Zalo đi ạ.
  - model: Yeah yeah lô lia
- `train_0430` WER 1.00 [ảo giác kiểu video/YouTube] — tool: `sys.ask_missing_info`
  - nhãn: Tạo giúp anh một phòng họp online cho buổi rà soát ngân sách ngày mai nhé.
  - model: Cảm ơn các bạn đã theo dõi và hẹn gặp lại.
- `train_0443` WER 1.00 [sai gần hết (không rõ nguyên nhân)] — tool: `smart_home.light_set`
  - nhãn: Vậy bật cái đèn đó lên.
  - model: Bạn kia đi ăn rồi đấy
- `train_0717` WER 1.00 [sai gần hết (không rõ nguyên nhân)] — tool: `calendar.create_event`
  - nhãn: Lấy ca tám giờ sáng hôm hai tư nhé.
  - model: Các bạn có muốn làm gì?
- `train_0797` WER 1.00 [sai gần hết (không rõ nguyên nhân)] — tool: `—`
  - nhãn: Nói chuyện với em vui thật đấy, cảm ơn em nhiều nha.
  - model: Hãy trở nên nhiều thằng đấy, dù không bao giờ em nhìn ai.
- `train_0865` WER 1.00 [ảo giác kiểu video/YouTube] — tool: `home_energy.get_usage`
  - nhãn: Này, xem giúp tôi tuần này nhà mình dùng hết bao nhiêu số điện rồi.
  - model: Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn
- `train_0878` WER 1.00 [ảo giác kiểu video/YouTube, cắt cụt / thiếu nội dung] — tool: `contacts.search`
  - nhãn: Tìm giúp anh số chị Lan trong danh bạ, rồi ghi chú lại ba cái cần hỏi chị ấy là doanh thu miền Trung, chi phí vận chuyển với tỷ lệ hoàn hàng nhé.
  - model: Hãy subscribe cho kênh Ghiền Mì Gõ Để không bỏ lỡ những video hấp dẫn
- `train_1052` WER 1.00 [cắt cụt / thiếu nội dung] — tool: `ride_hailing.get_ride`
  - nhãn: Xe chở Út với hai đơn đồ ăn, cái nào tới nơi rồi thì gạch việc đó giùm tôi nghen.
  - model: Xin chào các bạn.
- `train_0510` WER 0.96 [cắt cụt / thiếu nội dung] — tool: `smart_home.get_status`
  - nhãn: Cài cho anh hẹn giờ tự bật quạt phòng ngủ lúc chín giờ tối hằng ngày, xong cho con robot dọn phòng ngủ luôn bây giờ nhé.
  - model: Cảm ơn anh.
- `train_0683` WER 0.95 [ảo giác kiểu video/YouTube] — tool: `sys.ask_missing_info`
  - nhãn: Tạo sẵn giúp tôi một phòng họp trực tuyến cho buổi hướng dẫn bà con trong tổ dân phố sắp tới nhé.
  - model: Hãy đăng ký kênh để ủng hộ kênh của mình nhé!
- `train_0258` WER 0.94 [sai gần hết (không rõ nguyên nhân)] — tool: `smart_home.get_status`
  - nhãn: Tắt hết đèn tầng hai, trừ phòng nào đang bật điều hoà thì để nguyên nhé.
  - model: Tất cả các bạn có thể tìm ra những điều này
- `train_0421` WER 0.90 [cắt cụt / thiếu nội dung, thừa nội dung, lặp / vòng lặp sinh token] — tool: `gold_price.get_price`
  - nhãn: Nếu vàng SJC bán ra quá 121 triệu thì nhắc chị mai chín giờ sáng bán vàng, không thì lấy tài khoản thanh toán chuyển Phạm Hùng 2 triệu tiền góp mua vàng.
  - model: Nếu vang adress bán ra qua 121.1.9.9.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0
- `train_1061` WER 0.88 [sai gần hết (không rõ nguyên nhân)] — tool: `medication.list`
  - nhãn: Ủa siro ho bác sĩ kê đó, một ngày tui phải uống mấy lần, mấy giờ vậy?
  - model: Hồ Sư Thu Hạ bác sĩ kia đã muốn làm phẫu huynh với lần mới ra đây.
- `train_0923` WER 0.86 [sai gần hết (không rõ nguyên nhân)] — tool: `fuel_price.get_price`
  - nhãn: Ờ... RON 95, à không, xe anh giờ đổ E5 rồi, coi E5 dùm nha, chiều anh chạy xuống Long An.
  - model: Chào các bạn, hôm nay tôi sẽ đổ E5 cho anh.
- `train_0983` WER 0.86 [sai gần hết (không rõ nguyên nhân)] — tool: `library.search`
  - nhãn: Giữ cuốn nớ, à thôi khỏi. Ghi việc cần làm: qua thư viện mượn hai cuốn, nếu Nhà giả kim còn.
  - model: Chữ con nữ, hãy đòi khỏi, nghe về con lòng có thư viện mà ai cũng nói nhẹ dậy kìa.
- `train_0230` WER 0.84 [sai gần hết (không rõ nguyên nhân)] — tool: `bill_payment.list_bills`
  - nhãn: Cho mình chuyển qua Nguyễn Thị Lan y số tiền nước tháng này nha, trừ tài khoản thanh toán, mà thiếu thì dùng tiết kiệm.
  - model: Trong mặt trận của ông Nguyễn Thị Lâm, khi số tiền được thăng lên, rồi họ được quản lý theo điểm thích kiệm.
- `train_0596` WER 0.83 [sai gần hết (không rõ nguyên nhân)] — tool: `door_lock.list_locks`
  - nhãn: Ừ, khóa lại giúp chị đi.
  - model: Ở Hoa Lạch Chút Chí Đi
