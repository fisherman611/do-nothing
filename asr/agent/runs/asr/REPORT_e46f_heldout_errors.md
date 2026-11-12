# Phân tích lỗi ASR `e46f` trên 405 câu held-out (12/11/2026)

Nguồn: `agent/runs/asr/nbest_e46f/nbest_B.json` (phần `set=heldout`) — e46f nghe beam 5 trên 405 đoạn giữ riêng (123 câu cuối + 282 câu lịch sử), lấy bản 1 làm 1-best. Script: `agent/asr_errors_e46f.py` (chạy từ thư mục `asr/`).

**Model:** `e46f` = PhoWhisper-large fine-tune LoRA r32/α64 trên audio thật v1 của BTC, prompt v1 từ ngữ cảnh lượt trước, label smoothing 0,1, CTC 0,3. Là ASR của bản nộp 62.76. Mọi bản thử sau (L2 data v2.1; B4 từ gốc và B4b từ e46f với data TTS v3; Qwen3-ASR-0.6B 6 epoch) đều không vượt (PLAN_72h_v3 §0, NOTES §1q–§1s, §1z7).

**Chuẩn hoá:** thường, bỏ dấu câu, giữ chữ số (module `vi_num` không có trong repo local nên không đổi số → chữ). Vì vậy WER ở đây cao hơn số trong `gate_v3.json` (6,75% so với 5,77%): lệch do lỗi dạng viết `9 ↔ chín`; cấu trúc lỗi không đổi.

## 1. Số liệu

| Nhóm | n | WER |
|---|---|---|
| Tất cả | 405 | 6,75% |
| Câu cuối (final) | 123 | **7,80%** |
| Lịch sử | 282 | 5,80% |
| Vùng Bắc | 154 | 6,77% |
| Vùng Nam | 132 | 5,66% |
| Vùng Trung | 119 | **7,97%** |

- 218/405 câu đúng hết. 187 câu sai: 99 câu sai 1 từ, 49 sai 2, 15 sai 3, 24 sai ≥ 4.
- **Câu cuối** (chỗ FC đọc): 44 đúng, 8 chỉ khác dạng viết số / từ đệm, **71 sai thật (58%)**. Đây là nguồn của M2 76,5 → 63,5 (NOTES §1, PLAN §1).
- **Beam 2–5 bù được:** 66/187 câu sai có bản khác đúng hoàn toàn, 108/187 có bản tốt hơn 1-best. Riêng câu cuối: 15/71 đúng hẳn, 37/71 tốt hơn.

## 2. Phân loại lỗi (257 cụm, toàn bộ 405 câu)

| Loại | Cụm | Tỉ lệ | Ví dụ |
|---|---|---|---|
| Từ thường nghe nhầm | 179 | **63,5%** | `tạm dừng → tập trường`, `vặn → đặt`, `pháp chế → khách chị`, `lượng → phần`, `dọn chỗ đó luôn → chấm đồ giòn chờ đổi muộn` |
| Số (kể cả dạng viết) | 26 | 9,2% | `chín ↔ 9`, `năm ↔ 5`, `hông → không`, `quạt → một`; **sai giá trị số thật ≈ 0** (plan đếm 9/405 bằng `_nums`, phần lớn là dạng viết, NOTES §1z) |
| Tên Latin / từ nước ngoài | 25 | 8,9% | `Brightpath → Bia Pratt`, `Ryan Cooper → ra hàng khách với`, `S+ Sport → đó trọng sport`, `guide → gai`, `edura → endora` |
| Mất từ | 23 | 8,2% | `rứa con`, `nghe con`, `chi`, `bao`, `gói` — chủ yếu từ phương ngữ Trung |
| Thừa từ | 14 | 5,0% | `train_1009` thêm cả câu `Đèn em kêu em thu. Tiếp tục.` ở đầu (nghi ảo giác từ prompt) |
| Từ đệm đổi | 11 | 3,9% | `nhé ↔ nha`, `ừ ↔ ờ`, `ấy ↔ đấy` — vô hại với FC |
| Mất/thừa từ đệm | 4 | 1,5% | `nha`, `ạ`, `ờ`, `đi` |

Riêng 71 câu cuối sai thật: từ thường 87 cụm, tên Latin 16, mất từ 11, thừa từ 10, số 6, từ đệm 7.

## 3. Nhận xét

1. **Lỗi đắt nhất là thực thể trong câu cuối, không phải số.** `Brightpath`, `Ryan Cooper`, `Mọt Sách → họp khách`, `Nhã Văn → Chảo Phan`, `buổi rà soát ngân sách → quả gà sáng danh sách`. Những lỗi này làm FC tìm rỗng → chuỗi chết; N1/N1b không cứu được vì truy vấn sai từ gốc (khớp §1s). Số sai giá trị thật gần 0, nên ITN/chuẩn hoá số không có gì để sửa (§1z).
2. **Phương ngữ Trung tệ nhất** (7,97% so với Bắc 6,77 / Nam 5,66), chủ yếu mất từ đệm `rứa/chi/nghe con`, `ni → đi`, `hè → Hai Hà`. Phần lớn là từ đệm, ít đổi ý nghĩa cho FC.
3. **"Từ thường" 63%** là nghe nhầm cục bộ ở câu nói tự nhiên, không có cấu trúc để sửa bằng luật; chỉ sửa được bằng thêm audio thật, mà không có. B4/B4b thêm TTS đã làm xấu đi (§1z7–§1z8).
4. **Beam 5 có giá trị:** 37/71 câu cuối sai có bản tốt hơn trong beam. Căn cứ cho A4 (5) "nhiều cách nghe trong prompt FC" (Sun et al. 2311.07418: chỉ lợi khi train FC trên định dạng đó). Chi phí: đổi định dạng, train lại FC, đổi Docker → plan xếp "chỉ khi còn ≥ 1,5 ngày".

## 4. Hệ quả cho plan

- **Không train ASR tiếp.** Không còn nhóm lỗi nào đủ lớn để nhắm data riêng; hai lần thử đều thua e46f.
- **Điểm còn ăn được nằm ở FC chịu lỗi** trên 71 câu cuối kiểu này → A2 (train trên `fmtA_v3`) là ưu tiên 1.
- Nhiễu B0 (PhoWhisper-medium gốc, WER 20%) nặng hơn lỗi e46f thật (WER 6–8%, đa số 1–2 từ/câu). Bộ `fmtA_a4` (nhiễu từ chính e46f trên TTS) khớp lúc thi hơn, nhưng phải dựng lại theo A6 (loại giọng Trung ElevenLabs) trước khi train vòng 2.

## Phụ lục — 71 câu cuối sai thật

Cột **Beam**: ✓ có bản 2–5 đúng hoàn toàn · ↑ có bản ít lỗi hơn · – không bản nào tốt hơn. Cột **Lỗi** = số từ sai sau chuẩn hoá.

no _nums [Errno 2] No such file or directory: 'd:\\GitHub\\do-nothing\\asr\\agent\\data\\places_vn.json'
| # | id | Vùng | Lỗi | Beam | Nhãn | e46f | Cụm sai |
|---|---|---|---|---|---|---|---|
| 1 | train_1151 | Bắc | 13 | ↑ | Tuần sau em ôn thi cuối kỳ nên cần chỗ ngồi học. Nếu đi xe buýt tới quán Mọt Sách cũng dưới nửa tiếng thì lưu luôn cả hai quán, không thì chỉ lưu Nhã Văn thôi ạ. | Tuần đâu em. Em thư thuýt từ chỗ ngồi học, nếu thì xe buýt tới quán họp khách, cũng nếu hủy tiếng thì ghi luôn cả hai quán, không thì chỉ lưu nhã Văn thôi ạ. | `sau` → `đâu`; `ôn thi cuối kỳ nên cần` → `em thư thuýt từ`; `đi` → `thì`; `mọt sách` → `họp khách`; `dưới nửa` → `nếu hủy`; `lưu` → `ghi` |
| 2 | train_1009 | Nam | 12 | – | Chiều nay hình như chị phải đón ai á, quên mất tiêu. Nhắc có vụ đón thì tắt đèn trên lầu còn mở, gọi số thường cho người đó nha. | Đèn em kêu em thu. Tiếp tục. Chiều nay hình như chị phải đi đón ai ấy, với mục tiêu. Nãy có vụ đón thì tắt đèn trên lầu còn mở, gọi số thường cho người đó nha. | `∅` → `đèn em kêu em thu tiếp tục`; `∅` → `đi`; `á quên mất` → `ấy với mục`; `nhắc` → `nãy` |
| 3 | train_1483 | Nam | 9 | – | Còn hai lần bấm chuông đó, có một lần là Ryan Cooper trên lầu quên chìa khóa, con đánh dấu lần của ổng thôi nha. | Còn hai lần bấm chuông đó, có một lần là ra hàng khách với trên lầu quanh trực khoá, con đánh dấu lần khoảng thôi nha. | `ryan cooper` → `ra hàng khách với`; `quên chìa khóa` → `quanh trực khoá`; `của ổng` → `khoảng` |
| 4 | train_1078 | Trung | 8 | ↑ | Anh đi tour đây. Cái camera hôm kia báo mất kết nối á, giờ vẫn chưa ghi hình thì bật chế độ đi vắng giùm nghe. | Anh đi tour đấy, gửi camera hôm kia bảo mới kẹt nổi á. Giờ còn chưa ghi hình thì bật chế độ đi vãn giùm nghe. | `đây cái` → `đấy gửi`; `báo mất kết nối` → `bảo mới kẹt nổi`; `vẫn` → `còn`; `vắng` → `vãn` |
| 5 | train_0933 | Nam | 7 | ↑ | Bảy giờ, ờ mà khoan, bảy rưỡi tui mới chạy xe từ công ty về. Ghi dùm cái trận S+ đang đá lúc tui tới nhà nha. | Bảy giờ... à mà khoan, bảy rưỡi tôi mới chạy xe từ công ty về, ghi giùm cái chặn S cộng đăng đá lúc tôi tới nhà nha. | `ờ` → `à`; `tui` → `tôi`; `dùm` → `giùm`; `trận` → `chặn`; `đang` → `cộng đăng`; `tui` → `tôi` |
| 6 | train_1430 | Bắc | 7 | ↑ | Thêm nhắc anh uống kẽm Brightpath mỗi ngày nhé, uống cùng giờ với thuốc Brightpath em vừa đọc. | Thêm nhắc anh uống kẹo Bia Pratt mỗi ngày nha, uống cùng giờ với thuốc Bia Pratt anh vừa đọc. | `kẽm brightpath` → `kẹo bia pratt`; `nhé` → `nha`; `brightpath em` → `bia pratt anh` |
| 7 | train_0430 | Bắc | 6 | – | Tạo giúp anh một phòng họp online cho buổi rà soát ngân sách ngày mai nhé. | Tạo giúp anh một phòng họp online cho quả gà sáng danh sách như vậy nhé. | `buổi rà soát ngân` → `quả gà sáng danh`; `ngày mai` → `như vậy` |
| 8 | train_0483 | Bắc | 6 | ↑ | Vậy cho nó dọn chỗ đó luôn bây giờ đi em. | Vậy chấm đồ giòn chờ đổi muộn bây giờ đi em. | `cho nó dọn chỗ đó luôn` → `chấm đồ giòn chờ đổi muộn` |
| 9 | train_0926 | Nam | 6 | – | Ờ... chiều nay thằng Huy nó qua nhà cho con mèo ăn giùm em, em tính tạo cho nó cái mã mở cổng tới năm giờ á... à mà thôi thôi khỏi, chị Hai mới nhắn là chỉ ở nhà cả chiều rồi. Ok bỏ đi nha, hông cần nữa. | Ờ chiều nay thằng Huy nói qua nhà cho con mèo ăn giùm em, em tính tạo cho nó cái mã mở cổng tới năm giờ ấy. À mà thôi thôi, chị Hai mới nhắn là chị ở nhà cả chiều rồi. Ok, bỏ đi nha, không còn nữa. | `nó` → `nói`; `á` → `ấy`; `khỏi` → `∅`; `chỉ` → `chị`; `hông cần` → `không còn` |
| 10 | train_1505 | Trung | 5 | ↑ | Con ơi, cái xe cô đặt sáng mai ra ga... à không, ra sân bay nớ, mấy giờ hắn tới đón cô rứa con? | Con ơi, cái xe cô đặt sáng mai ra ga... à không, ra sân bay nữa. Mấy giờ hạn tối đón bố rồi con? | `nớ` → `nữa`; `hắn tới` → `hạn tối`; `cô rứa` → `bố rồi` |
| 11 | train_1695 | Trung | 5 | ↑ | Khách check-out sớm, tui lu bu quá. Em coi lịch thuốc rồi nhắc tui đúng giờ uống vitamin D sáng ni nghe. | Khách check-out sáng tư lưu bù quá, em coi lịch thuốc rồi nhắc tui đúng giờ uống vitamin D sáng ni nha. | `sớm tui lu bu` → `sáng tư lưu bù`; `nghe` → `nha` |
| 12 | train_1057 | Trung | 5 | – | Ờ, theo dõi kênh Khoa học mỗi ngày giùm em, à không, kênh nớ theo rồi, kênh Thành phố xanh. Tập mới nhất ghi vô ghi chú Nghe sau, với lưu quán Trạm Sách trên bản đồ. | Ờ thế giải kênh Khoa học mỗi ngày giùm em, à không, kênh nớ thuê em rồi, kênh Thành Phố Xanh, tập mới nhất ghi vô ghi chú nghe sau, với lưu quán trạm sách trên bản lồ. | `theo dõi` → `thế giải`; `theo` → `thuê em`; `đồ` → `lồ` |
| 13 | train_1675 | Trung | 4 | – | Tối ni cô học online lớp nhiếp ảnh bên Edura từ bảy rưỡi tới tám giờ mười, ghi vô tiến độ giùm cô nghe. | Tối nay cô học online lớp nhiếp ảnh bên Endora, từ bảy rưỡi tới tám giờ mười ghi vô tiếng Nội giùm cô nghe. | `ni` → `nay`; `edura` → `endora`; `tiến độ` → `tiếng nội` |
| 14 | train_0942 | Trung | 4 | ↑ | Rứa con mở cho cô cái tập Khoa học mỗi ngày dài cỡ chừng nớ, cô nhớ có một tập dài rứa á. | Rứa con mở cho cô cái tập khoá học mỗi ngày dài cửa chân nớ, cô nhớ có một tập dài rứa? | `khoa` → `khoá`; `cỡ chừng` → `cửa chân`; `á` → `∅` |
| 15 | train_0675 | Bắc | 4 | ↑ | Thế thì hủy gói đó đi em. Xong soạn giúp anh cái email nháp gửi team chấm data a còng example chấm test, báo là quyền truy cập sẽ dừng vào cuối kỳ này nhé. | Thế thì huỷ gói đó đi em. Xong soạn giúp anh cái email nháp gửi tin chấm data anh không, example chấm test, báo là quyền truy cập sẽ dừng vào cuối kỳ này nhé. | `hủy` → `huỷ`; `team` → `tin`; `a còng` → `anh không` |
| 16 | train_1019 | Trung | 4 | ↑ | Tuần ni anh đi tour suốt, coi giùm bảy ngày qua nhà xài hết mấy số điện, tốn bao nhiêu tiền rứa? | Tuần này anh đi tour suốt, coi giùm bảy ngày qua nhà anh xài hết bao nhiêu số điện, tốn bao nhiêu tiền rứa? | `ni` → `này`; `∅` → `anh`; `mấy` → `bao nhiêu` |
| 17 | train_1254 | Nam | 3 | – | Dọn nhà vệ sinh thì để Tuấn làm, còn rửa chén đổi qua bà nội dùm anh nha. | Dọn nhà vệ sinh thì để Tuấn làm, còn rửa chén online qua bà nội giùm anh. | `đổi` → `online`; `dùm` → `giùm`; `nha` → `∅` |
| 18 | train_1451 | Nam | 3 | – | Còn cái báo cửa phòng Lemon Tree con mới đọc đó, cũng đánh dấu xem luôn nha con. | Còn cái báo cửa phòng Lê Minh chi con mới đọc đó, cũng đánh dấu xem luôn nha con. | `lemon tree` → `lê minh chi` |
| 19 | train_0826 | Bắc | 3 | ↑ | Này, cái hội nghị trong Sài Gòn tôi ghi trong lịch ấy, xem giờ rồi đặt cho tôi một vé bay từ Hà Nội vào hôm đó, chuyến nào hạ cánh trước giờ hội nghị ít nhất bốn tiếng. Xong gọi sẵn xe bốn chỗ ra sân bay Tân Sơn Nhất đón tôi lúc máy bay hạ cánh, chở về khách sạn Riverside nhé. | Này, cái hội nghị trong Sài Gòn tôi ghi trong lịch đấy, xem từ giờ rồi đặt cho tôi một vé bay từ Hà Nội vào hôm đó, chuyến nào hạ cánh trước giờ hội nghị ít nhất bốn tiếng. Xong gọi sẵn xe bốn chỗ ra sân bay Tân Sơn Nhất đón tôi lúc máy bay hạ cánh, chở về khách sạn Riverside nh | `ấy` → `đấy`; `∅` → `từ`; `nhé` → `nh` |
| 20 | train_1678 | Trung | 3 | ↑ | Đợt thuốc chị uống xong rồi, xoá giùm chị lịch mấy thuốc nào có cữ uống buổi tối nghe em. | Đọc thuốc chị uống xong rồi, xoá giùm chị lịch mấy thuốc nào có cữ buổi tối nghe. | `đợt` → `đọc`; `uống` → `∅`; `em` → `∅` |
| 21 | train_1366 | Nam | 3 | ↑ | Đánh dấu đã coi cái chuông của Peter Hale với cửa sau nghen, rồi đặt báo thức trước giờ chở ổng bốn lăm phút. | Đánh dấu đã coi cái chuông của Peter Ham với cửa sau nghen, rồi đặt báo thức trước giờ chạy động bốn lăm phút. | `hale` → `ham`; `chở ổng` → `chạy động` |
| 22 | train_1636 | Bắc | 3 | – | Ờ sáng nay hình như có thư tặng voucher ấy, em xem giúp tớ voucher đó hạn đến hôm nào nha. | Sáng nay hình như có thư tặng voucher ấy, em xem giúp tớ voucher ra hạn đến mức hôm nào nha. | `ờ` → `∅`; `đó` → `ra`; `∅` → `mức` |
| 23 | train_1171 | Trung | 3 | ↑ | Đắt quá mà đi có mấy bữa, gói mô gia hạn vô thứ Sáu tới thì hủy giùm em. | Đắc quán mà đi có mấy bữa gói mô gia hạn vô thứ sáu tới thì huỷ giùm em. | `đắt quá` → `đắc quán`; `hủy` → `huỷ` |
| 24 | train_1428 | Nam | 3 | ↑ | Lấy số tiền Greenhaven Foods mới chuyển, gửi tiết kiệm sáu tháng từ tài khoản vựa giùm tôi. | Lấy số tiền Green Harbor Foods mới chuyển, gửi tiết kiệm sáu tháng từ tài khoản vừa giùm tôi. | `greenhaven` → `green harbor`; `vựa` → `vừa` |
| 25 | train_0809 | Bắc | 3 | – | Tiêu đề là tóm tắt hợp đồng ngày bốn tháng mười, còn nội dung thì ghi là gửi anh bản tóm tắt hợp đồng đã thống nhất hôm mùng bốn. Để nháp thôi nhé, khoan gửi. | Tiêu đề là tổng tắt hợp đồng ngày bốn tháng mười, còn nội dung thì ghi là gửi anh bản tổng tắt hợp đồng đã thống nhất hôm mùng bốn, để nhắc thôi nhé, khoan gửi. | `tóm` → `tổng`; `tóm` → `tổng`; `nháp` → `nhắc` |
| 26 | train_0892 | Bắc | 2 | – | Thế thì trước hôm đó một ngày, sáu giờ tối, nhắc tôi đi mua hoa nhé. | Thế thì hôm trước đó một ngày sáu giờ tối nhắc tôi đi mua hoa nhé. | `trước hôm` → `hôm trước` |
| 27 | train_1622 | Trung | 2 | ↑ | Tiền vé chuyển anh Kha rồi, còn bắp nước thì em lấy điểm rạp phim đổi một phần luôn nghe. | Điện vé chuyển anh Kha rồi, còn bắp nước thì lấy điểm rạp phim đổi một phần luôn nghe. | `tiền` → `điện`; `em` → `∅` |
| 28 | train_1276 | Nam | 2 | – | Mưa vậy thôi hủy chiếc xe thứ sáu đi con, gọi lại cái số lạ hồi trưa giùm cô. | Mưa vậy thôi, hủy chiếc xe thứ sáu đi con, rồi lấy cái số lạ hồi trưa giùm cô. | `gọi lại` → `rồi lấy` |
| 29 | train_0242 | Nam | 2 | ↑ | Trả bằng ngân hàng giúp tôi mấy hoá đơn đến hạn trong 5 ngày tới, riêng hoá đơn nào trên 500 nghìn thì giữ lại, chưa trả nhé. | Trả bằng ngân hàng giúp tôi mấy hoá đơn tính hẹn trong 5 ngày tới, riêng hoá đơn nào trên 500 nghìn thì giữ lại, chưa trả nhé. | `đến hạn` → `tính hẹn` |
| 30 | train_0839 | Bắc | 2 | ↑ | Ừ, nếu đúng là cái cửa đó vẫn đang mở thì khoá lại giúp chị, rồi giao cho chị Lan việc kiểm tra pin khoá nhé. | Ừ, nếu đúng là cái cửa đó vẫn đang mở thì khóa lại giúp chị, rồi giao cho chị Lan việc kiểm tra pin nhé. | `khoá` → `khóa`; `khoá` → `∅` |
| 31 | train_0860 | Bắc | 2 | – | Em xem ngoài trời giờ có dưới hai mươi độ không, lạnh thế thì mười giờ nhắc anh mặc áo khoác trước khi ra ngoài nhé. | Em xem ngoài trời giờ có dưới hai mươi độ không, lạnh thế thì mười giờ nhắc anh mang áo khoác trước khi ra ngoài nha. | `mặc` → `mang`; `nhé` → `nha` |
| 32 | train_0406 | Bắc | 2 | – | Mai nếu mưa thì huỷ lịch khám răng rồi nhắn SMS báo sếp Dũng, còn không mưa thì đặt ô tô bốn chỗ từ nhà đến đó, đón sớm hơn giờ hẹn nửa tiếng. | Mai nếu mưa thì huỷ lịch khám răng rồi nhắn SMS báo sếp Dũng, còn không mưa thì đặt ô tô 4 chỗ từ nhà tới đó, đón sớm hơn giờ hẹn nửa tiếng. | `bốn` → `4`; `đến` → `tới` |
| 33 | train_1059 | Trung | 2 | ↑ | Chuyển qua lau... à thôi khỏi, dừng hẳn cho nó về sạc đi em. | Chuyển qua lao. À thôi, khỏi dừng, hoãn cho nó về sạc đi em. | `lau` → `lao`; `hẳn` → `hoãn` |
| 34 | train_1413 | Trung | 2 | ✓ | Rứa giữ giùm tui cuốn Frankfurt em vừa đọc hồi nãy nghe. | Ghi giữ giùm tui cuốn Frankfurt em vừa đặt hồi nãy nghe. | `rứa` → `ghi`; `đọc` → `đặt` |
| 35 | train_1022 | Bắc | 2 | – | Mỗi tờ trúng thì chuyển nó hai trăm nghìn, trượt cả hai thì thôi nhé. | Mỗi tờ trúng thì chuyển cho nó hai trăm nghìn, trừ cả hai thì thôi nhé. | `∅` → `cho`; `trượt` → `trừ` |
| 36 | train_1570 | Nam | 2 | – | Mấy gói Taipei trừ tiền ngày mười tháng mười hai á, em hủy bớt một gói thôi. | Mấy gói chạy bầy trừ tiền ngày mười tháng mười hai á, em hủy bớt một gói thôi. | `taipei` → `chạy bầy` |
| 37 | train_0690 | Bắc | 2 | – | Ờ, buổi tập huấn hôm mùng sáu tháng mười ấy, nếu nó đang bị hủy thì em khôi phục lại giúp anh nhé. | Ờ buổi tập huấn mùng sáu tháng mười ấy, nếu nó đang bị hủy thì anh khôi phục lại giúp anh nhé. | `hôm` → `∅`; `em` → `anh` |
| 38 | train_1363 | Trung | 2 | – | Thẻ nào đủ điểm thì đổi giùm chị phần quà rẻ nhất, mỗi thẻ một phần, trừ thẻ JetXanh ra nghe. | Thẻ nào đủ điểm thì đổi giùm chị phần quà rẻ nhất, mỗi thẻ một phần, trừ thẻ giá xăng ra nghe. | `jetxanh` → `giá xăng` |
| 39 | train_0270 | Nam | 2 | ✓ | Tắt hết đèn tầng hai đi, trừ phòng nào đang bật quạt thì cứ để đèn, rồi bật điều hoà phòng ngủ 26 độ nhé. | Tắt hết đèn tầng 2 đi, trừ phòng mà đang bật quạt thì cứ để đèn, rồi bật điều hoà phòng ngủ 26 độ nhé. | `hai` → `2`; `nào` → `mà` |
| 40 | train_1118 | Nam | 2 | ↑ | Dò tờ bảy hai không chín một một. Thêm chục hột gà vô việc cần làm, lộn, vô đi chợ. | Giờ tờ bảy hai không chín một một, thêm chục hộp gà vô việc cần làm, lộn, vô đi chợ. | `dò` → `giờ`; `hột` → `hộp` |
| 41 | train_1519 | Nam | 2 | – | Dọn nhà vệ sinh giao cho Bé Na, còn nhận thùng hàng bên Marigold Foods thì giao chú Năm nha con. | Dọn nhà vệ sinh giao cho Bé Na, còn những thùng hàng bên Marygold Foods thì giao chú Năm nha con. | `nhận` → `những`; `marigold` → `marygold` |
| 42 | train_1521 | Nam | 2 | ↑ | Lưu giùm tôi bài nhà nhập khẩu Daniel Lee, rồi đặt báo thức trước buổi tọa đàm của ổng ba mươi phút nha. | Lưu giùm tôi bài Nhà nhập khẩu Daniel Lee, rồi đặt báo thức trước buổi tọa đàm của ông Ba mười phút nha. | `ổng` → `ông`; `mươi` → `mười` |
| 43 | train_1414 | Nam | 2 | ✓ | Anh lộn rồi, vô cuộc họp với Daniel Lee giùm anh. Thuốc dạ dày thì bỏ cữ trưa, chỉ giữ cữ sáng thôi. | Anh lộn rồi. Vô cuộc họp với Daniel Lee giùm anh, thuốc dạ dày thì bỏ cửa trưa, chỉ giữ cửa sáng thôi. | `cữ` → `cửa`; `cữ` → `cửa` |
| 44 | train_0701 | Bắc | 2 | ↑ | Cho em tập Phản xạ khi sơ cứu vào danh sách nghe sau nha, với tạo giúp em việc thực hành động tác sơ cứu trong tập vừa nghe, hạn tám giờ tối chủ nhật mười một tháng mười. | Cho em tập phản xạ ghi sơ cứu vào danh sách nghe sau nha, với tạo giúp em việc thực hành động tác sơ cứu trong tập vừa nghe, hẹn tám giờ tối chủ nhật mười một tháng mười. | `khi` → `ghi`; `hạn` → `hẹn` |
| 45 | train_1553 | Trung | 2 | – | Giữ giùm chiếc xe đôi với chiếc rẻ nhứt lúc nãy, rồi nhắn Isabel Moreno là xe để ở trạm Đại Nội, ra lấy liền nghe. | Giữ giùm chiếc xe đôi với chiếc rẻ nhất lúc nãy, rồi nhắn Isabel là Moreno là xe để ở trạm Đại Nội, ra lấy liền nghe. | `nhứt` → `nhất`; `∅` → `là` |
| 46 | train_0887 | Bắc | 1 | ✓ | Thôi lấy chuyến sớm hơn đi, một ghế thôi. | Thôi lấy chuyến sớm hơn đi, một ghé thôi. | `ghế` → `ghé` |
| 47 | train_1487 | Bắc | 1 | ✓ | Thế thì tôi nhầm tên rồi, gọi cô Olivia Brown giúp tôi. Với huỷ luôn cái gói năm trăm nghìn ấy đi. | Thế thì tôi nhầm tên rồi. Gọi cô Olivia Brown giúp tôi, với hủy luôn cái gói năm trăm nghìn ấy đi. | `huỷ` → `hủy` |
| 48 | train_1017 | Bắc | 1 | ✓ | Hôm qua chị chưa đủ mục tiêu bước thì bỏ bánh su kem đi, đủ thì thêm hộp sữa chua. | Hôm qua chị chưa đủ mục tiêu bước thì bỏ bánh xô kem đi, đủ thì thêm hộp sữa chua. | `su` → `xô` |
| 49 | train_1163 | Nam | 1 | ✓ | Ghi dùm tôi năm giờ sáng nay đi bộ bốn chục phút, rồi coi hôm qua tôi đi chưa tới sáu ngàn bước thì hạ mục tiêu còn sáu ngàn nghen. | Ghi giùm tôi năm giờ sáng nay đi bộ bốn chục phút, rồi coi hôm qua tôi đi chưa tới sáu ngàn bước thì hạ mục tiêu còn sáu ngàn nghen. | `dùm` → `giùm` |
| 50 | train_0121 | Nam | 1 | ✓ | Trong năm ngày tới có hoá đơn nào tới hạn thì lấy ngân hàng trả hết giúp mình, riêng hoá đơn nào cao hơn 500 nghìn thì đừng trả, chờ mình xem đã. | Trong năm ngày tới có hoá đơn nào tới hạn thì lấy ngân hàng trả hết giúp mình, riêng hoá đơn nào cao hơn 500 nghìn thì đừng trả, chờ mình xem nữa. | `đã` → `nữa` |
| 51 | train_1495 | Trung | 1 | – | Nếu mai mưa thì đặt mục tiêu bước bằng nửa số bước tour của Emily thôi, không thì đặt đúng số đó nghe. | Nếu mai mưa thì đặt mục tiêu bước bằng nửa số bước tour của Emily thôi, không đặt đúng số đó nghe. | `thì` → `∅` |
| 52 | train_0343 | Trung | 1 | ✓ | Cho tôi gửi Lê Minh đúng tiền mạng tháng này, chủ nhà Phạm Hùng đúng tiền nước tháng này, lấy ở tài khoản thanh toán, không đủ cả hai khoản thì lấy tiết kiệm. | Cho tôi gửi Lê Minh đúng tiền mạng tháng này, chủ nhà Phạm Hùng đúng tiền nước tháng này, lấy ở tài khoản thanh toán, không đủ cả hai khoản thì lấy tiền tiết kiệm. | `∅` → `tiền` |
| 53 | train_1025 | Nam | 1 | – | Tôi tính mai đãi mối ly cà phê Cao Nguyên. Phiếu bên đó mà hổng còn cái nào xài được thì lấy điểm đổi ly phin sữa đá giùm tôi nha. | Tôi tính mai đãi mối ly cà phê Cao Nguyên. Phía bên đó mà hổng còn cái nào xài được thì lấy điểm đổi ly phin sữa đá giùm tôi nha. | `phiếu` → `phía` |
| 54 | train_1091 | Trung | 1 | – | Chết rồi, chiều ni em dạy thử ở trường tới tối mới về, ghi lại giùm em tập Gia đình là số một trên VTV3 tối ni nghe. | Chết rồi. Chiều ni em dạy cửa ở trường tới tối mới về, ghi lại giùm em tập Gia đình là số một trên VTV3 tối ni nghe. | `thử` → `cửa` |
| 55 | train_0206 | Bắc | 1 | – | Trần Văn Tuấn thì chuyển đúng tiền điện tháng này, chủ nhà Phạm Hùng đúng tiền mạng tháng này, trừ vào tài khoản thanh toán, không đủ cả hai thì lấy tiết kiệm nha. | Trần Văn Tuấn thì chuyển đúng số tiền điện tháng này, chủ nhà Phạm Hùng đúng tiền mạng tháng này, trừ vào tài khoản thanh toán, không đủ cả hai thì lấy tiết kiệm nha. | `∅` → `số` |
| 56 | train_0805 | Bắc | 1 | ✓ | Từ nhà tôi ra hồ Tây bao xa hả em? Tôi chạy ra đó rồi chạy về thì tính xem tổng cộng mấy cây số, xong thêm vào danh sách việc cần làm cho tôi là sáu giờ sáng mai chạy đúng chừng ấy nhé. | Từ nhà tôi ra Hồ Tây bao xa hả em? Tôi chạy ra đó rồi chạy về thì tính xem tổng cộng mấy cái số, xong thêm vào danh sách việc cần làm cho tôi là sáu giờ sáng mai chạy đúng chừng ấy nhé. | `cây` → `cái` |
| 57 | train_0699 | Bắc | 1 | – | Mai Anh vừa báo đổi số đấy, em sửa trong danh bạ thành không chín không tám, không không không, không hai ba hai nhé, xong nhắn Zalo cho chị ấy là anh lưu số mới rồi. | Mai anh vừa báo đổi số đấy, em sửa trong danh bạ thành không chín không tám, không không không, không hai ba nhé. Xong nhắn Zalo cho chị ấy là anh lưu số mới rồi. | `hai` → `∅` |
| 58 | train_0613 | Bắc | 1 | – | Soạn giúp em một cái email nháp gửi team chấm qa a còng example chấm test, tiêu đề là Kiểm tra phụ lục, nội dung nhờ mọi người rà lại file đính kèm, xong tạo thêm việc rà phụ lục, hạn năm giờ chiều mai nha. | Soạn giúp em một cái email nháp gửi team chấm qua a còng example chấm test, tiêu đề là kiểm tra phụ lục, nội dung nhờ mọi người rà lại file đính kèm, xong tạo thêm việc rà phụ lục, hạn năm giờ chiều mai nha. | `qa` → `qua` |
| 59 | train_0852 | Bắc | 1 | ✓ | Thế huỷ giúp chị cái lịch bốn giờ đấy đi. | Thế huỷ giúp chị cái lịch bốn giờ nãy đi. | `đấy` → `nãy` |
| 60 | train_1725 | Trung | 1 | – | Con check coi mai cô khám răng mấy giờ, cô muốn đi trước nửa tiếng. | Con check coi mai cô khám ra mấy giờ, cô muốn đi trước nửa tiếng. | `răng` → `ra` |
| 61 | train_0735 | Bắc | 1 | ✓ | Hà Nội giờ đang bao nhiêu độ ấy nhỉ, mở cái app thời tiết Sao Mai lên xem giúp anh với. | Hà Nội giờ đây bao nhiêu độ ấy nhỉ, mở cái app thời tiết sao mai lên xem giúp anh với. | `đang` → `đây` |
| 62 | train_1415 | Trung | 1 | – | Em giao nhầm rồi, việc voucher đặt hoa mới là của Na, còn bánh kem thì trả lại cho ba nghe. | Em giao nhầm rồi, việc voucher đặt khoa mới là của Na, còn bánh kem thì trả lại cho Ba nghe. | `hoa` → `khoa` |
| 63 | train_0622 | Bắc | 1 | ✓ | Thế em lưu cái kết luận ấy vào ghi chú, đặt tên là Tổng kết họp cho chị nhé. | Thế em lưu cái kết luận ấy vào ghi chú, đặt tên là Tổng Kết Hợp cho chị nhé. | `họp` → `hợp` |
| 64 | train_0563 | Bắc | 1 | ✓ | Này, viết giúp tôi cái email gửi Ngân, bảo là tôi nhận được hồ sơ rồi nhé. | Này, ghi giúp tôi cái email gửi Ngân, bảo là tôi nhận được hồ sơ rồi nhé. | `viết` → `ghi` |
| 65 | train_1224 | Nam | 1 | – | Nhắn Minh dùm anh là mai mưa, hủy cà phê nha... mà thôi khỏi hủy, nói dời qua tám giờ tối đi. | Nhắn Minh dùm anh làm mai mưa, hủy cà phê nha... mà thôi, khỏi hủy, nói dời qua tám giờ tối đi. | `là` → `làm` |
| 66 | train_0318 | Nam | 1 | ✓ | Hoá đơn nào tới hạn trong 5 ngày tới thì trả bằng ngân hàng hết nha, còn cái nào quá 500 nghìn thì cứ để đó, chưa trả. | Hoá lương nào tới hạn trong 5 ngày tới thì trả bằng ngân hàng hết nha, còn cái nào quá 500 nghìn thì cứ để đó, chưa trả. | `đơn` → `lương` |
| 67 | train_1526 | Bắc | 1 | – | Xem gần phòng tập bên Lê Đức Thọ có quán The Lemon Tree Café không, không có thì tìm gần trường tớ nhé. | Xem gần phòng tập bên Lê Đức Thọ có quán The Lemon Tree Café không, không thì tìm gần trường tớ nhé. | `có` → `∅` |
| 68 | train_1696 | Nam | 1 | – | Khoản Northwind Trading chuyển sáng nay, em gửi tiết kiệm sáu tháng nguyên số đó, rồi nhắn Zalo bà Hồng là tui gửi rồi nha. | Tài khoản Northwind Trading chuyển sáng nay, em gửi tiết kiệm sáu tháng nguyên số đó, rồi nhắn Zalo bà Hồng là tui gửi rồi nha. | `∅` → `tài` |
| 69 | train_1315 | Trung | 1 | – | Cái bật đèn phòng khách bỏ đi hỉ, tốn điện lắm. Với sáng mai sáu rưỡi nhắc em gọi cho mạ nghe. | Cái bật đèn phòng khách bỏ đi hỉ tốn điện lắm, với sáng mai sáu rưỡi em gọi cho Mạ nghe. | `nhắc` → `∅` |
| 70 | train_0657 | Bắc | 1 | – | Gọi ngay giúp em số không chín tám bảy, không không một, một hai hai không, số anh trực ca hôm nay đó. | Gọi nhắn giúp em số không chín tám bảy không không một một hai hai không, số anh trực ca hôm nay đó. | `ngay` → `nhắn` |
| 71 | train_1427 | Nam | 1 | – | Tắt cái quạt phòng làm việc đi em. Còn mail gia hạn OrbitCloud+ thì lưu trữ hay xoá, em hỏi lại anh nha. | Tắt cái quạt phòng làm việc đi em. Còn mail gia hạn OrbitCloud Plus thì lưu trữ hay xoá, em hỏi lại anh nha. | `∅` → `plus` |
