# PLAN v3 — cải thiện sau lần nộp đầu (public 59) — 11/11/2026

Căn cứ: `plans/REPORT_public_rootcause.md`. Bản đang nộp: ASR v2 medium (`w-m4q9`) → bộ lọc `pack_b` → FC `fmtA_48e`
(`q-f48e`). Mọi con số "kỳ vọng" trong file này là **ước lượng, chưa đo**; chỉ giữ một thay đổi khi nó thắng trên thước đo
ở mục 0.

## Tóm tắt

- Mất điểm ở hai chỗ: **ASR nghe sai thực thể của miền mới** (khớp thực thể 60.5% so với 78.8% ở app cũ) và **FC làm không
  trọn quy trình** (tìm rỗng thì dừng: 25 mẫu; tìm thấy vẫn dừng: 18; không gọi gì dù yêu cầu rõ: ~10–12; bỏ sót yêu cầu
  thứ hai). Bộ lọc app và schema không phải nguyên nhân.
- Thứ tự làm: **(0) dựng thước đo → (1) sửa lúc infer, không train → (2) train lại ASR rồi FC → (3) ý tưởng lớn**, song song
  là **Docker** (bắt buộc cho vòng private).
- Phát hiện đáng giá nhất về đo lường: model cũ được **59.4 trên val_ood sạch**, public được **59** → val_ood (5 app không
  có trong train) là thước đo gần public nhất mà mình có. `fmtA_48e` đã train trên val_ood nên mất thước đo này; phải lấy lại.

## Ràng buộc

| | |
|---|---|
| Tham số | < 6B. Hiện 4.206 (FC) + 0.77 (ASR medium) + 0.135 (lọc) = 5.11B → **còn 0.89B, vừa đủ đổi ASR sang large (1.54B) = 5.88B** |
| Thời gian | 30 s/mẫu; Docker offline cho vòng private |
| Luật tự đặt | Không train/tune theo public test; không đọc `*_environment/`; không lấy tên/tool của test đưa vào data |
| Tiền | Modal `duytungihb` còn ~$20 (ước); 3 workspace kia hết. Kaggle 2×T4 miễn phí. myserver 2×A100-40GB chỉ khi chủ repo cho phép |

---

## 0. Dựng thước đo trước (không train, ~2 giờ, ~$1)

| # | Thước đo | Cách dựng | Dùng để |
|---|---|---|---|
| M1 ✅ | `val_in` sạch (128) | **11/11 đo lại đúng cấu hình đã nộp (không `--ask-tau`): 76.5, F1 .803, state .891, 0 lỗi; khớp 128/128 mẫu với lần `--ask-tau 0.2` (ngưỡng không kích hoạt lần nào). latency p50 13.8 s / p90 33.3 s (trước khi sửa race: 18.6 / 204 s). Lệnh: `agent/eval_val.py`, report `agent/runs/clef/reports/fmtA_48e_notau.val_in.json`** | chặn hồi quy |
| M2 ✅ | `val_in` + `val_ood` **có lỗi ASR giống điều kiện public** — **11/11: `agent/runs/asr/val.M2.asr.json` (lỗi của PhoWhisper-medium GỐC, pha loãng q=0.45 bằng `agent/make_m2.py`; WER 11.3%, 59% câu có lỗi; hiệu chỉnh trên 405 đoạn: WER 11.2%, thực thể khớp 76.4%). Mốc `fmtA_48e` val_in M2: 63.5 (F1 .719, state .789) vs sạch 76.5 → −13.0 ± 3.2 (kém 24, tốt 2, bằng 102). Report `fmtA_48e_M2.val_in.json`** | Xem **§0.1** bên dưới: chọn máy sinh lỗi bằng cách hiệu chỉnh trên 405 đoạn giữ riêng cho khớp mức lỗi của v2 trên public, rồi chạy trên 1 282 đoạn audio val | đo chống chịu ASR (trước giờ không đo được vì ASR của mình đã thuộc audio train) |
| M3 | `val_ood` (259, 5 app lạ) | chỉ hợp lệ với model **không** train trên val_ood: `fmtA_mid_e1` hôm nay, FC v3 ở mục 2 | thước đo app lạ, gần public nhất |
| M4 | Tỉ lệ thực thể khớp transcript | `agent/analyze_public.py` (đã có), chạy trên 405 đoạn giữ riêng + tập TTS giọng lạ | chọn ASR |
| M5 ✅ | Chẩn đoán trên public (không tune) — **11/11: Q1 bước đầu đúng tool như assistant: app cũ 97.0% (810/835), app mới 84.5% (224/265), DỪNG 85.0%, tổng 92.3%. 41 lỗi ở app mới: 20 chọn DỪNG (không gọi gì), 14 sang app khác, 6 sai tool cùng app; chỉ 2 do bộ lọc. Yếu nhất: habit_tracker 55.6%, parking 70.8%, grocery 78.3%. Kết quả: `agent/runs/clef/diag_public_q1.jsonl`, script `agent/diag_public_q1.py`** | Q1 bước đầu cho ~1 100 lượt history có nhãn sẵn trong đầu vào, tách app mới/cũ (~$0.5) | biết FC yếu ở app mới tới đâu |

Quy tắc chọn: một thay đổi được giữ khi **M2/M3 tăng** và **M1 không giảm quá 1 điểm**.

### 0.1 Thiết kế M2 (đã chỉnh)

**Vấn đề của thiết kế cũ:** PhoWhisper gốc chưa fine-tune lỗi theo kiểu khác hẳn v2 (chưa học cách viết số, tên app,
khẩu ngữ trong data; chưa học dùng prompt) → mức lỗi và loại lỗi có thể lệch xa điều kiện thi, làm M2 phạt oan hoặc
khen oan. Phải **hiệu chỉnh** trước khi dùng.

**Mục tiêu hiệu chỉnh** (đo được, không cần nhãn public):

| Chỉ số | Mục tiêu | Nguồn |
|---|---|---|
| Thực thể khớp transcript (M4) | **~72%** tổng; ~60% với thực thể lạ | v2 trên public: 78.8% app cũ / 60.5% app mới / 71.8% tổng |
| WER trên 405 đoạn giữ riêng | 7–12% (v2: 6.85% — public khó hơn nên cho phép cao hơn) | `heldout.v2_medium` |
| Tỉ lệ câu có lỗi | 45–60% (v2 held-out: 44.7%) | A5b |

**Máy sinh lỗi ứng viên** (chạy trên 405 đoạn giữ riêng trước, chọn cái gần mục tiêu nhất):

| # | Ứng viên | Ghi chú |
|---|---|---|
| G1 | PhoWhisper-**medium** gốc, có prompt lượt trước | cùng cỡ với v2, chưa nghe audio train |
| G2 | PhoWhisper-**small** gốc, có prompt | lỗi nhiều hơn — dùng nếu G1 quá sạch |
| G3 | v2 + lấy mẫu T=0.7 (một mẫu/đoạn) | đúng kiểu lỗi của v2, nhưng in-sample trên val → chỉ dùng nếu G1/G2 lệch kiểu lỗi quá xa |

Chọn theo M4 trước, WER sau. Cùng một chuẩn hoá đầu ra như v2 (số, dấu câu) để FC không bị phạt vì định dạng.

**Các bước:**
1. Chạy G1 (và G2 nếu cần) trên 405 đoạn giữ riêng → `agent/asr_score.py` (WER) + M4 (`agent/analyze_public.py`, phần B) → chọn máy sinh lỗi.
2. Chạy máy đã chọn trên 1 282 đoạn val (val_in + val_ood), prompt bằng transcript lượt trước của chính nó (như lúc thi) → `agent/runs/asr/val.M2.asr.json`.
3. Kiểm: WER + M4 của transcript M2 trên val so với gold; ghi vào report.
4. Mốc FC trên M2: `eval_val.py --asr val.M2.asr.json` cho `fmtA_48e` (val_in) và `fmtA_mid_e1` (val_in + val_ood, vì val_ood không nằm trong train của model này).

**Kết quả mong đợi:** điểm `fmtA_48e` tụt từ 76.5 về khoảng 60–70 trên val_in M2. Nếu tụt quá ít (< 3 điểm), máy sinh lỗi
quá sạch → quay lại bước 1 chọn G2.

**Nơi chạy và tiền:** ASR ~1 700 đoạn ≈ 10 phút L40S (Modal ~$0.3, cần upload ~770 MB audio train) hoặc Kaggle 2×T4 (miễn phí,
cần tạo dataset audio). FC 2 lượt val_in + 1 lượt val_ood ≈ $0.6. **Tổng ≈ $1 hoặc ≈ $0.6 nếu ASR chạy Kaggle.**

### 0.1b Kết quả M2 (11/11)

| | val_in sạch | val_in M2 (lỗi ASR) |
|---|---|---|
| `fmtA_48e` | 76.5 | **63.5** (−13.0 ± 3.2, 24 mẫu kém / 2 tốt) |

- Khoảng tụt 13 điểm trên val_in ≈ 75% khoảng chênh val_in→public (17.5) → **lỗi ASR là phần lớn khoảng chênh**, kể cả với app quen. Phần còn lại là app lạ.
- 24 mẫu tụt (đọc tay): 11 khác tool/app (từ khoá hành động bị nghe sai: "huỷ" → "huyết", "họp" → "họ", "quạt" → "và"; câu điều kiện dài mất nửa sau), 9 cùng tool nhưng sai tham số (tên, nội dung), 2 không gọi gì, 2 thừa/thiếu call.
- Kiểu lỗi chủ đạo là **mất hoặc sai động từ hành động trong câu dài nhiều yêu cầu** — không phải tìm kiếm rỗng (val_in ít tìm kiếm). → N1/N4 sẽ ít tác dụng trên M2; **ASR tốt hơn (N3) và FC học chịu nhiễu ở cả động từ (F2) là đòn bẩy chính**.
- Hạn chế: M2 chỉ đo được thay đổi phía FC trên cùng transcript; 128 mẫu, sai số ±3 điểm.

### 0.1c Nhận xét từ M2 → điều chỉnh plan (11/11)

**Tụt điểm theo kiểu câu (val_in, `fmtA_48e`, sạch → M2):**

| Nhóm | n | Sạch | M2 | Tụt |
|---|---|---|---|---|
| Câu cuối **không có lỗi** ASR (chỉ history có lỗi) | 38 | 76.1 | 76.1 | **0.0** |
| Câu cuối **có lỗi** ASR | 90 | 76.7 | 58.1 | **18.6** |
| **Câu điều kiện** ("nếu… thì…") | 33 | 75.4 | 50.2 | **25.2** |
| Câu không điều kiện | 95 | 76.9 | 68.1 | 8.8 |
| Câu dài ≥ 20 từ | 81 | 70.1 | 55.3 | 14.7 |
| Câu ngắn < 20 từ | 47 | 87.6 | 77.5 | 10.1 |

**Nhận xét:**
1. **Lỗi ở các lượt history gần như vô hại** (38 mẫu câu cuối sạch: tụt 0.0) → toàn bộ thiệt hại đến từ **câu cuối**. Ưu tiên chất lượng ASR ở câu cuối; không cần tốn công cho nhiễu history.
2. **Câu điều kiện là điểm gãy lớn nhất** (−25 điểm): một từ nghe sai ở vế điều kiện hoặc vế hành động ("huỷ" → "huyết", "họp" → "họ") làm hỏng cả chuỗi tool. Câu dài nhiều yêu cầu cũng tụt mạnh.
3. **Huấn luyện chống nhiễu hiện tại chưa đủ:** `fmtA_48e` đã train với nhiễu ASR (40% mẫu: 297 câu thay nguyên câu + 6 836 câu theo bảng 441 cụm của v2) mà vẫn tụt 13 điểm. Bảng nhiễu quá hẹp (chỉ từ 405 đoạn), phần lớn là tên riêng và từ đệm, gần như không đụng **động từ hành động** và **từ nối điều kiện**.
4. **Máy sinh lỗi M2 dùng được để sinh data train:** PhoWhisper gốc + pha loãng q=0.45 cho lỗi nghe nhầm thật, phủ mọi loại từ, đã hiệu chỉnh khớp mức lỗi thi. Chạy nó trên **toàn bộ audio train + TTS** (khác audio của val → không rò) là nguồn nhiễu cho F2 thay cho bảng 441 cụm.
5. **Độ nhạy của M2:** SE ±3.2 trên 128 mẫu → chỉ thấy được thay đổi ≳ 4–6 điểm; thay đổi nhỏ hơn phải xem so cặp từng mẫu và tập trung vào 90 mẫu có câu cuối lỗi.

**Điều chỉnh plan:**

| Mục | Điều chỉnh |
|---|---|
| **F2** (nhiễu ASR cho FC) | Đổi nguồn: dùng **máy sinh lỗi M2** (PhoWhisper-medium gốc trên audio train/TTS, q≈0.45) thay bảng 441 cụm; tỉ lệ mẫu nhiễu 40% → **60%**; nhiễu **chỉ ở câu cuối** là chính (history nhiễu nhẹ). Chạy trên Kaggle (~1 h, $0) |
| **F4 / D5** | Tăng tỉ trọng **câu điều kiện và câu nhiều yêu cầu** có nhiễu ở động từ / từ nối; mục tiêu ≥ 25% data FC |
| **N3** (ASR v1 large) | Giữ ưu tiên số 1 trong mục không train: lợi ích của nó rơi đúng vào câu cuối |
| **N8 / I4** (ý kiến thứ hai) | Đưa **I4 (nhiều giả thuyết ASR cho câu cuối)** lên cùng N8: lấy 2–3 giả thuyết cho câu cuối, chạy Q1 trên từng cái; lệch nhau thì chọn theo đồng thuận. Đánh thẳng vào lỗi động từ |
| **N1, N4** | Giữ, nhưng lợi ích chỉ thấy trên public (val_in ít tìm kiếm); không kỳ vọng đổi M2 |
| **Mục tiêu FC v3 (F6)** | Thêm tiêu chí: **M2 val_in ≥ 70** (từ 63.5) và tụt ở câu điều kiện ≤ 12 điểm |

### 0.2 Nhận xét từ M5 (11/11) — cải thiện được gì

M5 đo **bước đầu tiên** (chọn tool) trên 1 299 lượt history của public, đầu vào là transcript ASR v2. Không đo tham số và các
bước sau, nên đây là cận dưới của lỗi FC.

| Nhóm | Đúng | Nhận xét |
|---|---|---|
| App cũ | 97.0% (810/835) | FC chọn tool ổn kể cả khi đầu vào có lỗi ASR → điểm mất ở app cũ nằm ở **tham số và bước sau**, không ở chọn tool |
| App mới | 84.5% (224/265) | kém app cũ 12.5 điểm; lượt đầu 87.3%, **lượt sau 82.0%** (ngữ cảnh ASR nhiễu dồn lại + câu hỏi tiếp dùng đại từ) |
| Không cần gọi gì | 85.0% (164/193) | 29 lần gọi thừa, **9 lần ở ecommerce** (người dùng kể/phàn nàn về đơn hàng, model vội tra đơn hoặc tạo đơn trả) |
| Cần hỏi lại | 1/6 | model gần như không hỏi lại (mẫu nhỏ) |

**41 lỗi ở app mới chia ba loại:**

| Loại | Số | Nguyên nhân (đọc tay) | Sửa bằng |
|---|---|---|---|
| Chọn DỪNG, không gọi gì | 20 | ~10 do transcript hỏng nặng ("thịt ba chỉ" → "thì ba chỉ", "lễ hội ẩm thực" → "lấy hỏi âm lịch", "Cảnh phát hôm ni bao nhiêu"); ~10 do câu hỏi tiếp trong miền lạ ("Còn chi khác rẻ hơn không", "Nấu cho sáu người thì cần nhiêu gà") | ASR v3 (D3); FC: mẫu hỏi tiếp trong miền "vỏ mới" (F3, D5) |
| Sang app khác | 14 | cặp app trùng chức năng: `public_transit`↔`maps` (2), `train_bus_tickets`↔`flights`/`calendar` (2), `ecommerce`↔`grocery` (2), `parking`↔`ride_hailing` (2), `habit_tracker`↔`shopping_list`; vài cái do ASR ("vé tàu" → "vé bầu" → chọn `flights`) | **F3b (mới): mẫu tương phản cho cặp app trùng chức năng** — cùng một câu, hai bộ app đã cài khác nhau → hai tool khác nhau |
| Cùng app, sai tool | 6 | `parking.find`↔`list_sessions`/`reserve`, `stocks.get_quote`↔`get_portfolio` | F3 (đổi vỏ có đủ nhóm tool tìm / xem của tôi / đặt) |

**Điều chỉnh ưu tiên rút ra:**

1. **Xác suất của tool đúng khi model sai rất thấp** (trung vị 0.078; chỉ 4/20 lỗi DỪNG có p > 0.2) → **N5 (chặn dừng sớm bằng ngưỡng) cứu được ít ở bước đầu** → hạ ưu tiên; lỗi này phải sửa bằng train (F3/F3b) hoặc ASR tốt hơn.
2. **Bộ lọc chỉ gây 2/41 lỗi** → **bỏ N7**; giữ nguyên bộ lọc.
3. **Độ tự tin dùng được làm cờ báo:** `conf < 0.7` chỉ chiếm 5% số lượt (70) nhưng chứa **40% tổng lỗi** (57% lượt trong nhóm này sai) → **I6 (hỏi ý kiến thứ hai khi conf thấp) đáng làm sớm**, chi phí chỉ +5% request. Ý kiến thứ hai nên khác nguồn: catalog đầy đủ thay vì hai tầng, hoặc giả thuyết ASR thứ hai (I4).
4. **Khoảng một nửa lỗi DỪNG/nhầm app bắt nguồn từ transcript** → ASR vẫn là đòn bẩy số 1 (N3 ngay, ASR v3 sau).
5. **Thêm vào data FC:** (a) F3b cặp app trùng chức năng; (b) câu hỏi tiếp ngắn dùng đại từ trong miền mới; (c) câu kể/phàn nàn không yêu cầu hành động (đặc biệt mua sắm) với nhãn không gọi tool; (d) câu thiếu thông tin → hỏi lại.

**Ước lượng thô phần điểm do chọn sai tool bước đầu ở app mới:** 44% mẫu public dùng app mới × (15.5% − 3.0%) ≈ **5–6 điểm**
trong khoảng chênh 17.5 điểm so với val_in. Phần còn lại nằm ở tham số (ASR) và các bước sau (tìm rỗng, dừng sớm).

---

## 1. Cải thiện KHÔNG cần train lại (làm ngay, ~1 ngày công, ~$4–6 Modal để đo)

Xếp theo lợi ích ước lượng / công sức.

| # | Việc | Cách làm | Sửa lỗi nào | Kỳ vọng | Rủi ro |
|---|---|---|---|---|---|
| N1 ✅ **đã cài (`--retry-empty`), 11/11** — M2 val_in 63.4 vs 63.5 (127 bằng, 1 kém 0.1); public: luật bật 9/634 mẫu, 7 mẫu đổi state (event_tickets: tìm lại → đặt vé → làm tiếp); file `result/submission_fmtA_48e_e46f_n1.jsonl`, **public 62.76 (+0.72 so với 62.04)** | **Phục hồi khi tìm kiếm rỗng** | Trong `clef_agent`: tool đọc trả danh sách rỗng → thử lại **một lần**, **chỉ với tool có tham số tìm kiếm tuỳ chọn** (gọi lại không kèm từ khoá để lấy danh sách, vd `event_tickets.search`: cứu 9/9 trên public). **Không áp cho `contacts.search`** (liệt kê cả danh bạ dễ gửi nhầm người). Tool có tham số bắt buộc (flights, train_bus_tickets, public_transit) nới truy vấn vẫn rỗng → để N4 lo | B1 (25 mẫu đang 0 điểm) | +0.5–1 (đã hạ sau kiểm offline: 20/59 lần rỗng nới ra kết quả, trừ contacts còn ~12) | call thừa làm giảm F1 (mẫu cứu được chỉ lấy ~0.4–0.6 điểm thay vì 1) |
| N2 | **Chọn kết quả theo âm thanh** (xem Ý tưởng I1) | Khi model chọn 1 mục trong danh sách kết quả, cộng thêm điểm "audio nghe giống tên mục nào nhất" (Whisper chấm log-likelihood từng tên ứng viên) | A1 khi danh sách đã có (vd "liên hoan tin nhắn" → "Liên hoan phim ngắn") | +1–3 | thêm ~0.5–1 s/mẫu |
| N3 ✅ **public 62.04 (+3.04 so với 59), 11/11** — `result/submission_fmtA_48e_e46f.jsonl`; FC giữ nguyên; khớp thực thể public app cũ 78.8→79.8, app mới 60.5→62.1; 138/634 mẫu đổi state | **Đổi ASR sang v1 large đã có** (`pw-tmp-e46f`) | Không train: trên 405 đoạn giữ riêng WER 6.23 vs 6.85, lỗi tên 19 vs 28. Tổng tham số 5.88B. Chọn bằng M4 + M2 | A1, A2 | +0.5–2 | chênh trong nhiễu; FC được train với nhiễu của v2; chậm gấp đôi |
| N4 | **Sửa địa danh ở mức tham số** | Với tham số kiểu nơi chốn (origin/destination/near/city), đưa giá trị model sinh về địa danh gần âm nhất trong danh bạ địa danh VN chung (63 tỉnh thành, ga, sân bay), chỉ khi giá trị không phải địa danh hợp lệ và có đúng một ứng viên gần | A1 (Vin→Vinh, Hủy→Huế, Hàng Sơn→Lạng Sơn) | +0.5–1.5 | sửa nhầm; phải đo tỉ lệ sửa sai trên M2 |
| N5 ⬇ | **Chặn dừng sớm** — *hạ ưu tiên sau M5: khi model sai, xác suất tool đúng trung vị chỉ 0.078 → ngưỡng cứu được ít; chỉ thử cho trường hợp "đã tìm thấy kết quả mà dừng" (18 mẫu), làm sau cùng* | Model chọn DỪNG ngay sau một tool đọc có kết quả, câu có động từ hành động chưa được thực hiện và P(DỪNG) < τ → lấy tool xếp sau. τ chọn trên M1/M2 bằng `probs` đã ghi | B2 (18 mẫu) | +0.5–1.5 | gọi thừa tool ghi làm sai state → mẫu về 0; chỉ bật nếu M1 không giảm |
| N6 | Chỉnh lại `--ask-tau` cho model mới | Ngưỡng 0.2 gần như không kích hoạt với `fmtA_48e` (0 lần trên val_in); quét 0.05–0.2 | hỏi lại | ±0.5 | nhỏ |
| ~~N7~~ | ~~App chưa từng thấy luôn ở tầng 1~~ — **BỎ sau M5:** bộ lọc chỉ gây 2/41 lỗi ở app mới | | | | |
| **N8** (= I6, đưa lên sau M5) | **Ý kiến thứ hai khi độ tự tin thấp** | Bước Q1 có `conf < 0.7` (≈5% số bước, chứa 40% lỗi bước đầu trên public) → hỏi lại cùng bước với **catalog đầy đủ** (không hai tầng); hai lần khớp thì giữ, lệch thì theo lần có xác suất cao hơn. Mở rộng sau: ý kiến thứ hai từ giả thuyết ASR thứ hai (I4). Ngưỡng 0.7 lấy từ phân bố `conf`, kiểm hồi quy trên M1 | B3, nhầm app | +0.5–2 | +5% request; hai lần cùng sai thì không cứu được |

Không làm: thêm từ khoá bộ lọc cho app mới (recall đã 99–100%); A4b (đã bỏ).

**Kiểm offline trước khi làm N (11/11, $0):**
- *Nhãn train sau một tool đọc trả rỗng:* 77 lần đi tiếp tool ghi (hầu hết `tv.guide` trong câu điều kiện), 11 lần dừng, **1 lần gọi lại** → nhãn gần như không bao giờ "tìm lại". Retry luôn làm thêm call ngoài nhãn (giảm F1), chỉ đáng khi nó cứu được hành động phía sau.
- *Trần của N1 trên public* (chạy lại call trong `ToolEnv`, thử nới truy vấn): 59 lần tìm rỗng → **20 lần nới ra kết quả**: `event_tickets.search` 9/9 (bỏ query → liệt kê sự kiện, có đúng tên cần tìm: "liên hoan tin nhắn" → "Liên hoan phim ngắn", "gọi tiệm ngày mưa" → "Vở kịch Ngày Mưa"), `contacts.search` 8 (**không làm**: liệt kê cả danh bạ dễ gửi nhầm người, tên không có trong danh bạ thì nhãn có thể là hỏi lại), train/grocery/ecommerce 1 mỗi loại. `flights` 8, `train_bus_tickets` 6, `public_transit` 6 vẫn rỗng → cần **N4** (đưa địa danh về tên đúng), không phải nới truy vấn.
- *N3 (v1 large) trên 405 đoạn giữ riêng:* thực thể khớp **80.0% vs 74.5%** của v2 (n=55, chênh 3 thực thể), lỗi tên 19 vs 28 / 289 tên → nghiêng về v1 large nhưng mẫu nhỏ.
- *M1 không đo được lợi ích của N1/N4/N5* (val_in chỉ 2/128 lần tìm rỗng, 2/128 dừng sớm) → M1 chỉ chặn hồi quy; lợi ích chỉ thấy qua lượt nộp public (hoặc M2).

**Thứ tự chạy (đã chỉnh sau M5 + kiểm offline):** M2 → **N3** (ASR v1 large) → **N1 bản hẹp** → **N4** (địa danh) → **N8** (ý kiến thứ hai khi conf thấp) → N2/I1 (chọn theo âm thanh) → N6 → N5 (nếu còn giờ). N7 bỏ. Mỗi bước một lượt val (~$0.2–0.5) để chặn hồi quy trên M1; lợi ích của N1/N3/N4 chỉ thấy được trên M2 hoặc lượt nộp public. Gộp các thay đổi thắng → chạy public → nộp lần 2.

### Việc bắt buộc song song: Docker + độ trễ (không train)

- Image offline: PhoWhisper + FC gộp + bộ lọc + vLLM (hoặc HF generate) + `submit.py`; đọc `/data`, ghi `result/submission.jsonl`.
- Đo **30 s/mẫu** thật (ASR từng lượt + FC nối tiếp trên một GPU); đặt ngân sách thời gian: gần hết giờ thì dừng và nộp các call đã có.
- Lỗi đã biết phải giữ bản sửa: khoá nạp bộ lọc (đã sửa), retry request.
- Vòng private chấm bằng Docker, nên việc này quan trọng hơn mọi điểm cộng ở trên.

---

## 2. Train lại từng bộ phận

Ngân sách tham số quyết định cấu hình: **FC giữ Qwen3.5-4B, phần dư dồn cho ASR large** (ASR là nguyên nhân số 1).

| Bộ phận | Hiện tại | Đề xuất | Tham số |
|---|---|---|---|
| ASR | PhoWhisper-medium v2 | **PhoWhisper-large v3** (recipe v2 + data mới) | 0.77 → 1.54B |
| FC | Qwen3.5-4B + LoRA `fmtA_48e` | cùng nền, **LoRA v3** train trên data sửa | 4.206B |
| Bộ lọc | bi-encoder B + BM25 | **giữ nguyên** | 0.135B |
| Tổng | 5.11B | **5.88B** | < 6B |

### 2.0 Dữ liệu phải đa dạng thật (yêu cầu của chủ repo, 11/11)

**Hiện trạng (đo trên 2 066 câu TTS hôm 10/11):** bám sát phân bố lệch của train, bỏ trống miền mới.
"vàng" 100 lần nhưng 78 lần là SJC, PNJ/DOJI/9999/24K: 0; "tỉ giá" 0, USD 0; vé tàu 0, bãi xe 0, công thức nấu 0, cổ phiếu 1.
Nhãn train cũng lệch: `gold_price` sjc 103 / ring 17 / world 2; tỉ giá USD 32 / EUR 8 / JPY 7.

**Nguyên tắc sinh data v3 (áp cho cả câu TTS của ASR và mẫu FC):**

| # | Trục đa dạng | Cách làm | Kiểm bằng số |
|---|---|---|---|
| D1 | **Cách nói của từng giá trị enum** | mỗi giá trị enum trong catalog train ≥ 8 cách nói: vàng miếng / vàng SJC / vàng nhẫn trơn 9999 / vàng PNJ, DOJI, Bảo Tín / vàng thế giới, quốc tế; đô, đô la Mỹ, USD, tiền Mỹ, euro, yên, won, nhân dân tệ, bảng Anh; RON 95 / xăng chín lăm / E5 / dầu DO | mỗi giá trị enum ≥ 8 biến thể; không giá trị nào > 40% trong nhóm của nó |
| D2 | **Cân lại phân bố lệch** | lấy mẫu đều theo giá trị (không theo tần suất train): ring/world ngang sjc, EUR/JPY/KRW/CNY ngang USD | tỉ lệ max/min giữa các giá trị ≤ 3 |
| D3 | **Thực thể miền mở từ nguồn chung** (không lấy từ test) | 63 tỉnh thành + quận/huyện lớn; ga, bến xe, sân bay; mã chuyến (SE1–SE8, TN, SPT, VN/VJ/QH + số); sự kiện/phim/lễ hội; món ăn, nguyên liệu; sản phẩm có mã; mã cổ phiếu đọc từng chữ; tên người/tên quán mới | ≥ 1 500 thực thể khác nhau, mỗi thực thể ≤ 3 câu; ≥ 30% thực thể không có trong train |
| D4 | **Số và đơn vị** | chỉ/lượng/cây, phần trăm, độ C/F, km, kg, triệu/tỉ/nghìn/"k", giờ kém, ngày âm, số điện thoại, email đánh vần | mỗi kiểu ≥ 40 câu |
| D5 | **Kiểu câu** | 1 yêu cầu / 2–3 yêu cầu / điều kiện ("nếu… thì") / tự sửa ("à thôi…") / hỏi tiếp dùng đại từ ("cái đó", "chuyến đấy") / câu cụt | mỗi kiểu ≥ 10% |
| D6 | **Giọng và vùng miền** | Bắc / Trung / Nam + khẩu ngữ (rứa, hỉ, nghen, tui); xưng hô anh/chị/cô/chú/con/tớ | mỗi vùng ≥ 20% |
| D7 | **Âm thanh** | thêm giọng (`tts-next`), tốc độ 0.85–1.2, cao độ, nhiễu nền SNR 0–20 dB, vang, băng tần điện thoại; **giữ riêng 1–2 giọng** không cho train | ≥ 8 giọng; tập giữ riêng theo giọng |
| D8 | **Miền "vỏ mới"** (cho FC, mục F3) | đổi vỏ app thật sang ≥ 10 miền khác nhau, tên không trùng tool của test | ≥ 20% mẫu FC |

**Kiểm đa dạng trước khi sinh audio** (script `synth/tts_text/qa_diversity.py`, chưa viết): in bảng các chỉ số ở cột phải; trượt
ngưỡng nào thì sinh bù trước khi tốn token TTS. Chặn rò: không câu nào trùng 405 đoạn giữ riêng, val_in/val_ood, hay public test.

### 2.1 ASR v3

| # | Việc | Chi tiết |
|---|---|---|
| A1 | **Data TTS đa dạng theo §2.0 (D1–D7)** (~3 000–4 000 câu × 3 giọng) | địa danh VN (tỉnh, huyện, ga, bến xe, sân bay), mã chuyến (chữ + số: SE5, TN1, VN213), tên sự kiện/phim/lễ hội, món ăn và nguyên liệu, sản phẩm có mã (MiniMix M2), mã cổ phiếu đọc từng chữ, email đánh vần ("a còng… chấm…"), câu dài 2–3 yêu cầu. Nguồn: danh sách chung + tự viết, **không lấy từ public test**. Còn ~577k token `tts-flash` + 1M `tts-next` |
| A2 | **Đa dạng giọng** | A5 cho thấy TTS cùng 4 giọng quá dễ (WER 0.44%) → thêm giọng của `tts-next`, đổi tốc độ/cao độ, nhiễu nền mạnh hơn (SNR 0–15 dB, giới hạn băng tần kiểu điện thoại) |
| A3 | **Giữ riêng theo giọng** | để nguyên 1–2 giọng không cho train → đo M4 trên giọng lạ (gần điều kiện public hơn giữ riêng theo câu) |
| A4 | Train | PhoWhisper-large, LoRA như v2 (CE+LS, CTC cuối + lớp giữa, KWS, prompt V2.1, nhiễu lượt trước), chọn checkpoint theo **lỗi tên + M4**, không chỉ WER |
| A5 | Tiêu chí | trên 405 đoạn: lỗi tên ≤ 19 (mức v1 large), WER ≤ 6.2%; M4 giọng lạ tăng ≥ 10 điểm so với v2 |

Chi phí train: ~3 h H100 ≈ $12, hoặc ~5–6 h A100 myserver ($0, cần phép), hoặc Kaggle T4 (large + base fp16, ~10 epoch có thể
vượt 12 h/phiên → giảm còn 5 epoch). Phương án rẻ dự phòng: train tiếp v2 medium 2–3 epoch với data A1/A2 (~$2).

### 2.2 FC v3

| # | Việc | Chi tiết | Sửa lỗi |
|---|---|---|---|
| F1 | **Trả lại thước đo:** bỏ val_ood khỏi train | train trên `train` (+ synth), chọn model theo M3 (val_ood sạch và có lỗi ASR) | quy trình |
| F2 | **Nhiễu ASR thật, phủ thực thể** — *đã chỉnh sau M2 (§0.1c): nguồn = máy sinh lỗi M2 (PhoWhisper gốc pha loãng), 60% mẫu, nhiễu tập trung ở câu cuối* | chạy PhoWhisper gốc chưa fine-tune + checkpoint epoch sớm trên toàn bộ audio train và TTS → cặp (câu chuẩn, câu nghe sai) cho **mọi** câu train, thay bảng 441 cụm hiện tại; nhãn giữ thực thể chuẩn → model học sửa "Hàng Sơn" → "Lạng Sơn" ngay ở lần gọi đầu | A1, A2 |
| F3 | **"Khoác áo miền mới" cho app thật** | hiện mẫu đổi giao diện (`iface`) chỉ 2% data. Tăng lên ~20%: đổi tên app/tool/tham số/mô tả và thực thể của app thật sang miền khác (đặt vé bay → vé tàu xe; giao đồ ăn → đi chợ; danh sách mua → tủ bếp; lịch TV → vé sự kiện; tỉ giá/vàng → chứng khoán; đường đi → xe buýt), môi trường thật vẫn chạy bên dưới nên kết quả tool là thật. Đặt tên khác tool của test | B2, B3 (app lạ) |
| F3b | **Tương phản cặp app trùng chức năng** (từ M5) | cùng một câu nói, hai bộ app đã cài khác nhau → hai tool khác nhau (xe buýt↔bản đồ, tàu xe↔máy bay, đi chợ↔sàn TMĐT, gửi xe↔gọi xe, thói quen↔danh sách); thêm câu kể/phàn nàn không cần gọi tool và câu thiếu thông tin → hỏi lại | nhầm app (14), gọi thừa (29) |
| F4 | **Làm trọn quy trình** | tăng tỉ trọng chuỗi "tìm → hành động" và câu 2–3 yêu cầu; thêm mẫu "tìm rỗng → tìm lại → làm tiếp" sinh bằng `ToolEnv` của train (truy vấn bị nhiễu F2) | B1, B4 |
| F5 | Train (xem §2.3) | khởi đầu từ adapter `fmtA_48e` bằng `--init-adapter`, ~10k ví dụ (60% mới, 40% lặp lại data cũ), LR 5e-5, 1 epoch ≈ 2.5 h H100 ≈ $10. Train từ đầu 20k ≈ $18 nếu đủ tiền | |
| F6 | Tiêu chí | M3 có lỗi ASR tăng ≥ 3 điểm so với `fmtA_mid_e1`; M1 ≥ 75.5 | |

Thứ tự: ASR v3 trước (FC cần nhiễu của đúng ASR sẽ dùng), F2/F3/F4 dựng data song song lúc ASR train.

### 2.3 Train tiếp từ checkpoint hôm qua bằng data mới — được, với điều kiện

**FC (`fmtA_48e_e1`, 1 epoch):** làm được. Adapter ở HF `fisherman611/q-f48e-lora/final` (và myserver `runs/clef/fmtA_48e_e1/final`).
- `--resume` **không dùng được** cho data mới (nó kiểm đúng số ví dụ + kích thước file của data cũ, để chạy tiếp đúng chỗ dừng).
- Đã thêm cờ **`--init-adapter <thư mục adapter>`** vào `agent/clef/train.py` (11/11, chưa chạy thử): nạp trọng số LoRA cũ, optimizer và lịch LR mới.
- Cấu hình đề xuất: LR **5e-5** (nửa LR cũ), warmup 3%, 1 epoch; data = **60% mới + 40% lặp lại mẫu cũ** (chống quên — train chỉ data mới sẽ làm tụt val_in).
- Lợi: rẻ hơn ~2 lần so với train từ đầu (10k ví dụ ≈ 2.5 h H100 ≈ $10; A100 myserver ≈ 4.5 h, $0).
- **Giá phải trả:** checkpoint này đã học val_ood → train tiếp thì val_ood vẫn bị "nhiễm", **không dùng M3 để chọn model được**. Thước đo app lạ thay thế: (a) giữ riêng 3–4 miền "vỏ mới" (D8) chưa từng cho train làm val OOD mới; (b) M5 trên public (chỉ chẩn đoán). Nếu muốn M3 sạch thì phải train từ đầu trên `train` (~$18).

**ASR (v2 medium, 10 epoch):** cũng train tiếp được: LoRA mới trên model đã gộp `w-m4q9`, 2–3 epoch với data D1–D7 trộn audio thật
(~$2–3 H100). Rẻ, nhưng trần thấp hơn large; chọn giữa "medium train tiếp" và "large train mới" bằng M4 trên giọng giữ riêng.

---

## 3. Ý tưởng lớn

| # | Ý tưởng | Vì sao đáng | Cách làm | Cần train? |
|---|---|---|---|---|
| I1 | **Chọn theo âm thanh (acoustic-grounded choice)** | Lỗi lớn nhất là tên riêng nghe sai, nhưng lúc model phải chọn 1 mục trong kết quả tìm kiếm thì **danh sách tên đúng đã nằm trước mặt**. Văn bản ASR đã mất thông tin, audio thì chưa | Với mỗi tên ứng viên, cho Whisper chấm xác suất "audio này nói tên đó" (ép giải mã), rồi nhân vào xác suất chọn id của Clef: `P(id) ∝ P_FC(id) · P_ASR(tên│audio)^λ`. Không thêm tool call nào | Không (λ chọn trên M2) |
| I2 | **ASR chưa fine-tune làm máy sinh lỗi** | ASR của mình thuộc lòng audio train nên không sinh được lỗi để dạy FC. Model gốc cùng họ thì chưa nghe audio đó → lỗi thật, đúng kiểu âm học, phủ mọi câu, miễn phí | Chạy model gốc + checkpoint sớm trên audio train/val → vừa là data nhiễu cho FC (F2), vừa là tập val có lỗi ASR (M2) | Không |
| I3 | **Khoác áo miền mới** (F3) | Muốn tổng quát app lạ thì model phải thấy thật nhiều "app lạ" lúc train; viết môi trường giả thì đắt, đổi vỏ app thật thì gần như miễn phí và kết quả tool vẫn thật | Bảng ánh xạ miền → sinh tự động | Có (FC) |
| I4 | **Bỏ phiếu qua nhiều giả thuyết ASR** | Một câu nghe sai có thể kéo sai cả mẫu; nhiều giả thuyết cùng dẫn tới một tool call thì đáng tin, lệch nhau là tín hiệu "chưa chắc" | Lấy 3 giả thuyết ASR → chạy Q1 trên từng cái → đồng thuận thì đi tiếp, lệch thì dùng thực thể được nhiều giả thuyết ủng hộ hoặc hỏi lại. Bản train: đưa cả 3 vào prompt cho FC tự chọn | Bản bỏ phiếu: không |
| I5 | **Gợi ý luồng từ catalog** | App lạ: model tìm xong không biết bước sau là đặt. Catalog đã ghi tool nào cần id từ tool nào (x-ref) → suy ra được chuỗi "search → book" mà không cần nhãn | Dựng đồ thị x-ref từ catalog lúc infer, chèn gợi ý ngắn vào Q1 ("còn bước dùng kết quả vừa tìm: book, add_to_cart") | Nên có (FC học dùng gợi ý) |
| I6 → **N8** | **Hai góc nhìn, một model** (đã đưa lên mục 1 sau M5) | 48 mẫu public có độ tự tin < 0.5. Không đủ tham số cho 2 model FC, nhưng cùng một model với catalog đầy đủ và catalog hai tầng là hai "ý kiến" khác nhau | Khi `conf` thấp: chạy lại bước đó với catalog đầy đủ; hai lần khớp thì giữ, lệch thì theo lần có xác suất cao hơn | Không |

Mình đánh giá **I1 + I2** là hai ý có tỉ lệ lợi ích/chi phí cao nhất: không cần train, đánh thẳng vào nguyên nhân số 1, và I2 còn
giải quyết luôn việc thiếu thước đo.

---

## Lộ trình đề xuất

> **Cập nhật 11/11 16:40:** lộ trình và ưu tiên hiện hành nằm ở `plans/PLAN_72h_v3.md` (4 luồng song song: FC v3, ASR, luật lúc infer, Docker). Bảng dưới là bản cũ.

| Giai đoạn | Việc | Thời gian | Tiền |
|---|---|---|---|
| 1 | M1 ✅, M5 ✅, M2 (đang chạy) → **N3, N1 bản hẹp** | ~3 h | ~$2 |
| 2 | **N4, N8**, rồi N2/I1, N6 (N5 nếu còn giờ) → **nộp lần 2** | ~4 h | ~$3 |
| 3 (song song) | Docker + đo 30 s | ~4 h | ~$1 |
| 4 | TTS A1/A2 → ASR v3 large | ~5 h (3 h train) | $12 hoặc $0 (myserver/Kaggle) |
| 5 | Data F2–F4 → FC v3 → M3 → **nộp lần 3** | ~6 h | ~$12 |

Giai đoạn 1–3 nằm trong ngân sách hiện có. Giai đoạn 4–5 cần **~$24 Modal** hoặc dùng myserver/Kaggle.

## Cần chủ repo quyết

1. Hạn chót thật và số lượt nộp leaderboard còn lại.
2. Có cho phép dùng myserver để train ASR v3 / FC v3 không (tiết kiệm ~$24).
3. Ngân sách Modal thêm (nếu không dùng myserver).
4. Có được dùng corpus tiếng Việt bên ngoài cho ASR không (theo luật BTC).
5. FC v3: train tiếp từ adapter `fmtA_48e` (rẻ) hay train từ đầu.


## Ghi nhận thêm 11/11 (ASR)

- **Qwen3-ASR-0.6B** (0.94B tham số) trên 405 đoạn giữ riêng: zero-shot WER 32.3% / khớp thực thể 54.5% (có ngữ cảnh), kém cả PhoWhisper-medium gốc (25.8%). Fine-tune 1 epoch (audio train + TTS, Kaggle 2×T4): WER 13.2% / 65.5% (có ngữ cảnh), 15.7% không ngữ cảnh → ngữ cảnh giúp ~2.5 điểm WER nhưng vẫn xa v1 large (6.23% / 80.0%). Đang chạy 6 epoch (lr 5e-5, đã sửa lỗi cộng dồn loss) để kết luận. Transcript: `agent/runs/asr/heldout.q3asr06*.asr.json`; so bằng `agent/asr_compare.py`.
- **N3 cho thấy model lớn quan trọng hơn data mới**: v1 large (data cũ) 62.04 > v2 medium (TTS + aug + prompt V2.1) 59.
- **L2** (Kaggle, đang chạy): v1 large `e46f` + 2 epoch data v2.1 (TTS, aug, prompt V2.1, CTC/inter-CTC/KWS). L1 (large gốc + v2.1, 4 epoch) đã dừng; chạy trên Modal A100-80GB/H100 ước $7–8.
- Sửa `asr_train.py`: inter-CTC tính layer_norm ở fp32 (lỗi Float/Half khi `--base-fp16` trên T4).
- **Metric chẩn đoán mới** (`python -m metrics calls`, `python -m metrics asr-detail`) và kết quả phân tích: xem `plans/PLAN_72h_v3.md` §1b.
