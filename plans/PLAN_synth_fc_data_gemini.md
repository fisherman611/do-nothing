# Plan sinh dữ liệu FC: tool đồng nghĩa (alias), biến thể giao diện, app mới tự tạo, và học tương phản (Gemini teamwork)

Cập nhật: 2026-11-09 (viết lại). Đi kèm `PLAN_phonetic_augment_gemini.md`, `PLAN_FC_clef_qwen35.md`. Người thực hiện: nhóm agent Gemini. Người duyệt: chủ repo.

## 0. Mục tiêu

Model FC (Qwen3.5-4B, định dạng Clef-token) cần chọn tool theo **ý nghĩa** (mô tả, tham số), không theo tên đã thuộc lòng.

- Hiện trạng (Clef P2, 1 epoch): val_in 78.4, val_ood 65.0.
- Trên val_ood, sai tool là 57/259 mẫu.
- Train có 46 app, 147 tool.

Ba phần:

1. **Phần A, tool train:**
   - **A1, alias:** mỗi tool train sinh ra vài tool cùng nghĩa nhưng khác tên (tên app, tên tool, tên tham số, mô tả viết lại). Mẫu giữ nguyên. Nhãn giữ nguyên về nghĩa, chỉ đổi tên theo bảng ánh xạ.
   - **A2, biến thể giao diện (mục 2.4):** tạo tool mới bằng cách gộp, đổi `enum`, thêm hoặc bớt tham số tuỳ chọn (không tách tool, vì dùng lại hội thoại gốc). Mẫu giữ nguyên, nhãn biến đổi bằng luật cố định.
2. **Phần B, app mới (tool chưa có trong train):**
   - **B1, tương tự tool test:** **tham khảo** catalog tool của public_test để biết loại chức năng, rồi tự tạo tool **tương tự nhưng không dùng y nguyên** (khác tên, mô tả viết lại, khác giao diện).
   - **B2, app tự nghĩ:** app ở lĩnh vực train chưa có, từ hiểu biết chung.
   - Cả hai: tự viết app chạy được, rồi sinh mẫu tổng hợp theo mục 4.
3. **Phần C, học tương phản:** cách dùng các tool alias và tool dễ nhầm làm cặp dương / âm khi train (mục 5).

## 1. Luật (bắt buộc)

| Nguồn | Dùng được? | Ghi chú |
|---|---|---|
| `train/train.jsonl`, `train_tools_catalog.json`, `train/zaic_tools/` (code) | Có | Nguồn chính của phần A |
| `public_test/public_test_tools_catalog.json` (định nghĩa tool) | **Chỉ để tham khảo ở B1** | Đọc để biết có loại chức năng gì, rồi tạo tool tương tự. **Không dùng y nguyên:** không copy tên app, tên tool, tên tham số, câu mô tả hay schema. Không đưa file này hay nội dung của nó vào data train. Chủ repo đã quyết định dùng theo cách này (2026-11-09). |
| Phần còn lại của `public_test/` và `private_test/`: code app, `*.jsonl`, environment, audio | **Không** | Không mở. `private_test/` không đọc gì cả, kể cả catalog. |
| Mọi `*_environment/`, kể cả của train | **Không** | AGENT.md dòng 24. State của phần B do code sinh ra; kết quả tool của train lấy từ `label.tool_results`. |
| Transcript hay nhãn ASR của audio public_test (kể cả do đội tự tạo) | **Không** | AGENT.md dòng 25, 27. Không dùng kể cả để tham khảo. |

Hệ quả cho đánh giá:

- Sau B1, **điểm public test không còn đo sạch được khả năng generalize với app lạ**, vì train đã có tool tương tự. Báo điểm public test tách riêng, và không dùng nó để chọn gì.
- Thước đo sạch là `val_ood` (5 app train bị giữ lại: alarm, ewallet, flights, loyalty, podcast_radio) và `val_alias` (mục 6).
- Mọi quyết định chọn checkpoint và siêu tham số **không được** dựa trên public test.

## 2. Phần A: alias cho tool train

### 2.1 Phạm vi

- **Tool cần phủ:** mọi tool của các app **không** thuộc `held_out_apps` (`asr/agent/clef/splits.json`), khoảng 127 tool trên 41 app.
- **Không tạo alias cho 5 app giữ lại,** nếu không thì val_ood bị rò rỉ.
- **Mỗi tool: 2–4 alias.** Mỗi alias là một bộ đổi tên đầy đủ (mục 2.2).
- **Chỉ augment mẫu thuộc `splits.json["train"]`.**
- **Mỗi tool có ít nhất `min(6, số mẫu có gọi tool đó)` mẫu được augment.** Tổng cộng khoảng 1.0–1.5 lần số mẫu train (khoảng 1200–1700 mẫu mới). Không cần augment hết.
- Tool chưa có mẫu nào trong split train (ví dụ `contacts.get`) chỉ có alias trong catalog, xuất hiện như tool được cài. Không bịa mẫu cho chúng ở phần A.

### 2.2 Một alias gồm những gì

Ghi vào `synth/alias_map.json`:

```json
{"alias_id": "sms@a2",
 "app": {"from": "sms", "to": "tin_nhan"},
 "tools": {"sms.send": {"to": "tin_nhan.gui", "params": {"contact_id": "nguoi_nhan_id", "content": "noi_dung", "phone_number": "so_dien_thoai"}},
           "sms.read_recent": {"to": "tin_nhan.doc_gan_day", "params": {}}},
 "descriptions": {"tin_nhan.gui": "Gửi tin nhắn SMS tới liên hệ hoặc số điện thoại", "...": "..."}}
```

Quy tắc:

1. **Đổi tên theo cả app.** Các tool của cùng một app dùng chung tiền tố mới.
   - `x-ref` có dạng `app.collection` và trỏ tới **app được tham chiếu**, không phải app của tool. Ví dụ `sms.send.contact_id` có `"x-ref": "contacts.contacts"`.
   - Chỉ đổi tiền tố trong `x-ref` khi **app được tham chiếu** cũng được alias trong mẫu đó (`contacts` → `danh_ba` thì thành `"danh_ba.contacts"`). Phần `collection` giữ nguyên.
   - Câu "lấy từ contacts.search" trong mô tả tham số cũng đổi theo tên alias.
   - Ví dụ chạy được: `synth/examples/alias_examples.jsonl`.
2. **Kiểu tên đa dạng,** trộn đều giữa các alias:
   - đồng nghĩa tiếng Anh: `messaging.send_text`;
   - tiếng Việt không dấu: `tin_nhan.gui`;
   - camelCase hoặc viết tắt: `msg.sendSms`;
   - **Mọi tên phải có nghĩa,** đọc tên là đoán được tool làm gì: `people.findPerson`, `textMsg.sendNow`, tham số `personId`, `body`. **Không dùng tên vô nghĩa** như `app_k3.fn_07` hay `p1`, `p2`.
   - Tên nên **khác hẳn tên gốc về từ ngữ** (không chỉ đổi kiểu viết), để model phải hiểu nghĩa chứ không khớp chuỗi. Ví dụ `sms.send` → `textMsg.sendNow`, không phải `Sms.Send`.
3. **Mô tả tool viết lại cùng nghĩa** (không copy câu gốc), tiếng Việt **có dấu** (không viết không dấu).
   - **Mô tả tham số giữ nguyên phần luật của mô tả gốc**: định dạng ("ISO-8601 đầy đủ giây và offset múi giờ, ví dụ …", "Số nguyên, không dấu chấm hay phẩy", "giữ nguyên chuỗi"), luật mặc định ("'nhà' → home_address"), ý nghĩa từng giá trị enum, nguồn lấy id ("lấy từ <tên tool alias>"). Không tự đặt định dạng khác (vd "YYYY-MM-DD HH:MM") vì sẽ ngược với nhãn.
   - Mô tả liệt kê giá trị enum phải đúng y tập `enum`.
   - Tên app/tool mồi (mục 5.2) phải tự nhiên như alias, **không** chứa chữ "decoy"; `val_alias` dùng tên mồi riêng.
   - Bản sửa tự động: `synth/fix_synth.py` (2026-11-09).
4. **Không đổi:**
   - kiểu dữ liệu, `enum` (giá trị), `required`, `minimum`/`maximum`;
   - nội dung kết quả tool (`tool_results`), kể cả tên khoá trong kết quả. Như vậy `ref_candidates` (khoá `<số ít>_id`, khoá collection) vẫn chạy.
5. Tên mới không được trùng tên của bất kỳ tool train, tool alias khác hay tool test nào (kể cả tên tham số).

### 2.3 Tạo mẫu alias

Với mỗi mẫu được chọn:

1. **Chọn alias cho 1–3 app trong `installed_apps` của mẫu,** ưu tiên app có trong nhãn. Các app còn lại giữ tên gốc. Như vậy model gặp cả tên gốc lẫn tên mới trong cùng một prompt.
2. **Ánh xạ tên** trong `context.installed_apps`, trong `history[*].tool_calls` của assistant và trong `label.tool_calls`: tên tool và **tên khoá tham số**, theo `alias_map.json`. Giá trị tham số giữ nguyên.
3. **`audio_text`, `tool_results`, `final_state` giữ nguyên.**
4. Ghi:
   - `id = <id gốc>#al<k>`;
   - `aug = {"alias": ["sms@a2", ...]}`;
   - `catalog_overlay`: các tool alias mà mẫu cần. `build_data.py` sẽ ghép phần này vào catalog khi dựng prompt.

**Ràng buộc quan trọng: không bao giờ cài hai alias của cùng một tool (hoặc alias cùng tool gốc) trong một mẫu.** Nếu cài cùng lúc thì có hai đáp án đúng ngang nhau và nhãn trở nên mơ hồ. Trường hợp tool gần nghĩa nhưng **khác** nghĩa thì xem mục 5.2.

### 2.4 Phần A2: biến thể giao diện cho tool train (tạo tool mới)

Mục 2.1–2.3 (A1) chỉ đổi tên. A2 tạo **tool mới có giao diện khác** nhưng vẫn làm được đúng việc của tool gốc. Câu lệnh của mẫu giữ nguyên. **Nhãn được biến đổi bằng luật cố định**, Gemini không viết tay nhãn. Mục đích là để model học đọc schema thật sự (tham số nào, `enum` nào), chứ không chỉ chịu được việc đổi tên.

**Không tách một tool thành nhiều tool ở A2.** A2 dùng lại nguyên hội thoại của mẫu gốc, mà hội thoại đó được viết cho một tool. Thiết kế nhiều tool nhỏ trong cùng app chỉ làm ở Phần B, nơi hội thoại được viết mới cho đúng giao diện.

**Bốn kiểu biến đổi:**

| Kiểu | Ví dụ với tool train | Luật biến đổi nhãn |
|---|---|---|
| **Gộp tool** | `reminders.create` + `reminders.complete` + `reminders.delete` → `nhac_viec.xu_ly(thao_tac: tao\|xong\|xoa, …)` | thêm `thao_tac` theo tool gốc, các tham số còn lại đổi tên |
| **Đổi giá trị `enum`** | `fuel_price.get_price(type: ron95\|e5\|diesel)` → `giaXang.traCuu(loai: xang_95\|xang_e5\|dau_do)` | ánh xạ giá trị 1–1, áp cho cả tham số nhãn **và** cùng khoá trong `tool_results` |
| **Thêm tham số tuỳ chọn** | `smart_home.light_set` thêm `transition_sec?`, `room_name?` | nhãn không đổi (chỉ đổi tên); dạy model **không điền** tham số tuỳ chọn khi người dùng không nói |
| **Bớt tham số tuỳ chọn** | `reminders.create` bỏ `repeat` | chỉ dùng cho mẫu mà nhãn **không** dùng tham số đó |

**Quy tắc:**

1. **Tên luôn có nghĩa** và khác tên gốc về từ ngữ, như mục 2.2.2. Mô tả viết lại. Với kiểu gộp, mô tả nói rõ từng giá trị của tham số thao tác làm gì.
2. **Mỗi biến thể là một bản ghi trong `synth/iface_map.json`,** gồm `kind` (merge / enum / add_opt / drop_opt), schema tool mới và luật ánh xạ khai báo được. Ví dụ:
   ```json
   {"iface_id": "fuel_price@e1", "kind": "enum", "from": "fuel_price.get_price",
    "to": {"name": "giaXang.traCuu", "params": {"type": "loai"},
           "value_map": {"type": {"ron95": "xang_95", "e5": "xang_e5", "diesel": "dau_do"}}}}
   ```
   Luật phải viết được bằng `has` / `equals` / `value_map`. **Không dùng luật cần suy luận.**
3. **Script áp luật:** `synth/iface_apply.py` có hàm `forward` (gốc → biến thể) và `inverse` (biến thể → gốc). Áp cho `label.tool_calls`, `history[*].tool_calls` và (với kiểu `enum`) các khoá tương ứng trong `tool_results`. Gemini viết `iface_map.json`; mẫu do script sinh, không viết tay.
4. **Không đổi:** câu nói, `final_state`, tên khoá trong `tool_results` (để `ref_candidates` vẫn chạy).
5. **Không cài chung** tool gốc với biến thể A2 của nó, hay hai biến thể của cùng tool gốc trong một mẫu (cùng ràng buộc như mục 2.3). Với kiểu gộp, tool gộp thay cho **cả nhóm** tool gốc; không cài lẫn tool gốc nào của nhóm.
6. **Không làm cho 5 app giữ lại.**
7. **Phạm vi:** khoảng 30–40 tool có cấu trúc phù hợp (có `x-one-of`, có `enum` hành động, hoặc có nhóm tool cùng app), mỗi tool 1–2 biến thể. Mỗi biến thể có `min(6, số mẫu)` mẫu, tổng khoảng 400–700 mẫu. `id = <id gốc>#if<k>`, `aug = {"iface": [...], "pair_id": <id gốc>}`, cùng `catalog_overlay`.
8. **Phiên âm:** chạy `synth_phonetic/augment.py` trên `train_iface.jsonl` như mục 4.5, kể cả history.
9. Consistency loss (5.3) chỉ áp cho A1 (ánh xạ 1–1). A2 chỉ ghi `pair_id` để phân tích.

## 3. Phần B: app mới (B1 tương tự tool test, B2 tự nghĩ)

### 3.1 B1: tương tự tool test, không dùng y nguyên

1. **Đọc `public_test_tools_catalog.json`** để lập danh sách **loại chức năng** test có mà train chưa có, ví dụ "tra cứu X", "đặt Y". Chỉ ghi lại chức năng bằng lời của mình, không chép schema.
2. **Mỗi chức năng: 1–2 tool tương tự,** tự thiết kế từ đầu. Mỗi tool phải thoả cả bốn điều kiện:
   - **Tên khác về từ ngữ:** quy tắc như mục 2.2.2. Không dùng tên app, tên tool, tên tham số của test.
   - **Mô tả tự viết,** không dịch hay diễn đạt lại từng câu của test.
   - **Giao diện khác ít nhất ở một điểm:** thêm hoặc bớt tham số, đổi kiểu hay `enum`, đổi cấu trúc kết quả trả về, hoặc chia thao tác thành tool khác cách test chia.
   - **Không phải tool test đổi tên:** giám khảo LLM so tool B1 với tool test tương ứng, chấm "chép" thì làm lại.
   - Ghi `synth/testlike_map.json` (tool B1 → loại chức năng đã tham khảo) **chỉ để audit**.
3. Viết app, sinh state và mẫu như B2 (mục 3.2, bước 3–5). Hội thoại viết mới hoàn toàn, không dựa vào `public_test.jsonl`.
4. **Tách một nửa số app B1 làm `val_testlike`.** Chỉ theo dõi, không chọn checkpoint.

### 3.2 B2: app tự nghĩ

1. **Chọn lĩnh vực** từ hiểu biết chung về trợ lý ảo trên điện thoại ở Việt Nam. B2 **không dựa vào catalog test**; lĩnh vực nào đã có ở B1 thì bỏ khỏi B2. Ưu tiên lĩnh vực train **chưa có**, ví dụ: đặt lịch cắt tóc/spa, tra điểm thi, nộp tiền điện nước, đặt sân thể thao, tra cứu bảo hiểm, gửi hàng chuyển phát, đặt bàn nhà hàng, đăng ký khám bệnh. **Không làm app trùng lĩnh vực 5 app giữ lại** (báo thức, ví điện tử, vé máy bay, điểm thành viên, podcast/radio), nếu không `val_ood` bị rò rỉ. Lập danh sách khoảng 15–25 app, mỗi app 2–5 tool, ghi vào `synth/novel_apps.json`. Chủ repo duyệt danh sách trước khi viết code.
2. **Thiết kế tool** theo đúng định dạng catalog train (`name`, `kind`, `description`, `parameters` có `type`, `enum`, `required`, `x-one-of`, `x-ref`, `returns`):
   - tên có nghĩa, theo các kiểu ở mục 2.2.2 (trộn tiếng Anh, tiếng Việt không dấu, camelCase);
   - mỗi app có cả tool đọc (tìm, xem) và tool ghi (đặt, huỷ, sửa); tool ghi lấy id từ tool đọc qua `x-ref`, giống `contacts.search` → `sms.send`;
   - có app thiết kế nhiều tool nhỏ cho các thao tác gần nhau (ví dụ `datLich` / `doiLich` / `huyLich`), có app gộp thành một tool với tham số thao tác. Hội thoại phần B viết mới nên dùng được cả hai kiểu;
   - có chỗ dễ nhầm với app train (ví dụ "đặt sân" với `calendar.create_event`) để model phải đọc mô tả.
3. **Viết app chạy được** trong `synth/zaic_tools/apps/<app>.py`, theo khung `App` của train (`apps/base.py`). Có unit test.
4. **Sinh state và mẫu tổng hợp** theo mục 4. Mỗi tool 4–8 mẫu. Mỗi mẫu cài thêm app train (alias hoặc gốc) như mẫu thật.
5. **Tách một nửa số app mới làm `val_novel`:** không mẫu nào của các app này vào train. Dùng để đo generalize với app lạ, cùng `val_ood`.

## 4. Sinh mẫu tổng hợp (cho phần B)

Áp dụng cho tool chưa có mẫu thật.

1. **State ban đầu:** sinh bằng code có seed (`synth/gen_state.py`).
   - Mỗi collection 3–12 bản ghi, có mục gây nhiễu: tên gần giống nhau, hai mục cùng ngày.
   - Khoảng 10% mẫu có state làm yêu cầu không thực hiện được (hết chỗ, không tìm thấy), để tool trả `noop`.
   - Thời gian nhất quán với `context.current_time`.
2. **Câu lệnh và lịch sử:** Gemini viết dựa trên state và danh sách app đã cài. Câu phải tự nhiên như lời nói: có từ đệm, có vùng miền, có tự sửa lời, số viết lẫn chữ và chữ số như train. Không dùng tên tool hay tên tham số trong câu người dùng.
3. **Chuỗi tool call:** chạy từng call qua `ToolEnv` trên split `synth` để lấy `tool_results` thật. Ghi `final_state = env.get_state()`. Không được bịa kết quả tool.
4. **Phân phối mục tiêu:** lấy từ **train**, giữ sai lệch trong khoảng ±5 điểm phần trăm.

   | Thuộc tính | Train |
   |---|---|
   | Số call ở lượt cuối: 0 / 1 / 2 / 3 / 4 / ≥5 | 7 / 24 / 28 / 23 / 10 / 8% |
   | Có `sys.ask_missing_info` | 14% |
   | Số lượt user trong history: 0 / 1 / 2 / 3 / 4 / ≥5 | 34 / 19 / 17 / 9 / 5 / 16% |
   | Độ dài câu cuối | trung bình 22.5 từ (p10 13, p90 34) |
   | Mẫu có ≥ 1 từ nước ngoài (câu cuối hoặc history) | khoảng 49% |
   | Loại từ nước ngoài: english_word / brand / acronym / person / code / channel | 46 / 20 / 16 / 11 / 4 / 3% |

5. **Từ nước ngoài và biến thể phiên âm** (giống `PLAN_phonetic_augment_gemini.md`, áp cho cả câu cuối lẫn history):
   - **Khi viết mẫu:** chèn từ nước ngoài theo tỉ lệ ở bảng trên, viết bằng **chữ chuẩn** (SJC, Peter Hansen, check-in, VTV6). State cũng phải có tên nước ngoài: danh bạ, địa điểm, thương hiệu, mã sản phẩm. Nhãn dùng đúng chữ chuẩn đó.
   - **Từ điển:** từ mới chưa có trong `synth_phonetic/terms.json` ghi vào `synth/terms_novel.json`, cùng định dạng. Phiên âm theo mục 4.1 của plan phonetic (phiên âm chuẩn, 1–2 cách). Không dùng từ đồng âm với từ Việt khác nghĩa, ví dụ share → "xe", full → "phun".
   - **Sinh biến thể:** chạy `synth_phonetic/augment.py` trên `train_novel.jsonl`, dùng từ điển gộp và catalog đã ghép overlay. Cần thêm tuỳ chọn `--ids-from-input` để bỏ kiểm tra `splits.json` và số 1158, vì đây là data tổng hợp. Kết quả:
     - `#ph1`: phiên âm mọi từ ở câu cuối, mỗi từ ở history với xác suất 0.3. Nếu câu cuối không có từ nước ngoài thì phiên âm hết history.
     - `#ph2`: chỉ phiên âm 1–2 từ, giữ chữ gốc ở các từ còn lại.
     - Lọc theo mục 5 của plan phonetic: brand hoặc tên người mà nhãn cần chữ gốc nhưng chữ gốc không thấy ở đâu model nhìn được (context, catalog, history, kết quả tool trước đó) thì **không phiên âm**. Luật này xét cả nhãn chính lẫn từng lượt history.
   - **Đầu ra:** `train_novel_phonetic.jsonl`. `build_data.py --history --phonetic` tự tách lượt history bị phiên âm thành mẫu phụ.
   - **Mẫu thuộc `val_novel` / `val_testlike`:** biến thể phiên âm vào cùng tập đó theo `id` gốc, không vào train. Báo điểm riêng cho bản gốc và bản phiên âm.

## 5. Phần C: học tương phản

### 5.1 Clef đã là tương phản sẵn, cái cần thêm là âm tính khó

Loss của Clef là softmax **giới hạn vào các phương án** của câu hỏi (Q1: tool nào; Q2: id nào, enum nào). Như vậy nó đã là dạng InfoNCE: kéo xác suất phương án đúng lên so với các phương án khác trong cùng tập. Độ mạnh của nó phụ thuộc vào **các phương án sai có khó không**. Phần A và B cung cấp đúng thứ đó. Thêm ba loại tín hiệu:

### 5.2 Âm tính khó: tool dễ nhầm (khác nghĩa, tên hoặc mô tả gần giống)

- **Tạo tool mồi:** với mỗi tool trong nhãn, tạo 1–2 tool mồi **khác nghĩa thật**, và cài tool mồi vào mẫu như một app bình thường.
  - Ví dụ: `sms.send` (gửi ngay) với `sms.schedule` (hẹn giờ gửi); `banking.transfer` với `banking.request_money`; `calendar.cancel_event` với `calendar.decline_invite`.
  - Tool mồi cũng phải chạy được (thêm vào app tổng hợp `synth_decoy_*`). Nếu model chọn nhầm khi chạy agent, kết quả vẫn hợp lệ.
- **Cặp tool có sẵn giữa các app:** tận dụng các cặp như `sms` / `chat_ott`, `banking` / ví điện tử tổng hợp, `reminders` / `calendar`.
  - Cài cả hai trong một mẫu.
  - Câu lệnh phải có chi tiết quyết định, ví dụ "nhắn Zalo" hay "nhắn tin thường".
  - Một tỉ lệ mẫu chỉ cài một trong hai, để model không mặc định luôn chọn một bên.
- **Nhãn:** nhãn không đổi, chỉ thêm phương án sai khó vào Q1.
  - Cách làm: mục 2.3 bước 1, cộng thêm app mồi vào `installed_apps`.
  - Mỗi mẫu tối đa 2 app mồi, để không đổi quá nhiều phân phối số app đã cài (trung bình 9.2, khoảng 3–18).

### 5.3 Dương tính: nhất quán giữa các alias (consistency loss)

- **Ghép cặp:** mẫu gốc và mẫu alias của nó (cùng `id` gốc, cùng `step`) là cặp dương. Hai phương án Q1 tương ứng với nhau qua `alias_map` (cùng tool gốc) phải có cùng xác suất.
- **Loss thêm:**
  - KL đối xứng giữa hai phân phối Q1 (và Q2 nếu có) sau khi **ánh xạ phương án về tool gốc**: `λ · symKL(p_gốc, p_alias)`.
  - Chỉ áp dụng cho khoảng 20–30% mẫu alias, để không tăng gấp đôi chi phí.
  - Bắt đầu với `λ = 0.5`.
- **Cần sửa code:**
  - `build_data.py` ghi thêm `pair_id` và với mỗi phương án Q1 một `canon` (tên tool gốc).
  - `train.py` gom hai ví dụ cùng `pair_id` vào cùng bước và tính KL trên các phương án đã ánh xạ.
  - Agent Gemini **không** sửa `train.py`. Chỉ cần ghi đủ `pair_id` và `canon` trong data. Chủ repo hoặc Claude sẽ cài loss.

### 5.4 Biên (margin) giữa đáp án và tool mồi khó nhất (tuỳ chọn)

- **Loss:** với các mẫu có tool mồi (mục 5.2), thêm `max(0, m − (z_đúng − z_mồi_cao_nhất))` trên logit đã giới hạn, với `m ≈ 2`.
- **Khi nào thêm:** chỉ khi đánh giá ở mục 6 cho thấy model vẫn hay nhầm sang tool mồi.

### 5.5 Thứ tự thử

1. Chỉ data, tức phần A cộng 5.2, với loss Clef như cũ.
2. Thêm 5.3.
3. Thêm 5.4 nếu cần.

So từng bước trên `val_ood` và `val_alias`. Chỉ giữ thay đổi nào có cải thiện.

## 6. Tập đánh giá

| Tập | Tạo thế nào | Đo gì |
|---|---|---|
| `val_in` (128) | có sẵn | không được giảm |
| `val_ood` (259) | có sẵn, 5 app train giữ lại | generalize thật (thước đo chính) |
| `val_alias` | val_in với **alias mới** (bộ tên chưa dùng trong train) cho 1–3 app mỗi mẫu, kèm tool mồi | độ bền với đổi tên; tỉ lệ đổi đáp án giữa gốc và alias (thấp là tốt) |
| `val_novel` | một nửa số app B2 (không có mẫu nào của các app này trong train) | generalize với app lạ, bổ sung cho `val_ood` |
| `val_testlike` | một nửa số app B1 | chỉ theo dõi, không dùng để chọn checkpoint |

## 7. QA (tự động, bắt buộc)

Script `synth/qa.py`, báo cáo ghi vào `synth/reports/`:

1. **Ánh xạ ngược:** đổi mẫu alias về tên gốc theo `alias_map`. Kết quả phải trùng y hệt mẫu gốc (so hash của `label`, `history`, `context` sau khi ánh xạ ngược).
2. **Không mơ hồ:** không có mẫu nào cài hai tool cùng tool gốc (mục 2.3).
2b. **Phần A2:**
   - `inverse(forward(mẫu)) == mẫu gốc` (so hash `label`, `history`, `tool_results`);
   - mọi call trong nhãn mới khớp đúng một nhánh `when`;
   - với kiểu gộp, không còn tool gốc nào của nhóm được cài;
   - không có bản ghi `kind = split`;
   - với kiểu `drop_opt`, nhãn gốc không dùng tham số bị bỏ.
3. **Schema:**
   - mọi call trong nhãn qua `validate` với catalog đã ghép `catalog_overlay`;
   - `x-ref` của alias trỏ đúng tiền tố app mới.
4. **Tên không trùng:** giữa tool train, mọi alias, tool A2, tool mồi, tool phần B và **tool test** (kể cả tên tham số). Tool B1 không có câu mô tả nào trùng hay gần như chép câu mô tả của tool test (kiểm bằng so khớp chuỗi và giám khảo LLM).
5. **Mô tả:** giám khảo LLM (model khác với model sinh) xác nhận mô tả alias cùng nghĩa với gốc, và mô tả tool mồi khác nghĩa với tool đúng mà câu lệnh vẫn phân biệt được. Điểm ≥ 4/5.
6. **Phần B:**
   - unit test app pass;
   - replay mẫu khớp `state_hash`;
   - log của agent không có lần mở file nào trong `public_test/` ngoài `public_test_tools_catalog.json`, và không mở gì trong `private_test/`;
   - grep data train không thấy tên tool, tên tham số hay câu mô tả nào của catalog test;
   - `testlike_map.json` (tool B1 → loại chức năng test đã tham khảo) chỉ để audit, không nằm trong data train;
   - bản phiên âm (mục 4.5) qua `synth_phonetic/qa.py`, kiểm 6.1–6.4 và 6.6: nhãn, context và lượt assistant trùng y hệt bản gốc; mọi `to` có trong từ điển; âm tiết hợp lệ; không còn dòng `#ph` nào không thay gì;
   - tỉ lệ mẫu có từ nước ngoài và phân bố loại từ nằm trong ±5 điểm % so với bảng ở mục 4.4.
7. **Chống rò rỉ val:**
   - không có `id` nào thuộc `val_in` / `val_ood`;
   - không có alias nào cho 5 app giữ lại.
8. **Duyệt tay:** chủ repo duyệt 30 mẫu alias, 30 mẫu A2, 30 mẫu có tool mồi, 30 mẫu phần B.

## 8. Phân công nhóm Gemini

| Vai | Việc | Đầu ra |
|---|---|---|
| Alias designer | Mục 2.2: 2–4 alias cho khoảng 127 tool | `alias_map.json` |
| Interface designer | Mục 2.4: chọn 30–40 tool, viết `iface_map.json` + `iface_apply.py` (forward/inverse) + test | `iface_map.json`, `iface_apply.py` |
| Interface augmenter | Áp `iface_apply.py` lên mẫu train, ghi `catalog_overlay`, chạy phiên âm | `train_iface.jsonl`, `train_iface_phonetic.jsonl` |
| Decoy designer | Mục 5.2: tool mồi khác nghĩa + app `synth_decoy_*` chạy được + unit test | `decoy_map.json`, `apps/synth_decoy_*.py` |
| Sample augmenter | Mục 2.3 + gắn tool mồi + `pair_id`/`canon` (5.3) | `train_alias.jsonl` |
| Test-like designer | Mục 3.1: đọc catalog test lấy loại chức năng, tự thiết kế tool tương tự (không chép) + app + unit test | `testlike_map.json`, `apps/*.py`, test |
| Novel app designer | Mục 3.2: nghĩ app mới, viết schema + app chạy được + unit test | `novel_apps.json`, `apps/*.py`, test |
| Sample generator | Mục 4.1–4.4 cho phần B (có chèn từ nước ngoài) | `synth_environment/`, `train_novel.jsonl` |
| Phonetic augmenter | Mục 4.5: từ điển mới, chạy `augment.py`, lọc không suy ra được | `terms_novel.json`, `train_novel_phonetic.jsonl` |
| Verifier | Mục 7.1–7.4, 7.6, 7.7 | `reports/` |
| Reviewer (model khác) | Mục 7.5, review code app | điểm, nhận xét |

**Thứ tự:**

1. Phần A, đợt thử 50 mẫu, rồi QA và chủ repo duyệt.
2. Phần A chạy đủ, cộng tool mồi.
2b. Phần A2: đợt thử 4 tool (mỗi kiểu một tool), QA, chủ repo duyệt, rồi chạy đủ.
3. `val_alias`.
4. Train và đánh giá (5.5 bước 1).
5. Phần B (B1 và B2), rồi đến bản phiên âm của nó (mục 4.5).
6. Consistency loss (5.3).

## 9. Tiêu chí chấp nhận

- **Phủ tool:** 100% tool ngoài các app giữ lại có ≥ 2 alias. Mỗi tool có mẫu trong train được augment ≥ `min(6, số mẫu)` lần.
- **Phần A2:** đủ 4 kiểu biến đổi, 30–40 tool. QA 7.2b không có lỗi nào.
- **Code phía repo (không phải việc của Gemini):** `build_data.py` ghép `catalog_overlay` vào catalog khi dựng prompt. Hiện code chưa hỗ trợ, phải làm xong trước khi train với data A1/A2.
- **QA:** 7.1–7.4 và 7.7 không có lỗi nào. Mục 7.5 đạt ≥ 90%. Duyệt tay ≤ 2 lỗi trên mỗi nhóm 30 mẫu.
- **Kết quả train sau 5.5 bước 1:**
  - `val_alias` tăng so với không augment;
  - `val_in` không giảm quá 1 điểm;
  - báo cáo `val_ood`.

## 10. Rủi ro

- **Luật A2 sai làm nhãn sai hàng loạt.** Xử lý: luật chỉ dùng `has`/`equals`/`value_map`, kiểm `inverse(forward)`, đợt thử 4 tool trước khi chạy đủ.
- **Model học thuộc chính các alias.** Xử lý: `val_alias` dùng bộ tên chưa từng xuất hiện trong train. Tên alias trộn nhiều kiểu (đồng nghĩa tiếng Anh, tiếng Việt không dấu, camelCase) và luôn có nghĩa.
- **Tool mồi làm nhãn mơ hồ** khi tool mồi gần nghĩa quá. Xử lý: giám khảo LLM ở mục 7.5, và câu lệnh phải có chi tiết quyết định.
- **B1 làm điểm public test cao giả** và dẫn đến chọn sai hướng. Xử lý: không chọn bất cứ thứ gì theo public test (mục 1), báo điểm public test tách riêng, so sánh có và không có B1 trên `val_ood`.
- **Tool B1 vô tình gần như chép tool test.** Xử lý: QA 7.4, giám khảo LLM, chủ repo duyệt `testlike_map.json`.
- **App mới của phần B quá dễ hoặc quá giống app train,** nên không dạy được gì. Xử lý: ưu tiên lĩnh vực train chưa có, giám khảo LLM kiểm độ khác biệt, theo dõi `val_novel`.
- **Prompt dài hơn khi cài thêm app mồi,** nên train chậm hơn. Xử lý: tối đa 2 app mồi mỗi mẫu, theo dõi số token.
