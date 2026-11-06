# LifeMate — Zalo AI Challenge 2026

Kho dữ liệu hiện có trong repository phục vụ cuộc thi LifeMate. Đây là bài toán trợ lý giọng nói tiếng Việt: hiểu yêu cầu của người dùng, chọn và gọi công cụ trong các ứng dụng mô phỏng, rồi tạo đúng trạng thái ứng dụng sau khi thực thi.

## Dữ liệu hiện có

Gói dữ liệu nằm tại `data/train+public_test/`.

| Phần dữ liệu | Số mẫu | Bản chép lời | Nhãn |
| --- | ---: | --- | --- |
| `train` | 1.545 | Có `audio_text` cho yêu cầu cuối | Có `tool_calls`, `tool_results` và `final_state` |
| `public_test` | 634 | Không | Không; có `sample_submission.jsonl` với 3 dòng ví dụ |
| `private_test` | Chưa có trong gói hiện tại | — | Theo tài liệu cuộc thi, được cung cấp sau khi hoàn thiện giải pháp Docker cuối cùng |

Các file WAV chứa yêu cầu cuối và những lượt user trước đó trong lịch sử. Theo tài liệu cuộc thi, audio là tiếng Việt, định dạng WAV 16 kHz, mono, 16-bit, được ghi âm bởi người nói với độ tuổi, giới tính và vùng miền khác nhau; một số yêu cầu có thể xen từ tiếng Anh thông dụng.

Một vài thống kê trực tiếp từ gói hiện tại:

- Có 1.018/1.545 mẫu train và 419/634 mẫu public test có lịch sử hội thoại.
- Catalog train có 46 ứng dụng, 147 công cụ; catalog public test có 36 ứng dụng, 122 công cụ. Hai catalog có 25 ứng dụng chung. Public test có thêm 11 ứng dụng không có trong catalog train: `ecommerce`, `event_tickets`, `grocery`, `habit_tracker`, `pantry`, `parking`, `public_transit`, `recipes`, `stocks`, `train_bus_tickets` và `translator`.
- Trong train, 217 mẫu có gọi `sys.ask_missing_info`, 111 mẫu có `tool_calls` rỗng, và 791 mẫu có chuỗi lệnh gọi qua nhiều ứng dụng.

## Cấu trúc thư mục

```text
data/train+public_test/
├── VERSION
├── README.md
├── train/
│   ├── train.jsonl
│   ├── train_tools_catalog.json
│   ├── train_environment/<id>/state.json
│   ├── audio/<id>/
│   └── zaic_tools/
└── public_test/
    ├── public_test.jsonl
    ├── public_test_tools_catalog.json
    ├── public_test_environment/<id>/state.json
    ├── sample_submission.jsonl
    ├── audio/<id>/
    └── zaic_tools/
```

## Một mẫu dữ liệu gồm những gì?

Mỗi dòng trong `<split>.jsonl` là một đối tượng JSON:

- `id`: mã mẫu, đồng thời là tên thư mục môi trường của mẫu đó.
- `audio`: đường dẫn tới WAV của yêu cầu cuối cần xử lý.
- `context`: thời gian hiện tại, múi giờ, vị trí, hồ sơ người dùng và `installed_apps`.
- `history`: các lượt hội thoại trước. Lượt user trỏ tới file audio; lượt assistant có thể có lời thoại, tool call và kết quả tool.
- `audio_text` và `label`: chỉ có trong train. `audio_text` là bản chép lời yêu cầu cuối. Nhãn gồm chuỗi tool call chuẩn, kết quả tương ứng và `final_state` của các ứng dụng đã bị thay đổi.

Chỉ được gọi những công cụ được liệt kê trong `context.installed_apps` của chính mẫu đó, cùng với `sys.ask_missing_info`. Vì vậy, việc một công cụ có trong catalog tổng không có nghĩa là công cụ đó khả dụng ở mọi mẫu.

`<split>_tools_catalog.json` mô tả ứng dụng, công cụ, tham số và kết quả. Loại công cụ gồm `read` (đọc dữ liệu), `write` (thay đổi dữ liệu) và `meta` (công cụ hệ thống như `sys.ask_missing_info`). Schema có thể dùng thêm `x-ref` để yêu cầu ID hợp lệ từ dữ liệu ứng dụng, `x-match: "fuzzy"` cho văn bản được so khớp gần đúng và `x-one-of` để yêu cầu ít nhất một lựa chọn trong nhóm tham số.

Mỗi mẫu có một file `state.json` chứa dữ liệu ban đầu của các ứng dụng tại thời điểm yêu cầu. Các hành động trong `history` đã được áp dụng vào trạng thái này. Theo quy định dữ liệu cuộc thi, agent/giải pháp không được đọc, sao chép hoặc phân tích trực tiếp các file trong `<split>_environment/`; cần truy cập dữ liệu qua thư viện `zaic_tools`.

## Những dạng tác vụ trong dữ liệu

Dữ liệu có các yêu cầu tra cứu và thao tác như xem thời tiết, lịch hoặc số dư; tạo nhắc việc; đổi lịch; gửi tin nhắn; thanh toán; đặt dịch vụ; và điều khiển thiết bị nhà thông minh. Nhiều mẫu cần kết hợp nhiều app, giải quyết tên người qua danh bạ, dùng lịch sử hội thoại, hoặc suy luận điều kiện và thời gian.

Ví dụ, mẫu `train_0001` yêu cầu nếu ngày mai mưa thì dời lịch chạy bộ một giờ và nhắn giờ mới cho sếp; nếu không mưa thì giữ nguyên. Chuỗi nhãn tra cứu dự báo, tìm và cập nhật sự kiện lịch, tra danh bạ rồi gửi SMS. Một số mẫu cần hỏi lại thông tin thiếu; một số khác không cần gọi tool, chẳng hạn lời chào, yêu cầu đã được hủy hoặc yêu cầu ngoài khả năng các ứng dụng có sẵn.

## Chạy công cụ mô phỏng

`zaic_tools` là thư viện Python dùng để kiểm tra schema, chạy công cụ và cập nhật trạng thái trong bộ nhớ. Mỗi mẫu phải dùng đúng một `ToolEnv`; mọi lệnh gọi trên môi trường đó đều được ghi vào kết quả, kể cả lệnh gọi thất bại. Không tạo lại môi trường để thử lại từ trạng thái ban đầu.

Chạy từ thư mục `data/train+public_test/`:

```python
from pathlib import Path
import sys

ROOT = Path(".")
SPLIT = "public_test"
sys.path.insert(0, str(ROOT / SPLIT))

from zaic_tools import ToolEnv

env = ToolEnv(SPLIT, "public_0001", root=ROOT)

# Kiểm tra schema, không thực thi lệnh.
env.validate("smart_home.light_set", {"power": "off"})

# Thực thi lệnh và nhận kết quả.
result = env.call("smart_home.get_status", {})

# Các giá trị dùng để tạo dòng nộp.
tool_calls = env.get_tool_calls()
state_hash = env.get_state_hash()
```

`env.call()` trả về kết quả thành công (`ok: true`) hoặc lỗi (`ok: false`). Các lệnh bị từ chối vẫn nằm trong nhật ký tool call, nên cần tránh gọi thử những lệnh không muốn đưa vào bài nộp.

## Định dạng bài nộp

File `submission.jsonl` phải là UTF-8 JSON Lines, có đúng một dòng cho mỗi mẫu trong split. Mỗi dòng gồm `id`, `tool_calls` và `state_hash`; không cần nộp câu trả lời văn bản:

```json
{"id":"public_0001","tool_calls":[],"state_hash":"..."}
```

`tool_calls` là toàn bộ lệnh đã gọi trên `ToolEnv` của mẫu; nếu không gọi công cụ nào thì dùng `[]`. `state_hash` lấy từ `env.get_state_hash()`. Xem `data/train+public_test/public_test/sample_submission.jsonl` để biết ví dụ định dạng. Hướng dẫn chi tiết và các quy tắc còn lại nằm trong `data/train+public_test/README.md`.

## Cách tính điểm

Mỗi mẫu được chấm theo trạng thái ứng dụng cuối cùng và mức độ khớp giữa các tool call đã nộp với danh sách chuẩn (`gold`). Điểm mẫu nằm trong khoảng 0–1; điểm tổng là trung bình điểm của `N` mẫu nhân 100.

### Khớp trạng thái

`state_hash` đã nộp phải trùng chính xác với hash chuẩn. Nếu không trùng, điểm của mẫu bằng 0, bất kể các tool call có khớp hay không. Hash phải được lấy từ `env.get_state_hash()` của phiên bản `zaic_tools` tương ứng. Ban tổ chức phát lại các tool call theo thứ tự đã nộp; nếu kết quả phát lại khác, kết quả của ban tổ chức được dùng làm chuẩn cuối cùng.

### Khớp tool call

Tool call đã nộp khớp với một tool call chuẩn khi tên tool giống nhau và:

- Mọi tham số không fuzzy có giá trị giống hệt nhau.
- Với mỗi tham số fuzzy, token-F1 phải **lớn hơn 0,6** sau khi chuyển về chữ thường và chuẩn hóa Unicode NFC.

Mỗi tool call chỉ được ghép với tối đa một tool call chuẩn; thứ tự các lệnh không ảnh hưởng đến việc ghép. `sys.ask_missing_info` được chấm như các tool khác. Tool call không khớp, bao gồm cả call bị từ chối do lỗi schema, đều được tính là call chưa khớp.

Với mẫu `i`, ký hiệu `Mᵢ` là số call đã ghép được, `Uᵢ` là số call đã nộp nhưng không ghép được, và `|Gᵢ|` là số call chuẩn. Khi danh sách chuẩn không rỗng:

```text
Pᵢ = Mᵢ / (Mᵢ + Uᵢ)       # precision
Rᵢ = Mᵢ / |Gᵢ|             # recall
F1ᵢ = 2 × Pᵢ × Rᵢ / (Pᵢ + Rᵢ)
```

Nếu không có call nào khớp thì `F1ᵢ = 0`. Gọi `Cᵢ` là điểm khớp tool call: khi danh sách chuẩn không rỗng, `Cᵢ = F1ᵢ`; khi danh sách chuẩn rỗng (mẫu ngoài phạm vi), `Cᵢ = 1` nếu danh sách đã nộp cũng rỗng, ngược lại `Cᵢ = 0`. Điểm mẫu vẫn phải thỏa điều kiện hash trạng thái ở trên; điểm tổng được tính như sau:

```text
Điểm mẫuᵢ = 1[state_hash đã nộp = state_hash chuẩn] × Cᵢ
Điểm tổng = (100 / N) × Σᵢ Điểm mẫuᵢ
```

Đã đối chiếu với [ảnh công thức chính thức](metrics/reference/official_formula.png) từ mục Evaluation trên [trang LifeMate](https://challenge.zalo.ai/portal/lifemate). Bộ chấm local, xác minh hash qua `zaic_tools`, xử lý runtime và tie-break nằm trong [metrics/](metrics/README.md).

## Lưu ý về dữ liệu

Theo tài liệu gói dữ liệu, dữ liệu này chỉ dành cho các đội đã đăng ký Zalo AI Challenge 2026; không chia sẻ hoặc phát tán ra ngoài phạm vi cuộc thi.
