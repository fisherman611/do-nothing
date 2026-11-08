# Tạo transcript tham chiếu bằng Alibaba Cloud

Runner local dùng `qwen-audio-3.1-asr-flash` (HTTP, đồng bộ) để tạo **nhãn giả phục vụ đánh giá ASR**. Không dùng nhãn test này để train/tuning, không tích hợp API vào Docker inference của cuộc thi. WER/CER so với các nhãn này chỉ thể hiện độ lệch so với Qwen, chưa phải điểm so với nhãn chuẩn do người xác minh.

Chạy từ thư mục gốc repo, PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-asr-cloud.txt
.\.venv\Scripts\python.exe -m asr.qwen_cloud_label --dry-run
.\.venv\Scripts\python.exe -m asr.qwen_cloud_label --limit 20
```

Lệnh cuối **gọi API và tiêu quota**, tạo 20 transcript đầu tiên. Kiểm tra `predictions.jsonl` (text, `usage`, lỗi nếu có) rồi bỏ `--limit` để chạy tiếp cùng lượt:

```powershell
.\.venv\Scripts\python.exe -m asr.qwen_cloud_label --token-budget 900000
```

Mặc định xử lý audio cuối và mọi audio user trong history: **1.933 WAV, khoảng 2,762 giờ**, gồm 634 yêu cầu cuối. Chỉ lấy audio cuối: thêm `--scope final` và một `--output-dir` riêng.

## Hotword và context

Mỗi request gửi kèm gợi ý để model nghe đúng tên riêng:

- **Hotword** (`parameters.vocabulary`): lấy từ `context.user_profile` (tên, địa chỉ tách theo dấu phẩy, thành phố) rồi đến state ban đầu của các app: tên liên hệ, nhóm chat, bài hát/ca sĩ, playlist, ngân hàng, quán, địa điểm, bến/tuyến, kênh TV, phòng, nhãn báo thức... Bỏ ID, số, ngày, email và chuỗi dài hơn 40 ký tự. Trên public_test trung bình khoảng 59 hotword/mẫu, tối đa 120. Mặc định trọng số 3 (`--hotword-weight`, 1–5), tối đa 300 từ/mẫu (`--max-hotwords`).
- State được đọc qua `zaic_tools.ToolEnv(...).get_state()` của chính split, không mở trực tiếp `*_environment/` (guard trong `metrics.io` vẫn giữ nguyên). Không gọi tool nào. State là trạng thái lúc gửi yêu cầu cuối, nên audio history cũng dùng chung danh sách này.
- **Context** (`input.messages`): tối đa 3 câu trả lời assistant ngay trước audio đó trong `history`, mỗi câu cắt còn 400 ký tự theo giới hạn API. Không gửi tool calls, tool results hoặc nhãn train.
- Tắt từng phần: `--hotwords profile` (chỉ profile), `--hotwords none`, `--context none`. Đổi bất kỳ tuỳ chọn nào thì dùng `--output-dir` mới.

Rủi ro: hotword mạnh có thể khiến model "nghe" ra tên có trong state dù người nói không nói. Nên đo trên **train** (có `audio_text` thật) trước, với `--input data/train+public_test/train/train.jsonl --limit 100`, so WER khi có và khi không có hotword/context.

Model có **chuẩn hoá văn bản sẵn** (bỏ từ đệm, lặp, tự sửa lời, chuẩn hoá dấu câu). Tài liệu không có tham số tắt tính năng này, nên transcript không hoàn toàn nguyên văn.

## Cấu hình key

Runner đọc `.env` bằng `python-dotenv`; biến môi trường của tiến trình được ưu tiên:

```dotenv
DASHSCOPE_API_KEY=<your-api-key>
API_HOST=<workspace-id>.ap-southeast-1.maas.aliyuncs.com
```

Cũng có thể lấy hostname từ `DASHSCOPE_URL` nếu không có `API_HOST`. Endpoint thực dùng là `https://<host>/api/v1/services/aigc/multimodal-generation/generation`. Không theo redirect để không gửi key sang host khác. `--dry-run` không cần key, không gọi mạng, không ghi output.

Mỗi audio là một request POST, WAV 16 kHz mono gửi dạng base64 data URI, `language_hints` = `vi`, `en`. Giới hạn API: audio tối đa 5 phút, 600 request/phút.

## Tốc độ

`--workers 4` (mặc định) chạy 4 request song song; tối đa 16. Một request không phải stream theo thời gian thực như WebSocket trước đây nên không bị chặn dưới bởi độ dài audio. Gặp HTTP 429 nhiều thì giảm `--workers`.

## Output và chạy tiếp

Output mặc định: `data/results/asr/qwen-cloud/public-test-asr-flash/` (khác thư mục của lượt WebSocket cũ, để không trộn kết quả):

- `predictions.jsonl`: kết quả mới nhất của từng audio; text, status, câu/từ kèm timestamp nếu API trả về, usage, `hotword_count`, `context_count`, `request_id` trong `diagnostics`.
- `references.jsonl`: chỉ audio cuối thành công, schema `id` + `audio_text` dùng với metrics ASR hiện có.
- `history_references.jsonl`: nhãn giả cho history, có `sample_id`, `history_index` và đường dẫn audio.
- `attempts.jsonl`: nhật ký append-only ghi trước và sau mỗi request.
- `summary.json`: số thành công/lỗi/chưa chạy, token đã hạch toán, số attempt thiếu usage cuối.
- `manifest.jsonl` (gồm hotword và context của từng audio), `run_config.json`, `last_invocation.json`: không chứa key.

`asr/review_ui.py` mặc định vẫn đọc thư mục cũ; truyền `--results data/results/asr/qwen-cloud/public-test-asr-flash` để xem kết quả mới.

Chạy lại cùng lệnh sẽ bỏ qua audio thành công và không tự thử lại lỗi/attempt bị ngắt **từ lượt trước**; thêm `--retry-errors` để chọn lại. Trong lượt hiện tại, lỗi mạng/timeout, HTTP 408/429/5xx hoặc mã `Throttling`/`InternalError`/`ServiceUnavailable` được thử lại tối đa **3 lần**, chờ **2/4/8 giây** (`--max-retries`, `--retry-delay`). Lỗi khác (sai tham số, xác thực...) dừng lượt chạy sau khi các request đang bay hoàn tất. Chỉ một tiến trình được dùng output tại một thời điểm; nếu bị kill và còn `.runner.lock`, chỉ xóa lock sau khi chắc chắn runner cũ đã dừng.

## Ngân sách token

`--token-budget 900000` là ngân sách hạch toán cộng dồn trong **cùng thư mục output**. Nó không biết quota còn lại thực tế hay usage ở nơi khác; theo dõi trên Alibaba Cloud Console.

Trước mỗi request, kể cả retry và các request đang chạy song song, runner yêu cầu còn ít nhất `--reserve-tokens 2000` token. Khi API trả `usage` (`input_tokens`, `output_tokens`), lấy số thực tế. Nếu request lỗi/bị ngắt hoặc thiếu usage, giữ reserve. Lượt WebSocket cũ tốn khoảng 16 token/giây audio; hotword và context làm tăng input token, nên chạy 20 mẫu rồi xem `summary.json` để ngoại suy. Đây là **ước tính bảo vệ ngân sách, không phải hard cap billing**.

## Kiểm tra offline

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s asr/tests -p test_qwen_cloud_label.py -v
```

Tests giả lập HTTP và state, không gửi audio hoặc key thật, không tiêu quota.

Tài liệu API: [HTTP API của qwen-audio-3.1-asr-flash](https://www.alibabacloud.com/help/en/model-studio/qwen-audio-asr-http-api), [thông tin model](https://www.alibabacloud.com/help/en/model-studio/qwen-audio-3-1-asr-flash), [tổng quan nhận dạng không thời gian thực](https://www.alibabacloud.com/help/en/model-studio/qwen-speech-recognition).
