# Tạo transcript tham chiếu bằng Alibaba Cloud

Runner local dùng `qwen-audio-3.1-asr-flash-message` để tạo **nhãn giả phục vụ đánh giá ASR**. Không dùng nhãn test này để train/tuning, không tích hợp API vào Docker inference của cuộc thi. Không ghi đè dataset hoặc đọc `*_environment/`. WER/CER so với các nhãn này chỉ thể hiện độ lệch so với Qwen, chưa phải điểm so với nhãn chuẩn do người xác minh.

Chạy từ thư mục gốc repo, PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-asr-cloud.txt
.\.venv\Scripts\python.exe -m asr.qwen_cloud_label --dry-run
.\.venv\Scripts\python.exe -m asr.qwen_cloud_label --limit 20
```

Lệnh cuối **gọi API và tiêu quota**, tạo 20 transcript đầu tiên. Khi đã kiểm tra transcript và usage, bỏ `--limit` để tiếp tục cùng lượt:

```powershell
.\.venv\Scripts\python.exe -m asr.qwen_cloud_label --token-budget 900000
```

Mặc định terminal dùng thanh **tqdm**: tiến độ/ETA, ID đang xử lý, số thành công/lỗi, `tokens` đã được API xác nhận, `reserve` cho các request chưa biết usage và `unknown` là số attempt chưa biết usage. Chạy tiếp từ một lượt có mẫu lỗi:

```powershell
.\.venv\Scripts\python.exe -m asr.qwen_cloud_label --token-budget 900000 --retry-errors
```

Nếu cần log JSON Lines, thêm `--log-format jsonl`. Log lỗi mới có `diagnostics`: bước kết nối/gửi task/stream audio/chờ kết thúc, số bytes đã gửi, event cuối, thông báo đã che key, HTTP status hoặc WebSocket close code/reason nếu server cung cấp. Log cũ chỉ có tên exception nên không thể suy ra chính xác nguyên nhân đóng kết nối từ các attempt cũ.

Mặc định xử lý audio cuối và mọi audio user trong history: **1.933 WAV, khoảng 2,762 giờ** trên dataset hiện tại, gồm 634 yêu cầu cuối. Để chỉ lấy audio cuối, chọn scope và thư mục output riêng:

```powershell
.\.venv\Scripts\python.exe -m asr.qwen_cloud_label --scope final --limit 20 --output-dir data/results/asr/qwen-cloud/public-test-final
```

`--limit` chọn N audio đầu trong manifest, không phải N hội thoại hay N audio chưa chạy. Manifest giữ thứ tự hội thoại trong JSONL, audio cuối trước rồi các lượt user trong history. ID audio cuối là ID hội thoại; ID history có dạng `<sample_id>:history:<index>` với index bắt đầu từ 0 trong mảng `history`.

## Cấu hình key

Runner đọc `.env` bằng `python-dotenv`; biến môi trường của tiến trình được ưu tiên. Chỉ cần hai biến đã có trong `.env` của bạn:

```dotenv
DASHSCOPE_API_KEY=<your-api-key>
API_HOST=<workspace-id>.ap-southeast-1.maas.aliyuncs.com
```

Cũng có thể lấy hostname từ `DASHSCOPE_URL` nếu không có `API_HOST`. Endpoint thực dùng là `wss://<host>/api-ws/v1/inference`, không phải endpoint OpenAI-compatible. Không dùng biến `ID` để suy ra workspace. Key chỉ ở trong bộ nhớ; `.env` đã được `.gitignore` bỏ qua. `--dry-run` không cần key, không gọi mạng, không ghi output.

Runner gửi PCM 16 kHz, mono, 16 bit theo chunk 100 ms, một audio mỗi request, một request tại một thời điểm. Ngôn ngữ gợi ý `vi` và `en` cho câu tiếng Việt xen tiếng Anh. Tắt `disfluency_removal_enabled` để không yêu cầu biên tập từ đệm/tự sửa lời. Chỉ ghép câu có `sentence_end=true`, cập nhật theo `sentence_id` để không nối lặp kết quả trung gian. Không gửi history text, tool results hoặc nhãn train làm ngữ cảnh.

## Output và chạy tiếp

Output mặc định: `data/results/asr/qwen-cloud/public-test-references/`:

- `predictions.jsonl`: kết quả mới nhất của từng audio đã thử; text, status, thời gian, timestamp câu/từ nếu API trả về, usage và task ID.
- `references.jsonl`: chỉ audio cuối thành công, schema `id` + `audio_text` dùng với metrics ASR hiện có.
- `history_references.jsonl`: nhãn giả cho history, có `sample_id`, `history_index` và đường dẫn audio.
- `attempts.jsonl`: nhật ký append-only ghi trước và sau mỗi request để giữ thông tin khi bị ngắt.
- `summary.json`: số thành công/lỗi/chưa chạy, token đã hạch toán, số attempt thiếu usage cuối.
- `manifest.jsonl`, `run_config.json`, `last_invocation.json`: input checksum, cấu hình và tham số lượt chạy; không chứa key.

References có `reference_type=pseudo_label`, `human_reviewed=false`; chỉ gồm request thành công. Transcript rỗng vẫn giữ để kiểm tra thủ công. Không có file submission tool calling hoặc dataset train mới.

Chạy lại cùng lệnh sẽ bỏ qua audio thành công và mặc định không tự thử lại lỗi/attempt bị ngắt **từ lượt chạy trước**. Thêm `--retry-errors` để chọn lại các mẫu đó. Trong lượt hiện tại, lỗi transport tạm thời hoặc HTTP 408/429/5xx được thử lại tối đa **3 lần**, chờ **2/4/8 giây** (`--max-retries`, `--retry-delay`). Không tự retry lỗi xác thực, protocol/policy hoặc thành công nhưng thiếu usage cuối. Mỗi attempt được ghi riêng và giữ reserve khi usage chưa rõ; retry không xóa lịch sử chi phí. Request gửi lại có thể tiếp tục bị tính phí. Dùng `--max-retries 0` để tắt retry. Nếu đổi input, audio, scope hoặc endpoint, dùng output mới. Chỉ một tiến trình được phép dùng output tại một thời điểm; nếu bị kill và còn `.runner.lock`, chỉ xóa lock sau khi chắc chắn runner cũ đã dừng.

## Ngân sách token

`--token-budget 900000` là ngân sách hạch toán cộng dồn trong **cùng thư mục output**, chừa khoảng trống so với quota 1M. Nó không biết quota còn lại thực tế, usage ở ứng dụng khác hoặc output khác. Theo dõi quota/hạn dùng trên Alibaba Cloud Console.

Trước mỗi request, kể cả retry, runner yêu cầu còn ít nhất `--reserve-tokens 10000` token. Khi API trả usage cuối, lấy `input_tokens + output_tokens` thực tế; các event usage là số cộng dồn nên không cộng lặp từng event. Nếu request lỗi/bị ngắt hoặc thiếu usage cuối, giữ reserve (hoặc usage đã thấy nếu lớn hơn). Dừng khi hết lượt retry, gặp lỗi không retry được, thiếu usage cuối sau thành công hoặc chạm budget. Pending attempt vẫn bị tính reserve khi chạy tiếp. `summary.json` tách `confirmed_tokens` và `reserved_unknown_tokens`; `accounted_tokens` là tổng hai phần. Đây là **ước tính bảo vệ ngân sách, không phải hard cap billing**: request có thể tốn hơn reserve; lỗi hoặc ngắt mạng có thể đã bị tính phí mà chưa trả usage.

Kiểm tra 20 mẫu trước rồi xem `summary.json` và `predictions.jsonl` để ngoại suy theo thời lượng audio. Không mặc định quy đổi 1M token thành một số giờ cố định cho model này. Nếu file journal bị ghi dở/corrupt, runner báo lỗi thay vì tự bỏ qua request đã có thể bị tính phí; giữ bản gốc để đối chiếu usage trước khi sửa.

## Kiểm tra offline

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s asr/tests -p test_qwen_cloud_label.py -v
```

Tests giả lập transport, không gửi audio hoặc key thật, không tiêu quota.

Protocol đối chiếu với tài liệu chính thức: [WebSocket endpoint và flow](https://help.aliyun.com/en/model-studio/qwen-asr-message-websocket-api), [client events](https://www.alibabacloud.com/help/en/model-studio/qwen-asr-message-client-events), [server events và usage](https://docs.qwencloud.com/api-reference/speech-recognition/qwen-asr-message/server-events).
