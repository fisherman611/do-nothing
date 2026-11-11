# Metrics LifeMate

Bộ chấm local bằng Python 3.10+. Phần tool-call/state chỉ dùng thư viện chuẩn; ASR WER/CER dùng `jiwer==4.0.0` trong `requirements.txt`. Đã đối chiếu [ảnh công thức chính thức](reference/official_formula.png) trong mục Evaluation LifeMate ngày 06/11/2026; [nguồn](reference/README.md). Chạy các lệnh bên dưới từ thư mục gốc repository.

## Quy tắc đang triển khai

- So hash dưới dạng chuỗi: hash khác gold thì mẫu được 0 điểm. Với `--replay`, hash tính lại bằng `ToolEnv` là kết quả có thẩm quyền; báo riêng việc hash khai báo khác hash replay.
- Hai call khớp khi tên tool giống nhau, các tham số không fuzzy giống nhau, và từng tham số `x-match: "fuzzy"` có token-F1 **> 0,6**. Catalog phải thuộc đúng split.
- Ghép một-một theo số cặp khớp tối đa (maximum bipartite matching), bỏ qua thứ tự call và giữ nguyên số lần xuất hiện của call trùng nhau.
- `P = M / số call nộp`, `R = M / số call gold`, `F1 = 2M / (số call nộp + số call gold)`. Không có call khớp thì F1 bằng 0. Call không hợp lệ hoặc không có tool trong catalog vẫn tính vào số call nộp và không khớp.
- Khi gold rỗng, phần điểm call bằng 1 chỉ khi danh sách nộp cũng rỗng. Điều kiện hash vẫn áp dụng.
- `sys.ask_missing_info` được so như các tool khác.
- Mẫu timeout hoặc lỗi thực thi được 0 điểm, kể cả khi hash/calls đúng; không được gỡ mẫu đó khỏi mẫu số.
- Điểm cuối là **100 × trung bình điểm mẫu** trên toàn bộ ID trong gold. `score` nằm trong `[0, 100]`, `mean_sample_score` và điểm từng mẫu nằm trong `[0, 1]`.
- `tool_call_f1` là F1 trung bình để chẩn đoán, chưa áp điều kiện thực thi. `tie_break_score` trên thang 100 bỏ điều kiện hash nhưng vẫn cho 0 với mẫu thiếu/lỗi/timeout.

Token-F1 dùng chữ thường, Unicode NFC, tách token theo khoảng trắng và đếm số lần xuất hiện của mỗi token. Dấu câu và dấu tiếng Việt được giữ nguyên. Hai chuỗi rỗng khớp nhau. Thiếu tham số khác với truyền `null`; tham số dư không khớp. Số JSON `1` và `1.0` khớp nhau, còn `true` không khớp `1`. Các danh sách tham số giữ thứ tự.

Công thức điểm và thang 100 đã được xác minh từ ảnh chính thức. Văn bản công khai chưa xác định chi tiết tokenizer/dấu câu hoặc thuật toán ghép khi nhiều cặp cùng khớp; các lựa chọn local trên được nêu rõ để đối chiếu khi có mã scorer của BTC. Không khẳng định đã đối chiếu được tokenizer nội bộ của BTC.

## Tạo gold có hash cho train

Nhãn trong `train.jsonl` có `label.tool_calls`, nhưng không có hash chuẩn. Không thể tự hash `label.final_state` vì nó chỉ chứa các app bị thay đổi. Lệnh sau chạy chuỗi call gold bằng đúng `train/zaic_tools`, lấy hash từ `env.get_state_hash()` rồi lưu cache:

```powershell
python -m metrics prepare-gold --data-root "data/train+public_test" --output "data/metrics/gold_train.jsonl"
```

Mỗi mẫu dùng một `ToolEnv` duy nhất, chỉ gọi `call`, `get_tool_calls`, `get_state_hash`. Lệnh không tự đọc file dưới `*_environment/`, không dùng `get_state`, `Executor` hay `Simulator`, và không sửa dữ liệu gốc. Nếu call gold bị từ chối thì dừng, không tạo lại môi trường. Output được thay thế nguyên tử sau khi hoàn tất; dùng lại cache này cho các lần chấm tiếp theo. Tạo gold và xác minh submission là các lượt chạy độc lập, mỗi lượt chỉ có một phiên cho mỗi mẫu.

Để chỉ tạo gold của tập validation, truyền `--labels path/to/validation.jsonl` (các dòng train có nhãn được chọn). Lệnh này chỉ phục vụ train có nhãn; public/private test không có gold để chấm local.

## Chấm submission

Gold và submission dùng cùng cấu trúc, một dòng cho mỗi mẫu:

```json
{"id": "train_0001", "tool_calls": [{"name": "weather.get_forecast", "arguments": {"date": "2026-11-29"}}], "state_hash": "hash-from-env"}
```

```powershell
python -m metrics score --gold "data/metrics/gold_train.jsonl" --submission "result/submission.jsonl" --catalog "data/train+public_test/train/train_tools_catalog.json" --replay --data-root "data/train+public_test" --split train --timings "result/timings.jsonl" --output "result/metrics.json"
```

Submission phải dùng ID thuộc tập gold đang chấm. Nếu chỉ đánh giá một subset thì chuẩn bị gold của đúng subset đó; nếu không, các mẫu bị thiếu vẫn nằm trong mẫu số và nhận 0 điểm. Thêm `--require-all` để báo lỗi khi thiếu ID. ID dư, ID trùng, thiếu hash hoặc sai cấu trúc dòng gây lỗi (exit code 2).

Console in thống kê tổng. `--output` lưu thêm điểm từng mẫu, cặp chỉ số call đã ghép, số call thừa/thiếu, precision, recall, trạng thái thực thi, hash khai báo/thực tế, thời gian inference/replay và cờ `missing_prediction`. Chỉ số call bắt đầu từ 0.

`--replay` phát lại tuần tự các call trong tiến trình riêng cho mỗi mẫu, bằng đúng `zaic_tools` của split. Không đọc state trực tiếp, không clone/reset env. Call sai schema vẫn được thực hiện qua API và được đánh dấu không khớp; replay tiếp tục cho những call sau. Call sai cấu trúc không thể truyền vào API vẫn tính là call chưa khớp. Hash sau replay quyết định điểm, kể cả khi khác hash khai báo. Các trường fuzzy được loại khỏi hash bởi `zaic_tools` chính thức, không tự mô phỏng lại tại đây.

Replay lỗi/treo cho mẫu đó 0 điểm và không thử lại. `--replay-timeout` mặc định 30 giây là giới hạn của **bộ xác minh**, không phải thời gian inference. SHA-256 của mã runtime và catalog được ghi vào report để nhận biết phiên bản; catalog truyền vào phải trùng bản dùng bởi runtime.

Có thể bỏ `--replay` để so nhanh hash khai báo, nhưng report sẽ ghi `hash_verification: "declared_only"`. Không được trộn mode này với report đã replay khi xếp hạng. Có thể bỏ `--timings` khi chưa có pipeline; lúc đó không áp được timeout inference và không có tie-break thời gian.

## Runtime và giới hạn 30 giây

Lưu thời gian **ASR + LLM + tool loop** của từng mẫu vào file riêng, không thêm trường vào submission:

```jsonl
{"id": "train_0001", "elapsed_seconds": 12.4, "status": "ok"}
{"id": "train_0002", "elapsed_seconds": 30.1, "status": "timeout"}
{"id": "train_0003", "elapsed_seconds": 0.8, "status": "error", "error": "inference failed"}
```

Đúng 30 giây vẫn hợp lệ; lớn hơn 30 giây, `status: "timeout"` hoặc `status: "error"` cho điểm 0. Nếu truyền `--timings`, file phải bao phủ chính xác toàn bộ ID của tập gold, kể cả mẫu thất bại hoặc thiếu prediction. Không nhận số âm, NaN, vô cực, ID lặp/thiếu/dư. Thời gian trung bình được tính trên tất cả mẫu; tuyệt đối không lấy thời gian replay làm tốc độ model.

Thư viện có helper hard timeout cho pipeline sau này:

```python
from dataclasses import asdict
from metrics import run_timed_sample

# infer_one là hàm cấp module do pipeline cung cấp, tạo ToolEnv bên trong
# và trả lại {id, tool_calls, state_hash}. Đặt lệnh chạy dưới main guard.
if __name__ == "__main__":
    prediction, timing = run_timed_sample(
        item["id"], infer_one, args=(item,), timeout_seconds=30,
    )
    timing_row = asdict(timing)
```

Helper dùng process `spawn`, đo cả startup, nạp model trong callback và truyền kết quả. Nó dừng process khi quá hạn, trả `None` với status tương ứng khi lỗi/timeout, và không chạy lại mẫu. Callback phải là hàm cấp module, không dùng lambda hay truyền env/model đã tạo ở process cha. Đây là helper kiểm thử theo mẫu; pipeline tối ưu có thể nạp model một lần trong worker riêng và ghi cùng định dạng timing. Không giả định helper này đã đo trên GPU A30 hoặc đã kiểm tra Docker/network.

Mẫu thất bại trả `None` có thể thiếu khỏi prediction trong đánh giá local và vẫn nhận 0 điểm. File như vậy chưa phải submission đầy đủ theo quy định; dùng `--require-all` để bắt thiếu ID trước khi nộp.

## Tie-break và tính tái lập

Xếp hạng theo điểm chính cao hơn, rồi `tie_break_score` cao hơn (F1 bỏ điều kiện hash), rồi thời gian inference trung bình thấp hơn:

```powershell
python -m metrics rank --entry "baseline=result/baseline.json" --entry "candidate=result/candidate.json" --output "result/ranking.json"
```

Các report phải dùng cùng gold, catalog, mode xác minh và số lượt chạy. Nếu cần tie-break thời gian mà thiếu timing thì báo lỗi, không tự đoán. Không làm tròn trước khi so sánh. Cả ba tiêu chí giống nhau thì đồng hạng.

Khi áp quy tắc chạy lại 5 lần, chấm riêng từng lần với prediction/timing thực tế của lần đó, rồi lấy trung bình:

```powershell
python -m metrics average-runs --reports result/run1.json result/run2.json result/run3.json result/run4.json result/run5.json --output result/average5.json
```

Lệnh yêu cầu đúng 5 report đơn, cùng tập mẫu, gold và runtime. Mẫu thất bại vẫn nằm trong mẫu số của từng lượt. Điểm chính và F1 tie-break được lấy trung bình qua 5 lượt; thời gian trung bình chỉ có khi cả 5 lượt có đủ timing. Lệnh này tổng hợp các lần inference đã chạy độc lập; không tự chạy model/Docker và không chứng minh rằng các file report đến từ những lần chạy độc lập.

## Dùng trong Python

```python
from metrics import ToolCatalog, evaluate
from metrics.io import load_json, read_jsonl

catalog = ToolCatalog.from_dict(load_json("data/train+public_test/train/train_tools_catalog.json"))
report = evaluate(
    read_jsonl("result/submission.jsonl"),
    read_jsonl("data/metrics/gold_train.jsonl"),
    catalog,
)
print(report.summary)
```

## Kiểm tra

```powershell
python -m unittest discover -s metrics/tests -v
```

Các case kiểm tra gồm Unicode NFC, dấu câu, token lặp, ngưỡng chính xác 0,6, ghép tối đa, call trùng/sai, thang 100, hash replay có thẩm quyền, thứ tự replay, gold rỗng, ID thiếu/trùng/dư, mốc 30 giây, hard timeout/error, timing coverage, ba bước tie-break, trung bình 5 lượt và CLI.

## Metrics ASR

Đầu vào tham chiếu là các dòng train có `id`, `audio_text`. Prediction có dạng:

```json
{"id":"train_0001","text":"transcript do ASR sinh ra","status":"ok","elapsed_seconds":0.5,"audio_seconds":4.2}
```

```powershell
python -m metrics asr --references data/train+public_test/train/train.jsonl --predictions data/results/asr/phowhisper-small/train-full/predictions.jsonl --output data/results/asr/phowhisper-small/train-full/metrics.json --worst-output data/results/asr/phowhisper-small/train-full/worst_samples.json --require-all
```

`metrics.asr.evaluate_asr` báo WER/CER corpus theo tổng số lỗi chia tổng độ dài tham chiếu, không lấy trung bình WER từng câu. Các metric gồm:

- Raw: Unicode NFC và gộp khoảng trắng, giữ nguyên hoa/thường và dấu câu.
- Normalized: thêm lowercase và bỏ dấu câu; giữ dấu tiếng Việt, chữ số, dấu âm như `-5`, ký hiệu tiền, `%` và dấu phân cách nằm giữa chữ số như `09:30`, `1.500.000`, `2/3`. Không tự chuyển số thành chữ hoặc ngược lại.
- CER tính cả khoảng trắng sau chuẩn hóa. WER/CER có thể vượt 100% khi có nhiều insertion. Quy ước tham chiếu rỗng theo jiwer 4: rỗng/rỗng có lỗi 0; rỗng/prediction có rate bằng số token/ký tự chèn.
- Exact match, counts substitution/deletion/insertion, số mẫu thiếu/lỗi/timeout, latency mean/p50/p95/max và real-time factor (tổng latency/tổng giây audio có đủ timing).

Prediction thiếu hoặc có status lỗi/timeout được chấm như chuỗi rỗng, giữ nguyên mẫu số. Transcript thực tế vẫn lưu trong report để chẩn đoán. ID lặp/dư, thiếu transcript tham chiếu, timing âm/NaN/vô cực gây lỗi. `--require-all` bắt thiếu ID; khi chấm subset hãy truyền đúng file references của subset. Các mẫu chưa có timing không bị gán số đo giả; report nêu rõ số timing đã đo.

Đây là metric nhận dạng tiếng nói, không phải điểm tool-call/state chính thức của cuộc thi. Chi tiết chuẩn edit-distance và empty reference dựa trên [jiwer](https://jitsi.github.io/jiwer/usage/).

## Chỉ số chẩn đoán (không phải điểm chính thức)

Hai lệnh dưới chỉ để phân tích lỗi; không ảnh hưởng `score`.

### Chọn tool và độ đúng tham số

```bash
python -m metrics calls --gold agent/runs/clef/gold_train.jsonl --submission <run.jsonl> \
    --catalog data/train+public_test/train/train_tools_catalog.json --output <calls.json>
```

- **Tool selection**: so tập tên tool (có đếm số lần) giữa bài nộp và gold → precision / recall / F1.
- **Argument accuracy**: ghép các call cùng tên tool, rồi xếp từng tham số vào `match` / `wrong` / `missing` / `extra`, chia theo kiểu
  tham số: `id`, `enum`, `bool`, `number`, `datetime`, `fuzzy` (`x-match: fuzzy`, token-F1 > 0,6), `text`, `other`.
- Mỗi mẫu thuộc một lớp: `exact`, `tool_selection` (thiếu/thừa tool), `arguments` (đúng tool, sai tham số), `both`.
- ID có trong gold mà không có trong bài nộp thì bỏ qua (dùng được gold của cả train cho một split con).

### ASR chi tiết

```bash
python -m metrics asr-detail --references data/tts_v2/manifest_v2.json --split val \
    --predictions agent/runs/asr/heldout.e46f.asr.json --audio-root data/train+public_test/train --output <asr_detail.json>
```

- **Entity error rate**: thực thể có trong câu chuẩn mà không có trong transcript. Thực thể có nhãn lấy từ `label_ents` / `hist_ents`
  của manifest (loại `person`, `place`, `other_name`); thực thể bề mặt lấy từ chính câu chuẩn: `number` (token toàn chữ số),
  `code` (chữ lẫn số hoặc viết hoa: SJC, VTV3, RV305), `foreign_word` (từ ASCII không phải âm tiết tiếng Việt).
- **Code-switching WER**: WER của các câu có `foreign_word` hoặc `code`, so với câu thuần Việt.
- WER/CER theo loại lượt (`final` / `history`).
- **Real-time factor** = tổng giây xử lý / tổng giây audio, khi file dự đoán có `secs` và có `--audio-root`.
- Chưa có: các chỉ số streaming (first partial, finalization, stability, endpointing) và concurrent streams — bài thi chấm file wav theo lô.
