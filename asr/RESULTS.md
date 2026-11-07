# Kết quả baseline ASR

## So sánh các baseline ASR — 07/11/2026

5 model mới đã chạy đủ 1.545 audio yêu cầu cuối của train, cùng manifest, T4, FP16, greedy và seed 42. Nhãn chỉ dùng local để chấm; worker inference chặn network.

| Model | Tham số thực | WER chuẩn hóa | CER chuẩn hóa | Mean / p95 (giây) | Lỗi / timeout |
| --- | ---: | ---: | ---: | ---: | ---: |
| vinai/PhoWhisper-large | 1,543,304,960 | 25.69% | 14.76% | 2.306 / 3.826 | 0 / 0 |
| Qwen/Qwen3-ASR-1.7B | 2,038,052,480 | 27.58% | 16.00% | 1.880 / 2.971 | 0 / 0 |
| vinai/PhoWhisper-medium | 763,857,920 | 28.33% | 16.30% | 1.728 / 2.897 | 0 / 0 |
| openai/whisper-large-v3-turbo | 808,878,080 | 32.07% | 19.40% | 0.469 / 0.671 | 0 / 0 |
| Qwen/Qwen3-ASR-0.6B | 782,426,112 | 33.88% | 19.29% | 1.939 / 3.080 | 0 / 0 |
| vinai/PhoWhisper-small, baseline cũ* | 241,734,912 | 34.20% | 19.49% | 0.970 / 1.607 | 0 / 0 |

*Dòng PhoWhisper-small trích báo cáo ngày 06/11, không phải một job mới. Prediction gốc chưa có trong repo D nên chưa xác minh được input hash của dòng lịch sử. Bản nguồn được lưu tại `historical_phowhisper_small.md`.

WER/CER tính corpus với cùng phiên bản metrics cho 5 model mới, gồm raw và normalized trong CSV/JSON. Normalized giữ dấu tiếng Việt, số, dấu âm và dấu phân cách số; không chuyển số thành chữ. CER tính cả khoảng trắng.

Latency đo riêng WAV decode + preprocess + generation + text decode, không gồm load/warmup/RPC. Các prefix full hợp lệ từ lượt v3 được giữ với timing đã đo và source hash không đổi; config ghi `resumed_from`, số prediction tái sử dụng và các attempt. Không gán latency bằng 0 cho dữ liệu được tiếp tục.

Số tham số lấy từ `sum(p.numel() for p in model.parameters())`, tính trọng số chia sẻ một lần. Tổng tensor trên Hugging Face có cách đếm khác và được lưu riêng cho Qwen. Ghép ASR + LLM vẫn cần kiểm tra tổng <6B, LLM <5B, context ≤16.384 token và thời gian toàn pipeline trên A30.

### Kiểm tra tái lập

- qwen3-asr-1.7b: 100/100 transcript giống lượt full; all_match=True.
- phowhisper-large: 100/100 transcript giống lượt full; all_match=True.

### Chi phí và artifact

Compute ước tính của launcher (full tiếp tục + repeat, chưa gồm các lượt trước): **3.081 USD**. Các cap cộng 7,65 USD cho full và 0,40 USD cho repeat. Chi phí theo model trong CSV chỉ tính các attempt ở launcher, không phải toàn bộ chi phí từ đầu smoke.

Snapshot billing workspace: credit đã dùng tháng 11 là **15.82 USD**, tăng **3.97 USD** từ mốc trước đợt chạy; billed_cost **0.00 USD**. Chênh lệch gồm các smoke/khởi chạy đã dừng và có thể cập nhật trễ hoặc chứa workload khác; không phải hóa đơn tách riêng app.

- `comparison.csv` / `comparison.json`: 5 model mới, thông số đầy đủ và đường dẫn prediction.
- `summary.json`: kết quả tất cả job/attempt, repeat và compute ước tính.
- Các thư mục `<model>-<phase>-attemptN/`: manifest, config, metrics, predictions và provenance.

Đây là benchmark trên train, chưa có kết quả holdout, ASR của history hoặc toàn pipeline ASR + LLM + tool loop trên A30.

Artifact lượt mới: [report](../data/results/asr/comparison/20261107-isolated/report.md), [CSV](../data/results/asr/comparison/20261107-isolated/comparison.csv).

---

## Chi tiết PhoWhisper-small — 06/11/2026

Đã chạy đủ **1.545 audio yêu cầu cuối của train**, không fine-tune, không lỗi/timeout/thiếu prediction. Modal profile `vyvy372004`, một GPU **Tesla T4**, batch size 1. Lượt đầy đủ chạy từ 15:59:25 đến 16:25:18 UTC, khoảng 25 phút 54 giây.

### Chất lượng nhận dạng

| Metric corpus | Raw | Chuẩn hóa |
| --- | ---: | ---: |
| WER | 45,02% | **34,20%** |
| CER | 22,58% | **19,49%** |
| Exact transcript match | 0/1.545 | 24/1.545 (1,55%) |

Raw áp Unicode NFC và gộp khoảng trắng. Chuẩn hóa thêm lowercase và bỏ dấu câu; giữ dấu tiếng Việt, số, dấu âm, dấu phân cách số và ký hiệu đơn vị/tiền. Không chuyển số thành chữ hoặc ngược lại. CER gồm khoảng trắng. Corpus WER/CER lấy tổng lỗi chia tổng độ dài tham chiếu, không lấy trung bình điểm từng câu. Report được chấm lại từ prediction đã lưu bằng CLI metrics hiện tại.

Ví dụ đúng sau chuẩn hóa, `train_0004`:

- Nhãn: “Trợ lý ơi, giảm độ sáng đèn phòng ngủ nhỏ xuống còn một nửa nhé.”
- ASR: “trợ lý ơi giảm độ sáng đèn phòng ngủ nhỏ xuống còn một nửa nhé.”

Ví dụ sai nội dung, `train_0005`: đoạn “đi họp phụ huynh chiều nay” được nhận thành “đi học của quân triều nay”. Ở `train_0003`, model viết “năm mươi ngàn” trong khi nhãn là “50 nghìn”; WER hiện vẫn tính khác biệt cách biểu diễn này. Vì vậy kết quả gồm cả lỗi nhận dạng và khác biệt cách viết.

### Tốc độ và cấu hình

| Đại lượng | Kết quả |
| --- | ---: |
| Thành công | 1.545/1.545 |
| Lỗi / timeout / thiếu | 0 / 0 / 0 |
| Latency trung bình | 0,970 giây/mẫu |
| p50 / p95 / max | 0,942 / 1,607 / 2,712 giây |
| Tổng thời gian xử lý từng mẫu | 1.498,15 giây |
| Tổng độ dài audio | 11.165,50 giây |
| Real-time factor | 0,134 |
| Peak GPU memory do PyTorch cấp phát | 576,42 MiB |
| Model load / warmup | 2,205 / 0,999 giây |

- Model: `vinai/PhoWhisper-small`, revision `a86b604c346caf7148c37512eafe783a16420adb`; đo được 241.734.912 tham số.
- FP16, SDPA, greedy (`do_sample=False`, `num_beams=1`), `language=vi`, `task=transcribe`, `max_new_tokens=256`, seed 42, bật deterministic algorithms.
- PyTorch 2.7.1+cu126, Transformers 4.57.6; các phiên bản khác lưu trong `run_config.json`.
- Model được tải khi build image. Worker inference chặn network và chỉ nhận ID + WAV; nhãn để local khi chấm.
- Smoke test 20 mẫu thành công; 20 transcript giao với lượt đầy đủ giống nhau. Đây chưa phải kiểm tra tái lập nhiều lượt đầy đủ.

Latency đo decode WAV + preprocessing + generate + text decode trong worker, không gồm model load/warmup hoặc truyền dữ liệu local–Modal. Peak memory là số PyTorch cấp phát, không phải toàn bộ bộ nhớ GPU mà tiến trình giữ.

Đây là baseline trên **train**, chỉ cho audio lượt cuối có transcript. Chưa đo ASR của history, chất lượng holdout, tốc độ trên A30 hoặc giới hạn 30 giây của toàn pipeline ASR + LLM + tool loop.

### Artifact và kiểm tra

- [Predictions](../data/results/asr/phowhisper-small/train-full-20261106/predictions.jsonl)
- [Metrics tổng hợp và từng mẫu](../data/results/asr/phowhisper-small/train-full-20261106/metrics.json)
- [25 mẫu có WER cao nhất](../data/results/asr/phowhisper-small/train-full-20261106/worst_samples.json)
- [Cấu hình, revision, phần cứng và thời điểm chạy](../data/results/asr/phowhisper-small/train-full-20261106/run_config.json)
- [Modal run](https://modal.com/apps/vyvy372004/main/ap-5U4BnfjufrKm9rqQdsckOp)

Đã kiểm tra đủ 1.545 ID duy nhất và trùng danh sách đầu vào; CLI chấm với `--require-all` thành công. Bộ unit test metrics có **47 test qua**, gồm 37 test tool/state và 10 test ASR.

Các artifact trong `data/` được lưu local và đang bị Git ignore. Cách chạy lại: [README ASR](README.md).
