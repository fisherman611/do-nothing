# PhoWhisper-small: baseline train ngày 06/11/2026

Đã chạy đủ **1.545 audio yêu cầu cuối của train**, không fine-tune, không lỗi/timeout/thiếu prediction. Modal profile `vyvy372004`, một GPU **Tesla T4**, batch size 1. Lượt đầy đủ chạy từ 15:59:25 đến 16:25:18 UTC, khoảng 25 phút 54 giây.

## Chất lượng nhận dạng

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

## Tốc độ và cấu hình

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

## Artifact và kiểm tra

- [Predictions](../data/results/asr/phowhisper-small/train-full-20261106/predictions.jsonl)
- [Metrics tổng hợp và từng mẫu](../data/results/asr/phowhisper-small/train-full-20261106/metrics.json)
- [25 mẫu có WER cao nhất](../data/results/asr/phowhisper-small/train-full-20261106/worst_samples.json)
- [Cấu hình, revision, phần cứng và thời điểm chạy](../data/results/asr/phowhisper-small/train-full-20261106/run_config.json)
- [Modal run](https://modal.com/apps/vyvy372004/main/ap-5U4BnfjufrKm9rqQdsckOp)

Đã kiểm tra đủ 1.545 ID duy nhất và trùng danh sách đầu vào; CLI chấm với `--require-all` thành công. Bộ unit test metrics có **47 test qua**, gồm 37 test tool/state và 10 test ASR.

Các artifact trong `data/` được lưu local và đang bị Git ignore. Cách chạy lại: [README ASR](README.md).
