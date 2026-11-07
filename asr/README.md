# Test PhoWhisper-small bằng Modal

Kế hoạch: [plans/asr_baseline.md](../plans/asr_baseline.md). Runner chỉ dùng audio cuối của train; nhãn `audio_text` được giữ local để chấm sau inference. Mã không truy cập app environment.

Kết quả lượt đầu: [PhoWhisper-small trên đủ 1.545 mẫu train](RESULTS.md).

Chuẩn bị ở local: Modal 1.6.3 và jiwer 4.0.0. Các thư viện Torch/Transformers và trọng số được cài trong image GPU, không cần tải model về máy local.

Chạy các lệnh từ thư mục gốc repo. Nếu PowerShell chặn activate virtualenv, thay `python` trong các lệnh dưới bằng `.\.venv\Scripts\python.exe`; không cần sửa Execution Policy.

```powershell
python -m pip install modal==1.6.3 jiwer==4.0.0
modal profile activate vyvy372004
```

Chạy 20 mẫu đầu để kiểm tra:

```powershell
python -X utf8 -m modal run --profile vyvy372004 -m asr.modal_phowhisper --limit 20 --output-dir data/results/asr/phowhisper-small/smoke-20261106
```

Chạy đủ train:

```powershell
python -X utf8 -m modal run --profile vyvy372004 -m asr.modal_phowhisper --output-dir data/results/asr/phowhisper-small/train-full-20261106
```

Mỗi lượt chọn thư mục output mới; runner không ghi đè prediction cũ. Có thể dùng `--train-dir` nếu gói dữ liệu nằm ở vị trí khác. `-X utf8` tránh lỗi terminal Windows khi Modal in ký hiệu tiến độ.

Model được pin revision, download khi build image; inference chặn network, chỉ load trọng số local. Một worker T4 giữ model, xử lý từng WAV bằng greedy decoding, seed 42 và batch size 1. Mọi call có `retries=0`; lỗi/timeout được ghi theo mẫu. Model load/warmup được báo riêng, latency mỗi mẫu đo decode WAV + preprocessing + generate + text decode, chưa gồm truyền dữ liệu local–Modal hoặc toàn pipeline agent.

Output gồm `predictions.jsonl`, `references.jsonl` (local), `metrics.json`, `worst_samples.json`, `run_config.json` và `run.log`. Raw/normalized WER/CER, thời gian, GPU và revision được lưu để tái lập. Report này đo trên train; không dùng để tuyên bố chất lượng holdout hay tốc độ trên A30.

## Chạy 5 baseline mới song song trên Windows

Chạy tại repo `D:\GitHub\do-nothing`. Modal client có lỗi kết nối khi nhiều luồng dùng chung các iterator map dài trên Windows, nên launcher dùng **một tiến trình Modal CLI/model**, tối đa 5 GPU. Các checkpoint và license được pin trong `asr/registry.py`.

Lượt smoke đã xác minh đủ 100 ID/model ở `data/results/asr/comparison/20261107-baselines-v3`. Để chạy full hoặc tiếp tục các prediction hợp lệ của lượt đó:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m asr.launch_suite --smoke-from data/results/asr/comparison/20261107-baselines-v3 --resume-from data/results/asr/comparison/20261107-baselines-v3 --output-dir data/results/asr/comparison/my-new-run
```

Chọn output mới; không chạy trùng lệnh khi launcher đang hoạt động. Lượt đang triển khai dùng output `20261107-isolated`. Launcher có cap ước tính tổng 7,65 USD cho full và 0,40 USD cho kiểm tra tái lập, ngoài chi phí các smoke trước; xem `CAPS` trong mã. Mỗi client dành 0,10 USD dự phòng; đây không phải hard cap billing của Modal.

Runner gửi từng nhóm 25 WAV vào hàng đợi, GPU vẫn xử lý **batch size 1**, greedy FP16, seed 42 và không nhận nhãn. Worker giữ tối đa 60 giây khi idle. Manifest, input và mã inference được kiểm tra trước khi tiếp tục; chỉ giữ prediction thành công với latency gốc, không gán timing giả. Call không có kết quả đã ghi có thể được chạy lại khi khôi phục transport; các attempt và chi phí được lưu rõ.

Để tạo smoke mới, dùng `asr.modal_suite --smoke-only`; có thể chọn một model bằng `--models phowhisper-medium`. Full trực tiếp một model có `--phase full --models <slug> --smoke-from <smoke-run>`. Các model cần cùng manifest và phiên bản metrics. Multi-model full dùng launcher thay vì chung Modal client trên Windows.

Output launcher gồm log riêng từng client, `status.json`, `summary.json`; mỗi attempt có predictions/metrics/config, manifest và provenance riêng. Sau full, launcher chạy lại 100 ID cho 2 model WER tốt nhất. Cấu hình lỗi/timeout và lượt chưa đủ ID được báo rõ, không bỏ mẫu khỏi WER/CER.

Kiểm tra: `python -m unittest discover -s asr/tests -v` và `python -m unittest discover -s metrics/tests -v`.

## Benchmark Whisper Large V3

Model `openai/whisper-large-v3` dùng revision `06f233fe06e710322aca913c1bc4249a0d71fce1`, license Apache-2.0. Worker dùng image riêng để chỉ tải checkpoint này, T4, FP16, greedy, cùng manifest và metrics với các baseline. Chạy smoke 100 mẫu rồi full khi smoke đạt và chi phí dự kiến nằm trong ngân sách:

```bash
ASR_ONLY_LARGE_V3=1 python -m modal run --profile vyvy372004 -m asr.modal_suite --models whisper-large-v3 --output-dir data/results/asr/comparison/whisper-large-v3-run --budget-usd 10
```

Output phải là thư mục mới. Ngân sách là ước tính compute, không phải hard cap của Modal. Chỉ audio yêu cầu cuối của train được dùng; nhãn được chấm local.
