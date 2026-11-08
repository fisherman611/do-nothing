# Train ASR v2 trên Kaggle (3 biến thể, 3 tài khoản)

Plan: `../PLAN_ASR_bias_v2.md`. Code train: `../modal/asr_train.py` (không phụ thuộc Modal) + `../modal/bias_prompt.py`.

- **GPU**: P100 đã bị Kaggle ngừng (CLI tự chuyển sang T4), nên kernel xin `NvidiaTeslaT4`, chỉ dùng 1 GPU. T4 không có bf16 → fp16, model gốc nạp fp16, LoRA fp32 (`--base-fp16`).
- **Mỗi tài khoản 30 giờ GPU/tuần**, phiên tối đa ~12 giờ. Kernel train tối đa `TIME_BUDGET_H = 8` giờ (tính từ lúc kernel bắt đầu), chừa thời gian cho eval cuối. Hết giờ thì lưu checkpoint và dừng; đẩy phiên `--resume 1` để train tiếp từ output phiên trước.
- **Batch**: 4 × accum 8 = 32 như v1; OOM thì driver tự chạy lại với 2 × 16. Eval giữa các epoch trên 150 utt đầu của val (để đỡ tốn giờ), eval cuối trên đủ 405 utt ở 3 điều kiện.
- **Dữ liệu**: dataset **riêng tư** `<user>/zaic-asr-v2-data` (manifest_v2.json, lexicon_v2.json, train/audio). Không có `*_environment/`, không có public test.

| Biến thể | Tài khoản (dòng trong file token) | Cờ thêm |
|---|---|---|
| V1 | 1 | – |
| V2 | 2 | `--label-ent-prob 0.5` |
| V3 | 3 | `--label-ent-prob 0.5 --all-distract-prob 0.3` |

```
export KAGGLE_TOKENS=<file, mỗi dòng một token KGAT_..., thứ tự V1 V2 V3>   # không để trong repo
python modal/make_manifest_v2.py data/train+public_test
python kaggle/launch.py data                 # upload dataset cho 3 tài khoản
python kaggle/launch.py push V1 --smoke      # chạy thử 24 hội thoại, 1 epoch
python kaggle/launch.py push V1              # train thật (tương tự V2, V3)
python kaggle/launch.py status
python kaggle/launch.py output V1            # status.json, eval.json, train.log → .kaggle_work/outputs/
python kaggle/launch.py push V1 --resume 1   # nếu status.json báo "incomplete"
```

Output của kernel (`/kaggle/working/out/<tag>`): model gộp fp16 + `eval.json` (WER/CER, lỗi tên, chép nhầm ở điều kiện (a), (b) lexicon, không prompt), `<tag>_adapter` (LoRA tốt nhất), `<tag>_ckpt` (để resume).
