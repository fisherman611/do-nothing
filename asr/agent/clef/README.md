# Clef-token cho tool call (Qwen3.5-4B cắt, không thinking)

Plan: `../../PLAN_FC_clef_qwen35.md`. Mọi lệnh chạy từ thư mục `zaic/`.

| File | Việc |
|---|---|
| `fmt.py` | Định dạng chung: nhãn 1 token, khối Q1/Q2, id ứng viên, render tool call giống chat template |
| `core.py` | `run_step`: một bước Q1 → Q2 → tool call, hỏi một `oracle` (train: `GoldOracle`; suy luận: vLLM) |
| `splits.py` → `splits.json` | val_ood (5 app giữ lại: ewallet, alarm, podcast_radio, loyalty, flights; 259 mẫu), val_in (128), train (1158) |
| `build_data.py` → `data/*.jsonl` | Ví dụ theo bước từ nhãn, kèm vị trí tính loss |
| `train.py` | LoRA + loss giới hạn nhãn (LS-CE + Brier) + CE cho tham số tự do |
| `modal_train.py` | Chạy `train.py` trên Modal A100-80GB (chưa chạy) |
| `../clef_agent.py` | Vòng agent qua vLLM, cùng output với `agent.py` |
| `check_vllm.py` | Bước 0: allowed_token_ids, tất định, prefix cache với Gated DeltaNet, logprob |
| `score_subsets.py` | Chấm `metrics/` trên val_in / val_ood |
| `test_fmt.py`, `test_loop.py` | Kiểm tra định dạng và vòng lặp với ToolEnv thật (oracle theo nhãn vàng → 100 điểm) |

```
python agent/clef/splits.py data/train+public_test
python agent/clef/build_data.py --data data/train+public_test --split-name train --out agent/clef/data/train.jsonl --history
python agent/clef/build_data.py --data data/train+public_test --split-name val_in --out agent/clef/data/val_in.jsonl --shuffle-seed -1
python agent/clef/test_fmt.py --data data/train+public_test
GOLD=<gold_train.jsonl> python agent/clef/test_loop.py --data data/train+public_test --out /tmp/loop.jsonl
python agent/clef/check_vllm.py --url <vLLM> --data data/train+public_test
python agent/clef_agent.py --data data/train+public_test --url <vLLM> --ids-split val_in --out agent/runs/clef/zs_val_in.jsonl
python agent/clef/score_subsets.py --gold <gold_train.jsonl> --run agent/runs/clef/zs_val_in.jsonl --sets val_in
```

Baseline zero-shot (agent.py, chấm lại trên đúng tập): không thinking val_in 39.1 / val_ood 27.2; thinking 46.0 / 35.7.
