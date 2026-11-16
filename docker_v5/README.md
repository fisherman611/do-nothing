# Docker offline (D1)

Luồng trong container: `asr_stage.py` (PhoWhisper, venv riêng) → vLLM phục vụ FC → `clef_agent.py` (bộ lọc app + FC + ToolEnv) → `submit.py`.
ASR chạy xong rồi thoát để nhả GPU trước khi bật vLLM, nên một GPU 24 GB+ là đủ.

```bash
PACK=<pack_b> bash docker/build.sh lifemate:dev          # cần mạng, kéo model ghim commit
docker run --rm --gpus all -v <data>:/data:ro -v <out>:/result -e SPLIT=public_test lifemate:dev
```

## Giả định chưa xác nhận (tài liệu đóng gói của BTC "sẽ có riêng")
- `/data/<split>/{<split>.jsonl, audio/, zaic_tools/}` và `<split>_environment/` như trong `data/train+public_test`.
- Kết quả ở `/result/submission.jsonl` (mỗi mẫu: id, tool_calls, state_hash).
- Ngân sách 30 s mỗi mẫu tính cả ASR: `clef_agent --time-budget 30 --asr-secs ...` dừng sớm và nộp các call đã có.

## Chưa làm / chưa kiểm
- **Chưa build, chưa chạy thử** (máy này không có GPU cho vLLM): cần chạy thử 20 mẫu tắt mạng, đo p50/p95 và bộ nhớ GPU (bước D1 trong plan).
- Đổi sang ASR v3 (prompt thiên vị từ khoá `bias_prompt`): `asr_stage.py` mới có chế độ v1; chế độ v21 cần nối `asr_srv/asr_infer_v2.run`.
- Tất định: vLLM `--seed 0`, request temperature 0; hai lần chạy từng lệch ~6/634 mẫu (batch khác nhau). Thử `VLLM_BATCH_INVARIANT=1` nếu vLLM 0.31 hỗ trợ trên GPU đích.
- Cờ N8/I5: bật qua `-e AGENT_FLAGS="--retry-empty --n8-tau 0.7 --i5 0.5"` sau khi đo xong (C1).
