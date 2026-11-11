#!/usr/bin/env bash
# D1: chạy trong container. Biến môi trường (có mặc định):
#   SPLIT=private_test  DATA=/data  OUT=/result  WORK=/work  WORKERS=16
#   AGENT_FLAGS="--retry-empty"        # cờ thêm cho clef_agent (vd --n8-tau 0.7 --i5 0.5 sau khi đo xong)
#   TIME_BUDGET=30                     # giây mỗi mẫu (ASR + FC); hết thì nộp các call đã có
#   LIMIT=20                           # chỉ chạy N mẫu đầu để đo thử (không tạo submission.jsonl)
set -euo pipefail
SPLIT=${SPLIT:-private_test}; DATA=${DATA:-/data}; OUT=${OUT:-/result}; WORK=${WORK:-/work}
WORKERS=${WORKERS:-16}; AGENT_FLAGS=${AGENT_FLAGS:---retry-empty}; TIME_BUDGET=${TIME_BUDGET:-30}
mkdir -p "$OUT" "$WORK"
log() { echo "[$(date +%T)] $*"; }
[ -f "$DATA/$SPLIT/$SPLIT.jsonl" ] || { echo "không thấy $DATA/$SPLIT/$SPLIT.jsonl" >&2; exit 2; }
[ -d "$DATA/$SPLIT/zaic_tools" ] || log "CẢNH BÁO: không thấy $DATA/$SPLIT/zaic_tools (clef_agent cần để tạo ToolEnv)"
T0=$(date +%s)
LIM=""; [ -n "${LIMIT:-}" ] && LIM="--limit $LIMIT"
# ghi bộ nhớ GPU mỗi 5 s để đo đỉnh (D1)
( nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader -l 5 > "$WORK/gpu_mem.csv" 2>/dev/null & echo $! > "$WORK/gpu_mem.pid" ) || true

# 1) ASR: nghe mọi lượt user của mọi mẫu; thoát tiến trình để nhả GPU trước khi bật vLLM
log "ASR"
/opt/asr/bin/python /app/docker/asr_stage.py --data "$DATA" --split "$SPLIT" --model /models/asr \
    --out "$WORK/asr.json" --secs "$WORK/asr_secs.json" ${LIMIT:+--limit $LIMIT}
log "ASR xong sau $(( $(date +%s) - T0 )) s"

# 2) vLLM phục vụ FC (cố định seed, greedy do clef_agent gửi temperature=0, seed=0)
log "bật vLLM"
vllm serve /models/fc --served-model-name llm --host 127.0.0.1 --port 8000 --max-model-len 16384 \
    --gpu-memory-utilization "${GPU_UTIL:-0.85}" --seed 0 --enable-prefix-caching \
    --max-logprobs 64 --logprobs-mode processed_logprobs --api-key local > "$WORK/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for i in $(seq 1 300); do
    curl -sf -H "Authorization: Bearer local" http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM thoát sớm:"; tail -n 30 "$WORK/vllm.log"; exit 3; }
    sleep 2
done
curl -sf -H "Authorization: Bearer local" http://127.0.0.1:8000/v1/models >/dev/null || { echo "vLLM không lên sau 600 s"; tail -n 30 "$WORK/vllm.log"; exit 3; }
log "vLLM sẵn sàng"

# 3) agent (bộ lọc app + FC từng bước + ToolEnv); mỗi mẫu một ToolEnv
python3 /app/agent/clef_agent.py --data "$DATA" --split "$SPLIT" --url http://127.0.0.1:8000 --key local \
    --out "$WORK/run.jsonl" --asr "$WORK/asr.json" --workers "$WORKERS" --fmt-from /models/fmt.json \
    --app-filter /models/pack --time-budget "$TIME_BUDGET" --asr-secs "$WORK/asr_secs.json" $LIM $AGENT_FLAGS
log "agent xong"

[ -n "${LIMIT:-}" ] && { log "LIMIT=$LIMIT: bỏ bước submit (chỉ đo thử)"; kill "$(cat "$WORK/gpu_mem.pid")" 2>/dev/null || true; exit 0; }
# 4) gom file nộp (dừng nếu thiếu/thừa/trùng id hoặc state_hash rỗng)
python3 /app/agent/submit.py --run "$WORK/run.jsonl" --data "$DATA" --split "$SPLIT" --out "$OUT/submission.jsonl"
log "xong: $OUT/submission.jsonl, tổng $(( $(date +%s) - T0 )) s"
