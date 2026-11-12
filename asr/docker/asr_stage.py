"""D1 (PLAN_72h_v3): giai đoạn ASR trong Docker. Đọc <data>/<split>/<split>.jsonl, nghe mọi lượt user theo thứ tự trong từng
mẫu, ghi transcript và số giây ASR từng mẫu. Điều kiện giống lúc đo: prompt = transcript do chính model sinh cho các lượt user
trước + lời assistant có sẵn (PhoWhisper v1: prompt_ids kiểu `agent/asr_kaggle.py`), beam 5, fp16, greedy theo seed cố định.

    python docker/asr_stage.py --data /data --split private_test --model /models/asr --out /work/asr.json --secs /work/asr_secs.json
Ra: --out {uid: câu} (uid lượt cuối = id mẫu, giống `clef_agent --asr`); --secs {id mẫu: tổng giây ASR của mẫu}.
Chỉ đọc <split>.jsonl và file audio; không mở *_environment/.

Chế độ --serve (dùng bởi /code/predict.py và predict_notebook.ipynb theo hướng dẫn Docker của BTC): nạp model MỘT lần rồi
phục vụ từng mẫu qua stdin/stdout, để ASR nằm trong thời gian đo của chính mẫu đó:
    python docker/asr_stage.py --serve --model /code/saved_models/asr
    vào (mỗi dòng một JSON): {"row": <một dòng của <split>.jsonl>, "data": "/data", "split": "private_test"}   |   {"cmd": "quit"}
    ra  (dòng bắt đầu bằng "@@ASR "): {"ready": true, "load_s": ...} rồi mỗi yêu cầu một {"hyps": {uid: câu}, "errors": {...}, "secs": ...}
Điều kiện giải mã giống hệt chế độ theo lô (cùng hàm transcribe_row).

ASR_MODE=v1 (mặc định, e46f). Hook `v21` cho ASR v3 dùng prompt thiên vị từ khoá: gọi asr_srv/asr_infer_v2.run — chưa nối (xem README).
"""
import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))
from asr_manifest import items_of                      # noqa: E402

SOP = "<|startofprev|>"
MAX_PROMPT_TOKENS = 180


def prompt_ids(tok, prev, hyps):
    parts = []
    for role, text, uid in prev:
        t = hyps.get(uid, "") if (role == "user" and uid is not None) else (text or "")
        if t.strip():
            parts.append(t.strip())
    if not parts:
        return []
    ids = tok(" " + " ".join(parts), add_special_tokens=False).input_ids[-MAX_PROMPT_TOKENS:]
    return [tok.convert_tokens_to_ids(SOP)] + ids


def load_model(model_dir, beams):
    """Nạp PhoWhisper (fp16, cuda) → (run(wav, prompt_ids) -> câu, tokenizer). Seed cố định, beam search tất định."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    torch.manual_seed(0)
    proc = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float16).to("cuda").eval()
    sot, tok = model.config.decoder_start_token_id, proc.tokenizer

    def run(wav, pid):
        f = proc.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", dtype=torch.float16)
        kw = {"prompt_ids": torch.tensor(pid, device="cuda")} if pid else {}
        with torch.no_grad():
            ids = model.generate(input_features=f, language="vi", task="transcribe", num_beams=beams, max_new_tokens=200, **kw)[0].tolist()
        if sot in ids:
            ids = ids[ids.index(sot):]
        return proc.batch_decode([ids], skip_special_tokens=True)[0].strip()

    return run, tok


def read_wav(path):
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    assert sr == 16000, sr
    return wav


def transcribe_row(row, data, split, run, tok, hyps, errors):
    """Nghe mọi lượt user của MỘT mẫu (history theo thứ tự, lượt cuối ở cuối); ghi vào hyps/errors, trả số giây đã dùng."""
    t_row = time.time()
    for it in items_of(row):
        try:
            hyps[it["uid"]] = run(read_wav(os.path.join(data, split, it["audio"])), prompt_ids(tok, it["prev"], hyps))
        except Exception as e:  # noqa: BLE001  — một đoạn lỗi không được làm hỏng cả mẫu: để rỗng, FC vẫn chạy tiếp
            hyps[it["uid"]] = ""
            errors[it["uid"]] = f"{type(e).__name__}: {e}"
    return time.time() - t_row


MARK = "@@ASR "          # dòng giao thức trên stdout; mọi dòng khác (log của thư viện) bị phía đọc bỏ qua


def emit(obj):
    sys.stdout.write(MARK + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def serve(a, load=load_model):
    """Nạp model một lần, rồi mỗi dòng stdin là một mẫu. `load` thay được khi kiểm thử không có GPU."""
    t0 = time.time()
    run, tok = load(a.model, a.beams)
    try:                                    # làm nóng (cuDNN, cấp phát) để mẫu đầu không gánh thời gian khởi động
        import numpy as np
        run(np.zeros(16000, dtype="float32"), [])
    except Exception as e:  # noqa: BLE001
        print(f"[asr] làm nóng lỗi (bỏ qua): {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    emit({"ready": True, "load_s": round(time.time() - t0, 3)})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "quit":
                break
            hyps, errors = {}, {}
            secs = transcribe_row(req["row"], req["data"], req["split"], run, tok, hyps, errors)
            emit({"hyps": hyps, "errors": errors, "secs": round(secs, 3)})
        except Exception as e:  # noqa: BLE001  — luôn trả một dòng cho mỗi yêu cầu để phía gọi không treo
            emit({"hyps": {}, "errors": {"_request": f"{type(e).__name__}: {e}"}, "secs": 0.0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--split")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    ap.add_argument("--secs")
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--limit", type=int, help="chỉ N mẫu đầu (chạy thử)")
    ap.add_argument("--serve", action="store_true", help="phục vụ từng mẫu qua stdin/stdout (xem docstring)")
    a = ap.parse_args()
    if os.environ.get("ASR_MODE", "v1") != "v1":
        sys.exit("ASR_MODE khác v1 chưa được nối vào docker/asr_stage.py")
    if a.serve:
        return serve(a)
    for k in ("data", "split", "out", "secs"):
        if not getattr(a, k):
            ap.error(f"--{k} là bắt buộc khi không dùng --serve")
    rows = [json.loads(l) for l in open(os.path.join(a.data, a.split, f"{a.split}.jsonl"), encoding="utf-8") if l.strip()]
    rows = rows[: a.limit]
    run, tok = load_model(a.model, a.beams)
    hyps, secs, errors = {}, collections.defaultdict(float), {}
    t_all = time.time()
    for n, row in enumerate(rows):
        secs[row["id"]] += transcribe_row(row, a.data, a.split, run, tok, hyps, errors)
        if n % 50 == 0:
            print(f"[asr] {n}/{len(rows)} mẫu, {time.time() - t_all:.0f}s", flush=True)
    json.dump(hyps, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump({k: round(v, 3) for k, v in secs.items()}, open(a.secs, "w"))
    print(f"[asr] xong {len(rows)} mẫu, {len(hyps)} đoạn, lỗi {len(errors)}, {time.time() - t_all:.0f}s", flush=True)
    if errors:
        print("[asr] lỗi:", list(errors.items())[:5], flush=True)


if __name__ == "__main__":
    main()
