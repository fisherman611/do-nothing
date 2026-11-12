"""Lối chạy theo "Hướng dẫn dùng docker để nộp bài ZAC 2026 - LifeMate" của BTC.

    bash /code/predict.sh            # = python3 /code/predict.py → /result/submission.jsonl
    predict_notebook.ipynb           # cùng các hàm dưới đây → /result/jupyter_submission.jsonl

Chạy LẦN LƯỢT TỪNG MẪU, model nạp một lần:  t1 → ToolEnv (đúng một cái cho mẫu) → ASR mọi lượt user của mẫu → bộ lọc app
+ FC Clef-token (agent/clef_agent.run_one) ⇄ env.validate / env.call → get_tool_calls + get_state_hash → `time` (ms).
predict.sh và notebook đi qua cùng hàm `infer`, cùng một request tại một thời điểm, nên hai file kết quả phải khớp nhau.

Tiến trình:
  - ASR: `docker/asr_stage.py --serve` trong venv /opt/asr (transformers 4.57.1 như lúc train), giữ model trên GPU.
  - FC: `vllm serve saved_models/fc` (seed 0, greedy). Bật SAU ASR và chỉ lấy GPU_UTIL phần bộ nhớ GPU vì ASR đã nằm sẵn.
Luật: chỉ validate / call / get_tool_calls / get_state_hash; không mở *_environment/; không cần mạng.

Mọi tham số có mặc định ở CFG bên dưới (BTC chạy không truyền gì). Biến môi trường cùng tên chỉ để thử lúc phát triển.
"""
import argparse
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time
import urllib.request

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)

_DEF = {
    "DATA": "/data",                                   # BTC mount thư mục chứa <split>/ vào đây
    "SPLIT": "",                                       # rỗng = tự tìm thư mục có <split>_tools_catalog.json
    "OUT": "/result/submission.jsonl",
    "WORK": "/tmp/lifemate",                           # log vLLM/ASR, run.jsonl chi tiết (không phải kết quả nộp)
    "MODELS": os.path.join(CODE, "saved_models"),      # asr/  fc/  pack/  fmt.json
    "ASR_PY": "/opt/asr/bin/python",
    "ASR_WORKER": os.path.join(CODE, "docker", "asr_stage.py"),
    "ASR_BEAMS": "5",
    "LLM_URL": "",                                     # rỗng = tự bật vLLM tại 127.0.0.1:VLLM_PORT
    "VLLM_PORT": "8000",
    "GPU_UTIL": "0.70",                                # phần bộ nhớ GPU cho vLLM; phần còn lại cho PhoWhisper (đo lại trên GPU đích)
    "SEED": "0",
    "TIME_BUDGET": "30",                               # giây mỗi mẫu, tính cả ASR; hết thì nộp các call đã có
    "BUDGET_MARGIN": "2",
    "RETRY_EMPTY": "1",                                # N1: tool đọc trả rỗng → tìm lại một lần
    "REQ_RETRIES": "2",                                # số lần thử một request tới vLLM (mặc định của clef_agent là 8, chờ tới 30 s/lần)
    "REQ_TIMEOUT": "30",                               # giây chờ một request
    "ASR_TIMEOUT": "20",                               # giây tối đa chờ ASR của một mẫu; quá thì FC chạy với transcript rỗng
    "LIMIT": "",                                       # chỉ N mẫu đầu (chạy thử)
    "GPU_LOG": "",                                     # "1" = ghi nvidia-smi mỗi 5 s vào WORK/gpu_mem.csv
}
CFG = {k: os.environ.get(k, v) for k, v in _DEF.items()}
API_KEY = "local"
MARK = "@@ASR "


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seed_everything(seed=None):
    seed = int(CFG["SEED"] if seed is None else seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:                                               # torch chỉ dùng cho embedder của bộ lọc app (CPU); ASR và vLLM tự đặt seed
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


def detect_split(data):
    """Thư mục con của /data có <tên>_tools_catalog.json (như notebook mẫu của BTC); nhiều hơn một thì ưu tiên private_test."""
    if CFG["SPLIT"]:
        return CFG["SPLIT"]
    found = sorted(d for d in os.listdir(data) if os.path.isfile(os.path.join(data, d, f"{d}_tools_catalog.json")))
    if not found:
        raise FileNotFoundError(f"không thấy <split>/<split>_tools_catalog.json trong {data}")
    return "private_test" if "private_test" in found else found[0]


# ---------------------------------------------------------------- ASR (tiến trình riêng, nạp một lần)
class AsrWorker:
    def __init__(self, work):
        self.work, self.p, self.q, self.load_s = work, None, None, None

    def start(self):
        cmd = [CFG["ASR_PY"], CFG["ASR_WORKER"], "--serve", "--model", os.path.join(CFG["MODELS"], "asr"),
               "--beams", CFG["ASR_BEAMS"]]
        self.err = open(os.path.join(self.work, "asr.log"), "a", encoding="utf-8")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True,
                                  encoding="utf-8", bufsize=1)
        self.q = queue.Queue()
        threading.Thread(target=self._pump, args=(self.p, self.q), daemon=True).start()
        msg = self._read(900)                           # nạp model lần đầu
        if not (msg and msg.get("ready")):
            raise RuntimeError(f"ASR không khởi động được (xem {self.work}/asr.log): {msg}")
        self.load_s = msg.get("load_s")

    @staticmethod
    def _pump(p, q):
        for line in p.stdout:                           # chỉ nhận dòng giao thức; log của thư viện bị bỏ qua
            if line.startswith(MARK):
                try:
                    q.put(json.loads(line[len(MARK):]))
                except ValueError:
                    pass
        q.put(None)                                     # tiến trình đã thoát

    def _read(self, timeout):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def transcribe(self, row, data, split, timeout):
        """→ ({uid: câu}, {uid: lỗi}). Không bao giờ ném lỗi: ASR hỏng thì trả rỗng để FC vẫn chạy (và thử bật lại ASR)."""
        try:
            if self.p is None or self.p.poll() is not None:
                self.start()
            self.p.stdin.write(json.dumps({"row": row, "data": data, "split": split}, ensure_ascii=False) + "\n")
            self.p.stdin.flush()
            msg = self._read(timeout)
            if msg is None:                             # quá giờ hoặc tiến trình chết: bỏ tiến trình này, mẫu sau bật lại
                self.close(kill=True)
                return {}, {"_asr": "timeout hoặc tiến trình ASR thoát"}
            return msg.get("hyps") or {}, msg.get("errors") or {}
        except Exception as e:  # noqa: BLE001
            self.close(kill=True)
            return {}, {"_asr": f"{type(e).__name__}: {e}"}

    def close(self, kill=False):
        p, self.p = self.p, None
        if p is None:
            return
        try:
            if not kill and p.poll() is None:
                p.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                p.stdin.flush()
                p.wait(timeout=20)
        except Exception:  # noqa: BLE001
            pass
        if p.poll() is None:
            p.kill()


# ---------------------------------------------------------------- vLLM (FC)
def _llm_up(url):
    try:
        req = urllib.request.Request(url.rstrip("/") + "/v1/models", headers={"Authorization": f"Bearer {API_KEY}"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def start_vllm(work):
    """→ (url, Popen | None). LLM_URL đặt sẵn (thử lúc phát triển) thì không bật gì."""
    if CFG["LLM_URL"]:
        return CFG["LLM_URL"], None
    url = f"http://127.0.0.1:{CFG['VLLM_PORT']}"
    logf = open(os.path.join(work, "vllm.log"), "a", encoding="utf-8")
    cmd = ["vllm", "serve", os.path.join(CFG["MODELS"], "fc"), "--served-model-name", "llm", "--host", "127.0.0.1",
           "--port", CFG["VLLM_PORT"], "--max-model-len", "16384", "--gpu-memory-utilization", CFG["GPU_UTIL"],
           "--seed", CFG["SEED"], "--enable-prefix-caching", "--max-logprobs", "64",
           "--logprobs-mode", "processed_logprobs", "--api-key", API_KEY]
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    for _ in range(450):                                # tối đa 15 phút
        if _llm_up(url):
            return url, p
        if p.poll() is not None:
            break
        time.sleep(2)
    tail = open(os.path.join(work, "vllm.log"), encoding="utf-8", errors="replace").read()[-3000:]
    if p.poll() is None:
        p.kill()
    raise RuntimeError("vLLM không lên:\n" + tail)


def ensure_vllm(rt):
    """vLLM do mình bật mà đã thoát (OOM, lỗi driver) → bật lại một lần cho mẫu này thay vì để mọi mẫu sau chờ rồi nộp rỗng."""
    if rt.vllm is None or rt.vllm.poll() is None:
        return
    log(f"CẢNH BÁO: vLLM đã thoát (mã {rt.vllm.returncode}) → bật lại")
    try:
        _, rt.vllm = start_vllm(rt.work)
    except Exception as e:  # noqa: BLE001
        log(f"bật lại vLLM thất bại: {e}")


# ---------------------------------------------------------------- nạp model + tài nguyên
class Runtime:
    pass


_RT = None


def load_models():
    """Bước 2 của notebook: nạp ASR, vLLM, tokenizer, catalog, bộ lọc app. Gọi lại lần nữa thì trả runtime đang sống."""
    global _RT
    if _RT is not None and not _RT.closed:
        return _RT
    t0 = time.perf_counter()
    rt = Runtime()
    rt.closed, rt.vllm, rt.gpu_log = False, None, None
    rt.data, rt.work = CFG["DATA"], CFG["WORK"]
    os.makedirs(rt.work, exist_ok=True)
    rt.split = detect_split(rt.data)
    log(f"split = {rt.split} | data = {rt.data}")
    if CFG["GPU_LOG"]:
        try:
            rt.gpu_log = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,memory.used,memory.total", "--format=csv,noheader",
                                           "-l", "5"], stdout=open(os.path.join(rt.work, "gpu_mem.csv"), "w"),
                                          stderr=subprocess.DEVNULL)
        except OSError:
            pass

    rt.asr = AsrWorker(rt.work)                         # 1) ASR trước: chiếm phần GPU của nó
    rt.asr.start()
    log(f"ASR sẵn sàng ({rt.asr.load_s} s)")
    url, rt.vllm = start_vllm(rt.work)                  # 2) vLLM lấy GPU_UTIL phần còn lại
    log(f"vLLM sẵn sàng tại {url}")

    from agent import clef_agent as C                   # 3) phần FC: cùng mã với lúc đo (clef_agent.main)
    rt.C, F, A = C, C.F, C.A
    C.REQ_RETRIES, C.REQ_TIMEOUT = int(CFG["REQ_RETRIES"]), float(CFG["REQ_TIMEOUT"])
    fmt_path = os.path.join(CFG["MODELS"], "fmt.json")
    if os.path.isfile(fmt_path):
        m = json.load(open(fmt_path, encoding="utf-8"))
        F.set_style_from(m.get("fmt", m))
    else:
        log(f"CẢNH BÁO: thiếu {fmt_path} → dùng format mặc định (lệch format lúc train thì điểm tụt)")
        F.set_style_from({})
    pack_dir = os.path.join(CFG["MODELS"], "pack")
    if not os.path.isdir(pack_dir):
        log(f"CẢNH BÁO: thiếu {pack_dir} → chạy KHÔNG có bộ lọc app (catalog một tầng)")
        pack_dir = None
    log(f"format: {F.fmt_opts()} | bộ lọc app: {pack_dir}")
    rt.args = argparse.Namespace(
        data=rt.data, split=rt.split, url=url, key=API_KEY, model_name="llm", app_filter=pack_dir,
        time_budget=float(CFG["TIME_BUDGET"]), budget_margin=float(CFG["BUDGET_MARGIN"]),
        retry_empty=CFG["RETRY_EMPTY"] == "1", log_probs=False, ask_tau=None, n8_tau=None, i5=None)
    sys.path.insert(0, os.path.join(rt.data, rt.split))  # zaic_tools CỦA SPLIT, không dùng bản đóng trong image
    from zaic_tools import ToolEnv
    from transformers import AutoTokenizer
    rt.ToolEnv = ToolEnv
    rt.tok = AutoTokenizer.from_pretrained(C.TOKENIZER[0], revision=C.TOKENIZER[1])
    rt.labels = F.make_labels(rt.tok)
    rt.catalog = A.load_catalog(os.path.join(rt.data, rt.split, f"{rt.split}_tools_catalog.json"))
    C.DEPS.update(C.build_deps(rt.catalog))
    if pack_dir:
        C.get_pack(rt.args)                             # nạp embedder ngay, không để mẫu đầu gánh
    try:                                                # làm nóng vLLM (không ảnh hưởng kết quả: greedy, seed cố định)
        C.request_json(url, API_KEY, "llm", "Xin chào", 1, retries=1)
    except Exception as e:  # noqa: BLE001
        log(f"làm nóng vLLM lỗi (bỏ qua): {type(e).__name__}: {e}")
    rt.load_ms = int((time.perf_counter() - t0) * 1000)
    rt.records = []
    print(f"load model: {rt.load_ms} ms", flush=True)
    _RT = rt
    return rt


def read_cases(rt):
    """Bước 3 của notebook: đọc <split>.jsonl."""
    path = os.path.join(rt.data, rt.split, f"{rt.split}.jsonl")
    cases = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if CFG["LIMIT"]:
        cases = cases[: int(CFG["LIMIT"])]
    return cases


# ---------------------------------------------------------------- một mẫu
def infer(rt, item):
    """→ (tool_calls, state_hash, ms). Đồng hồ bắt đầu trước khi đọc audio; đúng một ToolEnv cho mẫu."""
    t1 = time.perf_counter()
    C = rt.C
    env = rt.ToolEnv(rt.split, item["id"], root=rt.data)
    hyps, asr_err = rt.asr.transcribe(item, rt.data, rt.split, float(CFG["ASR_TIMEOUT"]))
    asr_s = time.perf_counter() - t1
    ensure_vllm(rt)
    C.ASR_SECS[item["id"]] = asr_s                      # run_one trừ vào ngân sách TIME_BUDGET của mẫu
    rec = {}
    try:                                                # run_one nhận "hàm tạo ToolEnv": đưa lại env đã tạo ở trên
        rec = C.run_one(item, rt.args, rt.tok, rt.labels, rt.catalog, lambda *a, **k: env, hyps)
    except Exception as e:  # noqa: BLE001  — vẫn nộp các call env đã ghi
        rec = {"error": f"{type(e).__name__}: {e}"}
    tool_calls = [{"name": c["name"], "arguments": c.get("arguments", {})} for c in env.get_tool_calls()]
    state_hash = env.get_state_hash()
    ms = int((time.perf_counter() - t1) * 1000)
    rt.records.append({"id": item["id"], "time": ms, "asr_s": round(asr_s, 3), "asr": hyps, "asr_errors": asr_err,
                       **{k: rec.get(k) for k in ("steps", "n_invalid", "n_retry", "timed_out", "n_requests", "anomaly",
                                                  "latency", "error", "decisions")}})
    return tool_calls, state_hash, ms


def run_all(rt, cases, out_path):
    """Bước 4: lần lượt từng mẫu, ghi từng dòng {id, tool_calls, state_hash, time}. In thống kê thời gian ra stdout."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    times, t0 = [], time.perf_counter()
    with open(out_path, "w", encoding="utf-8") as f:
        for n, item in enumerate(cases, 1):
            tool_calls, state_hash, ms = infer(rt, item)
            f.write(json.dumps({"id": item["id"], "tool_calls": tool_calls, "state_hash": state_hash, "time": ms},
                               ensure_ascii=False) + "\n")
            f.flush()
            times.append(ms)
            if n % 20 == 0 or n == len(cases):
                log(f"{n}/{len(cases)} mẫu | mẫu vừa xong {ms} ms | trung bình {sum(times) / len(times):.0f} ms")
    report(rt, times, time.perf_counter() - t0, out_path)
    return times


def report(rt, times, total_s, out_path):
    s = sorted(times)
    if s:
        over = sum(t > float(CFG["TIME_BUDGET"]) * 1000 for t in s)
        print(f"predict: {len(s)} mẫu, {int(total_s * 1000)} ms, trung bình {sum(s) / len(s):.1f} ms/mẫu | "
              f"p50 {s[len(s) // 2]} ms, p95 {s[min(len(s) - 1, int(len(s) * .95))]} ms, max {s[-1]} ms | "
              f"quá {CFG['TIME_BUDGET']} s: {over} mẫu", flush=True)
    errs = [r["id"] for r in rt.records if r.get("error")]
    if errs:
        print(f"mẫu có lỗi agent (vẫn nộp call + hash đã có): {len(errs)} {errs[:10]}", flush=True)
    print(f"output will be saved in {out_path}", flush=True)
    try:                                                # log chi tiết để chẩn đoán; không phải file nộp
        tag = os.path.splitext(os.path.basename(out_path))[0]
        with open(os.path.join(rt.work, f"{tag}.run.jsonl"), "w", encoding="utf-8") as f:
            for r in rt.records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    except OSError:
        pass


def shutdown(rt=None):
    """Tắt ASR và vLLM, nhả GPU."""
    global _RT
    rt = rt or _RT
    if rt is None or rt.closed:
        return
    rt.closed = True
    rt.asr.close()
    for p in (rt.vllm, rt.gpu_log):
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
    if _RT is rt:
        _RT = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=CFG["OUT"])
    ap.add_argument("--limit", type=int, help="chỉ N mẫu đầu (chạy thử)")
    a = ap.parse_args()
    if a.limit:
        CFG["LIMIT"] = str(a.limit)
    seed_everything()
    rt = load_models()
    try:
        cases = read_cases(rt)
        log(f"{len(cases)} mẫu → {a.out}")
        run_all(rt, cases, a.out)
    finally:
        shutdown(rt)


if __name__ == "__main__":
    main()
