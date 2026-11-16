"""Luồng suy luận dùng chung cho predict.py và predict_notebook.ipynb.

Mỗi mẫu:  audio (mọi lượt user, theo thứ tự)  →  ASR PhoWhisper-large (e46f)  →  bộ lọc app (BM25 ∪ embedding ∪ luật)
          →  FC Qwen3.5-4B (v5, Clef-token, qua vLLM chạy cục bộ)  →  ToolEnv.validate / call  →  tool_calls + state_hash.

ASR chạy ngay trong tiến trình này (GPU, fp16). FC chạy trong server vLLM nền (127.0.0.1, không cần mạng ngoài) vì vLLM
cần môi trường Python riêng. Mẫu được xử lý tuần tự, một ToolEnv cho mỗi mẫu, nên predict.py và notebook cho cùng kết quả.
"""
import json
import os
import random
import signal
import subprocess
import sys
import time
import types
import urllib.request
from pathlib import Path

CODE = Path(os.environ.get("CODE_DIR", Path(__file__).resolve().parent.parent))
MODELS = Path(os.environ.get("MODELS_DIR", CODE / "saved_models"))
VLLM_BIN = os.environ.get("VLLM_BIN", "/opt/fc/bin/vllm")
PORT = int(os.environ.get("VLLM_PORT", "8000"))
KEY = "local"
SEED = 42

# cấu hình mặc định của bản nộp (BTC không cần đổi)
CFG = {
    "time_budget": float(os.environ.get("TIME_BUDGET", "30")),   # giây/mẫu, tính cả ASR; hết thì nộp các call đã có
    "budget_margin": 2.0,
    "retry_empty": True,                                          # N1: tool đọc trả rỗng → tìm lại, bỏ tham số chữ tuỳ chọn
    "asr_beams": 5,
    "asr_max_prompt_tokens": 180,
    "gpu_util": float(os.environ.get("GPU_UTIL", "0.6")),        # phần GPU cho vLLM; phần còn lại cho ASR
}

sys.path.insert(0, str(CODE / "src"))


def seed_everything(seed=SEED):
    import numpy as np
    import torch
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_split(root):
    """/data/<split>/ chứa <split>_tools_catalog.json (như guide BTC). Trả (root, split)."""
    root = Path(root)
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / f"{p.name}_tools_catalog.json").exists():
            return root, p.name
    # phòng trường hợp /data chính là thư mục split
    cats = list(root.glob("*_tools_catalog.json"))
    if cats:
        split = cats[0].name[: -len("_tools_catalog.json")]
        return root.parent, split
    raise FileNotFoundError(f"không thấy <split>/<split>_tools_catalog.json trong {root}")


def load_items(root, split):
    with open(Path(root) / split / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- ASR
class ASR:
    SOP = "<|startofprev|>"

    def __init__(self, model_dir):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.torch = torch
        self.proc = WhisperProcessor.from_pretrained(model_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float16).to("cuda").eval()
        self.tok = self.proc.tokenizer
        self.sot = self.model.config.decoder_start_token_id
        self.sop = self.tok.convert_tokens_to_ids(self.SOP)

    def _prompt_ids(self, prev, hyps):
        parts = []
        for role, text, uid in prev:
            t = hyps.get(uid, "") if (role == "user" and uid is not None) else (text or "")
            if t.strip():
                parts.append(t.strip())
        if not parts:
            return []
        ids = self.tok(" " + " ".join(parts), add_special_tokens=False).input_ids[-CFG["asr_max_prompt_tokens"]:]
        return [self.sop] + ids

    def _run(self, wav, pid):
        torch = self.torch
        f = self.proc.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", dtype=torch.float16)
        kw = {"prompt_ids": torch.tensor(pid, device="cuda")} if pid else {}
        with torch.no_grad():
            ids = self.model.generate(input_features=f, language="vi", task="transcribe", num_beams=CFG["asr_beams"],
                                      max_new_tokens=200, **kw)[0].tolist()
        if self.sot in ids:
            ids = ids[ids.index(self.sot):]
        return self.proc.batch_decode([ids], skip_special_tokens=True)[0].strip()

    def transcribe_item(self, item, split_dir):
        """→ {uid: transcript} cho mọi lượt user có audio của mẫu (lượt cuối có uid = id mẫu)."""
        import soundfile as sf
        from agent.asr_manifest import items_of
        hyps = {}
        for it in items_of(item):
            try:
                wav, sr = sf.read(os.path.join(split_dir, it["audio"]), dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(1)
                if sr != 16000:
                    raise ValueError(f"sample rate {sr}")
                hyps[it["uid"]] = self._run(wav, self._prompt_ids(it["prev"], hyps))
            except Exception as e:  # noqa: BLE001 — một đoạn lỗi không làm hỏng cả mẫu
                print(f"[asr] lỗi {it['uid']}: {type(e).__name__}: {e}", flush=True)
                hyps[it["uid"]] = ""
        return hyps


# ---------------------------------------------------------------- vLLM (FC)
def _vllm_ready():
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/models", headers={"Authorization": f"Bearer {KEY}"})
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def start_vllm(log_path="/tmp/vllm.log", timeout=900):
    if _vllm_ready():
        return None
    cmd = [VLLM_BIN, "serve", str(MODELS / os.environ.get("FC_DIRNAME", "fc")), "--served-model-name", "llm", "--host", "127.0.0.1", "--port", str(PORT),
           "--max-model-len", "16384", "--gpu-memory-utilization", str(CFG["gpu_util"]), "--seed", "0",
           "--enable-prefix-caching", "--max-num-seqs", "8", "--max-logprobs", "64",
           "--logprobs-mode", "processed_logprobs", "--api-key", KEY]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    env = {**env, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1",
           "DO_NOT_TRACK": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _vllm_ready():
            return proc
        if proc.poll() is not None:
            log.flush()
            raise RuntimeError("vLLM thoát sớm:\n" + Path(log_path).read_text()[-3000:])
        time.sleep(2)
    raise RuntimeError(f"vLLM không lên sau {timeout}s, xem {log_path}")


# ---------------------------------------------------------------- toàn luồng
class Pipeline:
    def __init__(self, root, split):
        self.root, self.split = str(root), split
        self.split_dir = os.path.join(self.root, split)
        t0 = time.time()
        # vLLM trước (chiếm phần GPU cố định), ASR sau
        self.vllm = start_vllm()
        t_vllm = time.time() - t0
        self.asr = ASR(str(MODELS / "asr"))
        # zaic_tools của split (không dùng bản đóng trong image)
        sys.path.insert(0, self.split_dir)
        from zaic_tools import ToolEnv
        from transformers import AutoTokenizer
        from agent import clef_agent as CA
        from agent.clef import fmt as F
        self.ToolEnv, self.CA = ToolEnv, CA
        F.set_style_from(json.load(open(CODE / "config" / "fmt.json", encoding="utf-8")))
        self.tok = AutoTokenizer.from_pretrained(str(MODELS / "tokenizer"))
        self.labels = F.make_labels(self.tok)
        self.catalog = CA.A.load_catalog(os.path.join(self.split_dir, f"{split}_tools_catalog.json"))
        CA.DEPS.update(CA.build_deps(self.catalog))
        self.args = types.SimpleNamespace(
            data=self.root, split=split, url=f"http://127.0.0.1:{PORT}", key=KEY, model_name="llm",
            app_filter=str(MODELS / "filter"), time_budget=CFG["time_budget"], budget_margin=CFG["budget_margin"],
            retry_empty=CFG["retry_empty"], retry_star=False, c3=False, n8_tau=None, i5=None, i5b=None,
            q1_debias=False, ask_tau=None, log_probs=False)
        CA.get_pack(self.args)                         # nạp bộ lọc (BM25 + embedding, CPU) một lần
        self.load_secs = time.time() - t0
        print(f"[load] vLLM {t_vllm:.1f}s, tổng nạp model {self.load_secs:.1f}s", flush=True)

    def run_sample(self, item):
        """→ (tool_calls, state_hash, ms, chi tiết). Thời gian tính từ trước khi đọc audio tới khi có kết quả."""
        t1 = time.perf_counter()
        hyps = self.asr.transcribe_item(item, self.split_dir)
        self.CA.ASR_SECS[item["id"]] = time.perf_counter() - t1
        r = self.CA.run_one(item, self.args, self.tok, self.labels, self.catalog, self.ToolEnv, hyps)
        ms = int((time.perf_counter() - t1) * 1000)
        r["asr_final"] = hyps.get(item["id"], "")
        return r["tool_calls"], r["state_hash"], ms, r

    def close(self):
        if self.vllm is not None and self.vllm.poll() is None:
            try:
                os.killpg(self.vllm.pid, signal.SIGTERM)
                self.vllm.wait(30)
            except Exception:
                try:
                    os.killpg(self.vllm.pid, signal.SIGKILL)
                except Exception:
                    pass
        self.vllm = None
