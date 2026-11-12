"""Tự kiểm image, KHÔNG cần GPU và không cần /data. Chạy lúc build (Dockerfile) và chạy tay được:  python3 /code/selfcheck.py
Bắt các lỗi chỉ lộ ra khi đã ở máy chấm: thiếu file, thiếu pyvi, tokenizer không nạp được khi tắt mạng, bộ lọc app hỏng."""
import json
import os
import sys

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
M = os.path.join(CODE, "saved_models")
bad = []


def check(name, fn):
    try:
        print(f"[ok] {name}: {fn()}", flush=True)
    except Exception as e:  # noqa: BLE001
        bad.append(name)
        print(f"[LỖI] {name}: {type(e).__name__}: {e}", flush=True)


def files():
    need = ["predict.sh", "predict.py", "start_jupyter.sh", "predict_notebook.ipynb", "README.md", "requirements.txt",
            "training_code/README.md", "docker/asr_stage.py", "agent/clef_agent.py", "saved_models/fmt.json",
            "saved_models/pack/pack.json", "saved_models/pack/keywords.json", "saved_models/asr/config.json",
            "saved_models/fc/config.json"]
    miss = [p for p in need if not os.path.exists(os.path.join(CODE, p))]
    assert not miss, f"thiếu {miss}"
    return f"{len(need)} file/thư mục bắt buộc"


def pyvi():
    from pyvi import ViTokenizer
    s = ViTokenizer.tokenize("Trường đại học bách khoa hà nội")
    assert "_" in s, s
    return s


def fmt():
    from agent import clef_agent as C
    m = json.load(open(os.path.join(M, "fmt.json"), encoding="utf-8"))
    C.F.set_style_from(m.get("fmt", m))
    o = C.F.fmt_opts()
    assert o["sys"] == "compact", f"bộ lọc app cần sys=compact, đang là {o}"
    return o


def tokenizer():
    from agent import clef_agent as C
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.TOKENIZER[0], revision=C.TOKENIZER[1])
    n = len(C.F.make_labels(tok))
    assert n >= 150, n
    return f"{type(tok).__name__}, {n} nhãn một token"


def app_filter():
    sys.path.insert(0, os.path.join(CODE, "agent", "clef", "app_filter"))
    from pack import AppFilterPack, tier1_for
    pack = AppFilterPack.load(os.path.join(M, "pack"))
    emb = pack.filt.emb
    phobert = any(k in emb.name.lower() for k in ("vietnamese-bi-encoder", "vietnamese-embedding", "phobert", "vietnamese-sbert"))
    assert emb.seg or not phobert, f"embedder {emb.name} gốc PhoBERT nhưng KHÔNG tách từ bằng pyvi"
    apps = sorted(emb.apps)
    item = {"id": "x", "audio_text": "đặt lịch họp với anh Nam sáng mai rồi nhắn cho mẹ", "history": [],
            "context": {"installed_apps": {a: {"tools": []} for a in apps}}}
    keep = tier1_for(pack, item)
    assert keep, "bộ lọc không giữ app nào"
    return f"{emb.name}, pyvi={emb.seg}, {len(apps)} app trong gói, k_bm25={pack.k_bm25} k_emb={pack.k_emb}, giữ {sorted(keep)[:6]}…"


def jupyter():
    import jupyterlab
    nb = json.load(open(os.path.join(CODE, "predict_notebook.ipynb"), encoding="utf-8"))
    return f"jupyterlab {jupyterlab.__version__}, notebook {sum(c['cell_type'] == 'code' for c in nb['cells'])} ô code"


def predict_module():
    import predict
    return f"predict.py nạp được; mặc định OUT={predict._DEF['OUT']} TIME_BUDGET={predict._DEF['TIME_BUDGET']} GPU_UTIL={predict._DEF['GPU_UTIL']}"


for name, fn in (("file", files), ("pyvi", pyvi), ("định dạng prompt", fmt), ("tokenizer (offline)", tokenizer),
                 ("bộ lọc app", app_filter), ("jupyter", jupyter), ("predict.py", predict_module)):
    check(name, fn)
if bad:
    sys.exit(f"selfcheck THẤT BẠI: {bad}")
print("selfcheck: đạt")
