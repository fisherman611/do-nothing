"""Trình xem data train (mẫu gốc, phonetic, alias, iface, val_alias) + ví dụ Clef đã build. Chỉ chạy local.

    python tools/data_viewer/server.py [--port 8765]      (từ thư mục zaic/)  → mở http://127.0.0.1:8765

Chỉ đọc: data/train+public_test/train/ (train.jsonl, audio), synth*/ *.jsonl, agent/clef/data/*.jsonl, splits.json.
Không đọc gì trong public_test/ hay *_environment/.
"""
import argparse
import glob
import json
import os
import re
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(ROOT, "data", "train+public_test", "train")
ORIG = "data/train+public_test/train/train.jsonl"


def sample_files():
    fs = [ORIG] + sorted(glob.glob("synth_phonetic/*.jsonl", root_dir=ROOT)) + sorted(glob.glob("synth/*.jsonl", root_dir=ROOT))
    return [f for f in fs if os.path.exists(os.path.join(ROOT, f))]


def clef_files():
    fs = glob.glob("agent/clef/data/*.jsonl", root_dir=ROOT) + glob.glob("agent/clef/data/*/*.jsonl", root_dir=ROOT)
    return sorted(f for f in fs if "/old_" not in f)


def clef_meta(rel):
    p = os.path.join(ROOT, rel + ".meta.json")
    return json.load(open(p)) if os.path.exists(p) else {}


_cache, _clef_idx = {}, {}


def load_samples(rel):
    p = os.path.join(ROOT, rel)
    key = (rel, os.path.getmtime(p))
    if key not in _cache:
        with open(p, encoding="utf-8") as f:
            _cache[key] = [json.loads(l) for l in f if l.strip()]
    return _cache[key]


ID_RE = re.compile(rb'^\{"id": "([^"]+)"')


def clef_steps(rel, sid):
    p = os.path.join(ROOT, rel)
    key = (rel, os.path.getmtime(p))
    if key not in _clef_idx:                      # chỉ mục offset theo id, dựng một lần (file ~200MB)
        idx, off = {}, 0
        with open(p, "rb") as f:
            for line in f:
                m = ID_RE.match(line)
                if m:
                    idx.setdefault(m.group(1).decode(), []).append(off)
                off += len(line)
        _clef_idx[key] = idx
    out = []
    with open(p, "rb") as f:
        for off in _clef_idx[key].get(sid, []):
            f.seek(off)
            e = json.loads(f.readline())
            out.append({"step": e["step"], "name": e["name"], "args": e["args"], "prefix_chars": e["prefix_chars"],
                        "prefix_tail": e["text"][max(0, e["prefix_chars"] - 1500):e["prefix_chars"]],
                        "body": e["text"][e["prefix_chars"]:], "spans": e["spans"]})
    return out


def clef_ids(rel):
    clef_steps(rel, "")                           # dựng chỉ mục
    p = os.path.join(ROOT, rel)
    return sorted(_clef_idx[(rel, os.path.getmtime(p))])


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        try:
            if u.path == "/":
                return self.send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
            if u.path == "/api/files":
                return self.send(200, {"samples": sample_files(), "clef": clef_files(), "orig": ORIG,
                                       "clef_meta": {f: clef_meta(f) for f in clef_files()}})
            if u.path == "/api/splits":
                return self.send(200, json.load(open(os.path.join(ROOT, "agent", "clef", "splits.json"))))
            if u.path == "/api/samples":
                if q.get("file") not in sample_files():
                    return self.send(404, {"error": "file"})
                return self.send(200, load_samples(q["file"]))
            if u.path == "/api/clef":
                if q.get("file") not in clef_files():
                    return self.send(404, {"error": "file"})
                return self.send(200, clef_steps(q["file"], q.get("id", "")))
            if u.path == "/api/clef_ids":
                if q.get("file") not in clef_files():
                    return self.send(404, {"error": "file"})
                return self.send(200, clef_ids(q["file"]))
            if u.path.startswith("/audio/"):
                rel = urllib.parse.unquote(u.path[len("/audio/"):])
                p = os.path.realpath(os.path.join(TRAIN_DIR, rel))
                if not p.startswith(os.path.realpath(TRAIN_DIR) + os.sep) or not p.endswith(".wav") or not os.path.exists(p):
                    return self.send(404, b"", "text/plain")
                return self.send(200, open(p, "rb").read(), "audio/wav")
            return self.send(404, {"error": "not found"})
        except Exception as e:                    # noqa: BLE001
            return self.send(500, {"error": repr(e)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    print(f"http://127.0.0.1:{a.port}  (Ctrl+C để dừng)")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
