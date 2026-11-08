"""Giao diện local để nghe audio và sửa/duyệt transcript của Qwen cloud.

Chạy:
    python asr/review_ui.py
rồi mở http://127.0.0.1:8765

- Trái: audio. Phải: text prediction (sửa được).
- "Lưu nháp" ghi vào review_state.json.
- Tick "Đã chuẩn" ghi ngay vào final_reviewed.jsonl (bỏ tick thì bị gỡ khỏi file).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "data/results/asr/qwen-cloud/public-test-asr-flash"
DEFAULT_AUDIO_ROOT = ROOT / "data/train+public_test/public_test"

LOCK = threading.Lock()


def is_suspicious(text: str, seconds: float) -> bool:
    cps = len(text) / max(seconds, 0.1)
    return bool(
        not text.strip()
        or re.search(r"[一-鿿]", text)
        or re.search(r"(\b\w+\b)( \1){2,}", text.lower())
        or cps > 22
        or (cps < 3 and seconds > 2)
    )


def atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


class Store:
    def __init__(self, results: Path, audio_root: Path):
        self.audio_root = audio_root
        self.state_path = results / "review_state.json"
        self.final_path = results / "final_reviewed.jsonl"
        self.samples: list[dict] = []
        with (results / "predictions.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                p = json.loads(line)
                self.samples.append(
                    {
                        "id": p["id"],
                        "sample_id": p.get("sample_id", p["id"]),
                        "audio": p["audio"],
                        "seconds": p["audio_seconds"],
                        "pred": p["text"],
                        "suspicious": is_suspicious(p["text"], p["audio_seconds"]),
                    }
                )
        self.by_id = {s["id"]: s for s in self.samples}
        self.state: dict[str, dict] = {}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))

    def payload(self) -> list[dict]:
        out = []
        for s in self.samples:
            st = self.state.get(s["id"], {})
            out.append(
                {
                    **s,
                    "text": st.get("text", s["pred"]),
                    "edited": "text" in st and st["text"] != s["pred"],
                    "approved": bool(st.get("approved")),
                }
            )
        return out

    def save(self, sid: str, text: str, approved: bool) -> None:
        if sid not in self.by_id:
            raise KeyError(sid)
        self.state[sid] = {"text": text, "approved": approved}
        atomic_write(self.state_path, json.dumps(self.state, ensure_ascii=False, indent=1))
        self._write_final()

    def _write_final(self) -> None:
        lines = []
        for s in self.samples:  # giữ thứ tự gốc, ổn định
            st = self.state.get(s["id"])
            if not st or not st.get("approved"):
                continue
            lines.append(
                json.dumps(
                    {
                        "id": s["id"],
                        "sample_id": s["sample_id"],
                        "audio": s["audio"],
                        "audio_text": st["text"].strip(),
                        "original_text": s["pred"],
                        "source": "human_review",
                        "reference_type": "human_verified",
                        "human_reviewed": True,
                    },
                    ensure_ascii=False,
                )
            )
        atomic_write(self.final_path, "\n".join(lines) + ("\n" if lines else ""))


class Handler(BaseHTTPRequestHandler):
    store: Store

    def log_message(self, *a):  # im lặng
        pass

    def _send(self, code, body: bytes, ctype: str, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/":
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/samples":
            with LOCK:
                body = json.dumps(self.store.payload(), ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
        elif path.startswith("/audio/"):
            self._audio(path[len("/audio/"):])
        else:
            self._send(404, b"not found", "text/plain")

    def _audio(self, sid: str):
        s = self.store.by_id.get(sid)
        if not s:
            return self._send(404, b"unknown id", "text/plain")
        root = self.store.audio_root.resolve()
        f = (root / s["audio"]).resolve()
        if root not in f.parents or not f.is_file():
            return self._send(404, b"missing audio", "text/plain")
        data = f.read_bytes()
        size = len(data)
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m and (m.group(1) or m.group(2)):
            start = int(m.group(1)) if m.group(1) else max(size - int(m.group(2)), 0)
            end = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
            end = min(end, size - 1)
            chunk = data[start : end + 1]
            return self._send(
                HTTPStatus.PARTIAL_CONTENT, chunk, "audio/wav",
                {"Content-Range": f"bytes {start}-{end}/{size}", "Accept-Ranges": "bytes"},
            )
        self._send(200, data, "audio/wav", {"Accept-Ranges": "bytes"})

    def do_POST(self):
        if urlparse(self.path).path != "/api/save":
            return self._send(404, b"not found", "text/plain")
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            with LOCK:
                self.store.save(req["id"], str(req["text"]), bool(req["approved"]))
            self._send(200, b'{"ok":true}', "application/json")
        except Exception as e:  # noqa: BLE001
            self._send(400, json.dumps({"ok": False, "error": str(e)}).encode(), "application/json")


HTML = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ASR Review</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1c1f24;--mut:#6b7280;--bd:#dfe3e8;--ac:#2563eb;--ok:#16a34a;--warn:#d97706}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--card:#1d2026;--fg:#e6e8eb;--mut:#9aa1ab;--bd:#2d323a;--ac:#6ea0ff;--ok:#4ade80;--warn:#fbbf24}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:10px 16px;border-bottom:1px solid var(--bd);background:var(--card);position:sticky;top:0}
header .sp{flex:1}
select,button,input[type=number]{font:inherit;color:inherit;background:var(--card);border:1px solid var(--bd);border-radius:6px;padding:5px 10px}
button{cursor:pointer}button:hover{border-color:var(--ac)}
button.primary{background:var(--ac);color:#fff;border-color:var(--ac)}
main{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px;max-width:1200px;margin:0 auto}
@media(max-width:760px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px}
h2{margin:0 0 10px;font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut)}
audio{width:100%}
.meta{color:var(--mut);font-size:13px;margin-top:10px;word-break:break-all}
.badge{display:inline-block;font-size:12px;padding:1px 8px;border-radius:99px;border:1px solid var(--bd);margin-right:6px}
.badge.warn{color:var(--warn);border-color:var(--warn)}.badge.ok{color:var(--ok);border-color:var(--ok)}
textarea{width:100%;min-height:150px;font:inherit;font-size:17px;line-height:1.6;color:inherit;background:var(--bg);border:1px solid var(--bd);border-radius:8px;padding:10px;resize:vertical}
textarea:focus{outline:2px solid var(--ac)}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:12px}
label.tick{display:flex;gap:8px;align-items:center;font-weight:600;cursor:pointer}
label.tick input{width:20px;height:20px;accent-color:var(--ok)}
.orig{margin-top:12px;font-size:13px;color:var(--mut)}
#status{color:var(--mut);font-size:13px}
kbd{border:1px solid var(--bd);border-radius:4px;padding:0 5px;font-size:12px}
</style></head><body>
<header>
  <button id="prev">←</button>
  <span id="pos">–</span>
  <button id="next">→</button>
  <select id="filter">
    <option value="todo">Chưa duyệt</option>
    <option value="sus">Nghi ngờ (chưa duyệt)</option>
    <option value="done">Đã duyệt</option>
    <option value="all">Tất cả</option>
  </select>
  <span class="sp"></span>
  <span id="prog"></span>
</header>
<main>
  <section class="card">
    <h2>Audio</h2>
    <audio id="player" controls></audio>
    <div class="row"><button id="replay">▶ Nghe lại <kbd>Ctrl+Space</kbd></button>
      <label><input type="checkbox" id="auto" checked> tự phát khi chuyển</label>
      <label>tốc độ <select id="rate"><option>0.75</option><option selected>1</option><option>1.25</option><option>1.5</option></select></label></div>
    <div class="meta" id="meta"></div>
  </section>
  <section class="card">
    <h2>Transcript</h2>
    <div id="badges"></div>
    <textarea id="text" spellcheck="false"></textarea>
    <div class="orig" id="orig"></div>
    <div class="row">
      <button id="save">Lưu nháp <kbd>Ctrl+S</kbd></button>
      <button id="reset">Khôi phục bản gốc</button>
      <label class="tick"><input type="checkbox" id="approved"> Đã chuẩn</label>
      <button class="primary" id="approveNext">Chuẩn &amp; tiếp <kbd>Ctrl+Enter</kbd></button>
      <span id="status"></span>
    </div>
  </section>
</main>
<script>
const $=id=>document.getElementById(id);
let all=[],view=[],cur=0;
const key='asr_review_filter';
try{$('filter').value=localStorage.getItem(key)||'todo'}catch(e){}

async function load(keepId){
  all=await (await fetch('/api/samples')).json();
  rebuild(keepId);
}
function rebuild(keepId){
  const f=$('filter').value;
  view=all.filter(s=>f==='all'||(f==='todo'&&!s.approved)||(f==='sus'&&!s.approved&&s.suspicious)||(f==='done'&&s.approved));
  const i=keepId?view.findIndex(s=>s.id===keepId):-1;
  cur=i>=0?i:Math.min(cur,Math.max(view.length-1,0));
  show();
}
function progress(){
  const d=all.filter(s=>s.approved).length;
  $('prog').textContent=`Đã duyệt ${d}/${all.length}`;
}
function show(autoplay=false){
  progress();
  const s=view[cur];
  $('pos').textContent=view.length?`${cur+1} / ${view.length}`:'0 / 0';
  if(!s){$('text').value='';$('meta').textContent='Không có mẫu nào trong bộ lọc này.';$('player').removeAttribute('src');$('badges').innerHTML='';$('orig').textContent='';return}
  $('player').src='/audio/'+encodeURIComponent(s.id);
  $('player').playbackRate=+$('rate').value;
  $('text').value=s.text;
  $('approved').checked=s.approved;
  $('meta').textContent=`${s.id} · ${s.seconds.toFixed(1)}s · ${s.audio}`;
  $('badges').innerHTML=(s.suspicious?'<span class="badge warn">nghi ngờ</span>':'')+(s.edited?'<span class="badge">đã sửa</span>':'')+(s.approved?'<span class="badge ok">đã chuẩn</span>':'');
  $('orig').textContent=s.edited?'Bản gốc: '+s.pred:'';
  $('status').textContent='';
  $('text').focus();
  if($('auto').checked){$('player').play().catch(()=>{})}
}
async function persist(approved){
  const s=view[cur];if(!s)return false;
  const text=$('text').value;
  const r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:s.id,text,approved})});
  if(!(await r.json()).ok){$('status').textContent='Lỗi lưu!';return false}
  s.text=text;s.approved=approved;s.edited=text!==s.pred;
  $('status').textContent=approved?'Đã lưu vào file final ✔':'Đã lưu nháp ✔';
  $('badges').innerHTML=(s.suspicious?'<span class="badge warn">nghi ngờ</span>':'')+(s.edited?'<span class="badge">đã sửa</span>':'')+(s.approved?'<span class="badge ok">đã chuẩn</span>':'');
  $('orig').textContent=s.edited?'Bản gốc: '+s.pred:'';
  progress();return true;
}
function go(d){ if(!view.length)return; cur=Math.min(Math.max(cur+d,0),view.length-1); show(); }
async function approveNext(){
  const s=view[cur];if(!s)return;
  if(!await persist(true))return;
  // mẫu vừa duyệt rời khỏi danh sách "chưa duyệt": giữ nguyên vị trí để mẫu kế tiếp trượt lên
  const f=$('filter').value;
  if(f==='todo'||f==='sus'){view.splice(cur,1);if(cur>=view.length)cur=Math.max(view.length-1,0)}
  else if(cur<view.length-1)cur++;
  show();
}
$('prev').onclick=()=>go(-1);$('next').onclick=()=>go(1);
$('filter').onchange=()=>{try{localStorage.setItem(key,$('filter').value)}catch(e){};cur=0;rebuild()};
$('save').onclick=()=>persist($('approved').checked);
$('approved').onchange=()=>persist($('approved').checked);
$('approveNext').onclick=approveNext;
$('reset').onclick=()=>{const s=view[cur];if(s){$('text').value=s.pred}};
$('replay').onclick=()=>{$('player').currentTime=0;$('player').play()};
$('rate').onchange=()=>{$('player').playbackRate=+$('rate').value};
document.addEventListener('keydown',e=>{
  if(e.ctrlKey&&e.key==='Enter'){e.preventDefault();approveNext()}
  else if(e.ctrlKey&&e.key.toLowerCase()==='s'){e.preventDefault();persist($('approved').checked)}
  else if(e.ctrlKey&&e.code==='Space'){e.preventDefault();$('replay').click()}
  else if(e.altKey&&e.key==='ArrowRight'){e.preventDefault();go(1)}
  else if(e.altKey&&e.key==='ArrowLeft'){e.preventDefault();go(-1)}
});
load();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="thư mục chứa predictions.jsonl")
    ap.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT, help="thư mục gốc của đường dẫn audio")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    Handler.store = Store(args.results, args.audio_root)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Mở http://127.0.0.1:{args.port}")
    print(f"Final: {Handler.store.final_path}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
