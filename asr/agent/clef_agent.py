"""Vòng agent Clef-token qua vLLM (PLAN_FC_clef_qwen35.md mục 5). Cùng output với agent.py để dùng lại metrics/.

Mỗi bước: Q1 (1 token, allowed_token_ids = nhãn) → Q2 (mỗi câu 1 token) → sinh giá trị tham số tự do (dừng ở
"\\n</parameter>") → env.validate → env.call. Chuỗi gửi đi được dựng bằng agent/clef/core.run_step, đúng hàm dùng để
dựng dữ liệu train.

Luật: một ToolEnv mỗi mẫu; chỉ validate/call/get_tool_calls/get_state_hash; không mở *_environment/; greedy, seed cố
định, thứ tự phương án cố định; prompt + sinh ≤ 16,384 token (dừng lượt nếu không đủ chỗ).

    python agent/clef_agent.py --data data/train+public_test --split train --url https://...modal.run \\
        --out agent/runs/clef/zs_val_in.jsonl --ids-split val_in [--model-name llm] [--workers 32]
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.clef import fmt as F                 # noqa: E402
from agent.clef.core import run_step            # noqa: E402

A = F.A
TOKENIZER = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
MAX_FREE_TOKENS = 200
MAX_TAIL_TOKENS = 400       # đuôi call: nhiều tham số tuỳ chọn (q2=discrete/none)
REQ_RETRIES = 8             # số lần thử một request; docker/predict.py hạ xuống (vLLM cục bộ, ngân sách 30 s/mẫu)
REQ_TIMEOUT = 600           # giây chờ một request


class ContextFull(Exception):
    pass


def request_json(url, key, model, prompt, max_tokens, retries=None, **extra):
    """POST /v1/completions → JSON đầy đủ (choices[0].text, .logprobs, usage)."""
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "seed": 0, **extra}
    data = json.dumps(body).encode()
    retries = REQ_RETRIES if retries is None else retries
    for k in range(retries):
        try:
            req = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=data,
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as r:
                return json.loads(r.read())
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(min(5 * (k + 1), 30))      # mạng chập chờn: 8 lần, chờ tối đa 30 s/lần (~2.5 phút tổng)


def request(url, key, model, prompt, max_tokens, retries=None, **extra):
    d = request_json(url, key, model, prompt, max_tokens, retries, **extra)
    return d["choices"][0]["text"], d.get("usage", {})


def label_probs(d, token_ids):
    """Xác suất trên các nhãn (theo thứ tự token_ids) từ logprobs của token đầu; None nếu server không trả logprobs.
    Server cần `--logprobs-mode processed_logprobs` để logprob tính sau mask allowed_token_ids (softmax trên nhãn)."""
    lp = (d["choices"][0].get("logprobs") or {}).get("top_logprobs") or []
    if not lp or not lp[0]:
        return None
    got = {}
    for k, v in lp[0].items():
        if k.startswith("token_id:"):
            got[int(k.split(":", 1)[1])] = v
    p = [math.exp(got[i]) if i in got else 0.0 for i in token_ids]
    z = sum(p)
    return [x / z for x in p] if z > 0 else None


def confidence(p):
    """typesafe.ai: (max p − 1/n) / (1 − 1/n); 0 = ngang ngẫu nhiên, 1 = chắc chắn."""
    n = len(p)
    return (max(p) - 1 / n) / (1 - 1 / n) if n > 1 else 1.0


class VLLMOracle:
    def __init__(self, args, tok, labels):
        self.a, self.tok = args, tok
        self.tid = dict(zip(labels, F.label_token_ids(tok, labels)))
        for w in (F.YES, F.NO):
            self.tid[w] = tok(" " + w, add_special_tokens=False).input_ids[0]
        self.log, self.n_req, self.prompt_tokens, self.anomaly = [], 0, 0, 0
        self.plog = []                 # song song với log: {kind, choice, p: {giá trị: xác suất}, conf, override}
        self.want_lp = bool(getattr(args, "log_probs", False) or getattr(args, "ask_tau", None) is not None)

    def _req_json(self, text, max_tokens, **extra):
        d = request_json(self.a.url, self.a.key, self.a.model_name, text, max_tokens, **extra)
        self.n_req += 1
        self.prompt_tokens += (d.get("usage") or {}).get("prompt_tokens") or 0
        return d

    def _req(self, text, max_tokens, **extra):
        return self._req_json(text, max_tokens, **extra)["choices"][0]["text"]

    def choose(self, text, labels, kind, info):
        if kind == "q1" and getattr(self, "force_q1", None):      # I5: ép Q1 vào tool đã chọn bằng Q1 thu hẹp (không gọi server)
            want, self.force_q1 = self.force_q1, None
            lab = next((l for l in labels if info.get(l) == want), None)
            if lab is not None:
                self.log.append((kind, want))
                self.plog.append({"kind": "q1_forced", "choice": want, "conf": 1.0, "p": {want: 1.0}})
                return lab
        if F.n_tokens(self.tok, text) + 1 > A.CONTEXT_LIMIT:
            raise ContextFull(kind)
        ids = [self.tid[l] for l in labels]
        extra = {"logprobs": len(ids), "return_tokens_as_token_ids": True} if self.want_lp else {}
        d = self._req_json(text, 1, allowed_token_ids=ids, **extra)
        lab = d["choices"][0]["text"].strip()
        if lab.startswith("token_id:"):    # return_tokens_as_token_ids cũng đổi text thành "token_id:<id>"
            tid = int(lab.split(":", 1)[1])
            lab = next((l for l, i in zip(labels, ids) if i == tid), lab)
        if lab not in labels:          # không nên xảy ra khi allowed_token_ids hoạt động
            self.anomaly += 1
            lab = labels[0]
        p = label_probs(d, ids) if self.want_lp else None
        override = None
        # §3.1 ngưỡng hỏi lại: ở Q1, nếu P(sys.ask_missing_info) > τ₁ và tool argmax là tool GHI → hỏi lại
        if p is not None and kind == "q1" and self.a.ask_tau is not None:
            ask = [l for l in labels if info.get(l) == F.ASK]
            if ask and info.get(lab) != F.ASK and p[labels.index(ask[0])] > self.a.ask_tau \
                    and self.kind_of(info.get(lab)) == "write":
                override, lab = info.get(lab), ask[0]
        self.log.append((kind, lab if kind != "q1" else info[lab]))
        if p is not None:
            name = (lambda l: info.get(l, l)) if kind in ("q1", "ref", "enum") and isinstance(info, dict) else \
                (lambda l: (info[2].get(l, l) if isinstance(info, tuple) and len(info) > 2 and isinstance(info[2], dict)
                            else l))
            self.plog.append({"kind": kind, "choice": name(lab), "conf": round(confidence(p), 4),
                              "p": {str(name(l)): round(x, 4) for l, x in sorted(zip(labels, p), key=lambda t: -t[1])[:8]},
                              **({"override": override} if override else {})})
        return lab

    def kind_of(self, tool):
        return (self.catalog.get(tool) or {}).get("kind", "write") if tool and tool != F.STOP else "meta"

    def choose_name(self, text, names):
        if F.n_tokens(self.tok, text) + 40 > A.CONTEXT_LIMIT:
            raise ContextFull("q1")
        out = self._req(text, 48, structured_outputs={"choice": [" " + n for n in names]})
        name = out.strip()
        if name not in names:
            self.anomaly += 1
            name = F.STOP
        self.log.append(("q1", name))
        return name

    def gen(self, text, k, s):
        mt = A.fit_max_tokens(self.tok, text, MAX_FREE_TOKENS, min_tokens=8)
        if mt is None:
            raise ContextFull(k)
        out = self._req(text, mt, stop=[F.FREE_STOP, "</parameter>", "</function>", "<|im_end|>"])
        return out

    def gen_tail(self, text, params):
        """Đuôi call (tham số tuỳ chọn, q2=discrete/none): model sinh tới </function>; parse các khối <parameter>."""
        mt = A.fit_max_tokens(self.tok, text, MAX_TAIL_TOKENS, min_tokens=8)
        if mt is None:
            raise ContextFull("tail")
        out = self._req(text, mt, stop=[F.TAIL_STOP, "</tool_call>", "<|im_end|>"])
        out = out.strip("\n")
        return (out + "\n" if out else ""), F.parse_tail(out, params)


_PACK = {}
_PACK_INIT = __import__("threading").Lock()      # chặn nhiều luồng cùng nạp bộ lọc (trước đây: 48 luồng → 52 bản model, tràn RAM/swap)


def get_pack(args):
    """Nạp bộ lọc một lần (catalog của split hiện tại, embedder CPU); mã hoá câu qua khoá để an toàn với nhiều luồng."""
    with _PACK_INIT:
      if "p" not in _PACK:
        import threading
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "clef", "app_filter"))
        from pack import AppFilterPack
        cat = json.load(open(f"{args.data}/{args.split}/{args.split}_tools_catalog.json", encoding="utf-8"))
        _PACK["p"], _PACK["lock"] = AppFilterPack.load(args.app_filter, catalog=cat), threading.Lock()
    return _PACK["p"], _PACK["lock"]


DEPS = {}
ASR_SECS = {}            # {id mẫu: giây ASR đã dùng} (D1: trừ vào ngân sách 30 s)


def build_deps(catalog):
    """I5: {tool đọc: {tool ghi cần id từ kết quả của nó}} suy từ x-ref của catalog ("app.field" ↔ tool đọc trả `field[]`)."""
    prod = {}
    for n, t in catalog.items():
        r = t.get("returns") or ""
        if t.get("kind") == "read" and r.endswith("[]"):
            prod.setdefault(f"{t.get('app')}.{r[:-2]}", []).append(n)
    deps = {}
    for n, t in catalog.items():
        if t.get("kind") == "read":
            continue
        for sch in ((t.get("parameters") or {}).get("properties") or {}).values():
            ref = sch.get("x-ref") or (sch.get("items") or {}).get("x-ref")
            for src in prod.get(ref, []):
                deps.setdefault(src, set()).add(n)
    return deps


def q1_probe(prefix, tools, labels, o, tok, hint, kind):
    """Hỏi riêng bước Q1 (không chạy Q2): trả (tên phương án chọn, bản ghi xác suất). Không đưa vào `decisions`."""
    q1, m = F.q1_block(tools, labels, None, F.n_tokens(tok, prefix) > F.SHORT_Q1_AT, hint)
    lab = o.choose(prefix + q1, list(m), "q1", m)
    o.log.pop()
    e = o.plog[-1]
    e["kind"] = kind
    return m[lab], e


def relax_empty(call, res, catalog):
    """N1 (PLAN_v3_improve): tool đọc trả danh sách rỗng → call mới bỏ các tham số chữ TUỲ CHỌN (thường là truy vấn bị
    ASR nghe sai, vd event_tickets.search{query}). Không áp dụng cho contacts, tham số id/enum. None nếu không nới được."""
    spec = catalog.get(call["name"]) or {}
    L = [v for v in res.values() if isinstance(v, list)] if isinstance(res, dict) and res.get("ok") else []
    if spec.get("kind") != "read" or spec.get("app") == "contacts" or not L or any(L):
        return None
    P = spec.get("parameters", {}).get("properties", {})
    req = set(spec.get("parameters", {}).get("required", []))
    drop = [k for k, v in call["arguments"].items() if k not in req and isinstance(v, str)
            and P.get(k, {}).get("type") == "string" and "enum" not in P.get(k, {}) and not k.endswith("id")]
    if not drop:
        return None
    return {"name": call["name"], "arguments": {k: v for k, v in call["arguments"].items() if k not in drop}}


def run_one(item, args, tok, labels, catalog, ToolEnv, asr):
    t0 = time.time()
    env = ToolEnv(args.split, item["id"], root=args.data)       # ToolEnv duy nhất của mẫu
    msgs = F.build_messages(item, asr)
    tools = A.tool_specs(item, catalog)
    seen = [(c["name"], r) for t in item["history"] if t["role"] == "assistant"
            for c, r in zip(t.get("tool_calls") or [], t.get("tool_results") or [])]
    o = VLLMOracle(args, tok, labels)
    o.catalog = catalog
    n_invalid, sigs, err, steps = 0, {}, None, 0
    retried, n_retry, n_n8, n_n8_switch, n_i5, i5_used, timed_out = set(), 0, 0, 0, 0, False, False
    request = (asr.get(item["id"]) if asr is not None else item.get("audio_text", "")) or ""
    done = []
    tier1 = None
    try:
        if args.app_filter:
            pack, lock = get_pack(args)               # thêm app_filter vào sys.path
            from pack import app_of, tier1_for        # noqa: E402
            with lock:
                tier1 = tier1_for(pack, item, asr)
        for step in range(A.MAX_STEPS):
            if args.time_budget and time.time() - t0 > args.time_budget - ASR_SECS.get(item["id"], 0.0) - args.budget_margin:
                timed_out = True                  # D1: hết ngân sách thời gian của mẫu → nộp các call đã có
                break
            steps += 1
            t1 = None if tier1 is None else tier1 | {app_of(d) for d in done}
            prefix = F.render_prefix(tok, msgs, tools, t1)
            hint = {"request": request, "done": done}
            n0 = len(o.plog)
            r = run_step(prefix, tools, catalog, seen, labels, o, tok, hint=hint)
            if args.n8_tau is not None and t1 is not None:           # N8: Q1 kém chắc → hỏi lại với catalog đầy đủ (không hai tầng)
                q1 = next((x for x in o.plog[n0:] if x["kind"] == "q1"), None)
                if q1 is not None and q1["conf"] < args.n8_tau:
                    n_n8 += 1
                    prefix_f = F.render_prefix(tok, msgs, tools, None)
                    name1 = F.STOP if r["stop"] else r["name"]
                    name2, e2 = q1_probe(prefix_f, tools, labels, o, tok, hint, "q1_n8")
                    if name2 != name1 and e2["p"].get(name2, 0) > q1["p"].get(name1, 0):
                        n_n8_switch += 1
                        r = run_step(prefix_f, tools, catalog, seen, labels, o, tok, hint=hint)
            if r["stop"] and args.i5 is not None and not i5_used and done and isinstance(seen[-1][1], dict):
                # I5: vừa có kết quả của tool đọc mà tool ghi (cần id từ nó) chưa gọi → Q1 thu hẹp {tool ghi đó, hỏi lại, DỪNG}
                last, res_last = seen[-1]
                cons = [d for d in sorted(DEPS.get(last, ())) if d in {t["function"]["name"] for t in tools} and d not in done]
                lists = [v for v in res_last.values() if isinstance(v, list)] if res_last.get("ok") else []
                if cons and lists and any(lists):
                    tools_r = [t for t in tools if t["function"]["name"] in cons]
                    name3, e3 = q1_probe(prefix, tools_r, labels, o, tok, hint, "q1_i5")
                    if name3 in cons and e3["p"].get(name3, 0) >= args.i5:
                        i5_used = True
                        n_i5 += 1
                        o.force_q1 = name3
                        r = run_step(prefix, tools, catalog, seen, labels, o, tok, hint=hint)
            if r["stop"]:
                break
            call = {"name": r["name"], "arguments": r["args"]}
            v = env.validate(call["name"], call["arguments"])
            if not v["ok"]:                                    # không chạy, không ghi vào tool_calls
                n_invalid += 1
                F.append_step(msgs, call, {"ok": False, "error": v["error"]})
                if n_invalid >= A.MAX_INVALID:
                    break
                continue
            res = env.call(call["name"], call["arguments"])
            done.append(call["name"])
            F.append_step(msgs, call, res)
            seen.append((call["name"], res))
            if args.retry_empty and call["name"] not in retried:          # N1: tìm rỗng → tìm lại không kèm chữ tuỳ chọn
                call2 = relax_empty(call, res, catalog)
                if call2 is not None and env.validate(call2["name"], call2["arguments"])["ok"]:
                    retried.add(call["name"])
                    res2 = env.call(call2["name"], call2["arguments"])
                    F.append_step(msgs, call2, res2)
                    seen.append((call2["name"], res2))
                    n_retry += 1
            sig = A._dump(call)
            sigs[sig] = sigs.get(sig, 0) + 1
            if call["name"] == F.ASK or sigs[sig] >= 3:
                break
    except ContextFull as e:
        err = f"context_limit:{e}"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    return {"id": item["id"], "style": F.STYLE.get("name", "v1"), "fmt": F.fmt_opts(), "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash(),
            "steps": steps, "n_invalid": n_invalid, "n_retry": n_retry, "n_n8": n_n8, "n_n8_switch": n_n8_switch, "n_i5": n_i5, "timed_out": timed_out, "decisions": o.log, **({"probs": o.plog} if o.plog else {}), "n_requests": o.n_req,
            "prompt_tokens_sum": o.prompt_tokens, "anomaly": o.anomaly,
            "latency": round(time.time() - t0, 2), "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--url", required=True)
    ap.add_argument("--key", default=os.environ.get("LLM_API_KEY", ""))
    ap.add_argument("--model-name", default="llm", help="tên model / LoRA adapter trên server vLLM")
    ap.add_argument("--out", required=True)
    ap.add_argument("--asr", help="JSON {uid: transcript}; mặc định audio_text (chỉ train)")
    ap.add_argument("--log-probs", action="store_true", help="lấy xác suất nhãn mỗi lựa chọn (server cần --max-logprobs 64 --logprobs-mode processed_logprobs), ghi vào trường probs")
    ap.add_argument("--n8-tau", type=float, help="N8: Q1 có conf < τ (vd 0.7) và đang dùng catalog hai tầng → hỏi lại với catalog đầy đủ; hai lần lệch thì theo lần có xác suất cao hơn. Tự bật --log-probs")
    ap.add_argument("--i5", type=float, help="I5: model chọn DỪNG ngay sau tool đọc có kết quả mà tool ghi dùng id của nó chưa gọi → hỏi lại Q1 thu hẹp; nếu xác suất tool ghi ≥ ngưỡng (vd 0.5) thì làm tiếp. Tự bật --log-probs")
    ap.add_argument("--time-budget", type=float, help="D1: giây tối đa cho mỗi mẫu (vd 30); hết thì dừng và nộp các call đã có. Tính cả giây ASR nếu có --asr-secs")
    ap.add_argument("--budget-margin", type=float, default=2.0, help="chừa cho request đang bay + ghi kết quả (giây)")
    ap.add_argument("--asr-secs", help="JSON {id mẫu: giây ASR} (docker/asr_stage.py ghi)")
    ap.add_argument("--retry-empty", action="store_true", help="N1: tool đọc trả rỗng → tìm lại một lần, bỏ tham số chữ tuỳ chọn")
    ap.add_argument("--ask-tau", type=float, help="Q1: hỏi lại nếu P(sys.ask_missing_info) > τ và tool argmax là tool ghi (PLAN_generalize_tools §3.1; τ = 0.2). Tự bật --log-probs")
    ap.add_argument("--asr-timings", help="JSON {id: giây ASR} cộng vào elapsed_seconds của <out>.timings.jsonl")
    ap.add_argument("--ids", help="file id, mỗi dòng một id")
    ap.add_argument("--ids-split", help="khóa trong agent/clef/splits.json (val_in / val_ood / train)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--style", default="v1", choices=list(F.STYLES), help="kiểu prompt (phải giống lúc dựng dữ liệu train)")
    ap.add_argument("--q2", default="all", choices=F.Q2_MODES)
    ap.add_argument("--q1-desc", default="full", choices=F.Q1_DESC_MODES)
    ap.add_argument("--sys", default="json", choices=F.SYS_MODES)
    ap.add_argument("--rules", default="full", choices=F.RULES_MODES)
    ap.add_argument("--app-filter", help="thư mục gói bộ lọc app (app_filter/pack.py): prompt catalog hai tầng; chỉ với --sys compact. "
                    "Catalog lấy từ <data>/<split>/<split>_tools_catalog.json")
    ap.add_argument("--fmt-from", help="config.json của run train hoặc <data>.meta.json: lấy style/q2/q1_desc từ đó, "
                    "ghi đè 3 cờ trên (tránh lệch format train / suy luận)")
    args = ap.parse_args()
    if args.n8_tau is not None or args.i5 is not None:
        args.log_probs = True
    if args.fmt_from:
        m = json.load(open(args.fmt_from, encoding="utf-8"))
        F.set_style_from(m.get("fmt", m))
    else:
        F.set_style(args.style, q2=args.q2, q1_desc=args.q1_desc, sys=args.sys, rules=args.rules)
    print("format:", F.fmt_opts(), flush=True)

    sys.path.insert(0, os.path.join(args.data, args.split))
    from zaic_tools import ToolEnv
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    labels = F.make_labels(tok)
    catalog = A.load_catalog(os.path.join(args.data, args.split, f"{args.split}_tools_catalog.json"))
    DEPS.update(build_deps(catalog))
    if args.asr_secs:
        ASR_SECS.update(json.load(open(args.asr_secs, encoding="utf-8")))
    items = [json.loads(l) for l in open(os.path.join(args.data, args.split, f"{args.split}.jsonl"), encoding="utf-8")
             if l.strip()]
    keep = None
    if args.ids:
        keep = {l.strip() for l in open(args.ids) if l.strip()}
    if args.ids_split:
        sp = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "clef", "splits.json")))
        keep = set(sp[args.ids_split]) & keep if keep else set(sp[args.ids_split])
    if keep is not None:
        items = [x for x in items if x["id"] in keep]
    items = items[: args.limit]
    asr = json.load(open(args.asr, encoding="utf-8")) if args.asr else None
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8") if l.strip()}
    todo = [x for x in items if x["id"] not in done]
    print(f"clef: {len(todo)} mẫu cần chạy ({len(done)} đã có)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    asr_time = json.load(open(args.asr_timings, encoding="utf-8")) if args.asr_timings else {}
    t0 = time.time()
    tpath = os.path.splitext(args.out)[0] + ".timings.jsonl"       # cho `metrics score --timings` (luật 30 s)
    with open(args.out, "a", encoding="utf-8") as f, open(tpath, "a", encoding="utf-8") as ft, \
            cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_one, x, args, tok, labels, catalog, ToolEnv, asr) for x in todo]
        for k, fu in enumerate(cf.as_completed(futs), 1):
            r = fu.result()
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            # elapsed = agent (+ ASR nếu có). workers > 1 làm thời gian bị đội lên do chen GPU: chỉ để chẩn đoán;
            # số áp luật 30 s phải đo với --workers 1 trên máy chạy nộp bài.
            ft.write(json.dumps({"id": r["id"], "elapsed_seconds": round(r["latency"] + asr_time.get(r["id"], 0.0), 3),
                                 "status": "error" if r["error"] else "ok", "error": r["error"],
                                 "workers": args.workers, "asr_included": bool(asr_time)}) + "\n")
            ft.flush()
            if k % 50 == 0 or k == len(futs):
                print(f"  {k}/{len(futs)}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
