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


class ContextFull(Exception):
    pass


def request(url, key, model, prompt, max_tokens, retries=4, **extra):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "seed": 0, **extra}
    data = json.dumps(body).encode()
    for k in range(retries):
        try:
            req = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=data,
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            return d["choices"][0]["text"], d.get("usage", {})
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(5 * (k + 1))


class VLLMOracle:
    def __init__(self, args, tok, labels):
        self.a, self.tok = args, tok
        self.tid = dict(zip(labels, F.label_token_ids(tok, labels)))
        for w in (F.YES, F.NO):
            self.tid[w] = tok(" " + w, add_special_tokens=False).input_ids[0]
        self.log, self.n_req, self.prompt_tokens, self.anomaly = [], 0, 0, 0

    def _req(self, text, max_tokens, **extra):
        out, usage = request(self.a.url, self.a.key, self.a.model_name, text, max_tokens, **extra)
        self.n_req += 1
        self.prompt_tokens += usage.get("prompt_tokens") or 0
        return out

    def choose(self, text, labels, kind, info):
        if F.n_tokens(self.tok, text) + 1 > A.CONTEXT_LIMIT:
            raise ContextFull(kind)
        out = self._req(text, 1, allowed_token_ids=[self.tid[l] for l in labels])
        lab = out.strip()
        if lab not in labels:          # không nên xảy ra khi allowed_token_ids hoạt động
            self.anomaly += 1
            lab = labels[0]
        self.log.append((kind, lab if kind != "q1" else info[lab]))
        return lab

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


def run_one(item, args, tok, labels, catalog, ToolEnv, asr):
    t0 = time.time()
    env = ToolEnv(args.split, item["id"], root=args.data)       # ToolEnv duy nhất của mẫu
    msgs = A.build_messages(item, asr)
    tools = A.tool_specs(item, catalog)
    seen = [(c["name"], r) for t in item["history"] if t["role"] == "assistant"
            for c, r in zip(t.get("tool_calls") or [], t.get("tool_results") or [])]
    o = VLLMOracle(args, tok, labels)
    n_invalid, sigs, err, steps = 0, {}, None, 0
    request = (asr.get(item["id"]) if asr is not None else item.get("audio_text", "")) or ""
    done = []
    try:
        for step in range(A.MAX_STEPS):
            steps += 1
            r = run_step(F.render_prefix(tok, msgs, tools), tools, catalog, seen, labels, o, tok,
                         hint={"request": request, "done": done})
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
            sig = A._dump(call)
            sigs[sig] = sigs.get(sig, 0) + 1
            if call["name"] == F.ASK or sigs[sig] >= 3:
                break
    except ContextFull as e:
        err = f"context_limit:{e}"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    return {"id": item["id"], "style": F.STYLE.get("name", "v1"), "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash(),
            "steps": steps, "n_invalid": n_invalid, "decisions": o.log, "n_requests": o.n_req,
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
    ap.add_argument("--ids", help="file id, mỗi dòng một id")
    ap.add_argument("--ids-split", help="khóa trong agent/clef/splits.json (val_in / val_ood / train)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--style", default="v1", choices=list(F.STYLES), help="kiểu prompt (phải giống lúc dựng dữ liệu train)")
    args = ap.parse_args()
    F.set_style(args.style)

    sys.path.insert(0, os.path.join(args.data, args.split))
    from zaic_tools import ToolEnv
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER[0], revision=TOKENIZER[1])
    labels = F.make_labels(tok)
    catalog = A.load_catalog(os.path.join(args.data, args.split, f"{args.split}_tools_catalog.json"))
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
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_one, x, args, tok, labels, catalog, ToolEnv, asr) for x in todo]
        for k, fu in enumerate(cf.as_completed(futs), 1):
            f.write(json.dumps(fu.result(), ensure_ascii=False) + "\n")
            f.flush()
            if k % 50 == 0 or k == len(futs):
                print(f"  {k}/{len(futs)}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
