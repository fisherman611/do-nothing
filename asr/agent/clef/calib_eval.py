"""Đo hiệu chỉnh xác suất các câu hỏi chọn (Q1 / Q2) kiểu teacher-forcing (PLAN_generalize_tools §3.0).

Với mỗi span `choice` của một bộ Clef đã build (val_in / val_ood, không xáo = đúng phân phối lúc infer): prompt =
text[:pos] (prefix gold tới ngay trước đáp án), hỏi server vLLM 1 token với allowed_token_ids = nhãn + logprobs, lấy
phân phối trên nhãn. Server phải chạy `--max-logprobs 64 --logprobs-mode processed_logprobs` (modal_llm_server.py
`mid_lp`) để logprob tính sau mask → softmax chỉ trên nhãn.

    python agent/clef/calib_eval.py collect --data agent/clef/data/calib/val_in_noshuf.jsonl --url <url> --out <x>.jsonl
    python agent/clef/calib_eval.py report --fit <val_in>.jsonl --eval <val_in>.jsonl <val_ood>.jsonl [...]

confidence (typesafe.ai): (max p − 1/n) / (1 − 1/n). Temperature T: p_T ∝ p^(1/T), fit trên --fit (chỉ val_in).
"""
import argparse
import collections
import concurrent.futures as cf
import json
import math
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOKENIZER = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
Q2_HEAD = re.compile(r"\[Q2\] (\S+): (.+)$")


def kind_of(text, pos, j, catalog):
    """q1 / present / bool / enum / ref; kèm (tool, tham số) cho Q2."""
    if j == 0:
        return "q1", None, None
    head = text[:pos].rsplit("\n[Q2] ", 1)[-1]
    line = "[Q2] " + head.split("\n", 1)[0]
    m = Q2_HEAD.match(line)
    if not m:
        return "q2?", None, None
    tool, rest = m.group(1), m.group(2)
    if rest.startswith("điền tham số "):
        return "present", tool, rest[len("điền tham số "):].split(" ", 1)[0].rstrip("?")
    k = rest.split(" = ", 1)[0]
    if rest.endswith("= true?"):
        return "bool", tool, k
    s = catalog.get(tool, {}).get("parameters", {}).get("properties", {}).get(k, {})
    return ("ref" if "x-ref" in s else "enum"), tool, k


def q1_options(text, pos):
    """nhãn → tên tool của khối Q1 ngay trước pos."""
    blk = text[:pos].rsplit("[Q1]", 1)[-1]
    return {m.group(1): m.group(2) for m in re.finditer(r"^(\S)\) (\S+)", blk, re.M)}


def collect(a):
    from transformers import AutoTokenizer
    from agent import agent as A
    tok = AutoTokenizer.from_pretrained(*[TOKENIZER[0]], revision=TOKENIZER[1])
    catalog = A.load_catalog(os.path.join(ROOT, "data/train+public_test/train/train_tools_catalog.json"))
    key = a.key or os.environ.get("LLM_API_KEY", "")
    tid = {}

    def ids(labels):
        for l in labels:
            if l not in tid:
                tid[l] = tok(" " + l, add_special_tokens=False).input_ids[0]
        return [tid[l] for l in labels]

    jobs = []
    for l in open(a.data, encoding="utf-8"):
        e = json.loads(l)
        ch = [sp for sp in e["spans"] if sp["type"] == "choice"]
        for j, sp in enumerate(ch):
            kind, tool, k = kind_of(e["text"], sp["pos"], j, catalog)
            rec = {"id": e["id"], "step": e["step"], "j": j, "kind": kind, "tool": tool, "param": k,
                   "labels": sp["labels"], "gold": sp["labels"].index(sp["gold"])}
            if kind == "q1":
                opt = q1_options(e["text"], sp["pos"])
                rec["gold_name"] = opt.get(sp["gold"])
                rec["names"] = [opt.get(x) for x in sp["labels"]]
            jobs.append((e["text"][: sp["pos"]], rec))

    def run(job):
        prompt, rec = job
        allowed = ids(rec["labels"])
        body = {"model": "llm", "prompt": prompt, "max_tokens": 1, "temperature": 0, "seed": 0,
                "allowed_token_ids": allowed, "logprobs": len(allowed), "return_tokens_as_token_ids": True}
        for t in range(8):
            try:
                r = urllib.request.Request(a.url.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                                           headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
                d = json.loads(urllib.request.urlopen(r, timeout=600).read())
                top = d["choices"][0]["logprobs"]["top_logprobs"][0]
                lp = {int(x.split(":")[1]): v for x, v in top.items()}
                p = [math.exp(lp.get(i, -1e9)) for i in allowed]
                z = sum(p) or 1.0
                return {**rec, "p": [round(x / z, 6) for x in p], "mass": round(z, 5)}
            except Exception as ex:  # noqa: BLE001
                err = str(ex)
                time.sleep(5 * (t + 1))
        return {**rec, "error": err}

    n = 0
    with open(a.out, "w", encoding="utf-8") as f, cf.ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(run, jobs):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
            if n % 500 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)
    print(f"{a.out}: {n} câu hỏi")


# ---------------- report ----------------

def temp(p, T):
    if T == 1.0:
        return p
    q = [x ** (1 / T) if x > 0 else 0.0 for x in p]
    z = sum(q) or 1.0
    return [x / z for x in q]


def conf(p):
    n = len(p)
    return (max(p) - 1 / n) / (1 - 1 / n) if n > 1 else 1.0


def auroc(scores, pos):
    """AUROC của score dự báo pos (True)."""
    P = [s for s, y in zip(scores, pos) if y]
    N = [s for s, y in zip(scores, pos) if not y]
    if not P or not N:
        return float("nan")
    allv = sorted([(s, 1) for s in P] + [(s, 0) for s in N])
    rank, i, rs = 0.0, 0, 0.0
    while i < len(allv):                       # rank trung bình cho giá trị trùng
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        r = (i + j + 1) / 2
        rs += r * sum(1 for k in range(i, j) if allv[k][1])
        i = j
    return (rs - len(P) * (len(P) + 1) / 2) / (len(P) * len(N))


def metrics(rows, T=1.0, bins=15):
    acc, nll, bri, ece_b = [], [], [], [[] for _ in range(bins)]
    mx, cf_, ok = [], [], []
    for r in rows:
        p = temp(r["p"], T)
        g = r["gold"]
        am = max(range(len(p)), key=p.__getitem__)
        c = am == g
        acc.append(c)
        nll.append(-math.log(max(p[g], 1e-12)))
        bri.append(sum((x - (1.0 if i == g else 0.0)) ** 2 for i, x in enumerate(p)))
        m = max(p)
        mx.append(m)
        cf_.append(conf(p))
        ok.append(c)
        ece_b[min(bins - 1, int(m * bins))].append((m, c))
    n = len(rows)
    ece = sum(len(b) / n * abs(sum(x for x, _ in b) / len(b) - sum(y for _, y in b) / len(b)) for b in ece_b if b)
    right = [c for c, y in zip(cf_, ok) if y]
    wrong = [c for c, y in zip(cf_, ok) if not y]
    return {"n": n, "acc": sum(acc) / n, "nll": sum(nll) / n, "brier": sum(bri) / n, "ece": ece,
            "mean_maxp": sum(mx) / n, "auroc": auroc(cf_, ok),
            "conf_right": sum(right) / len(right) if right else float("nan"),
            "conf_wrong": sum(wrong) / len(wrong) if wrong else float("nan"), "n_wrong": len(wrong)}


def fit_T(rows):
    best = (1e9, 1.0)
    for T in [x / 100 for x in range(20, 301, 2)]:
        v = sum(-math.log(max(temp(r["p"], T)[r["gold"]], 1e-12)) for r in rows) / len(rows)
        best = min(best, (v, T))
    return best[1]


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    bad = [r for r in rows if "error" in r]
    if bad:
        print(f"CẢNH BÁO {path}: {len(bad)} câu hỏi lỗi request, bỏ qua")
    return [r for r in rows if "p" in r]


def report(a):
    fit = load(a.fit)
    T_all = fit_T(fit)
    T_kind = {k: fit_T([r for r in fit if r["kind"] == k]) for k in sorted({r["kind"] for r in fit})}
    print(f"Temperature fit trên {os.path.basename(a.fit)}: T toàn cục = {T_all:.2f}; theo loại: "
          + ", ".join(f"{k} {v:.2f}" for k, v in T_kind.items()))
    out = {"T": T_all, "T_kind": T_kind, "fit": a.fit, "sets": {}}
    hdr = f"{'tập':22} {'loại':8} {'n':>5} {'acc':>6} {'ECE':>6} {'ECE@T':>6} {'NLL':>6} {'NLL@T':>6} {'maxp':>5} {'AUROC':>6} {'conf đúng':>9} {'conf sai':>8} {'n sai':>5}"
    for path in a.eval:
        rows = load(path)
        name = os.path.splitext(os.path.basename(path))[0]
        print("\n" + hdr)
        groups = [("TẤT CẢ", rows)] + [(k, [r for r in rows if r["kind"] == k]) for k in ("q1", "present", "ref", "enum", "bool")]
        out["sets"][name] = {}
        for k, rs in groups:
            if not rs:
                continue
            m0, mT = metrics(rs), metrics(rs, T_all)
            out["sets"][name][k] = {"raw": m0, "T": mT}
            print(f"{name:22} {k:8} {m0['n']:5} {m0['acc']:6.3f} {m0['ece']:6.3f} {mT['ece']:6.3f} {m0['nll']:6.3f} "
                  f"{mT['nll']:6.3f} {m0['mean_maxp']:5.2f} {m0['auroc']:6.3f} {m0['conf_right']:9.3f} {m0['conf_wrong']:8.3f} {m0['n_wrong']:5}")
        # Q1: hỏi lại / dừng
        q1 = [r for r in rows if r["kind"] == "q1" and r.get("names")]
        for lab, nm in (("ask", "sys.ask_missing_info"), ("DỪNG", "DỪNG")):
            g = [r for r in q1 if r["gold_name"] == nm]
            pred = [r for r in q1 if r["names"][max(range(len(r["p"])), key=r["p"].__getitem__)] == nm]
            hit = [r for r in g if r["names"][max(range(len(r["p"])), key=r["p"].__getitem__)] == nm]
            pg = [r["p"][r["names"].index(nm)] for r in g if nm in r["names"]]
            png = [r["p"][r["names"].index(nm)] for r in q1 if r["gold_name"] != nm and nm in r["names"]]
            if g:
                print(f"   Q1 {lab}: gold {len(g)}, model chọn {len(pred)}, đúng {len(hit)} (recall {len(hit) / len(g):.2f}); "
                      f"P({lab}) TB khi gold là {lab}: {sum(pg) / len(pg):.3f}, khi không: {sum(png) / max(1, len(png)):.3f}")
        # thiên vị vị trí: P trung bình theo vị trí vs tần suất gold theo vị trí (nhóm có ≥ 3 phương án)
        multi = [r for r in rows if len(r["p"]) >= 3]
        if multi:
            pm, gm = collections.Counter(), collections.Counter()
            for r in multi:
                for i, x in enumerate(r["p"][:5]):
                    pm[i] += x
                if r["gold"] < 5:
                    gm[r["gold"]] += 1
            print("   vị trí A..E (≥3 phương án): P TB " + " ".join(f"{pm[i] / len(multi):.3f}" for i in range(5))
                  + " | gold " + " ".join(f"{gm[i] / len(multi):.3f}" for i in range(5))
                  + f" | argmax=A {sum(max(range(len(r['p'])), key=r['p'].__getitem__) == 0 for r in multi) / len(multi):.3f}")
    if a.out:
        json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--data", required=True)
    c.add_argument("--url", required=True)
    c.add_argument("--key")
    c.add_argument("--out", required=True)
    c.add_argument("--workers", type=int, default=32)
    r = sub.add_parser("report")
    r.add_argument("--fit", required=True, help="tập fit temperature (val_in không xáo)")
    r.add_argument("--eval", nargs="+", required=True)
    r.add_argument("--out", help="lưu T + số liệu JSON")
    a = ap.parse_args()
    collect(a) if a.cmd == "collect" else report(a)


if __name__ == "__main__":
    main()
