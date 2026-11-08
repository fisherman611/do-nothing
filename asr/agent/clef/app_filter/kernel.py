"""Lọc app (PLAN_app_filter.md): so BM25, multilingual-e5-small, vietnamese-bi-encoder, paraphrase-multilingual-MiniLM-L12.

- Zero-shot: mọi mẫu train (lượt cuối), tách theo split clef (train / val_in / val_ood).
- Fine-tune nhanh (InfoNCE nhiều dương trên các app ĐÃ CÀI của chính mẫu) trên split train (lượt cuối + lượt history),
  app giữ lại của val_ood bị loại khỏi mọi tập ứng viên khi train. Đánh giá trên val_in / val_ood.
- Hybrid: RRF giữa BM25 và embedding.
Chỉ số: recall = tỉ lệ mẫu giữ đủ MỌI app vàng, theo top-k (có/không luật: + app đã gọi trong history, + app cung
cấp id theo x-ref), và số app giữ trung bình.
"""
import glob
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import unicodedata
from collections import Counter

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyvi"], check=False)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/kaggle/working"
D = os.path.dirname(glob.glob("/kaggle/input/**/train.jsonl", recursive=True)[0])
print("data:", D, "| device:", DEV, torch.cuda.get_device_name() if DEV == "cuda" else "")
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


cat = json.load(open(f"{D}/train_tools_catalog.json", encoding="utf-8"))
SPL = json.load(open(f"{D}/splits.json", encoding="utf-8"))
HELD = set(SPL["held_out_apps"])
rows = [json.loads(l) for l in open(f"{D}/train.jsonl", encoding="utf-8")]
split_of = {i: s for s in ("train", "val_in", "val_ood") for i in SPL[s]}

DOC, DEPS = {}, {}
for app in cat["apps"]:
    DOC[app["id"]] = f"{app['name_vi']}. {app['description']}. Chức năng: " + "; ".join(t["description"] for t in app["tools"])
    DEPS[app["id"]] = {p["x-ref"].split(".")[0] for t in app["tools"]
                       for p in t["parameters"].get("properties", {}).values() if "x-ref" in p} - {app["id"]}
APPS = sorted(DOC)


def apps_of(calls):
    return {c["name"].split(".")[0] for c in calls or []} - {"sys"}


def make_items():
    final, hist = [], []
    for r in rows:
        inst = sorted(a for a in r["context"]["installed_apps"] if a != "sys" and a in DOC)
        h = r["history"]
        prev_asst = next((t.get("text", "") for t in reversed(h) if t["role"] == "assistant"), "")
        final.append(dict(id=r["id"], split=split_of.get(r["id"]), q=r["audio_text"], prev=prev_asst,
                          gold=apps_of(r["label"]["tool_calls"]), inst=inst,
                          hist_apps={a for t in h if t["role"] == "assistant" for a in apps_of(t.get("tool_calls"))}))
        if split_of.get(r["id"]) != "train":
            continue
        for j, t in enumerate(h):   # lượt history: user j-1 → assistant j có call
            if t["role"] != "assistant" or not t.get("tool_calls") or j == 0 or h[j - 1]["role"] != "user":
                continue
            g = apps_of(t["tool_calls"])
            if not g or g & HELD:
                continue
            pa = next((x.get("text", "") for x in reversed(h[:j - 1]) if x["role"] == "assistant"), "")
            hist.append(dict(id=f"{r['id']}#h{j}", split="train", q=h[j - 1].get("audio_text", ""), prev=pa, gold=g,
                             inst=inst, hist_apps={a for x in h[:j - 1] if x["role"] == "assistant"
                                                   for a in apps_of(x.get("tool_calls"))}))
    return final, hist


FINAL, HIST = make_items()
log("lượt cuối:", len(FINAL), "| lượt history train:", len(HIST), "| app giữ lại:", sorted(HELD))


def query_text(it, with_prev):
    return (it["prev"] + " " + it["q"]).strip() if with_prev and it["prev"] else it["q"]


# ---------------- BM25 ----------------
def toks(s):
    w = re.findall(r"\w+", unicodedata.normalize("NFC", s.lower()))
    return w + [a + "_" + b for a, b in zip(w, w[1:])]


DT = {a: toks(d) for a, d in DOC.items()}
DF = Counter(w for d in DT.values() for w in set(d))
AVG = sum(map(len, DT.values())) / len(DT)


def bm25_scores(q, apps, k1=1.2, b=0.75):
    qt, N = set(toks(q)), len(DT)
    out = []
    for a in apps:
        tf, d = Counter(DT[a]), DT[a]
        out.append(sum(math.log(1 + (N - DF[w] + 0.5) / (DF[w] + 0.5)) * tf[w] * (k1 + 1) /
                       (tf[w] + k1 * (1 - b + b * len(d) / AVG)) for w in qt if w in tf))
    return np.array(out)


# ---------------- embedding ----------------
MODELS = {
    "e5-small": dict(id="intfloat/multilingual-e5-small", q="query: ", p="passage: ", seg=False),
    "vi-bi-encoder": dict(id="bkai-foundation-models/vietnamese-bi-encoder", q="", p="", seg=True),
    "mMiniLM-L12": dict(id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", q="", p="", seg=False),
}
_vi = None


def prep(cfg, texts, kind):
    global _vi
    if cfg["seg"]:
        if _vi is None:
            from pyvi import ViTokenizer
            _vi = ViTokenizer
        texts = [_vi.tokenize(t) for t in texts]
    return [cfg[kind] + t for t in texts]


def encode(model, cfg, texts, kind, grad=False):
    texts = prep(cfg, texts, kind)
    feats = model.tokenize(texts)
    feats = {k: (v.to(DEV) if hasattr(v, "to") else v) for k, v in feats.items()}
    with torch.set_grad_enabled(grad):
        e = model(feats)["sentence_embedding"]
    return torch.nn.functional.normalize(e, dim=-1)


def emb_scorer(model, cfg, items, with_prev):
    model.eval()
    with torch.no_grad():
        de = encode(model, cfg, [DOC[a] for a in APPS], "p")
        qe = torch.cat([encode(model, cfg, [query_text(it, with_prev) for it in items[i:i + 64]], "q")
                        for i in range(0, len(items), 64)])
    sims = (qe @ de.T).cpu().numpy()
    idx = {a: k for k, a in enumerate(APPS)}
    return {it["id"]: sims[n, [idx[a] for a in it["inst"]]] for n, it in enumerate(items)}


# ---------------- đánh giá ----------------
def ranks_from(scores):
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), int)
    r[order] = np.arange(len(scores))
    return r


def rrf(s1, s2, k=10):
    return 1 / (k + ranks_from(s1)) + 1 / (k + ranks_from(s2))


def evaluate(items, score_of):
    """score_of(it) → điểm trên it['inst']. Trả recall/kept theo k, có và không có luật; và trung vị hạng app vàng tệ nhất."""
    res = {}
    for k in range(1, 7):
        for rules in (False, True):
            hit = kept = 0
            for it in items:
                s = score_of(it)
                top = {it["inst"][j] for j in np.argsort(-s, kind="stable")[:k]}
                if rules:
                    top |= it["hist_apps"] & set(it["inst"])
                    top |= {d for a in list(top) for d in DEPS.get(a, ())} & set(it["inst"])
                hit += it["gold"] <= top
                kept += len(top)
            res[f"k{k}{'+r' if rules else ''}"] = (round(hit / len(items), 4), round(kept / len(items), 2))
    return res


def report(name, items_by_set, score_of):
    out = {}
    for sname, items in items_by_set.items():
        items = [it for it in items if it["gold"]]          # mẫu không gọi app nào: luôn "đúng", bỏ khỏi recall
        out[sname] = evaluate(items, score_of)
        r = out[sname]
        log(f"  {name:<34} {sname:<8} n={len(items):<4} " + " | ".join(
            f"k{k}+r {r[f'k{k}+r'][0]:.1%}/{r[f'k{k}+r'][1]:.1f}" for k in (2, 3, 4, 5)))
    return out


SETS_ZS = {s: [it for it in FINAL if it["split"] == s] for s in ("train", "val_in", "val_ood")}
SETS_ZS["all"] = FINAL
SETS_VAL = {s: SETS_ZS[s] for s in ("val_in", "val_ood")}
RESULTS = {"note": "giá trị (recall, số app giữ TB); k<n>+r = top-n + luật history/x-ref; chỉ mẫu có app vàng"}

log("\n=== BM25 ===")
for wp in (False, True):
    RESULTS[f"bm25{'+prev' if wp else ''}"] = report(f"bm25{' +prev' if wp else ''}", SETS_ZS,
                                                    lambda it, wp=wp: bm25_scores(query_text(it, wp), it["inst"]))


def train_model(model, cfg, epochs=3, lr=2e-5, bs=32, scale=20.0):
    data = [it for it in FINAL if it["split"] == "train" and it["gold"]] + HIST
    for it in data:
        it["cand"] = [a for a in it["inst"] if a not in HELD]
    data = [it for it in data if it["gold"] <= set(it["cand"])]
    steps = epochs * math.ceil(len(data) / bs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, int(0.1 * steps))) *
                                            max(0.0, (steps - s) / steps))
    rng = random.Random(SEED)
    model.train()
    t0, step = time.time(), 0
    for ep in range(epochs):
        rng.shuffle(data)
        tot = 0.0
        for i in range(0, len(data), bs):
            b = data[i:i + bs]
            apps = sorted({a for it in b for a in it["cand"]})
            ai = {a: k for k, a in enumerate(apps)}
            qe = encode(model, cfg, [query_text(it, rng.random() < 0.5) for it in b], "q", grad=True)
            de = encode(model, cfg, [DOC[a] for a in apps], "p", grad=True)
            sim = scale * qe @ de.T
            loss = 0
            for n, it in enumerate(b):        # softmax trên app đã cài của chính mẫu, nhiều dương
                c = torch.tensor([ai[a] for a in it["cand"]], device=DEV)
                pos = torch.tensor([a in it["gold"] for a in it["cand"]], device=DEV)
                lg = sim[n, c]
                loss = loss + (torch.logsumexp(lg, 0) - torch.logsumexp(lg[pos], 0))
            loss = loss / len(b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            tot += loss.item()
            step += 1
        log(f"    epoch {ep + 1}/{epochs}: loss {tot / math.ceil(len(data) / bs):.4f} ({time.time() - t0:.0f}s, {len(data)} ví dụ)")
    model.eval()
    return model


for name, cfg in MODELS.items():
    log(f"\n=== {name} ({cfg['id']}) ===")
    try:
        model = SentenceTransformer(cfg["id"], device=DEV)
        model.max_seq_length = 256
        nparam = sum(p.numel() for p in model.parameters())
        log(f"  tham số: {nparam / 1e6:.1f}M")
        RESULTS[name] = {"params_M": round(nparam / 1e6, 1)}
        for wp in (False, True):
            sc = emb_scorer(model, cfg, FINAL, wp)
            RESULTS[name][f"zs{'+prev' if wp else ''}"] = report(f"{name} zero-shot{' +prev' if wp else ''}", SETS_ZS,
                                                                 lambda it, sc=sc: sc[it["id"]])
        model = train_model(model, cfg)
        for wp in (False, True):
            sc = emb_scorer(model, cfg, FINAL, wp)
            RESULTS[name][f"ft{'+prev' if wp else ''}"] = report(f"{name} fine-tune{' +prev' if wp else ''}", SETS_VAL,
                                                                 lambda it, sc=sc: sc[it["id"]])
            RESULTS[name][f"ft{'+prev' if wp else ''}+bm25"] = report(
                f"{name} fine-tune{' +prev' if wp else ''} ⊕ BM25", SETS_VAL,
                lambda it, sc=sc, wp=wp: rrf(sc[it["id"]], bm25_scores(query_text(it, wp), it["inst"])))
        # độ trễ: 1 query (docs tính sẵn), CPU fp32
        mc = model.to("cpu")
        DEV = "cpu"
        qs = [it["q"] for it in SETS_ZS["val_in"][:50]]
        t0 = time.time()
        for q in qs:
            encode(mc, cfg, [q], "q")
        RESULTS[name]["cpu_ms_per_query"] = round(1000 * (time.time() - t0) / len(qs), 1)
        log(f"  CPU: {RESULTS[name]['cpu_ms_per_query']} ms/query (đã gồm tách từ nếu có)")
        DEV = "cuda" if torch.cuda.is_available() else "cpu"
        del model, mc
        torch.cuda.empty_cache()
    except Exception as e:   # một model lỗi không làm hỏng cả lượt
        import traceback
        log(f"  LỖI {name}: {type(e).__name__}: {e}")
        log(traceback.format_exc()[-1500:])
        RESULTS[name] = {"error": f"{type(e).__name__}: {e}"}
        DEV = "cuda" if torch.cuda.is_available() else "cpu"

json.dump(RESULTS, open(f"{OUT}/results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
open(f"{OUT}/summary.log", "w", encoding="utf-8").write("\n".join(LOG))
log("\nXONG")
