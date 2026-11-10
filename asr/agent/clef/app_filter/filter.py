"""Lọc app ứng viên cho một mẫu (PLAN_v2_pipeline §2.2): BM25 ∥ embedding tiếng Việt → HỢP top-k, + luật (app đã gọi
trong history, đóng bao x-ref), ưu tiên recall. Dùng được như module (build_data / clef_agent) và CLI đo recall.

    # đo recall theo k trên train / val_in / val_ood (BM25 không cần model; --emb <tên HF> để thêm embedding)
    python agent/clef/app_filter/filter.py eval --data data/train+public_test [--emb bkai-foundation-models/vietnamese-bi-encoder]
        [--emb-cache agent/clef/app_filter/emb_cache] [--out agent/clef/app_filter/eval_<tag>.json]

Tài liệu app = name_vi + mô tả app + tên/mô tả tool + mô tả tham số + enum + từ khoá đồng nghĩa (`app_keywords.json`).
Query = câu cuối (+ câu user cuối trong history). BM25 trên âm tiết có dấu và không dấu + bigram.
Chỉ đọc catalog train (và overlay nếu truyền vào); không đọc gì trong public_test/.
"""
import argparse
import collections
import json
import math
import os
import re
import sys
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
KEYWORDS = os.path.join(HERE, "app_keywords.json")


def strip_accents(s):
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def toks(s, bigram=True):
    s = unicodedata.normalize("NFC", (s or "").lower())
    w = re.findall(r"\w+", s)
    out = w + [strip_accents(x) for x in w if strip_accents(x) != x]
    if bigram:
        out += [a + "_" + b for a, b in zip(w, w[1:])]
    return out


def app_docs(catalog, keywords=None):
    """{app_id: (text, deps)}; deps = app khác được tham chiếu qua x-ref của tool trong app."""
    kw = keywords or {}
    docs, deps = {}, {}
    for app in catalog["apps"]:
        if app["id"] == "sys":
            continue
        parts = [app.get("name_vi", ""), app.get("name_vi", ""), app.get("description", "")] + kw.get(app["id"], [])
        d = set()
        for t in app["tools"]:
            parts.append(t["name"].split(".")[-1].replace("_", " "))
            parts.append(t.get("description", ""))
            for k, p in (t.get("parameters") or {}).get("properties", {}).items():
                parts.append(k.replace("_", " "))
                parts.append(p.get("description", ""))
                parts += [str(e) for e in p.get("enum", [])]
                if "x-ref" in p:
                    d.add(p["x-ref"].split(".")[0])
        docs[app["id"]] = " ".join(x for x in parts if x)
        deps[app["id"]] = d - {app["id"]}
    return docs, deps


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.tok = {a: toks(t) for a, t in docs.items()}
        self.tf = {a: collections.Counter(t) for a, t in self.tok.items()}
        self.N = len(docs)
        self.df = collections.Counter(w for t in self.tok.values() for w in set(t))
        self.avg = sum(len(t) for t in self.tok.values()) / max(1, self.N)
        self.k1, self.b = k1, b

    def score(self, q_toks, app):
        tf = self.tf.get(app)
        if not tf:
            return 0.0
        L = len(self.tok[app])
        s = 0.0
        for w in set(q_toks):
            if w in tf:
                idf = math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5))
                s += idf * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * L / self.avg))
        return s


class Embedder:
    """sentence-transformers; tài liệu app mã hoá một lần, cache theo (model, hash doc)."""

    def __init__(self, model_name, docs, cache_dir=None, device="cpu"):
        import hashlib
        import numpy as np
        from sentence_transformers import SentenceTransformer
        self.np = np
        self.model = SentenceTransformer(model_name, device=device)
        self.name = model_name
        cache_id = model_name
        if os.path.isdir(model_name):                            # cache tài liệu app phải khác bản zero-shot của base
            cache_id = os.path.abspath(model_name) + str(os.path.getmtime(os.path.join(model_name, "finetune.json")))
        ft = os.path.join(model_name, "finetune.json")           # model fine-tune lưu cục bộ: lấy tên base để biết pyvi / tiền tố e5
        if os.path.isfile(ft):
            self.name = json.load(open(ft, encoding="utf-8")).get("base", model_name)
            model_name = self.name
        # model gốc PhoBERT (bkai vietnamese-bi-encoder, dangvantuan/vietnamese-embedding, SimCSE-PhoBERT, vietnamese-sbert):
        # model card yêu cầu tách từ bằng pyvi trước khi mã hoá
        self.seg = any(k in model_name.lower() for k in ("vietnamese-bi-encoder", "vietnamese-embedding", "phobert", "vietnamese-sbert"))
        if self.seg:
            try:
                from pyvi import ViTokenizer
                self.vt = ViTokenizer
            except ImportError:
                self.seg = False
        if self.seg:                                   # PhoBERT: 258 vị trí → cắt 256 token, nếu không tài liệu dài gây IndexError
            self.model.max_seq_length = min(self.model.max_seq_length or 256, 256)
        self.apps = sorted(docs)
        key = hashlib.sha1((cache_id + "\n" + "\n".join(f"{a}\t{docs[a]}" for a in self.apps)).encode()).hexdigest()[:16]
        p = os.path.join(cache_dir, f"docs_{key}.npy") if cache_dir else None
        if p and os.path.exists(p):
            self.E = np.load(p)
        else:
            self.E = self.encode([docs[a] for a in self.apps], passage=True)
            if p:
                os.makedirs(cache_dir, exist_ok=True)
                np.save(p, self.E)
        self.idx = {a: i for i, a in enumerate(self.apps)}

    def prep(self, texts, passage):
        if self.seg:
            texts = [self.vt.tokenize(t) for t in texts]
        if "e5" in self.name:
            texts = [("passage: " if passage else "query: ") + t for t in texts]
        return texts

    def encode(self, texts, passage=False):
        return self.model.encode(self.prep(texts, passage), normalize_embeddings=True, batch_size=32, show_progress_bar=False)

    def scores(self, q_emb, apps):
        return {a: float(q_emb @ self.E[self.idx[a]]) if a in self.idx else 0.0 for a in apps}


class CrossReranker:
    """Cross-encoder (query, tài liệu app) → điểm; dùng thay Embedder: `q_emb` là dict {app: điểm} tính sẵn cho mẫu."""

    def __init__(self, model_name, docs, device="cpu"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device, max_length=512)
        self.docs = docs
        self.name = model_name

    def score_all(self, queries, inst_lists, batch_size=64):
        pairs, where = [], []
        for i, (q, inst) in enumerate(zip(queries, inst_lists)):
            for a in inst:
                if a in self.docs:
                    pairs.append((q, self.docs[a]))
                    where.append((i, a))
        s = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        out = [{} for _ in queries]
        for (i, a), v in zip(where, s):
            out[i][a] = float(v)
        return out

    def scores(self, q, apps):
        return {a: q.get(a, float("-inf")) for a in apps}


class AppFilter:
    """keep = top-k₁(BM25) ∪ top-k₂(emb) ∪ {app điểm ≥ rel·max} ∪ app trong history ∪ đóng bao x-ref."""

    def __init__(self, catalog, keywords=None, emb=None, k_bm25=3, k_emb=3, rel=0.0, rules=True, use_history=True):
        self.docs, self.deps = app_docs(catalog, keywords)
        self.bm25 = BM25(self.docs)
        self.emb = emb
        self.k_bm25, self.k_emb, self.rel, self.rules, self.use_history = k_bm25, k_emb, rel, rules, use_history

    @staticmethod
    def query(item):
        hist_user = [t.get("audio_text", "") for t in item.get("history", []) if t["role"] == "user"]
        return (item.get("audio_text", "") or "") + (" " + hist_user[-1] if hist_user else "")

    def rank(self, item, q_emb=None):
        inst = [a for a in item["context"]["installed_apps"] if a != "sys"]
        q = toks(self.query(item))
        sb = {a: self.bm25.score(q, a) for a in inst}
        se = self.emb.scores(q_emb, inst) if (self.emb is not None and q_emb is not None) else {}
        return inst, sb, se

    def keep(self, item, q_emb=None, k_bm25=None, k_emb=None, rel=None):
        inst, sb, se = self.rank(item, q_emb)
        kb, ke = self.k_bm25 if k_bm25 is None else k_bm25, self.k_emb if k_emb is None else k_emb
        r = self.rel if rel is None else rel
        keep = set(sorted(inst, key=lambda a: -sb[a])[:kb])
        if r > 0 and sb:                                  # ngưỡng tương đối BM25: câu nhiều ý → nhiều app có điểm cao
            top = max(sb.values())
            if top > 0:
                keep |= {a for a in inst if sb[a] >= r * top}
        if se:
            keep |= set(sorted(inst, key=lambda a: -se[a])[:ke])
            if r > 0:
                top = max(se.values())
                lo = min(se.values())
                keep |= {a for a in inst if se[a] - lo >= r * (top - lo)}   # cosine: so trên khoảng [min, max]
        if self.rules:
            if self.use_history:
                keep |= {c["name"].split(".")[0] for t in item.get("history", []) if t["role"] == "assistant"
                         for c in (t.get("tool_calls") or [])} & set(inst)
            for _ in range(2):                           # đóng bao x-ref (contacts cho sms/ewallet, ...)
                keep |= {d for a in list(keep) for d in self.deps.get(a, ())} & set(inst)
        return keep, sb, se


def gold_apps(item):
    return {c["name"].split(".")[0] for c in item["label"]["tool_calls"]} - {"sys"}


def evaluate(a):
    catalog = json.load(open(f"{a.data}/train/train_tools_catalog.json", encoding="utf-8"))
    kw = json.load(open(KEYWORDS, encoding="utf-8")) if (a.keywords and os.path.exists(KEYWORDS)) else None
    rows = [json.loads(l) for l in open(f"{a.data}/train/train.jsonl", encoding="utf-8") if l.strip()]
    sp = json.load(open(a.splits or os.path.join(ROOT, "agent/clef/splits.json")))
    emb = None
    docs, _ = app_docs(catalog, kw)
    if a.rerank:
        t0 = time.time()
        emb = CrossReranker(a.rerank, docs, device=a.device)
        print(f"reranker {a.rerank}: tải {time.time() - t0:.0f}s", flush=True)
    elif a.emb:
        t0 = time.time()
        emb = Embedder(a.emb, docs, a.emb_cache, device=a.device)
        print(f"embedding {a.emb}: {len(docs)} app docs, {time.time() - t0:.0f}s", flush=True)
    f = AppFilter(catalog, kw, emb)
    qe = {}
    if a.rerank:
        t0 = time.time()
        S = emb.score_all([AppFilter.query(r) for r in rows], [[x for x in r["context"]["installed_apps"] if x != "sys"] for r in rows])
        qe = {r["id"]: S[i] for i, r in enumerate(rows)}
        print(f"chấm {len(rows)} query × app: {time.time() - t0:.0f}s ({1000 * (time.time() - t0) / len(rows):.0f} ms/câu, CPU batch)", flush=True)
    elif emb:
        t0 = time.time()
        E = emb.encode([AppFilter.query(r) for r in rows])
        qe = {r["id"]: E[i] for i, r in enumerate(rows)}
        print(f"mã hoá {len(rows)} query: {time.time() - t0:.0f}s ({1000 * (time.time() - t0) / len(rows):.0f} ms/câu, CPU batch)", flush=True)
    sets = {s: [r for r in rows if r["id"] in set(sp[s])] for s in ("train", "val_in", "val_ood")}
    sets["all"] = rows
    configs = []
    for kb in a.k_bm25:
        for ke in (a.k_emb if emb else [0]):
            for rel in a.rel:
                configs.append((kb, ke, rel))
    out = {"emb": a.emb or a.rerank, "keywords": bool(kw), "configs": {}}
    hdr = f"{'cấu hình':26} " + " ".join(f"{s:>22}" for s in sets)
    print("\n" + hdr + "\n" + " " * 27 + " ".join(f"{'recall / app giữ':>22}" for _ in sets))
    for kb, ke, rel in configs:
        name = f"bm25@{kb}" + (f" ∪ emb@{ke}" if emb else "") + (f" ∪ rel≥{rel}" if rel else "") + " +luật"
        line, res = f"{name:26} ", {}
        for s, rs in sets.items():
            n = hit = kept = 0
            miss = []
            for r in rs:
                g = gold_apps(r)
                keep, sb, se = f.keep(r, qe.get(r["id"]), kb, ke, rel)
                kept += len(keep)
                if not g:
                    continue
                n += 1
                if g <= keep:
                    hit += 1
                elif len(miss) < 400:
                    miss.append({"id": r["id"], "miss": sorted(g - keep), "kept": sorted(keep), "utt": r.get("audio_text", "")[:160]})
            rec = hit / max(1, n)
            res[s] = {"recall": round(rec, 4), "n": n, "kept_avg": round(kept / max(1, len(rs)), 2), "miss": miss}
            line += f"{100 * rec:9.1f}% / {kept / max(1, len(rs)):5.2f}      "
        print(line, flush=True)
        out["configs"][name] = res
    if a.out:
        json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("→", a.out)
    # app bị sót nhiều nhất ở cấu hình đầu (train)
    first = next(iter(out["configs"].values()))
    c = collections.Counter(m for x in first["train"]["miss"] for m in x["miss"])
    print("\napp bị sót nhiều nhất (train, cấu hình đầu):", c.most_common(12))
    for x in first["train"]["miss"][:8]:
        print(f"  {x['id']} sót {x['miss']} | giữ {x['kept']} | {x['utt']}")


def prep_texts(model_name, texts, passage, seg=None):
    if seg is not None:
        texts = [seg.tokenize(t) for t in texts]
    if "e5" in model_name:
        texts = [("passage: " if passage else "query: ") + t for t in texts]
    return texts


def finetune(a):
    """InfoNCE nhiều dương (như app_filter/kernel.py): softmax trên các app ĐÃ CÀI của chính mẫu, dương = app gold.
    Dữ liệu: lượt cuối + lượt history của split train; app held-out (val_ood) bị loại khỏi mọi tập ứng viên để val_ood
    vẫn là app lạ với embedder. Lưu model sentence-transformers vào --out (dùng lại bằng `eval --emb <out>`)."""
    import random
    import torch
    from sentence_transformers import SentenceTransformer
    catalog = json.load(open(f"{a.data}/train/train_tools_catalog.json", encoding="utf-8"))
    kw = json.load(open(KEYWORDS, encoding="utf-8")) if (a.keywords and os.path.exists(KEYWORDS)) else None
    docs, _ = app_docs(catalog, kw)
    sp = json.load(open(a.splits or os.path.join(ROOT, "agent/clef/splits.json")))
    held = set(sp["held_out_apps"])
    rows = [json.loads(l) for l in open(f"{a.data}/train/train.jsonl", encoding="utf-8") if l.strip()]
    train_ids = set(sp["train"])
    data = []
    for r in rows:
        if r["id"] not in train_ids:
            continue
        inst = [x for x in r["context"]["installed_apps"] if x != "sys" and x not in held]
        g = gold_apps(r)
        if g and g <= set(inst):
            data.append((AppFilter.query(r), inst, g))
        hist = r["history"]
        for j, t in enumerate(hist):                        # lượt history: user j-1 → assistant j có tool_calls
            if t["role"] == "assistant" and j > 0 and hist[j - 1]["role"] == "user" and t.get("tool_calls"):
                gh = {c["name"].split(".")[0] for c in t["tool_calls"]} - {"sys"}
                prev_user = [x.get("audio_text", "") for x in hist[:j - 1] if x["role"] == "user"]
                q = (hist[j - 1].get("audio_text", "") or "") + (" " + prev_user[-1] if prev_user else "")
                if gh and gh <= set(inst):
                    data.append((q, inst, gh))
    print(f"fine-tune {a.base}: {len(data)} ví dụ (lượt cuối + history, split train, bỏ app held-out {sorted(held)})", flush=True)
    dev = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(a.base, device=dev)
    seg = None
    if "vietnamese-bi-encoder" in a.base:
        from pyvi import ViTokenizer
        seg = ViTokenizer

    def enc(texts, passage):
        feats = model.tokenize(prep_texts(a.base, texts, passage, seg))
        feats = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in feats.items()}   # st 5.x có khoá không phải tensor
        e = model(feats)["sentence_embedding"]
        return torch.nn.functional.normalize(e, dim=-1)

    steps = a.epochs * math.ceil(len(data) / a.bs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, int(0.1 * steps))) * max(0.0, (steps - s) / steps))
    rng = random.Random(a.seed)
    model.train()
    t0 = time.time()
    for ep in range(a.epochs):
        rng.shuffle(data)
        tot = 0.0
        for i in range(0, len(data), a.bs):
            b = data[i:i + a.bs]
            apps = sorted({x for _, inst, _ in b for x in inst})
            ai = {x: k for k, x in enumerate(apps)}
            qe = enc([q for q, _, _ in b], False)
            de = enc([docs[x] for x in apps], True)
            sim = a.scale * qe @ de.T
            loss = 0
            for n, (_, inst, g) in enumerate(b):
                c = torch.tensor([ai[x] for x in inst], device=dev)
                pos = torch.tensor([x in g for x in inst], device=dev)
                lg = sim[n, c]
                loss = loss + (torch.logsumexp(lg, 0) - torch.logsumexp(lg[pos], 0))
            loss = loss / len(b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            tot += loss.item()
        print(f"  epoch {ep + 1}/{a.epochs}: loss {tot / math.ceil(len(data) / a.bs):.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    os.makedirs(a.out, exist_ok=True)
    model.save(a.out)
    json.dump({"base": a.base, "epochs": a.epochs, "lr": a.lr, "bs": a.bs, "scale": a.scale, "n_examples": len(data),
               "held_out_excluded": sorted(held), "keywords": bool(kw)}, open(os.path.join(a.out, "finetune.json"), "w"), indent=1)
    print("→", a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default=os.path.join(ROOT, "data/train+public_test"))
    t.add_argument("--splits")
    t.add_argument("--base", default="bkai-foundation-models/vietnamese-bi-encoder")
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--bs", type=int, default=32)
    t.add_argument("--scale", type=float, default=20.0)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", default="auto")
    t.add_argument("--no-keywords", dest="keywords", action="store_false")
    e = sub.add_parser("eval")
    e.add_argument("--data", default=os.path.join(ROOT, "data/train+public_test"))
    e.add_argument("--emb", help="tên model sentence-transformers (bỏ trống = chỉ BM25)")
    e.add_argument("--rerank", help="cross-encoder (query, tài liệu app) thay cho embedding")
    e.add_argument("--emb-cache", default=os.path.join(HERE, "emb_cache"))
    e.add_argument("--device", default="cpu")
    e.add_argument("--k-bm25", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    e.add_argument("--k-emb", type=int, nargs="+", default=[2, 3, 4, 5])
    e.add_argument("--rel", type=float, nargs="+", default=[0.0])
    e.add_argument("--no-keywords", dest="keywords", action="store_false")
    e.add_argument("--splits")
    e.add_argument("--out")
    a = ap.parse_args()
    finetune(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()
