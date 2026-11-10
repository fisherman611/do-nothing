"""E2/E3: fine-tune embedding lọc app v2 (plans/PLAN_48h.md §1) + chấm trên mọi tập val.

    python train_v2.py --root <thư mục dữ liệu phẳng> --config A|B|C --out <dir> [--dry-run] [--eval-only]

Thư mục dữ liệu phẳng (`make_kaggle_ds.sh` dựng): train.jsonl, train_tools_catalog.json, public_test_tools_catalog.json
(chỉ catalog), splits.json, app_keywords.json, train_alias.jsonl, train_iface.jsonl, val_alias.jsonl, train_phonetic.jsonl,
alias_map.json, iface_map.json, newapps_catalog.json, newapps_keywords.json, newapps_utts.jsonl, filter.py.

Cấu hình (cùng base vietnamese-bi-encoder, lr 2e-5, bs 32, 3 epoch, warmup 10%, scale 20, seed 42):
  A  train gốc (lượt cuối + history) + alias + iface + phonetic; InfoNCE trên app đã cài
  B  A + newapps train + mô tả diễn đạt lại (doc ngẫu nhiên) + rụng 30% từ khoá + nhiễu ASR giả
  C  B + 0.5·MNRL trong batch + 0.5·InfoNCE có thêm 2 âm tính khó ngoài tập cài
Chống rò: bỏ mọi mẫu/ứng viên thuộc app held-out (val_ood) — kể cả app alias/iface dựng từ app held-out; từ khoá và
diễn đạt lại chỉ của app mới split=train; app mới val/test không bao giờ vào train. Chọn epoch theo emb-only recall@5
trên val_ood (đúng plan), chấm thêm val_in, val_alias, newapps (val_in/test_in/val_newapps/test_newapps/held_cross).
"""
import argparse
import ast
import collections
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def as_obj(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except ValueError:
            return ast.literal_eval(x)
    return x


# ---------------------------------------------------------------- tài liệu app
def camel_words(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return s.replace("_", " ").replace(".", " ").lower()


def doc_text(app, kw=None, desc=None):
    """Như filter.app_docs: tên + mô tả + tên/mô tả tool + tham số + enum + từ khoá."""
    name = app.get("name_vi") or camel_words(app["id"])
    parts = [name, name, desc if desc is not None else app.get("description", "")] + list(kw or [])
    deps = set()
    for t in app["tools"]:
        parts.append(camel_words(t["name"].split(".")[-1]))
        parts.append(t.get("description", ""))
        for k, p in (t.get("parameters") or {}).get("properties", {}).items():
            parts.append(camel_words(k))
            parts.append(p.get("description", ""))
            parts += [str(e) for e in p.get("enum", [])]
            if "x-ref" in p:
                deps.add(p["x-ref"].split(".")[0])
    return " ".join(x for x in parts if x), deps - {app["id"]}


class Docs:
    """Kho tài liệu: key → (app dict, danh sách từ khoá, các mô tả thay thế). Một app có thể có nhiều biến thể."""

    def __init__(self):
        self.apps, self.kw, self.paras = {}, {}, {}

    def add(self, key, app, kw=(), paras=()):
        self.apps[key] = app
        self.kw[key] = list(kw)
        self.paras[key] = list(paras)

    def text(self, key, rng=None, kw_drop=0.0, para=False):
        app = self.apps[key]
        kw = self.kw[key]
        if rng is not None and kw_drop > 0:
            kw = [k for k in kw if rng.random() >= kw_drop]
        desc = None
        if para and rng is not None and self.paras[key] and rng.random() < 0.5:
            desc = rng.choice(self.paras[key])
        return doc_text(app, kw, desc)[0]

    def deps(self, key):
        return doc_text(self.apps[key])[1]


# ---------------------------------------------------------------- nhiễu ASR giả
SWAPS = [("s", "x"), ("x", "s"), ("tr", "ch"), ("ch", "tr"), ("gi", "d"), ("d", "gi"), ("r", "d"), ("l", "n"), ("n", "l")]


def strip_acc(s):
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def asr_noise(q, rng):
    if rng.random() < 0.5:                                   # ASR: chữ thường, không dấu câu
        q = re.sub(r"[^\w\s]", " ", q.lower())
    w = q.split()
    out = []
    for x in w:
        r = rng.random()
        if r < 0.04 and len(w) > 5:                          # rơi từ
            continue
        if r < 0.08:                                         # phụ âm đầu vùng miền
            for a, b in SWAPS:
                if x.startswith(a):
                    x = b + x[len(a):]
                    break
        elif r < 0.11:                                       # mất dấu một từ
            x = strip_acc(x)
        out.append(x)
    return " ".join(out) if out else q


# ---------------------------------------------------------------- dữ liệu
class Item:
    __slots__ = ("id", "q", "inst", "gold", "hist", "src")

    def __init__(self, id, q, inst, gold, hist=(), src=""):
        self.id, self.q, self.inst, self.gold, self.hist, self.src = id, q, list(inst), set(gold), set(hist), src


def query_of(r):
    hu = [t.get("audio_text", "") for t in r.get("history", []) if t["role"] == "user"]
    return (r.get("audio_text", "") or "") + (" " + hu[-1] if hu else "")


def gold_of(calls):
    return {c["name"].split(".")[0] for c in calls} - {"sys"}


def hist_apps(r):
    return {c["name"].split(".")[0] for t in r.get("history", []) if t["role"] == "assistant"
            for c in (t.get("tool_calls") or [])} - {"sys"}


def load_all(root):
    J = lambda f: json.load(open(os.path.join(root, f), encoding="utf-8"))   # noqa: E731
    sp = J("splits.json")
    held = set(sp["held_out_apps"])
    tr_cat = J("train_tools_catalog.json")["apps"]
    te_cat = J("public_test_tools_catalog.json")["apps"]
    kw = {k: v for k, v in J("app_keywords.json").items() if not k.startswith("_")}
    new_cat = J("newapps_catalog.json")["apps"]
    new_kw = J("newapps_keywords.json")
    D = Docs()
    for a in tr_cat + te_cat:                                # app gốc: key = id
        if a["id"] != "sys" and a["id"] not in D.apps:
            D.add(a["id"], a, kw.get(a["id"], []))
    new_split = {}
    for a in new_cat:                                        # app mới: từ khoá + diễn đạt lại chỉ cho split=train
        new_split[a["id"]] = a["split"]
        tr = a["split"] == "train"
        D.add(a["id"], a, new_kw.get(a["id"], []) if tr else [], a.get("description_paraphrases", []) if tr else [])
    # app alias/iface → app gốc (để chặn app held-out)
    origin = {}
    for f in ("alias_map.json", "val_alias_map.json"):
        p = os.path.join(root, f)
        if os.path.exists(p):
            for v in json.load(open(p, encoding="utf-8")).values():
                origin[v["app"]["to"]] = v["app"]["from"]
    for v in J("iface_map.json").values():
        origin[v["app_to"]] = v["app_from"]
    return sp, held, D, new_split, origin


def overlay_keys(r, D, tag):
    """catalog_overlay của mẫu alias/iface → key tài liệu riêng của mẫu cho các app bị đổi tên/đổi giao diện."""
    ov = as_obj(r.get("catalog_overlay") or [])
    by = collections.defaultdict(list)
    for t in ov:
        by[t.get("app") or t["name"].split(".")[0]].append(t)
    m = {}
    for app, tools in by.items():
        if app in D.apps and doc_text(D.apps[app])[0] == doc_text({"id": app, "tools": tools, "name_vi": D.apps[app].get("name_vi"), "description": D.apps[app].get("description")})[0]:
            m[app] = app
            continue
        h = hashlib.md5(json.dumps(tools, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:10]
        key = f"{tag}:{app}:{h}"
        if key not in D.apps:
            D.add(key, {"id": app, "tools": tools, "name_vi": camel_words(app), "description": ""})
        m[app] = key
    return m


def mk_item(r, D, origin, held, src, base_id=None, ov=True):
    m = overlay_keys(r, D, src) if (ov and r.get("catalog_overlay")) else {}
    inst_raw = [a for a in r["context"]["installed_apps"] if a != "sys"]
    is_held = lambda a: a in held or origin.get(a) in held             # noqa: E731
    return m, inst_raw, is_held


def build_sets(root, cfg):
    sp, held, D, new_split, origin = load_all(root)
    ids = {k: set(v) for k, v in sp.items() if isinstance(v, list)}
    train_ids = ids["train"]
    rows = jl(os.path.join(root, "train.jsonl"))
    by_id = {r["id"]: r for r in rows}
    T, V = [], collections.defaultdict(list)
    counts = collections.Counter()

    def to_item(r, src, train, idx_hist=None):
        m = overlay_keys(r, D, src) if r.get("catalog_overlay") else {}
        inst = [a for a in r["context"]["installed_apps"] if a != "sys"]
        bad = lambda a: a in held or origin.get(a) in held             # noqa: E731
        key = lambda a: m.get(a, a)                                    # noqa: E731
        if idx_hist is None:
            g = gold_of(r["label"]["tool_calls"])
            q = query_of(r)
            h = hist_apps(r)
        else:
            j = idx_hist
            hist = r["history"]
            g = gold_of(hist[j]["tool_calls"])
            prev_user = [x.get("audio_text", "") for x in hist[:j - 1] if x["role"] == "user"]
            q = (hist[j - 1].get("audio_text", "") or "") + (" " + prev_user[-1] if prev_user else "")
            h = {c["name"].split(".")[0] for t in hist[:j - 1] if t["role"] == "assistant" for c in (t.get("tool_calls") or [])} - {"sys"}
        if train:
            if any(bad(a) for a in g):
                return None
            inst = [a for a in inst if not bad(a)]
        if not g or not g <= set(inst):
            return None
        for a in inst:
            if key(a) not in D.apps:
                return None
        return Item(r["id"] + ("" if idx_hist is None else f"#h{idx_hist}"), q, [key(a) for a in inst], {key(a) for a in g},
                    {key(a) for a in h if a in inst}, src)

    def add_train(r, src):
        it = to_item(r, src, True)
        if it:
            T.append(it)
            counts[src] += 1
        hist = r.get("history", [])
        for j, t in enumerate(hist):
            if t["role"] == "assistant" and j > 0 and hist[j - 1]["role"] == "user" and t.get("tool_calls"):
                it = to_item(r, src, True, j)
                if it:
                    T.append(it)
                    counts[src + "_hist"] += 1

    for r in rows:
        if r["id"] in train_ids:
            add_train(r, "orig")
    for f, src in (("train_alias.jsonl", "alias"), ("train_iface.jsonl", "iface"), ("train_phonetic.jsonl", "phonetic")):
        p = os.path.join(root, f)
        if not os.path.exists(p):
            continue
        for r in jl(p):
            if r["id"].split("#")[0] in train_ids:
                add_train(r, src)
    # newapps
    nrows = jl(os.path.join(root, "newapps_utts.jsonl"))

    def new_item(r):
        inst = [a for a in r["installed"] if a in D.apps]
        return Item(r["id"], r["text"], inst, set(r["gold_apps"]), (), "newapps")

    if cfg in ("B", "C"):
        for r in nrows:
            if r["split"] == "train":
                it = new_item(r)
                if any(a in held for a in it.gold):
                    continue
                it.inst = [a for a in it.inst if a not in held]
                T.append(it)
                counts["newapps"] += 1
    # tập đánh giá
    for name in ("val_in", "val_ood"):
        for i in ids[name]:
            it = to_item(by_id[i], name, False)
            if it:
                V[name].append(it)
    p = os.path.join(root, "val_alias.jsonl")
    if os.path.exists(p):
        for r in jl(p):
            it = to_item(r, "val_alias", False)
            if it:
                V["val_alias"].append(it)
    for r in nrows:
        if r["split"] != "train":
            V["new_" + r["split"]].append(new_item(r))
    # rò: không app held-out / app mới val-test nào trong dữ liệu train
    leak_new = {a for a, s in new_split.items() if s != "train"}
    for it in T:
        for k in it.inst:
            app = D.apps[k]["id"]
            assert app not in held and origin.get(app) not in held, (it.id, app)
            assert app not in leak_new, (it.id, app)
    return T, V, D, counts, held, new_split


# ---------------------------------------------------------------- BM25 + luật (dùng lại filter.py)
def bm25_index(D, root):
    sys.path.insert(0, root)
    sys.path.insert(0, HERE)
    import filter as F
    texts = {k: D.text(k) for k in D.apps}
    return F, F.BM25(texts)


# ---------------------------------------------------------------- huấn luyện / chấm
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", choices=["A", "B", "C"], required=True)
    ap.add_argument("--base", default="bkai-foundation-models/vietnamese-bi-encoder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--kw-drop", type=float, default=0.3)
    ap.add_argument("--w-mnrl", type=float, default=0.5)
    ap.add_argument("--w-hard", type=float, default=0.5)
    ap.add_argument("--n-hard", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help="chỉ lấy N ví dụ train và N mẫu mỗi tập chấm (thử nhanh)")
    ap.add_argument("--no-zs", action="store_true", help="bỏ chấm zero-shot trước khi train")
    a = ap.parse_args()
    random.seed(a.seed)
    T, V, D, counts, held, new_split = build_sets(a.root, a.config)
    print(f"config {a.config}: {len(T)} ví dụ train — " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), flush=True)
    print("tập chấm: " + ", ".join(f"{k} {len(v)}" for k, v in V.items()), f"| tài liệu {len(D.apps)} | held-out {sorted(held)}", flush=True)
    if a.dry_run:
        return
    if a.smoke:
        rng0 = random.Random(0)
        T = rng0.sample(T, min(a.smoke, len(T)))
        V = {k: v[:a.smoke] for k, v in V.items()}
    F, bm = bm25_index(D, a.root)
    deps = {k: set() for k in D.apps}
    for k in D.apps:                                          # x-ref theo id app → key cùng mẫu xử lý lúc chấm
        deps[k] = D.deps(k)

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(a.base, device=dev)
    model.max_seq_length = min(a.max_len, 256)
    try:
        model[0].auto_model.gradient_checkpointing_enable()
    except Exception as e:  # noqa: BLE001
        print("không bật được gradient checkpointing:", e)
    from pyvi import ViTokenizer
    seg_cache = {}

    def seg(t):
        if t not in seg_cache:
            seg_cache[t] = ViTokenizer.tokenize(t)
        return seg_cache[t]

    def enc_train(texts):
        f = model.tokenize([seg(t) for t in texts])
        f = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in f.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            e = model(f)["sentence_embedding"]
        return torch.nn.functional.normalize(e.float(), dim=-1)

    def enc_eval(texts):
        return model.encode([seg(t) for t in texts], normalize_embeddings=True, batch_size=64, show_progress_bar=False,
                            convert_to_numpy=True)

    # BM25 trước cho mọi item (cố định), dùng cho chấm và âm tính khó
    def bm25_scores(it, cands):
        q = F.toks(it.q)
        return {k: bm.score(q, k) for k in cands}

    def evaluate(tag):
        model.eval()
        keys = sorted({k for its in V.values() for it in its for k in it.inst})
        E = dict(zip(keys, enc_eval([D.text(k) for k in keys])))
        res = {}
        for name, its in V.items():
            Q = enc_eval([it.q for it in its])
            agg = collections.defaultdict(lambda: [0, 0])
            for n, it in enumerate(its):
                se = {k: float(Q[n] @ E[k]) for k in it.inst}
                sb = bm25_scores(it, it.inst)
                rb = sorted(it.inst, key=lambda k: -sb[k])
                re_ = sorted(it.inst, key=lambda k: -se[k])
                id2key = {D.apps[k]["id"]: k for k in it.inst}

                def close(keep):
                    keep = set(keep) | (it.hist & set(it.inst))
                    for _ in range(2):
                        keep |= {id2key[d] for k in list(keep) for d in deps.get(k, ()) if d in id2key}
                    return keep
                cfgs = {f"emb@{k}": close(re_[:k]) for k in (3, 4, 5)}
                for kb, ke in ((3, 4), (4, 5), (3, 5), (4, 6)):
                    cfgs[f"bm25@{kb}+emb@{ke}"] = close(rb[:kb] + re_[:ke])
                for c, keep in cfgs.items():
                    agg[c][0] += it.gold <= keep
                    agg[c][1] += len(keep)
            res[name] = {c: {"recall": round(h / max(1, len(its)), 4), "kept": round(k / max(1, len(its)), 2)} for c, (h, k) in agg.items()}
            res[name]["n"] = len(its)
        print(f"\n[{tag}]  recall % / app giữ")
        cols = ["emb@5", "bm25@3+emb@4", "bm25@4+emb@5"]
        print(f"{'tập':18}" + "".join(f"{c:>20}" for c in cols))
        for name, r in res.items():
            print(f"{name:18}" + "".join(f"{100 * r[c]['recall']:12.1f} /{r[c]['kept']:5.1f} " for c in cols) + f"  (n={r['n']})")
        sys.stdout.flush()
        model.train()
        return res

    os.makedirs(a.out, exist_ok=True)
    log = {"config": a.config, "args": vars(a), "counts": dict(counts), "n_train": len(T), "evals": {}}
    if not a.no_zs:
        log["evals"]["zero_shot"] = evaluate("zero-shot")
    # âm tính khó: app gốc + app mới split=train, không held-out, không trong tập cài; theo BM25 của câu
    pool = [k for k in D.apps if ":" not in k and k not in held and new_split.get(k, "train") == "train"]
    hard = {}
    if a.config == "C":
        for it in T:
            s = bm25_scores(it, [k for k in pool if k not in it.inst])
            hard[it.id] = [k for k in sorted(s, key=lambda k: -s[k])[:a.n_hard] if s[k] > 0]
    rng = random.Random(a.seed)
    steps = a.epochs * math.ceil(len(T) / a.bs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, int(0.1 * steps))) * max(0.0, (steps - s) / steps))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    aug = a.config in ("B", "C")
    best, best_ep = -1, -1
    model.train()
    t0 = time.time()
    for ep in range(a.epochs):
        rng.shuffle(T)
        tot, nb = 0.0, 0
        for i in range(0, len(T), a.bs):
            b = T[i:i + a.bs]
            keys = sorted({k for it in b for k in it.inst} | {k for it in b for k in hard.get(it.id, [])})
            ki = {k: j for j, k in enumerate(keys)}
            dtexts = [D.text(k, rng, a.kw_drop if aug else 0.0, aug) for k in keys]   # một biến thể/app/batch
            qs = [asr_noise(it.q, rng) if (aug and rng.random() < 0.5) else it.q for it in b]
            qe = enc_train(qs)
            de = enc_train(dtexts)
            sim = a.scale * qe @ de.T
            loss = 0
            for n, it in enumerate(b):
                c = torch.tensor([ki[k] for k in it.inst], device=dev)
                pos = torch.tensor([k in it.gold for k in it.inst], device=dev)
                lg = sim[n, c]
                l_inst = torch.logsumexp(lg, 0) - torch.logsumexp(lg[pos], 0)
                if a.config == "C":
                    allc = torch.arange(len(keys), device=dev)                       # MNRL: mọi tài liệu trong batch
                    posa = torch.tensor([k in it.gold for k in keys], device=dev)
                    l_mnrl = torch.logsumexp(sim[n, allc], 0) - torch.logsumexp(sim[n, allc][posa], 0)
                    hk = it.inst + [k for k in hard.get(it.id, []) if k not in it.inst]
                    ch = torch.tensor([ki[k] for k in hk], device=dev)
                    ph = torch.tensor([k in it.gold for k in hk], device=dev)
                    lh = sim[n, ch]
                    l_hard = torch.logsumexp(lh, 0) - torch.logsumexp(lh[ph], 0)
                    loss = loss + l_inst + a.w_mnrl * l_mnrl + a.w_hard * l_hard
                else:
                    loss = loss + l_inst
            loss = loss / len(b)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            tot += loss.item()
            nb += 1
            if nb % 50 == 0:
                print(f"  ep {ep + 1} bước {nb}/{math.ceil(len(T) / a.bs)} loss {tot / nb:.4f} ({time.time() - t0:.0f}s)", flush=True)
        print(f"epoch {ep + 1}/{a.epochs}: loss {tot / max(1, nb):.4f} ({time.time() - t0:.0f}s)", flush=True)
        r = evaluate(f"epoch {ep + 1}")
        log["evals"][f"epoch{ep + 1}"] = r
        score = r["val_ood"]["emb@5"]["recall"]
        if score > best:
            best, best_ep = score, ep + 1
            model.save(os.path.join(a.out, "best"))
            json.dump({"base": a.base, "config": a.config, "epoch": best_ep, "val_ood_emb@5": best, "n_examples": len(T),
                       "held_out_excluded": sorted(held)}, open(os.path.join(a.out, "best", "finetune.json"), "w"), indent=1)
        json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)
    log["best_epoch"] = best_ep
    log["best"] = log["evals"][f"epoch{best_ep}"]
    json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)
    print(f"\nchọn epoch {best_ep} (val_ood emb@5 = {100 * best:.1f}%) → {a.out}/best", flush=True)


if __name__ == "__main__":
    main()
