"""Phân tích nguyên nhân điểm public test (CHỈ CHẨN ĐOÁN — không dùng để tune/train, giữ luật không fit theo test).

Public test không có nhãn câu cuối và không có transcript tham chiếu. Nguồn đối chiếu duy nhất có sẵn trong ĐẦU VÀO:
các lượt history — mỗi lượt user (có audio, có transcript ASR của mình) đi kèm lượt assistant với `tool_calls`.
  A. Độ phủ bộ lọc app trên cặp history (app gold = app trong tool_calls của lượt assistant ngay sau).
  B. Kiểm thực thể ASR: tham số chuỗi lẽ ra đến từ lời nói (tên, nơi, tiêu đề...) có xuất hiện trong transcript không.
  C. Hành vi FC ở câu cuối (output của run public) so với run val_in cùng model.
Mốc đối chiếu cho A/B: 405 đoạn giữ riêng của train (transcript gold `audio_text` và transcript ASR v2).

    python agent/analyze_public.py --pack <pack_b> --out-md plans/REPORT_public_rootcause_data.md
"""
import argparse
import collections
import json
import os
import re
import statistics as st
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "clef", "app_filter"))

NEW_APPS = {"ecommerce", "event_tickets", "grocery", "habit_tracker", "pantry", "parking", "public_transit", "recipes",
            "stocks", "train_bus_tickets", "translator"}
SKIP_RE = re.compile(r"^[a-z0-9_\-:.+/ ]+$")          # id, enum, ngày/giờ ISO, số
PUNCT = re.compile(r"[^\w\s]", re.U)


def norm(s, accents=True):
    s = unicodedata.normalize("NFC", str(s)).lower()
    if not accents:
        s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").replace("đ", "d")
    return " ".join(PUNCT.sub(" ", s).split())


def app_of(name):
    return name.split(".")[0]


def str_args(args):
    for k, v in (args or {}).items():
        for x in (v if isinstance(v, list) else [v]):
            if isinstance(x, str):
                yield k, x


def history_pairs(row):
    """(j, uid, gold_calls) cho mỗi lượt user có lượt assistant gọi tool ngay sau."""
    H = row["history"]
    for j, t in enumerate(H):
        if t["role"] != "user" or j + 1 >= len(H) or H[j + 1]["role"] != "assistant":
            continue
        calls = [c for c in (H[j + 1].get("tool_calls") or []) if c["name"] != "sys.ask_missing_info"]
        if calls:
            yield j, os.path.basename(t["audio"])[:-4], calls


def context_text(row, j):
    """Văn bản model đã thấy trước lượt j ngoài lời nói: kết quả tool, câu assistant, hồ sơ — để loại tham số chép từ đó."""
    parts = [json.dumps(row["context"].get("user_profile", {}), ensure_ascii=False)]
    for t in row["history"][:j]:
        if t["role"] == "assistant":
            parts.append(t.get("text", ""))
            parts.append(json.dumps(t.get("tool_results") or [], ensure_ascii=False))
    return norm(" ".join(parts))


def entity_check(rows, text_of, ctx_cache):
    """Tham số chuỗi lẽ ra đến từ lời nói → có trong transcript không (đủ dấu / bỏ dấu / ≥80% token)."""
    res = collections.defaultdict(lambda: collections.Counter())
    examples = collections.defaultdict(list)
    for r in rows:
        for j, uid, calls in history_pairs(r):
            hyp = text_of(r, j, uid)
            if hyp is None:
                continue
            h1, h0 = norm(hyp), norm(hyp, accents=False)
            ctx = ctx_cache(r, j)
            for c in calls:
                grp = "new" if app_of(c["name"]) in NEW_APPS else "known"
                for k, v in str_args(c.get("arguments")):
                    v1 = norm(v)
                    if len(v1) < 2 or SKIP_RE.match(v.strip().lower()) or len(v1.split()) > 8 or (v1 and v1 in ctx):
                        continue
                    toks = v1.split()
                    exact = f" {v1} " in f" {h1} "
                    noacc = f" {norm(v, False)} " in f" {h0} "
                    tokrec = sum(t in h1.split() for t in toks) / len(toks) >= 0.8
                    res[grp]["n"] += 1
                    res[grp]["exact"] += exact
                    res[grp]["noacc"] += noacc
                    res[grp]["tok80"] += tokrec
                    if not noacc and len(examples[grp]) < 25:
                        examples[grp].append((c["name"], k, v, hyp))
    return res, examples


def rate(c, k):
    return f"{100 * c[k] / c['n']:.1f}%" if c["n"] else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "train+public_test"))
    ap.add_argument("--pack", required=True)
    ap.add_argument("--public-asr", default=os.path.join(HERE, "runs", "asr", "public.v2_medium.asr.json"))
    ap.add_argument("--heldout-asr", default=os.path.join(HERE, "runs", "asr", "heldout.v2_medium.asr.json"))
    ap.add_argument("--heldout-manifest", default=os.path.join(HERE, "runs", "asr", "asr_heldout.manifest.json"))
    ap.add_argument("--public-run", default=os.path.join(HERE, "runs", "clef", "public_fmtA_48e_notau.jsonl"))
    ap.add_argument("--valin-run", default=os.path.join(HERE, "runs", "clef", "fmtA_48e_tau02.val_in.jsonl"))
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", default=os.path.join(HERE, "runs", "clef", "public_rootcause.json"))
    a = ap.parse_args()

    pub = [json.loads(l) for l in open(f"{a.data}/public_test/public_test.jsonl", encoding="utf-8")]
    tr_all = {json.loads(l)["id"]: json.loads(l) for l in open(f"{a.data}/train/train.jsonl", encoding="utf-8")}
    held_sids = {x["sid"] for x in json.load(open(a.heldout_manifest))["items"]}
    held = [tr_all[s] for s in sorted(held_sids)]
    pasr = json.load(open(a.public_asr, encoding="utf-8"))
    hasr = json.load(open(a.heldout_asr, encoding="utf-8"))["hyps"]
    ctxc = {}

    def ctx_cache(r, j):
        k = (r["id"], j)
        if k not in ctxc:
            ctxc[k] = context_text(r, j)
        return ctxc[k]

    out = {}
    # ---------- B. kiểm thực thể ASR ----------
    eb = {}
    eb["public_v2"] = entity_check(pub, lambda r, j, uid: pasr.get(uid), ctx_cache)
    eb["held_v2"] = entity_check(held, lambda r, j, uid: hasr.get(uid), ctx_cache)
    eb["held_gold"] = entity_check(held, lambda r, j, uid: r["history"][j].get("audio_text"), ctx_cache)
    out["entity"] = {k: {g: dict(c) for g, c in v[0].items()} for k, v in eb.items()}

    # ---------- A. độ phủ bộ lọc trên cặp history ----------
    from pack import AppFilterPack, tier1_for
    cat_pub = json.load(open(f"{a.data}/public_test/public_test_tools_catalog.json", encoding="utf-8"))
    cat_tr = json.load(open(f"{a.data}/train/train_tools_catalog.json", encoding="utf-8"))
    pack_pub = AppFilterPack.load(a.pack, catalog=cat_pub)
    pack_tr = AppFilterPack.load(a.pack, catalog=cat_tr)

    def filt(rows, pk, asr, use_gold_text=False):
        c = collections.defaultdict(collections.Counter)
        misses = collections.Counter()
        sizes = []
        for r in rows:
            for j, uid, calls in history_pairs(r):
                gold = {app_of(x["name"]) for x in calls} - {"sys"}
                if not gold:
                    continue
                pseudo = {**r, "id": uid, "history": r["history"][:j], "audio": r["history"][j]["audio"],
                          "audio_text": r["history"][j].get("audio_text", "")}
                t1 = set(tier1_for(pk, pseudo, None if use_gold_text else asr))
                sizes.append(len(t1))
                grp = "new" if gold & NEW_APPS else "known"
                c[grp]["n"] += 1
                c[grp]["all_in"] += gold <= t1
                for g in gold - t1:
                    misses[g] += 1
        return c, misses, sizes

    fa = {"public_v2": filt(pub, pack_pub, pasr), "held_v2": filt(held, pack_tr, hasr),
          "held_gold": filt(held, pack_tr, None, use_gold_text=True)}
    out["filter"] = {k: {"groups": {g: dict(c) for g, c in v[0].items()}, "misses": dict(v[1].most_common(15)),
                         "tier1_mean": round(st.mean(v[2]), 2) if v[2] else None} for k, v in fa.items()}

    # ---------- C. hành vi FC ở câu cuối ----------
    def prof(path):
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        g = collections.defaultdict(list)
        for z in rows:
            apps = {app_of(c["name"]) for c in z["tool_calls"]}
            grp = "new" if apps & NEW_APPS else ("empty" if not z["tool_calls"] else "known")
            q1 = [p for p in z.get("probs", []) if p["kind"] == "q1"]
            g[grp].append({"n_calls": len(z["tool_calls"]), "n_invalid": z.get("n_invalid", 0),
                           "anomaly": z.get("anomaly", 0), "ask": any(c["name"] == "sys.ask_missing_info" for c in z["tool_calls"]),
                           "conf_min": min((p["conf"] for p in q1), default=None),
                           "conf_first": q1[0]["conf"] if q1 else None, "steps": z.get("steps", 0)})
        summ = {}
        for grp, L in g.items():
            cm = [x["conf_min"] for x in L if x["conf_min"] is not None]
            summ[grp] = {"mẫu": len(L), "call/mẫu": round(st.mean(x["n_calls"] for x in L), 2),
                         "mẫu có call bị từ chối": sum(x["n_invalid"] > 0 for x in L),
                         "tổng call bị từ chối": sum(x["n_invalid"] for x in L),
                         "anomaly": sum(bool(x["anomaly"]) for x in L), "hỏi lại": sum(x["ask"] for x in L),
                         "conf_min trung vị": round(st.median(cm), 3) if cm else None,
                         "conf_min<0.5": sum(c < 0.5 for c in cm)}
        return summ, rows

    pub_prof, pub_rows = prof(a.public_run)
    val_prof, _ = prof(a.valin_run)
    out["fc_profile"] = {"public": pub_prof, "val_in": val_prof}
    json.dump(out, open(a.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---------- markdown ----------
    L = ["# Dữ liệu chẩn đoán public test (tự sinh bởi agent/analyze_public.py)", ""]
    L += ["## A. Độ phủ bộ lọc app trên cặp history (app gold ⊆ tầng 1)", "",
          "| Nguồn | nhóm | cặp | đủ app gold trong tầng 1 | tầng 1 TB |", "|---|---|---|---|---|"]
    for k, v in fa.items():
        for g, c in sorted(v[0].items()):
            L.append(f"| {k} | {g} | {c['n']} | {rate(c, 'all_in')} | {out['filter'][k]['tier1_mean']} |")
    L += ["", "App hay bị bỏ sót (public_v2): " + ", ".join(f"{k} {n}" for k, n in fa["public_v2"][1].most_common(12)), ""]
    L += ["## B. Thực thể trong tham số có nằm trong transcript không", "",
          "| Nguồn transcript | nhóm | số tham số | khớp đủ dấu | khớp bỏ dấu | ≥80% token |", "|---|---|---|---|---|---|"]
    for k, (res, _) in eb.items():
        for g, c in sorted(res.items()):
            L.append(f"| {k} | {g} | {c['n']} | {rate(c, 'exact')} | {rate(c, 'noacc')} | {rate(c, 'tok80')} |")
    for g in ("new", "known"):
        ex = eb["public_v2"][1].get(g, [])
        if ex:
            L += ["", f"Ví dụ trượt (public, nhóm {g}):", ""]
            L += [f"- `{n}` {k}=`{v}` ← transcript: \"{h}\"" for n, k, v, h in ex[:15]]
    L += ["", "## C. Hành vi FC ở câu cuối", "", "| run | nhóm | " + " | ".join(next(iter(pub_prof.values())).keys()) + " |",
          "|---|---|" + "---|" * len(next(iter(pub_prof.values())))]
    for name, pf in (("public", pub_prof), ("val_in", val_prof)):
        for g, s in sorted(pf.items()):
            L.append(f"| {name} | {g} | " + " | ".join(str(v) for v in s.values()) + " |")
    # mẫu đọc tay: câu cuối của mẫu có cài app mới
    L += ["", "## D. 30 câu cuối (mẫu có cài app mới) — transcript và call của model", ""]
    byid = {z["id"]: z for z in pub_rows}
    shown = 0
    for r in pub:
        if shown >= 30:
            break
        inst_new = sorted(set(r["context"]["installed_apps"]) & NEW_APPS)
        if not inst_new:
            continue
        z = byid[r["id"]]
        q1 = [p for p in z.get("probs", []) if p["kind"] == "q1"]
        L.append(f"- **{r['id']}** (app mới đã cài: {', '.join(inst_new)}) \"{pasr.get(r['id'], '')}\" → "
                 + (", ".join(f"`{c['name']}`" for c in z["tool_calls"]) or "_(không gọi)_")
                 + (f" · conf_min {min(p['conf'] for p in q1):.2f}" if q1 else "")
                 + (f" · call bị từ chối {z['n_invalid']}" if z.get("n_invalid") else ""))
        shown += 1
    open(a.out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
