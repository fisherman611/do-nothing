"""Lấy tập con từ một bộ Clef đã build (vd fmtA_full) với coverage đầy đủ + ưu tiên app có trong catalog public test.

Đơn vị chọn là MẪU (mọi bước của một id đi cùng nhau). Chỉ đọc catalog tool của public test (tên app) — không mở
public_test.jsonl hay *_environment/.

    python agent/clef/make_subset.py --src agent/clef/data/fmtA_full --out agent/clef/data/fmtA_mid --target 11000

Thứ tự chọn:
  1. coverage: mỗi tool gold xuất hiện >= --min-tool lần (hoặc hết số có), mỗi loại mẫu (gốc/#h/#ph1/#ph2/#al/#if)
     >= --min-kind mẫu, mỗi loại kết thúc (DỪNG ngay / hỏi lại / gọi tool) có mặt;
  2. lấp tới --target ví dụ bằng lấy mẫu có trọng số: mẫu chạm app có trong test = 1.0, chỉ app train-only =
     --w-train-only, mẫu không gọi tool = 1.0; mẫu gốc (không augment) luôn ưu tiên giữ.
"""
import argparse
import collections
import json
import os
import random
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def kind_of(i):
    tag = i.split("#")[1:] if "#" in i else []
    if not tag:
        return "base"
    k = [t for t in tag if not t.startswith("h")]
    base = (k[0][:3] if k[0].startswith("ph") else k[0][:2]) if k else "base"
    return base + ("+h" if any(t.startswith("h") for t in tag) else "")


def apps_of(catalog_path):
    c = json.load(open(catalog_path, encoding="utf-8"))
    A = c["apps"]
    return {(a.get("name") or a.get("app") or a.get("id")) for a in A} if isinstance(A, list) else set(A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=11000, help="số ví dụ (bước) mong muốn")
    ap.add_argument("--min-tool", type=int, default=8)
    ap.add_argument("--min-kind", type=int, default=150)
    ap.add_argument("--w-train-only", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=os.path.join(ROOT, "data/train+public_test"))
    a = ap.parse_args()
    rng = random.Random(a.seed)

    test_apps = apps_of(f"{a.data}/public_test/public_test_tools_catalog.json") - {"sys"}
    orig = {}
    for l in open(f"{a.data}/train/train.jsonl", encoding="utf-8"):
        d = json.loads(l)
        orig[d["id"]] = d

    groups = collections.OrderedDict()
    for l in open(f"{a.src}/train.jsonl", encoding="utf-8"):
        e = json.loads(l)
        groups.setdefault(e["id"], []).append(l)
    info = {}
    for i, lines in groups.items():
        ex = [json.loads(l) for l in lines]
        base = orig.get(i.split("#")[0])
        # app của mẫu lấy từ nhãn GỐC (biến thể alias/iface đổi tên tool nhưng cùng ý định): mẫu chính → label,
        # mẫu phụ #h<j> → lượt history j của mẫu gốc (biến thể giữ nguyên cấu trúc history)
        htag = [t for t in i.split("#")[1:] if t.startswith("h") and t[1:].isdigit()]
        if base is None:
            calls = [e["name"] for e in ex if e["name"] and e["name"] != "DỪNG"]
        elif htag:
            j = int(htag[0][1:])
            calls = [c["name"] for c in (base["history"][j].get("tool_calls") or [])] if j < len(base["history"]) else []
        else:
            calls = [c["name"] for c in base["label"]["tool_calls"]]
        apps = {n.split(".")[0] for n in calls} - {"sys"}
        names = [e["name"] for e in ex if e["name"]]
        end = "ask" if any(n == "sys.ask_missing_info" for n in names) else \
            ("stop0" if len(ex) == 1 else "call")
        w = 1.0 if (not apps or apps & test_apps) else a.w_train_only
        info[i] = {"n": len(ex), "names": names, "canon": calls, "kind": kind_of(i), "end": end, "w": w,
                   "test_app": bool(apps & test_apps), "apps": apps}

    chosen, n_ex = set(), 0

    def take(i):
        nonlocal n_ex
        if i not in chosen:
            chosen.add(i)
            n_ex += info[i]["n"]

    ids = list(info)
    rng.shuffle(ids)
    # 0. giữ toàn bộ mẫu gốc (lượt cuối thật, không augment)
    for i in ids:
        if info[i]["kind"] == "base":
            take(i)
    # 1a. mỗi tool GỐC (tên trong catalog train; alias/iface quy về tool gốc) >= min_tool lần, ưu tiên mẫu ngắn
    cnt = collections.Counter(n for i in chosen for n in info[i]["canon"])
    by_tool = collections.defaultdict(list)
    for i in ids:
        for n in set(info[i]["canon"]):
            by_tool[n].append(i)
    for n, lst in sorted(by_tool.items(), key=lambda x: len(x[1])):
        for i in sorted(lst, key=lambda i: info[i]["n"]):
            if cnt[n] >= a.min_tool:
                break
            if i not in chosen:
                take(i)
                cnt.update(info[i]["canon"])
    # 1b. mỗi loại mẫu + mỗi kiểu kết thúc
    for key, m in (("kind", a.min_kind), ("end", a.min_kind)):
        have = collections.Counter(info[i][key] for i in chosen)
        for i in ids:
            v = info[i][key]
            if have[v] < m and i not in chosen:
                take(i)
                have[v] += 1
    n_cov = n_ex
    # 2. lấp tới target theo trọng số (Efraimidis–Spirakis: khoá u^(1/w))
    rest = sorted((i for i in ids if i not in chosen), key=lambda i: -rng.random() ** (1 / info[i]["w"]))
    for i in rest:
        if n_ex >= a.target:
            break
        take(i)

    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/train.jsonl", "w", encoding="utf-8") as f:
        for i, lines in groups.items():          # giữ thứ tự gốc
            if i in chosen:
                f.writelines(lines)
    for fn in ("val_in.jsonl", "val_in.jsonl.meta.json"):
        if os.path.exists(f"{a.src}/{fn}"):
            shutil.copy(f"{a.src}/{fn}", f"{a.out}/{fn}")

    def dist(sel, key):
        c = collections.Counter(info[i][key] for i in sel)
        return dict(sorted(c.items()))
    all_tools = set(by_tool)
    got_tools = {n for i in chosen for n in info[i]["canon"]}
    src_names = {n for v in info.values() for n in v["names"]}
    got_names = {n for i in chosen for n in info[i]["names"]}
    stats = {"n_items": len(chosen), "n_examples": n_ex, "n_examples_coverage_phase": n_cov,
             "src_items": len(info), "src_examples": sum(v["n"] for v in info.values()),
             "tools_covered": f"{len(got_tools)}/{len(all_tools)}",
             "min_tool_count": min(collections.Counter(n for i in chosen for n in info[i]["canon"]).values()),
             "names_incl_alias_covered": f"{len(got_names)}/{len(src_names)}",
             "kind": dist(chosen, "kind"), "kind_src": dist(info, "kind"),
             "end": dist(chosen, "end"), "end_src": dist(info, "end"),
             "test_app_share": round(sum(info[i]["test_app"] for i in chosen) / len(chosen), 3),
             "test_app_share_src": round(sum(v["test_app"] for v in info.values()) / len(info), 3)}
    meta = json.load(open(f"{a.src}/train.jsonl.meta.json", encoding="utf-8"))
    meta.update({"subset_of": a.src, "subset_args": {k: v for k, v in vars(a).items() if k != "data"},
                 "n_items": len(chosen), "n_examples": n_ex, "subset_stats": stats})
    json.dump(meta, open(f"{a.out}/train.jsonl.meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
