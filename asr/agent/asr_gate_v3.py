"""Cổng đo B4 (PLAN_72h_v3): so ASR v3 với e46f trên 405 đoạn held-out (+ tuỳ chọn các bộ TTS val giọng chưa nghe).

Chuẩn hoá chung: thường, bỏ dấu câu, chữ số → chữ (vi_num). Lỗi số đếm theo GIÁ TRỊ (asr_noise_b0._nums), thực thể theo
label_ents của manifest_v2 (thiếu nguyên văn sau khi bỏ dấu = sai), vùng theo vĩ độ người dùng trong context train.

    python agent/asr_gate_v3.py --new <eval.json của B4> [--base agent/runs/asr/heldout.e46f.asr.json]
           [--new-tts <hyps_tts.json {id: text}>] [--base-tts agent/runs/asr/e46f_tts.asr.json]
In bảng và kết luận qua/không qua; ghi agent/runs/asr/gate_v3.json.
"""
import argparse, json, os, re, sys, unicodedata

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/synth/tts_text"); sys.path.insert(0, ROOT)
from vi_num import to_spoken
from agent.clef.asr_noise_b0 import _nums


def norm(s):
    s = to_spoken(unicodedata.normalize("NFC", (s or "").lower()))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def fold(s):
    s = unicodedata.normalize("NFD", (s or "").lower()).replace("đ", "d")
    return " ".join(re.findall(r"[a-z0-9]+", "".join(c for c in s if unicodedata.category(c) != "Mn")))


def ed(r, h):
    r, h = r.split(), h.split(); d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        p, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a != b))
    return d[-1]


def regions():
    out = {}
    for l in open(f"{ROOT}/data/train+public_test/train/train.jsonl", encoding="utf-8"):
        r = json.loads(l); c = r["context"]
        lat = c["location"]["lat"] if isinstance(c, dict) else float(re.search(r"'lat':\s*([\d.]+)", c).group(1))
        out[r["id"]] = "Bắc" if lat >= 17.5 else ("Trung" if lat >= 13.5 else "Nam")
    return out


def score(items, hyps):
    """items: [{uid, ref, kind, region, ents}] → chỉ số."""
    acc = {}
    def add(k, e, w):
        a = acc.setdefault(k, [0, 0]); a[0] += e; a[1] += w
    ok = num_bad = ent_bad = ent_n = 0
    for it in items:
        h = hyps.get(it["uid"], ""); r = norm(it["ref"]); e = ed(r, norm(h)); w = len(r.split())
        for k in ("tất cả", it["kind"], f"vùng {it['region']}"): add(k, e, w)
        ok += e == 0
        num_bad += _nums(it["ref"]) != _nums(h)
        fh = f" {fold(h)} "
        for ent in it.get("ents", []):
            ent_n += 1; ent_bad += f" {fold(ent)} " not in fh
    out = {k: round(a[0] / max(1, a[1]), 4) for k, a in acc.items()}
    out.update({"câu đúng hết": ok, "câu sai giá trị số": num_bad, "thực thể sai": ent_bad, "thực thể": ent_n, "n": len(items)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True); ap.add_argument("--base", default=f"{ROOT}/agent/runs/asr/heldout.e46f.asr.json")
    ap.add_argument("--new-tts", default=""); ap.add_argument("--base-tts", default=f"{ROOT}/agent/runs/asr/e46f_tts.asr.json")
    a = ap.parse_args()
    reg = regions()
    m2 = {x["uid"]: x for x in json.load(open(f"{ROOT}/modal/manifest_v2.json", encoding="utf-8")) if x["split"] == "val"}
    held = json.load(open(f"{ROOT}/agent/runs/asr/asr_heldout.manifest.json"))["items"]
    items = [{"uid": h["uid"], "ref": h["ref"], "kind": h["kind"], "region": reg.get(h["sid"], "?"),
              "ents": sorted({e[2] for e in (m2.get(h["uid"], {}).get("label_ents") or [])
                              if len(e) > 2 and e[2] and f" {fold(e[2])} " in f" {fold(h['ref'])} "})} for h in held]   # chỉ thực thể có trong câu nói
    new = json.load(open(a.new, encoding="utf-8"))
    new_h = {u: h for u, _, h in new["pairs"]} if "pairs" in new else new.get("hyps", new)
    base_h = json.load(open(a.base, encoding="utf-8"))["hyps"]
    S = {"e46f": score(items, base_h), "ASR v3": score(items, new_h)}
    rows = ["tất cả", "final", "history", "vùng Bắc", "vùng Trung", "vùng Nam", "câu đúng hết", "câu sai giá trị số", "thực thể sai"]
    print(f"{'held-out 405':22s} {'e46f':>9s} {'ASR v3':>9s}")
    for k in rows:
        b, n = S["e46f"].get(k), S["ASR v3"].get(k)
        f = (lambda v: f"{v:9.2%}") if isinstance(b, float) else (lambda v: f"{v:9d}")
        print(f"{k:22s} {f(b)} {f(n)}")
    tts = {}
    if a.new_tts and os.path.exists(a.base_tts):
        nt = json.load(open(a.new_tts, encoding="utf-8")); bt = json.load(open(a.base_tts, encoding="utf-8"))["hyps"]
        man = json.load(open(f"{ROOT}/data/tts_v3/tts_manifest_v3.json", encoding="utf-8"))
        for grp, cond in (("TTS v3 val chuẩn (giọng chưa nghe)", lambda x: x["src_set"] == "v3" and x["split"] == "val"),
                          ("phương ngữ val Trung (Vinh)", lambda x: x["src_set"] == "v3el" and x["split"] == "val" and x["region"] == "Trung"),
                          ("phương ngữ val Nam (Cẩm Hồng)", lambda x: x["src_set"] == "v3el" and x["split"] == "val" and x["region"] == "Nam"),
                          ("TTS v2 giọng val", lambda x: x["src_set"] == "v2" and x["split"] == "val_voice")):
            its = [{"uid": f"{x['root']}/{x['audio'][:-4]}", "ref": x["text"], "kind": "tts", "region": x["region"]} for x in man if cond(x)]
            its = [i for i in its if i["uid"] in nt]
            if its: tts[grp] = {"e46f": score(its, bt)["tất cả"], "ASR v3": score(its, nt)["tất cả"], "n": len(its)}
        for g, v in tts.items(): print(f"{g:38s} n={v['n']:4d}  e46f {v['e46f']:.2%}  ASR v3 {v['ASR v3']:.2%}")
    b, n = S["e46f"], S["ASR v3"]
    gate = {"câu cuối không tệ hơn": n["final"] <= b["final"], "lỗi số không nhiều hơn": n["câu sai giá trị số"] <= b["câu sai giá trị số"],
            "thực thể sai ít hơn": n["thực thể sai"] < b["thực thể sai"],
            "WER Trung/Nam giảm": n["vùng Trung"] < b["vùng Trung"] and n["vùng Nam"] < b["vùng Nam"],
            "Bắc không tăng quá 0,3 điểm": n["vùng Bắc"] <= b["vùng Bắc"] + 0.003}
    print("\nCổng:"); [print(f"  {'✅' if v else '✗'} {k}") for k, v in gate.items()]
    print("KẾT LUẬN:", "QUA → chạy public" if all(gate.values()) else "CHƯA QUA → xem từng mục")
    json.dump({"heldout": S, "tts": tts, "gate": gate}, open(f"{ROOT}/agent/runs/asr/gate_v3.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
