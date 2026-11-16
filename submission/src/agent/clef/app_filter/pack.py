"""Gói bộ lọc app dùng lúc infer / build dữ liệu (E4): model embedding + BM25 + từ khoá + k + luật trong MỘT thư mục.

    pack = AppFilterPack.load("/path/to/pack")          # pack.json + model/ + app_keywords.json (+ newapps_keywords.json)
    order, tier1 = pack.rank(item)                      # order: app đã cài xếp theo điểm hợp nhất; tier1: tập app giữ đầy đủ
    F.compact_tools(tool_specs, tier1=tier1)            # catalog hai tầng (fmt.py)

Tạo gói: `python pack.py build --model <dir model> --catalog <train_tools_catalog.json> [--catalog ...] --out <dir>`.
Catalog dùng cho tài liệu app: lấy từ chính `item` lúc rank nếu truyền `catalog=` vào load; mặc định gói lưu snapshot
catalog (app_docs) để không phụ thuộc file ngoài. Chỉ dùng catalog, không đọc nhãn hay *_environment/.
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import filter as F  # noqa: E402


class AppFilterPack:
    def __init__(self, cfg, filt):
        self.cfg, self.filt = cfg, filt
        self.k_bm25, self.k_emb = cfg["k_bm25"], cfg["k_emb"]

    @classmethod
    def load(cls, d, catalog=None, device="cpu", emb_cache=None):
        cfg = json.load(open(os.path.join(d, "pack.json"), encoding="utf-8"))
        kw = json.load(open(os.path.join(d, "keywords.json"), encoding="utf-8"))
        cat = catalog or json.load(open(os.path.join(d, "catalog.json"), encoding="utf-8"))
        docs, _ = F.app_docs(cat, kw)
        emb = F.Embedder(os.path.join(d, "model"), docs, emb_cache, device=device)
        return cls(cfg, F.AppFilter(cat, kw, emb, k_bm25=cfg["k_bm25"], k_emb=cfg["k_emb"], rel=cfg.get("rel", 0.0)))

    def encode(self, items):
        return self.filt.emb.encode([F.AppFilter.query(i) for i in items])

    def rank(self, item, q_emb=None):
        """→ (order, tier1). order: app đã cài, điểm giảm dần (cosine + BM25 chuẩn hoá); tier1: bộ lọc (top-k ∪ luật)."""
        if q_emb is None:
            q_emb = self.encode([item])[0]
        keep, sb, se = self.filt.keep(item, q_emb)
        mb, me = (max(sb.values()) or 1.0) if sb else 1.0, (max(se.values()) if se else 1.0)
        order = sorted(sb, key=lambda a: -(sb[a] / mb + (se.get(a, 0.0) / me if me else 0.0)))
        return order, keep


def app_of(tool_name):
    return tool_name.rpartition(".")[0]


def item_with_asr(item, asr):
    """Bản item cho bộ lọc lúc infer: câu cuối và các lượt user trong history lấy từ transcript ASR (asr = {uid: text})."""
    if asr is None:
        return item
    hist = []
    for t in item.get("history", []):
        if t["role"] == "user" and t.get("audio"):
            t = {**t, "audio_text": asr.get(os.path.basename(t["audio"])[:-4], "")}
        hist.append(t)
    return {**item, "history": hist, "audio_text": asr.get(item["id"], "")}


def tier1_for(pack, item, asr=None, gold_apps=(), decoys=0, rng=None, q_emb=None):
    """Tập app tầng 1. Lúc infer: bộ lọc (top-k ∪ luật). Lúc dựng dữ liệu train: ∪ app gold ∪ `decoys` app nhiễu ngẫu nhiên
    (không gold) để model quen có app thừa. App lạ với bộ lọc (alias/iface đổi tên) không có điểm → chỉ vào tầng 1 nhờ gold/decoy."""
    view = item_with_asr(item, asr)
    _, keep = pack.rank(view, q_emb)
    keep = set(keep) | set(gold_apps)
    if decoys and rng is not None:
        inst = [x for x in item["context"]["installed_apps"] if x != "sys" and x not in keep]
        rng.shuffle(inst)
        keep |= set(inst[:decoys])
    return keep


def build(a):
    os.makedirs(a.out, exist_ok=True)
    shutil.copytree(a.model, os.path.join(a.out, "model"), dirs_exist_ok=True)
    apps, seen = [], set()
    for p in a.catalog:
        for x in json.load(open(p, encoding="utf-8"))["apps"]:
            if x["id"] not in seen:
                seen.add(x["id"])
                apps.append(x)
    json.dump({"apps": apps}, open(os.path.join(a.out, "catalog.json"), "w", encoding="utf-8"), ensure_ascii=False)
    kw = {}
    for p in a.keywords:
        kw.update({k: v for k, v in json.load(open(p, encoding="utf-8")).items() if not k.startswith("_")})
    json.dump(kw, open(os.path.join(a.out, "keywords.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump({"k_bm25": a.k_bm25, "k_emb": a.k_emb, "rel": 0.0, "model": "model/", "note": "hợp BM25 top-k ∪ emb top-k ∪ luật history/x-ref"},
              open(os.path.join(a.out, "pack.json"), "w"), indent=1)
    print("→", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--model", required=True)
    b.add_argument("--catalog", nargs="+", required=True)
    b.add_argument("--keywords", nargs="+", default=[os.path.join(HERE, "app_keywords.json")])
    b.add_argument("--k-bm25", type=int, default=4)
    b.add_argument("--k-emb", type=int, default=5)
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a)
