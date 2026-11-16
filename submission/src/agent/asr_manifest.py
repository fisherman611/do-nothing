"""Dựng manifest ASR cho một split: mọi đoạn audio user (lượt cuối + history) kèm ngữ cảnh lượt trước, theo đúng
điều kiện thi (prompt chỉ từ transcript ASR của các lượt user trước + text assistant có sẵn).

    python agent/asr_manifest.py --data data/train+public_test --split public_test --out agent/runs/asr/public_test.manifest.json
    python agent/asr_manifest.py --data data/train+public_test --split train --ids-split val_in,val_ood --out agent/runs/asr/val.manifest.json

Mỗi item: {sid, uid, kind: final|history, audio (đường dẫn tương đối trong split), prev: [[role, text|null, uid|null], ...],
ref (chỉ train: audio_text)}. uid = tên file không đuôi (khớp `--asr` của clef_agent / agent.py).
Chỉ đọc <split>.jsonl; không đọc *_environment/.
"""
import argparse
import json
import os


def items_of(sample):
    out, prev = [], []
    for t in sample["history"]:
        if t["role"] == "user":
            if t.get("audio"):
                uid = os.path.basename(t["audio"])[:-4]
                out.append({"sid": sample["id"], "uid": uid, "kind": "history", "audio": t["audio"],
                            "prev": list(prev), "ref": t.get("audio_text")})
                prev.append(["user", None, uid])          # text điền bằng transcript ASR lúc chạy
            elif t.get("audio_text"):
                prev.append(["user", t["audio_text"], None])
        else:
            if t.get("text"):
                prev.append(["assistant", t["text"], None])
    out.append({"sid": sample["id"], "uid": sample["id"], "kind": "final", "audio": sample["audio"],
                "prev": list(prev), "ref": sample.get("audio_text")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="public_test")
    ap.add_argument("--ids-split", help="train: khoá splits.json, nhiều khoá cách nhau bằng dấu phẩy")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(f"{a.data}/{a.split}/{a.split}.jsonl", encoding="utf-8") if l.strip()]
    if a.ids_split:
        sp = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "clef", "splits.json")))
        keep = {i for k in a.ids_split.split(",") for i in sp[k]}
        rows = [r for r in rows if r["id"] in keep]
    items = [it for r in rows for it in items_of(r)]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"split": a.split, "items": items}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    n_f = sum(it["kind"] == "final" for it in items)
    print(f"{a.split}: {len(rows)} mẫu → {len(items)} đoạn ({n_f} lượt cuối, {len(items) - n_f} history) → {a.out}")


if __name__ == "__main__":
    main()
