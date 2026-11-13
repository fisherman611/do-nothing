"""Manifest TTS P1 cho asr_train.py (cùng schema manifest.json của audio thật: uid, audio, text, sid, kind, split, prev) + root.
Audio = bản đã augment (data/asr_pub/wav, root 'tts_pub' trên myserver). text = nhãn P1 (không dấu câu).
prev = các lượt trước của cùng hội thoại public: lượt user lấy nhãn P1 của lượt đó, lượt assistant lấy text history.
    python synth/tts_text_pub/make_manifest_pub.py  → data/asr_pub/tts_manifest_pub.json
"""
import collections, json, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PT = os.path.join(ROOT, "data", "train+public_test", "public_test")


def main():
    R = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "synth", "tts_text_pub", "tts_text_pub.jsonl"), encoding="utf-8")}
    lab_by_uid = {r["uid"]: r["label"] for r in R.values() if r["src"] == "orig"}
    items = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(PT, "public_test.jsonl"), encoding="utf-8")}
    L = [json.loads(l) for l in open(os.path.join(ROOT, "data", "tts_pub_raw", "synth_pub_log.jsonl"), encoding="utf-8")]
    L = [x for x in L if x.get("provider") == "v4"]
    wavdir = os.path.join(ROOT, "data", "asr_pub", "wav")

    def prev_of(sid, uid, kind):
        H = items[sid]["history"]; out = []; u = 0
        stop_u = int(uid.rsplit("_u", 1)[1]) if kind == "history" else 10 ** 9
        for t in H:
            if t["role"] == "user":
                u += 1
                if u >= stop_u: break
                uu = f"{sid}_u{u}"; out.append(["user", lab_by_uid.get(uu, ""), uu])
            else:
                out.append(["assistant", t.get("text", "") or "", None])
        return out

    man, miss = [], 0
    for x in L:
        r = R[x["id"]]; fn = os.path.basename(x["wav"])
        if not os.path.exists(os.path.join(wavdir, fn)): miss += 1; continue
        man.append({"uid": f"{x['id']}__{x['voice']}", "audio": fn, "text": r["label"], "sid": r["sid"], "kind": r["kind"],
                    "split": "train", "root": "tts_pub", "prev": prev_of(r["sid"], r["uid"], r["kind"]),
                    "region": r["region"], "voice": x["voice"], "speed": x.get("speed"), "src": r["src"], "spoken": r["spoken"]})
    os.makedirs(os.path.join(ROOT, "data", "asr_pub"), exist_ok=True)
    json.dump(man, open(os.path.join(ROOT, "data", "asr_pub", "tts_manifest_pub.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(len(man), "item | thiếu wav", miss, "| miền", dict(collections.Counter(m["region"] for m in man)),
          "| có prev", sum(bool(m["prev"]) for m in man))


if __name__ == "__main__":
    main()
