"""So nhiều transcript ASR trên 405 đoạn giữ riêng: WER (asr_score) + tỉ lệ khớp thực thể (M4, cùng cách analyze_public).
    python agent/asr_compare.py agent/runs/asr/heldout.e46f.asr.json agent/runs/asr/heldout.v2_medium.asr.json ...
"""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analyze_public as AP  # noqa: E402

MAN = os.path.join(HERE, "runs", "asr", "asr_heldout.manifest.json")
data = os.path.join(ROOT, "data", "train+public_test")
tr = {json.loads(l)["id"]: json.loads(l) for l in open(f"{data}/train/train.jsonl", encoding="utf-8")}
held = [tr[s] for s in sorted({x["sid"] for x in json.load(open(MAN))["items"]})]
ctxc = {}
def cc(r, j):
    return ctxc.setdefault((r["id"], j), AP.context_text(r, j))
for f in sys.argv[1:]:
    h = json.load(open(f, encoding="utf-8"))["hyps"]
    res, _ = AP.entity_check(held, lambda r, j, uid: h.get(uid), cc)
    c = res["known"] + res["new"]
    out = subprocess.run([sys.executable, os.path.join(HERE, "asr_score.py"), "--manifest", MAN, "--asr", f],
                         capture_output=True, text=True).stdout.strip().splitlines()
    print(f"== {os.path.basename(f)} | thực thể n={c['n']} khớp {AP.rate(c, 'exact')} bỏ dấu {AP.rate(c, 'noacc')} tok80 {AP.rate(c, 'tok80')}")
    print("   " + "\n   ".join(out[:6]))
