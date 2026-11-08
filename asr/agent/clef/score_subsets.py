"""Chấm một file kết quả agent (agent.py / clef_agent.py) bằng metrics/ trên các tập trong splits.json.

    python agent/clef/score_subsets.py --gold <gold_train.jsonl> --run agent/runs/full/qwen35_4b_text.jsonl \\
        [--sets val_in val_ood]
gold: `python -m metrics prepare-gold --data-root data/train+public_test --output gold_train.jsonl`
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def score(gold, run, ids, catalog):
    with tempfile.TemporaryDirectory() as d:
        g, s, o = (os.path.join(d, x) for x in ("g.jsonl", "s.jsonl", "o.json"))
        for src, dst in ((gold, g), (run, s)):
            with open(dst, "w") as f:
                for l in open(src):
                    if l.strip() and json.loads(l)["id"] in ids:
                        f.write(l)
        subprocess.run([sys.executable, "-m", "metrics", "score", "--gold", g, "--submission", s,
                        "--catalog", catalog, "--output", o], cwd=ROOT, check=True, capture_output=True)
        return json.load(open(o))["summary"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--run", required=True, nargs="+")
    ap.add_argument("--sets", nargs="+", default=["val_in", "val_ood"])
    ap.add_argument("--catalog", default=os.path.join(ROOT, "data/train+public_test/train/train_tools_catalog.json"))
    a = ap.parse_args()
    sp = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits.json")))
    for run in a.run:
        have = {json.loads(l)["id"] for l in open(run) if l.strip()}
        for name in a.sets:
            ids = set(sp[name]) & have
            s = score(a.gold, run, ids, a.catalog)
            print(f"{os.path.basename(run):40} {name:8} n={len(ids):4}  điểm {s['score']:5.1f}  "
                  f"F1 {s['tie_break_score']:5.1f}  state {s['state_match_rate']:.3f}")


if __name__ == "__main__":
    main()
