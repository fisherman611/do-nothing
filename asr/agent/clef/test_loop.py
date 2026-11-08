"""Chạy clef_agent.run_one với ToolEnv thật trên train, thay vLLM bằng oracle đi theo nhãn vàng, rồi chấm bằng
metrics/. Nếu định dạng + parse + lịch sử đúng thì điểm phải gần 100.

    python agent/clef/test_loop.py --data data/train+public_test --limit 200 --out /tmp/x.jsonl
"""
import argparse
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agent import clef_agent as CA               # noqa: E402
from agent.clef import fmt as F                  # noqa: E402
from agent.clef.core import GoldOracle           # noqa: E402


class GoldSeq(GoldOracle):
    def __init__(self, calls):
        super().__init__(None)
        self.calls, self.i, self.log, self.n_req, self.prompt_tokens, self.anomaly = calls, -1, [], 0, 0, 0

    def choose(self, text, labels, kind, info):
        if kind == "q1":
            self.i += 1
            self.gold = self.calls[self.i] if self.i < len(self.calls) else None
        return super().choose(text, labels, kind, info)

    def choose_name(self, text, names):
        self.i += 1
        self.gold = self.calls[self.i] if self.i < len(self.calls) else None
        return super().choose_name(text, names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--style", default="v1")
    a = ap.parse_args()
    F.set_style(a.style)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(*CA.TOKENIZER[:1], revision=CA.TOKENIZER[1])
    labels = F.make_labels(tok)
    sys.path.insert(0, os.path.join(a.data, "train"))
    from zaic_tools import ToolEnv
    catalog = F.A.load_catalog(f"{a.data}/train/train_tools_catalog.json")
    rows = [json.loads(l) for l in open(f"{a.data}/train/train.jsonl", encoding="utf-8")]
    random.Random(1).shuffle(rows)
    rows = rows[: a.limit]
    args = argparse.Namespace(split="train", data=a.data, url="", key="", model_name="llm")
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            CA.VLLMOracle = lambda _a, _t, _l, calls=r["label"]["tool_calls"]: GoldSeq(calls)
            f.write(json.dumps(CA.run_one(r, args, tok, labels, catalog, ToolEnv, None), ensure_ascii=False) + "\n")
    errs = [json.loads(l) for l in open(a.out)]
    print("lỗi:", sum(bool(x["error"]) for x in errs), [x["error"] for x in errs if x["error"]][:3],
          "| invalid:", sum(x["n_invalid"] for x in errs))
    ids = {x["id"] for x in errs}
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gold_all = os.environ.get("GOLD", "")
    if gold_all:
        sub = a.out + ".gold"
        with open(sub, "w") as g:
            for l in open(gold_all):
                if json.loads(l)["id"] in ids:
                    g.write(l)
        cmd = [sys.executable, "-m", "metrics", "score", "--gold", sub, "--submission", a.out, "--catalog",
               f"{a.data}/train/train_tools_catalog.json", "--output", a.out + ".score.json"]
        subprocess.run(cmd, cwd=root, check=True, capture_output=True)
        s = json.load(open(a.out + ".score.json"))["summary"]
        print({k: round(s[k], 4) for k in ("score", "tie_break_score", "state_match_rate")})


if __name__ == "__main__":
    main()
