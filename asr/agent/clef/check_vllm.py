"""Bước 0 của PLAN_FC_clef_qwen35.md mục 5: kiểm tra server vLLM có đủ tính năng cho Clef-token không.

    python agent/clef/check_vllm.py --url https://...modal.run [--model-name llm] --data data/train+public_test

1. allowed_token_ids + max_tokens=1 trả về đúng một nhãn trong tập cho phép.
2. Tất định: cùng request hai lần → cùng output.
3. Prefix caching với kiến trúc hybrid (Gated DeltaNet): bộ đếm prefix cache trong /metrics tăng ở request thứ hai
   dùng chung prefix dài; so thời gian request đầu và sau.
4. logprob sau khi giới hạn nhãn (chỉ cần nếu dùng ngưỡng xác suất).
5. Danh sách model / LoRA adapter trên server.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agent import clef_agent as CA      # noqa: E402
from agent.clef import fmt as F         # noqa: E402


def get(url, key, path):
    req = urllib.request.Request(url.rstrip("/") + path, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode()


def cache_counters(url, key):
    txt = get(url, key, "/metrics")
    out = {}
    for name in ("prefix_cache_queries", "prefix_cache_hits"):
        vals = [float(m) for m in re.findall(rf"^vllm:{name}(?:_total)?(?:{{[^}}]*}})?\s+([0-9.eE+]+)$", txt, re.M)]
        out[name] = sum(vals) if vals else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--key", default=os.environ.get("LLM_API_KEY", ""))
    ap.add_argument("--model-name", default="llm")
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CA.TOKENIZER[0], revision=CA.TOKENIZER[1])
    labels = F.make_labels(tok)
    ids = F.label_token_ids(tok, labels)

    print("5. models:", json.loads(get(a.url, a.key, "/v1/models"))["data"][0]["id"], "…")

    # prompt thật: mẫu train đầu tiên, khối Q1
    item = json.loads(open(f"{a.data}/train/train.jsonl", encoding="utf-8").readline())
    catalog = F.A.load_catalog(f"{a.data}/train/train_tools_catalog.json")
    tools = F.A.tool_specs(item, catalog)
    prefix = F.render_prefix(tok, F.A.build_messages(item), tools)
    q1, m = F.q1_block(tools, labels)
    prompt = prefix + q1
    allowed = ids[: len(m)]
    print(f"prompt: {F.n_tokens(tok, prompt)} token, {len(m)} phương án")

    c0 = cache_counters(a.url, a.key)
    t = time.time()
    o1, u1 = CA.request(a.url, a.key, a.model_name, prompt, 1, allowed_token_ids=allowed)
    t1 = time.time() - t
    t = time.time()
    o2, _ = CA.request(a.url, a.key, a.model_name, prompt, 1, allowed_token_ids=allowed)
    t2 = time.time() - t
    lab = o1.strip()
    print(f"1. allowed_token_ids: output {o1!r} → {'OK' if lab in m else 'SAI'} ({m.get(lab)})")
    print(f"2. tất định: {'OK' if o1 == o2 else 'KHÁC'} ({o1!r} / {o2!r})")
    # request thứ ba: cùng prefix, đuôi khác (giống Q2 sau Q1)
    t = time.time()
    CA.request(a.url, a.key, a.model_name, prompt + " " + lab + "\n[Q2] thử: điền tham số x?\nĐáp án:", 1,
               allowed_token_ids=[tok(" " + F.YES, add_special_tokens=False).input_ids[0],
                                  tok(" " + F.NO, add_special_tokens=False).input_ids[0]])
    t3 = time.time() - t
    c1 = cache_counters(a.url, a.key)
    if c0["prefix_cache_queries"] is None:
        print("3. prefix cache: /metrics không có bộ đếm prefix_cache_* → kiểm tay log server")
    else:
        q = c1["prefix_cache_queries"] - c0["prefix_cache_queries"]
        h = c1["prefix_cache_hits"] - c0["prefix_cache_hits"]
        print(f"3. prefix cache: {h:.0f}/{q:.0f} token trúng cache ({h / max(q, 1):.0%}) "
              f"| thời gian request 1/2/3: {t1:.2f}s / {t2:.2f}s / {t3:.2f}s")
    body = json.dumps({"model": a.model_name, "prompt": prompt, "max_tokens": 1, "temperature": 0,
                       "allowed_token_ids": allowed, "logprobs": 5}).encode()
    req = urllib.request.Request(a.url.rstrip("/") + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {a.key}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        lp = json.loads(r.read())["choices"][0].get("logprobs") or {}
    top = (lp.get("top_logprobs") or [{}])[0]
    print("4. logprobs top-5:", top, "| tổng xác suất top-5:",
          round(sum(2.718281828 ** v for v in top.values()), 3), "(≈1 nếu logprob tính sau khi giới hạn nhãn)")


if __name__ == "__main__":
    main()
