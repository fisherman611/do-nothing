"""Kiểm toán NHÃN của bộ train Clef (PLAN_72h_v3 A1): mọi ví dụ trong <dir>/train.jsonl được đối chiếu với nguồn gold.

Phần 1 — cấu trúc + trung thực với gold (không cần GPU, không cần tokenizer):
  S1  bước của mỗi mẫu liền mạch 0..n, kết thúc đúng (DỪNG hoặc sys.ask_missing_info)
  S2  name/args của ví dụ == tool call gold cùng bước của mẫu nguồn (train.jsonl / synth / phonetic / lượt history)
  S3  khối <tool_call> trong text == name/args của ví dụ
  S4  Q1: chữ cái đáp án trỏ đúng dòng phương án mang tên tool gold (hoặc DỪNG)
  S5  span 'choice': ký tự tại pos đúng là đáp án gold và gold nằm trong labels; span 'lm': đoạn text đúng là giá trị tham số
  S6  Q2 có/không khớp việc tham số có trong args; Q2 chọn id: phương án gold mang đúng giá trị trong args
  S7  mọi tham số kiểu id trong nhãn đều xuất hiện trong phần prompt (kết quả tool / ngữ cảnh) — nhãn có căn cứ
  S8  câu user cuối trong prompt: bản sạch hay bản nhiễu (đếm)
  N1  ví dụ Q1 app lạ (#na): tên tool == gold_tools[0] của synth/newapps, câu trong prompt == câu nguồn,
      tool gold KHÔNG có tham số bắt buộc kiểu id (nếu có thì bước đầu đúng phải là tool đọc lấy id → nhãn sai)

    python3 agent/clef/audit_v3.py --dir agent/clef/data/fmtA_v3 --out agent/runs/audit/fmtA_v3.audit.json
"""
import argparse
import collections
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from agent.clef.build_data import history_items      # noqa: E402
from agent.clef import fmt as F                       # noqa: E402

ASK = "sys.ask_missing_info"
DUP = {}
RE_CALL = re.compile(r"<tool_call>\n<function=([^>\n]+)>\n(.*?)</function>\n</tool_call>$", re.S)
RE_PARAM = re.compile(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>\n", re.S)
RE_USER = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)


def load_sources(data, synth_paths):
    src = {}
    for l in open(f"{data}/train/train.jsonl", encoding="utf-8"):
        r = json.loads(l)
        src[r["id"]] = r
    for p in synth_paths:
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] in src:                       # id trùng (synth iface: 4 id dùng cho 2 biến thể khác nhau)
                DUP.setdefault(r["id"], [src[r["id"]]]).append(r)
            src[r["id"]] = r
    return src


def source_of(eid, src, cache):
    if eid in cache:
        return cache[eid]
    it = src.get(eid)
    if it is None and "#h" in eid:
        parent = eid.rsplit("#h", 1)[0]
        p = src.get(parent)
        if p is not None:
            for h in history_items(p):
                cache[h["id"]] = h
            it = cache.get(eid)
    cache[eid] = it
    return it


def render(v):
    """Giá trị tham số như in trong <parameter> — đúng hàm của pipeline (fmt.render_value: bool/số theo str(), list/dict theo JSON)."""
    return F.render_value(v)


def same_value(a, b):
    if a == b:
        return True
    try:
        return json.dumps(a, ensure_ascii=False, sort_keys=True) == json.dumps(b, ensure_ascii=False, sort_keys=True) or str(a) == str(b)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "train+public_test"))
    ap.add_argument("--synth", action="append", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    synth = a.synth or [os.path.join(ROOT, p) for p in ("synth/train_alias.jsonl", "synth/train_iface.jsonl", "synth_phonetic/train_phonetic.jsonl")]
    src, cache = load_sources(a.data, synth), {}
    NA_ROWS, NA_NEED_ID = {}, set()
    na_dir = os.path.join(ROOT, "synth", "newapps")
    if os.path.exists(os.path.join(na_dir, "newapps_utts.jsonl")):
        for l in open(os.path.join(na_dir, "newapps_utts.jsonl"), encoding="utf-8"):
            r = json.loads(l)
            NA_ROWS[r["id"]] = r
        for x in json.load(open(os.path.join(na_dir, "newapps_catalog.json"), encoding="utf-8"))["apps"]:
            for t in x["tools"]:
                if any(k.endswith("_id") or k == "id" for k in t.get("parameters", {}).get("required", [])):
                    NA_NEED_ID.add(t["name"])
    issues = collections.defaultdict(list)
    cnt = collections.Counter()
    steps = collections.defaultdict(dict)

    def bad(kind, e, msg):
        cnt["issue:" + kind] += 1
        if len(issues[kind]) < 40:
            issues[kind].append({"id": e["id"], "step": e["step"], "msg": msg})

    for line in open(f"{a.dir}/train.jsonl", encoding="utf-8"):
        e = json.loads(line)
        cnt["examples"] += 1
        text, pc = e["text"], e["prefix_chars"]
        tail = text[pc:]
        newapp = "#na" in e["id"]
        cnt["newapps" if newapp else "main"] += 1
        # ---- S3: khối tool_call
        m = RE_CALL.search(tail)
        if e["name"] is None:
            if m:
                bad("S3_stop_has_call", e, m.group(1))
        elif not newapp:
            if not m:
                bad("S3_no_call_block", e, e["name"])
            else:
                if m.group(1) != e["name"]:
                    bad("S3_call_name", e, f"{m.group(1)} != {e['name']}")
                got = dict(RE_PARAM.findall(m.group(2)))
                exp = {k: render(v) for k, v in e["args"].items()}
                if got != exp:
                    bad("S3_call_args", e, f"text {got} != args {exp}")
        # ---- S4/S5: Q1 + span
        ch = [s for s in e["spans"] if s["type"] == "choice"]
        if not ch:
            bad("S5_no_choice_span", e, "")
            continue
        for s in e["spans"]:
            if s["type"] == "choice":
                g = s["gold"]
                if g not in s["labels"]:
                    bad("S5_gold_not_in_labels", e, g)
                if not (text[s["pos"]:].startswith(g) or text[s["pos"]:].startswith(" " + g)):
                    bad("S5_choice_pos", e, repr(text[s["pos"] - 12:s["pos"] + 8]))
                if s["pos"] < pc:
                    bad("S5_span_in_prefix", e, s["pos"])
            else:
                if s["start"] < pc or s["end"] > len(text) or s["end"] <= s["start"]:
                    bad("S5_lm_bounds", e, f"{s['start']}..{s['end']}")
        q1 = ch[0]["gold"]
        mm = re.search(r"^" + re.escape(q1) + r"\) (\S+)", tail, re.M)
        want = e["name"] if e["name"] is not None else "DỪNG"
        if not mm:
            bad("S4_q1_option_missing", e, q1)
        elif mm.group(1) != want:
            bad("S4_q1_wrong_option", e, f"{q1}) {mm.group(1)} != {want}")
        if len(re.findall(r"^\S+\) " + re.escape(want) + r" ", tail, re.M)) != 1 and want != "DỪNG":
            bad("S4_q1_gold_option_count", e, want)
        if newapp:
            r = NA_ROWS.get(e["id"].rsplit("#na", 1)[0])
            if r is None:
                bad("N1_no_source", e, "")
            else:
                if e["name"] != r["gold_tools"][0]:
                    bad("N1_name", e, f"{e['name']} != {r['gold_tools'][0]}")
                us = RE_USER.findall(text[:pc])
                if not us or us[-1].strip() != r["text"].strip():
                    bad("N1_utterance", e, (us[-1][:60] if us else ""))
                if e["name"] in NA_NEED_ID:
                    bad("N1_gold_needs_id", e, e["name"])
            continue
        # ---- S5: lm span == giá trị tham số
        lm_txt = [text[s["start"]:s["end"]] for s in e["spans"] if s["type"] == "lm"]
        lm_txt = [t[:-len(F.FREE_STOP)] if t.endswith(F.FREE_STOP) else t for t in lm_txt]
        vals = [render(v) for v in (e["args"] or {}).values()]
        for t in lm_txt:
            if not any(t == v for v in vals):
                bad("S5_lm_text_not_arg", e, t[:80])
        # ---- S6: Q2
        if e["name"] is not None:
            args = e["args"]
            for k, ans in re.findall(r"^\[Q2\] \S+: điền tham số (\w+)[^\n]*\?\nĐáp án: (có|không)$", tail, re.M):
                if (ans == "có") != (k in args):
                    bad("S6_present_mismatch", e, f"{k}: {ans} nhưng args {list(args)}")
            for k, block, ans in re.findall(r"^\[Q2\] \S+: (\w+) = \?\n((?:\S+\) [^\n]*\n)+)Đáp án: (\S+)$", tail, re.M):
                opt = dict(re.findall(r"^(\S+)\) ([^\n]*)$", block, re.M)).get(ans)
                if k not in args:
                    bad("S6_choice_param_not_in_args", e, k)
                elif opt is None:
                    bad("S6_choice_option_missing", e, f"{k} {ans}")
                else:
                    v = args[k]
                    vs = [render(x) for x in v] if isinstance(v, list) else [render(v)]
                    if not any(opt == x or opt.startswith(x + " ") or opt.startswith(x + ",") for x in vs) and str(v).lower() not in ("true", "false"):
                        bad("S6_choice_value", e, f"{k}: phương án '{opt[:60]}' vs args {v}")
        # ---- S2: so với gold nguồn
        it = source_of(e["id"], src, cache)
        if it is None:
            bad("S2_no_source", e, "")
            continue
        i = int(e["step"])
        if e["id"].split("#h")[0] in DUP:            # chọn biến thể khớp tên tool ở bước này
            par = e["id"].split("#h")[0]
            for cand in DUP[par]:
                c2 = cand if "#h" not in e["id"] else next((h for h in history_items(cand) if h["id"] == e["id"]), None)
                if c2 and (i < len(c2["label"]["tool_calls"]) and c2["label"]["tool_calls"][i]["name"] == e["name"]
                           or i == len(c2["label"]["tool_calls"]) and e["name"] is None):
                    it = c2
                    break
            cnt["dup_id_examples"] += 1
        calls = it["label"]["tool_calls"]
        steps[e["id"]][i] = e["name"]
        if i < len(calls):
            g = calls[i]
            if e["name"] != g["name"]:
                bad("S2_name", e, f"{e['name']} != gold {g['name']}")
            else:
                ga = g["arguments"]
                if set(ga) != set(e["args"]):
                    bad("S2_arg_keys", e, f"{sorted(e['args'])} != gold {sorted(ga)}")
                for k in set(ga) & set(e["args"]):
                    if not same_value(ga[k], e["args"][k]):
                        bad("S2_arg_value", e, f"{k}: {e['args'][k]!r} != gold {ga[k]!r}")
        elif i == len(calls):
            if e["name"] is not None:
                bad("S2_expected_stop", e, e["name"])
        else:
            bad("S2_step_beyond_gold", e, f"{i} > {len(calls)}")
        # ---- S7: id có căn cứ trong prompt
        if e["name"] is not None:
            prefix = text[:pc]
            for k, v in e["args"].items():
                if (k.endswith("_id") or k == "id") and isinstance(v, str) and v not in prefix and v not in tail.split("<tool_call>")[0]:
                    bad("S7_id_not_grounded", e, f"{k}={v}")
        # ---- S8: câu user cuối sạch / nhiễu
        users = [u for u in RE_USER.findall(text[:pc]) if not u.startswith("<tool_response>")]
        last = users[-1].strip() if users else ""
        if i == 0:
            cnt["items"] += 1
            cnt["final_clean" if last == (it.get("audio_text") or "").strip() else "final_noised"] += 1

    # ---- S1: bước liền mạch + kết thúc
    for eid, st in steps.items():
        it = source_of(eid, src, cache)
        calls = it["label"]["tool_calls"]
        n_expect = len(calls) if (calls and calls[-1]["name"] == ASK) else len(calls) + 1
        miss = [i for i in range(n_expect) if i not in st]
        if miss:
            cnt["issue:S1_missing_steps"] += 1
            if len(issues["S1_missing_steps"]) < 40:
                issues["S1_missing_steps"].append({"id": eid, "missing": miss, "expect": n_expect})
    rep = {"counts": dict(cnt), "issues": issues}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(dict(cnt), ensure_ascii=False, indent=1))
    for k, v in issues.items():
        print(f"\n## {k} ({cnt['issue:' + k]})")
        for x in v[:6]:
            print("  ", json.dumps(x, ensure_ascii=False)[:260])


if __name__ == "__main__":
    main()
