"""Một bước của vòng agent Clef-token, viết một lần cho cả train và suy luận.

`run_step` dựng chuỗi Q1 → Q2 → tool call và hỏi `oracle` ở mỗi chỗ cần quyết định:
- `oracle.choose(text, labels, kind, info)` → một nhãn trong `labels` (kind: q1 / present / enum / ref / bool)
- `oracle.gen(text, k, schema)` → chuỗi giá trị của tham số tự do k (không gồm "\\n</parameter>")
- `oracle.gen_tail(text, {k: schema})` → (các khối <parameter> tuỳ chọn model tự điền, {k: giá trị}); chỉ khi
  STYLE["q2"] là discrete / none (PLAN_prompt_format_v2.md 1.1)

Train dùng `GoldOracle` (đáp án từ nhãn, ghi lại vị trí cần tính loss); suy luận dùng oracle gọi vLLM (clef_agent.py).
Vì cả hai đi qua cùng một hàm nên chuỗi train và prompt suy luận giống hệt nhau.
"""
from . import fmt as F


class SkipStep(Exception):
    """Đáp án vàng không nằm trong phương án (id ngoài tập ứng viên, enum lạ, tool chưa cài): bỏ ví dụ này."""


def run_step(prefix, tools, catalog, results_seen, labels, oracle, tok, rng=None, hint=None):
    """prefix: prompt chuẩn (kết thúc bằng generation prompt). results_seen: [(tên tool, kết quả)] của history + lượt này.
    rng: xáo thứ tự phương án (chỉ khi train). Trả về dict: stop, name, args, text (toàn bộ, gồm prefix)."""
    short = F.n_tokens(tok, prefix) > F.SHORT_Q1_AT
    opts, _ = F.q1_options(tools)
    perm = F.shuffled(len(opts), rng) if rng else None
    q1, m = F.q1_block(tools, labels, perm, short, hint)
    text = prefix + q1
    if F.STYLE.get("names"):          # model viết tên phương án (structured output trong tập tên)
        name = oracle.choose_name(text, list(m))
        text += " " + name + "\n"
    else:
        lab = oracle.choose(text, list(m), "q1", m)
        text += " " + lab + "\n"
        name = m[lab]
    if name == F.STOP:
        return {"stop": True, "name": None, "args": {}, "text": text}
    schema = catalog.get(name, {"parameters": {"properties": {}}})
    mode = F.STYLE.get("q2", "all")
    disc, free, tail = {}, [], []      # chọn trong Q2 / sinh từng tham số / sinh gộp ở đuôi call (tuỳ chọn)
    for k, s, req, kind in F.q2_plan(schema):
        ask = mode == "all" or (mode == "discrete" and kind != "free")
        if not req:
            if not ask:
                tail.append((k, s))
                continue
            text += F.q_present(name, k, s, schema.get("parameters", {}).get("x-one-of"))
            a = oracle.choose(text, [F.YES, F.NO], "present", (name, k))
            text += " " + a + "\n"
            if a == F.NO:
                continue
        if mode == "none" and kind != "free":
            free.append((k, s))        # chỉ Q1: id / enum / bool bắt buộc cũng sinh như text
            continue
        if kind == "bool":
            text += F.q_bool(name, k, s)
            a = oracle.choose(text, [F.YES, F.NO], "bool", (name, k))
            text += " " + a + "\n"
            disc[k] = a == F.YES
            continue
        if kind == "enum":
            vals = F.enum_options(s)
            p = F.shuffled(len(vals), rng) if rng else list(range(len(vals)))
            vals = [vals[i] for i in p]
            labs = labels[: len(vals)]
            text += F.q_choice(name, k, [F.render_value(v) for v in vals], labs)
            a = oracle.choose(text, labs, "enum", (name, k, dict(zip(labs, vals))))
            text += " " + a + "\n"
            disc[k] = vals[labs.index(a)]
            continue
        if kind == "ref":
            cands = F.ref_candidates(s["x-ref"], results_seen)
            if cands:
                p = F.shuffled(len(cands), rng) if rng else list(range(len(cands)))
                cands = [cands[i] for i in p]
                labs = labels[: len(cands)]
                text += F.q_choice(name, k, F.ref_option_text(cands), labs)
                a = oracle.choose(text, labs, "ref", (name, k, {l: c[0] for l, c in zip(labs, cands)}))
                text += " " + a + "\n"
                disc[k] = cands[labs.index(a)][0]
                continue
        free.append((k, s))            # tự do (hoặc ref không có ứng viên): model sinh
    text += F.call_head(name)
    for k, v in F.canonical_args(disc, schema).items():
        text += F.param_text(k, v)
    args = dict(disc)
    order = list(F.canonical_args({k: None for k, _ in free}, schema))   # cùng thứ tự với chat template
    for k, s in sorted(free, key=lambda x: order.index(x[0])):
        text += f"<parameter={k}>\n"
        v = oracle.gen(text, k, s)
        text += v + F.FREE_STOP + "\n"
        args[k] = F.A._coerce(v, s)
    if tail:                           # model tự quyết điền tham số tuỳ chọn nào, rồi đóng bằng </function>
        t, extra = oracle.gen_tail(text, dict(tail))
        text += t
        args.update(extra)
    text += F.CALL_END
    return {"stop": False, "name": name, "args": F.canonical_args(args, schema), "text": text}


class GoldOracle:
    """Đáp án từ tool call vàng; ghi lại vị trí (ký tự) cần tính loss trong chuỗi."""

    def __init__(self, gold_call):
        self.gold = gold_call          # None = bước dừng
        self.spans = []                # {"type": "choice", "pos", "labels", "gold"} | {"type": "lm", "start", "end"}

    def choose(self, text, labels, kind, info):
        g = self.gold
        if kind == "q1":
            want = F.STOP if g is None else g["name"]
            hit = [l for l, o in info.items() if o == want]
            if not hit:
                raise SkipStep(f"tool {want} không có trong Q1")
            lab = hit[0]
        elif kind == "present":
            lab = F.YES if info[1] in g["arguments"] else F.NO
        elif kind == "bool":
            lab = F.YES if g["arguments"].get(info[1]) is True else F.NO
        else:   # enum / ref
            want = g["arguments"].get(info[1])
            hit = [l for l, v in info[2].items() if v == want]
            if not hit:
                raise SkipStep(f"{kind} {info[1]}={want!r} không có trong phương án")
            lab = hit[0]
        self.spans.append({"type": "choice", "pos": len(text), "labels": list(labels), "gold": lab})
        return lab

    def choose_name(self, text, names):
        want = F.STOP if self.gold is None else self.gold["name"]
        if want not in names:
            raise SkipStep(f"tool {want} không có trong Q1")
        self.spans.append({"type": "lm", "start": len(text), "end": len(text) + 1 + len(want), "q1": True})
        return want

    def gen(self, text, k, s):
        v = F.render_value(self.gold["arguments"][k])
        self.spans.append({"type": "lm", "start": len(text), "end": len(text) + len(v) + len(F.FREE_STOP)})
        return v

    def gen_tail(self, text, params):
        """Đuôi call: các tham số tuỳ chọn có trong nhãn, thứ tự canonical; loss gồm cả '</function>' để học dừng."""
        g = self.gold["arguments"]
        keys = [k for k in F.canonical_args({k: None for k in params}, {"parameters": {"properties": params}}) if k in g]
        t = "".join(F.param_text(k, g[k]) for k in keys)
        self.spans.append({"type": "lm", "start": len(text), "end": len(text) + len(t) + len(F.TAIL_STOP)})
        return t, {k: g[k] for k in keys}
