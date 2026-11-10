"""Định dạng "Clef ở mức token" (PLAN_FC_clef_qwen35.md), dùng chung cho dữ liệu train (build_data.py) và suy luận
(clef_agent.py) để hai bên tạo ra đúng cùng một chuỗi.

Một bước của vòng agent là phần nối thêm sau prompt chuẩn của `agent.py` (chat template Qwen3.5, không thinking):

    <prefix chuẩn: ... <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n>
    [Q1] Bước tiếp theo là gì?          ← chọn tool / sys.ask_missing_info / DỪNG
    A) calendar.list_events – Xem lịch
    ...
    Đáp án: B                            ← 1 token nhãn
    [Q2] calendar.update_event: điền location (...)?
    Đáp án: không                        ← " có" / " không"
    [Q2] calendar.update_event: event_id = ?
    A) evt_01 (chạy bộ, 2026-11-29T08:00:00+07:00)
    Đáp án: A
    <tool_call>\\n<function=calendar.update_event>\\n<parameter=event_id>\\nevt_01\\n</parameter>\\n
    <parameter=start>\\n...model sinh...\\n</parameter>\\n</function>\\n</tool_call>

Khối Q1/Q2 không được giữ trong lịch sử: bước sau chỉ thấy tool call (render bằng chat template) + kết quả.
"""
import json
import os
import random
import re
import string
import sys

import importlib.util

# agent/agent.py (prompt, catalog, parse) nạp theo đường dẫn: tên "agent" trùng với package agent/
_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent.py")
if "lifemate_agent" in sys.modules:
    A = sys.modules["lifemate_agent"]
else:
    _spec = importlib.util.spec_from_file_location("lifemate_agent", _p)
    A = importlib.util.module_from_spec(_spec)
    sys.modules["lifemate_agent"] = A
    _spec.loader.exec_module(A)

STOP = "DỪNG"
ASK = "sys.ask_missing_info"
YES, NO = "có", "không"
ANS = "Đáp án:"
SHORT_Q1_AT = 13000          # prefix dài hơn (token) thì Q1 chỉ ghi tên tool, bỏ mô tả
MAX_DESC = 120               # ký tự mô tả tham số tối đa trong Q2 (70 cắt giữa ví dụ định dạng, PLAN_prompt_format_v2 1.3)
Q1_DESC = {"full": 70, "short": 40, "none": 0}   # ký tự mô tả tool trong Q1 theo STYLE["q1_desc"]
MAX_CANDS = 32               # id ứng viên tối đa cho một tham số
STOP_DESC = "Đã xong hoặc không cần gọi tool nào nữa"
STOP_DESC_V2 = "Đã làm xong MỌI việc được yêu cầu, hoặc yêu cầu không cần / không thể dùng tool"

# Kiểu prompt (zero-shot 2026-11-08, xem agent/runs/clef/RUNLOG.md). Train và suy luận phải dùng cùng một kiểu.
#   v1: bản đầu. p1: câu hỏi "điền tham số" hỏi theo yêu cầu người dùng + ràng buộc "một trong".
#   p2: p1 + Q1 nhắc lại yêu cầu cuối và các tool đã gọi trong lượt. p3: p2 + Q1 cho model viết tên tool
#   (structured output trong tập tên) thay cho nhãn chữ cái.
STYLES = {"v1": {}, "p1": {"present": "v2"}, "p2": {"present": "v2", "recap": True},
          "p3": {"present": "v2", "recap": True, "names": True}}
STYLE = {}
# Tuỳ chọn format thêm (PLAN_prompt_format_v2.md), độc lập với style:
#   q2:      all      = hỏi Q2 cho mọi tham số (tuỳ chọn: "có điền không?" rồi chọn / sinh)
#            discrete = chỉ hỏi Q2 cho ref / enum / bool; tham số tự do tuỳ chọn do model sinh trong phần đuôi call
#            none     = chỉ Q1; tham số bắt buộc sinh từng cái, tham số tuỳ chọn sinh trong phần đuôi
#   q1_desc: full | short | none = độ dài mô tả tool trong Q1 (JSON tools ở đầu prompt đã có mô tả đầy đủ)
Q2_MODES = ("all", "discrete", "none")
Q1_DESC_MODES = ("full", "short", "none")
#   sys:     json    = tools dạng JSON schema qua chat template Qwen (kèm hướng dẫn gọi hàm của template, ~4.3k token/43 tool)
#            compact = tools dạng text gọn trong system prompt, nhóm theo app, giữ đủ mô tả / kiểu / enum / x-ref /
#                      bắt buộc (~2.1k token/43 tool); không truyền `tools` cho template (bỏ ~250 token hướng dẫn)
#   rules:   full | short = quy tắc LifeMate trong system prompt (776 → ~370 token)
SYS_MODES = ("json", "compact")
RULES_MODES = ("full", "short")


def set_style(name, q2="all", q1_desc="full", sys="json", rules="full"):
    assert q2 in Q2_MODES and q1_desc in Q1_DESC_MODES, (q2, q1_desc)
    assert sys in SYS_MODES and rules in RULES_MODES, (sys, rules)
    STYLE.clear()
    STYLE.update(STYLES[name])
    STYLE.update(name=name, q2=q2, q1_desc=q1_desc, sys=sys, rules=rules)


FMT_KEYS = ("style", "q2", "q1_desc", "sys", "rules")
FMT_DEFAULT = {"style": "v1", "q2": "all", "q1_desc": "full", "sys": "json", "rules": "full"}


def fmt_opts():
    """Các khoá format cần giống nhau giữa build data, train và suy luận (ghi vào *.meta.json / config.json)."""
    return {"style": STYLE.get("name", "v1"), **{k: STYLE.get(k, FMT_DEFAULT[k]) for k in FMT_KEYS[1:]}}


def set_style_from(opts):
    """Đặt format từ dict fmt_opts() (config.json của run / *.meta.json); khoá thiếu lấy mặc định cũ."""
    o = {**FMT_DEFAULT, **{k: v for k, v in opts.items() if k in FMT_KEYS}}
    set_style(o["style"], q2=o["q2"], q1_desc=o["q1_desc"], sys=o["sys"], rules=o["rules"])


# ---------- system prompt / tools ----------
_TY = {"string": "str", "integer": "int", "number": "num", "boolean": "bool", "array": "list", "object": "obj"}


def _param_line(k, s, required):
    t = _TY.get(s.get("type"), s.get("type") or "str")
    if s.get("type") == "array":
        t = f"list[{_TY.get((s.get('items') or {}).get('type'), 'str')}]"
    if "enum" in s:
        t = "|".join(map(str, s["enum"]))
    if "x-ref" in s:
        t = f"id←{s['x-ref']}"
    d = re.sub(r"\s+", " ", s.get("description", "") or "").strip()
    return f"{k}{'' if required else '?'}: {t}" + (f" — {d}" if d else "")


def compact_tools(tools):
    """Tools (dạng chat-template) → text gọn, nhóm theo app. Giữ đủ thông tin schema: mô tả tool, từng tham số với kiểu /
    enum / x-ref / bắt buộc (`?` = tuỳ chọn) / mô tả, ràng buộc x-one-of."""
    apps = {}
    for t in tools:
        f = t["function"]
        app, _, _ = f["name"].rpartition(".")
        apps.setdefault(app, []).append(f)
    out = []
    for app, fs in apps.items():
        out.append(f"[{app}]")
        for f in fs:
            P = f.get("parameters") or {}
            props, req = P.get("properties") or {}, set(P.get("required") or [])
            out.append(f"- {f['name'].split('.', 1)[1] if '.' in f['name'] else f['name']}: "
                       f"{re.sub(r'\s+', ' ', f.get('description', '') or '').strip()}")
            out += [f"    · {_param_line(k, s, k in req)}" for k, s in props.items()]
            if P.get("x-one-of"):
                out.append("    · một trong: " + " / ".join(", ".join(g) for g in P["x-one-of"]))
    return "\n".join(out)


def build_messages(item, asr=None):
    """A.build_messages + quy tắc theo STYLE['rules']."""
    msgs = A.build_messages(item, asr, system=A.SYSTEM_SHORT if STYLE.get("rules") == "short" else A.SYSTEM)
    return msgs
ASK_DESC = "Hỏi lại người dùng thông tin bắt buộc còn thiếu rồi dừng"


# ---------- nhãn ----------
def make_labels(tok):
    """Nhãn mà " X" là đúng 1 token (A–Z, a–z, AA–ZZ...). Thứ tự cố định."""
    cands = list(string.ascii_uppercase) + list(string.ascii_lowercase) + \
        [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    return [c for c in cands if len(tok(" " + c, add_special_tokens=False).input_ids) == 1]


def label_token_ids(tok, labels):
    return [tok(" " + c, add_special_tokens=False).input_ids[0] for c in labels]


# ---------- kiểu tham số ----------
def param_kind(schema):
    if "x-ref" in schema:
        return "ref"
    if "enum" in schema:
        return "enum"
    if schema.get("type") == "boolean":
        return "bool"
    return "free"


def _short(s, n=MAX_DESC):
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


# ---------- Q1 ----------
def q1_options(tools):
    """Phương án Q1 theo thứ tự cố định: tool của các app đã cài (thứ tự catalog), sys.ask_missing_info, DỪNG."""
    names = [t["function"]["name"] for t in tools if t["function"]["name"] != ASK]
    desc = {t["function"]["name"]: t["function"].get("description", "") for t in tools}
    desc.setdefault(ASK, ASK_DESC)
    desc[STOP] = STOP_DESC
    return names + [ASK, STOP], desc


def q1_header(hint):
    if not STYLE.get("recap"):
        return ["[Q1] Bước tiếp theo là gì?"]
    req = _short((hint or {}).get("request", ""), 300)
    done = (hint or {}).get("done") or []
    return [f"[Q1] Yêu cầu cuối của người dùng: «{req}»",
            ("Đã gọi trong lượt này: " + ", ".join(done)) if done else "Chưa gọi tool nào trong lượt này.",
            "Bước tiếp theo là gì? Chỉ chọn DỪNG khi đã làm xong mọi việc được yêu cầu, "
            "hoặc yêu cầu không cần / không thể dùng tool."]


def q1_block(tools, labels, perm=None, short=False, hint=None):
    """Trả về (text kết thúc bằng 'Đáp án:', {nhãn: phương án}); kiểu "names": {tên: tên}, không có nhãn."""
    opts, desc = q1_options(tools)
    if STYLE.get("recap"):
        desc[STOP] = STOP_DESC_V2
    if perm is not None:
        opts = [opts[i] for i in perm]
    lines = q1_header(hint)
    n_desc = 0 if short else Q1_DESC[STYLE.get("q1_desc", "full")]
    def opt_line(head, o):
        return f"{head} {o}" if n_desc == 0 else f"{head} {o} – {_short(desc.get(o, ''), n_desc)}"
    if STYLE.get("names"):
        for o in opts:
            lines.append(opt_line("-", o))
        return "\n".join(lines) + "\nĐáp án (viết đúng tên một lựa chọn):", {o: o for o in opts}
    assert len(opts) <= len(labels), f"{len(opts)} phương án > {len(labels)} nhãn"
    for lab, o in zip(labels, opts):
        lines.append(opt_line(f"{lab})", o))
    return "\n".join(lines) + "\n" + ANS, dict(zip(labels, opts))


# ---------- id ứng viên ----------
def _info(d):
    vals = []
    for k, v in d.items():
        if k == "id" or isinstance(v, (dict, list, bool)) or v is None or v == "":
            continue
        vals.append(_short(str(v), 32))
        if len(vals) == 3:
            break
    return ", ".join(vals)


def _singular(collection):
    """contacts → contact, beneficiaries → beneficiary (đủ cho tên collection trong catalog)."""
    if collection.endswith("ies"):
        return collection[:-3] + "y"
    return collection[:-1] if collection.endswith("s") else collection


def _dicts_with_parent(o, parent, out):
    """(dict, khoá cha) cho mọi dict trong kết quả tool; phần tử của list nhận khoá của list làm cha."""
    if isinstance(o, dict):
        out.append((o, parent))
        for k, v in o.items():
            _dicts_with_parent(v, k, out)
    elif isinstance(o, list):
        for v in o:
            _dicts_with_parent(v, parent, out)
    return out


def ref_candidates(xref, results):
    """Id ứng viên cho tham số x-ref "app.collection" từ các cặp (tên tool, kết quả) đã thấy (history + lượt hiện tại).

    Ưu tiên id thuộc đúng collection:
    - giá trị của khoá "<số ít>_id" ở bất kỳ kết quả nào (contact_id trong tin nhắn / cuộc gọi, device_id trong rule…);
    - "id" của object do tool cùng app trả về, nằm dưới khoá "<collection>", "<số ít>" hoặc "item".
    Không có thì lùi về id của tool cùng app, rồi mọi id (như bản cũ). Giữ thứ tự xuất hiện, bỏ trùng, tối đa
    MAX_CANDS (giữ các id gần nhất). Độ phủ id vàng: train 2219/2220, history public_test 299/299 (bản cũ: 2203, 292).
    """
    app, coll = xref.split(".", 1)
    sing = _singular(coll)
    ref_key = f"{sing}_id"
    exact, same, every = [], [], []
    for name, res in results:
        tool_app = name.split(".")[0]
        for d, parent in _dicts_with_parent(res, None, []):
            v = d.get(ref_key)
            if isinstance(v, str):
                exact.append((v, _info({k: x for k, x in d.items() if k != ref_key})))
            if isinstance(d.get("id"), str):
                c = (d["id"], _info(d))
                every.append(c)
                if tool_app == app:
                    same.append(c)
                    if parent in (coll, sing, "item"):
                        exact.append(c)
    seen, out = set(), []
    for i, info in (exact or same or every):
        if i not in seen:
            seen.add(i)
            out.append((i, info))
    return out[-MAX_CANDS:]


# ---------- Q2 ----------
def q2_plan(schema):
    """Danh sách tham số theo thứ tự schema: (tên, schema, bắt buộc?, kiểu)."""
    params = schema.get("parameters", {})
    req = set(params.get("required", []))
    return [(k, s, k in req, param_kind(s)) for k, s in params.get("properties", {}).items()]


def q_present(tool, k, s, oneof=None):
    d = _short(s.get("description", ""))
    if STYLE.get("present") == "v2":
        grp = [g for g in (oneof or []) if k in g]
        o = f"; cần ít nhất một trong: {', '.join(x for g in oneof for x in g)}" if grp else ""
        return (f"[Q2] {tool}: yêu cầu của người dùng có nói tới hoặc cần giá trị cho {k}"
                f"{' (' + d + o + ')' if d or o else ''} không?\n{ANS}")
    return f"[Q2] {tool}: điền tham số {k}{' (' + d + ')' if d else ''}?\n{ANS}"


def q_bool(tool, k, s):
    return f"[Q2] {tool}: {k} = true?\n{ANS}"


def q_choice(tool, k, opts_text, labels):
    lines = [f"[Q2] {tool}: {k} = ?"] + [f"{lab}) {o}" for lab, o in zip(labels, opts_text)]
    return "\n".join(lines) + "\n" + ANS


def enum_options(s):
    return list(s["enum"])


def ref_option_text(cands):
    return [f"{i} ({info})" if info else i for i, info in cands]


# ---------- tool call ----------
def render_value(v):
    """Giống chat template Qwen3.5: chuỗi giữ nguyên, bool/số theo str() của Python, list/dict theo JSON."""
    if isinstance(v, str):
        return v
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def call_head(name):
    return f"<tool_call>\n<function={name}>\n"


def param_text(k, v):
    return f"<parameter={k}>\n{render_value(v)}\n</parameter>\n"


CALL_END = "</function>\n</tool_call>"
FREE_STOP = "\n</parameter>"


def canonical_args(args, schema):
    """Thứ tự tham số khi render: rời rạc (ref/enum/bool) → tự do bắt buộc → tự do tuỳ chọn; trong mỗi nhóm theo
    schema. Khớp thứ tự sinh của run_step (phần đuôi = tham số tuỳ chọn do model tự sinh) ở mọi chế độ q2."""
    plan = {k: (i, kind, req) for i, (k, _, req, kind) in enumerate(q2_plan(schema))}
    none_mode = STYLE.get("q2") == "none"        # chỉ Q1: bắt buộc (sinh từng cái) → tuỳ chọn (đuôi), theo schema
    def key(k):
        i, kind, req = plan.get(k, (10 ** 6, "free", False))
        if none_mode:
            return (not req, i)
        return (kind == "free", kind == "free" and not req, i)
    return {k: args[k] for k in sorted(args, key=key)}


TAIL_STOP = "</function>"    # phần đuôi call do model sinh kết thúc ở đây (nằm trong CALL_END)


def parse_tail(text, params):
    """Tham số trong phần đuôi model sinh: chỉ nhận khoá thuộc `params` {tên: schema}, ép kiểu theo schema."""
    out = {}
    for k, v in A.XML_PARAM.findall(text):
        if k in params and k not in out:
            out[k] = A._coerce(v, params[k])
    return out


# ---------- prompt chuẩn ----------
def render_prefix(tok, msgs, tools):
    if STYLE.get("sys") == "compact":          # tools gọn trong system prompt; không đưa `tools` cho template
        m = [{"role": "system", "content": msgs[0]["content"] + "\n\n# Tools (tên đầy đủ: app.tool)\n" + compact_tools(tools)},
             *msgs[1:]]
        return tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)


def n_tokens(tok, text):
    return len(tok(text, add_special_tokens=False).input_ids)


def append_step(msgs, call, result):
    """Ghi bước đã chạy vào lịch sử (không kèm khối câu hỏi)."""
    msgs.append(A._call_msg([call]))
    msgs.append({"role": "tool", "content": A._dump(A.clip_result(result))})


def shuffled(n, rng):
    p = list(range(n))
    rng.shuffle(p)
    return p
