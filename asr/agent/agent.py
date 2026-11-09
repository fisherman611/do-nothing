"""Zero-shot tool-calling agent loop for LifeMate (runs where the dataset is; the LLM is a vLLM endpoint).

For each sample: build messages (system rules + date table + profile, history with real tool calls/results,
final user text) -> LLM proposes tool call(s) -> env.validate -> env.call -> result back as a tool message -> repeat.
Stops when the model answers without a tool call, after sys.ask_missing_info, or after MAX_STEPS.

Rules respected (AGENT.md): exactly one ToolEnv per sample; only validate/call/get_tool_calls/get_state_hash;
nothing under *_environment/ is opened by this code (zaic_tools loads it). Greedy decoding, fixed seed.
User turns use `audio_text` (train only) or an ASR transcript file (--asr, {uid: text}).

    python agent.py --data <dir containing train/> --split train --model qwen3_4b_2507 \
        --url https://...modal.run --out runs/qwen3_4b_2507.jsonl [--limit 10] [--workers 48]
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.request

MAX_STEPS = 8
MAX_INVALID = 3
CONTEXT_LIMIT = 16384   # luật: context LLM ≤ 16,384 token (prompt + phần sinh)
MAX_LIST = 20           # kết quả tool dạng danh sách dài hơn thì cắt (train: tối đa 14 phần tử, không bị ảnh hưởng)
WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]

# key -> (tokenizer repo used to render the prompt, revision, output format, chat-template kwargs)
# Sailor2's own template has no tool support, so it is rendered with the Qwen3 (Hermes) template; same <|im_start|> tokens.
RENDER = {
    "qwen3_4b_2507": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554", "hermes", {}),
    "qwen35_4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "xml", {"enable_thinking": False}),
    "qwen3_1p7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "hermes", {"enable_thinking": False}),
    "sailor2_3b": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554", "hermes", {}),
    # Qwen3.5-4B stripped to text-only (strip_qwen35.py), same tokenizer/template as the original
    "qwen35_4b_text": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "xml", {"enable_thinking": False}),
    "qwen35_4b_text_think": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "xml", {"enable_thinking": True}),
}
# output budget per step; thinking models need room for the <think> block (prompt max 6.3k + 6k < 16,384)
MAX_TOKENS = {"qwen35_4b_text_think": 6000}
THINK_END = "</think>"

SYSTEM = """Bạn là LifeMate, trợ lý ảo trên điện thoại của người dùng. Nhiệm vụ: thực hiện YÊU CẦU CUỐI CÙNG của người dùng bằng cách gọi các tool được cung cấp. Các lượt trước chỉ là ngữ cảnh, những gì đã làm ở đó KHÔNG làm lại.

Quy tắc:
- Mỗi lần gọi một tool, xem kết quả rồi mới quyết định bước tiếp theo. Khi đã xong, hoặc không cần/không thể gọi tool, trả lời ngắn gọn bằng tiếng Việt và KHÔNG gọi tool nữa.
- Chỉ gọi tool có trong danh sách. Không gọi tool đọc dữ liệu nếu không cần cho yêu cầu.
- ID (contact_id, event_id, bill_id, device_id, ...) phải lấy từ kết quả tool hoặc lịch sử hội thoại, không được tự đặt.
- Chỉ điền tham số bắt buộc và những tham số người dùng đã nói rõ. KHÔNG tự thêm tham số tùy chọn (location, label, repeat, ...) khi người dùng không nhắc tới.
- Một yêu cầu thường gồm nhiều bước (kiểm tra điều kiện, tra cứu ID, thực hiện, nhắn tin). Tiếp tục gọi tool cho tới khi hoàn thành MỌI việc được yêu cầu; chỉ trả lời bằng văn bản khi đã xong hết.
- sys.ask_missing_info CHỈ dùng khi thật sự không thể xác định một tham số bắt buộc (ví dụ tìm danh bạ ra nhiều người cùng tên, người dùng không nói số tiền hay giờ). Nếu thông tin đã có trong yêu cầu (giờ, số điện thoại, số tiền, tên) thì phải điền vào tool, không hỏi lại. Khi dùng, truyền tool_name và danh sách params còn thiếu rồi dừng.
- Không gọi tool nào nếu yêu cầu nằm ngoài khả năng của các app đã cài, nếu người dùng đổi ý hoặc huỷ yêu cầu, hoặc nếu chỉ là chào hỏi, cảm ơn.
- Yêu cầu có điều kiện ("nếu mai mưa thì ..."): kiểm tra điều kiện trước, chỉ làm tiếp khi điều kiện đúng.
- Thời điểm dùng ISO-8601 đầy đủ giây và múi giờ, ví dụ 2026-11-11T05:30:00+07:00; ngày dùng YYYY-MM-DD. Số tiền là số nguyên đồng (hai triệu tám = 2800000).
- Nội dung tin nhắn, ghi chú, tiêu đề: viết ngắn gọn, giữ đủ thông tin người dùng nói.

{time_block}

Người dùng: {name}. Nhà: {home}. Nơi làm việc: {work}. Thành phố hiện tại: {city}. Vị trí hiện tại: lat {lat}, lng {lng}."""


def time_block(ctx):
    now = dt.datetime.fromisoformat(ctx["current_time"])
    lines = [f"Bây giờ là {now.strftime('%H:%M')} {WEEKDAYS[now.weekday()]}, ngày {now.date().isoformat()} "
             f"(múi giờ {ctx.get('timezone', '')}, offset {now.strftime('%z')[:3]}:{now.strftime('%z')[3:]})."]
    names = {0: "hôm nay", 1: "ngày mai", 2: "ngày kia"}
    days = []
    for k in range(0, 8):
        d = now.date() + dt.timedelta(days=k)
        days.append(f"{names.get(k, f'+{k} ngày')}: {WEEKDAYS[d.weekday()]} {d.isoformat()}")
    lines.append("Lịch: " + "; ".join(days) + ".")
    return "\n".join(lines)


def load_catalog(path):
    cat = json.load(open(path, encoding="utf-8"))
    return {t["name"]: t for a in cat["apps"] for t in a["tools"]}


def tool_specs(item, catalog):
    names = [t for v in item["context"]["installed_apps"].values() for t in v["tools"]]
    if "sys.ask_missing_info" not in names:
        names.append("sys.ask_missing_info")
    specs = []
    for n in names:
        t = catalog.get(n)
        if t is None:
            continue
        specs.append({"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                       "parameters": t.get("parameters", {"type": "object", "properties": {}})}})
    return specs


def _call_msg(calls):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": c.get("arguments", {})}}
                           for c in calls]}


def _dump(o):
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"))


def clip_result(o):
    """Cắt danh sách dài trong kết quả tool (giữ MAX_LIST phần tử đầu, ghi số phần tử bị bỏ) để prompt không vượt context."""
    if isinstance(o, dict):
        out = {}
        for k, v in o.items():
            if isinstance(v, list) and len(v) > MAX_LIST:
                out[k] = [clip_result(x) for x in v[:MAX_LIST]]
                out[f"{k}_omitted"] = len(v) - MAX_LIST
            else:
                out[k] = clip_result(v)
        return out
    if isinstance(o, list):
        return [clip_result(x) for x in o]
    return o


def fit_max_tokens(tok, prompt, max_tokens, min_tokens=64):
    """Số token sinh còn được phép sao cho prompt + sinh ≤ CONTEXT_LIMIT; None nếu không còn đủ chỗ."""
    n = len(tok(prompt, add_special_tokens=False).input_ids)
    room = CONTEXT_LIMIT - n
    return None if room < min_tokens else min(max_tokens, room)


def build_messages(item, asr=None):
    ctx = item["context"]
    p = ctx.get("user_profile", {})
    loc = ctx.get("location", {})
    msgs = [{"role": "system", "content": SYSTEM.format(
        time_block=time_block(ctx), name=p.get("name", ""), home=p.get("home_address", ""),
        work=p.get("work_address", ""), city=p.get("city", ""), lat=loc.get("lat", ""), lng=loc.get("lng", ""))}]
    for t in item["history"]:
        if t["role"] == "user":
            uid = os.path.basename(t.get("audio", ""))[:-4]
            text = asr.get(uid) if asr is not None else t.get("audio_text", "")
            msgs.append({"role": "user", "content": text or ""})
        else:
            if t.get("tool_calls"):
                msgs.append(_call_msg(t["tool_calls"]))
                for r in t.get("tool_results", []):
                    msgs.append({"role": "tool", "content": _dump(clip_result(r))})
            msgs.append({"role": "assistant", "content": t.get("text", "")})
    final = asr.get(item["id"]) if asr is not None else item.get("audio_text", "")
    msgs.append({"role": "user", "content": final or ""})
    return msgs


# ---------- parsing ----------
HERMES = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)
XML_FN = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.S)
XML_PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def _coerce(v, schema):
    typ = (schema or {}).get("type")
    if typ == "string":
        return v
    if typ == "boolean" and v.strip().lower() in ("true", "false"):   # chat template Qwen render bool kiểu Python ("True")
        return v.strip().lower() == "true"
    try:
        return json.loads(v)
    except Exception:
        return v


def parse(text, fmt, catalog, think=False):
    calls, bad = [], []
    if think:   # generation prompt ends with "<think>\n": everything before </think> is reasoning, not output
        text = text.split(THINK_END, 1)[1] if THINK_END in text else ""
    if fmt == "hermes":
        for blob in HERMES.findall(text):
            try:
                o = json.loads(blob)
                args = o.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                calls.append({"name": o["name"], "arguments": args})
            except Exception as e:
                bad.append(f"{type(e).__name__}: {blob[:200]}")
    else:
        for name, body in XML_FN.findall(text):
            props = catalog.get(name, {}).get("parameters", {}).get("properties", {})
            args = {k: _coerce(v.strip("\n"), props.get(k)) for k, v in XML_PARAM.findall(body)}
            calls.append({"name": name, "arguments": args})
    return calls, bad


# ---------- LLM ----------
def complete(url, key, prompt, max_tokens=400, retries=4):
    body = json.dumps({"model": "llm", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "seed": 0,
                       "stop": ["<|im_end|>", "<|endoftext|>"]}).encode()
    for k in range(retries):
        try:
            req = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=body,
                                         headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            return d["choices"][0]["text"], d.get("usage", {})
        except Exception as e:
            if k == retries - 1:
                raise
            time.sleep(5 * (k + 1))


def run_one(item, args, tok, catalog, ToolEnv, asr):
    fmt, tkw = RENDER[args.model][2], RENDER[args.model][3]
    think = tkw.get("enable_thinking", False)
    max_tokens = MAX_TOKENS.get(args.model, 400)
    t0 = time.time()
    env = ToolEnv(args.split, item["id"], root=args.data)       # the only env for this sample
    msgs = build_messages(item, asr)
    tools = tool_specs(item, catalog)
    raw, n_invalid, seen, err, ptoks, ctoks = [], 0, {}, None, [], []
    try:
        for step in range(MAX_STEPS):
            prompt = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False, **tkw)
            mt = fit_max_tokens(tok, prompt, max_tokens)
            if mt is None:                                  # hết chỗ trong 16,384 token: dừng lượt
                err = "context_limit"
                break
            out, usage = complete(args.url, args.key, prompt, max_tokens=mt)
            ptoks.append(usage.get("prompt_tokens"))
            ctoks.append(usage.get("completion_tokens"))
            raw.append(out)
            calls, bad = parse(out, fmt, catalog, think)
            if think:   # keep only the answer part in the conversation (the template drops old reasoning anyway)
                out = out.split(THINK_END, 1)[1].lstrip("\n") if THINK_END in out else ""
            if not calls:
                if bad:   # malformed call: tell the model once
                    n_invalid += 1
                    msgs.append({"role": "assistant", "content": out})
                    msgs.append({"role": "user", "content": "Lỗi định dạng tool call: " + bad[0] + ". Hãy gọi lại đúng định dạng JSON."})
                    if n_invalid >= MAX_INVALID:
                        break
                    continue
                break
            msgs.append(_call_msg(calls))
            stop = False
            for c in calls:
                v = env.validate(c["name"], c["arguments"])
                if not v["ok"]:                       # not executed, not recorded in tool_calls
                    n_invalid += 1
                    res = {"ok": False, "error": v["error"]}
                else:
                    sig = _dump(c)
                    seen[sig] = seen.get(sig, 0) + 1
                    res = env.call(c["name"], c["arguments"])
                    if c["name"] == "sys.ask_missing_info" or seen[sig] >= 3:
                        stop = True
                msgs.append({"role": "tool", "content": _dump(clip_result(res))})
            if stop or n_invalid >= MAX_INVALID:
                break
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    return {"id": item["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash(),
            "steps": len(raw), "n_invalid": n_invalid, "raw": raw, "prompt_tokens": ptoks, "completion_tokens": ctoks,
            "latency": round(time.time() - t0, 2), "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="thư mục chứa <split>/")
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", required=True, choices=list(RENDER))
    ap.add_argument("--url", required=True)
    ap.add_argument("--key", default=os.environ.get("LLM_API_KEY", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--asr", help="JSON {uid: transcript}; mặc định dùng audio_text (chỉ train)")
    ap.add_argument("--ids", help="file danh sách id (mỗi dòng một id)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(args.data, args.split))
    from zaic_tools import ToolEnv
    from transformers import AutoTokenizer
    repo, rev = RENDER[args.model][:2]
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    catalog = load_catalog(os.path.join(args.data, args.split, f"{args.split}_tools_catalog.json"))
    items = [json.loads(l) for l in open(os.path.join(args.data, args.split, f"{args.split}.jsonl"), encoding="utf-8") if l.strip()]
    if args.ids:
        keep = {l.strip() for l in open(args.ids) if l.strip()}
        items = [x for x in items if x["id"] in keep]
    if args.limit:
        items = items[: args.limit]
    asr = json.load(open(args.asr, encoding="utf-8")) if args.asr else None

    done = set()
    if os.path.exists(args.out):                       # resume: skip ids already written
        done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8") if l.strip()}
    todo = [x for x in items if x["id"] not in done]
    print(f"{args.model}: {len(todo)} mẫu cần chạy ({len(done)} đã có)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_one, x, args, tok, catalog, ToolEnv, asr) for x in todo]
        for k, fu in enumerate(cf.as_completed(futs), 1):
            f.write(json.dumps(fu.result(), ensure_ascii=False) + "\n")
            f.flush()
            if k % 50 == 0 or k == len(futs):
                print(f"  {k}/{len(futs)}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
