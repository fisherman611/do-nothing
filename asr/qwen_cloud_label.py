"""Create evaluation-only ASR references with Qwen's DashScope HTTP ASR API.

python -m asr.qwen_cloud_label --dry-run
python -m asr.qwen_cloud_label --limit 20

Hotwords come from user_profile and the sample's app state (read through zaic_tools.ToolEnv,
never by opening *_environment/ directly); context comes from earlier assistant replies.
Credentials are loaded at runtime, never persisted.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import uuid
import wave

from metrics.io import checked_path, read_jsonl, write_json, write_jsonl


MODEL = "qwen-audio-3.1-asr-flash"
PARAMETERS = {"format": "wav", "sample_rate": "16000", "language_hints": ["vi", "en"]}
HOTWORD_VERSION = 1
# State fields that hold names people say aloud (contacts, songs, places, shops...).
HOTWORD_KEYS = (
    "name", "title", "artist", "members", "from_name", "bank", "seller", "venue", "location",
    "address", "area", "stop", "stops", "origin", "destination", "operator", "airline",
    "provider", "channel", "channels", "room", "label", "symbol", "tags",
)
CONTEXT_TURNS, CONTEXT_CHARS = 3, 400  # API keeps at most 5 messages, 400 chars per turn
DEFAULT_INPUT = Path("data/train+public_test/public_test/public_test.jsonl")
DEFAULT_OUTPUT = Path("data/results/asr/qwen-cloud/public-test-asr-flash")
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
RETRYABLE_CODES = {"throttling", "throttling.ratequota", "toomanyrequests", "internalerror",
                   "internalservererror", "serviceunavailable", "requesttimeout"}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(checked_path(path).read_bytes()).hexdigest()


_TOOL_ENV = None


def load_state(input_path, sample_id):
    """Initial app state of one sample via the official ToolEnv (no tool calls are made)."""
    global _TOOL_ENV
    split_dir = input_path.parent
    if _TOOL_ENV is None:
        sys.path.insert(0, str(split_dir))
        try:
            from zaic_tools import ToolEnv
        except ImportError:
            raise ValueError(f"zaic_tools not found in {split_dir}; use --hotwords profile or none") from None
        _TOOL_ENV = ToolEnv
    state = _TOOL_ENV(split_dir.name, sample_id, root=str(split_dir.parent)).get_state()
    return {k: v for k, v in state.items() if k != "_meta"}


def _phrases(value):
    if isinstance(value, str):
        for part in value.split(","):
            part = " ".join(part.split())
            # Skip codes, dates, e-mails and long sentences; keep spoken names.
            if 2 <= len(part) <= 40 and "@" not in part and not re.fullmatch(r"[\d\s:/+.T-]+", part):
                yield part
    elif isinstance(value, list):
        for x in value:
            yield from _phrases(x)


def extract_hotwords(profile, state, limit):
    found, seen = [], set()

    def add(text):
        for phrase in _phrases(text):
            if phrase.casefold() not in seen:
                seen.add(phrase.casefold())
                found.append(phrase)

    for key in ("name", "home_address", "work_address", "city"):
        add((profile or {}).get(key))
    by_key = {key: [] for key in HOTWORD_KEYS}

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in by_key and isinstance(child, (str, list)) and not (
                        isinstance(child, list) and child and isinstance(child[0], dict)):
                    by_key[key].append(child)
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(state or {})
    for key in HOTWORD_KEYS:  # priority order: people and titles before rooms and tags
        for value in by_key[key]:
            add(value)
    return found[:limit]


def build_context(history, before):
    texts = [t["text"] for t in history[:before] if t.get("role") == "assistant" and t.get("text")]
    return [" ".join(t.split())[:CONTEXT_CHARS] for t in texts[-CONTEXT_TURNS:]]


def prepare_manifest(input_path, scope="all", hotwords="profile+state", context=True,
                     max_hotwords=300, hotword_weight=3):
    """Only follow WAV references in the split JSONL, never scan directories."""
    input_path = checked_path(input_path).resolve()
    root = input_path.parent
    audio_root = checked_path(root / "audio").resolve()
    if not audio_root.is_relative_to(root):
        raise ValueError("Audio directory must stay inside the split directory")
    items, ids, sample_ids = [], set(), set()
    for row in read_jsonl(input_path):
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise ValueError("Invalid or duplicate sample ID")
        sample_ids.add(sample_id)
        history = row.get("history", [])
        vocabulary = {}
        if hotwords != "none":
            profile = (row.get("context") or {}).get("user_profile")
            state = load_state(input_path, sample_id) if hotwords == "profile+state" else {}
            vocabulary = {w: hotword_weight for w in extract_hotwords(profile, state, max_hotwords)}
        refs = [(sample_id, row.get("audio"), None, len(history))]
        if scope == "all":
            refs += [(f"{sample_id}:history:{i}", turn.get("audio"), i, i)
                     for i, turn in enumerate(history) if turn.get("role") == "user"]
        for item_id, relative, history_index, before in refs:
            if item_id in ids or not isinstance(relative, str) or not relative:
                raise ValueError("Invalid audio reference or duplicate utterance ID")
            ids.add(item_id)
            path = checked_path(root / relative).resolve()
            if not path.is_relative_to(audio_root) or path.suffix.lower() != ".wav":
                raise ValueError("Audio must be a WAV under the split/audio directory")
            with wave.open(str(path), "rb") as audio:
                if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getcomptype()) != (16000, 1, 2, "NONE"):
                    raise ValueError("Expected 16-kHz mono PCM16 WAV")
                frames = audio.getnframes()
                if not frames or len(audio.readframes(frames)) != frames * 2:
                    raise ValueError("Empty or truncated WAV")
            if frames / 16000 > 300:
                raise ValueError("Audio longer than the 5-minute API limit")
            items.append({
                "id": item_id, "sample_id": sample_id, "audio": relative,
                "history_index": history_index, "audio_seconds": frames / 16000,
                "audio_sha256": sha256(path), "vocabulary": vocabulary,
                "context": build_context(history, before) if context else [],
            })
    if not items:
        raise ValueError("Input contains no audio")
    return items


def slim(item):
    """Journal/prediction rows omit the bulky per-request hints kept in manifest.jsonl."""
    return {k: v for k, v in item.items() if k not in ("vocabulary", "context")} | {
        "hotword_count": len(item.get("vocabulary") or {}), "context_count": len(item.get("context") or [])}


def load_credentials(env_file):
    try:
        from dotenv import dotenv_values
    except ImportError:
        raise ValueError("Install requirements-asr-cloud.txt first") from None
    # Process environment takes precedence; do not modify or print .env.
    values = {**dotenv_values(checked_path(env_file), encoding="utf-8-sig", interpolate=False), **os.environ}
    key = values.get("DASHSCOPE_API_KEY", "")
    if not key or "\n" in key or "\r" in key:
        raise ValueError("Missing or invalid DASHSCOPE_API_KEY in .env/environment")
    host = values.get("API_HOST")
    if not host:
        host = urlsplit(values.get("DASHSCOPE_URL", "")).hostname
    if not host:
        raise ValueError("Set API_HOST or DASHSCOPE_URL in .env/environment")
    parsed = urlsplit("https://" + host)
    hostname = parsed.hostname or ""
    if (parsed.netloc != hostname or parsed.path or parsed.query or parsed.fragment
            or not (hostname.endswith(".maas.aliyuncs.com")
                    or hostname in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"})):
        raise ValueError("API_HOST must be an Alibaba DashScope hostname without a path")
    return key, f"https://{hostname}/api/v1/services/aigc/multimodal-generation/generation"


def valid_usage(value):
    if not isinstance(value, dict):
        return None
    keys = ("input_tokens", "output_tokens", "total_tokens")
    if any(type(value.get(k)) is not int or value[k] < 0 for k in keys):
        return None
    if value["input_tokens"] + value["output_tokens"] != value["total_tokens"]:
        return None
    return {k: value[k] for k in keys}


def safe_message(value, api_key):
    """Keep useful transport errors without persisting credentials or headers."""
    message = str(value).replace(api_key, "[REDACTED]")
    message = re.sub(r"sk-[A-Za-z0-9_.-]+", "[REDACTED]", message)
    message = re.sub(r"(?i)(?:authorization\s*[:=]\s*)?bearer\s+[^\s,'\"}]+", "[REDACTED]", message)
    return message.replace("\r", " ").replace("\n", " ")[:500]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # never forward the Authorization header to another host


def _post(url, data, headers, timeout):
    opener = urllib.request.build_opener(_NoRedirect)
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with opener.open(request, timeout=timeout) as response:
        return response.status, response.read()


def build_request(path, vocabulary=None, context=None):
    audio = base64.b64encode(checked_path(path).read_bytes()).decode("ascii")
    messages = [{"role": "assistant", "content": [{"type": "text", "text": text}]} for text in context or []]
    messages.append({"role": "user", "content": [
        {"type": "input_audio", "input_audio": {"data": f"data:audio/wav;base64,{audio}"}}]})
    parameters = dict(PARAMETERS)
    if vocabulary:
        parameters["vocabulary"] = vocabulary
    return {"model": MODEL, "input": {"messages": messages}, "parameters": parameters}


def transcribe(path, api_key, endpoint, timeout=120, vocabulary=None, context=None):
    """One synchronous HTTP request per audio, with sanitized diagnostics and no internal retries."""
    started = time.monotonic()
    diagnostics = {"phase": "requesting"}
    result = {"status": "error", "text": "", "usage": None, "usage_final": False,
              "retryable": False, "diagnostics": diagnostics}
    body = json.dumps(build_request(path, vocabulary, context), ensure_ascii=False).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "X-DashScope-SSE": "disable"}
    try:
        status, raw = _post(endpoint, body, headers, timeout)
        diagnostics.update(phase="parsing", http_status=status)
        data = json.loads(raw)
        diagnostics["request_id"] = data.get("request_id")
        if data.get("code"):
            raise _ServerError(status, data)
        output = data.get("output") or {}
        text = output.get("text")
        if not isinstance(text, str):
            sentences = output.get("sentences") or [output.get("sentence") or {}]
            text = " ".join(s.get("text", "") for s in sentences if isinstance(s, dict))
        usage = dict(data.get("usage") or {})
        if "total_tokens" not in usage and all(type(usage.get(k)) is int for k in ("input_tokens", "output_tokens")):
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        tokens = valid_usage(usage)
        result.update(status="ok", text=text.strip(), usage=tokens, usage_final=tokens is not None,
                      audio_duration=usage.get("duration"),
                      sentence=output.get("sentence"), words=output.get("words"))
        diagnostics["phase"] = "finished"
    except urllib.error.HTTPError as error:
        try:
            data = json.loads(error.read() or b"{}")
        except ValueError:
            data = {}
        _record_server_error(result, error.code, data if isinstance(data, dict) else {}, api_key)
    except _ServerError as error:
        _record_server_error(result, error.status, error.data, api_key)
    except Exception as error:
        timed_out = isinstance(error, TimeoutError) or "timed out" in str(error).lower()
        result.update(status="timeout" if timed_out else "error", error=type(error).__name__,
                      retryable=isinstance(error, (urllib.error.URLError, ConnectionError, TimeoutError, OSError)))
        diagnostics["message"] = safe_message(error, api_key)
    result["elapsed_seconds"] = time.monotonic() - started
    return result


class _ServerError(Exception):
    def __init__(self, status, data):
        super().__init__(data.get("code"))
        self.status, self.data = status, data


def _record_server_error(result, status, data, api_key):
    code = re.sub(r"[^A-Za-z0-9_.-]", "_", str(data.get("code") or f"HTTP{status}").replace(api_key, "REDACTED"))[:100]
    result.update(error=f"DashScope {code}",
                  retryable=status in RETRYABLE_STATUS or code.lower() in RETRYABLE_CODES)
    result["diagnostics"].update(http_status=status, request_id=data.get("request_id"),
                                 server_message=safe_message(data.get("message", ""), api_key))


def append_event(path, value):
    with checked_path(path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_journal(path):
    """Fail closed on corruption; do not silently lose billed requests."""
    attempts = {}
    for event in read_jsonl(path) if path.exists() else []:
        attempt = event["attempt_id"]
        if event["event"] == "started":
            if attempt in attempts or type(event.get("reserved_tokens")) is not int or event["reserved_tokens"] <= 0:
                raise ValueError("Invalid started event in attempts journal")
            attempts[attempt] = event
        elif event["event"] == "completed":
            if attempt not in attempts or "result" in attempts[attempt]:
                raise ValueError("Invalid completed event in attempts journal")
            attempts[attempt]["result"] = event["result"]
        else:
            raise ValueError("Invalid event in attempts journal")
    latest, charged, unknown = {}, 0, 0
    for attempt in attempts.values():
        result = attempt.get("result", {"status": "interrupted", "text": ""})
        usage = valid_usage(result.get("usage"))
        known = result.get("usage_final") is True and usage is not None
        charged += usage["total_tokens"] if known else max(attempt["reserved_tokens"], (usage or {}).get("total_tokens", 0))
        unknown += not known
        latest[attempt["item"]["id"]] = {**attempt["item"], **result, "attempt_id": attempt["attempt_id"]}
    return latest, charged, unknown, len(attempts)


def confirmed_tokens(path):
    return sum(
        event["result"]["usage"]["total_tokens"]
        for event in read_jsonl(path) if event.get("event") == "completed"
        and event["result"].get("usage_final") is True and valid_usage(event["result"].get("usage")) is not None
    ) if path.exists() else 0


def export_results(output, manifest):
    latest, charged, unknown, attempts = read_journal(output / "attempts.jsonl")
    rows = [latest[item["id"]] for item in manifest if item["id"] in latest]
    write_jsonl(output / "predictions.jsonl", rows)
    references = [{
        "id": row["id"], "sample_id": row["sample_id"], "audio": row["audio"],
        "history_index": row["history_index"], "audio_text": row["text"],
        "source": MODEL, "reference_type": "pseudo_label", "human_reviewed": False,
    } for row in rows if row["status"] == "ok"]
    write_jsonl(output / "references.jsonl", [r for r in references if r["history_index"] is None])
    write_jsonl(output / "history_references.jsonl", [r for r in references if r["history_index"] is not None])
    confirmed = confirmed_tokens(output / "attempts.jsonl")
    summary = {
        "model": MODEL, "purpose": "evaluation_only", "updated_at": now(),
        "audio_count": len(manifest), "attempt_count": attempts,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "unattempted_count": len(manifest) - len(rows),
        "accounted_tokens": charged, "attempts_with_unknown_usage": unknown,
        "confirmed_tokens": confirmed, "reserved_unknown_tokens": charged - confirmed,
        "accounting_note": "Final API usage where available; otherwise reserve per attempt. Not a billing hard cap.",
    }
    write_json(output / "summary.json", summary)
    return summary


@contextmanager
def output_lock(output):
    lock = output / ".runner.lock"
    try:
        handle = lock.open("x", encoding="ascii")
    except FileExistsError:
        raise ValueError("Output is locked. If no runner is alive, remove .runner.lock and resume.") from None
    try:
        with handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def run(args, recognizer=transcribe):
    if (args.limit is not None and args.limit <= 0) or args.token_budget <= 0 or args.reserve_tokens <= 0:
        raise ValueError("Limit and token settings must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("Timeout must be finite and positive")
    if args.max_retries < 0 or not math.isfinite(args.retry_delay) or args.retry_delay < 0:
        raise ValueError("Retry count and delay must be non-negative")
    if not 1 <= args.workers <= 16 or args.max_hotwords <= 0:
        raise ValueError("workers must be 1-16 and max-hotwords positive")
    if not (1 <= args.hotword_weight <= 5 or args.hotword_weight == 50):
        raise ValueError("hotword-weight must be 1-5 (or 50)")
    input_path = checked_path(args.input).resolve()
    manifest = prepare_manifest(input_path, args.scope, args.hotwords, args.context == "history",
                                args.max_hotwords, args.hotword_weight)
    selected = manifest[:args.limit] if args.limit is not None else manifest
    counts = [len(r["vocabulary"]) for r in selected]
    overview = {"model": MODEL, "scope": args.scope, "audio_count": len(manifest),
                "selected_audio_count": len(selected),
                "selected_audio_hours": round(sum(r["audio_seconds"] for r in selected) / 3600, 4),
                "hotwords_mean": round(sum(counts) / len(counts), 1), "hotwords_max": max(counts),
                "with_context": sum(bool(r["context"]) for r in selected),
                "token_budget": args.token_budget, "workers": args.workers}
    if args.dry_run or args.log_format == "jsonl":
        print(json.dumps(overview, ensure_ascii=True), flush=True)
    else:
        print(f"Qwen Cloud {MODEL} | {len(selected)}/{len(manifest)} audio | scope={args.scope} | "
              f"{overview['selected_audio_hours']:.3f} h | hotwords~{overview['hotwords_mean']} | "
              f"workers={args.workers} | budget={args.token_budget:,} tokens", flush=True)
    if args.dry_run:
        return 0
    if args.reserve_tokens > args.token_budget:
        raise ValueError("reserve-tokens must not exceed token-budget")
    if args.log_format == "tqdm":
        try:
            from tqdm import tqdm
        except ImportError:
            raise ValueError("Install requirements-asr-cloud.txt first (tqdm is required)") from None
    api_key, endpoint = load_credentials(args.env_file)
    output = checked_path(args.output_dir).resolve()
    if output.is_relative_to(input_path.parent):
        raise ValueError("Output must be outside the input split directory")
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 2, "model": MODEL, "parameters": PARAMETERS,
        "input_sha256": sha256(input_path), "scope": args.scope,
        "hotwords": args.hotwords, "hotword_version": HOTWORD_VERSION, "context": args.context,
        "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "endpoint": endpoint, "purpose": "evaluation_only",
    }
    with output_lock(output):
        config_path, journal_path = output / "run_config.json", output / "attempts.jsonl"
        if config_path.exists():
            if json.loads(config_path.read_text(encoding="utf-8")) != config:
                raise ValueError("Existing run has different input/audio/model/settings; choose a new output-dir")
        else:
            if journal_path.exists():
                raise ValueError("Journal exists without run_config.json; choose a new output-dir")
            write_json(config_path, config)
            write_jsonl(output / "manifest.jsonl", manifest)
        latest, charged, unknown, _ = read_journal(journal_path)
        confirmed = confirmed_tokens(journal_path)
        todo = deque((item, 0) for item in selected if not (
            item["id"] in latest and (latest[item["id"]]["status"] == "ok" or not args.retry_errors)))
        progress = None
        if args.log_format == "tqdm":
            progress = tqdm(total=len(selected), initial=len(selected) - len(todo), desc="ASR",
                            unit="audio", dynamic_ncols=True)

        def log(message):
            if progress is not None:
                progress.write(message)
            else:
                print(json.dumps({"event": "message", "message": message}, ensure_ascii=True), flush=True)

        def refresh():
            if progress is not None:
                done = Counter(latest[item["id"]]["status"] for item in selected if item["id"] in latest)
                progress.set_postfix(ok=done["ok"], errors=sum(v for k, v in done.items() if k != "ok"),
                                     tokens=confirmed, reserve=charged - confirmed, unknown=unknown)

        refresh()
        exit_code, stopping = 0, False
        inflight, delayed, sequence = {}, [], 0
        pool = ThreadPoolExecutor(max_workers=args.workers)
        try:
            while todo or delayed or inflight:
                clock = time.monotonic()
                while delayed and delayed[0][0] <= clock and not stopping:
                    _, _, item, retry = heapq.heappop(delayed)
                    todo.appendleft((item, retry))
                while todo and not stopping and len(inflight) < args.workers:
                    if charged + args.reserve_tokens > args.token_budget:
                        log("Stopped before next request: token budget reserve reached.")
                        exit_code, stopping = 2, True
                        break
                    item, retry = todo.popleft()
                    path = checked_path(input_path.parent / item["audio"])
                    if sha256(path) != item["audio_sha256"]:
                        raise ValueError("Audio changed since preflight")
                    attempt_id = uuid.uuid4().hex
                    append_event(journal_path, {
                        "event": "started", "attempt_id": attempt_id, "started_at": now(),
                        "item": slim(item), "reserved_tokens": args.reserve_tokens,
                    })
                    charged += args.reserve_tokens
                    unknown += 1
                    future = pool.submit(recognizer, path, api_key, endpoint, args.timeout,
                                         item["vocabulary"], item["context"])
                    inflight[future] = (item, retry, attempt_id)
                if stopping and not inflight:
                    break
                if not inflight:
                    if delayed:
                        time.sleep(max(0.0, delayed[0][0] - time.monotonic()))
                        continue
                    break
                finished, _ = wait(inflight, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in finished:
                    item, retry, attempt_id = inflight.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:  # recognizer bug: record, never retry
                        result = {"status": "error", "text": "", "usage_final": False, "retryable": False,
                                  "error": type(error).__name__, "diagnostics": {"message": safe_message(error, api_key)}}
                    append_event(journal_path, {
                        "event": "completed", "attempt_id": attempt_id, "completed_at": now(), "result": result,
                    })
                    usage = valid_usage(result.get("usage"))
                    if result.get("usage_final") is True and usage is not None:
                        charged += usage["total_tokens"] - args.reserve_tokens
                        confirmed += usage["total_tokens"]
                        unknown -= 1
                    else:
                        charged += max(0, (usage or {}).get("total_tokens", 0) - args.reserve_tokens)
                    latest[item["id"]] = {**slim(item), **result, "attempt_id": attempt_id}
                    if args.log_format == "jsonl":
                        print(json.dumps({"id": item["id"], "status": result["status"],
                                          "accounted_tokens": charged, "attempts_with_unknown_usage": unknown,
                                          "retry": retry, "error": result.get("error"),
                                          "diagnostics": result.get("diagnostics")}, ensure_ascii=True), flush=True)
                    if result["status"] == "ok":
                        if progress is not None:
                            progress.update(1)
                        if not result.get("usage_final"):
                            log(f"{item['id']}: final usage missing; retained token reserve and continued.")
                    else:
                        details = result.get("diagnostics") or {}
                        log(f"{item['id']}: {result.get('error')} | http={details.get('http_status', '-')} | "
                            f"request_id={details.get('request_id', '-')} | "
                            f"detail={details.get('server_message') or details.get('message', '')}")
                        if result.get("retryable") and retry < args.max_retries and not stopping:
                            delay = args.retry_delay * 2 ** retry
                            log(f"Retry {retry + 1}/{args.max_retries} in {delay:g}s (prior request may have been billed).")
                            heapq.heappush(delayed, (time.monotonic() + delay, sequence, item, retry + 1))
                            sequence += 1
                        else:
                            if progress is not None:
                                progress.update(1)
                            if not stopping:
                                log("Stopped after failed request; finishing in-flight requests. Inspect predictions.jsonl.")
                            exit_code, stopping = exit_code or 1, True
                    refresh()
        except KeyboardInterrupt:
            log("Interrupted. Pending attempts keep their token reserve; rerun to resume.")
            exit_code = 130
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
            if progress is not None:
                progress.close()
            summary = export_results(output, manifest)
            write_json(output / "last_invocation.json", {
                "updated_at": now(), "token_budget": args.token_budget,
                "reserve_tokens": args.reserve_tokens, "limit": args.limit,
                "retry_errors": args.retry_errors, "timeout": args.timeout,
                "max_retries": args.max_retries, "retry_delay": args.retry_delay,
                "workers": args.workers, "log_format": args.log_format,
            })
            if args.log_format == "jsonl":
                print(json.dumps(summary, ensure_ascii=True), flush=True)
            else:
                print(f"Saved: {output}\nStatus: {summary['status_counts']} | "
                      f"confirmed={summary['confirmed_tokens']:,} tokens | "
                      f"reserved_unknown={summary['reserved_unknown_tokens']:,} tokens | "
                      f"unattempted={summary['unattempted_count']}", flush=True)
        if any(latest.get(item["id"], {}).get("status") != "ok" for item in selected):
            return exit_code or 2
        return exit_code


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--env-file", type=Path, default=Path(".env"))
    result.add_argument("--scope", choices=("all", "final"), default="all", help="All user audio, or final requests only")
    result.add_argument("--hotwords", choices=("profile+state", "profile", "none"), default="profile+state",
                        help="Hotword source: user_profile plus app state (via ToolEnv), profile only, or none")
    result.add_argument("--max-hotwords", type=int, default=300, help="Hotwords per sample, profile first")
    result.add_argument("--hotword-weight", type=int, default=3, help="Hotword weight, 1-5 (50 = super hotword)")
    result.add_argument("--context", choices=("history", "none"), default="history",
                        help="Send up to 3 earlier assistant replies as context")
    result.add_argument("--limit", type=int, help="Select first N audio entries; omit for entire manifest")
    result.add_argument("--workers", type=int, default=4, help="Concurrent requests (API limit: 600 requests/minute)")
    result.add_argument("--token-budget", type=int, default=900_000, help="Cumulative accounting limit in this output directory")
    result.add_argument("--reserve-tokens", type=int, default=2_000, help="Headroom required before each request; estimate, not hard cap")
    result.add_argument("--timeout", type=float, default=60, help="Socket timeout per request, in seconds")
    result.add_argument("--retry-errors", action="store_true", help="Explicitly retry prior failures/interrupted requests (may be billed again)")
    result.add_argument("--max-retries", type=int, default=3, help="Additional attempts per audio for transient transport/server errors")
    result.add_argument("--retry-delay", type=float, default=2, help="Exponential retry backoff starts at this many seconds")
    result.add_argument("--log-format", choices=("tqdm", "jsonl"), default="tqdm", help="Progress bar by default; jsonl for machine-readable logs")
    result.add_argument("--dry-run", action="store_true", help="Validate audio and hotwords; no credentials, network or output writes")
    return result


def main():
    try:
        return run(parser().parse_args())
    except (ValueError, OSError, KeyError, TypeError) as error:
        # Configuration/data failures only; network exceptions are sanitized in transcribe.
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
