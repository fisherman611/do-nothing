"""Create evaluation-only ASR references using Qwen's DashScope WebSocket API.

python -m asr.qwen_cloud_label --dry-run
python -m asr.qwen_cloud_label --limit 20

Credentials are loaded at runtime, never persisted. No app environment access.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from urllib.parse import urlsplit
import uuid
import wave

from metrics.io import checked_path, read_jsonl, write_json, write_jsonl


MODEL = "qwen-audio-3.1-asr-flash-message"
PARAMETERS = {
    "format": "pcm", "sample_rate": 16000, "language_hints": ["vi", "en"],
    "disfluency_removal_enabled": False, "intermediate_result_enabled": False,
}
DEFAULT_INPUT = Path("data/train+public_test/public_test/public_test.jsonl")
DEFAULT_OUTPUT = Path("data/results/asr/qwen-cloud/public-test-references")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(checked_path(path).read_bytes()).hexdigest()


def prepare_manifest(input_path, scope="all"):
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
        refs = [(sample_id, row.get("audio"), None)]
        if scope == "all":
            refs += [
                (f"{sample_id}:history:{i}", turn.get("audio"), i)
                for i, turn in enumerate(row.get("history", []))
                if turn.get("role") == "user"
            ]
        for item_id, relative, history_index in refs:
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
            items.append({
                "id": item_id, "sample_id": sample_id, "audio": relative,
                "history_index": history_index, "audio_seconds": frames / 16000,
                "audio_sha256": sha256(path),
            })
    if not items:
        raise ValueError("Input contains no audio")
    return items


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
    # Derive the WS path from the host, not the OpenAI-compatible HTTP endpoint.
    parsed = urlsplit("wss://" + host)
    hostname = parsed.hostname or ""
    if (parsed.netloc != hostname or parsed.path or parsed.query or parsed.fragment
            or not (hostname.endswith(".maas.aliyuncs.com")
                    or hostname in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"})):
        raise ValueError("API_HOST must be an Alibaba DashScope hostname without a path")
    return key, f"wss://{hostname}/api-ws/v1/inference"


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


def transcribe(path, api_key, endpoint, timeout=120):
    """One task per audio, with sanitized diagnostics and no transport retries."""
    import websocket

    with wave.open(str(path), "rb") as audio:
        pcm = audio.readframes(audio.getnframes())
    task_id = uuid.uuid4().hex
    started = time.monotonic()
    stopped = threading.Event()
    sender_errors, sentences = [], {}
    sock, sender, usage = None, None, None
    diagnostics = {"phase": "connecting", "audio_bytes_sent": 0, "events_received": 0,
                   "last_event": None, "task_started": False, "finish_sent": False}
    result = {"status": "error", "text": "", "usage": None, "usage_final": False,
              "task_id": task_id, "retryable": False, "diagnostics": diagnostics}

    def send_audio():
        try:
            # Pace 100 ms PCM chunks; receive events concurrently to avoid backpressure.
            for offset in range(0, len(pcm), 3200):
                if stopped.is_set():
                    return
                sock.send_binary(pcm[offset:offset + 3200])
                diagnostics["audio_bytes_sent"] += min(3200, len(pcm) - offset)
                if stopped.wait(min(3200, len(pcm) - offset) / 32000):
                    return
            sock.send(json.dumps({
                "header": {"action": "finish-task", "task_id": task_id, "streaming": "duplex"},
                "payload": {"input": {}},
            }))
            diagnostics.update(finish_sent=True, phase="waiting_task_finished")
        except Exception as error:
            sender_errors.append(error)

    try:
        sock = websocket.create_connection(
            endpoint, header={"Authorization": f"Bearer {api_key}"},
            timeout=min(15, timeout), enable_multithread=True,
            redirect_limit=0,
        )
        sock.settimeout(min(1, timeout))
        diagnostics["phase"] = "sending_run_task"
        sock.send(json.dumps({
            "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
            "payload": {"task_group": "audio", "task": "asr", "function": "recognition",
                        "model": MODEL, "parameters": PARAMETERS, "input": {}},
        }))
        diagnostics["phase"] = "waiting_task_started"
        while True:
            if time.monotonic() - started >= timeout:
                raise TimeoutError("Task deadline exceeded")
            if sender_errors:
                raise sender_errors[0]
            try:
                opcode, raw = sock.recv_data(control_frame=True)
            except websocket.WebSocketTimeoutException:
                continue
            if opcode == websocket.ABNF.OPCODE_CLOSE:
                diagnostics["close_code"] = int.from_bytes(raw[:2], "big") if len(raw) >= 2 else None
                diagnostics["close_reason"] = safe_message(raw[2:].decode("utf-8", errors="replace"), api_key)
                raise ConnectionError("Server sent a close frame before task-finished")
            if opcode in (websocket.ABNF.OPCODE_PING, websocket.ABNF.OPCODE_PONG):
                continue
            if not raw:
                raise ConnectionError("WebSocket closed before task-finished")
            event = json.loads(raw)
            header, payload = event.get("header", {}), event.get("payload", {})
            if header.get("task_id") != task_id:
                raise ValueError("Unexpected task ID")
            reported = valid_usage(payload.get("usage"))
            if reported is not None:
                usage = reported
            kind = header.get("event")
            diagnostics["events_received"] += 1
            diagnostics["last_event"] = kind
            if kind == "task-failed":
                code = str(header.get("error_code", "UNKNOWN"))
                # Never persist raw error bodies/headers, which can contain credentials.
                code = re.sub(r"[^A-Za-z0-9_.-]", "_", code.replace(api_key, "REDACTED"))[:100]
                result["error"] = f"DashScope task-failed: {code}"
                diagnostics["server_message"] = safe_message(header.get("error_message", ""), api_key)
                result["retryable"] = code.lower() in {"throttling", "toomanyrequests", "internalerror", "internalservererror", "serviceunavailable"}
                break
            if kind == "task-started":
                if sender is not None:
                    raise ValueError("Duplicate task-started event")
                diagnostics.update(task_started=True, phase="streaming_audio")
                sender = threading.Thread(target=send_audio, daemon=True)
                sender.start()
            elif kind == "result-generated":
                sentence = payload.get("output", {}).get("sentence", {})
                if sentence.get("sentence_end") is True and not sentence.get("heartbeat"):
                    sentence_id = sentence.get("sentence_id")
                    if type(sentence_id) is not int or not isinstance(sentence.get("text"), str):
                        raise ValueError("Invalid final sentence")
                    sentences[sentence_id] = sentence
            elif kind == "task-finished":
                if sender is None:
                    raise ValueError("task-finished before task-started")
                result.update(status="ok", usage_final=reported is not None)
                diagnostics["phase"] = "finished"
                break
    except Exception as error:
        result["status"] = "timeout" if isinstance(error, (TimeoutError, websocket.WebSocketTimeoutException)) else "error"
        result["error"] = type(error).__name__
        status_code = getattr(error, "status_code", None)
        if status_code is not None:
            # Handshake exceptions embed raw response headers and body in str(error).
            diagnostics["http_status"] = status_code
            diagnostics["message"] = f"WebSocket handshake HTTP {status_code}"
        else:
            diagnostics["message"] = safe_message(error, api_key)
        result["retryable"] = (
            isinstance(error, (ConnectionError, TimeoutError, websocket.WebSocketConnectionClosedException,
                               websocket.WebSocketTimeoutException))
            or status_code in {408, 429, 500, 502, 503, 504}
            or isinstance(error, OSError) and getattr(error, "errno", None) in {54, 104, 110, 10053, 10054, 10060}
        ) and diagnostics.get("close_code") not in {1002, 1003, 1007, 1008, 1009, 1010}
    finally:
        stopped.set()
        if sock is not None:
            try:
                sock.close(timeout=1)
            except Exception:
                # Closing a failed transport must not discard its partial usage.
                pass
        if sender is not None:
            sender.join(timeout=2)
    ordered = [sentences[k] for k in sorted(sentences)]
    result.update(
        text=" ".join(s["text"].strip() for s in ordered).strip(),
        sentences=ordered, usage=usage, elapsed_seconds=time.monotonic() - started,
    )
    return result


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
    input_path = checked_path(args.input).resolve()
    manifest = prepare_manifest(input_path, args.scope)
    selected = manifest[:args.limit] if args.limit is not None else manifest
    overview = {"model": MODEL, "scope": args.scope, "audio_count": len(manifest),
                      "selected_audio_count": len(selected),
                      "selected_audio_hours": round(sum(r["audio_seconds"] for r in selected) / 3600, 4),
                      "token_budget": args.token_budget}
    if args.dry_run or args.log_format == "jsonl":
        print(json.dumps(overview, ensure_ascii=True), flush=True)
    else:
        print(f"Qwen Cloud | {len(selected)}/{len(manifest)} audio | scope={args.scope} | "
              f"{overview['selected_audio_hours']:.3f} h | budget={args.token_budget:,} tokens", flush=True)
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
    if recognizer is transcribe:
        try:
            import websocket  # noqa: F401
        except ImportError:
            raise ValueError("Install requirements-asr-cloud.txt first") from None
    output = checked_path(args.output_dir).resolve()
    if output.is_relative_to(input_path.parent):
        raise ValueError("Output must be outside the input split directory")
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1, "model": MODEL, "parameters": PARAMETERS,
        "input_sha256": sha256(input_path), "scope": args.scope,
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
        progress = None
        if args.log_format == "tqdm":
            initial = sum(item["id"] in latest and (latest[item["id"]]["status"] == "ok" or not args.retry_errors)
                          for item in selected)
            progress = tqdm(total=len(selected), initial=initial, desc="ASR", unit="audio", dynamic_ncols=True)

        def log(message):
            if progress is not None:
                progress.write(message)
            else:
                print(json.dumps({"event": "message", "message": message}, ensure_ascii=True), flush=True)

        def refresh(item_id):
            if progress is not None:
                counts = Counter(latest[item["id"]]["status"] for item in selected if item["id"] in latest)
                progress.set_postfix(id=item_id, ok=counts["ok"], errors=sum(v for k, v in counts.items() if k != "ok"),
                                     tokens=confirmed, reserve=charged - confirmed, unknown=unknown)

        refresh("resume")
        exit_code = 0
        try:
            for item in selected:
                previous = latest.get(item["id"])
                if previous and (previous["status"] == "ok" or not args.retry_errors):
                    continue
                path = checked_path(input_path.parent / item["audio"])
                if sha256(path) != item["audio_sha256"]:
                    raise ValueError("Audio changed since preflight")
                result = None
                for retry in range(args.max_retries + 1):
                    if charged + args.reserve_tokens > args.token_budget:
                        log("Stopped before next request: token budget reserve reached.")
                        exit_code = 2
                        break
                    refresh(item["id"])
                    attempt_id = uuid.uuid4().hex
                    append_event(journal_path, {
                        "event": "started", "attempt_id": attempt_id, "started_at": now(),
                        "item": item, "reserved_tokens": args.reserve_tokens,
                    })
                    result = recognizer(path, api_key, endpoint, args.timeout)
                    append_event(journal_path, {
                        "event": "completed", "attempt_id": attempt_id, "completed_at": now(), "result": result,
                    })
                    latest, charged, unknown, _ = read_journal(journal_path)
                    if result.get("usage_final") and valid_usage(result.get("usage")) is not None:
                        confirmed += result["usage"]["total_tokens"]
                    refresh(item["id"])
                    if args.log_format == "jsonl":
                        print(json.dumps({"id": item["id"], "status": result["status"],
                                        "accounted_tokens": charged, "attempts_with_unknown_usage": unknown,
                                        "retry": retry, "error": result.get("error"),
                                        "diagnostics": result.get("diagnostics")}, ensure_ascii=True), flush=True)
                    if result["status"] == "ok":
                        break
                    details = result.get("diagnostics", {})
                    log(f"{item['id']}: {result.get('error')} | phase={details.get('phase', 'unknown')} | "
                        f"sent={details.get('audio_bytes_sent', '?')} bytes | "
                        f"detail={details.get('message', details.get('server_message', ''))}"
                        + (f" | close={details['close_code']} {details.get('close_reason', '')}"
                           if "close_code" in details else ""))
                    if not result.get("retryable") or retry == args.max_retries:
                        break
                    delay = args.retry_delay * 2 ** retry
                    log(f"Retry {retry + 1}/{args.max_retries} in {delay:g}s (prior request may have been billed).")
                    time.sleep(delay)
                if progress is not None and result is not None:
                    progress.update(1)
                if exit_code == 2:
                    break
                if result["status"] != "ok" or not result.get("usage_final"):
                    log("Stopped after failed request or missing final usage; inspect predictions.jsonl.")
                    exit_code = 1
                    break
        except KeyboardInterrupt:
            log("Interrupted. The pending attempt keeps its token reserve; rerun to resume.")
            exit_code = 130
        finally:
            if progress is not None:
                progress.close()
            summary = export_results(output, manifest)
            write_json(output / "last_invocation.json", {
                "updated_at": now(), "token_budget": args.token_budget,
                "reserve_tokens": args.reserve_tokens, "limit": args.limit,
                "retry_errors": args.retry_errors, "timeout": args.timeout,
                "max_retries": args.max_retries, "retry_delay": args.retry_delay,
                "log_format": args.log_format,
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
    result.add_argument("--limit", type=int, help="Select first N audio entries; omit for entire manifest")
    result.add_argument("--token-budget", type=int, default=900_000, help="Cumulative accounting limit in this output directory")
    result.add_argument("--reserve-tokens", type=int, default=10_000, help="Headroom required before each request; estimate, not hard cap")
    result.add_argument("--timeout", type=float, default=120, help="Deadline per audio including streaming, in seconds")
    result.add_argument("--retry-errors", action="store_true", help="Explicitly retry prior failures/interrupted requests (may be billed again)")
    result.add_argument("--max-retries", type=int, default=3, help="Additional attempts per audio for transient transport/server errors")
    result.add_argument("--retry-delay", type=float, default=2, help="Exponential retry backoff starts at this many seconds")
    result.add_argument("--log-format", choices=("tqdm", "jsonl"), default="tqdm", help="Progress bar by default; jsonl for machine-readable logs")
    result.add_argument("--dry-run", action="store_true", help="Validate audio and print counts without credentials, network or output writes")
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
