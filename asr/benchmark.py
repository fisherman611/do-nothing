"""Dataset manifests, accounting and result persistence, without GPU imports."""

import hashlib
import json
from pathlib import Path
import random
import threading
import time
import wave

from metrics.asr import evaluate_asr
from metrics.io import checked_path, read_jsonl, write_json, write_jsonl
from asr.registry import WORKER_RATE_USD_SECOND


def digest(path):
    return hashlib.sha256(checked_path(path).read_bytes()).hexdigest()


def prepare_dataset(train_dir):
    root = checked_path(train_dir).resolve()
    rows = read_jsonl(root / "train.jsonl")
    if not rows:
        raise ValueError("No train samples")
    seen, manifest = set(), []
    for row in rows:
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError("Duplicate or invalid train ID")
        seen.add(sample_id)
        if not isinstance(row.get("audio_text"), str):
            raise ValueError("Expected labeled train audio_text")
        path = checked_path(root / row["audio"]).resolve()
        if not path.is_relative_to(root / "audio") or path.suffix.lower() != ".wav":
            raise ValueError("Audio must stay under train/audio")
        with wave.open(str(path), "rb") as audio:
            if audio.getframerate() != 16000 or audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                raise ValueError("Expected 16-kHz mono PCM16 WAV")
            duration = audio.getnframes() / 16000
        if not 0 < duration <= 30:
            raise ValueError("Expected short audio of 0 to 30 seconds")
        manifest.append({"id": sample_id, "audio": row["audio"], "audio_seconds": duration, "sha256": digest(path)})
    return root, rows, manifest


def select_smoke(manifest, size=100):
    if type(size) is not int or size <= 0:
        raise ValueError("Smoke size must be positive")
    ordered = sorted(manifest, key=lambda r: r["id"])
    if len({r["id"] for r in ordered}) != len(ordered):
        raise ValueError("Duplicate manifest ID")
    if len(ordered) <= size:
        return ordered
    longest = sorted(ordered, key=lambda r: (-r["audio_seconds"], r["id"]))[:min(20, size)]
    selected = {r["id"] for r in longest}
    rest = [r for r in ordered if r["id"] not in selected]
    return sorted(longest + random.Random(42).sample(rest, size - len(longest)), key=lambda r: r["id"])


class Budget:
    """Conservative wall-clock estimate including RPC/queue time, not a billing hard cap."""
    def __init__(self, limit, clock=time.monotonic, reserve=1.0):
        if not isinstance(limit, (int, float)) or not 0 < limit <= 10:
            raise ValueError("Suite budget must be in (0, 10] USD")
        if not 0 <= reserve < limit:
            raise ValueError("Reserve must be nonnegative and below budget")
        self.limit, self.clock, self.reserve = float(limit), clock, reserve
        self.active, self.spent = {}, 0.0
        self.lock = threading.Lock()

    def start(self, key):
        with self.lock:
            if key in self.active:
                raise ValueError("Budget worker already active")
            self.active[key] = self.clock()

    def stop(self, key):
        with self.lock:
            self.spent += (self.clock() - self.active.pop(key) + 65) * WORKER_RATE_USD_SECOND

    def estimate(self):
        with self.lock:
            now = self.clock()
            return self.spent + sum(now - start + 65 for start in self.active.values()) * WORKER_RATE_USD_SECOND

    def available(self):
        # Reserve 1 USD for image build, idle tail, and accounting uncertainty.
        return max(0.0, self.limit - self.reserve - self.estimate())


def save_report(folder, predictions, references):
    report = evaluate_asr(predictions, references, require_all=True)
    write_json(folder / "metrics.json", report)
    write_json(folder / "worst_samples.json", sorted(report["samples"], key=lambda r: r["normalized"]["wer"], reverse=True)[:25])
    return report["summary"]


def run_phase(worker, slug, phase, manifest, rows, root, output_root, budget, resume_from=None):
    from datetime import datetime, timezone
    folder = output_root / slug / phase
    folder.mkdir(parents=True, exist_ok=False)
    row_by_id = {r["id"]: r for r in rows}
    refs = [{"id": m["id"], "audio_text": row_by_id[m["id"]]["audio_text"]} for m in manifest]
    write_jsonl(folder / "references.jsonl", refs)
    config = {"model_slug": slug, "phase": phase, "num_samples": len(manifest), "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "starting"}
    write_json(folder / "run_config.json", config)
    key, predictions, started = f"{slug}/{phase}", [], time.monotonic()
    if resume_from:
        source = checked_path(resume_from)
        previous = read_jsonl(source)
        expected = {m['id'] for m in manifest}
        if len({p['id'] for p in previous}) != len(previous) or any(p['id'] not in expected for p in previous):
            raise ValueError("Resume predictions have duplicate or unexpected IDs")
        predictions = [p for p in previous if p.get('status') == 'ok' and isinstance(p.get('text'), str)
                       and isinstance(p.get('elapsed_seconds'), (int, float)) and 0 <= p['elapsed_seconds'] <= 30]
        config.update(resumed_from=str(source), resume_source_sha256=digest(source),
                      num_reused_predictions=len(predictions),
                      resume_note="Original measured per-sample latencies retained. Unrecorded calls from interrupted transport may execute again; stateless ASR only.")
    completed_ids = {p['id'] for p in predictions}
    pending = [m for m in manifest if m['id'] not in completed_ids]
    budget.start(key)
    try:
        metadata = worker.describe.remote()
        config.update(metadata)
        config["status"] = "running"
        write_json(folder / "run_config.json", config)
        print(json.dumps({"event": "model_ready", "model": slug, "phase": phase, **metadata}), flush=True)
        with (folder / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as output:
            for previous in predictions:
                output.write(json.dumps(previous, ensure_ascii=False) + "\n")
            output.flush()
            for offset in range(0, len(pending), 25):
                if budget.available() <= 0:
                    break
                chunk = pending[offset:offset + 25]

                def payloads():
                    for item in chunk:
                        audio_bytes = checked_path(root / item["audio"]).read_bytes()
                        if hashlib.sha256(audio_bytes).hexdigest() != item["sha256"]:
                            raise ValueError("WAV changed after manifest was created")
                        yield {"id": item["id"], "audio_bytes": audio_bytes}

                # Keep inputs queued to avoid cold starts; max_containers=1 and no
                # @concurrent means batch-size-1 inference, not GPU batching.
                for index, result in enumerate(worker.transcribe.map(payloads(), order_outputs=True, return_exceptions=True)):
                    item = chunk[index]
                    if isinstance(result, Exception):
                        error = result
                        result = {"id": item["id"], "text": "", "status": "timeout" if "timeout" in type(error).__name__.lower() else "error", "elapsed_seconds": None, "error": f"{type(error).__name__}: {error}"}
                    if result.get("id") != item["id"] or not isinstance(result.get("text"), str):
                        raise ValueError("Unexpected worker prediction")
                    result["audio_seconds"] = item["audio_seconds"]
                    predictions.append(result)
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output.flush()
                    if len(predictions) % 25 == 0 or len(predictions) == len(manifest):
                        progress = {"model": slug, "phase": phase, "completed": len(predictions), "total": len(manifest), "errors": sum(r["status"] != "ok" for r in predictions), "estimated_spend_usd": round(budget.estimate(), 3)}
                        print(json.dumps(progress), flush=True)
                        with (folder / "run.log").open("a", encoding="utf-8") as log:
                            log.write(json.dumps(progress) + "\n")
        if len(predictions) != len(manifest):
            from metrics.asr import evaluate_asr
            report = evaluate_asr(predictions, refs)
            write_json(folder / "metrics.json", report)
            config["status"] = "incomplete_budget"
            summary = report["summary"]
        else:
            summary = save_report(folder, predictions, refs)
            config["status"] = "completed"
        config["summary"] = summary
        config["hit_token_limit_count"] = sum(bool(r.get("hit_token_limit")) for r in predictions)
        config["wall_seconds"] = time.monotonic() - started
        config["num_predictions"] = len(predictions)
    except Exception as error:
        config.update(status="failed", error=f"{type(error).__name__}: {error}", num_predictions=len(predictions), wall_seconds=time.monotonic() - started)
        print(json.dumps({"event": "phase_failed", "model": slug, "phase": phase, "error": config["error"]}), flush=True)
    finally:
        budget.stop(key)
        config["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        config["estimated_suite_spend_usd"] = budget.estimate()
        write_json(folder / "run_config.json", config)
    return config
