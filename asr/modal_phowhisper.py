"""Run PhoWhisper-small on train audio with Modal; labels remain local.

modal run --profile vyvy372004 -m asr.modal_phowhisper --limit 20 --output-dir data/results/asr/phowhisper-small/smoke
"""

from pathlib import Path
import io
import json
import time

import modal

MODEL_ID = "vinai/PhoWhisper-small"
MODEL_REVISION = "a86b604c346caf7148c37512eafe783a16420adb"
MODEL_PATH = "/opt/models/phowhisper-small"


def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, local_dir=MODEL_PATH,
        allow_patterns=["*.json", "*.txt", "*.bin", "*.model", "*.safetensors", "LICENSE*"],
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"HF_HUB_DISABLE_XET": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install(
        "transformers==4.57.6", "accelerate==1.12.0", "numpy==2.2.6",
        "soundfile==0.13.1", "librosa==0.11.0", "huggingface-hub==0.36.0", "safetensors==0.8.0",
    )
    .run_function(download_model)
    .env({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_python_source("asr")
)
app = modal.App("lifemate-phowhisper-small-asr")


@app.cls(
    image=image, gpu="T4", cpu=2, memory=8192,
    timeout=30, startup_timeout=600, max_containers=1,
    scaledown_window=60, retries=0, block_network=True, include_source=False,
)
class PhoWhisper:
    @modal.enter()
    def load_model(self):
        import importlib.metadata
        import random
        import numpy as np
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_PATH, local_files_only=True, dtype=torch.float16,
            use_safetensors=False, attn_implementation="sdpa",
        ).to("cuda").eval()
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        features = self.processor(np.zeros(16000, dtype=np.float32), sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        with torch.inference_mode():
            self.model.generate(
                features.input_features.to("cuda", dtype=torch.float16),
                attention_mask=features.attention_mask.to("cuda"),
                language="vi", task="transcribe", do_sample=False, num_beams=1, max_new_tokens=1,
            )
        torch.cuda.synchronize()
        self.metadata = {
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "gpu": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda,
            "model_load_seconds": load_seconds, "warmup_seconds": time.perf_counter() - started,
            "versions": {name: importlib.metadata.version(name) for name in (
                "torch", "transformers", "accelerate", "numpy", "soundfile", "huggingface-hub",
            )},
            "decoding": {"language": "vi", "task": "transcribe", "do_sample": False, "num_beams": 1, "max_new_tokens": 256},
            "seed": 42, "dtype": "float16", "batch_size": 1,
            "network_blocked": True, "timing_scope": "remote audio decode + preprocessing + generation + text decode; excludes model load/warmup/RPC",
        }

    @modal.method()
    def describe(self):
        return self.metadata

    @modal.method()
    def transcribe(self, payload: dict):
        import soundfile as sf
        import torch

        started = time.perf_counter()
        result = {"id": payload["id"], "text": "", "status": "error", "error": None}
        try:
            audio, rate = sf.read(io.BytesIO(payload["audio_bytes"]), dtype="float32")
            if rate != 16000 or audio.ndim != 1:
                raise ValueError("Expected mono WAV at 16 kHz")
            result["audio_seconds"] = len(audio) / rate
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            features = self.processor(audio, sampling_rate=rate, return_tensors="pt", return_attention_mask=True)
            with torch.inference_mode():
                tokens = self.model.generate(
                    features.input_features.to("cuda", dtype=torch.float16),
                    attention_mask=features.attention_mask.to("cuda"),
                    language="vi", task="transcribe", do_sample=False, num_beams=1,
                    max_new_tokens=256, return_timestamps=False,
                )
            result["text"] = self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            torch.cuda.synchronize()
            result["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2
            result["status"] = "ok"
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        result["elapsed_seconds"] = time.perf_counter() - started
        if result["elapsed_seconds"] > 30:
            result["status"] = "timeout"
        return result


@app.local_entrypoint()
def main(
    limit: int = 0,
    output_dir: str = "data/results/asr/phowhisper-small/train-full",
    train_dir: str = "data/train+public_test/train",
):
    from datetime import datetime, timezone
    import hashlib
    import wave
    from metrics.asr import evaluate_asr
    from metrics.io import checked_path, read_jsonl, write_json, write_jsonl

    if limit < 0:
        raise ValueError("limit must be non-negative; 0 means all train samples")
    root = checked_path(train_dir).resolve()
    rows = read_jsonl(root / "train.jsonl")
    if limit:
        rows = rows[:limit]
    if not rows or any(not isinstance(row.get("audio_text"), str) for row in rows):
        raise ValueError("Expected labeled train samples")
    seen, paths = set(), []
    for row in rows:
        if row["id"] in seen:
            raise ValueError("Duplicate train id")
        seen.add(row["id"])
        path = checked_path(root / row["audio"]).resolve()
        if not path.is_relative_to(root / "audio") or path.suffix.lower() != ".wav":
            raise ValueError("Audio path must stay under train/audio")
        with wave.open(str(path), "rb") as audio:
            if audio.getframerate() != 16000 or audio.getnchannels() != 1:
                raise ValueError(f"Unexpected audio format for {row['id']}")
        paths.append(path)
    folder = checked_path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "predictions.jsonl").exists():
        raise ValueError("Output already exists; choose a new output_dir for an independent run")
    write_jsonl(folder / "references.jsonl", [{"id": r["id"], "audio_text": r["audio_text"]} for r in rows])
    config = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_ids": [row["id"] for row in rows], "num_samples": len(rows),
        "train_jsonl_sha256": hashlib.sha256((root / "train.jsonl").read_bytes()).hexdigest(),
        "dataset_scope": "labeled final-turn train audio only; no training or tuning",
        "out_of_sample_estimate": False,
    }
    write_json(folder / "run_config.json", config)
    worker = PhoWhisper()
    metadata = worker.describe.remote()
    config.update(metadata)
    write_json(folder / "run_config.json", config)
    print(json.dumps({"event": "model_ready", "samples": len(rows), **metadata}, ensure_ascii=True), flush=True)
    predictions = []

    def payloads():
        for row, path in zip(rows, paths):
            # Never send reference text, labels, history or app state to GPU.
            yield {"id": row["id"], "audio_bytes": path.read_bytes()}

    with (folder / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as output, (folder / "run.log").open("w", encoding="utf-8") as log:
        for index, result in enumerate(worker.transcribe.map(payloads(), order_outputs=True, return_exceptions=True), 1):
            row = rows[index - 1]
            if isinstance(result, Exception):
                error_type = type(result).__name__
                result = {
                    "id": row["id"], "text": "", "status": "timeout" if "timeout" in error_type.lower() else "error",
                    "elapsed_seconds": None, "error": f"{error_type}: {result}",
                }
            if result["id"] != row["id"]:
                raise ValueError("Modal returned an unexpected sample id")
            predictions.append(result)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            if index % 25 == 0 or index == len(rows):
                progress = {"completed": index, "total": len(rows), "errors": sum(p["status"] != "ok" for p in predictions)}
                line = json.dumps(progress)
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
    report = evaluate_asr(predictions, rows, require_all=True)
    write_json(folder / "metrics.json", report)
    worst = sorted(report["samples"], key=lambda r: r["normalized"]["wer"], reverse=True)[:25]
    write_json(folder / "worst_samples.json", worst)
    config["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    config["num_predictions"] = len(predictions)
    write_json(folder / "run_config.json", config)
    print(json.dumps({"event": "completed", "output_dir": str(folder), "summary": report["summary"]}, ensure_ascii=True), flush=True)
