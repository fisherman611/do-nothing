"""Five ASR baselines on one T4 each; smoke, estimate, full train, repeat."""

from pathlib import Path
import json
import os
import time

import modal
from asr.registry import MODELS, WORKER_RATE_USD_SECOND, download_family

app = modal.App("lifemate-asr-baselines")
base = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"HF_HUB_DISABLE_XET": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install(
        "transformers==4.57.6", "accelerate==1.12.0", "numpy==2.2.6",
        "soundfile==0.13.1", "librosa==0.11.0", "huggingface-hub==0.36.0", "safetensors==0.8.0",
    )
)
offline = {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TOKENIZERS_PARALLELISM": "false",
}
whisper_image = base.run_function(download_family, args=("whisper",)).env(offline).add_local_python_source("asr")
qwen_image = (
    base.apt_install("sox", "libsndfile1")
    .pip_install("torchaudio==2.7.1", "torchvision==0.22.1", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install("qwen-asr==0.0.6")
    .run_function(download_family, args=("qwen",)).env(offline).add_local_python_source("asr")
)
options = dict(gpu="T4", cpu=2, memory=8192, timeout=30, startup_timeout=600,
               max_containers=1, scaledown_window=60, retries=0, block_network=True, include_source=False)


@app.cls(image=whisper_image, **options)
class WhisperWorker:
    model_slug: str = modal.parameter()

    @modal.enter()
    def load(self):
        from asr.runtime import Runtime
        self.engine = Runtime(self.model_slug)

    @modal.method()
    def describe(self):
        return self.engine.metadata

    @modal.method()
    def transcribe(self, payload: dict):
        return self.engine.transcribe(payload)


@app.cls(image=qwen_image, **options)
class QwenWorker:
    model_slug: str = modal.parameter()

    @modal.enter()
    def load(self):
        from asr.runtime import Runtime
        self.engine = Runtime(self.model_slug)

    @modal.method()
    def describe(self):
        return self.engine.metadata

    @modal.method()
    def transcribe(self, payload: dict):
        return self.engine.transcribe(payload)


@app.local_entrypoint()
def main(
    train_dir: str = "data/train+public_test/train",
    output_dir: str = "",
    budget_usd: float = 10.0,
    smoke_only: bool = False,
    models: str = "",
    phase: str = "all",
    smoke_from: str = "",
    resume_from: str = "",
    budget_reserve_usd: float = 1.0,
):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import csv
    from datetime import datetime, timezone
    import importlib.metadata
    from asr.benchmark import Budget, digest, prepare_dataset, run_phase, select_smoke
    from metrics.io import checked_path, write_json, write_jsonl

    budget = Budget(budget_usd, reserve=budget_reserve_usd)
    slugs = models.split(",") if models else list(MODELS)
    if len(slugs) != len(set(slugs)) or not slugs or any(s not in MODELS for s in slugs):
        raise ValueError("Select unique model slugs from asr.registry.MODELS")
    if phase not in ('all', 'full', 'repeat100') or (phase != 'all' and len(slugs) != 1):
        raise ValueError("full/repeat100 mode requires exactly one model per process")
    if os.name == 'nt' and len(slugs) > 1 and not smoke_only:
        raise ValueError('On Windows use --smoke-only, then python -m asr.launch_suite with separate clients per model')
    root, rows, manifest = prepare_dataset(train_dir)
    if len(rows) != 1545:
        raise ValueError("Expected the complete 1545-sample train split")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = checked_path(output_dir or f"data/results/asr/comparison/{stamp}").resolve()
    if output.exists():
        raise ValueError("Choose a new output directory; existing runs are never overwritten")
    output.mkdir(parents=True)
    smoke = select_smoke(manifest)
    write_jsonl(output / "manifest-full.jsonl", manifest)
    write_jsonl(output / "manifest-smoke100.jsonl", smoke)
    write_json(output / "provenance.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "profile": "vyvy372004",
        "dataset_scope": "train final-turn audio; no fine-tuning", "num_samples": len(rows),
        "train_jsonl_sha256": digest(root / "train.jsonl"),
        "manifest_sha256": digest(output / "manifest-full.jsonl"),
        "smoke_sha256": digest(output / "manifest-smoke100.jsonl"),
        "source_sha256": {name: digest(Path("asr") / name) for name in ("registry.py", "runtime.py", "benchmark.py", "modal_suite.py")},
        "metrics_sha256": digest(Path("metrics/asr.py")), "jiwer_version": importlib.metadata.version("jiwer"),
        "models": {s: MODELS[s] for s in slugs}, "budget_usd": budget_usd,
        "worker_rate_usd_second": WORKER_RATE_USD_SECOND,
        "budget_scope": "wall-clock compute estimate, not a provider hard cap", "reserve_usd": budget_reserve_usd,
    })
    workers = {s: (WhisperWorker if MODELS[s]["family"] == "whisper" else QwenWorker)(model_slug=s) for s in slugs}
    results = []

    if phase != 'all':
        from metrics.io import load_json
        slug = slugs[0]
        prior = checked_path(smoke_from).resolve()
        prior_provenance = load_json(prior / 'provenance.json')
        if prior_provenance['train_jsonl_sha256'] != digest(root / 'train.jsonl'):
            raise ValueError('Train data differs from the validated smoke run')
        if prior_provenance['manifest_sha256'] != digest(output / 'manifest-full.jsonl'):
            raise ValueError('WAV data differs from the validated smoke run')
        if prior_provenance['source_sha256']['runtime.py'] != digest(Path('asr/runtime.py')):
            raise ValueError('Inference source differs from the validated smoke run')
        prior_config = load_json(prior / slug / 'smoke100/run_config.json')
        if (prior_config['status'] != 'completed' or prior_config['summary']['num_success'] != 100
                or prior_config.get('hit_token_limit_count')
                or prior_config['model_id'] != MODELS[slug]['model_id']
                or prior_config['revision'] != MODELS[slug]['revision']):
            raise ValueError('Requires a healthy smoke run of the same checkpoint')
        resume_file = None
        if resume_from and phase == 'full':
            source_root = checked_path(resume_from).resolve()
            source_provenance = load_json(source_root / 'provenance.json')
            if (source_provenance['train_jsonl_sha256'] != prior_provenance['train_jsonl_sha256']
                    or source_provenance['manifest_sha256'] != digest(output / 'manifest-full.jsonl')
                    or source_provenance['source_sha256']['runtime.py'] != digest(Path('asr/runtime.py'))
                    or source_provenance['models'][slug]['revision'] != MODELS[slug]['revision']):
                raise ValueError('Resume requires identical input, checkpoint and inference source')
            candidate = source_root / slug / 'full/predictions.jsonl'
            if candidate.exists():
                resume_file = candidate
        config = run_phase(workers[slug], slug, phase, manifest if phase == 'full' else smoke,
                           rows, root, output, budget, resume_from=resume_file)
        write_json(output / 'jobs.json', [config])
        write_json(output / 'summary.json', {'jobs': [config], 'estimated_compute_usd': budget.estimate()})
        print(json.dumps({'event': 'single_model_finished', 'output': str(output), 'model': slug,
                          'status': config['status'], 'estimated_compute_usd': budget.estimate()}), flush=True)
        return

    def execute(phase, names, selected):
        found = []
        if len(names) == 1:
            slug = names[0]
            config = run_phase(workers[slug], slug, phase, selected, rows, root, output, budget)
            results.append(config)
            write_json(output / "jobs.json", results)
            return [config]
        with ThreadPoolExecutor(max_workers=min(5, len(names))) as pool:
            futures = {pool.submit(run_phase, workers[s], s, phase, selected, rows, root, output, budget): s for s in names}
            for future in as_completed(futures):
                config = future.result()
                found.append(config)
                results.append(config)
                write_json(output / "jobs.json", results)
        return found

    smoke_results = execute("smoke100", slugs, smoke)
    eligible, estimates = [], []
    for config in smoke_results:
        slug = config["model_slug"]
        summary = config.get("summary", {})
        healthy = config["status"] == "completed" and summary.get("num_success") == len(smoke) and not config.get("hit_token_limit_count")
        seconds = config["wall_seconds"] / len(smoke) * len(rows) * 1.25 + 60
        estimates.append({"model": slug, "smoke_healthy": healthy, "full_seconds_with_25_percent_margin": seconds, "full_estimated_usd": seconds * WORKER_RATE_USD_SECOND})
        if healthy:
            eligible.append(slug)
    projected = sum(e["full_estimated_usd"] for e in estimates if e["smoke_healthy"])
    estimate = {"models": estimates, "compute_estimate_so_far_usd": budget.estimate(), "full_projected_usd": projected,
                "remaining_after_1_usd_reserve": budget.available(), "smoke_only": smoke_only}
    write_json(output / "budget_estimate.json", estimate)
    print(json.dumps({"event": "smoke_completed", **estimate}), flush=True)
    if not smoke_only and eligible and projected <= budget.available():
        full_results = execute("full", eligible, manifest)
        completed = [c for c in full_results if c["status"] == "completed"]
        best = sorted(completed, key=lambda c: c["summary"]["normalized"]["wer"])[:2]
        repeat_cost = sum(c["wall_seconds"] / len(rows) * len(smoke) * 1.25 * WORKER_RATE_USD_SECOND + 0.02 for c in best)
        if best and repeat_cost <= budget.available():
            repeated = execute("repeat100", [c["model_slug"] for c in best], smoke)
            for config in repeated:
                if config["status"] != "completed":
                    continue
                from metrics.io import read_jsonl
                original = {r["id"]: r["text"] for r in read_jsonl(output / config["model_slug"] / "full/predictions.jsonl")}
                replay = read_jsonl(output / config["model_slug"] / "repeat100/predictions.jsonl")
                write_json(output / config["model_slug"] / "repeat100/reproducibility.json", {
                    "num_samples": len(replay), "same_text": sum(r["text"] == original[r["id"]] for r in replay),
                    "all_match": all(r["status"] == "ok" and r["text"] == original[r["id"]] for r in replay),
                })
    elif not smoke_only:
        print(json.dumps({"event": "full_not_started", "reason": "No healthy model or projected cost exceeds remaining budget; review smoke results"}), flush=True)
    flattened = []
    for config in results:
        summary = config.get("summary", {})
        flattened.append({"model": config["model_slug"], "phase": config["phase"], "status": config["status"],
                          "num_samples": config.get("num_samples"), "num_predictions": config.get("num_predictions", 0),
                          "wer_percent": summary.get("normalized", {}).get("wer_percent"),
                          "cer_percent": summary.get("normalized", {}).get("cer_percent"),
                          "mean_seconds": summary.get("latency_seconds", {}).get("mean"),
                          "p95_seconds": summary.get("latency_seconds", {}).get("p95"),
                          "errors": summary.get("num_errors"), "timeouts": summary.get("num_timeouts"),
                          "num_parameters": config.get("num_parameters"), "wall_seconds": config["wall_seconds"],
                          "error": config.get("error")})
    write_json(output / "summary.json", {"jobs": flattened, "estimated_compute_usd": budget.estimate(), "budget_usd": budget_usd})
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
    print(json.dumps({"event": "suite_finished", "output": str(output), "jobs": flattened, "estimated_compute_usd": budget.estimate()}), flush=True)
