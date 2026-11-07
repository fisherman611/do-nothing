"""Public checkpoint locks; all revisions predate 2026-10-26."""

MODELS = {
    "phowhisper-medium": {
        "model_id": "vinai/PhoWhisper-medium", "family": "whisper",
        "revision": "55a7e3eb6c906de891f8f06a107754427dd3be79",
        "checkpoint_date": "2024-11-12", "license": "bsd-3-clause", "safetensors": False,
    },
    "phowhisper-large": {
        "model_id": "vinai/PhoWhisper-large", "family": "whisper",
        "revision": "b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb",
        "checkpoint_date": "2024-11-12", "license": "bsd-3-clause", "safetensors": False,
    },
    "whisper-large-v3-turbo": {
        "model_id": "openai/whisper-large-v3-turbo", "family": "whisper",
        "revision": "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
        "checkpoint_date": "2024-10-04", "license": "mit", "safetensors": True,
    },
    "qwen3-asr-0.6b": {
        "model_id": "Qwen/Qwen3-ASR-0.6B", "family": "qwen",
        "revision": "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
        "checkpoint_date": "2026-01-30", "license": "apache-2.0", "safetensors": True,
        "serialized_parameters": 938008576,
    },
    "qwen3-asr-1.7b": {
        "model_id": "Qwen/Qwen3-ASR-1.7B", "family": "qwen",
        "revision": "7278e1e70fe206f11671096ffdd38061171dd6e5",
        "checkpoint_date": "2026-01-30", "license": "apache-2.0", "safetensors": True,
        "serialized_parameters": 2349217408,
    },
}

WORKER_RATE_USD_SECOND = 0.000164 + 2 * 0.0000131 + 8 * 0.00000222
MODEL_ROOT = "/opt/models"


def download_family(family):
    from huggingface_hub import snapshot_download
    for slug, spec in MODELS.items():
        if spec["family"] != family:
            continue
        weight = "*.safetensors" if spec["safetensors"] else "pytorch_model*.bin"
        snapshot_download(
            spec["model_id"], revision=spec["revision"], local_dir=f"{MODEL_ROOT}/{slug}",
            allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", weight, "LICENSE*"],
        )
