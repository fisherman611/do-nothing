"""GPU runtime. Inputs contain only an ID and a single WAV."""

import io
import time


class Runtime:
    def __init__(self, slug):
        import importlib.metadata
        import random
        import numpy as np
        import torch
        from asr.registry import MODELS, MODEL_ROOT

        started = time.perf_counter()
        self.spec = MODELS[slug]
        self.family = self.spec["family"]
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
        path = f"{MODEL_ROOT}/{slug}"
        if self.family == "whisper":
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                path, local_files_only=True, dtype=torch.float16,
                use_safetensors=self.spec["safetensors"], attn_implementation="sdpa",
            ).to("cuda").eval()
        else:
            from qwen_asr import Qwen3ASRModel
            self.wrapper = Qwen3ASRModel.from_pretrained(
                path, local_files_only=True, dtype=torch.float16, device_map="cuda:0",
                attn_implementation="sdpa", max_inference_batch_size=1, max_new_tokens=256,
            )
            self.model = self.wrapper.model.eval()
            self.processor = self.wrapper.processor
            self.model.generation_config.do_sample = False
            self.model.generation_config.num_beams = 1
            self.model.generation_config.return_dict_in_generate = True
        count = sum(p.numel() for p in self.model.parameters())
        if count >= 6_000_000_000:
            raise ValueError("ASR alone exceeds the system parameter limit")
        torch.cuda.synchronize()
        load = time.perf_counter() - started
        started = time.perf_counter()
        self.decode(np.zeros(16000, dtype=np.float32), max_tokens=1)
        torch.cuda.synchronize()
        versions = {}
        for name in ("torch", "transformers", "accelerate", "numpy", "soundfile", "huggingface-hub", "qwen-asr"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        self.metadata = {
            **self.spec, "model_slug": slug, "num_parameters": count,
            "gpu": torch.cuda.get_device_name(0), "dtype": "float16", "attention": "sdpa",
            "cuda_version": torch.version.cuda, "versions": versions,
            "model_load_seconds": load, "warmup_seconds": time.perf_counter() - started,
            "decoding": {"language": "Vietnamese", "do_sample": False, "num_beams": 1, "max_new_tokens": 256},
            "seed": 42, "batch_size": 1, "network_blocked": True,
            "timing_scope": "WAV decode + preprocessing + generation + text decode; excludes load/warmup/RPC",
        }

    def decode(self, audio, max_tokens=256):
        import torch
        with torch.inference_mode():
            if self.family == "whisper":
                features = self.processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
                tokens = self.model.generate(
                    features.input_features.to("cuda", dtype=torch.float16),
                    attention_mask=features.attention_mask.to("cuda"), language="vi", task="transcribe",
                    do_sample=False, num_beams=1, max_new_tokens=max_tokens, return_timestamps=False,
                )
                return self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip(), int(tokens.shape[-1]) >= max_tokens + 3
            # Use the package's prompt and parser, while explicitly enforcing greedy generation.
            from qwen_asr.inference.utils import parse_asr_output
            prompt = self.wrapper._build_text_prompt(context="", force_language="Vietnamese")
            features = self.processor(text=[prompt], audio=[audio], return_tensors="pt", padding=True)
            features = features.to(self.model.device).to(self.model.dtype)
            tokens = self.model.generate(
                **features, max_new_tokens=max_tokens, do_sample=False, num_beams=1,
                pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
            ).sequences[:, features["input_ids"].shape[1]:]
            raw = self.processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            _, text = parse_asr_output(raw, user_language="Vietnamese")
            return text.strip(), int(tokens.shape[-1]) >= max_tokens

    def transcribe(self, payload):
        import soundfile as sf
        import torch
        started = time.perf_counter()
        result = {"id": payload["id"], "text": "", "status": "error", "error": None}
        try:
            audio, rate = sf.read(io.BytesIO(payload["audio_bytes"]), dtype="float32")
            if rate != 16000 or audio.ndim != 1 or not len(audio):
                raise ValueError("Expected nonempty mono 16-kHz WAV")
            if len(audio) > 30 * rate:
                raise ValueError("This short-audio baseline does not silently truncate WAVs over 30 seconds")
            result["audio_seconds"] = len(audio) / rate
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            result["text"], result["hit_token_limit"] = self.decode(audio)
            torch.cuda.synchronize()
            result["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 1024**2
            result["status"] = "ok"
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        result["elapsed_seconds"] = time.perf_counter() - started
        if result["elapsed_seconds"] > 30:
            result["status"] = "timeout"
        return result
