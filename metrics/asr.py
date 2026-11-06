"""Corpus WER/CER for labeled ASR data. Requires jiwer==4.0.0."""

from collections import Counter
from dataclasses import dataclass
from math import fsum, isfinite
from typing import Iterable
import unicodedata


def normalize_text(text: str, *, normalized: bool = True) -> str:
    if not isinstance(text, str):
        raise ValueError("ASR transcript must be a string")
    text = unicodedata.normalize("NFC", text)
    if normalized:
        text = text.lower()
        # Keep semantic symbols, negative signs and numeric separators.
        text = "".join(
            char for i, char in enumerate(text)
            if not unicodedata.category(char).startswith("P") or char == "%"
            or (char == "-" and i + 1 < len(text) and text[i + 1].isdigit())
            or (char in ".,:/-" and i > 0 and i + 1 < len(text)
                and text[i - 1].isdigit() and text[i + 1].isdigit())
        )
    return " ".join(text.split())


@dataclass(frozen=True)
class ErrorCounts:
    hits: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def reference_length(self):
        return self.hits + self.substitutions + self.deletions

    @property
    def errors(self):
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self):
        # jiwer 4: insertions against an empty reference are counted directly.
        return self.errors / self.reference_length if self.reference_length else float(self.errors)

    def to_dict(self):
        return {
            "hits": self.hits, "substitutions": self.substitutions,
            "deletions": self.deletions, "insertions": self.insertions,
            "reference_length": self.reference_length, "errors": self.errors,
        }


def error_counts(reference: str, prediction: str, *, characters: bool = False) -> ErrorCounts:
    try:
        import jiwer
    except ImportError as error:
        raise ValueError("ASR metrics need jiwer==4.0.0; install requirements.txt") from error
    output = (jiwer.process_characters if characters else jiwer.process_words)(reference, prediction)
    return ErrorCounts(output.hits, output.substitutions, output.deletions, output.insertions)


def _combined(counts: Iterable[dict]) -> dict:
    totals = Counter()
    for count in counts:
        totals.update({key: count[key] for key in ("hits", "substitutions", "deletions", "insertions")})
    combined = ErrorCounts(**{key: totals[key] for key in ("hits", "substitutions", "deletions", "insertions")})
    return {**combined.to_dict(), "rate": combined.rate, "percent": 100 * combined.rate}


def _duration(row: dict, name: str, *, positive: bool = False):
    value = row.get(name)
    if value is None:
        return None
    if type(value) not in (int, float) or not isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{row['id']}: {name} must be finite and {'positive' if positive else 'non-negative'}")
    return value


def _percentile(values: list[float], fraction: float):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def evaluate_asr(predictions: Iterable[dict], references: Iterable[dict], *, require_all: bool = False) -> dict:
    """Score raw train rows (id/audio_text) against id/text ASR predictions.

    Missing/error/timeout predictions are evaluated as empty strings, never
    dropped from corpus denominators. A successful empty transcript is valid.
    Raw = NFC + whitespace collapse. Normalized adds lowercase/punctuation
    removal, preserving accents, digits, numeric signs/separators and percent signs.
    """
    gold, submitted = {}, {}
    for row in references:
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in gold:
            raise ValueError("Invalid or duplicate ASR reference id")
        if not isinstance(row.get("audio_text"), str):
            raise ValueError(f"{row['id']}: reference needs audio_text (train only)")
        gold[row["id"]] = row
    if not gold:
        raise ValueError("ASR references must contain at least one sample")
    for row in predictions:
        if not isinstance(row.get("id"), str) or row["id"] in submitted:
            raise ValueError("Invalid or duplicate ASR prediction id")
        if not isinstance(row.get("text"), str):
            raise ValueError(f"{row['id']}: prediction needs a text string")
        if row.get("status", "ok") not in ("ok", "error", "timeout"):
            raise ValueError(f"{row['id']}: invalid ASR status")
        submitted[row["id"]] = row
    if submitted.keys() - gold.keys():
        raise ValueError("ASR prediction contains IDs outside references")
    if require_all and gold.keys() - submitted.keys():
        raise ValueError("Missing ASR prediction IDs")
    samples = []
    for sample_id, reference in gold.items():
        predicted = submitted.get(sample_id)
        status = predicted.get("status", "ok") if predicted is not None else "missing"
        elapsed = _duration(predicted, "elapsed_seconds") if predicted is not None else None
        audio_seconds = _duration(predicted, "audio_seconds", positive=True) if predicted is not None else None
        if elapsed is not None and elapsed > 30:
            status = "timeout"
        text = predicted["text"] if predicted is not None and status == "ok" else ""
        sample = {
            "id": sample_id, "status": status,
            "reference": reference["audio_text"],
            "prediction": predicted["text"] if predicted else "",
            "elapsed_seconds": elapsed, "audio_seconds": audio_seconds,
            "error": predicted.get("error") if predicted else "Prediction missing",
        }
        for mode in ("raw", "normalized"):
            ref = normalize_text(reference["audio_text"], normalized=mode == "normalized")
            hyp = normalize_text(text, normalized=mode == "normalized")
            words = error_counts(ref, hyp)
            chars = error_counts(ref, hyp, characters=True)
            sample[mode] = {
                "reference": ref, "prediction": hyp,
                "wer": words.rate, "cer": chars.rate,
                "word_counts": words.to_dict(), "character_counts": chars.to_dict(),
                "exact_match": status == "ok" and ref == hyp,
            }
        samples.append(sample)
    count = len(samples)
    elapsed = [s["elapsed_seconds"] for s in samples if s["elapsed_seconds"] is not None]
    timed_audio = [s for s in samples if s["elapsed_seconds"] is not None and s["audio_seconds"] is not None]
    audio_total = fsum(s["audio_seconds"] for s in timed_audio)
    summary = {
        "num_samples": count, "num_predictions": len(submitted),
        "num_success": sum(s["status"] == "ok" for s in samples),
        "num_errors": sum(s["status"] == "error" for s in samples),
        "num_timeouts": sum(s["status"] == "timeout" for s in samples),
        "num_missing": sum(s["status"] == "missing" for s in samples),
        "latency_seconds": {
            "num_measured": len(elapsed), "total": fsum(elapsed),
            "mean": fsum(elapsed) / len(elapsed) if elapsed else None,
            "p50": _percentile(elapsed, 0.5), "p95": _percentile(elapsed, 0.95),
            "max": max(elapsed) if elapsed else None,
        },
        "real_time_factor": fsum(s["elapsed_seconds"] for s in timed_audio) / audio_total if audio_total else None,
        "rtf_audio_seconds": audio_total,
        "normalization": "NFC; collapse whitespace; normalized adds lowercase and punctuation removal; preserve diacritics, digits, numeric signs/separators, symbols and %",
        "cer_includes_spaces": True,
    }
    for mode in ("raw", "normalized"):
        words = _combined(s[mode]["word_counts"] for s in samples)
        chars = _combined(s[mode]["character_counts"] for s in samples)
        summary[mode] = {
            "wer": words["rate"], "wer_percent": words["percent"],
            "cer": chars["rate"], "cer_percent": chars["percent"],
            "word_counts": words, "character_counts": chars,
            "exact_match_rate": sum(s[mode]["exact_match"] for s in samples) / count,
        }
    return {"summary": summary, "samples": samples}
