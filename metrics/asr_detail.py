"""Diagnostic ASR metrics beyond corpus WER/CER. Requires jiwer==4.0.0.

* entity error rate: entities (from labels, by type) and surface entities
  (digit tokens, alphanumeric codes, foreign words) that occur in the
  reference and are missing from the hypothesis;
* code-switching WER: WER on utterances that contain foreign words or codes
  versus purely Vietnamese utterances;
* WER by utterance kind (final turn / history);
* real-time factor when processing seconds and audio durations are given.
"""

from collections import Counter, defaultdict
from typing import Iterable, Mapping
import re
import unicodedata

from .asr import error_counts

_PUNCT = re.compile(r"[^\w\s]", re.U)
_VI_SYLLABLE = re.compile(
    r"(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdđghklmnprstvx])?[aeiouy]+(?:ch|ng|nh|[cmnpt])?$")
_PERSON_KEYS = {"assignee", "members", "from_name", "recipient", "to", "contact", "participants", "attendees"}
_PLACE_KEYS = {"location", "destination", "origin", "station", "near", "city", "address", "stop", "place", "room"}
SURFACE_TYPES = ("number", "code", "foreign_word")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", (text or "").lower())
    return " ".join(_PUNCT.sub(" ", text).split())


def _plain(token: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", token) if unicodedata.category(c) != "Mn").replace("đ", "d")


def surface_type(raw_token: str) -> str | None:
    """Type of a reference token that is risky for a Vietnamese ASR, or None."""
    token = _PUNCT.sub("", unicodedata.normalize("NFC", raw_token))
    if not token:
        return None
    if token.isdigit():
        return "number"
    if any(c.isdigit() for c in token) or (token.isupper() and token.isascii() and len(token) >= 2):
        return "code"
    lowered = token.lower()
    if lowered.isascii() and lowered.isalpha() and len(lowered) >= 3 and not _VI_SYLLABLE.match(lowered):
        return "foreign_word"
    return None


def entity_type(app: str, key: str) -> str:
    if app == "contacts" or key in _PERSON_KEYS:
        return "person"
    if key in _PLACE_KEYS:
        return "place"
    return "other_name"


def reference_entities(text: str, labeled: Iterable[tuple[str, str, str]]) -> list[tuple[str, str]]:
    """(type, normalized value) pairs that actually occur in the reference text."""
    padded = f" {normalize(text)} "
    found, seen = [], set()
    for app, key, value in labeled:
        value = normalize(str(value))
        if len(value) >= 2 and len(value.split()) <= 8 and f" {value} " in padded and value not in seen:
            seen.add(value)
            found.append((entity_type(app, key), value))
    for raw in (text or "").split():
        kind = surface_type(raw)
        value = normalize(raw)
        if kind and value and (kind, value) not in seen:
            seen.add((kind, value))
            found.append((kind, value))
    return found


def evaluate_asr_detail(predictions: Mapping[str, str], references: Iterable[Mapping], *,
                        seconds: Mapping[str, float] | None = None,
                        durations: Mapping[str, float] | None = None) -> dict:
    """references: rows with uid, text, optional kind and entities [(app, key, value)]."""
    groups = defaultdict(list)
    entities = defaultdict(Counter)
    samples = []
    spent = audio = 0.0
    for row in references:
        uid, text = row["uid"], row["text"]
        hypothesis = predictions.get(uid, "")
        ref_n, hyp_n = normalize(text), normalize(hypothesis)
        if not ref_n:
            continue
        counts = error_counts(ref_n, hyp_n)
        chars = error_counts(ref_n, hyp_n, characters=True)
        found = reference_entities(text, row.get("entities") or [])
        switched = any(kind in ("foreign_word", "code") for kind, _ in found)
        for name in ("all", f"kind:{row.get('kind', 'unknown')}", "code_switching" if switched else "vietnamese_only"):
            groups[name].append((counts, chars))
        padded = f" {hyp_n} "
        plain = f" {_plain(hyp_n)} "
        missed = []
        for kind, value in found:
            hit = f" {value} " in padded
            entities[kind].update(total=1, hit=int(hit), hit_no_accent=int(hit or f" {_plain(value)} " in plain))
            if not hit:
                missed.append([kind, value])
        if seconds and durations and uid in seconds and uid in durations:
            spent += seconds[uid]
            audio += durations[uid]
        samples.append({"uid": uid, "kind": row.get("kind"), "wer": counts.rate, "code_switching": switched,
                        "missed_entities": missed, "reference": text, "prediction": hypothesis})

    def rates(items: list) -> dict:
        words = [c for c, _ in items]
        letters = [c for _, c in items]
        def total(parts, name):
            return sum(getattr(p, name) for p in parts)
        def rate(parts):
            length = total(parts, "hits") + total(parts, "substitutions") + total(parts, "deletions")
            errors = total(parts, "substitutions") + total(parts, "deletions") + total(parts, "insertions")
            return errors / length if length else None
        return {"utterances": len(items), "wer": rate(words), "cer": rate(letters),
                "exact_utterances": sum(c.rate == 0 for c in words) / len(items) if items else None}

    entity_summary = {}
    for kind, counter in sorted(entities.items()):
        entity_summary[kind] = {"total": counter["total"], "missed": counter["total"] - counter["hit"],
                                "error_rate": 1 - counter["hit"] / counter["total"],
                                "error_rate_no_accent": 1 - counter["hit_no_accent"] / counter["total"]}
    labeled = [k for k in entity_summary if k not in SURFACE_TYPES]
    total = sum(entity_summary[k]["total"] for k in labeled)
    missed = sum(entity_summary[k]["missed"] for k in labeled)
    summary = {
        "groups": {name: rates(items) for name, items in sorted(groups.items())},
        "entity_error_rate": {"labeled_total": total, "labeled_missed": missed,
                              "labeled_error_rate": missed / total if total else None, "by_type": entity_summary},
        "real_time_factor": spent / audio if audio else None,
    }
    return {"summary": summary, "samples": samples}


def _listed(value) -> list:
    if isinstance(value, str):      # manifests store these lists as Python literals
        import ast
        value = ast.literal_eval(value) if value.strip() else []
    return [tuple(item) for item in value or [] if len(item) == 3]


def load_references(data, split: str | None = None) -> list[dict]:
    """Normalize a manifest (list or {items: [...]}) into rows for evaluate_asr_detail."""
    rows = data["items"] if isinstance(data, dict) else data
    out = []
    for row in rows:
        if split and row.get("split") != split:
            continue
        text = row.get("text") if row.get("text") is not None else row.get("ref")
        if not isinstance(text, str):
            continue
        entities = _listed(row.get("entities")) + _listed(row.get("label_ents")) + _listed(row.get("hist_ents"))
        out.append({"uid": row["uid"], "text": text, "kind": row.get("kind"), "entities": entities,
                    "audio": row.get("audio"), "root": row.get("root")})
    return out


def wav_durations(rows: Iterable[Mapping], audio_root) -> dict[str, float]:
    """Durations in seconds from WAV headers; rows without a readable file are skipped."""
    import wave
    from pathlib import Path
    out = {}
    for row in rows:
        if not row.get("audio"):
            continue
        path = Path(audio_root) / (row.get("root") or "") / row["audio"]
        try:
            with wave.open(str(path), "rb") as handle:
                out[row["uid"]] = handle.getnframes() / handle.getframerate()
        except (OSError, wave.Error, EOFError):
            continue
    return out
