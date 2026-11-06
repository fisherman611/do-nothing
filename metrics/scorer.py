"""State-gated, macro-averaged tool-call F1 for LifeMate.

This module compares supplied hashes. It never reads app environments, computes
hashes itself, or replays submitted tool calls.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from math import fsum
from typing import Any, Iterable, Mapping
import unicodedata

from .records import ReplayRecord, TimingRecord


def token_f1(prediction: str, reference: str) -> float:
    """Multiset token F1: lowercase, NFC, then whitespace tokenization.

    Punctuation and accents are preserved. Two empty strings score 1; only one
    empty string scores 0. Non-string fuzzy values are never coerced to text.
    """
    if not isinstance(prediction, str) or not isinstance(reference, str):
        return 0.0
    predicted = Counter(unicodedata.normalize("NFC", prediction.lower()).split())
    expected = Counter(unicodedata.normalize("NFC", reference.lower()).split())
    length = sum(predicted.values()) + sum(expected.values())
    if length == 0:
        return 1.0
    overlap = sum((predicted & expected).values())
    return 2.0 * overlap / length


def _arguments_match(prediction: Any, reference: Any, schema: dict) -> bool:
    if schema.get("x-match") == "fuzzy":
        return token_f1(prediction, reference) > 0.6
    if isinstance(reference, dict):
        if not isinstance(prediction, dict) or prediction.keys() != reference.keys():
            return False
        properties = schema.get("properties", {})
        return all(
            _arguments_match(prediction[key], value, properties.get(key, {}))
            for key, value in reference.items()
        )
    if isinstance(reference, list):
        return (
            isinstance(prediction, list)
            and len(prediction) == len(reference)
            and all(
                _arguments_match(p, r, schema.get("items", {}))
                for p, r in zip(prediction, reference)
            )
        )
    # JSON numbers compare by value; booleans must not match 0/1 in Python.
    if type(reference) in (int, float):
        return type(prediction) in (int, float) and prediction == reference
    return type(prediction) is type(reference) and prediction == reference


def _valid_call(call: Any) -> bool:
    return (
        isinstance(call, dict)
        and set(call) == {"name", "arguments"}
        and isinstance(call["name"], str)
        and isinstance(call["arguments"], dict)
    )


@dataclass(frozen=True)
class ToolCatalog:
    """Tool argument schemas, including x-match annotations, from one split."""

    parameters: dict[str, dict]

    @classmethod
    def from_dict(cls, catalog: dict) -> ToolCatalog:
        if not isinstance(catalog, dict) or not isinstance(catalog.get("apps"), list):
            raise ValueError("Catalog must contain an apps list")
        parameters = {}
        for app in catalog["apps"]:
            if not isinstance(app, dict) or not isinstance(app.get("tools"), list):
                raise ValueError("Each catalog app must contain a tools list")
            for tool in app["tools"]:
                if (
                    not isinstance(tool, dict)
                    or not isinstance(tool.get("name"), str)
                    or not tool["name"]
                    or not isinstance(tool.get("parameters"), dict)
                ):
                    raise ValueError("Each tool needs a name and parameter schema")
                if tool["name"] in parameters:
                    raise ValueError(f"Duplicate catalog tool: {tool['name']}")
                parameters[tool["name"]] = tool["parameters"]
        return cls(parameters)

    def calls_match(self, prediction: Any, reference: Any) -> bool:
        if not _valid_call(prediction) or not _valid_call(reference):
            return False
        name = prediction["name"]
        return (
            name == reference["name"]
            and name in self.parameters
            and _arguments_match(
                prediction["arguments"], reference["arguments"], self.parameters[name]
            )
        )


@dataclass(frozen=True)
class CallMetrics:
    matched_calls: int
    submitted_calls: int
    gold_calls: int
    unmatched_submitted_calls: int
    missing_gold_calls: int
    precision: float
    recall: float
    tool_f1: float
    matched_pairs: tuple[tuple[int, int], ...]  # (submitted index, gold index)


def match_tool_calls(
    predictions: list, references: list, catalog: ToolCatalog,
    *, rejected_call_indices: tuple[int, ...] = (),
) -> CallMetrics:
    """Find maximum one-to-one matching, independent of call order.

    All submitted entries count, including malformed/unknown calls. Reference
    calls must be well-formed and their tools must exist in the chosen catalog.
    """
    if not isinstance(predictions, list) or not isinstance(references, list):
        raise ValueError("tool_calls must be a list")
    for call in references:
        if not _valid_call(call) or call["name"] not in catalog.parameters:
            raise ValueError(f"Invalid gold call or wrong tool catalog: {call!r}")
    rejected = set(rejected_call_indices)
    if any(type(index) is not int or not 0 <= index < len(predictions) for index in rejected):
        raise ValueError("Rejected call index outside submitted tool_calls")
    adjacency = [
        [j for j, reference in enumerate(references) if catalog.calls_match(call, reference)]
        if i not in rejected else []
        for i, call in enumerate(predictions)
    ]
    # Augmenting paths can reassign a previous pair; greedy matching cannot.
    matched_gold: dict[int, int] = {}

    def augment(predicted_index: int, seen: set[int]) -> bool:
        for gold_index in adjacency[predicted_index]:
            if gold_index in seen:
                continue
            seen.add(gold_index)
            if gold_index not in matched_gold or augment(matched_gold[gold_index], seen):
                matched_gold[gold_index] = predicted_index
                return True
        return False

    for index in range(len(predictions)):
        augment(index, set())
    matched = len(matched_gold)
    submitted_count, gold_count = len(predictions), len(references)
    both_empty = submitted_count == gold_count == 0
    precision = matched / submitted_count if submitted_count else float(both_empty)
    recall = matched / gold_count if gold_count else float(both_empty)
    # Equivalent to 2PR/(P+R), including the zero-match convention.
    f1 = 2.0 * matched / (submitted_count + gold_count) if not both_empty else 1.0
    return CallMetrics(
        matched, submitted_count, gold_count, submitted_count - matched,
        gold_count - matched, precision, recall, f1,
        tuple(sorted((predicted, gold) for gold, predicted in matched_gold.items())),
    )


@dataclass(frozen=True)
class SampleMetrics:
    id: str
    score: float
    state_match: bool
    missing_prediction: bool
    calls: CallMetrics
    tie_break_f1: float
    execution_status: str
    elapsed_seconds: float | None
    replay_status: str | None
    replay_elapsed_seconds: float | None
    reported_state_hash: str | None
    effective_state_hash: str | None
    hash_consistent: bool | None
    rejected_call_indices: tuple[int, ...]
    execution_error: str | None
    replay_error: str | None


def _validate_record(record: Any, *, gold: bool) -> None:
    role = "Gold" if gold else "Submission"
    if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"]:
        raise ValueError(f"{role} row needs a non-empty string id")
    if not isinstance(record.get("tool_calls"), list):
        hint = "; run prepare-gold for train.jsonl labels" if gold and "label" in record else ""
        raise ValueError(f"{role} {record['id']}: tool_calls must be a list{hint}")
    if not isinstance(record.get("state_hash"), str) or not record["state_hash"]:
        raise ValueError(f"{role} {record['id']}: state_hash must be a non-empty string")


def score_sample(
    prediction: Mapping | None, reference: Mapping, catalog: ToolCatalog,
    *, timing: TimingRecord | None = None, replay: ReplayRecord | None = None,
) -> SampleMetrics:
    """Hash equality gates call F1, including samples with empty gold calls."""
    _validate_record(reference, gold=True)
    if prediction is not None:
        _validate_record(prediction, gold=False)
        if prediction["id"] != reference["id"]:
            raise ValueError("Submission id and gold id must match")
    for record in (timing, replay):
        if record is not None and record.id != reference["id"]:
            raise ValueError("Execution evidence id and gold id must match")
    calls = match_tool_calls(
        prediction["tool_calls"] if prediction is not None else [],
        reference["tool_calls"], catalog,
        rejected_call_indices=replay.rejected_call_indices if replay else (),
    )
    if prediction is None:
        # A missing row is a failed sample, even when its gold list is empty.
        calls = CallMetrics(
            0, 0, calls.gold_calls, 0, calls.gold_calls, 0.0, 0.0, 0.0, (),
        )
    reported_hash = prediction["state_hash"] if prediction is not None else None
    effective_hash = replay.state_hash if replay else reported_hash
    status = "ok"
    if prediction is None:
        status = "missing_prediction"
    if replay and replay.status != "ok":
        status = "replay_" + replay.status
    if timing:
        if timing.status == "timeout" or timing.elapsed_seconds > 30:
            status = "timeout"
        elif timing.status == "error":
            status = "error"
    state_match = effective_hash is not None and effective_hash == reference["state_hash"]
    eligible = status == "ok"
    return SampleMetrics(
        reference["id"], calls.tool_f1 if state_match and eligible else 0.0,
        state_match, prediction is None, calls, calls.tool_f1 if eligible else 0.0,
        status, timing.elapsed_seconds if timing else None,
        replay.status if replay else None, replay.elapsed_seconds if replay else None,
        reported_hash, effective_hash,
        (reported_hash == replay.state_hash) if replay and replay.status == "ok" else None,
        replay.rejected_call_indices if replay else (),
        (timing.error or ("Inference exceeded 30 seconds" if timing.elapsed_seconds > 30 else None)) if timing else None,
        replay.error if replay else None,
    )


@dataclass(frozen=True)
class EvaluationReport:
    summary: dict[str, Any]
    samples: tuple[SampleMetrics, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _index_records(records: Iterable[Mapping], *, gold: bool) -> dict[str, Mapping]:
    indexed = {}
    for record in records:
        _validate_record(record, gold=gold)
        if record["id"] in indexed:
            raise ValueError(f"Duplicate {'gold' if gold else 'submission'} id: {record['id']}")
        indexed[record["id"]] = record
    return indexed


def evaluate(
    predictions: Iterable[Mapping], references: Iterable[Mapping],
    catalog: ToolCatalog, *, require_all: bool = False,
    timings: Iterable[TimingRecord] | None = None,
    replays: Iterable[ReplayRecord] | None = None,
) -> EvaluationReport:
    """Macro average over all gold IDs; omitted samples score zero.

    Duplicate IDs, unknown submitted IDs and invalid JSONL records are errors,
    rather than silently ignored rows. require_all also rejects missing rows.
    """
    gold = _index_records(references, gold=True)
    submitted = _index_records(predictions, gold=False)
    if not gold:
        raise ValueError("Gold dataset must contain at least one sample")
    extra = submitted.keys() - gold.keys()
    if extra:
        raise ValueError(f"Unknown submitted ids: {', '.join(sorted(extra))}")
    missing = gold.keys() - submitted.keys()
    if missing and require_all:
        raise ValueError(f"Missing submitted ids: {', '.join(sorted(missing))}")
    def evidence_index(records, expected_ids, label):
        if records is None:
            return {}
        indexed = {}
        for record in records:
            if record.id in indexed:
                raise ValueError(f"Duplicate {label} id: {record.id}")
            indexed[record.id] = record
        if indexed.keys() != expected_ids:
            raise ValueError(f"{label} IDs must exactly match {'gold' if label == 'timing' else 'submitted'} IDs")
        return indexed

    times = evidence_index(timings, gold.keys(), "timing")
    replayed = evidence_index(replays, submitted.keys(), "replay")
    samples = tuple(score_sample(
        submitted.get(key), value, catalog, timing=times.get(key), replay=replayed.get(key),
    ) for key, value in gold.items())
    count = len(samples)
    mean_score = fsum(s.score for s in samples) / count

    def fingerprint(value):
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    return EvaluationReport(
        {
            "metric_version": 2,
            "score_scale": 100,
            "num_runs": 1,
            "catalog_fingerprint": fingerprint(catalog.parameters),
            "gold_fingerprint": fingerprint(gold),
            "num_samples": count,
            "num_submitted_samples": len(submitted),
            "num_missing_samples": len(missing),
            "score": 100.0 * mean_score,
            "mean_sample_score": mean_score,
            "tool_call_f1": fsum(s.calls.tool_f1 for s in samples) / count,
            "tie_break_score": 100.0 * fsum(s.tie_break_f1 for s in samples) / count,
            "state_match_rate": sum(s.state_match for s in samples) / count,
            "hash_verification": "replay" if replays is not None else "declared_only",
            "num_hash_discrepancies": sum(s.hash_consistent is False for s in samples),
            "num_failed_samples": sum(s.execution_status != "ok" for s in samples),
            "num_timeout_samples": sum(s.execution_status == "timeout" for s in samples),
            "num_replay_timeouts": sum(s.replay_status == "timeout" for s in samples),
            "timing_available": timings is not None,
            "time_limit_seconds": 30,
            "average_processing_seconds": fsum(t.elapsed_seconds for t in times.values()) / count if times else None,
            "max_processing_seconds": max(t.elapsed_seconds for t in times.values()) if times else None,
            "matched_calls": sum(s.calls.matched_calls for s in samples),
            "submitted_calls": sum(s.calls.submitted_calls for s in samples),
            "gold_calls": sum(s.calls.gold_calls for s in samples),
        }, samples,
    )
