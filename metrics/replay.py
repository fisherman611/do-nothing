"""Verify calls using only the official ToolEnv participant API."""

from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import sys

from .execution import run_isolated
from .io import checked_path
from .records import ReplayRecord, valid_duration
from .scorer import _valid_call, _validate_record


def runtime_paths(data_root: Path, split: str) -> tuple[Path, Path]:
    if split not in ("train", "public_test", "private_test"):
        raise ValueError("split must be train, public_test or private_test")
    directory = checked_path(data_root).resolve() / split
    catalog = directory / f"{split}_tools_catalog.json"
    if not (directory / "zaic_tools" / "__init__.py").is_file() or not catalog.is_file():
        raise ValueError(f"Missing official {split}/zaic_tools or split catalog under {data_root}")
    return directory, catalog


def load_tool_env(data_root: Path, split: str):
    directory, _ = runtime_paths(data_root, split)
    sys.path.insert(0, str(directory))
    runtime = importlib.import_module("zaic_tools")
    if Path(runtime.__file__).resolve().parent != (directory / "zaic_tools").resolve():
        raise ValueError("A different zaic_tools is already imported; use a fresh process")
    return runtime.ToolEnv


def runtime_fingerprint(data_root: Path, split: str, catalog: Path) -> dict:
    directory, official_catalog = runtime_paths(data_root, split)
    expected = hashlib.sha256(official_catalog.read_bytes()).hexdigest()
    if hashlib.sha256(checked_path(catalog).read_bytes()).hexdigest() != expected:
        raise ValueError("Scoring catalog differs from the official catalog used for replay")
    digest = hashlib.sha256()
    for source in sorted((directory / "zaic_tools").rglob("*.py")):
        digest.update(source.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return {"split": split, "catalog_sha256": expected, "zaic_tools_sha256": digest.hexdigest()}


def _replay_one(data_root: Path, split: str, sample_id: str, calls: list) -> dict:
    if (
        Path(sample_id).name != sample_id or "/" in sample_id or "\\" in sample_id
        or sample_id in (".", "..") or ":" in sample_id
    ):
        raise ValueError("Invalid sample id for ToolEnv")
    tool_env = load_tool_env(data_root, split)
    env = tool_env(split, sample_id, root=data_root)  # one session; never reset
    rejected = []
    for index, call in enumerate(calls):
        if not _valid_call(call):
            rejected.append(index)
            continue
        result = env.call(call["name"], call["arguments"])
        if not isinstance(result, dict):
            raise ValueError("Official tool returned an invalid result")
        if result.get("ok") is False:
            rejected.append(index)
    return {"state_hash": env.get_state_hash(), "rejected_call_indices": tuple(rejected)}


@dataclass(frozen=True)
class ReplayBatch:
    records: tuple[ReplayRecord, ...]
    runtime: dict


def verify_submission(
    predictions: list[dict], data_root: str | Path, split: str, catalog: str | Path,
    *, replay_timeout_seconds: float = 30.0,
) -> ReplayBatch:
    """One isolated replay/session per submitted sample, no inference timings.

    The recomputed hash is authoritative even when the claimed hash differs.
    Rejected calls stay in the submitted list and are excluded from matching.
    """
    root = checked_path(data_root).resolve()
    valid_duration(replay_timeout_seconds)
    if replay_timeout_seconds == 0:
        raise ValueError("Replay deadline must be positive")
    fingerprint = runtime_fingerprint(root, split, Path(catalog))
    seen = set()
    for row in predictions:
        _validate_record(row, gold=False)
        if row["id"] in seen:
            raise ValueError(f"Duplicate submission id: {row['id']}")
        seen.add(row["id"])
    records = []
    for row in predictions:
        result = run_isolated(
            _replay_one, (root, split, row["id"], row["tool_calls"]),
            timeout_seconds=replay_timeout_seconds,
        )
        value = result.value if result.status == "ok" else {}
        records.append(ReplayRecord(
            row["id"], result.status, result.elapsed_seconds,
            value.get("state_hash"), value.get("rejected_call_indices", ()), result.error,
        ))
    return ReplayBatch(tuple(records), fingerprint)
