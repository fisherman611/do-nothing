"""Strict JSON input and atomic output for metric artifacts."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator


def checked_path(path: str | Path) -> Path:
    path = Path(path)
    if any(part.endswith("_environment") for part in path.resolve().parts):
        raise ValueError("App environment files may only be accessed through ToolEnv")
    return path


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is not allowed: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite JSON number is not allowed: {value}")
    return number


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode(text: str):
    return json.loads(text, parse_constant=_reject_constant, parse_float=_finite_float, object_pairs_hook=_unique_object)


def load_json(path: str | Path) -> Any:
    return _decode(checked_path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path: str | Path) -> list[dict]:
    path = checked_path(path)
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = _decode(line)
                if not isinstance(row, dict):
                    raise ValueError("row must be a JSON object")
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            rows.append(row)
    return rows


def _atomic_write(path: str | Path, chunks: Iterable[str]) -> None:
    path = checked_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for chunk in chunks:
                handle.write(chunk)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: str | Path, value: Any) -> None:
    _atomic_write(path, [json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"])


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    def chunks() -> Iterator[str]:
        for row in rows:
            yield json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"

    _atomic_write(path, chunks())
