"""Execution evidence kept separately from official submission JSONL."""

from dataclasses import dataclass
import math


def valid_id(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("Sample id must be a non-empty string")


def valid_duration(value: float) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("elapsed_seconds must be a finite, non-negative number")


@dataclass(frozen=True)
class TimingRecord:
    id: str
    elapsed_seconds: float
    status: str = "ok"
    error: str | None = None

    def __post_init__(self):
        valid_id(self.id)
        valid_duration(self.elapsed_seconds)
        if self.status not in ("ok", "error", "timeout"):
            raise ValueError("Timing status must be ok, error or timeout")
        if self.error is not None and not isinstance(self.error, str):
            raise ValueError("Timing error must be a string or null")

    @classmethod
    def from_dict(cls, value: dict):
        if not isinstance(value, dict) or not {"id", "elapsed_seconds", "status"} <= value.keys():
            raise ValueError("Timing row needs id, elapsed_seconds and status")
        return cls(value["id"], value["elapsed_seconds"], value["status"], value.get("error"))


@dataclass(frozen=True)
class ReplayRecord:
    id: str
    status: str
    elapsed_seconds: float
    state_hash: str | None = None
    rejected_call_indices: tuple[int, ...] = ()
    error: str | None = None

    def __post_init__(self):
        valid_id(self.id)
        valid_duration(self.elapsed_seconds)
        if self.status not in ("ok", "error", "timeout"):
            raise ValueError("Replay status must be ok, error or timeout")
        if self.status == "ok" and (not isinstance(self.state_hash, str) or not self.state_hash):
            raise ValueError("Successful replay needs a state_hash from ToolEnv")
        if any(type(i) is not int or i < 0 for i in self.rejected_call_indices):
            raise ValueError("Rejected call indices must be non-negative integers")
        if self.error is not None and not isinstance(self.error, str):
            raise ValueError("Replay error must be a string or null")
