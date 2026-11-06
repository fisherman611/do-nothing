"""Hard process deadlines; a timed-out sample is never retried."""

from dataclasses import dataclass
import multiprocessing as mp
import time
from typing import Any, Callable

from .records import TimingRecord, valid_duration, valid_id


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    elapsed_seconds: float
    value: Any = None
    error: str | None = None


def _worker(connection, function, args, kwargs):
    try:
        result = function(*args, **kwargs)
        connection.send(("ok", result, None))
    except BaseException as error:
        try:
            connection.send(("error", None, f"{type(error).__name__}: {error}"))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


def run_isolated(
    function: Callable, args: tuple = (), kwargs: dict | None = None,
    *, timeout_seconds: float = 30.0,
) -> ExecutionResult:
    """Run a picklable top-level function in a fresh spawn process.

    Wall time includes process startup, callback/model loading and result
    transfer. Models/envs are not inherited from the parent. Use the per-sample
    function contract only; this is not a Docker or network sandbox.
    """
    valid_duration(timeout_seconds)
    if timeout_seconds == 0:
        raise ValueError("timeout_seconds must be positive")
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, function, args, kwargs or {}))
    started = time.perf_counter()
    try:
        process.start()
        sender.close()
        remaining = max(0.0, timeout_seconds - (time.perf_counter() - started))
        if not receiver.poll(remaining):
            return ExecutionResult("timeout", time.perf_counter() - started, error="Sample deadline exceeded")
        status, value, error = receiver.recv()
        elapsed = time.perf_counter() - started
        if elapsed > timeout_seconds:
            return ExecutionResult("timeout", elapsed, error="Sample deadline exceeded")
        return ExecutionResult(status, elapsed, value, error)
    except Exception as error:
        return ExecutionResult("error", time.perf_counter() - started, error=f"{type(error).__name__}: {error}")
    finally:
        sender.close()
        receiver.close()
        if process.pid is not None:
            process.join(timeout=0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()


def run_timed_sample(
    sample_id: str, function: Callable, args: tuple = (), kwargs: dict | None = None,
    *, timeout_seconds: float = 30.0,
) -> tuple[Any, TimingRecord]:
    """Helper for the future inference pipeline: result + separate timing row.

    The callback must create its single ToolEnv inside the child and return its
    submission row. Failure returns None, with no reset or retry of that sample.
    """
    valid_id(sample_id)
    valid_duration(timeout_seconds)
    if timeout_seconds > 30:
        raise ValueError("Inference deadline may not exceed the competition's 30 seconds")
    result = run_isolated(function, args, kwargs, timeout_seconds=timeout_seconds)
    return result.value, TimingRecord(sample_id, result.elapsed_seconds, result.status, result.error)
