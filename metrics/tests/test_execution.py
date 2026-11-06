"""Runtime evidence, authoritative replay, ranking and five-run scoring."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from metrics import (
    ReplayRecord, TimingRecord, ToolCatalog, aggregate_five_runs, evaluate,
    rank_reports, run_timed_sample, score_sample,
)
from metrics.io import load_json, read_jsonl, write_json, write_jsonl
from metrics.replay import verify_submission


CATALOG = {"apps": [{"id": "demo", "tools": [
    {"name": "demo.set", "parameters": {"properties": {"value": {"type": "integer"}}}},
    {"name": "demo.read", "parameters": {"properties": {}}},
]}]}


def call(value):
    return {"name": "demo.set", "arguments": {"value": value}}


def row(calls=None, state_hash="2"):
    return {"id": "s1", "tool_calls": [call(1), call(2)] if calls is None else calls, "state_hash": state_hash}


def fast_callback():
    return row()


def failing_callback():
    raise RuntimeError("inference failed")


def slow_callback():
    time.sleep(60)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.catalog = ToolCatalog.from_dict(CATALOG)

    def test_official_percentage_scale(self):
        report = evaluate([row()], [row()], self.catalog)
        self.assertEqual(report.summary["score"], 100)
        self.assertEqual(report.summary["mean_sample_score"], 1)
        self.assertEqual(report.summary["tie_break_score"], 100)

    def test_deadline_boundary_and_execution_failure(self):
        for status, elapsed, score in [("ok", 30, 1), ("ok", 30.000001, 0), ("error", 1, 0), ("timeout", 1, 0)]:
            with self.subTest(status=status, elapsed=elapsed):
                result = score_sample(row(), row(), self.catalog, timing=TimingRecord("s1", elapsed, status))
                self.assertEqual(result.score, score)
                self.assertEqual(result.tie_break_f1, score)
        empty = score_sample(row([]), row([]), self.catalog, timing=TimingRecord("s1", 31))
        self.assertEqual(empty.score, 0)

    def test_timing_coverage_is_mandatory_when_supplied(self):
        for timings in [[], [TimingRecord("other", 1)], [TimingRecord("s1", 1), TimingRecord("s1", 2)]]:
            with self.subTest(timings=timings), self.assertRaises(ValueError):
                evaluate([row()], [row()], self.catalog, timings=timings)

    def test_invalid_times_and_statuses_are_rejected(self):
        for value in [-1, float("nan"), float("inf"), True, "1"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                TimingRecord("s1", value)
        with self.assertRaises(ValueError):
            TimingRecord("s1", 1, "unknown")

    def test_replay_hash_overrides_claim_in_both_directions(self):
        bad_replay = ReplayRecord("s1", "ok", 0.1, "wrong")
        good_replay = ReplayRecord("s1", "ok", 0.1, "2")
        result = score_sample(row(), row(), self.catalog, replay=bad_replay)
        self.assertEqual(result.score, 0)
        self.assertFalse(result.hash_consistent)
        result = score_sample(row(state_hash="fake"), row(), self.catalog, replay=good_replay)
        self.assertEqual(result.score, 1)
        self.assertFalse(result.hash_consistent)

    def test_rejected_call_cannot_match_even_if_arguments_are_equal(self):
        replay = ReplayRecord("s1", "ok", 0.1, "2", (0,))
        result = score_sample(row(), row(), self.catalog, replay=replay)
        self.assertEqual(result.calls.matched_calls, 1)
        self.assertEqual(result.calls.unmatched_submitted_calls, 1)
        self.assertEqual(result.score, 0.5)

    def test_failed_replay_does_not_fall_back_to_claimed_hash(self):
        for status in ("error", "timeout"):
            result = score_sample(row(), row(), self.catalog, replay=ReplayRecord("s1", status, 0.2, error="verification failed"))
            self.assertEqual(result.score, 0)
            self.assertIsNone(result.effective_state_hash)
            self.assertEqual(result.tie_break_f1, 0)
            self.assertEqual(result.replay_error, "verification failed")
        with self.assertRaises(ValueError):
            evaluate([row()], [row()], self.catalog, replays=[])

    def test_time_statistics_use_inference_not_replay_time(self):
        result = evaluate([row()], [row()], self.catalog,
            timings=[TimingRecord("s1", 12)], replays=[ReplayRecord("s1", "ok", 0.01, "2")])
        self.assertEqual(result.summary["average_processing_seconds"], 12)
        self.assertEqual(result.samples[0].replay_elapsed_seconds, 0.01)

    def test_hard_deadline_stops_worker_and_reports_failure(self):
        started = time.perf_counter()
        value, timing = run_timed_sample("s1", slow_callback, timeout_seconds=0.3)
        self.assertIsNone(value)
        self.assertEqual(timing.status, "timeout")
        self.assertLess(time.perf_counter() - started, 3)
        self.assertEqual(score_sample(row(), row(), self.catalog, timing=timing).score, 0)

    def test_measured_callback_success_and_error(self):
        value, timing = run_timed_sample("s1", fast_callback)
        self.assertEqual(value, row())
        self.assertEqual(timing.status, "ok")
        value, timing = run_timed_sample("s1", failing_callback)
        self.assertIsNone(value)
        self.assertEqual(timing.status, "error")
        self.assertIn("inference failed", timing.error)
        with self.assertRaises(ValueError):
            run_timed_sample("s1", fast_callback, timeout_seconds=31)


class RankingTests(unittest.TestCase):
    def report(self, prediction=None, duration=None):
        return evaluate([prediction or row()], [row()], ToolCatalog.from_dict(CATALOG),
            timings=[TimingRecord("s1", duration)] if duration is not None else None).to_dict()

    def test_all_three_ranking_levels(self):
        perfect = self.report(duration=20)
        fast = self.report(row(state_hash="wrong"), 1)
        slow = self.report(row(state_hash="wrong"), 2)
        weak = self.report(row([], "wrong"), 0.1)
        ranking = rank_reports([("weak", weak), ("slow", slow), ("fast", fast), ("perfect", perfect)])
        self.assertEqual([r["name"] for r in ranking], ["perfect", "fast", "slow", "weak"])
        self.assertEqual([r["rank"] for r in ranking], [1, 2, 3, 4])

    def test_exact_ties_share_rank_and_missing_times_are_not_invented(self):
        ranking = rank_reports([("b", self.report(duration=2)), ("a", self.report(duration=2))])
        self.assertEqual([r["rank"] for r in ranking], [1, 1])
        with self.assertRaisesRegex(ValueError, "timings"):
            rank_reports([("a", self.report()), ("b", self.report())])

    def test_five_run_average_and_runtime(self):
        reports = [self.report(row(state_hash="2" if i < 2 else "wrong"), i + 1) for i in range(5)]
        result = aggregate_five_runs(reports)
        self.assertEqual(result["summary"]["score"], 40)
        self.assertEqual(result["summary"]["tie_break_score"], 100)
        self.assertEqual(result["summary"]["average_processing_seconds"], 3)
        self.assertEqual(result["samples"][0]["score"], 0.4)
        with self.assertRaises(ValueError):
            aggregate_five_runs(reports[:4])
        changed = self.report(duration=1)
        changed["summary"]["gold_fingerprint"] = "different"
        with self.assertRaises(ValueError):
            aggregate_five_runs(reports[:4] + [changed])

    def test_cli_average_and_rank(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            runs = []
            for i in range(5):
                path = folder / f"run{i}.json"
                write_json(path, self.report(duration=i + 1))
                runs.append(str(path))
            average = folder / "average.json"
            command = [sys.executable, "-m", "metrics", "average-runs", "--reports", *runs, "--output", str(average)]
            result = subprocess.run(command, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(load_json(average)["summary"]["score"], 100)
            second = load_json(average)
            second["summary"]["average_processing_seconds"] = 4
            slower = folder / "slower.json"
            write_json(slower, second)
            output = folder / "ranking.json"
            result = subprocess.run([
                sys.executable, "-m", "metrics", "rank", "--entry", f"fast={average}",
                "--entry", f"slow={slower}", "--output", str(output),
            ], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(load_json(output)[0]["name"], "fast")


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        package = self.root / "train" / "zaic_tools"
        package.mkdir(parents=True)
        # Synthetic runtime fixture: no app environment files exist or are read.
        (package / "__init__.py").write_text('''
_created = 0
class Env:
    def __init__(self): self.value = 0
    def call(self, name, arguments):
        if name == "demo.set" and type(arguments.get("value")) is int:
            self.value = arguments["value"]
            return {"ok": True}
        if name == "demo.read": return {"ok": True}
        return {"ok": False, "error": "schema rejected"}
    def get_state_hash(self): return str(self.value)
def ToolEnv(split, sample_id, root):
    global _created
    _created += 1
    if _created != 1: raise RuntimeError("environment was recreated")
    return Env()
''', encoding="utf-8")
        self.catalog_path = self.root / "train" / "train_tools_catalog.json"
        write_json(self.catalog_path, CATALOG)
        self.catalog = ToolCatalog.from_dict(CATALOG)

    def tearDown(self):
        self.temp.cleanup()

    def test_actual_replay_order_is_used_despite_orderless_call_matching(self):
        predicted = row([call(2), call(1)])
        batch = verify_submission([predicted], self.root, "train", self.catalog_path)
        self.assertEqual(batch.records[0].status, "ok", batch.records[0].error)
        self.assertEqual(batch.records[0].state_hash, "1")
        report = evaluate([predicted], [row()], self.catalog, replays=batch.records)
        self.assertEqual(report.summary["score"], 0)
        self.assertEqual(report.summary["tie_break_score"], 100)
        self.assertEqual(report.summary["num_hash_discrepancies"], 1)

    def test_rejected_calls_continue_replay_and_count_as_unmatched(self):
        predicted = row([call(1), call(True), None, call(2)])
        batch = verify_submission([predicted], self.root, "train", self.catalog_path)
        self.assertEqual(batch.records[0].rejected_call_indices, (1, 2))
        report = evaluate([predicted], [row()], self.catalog, replays=batch.records)
        self.assertAlmostEqual(report.summary["score"], 100 * 4 / 6)
        self.assertEqual(report.samples[0].calls.unmatched_submitted_calls, 2)

    def test_wrong_catalog_rejected_before_replay(self):
        path = self.root / "wrong.json"
        write_json(path, {"apps": []})
        with self.assertRaisesRegex(ValueError, "catalog differs"):
            verify_submission([row()], self.root, "train", path)

    def test_cli_combines_replay_and_actual_timing_sidecar(self):
        gold, submitted, timings, output = [self.root / n for n in ("gold.jsonl", "prediction.jsonl", "timings.jsonl", "score.json")]
        write_jsonl(gold, [row()])
        write_jsonl(submitted, [row()])
        write_jsonl(timings, [asdict(TimingRecord("s1", 31))])
        result = subprocess.run([
            sys.executable, "-m", "metrics", "score", "--gold", str(gold),
            "--submission", str(submitted), "--catalog", str(self.catalog_path),
            "--data-root", str(self.root), "--replay", "--timings", str(timings), "--output", str(output),
        ], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = load_json(output)
        self.assertEqual(report["summary"]["score"], 0)
        self.assertEqual(report["summary"]["num_timeout_samples"], 1)
        self.assertEqual(report["summary"]["average_processing_seconds"], 31)
        self.assertIn("zaic_tools_sha256", report["verification"])


if __name__ == "__main__":
    unittest.main()
