"""Boundary cases that change leaderboard scores, using synthetic data only."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from metrics import ToolCatalog, evaluate, match_tool_calls, score_sample, token_f1
from metrics.__main__ import _gold_rows
from metrics.io import read_jsonl, write_jsonl


CATALOG = {
    "apps": [{"id": "demo", "tools": [
        {"name": "demo.read", "parameters": {"type": "object", "properties": {}}},
        {"name": "demo.send", "parameters": {"type": "object", "properties": {
            "to": {"type": "string"}, "text": {"type": "string", "x-match": "fuzzy"},
        }}},
        {"name": "demo.set", "parameters": {"type": "object", "properties": {
            "value": {}, "nested": {"type": "object", "properties": {
                "text": {"type": "string", "x-match": "fuzzy"},
            }},
        }}},
    ]}]
}


def call(name="demo.read", **arguments):
    return {"name": name, "arguments": arguments}


def row(sample_id="s1", calls=None, state_hash="same"):
    return {"id": sample_id, "tool_calls": [] if calls is None else calls, "state_hash": state_hash}


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.catalog = ToolCatalog.from_dict(CATALOG)

    def test_unicode_normalization_and_multiset_overlap(self):
        self.assertEqual(token_f1("CA\u0300 PHE\u0302", "cà phê"), 1.0)
        self.assertAlmostEqual(token_f1("a a", "a"), 2 / 3)
        self.assertEqual(token_f1("xin chào!", "xin chào"), 0.5)
        self.assertEqual(token_f1("", "  "), 1.0)
        self.assertEqual(token_f1("", "a"), 0.0)

    def test_fuzzy_threshold_is_strict(self):
        gold = call("demo.send", to="c1", text="a b c y z w")
        predicted = call("demo.send", to="c1", text="a b c x")
        self.assertEqual(token_f1(predicted["arguments"]["text"], gold["arguments"]["text"]), 0.6)
        self.assertFalse(self.catalog.calls_match(predicted, gold))
        self.assertTrue(self.catalog.calls_match(call("demo.send", to="c1", text="a b c"), gold))

    def test_maximum_matching_reassigns_greedy_pair(self):
        gold = [call("demo.send", text="a b c"), call("demo.send", text="a b d")]
        predicted = [call("demo.send", text="a b"), call("demo.send", text="b c")]
        for order in (predicted, list(reversed(predicted))):
            metrics = match_tool_calls(order, gold, self.catalog)
            self.assertEqual(metrics.matched_calls, 2)
            self.assertEqual(metrics.tool_f1, 1.0)
            self.assertEqual(len({g for _, g in metrics.matched_pairs}), 2)

    def test_duplicate_and_rejected_calls_reduce_precision(self):
        predicted = [call(), call(), call("unknown.tool"), None, {"name": "demo.read"}]
        metrics = match_tool_calls(predicted, [call()], self.catalog)
        self.assertEqual(metrics.matched_calls, 1)
        self.assertEqual(metrics.unmatched_submitted_calls, 4)
        self.assertEqual(metrics.precision, 0.2)
        self.assertEqual(metrics.recall, 1.0)
        self.assertAlmostEqual(metrics.tool_f1, 1 / 3)

    def test_non_fuzzy_arguments_are_exact(self):
        pairs = [
            (call("demo.send", to="C1", text="a"), call("demo.send", to="c1", text="a")),
            (call("demo.set", value=True), call("demo.set", value=1)),
            (call("demo.set", value="1"), call("demo.set", value=1)),
            (call("demo.set", value=None), call("demo.set")),
            (call("demo.set", value=[2, 1]), call("demo.set", value=[1, 2])),
            (call("demo.read", extra=1), call("demo.read")),
        ]
        for prediction, gold in pairs:
            with self.subTest(prediction=prediction):
                self.assertFalse(self.catalog.calls_match(prediction, gold))
        self.assertTrue(self.catalog.calls_match(call("demo.set", value=1.0), call("demo.set", value=1)))

    def test_nested_fuzzy_fields_follow_schema(self):
        self.assertTrue(self.catalog.calls_match(
            call("demo.set", nested={"text": "XIN CHÀO"}),
            call("demo.set", nested={"text": "xin chào"}),
        ))

    def test_hash_mismatch_gates_score_but_not_tiebreak(self):
        result = score_sample(row(calls=[call()], state_hash="wrong"), row(calls=[call()]), self.catalog)
        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.state_match)
        self.assertEqual(result.calls.tool_f1, 1.0)

    def test_empty_gold_and_missing_submission(self):
        cases = [
            (row(), 1.0),
            (row(calls=[call()]), 0.0),
            (row(state_hash="wrong"), 0.0),
            (None, 0.0),
        ]
        for prediction, expected in cases:
            with self.subTest(prediction=prediction):
                self.assertEqual(score_sample(prediction, row(), self.catalog).score, expected)
        self.assertEqual(score_sample(None, row(), self.catalog).calls.tool_f1, 0.0)

    def test_no_prediction_for_nonempty_gold(self):
        result = score_sample(row(), row(calls=[call()]), self.catalog)
        self.assertEqual(result.calls.precision, 0.0)
        self.assertEqual(result.calls.recall, 0.0)
        self.assertEqual(result.score, 0.0)

    def test_call_order_does_not_matter(self):
        calls = [call("demo.set", value=1), call()]
        result = score_sample(row(calls=list(reversed(calls))), row(calls=calls), self.catalog)
        self.assertEqual(result.score, 1.0)

    def test_macro_average_includes_missing_ids(self):
        gold = [row("a", [call()]), row("b", [call(), call()]), row("c")]
        submitted = [row("b", [call(), call()], "wrong"), row("a", [call()])]
        report = evaluate(submitted, gold, self.catalog)
        self.assertAlmostEqual(report.summary["score"], 100 / 3)
        self.assertAlmostEqual(report.summary["mean_sample_score"], 1 / 3)
        self.assertAlmostEqual(report.summary["tool_call_f1"], 2 / 3)
        self.assertEqual(report.summary["num_missing_samples"], 1)
        self.assertEqual([s.id for s in report.samples], ["a", "b", "c"])

    def test_invalid_record_sets_fail_instead_of_inflating_score(self):
        cases = [
            ([row(), row()], [row()]),
            ([row()], [row(), row()]),
            ([row("extra")], [row()]),
            ([], []),
            ([{"id": "s1", "tool_calls": []}], [row()]),
            ([row()], [{"id": "s1", "label": {"tool_calls": []}}]),
        ]
        for predicted, gold in cases:
            with self.subTest(predicted=predicted, gold=gold), self.assertRaises(ValueError):
                evaluate(predicted, gold, self.catalog)
        with self.assertRaises(ValueError):
            evaluate([], [row()], self.catalog, require_all=True)

    def test_wrong_catalog_rejects_gold(self):
        with self.assertRaises(ValueError):
            match_tool_calls([], [call("wrong.tool")], self.catalog)


class InputAndCliTests(unittest.TestCase):
    def test_jsonl_bom_blank_lines_and_line_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rows.jsonl"
            source.write_text('\ufeff{"id":"s1"}\n\n{"x":NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"rows.jsonl:3:"):
                read_jsonl(source)
            source.write_text('\ufeff{"id":"s1"}\n\n', encoding="utf-8")
            self.assertEqual(read_jsonl(source), [{"id": "s1"}])

    def test_environment_file_access_is_blocked(self):
        with self.assertRaisesRegex(ValueError, "ToolEnv"):
            read_jsonl(Path("train_environment") / "s1" / "state.json")

    def test_ambiguous_json_and_overflow_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.jsonl"
            for text in ('{"x": 1e999}', '{"id":"a", "id":"b"}'):
                with self.subTest(text=text):
                    source.write_text(text, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_jsonl(source)

    def test_cli_writes_summary_and_sample_report(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            gold, submitted, catalog, output = [directory / n for n in (
                "gold.jsonl", "submission.jsonl", "catalog.json", "report.json",
            )]
            write_jsonl(gold, [row("a", [call()]), row("b")])
            write_jsonl(submitted, [row("a", [call()])])
            catalog.write_text(json.dumps(CATALOG), encoding="utf-8")
            completed = subprocess.run([
                sys.executable, "-m", "metrics", "score", "--gold", str(gold),
                "--submission", str(submitted), "--catalog", str(catalog), "--output", str(output),
            ], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["score"], 50.0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["samples"][1]["missing_prediction"])

    def test_gold_replay_uses_one_session_and_keeps_calls(self):
        sessions = []

        class FakeEnv:
            def __init__(self, split, sample_id, root):
                sessions.append((split, sample_id))
                self.calls = []

            def call(self, name, arguments):
                self.calls.append({"name": name, "arguments": arguments})
                return {"ok": True}

            def get_tool_calls(self):
                return self.calls

            def get_state_hash(self):
                return "from-official-env"

        rows = [{"id": "a", "label": {"tool_calls": [call()]}}, {"id": "b", "label": {"tool_calls": []}}]
        result = list(_gold_rows(rows, FakeEnv, Path(".")))
        self.assertEqual(sessions, [("train", "a"), ("train", "b")])
        self.assertEqual(result[0]["tool_calls"], [call()])
        self.assertEqual(result[0]["state_hash"], "from-official-env")

    def test_atomic_write_keeps_previous_artifact_after_failure(self):
        def broken_rows():
            yield row()
            raise ValueError("replay failed")

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "gold.jsonl"
            target.write_text("previous artifact", encoding="utf-8")
            with self.assertRaises(ValueError):
                write_jsonl(target, broken_rows())
            self.assertEqual(target.read_text(encoding="utf-8"), "previous artifact")
            self.assertEqual(list(Path(directory).iterdir()), [target])


if __name__ == "__main__":
    unittest.main()
