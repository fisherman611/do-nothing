import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import wave

from asr.qwen_cloud_label import (
    append_event, load_credentials, parser, prepare_manifest, read_journal, run, transcribe,
)


def dataset(root):
    split = root / "public_test"
    (split / "audio").mkdir(parents=True)
    for name in ("final", "history"):
        with wave.open(str(split / "audio" / f"{name}.wav"), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\0\0" * 160)
    path = split / "public_test.jsonl"
    path.write_text(json.dumps({
        "id": "public_1", "audio": "audio/final.wav",
        "history": [{"role": "user", "audio": "audio/history.wav"},
                    {"role": "assistant", "text": "ignored"}],
    }) + "\n", encoding="utf-8")
    return path


class FakeSocket:
    def __init__(self):
        self.task_id = None
        self.finish = threading.Event()
        self.commands, self.binary = [], []
        self.position = 0

    def settimeout(self, timeout):
        pass

    def send_binary(self, data):
        self.binary.append(data)

    def send(self, text):
        command = json.loads(text)
        self.commands.append(command)
        self.task_id = command["header"]["task_id"]
        if command["header"]["action"] == "finish-task":
            self.finish.set()

    def recv(self):
        events = [
            ("task-started", {}),
            ("result-generated", {"output": {"sentence": {"sentence_id": 1, "sentence_end": False, "text": "partial"}}}),
            ("result-generated", {"output": {"sentence": {"sentence_id": 2, "sentence_end": True, "text": "world"}}}),
            ("result-generated", {"output": {"sentence": {"sentence_id": 1, "sentence_end": True, "text": "wrong"}}}),
            ("result-generated", {"output": {"sentence": {"sentence_id": 1, "sentence_end": True, "text": "hello"}}}),
            ("task-finished", {"usage": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}}),
        ]
        if self.position == len(events) - 1 and not self.finish.wait(2):
            raise TimeoutError("Sender did not finish")
        kind, payload = events[self.position]
        self.position += 1
        return json.dumps({"header": {"event": kind, "task_id": self.task_id}, "payload": payload})

    def recv_data(self, control_frame=False):
        return 1, self.recv().encode("utf-8")

    def close(self, timeout=1):
        pass


class CloudLabelTests(unittest.TestCase):
    def test_manifest_scopes_and_forbidden_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = dataset(root)
            items = prepare_manifest(source)
            self.assertEqual([r["id"] for r in items], ["public_1", "public_1:history:0"])
            self.assertEqual(len(prepare_manifest(source, "final")), 1)
            row = json.loads(source.read_text())
            for unsafe in ("../outside.wav", "public_test_environment/1/state.json", "audio/../../outside.wav"):
                row["audio"] = unsafe
                source.write_text(json.dumps(row), encoding="utf-8")
                with self.assertRaises(ValueError):
                    prepare_manifest(source)

    def test_credentials_use_workspace_host_without_printing_or_modifying_env(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {}, clear=True):
            env = Path(temp) / ".env"
            content = "DASHSCOPE_API_KEY=test-secret\nAPI_HOST=ws-example.ap-southeast-1.maas.aliyuncs.com\n"
            env.write_text(content, encoding="utf-8")
            key, endpoint = load_credentials(env)
            self.assertEqual(key, "test-secret")
            self.assertEqual(endpoint, "wss://ws-example.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/inference")
            self.assertEqual(env.read_text(), content)
            with patch.dict("os.environ", {"API_HOST": "example.com"}):
                with self.assertRaises(ValueError):
                    load_credentials(env)

    def test_full_duplex_final_sentences_replace_intermediate_and_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            path = dataset(Path(temp)).parent / "audio/final.wav"
            socket = FakeSocket()
            with patch("websocket.create_connection", return_value=socket) as connect:
                result = transcribe(path, "test-secret", "wss://example", 3)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["text"], "hello world")
            self.assertTrue(result["usage_final"])
            self.assertEqual(result["usage"]["total_tokens"], 10)
            self.assertEqual(len(b"".join(socket.binary)), 320)
            self.assertEqual([c["header"]["action"] for c in socket.commands], ["run-task", "finish-task"])
            self.assertFalse(socket.commands[0]["payload"]["parameters"]["disfluency_removal_enabled"])
            self.assertEqual(connect.call_args.kwargs["redirect_limit"], 0)

    def test_handshake_error_never_persists_secret(self):
        with tempfile.TemporaryDirectory() as temp:
            path = dataset(Path(temp)).parent / "audio/final.wav"
            with patch("websocket.create_connection", side_effect=RuntimeError("Bearer test-secret")):
                result = transcribe(path, "test-secret", "wss://example")
            self.assertEqual(result["status"], "error")
            self.assertNotIn("test-secret", json.dumps(result))
            self.assertEqual(result["diagnostics"]["phase"], "connecting")
            self.assertEqual(result["diagnostics"]["message"], "[REDACTED]")

    def test_close_frame_records_reason_and_does_not_retry_policy_failure(self):
        class PolicySocket(FakeSocket):
            def recv_data(self, control_frame=False):
                return 8, (1008).to_bytes(2, "big") + b"policy error Bearer test-secret"

        with tempfile.TemporaryDirectory() as temp:
            path = dataset(Path(temp)).parent / "audio/final.wav"
            with patch("websocket.create_connection", return_value=PolicySocket()):
                result = transcribe(path, "test-secret", "wss://example")
            self.assertEqual(result["diagnostics"]["close_code"], 1008)
            self.assertIn("policy error", result["diagnostics"]["close_reason"])
            self.assertFalse(result["retryable"])
            self.assertNotIn("test-secret", json.dumps(result))

    def test_task_deadline_stops_streaming_and_preserves_timeout_status(self):
        import websocket

        class StalledSocket(FakeSocket):
            def recv(self):
                if self.position == 0:
                    self.position += 1
                    return json.dumps({"header": {"task_id": self.task_id, "event": "task-started"}})
                raise websocket.WebSocketTimeoutException()

        with tempfile.TemporaryDirectory() as temp:
            path = dataset(Path(temp)).parent / "audio/final.wav"
            with patch("websocket.create_connection", return_value=StalledSocket()):
                result = transcribe(path, "test-secret", "wss://example", 0.03)
            self.assertEqual(result["status"], "timeout")
            self.assertFalse(result["usage_final"])

    def test_pending_request_keeps_reserve_and_corrupt_journal_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "attempts.jsonl"
            append_event(path, {"event": "started", "attempt_id": "attempt", "reserved_tokens": 100,
                                "item": {"id": "sample"}})
            latest, charged, unknown, count = read_journal(path)
            self.assertEqual((charged, unknown, count), (100, 1, 1))
            self.assertEqual(latest["sample"]["status"], "interrupted")
            with path.open("a") as handle:
                handle.write('{"incomplete":')
            with self.assertRaises(ValueError):
                read_journal(path)

    def test_resume_budget_counts_prior_requests_and_exports_only_successful_refs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = dataset(root), root / "results"
            calls = []

            def recognize(*args):
                calls.append(args[0])
                return {"status": "ok", "text": "xin chào", "usage_final": True,
                        "usage": {"input_tokens": 95, "output_tokens": 5, "total_tokens": 100}}

            base = ["--input", str(source), "--output-dir", str(output),
                    "--reserve-tokens", "200", "--token-budget", "250"]
            with patch("asr.qwen_cloud_label.load_credentials", return_value=("test-secret", "wss://example")), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(run(parser().parse_args(base), recognize), 2)
                self.assertEqual(len(calls), 1)
                self.assertEqual(run(parser().parse_args(base), recognize), 2)
                self.assertEqual(len(calls), 1)
                more = base + ["--token-budget", "500"]
                self.assertEqual(run(parser().parse_args(more), recognize), 0)
                self.assertEqual(len(calls), 2)
                self.assertEqual(run(parser().parse_args(more), recognize), 0)
                self.assertEqual(len(calls), 2)
            self.assertEqual(len((output / "references.jsonl").read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(len((output / "history_references.jsonl").read_text(encoding="utf-8").splitlines()), 1)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["accounted_tokens"], 200)
            self.assertEqual(summary["status_counts"], {"ok": 2})
            self.assertNotIn("test-secret", "".join(p.read_text(encoding="utf-8") for p in output.glob("*.json*")))

    def test_interruption_is_accounted_and_not_automatically_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = dataset(root), root / "results"
            args = parser().parse_args(["--input", str(source), "--output-dir", str(output), "--scope", "final"])
            with patch("asr.qwen_cloud_label.load_credentials", return_value=("test-secret", "wss://example")), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with patch("builtins.print"):
                    self.assertEqual(run(args, lambda *a: (_ for _ in ()).throw(KeyboardInterrupt())), 130)
                self.assertEqual(run(args, lambda *a: self.fail("Must not retry pending request")), 2)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["accounted_tokens"], args.reserve_tokens)
            self.assertEqual(summary["status_counts"], {"interrupted": 1})
            self.assertEqual((output / "references.jsonl").read_text(), "")

    def test_failed_request_stops_without_labeling_remaining_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = dataset(root), root / "results"
            args = parser().parse_args(["--input", str(source), "--output-dir", str(output)])
            calls = []

            def fail(*a):
                calls.append(a[0])
                return {"status": "error", "text": "partial", "usage_final": False,
                        "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}}

            with patch("asr.qwen_cloud_label.load_credentials", return_value=("test-secret", "wss://example")), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(run(args, fail), 1)
            self.assertEqual(len(calls), 1)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["accounted_tokens"], args.reserve_tokens)
            self.assertEqual(summary["unattempted_count"], 1)
            self.assertEqual((output / "references.jsonl").read_text(), "")

    def test_transient_retry_records_each_attempt_and_keeps_unknown_usage_reserve(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = dataset(root), root / "results"
            args = parser().parse_args(["--input", str(source), "--output-dir", str(output),
                                        "--scope", "final", "--retry-delay", "0", "--log-format", "jsonl"])
            calls = []

            def recognize(*a):
                calls.append(a[0])
                if len(calls) == 1:
                    return {"status": "error", "text": "", "usage_final": False,
                            "error": "WebSocketConnectionClosedException", "retryable": True}
                return {"status": "ok", "text": "hello", "usage_final": True,
                        "usage": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}}

            with patch("asr.qwen_cloud_label.load_credentials", return_value=("test-secret", "wss://example")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(args, recognize), 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["attempt_count"], 2)
            self.assertEqual(summary["confirmed_tokens"], 10)
            self.assertEqual(summary["reserved_unknown_tokens"], args.reserve_tokens)
            self.assertEqual(summary["status_counts"], {"ok": 1})

    def test_dry_run_does_not_load_key_call_api_or_write_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = dataset(root), root / "results"
            args = parser().parse_args(["--input", str(source), "--output-dir", str(output), "--dry-run"])
            with patch("asr.qwen_cloud_label.load_credentials", side_effect=AssertionError("No credentials needed")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(args, lambda *a: self.fail("No API calls allowed")), 0)
            self.assertFalse(output.exists())

    def test_retries_stop_at_attempt_or_budget_limit_and_jsonl_stays_parseable(self):
        for budget, expected_calls, expected_exit in [(100000, 2, 1), (10000, 1, 2)]:
            with self.subTest(budget=budget), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, output = dataset(root), root / "results"
                args = parser().parse_args(["--input", str(source), "--output-dir", str(output),
                                            "--scope", "final", "--log-format", "jsonl", "--retry-delay", "0",
                                            "--max-retries", "1", "--token-budget", str(budget)])
                failure = {"status": "error", "text": "", "error": "WebSocketConnectionClosedException",
                           "usage_final": False, "retryable": True}
                capture = io.StringIO()
                calls = []

                def recognize(*a):
                    calls.append(a[0])
                    return failure

                with patch("asr.qwen_cloud_label.load_credentials", return_value=("test-secret", "wss://example")), contextlib.redirect_stdout(capture):
                    self.assertEqual(run(args, recognize), expected_exit)
                self.assertEqual(len(calls), expected_calls)
                for line in capture.getvalue().splitlines():
                    self.assertIsInstance(json.loads(line), dict)
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["accounted_tokens"], expected_calls * args.reserve_tokens)


if __name__ == "__main__":
    unittest.main()
