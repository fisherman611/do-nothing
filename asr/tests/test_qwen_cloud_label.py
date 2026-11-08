import contextlib
import io
import json
from pathlib import Path
import tempfile
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

    def setUp(self):
        patcher = patch("asr.qwen_cloud_label.load_state", return_value={
            "contacts": {"contacts": [{"id": "c_1", "name": "Nguyễn Thị Lan", "phone": "0901234567"}]},
            "music": {"tracks": [{"id": "trk_1", "title": "Nơi này có anh", "artist": "Sơn Tùng M-TP"}]},
        })
        self.load_state = patcher.start()
        self.addCleanup(patcher.stop)

    def test_hotwords_from_profile_and_state_and_context_from_earlier_replies(self):
        with tempfile.TemporaryDirectory() as temp:
            source = dataset(Path(temp))
            row = json.loads(source.read_text())
            row["context"] = {"user_profile": {"name": "Anh Thành", "home_address": "77 Hoàng Văn Thụ, Phú Nhuận"}}
            source.write_text(json.dumps(row), encoding="utf-8")
            final, history = prepare_manifest(source)
            self.assertEqual(list(final["vocabulary"]), [
                "Anh Thành", "77 Hoàng Văn Thụ", "Phú Nhuận", "Nguyễn Thị Lan", "Nơi này có anh", "Sơn Tùng M-TP"])
            self.assertEqual(set(final["vocabulary"].values()), {3})
            self.assertEqual(final["context"], ["ignored"])
            self.assertEqual(history["context"], [])
            self.assertEqual(history["vocabulary"], final["vocabulary"])
            self.assertEqual(prepare_manifest(source, hotwords="none", context=False)[0]["vocabulary"], {})
            self.assertEqual(len(prepare_manifest(source, max_hotwords=2)[0]["vocabulary"]), 2)

    def test_credentials_use_workspace_host_without_printing_or_modifying_env(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {}, clear=True):
            env = Path(temp) / ".env"
            content = "DASHSCOPE_API_KEY=test-secret\nAPI_HOST=ws-example.ap-southeast-1.maas.aliyuncs.com\n"
            env.write_text(content, encoding="utf-8")
            key, endpoint = load_credentials(env)
            self.assertEqual(key, "test-secret")
            self.assertEqual(endpoint, "https://ws-example.ap-southeast-1.maas.aliyuncs.com"
                                       "/api/v1/services/aigc/multimodal-generation/generation")
            self.assertEqual(env.read_text(), content)
            with patch.dict("os.environ", {"API_HOST": "example.com"}):
                with self.assertRaises(ValueError):
                    load_credentials(env)

    def test_http_request_carries_audio_hotwords_context_and_parses_usage(self):
        with tempfile.TemporaryDirectory() as temp:
            path = dataset(Path(temp)).parent / "audio/final.wav"
            response = {"request_id": "r1", "output": {"text": " xin chào Lan ", "sentence": {"text": "xin chào Lan"}},
                        "usage": {"duration": 1, "input_tokens": 8, "output_tokens": 2}}
            with patch("asr.qwen_cloud_label._post", return_value=(200, json.dumps(response).encode())) as post:
                result = transcribe(path, "test-secret", "https://example", 3, {"Lan": 3}, ["Bạn muốn gọi ai?"])
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["text"], "xin chào Lan")
            self.assertTrue(result["usage_final"])
            self.assertEqual(result["usage"]["total_tokens"], 10)
            body = json.loads(post.call_args.args[1])
            self.assertEqual(body["model"], "qwen-audio-3.1-asr-flash")
            self.assertEqual(body["parameters"]["vocabulary"], {"Lan": 3})
            messages = body["input"]["messages"]
            self.assertEqual(messages[0], {"role": "assistant", "content": [{"type": "text", "text": "Bạn muốn gọi ai?"}]})
            self.assertTrue(messages[-1]["content"][0]["input_audio"]["data"].startswith("data:audio/wav;base64,"))
            self.assertEqual(post.call_args.args[2]["Authorization"], "Bearer test-secret")

    def test_http_errors_are_sanitized_and_classified(self):
        import urllib.error

        def http_error(status, code):
            body = io.BytesIO(json.dumps({"code": code, "message": "Bearer test-secret bad",
                                          "request_id": "r2"}).encode())
            return urllib.error.HTTPError("https://example", status, "x", {}, body)

        with tempfile.TemporaryDirectory() as temp:
            path = dataset(Path(temp)).parent / "audio/final.wav"
            for status, code, retryable in [(429, "Throttling", True), (400, "InvalidParameter", False),
                                            (503, "ServiceUnavailable", True)]:
                with patch("asr.qwen_cloud_label._post", side_effect=http_error(status, code)):
                    result = transcribe(path, "test-secret", "https://example")
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["retryable"], retryable)
                self.assertEqual(result["diagnostics"]["http_status"], status)
                self.assertNotIn("test-secret", json.dumps(result))
            with patch("asr.qwen_cloud_label._post", side_effect=TimeoutError("timed out")):
                result = transcribe(path, "test-secret", "https://example")
            self.assertEqual((result["status"], result["retryable"]), ("timeout", True))
            with patch("asr.qwen_cloud_label._post", side_effect=RuntimeError("Bearer test-secret")):
                result = transcribe(path, "test-secret", "https://example")
            self.assertEqual(result["diagnostics"]["message"], "[REDACTED]")
            self.assertFalse(result["retryable"])

    def test_concurrent_workers_label_every_audio_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = dataset(root), root / "results"
            args = parser().parse_args(["--input", str(source), "--output-dir", str(output),
                                        "--workers", "4", "--log-format", "jsonl"])
            calls = []

            def recognize(path, *a):
                calls.append(path.name)
                return {"status": "ok", "text": path.stem, "usage_final": True,
                        "usage": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}}

            with patch("asr.qwen_cloud_label.load_credentials", return_value=("test-secret", "https://example")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(args, recognize), 0)
            self.assertEqual(sorted(calls), ["final.wav", "history.wav"])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual((summary["confirmed_tokens"], summary["reserved_unknown_tokens"]), (20, 0))
            journal = (output / "attempts.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("vocabulary", journal)

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
            with patch("asr.qwen_cloud_label.load_credentials", return_value=("test-secret", "https://example")), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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
            with patch("asr.qwen_cloud_label.load_credentials", return_value=("test-secret", "https://example")), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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
            args = parser().parse_args(["--input", str(source), "--output-dir", str(output), "--workers", "1"])
            calls = []

            def fail(*a):
                calls.append(a[0])
                return {"status": "error", "text": "partial", "usage_final": False,
                        "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}}

            with patch("asr.qwen_cloud_label.load_credentials", return_value=("test-secret", "https://example")), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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

            with patch("asr.qwen_cloud_label.load_credentials", return_value=("test-secret", "https://example")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(args, recognize), 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["attempt_count"], 2)
            self.assertEqual(summary["confirmed_tokens"], 10)
            self.assertEqual(summary["reserved_unknown_tokens"], args.reserve_tokens)
            self.assertEqual(summary["status_counts"], {"ok": 1})

    def test_success_missing_final_usage_preserves_reserve_and_continues(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = dataset(root), root / "results"
            args = parser().parse_args(["--input", str(source), "--output-dir", str(output),
                                        "--log-format", "jsonl", "--reserve-tokens", "200"])
            calls = []

            def recognize(*a):
                calls.append(a[0])
                usage = {"input_tokens": 80, "output_tokens": 20, "total_tokens": 100}
                return {"status": "ok", "text": "xin chào", "usage": usage,
                        "usage_final": len(calls) > 1}

            with patch("asr.qwen_cloud_label.load_credentials", return_value=("test-secret", "https://example")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run(args, recognize), 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(calls), 2)
            self.assertEqual(summary["status_counts"], {"ok": 2})
            self.assertEqual(summary["confirmed_tokens"], 100)
            self.assertEqual(summary["reserved_unknown_tokens"], 200)

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
                                            "--max-retries", "1", "--token-budget", str(budget),
                                            "--reserve-tokens", "10000"])
                failure = {"status": "error", "text": "", "error": "WebSocketConnectionClosedException",
                           "usage_final": False, "retryable": True}
                capture = io.StringIO()
                calls = []

                def recognize(*a):
                    calls.append(a[0])
                    return failure

                with patch("asr.qwen_cloud_label.load_credentials", return_value=("test-secret", "https://example")), contextlib.redirect_stdout(capture):
                    self.assertEqual(run(args, recognize), expected_exit)
                self.assertEqual(len(calls), expected_calls)
                for line in capture.getvalue().splitlines():
                    self.assertIsInstance(json.loads(line), dict)
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["accounted_tokens"], expected_calls * args.reserve_tokens)


if __name__ == "__main__":
    unittest.main()
