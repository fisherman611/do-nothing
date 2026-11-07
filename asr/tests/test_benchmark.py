import json
from pathlib import Path
import tempfile
import unittest
import wave

from asr.benchmark import Budget, prepare_dataset, run_phase, select_smoke
from asr.registry import MODELS, WORKER_RATE_USD_SECOND


def dataset(folder, count=3):
    root = folder / "train"
    (root / "audio").mkdir(parents=True)
    rows = []
    for i in range(count):
        path = root / "audio" / f"{i}.wav"
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\0\0" * (1600 + i * 100))
        rows.append({"id": str(i), "audio": f"audio/{i}.wav", "audio_text": "xin chao"})
    (root / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return root


class BenchmarkTests(unittest.TestCase):
    def test_manifest_audio_integrity_and_path_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = dataset(Path(temp))
            _, rows, manifest = prepare_dataset(root)
            self.assertEqual(len(manifest), 3)
            self.assertEqual(len(manifest[0]["sha256"]), 64)
            rows[0]["audio"] = "../outside.wav"
            (root / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_dataset(root)

    def test_smoke_is_stable_unique_and_contains_longest_twenty(self):
        manifest = [{"id": f"{i:04}", "audio_seconds": i + 1} for i in range(200)]
        selected = select_smoke(manifest)
        self.assertEqual(selected, select_smoke(list(reversed(manifest))))
        ids = {r["id"] for r in selected}
        self.assertEqual(len(ids), 100)
        self.assertTrue({f"{i:04}" for i in range(180, 200)} <= ids)

    def test_accounting_adds_concurrent_workers_and_reserves_overhead(self):
        now = [0.0]
        budget = Budget(10, clock=lambda: now[0])
        budget.start("a")
        budget.start("b")
        now[0] = 100
        self.assertAlmostEqual(budget.estimate(), 330 * WORKER_RATE_USD_SECOND)
        budget.stop("a")
        now[0] = 200
        budget.stop("b")
        self.assertAlmostEqual(budget.estimate(), 430 * WORKER_RATE_USD_SECOND)
        self.assertAlmostEqual(budget.available(), 9 - budget.estimate())

    def test_worker_exception_is_scored_and_other_samples_continue(self):
        class Remote:
            def __init__(self, call):
                self.remote = call

            def map(self, payloads, **kwargs):
                for payload in payloads:
                    try:
                        yield self.remote(payload)
                    except Exception as error:
                        yield error

        class Worker:
            describe = Remote(lambda: {"num_parameters": 10})
            transcribe = Remote(lambda payload: ({"id": payload["id"], "text": "xin chao", "status": "ok", "elapsed_seconds": 0.1} if payload["id"] != "1" else (_ for _ in ()).throw(RuntimeError("inference failed"))))

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            root, rows, manifest = prepare_dataset(dataset(folder))
            result = run_phase(Worker(), "fake", "full", manifest, rows, root, folder / "results", Budget(10))
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["summary"]["num_errors"], 1)
            self.assertEqual(result["num_predictions"], 3)
            self.assertAlmostEqual(result["summary"]["normalized"]["wer"], 1 / 3)

    def test_no_inference_when_budget_is_already_exhausted(self):
        class Remote:
            def __init__(self, call):
                self.remote = call

        class Worker:
            describe = Remote(lambda: {})
            transcribe = Remote(lambda payload: self.fail("must not call inference"))

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            root, rows, manifest = prepare_dataset(dataset(folder))
            result = run_phase(Worker(), "fake", "full", manifest, rows, root, folder / "results", Budget(1.01, reserve=1.0))
            self.assertEqual(result["status"], "incomplete_budget")
            self.assertEqual(result["num_predictions"], 0)
            self.assertEqual(result["summary"]["num_missing"], 3)

    def test_checkpoint_locks_predate_cutoff(self):
        self.assertEqual(len(MODELS), 5)
        for spec in MODELS.values():
            self.assertLess(spec["checkpoint_date"], "2026-10-26")
            self.assertEqual(len(spec["revision"]), 40)

    def test_resume_keeps_original_latency_and_only_calls_missing_ids(self):
        called = []

        class Method:
            @staticmethod
            def map(payloads, **kwargs):
                for item in payloads:
                    called.append(item['id'])
                    yield {'id': item['id'], 'text': 'xin chao', 'status': 'ok', 'elapsed_seconds': 0.2}

        class Describe:
            @staticmethod
            def remote():
                return {}

        class Worker:
            describe, transcribe = Describe(), Method()

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            root, rows, manifest = prepare_dataset(dataset(folder))
            source = folder / 'partial.jsonl'
            source.write_text(json.dumps({'id': '0', 'text': 'xin chao', 'status': 'ok', 'elapsed_seconds': 0.123}) + '\n', encoding='utf-8')
            config = run_phase(Worker(), 'fake', 'full', manifest, rows, root, folder / 'results', Budget(10), resume_from=source)
            self.assertEqual(called, ['1', '2'])
            self.assertEqual(config['num_reused_predictions'], 1)
            self.assertEqual(config['num_predictions'], 3)
            self.assertAlmostEqual(config['summary']['latency_seconds']['mean'], (0.123 + 0.2 + 0.2) / 3)

    def test_resume_rejects_duplicate_or_foreign_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            root, rows, manifest = prepare_dataset(dataset(folder))
            source = folder / 'bad.jsonl'
            for index, ids in enumerate((['0', '0'], ['unknown'])):
                source.write_text('\n'.join(json.dumps({'id': i, 'text': '', 'status': 'ok', 'elapsed_seconds': 0}) for i in ids), encoding='utf-8')
                with self.assertRaises(ValueError):
                    run_phase(None, 'fake', 'full', manifest, rows, root, folder / f'results{index}', Budget(10), resume_from=source)


if __name__ == "__main__":
    unittest.main()
