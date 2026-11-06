import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from metrics.asr import evaluate_asr, error_counts, normalize_text
from metrics.io import write_jsonl


class AsrTests(unittest.TestCase):
    def test_normalization_preserves_vietnamese_and_semantic_numbers(self):
        self.assertEqual(normalize_text('  CA\u0300 PHE\u0302!  '), 'cà phê')
        self.assertEqual(normalize_text('Mai 09:30, chuyển 1.500.000đ, 5%.'), 'mai 09:30 chuyển 1.500.000đ 5%')
        self.assertNotEqual(normalize_text('1.5'), normalize_text('15'))
        self.assertNotEqual(normalize_text('không bật'), normalize_text('bật'))

    def test_normalization_preserves_negative_numbers(self):
        self.assertEqual(normalize_text('Nhiệt độ -5, giảm 2-3 độ.'), 'nhiệt độ -5 giảm 2-3 độ')
        self.assertNotEqual(normalize_text('-5'), normalize_text('5'))

    def test_edit_counts_include_insertions_deletions_and_duplicates(self):
        self.assertEqual(error_counts('a a', 'a').deletions, 1)
        self.assertEqual(error_counts('a b', 'a c').substitutions, 1)
        self.assertEqual(error_counts('a', 'a b').insertions, 1)
        self.assertEqual(error_counts('', '').rate, 0)
        self.assertEqual(error_counts('', 'a b').rate, 2)

    def test_corpus_micro_average_not_mean_of_utterance_rates(self):
        refs = [{'id': 'a', 'audio_text': 'one'}, {'id': 'b', 'audio_text': 'one two three'}]
        predictions = [{'id': 'a', 'text': ''}, {'id': 'b', 'text': 'one two three'}]
        report = evaluate_asr(predictions, refs)
        self.assertEqual(report['summary']['normalized']['wer'], 0.25)
        self.assertEqual(report['summary']['normalized']['word_counts']['reference_length'], 4)

    def test_raw_and_normalized_are_reported_separately(self):
        report = evaluate_asr([{'id': 'a', 'text': 'xin chào'}], [{'id': 'a', 'audio_text': 'Xin chào!'}])
        self.assertEqual(report['summary']['normalized']['wer'], 0)
        self.assertEqual(report['summary']['raw']['wer'], 1)
        self.assertEqual(report['summary']['normalized']['exact_match_rate'], 1)

    def test_missing_error_timeout_not_dropped(self):
        refs = [{'id': name, 'audio_text': 'xin chào'} for name in ('a', 'b', 'c', 'd')]
        predictions = [
            {'id': 'a', 'text': 'xin chào', 'status': 'ok'},
            {'id': 'b', 'text': 'xin chào', 'status': 'error'},
            {'id': 'c', 'text': 'xin chào', 'elapsed_seconds': 30.1},
        ]
        report = evaluate_asr(predictions, refs)
        self.assertEqual(report['summary']['normalized']['wer'], 0.75)
        self.assertEqual(report['summary']['num_errors'], 1)
        self.assertEqual(report['summary']['num_timeouts'], 1)
        self.assertEqual(report['summary']['num_missing'], 1)

    def test_spaces_are_included_in_cer(self):
        self.assertEqual(error_counts('a b', 'ab', characters=True).reference_length, 3)
        self.assertAlmostEqual(error_counts('a b', 'ab', characters=True).rate, 1 / 3)

    def test_latency_and_rtf(self):
        refs = [{'id': 'a', 'audio_text': 'a'}, {'id': 'b', 'audio_text': 'b'}]
        preds = [{'id': 'a', 'text': 'a', 'elapsed_seconds': 1, 'audio_seconds': 2},
                 {'id': 'b', 'text': 'b', 'elapsed_seconds': 3, 'audio_seconds': 6}]
        result = evaluate_asr(preds, refs)['summary']
        self.assertEqual(result['real_time_factor'], 0.5)
        self.assertEqual(result['latency_seconds']['mean'], 2)
        self.assertEqual(result['latency_seconds']['p50'], 2)

    def test_invalid_ids_transcripts_and_times(self):
        gold = [{'id': 'a', 'audio_text': 'a'}]
        cases = [
            [{'id': 'a', 'text': 'a'}, {'id': 'a', 'text': 'a'}],
            [{'id': 'extra', 'text': 'a'}], [{'id': 'a', 'text': None}],
            [{'id': 'a', 'text': 'a', 'elapsed_seconds': float('nan')}],
        ]
        for predictions in cases:
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                evaluate_asr(predictions, gold)
        with self.assertRaises(ValueError):
            evaluate_asr([], gold, require_all=True)
        with self.assertRaises(ValueError):
            evaluate_asr([], [{'id': 'a'}])

    def test_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            gold, predicted, output, worst = [directory / name for name in ('gold.jsonl', 'asr.jsonl', 'metrics.json', 'worst.json')]
            write_jsonl(gold, [{'id': 'a', 'audio_text': 'Xin chào!'}])
            write_jsonl(predicted, [{'id': 'a', 'text': 'xin chào'}])
            result = subprocess.run([
                sys.executable, '-m', 'metrics', 'asr', '--references', str(gold),
                '--predictions', str(predicted), '--output', str(output), '--worst-output', str(worst),
            ], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text(encoding='utf-8'))['summary']['normalized']['wer_percent'], 0)
            self.assertEqual(len(json.loads(worst.read_text(encoding='utf-8'))), 1)


if __name__ == '__main__':
    unittest.main()
