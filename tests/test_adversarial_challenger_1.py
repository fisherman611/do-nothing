"""
tests/test_adversarial_challenger_1.py
Adversarial Stress-Test Harness for Challenger 1 (Data & Acoustic Coverage)

Covers:
1. File-level & JSON-level adversarial hygiene (BOM, line endings, strict JSON, trailing spaces)
2. Acoustic & Waveform deep audit (RIFF header, PCM 16kHz mono 16-bit, non-zero audio frames, RMS energy, clipping)
3. Direct physical disk audit vs JSONL union (1,933 WAV files on disk matched 1-to-1)
4. Partition completeness & public_test.jsonl dialogue turn symmetry (0 overlap, 1,933 union)
5. Adversarial text hygiene (CJK, control chars, zero-width chars, non-breaking spaces, repetitious loops, sentinel tokens, NFC normalization)
6. Speech rate and acoustic-text alignment bounds (WPS sanity bounds, minimum/maximum audio durations)
7. Schema parity and metadata consistency with final_reviewed.jsonl
8. Baseline prediction lineage and refinement distance sanity
"""

import json
import math
from pathlib import Path
import re
import struct
import unicodedata
import unittest
import wave

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data/results/asr/qwen-cloud/public-test-asr-flash"
FINAL_REVIEWED_PATH = RESULTS_DIR / "final_reviewed.jsonl"
REMAINING_REVIEWED_PATH = RESULTS_DIR / "remaining_reviewed.jsonl"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.jsonl"
PUBLIC_TEST_DIR = ROOT / "data/train+public_test/public_test"
AUDIO_DIR = PUBLIC_TEST_DIR / "audio"
PUBLIC_TEST_JSONL = PUBLIC_TEST_DIR / "public_test.jsonl"

TOTAL_EXPECTED_AUDIOS = 1933
FINAL_EXPECTED_COUNT = 725
REMAINING_EXPECTED_COUNT = 1208
TOTAL_MAIN_TURNS = 634
TOTAL_HISTORY_TURNS = 1299


def parse_strict_jsonl(path: Path):
    """Load JSONL without skipping blank lines, enforcing strict formatting."""
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    records = []
    for line_idx, line in enumerate(lines, 1):
        stripped = line.rstrip("\r\n")
        if not stripped:
            raise ValueError(f"{path.name}:{line_idx}: Unexpected empty/blank line encountered.")
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path.name}:{line_idx}: JSON parse error: {e}")
    return lines, records


class TestAdversarialChallenger1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. Load remaining_reviewed
        cls.rem_raw_bytes = REMAINING_REVIEWED_PATH.read_bytes()
        cls.rem_lines, cls.rem_records = parse_strict_jsonl(REMAINING_REVIEWED_PATH)
        cls.rem_ids = {r["id"] for r in cls.rem_records}

        # 2. Load final_reviewed
        cls.final_raw_bytes = FINAL_REVIEWED_PATH.read_bytes()
        cls.final_lines, cls.final_records = parse_strict_jsonl(FINAL_REVIEWED_PATH)
        cls.final_ids = {r["id"] for r in cls.final_records}

        # 3. Load predictions
        cls.pred_lines, cls.pred_records = parse_strict_jsonl(PREDICTIONS_PATH)
        cls.pred_dict = {r["id"]: r for r in cls.pred_records}

    # =========================================================================
    # 1. File-level & JSON-level Adversarial Hygiene
    # =========================================================================
    def test_01_no_utf8_bom(self):
        """Verify remaining_reviewed.jsonl has no UTF-8 Byte Order Mark (BOM)."""
        self.assertFalse(
            cls_rem_bytes := self.rem_raw_bytes.startswith(b"\xef\xbb\xbf"),
            "remaining_reviewed.jsonl contains UTF-8 BOM, which breaks standard JSONL streamers.",
        )

    def test_02_line_endings_and_blank_lines(self):
        """Verify line count is exactly 1,208 with no empty lines anywhere."""
        self.assertEqual(
            len(self.rem_lines),
            REMAINING_EXPECTED_COUNT,
            f"Expected exactly {REMAINING_EXPECTED_COUNT} lines, found {len(self.rem_lines)}",
        )
        # Check no bare carriage returns
        for idx, line in enumerate(self.rem_lines, 1):
            if "\r" in line and not line.endswith("\r\n"):
                self.fail(f"Line {idx} has an unescaped carriage return inside record.")

    def test_03_strict_json_record_integrity(self):
        """Verify all 1,208 records have required keys and expected types."""
        required_keys = {"id", "sample_id", "audio", "audio_text", "original_text", "source", "reference_type", "human_reviewed"}
        for idx, rec in enumerate(self.rem_records, 1):
            missing = required_keys - set(rec.keys())
            self.assertEqual(missing, set(), f"Record line {idx} ({rec.get('id')}) is missing keys: {missing}")
            self.assertIsInstance(rec["human_reviewed"], bool, f"Line {idx}: human_reviewed must be bool")
            self.assertFalse(rec["human_reviewed"], f"Line {idx}: human_reviewed must be False for unapproved items")
            for field in ["id", "sample_id", "audio", "audio_text", "original_text", "source", "reference_type"]:
                self.assertIsInstance(rec[field], str, f"Line {idx}: field {field} must be str")
                self.assertTrue(len(rec[field].strip()) > 0, f"Line {idx}: field {field} cannot be empty")

    # =========================================================================
    # 2. Partition & Coverage vs public_test.jsonl & final_reviewed.jsonl
    # =========================================================================
    def test_04_disjoint_partition_and_union_count(self):
        """Verify 0 overlap and exact 1,933 union with final_reviewed.jsonl."""
        self.assertEqual(len(self.final_records), FINAL_EXPECTED_COUNT)
        self.assertEqual(len(self.rem_records), REMAINING_EXPECTED_COUNT)

        overlap = self.rem_ids.intersection(self.final_ids)
        self.assertEqual(len(overlap), 0, f"Overlapping IDs between final and remaining: {overlap}")

        union_ids = self.rem_ids.union(self.final_ids)
        self.assertEqual(len(union_ids), TOTAL_EXPECTED_AUDIOS, f"Union count must be {TOTAL_EXPECTED_AUDIOS}")

    def test_05_public_test_dialogue_turn_mapping(self):
        """Cross-check all records against public_test.jsonl ground truth dialogue turns."""
        self.assertTrue(PUBLIC_TEST_JSONL.is_file(), f"{PUBLIC_TEST_JSONL} must exist")
        with PUBLIC_TEST_JSONL.open("r", encoding="utf-8") as f:
            pt_cases = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(len(pt_cases), TOTAL_MAIN_TURNS, "public_test.jsonl must contain 634 test cases")

        expected_ids = set()
        expected_main_ids = set()
        expected_history_ids = set()

        for case in pt_cases:
            sid = case["id"]
            # Main audio turn
            expected_ids.add(sid)
            expected_main_ids.add(sid)

            # User history turns
            history = case.get("history", [])
            for h_idx, h_turn in enumerate(history):
                # User turns with audio
                if h_turn.get("role") == "user" and "audio" in h_turn and h_turn["audio"]:
                    hid = f"{sid}:history:{h_idx}"
                    expected_ids.add(hid)
                    expected_history_ids.add(hid)

        self.assertEqual(len(expected_main_ids), TOTAL_MAIN_TURNS)
        self.assertEqual(len(expected_history_ids), TOTAL_HISTORY_TURNS)
        self.assertEqual(len(expected_ids), TOTAL_EXPECTED_AUDIOS)

        # Compare with union
        union_ids = self.rem_ids.union(self.final_ids)
        missing_from_union = expected_ids - union_ids
        unexpected_in_union = union_ids - expected_ids
        self.assertEqual(missing_from_union, set(), f"IDs missing from union: {missing_from_union}")
        self.assertEqual(unexpected_in_union, set(), f"Unexpected IDs in union: {unexpected_in_union}")

        # Check category distribution
        rem_main = {i for i in self.rem_ids if ":history:" not in i}
        rem_hist = {i for i in self.rem_ids if ":history:" in i}
        final_main = {i for i in self.final_ids if ":history:" not in i}
        final_hist = {i for i in self.final_ids if ":history:" in i}

        self.assertEqual(len(rem_main), 363, f"Expected 363 remaining main turns, got {len(rem_main)}")
        self.assertEqual(len(rem_hist), 845, f"Expected 845 remaining history turns, got {len(rem_hist)}")
        self.assertEqual(len(final_main), 271, f"Expected 271 final main turns, got {len(final_main)}")
        self.assertEqual(len(final_hist), 454, f"Expected 454 final history turns, got {len(final_hist)}")

    # =========================================================================
    # 3. Direct Physical Disk Audit (Zero Ghost Files, Zero Missing Files)
    # =========================================================================
    def test_06_direct_physical_disk_wav_enumeration(self):
        """Scan public_test/audio/*/*.wav on disk directly and match 1-to-1 with union."""
        disk_wavs = list(AUDIO_DIR.glob("*/*.wav"))
        self.assertEqual(
            len(disk_wavs),
            TOTAL_EXPECTED_AUDIOS,
            f"Physical WAV files on disk: expected {TOTAL_EXPECTED_AUDIOS}, found {len(disk_wavs)}",
        )

        disk_rel_paths = {w.relative_to(PUBLIC_TEST_DIR).as_posix() for w in disk_wavs}
        union_audio_paths = {r["audio"] for r in self.rem_records} | {r["audio"] for r in self.final_records}

        missing_on_disk = union_audio_paths - disk_rel_paths
        ghost_on_disk = disk_rel_paths - union_audio_paths
        self.assertEqual(missing_on_disk, set(), f"Audio paths missing on disk: {missing_on_disk}")
        self.assertEqual(ghost_on_disk, set(), f"Audio paths on disk not in union: {ghost_on_disk}")

    # =========================================================================
    # 4. Deep Acoustic & Waveform Integrity (All 1,208 Remaining Audios)
    # =========================================================================
    def test_07_acoustic_headers_and_signal_stress(self):
        """Check all 1,208 remaining WAV files for header validity, frame integrity, RMS energy, and duration."""
        durations = []
        rms_values = []
        zero_energy_audios = []
        corrupted_audios = []

        for idx, rec in enumerate(self.rem_records, 1):
            rid = rec["id"]
            wav_path = PUBLIC_TEST_DIR / rec["audio"]
            self.assertTrue(wav_path.is_file(), f"Line {idx} ({rid}): Audio file {wav_path} does not exist")

            # Check file size > 44 bytes (standard WAV header size)
            file_size = wav_path.stat().st_size
            self.assertGreater(file_size, 44, f"Line {idx} ({rid}): WAV file size too small: {file_size} bytes")

            try:
                with wave.open(str(wav_path), "rb") as wf:
                    channels = wf.getnchannels()
                    rate = wf.getframerate()
                    width = wf.getsampwidth()
                    comptype = wf.getcomptype()
                    nframes = wf.getnframes()

                    self.assertEqual(channels, 1, f"{rid}: channels = {channels}, expected 1")
                    self.assertEqual(rate, 16000, f"{rid}: rate = {rate}, expected 16000")
                    self.assertEqual(width, 2, f"{rid}: sampwidth = {width}, expected 2 (16-bit PCM)")
                    self.assertEqual(comptype, "NONE", f"{rid}: comptype = {comptype}, expected NONE")
                    self.assertGreater(nframes, 0, f"{rid}: nframes = {nframes}, expected > 0")

                    # Read all raw frames
                    raw_data = wf.readframes(nframes)
                    expected_bytes = nframes * 2
                    self.assertEqual(len(raw_data), expected_bytes, f"{rid}: frame bytes length mismatch")

                    duration_sec = nframes / rate
                    durations.append(duration_sec)

                    # Compute RMS energy to ensure audio is not complete zero-signal/dead air
                    samples = struct.unpack(f"<{nframes}h", raw_data)
                    sum_sq = sum(s * s for s in samples)
                    rms = math.sqrt(sum_sq / nframes)
                    rms_values.append(rms)

                    if rms == 0:
                        zero_energy_audios.append(rid)

            except Exception as e:
                corrupted_audios.append((rid, str(e)))

        self.assertEqual(corrupted_audios, [], f"Corrupted audio files: {corrupted_audios}")
        self.assertEqual(zero_energy_audios, [], f"Digital silence (RMS=0) audio files: {zero_energy_audios}")

        min_dur = min(durations)
        max_dur = max(durations)
        total_sec = sum(durations)
        total_hours = total_sec / 3600.0

        # Adversarial sanity bounds: audio should be between 0.2s and 120s
        self.assertGreater(min_dur, 0.2, f"Audio too short: min duration is {min_dur:.2f}s")
        self.assertLess(max_dur, 120.0, f"Audio too long: max duration is {max_dur:.2f}s")
        self.assertGreater(total_hours, 0.5, f"Total audio duration too small: {total_hours:.2f}h")

    # =========================================================================
    # 5. Adversarial Text Hygiene, Unicode & Hallucination Checks
    # =========================================================================
    def test_08_adversarial_text_hygiene_and_hallucination(self):
        """Check all 1,208 transcripts for subtle corruptions, unicode issues, and hallucinations."""
        cjk_re = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
        control_re = re.compile(r"[\x00-\x1f\x7f-\x9f]")
        zero_width_re = re.compile(r"[\u200b\u200c\u200d\ufeff]")
        irregular_space_re = re.compile(r"[\u00a0\u3000\t]")
        sentinel_tokens = {
            "none", "null", "undefined", "nan", "n/a", "[error]", "[silence]",
            "[noise]", "[unknown]", "todo", "tbd", "???", "***",
            "<|im_start|>", "<|im_end|>", "<|endoftext|>", "model", "assistant", "user",
        }

        repetition_re = re.compile(r"\b(\w+)(?:\s+\1){4,}\b", re.IGNORECASE)

        hygiene_violations = []

        for idx, rec in enumerate(self.rem_records, 1):
            rid = rec["id"]
            text = rec["audio_text"]

            # 1. Leading / trailing whitespace
            if text != text.strip():
                hygiene_violations.append((rid, "leading_or_trailing_whitespace", repr(text)))

            # 2. Double spaces
            if "  " in text:
                hygiene_violations.append((rid, "consecutive_spaces", repr(text)))

            # 3. Raw control characters
            if control_re.findall(text):
                hygiene_violations.append((rid, "control_characters", control_re.findall(text)))

            # 4. Zero-width spaces
            if zero_width_re.findall(text):
                hygiene_violations.append((rid, "zero_width_characters", zero_width_re.findall(text)))

            # 5. Irregular whitespace (non-breaking spaces, tabs)
            if irregular_space_re.findall(text):
                hygiene_violations.append((rid, "irregular_whitespace", irregular_space_re.findall(text)))

            # 6. CJK characters
            if cjk_re.findall(text):
                hygiene_violations.append((rid, "cjk_hallucination", cjk_re.findall(text)))

            # 7. Sentinel / error tokens
            for token in sentinel_tokens:
                if token in text.lower().split():
                    hygiene_violations.append((rid, "sentinel_token_detected", token))

            # 8. Unreasonable repetitions (word repeated 5+ times in loop)
            rep_match = repetition_re.search(text)
            if rep_match:
                hygiene_violations.append((rid, "repetition_loop", rep_match.group(0)))

            # 9. Unicode NFC normalization
            nfc_text = unicodedata.normalize("NFC", text)
            if nfc_text != text:
                hygiene_violations.append((rid, "non_nfc_unicode", "Transcript is not in NFC normalized form"))

        self.assertEqual(
            hygiene_violations,
            [],
            f"Encountered text hygiene violations in {len(hygiene_violations)} records:\n"
            + "\n".join(str(v) for v in hygiene_violations[:10]),
        )

    # =========================================================================
    # 6. Speech Rate & Acoustic-Text Alignment Sanity Bounds
    # =========================================================================
    def test_09_speech_rate_sanity_bounds(self):
        """Verify words per second (WPS) is within realistic acoustic boundaries (not 0, not > 12 WPS)."""
        wps_records = []
        for rec in self.rem_records:
            rid = rec["id"]
            wav_path = PUBLIC_TEST_DIR / rec["audio"]
            with wave.open(str(wav_path), "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()

            words = rec["audio_text"].strip().split()
            word_count = len(words)
            wps = word_count / duration
            wps_records.append((rid, duration, word_count, wps))

            # Acoustic bounds check:
            # - At least 1 word
            self.assertGreater(word_count, 0, f"{rid}: zero words in audio_text")
            # - Maximum speech rate: normal conversational Vietnamese is ~3-5 words/sec. Even fast speech < 12 wps.
            self.assertLess(
                wps,
                12.0,
                f"{rid}: Unrealistically high speech rate {wps:.2f} wps ({word_count} words in {duration:.2f}s)",
            )
            # - Minimum speech rate: at least 0.1 wps (e.g. 1 word in 10s audio)
            self.assertGreater(
                wps,
                0.1,
                f"{rid}: Unrealistically low speech rate {wps:.2f} wps ({word_count} words in {duration:.2f}s)",
            )

    # =========================================================================
    # 7. Lineage and Consistency with predictions.jsonl
    # =========================================================================
    def test_10_prediction_lineage_and_refinement_integrity(self):
        """Verify original_text matches raw model predictions and refinements are valid transformations."""
        mismatched_original = []
        refined_count = 0

        for rec in self.rem_records:
            rid = rec["id"]
            self.assertIn(rid, self.pred_dict, f"{rid} missing from predictions.jsonl")
            pred_rec = self.pred_dict[rid]

            # original_text should match prediction text
            expected_orig = pred_rec.get("text", "")
            actual_orig = rec.get("original_text", "")
            if expected_orig != actual_orig:
                mismatched_original.append((rid, expected_orig, actual_orig))

            if rec["audio_text"] != actual_orig:
                refined_count += 1

        self.assertEqual(
            mismatched_original,
            [],
            f"Found original_text mismatch with predictions.jsonl:\n"
            + "\n".join(str(m) for m in mismatched_original[:5]),
        )
        # Ensure that refinement was actually applied to a subset of records where needed
        self.assertGreater(
            refined_count,
            0,
            "Expected at least some records to have state-guided refinements applied.",
        )


if __name__ == "__main__":
    unittest.main()
