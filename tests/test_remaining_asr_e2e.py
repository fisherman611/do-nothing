"""
tests/test_remaining_asr_e2e.py
Comprehensive End-to-End Test Suite for Public Test ASR Labeling and Review UI Preparation

This suite enforces requirements from ORIGINAL_REQUEST.md and PROJECT.md across 5 tiers:
  Tier 1: Feature Coverage (remaining_reviewed.jsonl exists, exactly 1,208 records, required fields)
  Tier 2: Boundary & Corner Cases (non-empty transcripts, UTF-8 integrity, WAV correspondence, schema parity,
          adversarial filtering of CJK hallucination / repetition / control chars)
  Tier 3: Cross-Feature / Partition Integrity (725 final + 1,208 remaining = 1,933 total, disjoint partition)
  Tier 4: Anti-Leak & Environment Access Compliance (AST audit, ToolEnv API validation, runtime file interceptor,
          adversarial traversal defense)
  Tier 5: Review UI Data Preparation & Inspection (Store loading, 725 approved, 1,208 todo/suggested, matching text,
          safety audit ensuring no premature approvals)

Run command:
  .\\.venv\\Scripts\\python.exe -m unittest tests/test_remaining_asr_e2e.py
"""

import ast
import builtins
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import traceback
import unittest
import warnings
import wave

# Project Root and Canonical File Paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data/results/asr/qwen-cloud/public-test-asr-flash"
FINAL_REVIEWED_PATH = RESULTS_DIR / "final_reviewed.jsonl"
REMAINING_REVIEWED_PATH = RESULTS_DIR / "remaining_reviewed.jsonl"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.jsonl"
REVIEW_STATE_PATH = RESULTS_DIR / "review_state.json"
REVIEW_SUGGESTIONS_PATH = RESULTS_DIR / "review_suggestions.jsonl"

PUBLIC_TEST_DIR = ROOT / "data/train+public_test/public_test"
AUDIO_DIR = PUBLIC_TEST_DIR / "audio"
PUBLIC_TEST_JSONL = PUBLIC_TEST_DIR / "public_test.jsonl"

# Benchmark Target Constants
TOTAL_AUDIO_COUNT = 1933
FINAL_REVIEWED_COUNT = 725
REMAINING_REVIEWED_COUNT = 1208
FINAL_MAIN_COUNT = 271
FINAL_HISTORY_COUNT = 454
REMAINING_MAIN_COUNT = 363
REMAINING_HISTORY_COUNT = 845
TOTAL_MAIN_COUNT = 634
TOTAL_HISTORY_COUNT = 1299
REQUIRED_FIELDS = ("id", "sample_id", "audio", "audio_text")


def load_jsonl(path: Path) -> list[dict]:
    """Helper to parse a JSONL file into a list of dicts with explicit error reporting."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                records.append(json.loads(line_str))
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {err}") from err
    return records


# ==============================================================================
# Tier 1: Feature Coverage
# ==============================================================================
class TestTier1FeatureCoverage(unittest.TestCase):
    """
    Tier 1 tests core feature presence: existence of remaining_reviewed.jsonl,
    exact record count of 1,208, and presence of mandatory fields.
    """

    def test_t1_1_remaining_reviewed_file_exists(self):
        """remaining_reviewed.jsonl must exist at the expected results path."""
        self.assertTrue(
            REMAINING_REVIEWED_PATH.is_file(),
            f"Target file '{REMAINING_REVIEWED_PATH}' does not exist. "
            "Milestone 1 must generate this file.",
        )

    def test_t1_2_remaining_reviewed_valid_jsonl_and_count(self):
        """remaining_reviewed.jsonl must contain valid JSON on every line and exactly 1,208 records."""
        if not REMAINING_REVIEWED_PATH.is_file():
            self.fail(
                f"Missing '{REMAINING_REVIEWED_PATH}'. Cannot verify record count (M1 pending)."
            )
        records = load_jsonl(REMAINING_REVIEWED_PATH)
        self.assertEqual(
            len(records),
            REMAINING_REVIEWED_COUNT,
            f"Expected exactly {REMAINING_REVIEWED_COUNT} records in remaining_reviewed.jsonl, "
            f"but found {len(records)}.",
        )

    def test_t1_3_remaining_reviewed_required_fields_present(self):
        """Every record in remaining_reviewed.jsonl must have id, sample_id, audio, and audio_text."""
        if not REMAINING_REVIEWED_PATH.is_file():
            self.fail(
                f"Missing '{REMAINING_REVIEWED_PATH}'. Cannot verify fields (M1 pending)."
            )
        records = load_jsonl(REMAINING_REVIEWED_PATH)
        for idx, rec in enumerate(records):
            rec_id = rec.get("id", f"<index {idx}>")
            for field in REQUIRED_FIELDS:
                self.assertIn(
                    field,
                    rec,
                    f"Record '{rec_id}' (line {idx + 1}) missing required field '{field}'.",
                )
                self.assertIsInstance(
                    rec[field],
                    str,
                    f"Record '{rec_id}' field '{field}' must be a string, got {type(rec[field]).__name__}.",
                )


# ==============================================================================
# Tier 2: Boundary & Corner Cases
# ==============================================================================
class TestTier2BoundaryAndCornerCases(unittest.TestCase):
    """
    Tier 2 tests boundary conditions: non-empty transcripts, UTF-8 encoding integrity,
    WAV path correspondence, audio validity, schema conformance, and adversarial text hygiene.
    """

    def test_t2_1_audio_text_non_empty_and_valid_string(self):
        """Transcripts must be non-empty, non-whitespace, and not placeholder/error tokens."""
        if not REMAINING_REVIEWED_PATH.is_file():
            self.fail(
                f"Missing '{REMAINING_REVIEWED_PATH}'. Cannot verify transcript content (M1 pending)."
            )
        records = load_jsonl(REMAINING_REVIEWED_PATH)
        invalid_tokens = {"none", "null", "undefined", "n/a", "nan", "[silence]", "...", "."}

        for idx, rec in enumerate(records):
            rec_id = rec.get("id", f"<index {idx}>")
            text = rec.get("audio_text", "")
            stripped = text.strip()

            self.assertGreater(
                len(stripped),
                0,
                f"Record '{rec_id}' has empty or whitespace-only audio_text.",
            )
            self.assertNotIn(
                stripped.lower(),
                invalid_tokens,
                f"Record '{rec_id}' has invalid placeholder audio_text: {stripped!r}.",
            )
            # Ensure no raw model generation sentinel tokens leaked into transcript
            self.assertNotIn(
                "<|endoftext|>",
                text,
                f"Record '{rec_id}' contains model special token '<|endoftext|>'.",
            )
            self.assertNotIn(
                "<|im_start|>",
                text,
                f"Record '{rec_id}' contains model special token '<|im_start|>'.",
            )

    def test_t2_2_valid_utf8_encoding_and_no_replacement_chars(self):
        """remaining_reviewed.jsonl must be valid UTF-8 without unicode replacement characters (\\ufffd)."""
        if not REMAINING_REVIEWED_PATH.is_file():
            self.fail(
                f"Missing '{REMAINING_REVIEWED_PATH}'. Cannot verify UTF-8 integrity (M1 pending)."
            )
        raw_bytes = REMAINING_REVIEWED_PATH.read_bytes()
        try:
            decoded = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as err:
            self.fail(f"remaining_reviewed.jsonl is not valid UTF-8: {err}")

        self.assertNotIn(
            "\ufffd",
            decoded,
            "Found unicode replacement character '\\ufffd' in remaining_reviewed.jsonl, "
            "indicating lossy decoding or corruption.",
        )

    def test_t2_3_audio_file_paths_exist_as_valid_wav(self):
        """All audio paths must point to existing, valid 16kHz mono 16-bit PCM WAV files."""
        if not REMAINING_REVIEWED_PATH.is_file():
            self.fail(
                f"Missing '{REMAINING_REVIEWED_PATH}'. Cannot verify WAV paths (M1 pending)."
            )
        records = load_jsonl(REMAINING_REVIEWED_PATH)

        for idx, rec in enumerate(records):
            rec_id = rec.get("id", f"<index {idx}>")
            audio_rel = rec.get("audio", "")
            self.assertFalse(
                Path(audio_rel).is_absolute(),
                f"Record '{rec_id}' audio path must be relative, got '{audio_rel}'.",
            )
            wav_path = (PUBLIC_TEST_DIR / audio_rel).resolve()
            self.assertTrue(
                wav_path.is_file(),
                f"Record '{rec_id}' audio file does not exist on disk: {wav_path}",
            )
            self.assertTrue(
                wav_path.name.lower().endswith(".wav"),
                f"Record '{rec_id}' audio file is not a .wav: {wav_path}",
            )

            # Validate WAV format header (16 kHz, 1 channel, 16-bit PCM)
            with wave.open(str(wav_path), "rb") as wf:
                self.assertEqual(
                    wf.getnchannels(),
                    1,
                    f"WAV '{wav_path.name}' for '{rec_id}' must be mono (1 channel), got {wf.getnchannels()}.",
                )
                self.assertEqual(
                    wf.getframerate(),
                    16000,
                    f"WAV '{wav_path.name}' for '{rec_id}' must be 16,000 Hz, got {wf.getframerate()}.",
                )
                self.assertEqual(
                    wf.getsampwidth(),
                    2,
                    f"WAV '{wav_path.name}' for '{rec_id}' must be 16-bit (2 bytes/sample), got {wf.getsampwidth()}.",
                )

    def test_t2_4_schema_and_id_format_conformance_with_final_reviewed(self):
        """Records must conform to final_reviewed.jsonl schema conventions and ID naming patterns."""
        if not REMAINING_REVIEWED_PATH.is_file():
            self.fail(
                f"Missing '{REMAINING_REVIEWED_PATH}'. Cannot verify schema parity (M1 pending)."
            )
        records = load_jsonl(REMAINING_REVIEWED_PATH)

        main_pattern = re.compile(r"^public_\d{4}$")
        history_pattern = re.compile(r"^public_\d{4}:history:\d+$")

        for idx, rec in enumerate(records):
            rec_id = rec.get("id", "")
            sample_id = rec.get("sample_id", "")
            audio = rec.get("audio", "")

            # Check ID format
            if ":history:" in rec_id:
                self.assertTrue(
                    history_pattern.match(rec_id),
                    f"Record ID '{rec_id}' at index {idx} does not match history format 'public_XXXX:history:K'.",
                )
                self.assertEqual(
                    rec_id.split(":history:")[0],
                    sample_id,
                    f"Record ID '{rec_id}' prefix does not match sample_id '{sample_id}'.",
                )
            else:
                self.assertTrue(
                    main_pattern.match(rec_id),
                    f"Record ID '{rec_id}' at index {idx} does not match main turn format 'public_XXXX'.",
                )
                self.assertEqual(
                    rec_id,
                    sample_id,
                    f"For main turn, ID '{rec_id}' must equal sample_id '{sample_id}'.",
                )

            # Check audio path format: audio/{sample_id}/{filename}.wav
            expected_prefix = f"audio/{sample_id}/"
            self.assertTrue(
                audio.startswith(expected_prefix),
                f"Record '{rec_id}' audio '{audio}' must start with '{expected_prefix}'.",
            )

            # Check optional fields if present
            if "human_reviewed" in rec:
                self.assertIsInstance(
                    rec["human_reviewed"],
                    bool,
                    f"Record '{rec_id}' field 'human_reviewed' must be boolean.",
                )
                self.assertFalse(
                    rec["human_reviewed"],
                    f"Record '{rec_id}' human_reviewed should be False for generated remaining labels.",
                )

    def test_t2_5_adversarial_no_cjk_hallucinations_or_control_chars(self):
        """Transcripts must be Vietnamese and must not contain CJK characters or raw control characters."""
        if not REMAINING_REVIEWED_PATH.is_file():
            self.fail(
                f"Missing '{REMAINING_REVIEWED_PATH}'. Cannot verify adversarial text hygiene (M1 pending)."
            )
        records = load_jsonl(REMAINING_REVIEWED_PATH)
        cjk_regex = re.compile(r"[\u4e00-\u9fff]")
        control_char_regex = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

        for idx, rec in enumerate(records):
            rec_id = rec.get("id", f"<index {idx}>")
            text = rec.get("audio_text", "")

            # CJK characters are a known acoustic hallucination mode on Vietnamese audio
            cjk_matches = cjk_regex.findall(text)
            self.assertEqual(
                cjk_matches,
                [],
                f"Record '{rec_id}' contains CJK hallucination characters: {cjk_matches} in {text!r}",
            )

            # Raw control characters
            control_matches = control_char_regex.findall(text)
            self.assertEqual(
                control_matches,
                [],
                f"Record '{rec_id}' contains raw control characters in audio_text: {text!r}",
            )


# ==============================================================================
# Tier 3: Partition & Cross-Feature Integrity
# ==============================================================================
class TestTier3PartitionIntegrity(unittest.TestCase):
    """
    Tier 3 tests dataset partition completeness:
    725 (final) + 1,208 (remaining) = 1,933 (total public test audios),
    with strictly 0 overlap and 0 omissions.
    """

    def setUp(self):
        self.assertTrue(
            FINAL_REVIEWED_PATH.is_file(),
            f"Reference file '{FINAL_REVIEWED_PATH}' must exist.",
        )
        self.assertTrue(
            PREDICTIONS_PATH.is_file(),
            f"Universe reference file '{PREDICTIONS_PATH}' must exist.",
        )
        self.final_records = load_jsonl(FINAL_REVIEWED_PATH)
        self.pred_records = load_jsonl(PREDICTIONS_PATH)
        self.final_ids = {r["id"] for r in self.final_records}
        self.pred_ids = {r["id"] for r in self.pred_records}

    def test_t3_1_final_reviewed_count_is_725(self):
        """final_reviewed.jsonl must strictly remain at exactly 725 human-verified records."""
        self.assertEqual(
            len(self.final_records),
            FINAL_REVIEWED_COUNT,
            f"Expected exactly {FINAL_REVIEWED_COUNT} records in final_reviewed.jsonl, "
            f"got {len(self.final_records)}. The approved set must not be altered.",
        )

    def test_t3_2_remaining_reviewed_count_is_1208(self):
        """remaining_reviewed.jsonl must contain exactly 1,208 records."""
        if not REMAINING_REVIEWED_PATH.is_file():
            self.fail(
                f"Missing '{REMAINING_REVIEWED_PATH}'. Cannot verify count (M1 pending)."
            )
        rem_records = load_jsonl(REMAINING_REVIEWED_PATH)
        self.assertEqual(
            len(rem_records),
            REMAINING_REVIEWED_COUNT,
            f"Expected exactly {REMAINING_REVIEWED_COUNT} records in remaining_reviewed.jsonl, "
            f"got {len(rem_records)}.",
        )

    def test_t3_3_zero_id_overlap_between_final_and_remaining(self):
        """There must be exactly zero ID overlap between final_reviewed and remaining_reviewed."""
        if not REMAINING_REVIEWED_PATH.is_file():
            self.fail(
                f"Missing '{REMAINING_REVIEWED_PATH}'. Cannot verify disjointness (M1 pending)."
            )
        rem_records = load_jsonl(REMAINING_REVIEWED_PATH)
        rem_ids = {r["id"] for r in rem_records}

        overlap = self.final_ids.intersection(rem_ids)
        self.assertEqual(
            len(overlap),
            0,
            f"Found {len(overlap)} overlapping IDs between final_reviewed and remaining_reviewed: "
            f"{sorted(list(overlap))[:5]}...",
        )

    def test_t3_4_union_coverage_exact_match_1933_public_test_audios(self):
        """The union of final_reviewed and remaining_reviewed must cover all 1,933 public test audios exactly."""
        if not REMAINING_REVIEWED_PATH.is_file():
            self.fail(
                f"Missing '{REMAINING_REVIEWED_PATH}'. Cannot verify union coverage (M1 pending)."
            )
        rem_records = load_jsonl(REMAINING_REVIEWED_PATH)
        rem_ids = {r["id"] for r in rem_records}

        self.assertEqual(
            len(self.pred_ids),
            TOTAL_AUDIO_COUNT,
            f"Expected universe size {TOTAL_AUDIO_COUNT}, got {len(self.pred_ids)}.",
        )

        union_ids = self.final_ids.union(rem_ids)
        self.assertEqual(
            len(union_ids),
            TOTAL_AUDIO_COUNT,
            f"Combined union size is {len(union_ids)}, expected {TOTAL_AUDIO_COUNT}.",
        )

        missing_ids = self.pred_ids - union_ids
        self.assertEqual(
            len(missing_ids),
            0,
            f"{len(missing_ids)} public test audio IDs missing from union: {sorted(list(missing_ids))[:5]}...",
        )

        unexpected_ids = union_ids - self.pred_ids
        self.assertEqual(
            len(unexpected_ids),
            0,
            f"{len(unexpected_ids)} unrecognized audio IDs found in union: {sorted(list(unexpected_ids))[:5]}...",
        )

    def test_t3_5_category_breakdown_partition(self):
        """Check exact category breakdown for main vs history turns in both partitions."""
        if not REMAINING_REVIEWED_PATH.is_file():
            self.fail(
                f"Missing '{REMAINING_REVIEWED_PATH}'. Cannot verify category breakdown (M1 pending)."
            )
        rem_records = load_jsonl(REMAINING_REVIEWED_PATH)

        final_main = [r for r in self.final_records if ":history:" not in r["id"]]
        final_hist = [r for r in self.final_records if ":history:" in r["id"]]
        self.assertEqual(len(final_main), FINAL_MAIN_COUNT)
        self.assertEqual(len(final_hist), FINAL_HISTORY_COUNT)

        rem_main = [r for r in rem_records if ":history:" not in r["id"]]
        rem_hist = [r for r in rem_records if ":history:" in r["id"]]
        self.assertEqual(
            len(rem_main),
            REMAINING_MAIN_COUNT,
            f"Expected {REMAINING_MAIN_COUNT} main turns in remaining_reviewed, got {len(rem_main)}.",
        )
        self.assertEqual(
            len(rem_hist),
            REMAINING_HISTORY_COUNT,
            f"Expected {REMAINING_HISTORY_COUNT} history turns in remaining_reviewed, got {len(rem_hist)}.",
        )

        self.assertEqual(len(final_main) + len(rem_main), TOTAL_MAIN_COUNT)
        self.assertEqual(len(final_hist) + len(rem_hist), TOTAL_HISTORY_COUNT)


# ==============================================================================
# Tier 4: Anti-Leak Compliance & Environment Access
# ==============================================================================
class TestTier4AntiLeakCompliance(unittest.TestCase):
    """
    Tier 4 tests anti-leak compliance:
    1. Static AST audit verifying no code in `asr/` directly opens `*_environment/` or `state.json`.
    2. Dynamic verification that `zaic_tools.ToolEnv.get_state()` is the authorized interface.
    3. Runtime interception hook verifying unauthorized direct opens of environment state are blocked.
    4. Adversarial path traversal detection (e.g. audio/../public_test_environment).
    """

    def test_t4_1_ast_inspection_no_direct_environment_access(self):
        """Static AST scan of asr/ implementation code ensuring zero direct opens of *_environment/ or state.json."""
        asr_dir = ROOT / "asr"
        self.assertTrue(asr_dir.is_dir(), f"Directory not found: {asr_dir}")

        violations = []
        for py_path in asr_dir.rglob("*.py"):
            # Exclude tests and test suites
            if "tests" in py_path.parts or py_path.name.startswith("test_"):
                continue

            try:
                code = py_path.read_text(encoding="utf-8")
                tree = ast.parse(code, filename=str(py_path))
            except Exception as e:
                self.fail(f"Failed to parse AST of {py_path}: {e}")

            for node in ast.walk(tree):
                # Check function / method calls
                if isinstance(node, ast.Call):
                    # Check arguments passed to calls (e.g. open, Path, etc.)
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            val = arg.value.replace("\\", "/")
                            # If argument references an environment directory or environment state.json
                            if ("_environment" in val or val.endswith("/state.json")) and "review_state.json" not in val:
                                violations.append(
                                    f"{py_path.relative_to(ROOT)}:{node.lineno}: "
                                    f"Forbidden environment path argument in call: {val!r}"
                                )
                # Check binary operations like path / "public_test_environment"
                elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                    for side in (node.left, node.right):
                        if isinstance(side, ast.Constant) and isinstance(side.value, str):
                            val = side.value.replace("\\", "/")
                            if "_environment" in val:
                                violations.append(
                                    f"{py_path.relative_to(ROOT)}:{node.lineno}: "
                                    f"Forbidden environment path segment in division: {val!r}"
                                )

        self.assertEqual(
            violations,
            [],
            "AST Anti-leak violations detected in asr/ code:\n" + "\n".join(violations),
        )

    def test_t4_2_toolenv_official_state_access(self):
        """Verify that zaic_tools.ToolEnv.get_state() is functional and provides expected app state."""
        sys.path.insert(0, str(PUBLIC_TEST_DIR))
        try:
            from zaic_tools import ToolEnv
        except ImportError as e:
            self.fail(f"Could not import zaic_tools.ToolEnv: {e}")

        # Ignore unclosed file resource warnings within zaic_tools library implementation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            sample_id = "public_0002"
            env = ToolEnv("public_test", sample_id, root=str(PUBLIC_TEST_DIR.parent))
            state = env.get_state()

        self.assertIsInstance(state, dict, "ToolEnv.get_state() must return a dict.")
        self.assertGreater(len(state), 0, "ToolEnv.get_state() returned empty state.")
        self.assertIn(
            "calendar",
            state,
            "Expected 'calendar' app state in public_0002.",
        )
        self.assertIn("events", state["calendar"])

    def test_t4_3_runtime_intercept_blocks_unauthorized_state_file_opens(self):
        """Runtime interceptor must detect and block any direct filesystem open of environment state files outside zaic_tools."""
        sys.path.insert(0, str(PUBLIC_TEST_DIR))
        from zaic_tools import ToolEnv

        orig_open = builtins.open
        orig_path_open = Path.open

        def audit_check(file_arg):
            # Normalize path representation to detect traversal tricks
            p = str(Path(file_arg).resolve()).replace("\\", "/")
            if "_environment" in p and p.endswith("state.json"):
                stack = traceback.extract_stack()
                callers = [f.filename.replace("\\", "/") for f in stack]
                in_zaic = any("zaic_tools" in c for c in callers)
                if not in_zaic:
                    raise PermissionError(
                        f"Anti-leak violation: unauthorized direct open of environment file '{file_arg}'."
                    )

        def audited_open(file, *args, **kwargs):
            audit_check(file)
            return orig_open(file, *args, **kwargs)

        def audited_path_open(self, *args, **kwargs):
            audit_check(self)
            return orig_path_open(self, *args, **kwargs)

        try:
            builtins.open = audited_open
            Path.open = audited_path_open

            # 1. Official ToolEnv call must succeed through zaic_tools
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                env = ToolEnv("public_test", "public_0002", root=str(PUBLIC_TEST_DIR.parent))
                st = env.get_state()
            self.assertIn("calendar", st)

            # 2. Direct open must be intercepted and rejected with PermissionError
            forbidden_target = (
                PUBLIC_TEST_DIR / "public_test_environment" / "public_0002" / "state.json"
            )
            with self.assertRaises(PermissionError):
                with open(forbidden_target, "r", encoding="utf-8") as _:
                    pass

            with self.assertRaises(PermissionError):
                _ = forbidden_target.read_text(encoding="utf-8")

        finally:
            builtins.open = orig_open
            Path.open = orig_path_open

    def test_t4_4_adversarial_traversal_detection(self):
        """Adversarial check: path traversal attempts (e.g. audio/../public_test_environment) must also be caught."""
        orig_open = builtins.open
        orig_path_open = Path.open

        def audit_check(file_arg):
            resolved = str(Path(file_arg).resolve()).replace("\\", "/")
            if "_environment" in resolved and resolved.endswith("state.json"):
                stack = traceback.extract_stack()
                callers = [f.filename.replace("\\", "/") for f in stack]
                in_zaic = any("zaic_tools" in c for c in callers)
                if not in_zaic:
                    raise PermissionError(f"Anti-leak violation: path traversal access to {file_arg}")

        def audited_open(file, *args, **kwargs):
            audit_check(file)
            return orig_open(file, *args, **kwargs)

        def audited_path_open(self, *args, **kwargs):
            audit_check(self)
            return orig_path_open(self, *args, **kwargs)

        try:
            builtins.open = audited_open
            Path.open = audited_path_open

            traversal_path = (
                PUBLIC_TEST_DIR / "audio" / ".." / "public_test_environment" / "public_0002" / "state.json"
            )
            with self.assertRaises(PermissionError):
                with open(traversal_path, "r", encoding="utf-8") as _:
                    pass

        finally:
            builtins.open = orig_open
            Path.open = orig_path_open


# ==============================================================================
# Tier 5: Review UI Data Preparation & Inspection
# ==============================================================================
class TestTier5ReviewUIPayloadPreparation(unittest.TestCase):
    """
    Tier 5 tests review UI data integration:
    - asr/review_ui.py Store loads successfully.
    - Store.payload() maintains 725 approved items and 1,208 todo items.
    - All 1,208 items have suggestion flags and newly labeled transcripts.
    - review_state.json maintains safety (no unapproved items marked approved).
    - review_ui Store save isolation check.
    """

    def setUp(self):
        try:
            from asr.review_ui import Store
            self.Store = Store
        except ImportError as e:
            self.fail(f"Failed to import Store from asr.review_ui: {e}")

        self.store = self.Store(RESULTS_DIR, PUBLIC_TEST_DIR)

    def test_t5_1_review_ui_store_loads_all_1933_samples(self):
        """Store must load all 1,933 public test samples from predictions.jsonl."""
        self.assertEqual(
            len(self.store.samples),
            TOTAL_AUDIO_COUNT,
            f"Expected {TOTAL_AUDIO_COUNT} samples in Store, got {len(self.store.samples)}.",
        )
        self.assertEqual(
            len(self.store.by_id),
            TOTAL_AUDIO_COUNT,
            f"Expected {TOTAL_AUDIO_COUNT} unique IDs in Store.by_id, got {len(self.store.by_id)}.",
        )

    def test_t5_2_review_ui_payload_preserves_725_approved(self):
        """Store.payload() must have exactly 725 approved items matching final_reviewed.jsonl."""
        payload = self.store.payload()
        approved = [s for s in payload if s["approved"]]
        self.assertEqual(
            len(approved),
            FINAL_REVIEWED_COUNT,
            f"Expected exactly {FINAL_REVIEWED_COUNT} approved samples in UI payload, got {len(approved)}.",
        )

        final_records = load_jsonl(FINAL_REVIEWED_PATH)
        final_ids = {r["id"] for r in final_records}
        approved_ids = {s["id"] for s in approved}
        self.assertEqual(
            approved_ids,
            final_ids,
            "Set of approved IDs in review UI does not match final_reviewed.jsonl.",
        )

    def test_t5_3_review_ui_payload_has_1208_todo_with_suggestions(self):
        """Store.payload() must have exactly 1,208 unapproved (todo) items, all with suggestions."""
        payload = self.store.payload()
        todo = [s for s in payload if not s["approved"]]
        self.assertEqual(
            len(todo),
            REMAINING_REVIEWED_COUNT,
            f"Expected {REMAINING_REVIEWED_COUNT} todo items in UI payload, got {len(todo)}.",
        )

        with_suggestions = [s for s in todo if s["has_suggestion"]]
        self.assertEqual(
            len(with_suggestions),
            REMAINING_REVIEWED_COUNT,
            f"Expected all {REMAINING_REVIEWED_COUNT} todo items to have suggestions, "
            f"but only {len(with_suggestions)} have suggestions.",
        )

    def test_t5_4_ui_payload_transcripts_match_remaining_reviewed(self):
        """When remaining_reviewed.jsonl exists, every todo item in the UI must display the new transcript."""
        if not REMAINING_REVIEWED_PATH.is_file():
            self.fail(
                f"Missing '{REMAINING_REVIEWED_PATH}'. Cannot verify UI transcript parity (M1/M2 pending)."
            )

        rem_records = load_jsonl(REMAINING_REVIEWED_PATH)
        rem_dict = {r["id"]: r["audio_text"] for r in rem_records}

        payload = self.store.payload()
        todo = [s for s in payload if not s["approved"]]

        mismatches = []
        for s in todo:
            sid = s["id"]
            if sid not in rem_dict:
                mismatches.append(f"Missing from remaining_reviewed: {sid}")
                continue
            expected_text = rem_dict[sid]
            ui_text = s["text"]
            if ui_text != expected_text:
                mismatches.append(
                    f"ID {sid}: UI text {ui_text!r} != remaining_reviewed text {expected_text!r}"
                )

        self.assertEqual(
            mismatches,
            [],
            f"Found {len(mismatches)} mismatches between UI payload and remaining_reviewed.jsonl:\n"
            + "\n".join(mismatches[:10]),
        )

    def test_t5_5_safety_review_state_unapproved_integrity(self):
        """review_state.json must not have any of the 1,208 remaining items marked as approved: true."""
        self.assertTrue(
            REVIEW_STATE_PATH.is_file(),
            f"Expected review_state.json to exist at {REVIEW_STATE_PATH}.",
        )
        state_data = json.loads(REVIEW_STATE_PATH.read_text(encoding="utf-8"))

        pred_records = load_jsonl(PREDICTIONS_PATH)
        final_records = load_jsonl(FINAL_REVIEWED_PATH)
        final_ids = {r["id"] for r in final_records}
        rem_expected_ids = {r["id"] for r in pred_records if r["id"] not in final_ids}

        prematurely_approved = []
        for sid in rem_expected_ids:
            if sid in state_data and state_data[sid].get("approved") is True:
                prematurely_approved.append(sid)

        self.assertEqual(
            prematurely_approved,
            [],
            f"Found {len(prematurely_approved)} remaining audio IDs marked 'approved: true' in review_state.json. "
            "This violates safety because it would trigger premature inclusion in final_reviewed.jsonl:\n"
            + str(prematurely_approved[:10]),
        )

    def test_t5_6_store_save_isolated_simulation(self):
        """Isolated test: Store.save with approved=False does not write to final_reviewed.jsonl, and unknown ID raises KeyError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_res = Path(temp_dir)
            # Copy predictions, state, suggestions to temp
            shutil.copy2(PREDICTIONS_PATH, temp_res / "predictions.jsonl")
            shutil.copy2(REVIEW_STATE_PATH, temp_res / "review_state.json")
            if REVIEW_SUGGESTIONS_PATH.is_file():
                shutil.copy2(REVIEW_SUGGESTIONS_PATH, temp_res / "review_suggestions.jsonl")
            shutil.copy2(FINAL_REVIEWED_PATH, temp_res / "final_reviewed.jsonl")

            temp_store = self.Store(temp_res, PUBLIC_TEST_DIR)

            # 1. Invalid ID must raise KeyError
            with self.assertRaises(KeyError):
                temp_store.save("non_existent_sample_id_xyz", "text", approved=True)

            # 2. Save with approved=False on an unreviewed sample
            sample_to_test = "public_0289"
            temp_store.save(sample_to_test, "Test draft transcript", approved=False)

            # Confirm final_reviewed.jsonl was NOT augmented with this unapproved item
            written_finals = load_jsonl(temp_res / "final_reviewed.jsonl")
            self.assertEqual(len(written_finals), FINAL_REVIEWED_COUNT)
            written_ids = {r["id"] for r in written_finals}
            self.assertNotIn(sample_to_test, written_ids)


if __name__ == "__main__":
    unittest.main()
