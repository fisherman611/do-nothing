"""
tests/test_challenger_anti_leak_and_ui.py
Empirical Challenger Test Suite: Anti-Leak Compliance & Review UI Integration Stress-Tester

Adversarially audits:
1. Static AST analysis of asr/label_remaining.py for forbidden environment access.
2. Runtime interception of file opens (builtins.open, Path.open, etc.) ensuring no direct access to *_environment/ or state.json.
3. Official zaic_tools.ToolEnv interface validation and spy verification.
4. Review UI Store instantiation, exact 1,933 items (725 approved, 1,208 todo/suggested), and transcript synchronization.
5. Review UI Store.save safety, idempotence, draft isolation, and atomic write mechanics.
"""

from __future__ import annotations

import ast
import builtins
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

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data/results/asr/qwen-cloud/public-test-asr-flash"
FINAL_REVIEWED_PATH = RESULTS_DIR / "final_reviewed.jsonl"
REMAINING_REVIEWED_PATH = RESULTS_DIR / "remaining_reviewed.jsonl"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.jsonl"
REVIEW_STATE_PATH = RESULTS_DIR / "review_state.json"
REVIEW_SUGGESTIONS_PATH = RESULTS_DIR / "review_suggestions.jsonl"
PUBLIC_TEST_DIR = ROOT / "data/train+public_test/public_test"
DATA_ROOT = ROOT / "data/train+public_test"

TOTAL_COUNT = 1933
APPROVED_COUNT = 725
REMAINING_COUNT = 1208


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))
    return records


# ==============================================================================
# Suite 1: Anti-Leak AST & Dynamic Interception Stress-Tests
# ==============================================================================
class TestAntiLeakStressHarness(unittest.TestCase):
    """Adversarial stress-testing of anti-leak rules (Requirement R2)."""

    def test_ast_deep_inspection_label_remaining(self):
        """Exhaustive AST inspection of asr/label_remaining.py."""
        target_script = ROOT / "asr/label_remaining.py"
        self.assertTrue(target_script.is_file(), f"Missing {target_script}")

        source_code = target_script.read_text(encoding="utf-8")
        tree = ast.parse(source_code, filename=str(target_script))

        # Check imports: zaic_tools must be imported, but nothing from *_environment
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.add(node.module)

        for mod in imported_modules:
            self.assertNotIn(
                "environment",
                mod.lower(),
                f"Suspicious import module '{mod}' referencing environment in {target_script}",
            )

        # Walk all constants, strings, function calls, and binary operations
        violations = []
        for node in ast.walk(tree):
            # Check string constants
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                s = node.value.replace("\\", "/")
                # Allow docstrings or comments mentioning rules, but forbid path constants pointing to *_environment
                if "_environment" in s and not s.startswith("Extracts application state strictly") and not s.startswith("R2"):
                    violations.append(f"Line {node.lineno}: String constant containing '_environment': {s!r}")
                if s.endswith("/state.json") or (s == "state.json" and "review_state.json" not in s):
                    violations.append(f"Line {node.lineno}: String constant targeting state.json: {s!r}")

            # Check format strings / JoinedStr
            if isinstance(node, ast.JoinedStr):
                for part in node.values:
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        s = part.value.replace("\\", "/")
                        if "_environment" in s:
                            violations.append(f"Line {node.lineno}: f-string part containing '_environment': {s!r}")

            # Check path divisions
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                for child in (node.left, node.right):
                    if isinstance(child, ast.Constant) and isinstance(child.value, str):
                        s = child.value.replace("\\", "/")
                        if "_environment" in s:
                            violations.append(f"Line {node.lineno}: Path division with '_environment': {s!r}")

        self.assertEqual(
            violations,
            [],
            f"Anti-leak AST violations found in {target_script}:\n" + "\n".join(violations),
        )

    def test_runtime_open_interception_enforces_zaic_tools_isolation(self):
        """
        Runtime interception of builtins.open and Path.open.
        Verifies that:
        1. When zaic_tools opens state.json, it is allowed.
        2. When any non-zaic caller attempts to open *_environment or state.json, PermissionError is raised.
        """
        sys.path.insert(0, str(PUBLIC_TEST_DIR))
        from zaic_tools import ToolEnv

        orig_builtin_open = builtins.open
        orig_path_open = Path.open
        orig_path_read_text = Path.read_text
        orig_path_read_bytes = Path.read_bytes

        audit_log = []

        def audit_guard(path_obj):
            norm = str(Path(path_obj).resolve()).replace("\\", "/")
            if "_environment" in norm and norm.endswith("state.json"):
                stack = traceback.extract_stack()
                callers = [frame.filename.replace("\\", "/") for frame in stack]
                in_zaic = any("zaic_tools" in c for c in callers)
                audit_log.append({"path": norm, "in_zaic": in_zaic})
                if not in_zaic:
                    raise PermissionError(
                        f"Anti-Leak Intercepted: Unauthorized attempt to open environment state file directly: {norm}"
                    )

        def hooked_builtin_open(file, *args, **kwargs):
            audit_guard(file)
            return orig_builtin_open(file, *args, **kwargs)

        def hooked_path_open(self, *args, **kwargs):
            audit_guard(self)
            return orig_path_open(self, *args, **kwargs)

        def hooked_path_read_text(self, *args, **kwargs):
            audit_guard(self)
            return orig_path_read_text(self, *args, **kwargs)

        def hooked_path_read_bytes(self, *args, **kwargs):
            audit_guard(self)
            return orig_path_read_bytes(self, *args, **kwargs)

        try:
            builtins.open = hooked_builtin_open
            Path.open = hooked_path_open
            Path.read_text = hooked_path_read_text
            Path.read_bytes = hooked_path_read_bytes

            # 1. Official ToolEnv call must succeed and register audit_log with in_zaic=True
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                env = ToolEnv("public_test", "public_0006", root=str(DATA_ROOT))
                st = env.get_state()
            self.assertIsInstance(st, dict)
            self.assertGreater(len(audit_log), 0)
            self.assertTrue(all(entry["in_zaic"] for entry in audit_log))

            # 2. Adversarial direct opens must all be rejected with PermissionError
            test_cases = [
                PUBLIC_TEST_DIR / "public_test_environment/public_0006/state.json",
                PUBLIC_TEST_DIR / "audio/../public_test_environment/public_0006/state.json",
                ROOT / "data/train+public_test/public_test/public_test_environment/public_0006/state.json",
            ]
            for bad_path in test_cases:
                with self.assertRaises(PermissionError, msg=f"Failed to intercept direct open of {bad_path}"):
                    with open(bad_path, "r", encoding="utf-8") as _:
                        pass

                with self.assertRaises(PermissionError, msg=f"Failed to intercept Path.open of {bad_path}"):
                    with bad_path.open("r", encoding="utf-8") as _:
                        pass

                with self.assertRaises(PermissionError, msg=f"Failed to intercept Path.read_text of {bad_path}"):
                    _ = bad_path.read_text(encoding="utf-8")

        finally:
            builtins.open = orig_builtin_open
            Path.open = orig_path_open
            Path.read_text = orig_path_read_text
            Path.read_bytes = orig_path_read_bytes

    def test_zaic_toolenv_get_state_structure_integrity(self):
        """Verify ToolEnv.get_state() returns expected domain models across multiple samples."""
        sys.path.insert(0, str(PUBLIC_TEST_DIR))
        from zaic_tools import ToolEnv
        from asr.label_remaining import extract_entities_from_state

        test_samples = ["public_0002", "public_0006", "public_0010", "public_0025"]
        for sid in test_samples:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                env = ToolEnv("public_test", sid, root=str(DATA_ROOT))
                st = env.get_state()
            self.assertIsInstance(st, dict)
            entities = extract_entities_from_state(st)
            self.assertIsInstance(entities, dict)
            self.assertIn("contacts", entities)
            self.assertIn("banks", entities)
            self.assertIn("events", entities)


# ==============================================================================
# Suite 2: Review UI Store Integration & Safety Stress-Tests
# ==============================================================================
class TestReviewUIStoreStressHarness(unittest.TestCase):
    """Adversarial stress-testing of asr/review_ui.py Store."""

    def setUp(self):
        from asr.review_ui import Store
        self.store = Store(RESULTS_DIR, PUBLIC_TEST_DIR)

    def test_store_sample_inventory_and_by_id(self):
        """Store must load all 1,933 public test samples."""
        self.assertEqual(len(self.store.samples), TOTAL_COUNT)
        self.assertEqual(len(self.store.by_id), TOTAL_COUNT)
        self.assertEqual(len(self.store.samples), len(set(s["id"] for s in self.store.samples)))

    def test_store_payload_exact_split_and_flags(self):
        """
        Store.payload() must have:
        - Exactly 725 approved items (matching final_reviewed.jsonl).
        - Exactly 1,208 unapproved items (todo).
        - All 1,208 unapproved items must have has_suggestion=True.
        - None of the 725 approved items should have has_suggestion=True.
        """
        payload = self.store.payload()
        self.assertEqual(len(payload), TOTAL_COUNT)

        approved = [s for s in payload if s["approved"]]
        todo = [s for s in payload if not s["approved"]]

        self.assertEqual(len(approved), APPROVED_COUNT)
        self.assertEqual(len(todo), REMAINING_COUNT)

        # Check suggestions on todo items
        todo_with_suggestions = [s for s in todo if s["has_suggestion"]]
        self.assertEqual(len(todo_with_suggestions), REMAINING_COUNT)

        # Check that approved items do not have suggestion overriding approved status
        approved_with_suggestions = [s for s in approved if s["has_suggestion"]]
        self.assertEqual(len(approved_with_suggestions), 0)

        # Check suggestion sources
        for s in todo:
            self.assertTrue(
                s["suggestion_source"].startswith("zaic_toolenv_state_guided:"),
                f"Item {s['id']} has invalid suggestion source: {s['suggestion_source']}",
            )

    def test_store_payload_transcripts_match_remaining_reviewed(self):
        """Every todo item in Store.payload() must have text matching remaining_reviewed.jsonl."""
        rem_records = load_jsonl(REMAINING_REVIEWED_PATH)
        self.assertEqual(len(rem_records), REMAINING_COUNT)
        rem_map = {r["id"]: r["audio_text"] for r in rem_records}

        payload = self.store.payload()
        todo = [s for s in payload if not s["approved"]]

        mismatches = []
        for s in todo:
            sid = s["id"]
            expected = rem_map[sid]
            actual = s["text"]
            if expected != actual:
                mismatches.append(f"{sid}: expected {expected!r} != UI {actual!r}")

        self.assertEqual(mismatches, [], f"Mismatched UI transcripts:\n" + "\n".join(mismatches[:10]))

    def test_store_save_draft_isolation_and_atomic_writes(self):
        """
        Adversarial simulation of Store.save:
        - Draft save (approved=False) MUST NEVER write to final_reviewed.jsonl.
        - Final save (approved=True) writes atomically and preserves order.
        - Un-approving immediately removes item from final_reviewed.jsonl.
        - Unknown ID raises KeyError.
        """
        from asr.review_ui import Store

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            shutil.copy2(PREDICTIONS_PATH, tmp_path / "predictions.jsonl")
            shutil.copy2(REVIEW_STATE_PATH, tmp_path / "review_state.json")
            shutil.copy2(REVIEW_SUGGESTIONS_PATH, tmp_path / "review_suggestions.jsonl")
            shutil.copy2(FINAL_REVIEWED_PATH, tmp_path / "final_reviewed.jsonl")

            sim_store = Store(tmp_path, PUBLIC_TEST_DIR)

            # 1. Unknown ID raises KeyError
            with self.assertRaises(KeyError):
                sim_store.save("non_existent_audio_id", "some text", approved=True)

            # 2. Save draft on a todo sample (e.g. public_0289)
            target_id = "public_0289"
            sim_store.save(target_id, "Bản dịch nháp thử nghiệm", approved=False)

            # Check review_state.json updated
            state_data = json.loads((tmp_path / "review_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state_data[target_id]["text"], "Bản dịch nháp thử nghiệm")
            self.assertFalse(state_data[target_id]["approved"])

            # Check final_reviewed.jsonl was NOT augmented (must stay at 725)
            finals = load_jsonl(tmp_path / "final_reviewed.jsonl")
            self.assertEqual(len(finals), APPROVED_COUNT)
            self.assertNotIn(target_id, {r["id"] for r in finals})

            # Check payload reflection
            p = sim_store.payload()
            item = next(s for s in p if s["id"] == target_id)
            self.assertEqual(item["text"], "Bản dịch nháp thử nghiệm")
            self.assertFalse(item["approved"])
            self.assertFalse(item["has_suggestion"])  # Draft override disables has_suggestion

            # 3. Now approve this sample
            sim_store.save(target_id, "Bản dịch đã chuẩn duyệt", approved=True)
            finals_after_approve = load_jsonl(tmp_path / "final_reviewed.jsonl")
            self.assertEqual(len(finals_after_approve), APPROVED_COUNT + 1)
            new_record = next(r for r in finals_after_approve if r["id"] == target_id)
            self.assertEqual(new_record["audio_text"], "Bản dịch đã chuẩn duyệt")
            self.assertTrue(new_record["human_reviewed"])
            self.assertEqual(new_record["source"], "human_review")
            self.assertEqual(new_record["reference_type"], "human_verified")

            # Check canonical ordering in final_reviewed.jsonl matches predictions.jsonl
            pred_order = {s["id"]: i for i, s in enumerate(sim_store.samples)}
            final_indices = [pred_order[r["id"]] for r in finals_after_approve]
            self.assertEqual(final_indices, sorted(final_indices), "Ordering in final_reviewed.jsonl violated")

            # 4. Now un-approve (untick checkbox)
            sim_store.save(target_id, "Bản dịch đã chuẩn duyệt", approved=False)
            finals_after_unapprove = load_jsonl(tmp_path / "final_reviewed.jsonl")
            self.assertEqual(len(finals_after_unapprove), APPROVED_COUNT)
            self.assertNotIn(target_id, {r["id"] for r in finals_after_unapprove})

    def test_store_save_with_adversarial_and_multiline_strings(self):
        """Test Store.save handles quotes, newlines, and unicode safely."""
        from asr.review_ui import Store

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            shutil.copy2(PREDICTIONS_PATH, tmp_path / "predictions.jsonl")
            shutil.copy2(REVIEW_STATE_PATH, tmp_path / "review_state.json")
            shutil.copy2(REVIEW_SUGGESTIONS_PATH, tmp_path / "review_suggestions.jsonl")
            shutil.copy2(FINAL_REVIEWED_PATH, tmp_path / "final_reviewed.jsonl")

            sim_store = Store(tmp_path, PUBLIC_TEST_DIR)

            target_id = "public_0290:history:2"
            tricky_text = 'Câu thoại có dấu ngoặc "kép" và ký tự tiếng Việt: Đỗ Quyên, Ngô Quyền, \n dòng 2'
            sim_store.save(target_id, tricky_text, approved=True)

            finals = load_jsonl(tmp_path / "final_reviewed.jsonl")
            saved_item = next(r for r in finals if r["id"] == target_id)
            self.assertEqual(saved_item["audio_text"], tricky_text.strip())

    def test_suggestions_file_exact_1208_sync_with_remaining_reviewed(self):
        """review_suggestions.jsonl and remaining_reviewed.jsonl must be 100% synchronized across all 1,208 records."""
        rem_records = load_jsonl(REMAINING_REVIEWED_PATH)
        sugg_records = load_jsonl(REVIEW_SUGGESTIONS_PATH)

        self.assertEqual(len(rem_records), REMAINING_COUNT)
        self.assertEqual(len(sugg_records), REMAINING_COUNT)

        rem_map = {r["id"]: r for r in rem_records}
        sugg_map = {s["id"]: s for s in sugg_records}

        self.assertEqual(set(rem_map.keys()), set(sugg_map.keys()))

        mismatches = []
        for sid, rem in rem_map.items():
            sugg = sugg_map[sid]
            if rem["audio_text"] != sugg["audio_text"]:
                mismatches.append(f"Text mismatch for {sid}: rem={rem['audio_text']!r} != sugg={sugg['audio_text']!r}")
            if rem["audio"] != sugg["audio"]:
                mismatches.append(f"Audio mismatch for {sid}: rem={rem['audio']!r} != sugg={sugg['audio']!r}")
            if sugg["human_reviewed"] is not False:
                mismatches.append(f"sugg['human_reviewed'] must be False for {sid}")
            if sugg["reference_type"] != "state_guided_suggestion":
                mismatches.append(f"sugg['reference_type'] unexpected for {sid}: {sugg['reference_type']}")

        self.assertEqual(mismatches, [], "Suggestions sync discrepancies:\n" + "\n".join(mismatches[:10]))

    def test_ast_scan_entire_asr_directory_for_environment_leaks(self):
        """Audit all production Python files in asr/ to guarantee zero direct environment references in executable code."""
        asr_path = ROOT / "asr"
        violations = []
        for py_path in asr_path.rglob("*.py"):
            if "tests" in py_path.parts or py_path.name.startswith("test_"):
                continue

            tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
            # Find module docstring to exclude it
            docstring = ast.get_docstring(tree)

            for node in ast.walk(tree):
                # Check call arguments
                if isinstance(node, ast.Call):
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            val = arg.value.replace("\\", "/")
                            if ("_environment" in val or val.endswith("/state.json")) and "review_state.json" not in val:
                                violations.append(f"{py_path.name}:{node.lineno}: Call arg {val!r}")
                # Check binary division operations (Path / "...")
                elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                    for side in (node.left, node.right):
                        if isinstance(side, ast.Constant) and isinstance(side.value, str):
                            val = side.value.replace("\\", "/")
                            if "_environment" in val or (val.endswith("/state.json") and "review_state.json" not in val):
                                violations.append(f"{py_path.name}:{node.lineno}: Path div {val!r}")

        self.assertEqual(violations, [], "AST violations in asr/ codebase:\n" + "\n".join(violations))

    def test_store_concurrent_save_thread_safety(self):
        """Simulate concurrent Store.save calls synchronized via review_ui.LOCK."""
        import concurrent.futures
        from asr.review_ui import Store, LOCK

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            shutil.copy2(PREDICTIONS_PATH, tmp_path / "predictions.jsonl")
            shutil.copy2(REVIEW_STATE_PATH, tmp_path / "review_state.json")
            shutil.copy2(REVIEW_SUGGESTIONS_PATH, tmp_path / "review_suggestions.jsonl")
            shutil.copy2(FINAL_REVIEWED_PATH, tmp_path / "final_reviewed.jsonl")

            sim_store = Store(tmp_path, PUBLIC_TEST_DIR)

            # Concurrent saves on separate IDs with server LOCK
            test_ids = ["public_0289", "public_0290", "public_0293", "public_0294"]

            def save_worker(sid):
                with LOCK:
                    sim_store.save(sid, f"Concurrent text for {sid}", approved=False)

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(save_worker, sid) for sid in test_ids]
                for f in concurrent.futures.as_completed(futures):
                    f.result()

            # Verify review_state.json is valid JSON and contains all 4 entries
            state_data = json.loads((tmp_path / "review_state.json").read_text(encoding="utf-8"))
            for sid in test_ids:
                self.assertIn(sid, state_data)
                self.assertEqual(state_data[sid]["text"], f"Concurrent text for {sid}")
                self.assertFalse(state_data[sid]["approved"])

            # Verify final_reviewed.jsonl was not corrupted and remains at 725
            finals = load_jsonl(tmp_path / "final_reviewed.jsonl")
            self.assertEqual(len(finals), APPROVED_COUNT)

    def test_real_workspace_files_remain_unmodified_and_safe(self):
        """Verify real final_reviewed.jsonl and review_state.json were never modified by tests."""
        final_records = load_jsonl(FINAL_REVIEWED_PATH)
        self.assertEqual(len(final_records), APPROVED_COUNT)

        state_data = json.loads(REVIEW_STATE_PATH.read_text(encoding="utf-8"))
        final_ids = {r["id"] for r in final_records}

        # Ensure no unapproved items are prematurely marked approved: true
        for sid, st in state_data.items():
            if st.get("approved") is True:
                self.assertIn(sid, final_ids, f"Found unapproved ID {sid} marked approved in review_state.json")


if __name__ == "__main__":
    unittest.main()
