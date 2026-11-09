# Test Infrastructure: Public Test ASR Labeling & Review UI Verification

## 1. Overview & Architecture

The End-to-End (E2E) verification infrastructure is built as an opaque-box, requirement-driven test suite implemented in `tests/test_remaining_asr_e2e.py`. It provides continuous validation of the public test ASR labeling pipeline and Review UI data preparation against the formal specifications in `ORIGINAL_REQUEST.md` and `PROJECT.md`.

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 E2E Test Suite Runner                   │
                  │        (tests/test_remaining_asr_e2e.py - 23 Tests)      │
                  └────────────────────────────┬────────────────────────────┘
                                               │
       ┌──────────────────┬────────────────────┼───────────────────┬───────────────────┐
       ▼                  ▼                    ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    Tier 1    │   │    Tier 2    │    │    Tier 3    │    │    Tier 4    │    │    Tier 5    │
│   Feature    │   │  Boundary &  │    │  Partition   │    │  Anti-Leak   │    │  Review UI   │
│   Coverage   │   │ Corner Cases │    │  Integrity   │    │  Compliance  │    │ Payload Prep │
└──────────────┘   └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

The test runner utilizes Python's built-in `unittest` framework and standard library modules (`ast`, `builtins`, `io`, `json`, `pathlib`, `re`, `traceback`, `wave`), ensuring zero external package dependencies while providing deterministic, rapid execution (< 2 seconds).

---

## 2. Test Tiers Specification

### Tier 1: Feature Coverage (`TestTier1FeatureCoverage`)
Enforces the primary deliverables of Milestone 1:
- **`test_t1_1_remaining_reviewed_file_exists`**: Verifies existence of `data/results/asr/qwen-cloud/public-test-asr-flash/remaining_reviewed.jsonl`.
- **`test_t1_2_remaining_reviewed_valid_jsonl_and_count`**: Validates JSON formatting on every line and asserts the exact count of 1,208 records.
- **`test_t1_3_remaining_reviewed_required_fields_present`**: Asserts that all 1,208 records contain required string fields: `id`, `sample_id`, `audio`, and `audio_text`.

### Tier 2: Boundary & Corner Cases (`TestTier2BoundaryAndCornerCases`)
Validates transcript fidelity, data encoding, physical audio correspondence, and schema compatibility:
- **`test_t2_1_audio_text_non_empty_and_valid_string`**: Ensures transcripts are non-empty, non-whitespace, not sentinel placeholders (`"none"`, `"null"`, `"[silence]"`), and free from LLM generation tokens (`<|endoftext|>`, `<|im_start|>`).
- **`test_t2_2_valid_utf8_encoding_and_no_replacement_chars`**: Guarantees strict UTF-8 decoding without replacement characters (`\ufffd`) or corruption.
- **`test_t2_3_audio_file_paths_exist_as_valid_wav`**: Validates that all audio paths are relative, point to existing files under `data/train+public_test/public_test/`, and conform to 16,000 Hz, 1-channel, 16-bit PCM WAV specifications.
- **`test_t2_4_schema_and_id_format_conformance_with_final_reviewed`**: Checks schema parity with `final_reviewed.jsonl`, verifying that main turns follow `public_XXXX` (`id == sample_id`) and history turns follow `public_XXXX:history:K`.
- **`test_t2_5_adversarial_no_cjk_hallucinations_or_control_chars`**: Adversarial check verifying that transcripts contain no Chinese/CJK characters (`[\u4e00-\u9fff]`) or raw ASCII control characters.

### Tier 3: Partition & Cross-Feature Integrity (`TestTier3PartitionIntegrity`)
Enforces dataset partition invariants across reviewed and remaining subsets:
- **`test_t3_1_final_reviewed_count_is_725`**: Protects `final_reviewed.jsonl` from accidental overwriting or truncation, asserting exactly 725 human-reviewed records.
- **`test_t3_2_remaining_reviewed_count_is_1208`**: Confirms exactly 1,208 records in `remaining_reviewed.jsonl`.
- **`test_t3_3_zero_id_overlap_between_final_and_remaining`**: Asserts that `set(final_ids) & set(remaining_ids) == ∅` (strictly 0 duplicate utterances).
- **`test_t3_4_union_coverage_exact_match_1933_public_test_audios`**: Asserts that `set(final_ids) | set(remaining_ids)` exactly matches all 1,933 public test audio files with zero omissions and zero extraneous IDs.
- **`test_t3_5_category_breakdown_partition`**: Audits dialogue turn partitioning:
  - Final reviewed: 271 main turns + 454 history turns = 725 records.
  - Remaining reviewed: 363 main turns + 845 history turns = 1,208 records.
  - Total universe: 634 main turns + 1,299 history turns = 1,933 audio files.

### Tier 4: Anti-Leak Compliance & Environment Access (`TestTier4AntiLeakCompliance`)
Validates benchmark data integrity rules (Requirement R2):
- **`test_t4_1_ast_inspection_no_direct_environment_access`**: Performs static AST traversal over all Python scripts in `asr/` to ensure no calls pass path arguments containing `_environment` or target benchmark `state.json`.
- **`test_t4_2_toolenv_official_state_access`**: Dynamically exercises `zaic_tools.ToolEnv(split, sample_id, root=...).get_state()`, validating authorized retrieval of application state dictionaries (e.g. `calendar`, `contacts`, `alarm`).
- **`test_t4_3_runtime_intercept_blocks_unauthorized_state_file_opens`**: Installs an auditing interceptor on `builtins.open` and `pathlib.Path.open`. Verifies that direct file open calls to `*_environment/*/state.json` outside of `zaic_tools` raise `PermissionError`, while calls originating from `zaic_tools` succeed.
- **`test_t4_4_adversarial_traversal_detection`**: Verifies that path traversal tricks (e.g., `audio/../public_test_environment/public_0002/state.json`) are detected and blocked by the audit hook.

### Tier 5: Review UI Data Preparation & Inspection (`TestTier5ReviewUIPayloadPreparation`)
Validates Review Web UI integration (Requirement R3):
- **`test_t5_1_review_ui_store_loads_all_1933_samples`**: Confirms `asr.review_ui.Store` loads all 1,933 samples from `predictions.jsonl`.
- **`test_t5_2_review_ui_payload_preserves_725_approved`**: Asserts that `Store.payload()` retains exactly 725 approved items matching `final_reviewed.jsonl`.
- **`test_t5_3_review_ui_payload_has_1208_todo_with_suggestions`**: Asserts that all 1,208 remaining items are classified as unapproved (`todo`), each flagged with `has_suggestion=True`.
- **`test_t5_4_ui_payload_transcripts_match_remaining_reviewed`**: Verifies 1:1 text equality between the UI payload and `remaining_reviewed.jsonl` for all 1,208 unreviewed items.
- **`test_t5_5_safety_review_state_unapproved_integrity`**: Confirms that none of the 1,208 remaining audio IDs are marked `"approved": true` in `review_state.json`, preventing accidental promotion during review sessions.
- **`test_t5_6_store_save_isolated_simulation`**: Performs isolated tests in a temporary directory verifying `Store.save` with `approved=False` does not augment `final_reviewed.jsonl` and raises `KeyError` on unknown sample IDs.

---

## 3. How to Run

Execute the test suite using the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_remaining_asr_e2e.py
```

### Run with Verbose Output
```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_remaining_asr_e2e.py -v
```

### Run Specific Tiers
```powershell
# Tier 1 only
.\.venv\Scripts\python.exe -m unittest tests.test_remaining_asr_e2e.TestTier1FeatureCoverage

# Tier 3 only
.\.venv\Scripts\python.exe -m unittest tests.test_remaining_asr_e2e.TestTier3PartitionIntegrity

# Tier 4 only
.\.venv\Scripts\python.exe -m unittest tests.test_remaining_asr_e2e.TestTier4AntiLeakCompliance

# Tier 5 only
.\.venv\Scripts\python.exe -m unittest tests.test_remaining_asr_e2e.TestTier5ReviewUIPayloadPreparation
```

---

## 4. Progressive Testability Lifecycle

| Phase | Milestone | Expected Passing Tests | Pending Implementation |
|---|---|---|---|
| **Phase 0** | Test Infrastructure Ready | 10 passed (T3.1, T4.1-T4.4, T5.1-T5.3, T5.5-T5.6) | 13 failed awaiting `remaining_reviewed.jsonl` |
| **Phase 1** | Milestone 1 (M1) Complete | 22 passed (Tiers 1, 2, 3, 4, and T5 partial) | 1 failed (T5.4 awaiting M2 suggestions sync) |
| **Phase 2** | Milestone 2 (M2) Complete | **23 passed (100% OK)** | None. All acceptance criteria met. |
