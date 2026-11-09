# Project: Public Test ASR Labeling and Review UI Preparation

## Architecture
The system consists of an ASR evaluation and labeling pipeline for the public test set:
1. **Data Source**:
   - `data/train+public_test/public_test/audio/*/*.wav`: 1,933 WAV audio files across 634 test cases.
   - `data/train+public_test/public_test/public_test.jsonl`: Dialogue turns, user history turns, context.
   - `zaic_tools.ToolEnv`: Approved API to access initial application states without reading `*_environment/` directly.
2. **ASR Model Predictions**:
   - `data/results/asr/qwen-cloud/public-test-asr-flash/predictions.jsonl`: 1,933 predictions from `qwen-audio-3.1-asr-flash`.
   - `data/results/asr/qwen-cloud/public-test-references/predictions.jsonl`: 1,933 predictions from `qwen-audio-3.1-asr-flash-message`.
3. **Reviewed & Remaining Labels**:
   - `data/results/asr/qwen-cloud/public-test-asr-flash/final_reviewed.jsonl`: 725 human-verified records.
   - `data/results/asr/qwen-cloud/public-test-asr-flash/remaining_reviewed.jsonl`: Target file with exactly 1,208 records.
4. **Review UI & State**:
   - `asr/review_ui.py`: Standalone Web UI (`http://127.0.0.1:8765`) loading `Store`.
   - `data/results/asr/qwen-cloud/public-test-asr-flash/review_suggestions.jsonl`: Review suggestions loaded into UI for unapproved items.
   - `data/results/asr/qwen-cloud/public-test-asr-flash/review_state.json`: Approval status and draft overrides.

## Feature Inventory
| # | Feature | Description | Milestone | Source | Status |
|---|---------|-------------|-----------|--------|--------|
| 1 | Remaining Audio Identification | Identify all 1,208 audio IDs in `public_test` not present in `final_reviewed.jsonl` | M1 | ORIGINAL_REQUEST §R1 | DONE |
| 2 | Anti-Leak State Extraction | Extract application state entities exclusively via `zaic_tools.ToolEnv.get_state()` | M1 | ORIGINAL_REQUEST §R2 | DONE |
| 3 | State-Guided Context Transcription | Refine acoustic predictions using entity names, history context, token shielding, and NFC tone preservation | M1 | ORIGINAL_REQUEST §R1 | DONE |
| 4 | Output Schema Conformance | Write `remaining_reviewed.jsonl` with required fields (`id`, `sample_id`, `audio`, `audio_text`) matching `final_reviewed.jsonl` | M1 | ORIGINAL_REQUEST §R1 | DONE |
| 5 | Review UI Suggestions Update | Update `review_suggestions.jsonl` with the 1,208 new transcripts so UI displays them under `todo` and `suggested` | M2 | ORIGINAL_REQUEST §R3 | DONE |
| 6 | Review State & Safety Preservation | Ensure `final_reviewed.jsonl` is not overwritten and remains strictly at 725 records | M2 | ORIGINAL_REQUEST §R3 | DONE |
| 7 | Automated E2E Verification Suite | Test suite verifying 1,208 count, 1,933 union coverage, schema, anti-leak compliance, and UI load | E2E | ORIGINAL_REQUEST §Acceptance Criteria | DONE |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Test Suite Track | Build automated verification runner checking 100% coverage, schema, anti-leak AST/runtime audit, and UI payload | none | DONE |
| M1 | State-Guided Labeling of 1,208 Audios | Generate `remaining_reviewed.jsonl` via `ToolEnv.get_state()` entity integration, token shielding, and NFC tone matching | none | DONE |
| M2 | Review UI Data Preparation | Update `review_suggestions.jsonl` and ensure `review_ui.py` loads all 1,208 items in `todo` and `suggested` | M1 | DONE |
| M3 | Final Acceptance & Adversarial Hardening | Run 100% E2E tests, verify all acceptance criteria, and conduct forensic audit | E2E, M1, M2 | DONE |

## Interface Contracts
### Transcription ↔ Review UI
- File: `data/results/asr/qwen-cloud/public-test-asr-flash/remaining_reviewed.jsonl`
- JSONL Schema:
  - `id`: str (e.g. `"public_0002"` or `"public_0006:history:0"`)
  - `sample_id`: str (e.g. `"public_0002"`)
  - `audio`: str (e.g. `"audio/public_0002/public_0002.wav"`)
  - `audio_text`: str (non-empty refined transcript)
  - `original_text`: str (raw prediction from `predictions.jsonl`)
  - `source`: str (truthful source: `qwen_flash_state_aligned`, `qwen_message_state_aligned`, etc.)
  - `reference_type`: str (`"model_refined"`)
  - `human_reviewed`: bool (`false`)

### Anti-Leak Interface Contract
- Import: `from zaic_tools import ToolEnv`
- Invocation: `ToolEnv("public_test", sample_id, root="data/train+public_test").get_state()`
- Restriction: Zero file open/read calls targeting paths containing `_environment` or `state.json`.

## Code Layout
- Implementation scripts: `asr/label_remaining.py`, `asr/review_ui.py`
- Test scripts: `tests/test_remaining_asr_e2e.py`, `asr/tests/test_label_remaining.py`
- Metadata & Coordination: `.agents/teamwork/`
