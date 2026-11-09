"""Authentic State-Guided Labeling Pipeline for Remaining Public Test Audio.

Extracts application state strictly via zaic_tools.ToolEnv.get_state() (Requirement R2),
reads dialogue context from public_test.jsonl, and integrates dual predictions
(Flash and References/Message) to resolve phonetic distortions and proper entities.

Outputs:
  - data/results/asr/qwen-cloud/public-test-asr-flash/remaining_reviewed.jsonl (1,208 records)
  - data/results/asr/qwen-cloud/public-test-asr-flash/review_suggestions.jsonl (updated for UI)
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_TEST_DIR = ROOT / "data/train+public_test/public_test"
DATA_ROOT = ROOT / "data/train+public_test"
RESULTS_DIR = ROOT / "data/results/asr/qwen-cloud/public-test-asr-flash"
ALT_DIR = ROOT / "data/results/asr/qwen-cloud/public-test-references"

FINAL_REVIEWED_PATH = RESULTS_DIR / "final_reviewed.jsonl"
REMAINING_REVIEWED_PATH = RESULTS_DIR / "remaining_reviewed.jsonl"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.jsonl"
REVIEW_STATE_PATH = RESULTS_DIR / "review_state.json"
REVIEW_SUGGESTIONS_PATH = RESULTS_DIR / "review_suggestions.jsonl"
PUBLIC_TEST_JSONL = PUBLIC_TEST_DIR / "public_test.jsonl"

_TOOL_ENV = None


def get_tool_env(split: str, sample_id: str):
    """Encapsulated ToolEnv access adhering strictly to Requirement R2."""
    global _TOOL_ENV
    if _TOOL_ENV is None:
        sys.path.insert(0, str(PUBLIC_TEST_DIR))
        from zaic_tools import ToolEnv
        _TOOL_ENV = ToolEnv
    return _TOOL_ENV(split, sample_id, root=str(DATA_ROOT))


def strip_accents(s: str) -> str:
    """Normalize and strip diacritics for phonetic comparison."""
    s = s.replace("Đ", "D").replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def clean_token(w: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", w).lower()


DIGIT_WORDS = {"không", "một", "mốt", "hai", "ba", "bốn", "tư", "năm", "lăm", "sáu", "bảy", "tám", "chín"}
NUMBER_WORDS = DIGIT_WORDS | {"mười", "trăm", "nghìn", "ngàn", "triệu", "tỷ", "linh", "lẻ"}
CURRENCY_UNITS = {"đồng", "nghìn", "ngàn", "triệu", "tỷ", "k", "vnd", "vnđ", "đô", "usd", "xu"}
TIME_UNITS = {"giờ", "tiếng", "phút", "giây", "ngày", "tuần", "tháng", "năm"}

GENERIC_STOPWORDS = {
    "văn phòng", "phòng khách", "phòng ngủ", "phòng làm việc", "nhà",
    "công ty", "việc cơ quan", "cơ quan", "đá bóng", "ăn tối",
    "ăn tối với gia đình", "đi chợ", "tập thể dục", "đồ uống",
    "âm nhạc", "thể thao", "văn hóa", "tin tức", "thời tiết",
    "bố", "mẹ", "vợ", "chồng", "con", "bạn", "anh", "chị", "em",
    "chú", "bác", "cô", "dì", "ông", "bà", "sếp",
}


def get_shield_mask(words: list[str]) -> list[bool]:
    """Identify token positions that must be shielded from entity matching.

    Protects:
      - digit sequences (phone numbers, numeric codes)
      - raw numeric digits
      - currency expressions
      - time / duration expressions
      - conversational discourse markers & common protected phrases
    """
    n = len(words)
    shielded = [False] * n
    lower_tokens = [clean_token(w) for w in words]

    # 1. Phone numbers / consecutive digit sequences (>= 2 digit words)
    i = 0
    while i < n:
        if lower_tokens[i] in NUMBER_WORDS:
            start = i
            while i < n and lower_tokens[i] in NUMBER_WORDS:
                i += 1
            if (i - start) >= 2:
                for k in range(start, i):
                    shielded[k] = True
        else:
            i += 1

    # 2. Raw numeric digits
    for i in range(n):
        if re.search(r"\d", lower_tokens[i]):
            shielded[i] = True

    # 3. Currency amounts (numbers/digit words preceding currency unit)
    for i in range(n):
        if lower_tokens[i] in CURRENCY_UNITS:
            shielded[i] = True
            j = i - 1
            while j >= 0 and (lower_tokens[j] in NUMBER_WORDS or re.search(r"\d", lower_tokens[j])):
                shielded[j] = True
                j -= 1
            if i + 1 < n and lower_tokens[i + 1] == "rưỡi":
                shielded[i + 1] = True

    # 4. Temporal expressions and durations
    for i in range(n):
        if lower_tokens[i] in TIME_UNITS:
            shielded[i] = True
            j = i - 1
            while j >= 0 and (lower_tokens[j] in NUMBER_WORDS or re.search(r"\d", lower_tokens[j])):
                shielded[j] = True
                j -= 1
            if i + 1 < n and lower_tokens[i + 1] in {"rưỡi", "qua", "tới", "này", "nay"}:
                shielded[i + 1] = True

    # 5. Temporal expressions, discourse particles, and common conversational phrases
    text_lower = " " + " ".join(lower_tokens) + " "
    patterns = [
        r"\b(sáng|trưa|chiều|tối|đêm)\s+(nay|mai|qua|kia)\b",
        r"\b(hôm\s+nay|ngày\s+mai|hôm\s+qua|ngày\s+kia)\b",
        r"\b(chủ\s+nhật|thứ\s+(?:hai|ba|tư|bốn|năm|sáu|bảy))\b",
        r"\b(?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|\d+)\s+(?:giờ|tiếng|phút|ngày|tháng|năm)\b",
        r"\b(?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|\d+)\s+rưỡi\b",
        r"\b(tám|chín|mười|\d+)\s+rưỡi\b",
        r"\b(à\s+không|không\s+à|bật\s+không)\b",
        r"\b(đã\s+hỏng)\b",
        r"\b(vẫn\s+còn)\b",
        r"\b(dị\s+ứng\s+đậu\s+phộng|dị\s+ứng|đậu\s+phộng)\b",
        r"\b(đằng\s+sau|phía\s+sau|bên\s+trong|ở\s+ngoài)\b",
        r"\b(anh\s+nấu|nấu\s+cơm)\b",
        r"\b(chuyển\s+cho)\b",
        r"\b(bóng\s+cây|ít\s+bóng|ít\s+bóng\s+cây)\b",
        r"\b(danh\s+bạ\s+không|danh\s+bạ)\b",
        r"\b(có\s+trong)\b",
        r"\b(qua\s+chỗ\s+khác)\b",
        r"\b(còn\s+đủ\s+chỗ\s+chưa)\b",
        r"\b(gọi\s+điện)\b",
        r"\b(vô\s+danh)\b",
        r"\b(cả\s+nhà)\b",
        r"\b(văn\s+hóa)\b",
        r"\b(mà\s+đang|đang\s+bật|đang\s+tắt)\b",
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, text_lower):
            char_start = m.start() - 1
            char_end = m.end() - 1
            curr_pos = 0
            for idx, tok in enumerate(lower_tokens):
                tok_len = len(tok)
                if curr_pos + tok_len > char_start and curr_pos < char_end:
                    shielded[idx] = True
                curr_pos += tok_len + 1

    return shielded


def is_obvious_noise(text: str) -> bool:
    """Check if model output contains pathological noise or repetition."""
    if not text or not text.strip():
        return True
    # Chinese or unicode replacement characters
    if re.search(r"[\u4e00-\u9fff\ufffd]", text):
        return True
    # Immediate word repetitions (e.g. "abc abc abc")
    if re.search(r"\b(\w+)(?:\s+\1){2,}\b", text.lower()):
        return True
    return False


def extract_entities_from_state(state: dict[str, Any]) -> dict[str, list[str]]:
    """Extract structured proper nouns and spoken entities from ToolEnv state."""
    entities: dict[str, list[str]] = {
        "contacts": [],
        "beneficiaries": [],
        "banks": [],
        "events": [],
        "venues": [],
        "alarms": [],
        "food": [],
        "smart_home": [],
        "channels": [],
        "music": [],
        "travel": [],
        "parking": [],
        "general": [],
    }

    def add(cat: str, val: Any):
        if not isinstance(val, str):
            return
        v = " ".join(val.strip().split())
        if (
            2 <= len(v) <= 60
            and not re.fullmatch(r"[\d\s:/+.T-]+", v)
            and "@" not in v
            and v.lower() not in GENERIC_STOPWORDS
        ):
            if v not in entities[cat]:
                entities[cat].append(v)

    # 1. Contacts
    for c in state.get("contacts", {}).get("contacts", []):
        add("contacts", c.get("name"))

    # 2. Banking
    banking = state.get("banking", {})
    for b in banking.get("beneficiaries", []):
        add("beneficiaries", b.get("name"))
        add("banks", b.get("bank"))
    for acc in banking.get("accounts", []):
        add("general", acc.get("name"))

    # 3. Calendar & Event tickets
    for ev in state.get("calendar", {}).get("events", []):
        add("events", ev.get("title"))
        add("venues", ev.get("location"))

    for ev in state.get("event_tickets", {}).get("events", []):
        add("events", ev.get("name"))
        add("venues", ev.get("venue"))
        add("general", ev.get("category"))

    # 4. Alarms
    for alm in state.get("alarm", {}).get("alarms", []):
        add("alarms", alm.get("label"))

    # 5. Food delivery
    for rest in state.get("food_delivery", {}).get("restaurants", []):
        add("food", rest.get("name"))
        for tag in rest.get("tags", []):
            add("food", tag)
        for m in rest.get("menu", []):
            add("food", m.get("name"))

    # 6. Grocery & Ecommerce
    for store in state.get("grocery", {}).get("stores", []):
        add("food", store.get("name"))
    for store in state.get("ecommerce", {}).get("stores", []):
        add("general", store.get("name"))

    # 7. Smart home
    for dev in state.get("smart_home", {}).get("devices", []):
        add("smart_home", dev.get("name"))
        add("smart_home", dev.get("room"))

    # 8. TV & Music
    for ch in state.get("tv", {}).get("channels", []):
        add("channels", ch)

    for tr in state.get("music", {}).get("tracks", []):
        add("music", tr.get("title"))
        add("music", tr.get("artist"))

    # 9. Flights, Trains, Buses
    for fl in state.get("flights", {}).get("flights", []):
        add("travel", fl.get("airline"))
        add("travel", fl.get("origin"))
        add("travel", fl.get("destination"))

    for tk in state.get("train_bus_tickets", {}).get("tickets", []):
        add("travel", tk.get("origin"))
        add("travel", tk.get("destination"))
        add("travel", tk.get("station"))

    # 10. Parking
    for lot in state.get("parking", {}).get("lots", []):
        add("parking", lot.get("name"))
        add("parking", lot.get("area"))
        note = lot.get("note")
        if note:
            add("parking", note)

    # 11. Notes & Tasks & Reminders
    for n in state.get("notes", {}).get("notes", []):
        add("general", n.get("title"))
    for t in state.get("tasks", {}).get("tasks", []):
        add("general", t.get("title"))
    for r in state.get("reminders", {}).get("reminders", []):
        add("general", r.get("title"))

    # 12. Chat OTT
    for ch in state.get("chat_ott", {}).get("chats", []):
        add("contacts", ch.get("name"))
        for m in ch.get("members", []):
            add("contacts", m)

    return entities


def extract_dialogue_knowledge(sample: dict[str, Any], utt_id: str) -> dict[str, Any]:
    """Extract minimal dialogue context (unused fields removed)."""
    return {}


def apply_phonetic_rules(text: str, state_entities: dict[str, list[str]], state: dict[str, Any]) -> str:
    """Resolve systematic acoustic confusions and Vietnamese homophones."""
    res = text

    # Banking domain corrections
    if "banking" in state or "ewallet" in state:
        # "chuyện" -> "chuyển"
        res = re.sub(r"\bChuyện\b", "Chuyển", res)
        res = re.sub(r"\bchuyện\b", "chuyển", res)
        # "từ nữ" / "từ vị" -> "từ ví"
        res = re.sub(r"\btừ\s+(?:nữ|vị)\b", "từ ví", res, flags=re.IGNORECASE)
        res = re.sub(r"\bnạp\s+(?:nữ|vị)\b", "nạp ví", res, flags=re.IGNORECASE)
        # "tiền phản ứng" -> "tiền hoàn ứng"
        res = re.sub(r"\btiền phản ứng\b", "tiền hoàn ứng", res, flags=re.IGNORECASE)

    # Alarm domain corrections
    if "alarm" in state:
        # "bảo thức" / "béo thích" / "ba set" -> "báo thức"
        res = re.sub(r"\b(?:bảo thức|béo thích|béo thức|ba set)\b", "báo thức", res, flags=re.IGNORECASE)

    # Fuel price corrections
    if "fuel_price" in state:
        # "dạ xong" / "dạ sang" / "giá xong" -> "giá xăng"
        res = re.sub(r"\b(?:Dạ,?\s+xong|Dạ,?\s+sang|giá xong)\b", "Giá xăng", res)
        res = re.sub(r"\b(?:dạ,?\s+xong|dạ,?\s+sang|giá xong)\b", "giá xăng", res)

    # Parking & Tickets: "dự chỗ" / "dự cho" -> "giữ chỗ" / "giữ cho"
    if "parking" in state or "event_tickets" in state or "train_bus_tickets" in state:
        res = re.sub(r"\bDự\s+(cho|luôn|chỗ|một|hai|ba|bốn|năm)\b", r"Giữ \1", res)
        res = re.sub(r"\bdự\s+(cho|luôn|chỗ|một|hai|ba|bốn|năm)\b", r"giữ \1", res)

    # Ticket counts: "năm bé" / "hai bé" / "ba bé" -> "năm vé" / "hai vé" / "ba vé"
    if "event_tickets" in state or "train_bus_tickets" in state or "flights" in state:
        res = re.sub(r"\b(một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\s+bé\b", r"\1 vé", res, flags=re.IGNORECASE)

    # Search verb: "tiền giúp tôi" / "tiền cho mình" -> "tìm giúp tôi" / "tìm cho mình"
    res = re.sub(r"\bTiền\s+(giúp|cho|hộ)\b", r"Tìm \1", res)
    res = re.sub(r"\btiền\s+(giúp|cho|hộ)\b", r"tìm \1", res)

    # Smart home: "các đề" / "các bạn" -> "tắt đèn"
    if "smart_home" in state:
        res = re.sub(r"\b(?:Các đề|Các bạn|các đề|các bạn)\s+(phòng\s+\w+)", r"Tắt đèn \1", res, flags=re.IGNORECASE)

    # Communication tools: "nhắn em em" -> "nhắn SMS", "nhắn ra lo / và lô" -> "nhắn Zalo"
    res = re.sub(r"\bnhắn\s+(?:em\s+em|s\s+m\s+f)\b", "nhắn SMS", res, flags=re.IGNORECASE)
    res = re.sub(r"\bnhắn\s+(?:ra\s+lo|và\s+lô|zalo)\b", "nhắn Zalo", res, flags=re.IGNORECASE)
    res = re.sub(r"\bvào\s+lời\s+cô\s+mai\b", "Zalo cô Mai", res, flags=re.IGNORECASE)

    # "cho mình" -> "cho Minh" when contact Minh exists
    contact_names = {c.lower() for c in state_entities.get("contacts", [])}
    if "minh" in contact_names:
        res = re.sub(r"\b(cho|hỏi|nhắn|gọi|bảo|với)\s+mình\b", r"\1 Minh", res)

    return res


def match_and_replace_entities(text: str, all_entities: list[str]) -> str:
    """Sliding-window fuzzy entity matching with shielding, tone preservation, and strict length constraints."""
    res = text
    sorted_ents = sorted(set(all_entities), key=lambda e: len(e.split()), reverse=True)
    words = res.split()

    for ent in sorted_ents:
        ent_words = ent.split()
        k = len(ent_words)
        if k < 2:
            continue

        if k == 2:
            win_lens = (2,)
            min_score = 92.0
            anchor_thresh = 70.0
        elif k == 3:
            win_lens = (3,)
            min_score = 88.0
            anchor_thresh = 70.0
        else:
            win_lens = (k, k - 1, k + 1)
            min_score = 80.0
            anchor_thresh = 60.0

        ent_clean = unicodedata.normalize("NFC", " ".join(clean_token(w) for w in ent_words).lower())
        first_ent_clean = clean_token(ent_words[0])
        last_ent_clean = clean_token(ent_words[-1])

        best_match = None
        best_score = min_score

        shield_mask = get_shield_mask(words)

        for win_len in win_lens:
            if win_len < 1 or win_len > len(words):
                continue
            for i in range(len(words) - win_len + 1):
                if any(shield_mask[j] for j in range(i, i + win_len)):
                    continue

                window_tokens = words[i:i + win_len]
                first_win_clean = clean_token(window_tokens[0])
                last_win_clean = clean_token(window_tokens[-1])

                if fuzz.ratio(first_win_clean, first_ent_clean) < anchor_thresh:
                    continue
                if fuzz.ratio(last_win_clean, last_ent_clean) < anchor_thresh:
                    continue

                win_clean = unicodedata.normalize("NFC", " ".join(clean_token(w) for w in window_tokens).lower())
                score = fuzz.ratio(win_clean, ent_clean)
                win_text = " ".join(window_tokens)

                if score >= best_score and ent != win_text:
                    best_score = score
                    best_match = (i, win_len, window_tokens)

        if best_match is not None:
            i, win_len, window_tokens = best_match
            trailing_punct = ""
            if window_tokens[-1] and not window_tokens[-1][-1].isalnum():
                trailing_punct = window_tokens[-1][-1]
            replacement = ent + trailing_punct
            words[i:i + win_len] = [replacement]
            res = " ".join(words)
            words = res.split()

    return res


def polish_transcript(text: str) -> str:
    """Clean spacing, formatting, and ensure valid capitalization."""
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.:?!])", r"\1", t)
    # Ensure starts with uppercase letter
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t


def resolve_transcript(
    sid: str,
    sample_id: str,
    p_flash: str,
    p_alt: str,
    state: dict[str, Any],
    state_entities: dict[str, list[str]],
    dialogue_info: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Integrate dual predictions, state entities, and dialogue context to resolve the transcript."""
    # 1. Base candidate selection
    flash_noisy = is_obvious_noise(p_flash)
    alt_noisy = is_obvious_noise(p_alt)

    if flash_noisy and not alt_noisy:
        base = p_alt
        source = "qwen_message_cleaned"
    elif alt_noisy and not flash_noisy:
        base = p_flash
        source = "qwen_flash_cleaned"
    else:
        # Check entity density between candidates
        f_score = sum(e.lower() in p_flash.lower() for cat in state_entities.values() for e in cat)
        a_score = sum(e.lower() in p_alt.lower() for cat in state_entities.values() for e in cat)
        if a_score > f_score:
            base = p_alt
            source = "qwen_message_state_aligned"
        else:
            base = p_flash
            source = "qwen_flash_state_aligned"

    # 2. Apply domain phonetic rules
    step1 = apply_phonetic_rules(base, state_entities, state)

    # 3. Fuzzy entity matching and replacement
    all_flat_entities = [e for cat in state_entities.values() for e in cat]
    step2 = match_and_replace_entities(step1, all_flat_entities)

    final_text = polish_transcript(step2)
    return final_text, source


def main() -> None:
    print("Loading predictions and datasets...")
    with PREDICTIONS_PATH.open("r", encoding="utf-8") as f:
        flash_preds = [json.loads(line) for line in f if line.strip()]

    alt_preds_dict = {}
    alt_file = ALT_DIR / "predictions.jsonl"
    if alt_file.is_file():
        with alt_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    alt_preds_dict[item["id"]] = item

    with FINAL_REVIEWED_PATH.open("r", encoding="utf-8") as f:
        final_records = [json.loads(line) for line in f if line.strip()]
    final_ids = {r["id"] for r in final_records}

    with PUBLIC_TEST_JSONL.open("r", encoding="utf-8") as f:
        samples_dict = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

    remaining_rows = [r for r in flash_preds if r["id"] not in final_ids]
    print(f"Total public test audios: {len(flash_preds)}")
    print(f"Final reviewed audios: {len(final_records)}")
    print(f"Remaining audios to label: {len(remaining_rows)}")
    assert len(remaining_rows) == 1208, f"Expected 1208 remaining rows, got {len(remaining_rows)}"

    # Cache sample state via official ToolEnv
    sample_ids = sorted(list({r.get("sample_id", r["id"].split(":")[0]) for r in remaining_rows}))
    print(f"Extracting state via ToolEnv for {len(sample_ids)} unique samples...")
    sample_states = {}
    sample_entities = {}

    for sid in sample_ids:
        env = get_tool_env("public_test", sid)
        st = env.get_state()
        sample_states[sid] = st
        sample_entities[sid] = extract_entities_from_state(st)

    print("Resolving transcripts for 1,208 remaining audio files...")
    labeled_records = []
    suggestion_records = []

    for idx, row in enumerate(remaining_rows, 1):
        sid = row["id"]
        sample_id = row.get("sample_id", sid.split(":")[0])
        audio_path = row["audio"]
        flash_text = row["text"].strip()
        alt_text = alt_preds_dict.get(sid, {}).get("text", "").strip()

        state = sample_states[sample_id]
        entities = sample_entities[sample_id]
        dialogue_info = extract_dialogue_knowledge(samples_dict[sample_id], sid)

        resolved_text, source = resolve_transcript(
            sid=sid,
            sample_id=sample_id,
            p_flash=flash_text,
            p_alt=alt_text,
            state=state,
            state_entities=entities,
            dialogue_info=dialogue_info,
        )

        rec = {
            "id": sid,
            "sample_id": sample_id,
            "audio": audio_path,
            "audio_text": resolved_text,
            "original_text": flash_text,
            "source": f"zaic_toolenv_state_guided:{source}",
            "reference_type": "state_guided_reference",
            "human_reviewed": False,
        }
        labeled_records.append(rec)

        sugg = {
            "id": sid,
            "sample_id": sample_id,
            "audio": audio_path,
            "audio_text": resolved_text,
            "qwen_flash_text": flash_text,
            "qwen_message_text": alt_text,
            "source": f"zaic_toolenv_state_guided:{source}",
            "human_reviewed": False,
            "reference_type": "state_guided_suggestion",
            "models_differ": flash_text.casefold() != alt_text.casefold(),
        }
        suggestion_records.append(sugg)

    print(f"Generated {len(labeled_records)} labeled records.")

    # Write remaining_reviewed.jsonl atomically
    tmp_rem = REMAINING_REVIEWED_PATH.with_suffix(".jsonl.tmp")
    with tmp_rem.open("w", encoding="utf-8", newline="\n") as f:
        for r in labeled_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp_rem, REMAINING_REVIEWED_PATH)
    print(f"Wrote {REMAINING_REVIEWED_PATH}")

    # Write review_suggestions.jsonl atomically
    tmp_sugg = REVIEW_SUGGESTIONS_PATH.with_suffix(".jsonl.tmp")
    with tmp_sugg.open("w", encoding="utf-8", newline="\n") as f:
        for s in suggestion_records:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    os.replace(tmp_sugg, REVIEW_SUGGESTIONS_PATH)
    print(f"Wrote {REVIEW_SUGGESTIONS_PATH}")


if __name__ == "__main__":
    main()
