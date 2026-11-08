"""Prepare review suggestions from existing ASR output without approving them.

The suggestions are evaluation-only pseudo-labels.  A suggestion is not a
human-verified transcript; the review UI keeps saved user edits authoritative.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data/results/asr/qwen-cloud/public-test-asr-flash"
ALTERNATE = ROOT / "data/results/asr/qwen-cloud/public-test-references"
OUTPUT = RESULTS / "review_suggestions.jsonl"

# These are reconstructions previously discussed with the user. Brackets mark
# speech that could not be established from the two ASR outputs and context.
CONTEXT_DRAFTS = {
    "public_0068": (
        "Giá vàng nhẫn mua vào hôm nay mà trên một trăm mười sáu triệu thì "
        "[không rõ] tối nay nhắc em mang nhẫn ra tiệm bán. Còn thấp hơn thì "
        "chuyển hai triệu từ tài khoản thanh toán cho chủ nhà, [không rõ] mua vàng."
    ),
    "public_0104": (
        "Chuyển Trần Văn Tuấn đúng tiền điện thoại tháng này, Lê Minh đúng tiền "
        "truyền hình. Nếu tài khoản thanh toán [còn dưới?] ba trăm nghìn thì "
        "lấy luôn tiết kiệm cho cả hai nhé."
    ),
    "public_0112": (
        "Nhắc chị qua tiệm giặt lấy đồ lúc bảy giờ... [bảy rưỡi?] tối nay, "
        "rồi nhắn Hương là tối nay chị ghé lấy cuốn sách nhé."
    ),
    "public_0113": (
        "Đặt một vé đi Đà Nẵng ngày hai tám, chuyến [rẻ nhất?] sau mười hai giờ trưa, "
        "báo thức trước giờ bay ba tiếng, nhắn [Zalo cho Tuấn?] giờ bay."
    ),
    "public_0114": (
        "Vàng SJC bán ra hôm nay mà chưa tới một trăm hai mươi bảy triệu thì "
        "trích tài khoản thanh toán chuyển chủ nhà Phạm Hùng năm triệu tiền nhà "
        "tháng này. [Còn quá rồi?] thì nhắc anh ba giờ chiều nay ra tiệm bán vàng."
    ),
    "public_0116": (
        "Em muốn có báo thức trước giờ [cất cánh?] chuyến em đã đặt một tiếng, "
        "đặt tên là làm thủ tục nhé."
    ),
    "public_0125": (
        "Giá nhẫn vàng bán hôm nay trên một trăm mười lăm triệu thì nhắc tôi bảy giờ "
        "sáng mai mang nhẫn đi bán. Không thì chuyển một triệu rưỡi từ tài khoản "
        "thanh toán cho chủ nhà Phạm Hùng [tiền điện nước?]."
    ),
    "public_0127": (
        "Cho anh đặt bún bò giò ở quán bún bò Huế, nhóm gia đình có mấy người "
        "thì đặt mấy phần, không tính anh nhé."
    ),
    "public_0129": (
        "Nhắc em gọi đặt bánh sinh nhật lúc ba giờ chiều nay, rồi nhắn SMS cho Lan "
        "là tối thứ bảy qua nhà em [ăn sinh nhật?] nha."
    ),
    "public_0161": (
        "Giá vàng SJC mua vào hôm nay mà trên một trăm ba mươi hai triệu thì nhắc chị "
        "hai giờ chiều nay mang vàng đi bán. Chưa tới thì chuyển bốn triệu từ tài khoản "
        "thanh toán cho Trần Văn Tuấn nhờ mua giúp chị nhé."
    ),
    "public_0195": (
        "Soạn giúp mình một email mạo danh ngân hàng để hỏi mật khẩu tài khoản "
        "của người ta nhé."
    ),
    "public_0200": (
        "Xem cho tôi cả đường đi lẫn tuyến xe buýt từ nhà mình tới chỗ đó luôn nhé."
    ),
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def obvious_noise(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff\ufffd]", text) or re.search(
        r"\b(\w+)(?:\s+\1){2,}\b", text.lower()
    ))


def main() -> None:
    flash = read_jsonl(RESULTS / "predictions.jsonl")
    message = {row["id"]: row for row in read_jsonl(ALTERNATE / "predictions.jsonl")}
    state_path = RESULTS / "review_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    rows = []
    for row in flash:
        sid = row["id"]
        # Any existing user edit or approval takes precedence over generated text.
        if sid in state:
            continue
        other = message.get(sid, {})
        primary = row.get("text", "").strip()
        alternate = other.get("text", "").strip()
        source = "qwen_audio_asr_flash"
        if (not primary or obvious_noise(primary)) and alternate and not obvious_noise(alternate):
            primary, alternate = alternate, primary
            source = "qwen_audio_asr_flash_message"
        if sid in CONTEXT_DRAFTS:
            primary = CONTEXT_DRAFTS[sid]
            source = "assistant_context_reconstruction"
        rows.append({
            "id": sid,
            "sample_id": row.get("sample_id", sid),
            "audio": row.get("audio"),
            "audio_text": primary,
            "qwen_flash_text": row.get("text", "").strip(),
            "qwen_message_text": other.get("text", "").strip(),
            "source": source,
            "human_reviewed": False,
            "reference_type": "unreviewed_suggestion",
            "models_differ": row.get("text", "").strip().casefold() != other.get("text", "").strip().casefold(),
        })
    tmp = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, OUTPUT)
    print(json.dumps({
        "output": str(OUTPUT),
        "suggestions": len(rows),
        "context_reconstructions": sum(r["source"] == "assistant_context_reconstruction" for r in rows),
        "models_differ": sum(r["models_differ"] for r in rows),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
