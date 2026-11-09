"""Unit tests for asr/label_remaining.py state-guided labeling pipeline."""

import unittest
from asr.label_remaining import (
    apply_phonetic_rules,
    clean_token,
    extract_entities_from_state,
    get_shield_mask,
    is_obvious_noise,
    match_and_replace_entities,
    polish_transcript,
    resolve_transcript,
    strip_accents,
)


class TestLabelRemainingUnit(unittest.TestCase):
    def test_strip_accents_and_clean_token(self):
        self.assertEqual(strip_accents("Đêm nhạc gây quỹ"), "Dem nhac gay quy")
        self.assertEqual(clean_token("“Hà Nội,”"), "hà nội")

    def test_is_obvious_noise(self):
        self.assertTrue(is_obvious_noise(""))
        self.assertTrue(is_obvious_noise("   "))
        self.assertTrue(is_obvious_noise("Text with \ufffd replacement"))
        self.assertTrue(is_obvious_noise("Text with 漢字 chinese chars"))
        self.assertTrue(is_obvious_noise("lại lại lại lại lại lặp từ"))
        self.assertFalse(is_obvious_noise("Đặt xe bốn chỗ từ công ty đi khám răng chiều nay"))

    def test_extract_entities_from_state(self):
        mock_state = {
            "contacts": {
                "contacts": [
                    {"name": "Mai Anh", "relation": "đồng nghiệp", "phone": "0901234567"},
                    {"name": "Bác Ba", "relation": "hàng xóm"},
                    {"name": "Bố", "relation": "bố"},
                    {"name": "Sếp Trân", "relation": "sếp"},
                ]
            },
            "banking": {
                "beneficiaries": [{"name": "Trần Văn Tuấn", "bank": "Quân Việt Bank"}],
                "accounts": [{"name": "Tài khoản thanh toán"}],
            },
            "calendar": {
                "events": [{"title": "khám răng", "location": "Nha khoa Hà Nội"}]
            },
            "event_tickets": {
                "events": [{"name": "Đêm thiên văn ngắm sao", "venue": "Nhà hát Lớn"}]
            },
            "alarm": {
                "alarms": [{"label": "đưa con đi học"}]
            },
            "food_delivery": {
                "restaurants": [
                    {"name": "Quán Cơm Tấm", "tags": ["đá bóng", "đồ uống"], "menu": [{"name": "Cơm tấm sườn"}]}
                ]
            },
        }
        entities = extract_entities_from_state(mock_state)
        self.assertIn("Mai Anh", entities["contacts"])
        self.assertIn("Trần Văn Tuấn", entities["beneficiaries"])
        self.assertIn("Quân Việt Bank", entities["banks"])
        self.assertIn("khám răng", entities["events"])
        self.assertIn("Đêm thiên văn ngắm sao", entities["events"])
        self.assertIn("Nhà hát Lớn", entities["venues"])
        self.assertIn("đưa con đi học", entities["alarms"])

        # Integrity checks: no synthetic concatenation or stopwords
        self.assertNotIn("bố Bố", entities["contacts"])
        self.assertNotIn("sếp Sếp Trân", entities["contacts"])
        self.assertNotIn("bố", entities["contacts"])
        self.assertNotIn("đá bóng", entities["food"])
        self.assertNotIn("đồ uống", entities["food"])

    def test_get_shield_mask_protects_critical_spans(self):
        # 1. Phone numbers
        words1 = "số của chị ấy là không chín tám tám bảy bảy sáu sáu năm năm một".split()
        mask1 = get_shield_mask(words1)
        # All digit words from 'không' to 'một' must be shielded
        for w, s in zip(words1[5:], mask1[5:]):
            self.assertTrue(s, f"Digit token '{w}' was not shielded")

        # 2. Currency
        words2 = "Chuyển ba triệu rưỡi nhé".split()
        mask2 = get_shield_mask(words2)
        # 'ba', 'triệu', 'rưỡi' must be shielded
        self.assertTrue(mask2[1])  # ba
        self.assertTrue(mask2[2])  # triệu
        self.assertTrue(mask2[3])  # rưỡi

        # 3. Durations and times
        words3 = "giữ chỗ tám rưỡi sáng ba tiếng nhé".split()
        mask3 = get_shield_mask(words3)
        self.assertTrue(mask3[2])  # tám
        self.assertTrue(mask3[3])  # rưỡi
        self.assertTrue(mask3[5])  # ba
        self.assertTrue(mask3[6])  # tiếng

        # 4. Discourse particles
        words4 = "từ tám giờ sáng à không, chín giờ đi".split()
        mask4 = get_shield_mask(words4)
        self.assertTrue(mask4[4])  # à
        self.assertTrue(mask4[5])  # không,

    def test_tone_sensitive_entity_matching_prevents_corruption(self):
        # Tone differences must prevent matching
        t1 = "Cái ampule tóc mới mua về đã hỏng rồi..."
        res1 = match_and_replace_entities(t1, ["đá bóng"])
        self.assertIn("đã hỏng", res1)
        self.assertNotIn("đá bóng", res1)

        t2 = "Chiều nay ngoài trời sao rồi con?"
        res2 = match_and_replace_entities(t2, ["Chú Năm"])
        self.assertIn("Chiều nay", res2)
        self.assertNotIn("Chú Năm", res2)

        t3 = "Trời, sáng chủ nhật mà nó vô quá trời luôn."
        res3 = match_and_replace_entities(t3, ["Chú Năm"])
        self.assertIn("chủ nhật", res3)
        self.assertNotIn("Chú Năm", res3)

        t4 = "số của chị ấy là không chín tám tám bảy bảy sáu sáu năm năm một."
        res4 = match_and_replace_entities(t4, ["Bác Ba"])
        self.assertIn("bảy bảy", res4)
        self.assertNotIn("Bác Ba", res4)

    def test_apply_phonetic_rules_domain_specific(self):
        # Banking
        state_banking = {"banking": {}}
        t1 = apply_phonetic_rules("Chuyện cho tôi một triệu từ nữ", {}, state_banking)
        self.assertEqual(t1, "Chuyển cho tôi một triệu từ ví")

        # Alarm
        state_alarm = {"alarm": {}}
        t2 = apply_phonetic_rules("Đặt bảo thức lúc 6 giờ sáng", {}, state_alarm)
        self.assertEqual(t2, "Đặt báo thức lúc 6 giờ sáng")

        # Fuel
        state_fuel = {"fuel_price": {}}
        t3 = apply_phonetic_rules("Dạ xong hôm nay thế nào em?", {}, state_fuel)
        self.assertEqual(t3, "Giá xăng hôm nay thế nào em?")

        # Parking
        state_parking = {"parking": {}}
        t4 = apply_phonetic_rules("Dự chỗ cho tôi từ hai giờ chiều", {}, state_parking)
        self.assertEqual(t4, "Giữ chỗ cho tôi từ hai giờ chiều")

        # Tickets
        state_tickets = {"event_tickets": {}}
        t5 = apply_phonetic_rules("Lấy năm bé đi em", {}, state_tickets)
        self.assertEqual(t5, "Lấy năm vé đi em")

    def test_match_and_replace_entities(self):
        text = "Nhà anh muốn đi cái đêm thênhênh ngắm sao tối nay"
        ents = ["Đêm thiên văn ngắm sao"]
        result = match_and_replace_entities(text, ents)
        self.assertEqual(result, "Nhà anh muốn đi cái Đêm thiên văn ngắm sao tối nay")

    def test_polish_transcript(self):
        raw = "   đặt xe   bốn chỗ đi làm ,  nhé . "
        self.assertEqual(polish_transcript(raw), "Đặt xe bốn chỗ đi làm, nhé.")

    def test_resolve_transcript_end_to_end(self):
        mock_state = {
            "calendar": {"events": [{"title": "khám răng chiều nay"}]},
            "ride_hailing": {},
        }
        mock_entities = {"events": ["khám răng chiều nay"]}
        mock_dialogue = {}

        p_flash = "Đặt xe bốn chỗ từ công ty đi tham gia dự này, đặt ngay bây giờ nhé."
        p_alt = "Đặt xe bốn chỗ từ công ty đi khám răng chiều nay, đặt ngay bây giờ nhé."

        res_text, src = resolve_transcript(
            sid="public_0002",
            sample_id="public_0002",
            p_flash=p_flash,
            p_alt=p_alt,
            state=mock_state,
            state_entities=mock_entities,
            dialogue_info=mock_dialogue,
        )
        self.assertIn("khám răng chiều nay", res_text)
        self.assertTrue(res_text.startswith("Đặt xe"))
        self.assertNotIn("dialogue_context_phone_reconstruction", src)
        self.assertIn("qwen_", src)


if __name__ == "__main__":
    unittest.main()
