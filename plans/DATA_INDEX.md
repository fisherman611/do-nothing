# Chỉ mục data (cập nhật 12/11/2026 tối)

Đường dẫn tính từ thư mục dự án trên myserver: `/mnt/hungpv/projects/do-nothing`.
Trạng thái: ✅ dùng được · ⚠️ chưa chuẩn, không dùng cho bản nộp, **không đẩy HF** · 📦 đã lên HF.

## 1. Data FC

| Đường dẫn trên myserver | Nội dung | Số mẫu | Dung lượng | Trạng thái | HF |
|---|---|---|---|---|---|
| `asr/agent/clef/data/fmtA_v3/train.jsonl` (+ `.meta.json`, `report.json`) | Data FC v3: hội thoại train BTC + phonetic + alias + iface + history, nhiễu ASR nguồn B0 (PhoWhisper-medium gốc), đã kiểm nhãn (A1, NOTES §1o) | 12 241 ví dụ (11 503 + 738 app lạ) | 166 MB | ✅ sẵn train A2 | 📦 `fisherman611/fc-8099` `fmtA_v3/` |
| `asr/agent/clef/data/fmtA_a4/train.jsonl` (+ meta, report) | Data FC A4: như v3 nhưng nhiễu lấy từ e46f nghe TTS (`make_a4.sh`) | 12 241 ví dụ | 166 MB | ⚠️ bảng nhiễu lẫn lỗi từ giọng Trung ElevenLabs kém → dựng lại theo A6 trước khi train | 📦 `fc-8099` `fmtA_a4/` |
| `asr/agent/clef/data/newapps_q1/{train,val_newapps,test_newapps}.jsonl` | Ví dụ Q1 app tổng hợp (train) + bộ đo app lạ M3b (val/test) | 738 / 95 / 81 | 13 MB | ✅ | 📦 `fc-8099` `newapps_q1/` |

## 2. Nguồn nhiễu ASR cho data FC

| Đường dẫn | Nội dung | Trạng thái | HF |
|---|---|---|---|
| `asr/agent/runs/asr/b0.base_medium.asr.json` | Lỗi nghe của PhoWhisper-medium gốc trên 9 854 câu (audio thật + TTS v2) — nguồn nhiễu của fmtA_v3 | ✅ | 📦 `fisherman611/x-8eb9` |
| `asr/agent/runs/asr/nbest_e46f/nbest_{A,B}.json` | e46f nghe beam 5 (5 bản + điểm): A = TTS v2 xử lý lại 6 198 câu; B = TTS v3 chuẩn 2 300 + phương ngữ 1 333 + held-out 405 | ✅ (số liệu); phần phương ngữ Trung ⚠️ | – |
| `asr/agent/runs/asr/e46f_tts.asr.json`, `e46f_tts.manifest.json`, `e46f_tts.nbest.json` | Chuyển từ N-best: 1-best mỗi câu TTS, manifest uid/text, 5 cách nghe | ✅ / Trung ⚠️ | – |
| `asr/agent/runs/asr/asr_subst_e46f_tts.json`, `asr_subst_a4.json` | Bảng cụm thay thế từ lỗi e46f trên TTS (738 cụm) và bản gộp với v2 (1 078 cụm) | ⚠️ lẫn giọng Trung ElevenLabs | – |
| `asr/agent/runs/asr/b0_e46f.asr.json` | Gộp: câu thật = lỗi medium gốc, câu TTS = lỗi e46f (nguồn của fmtA_a4) | ⚠️ như trên | – |

## 3. Data ASR v3 (⚠️ toàn bộ chưa chuẩn — B4/B4b không qua cổng, NOTES §1z7–§1z8)

| Đường dẫn | Nội dung | Số file | Dung lượng |
|---|---|---|---|
| `data/asr_v3/` | Thư mục DATA cho `asr_train.py`: `manifest.json` (audio thật 4 631 train + 405 val), `tts_manifest_v3.json`, `train → ../zaic_asr/train` (link audio thật) | – | 2,4 GB (cả thư mục) |
| `data/asr_v3/tts_v3/` | TTS v3 giọng chuẩn (DashScope `qwen-audio-3.1-tts-flash`, 3 giọng train + `longanfengyue` cho val), đã xử lý augment_v3 | 2 300 wav | 335 MB |
| `data/asr_v3/tts_v3el/` | Phương ngữ ElevenLabs `eleven_flash_v2_5` (Trung: Ethan, Zara, Nathan, Jay; val Vinh — Nam: Trâm, Tâm, Hùng, Khánh Tú, Nam Bến Tre; val Cẩm Hồng), đã xử lý | 1 333 wav | 215 MB |
| `data/asr_v3/tts_v2p/` | TTS v2 cũ (6 198 câu) xử lý lại theo augment_v3 | 6 198 wav | 998 MB |
| `data/asr_v3/noise_bank/` | Kho nhiễu thật (đoạn im lặng audio train, bỏ held-out) + `eq_dashscope.json`, `eq_eleven.json`, `manifest.json` | 2 850 wav | 45 MB |
| `data/tts_v3_raw/dashscope/`, `elevenlabs/` | Audio TTS v3 **bản gốc** trước xử lý + `synth_log.jsonl`, `synth_el_log.jsonl` | 2 300 + 1 333 wav | 418 MB |
| `data/tts_v3_meta/` | `tts_manifest_v3.json`, log sinh, `el_subset_ids.txt` | 4 file | 7,8 MB |
| `synth/tts_text_v3/tts_text_v3.jsonl` | Text TTS v3: tên/số/động từ (giọng chuẩn) + phương ngữ Trung/Nam, nhãn theo quy ước train, chống rò held-out + public | 3 633 câu | 1 MB |

## 4. Giọng và mẫu nghe

| Đường dẫn | Nội dung |
|---|---|
| `data/voices_regional_meta/selected_voices.json` | 8 + 3 giọng ElevenLabs đã dùng, kèm WER e46f |
| `data/voices_regional_meta/candidates.json`, `rank.json`, `sentences.json`, `index.csv` | 20 ứng viên, bảng chấm độ khó, câu thử, danh sách 165 giọng Trung/Nam |
| `data/voices_regional_meta/samples/difftest/` | 40 câu thử độ khó (20 giọng × 2) |
| `data/voices_regional_meta/samples/smoke_el/` | 16 câu smoke (raw + processed) |
| `data/voices_regional_meta/samples/model_ab/` | Cùng 1 câu Trung, giọng Zara: `eleven_flash_v2_5` / `eleven_v4` / `eleven_v3` |
| `data/voices_regional_meta/samples/probe/`, `probe_elevenlabs/` | Mẫu thử Alibaba (flash/plus/next) và 8 giọng ElevenLabs đầu tiên |

## 5. Không nằm trên myserver

| Ở đâu | Nội dung |
|---|---|
| Modal `thang-nt3092004`, volume `zaic-asr-v3-out` | Model ASR B4 (`lora_ls0.1_ctc0.3_prompt_tts_v3base`) và B4b (`..._e46f_tts_v3e46f`) — không qua cổng, không dùng |
| Modal volume `zaic-asr-v3-data` | Bản sao data ASR v3 + audio public (cho app `lifemate-asr-infer-v3`) |
| Kaggle (private): `zaic-asr-heldout405`, `zaic-voice-difftest`, `zaic-nbest-tts-v2p`, `zaic-nbest-tts-v3-heldout` | Audio held-out / câu thử / TTS cho các kernel đo; không chứa gì từ public |
| HF (đã có từ trước) | xem `plans/REPRO_INVENTORY.md` (model e46f, q-f48e, bộ lọc app, data fmtA_48e, TTS v2 `au-r7k2`) |
