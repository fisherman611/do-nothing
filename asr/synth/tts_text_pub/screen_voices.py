"""Lọc giọng ElevenLabs có `accent` (northern/central/southern) cho P2: mỗi miền ~7 ứng viên (dùng nhiều nhất, cân nam/nữ),
cùng 1 câu P1 của miền đó, đọc bằng eleven_v4; 2 giọng mỗi miền đọc thêm bằng eleven_v4_turbo để so chất lượng.
    /home/thanhlh13/.works/research/.venv/bin/python synth/tts_text_pub/screen_voices.py [--plan]
→ data/tts_pub_raw/voice_screen/*.wav, voice_screen.json, data/tts_pub_raw/voice_screen.html
"""
import concurrent.futures as cf, html, importlib.util, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
spec = importlib.util.spec_from_file_location("sv", os.path.join(HERE, "synth_vibi_pub.py")); sv = importlib.util.module_from_spec(spec); spec.loader.exec_module(sv)
OUT = os.path.join(ROOT, "data", "tts_pub_raw", "voice_screen"); os.makedirs(OUT, exist_ok=True)
ACC = {"Bac": "northern", "Nam": "southern", "Trung": "central"}
PER = 7


def candidates():
    V = json.load(open(os.path.join(ROOT, "data", "voices_vi_by_accent.json"), encoding="utf-8"))
    out = {}
    for reg, acc in ACC.items():
        vs = sorted(V[acc], key=lambda v: -(v.get("usage_character_count_1y") or 0))
        m = [v for v in vs if v.get("gender") == "male"]; f = [v for v in vs if v.get("gender") == "female"]
        pick = (m[:4] + f[:3]) if len(f) >= 3 else (m[:PER - len(f[:3])] + f[:3])
        out[reg] = pick[:PER]
    return out


def sentence(reg):
    rows = [json.loads(l) for l in open(sv.TEXT, encoding="utf-8")]
    return next(r for r in rows if r["region"] == reg and r["src"] == "orig" and r["kind"] == "final" and "num" in r["group"]
                and "selfcorr" in r["group"] or (r["region"] == reg and r["src"] == "orig" and r["kind"] == "final" and "num" in r["group"] and 70 < len(r["spoken"]) < 110))


def main():
    C = candidates(); S = {reg: sentence(reg) for reg in ACC}
    jobs = []
    for reg, vs in C.items():
        for i, v in enumerate(vs):
            jobs.append((reg, v, "eleven_v4"))
            if i < 2: jobs.append((reg, v, "eleven_v4_turbo"))
    est = sum(len(S[r]["spoken"]) * (1 if m == "eleven_v4" else 0.5) for r, _, m in jobs)
    print(f"{len(jobs)} job, ước {est:,.0f} credit, còn {sv.credits()}", flush=True)
    for reg in ACC: print(f"  {reg}: {S[reg]['spoken']}  |  " + ", ".join(f"{v['name'].split(' - ')[0][:18]}({v.get('gender', '?')[0]})" for v in C[reg]))
    if "--plan" in sys.argv: return
    import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
    res, errs = [], []

    def go(j):
        reg, v, model = j
        tag = f"{reg}__{v['voice_id']}__{model.replace('eleven_', '')}"
        wav = os.path.join(OUT, tag + ".wav")
        try:
            t = sv.req(f"/v1/text-to-speech/{v['voice_id']}", {"text": S[reg]["spoken"], "model_id": model, "provider": "elevenlabs",
                                                              "language_code": "vi", "voice_settings": {"speed": 1.0}})
            for _ in range(120):
                time.sleep(2); d = sv.req(f"/v1/history/{t['id']}")
                if d.get("status") == "completed": break
                if d.get("status") == "failed": raise RuntimeError(d.get("error"))
            sv.download(d["result"]["audio_url"], os.path.join(OUT, tag + ".mp3"), wav, ff)
            secs = round((os.path.getsize(wav) - 44) / 32000, 2)
            res.append({"region": reg, "voice_id": v["voice_id"], "name": v["name"], "gender": v.get("gender"), "age": v.get("age"),
                        "model": model, "secs": secs, "cps": round(len(S[reg]["spoken"]) / secs, 1), "credits": d.get("credits_deducted"), "wav": tag + ".wav"})
        except Exception as e:  # noqa: BLE001
            errs.append((tag, str(e)[:150]))

    with cf.ThreadPoolExecutor(5) as ex: list(ex.map(go, jobs))
    json.dump({"sentences": {k: v["spoken"] for k, v in S.items()}, "results": res, "errors": errs},
              open(os.path.join(OUT, "voice_screen.json"), "w"), ensure_ascii=False, indent=1)
    print(f"ok {len(res)} lỗi {len(errs)} {errs[:3]} | credit dùng {sum(x['credits'] or 0 for x in res)} | còn {sv.credits()}", flush=True)
    h = ['<!doctype html><meta charset=utf-8><title>Lọc giọng theo miền</title><style>body{font:15px system-ui;max-width:950px;margin:20px auto;padding:0 12px}h3{margin-top:28px}.r{margin:4px 0}.fast{color:#b00}small{color:#555}audio{height:30px;vertical-align:middle}</style>',
         '<h2>Lọc giọng ElevenLabs theo miền (eleven_v4; 2 giọng/miền có thêm v4_turbo)</h2><p>Tiếng nói thật BTC: trung vị 13,6 ký tự/giây. <span class=fast>Đỏ</span> = &gt; 17 ký tự/giây (nhanh).</p>']
    for reg in ACC:
        h.append(f"<h3>{reg} ({ACC[reg]})</h3><p>🗣️ {html.escape(S[reg]['spoken'])}</p>")
        for x in sorted([r for r in res if r["region"] == reg], key=lambda z: (z["name"], z["model"])):
            cls = " class=fast" if x["cps"] > 17 else ""
            h.append(f'<div class=r><small{cls}>{html.escape(x["name"][:45])} · {x["gender"]}/{x["age"]} · {x["model"].replace("eleven_", "")} · {x["cps"]} ký tự/s</small> '
                     f'<audio controls preload=none src="voice_screen/{x["wav"]}"></audio></div>')
    open(os.path.join(ROOT, "data", "tts_pub_raw", "voice_screen.html"), "w", encoding="utf-8").write("\n".join(h))


if __name__ == "__main__":
    main()
