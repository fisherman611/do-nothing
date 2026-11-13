"""Thử chỉnh tốc độ TTS cho khớp test: cùng câu public (lời đọc P1), 9 giọng đã chọn (3/miền), eleven_v4.
Mỗi (câu, giọng) 4 bản: A speed 1,0 · B speed 0,85 (cách 1) · C = A kéo giãn atempo (cách 2) · D = B kéo giãn (1+2).
Tốc độ đích của C/D = tốc độ thật của chính audio public câu đó (ký tự/giây, bỏ im lặng hai đầu), atempo giới hạn 0,7–1,3.
    /home/thanhlh13/.works/research/.venv/bin/python synth/tts_text_pub/speed_test.py [--plan]
→ data/tts_pub_raw/speed_test/*.wav, speed_test.json, data/tts_pub_raw/speed_test.html
"""
import concurrent.futures as cf, html, importlib.util, json, os, shutil, subprocess, sys, time
import numpy as np, soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
spec = importlib.util.spec_from_file_location("sv", os.path.join(HERE, "synth_vibi_pub.py")); sv = importlib.util.module_from_spec(spec); spec.loader.exec_module(sv)
PT = os.path.join(ROOT, "data", "train+public_test", "public_test")
OUT = os.path.join(ROOT, "data", "tts_pub_raw", "speed_test"); os.makedirs(OUT, exist_ok=True)
VOICES = {"Bac": ["Thắm", "Hien", "Trung Caha"], "Nam": ["Hung Tran", "Trung - Soft", "Tuyết"], "Trung": ["Ngan", "Vinh", "Son Tran"]}
N_SENT = 2


def speech_secs(p):
    x, sr = sf.read(p); x = x if x.ndim == 1 else x.mean(1)
    fr = int(0.02 * sr); n = len(x) // fr
    e = np.sqrt((x[:n * fr].reshape(n, fr) ** 2).mean(1) + 1e-12); db = 20 * np.log10(e)
    idx = np.where(db > max(db.max() - 35, -55))[0]
    return (idx[-1] - idx[0] + 1) * fr / sr if len(idx) else len(x) / sr


def voices():
    d = json.load(open(os.path.join(ROOT, "data", "tts_pub_raw", "voice_screen", "voice_screen.json"), encoding="utf-8"))
    out = {}
    for reg, names in VOICES.items():
        out[reg] = []
        for nm in names:
            import unicodedata
            nf = lambda s: unicodedata.normalize("NFC", s)
            x = next(r for r in d["results"] if r["region"] == reg and nf(r["name"]).startswith(nf(nm)))
            out[reg].append((x["voice_id"], nm.split(" - ")[0].replace(" ", "")))
    return out


def sentences():
    rows = [json.loads(l) for l in open(sv.TEXT, encoding="utf-8")]
    items = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(PT, "public_test.jsonl"), encoding="utf-8")}
    out = {}
    for reg in VOICES:
        c = [r for r in rows if r["region"] == reg and r["src"] == "orig" and r["kind"] == "final" and 60 < len(r["spoken"]) < 115]
        pick = [next(r for r in c if "num" in r["group"]), next(r for r in c if "num" not in r["group"])]
        for r in pick:
            src = os.path.join(PT, items[r["sid"]]["audio"]); dst = os.path.join(OUT, f"real__{r['sid']}.wav"); shutil.copy(src, dst)
            r["real_wav"] = os.path.basename(dst); r["real_cps"] = round(len(r["spoken"]) / speech_secs(dst), 1)
        out[reg] = pick[:N_SENT]
    return out


def gen(text, vid, speed, wav, ff):
    t = sv.req(f"/v1/text-to-speech/{vid}", {"text": text, "model_id": "eleven_v4", "provider": "elevenlabs", "language_code": "vi",
                                            "voice_settings": {"speed": speed}})
    for _ in range(120):
        time.sleep(2); d = sv.req(f"/v1/history/{t['id']}")
        if d.get("status") == "completed": break
        if d.get("status") == "failed": raise RuntimeError(d.get("error"))
    sv.download(d["result"]["audio_url"], wav[:-4] + ".mp3", wav, ff)
    return d.get("credits_deducted") or 0


def stretch(src, dst, cps_now, cps_target, ff):
    f = min(1.3, max(0.7, cps_target / cps_now))
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", src, "-filter:a", f"atempo={f:.3f}", "-ar", "16000", "-ac", "1", dst], check=True)
    return round(f, 3)


def main():
    V, S = voices(), sentences()
    jobs = [(reg, r, vid, tag, sp) for reg in V for r in S[reg] for vid, tag in V[reg] for sp in (1.0, 0.85)]
    print(f"{len(jobs)} job TTS, ước {sum(len(j[1]['spoken']) for j in jobs):,} credit, còn {sv.credits()}")
    for reg in S:
        for r in S[reg]: print(f"  {reg} {r['sid']} thật {r['real_cps']} ký tự/s: {r['spoken']}")
    if "--plan" in sys.argv: return
    import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
    res, errs, cred = [], [], [0]

    def go(j):
        reg, r, vid, tag, sp = j
        base = f"{r['sid']}__{tag}__s{int(sp * 100)}"
        wav = os.path.join(OUT, base + ".wav")
        try:
            cred[0] += gen(r["spoken"], vid, sp, wav, ff)
            cps = round(len(r["spoken"]) / speech_secs(wav), 1)
            st = os.path.join(OUT, base + "_st.wav"); f = stretch(wav, st, cps, r["real_cps"], ff)
            cps2 = round(len(r["spoken"]) / speech_secs(st), 1)
            res.append({"region": reg, "sid": r["sid"], "voice": tag, "speed": sp, "wav": base + ".wav", "cps": cps,
                        "st_wav": base + "_st.wav", "atempo": f, "cps_st": cps2})
        except Exception as e:  # noqa: BLE001
            errs.append((base, str(e)[:150]))

    with cf.ThreadPoolExecutor(5) as ex: list(ex.map(go, jobs))
    json.dump({"sentences": {k: [{x: r[x] for x in ("sid", "spoken", "real_wav", "real_cps")} for r in v] for k, v in S.items()}, "results": res, "errors": errs},
              open(os.path.join(OUT, "speed_test.json"), "w"), ensure_ascii=False, indent=1)
    print(f"ok {len(res)} lỗi {len(errs)} {errs[:3]} | credit dùng {cred[0]} | còn {sv.credits()}")
    lab = {1.0: ("A · speed 1,0 (gốc)", "C · A + kéo giãn (cách 2)"), 0.85: ("B · speed 0,85 (cách 1)", "D · B + kéo giãn (cách 1+2)")}
    h = ['<!doctype html><meta charset=utf-8><title>Thử tốc độ TTS</title><style>body{font:15px system-ui;max-width:1000px;margin:20px auto;padding:0 12px}'
         '.s{border-top:2px solid #888;margin-top:18px;padding-top:8px}.v{margin:8px 0 8px 12px}.r{margin:2px 0}small{color:#555}audio{height:30px;vertical-align:middle}.real{background:#eef6ee;padding:6px}</style>',
         '<h2>Thử chỉnh tốc độ TTS (eleven_v4) so với audio public thật</h2><p>Tốc độ = ký tự/giây, đã bỏ im lặng hai đầu. Đích kéo giãn = tốc độ của chính audio thật câu đó.</p>']
    for reg in S:
        for r in S[reg]:
            h.append(f'<div class=s><b>{reg}</b> · {r["sid"]}<div class=r>🗣️ {html.escape(r["spoken"])}</div>'
                     f'<div class="r real">🎧 <b>Audio thật public</b> · {r["real_cps"]} ký tự/s <audio controls preload=none src="speed_test/{r["real_wav"]}"></audio></div>')
            for vid, tag in V[reg]:
                h.append(f'<div class=v><b>{tag}</b>')
                for x in sorted([x for x in res if x["sid"] == r["sid"] and x["voice"] == tag], key=lambda z: -z["speed"]):
                    a, c = lab[x["speed"]]
                    h.append(f'<div class=r><small>{a} · {x["cps"]} ký tự/s</small> <audio controls preload=none src="speed_test/{x["wav"]}"></audio></div>')
                    h.append(f'<div class=r><small>{c} · atempo {x["atempo"]} → {x["cps_st"]} ký tự/s</small> <audio controls preload=none src="speed_test/{x["st_wav"]}"></audio></div>')
                h.append("</div>")
            h.append("</div>")
    open(os.path.join(ROOT, "data", "tts_pub_raw", "speed_test.html"), "w", encoding="utf-8").write("\n".join(h))


if __name__ == "__main__":
    main()
