"""A/B model TTS cho P2: cùng 6 câu public (số, thực thể, tự sửa lời, Trung, Nam, điều kiện) qua ElevenLabs v4/v3/multilingual_v2
(giọng nathan, tram) và MiniMax speech-2.8-hd (3 giọng vi). Ghi data/tts_pub_raw/ab_models/ + ab_index.html để nghe.
    /home/thanhlh13/.works/research/.venv/bin/python synth/tts_text_pub/ab_models.py [--plan]
"""
import concurrent.futures as cf, html, importlib.util, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
spec = importlib.util.spec_from_file_location("sv", os.path.join(HERE, "synth_vibi_pub.py")); sv = importlib.util.module_from_spec(spec); spec.loader.exec_module(sv)
OUT = os.path.join(ROOT, "data", "tts_pub_raw", "ab_models"); os.makedirs(OUT, exist_ok=True)
EL_VOICES = [("u8EWWYyBDfXFxHak7WM3", "nathan"), ("6Xq32rHt6oAu98TBDKfr", "tram")]
EL_MODELS = ["eleven_v4", "eleven_v3", "eleven_multilingual_v2"]
MM_VOICES = [("362703657091273", "mm_podcast_host"), ("262184394641600", "mm_male_1_v2"), ("226905123659934", "mm_kindhearted_girl")]


def pick(rows):
    O = [r for r in rows if r["src"] == "orig" and r["kind"] == "final"]
    f = lambda cond: next(r for r in O if cond(r))
    cs = [f(lambda r: "selfcorr" in r["group"]), f(lambda r: "num" in r["group"] and r["region"] == "Bac" and 60 < len(r["spoken"]) < 130),
          f(lambda r: r["region"] == "Trung" and "num" in r["group"]), f(lambda r: r["region"] == "Nam" and 60 < len(r["spoken"]) < 130),
          f(lambda r: any(x[0] == "ent" for x in r["fixes"]) and len(r["spoken"]) < 140), f(lambda r: "cond" in r["group"] and len(r["spoken"]) < 160)]
    seen, out = set(), []
    for r in cs:
        if r["id"] not in seen: out.append(r); seen.add(r["id"])
    return out


def jobs_for(rows):
    J = []
    for r in pick(rows):
        for m in EL_MODELS:
            for vid, tag in EL_VOICES: J.append((r, "elevenlabs", m, vid, f"{m.replace('eleven_', '')}_{tag}", "vi", 1.0))
        for vid, tag in MM_VOICES: J.append((r, "minimax", "speech-2.8-hd", vid, tag, "Vietnamese", 1.0))
    return J


def run(j, ff):
    r, prov, model, vid, tag, lang, mult = j
    wav = os.path.join(OUT, f"{r['id']}__{tag}.wav")
    if os.path.exists(wav): return None
    body = {"text": r["spoken"], "model_id": model, "provider": prov, "language_code": lang, "voice_settings": {"speed": 1.0}}
    t = sv.req(f"/v1/text-to-speech/{vid}", body); tid = t.get("id")
    import time
    for _ in range(120):
        time.sleep(2); d = sv.req(f"/v1/history/{tid}")
        if d.get("status") == "completed": break
        if d.get("status") == "failed": raise RuntimeError(d.get("error"))
    sv.download(d["result"]["audio_url"], os.path.join(OUT, f"{r['id']}__{tag}.mp3"), wav, ff)
    return {"id": r["id"], "tag": tag, "model": model, "chars": d.get("characters_used"), "credits": d.get("credits_deducted"),
            "secs": round((os.path.getsize(wav) - 44) / 32000, 2)}


def main():
    rows = [json.loads(l) for l in open(sv.TEXT, encoding="utf-8")]
    J = jobs_for(rows)
    est = sum(len(r["spoken"]) * (1.0) for r, *_ in J)
    print(f"{len(J)} job, {len({r['id'] for r, *_ in J})} câu, ước ≈ {est:,.0f} credit, còn {sv.credits()}", flush=True)
    if "--plan" in sys.argv: return
    import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
    log, errs = [], []
    def go(j):
        try:
            x = run(j, ff)
            if x: log.append(x)
        except Exception as e:  # noqa: BLE001
            errs.append((j[0]["id"], j[4], str(e)[:150]))
    with cf.ThreadPoolExecutor(5) as ex: list(ex.map(go, J))
    json.dump(log, open(os.path.join(OUT, "ab_log.json"), "w"), ensure_ascii=False, indent=1)
    print("ok", len(log), "lỗi", len(errs), errs[:5], "| credit dùng", sum(x["credits"] or 0 for x in log), "| còn", sv.credits(), flush=True)
    by = {}
    for x in log: by.setdefault(x["id"], []).append(x)
    R = {r["id"]: r for r in rows}
    h = ['<!doctype html><meta charset=utf-8><title>A/B model TTS</title><style>body{font:15px system-ui;max-width:900px;margin:20px auto;padding:0 12px}.c{border-top:2px solid #999;padding:12px 0}.t{margin:3px 0}small{color:#555}audio{height:30px;vertical-align:middle}</style><h2>A/B model TTS — cùng câu, khác model/giọng</h2><p>flash v2.5 đã nghe ở smoke; ở đây: v4 · v3 · multilingual_v2 (ElevenLabs, giọng nathan Trung / tram Nam) và MiniMax speech-2.8-hd (3 giọng vi).</p>']
    for sid, xs in by.items():
        r = R[sid]; h.append(f'<div class=c><div class=t><b>{html.escape(sid)}</b> <small>[{r["region"]} · {r["group"]}]</small></div><div class=t>🗣️ {html.escape(r["spoken"])}</div>')
        for x in sorted(xs, key=lambda z: z["tag"]):
            h.append(f'<div class=t><small>{x["tag"]} · {x["secs"]}s · {x["chars"] / x["secs"]:.1f} ký tự/s · {x["credits"]} credit</small> <audio controls preload=none src="ab_models/{sid}__{x["tag"]}.wav"></audio></div>')
        h.append("</div>")
    open(os.path.join(ROOT, "data", "tts_pub_raw", "ab_index.html"), "w", encoding="utf-8").write("\n".join(h))


if __name__ == "__main__":
    main()
