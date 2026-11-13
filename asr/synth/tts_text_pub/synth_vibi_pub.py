"""P2.2: sinh audio cho tts_text_pub.jsonl qua Vibi (ElevenLabs flash cho Trung/Nam, CapCut cho mọi miền).
Key VIBI_API_KEY trong /home/thanhlh13/.works/research/.env (không in). Async: POST → poll /v1/history/{id} → tải mp3 → wav 16 kHz mono
(ffmpeg của imageio-ffmpeg). Cache theo (id, voice): có wav thì bỏ qua → chạy lại được. Chạy bằng python của ~/.works/research/.venv.

    python synth/tts_text_pub/synth_vibi_pub.py --smoke 30                 # 30 câu thử, ghi data/tts_pub_raw/smoke/
    python synth/tts_text_pub/synth_vibi_pub.py --plan                     # chỉ đếm câu/ký tự/credit theo phân bổ
    python synth/tts_text_pub/synth_vibi_pub.py [--only flash|capcut] [--limit N] [--workers 4]
    python synth/tts_text_pub/synth_vibi_pub.py --credits
Log: data/tts_pub_raw/synth_pub_log.jsonl (id, voice, provider, model, chars, credits, secs, task). Lỗi: synth_pub_err.jsonl.
"""
import argparse, collections, concurrent.futures as cf, json, os, random, re, subprocess, sys, threading, time, urllib.error, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEXT = os.path.join(ROOT, "synth", "tts_text_pub", "tts_text_pub.jsonl")
OUT = os.path.join(ROOT, "data", "tts_pub_raw")
API = "https://api.vibi.pro"
KEY = next(m.group(1) for l in open("/home/thanhlh13/.works/research/.env", encoding="utf-8")
           for m in [re.match(r"\s*(?:export\s+)?VIBI_API_KEY\s*=\s*[\"']?([^\"'\s]+)", l)] if m)
H = {"xi-api-key": KEY, "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
     "Accept": "application/json", "Content-Type": "application/json"}
# ElevenLabs (flash v2.5) — 6 giọng train đã chọn 12/11 theo WER e46f; Jay/Cẩm Hồng chỉ val, không dùng ở đây
# 13/11 sau smoke: chỉ giữ giọng đọc không nhanh (tốc độ trung bình ≤ ~17 ký tự/giây; train thật BTC median 13,6, p90 17,2).
# Bỏ vì nhanh: zara 17,9 · tam 17,8 · hung 17,6 · cc_anhdung 21,0 · cc_trungcaha 19,9 · cc_lynam 18,0 · cc_chimai 17,9.
EL = {"Trung": [("EcENCamMHdzIoMysqdsv", "ethan"), ("u8EWWYyBDfXFxHak7WM3", "nathan")],
      "Nam": [("6Xq32rHt6oAu98TBDKfr", "tram")]}
# CapCut (0,01 credit/ký tự) — giữ 4 giọng: hanu 15,1 · sainu 15,9 · duybac 16,7 · namtram 16,9
CC = {"Bac": [("7491607926562361857", "cc_hanu"), ("7491608109043945985", "cc_duybac"), ("7493478233703191041", "cc_namtram")],
      "Nam": [("7491607853086544400", "cc_sainu"), ("7493478233703191041", "cc_namtram")],
      "Trung": [("7491607926562361857", "cc_hanu"), ("7493478233703191041", "cc_namtram")]}
# 13/11 tối (chủ repo): bỏ flash v2.5 và CapCut; dùng eleven_v4 + speed lúc sinh (không kéo giãn sau, vì nghe không tự nhiên).
# 9 giọng có accent (3/miền), speed hiệu chỉnh theo bài thử speed_test (đích ≈ tốc độ thật test: Bắc 14,7 · Nam 14,1 · Trung 13,2 ký tự/giây).
V4 = {"Bac": [("Thắm", 0.85), ("Hien", 0.85), ("Trung Caha", 0.75)],
      "Nam": [("Hung Tran", 0.95), ("Trung - Soft", 0.85), ("Tuyết", 0.85)],
      "Trung": [("Ngan", 0.8), ("Vinh", 0.8), ("Son Tran", 0.75)]}
SAMPLE_FRAC = 0.65          # lấy mẫu phân tầng theo miền → giữ tỉ lệ câu 45/32/23 ≈ test, vừa số dư credit
_lock = threading.Lock()


def v4_voices():
    import unicodedata
    nf = lambda s: unicodedata.normalize("NFC", s)
    d = json.load(open(os.path.join(ROOT, "data", "tts_pub_raw", "voice_screen", "voice_screen.json"), encoding="utf-8"))
    out = {}
    for reg, lst in V4.items():
        out[reg] = []
        for nm, sp in lst:
            x = next(r for r in d["results"] if r["region"] == reg and nf(r["name"]).startswith(nf(nm)))
            out[reg].append((x["voice_id"], "v4_" + nm.split(" - ")[0].replace(" ", "").lower(), sp))
    return out


def req(path, body=None, tries=4):
    for k in range(tries):
        try:
            r = urllib.request.Request(API + path, headers=H, data=json.dumps(body).encode() if body else None, method="POST" if body else "GET")
            return json.load(urllib.request.urlopen(r, timeout=60))
        except urllib.error.HTTPError as e:
            txt = e.read()[:200].decode(errors="replace")
            if e.code in (429, 500, 502, 503, 504) and k < tries - 1: time.sleep(3 * (k + 1)); continue
            raise RuntimeError(f"HTTP {e.code}: {txt}")
        except Exception as e:  # noqa: BLE001
            if k < tries - 1: time.sleep(3 * (k + 1)); continue
            raise


def credits():
    return req("/v1/auth/me").get("credit_balance")


def tts(provider, voice_id, text, speed=1.0):
    if provider == "v4":
        body = {"text": text, "model_id": "eleven_v4", "provider": "elevenlabs", "language_code": "vi", "voice_settings": {"speed": speed}}
    elif provider == "flash":
        body = {"text": text, "model_id": "eleven_flash_v2_5", "provider": "elevenlabs", "language_code": "vi", "voice_settings": {"speed": 1.0}}
    else:
        body = {"text": text, "provider": "capcut", "language_code": "vi", "voice_settings": {"speed": 1.0, "pitch": 0}}
    t = req(f"/v1/text-to-speech/{voice_id}", body)
    tid = t.get("id") or t.get("task_id")
    for _ in range(90):
        time.sleep(2)
        d = req(f"/v1/history/{tid}")
        if d.get("status") == "completed": return d
        if d.get("status") == "failed": raise RuntimeError(f"failed: {d.get('error')}")
    raise RuntimeError("timeout poll")


def download(url, dst_mp3, dst_wav, ff):
    data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": H["User-Agent"]}), timeout=120).read()
    open(dst_mp3, "wb").write(data)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", dst_mp3, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", dst_wav], check=True)


def plan_rows(rows, only=None):
    """v4: lấy mẫu phân tầng SAMPLE_FRAC câu mỗi miền (seed 42), mỗi câu 1 giọng của đúng miền, chia đều 3 giọng.
    → (row, "v4", voice_id, tag, speed). only="legacy" để dùng phân bổ cũ flash/CapCut."""
    if only != "legacy":
        V = v4_voices(); jobs = []
        for reg in ("Bac", "Nam", "Trung"):
            rr = sorted([r for r in rows if r["region"] == reg], key=lambda r: r["id"])
            rr = random.Random(42).sample(rr, round(len(rr) * SAMPLE_FRAC))
            for i, r in enumerate(sorted(rr, key=lambda r: r["id"])):
                vid, tag, sp = V[reg][i % len(V[reg])]
                jobs.append((r, "v4", vid, tag, sp))
        return jobs
    jobs = []
    for r in rows:
        rng = random.Random(r["id"])
        if r["region"] in EL and only != "capcut":
            for vid, tag in rng.sample(EL[r["region"]], min(2, len(EL[r["region"]]))): jobs.append((r, "flash", vid, tag))
        if only != "flash":
            for vid, tag in rng.sample(CC[r["region"]], min(2, len(CC[r["region"]]))): jobs.append((r, "capcut", vid, tag))
    return jobs


def pick_smoke(rows, n):
    rng = random.Random(13)
    num = [r for r in rows if r["src"] == "orig" and "num" in r["group"]]
    ent = [r for r in rows if r["src"] == "orig" and any(f[0] == "ent" for f in r["fixes"])]
    dia = [r for r in rows if r["src"] == "orig" and ("selfcorr" in r["group"] or r["region"] != "Bac")]
    var = [r for r in rows if r["src"] == "variant"]
    k = n // 4
    out, seen = [], set()
    for pool in (ent, num, dia, var):
        for r in rng.sample(pool, min(k, len(pool))):
            if r["id"] not in seen: out.append(r); seen.add(r["id"])
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--plan", action="store_true"); ap.add_argument("--credits", action="store_true")
    ap.add_argument("--only", choices=["flash", "capcut"]); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.credits: print("credit còn:", credits()); return
    rows = [json.loads(l) for l in open(TEXT, encoding="utf-8")]
    sub = "smoke" if a.smoke else "wav"
    outdir = os.path.join(OUT, sub); os.makedirs(outdir, exist_ok=True); os.makedirs(os.path.join(OUT, "mp3"), exist_ok=True)
    jobs = plan_rows(pick_smoke(rows, a.smoke) if a.smoke else rows, a.only)
    if a.limit: jobs = jobs[:a.limit]
    jobs = [j if len(j) == 5 else (*j, 1.0) for j in jobs]
    rate = {"v4": 1.0, "flash": 0.5, "capcut": 0.01}
    est = sum(len(r["spoken"]) * rate[p] for r, p, *_ in jobs)
    print(f"{len(jobs)} job {dict(collections.Counter(p for _, p, *_ in jobs))}, {sum(len(r['spoken']) for r, *_ in jobs):,} ký tự, "
          f"≈ {est:,.0f} credit; credit còn: {credits()}")
    print("  theo miền:", dict(collections.Counter(r["region"] for r, *_ in jobs)), "| theo giọng:", dict(collections.Counter(t for *_, t, _ in jobs)))
    if a.plan: return
    import imageio_ffmpeg; ff = imageio_ffmpeg.get_ffmpeg_exe()
    logp, errp = os.path.join(OUT, f"synth_pub_log{'_smoke' if a.smoke else ''}.jsonl"), os.path.join(OUT, "synth_pub_err.jsonl")
    done = set()
    if os.path.exists(logp):
        for l in open(logp, encoding="utf-8"): x = json.loads(l); done.add((x["id"], x["voice"]))
    todo = [j for j in jobs if (j[0]["id"], j[3]) not in done and not os.path.exists(os.path.join(outdir, f"{j[0]['id']}__{j[3]}.wav"))]
    print(f"đã có {len(jobs) - len(todo)}, cần sinh {len(todo)}")
    stats = {"ok": 0, "err": 0, "credits": 0}

    def run(j):
        r, prov, vid, tag, sp = j
        wav = os.path.join(outdir, f"{r['id']}__{tag}.wav"); mp3 = os.path.join(OUT, "mp3", f"{r['id']}__{tag}.mp3")
        try:
            d = tts(prov, vid, r["spoken"], sp)
            download(d["result"]["audio_url"], mp3, wav, ff)
            secs = round((os.path.getsize(wav) - 44) / 32000, 2)
            rec = {"id": r["id"], "voice": tag, "voice_id": vid, "provider": prov, "model": d.get("model_id"), "chars": d.get("characters_used"),
                   "credits": d.get("credits_deducted"), "speed": sp, "secs": secs, "task": d.get("id"), "region": r["region"], "wav": os.path.relpath(wav, ROOT)}
            with _lock:
                open(logp, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["ok"] += 1; stats["credits"] += rec["credits"] or 0
        except Exception as e:  # noqa: BLE001
            with _lock:
                open(errp, "a", encoding="utf-8").write(json.dumps({"id": r["id"], "voice": tag, "provider": prov, "err": str(e)[:200]}, ensure_ascii=False) + "\n")
                stats["err"] += 1
        with _lock:
            n = stats["ok"] + stats["err"]
            if n % 25 == 0 or n == len(todo): print(f"  {n}/{len(todo)} ok={stats['ok']} err={stats['err']} credit dùng≈{stats['credits']}", flush=True)

    with cf.ThreadPoolExecutor(a.workers) as ex: list(ex.map(run, todo))
    print("xong:", stats, "| credit còn:", credits())


if __name__ == "__main__":
    main()
