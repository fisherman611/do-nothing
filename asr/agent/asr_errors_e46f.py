import json, re, sys, unicodedata, collections
ROOT = __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "..", ".."))
sys.path.insert(0, f"{ROOT}/synth/tts_text"); sys.path.insert(0, f"{ROOT}/asr")
try:
    from vi_num import to_spoken
except Exception:
    to_spoken = lambda s: s  # vi_num không có trong repo local; chữ số giữ nguyên
try:
    from agent.clef.asr_noise_b0 import _nums
except Exception as e:
    print("no _nums", e); _nums = lambda s: None

def norm(s):
    s = to_spoken(unicodedata.normalize("NFC", (s or "").lower()))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()

def align(r, h):
    n, m = len(r), len(h)
    D = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): D[i][0] = i
    for j in range(m+1): D[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            D[i][j] = min(D[i-1][j]+1, D[i][j-1]+1, D[i-1][j-1]+(r[i-1] != h[j-1]))
    i, j, ops = n, m, []
    while i or j:
        if i and j and D[i][j] == D[i-1][j-1]+(r[i-1] != h[j-1]):
            ops.append(("=" if r[i-1]==h[j-1] else "S", r[i-1], h[j-1])); i-=1; j-=1
        elif i and D[i][j] == D[i-1][j]+1: ops.append(("D", r[i-1], "")); i-=1
        else: ops.append(("I", "", h[j-1])); j-=1
    return ops[::-1], D[n][m]

def chunks(ops):
    out, cur = [], None
    for op, r, h in ops:
        if op == "=":
            if cur: out.append(cur); cur = None
        else:
            cur = cur or {"ref": [], "hyp": []}
            if r: cur["ref"].append(r)
            if h: cur["hyp"].append(h)
    if cur: out.append(cur)
    return out

FILLER = set("ờ à ừ ạ nhé nhá nha nghen hen đi thôi ơi này ừm ấy mà cơ thế vậy hả hử ha đấy đó nhỉ".split())
NUMW = set("không một hai ba bốn năm sáu bảy tám chín mười mươi mốt lăm trăm nghìn ngàn triệu tỷ lẻ linh rưỡi tư hăm".split())

def cat(c):
    r, h = c["ref"], c["hyp"]
    if not h: return "mất từ (đệm)" if all(w in FILLER for w in r) else "mất từ"
    if not r: return "thừa từ (đệm)" if all(w in FILLER for w in h) else "thừa từ"
    if any(w in NUMW for w in r) or any(w in NUMW for w in h): return "số"
    if all(w in FILLER for w in r+h): return "từ đệm"
    if any(re.search(r"[a-z]", unicodedata.normalize("NFD", w).encode("ascii","ignore").decode()) and not re.search(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", w) and len(w) > 3 for w in r): return "từ nước ngoài/tên Latin"
    return "từ thường"

d = json.load(open(f"{ROOT}/asr/agent/runs/asr/nbest_e46f/nbest_B.json", encoding="utf-8"))
held = {k: v for k, v in d.items() if v["set"] == "heldout"}
# region from train.jsonl lat
reg = {}
try:
    for l in open(f"{ROOT}/data/train+public_test/train/train.jsonl", encoding="utf-8"):
        rr = json.loads(l); c = rr["context"]
        lat = c["location"]["lat"] if isinstance(c, dict) else float(re.search(r"'lat':\s*([\d.]+)", c).group(1))
        reg[rr["id"]] = "Bắc" if lat >= 17.5 else ("Trung" if lat >= 13.5 else "Nam")
except Exception as e:
    print("no train.jsonl:", e)

acc = collections.defaultdict(lambda: [0, 0, 0])  # err, words, n
cats = collections.Counter(); cat_ex = collections.defaultdict(list)
rows = []; num_bad = []; oracle_fix = 0; oracle_better = 0; wrong = 0
sent_err_dist = collections.Counter()
for uid, v in held.items():
    sid = uid.split("/")[1].split("_u")[0]
    region = reg.get(sid, "?")
    ref = norm(v["label"]); best = v["nbest"][0]["text"]; hyp = norm(best)
    ops, e = align(ref.split(), hyp.split()); w = len(ref.split())
    for k in ("tất cả", v["kind"], "vùng " + region):
        a = acc[k]; a[0] += e; a[1] += w; a[2] += 1
    sent_err_dist[min(e, 5)] += 1
    if e:
        wrong += 1
        es = [align(ref.split(), norm(b["text"]).split())[1] for b in v["nbest"][1:]]
        if min(es) == 0: oracle_fix += 1
        if min(es) < e: oracle_better += 1
        for c in chunks(ops):
            k = cat(c); cats[k] += 1
            if len(cat_ex[k]) < 8: cat_ex[k].append(f"{' '.join(c['ref'])!r} → {' '.join(c['hyp'])!r}")
        rows.append((e / w, e, uid, v["kind"], region, v["label"], best))
    try:
        if _nums(v["label"]) != _nums(best): num_bad.append((uid, v["label"], best))
    except Exception: pass

print("== WER e46f (1-best beam5), 405 held-out ==")
for k, (e, w, n) in sorted(acc.items()):
    print(f"{k:12s} n={n:3d}  WER={100*e/w:5.2f}%")
print("câu đúng hết:", 405 - wrong, "| câu sai:", wrong)
print("phân bố số lỗi/câu:", dict(sorted(sent_err_dist.items())))
print(f"oracle beam: bản 2–5 sửa hết {oracle_fix}/{wrong}; tốt hơn 1-best {oracle_better}/{wrong}")
print("\n== Loại lỗi (cụm) ==")
tot = sum(cats.values())
for k, c in cats.most_common():
    print(f"{k:24s} {c:4d} ({100*c/tot:4.1f}%)")
    for ex in cat_ex[k][:6]: print("     ", ex)
print(f"\n== Sai giá trị số: {len(num_bad)} ==")
for u, r, h in num_bad: print(f"  {u}\n    REF {r}\n    HYP {h}")
print("\n== 25 câu tệ nhất ==")
for wer, e, uid, kind, region, lab, hyp in sorted(rows, reverse=True)[:25]:
    print(f"[{wer:.2f} e={e}] {uid} {kind} {region}\n    REF {lab}\n    HYP {hyp}")
