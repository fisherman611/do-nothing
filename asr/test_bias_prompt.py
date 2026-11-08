"""Kiểm tra bias_prompt: prompt lúc test chỉ đọc context/history/transcript model; độ dài trong ngân sách.
    python test_bias_prompt.py <root data> [eval_v1.json]"""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer
import bias_prompt as bp

root = sys.argv[1]
tok = AutoTokenizer.from_pretrained("vinai/PhoWhisper-large", revision="b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb")
here = os.path.dirname(os.path.abspath(__file__))
lex = json.load(open(f"{here}/lexicon_v2.json"))
man = json.load(open(f"{here}/manifest_v2.json"))
pool = {}
for app in lex.values():
    for f, vals in app.items():
        pool.setdefault(f, []).extend(vals)

# 1) public_test: dòng không có audio_text/label; prompt eval chỉ dùng context + history + hyps giả
pt = [json.loads(l) for l in open(f"{root}/public_test/public_test.jsonl")]
n = 0
for row in pt:
    for it in bp.items_from_row(row, with_label=False):
        assert "label_ents" not in it and it["text"] is None
        hyps = {p[2]: "giả" for p in it["prev"] if p[0] == "user"}
        assert all(p[1] is None for p in it["prev"] if p[0] == "user")      # không có nhãn lượt user lúc test
        ids, ents = bp.build_prompt(tok, it, hyps=hyps,
                                    extra_ents=bp.select_lexicon(tok, lex, it["apps"], "mở kênh vtv3"))
        assert len(ids) <= 221, len(ids)
        n += 1
print("public_test: dựng prompt cho", n, "utt, OK")

# 2) train mode: độ dài, có/không label_ents
rng = random.Random(0)
cfg = dict(label_ent_prob=0.5, all_distract_prob=0.0, n_distract_max=10, section_drop=0.3, pool=pool)
L = [len(bp.build_prompt(tok, it, train=True, rng=rng, cfg=cfg)[0]) for it in man[:2000]]
assert max(L) <= 221, max(L)
print("train prompt len: max", max(L), "| rỗng", sum(l == 0 for l in L), "/", len(L))
it = next(x for x in man if x["hist_ents"] and x["label_ents"] and x["kind"] == "final")
print("ví dụ eval:", tok.decode(bp.build_prompt(tok, it)[0])[-300:])
print("ví dụ train V2:", tok.decode(bp.build_prompt(tok, it, train=True, rng=random.Random(3), cfg=cfg)[0])[-300:])

# 3) ước lượng điều kiện lexicon bằng hyp của v1 làm bản nháp (không cần train)
if len(sys.argv) > 2:
    d = json.load(open(sys.argv[2]))
    M = {x["uid"]: x for x in man}
    hyps = {u: h for u, _, h in d["pairs"]}
    refs, hs, pa, pb = [], [], [], []
    for u, ref, h in d["pairs"]:
        it = M[u]
        _, ea = bp.build_prompt(tok, it, hyps=hyps)
        _, eb = bp.build_prompt(tok, it, hyps=hyps, extra_ents=bp.select_lexicon(
            tok, lex, it["apps"], h, exclude=[e[2] for e in it["hist_ents"]]))
        refs.append(ref); hs.append(h); pa.append(ea); pb.append(eb)
    for name, pe in (("(a) history + profile", pa), ("(b) + lexicon", pb)):
        s = bp.name_stats(refs, hs, pe)
        print(name, {k: s[k] for k in ("names", "name_err", "in_prompt", "in_prompt_err")},
              "| TB tên/prompt", round(sum(map(len, pe)) / len(pe), 1))
