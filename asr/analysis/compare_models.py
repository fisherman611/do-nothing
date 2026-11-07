"""So sánh lỗi theo loại (từ nội dung / tên / số / từ đệm) giữa các model trên cùng val."""
import json, collections, sys, jiwer
from err_analysis import full, basic, name_tokens, NUMWORDS

NUMW = NUMWORDS - {"không"}
PART = set("nhé nha nhá nhỉ ạ à ờ ừ ơi thôi đi với nào ấy đấy đó này đây á hả hở hén hen nghe ni rứa nớ mô hông ok oke ừm ờm hử hà chớ chứ luôn giùm dùm giúp hộ cái mà thì là".split())
PRON = set("mình anh em tôi chị cô chú bác con cháu ông bà sếp tớ cậu bạn tui".split())
man = json.load(open('/home/thanhlh13/.works/research/zaic/modal/manifest.json'))
train_names = set()
for x in man:
    if x['split'] == 'train': train_names |= name_tokens(x['text'])

def is_num(rt, j):
    t = rt[j]
    if t in NUMW: return True
    return t == "không" and ((j > 0 and rt[j-1] in NUMW | {"không"}) or (j+1 < len(rt) and rt[j+1] in NUMW | {"không"}))

def analyze(path, dump=None):
    pairs = json.load(open(path))['pairs']
    R = [full(r) for _, r, _ in pairs]; H = [full(h) for _, _, h in pairs]
    Rb = [basic(r) for _, r, _ in pairs]; Hb = [basic(h) for _, _, h in pairs]
    cat = collections.Counter(); nt = ne = nseen = nseen_err = nun = nun_err = numt = nume = 0
    name_errs = []
    for (i, rr, hh), r, h in zip(pairs, R, H):
        names = name_tokens(rr); rt = r.split(); ht = h.split()
        bad = set(); ins = []
        if r != h:
            for ch in jiwer.process_words(r, h).alignments[0]:
                if ch.type == "equal": continue
                bad.update(range(ch.ref_start_idx, ch.ref_end_idx))
                if ch.type == "insert": ins += ht[ch.hyp_start_idx:ch.hyp_end_idx]
        kinds = set()
        for j, t in enumerate(rt):
            isn = t in names; isnum = is_num(rt, j)
            if isn:
                nt += 1
                if t in train_names: nseen += 1
                else: nun += 1
            numt += isnum
            if j in bad:
                if isn:
                    ne += 1; kinds.add("tên"); name_errs.append((t, rr, hh))
                    if t in train_names: nseen_err += 1
                    else: nun_err += 1
                elif isnum: nume += 1; kinds.add("số")
                elif t in PART: kinds.add("từ đệm")
                elif t in PRON: kinds.add("đại từ")
                else: kinds.add("từ nội dung")
        for t in ins: kinds.add("từ đệm" if t in PART else "chèn")
        if r != h:
            if kinds <= {"từ đệm", "đại từ", "chèn"}: cat["chỉ từ đệm/đại từ"] += 1
            else:
                for k in kinds - {"từ đệm", "đại từ", "chèn"}: cat[k] += 1
    if dump:
        with open(dump, "w") as f:
            for t, rr, hh in name_errs: f.write(f"[{t}]\n  R: {rr}\n  H: {hh}\n")
    return dict(wer=jiwer.wer(Rb, Hb), wer_norm=jiwer.wer(R, H), exact=sum(a == b for a, b in zip(R, H)), cat=cat,
                name=(ne, nt), name_seen=(nseen_err, nseen), name_unseen=(nun_err, nun), num=(nume, numt))

if __name__ == "__main__":
    runs = [("LoRA CE", "lora_eval.json"), ("LS 0.1", "ls01_eval.json"), ("Weighted CE", "w2_eval.json"), ("LS+CTC", "ctc_eval.json")]
    print(f"{'model':12s} {'WER':>6s} {'WERnorm':>8s} {'khớp':>5s} | {'nộidung':>7s} {'tên':>4s} {'số':>3s} {'đệm':>4s} | {'tên sai':>9s} {'thấy':>8s} {'chưa':>6s} | {'số sai':>8s}")
    for name, p in runs:
        a = analyze(p, dump=f"name_errs_{p.split('_')[0]}.txt"); c = a['cat']
        print(f"{name:12s} {a['wer']:6.3f} {a['wer_norm']:8.3f} {a['exact']:5d} | {c['từ nội dung']:7d} {c['tên']:4d} {c['số']:3d} {c['chỉ từ đệm/đại từ']:4d} | "
              f"{a['name'][0]:3d}/{a['name'][1]:<5d} {a['name_seen'][0]:2d}/{a['name_seen'][1]:<5d} {a['name_unseen'][0]:2d}/{a['name_unseen'][1]:<3d} | {a['num'][0]:3d}/{a['num'][1]}")
