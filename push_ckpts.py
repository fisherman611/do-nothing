"""Đẩy MỌI checkpoint FC (chỉ adapter: config + safetensors) lên HF, rồi (--delete) xóa ở myserver, trừ checkpoint mới nhất chưa eval.
   HF_TOKEN=... python push_ckpts.py            # chỉ liệt kê (dry-run)
   HF_TOKEN=... python push_ckpts.py --push     # đẩy lên HF, kiểm tra đủ file
   HF_TOKEN=... python push_ckpts.py --push --delete   # + xóa ở myserver (chỉ khi đã đẩy xong và kiểm tra khớp)"""
import os, re, shutil, sys
from huggingface_hub import HfApi
C = "/mnt/hungpv/projects/do-nothing/asr/agent/clef/runs/"
REPO = "fisherman611/q-v4-ckpts"
ck = []   # (thứ tự, tên trên HF, thư mục local, nhãn eval)
def add(order, name, d): 
    if os.path.exists(d + "/adapter_model.safetensors"): ck.append((order, name, d))
e1 = C + "fmtA_v4_e1/"
for s in (1318, 1319): add(s, f"e1_step{s}", e1 + f"step{s}")
for s in (2636, 2637): add(s, f"e2_step{s}", e1 + f"step{s}")
if os.path.isdir(e1 + "e3_ckpts"):
    for n in os.listdir(e1 + "e3_ckpts"):
        m = re.fullmatch(r"step(\d+)", n)
        if m: add(int(m[1]), f"e3_step{m[1]}", e1 + "e3_ckpts/" + n)
e4 = C + "fmtA_v4_e4/"
if os.path.isdir(e4 + "e4_ckpts"):
    for n in os.listdir(e4 + "e4_ckpts"):
        m = re.fullmatch(r"step(\d+)", n)
        if m: add(100000 + int(m[1]), f"e4_step{m[1]}", e4 + "e4_ckpts/" + n)
ck.sort()
ev = set()   # tên đã eval (dòng RESULT)
for tag, f in (("e3", e1 + "e3_eval.log"), ("e4", e4 + "e4_eval.log")):
    if os.path.exists(f):
        for l in open(f, encoding="utf-8"):
            m = re.search(r"RESULT step (\d+)", l)
            if m: ev.add(f"{tag}_step{m[1]}")
ev |= {"e2_step2637"}   # epoch 2 đã eval (528/634); epoch 1 qua oracle/e46f cũ
unev = [c for c in ck if c[1] not in ev]
keep = unev[-1] if unev else ck[-1]
print(f"{len(ck)} checkpoint; đã eval {len([c for c in ck if c[1] in ev])}; chưa eval {len(unev)}; GIỮ LẠI ở myserver: {keep[1]}")
for c in ck: print("  ", c[1], "eval" if c[1] in ev else "CHƯA eval", "<-- giữ" if c is keep else "")
if "--push" not in sys.argv: sys.exit(0)
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
for _, name, d in ck:
    api.upload_folder(folder_path=d, repo_id=REPO, repo_type="model", path_in_repo=name,
                      allow_patterns=["adapter_config.json", "adapter_model.safetensors"], commit_message=f"{name} (từ myserver)")
    print("đẩy xong", name, flush=True)
info = {s.rfilename: s.size for s in api.repo_info(REPO, repo_type="model", files_metadata=True).siblings}
bad = [n for _, n, d in ck if info.get(f"{n}/adapter_model.safetensors") != os.path.getsize(d + "/adapter_model.safetensors")]
print("KIỂM TRA: thiếu/lệch kích thước:", bad or "không có")
if "--delete" in sys.argv:
    if bad: sys.exit("có checkpoint lệch → KHÔNG xóa")
    for _, n, d in ck:
        if n == keep[1]: continue
        shutil.rmtree(d, ignore_errors=True); print("đã xóa", d)
