"""Đẩy adapter FC lên HF (fisherman611/q-v4-ckpts): ưu tiên bản có khả năng nộp, rồi toàn bộ phần còn lại. KHÔNG xóa gì.
HF_TOKEN lấy từ biến môi trường (truyền qua stdin, không ghi file)."""
import os, re, sys, time
from huggingface_hub import HfApi
C = "/mnt/hungpv/projects/do-nothing/asr/agent/clef/runs/"
REPO = "fisherman611/q-v4-ckpts"
E1, E4, E5 = C + "fmtA_v4_e1/", C + "fmtA_v5_e4/", C + "fmtA_v5_e5/"
items = []
def add(name, d):
    if os.path.exists(d + "/adapter_model.safetensors") and all(n != name for n, _ in items): items.append((name, d))
# ưu tiên
add("v5e5_step400", E5 + "e5_ckpts/step400"); add("v5e5_step600", E5 + "e5_ckpts/step600")
add("v5e4_step1100", E4 + "e4_ckpts/step1100"); add("v5e4_step1000", E4 + "e4_ckpts/step1000")
add("v5e4_step800", E4 + "e4_ckpts/step800")
# phần còn lại
for tag, base in (("v5e5", E5 + "e5_ckpts/"), ("v5e4", E4 + "e4_ckpts/"), ("e3", E1 + "e3_ckpts/")):
    if os.path.isdir(base):
        for n in sorted(os.listdir(base)):
            if re.fullmatch(r"step\d+", n): add(f"{tag}_{n}", base + n)
for s in (1318, 1319): add(f"e1_step{s}", E1 + f"step{s}")
for s in (2636, 2637): add(f"e2_step{s}", E1 + f"step{s}")
print(len(items), "adapter sẽ đẩy", flush=True)
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
for name, d in items:
    for k in range(3):
        try:
            api.upload_folder(folder_path=d, repo_id=REPO, repo_type="model", path_in_repo=name,
                              allow_patterns=["adapter_config.json", "adapter_model.safetensors"], commit_message=f"{name} (từ myserver)")
            print(time.strftime("%H:%M:%S"), "đẩy xong", name, flush=True); break
        except Exception as e:
            print("lỗi", name, k, repr(e)[:160], flush=True); time.sleep(20)
info = {s.rfilename: s.size for s in api.repo_info(REPO, repo_type="model", files_metadata=True).siblings}
bad = [n for n, d in items if info.get(f"{n}/adapter_model.safetensors") != os.path.getsize(d + "/adapter_model.safetensors")]
print("KIỂM TRA: thiếu/lệch kích thước:", bad or "không có", flush=True)
print("XONG", flush=True)
