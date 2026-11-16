import os, sys
from huggingface_hub import HfApi
REPO = "fisherman611/q-v5-6c1e"
D = "/mnt/hungpv/projects/do-nothing/models/qwen35_4b_v5_e5s400"
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
ci = api.upload_folder(folder_path=D, repo_id=REPO, repo_type="model", commit_message="FC v5 epoch5 step400 (đã gộp)")
sha = getattr(ci, "oid", None) or getattr(ci, "commit_oid", None)
info = api.repo_info(REPO, repo_type="model", files_metadata=True)
sz = {s.rfilename: s.size for s in info.siblings}
loc = {n: os.path.getsize(os.path.join(D, n)) for n in os.listdir(D)}
bad = [n for n in loc if sz.get(n) != loc[n]]
print("REPO", REPO, "REVISION", info.sha, "commit", sha)
print("KIỂM TRA kích thước lệch:", bad or "không có")
print("XONG")
