"""Đẩy file lên HF qua Modal (luồng local → Modal → HF, PLAN_48h): file đã nằm trên volume `hf-stage`,
HF token nằm trong Modal secret `hf-write` (biến HF_TOKEN, chủ repo tự tạo; không ghi token vào code/repo).

    modal volume put hf-stage <thư mục stage> /
    modal run modal/modal_hf_push.py --jobs '[["d48e","fisherman611/d-f48e","dataset"], ...]'

Mỗi job: thư mục con trên volume → repo HF. File *.tgz được giải nén trên Modal rồi mới upload (upload jsonl gốc,
không để file nén trên HF). Repo tạo public nếu chưa có (tài khoản hết hạn mức private).
"""
import json

import modal

app = modal.App("lifemate-hf-push")
stage = modal.Volume.from_name("hf-stage", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub>=1.0")


@app.function(image=image, volumes={"/stage": stage}, secrets=[modal.Secret.from_name("hf-write")], timeout=3600,
              cpu=2, memory=4096)
def push(jobs: list):
    import os
    import shutil
    import tarfile
    import tempfile

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    out = {}
    for sub, repo, rtype in jobs:
        src = f"/stage/{sub}"
        work = tempfile.mkdtemp()
        for root, _, files in os.walk(src):
            for f in files:
                p = os.path.join(root, f)
                rel = os.path.relpath(p, src)
                if f.endswith(".tgz"):
                    with tarfile.open(p) as t:
                        t.extractall(os.path.join(work, os.path.dirname(rel)), filter="data")
                else:
                    os.makedirs(os.path.join(work, os.path.dirname(rel)), exist_ok=True)
                    shutil.copy(p, os.path.join(work, rel))
        api.create_repo(repo, repo_type=rtype, private=False, exist_ok=True)
        api.upload_folder(folder_path=work, repo_id=repo, repo_type=rtype, commit_message=f"upload {sub} (qua Modal)")
        info = api.repo_info(repo, repo_type=rtype, files_metadata=True)
        out[repo] = {"files": len(info.siblings), "mb": round(sum((s.size or 0) for s in info.siblings) / 1e6, 1),
                     "private": info.private}
        print(repo, out[repo], flush=True)
    return out


@app.local_entrypoint()
def main(jobs: str):
    print(json.dumps(push.remote(json.loads(jobs)), ensure_ascii=False, indent=1))
