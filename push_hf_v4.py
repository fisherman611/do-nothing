import os
from huggingface_hub import HfApi
R = "/mnt/hungpv/projects/do-nothing/"
api = HfApi(token=os.environ["HF_TOKEN"])
print("user:", api.whoami()["name"], flush=True)
J = [
 ("fisherman611/q-v4e1-lora", "model", [(R+"asr/agent/clef/runs/fmtA_v4_e1/final", "final", ["adapter_config.json","adapter_model.safetensors"], None)]),
 ("fisherman611/pw-tmp-v4a", "model", [(R+"models/asr_out/v4/lora_ls0.1_prompt_e46f_tts_v4p1_adapter", "p1", None, None),
                                       (R+"models/asr_out/v4/lora_ls0.1_prompt_e46f_v4p2_real_adapter", "p2", None, None)]),
 ("fisherman611/pw-tmp-v4m", "model", [(R+"models/asr_out/v4m/lora_ls0.1_prompt_e46f_v4mp2_real", ".", None, ["eval.json"])]),
 ("fisherman611/fc-v4-3a7d", "dataset", [(R+"asr/agent/clef/data/fmtA_v4", "fmtA_v4", None, None)]),
 ("fisherman611/au-v4-pub", "dataset", [(R+"data/asr_v3/tts_pub", "tts_pub", None, None)]),
]
for repo, rt, parts in J:
    try:
        api.create_repo(repo, repo_type=rt, private=False, exist_ok=True)
        for loc, pir, allow, ign in parts:
            api.upload_folder(folder_path=loc, repo_id=repo, repo_type=rt, path_in_repo=pir, allow_patterns=allow, ignore_patterns=ign, commit_message=f"upload {pir} (từ myserver)")
        if "au-v4" in repo:
            for f in ["data/asr_v3/tts_manifest_pub.json", "data/asr_v3/tts_manifest_pub_f.json"]:
                api.upload_file(path_or_fileobj=R+f, path_in_repo=os.path.basename(f), repo_id=repo, repo_type=rt)
        i = api.repo_info(repo, repo_type=rt, files_metadata=True)
        print("OK", repo, len(i.siblings), "files", round(sum((s.size or 0) for s in i.siblings)/1e9, 2), "GB", i.sha, flush=True)
    except Exception as e:
        print("FAIL", repo, repr(e)[:300], flush=True)
print("DONE", flush=True)
