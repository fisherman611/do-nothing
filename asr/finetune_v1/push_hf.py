import modal,os
app=modal.App("zaic-push-hf")
out=modal.Volume.from_name("zaic-out")
img=modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub")
@app.function(image=img,volumes={"/out":out},secrets=[modal.Secret.from_name("hf-token")],timeout=1800)
def push(name:str):
    from huggingface_hub import HfApi
    api=HfApi(token=os.environ["HF_TOKEN"]); user=api.whoami()["name"]; rid=f"{user}/{name}"
    api.create_repo(rid,private=False,exist_ok=True)
    api.upload_folder(repo_id=rid,folder_path="/out/lora",ignore_patterns=["eval.json"])
    print("PUSHED",rid)
@app.local_entrypoint()
def main(name:str="pw-tmp-7f3a"): push.remote(name)
