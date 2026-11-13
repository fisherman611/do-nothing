"""Model soup cho ASR (PLAN §1z11 #2): trung bình trọng số các model PhoWhisper-large đã gộp LoRA (cùng gốc, cùng split),
không train thêm. Lưu bản soup lên volume `zaic-asr-v3-out/soup_<tên>` rồi chép lời 405 held-out đúng điều kiện thi bằng
app `lifemate-asr-infer-v3` (đã deploy). Chấm cục bộ bằng chuẩn hoá của agent/asr_errors_e46f.py (so tương đối; số chính
thức chạy `asr_gate_v3.py` trên myserver).

    modal deploy asr/modal_soup.py
    python asr/modal_soup.py make  e46f_c3d9   fisherman611/pw-tmp-e46f fisherman611/pw-tmp-c3d9 [--w 0.5,0.5]
    python asr/modal_soup.py eval  /out/soup_e46f_c3d9  [--out agent/runs/asr/heldout.soup_e46f_c3d9.asr.json]
    python asr/modal_soup.py score agent/runs/asr/heldout.soup_e46f_c3d9.asr.json
"""
import json
import os
import sys

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("lifemate-asr-soup")
out_vol = modal.Volume.from_name("zaic-asr-v3-out", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.8.0", "safetensors", "numpy<2.3", "huggingface_hub[hf_transfer]")
         .env({"HF_HOME": "/out/hf_cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"}))


@app.function(image=image, cpu=8, memory=32768, volumes={"/out": out_vol}, timeout=3600)
def make_soup(name: str, models: list, weights: list = None):
    """models: repo HF hoặc đường dẫn /out/<tag>; weights mặc định đều nhau. Ghi /out/soup_<name>."""
    import shutil
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    dirs = [m if m.startswith("/") else snapshot_download(m) for m in models]
    w = weights or [1.0 / len(dirs)] * len(dirs)
    assert abs(sum(w) - 1) < 1e-6 and len(w) == len(dirs)
    acc = None
    for d, wi in zip(dirs, w):
        sd = load_file(f"{d}/model.safetensors")
        if acc is None:
            acc = {k: v.float() * wi for k, v in sd.items()}
        else:
            assert set(sd) == set(acc), "khoá tensor không khớp"
            for k, v in sd.items():
                acc[k] += v.float() * wi
        print(f"+ {d} × {wi}: {len(sd)} tensor", flush=True)
    out = f"/out/soup_{name}"
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(dirs[0], out, ignore=shutil.ignore_patterns("model.safetensors", "*.msgpack", "*.h5", "*.bin", ".git*"))
    save_file({k: v.half() for k, v in acc.items()}, f"{out}/model.safetensors", metadata={"format": "pt"})
    json.dump({"models": models, "weights": w}, open(f"{out}/soup.json", "w"), indent=1)
    out_vol.commit()
    print("ghi", out, flush=True)
    return out


def held_manifest():
    """405 câu val của manifest ASR v1 (`modal volume get zaic-asr-v3-data manifest.json`), prev = [(role, text, uid)]."""
    m = json.load(open(f"{HERE}/agent/runs/asr/asr_v3_manifest.json", encoding="utf-8"))
    return [x for x in m if x["split"] == "val"]


def held_items():
    return [{"uid": h["uid"], "sid": h["sid"], "kind": h["kind"], "path": f"train/{h['audio']}",
             "prev": h["prev"] if isinstance(h["prev"], list) else json.loads(h["prev"]), "set": "heldout"} for h in held_manifest()]


def score(path):
    """WER tương đối (chuẩn hoá như agent/asr_errors_e46f.py: thường, bỏ dấu câu, giữ chữ số) trên 405 held-out."""
    sys.path.insert(0, HERE)
    from agent.asr_errors_e46f import norm, align   # noqa: E402  (chạy phần hàm; phần main của file đó nằm sau 'd = json.load')
    held = {h["uid"]: h for h in held_manifest()}
    hyps = json.load(open(path, encoding="utf-8"))["hyps"]
    acc = {}
    ok = 0
    for uid, h in held.items():
        r = norm(h["text"]).split(); e = align(r, norm(hyps.get(uid, "")).split())[1]
        for k in ("tất cả", h["kind"]):
            a = acc.setdefault(k, [0, 0]); a[0] += e; a[1] += len(r)
        ok += e == 0
    return {k: round(100 * a[0] / a[1], 2) for k, a in acc.items()} | {"câu đúng hết": ok}


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "make":
        w = [float(x) for x in args[args.index("--w") + 1].split(",")] if "--w" in args else None
        if "--w" in args:
            i = args.index("--w"); args = args[:i] + args[i + 2:]
        name, models = args[0], args[1:]
        with modal.enable_output(), app.run():
            print(make_soup.remote(name, models, w))
    elif cmd == "eval":
        model, out = args[0], (args[args.index("--out") + 1] if "--out" in args else f"{HERE}/agent/runs/asr/heldout.{os.path.basename(args[0])}.asr.json")
        f = modal.Function.from_name("lifemate-asr-infer-v3", "transcribe")
        res = f.remote(held_items(), model, 5, True)
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("ghi", out, "| lỗi:", len(res["errors"]), "| giây:", res["total_s"])
        print(score(out))
    elif cmd == "public":   # python modal_soup.py public /out/soup_x agent/runs/asr/public.soup_x.asr.json
        model, out = args[0], args[1]
        pub = json.load(open(f"{HERE}/agent/runs/asr/public_test.manifest.json", encoding="utf-8"))["items"]
        items = [{"uid": p["uid"], "sid": p["sid"], "kind": p["kind"], "path": f"public_test/{p['audio']}",
                  "prev": p.get("prev") or [], "set": "public"} for p in pub]
        res = modal.Function.from_name("lifemate-asr-infer-v3", "transcribe").remote(items, model, 5, True)
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("ghi", out, "| đoạn:", len(res["hyps"]), "| lỗi:", len(res["errors"]), "| giây:", res["total_s"])
    elif cmd == "score":
        for p in args: print(os.path.basename(p), score(p))
