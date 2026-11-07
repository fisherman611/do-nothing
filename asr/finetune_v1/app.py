import modal
app=modal.App("zaic-phowhisper")
vol=modal.Volume.from_name("zaic-data",create_if_missing=True)
out=modal.Volume.from_name("zaic-out",create_if_missing=True)
img=(modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg","libsndfile1")
     .pip_install("torch==2.8.0","transformers==4.57.1","peft==0.17.1","accelerate==1.10.1","jiwer","librosa","soundfile","numpy<2.3"))
M="vinai/PhoWhisper-large"

def norm(s):
    import re,unicodedata
    s=unicodedata.normalize("NFC",s.lower())
    s=re.sub(r"[^\w\s]"," ",s); return re.sub(r"\s+"," ",s).strip()

def load(items):
    import soundfile as sf
    return [sf.read(f"/data/train/{i['audio']}")[0].astype("float32") for i in items]

def transcribe(model,proc,items,bs=16):
    import torch
    model.eval(); hyp=[]
    for i in range(0,len(items),bs):
        wavs=load(items[i:i+bs])
        f=proc.feature_extractor(wavs,sampling_rate=16000,return_tensors="pt").input_features.to("cuda",dtype=model.dtype)
        with torch.no_grad():
            g=model.generate(input_features=f,language="vi",task="transcribe",num_beams=5,max_new_tokens=200)
        hyp+=proc.batch_decode(g,skip_special_tokens=True)
    return hyp

def score(refs,hyp):
    import jiwer
    r=[norm(x) for x in refs]; h=[norm(x) for x in hyp]
    return dict(wer=jiwer.wer(r,h),cer=jiwer.cer(r,h))

@app.function(image=img,gpu="A100-40GB",volumes={"/data":vol,"/out":out},timeout=3600)
def baseline():
    import json,torch
    from transformers import WhisperForConditionalGeneration,WhisperProcessor
    items=[x for x in json.load(open("/data/manifest.json")) if x["split"]=="val"]
    proc=WhisperProcessor.from_pretrained(M)
    model=WhisperForConditionalGeneration.from_pretrained(M,torch_dtype=torch.bfloat16).cuda()
    hyp=transcribe(model,proc,items); res=score([x["text"] for x in items],hyp)
    json.dump(dict(res=res,pairs=[(x["uid"],x["text"],h) for x,h in zip(items,hyp)]),open("/out/baseline.json","w"),ensure_ascii=False,indent=1)
    out.commit(); print("BASELINE",res); return res

@app.function(image=img,gpu="A100-80GB",volumes={"/data":vol,"/out":out},timeout=4*3600)
def train(mode:str="lora",epochs:int=6,lr:float=0.0,full_data:bool=False):
    import json,random,torch,numpy as np
    from transformers import WhisperForConditionalGeneration,WhisperProcessor,Seq2SeqTrainer,Seq2SeqTrainingArguments,EarlyStoppingCallback
    man=json.load(open("/data/manifest.json"))
    tr=[x for x in man if full_data or x["split"]=="train"]; va=[x for x in man if x["split"]=="val"]
    proc=WhisperProcessor.from_pretrained(M); proc.tokenizer.set_prefix_tokens(language="vi",task="transcribe")
    model=WhisperForConditionalGeneration.from_pretrained(M,torch_dtype=torch.float32)
    model.config.forced_decoder_ids=None; model.generation_config.forced_decoder_ids=None
    if mode=="lora":
        from peft import LoraConfig,get_peft_model
        model=get_peft_model(model,LoraConfig(r=32,lora_alpha=64,lora_dropout=0.05,target_modules=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"]))
        model.print_trainable_parameters(); lr=lr or 3e-4
    else: lr=lr or 5e-6
    model.config.apply_spec_augment=True; model.config.mask_time_prob=0.05; model.config.mask_feature_prob=0.05
    class DS(torch.utils.data.Dataset):
        def __init__(s,it): s.it=it
        def __len__(s): return len(s.it)
        def __getitem__(s,i):
            x=s.it[i]; w=load([x])[0]
            return dict(f=proc.feature_extractor(w,sampling_rate=16000).input_features[0],l=proc.tokenizer(x["text"]).input_ids)
    def coll(b):
        f=torch.tensor(np.stack([x["f"] for x in b])); L=max(len(x["l"]) for x in b)
        lab=torch.full((len(b),L),-100)
        for i,x in enumerate(b): lab[i,:len(x["l"])]=torch.tensor(x["l"])
        if (lab[:,0]==model.config.decoder_start_token_id).all(): lab=lab[:,1:]
        return dict(input_features=f,labels=lab)
    def cm(p):
        ids=p.predictions; lab=p.label_ids; lab[lab==-100]=proc.tokenizer.pad_token_id
        return score(proc.batch_decode(lab,skip_special_tokens=True),proc.batch_decode(ids,skip_special_tokens=True))
    tag=f"{mode}{'_full' if full_data else ''}"
    args=Seq2SeqTrainingArguments(output_dir=f"/tmp/{tag}",per_device_train_batch_size=16,gradient_accumulation_steps=2,
        learning_rate=lr,warmup_ratio=0.1,lr_scheduler_type="cosine",num_train_epochs=epochs,bf16=True,gradient_checkpointing=(mode!="lora"),
        eval_strategy="epoch",save_strategy="epoch",save_total_limit=1,save_only_model=True,load_best_model_at_end=True,metric_for_best_model="wer",greater_is_better=False,
        predict_with_generate=True,generation_max_length=200,per_device_eval_batch_size=16,logging_steps=20,report_to=[],remove_unused_columns=False,
        label_names=["labels"],dataloader_num_workers=4,seed=42)
    t=Seq2SeqTrainer(model=model,args=args,train_dataset=DS(tr),eval_dataset=DS(va),data_collator=coll,compute_metrics=cm,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    t.train()
    model=t.model
    if mode=="lora": model=model.merge_and_unload()
    model=model.to(torch.bfloat16).cuda()
    hyp=transcribe(model,proc,va); res=score([x["text"] for x in va],hyp)
    d=f"/out/{tag}"; model.save_pretrained(d,safe_serialization=True); proc.save_pretrained(d)
    json.dump(dict(res=res,log=t.state.log_history,pairs=[(x["uid"],x["text"],h) for x,h in zip(va,hyp)]),open(f"{d}/eval.json","w"),ensure_ascii=False,indent=1)
    out.commit(); print("FINAL",tag,res); return res

@app.local_entrypoint()
def main(step:str="baseline",mode:str="lora",epochs:int=6,full_data:bool=False):
    print(baseline.remote() if step=="baseline" else train.remote(mode=mode,epochs=epochs,full_data=full_data))
