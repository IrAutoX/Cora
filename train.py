import argparse,json,os,time
from pathlib import Path
import torch
from datasets import load_dataset
from peft import LoraConfig,get_peft_model
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).parent
OUT=ROOT/"checkpoints"; OUT.mkdir(exist_ok=True)
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
def python_stream():
    for row in load_dataset("jtatman/python-code-dataset-500k",split="train",streaming=True):
        a=str(row.get("instruction","")).strip(); b=str(row.get("output","")).strip()
        if a and b: yield [{"role":"user","content":a},{"role":"assistant","content":b}]
def persian_stream():
    for row in load_dataset("saied/Persian_Chat_Dataset",split="train",streaming=True):
        m=row.get("messages")
        if isinstance(m,list):
            clean=[{"role":str(x.get("role","user")),"content":str(x["content"])} for x in m if isinstance(x,dict) and x.get("content")]
            if len(clean)>=2: yield clean
def local_stream():
    p=ROOT/"data"/"fa.txt"
    if not p.exists(): return
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip(): yield [{"role":"user","content":"این متن فارسی را بخوان و پاسخ مناسب بده."},{"role":"assistant","content":line.strip()}]
def mixed_stream():
    py=python_stream(); fa=persian_stream(); local=iter(local_stream() or [])
    while True:
        ok=False
        try: yield next(py); ok=True
        except StopIteration: py=python_stream()
        try: yield next(fa); ok=True
        except StopIteration: fa=persian_stream()
        try: yield next(local); ok=True
        except StopIteration: local=iter(local_stream() or [])
        if not ok: return
def encode(tok,messages,n):
    try: text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
    except Exception: text="\n".join(str(x["role"])+": "+str(x["content"]) for x in messages)
    ids=tok(text,truncation=True,max_length=n,return_tensors="pt")["input_ids"][0]
    return ids
def main():
    p=argparse.ArgumentParser(); p.add_argument("--minutes",type=float,default=60); p.add_argument("--max-length",type=int,default=512); p.add_argument("--seed",type=int,default=42); a=p.parse_args()
    torch.manual_seed(a.seed); torch.set_num_threads(max(1,min(4,os.cpu_count() or 2)))
    tok=AutoTokenizer.from_pretrained(MODEL_NAME,use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(MODEL_NAME,torch_dtype=torch.float32)
    cfg=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    model=get_peft_model(base,cfg); model.train()
    opt=torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),lr=2e-4,weight_decay=0.01)
    stream=mixed_stream(); start=time.time(); deadline=start+a.minutes*60; step=0; loss_value=0.; saved2=False; next_save=120
    def save(name):
        d=OUT/name; d.mkdir(parents=True,exist_ok=True); model.save_pretrained(d); tok.save_pretrained(d)
        (d/"metrics.json").write_text(json.dumps({"step":step,"loss":loss_value,"elapsed_seconds":round(time.time()-start,2),"base_model":MODEL_NAME},indent=2),encoding="utf-8")
    while time.time()<deadline:
        ids=encode(tok,next(stream),a.max_length)
        if ids.numel()<8: continue
        out=model(input_ids=ids.unsqueeze(0),labels=ids.unsqueeze(0)); loss=out.loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        step+=1; loss_value=float(loss.detach()); elapsed=time.time()-start
        if not saved2 and elapsed>=120: save("cora-2m"); saved2=True; print(json.dumps({"event":"checkpoint","checkpoint":"cora-2m","step":step,"loss":loss_value,"elapsed_seconds":round(elapsed,2)}),flush=True)
        if elapsed>=next_save: save("cora-latest"); next_save+=600
        if step%10==0: print(json.dumps({"event":"train","step":step,"loss":loss_value,"elapsed_seconds":round(elapsed,2)}),flush=True)
    save("cora-final"); save("cora-latest")
    metrics={"status":"complete","steps":step,"loss":loss_value,"elapsed_seconds":round(time.time()-start,2),"base_model":MODEL_NAME,"trainable_parameters":sum(x.numel() for x in model.parameters() if x.requires_grad)}
    (OUT/"metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(metrics,ensure_ascii=False),flush=True)
if __name__=="__main__": main()
