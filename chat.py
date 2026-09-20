import argparse,json,torch
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer
from model import MODEL_NAME
from cora_tools import run_tool,list_tools
def load_model(path):
    tok=AutoTokenizer.from_pretrained(path); base=AutoModelForCausalLM.from_pretrained(MODEL_NAME,torch_dtype=torch.float32); m=PeftModel.from_pretrained(base,path); m.eval(); return tok,m
def generate(tok,m,msgs,n):
    text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True); x=tok(text,return_tensors="pt")
    with torch.no_grad(): y=m.generate(**x,max_new_tokens=n,do_sample=True,temperature=.7,top_p=.9,repetition_penalty=1.08,pad_token_id=tok.eos_token_id)
    return tok.decode(y[0][x["input_ids"].shape[-1]:],skip_special_tokens=True).strip()
def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",default="checkpoints/cora-latest"); p.add_argument("--allow-tools",action="store_true"); p.add_argument("--max-new-tokens",type=int,default=256); a=p.parse_args()
    tok,m=load_model(a.checkpoint)
    history=[{"role":"system","content":"تو Cora هستی، دستیار فارسی و انگلیسی برای برنامه‌نویسی. مستقیم و دقیق پاسخ بده. برای مسائل چندمرحله‌ای در ذهن برنامه‌ریزی کن و فقط نتیجه و مراحل لازم را نمایش بده؛ زنجیره فکر خصوصی را نمایش نده. برای اطلاعات جدید از ابزار وب استفاده کن."}]
    print("Cora آماده است. exit برای خروج، /tools برای ابزارها، /tool NAME ARG برای ابزار.")
    if a.allow_tools: print("Tools:",", ".join(list_tools()))
    while True:
        u=input("شما> ").strip()
        if not u: continue
        if u.lower()=="exit": break
        if u=="/tools": print(json.dumps(list_tools(),ensure_ascii=False,indent=2)); continue
        if a.allow_tools and u.startswith("/tool "):
            z=u.split(" ",2); print(run_tool(z[1],z[2] if len(z)>2 else "")); continue
        history.append({"role":"user","content":u}); ans=generate(tok,m,history,a.max_new_tokens); print("کورا>",ans); history.append({"role":"assistant","content":ans})
if __name__=="__main__": main()
