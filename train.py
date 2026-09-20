import argparse,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import CoraModel
ROOT=Path(__file__).parent
OUT=ROOT/"checkpoints"; OUT.mkdir(exist_ok=True)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--minutes",type=float,default=2); p.add_argument("--seed",type=int,default=42); a=p.parse_args()
    torch.manual_seed(a.seed)
    text=(ROOT/"data/fa.txt").read_text(encoding="utf-8")
    vocab=sorted(set(text)); stoi={c:i for i,c in enumerate(vocab)}
    ids=torch.tensor([stoi[c] for c in text],dtype=torch.long)
    block=min(128,max(16,len(ids)//4))
    model=CoraModel(len(vocab),block); opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    start=time.time(); deadline=start+a.minutes*60; step=0; loss_value=0.0
    while time.time()<deadline or step<5:
        ix=torch.randint(0,len(ids)-block-1,(16,))
        x=torch.stack([ids[i:i+block] for i in ix]); y=torch.stack([ids[i+1:i+block+1] for i in ix])
        loss=F.cross_entropy(model(x).reshape(-1,len(vocab)),y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        step+=1; loss_value=float(loss.item())
        if step%10==0: print(json.dumps({"step":step,"loss":loss_value,"elapsed_seconds":round(time.time()-start,2)}),flush=True)
    torch.save({"state_dict":model.state_dict(),"vocab":vocab,"block_size":block,"steps":step,"loss":loss_value},OUT/"cora-latest.pt")
    metrics={"steps":step,"loss":loss_value,"elapsed_seconds":round(time.time()-start,2),"vocab_size":len(vocab),"parameters":sum(x.numel() for x in model.parameters())}
    (OUT/"metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(metrics,ensure_ascii=False),flush=True)
if __name__=="__main__": main()
