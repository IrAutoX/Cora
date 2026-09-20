import argparse,torch
from model import CoraModel
from cora_tools import run_tool,list_tools
def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",default="checkpoints/cora-latest.pt"); p.add_argument("--allow-tools",action="store_true"); a=p.parse_args()
    c=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    m=CoraModel(len(c["vocab"]),c["block_size"]); m.load_state_dict(c["state_dict"]); m.eval()
    print("Cora آماده است. برای خروج exit بنویس.")
    if a.allow_tools: print("Tools:",", ".join(list_tools()))
    while True:
        prompt=input("شما> ").strip()
        if prompt.lower()=="exit": break
        if a.allow_tools and prompt.startswith("/tool "):
            z=prompt.split(); print(run_tool(z[1],z[2:])); continue
        ids=[c["vocab"].index(ch) for ch in prompt if ch in c["vocab"]]
        if not ids: print("کورا> داده کافی برای این ورودی ندارد."); continue
        x=torch.tensor([ids[-c["block_size"]:]])
        with torch.no_grad(): nxt=int(m(x)[0,-1].argmax())
        print("کورا>",prompt+c["vocab"][nxt])
if __name__=="__main__": main()
