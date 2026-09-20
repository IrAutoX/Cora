import torch
from torch import nn
class CoraModel(nn.Module):
    def __init__(self,vocab_size,block_size=128,dim=128,heads=4,layers=3):
        super().__init__()
        self.block_size=block_size
        self.token_emb=nn.Embedding(vocab_size,dim)
        self.pos_emb=nn.Embedding(block_size,dim)
        layer=nn.TransformerEncoderLayer(d_model=dim,nhead=heads,dim_feedforward=dim*4,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=layers)
        self.norm=nn.LayerNorm(dim)
        self.head=nn.Linear(dim,vocab_size)
    def forward(self,x):
        t=x.shape[1]
        pos=torch.arange(t,device=x.device).unsqueeze(0)
        h=self.token_emb(x)+self.pos_emb(pos)
        mask=torch.triu(torch.ones(t,t,device=x.device,dtype=torch.bool),diagonal=1)
        return self.head(self.norm(self.encoder(h,mask=mask)))
