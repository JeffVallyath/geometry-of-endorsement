"""Trainable map only. This is not a Transformers cache adapter.
Masks must be made from source-prefix character/token alignment, never labels.
The caller guarantees that z and all prefix inputs are computed before query tokens.
"""
import torch
from torch import nn

class AddressedLowRankEditor(nn.Module):
    def __init__(self,width:int,rank:int=16,seed:int=0):
        super().__init__();g=torch.Generator().manual_seed(seed)
        # Zero functional update, but nonzero input factor gives nonzero first
        # output-factor gradients. No both-factors-zero dead initialization.
        self.input_factor=nn.Parameter(torch.randn(2*width,rank,generator=g)*(.01/(2*width)**.5))
        self.output_factor=nn.Parameter(torch.zeros(rank,width))
        self.bias=nn.Parameter(torch.zeros(rank))
        self.register_buffer('center',torch.zeros(width))
    def forward(self,h,z,mask,cap=None):
        if z.ndim==1: z=z.expand_as(h)
        else: z=torch.broadcast_to(z,h.shape)
        x=torch.cat((h.float()-self.center,z.float()-self.center),dim=-1)
        d=(x@self.input_factor+self.bias)@self.output_factor
        if cap is not None:
            norm=d.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            d=d*torch.clamp(torch.as_tensor(cap,device=d.device)/norm,max=1.)
        d=d*mask.to(d.dtype).unsqueeze(-1)
        return h+d.to(h.dtype)
