import torch
import torch.nn as nn
from net.CIDNet_Mamba_separable_learning import CIDNet as CIDNet_Mamba
from net.CIDNet_base import CIDNet as CIDNet_Base
from huggingface_hub import PyTorchModelHubMixin

class TransWrapper:
    def __init__(self, trans1, trans2):
        self.trans1 = trans1
        self.trans2 = trans2
        
    @property
    def gated(self):
        return self.trans2.gated
        
    @gated.setter
    def gated(self, val):
        self.trans1.gated = val
        self.trans2.gated = val
        
    @property
    def gated2(self):
        return self.trans2.gated2
        
    @gated2.setter
    def gated2(self, val):
        self.trans1.gated2 = val
        self.trans2.gated2 = val
        
    @property
    def alpha(self):
        return self.trans2.alpha
        
    @alpha.setter
    def alpha(self, val):
        self.trans1.alpha = val
        self.trans2.alpha = val

class CIDNetTwoStage(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
        ):
        super(CIDNetTwoStage, self).__init__()
        
        # Stage 1: CIDNet + Mamba
        self.stage1 = CIDNet_Mamba(channels=channels, heads=heads, norm=norm)
        
        # Stage 2: CIDNet base (without Mamba)
        self.stage2 = CIDNet_Base(channels=channels, heads=heads, norm=norm)
        
        # Wrapper for trans color space matching during eval
        self._trans = TransWrapper(self.stage1.trans, self.stage2.trans)
        
    @property
    def trans(self):
        return self._trans
        
    def forward(self, x, stop_grad=False, return_all=False):
        # Stage 1 (Detail Restoration and mild lighting)
        out_s1 = self.stage1(x)
        
        # Detach gradient flow to Stage 1 if stop_grad is True
        if stop_grad:
            out_s1_in = out_s1.detach()
        else:
            out_s1_in = out_s1
            
        # Stage 2 (Global brightness enhancement)
        out_s2 = self.stage2(out_s1_in)
        
        if return_all:
            return out_s1, out_s2
        return out_s2
        
    def HVIT(self, x):
        return self.stage2.HVIT(x)
        
    def RGB2YCrCb(self, x):
        return self.stage2.RGB2YCrCb(x)
