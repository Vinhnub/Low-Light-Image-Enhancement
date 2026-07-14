import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from huggingface_hub import PyTorchModelHubMixin
from net.mmmamba import MMMamba

class CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
        ):
        super(CIDNet, self).__init__()
        
        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False)
            )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0,bias=False)
        )
        
        
        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
            )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
            )
        
        # Cross Attention
        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)
        
        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)

        # MMMamba
        self.MMMamba_1 = MMMamba(ch2)
        self.MMMamba_2 = MMMamba(ch3)
        self.MMMamba_3 = MMMamba(ch4)
        self.MMMamba_4 = MMMamba(ch4)
        self.MMMamba_5 = MMMamba(ch3)
        self.MMMamba_6 = MMMamba(ch2)

        self.Combine_I_1 = nn.Conv2d(ch2 * 2, ch2, kernel_size=1, stride=1, padding=0, bias=False)
        self.Combine_I_2 = nn.Conv2d(ch3 * 2, ch3, kernel_size=1, stride=1, padding=0, bias=False)
        self.Combine_I_3 = nn.Conv2d(ch4 * 2, ch4, kernel_size=1, stride=1, padding=0, bias=False)
        self.Combine_I_4 = nn.Conv2d(ch4 * 2, ch4, kernel_size=1, stride=1, padding=0, bias=False)
        self.Combine_I_5 = nn.Conv2d(ch3 * 2, ch3, kernel_size=1, stride=1, padding=0, bias=False)
        self.Combine_I_6 = nn.Conv2d(ch2 * 2, ch2, kernel_size=1, stride=1, padding=0, bias=False)

        self.Combine_HV_1 = nn.Conv2d(ch2 * 2, ch2, kernel_size=1, stride=1, padding=0, bias=False)
        self.Combine_HV_2 = nn.Conv2d(ch3 * 2, ch3, kernel_size=1, stride=1, padding=0, bias=False)
        self.Combine_HV_3 = nn.Conv2d(ch4 * 2, ch4, kernel_size=1, stride=1, padding=0, bias=False)
        self.Combine_HV_4 = nn.Conv2d(ch4 * 2, ch4, kernel_size=1, stride=1, padding=0, bias=False)
        self.Combine_HV_5 = nn.Conv2d(ch3 * 2, ch3, kernel_size=1, stride=1, padding=0, bias=False)
        self.Combine_HV_6 = nn.Conv2d(ch2 * 2, ch2, kernel_size=1, stride=1, padding=0, bias=False)

        self.trans = RGB_HVI()
        
    def forward(self, x, return_feats=False):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)
        # low
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0

        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)

        # My code
        i_enc2_m, hv_2_m = self.MMMamba_1([i_enc1, hv_1])
        i_enc2_c = torch.cat([i_enc2, i_enc2_m], dim=1)
        hv_2_c = torch.cat([hv_2, hv_2_m], dim=1)
        i_enc2 = self.Combine_I_1(i_enc2_c)
        hv_2 = self.Combine_HV_1(hv_2_c)
        # My code

        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)
        
        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        # My code
        i_enc3_m, hv_3_m = self.MMMamba_2([i_enc2, hv_2])
        i_enc3_c = torch.cat([i_enc3, i_enc3_m], dim=1)
        hv_3_c = torch.cat([hv_3, hv_3_m], dim=1)
        i_enc3 = self.Combine_I_2(i_enc3_c)
        hv_3 = self.Combine_HV_2(hv_3_c)
        # My code
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)
        
        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)
        # My code
        i_enc4_m, hv_4_m = self.MMMamba_3([i_enc3, hv_3])
        i_enc4_c = torch.cat([i_enc4, i_enc4_m], dim=1)
        hv_4_c = torch.cat([hv_4, hv_4_m], dim=1)
        i_enc4 = self.Combine_I_3(i_enc4_c)
        hv_4 = self.Combine_HV_3(hv_4_c)
        # My code

        i_dec4 = self.I_LCA4(i_enc4,hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)
        # My code
        i_dec4_m, hv_4_m = self.MMMamba_4([i_enc4, hv_4])
        i_dec4_c = torch.cat([i_dec4, i_dec4_m], dim=1)
        hv_4_c = torch.cat([hv_4, hv_4_m], dim=1)
        i_dec4 = self.Combine_I_4(i_dec4_c)
        hv_4 = self.Combine_HV_4(hv_4_c)
        # My code
        
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)

        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)
        # My code
        i_dec2_m, hv_2_m = self.MMMamba_5([i_dec3, hv_3])
        i_dec2_c = torch.cat([i_dec2, i_dec2_m], dim=1)
        hv_2_c = torch.cat([hv_2, hv_2_m], dim=1)
        i_dec2 = self.Combine_I_5(i_dec2_c)
        hv_2 = self.Combine_HV_5(hv_2_c)
        # My code
        
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)
        
        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)
        # My code
        i_dec1_m, hv_1_m = self.MMMamba_6([i_dec2, hv_2])
        i_dec1_c = torch.cat([i_dec1, i_dec1_m], dim=1)
        hv_1_c = torch.cat([hv_1, hv_1_m], dim=1)
        i_dec1 = self.Combine_I_6(i_dec1_c)
        hv_1 = self.Combine_HV_6(hv_1_c)
        # My code

        # =================================
        
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)
        
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        if return_feats:
            feats = {
                'i_enc2': i_enc2,
                'hv_2': hv_2,
                'i_enc3': i_enc3,
                'hv_3': hv_3,
                'i_enc4': i_enc4,
                'hv_4': hv_4,
                'i_dec2': i_dec2,
                'i_dec1': i_dec1
            }

            return output_rgb, feats

        return output_rgb
    
    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi
    
    def RGB2YCrCb(self, x):
        ycrcb = self.trans.RGB2YCrCb(x)
        return ycrcb
    
    

