import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from net.edge_filter import EdgeExtractor
from net.noise_filter import NoiseExtractor
from huggingface_hub import PyTorchModelHubMixin

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(1, channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
        ):
        super(CIDNet, self).__init__()
        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        # --- THÊM EDGE EXTRACTOR ---
        # Trích xuất cạnh cho nhánh HV (tính cạnh 2 kênh H, V rồi gộp lại làm 1)
        # self.edge_hv_ext = EdgeExtractor(in_channels=2)
        # Trích xuất cạnh cho nhánh I (1 kênh)
        self.edge_i_ext = EdgeExtractor(in_channels=1)
        # Trích xuất bản đồ nhiễu (Noise Prior) riêng rẽ cho từng nhánh
        self.noise_hv_ext = NoiseExtractor(in_channels=2, kernel_size=3)
        # self.noise_i_ext = NoiseExtractor(in_channels=1, kernel_size=3)
        
        # Chú ý kênh (SE Blocks) cho đầu vào 10 kênh và 6 kênh (đã thêm noise_map vào cả 2)
        self.se_hv = SEBlock(channel=9, reduction=3)
        self.se_i = SEBlock(channel=6, reduction=2)
        
        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(9, ch1, 3, stride=1, padding=0, bias=False)
            )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )
        
        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            # Đầu vào 6 kênh (I + edge_i + noise_map + 3 kênh new_i_channels)
            nn.Conv2d(6, ch1, 3, stride=1, padding=0, bias=False),
            )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm)
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
            )
        
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
        
        self.trans = RGB_HVI()
        
    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        
        hv = hvi[:, 0:2, :, :].to(dtypes)
        i = hvi[:, 2:3, :, :].to(dtypes)
        
        # --- TRÍCH XUẤT VÀ NỐI KÊNH EDGE / NOISE ---
        # edge_hv = self.edge_hv_ext(hv, average_channels=True).to(dtypes) # [B, 1, H, W]
        edge_i = self.edge_i_ext(i).to(dtypes) # [B, 1, H, W]
        noise_hv = self.noise_hv_ext(hv, average_channels=True).to(dtypes) # [B, 1, H, W]
        # noise_i = self.noise_i_ext(i).to(dtypes) # [B, 1, H, W]
        
        # Loại bỏ cạnh khỏi bản đồ nhiễu
        noise_hv = noise_hv * (1 - edge_i)
        
        # Tính toán các kênh bổ sung từ x
        R = x[:, 0:1, :, :]
        G = x[:, 1:2, :, :]
        B = x[:, 2:3, :, :]
        epsilon = 1e-6
        
        # Nhánh HV (Image 1)
        Cb = -0.168736 * R - 0.331264 * G + 0.5 * B
        Cr = 0.5 * R - 0.418688 * G - 0.081312 * B
        rgb_sum = R + G + B + epsilon
        r_norm = R / rgb_sum
        g_norm = G / rgb_sum
        max_RGB, _ = torch.max(x, dim=1, keepdim=True)
        min_RGB, _ = torch.min(x, dim=1, keepdim=True)
        #S = (max_RGB - min_RGB) / (max_RGB + epsilon)
        
        new_hv_channels = torch.cat([Cb, Cr, r_norm, g_norm], dim=1).to(dtypes)
        hvi_edge = torch.cat([hvi, edge_i, noise_hv, new_hv_channels], dim=1) # [B, 9, H, W]
        # Áp dụng SE block để đánh giá và lọc các kênh nhiễu
        hvi_edge = self.se_hv(hvi_edge)
        
        # Nhánh I (Image 2)
        Y_Rec709 = 0.2126 * R + 0.7152 * G + 0.0722 * B
        Y_lightness = 0.5 * (max_RGB + min_RGB)
        Y_L2 = torch.sqrt(R**2 + G**2 + B**2 + epsilon)
        
        new_i_channels = torch.cat([Y_Rec709, Y_lightness, Y_L2], dim=1).to(dtypes)
        i_edge = torch.cat([i, edge_i, noise_hv, new_i_channels], dim=1) # [B, 6, H, W]
        # Áp dụng SE block
        i_edge = self.se_i(i_edge)
        
        # low
        i_enc0 = self.IE_block0(i_edge)
        i_enc1 = self.IE_block1(i_enc0)
        
        hv_0 = self.HVE_block0(hvi_edge)
        hv_1 = self.HVE_block1(hv_0)
        
        i_jump0 = i_enc0
        hv_jump0 = hv_0
        
        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)
        
        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc3)
        hv_3 = self.HVE_block3(hv_3)
        
        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)
        
        i_dec4 = self.I_LCA4(i_enc4,hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)
        
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)
        
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)
        
        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)
        
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)
        
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb
    
    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi
