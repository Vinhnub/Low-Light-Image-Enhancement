import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseExtractor(nn.Module):
    """
    Trích xuất Bản đồ nhiễu (Noise Prior) dựa trên Phương sai cục bộ (Local Variance).
    Dành cho bài toán khử nhiễu ảnh trong điều kiện thiếu sáng (lấy cảm hứng từ FFDNet).
    """
    def __init__(self, in_channels=1, kernel_size=3):
        super(NoiseExtractor, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        # Tạo bộ lọc tính trung bình cho nhiều kênh (nhóm conv độc lập mỗi kênh)
        self.register_buffer('weight', torch.ones(in_channels, 1, kernel_size, kernel_size) / (kernel_size**2))

    def forward(self, x, average_channels=True):
        # 1. Tính giá trị trung bình cục bộ E[X] bằng grouped convolution
        mean_x = F.conv2d(x, self.weight, padding=self.kernel_size//2, groups=self.in_channels)
        
        # 2. Tính trung bình của bình phương E[X^2]
        mean_x2 = F.conv2d(x**2, self.weight, padding=self.kernel_size//2, groups=self.in_channels)
        
        # 3. Phương sai V[X] = E[X^2] - (E[X])^2
        variance = mean_x2 - mean_x**2
        
        # 4. Đảm bảo không bị âm do sai số dấu phẩy động
        noise_map = torch.clamp(variance, min=0.0)
        
        if average_channels and self.in_channels > 1:
            noise_map = torch.mean(noise_map, dim=1, keepdim=True)
        
        # Có thể dùng căn bậc hai để lấy Standard Deviation (Sigma) thay vì Variance
        # noise_map = torch.sqrt(noise_map + 1e-8)
        
        # Tùy chọn: Chuẩn hóa bản đồ nhiễu về khoảng [0, 1] trên mỗi ảnh trong batch
        # để mạng dễ học hơn. (Bỏ comment nếu muốn chuẩn hóa)
        # b, c, h, w = noise_map.size()
        # min_val = noise_map.view(b, c, -1).min(dim=2, keepdim=True)[0].unsqueeze(3)
        # max_val = noise_map.view(b, c, -1).max(dim=2, keepdim=True)[0].unsqueeze(3)
        # noise_map = (noise_map - min_val) / (max_val - min_val + 1e-6)
        
        return noise_map

if __name__ == "__main__":
    print("=== TEST NOISE EXTRACTOR ===")
    ext_1ch = NoiseExtractor(in_channels=1, kernel_size=3)
    dummy_input_1ch = torch.rand(2, 1, 64, 64)
    out_1 = ext_1ch(dummy_input_1ch)
    print(f"1-ch Noise map shape: {out_1.shape}")
    
    ext_2ch = NoiseExtractor(in_channels=2, kernel_size=3)
    dummy_input_2ch = torch.rand(2, 2, 64, 64)
    out_2 = ext_2ch(dummy_input_2ch, average_channels=True)
    print(f"2-ch Noise map (averaged) shape: {out_2.shape}")
