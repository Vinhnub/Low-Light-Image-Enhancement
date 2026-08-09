import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeExtractor(nn.Module):
    """
    Trích xuất bản đồ cạnh (edge map) từ ảnh đầu vào.
    Module này bao gồm 2 bước:
    1. Làm mờ (Gaussian Blur) để loại bỏ nhiễu hạt (noise), giúp các đường nét liền mạch hơn.
    2. Sử dụng bộ lọc Sobel (theo 2 hướng X và Y) để trích xuất cạnh.
    """
    def __init__(self, in_channels=3, blur_kernel_size=5, blur_sigma=1.0, rgb_to_gray=False):
        """
        Args:
            in_channels: Số kênh đầu vào (vd: 3 cho RGB, 2 cho HV, 1 cho Gray).
            blur_kernel_size: Kích thước kernel mờ.
            blur_sigma: Độ lệch chuẩn cho Gaussian Blur.
            rgb_to_gray: Nếu True và in_channels=3, sẽ chuyển RGB sang Gray (1 kênh) trước khi tìm cạnh.
        """
        super(EdgeExtractor, self).__init__()
        self.in_channels = in_channels
        self.rgb_to_gray = (rgb_to_gray and in_channels == 3)
        
        # Nếu chuyển RGB sang Gray, sau bước blur ta chỉ còn 1 kênh để qua Sobel
        self.sobel_channels = 1 if self.rgb_to_gray else in_channels
        
        # 1. Tạo Gaussian Blur Kernel
        self.blur_kernel_size = blur_kernel_size
        gaussian_kernel = self._create_gaussian_kernel(blur_kernel_size, blur_sigma)
        
        # Lặp lại kernel cho mỗi kênh đầu vào (để dùng cho depthwise convolution)
        gaussian_kernel = gaussian_kernel.repeat(in_channels, 1, 1, 1)
        self.register_buffer('gaussian_kernel', gaussian_kernel) 
        
        # 2. Tạo Sobel Kernel
        sobel_x = torch.tensor([[-1., 0., 1.], 
                                [-2., 0., 2.], 
                                [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
                                
        sobel_y = torch.tensor([[-1., -2., -1.], 
                                [ 0.,  0.,  0.], 
                                [ 1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
        
        # Lặp lại Sobel kernel cho từng kênh (để áp dụng tính cạnh độc lập cho từng kênh)
        sobel_x = sobel_x.repeat(self.sobel_channels, 1, 1, 1)
        sobel_y = sobel_y.repeat(self.sobel_channels, 1, 1, 1)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """Hàm sinh ra ma trận trọng số cho Gaussian Blur"""
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing='ij')
        variance = sigma ** 2.0
        
        gaussian = torch.exp(-(grid_x**2 + grid_y**2) / (2 * variance))
        gaussian = gaussian / torch.sum(gaussian) # Chuẩn hóa để tổng bằng 1
        
        return gaussian.view(1, 1, kernel_size, kernel_size)

    def forward(self, x, average_channels=False):
        """
        Args:
            x: Tensor đầu vào, kích thước [B, C, H, W]
            average_channels: Nếu True, sẽ tính trung bình các bản đồ cạnh của từng kênh để gộp thành 1 kênh.
        Returns:
            edge_map: Bản đồ cạnh, kích thước [B, C, H, W] (hoặc [B, 1, H, W] nếu average_channels=True hoặc rgb_to_gray=True)
        """
        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Module khởi tạo với in_channels={self.in_channels}, nhưng nhận được {C} kênh.")
            
        # --- BƯỚC 1: LÀM MỜ (BLUR) ---
        pad_blur = self.blur_kernel_size // 2
        # Dùng padding mode='replicate' (nhân bản viền) để tránh viền ảnh bị lỗi vệt đen
        x_padded = F.pad(x, (pad_blur, pad_blur, pad_blur, pad_blur), mode='replicate')
        
        # Tính convolution từng kênh (groups=C)
        blurred = F.conv2d(x_padded, self.gaussian_kernel, groups=C)
        
        # --- BƯỚC 2: CHUYỂN SANG GRAYSCALE (Tùy chọn cho RGB) ---
        if self.rgb_to_gray:
            # Công thức tính luma
            weight = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
            target = torch.sum(blurred * weight, dim=1, keepdim=True) # shape: [B, 1, H, W]
        else:
            target = blurred # Giữ nguyên số kênh: C (ví dụ: 2 kênh HV)
            
        # --- BƯỚC 3: TRÍCH XUẤT CẠNH BẰNG SOBEL ---
        target_padded = F.pad(target, (1, 1, 1, 1), mode='replicate')
        
        # Trích xuất đạo hàm theo 2 hướng độc lập cho mỗi kênh (groups=self.sobel_channels)
        edge_x = F.conv2d(target_padded, self.sobel_x, groups=self.sobel_channels)
        edge_y = F.conv2d(target_padded, self.sobel_y, groups=self.sobel_channels)
        
        # Tính biên độ (Magnitude) tổng hợp
        edge_mag = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        
        # Nếu muốn gộp (average) tất cả các kênh cạnh thành 1 kênh duy nhất
        if average_channels and edge_mag.shape[1] > 1:
            edge_mag = torch.mean(edge_mag, dim=1, keepdim=True)
            
        return edge_mag

if __name__ == '__main__':
    import os
    import sys
    # Thêm đường dẫn gốc HVI-CIDNet để có thể import HVI_transform
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    
    from net.HVI_transform import RGB_HVI
    import torchvision.transforms as T
    from torchvision.utils import save_image
    from PIL import Image

    print("=== TEST EDGE EXTRACTOR ===")
    # Tìm một ảnh mẫu trong thư mục HVI-CIDNet
    img_path = os.path.join(parent_dir, 'E:/PythonFile/Project/Low-Light-Image-Enhancement/mydata/dataset/dataset/LOLv2-real/Test/Input/00787.png')
    
    if os.path.exists(img_path):
        print(f"Đang đọc ảnh: {img_path}")
        # Đọc ảnh RGB
        img = Image.open(img_path).convert('RGB')
        # Chuyển thành tensor [1, 3, H, W] trong khoảng [0, 1]
        x = T.ToTensor()(img).unsqueeze(0)
        
        # 1. Khởi tạo bộ chuyển đổi HVI
        hvi_trans = RGB_HVI()
        hvi = hvi_trans.HVIT(x)
        print("Kích thước HVI:", hvi.shape)
        
        # 2. Tách thành các nhánh để thử nghiệm
        hv = hvi[:, 0:2, :, :] # 2 kênh H và V
        i = hvi[:, 2:3, :, :]  # kênh cường độ sáng I
        
        # 3. Thử nghiệm trên nhánh HV (2 kênh)
        hv_extractor = EdgeExtractor(in_channels=2, blur_kernel_size=9, blur_sigma=3.0)
        # Gộp thành 1 ảnh xám cạnh bằng average_channels=True
        edge_hv_avg = hv_extractor(hv, average_channels=True)
        # Hoặc giữ nguyên 2 kênh cạnh
        edge_hv_2ch = hv_extractor(hv, average_channels=False)
        
        # 4. Thử nghiệm trên nhánh I (1 kênh)
        i_extractor = EdgeExtractor(in_channels=1, blur_kernel_size=3)
        edge_i = i_extractor(i)
        
        # 5. Lưu ảnh kết quả ra thư mục output_test_edges
        out_dir = os.path.join(parent_dir, 'output_test_edges')
        os.makedirs(out_dir, exist_ok=True)
        
        # Normalize=True giúp stretch giá trị pixel ra khoảng [0, 1] để dễ nhìn
        save_image(x, os.path.join(out_dir, '0_original.png'))
        save_image(edge_hv_avg, os.path.join(out_dir, '1_edge_HV_avg.png'), normalize=True)
        save_image(edge_i, os.path.join(out_dir, '2_edge_I.png'), normalize=True)
        
        print(f"Đã trích xuất xong! Các ảnh kết quả được lưu tại: {out_dir}")
    else:
        print(f"Không tìm thấy ảnh mẫu {img_path}.")
