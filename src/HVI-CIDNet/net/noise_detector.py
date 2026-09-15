import os
import sys
import math
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# Đảm bảo import được module net khi chạy độc lập
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


class ImmerkaerNoiseEstimator(nn.Module):
    """
    Ước lượng mức độ nhiễu (Noise Level Estimation) bằng toán tử Immerkaer.
    Dựa trên bài báo kinh điển:
    J. Immerkaer, "Fast Noise Variance Estimation", Computer Vision and Image Understanding, 1996.

    Ưu điểm:
    - Rất nhanh, không cần huấn luyện (training-free).
    - Triệt tiêu gần như hoàn toàn các thành phần gradient bậc 1 (mặt phẳng dốc, cạnh mượt)
      và tập trung trích xuất thành phần nhiễu hạt ngẫu nhiên (Gaussian/Poisson noise).
    """
    def __init__(self, in_channels=1):
        super(ImmerkaerNoiseEstimator, self).__init__()
        self.in_channels = in_channels

        # Kernel Immerkaer 3x3:
        # [  1  -2   1 ]
        # [ -2   4  -2 ]
        # [  1  -2   1 ]
        kernel = torch.tensor([
            [ 1.0, -2.0,  1.0],
            [-2.0,  4.0, -2.0],
            [ 1.0, -2.0,  1.0]
        ], dtype=torch.float32)

        # Chuẩn hóa để conv theo từng kênh (grouped convolution)
        weight = kernel.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        self.register_buffer('weight', weight)

        # Hệ số chuẩn hóa: sqrt(pi / 2) / 6 ≈ 0.20888568
        self.coef = math.sqrt(math.pi / 2.0) / 6.0

    def forward(self, x):
        """
        x: Tensor [B, C, H, W]
        Trả về:
            noise_map:   [B, 1, H, W] hoặc [B, C, H, W] - Bản đồ phân bố nhiễu cục bộ
            noise_level: [B, 1] - Ước lượng độ lệch chuẩn sigma tổng thể của mỗi ảnh
        """
        B, C, H, W = x.shape
        # Replicate pad 1 pixel để giữ nguyên kích thước không gian [H, W]
        x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
        residual = F.conv2d(x_pad, self.weight, groups=self.in_channels)
        abs_residual = torch.abs(residual)

        # Bản đồ nhiễu cục bộ
        if C > 1:
            local_noise_map = torch.mean(abs_residual, dim=1, keepdim=True) * self.coef
        else:
            local_noise_map = abs_residual * self.coef

        # Tính sigma toàn cục cho mỗi ảnh trong batch
        # Immerkaer formula: sigma = sqrt(pi/2) / (6 * (W-2) * (H-2)) * sum(|I * M|)
        noise_level = local_noise_map.view(B, -1).mean(dim=1, keepdim=True)

        return local_noise_map, noise_level


class EdgeAwareNoiseDetector(nn.Module):
    """
    Phát hiện nhiễu loại trừ cạnh biên (Edge-Preserving / Edge-Aware Noise Detection).

    Vấn đề:
    Các bộ lọc tần số cao thông thường rất dễ nhầm lẫn giữa "Cạnh thật sắc nét" (Edges)
    và "Nhiễu ngẫu nhiên" (Noise).

    Giải pháp:
    1. Trích xuất phần dư tần số cao bằng High-pass Filter (hoặc Laplacian).
    2. Đồng thời tính độ lớn gradient biên cạnh (Sobel).
    3. Trọng số hóa: Giảm ảnh hưởng tại các vị trí có biên cạnh sắc nét (Edge Gating).
       Vùng nào có gradient cao -> Đó là CẠNH -> Triệt tiêu khỏi bản đồ nhiễu.
       Vùng nào gradient thấp nhưng dao động tần số cao -> Đó là NHIỄU!
    """
    def __init__(self, in_channels=1, kernel_size=3):
        super(EdgeAwareNoiseDetector, self).__init__()
        self.in_channels = in_channels

        # 1. Bộ lọc làm mờ (Low-pass) để tính phần dư High-pass
        blur_kernel = torch.ones(in_channels, 1, kernel_size, kernel_size) / (kernel_size ** 2)
        self.register_buffer('blur_kernel', blur_kernel)

        # 2. Bộ lọc Sobel để phát hiện biên cạnh
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        # Bước 1: Tính thành phần tần số cao (High-pass residual)
        x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
        smooth = F.conv2d(x_pad, self.blur_kernel, groups=self.in_channels)
        residual = torch.abs(x - smooth)

        # Bước 2: Tính độ lớn cạnh Sobel
        gx = F.conv2d(x_pad, self.sobel_x, groups=self.in_channels)
        gy = F.conv2d(x_pad, self.sobel_y, groups=self.in_channels)
        edge_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        # Gộp kênh nếu C > 1
        if C > 1:
            residual = residual.mean(dim=1, keepdim=True)
            edge_mag = edge_mag.mean(dim=1, keepdim=True)

        # Bước 3: Chuẩn hóa edge_mag về [0, 1] cho mỗi ảnh
        edge_flat = edge_mag.view(B, -1)
        edge_min = edge_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        edge_max = edge_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        edge_norm = (edge_mag - edge_min) / (edge_max - edge_min + 1e-6)

        # Bước 4: Edge Gating: Vùng có cạnh mạnh (edge_norm -> 1) sẽ bị triệt tiêu
        edge_gate = 1.0 - torch.sigmoid(10.0 * (edge_norm - 0.2))
        noise_map = residual * edge_gate

        # Ước lượng độ lệch chuẩn
        noise_level = noise_map.view(B, -1).mean(dim=1, keepdim=True)

        return noise_map, noise_level


class IlluminationGuidedNoiseDetector(nn.Module):
    """
    Module phát hiện nhiễu đặc thù cho HVI-CIDNet (Low-Light Image Enhancement).

    Đặc trưng vật lý của ảnh thiếu sáng:
    1. Nhiễu trong ảnh thiếu sáng tuân theo mô hình Poisson-Gaussian (Signal-dependent noise).
    2. Vùng tối (Độ rọi I thấp) có tỉ số tín hiệu trên nhiễu (SNR) cực kỳ thấp. Khi cảm biến
       hoặc thuật toán đẩy sáng, nhiễu tại vùng tối sẽ bùng nổ mạnh nhất.
    3. Kênh màu sắc (HV) thường bị nhiễu hạt màu (chroma noise), trong khi kênh I bị nhiễu luma.

    Hoạt động:
    - Kết hợp EdgeAwareNoiseDetector trên nhánh HV và/hoặc I.
    - Dùng bản đồ độ rọi I để điều tiết (Weighting) độ nhạy phát hiện nhiễu ở vùng tối.
    """
    def __init__(self, in_channels_hv=2, in_channels_i=1, dark_boost_factor=1.5):
        super(IlluminationGuidedNoiseDetector, self).__init__()
        self.hv_detector = EdgeAwareNoiseDetector(in_channels=in_channels_hv)
        self.i_detector = EdgeAwareNoiseDetector(in_channels=in_channels_i)
        self.dark_boost = dark_boost_factor

    def forward(self, hv, i):
        """
        hv: [B, 2, H, W] - Nhánh sắc độ & giá trị H, V
        i:  [B, 1, H, W] - Nhánh độ rọi Illumination I (trong khoảng [0, 1])

        Trả về:
            guided_noise_map: [B, 1, H, W] - Bản đồ nhiễu tổng hợp được dẫn đường bởi ánh sáng
            noise_info: dict chứa chi tiết noise_hv, noise_i, noise_level
        """
        # 1. Trích xuất bản đồ nhiễu từ 2 nhánh
        noise_hv, level_hv = self.hv_detector(hv)
        noise_i, level_i   = self.i_detector(i)

        # 2. Tạo trọng số khuếch đại cho vùng tối (Dark Weight)
        # Vùng càng tối (i càng nhỏ) -> Trọng số càng lớn
        dark_weight = 1.0 + self.dark_boost * torch.clamp(1.0 - i, min=0.0, max=1.0)

        # 3. Tổng hợp bản đồ nhiễu (kết hợp màu sắc + độ rọi + trọng số tối)
        combined_noise = (0.7 * noise_hv + 0.3 * noise_i) * dark_weight

        # 4. Chuẩn hóa bản đồ nhiễu về khoảng [0, 1] trên mỗi ảnh để dễ đưa vào Attention/Gate
        B = combined_noise.shape[0]
        c_flat = combined_noise.view(B, -1)
        c_min = c_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        c_max = c_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        norm_noise_map = (combined_noise - c_min) / (c_max - c_min + 1e-6)

        overall_level = norm_noise_map.view(B, -1).mean(dim=1, keepdim=True)

        return norm_noise_map, {
            'noise_hv_level': level_hv,
            'noise_i_level': level_i,
            'overall_level': overall_level
        }


class NoiseDetector(nn.Module):
    """
    Giao diện Detect Nhiễu tổng quát (Universal Noise Detector) cho HVI-CIDNet.

    Hỗ trợ 4 chế độ hoạt động:
    1. 'immerkaer': Thuật toán thống kê toán học Immerkaer cực nhanh, chính xác cho nhiễu ngẫu nhiên.
    2. 'edge_aware': Lọc tần số cao có trừ biên cạnh sắc nét (tránh nhầm lẫn cạnh với nhiễu).
    3. 'illumination_guided': Dành riêng cho cặp (HV, I) trong không gian HVI.
    4. 'hybrid': Kết hợp cả Immerkaer và Edge-Aware để cho kết quả cân bằng nhất.
    """
    def __init__(self, in_channels=3, mode='edge_aware', dark_boost=1.5):
        super(NoiseDetector, self).__init__()
        self.in_channels = in_channels
        self.mode = mode

        if mode == 'immerkaer':
            self.detector = ImmerkaerNoiseEstimator(in_channels=in_channels)
        elif mode == 'edge_aware':
            self.detector = EdgeAwareNoiseDetector(in_channels=in_channels)
        elif mode == 'illumination_guided':
            self.detector = IlluminationGuidedNoiseDetector(
                in_channels_hv=2 if in_channels >= 2 else 1,
                in_channels_i=1,
                dark_boost_factor=dark_boost
            )
        elif mode == 'hybrid':
            self.detector_imm = ImmerkaerNoiseEstimator(in_channels=in_channels)
            self.detector_edge = EdgeAwareNoiseDetector(in_channels=in_channels)
        else:
            raise ValueError(f"Mode '{mode}' không được hỗ trợ. Chọn: immerkaer, edge_aware, illumination_guided, hybrid")

    def forward(self, x, guide_i=None):
        """
        Tham số:
            x:       [B, C, H, W] - Ảnh đầu vào (RGB hoặc kênh HV hoặc toàn bộ HVI)
            guide_i: [B, 1, H, W] (Tùy chọn) - Kênh độ rọi I nếu dùng mode 'illumination_guided'

        Trả về:
            noise_map:   [B, 1, H, W] - Bản đồ phân bố mức độ nhiễu
            noise_level: [B, 1]       - Giá trị độ lệch chuẩn ước lượng tổng thể
        """
        if self.mode == 'illumination_guided':
            if guide_i is None:
                # Nếu không truyền guide_i riêng, tự tách kênh cuối của x làm guide_i
                if x.shape[1] == 3:
                    hv = x[:, :2, :, :]
                    i = x[:, 2:3, :, :]
                else:
                    raise ValueError("Cần cung cấp guide_i hoặc x phải có 3 kênh (HV + I)")
            else:
                hv = x
                i = guide_i
            return self.detector(hv, i)

        elif self.mode == 'hybrid':
            n_map1, lvl1 = self.detector_imm(x)
            n_map2, lvl2 = self.detector_edge(x)
            noise_map = 0.5 * (n_map1 + n_map2)
            noise_level = 0.5 * (lvl1 + lvl2)
            return noise_map, noise_level

        else:
            return self.detector(x)

    def detect_and_visualize(self, x, guide_i=None, original_img=None, output_dir='output_noise', prefix='noise'):
        """
        Thực hiện detect nhiễu và tự động xuất ra ảnh biểu đồ trực quan hóa:
        - {prefix}_dashboard.png: Ảnh tổng hợp 3 trong 1 (Ảnh gốc + Heatmap nhiễu + Biểu đồ phân bố)
        - {prefix}_heatmap.png:   Bản đồ nhiệt nhiễu độc lập
        - {prefix}_histogram.png: Biểu đồ phân bố tần suất nhiễu độc lập
        """
        res = self.forward(x, guide_i=guide_i)
        if isinstance(res, tuple):
            noise_map, noise_info = res
            if isinstance(noise_info, dict):
                noise_level = noise_info.get('overall_level', noise_info.get('noise_level', 0.0))
            else:
                noise_level = noise_info
        else:
            noise_map = res
            noise_level = noise_map.mean()

        # Hiển thị ảnh gốc
        img_to_show = original_img if original_img is not None else x

        saved_files = visualize_and_save_noise(
            image_input=img_to_show,
            noise_map=noise_map,
            noise_level=noise_level,
            output_dir=output_dir,
            filename_prefix=prefix,
            mode_name=self.mode
        )
        return noise_map, noise_level, saved_files


def visualize_and_save_noise(
    image_input,
    noise_map,
    noise_level,
    output_dir='output_noise',
    filename_prefix='noise',
    mode_name='Noise Detection'
):
    """
    Xuất ra file ảnh biểu thị trực quan hóa và biểu đồ phân bố nhiễu:
    1. {prefix}_dashboard.png: Bảng điều khiển gồm:
       - Panel 1: Ảnh đầu vào (Input Image)
       - Panel 2: Bản đồ nhiệt nhiễu (Noise Heatmap kèm Colorbar)
       - Panel 3: Biểu đồ phân bố tần suất nhiễu (Noise Intensity Histogram với các chỉ số thống kê)
    2. {prefix}_heatmap.png: Bản đồ nhiệt màu Inferno độc lập
    3. {prefix}_histogram.png: Biểu đồ phân bố tần suất độc lập
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # 1. Chuyển đổi ảnh đầu vào sang dạng numpy [H, W, C]
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert('RGB')
        img_np = np.array(pil_img) / 255.0
    elif isinstance(image_input, torch.Tensor):
        inp = image_input.detach().cpu()
        if inp.dim() == 4:
            inp = inp[0]  # Lấy ảnh đầu tiên trong batch
        if inp.shape[0] in [1, 2, 3]:
            if inp.shape[0] == 1:
                inp = inp.repeat(3, 1, 1)
            elif inp.shape[0] == 2:
                # Kênh HV: ghép thêm kênh thứ 3 để hiển thị RGB
                ch3 = torch.sqrt(inp[0:1]**2 + inp[1:2]**2)
                inp = torch.cat([(inp[0:1]+1)/2, (inp[1:2]+1)/2, ch3], dim=0)
            img_np = inp.permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0.0, 1.0)
        else:
            img_np = inp.numpy()
    elif isinstance(image_input, Image.Image):
        img_np = np.array(image_input.convert('RGB')) / 255.0
    else:
        img_np = np.array(image_input)

    # 2. Chuyển đổi Noise Map sang numpy 2D [H, W]
    if isinstance(noise_map, torch.Tensor):
        n_map = noise_map.detach().cpu()
        if n_map.dim() == 4:
            n_map = n_map[0, 0]
        elif n_map.dim() == 3:
            n_map = n_map[0]
        n_map_np = n_map.numpy()
    else:
        n_map_np = np.array(noise_map)

    # 3. Lấy giá trị scalar cho noise_level (sigma)
    if isinstance(noise_level, torch.Tensor):
        sigma_val = float(noise_level.detach().cpu().mean().item())
    elif isinstance(noise_level, (int, float)):
        sigma_val = float(noise_level)
    else:
        sigma_val = float(np.mean(n_map_np))

    # 4. Tính toán các thông số thống kê phân bố
    n_flat = n_map_np.flatten()
    mean_val = float(np.mean(n_flat))
    median_val = float(np.median(n_flat))
    std_val = float(np.std(n_flat))
    p95_val = float(np.percentile(n_flat, 95))
    max_val = float(np.max(n_flat))

    # --------------------------------------------------------------------------
    # A. VẼ DASHBOARD TỔNG HỢP (3 TRONG 1)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor('#f8f9fa')

    # Panel 1: Ảnh gốc
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image", fontsize=13, fontweight='bold', pad=10)
    axes[0].axis('off')

    # Panel 2: Bản đồ nhiệt nhiễu (Heatmap)
    im2 = axes[1].imshow(n_map_np, cmap='inferno')
    axes[1].set_title(f"Noise Heatmap [{mode_name.upper()}]", fontsize=13, fontweight='bold', pad=10)
    axes[1].axis('off')
    cbar = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Noise Magnitude", rotation=270, labelpad=14, fontsize=10)

    # Panel 3: Biểu đồ phân bố tần suất nhiễu (Histogram)
    axes[2].set_facecolor('#ffffff')
    counts, bins, patches = axes[2].hist(
        n_flat, bins=60, color='#e74c3c', alpha=0.75,
        density=True, edgecolor='black', linewidth=0.5, label='Residual Distribution'
    )
    # Vẽ các đường thống kê
    axes[2].axvline(mean_val, color='#2980b9', linestyle='--', linewidth=2, label=f'Mean (μ) = {mean_val:.4f}')
    axes[2].axvline(median_val, color='#8e44ad', linestyle=':', linewidth=2, label=f'Median = {median_val:.4f}')
    axes[2].axvline(p95_val, color='#d35400', linestyle='-.', linewidth=1.5, label=f'95th Percentile = {p95_val:.4f}')

    axes[2].set_title(f"Noise Histogram (Estimated σ ≈ {sigma_val:.4f})", fontsize=13, fontweight='bold', pad=10)
    axes[2].set_xlabel("Noise Residual Intensity", fontsize=11)
    axes[2].set_ylabel("Probability Density", fontsize=11)
    axes[2].grid(True, linestyle='--', alpha=0.4)
    axes[2].legend(fontsize=9, loc='upper right', framealpha=0.9)

    # Chèn hộp thông số tổng kết
    stats_text = (
        f"Statistics:\n"
        f"• Sigma (σ): {sigma_val:.4f}\n"
        f"• Mean (μ):  {mean_val:.4f}\n"
        f"• Std Dev:   {std_val:.4f}\n"
        f"• Max:       {max_val:.4f}"
    )
    axes[2].text(
        0.05, 0.95, stats_text, transform=axes[2].transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.9)
    )

    dashboard_path = os.path.join(output_dir, f"{filename_prefix}_dashboard.png")
    plt.tight_layout()
    plt.savefig(dashboard_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --------------------------------------------------------------------------
    # B. LƯU BẢN ĐỒ NHIỆT (HEATMAP) ĐỘ PHÂN GIẢI CAO ĐỘC LẬP
    # --------------------------------------------------------------------------
    fig_hm, ax_hm = plt.subplots(figsize=(7, 6))
    im_hm = ax_hm.imshow(n_map_np, cmap='inferno')
    ax_hm.set_title(f"Noise Heatmap ({mode_name})", fontsize=12, fontweight='bold')
    ax_hm.axis('off')
    cbar_hm = fig_hm.colorbar(im_hm, ax=ax_hm, fraction=0.046, pad=0.04)
    cbar_hm.set_label("Noise Intensity", rotation=270, labelpad=12)
    heatmap_path = os.path.join(output_dir, f"{filename_prefix}_heatmap.png")
    fig_hm.savefig(heatmap_path, dpi=200, bbox_inches='tight')
    plt.close(fig_hm)

    # --------------------------------------------------------------------------
    # C. LƯU RIÊNG BIỂU ĐỒ HISTOGRAM ĐỘC LẬP
    # --------------------------------------------------------------------------
    fig_hist, ax_hist = plt.subplots(figsize=(7, 5))
    ax_hist.hist(n_flat, bins=60, color='#e74c3c', alpha=0.75, density=True, edgecolor='black', linewidth=0.5)
    ax_hist.axvline(mean_val, color='#2980b9', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.4f}')
    ax_hist.axvline(median_val, color='#8e44ad', linestyle=':', linewidth=2, label=f'Median = {median_val:.4f}')
    ax_hist.set_title(f"Noise Distribution Histogram (σ ≈ {sigma_val:.4f})", fontsize=12, fontweight='bold')
    ax_hist.set_xlabel("Noise Residual Intensity")
    ax_hist.set_ylabel("Probability Density")
    ax_hist.grid(True, linestyle='--', alpha=0.4)
    ax_hist.legend()
    hist_path = os.path.join(output_dir, f"{filename_prefix}_histogram.png")
    fig_hist.savefig(hist_path, dpi=200, bbox_inches='tight')
    plt.close(fig_hist)

    print(f"-> [Saved] Noise Dashboard (3-in-1): {dashboard_path}")
    print(f"-> [Saved] Noise Heatmap:            {heatmap_path}")
    print(f"-> [Saved] Noise Histogram Chart:    {hist_path}")

    return {
        'dashboard': dashboard_path,
        'heatmap': heatmap_path,
        'histogram': hist_path
    }


# ==============================================================================
# HÀM KIỂM THỬ (UNIT TEST) VÀ CHẠY CLI
# ==============================================================================
if __name__ == '__main__':
    import sys
    import os
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    import argparse
    parser = argparse.ArgumentParser(description='Detect nhiễu ảnh và xuất biểu đồ phân bố nhiễu cho HVI-CIDNet.')
    parser.add_argument('--input', type=str, default=None, help='Đường dẫn ảnh đầu vào cần detect nhiễu')
    parser.add_argument('--output_dir', type=str, default='output_noise_detect', help='Thư mục lưu ảnh biểu đồ nhiễu')
    parser.add_argument('--mode', type=str, default='illumination_guided',
                        choices=['immerkaer', 'edge_aware', 'illumination_guided', 'hybrid'],
                        help='Thuật toán detect nhiễu')
    parser.add_argument('--dark_boost', type=float, default=2.0, help='Hệ số khuếch đại vùng tối trong mode illumination_guided')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 68)
    print(" [NOISE DETECTOR] PHÁT HIỆN NHIỄU & XUẤT BIỂU ĐỒ CHO HVI-CIDNET")
    print("=" * 68)
    print(f"Device: {device}")

    # TRƯỜNG HỢP 1: Người dùng chỉ định ảnh cụ thể qua --input
    if args.input and os.path.exists(args.input):
        print(f"\n[+] Đang xử lý ảnh đầu vào: {args.input}")
        pil_img = Image.open(args.input).convert('RGB')
        img_t = TF.to_tensor(pil_img).unsqueeze(0).to(device)

        detector = NoiseDetector(in_channels=3, mode=args.mode, dark_boost=args.dark_boost).to(device)

        if args.mode == 'illumination_guided':
            # Chuyển đổi sang HVI để lấy đúng nhánh HV và nhánh I
            try:
                from net.HVI_transform import RGB_HVI
                trans = RGB_HVI().to(device)
                hvi = trans.HVIT(img_t)
                hv = hvi[:, :2, :, :]
                i = hvi[:, 2:3, :, :]
                detector_hvi = NoiseDetector(in_channels=2, mode='illumination_guided', dark_boost=args.dark_boost).to(device)
                n_map, n_lvl, paths = detector_hvi.detect_and_visualize(
                    hv, guide_i=i, original_img=img_t, output_dir=args.output_dir, prefix='sample_hvi'
                )
            except Exception as e:
                print(f"Không import được RGB_HVI ({e}), dùng kênh RGB trực tiếp...")
                n_map, n_lvl, paths = detector.detect_and_visualize(
                    img_t, original_img=img_t, output_dir=args.output_dir, prefix='sample'
                )
        else:
            n_map, n_lvl, paths = detector.detect_and_visualize(
                img_t, original_img=img_t, output_dir=args.output_dir, prefix='sample'
            )

        print(f"\n=> ĐÃ XUẤT THÀNH CÔNG BIỂU ĐỒ NHIỄU VÀO THƯ MỤC: {args.output_dir}")

    # TRƯỜNG HỢP 2: Chạy kiểm thử tự động (Unit Test & Thử ảnh mẫu nếu có)
    else:
        print("\n[+] Đang chạy Unit Tests trên các chế độ phát hiện nhiễu...")
        B, C, H, W = 1, 3, 256, 256
        dummy_clean = torch.rand(B, C, H, W, device=device)
        noise = torch.randn_like(dummy_clean) * 0.08
        dummy_noisy = torch.clamp(dummy_clean + noise, 0.0, 1.0)

        # Test và xuất biểu đồ mẫu
        detector = NoiseDetector(in_channels=3, mode='edge_aware').to(device)
        n_map, n_lvl, paths = detector.detect_and_visualize(
            dummy_noisy, original_img=dummy_noisy, output_dir=args.output_dir, prefix='synthetic_noisy'
        )

        # Nếu có ảnh LOLv1 mẫu trong repo, test luôn trên ảnh thật
        sample_img_path = r'E:\PythonFile\Project\Low-Light-Image-Enhancement\mydata\dataset\dataset\LOLv1\test\low\778.png'
        if os.path.exists(sample_img_path):
            print(f"\n[+] Tìm thấy ảnh thiếu sáng thực tế: {sample_img_path}")
            real_pil = Image.open(sample_img_path).convert('RGB')
            real_t = TF.to_tensor(real_pil).unsqueeze(0).to(device)

            from net.HVI_transform import RGB_HVI
            trans = RGB_HVI().to(device)
            hvi = trans.HVIT(real_t)
            hv = hvi[:, :2, :, :]
            i = hvi[:, 2:3, :, :]

            detector_ig = NoiseDetector(in_channels=2, mode='illumination_guided', dark_boost=2.5).to(device)
            n_map_real, n_lvl_real, paths_real = detector_ig.detect_and_visualize(
                hv, guide_i=i, original_img=real_t, output_dir=args.output_dir, prefix='lol_778_real'
            )

        print(f"\n=> TẤT CẢ KIỂM THỬ ĐÃ HOÀN TẤT! Kết quả được lưu tại thư mục: {args.output_dir}")

