"""
generate_stage1_gt_retinex.py
─────────────────────────────────────────────────────────────────────────────
Tạo ảnh GT trung gian cho Stage 1 dựa trên Lý thuyết Retinex.

Theo Retinex: S (ảnh) = R (Reflectance - Bản chất vật lý/Chi tiết) × I (Illumination - Ánh sáng)

Thuật toán:
    1. Tách I_gt (ánh sáng của ảnh GT) = làm mờ ( max_channels(GT) )
    2. Tách R_gt (chi tiết/màu của GT) = GT / I_gt
    3. Tách I_in (ánh sáng của input)  = làm mờ ( max_channels(Input) )
    4. Trộn ánh sáng (I_mid): I_mid = I_in + alpha * (I_gt - I_in)
    5. Tạo GT_S1 = R_gt * I_mid

Cách dùng:
  # Debug 1 cặp ảnh để xem biểu đồ Retinex
  python generate_stage1_gt_retinex.py --input_dir path/low/1.png --gt_dir path/high/1.png --output_dir ./debug --alpha 0.3 --debug

  # Chạy batch toàn dataset
  python generate_stage1_gt_retinex.py --input_dir path/low/ --gt_dir path/high/ --output_dir path/stage1_gt_retinex/ --alpha 0.3
─────────────────────────────────────────────────────────────────────────────
"""

import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def gaussian_kernel(kernel_size=31, sigma=15.0):
    """Tạo kernel Gaussian 2D."""
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    kernel = (1./(2.*np.pi*variance)) * torch.exp(
        -torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance)
    )
    kernel = kernel / torch.sum(kernel)
    return kernel.view(1, 1, kernel_size, kernel_size)

def gaussian_blur(x, sigma=15.0):
    """Làm mờ Tensor [B, 1, H, W] để tạo Illumination Map trơn tru."""
    k_size = int(2 * np.ceil(3 * sigma) + 1)
    kernel = gaussian_kernel(k_size, sigma).to(x.device)
    pad = k_size // 2
    x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    return F.conv2d(x_padded, kernel, groups=1)


class RetinexStage1GTGenerator:
    """Tạo GT S1 dựa trên phân rã Retinex."""

    def __init__(self, alpha: float = 0.3, sigma: float = 15.0, device: str = 'cpu'):
        self.alpha = alpha
        self.sigma = sigma
        self.device = torch.device(device)

    def extract_illumination(self, img: torch.Tensor) -> torch.Tensor:
        """
        Ước lượng Illumination Map: Lấy max của 3 kênh màu, sau đó làm mờ
        để ánh sáng trở nên trơn tru (smooth), tránh chứa chi tiết của R.
        """
        # I_raw = max(R, G, B)
        I_raw = torch.max(img, dim=1, keepdim=True)[0]
        # Smooth Illumination
        I_smooth = gaussian_blur(I_raw, sigma=self.sigma)
        return I_smooth.clamp(1e-6, 1.0)

    @torch.no_grad()
    def generate(self, input_low: torch.Tensor, gt_rgb: torch.Tensor) -> dict:
        squeeze = (input_low.dim() == 3)
        if squeeze:
            input_low = input_low.unsqueeze(0)
            gt_rgb    = gt_rgb.unsqueeze(0)

        input_low = input_low.to(self.device)
        gt_rgb    = gt_rgb.to(self.device)

        # 1 & 2: Phân rã ảnh GT -> Reflectance (R_gt) và Illumination (I_gt)
        I_gt = self.extract_illumination(gt_rgb)
        R_gt = gt_rgb / I_gt
        # Không clamp R_gt ở đây vì I_gt là bản làm mờ, gt_rgb/I_gt có thể > 1.0. 
        # Nếu clamp sẽ làm mất các đỉnh sáng (peak brightness).

        # 3: Trích xuất Illumination của ảnh input (tối)
        I_in = self.extract_illumination(input_low)

        # 4: Nội suy ánh sáng (Illumination Blending)
        # Sáng hơn input 1 chút, nhưng vẫn giữ bản chất ánh sáng môi trường
        I_mid = I_in + self.alpha * (I_gt - I_in)

        # 5: Tái tạo ảnh Stage 1 GT
        # Kết hợp chi tiết hoàn hảo (R_gt) với ánh sáng yếu trung gian (I_mid)
        gt_s1_rgb = R_gt * I_mid
        gt_s1_rgb = gt_s1_rgb.clamp(0.0, 1.0)

        if squeeze:
            gt_s1_rgb = gt_s1_rgb.squeeze(0)

        return {
            'gt_s1_rgb': gt_s1_rgb,
            'debug': {
                'I_gt':  I_gt.squeeze(0) if squeeze else I_gt,
                'R_gt':  R_gt.squeeze(0) if squeeze else R_gt,
                'I_in':  I_in.squeeze(0) if squeeze else I_in,
                'I_mid': I_mid.squeeze(0) if squeeze else I_mid,
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# Utils & CLI
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.JPG', '.JPEG'}

def get_image_pairs(input_dir: Path, gt_dir: Path) -> list:
    input_files = {f.stem: f for f in input_dir.iterdir() if f.suffix in IMAGE_EXTENSIONS}
    gt_files    = {f.stem: f for f in gt_dir.iterdir()    if f.suffix in IMAGE_EXTENSIONS}
    common = sorted(set(input_files) & set(gt_files))
    if common:
        return [(str(input_files[s]), str(gt_files[s]), input_files[s].name) for s in common]
    return []

def run_batch(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.input_dir)
    gt_dir    = Path(args.gt_dir)

    pairs = get_image_pairs(input_dir, gt_dir)
    if not pairs:
        print("[ERROR] Không tìm thấy cặp ảnh nào.")
        return

    generator = RetinexStage1GTGenerator(alpha=args.alpha, sigma=args.sigma, device=args.device)
    to_tensor = transforms.ToTensor()
    to_pil    = transforms.ToPILImage()

    for input_path, gt_path, filename in tqdm(pairs, desc="Tạo Retinex GT"):
        input_img = to_tensor(Image.open(input_path).convert('RGB'))
        gt_img    = to_tensor(Image.open(gt_path).convert('RGB'))
        result    = generator.generate(input_img, gt_img)
        to_pil(result['gt_s1_rgb'].cpu()).save(str(output_dir / filename))

def run_debug(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    to_tensor = transforms.ToTensor()
    to_pil    = transforms.ToPILImage()

    input_img = to_tensor(Image.open(args.input_dir).convert('RGB'))
    gt_img    = to_tensor(Image.open(args.gt_dir).convert('RGB'))

    generator = RetinexStage1GTGenerator(alpha=args.alpha, sigma=args.sigma, device=args.device)
    result    = generator.generate(input_img, gt_img)
    debug     = result['debug']

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('Retinex Decomposition: S = Reflectance × Illumination', fontsize=16)

    def show_gray(ax, t, title):
        arr = t.squeeze().cpu().numpy()
        im  = ax.imshow(arr, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def show_rgb(ax, t, title):
        arr = t.permute(1, 2, 0).cpu().numpy().clip(0, 1)
        ax.imshow(arr)
        ax.set_title(title)
        ax.axis('off')

    # Hàng 1: Phân rã GT
    show_rgb(axes[0, 0], gt_img, '1. GT Gốc (S_gt)')
    show_gray(axes[0, 1], debug['I_gt'], '2. Illumination của GT (I_gt)')
    show_rgb(axes[0, 2], debug['R_gt'], '3. Reflectance của GT (R_gt)\n[Bản chất màu & chi tiết]')
    axes[0,3].axis('off')

    # Hàng 2: Áp dụng ánh sáng yếu
    show_rgb(axes[1, 0], input_img, '4. Input Tối (S_in)')
    show_gray(axes[1, 1], debug['I_in'], '5. Illumination của Input (I_in)')
    show_gray(axes[1, 2], debug['I_mid'], f'6. Ánh sáng hòa trộn (I_mid)\nI_in + {args.alpha}*(I_gt - I_in)')
    show_rgb(axes[1, 3], result['gt_s1_rgb'], '7. Kết quả GT S1\nGT_S1 = R_gt × I_mid')

    plt.tight_layout()
    fig_path = str(output_dir / 'retinex_debug.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n[✓] Đã lưu biểu đồ phân tích Retinex tại: {fig_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Tạo GT Stage 1 bằng Retinex Decomposition')
    parser.add_argument('--input_dir',  required=True, type=str)
    parser.add_argument('--gt_dir',     required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--alpha',      type=float, default=0.3, help='Hệ số trộn ánh sáng')
    parser.add_argument('--sigma',      type=float, default=15.0, help='Độ mờ của Illumination Map')
    parser.add_argument('--debug',      action='store_true', help='Debug 1 cặp ảnh (vẽ chart)')
    parser.add_argument('--device',     type=str, default='cpu')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        run_debug(args)
    else:
        run_batch(args)
