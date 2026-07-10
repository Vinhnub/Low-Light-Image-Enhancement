"""
generate_stage1_gt.py
─────────────────────────────────────────────────────────────────────────────
Tạo ảnh GT trung gian cho Stage 1: tăng ít sáng, giữ chi tiết rất tốt.

Nguyên lý (Luminance Blending):
    1. Lấy ảnh GT (có detail hoàn hảo, brightness đúng)
    2. Scale brightness GT xuống gần input → chỉ sáng hơn input một chút
    3. Detail 100% từ GT, không bị biến dạng

Công thức:
    lum_input = mean brightness của input_low
    lum_gt    = mean brightness của gt_rgb
    lum_mid   = lum_input + α × (lum_gt - lum_input)
    
    gt_s1 = gt_rgb × (lum_mid / lum_gt)

    α = 0.3 → gt_s1 sáng hơn input 30% khoảng cách tới GT
    α = 0.5 → gt_s1 sáng ở giữa input và GT
    α = 0.1 → gt_s1 gần như tối bằng input, nhưng chi tiết rõ

Cách dùng:
  # Debug 1 cặp ảnh
  python generate_stage1_gt.py --input_dir img_low.png --gt_dir img_gt.png --output_dir ./debug --alpha 0.3 --debug

  # Batch toàn dataset
  python generate_stage1_gt.py --input_dir path/to/low/ --gt_dir path/to/high/ --output_dir path/to/stage1_gt/ --alpha 0.3
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Core: Stage1GTGenerator
# ─────────────────────────────────────────────────────────────────────────────

class Stage1GTGenerator:
    """
    Tạo ảnh GT cho Stage 1: giữ 100% detail của GT, chỉ giảm brightness.

    Công thức:
        lum_mid = lum_input + α × (lum_gt - lum_input)
        scale   = lum_mid / lum_gt
        gt_s1   = gt_rgb × scale

    → Detail giữ nguyên (nhân tỷ lệ, không cộng/trừ)
    → Brightness giảm xuống mức trung gian giữa input và GT
    → Tương phản tương đối bảo toàn hoàn toàn
    """

    def __init__(self, alpha: float = 0.3, mode: str = 'global'):
        """
        Args:
            alpha : Mức sáng trung gian (0–1).
                      0.0 → gt_s1 tối bằng input (chỉ có detail)
                      0.3 → sáng hơn input 30% ← KHUYẾN NGHỊ
                      0.5 → giữa input và GT
                      1.0 → gt_s1 = GT gốc (vô nghĩa)
            mode  : 'global' → scale theo mean brightness toàn ảnh
                    'local'  → scale theo brightness cục bộ (per-pixel, qua blur)
        """
        self.alpha = alpha
        self.mode  = mode

        self.to_tensor = transforms.ToTensor()
        self.to_pil    = transforms.ToPILImage()

    def _luminance(self, rgb: torch.Tensor) -> torch.Tensor:
        """Tính luminance (brightness) theo ITU-R BT.601. Shape: [B,1,H,W]"""
        return (
            0.299 * rgb[:, 0:1] +
            0.587 * rgb[:, 1:2] +
            0.114 * rgb[:, 2:3]
        )

    @torch.no_grad()
    def generate(
        self,
        input_low: torch.Tensor,
        gt_rgb:    torch.Tensor,
    ) -> dict:
        """
        Tạo gt_s1 từ cặp (input_low, gt_rgb).

        Args:
            input_low : Tensor [3, H, W] hoặc [B, 3, H, W], range [0, 1]
            gt_rgb    : Tensor [3, H, W] hoặc [B, 3, H, W], range [0, 1]

        Returns:
            dict:
                'gt_s1_rgb' : GT cho Stage 1 (RGB)
                'debug'     : dict chứa các giá trị trung gian
        """
        squeeze = (input_low.dim() == 3)
        if squeeze:
            input_low = input_low.unsqueeze(0)
            gt_rgb    = gt_rgb.unsqueeze(0)

        eps = 1e-6

        # Tính luminance
        lum_input = self._luminance(input_low)  # [B,1,H,W]
        lum_gt    = self._luminance(gt_rgb)     # [B,1,H,W]

        if self.mode == 'global':
            # Mean brightness toàn ảnh → 1 giá trị scale cho toàn ảnh
            mean_input = lum_input.mean(dim=[2, 3], keepdim=True)  # [B,1,1,1]
            mean_gt    = lum_gt.mean(dim=[2, 3], keepdim=True)     # [B,1,1,1]

            # Mức sáng mục tiêu: nằm giữa input và GT
            lum_target = mean_input + self.alpha * (mean_gt - mean_input)

            # Scale ratio: giảm brightness GT xuống lum_target
            scale = lum_target / (mean_gt + eps)  # [B,1,1,1] → broadcast

        else:
            raise ValueError(f"Mode '{self.mode}' chưa được hỗ trợ. Dùng 'global'.")

        # Scale GT → giữ 100% detail, chỉ thay đổi brightness
        gt_s1_rgb = gt_rgb * scale
        gt_s1_rgb = gt_s1_rgb.clamp(0.0, 1.0)

        # Debug info
        lum_s1 = self._luminance(gt_s1_rgb)

        if squeeze:
            gt_s1_rgb = gt_s1_rgb.squeeze(0)

        return {
            'gt_s1_rgb': gt_s1_rgb,
            'debug': {
                'mean_lum_input':  lum_input.mean().item(),
                'mean_lum_gt':     lum_gt.mean().item(),
                'mean_lum_s1':     lum_s1.mean().item(),
                'scale':           scale.mean().item(),
                'alpha':           self.alpha,
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# File matching utils
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.JPG', '.JPEG'}


def get_image_pairs(input_dir: Path, gt_dir: Path) -> list:
    """Lấy danh sách cặp (input_path, gt_path, filename) khớp theo tên file."""
    input_files = {f.stem: f for f in input_dir.iterdir() if f.suffix in IMAGE_EXTENSIONS}
    gt_files    = {f.stem: f for f in gt_dir.iterdir()    if f.suffix in IMAGE_EXTENSIONS}

    common = sorted(set(input_files) & set(gt_files))

    if common:
        return [(str(input_files[s]), str(gt_files[s]), input_files[s].name)
                for s in common]

    # Fallback: sort rồi zip
    print("[WARN] Tên file không khớp → match theo thứ tự.")
    inp_sorted = sorted(input_files.values())
    gt_sorted  = sorted(gt_files.values())
    return [(str(i), str(g), i.name) for i, g in zip(inp_sorted, gt_sorted)]


# ─────────────────────────────────────────────────────────────────────────────
# Batch generation
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(args):
    """Xử lý toàn bộ dataset, lưu gt_s1 vào output_dir."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)
    gt_dir    = Path(args.gt_dir)

    print(f"  Input  : {input_dir}")
    print(f"  GT     : {gt_dir}")
    print(f"  Output : {output_dir}")
    print(f"  Alpha  : {args.alpha}\n")

    pairs = get_image_pairs(input_dir, gt_dir)
    if not pairs:
        print("[ERROR] Không tìm thấy cặp ảnh nào.")
        return

    print(f"Tìm thấy {len(pairs)} cặp ảnh.\n")

    generator = Stage1GTGenerator(alpha=args.alpha)
    to_tensor = transforms.ToTensor()
    to_pil    = transforms.ToPILImage()

    errors = []
    for input_path, gt_path, filename in tqdm(pairs, desc="Generating Stage1 GT"):
        try:
            input_img = to_tensor(Image.open(input_path).convert('RGB'))
            gt_img    = to_tensor(Image.open(gt_path).convert('RGB'))
            result    = generator.generate(input_img, gt_img)
            to_pil(result['gt_s1_rgb'].cpu()).save(str(output_dir / filename))
        except Exception as e:
            errors.append((filename, str(e)))

    print(f"\n{'='*50}")
    print(f"  Hoàn thành: {len(pairs) - len(errors)}/{len(pairs)} ảnh")
    if errors:
        for fname, err in errors:
            print(f"  [ERROR] {fname}: {err}")
    print(f"  Kết quả: {output_dir}")
    print(f"{'='*50}")


# ─────────────────────────────────────────────────────────────────────────────
# Debug: 1 cặp ảnh + visualize
# ─────────────────────────────────────────────────────────────────────────────

def run_debug(args):
    """Debug mode: xử lý 1 cặp ảnh, lưu so sánh."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    to_tensor = transforms.ToTensor()
    to_pil    = transforms.ToPILImage()

    input_img = to_tensor(Image.open(args.input_dir).convert('RGB'))
    gt_img    = to_tensor(Image.open(args.gt_dir).convert('RGB'))

    generator = Stage1GTGenerator(alpha=args.alpha)
    result    = generator.generate(input_img, gt_img)
    debug     = result['debug']

    # Lưu ảnh
    to_pil(input_img).save(str(output_dir / 'input_low.png'))
    to_pil(gt_img).save(str(output_dir / 'gt_rgb.png'))
    to_pil(result['gt_s1_rgb'].cpu()).save(str(output_dir / 'gt_s1.png'))

    # Vẽ so sánh
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f'Stage 1 GT — Luminance Blending (alpha={args.alpha})\n'
        f'gt_s1 = gt_rgb × scale   |   scale = {debug["scale"]:.4f}',
        fontsize=13
    )

    def show_rgb(ax, t, title, lum):
        arr = t.permute(1, 2, 0).cpu().numpy().clip(0, 1)
        ax.imshow(arr)
        ax.set_title(f'{title}\nmean lum = {lum:.4f}', fontsize=10)
        ax.axis('off')

    show_rgb(axes[0], input_img,           'input_low',          debug['mean_lum_input'])
    show_rgb(axes[1], result['gt_s1_rgb'], f'gt_s1 (α={args.alpha})', debug['mean_lum_s1'])
    show_rgb(axes[2], gt_img,              'gt_rgb (GT gốc)',    debug['mean_lum_gt'])

    plt.tight_layout()
    fig_path = str(output_dir / 'debug_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Thống kê
    print(f"\n─── Thống kê ────────────────────────────────────────────")
    print(f"  Mean Lum input : {debug['mean_lum_input']:.4f}")
    print(f"  Mean Lum GT    : {debug['mean_lum_gt']:.4f}")
    print(f"  Mean Lum gt_s1 : {debug['mean_lum_s1']:.4f}")
    print(f"  Scale factor   : {debug['scale']:.4f}")
    print(f"  Alpha          : {debug['alpha']}")
    print(f"─────────────────────────────────────────────────────────")
    print(f"\n  gt_s1 sáng hơn input {((debug['mean_lum_s1']/debug['mean_lum_input'])-1)*100:.1f}%")
    print(f"  gt_s1 tối hơn GT     {(1-(debug['mean_lum_s1']/debug['mean_lum_gt']))*100:.1f}%")
    print(f"\n  Chart: {fig_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Tạo ảnh GT cho Stage 1: tăng ít sáng + giữ 100% detail từ GT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Debug 1 cặp ảnh
  python generate_stage1_gt.py --input_dir low/1.png --gt_dir high/1.png --output_dir ./debug --alpha 0.3 --debug

  # Batch toàn dataset
  python generate_stage1_gt.py --input_dir path/to/low/ --gt_dir path/to/high/ --output_dir path/to/stage1_gt/ --alpha 0.3

Chọn alpha:
  0.1 → gt_s1 rất tối (gần input), chỉ có detail
  0.3 → sáng nhẹ hơn input ← khuyến nghị
  0.5 → giữa input và GT
  0.7 → gần GT
        """
    )
    parser.add_argument('--input_dir',  required=True, type=str)
    parser.add_argument('--gt_dir',     required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--alpha',      type=float, default=0.3,
                        help='Mức sáng: 0=tối bằng input, 1=sáng bằng GT (default: 0.3)')
    parser.add_argument('--debug',      action='store_true',
                        help='Debug mode: 1 cặp ảnh + chart so sánh')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print("=" * 50)
    print("  Stage 1 GT — Luminance Blending")
    print("=" * 50)
    print(f"  Alpha : {args.alpha}")
    print(f"  Mode  : {'DEBUG' if args.debug else 'BATCH'}")
    print("=" * 50 + "\n")

    if args.debug:
        run_debug(args)
    else:
        run_batch(args)
