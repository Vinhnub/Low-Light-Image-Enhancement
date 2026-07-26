import os
import random
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from os import listdir
from os.path import join

from data.options import option
from data.data import *
from data.util import is_image_file, load_img
from loss.losses import *

class PairedDatasetFromFolder(data.Dataset):
    """
    Dataset loader cho các cặp ảnh đầu vào (Low-light) và ảnh chuẩn (GT High-light).
    Hỗ trợ pad kích thước ảnh về bội số của 8 để khớp với kiến trúc mạng CIDNet.
    """
    def __init__(self, low_dir, high_dir, size=None):
        super(PairedDatasetFromFolder, self).__init__()
        self.low_files = sorted([join(low_dir, x) for x in listdir(low_dir) if is_image_file(x)])
        self.high_files = sorted([join(high_dir, x) for x in listdir(high_dir) if is_image_file(x)])
        
        if len(self.low_files) == 0:
            raise FileNotFoundError(f"Không tìm thấy ảnh trong thư mục low: {low_dir}")
        if len(self.low_files) != len(self.high_files):
            raise ValueError(f"Số lượng ảnh low ({len(self.low_files)}) và high GT ({len(self.high_files)}) không bằng nhau!")
            
        self.size = size
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        im1 = load_img(self.low_files[index])
        im2 = load_img(self.high_files[index])
        file1 = os.path.basename(self.low_files[index])
        file2 = os.path.basename(self.high_files[index])

        im1 = self.to_tensor(im1)
        im2 = self.to_tensor(im2)

        if self.size and self.size > 0:
            # Crop ngẫu nhiên theo size nếu có yêu cầu
            h, w = im1.shape[1], im1.shape[2]
            if h >= self.size and w >= self.size:
                i = random.randint(0, h - self.size)
                j = random.randint(0, w - self.size)
                im1 = im1[:, i:i+self.size, j:j+self.size]
                im2 = im2[:, i:i+self.size, j:j+self.size]
        else:
            # Pad về bội số của 8
            factor = 8
            h, w = im1.shape[1], im1.shape[2]
            H, W = ((h + factor - 1) // factor) * factor, ((w + factor - 1) // factor) * factor
            padh = H - h
            padw = W - w
            if padh > 0 or padw > 0:
                im1 = F.pad(im1.unsqueeze(0), (0, padw, 0, padh), 'reflect').squeeze(0)
                im2 = F.pad(im2.unsqueeze(0), (0, padw, 0, padh), 'reflect').squeeze(0)

        return im1, im2, file1, file2

    def __len__(self):
        return len(self.low_files)


def build_model(weights_path, model_arch='mamba', device='cuda'):
    """
    Kởi tạo mô hình CIDNet và load trọng số (weights) đã huấn luyện.
    """
    print(f"===> Khởi tạo mô hình architecture: CIDNet ({model_arch})")
    if model_arch == 'mamba':
        from net.CIDNet_Mamba_separable_learning import CIDNet
    else:
        from net.CIDNet_base import CIDNet

    model = CIDNet().to(device)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Không tìm thấy file weight tại path: {weights_path}")

    print(f"===> Đang load weight từ: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    cleaned_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict)
    model.eval()
    return model


def init_unweighted_losses(device):
    """
    Khởi tạo tất cả các hàm loss với hệ số trọng số = 1.0 (Không có hệ số nhân trọng số dataset / warmup multiplier).
    """
    L1_loss   = L1Loss(loss_weight=1.0, reduction='mean').to(device)
    L2_loss   = L2Loss(loss_weight=1.0, reduction='mean').to(device)
    D_loss    = SSIM(window_size=11, size_average=True, weight=1.0).to(device)
    E_loss    = EdgeLoss(loss_weight=1.0, reduction='mean').to(device)
    P_loss    = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1, 'conv3_4': 1, 'conv4_4': 1}, perceptual_weight=1.0, criterion='mse').to(device)
    LSGD_loss = RegionLSGDLoss(loss_weight=1.0, reduction='mean').to(device)
    EXP_loss  = ExposureControlLoss(patch_size=16, mean_val=0.6, loss_weight=1.0).to(device)

    return (L1_loss, L2_loss, D_loss, E_loss, P_loss, LSGD_loss, EXP_loss)


def get_dataset_loader(opt):
    """
    Lấy DataLoader phù hợp dựa trên dataset và split chọn (train hoặc val).
    """
    if opt.low_dir and opt.high_dir:
        print(f"===> Sử dụng thư mục ảnh tùy chỉnh: Low={opt.low_dir}, High={opt.high_dir}")
        dataset_set = PairedDatasetFromFolder(opt.low_dir, opt.high_dir, size=opt.cropSize if opt.use_crop else None)
    else:
        print(f"===> Loading dataset: {opt.dataset} (Split: {opt.split})")
        if opt.split == 'train':
            # Sử dụng tập train với cặp ảnh (Low, High)
            if opt.dataset == 'lol_v1':
                dataset_set = get_lol_training_set(opt.data_train_lol_v1, size=opt.cropSize if opt.use_crop else 0)
            elif opt.dataset == 'lol_blur':
                dataset_set = get_training_set_blur(opt.data_train_lol_blur, size=opt.cropSize if opt.use_crop else 0)
            elif opt.dataset == 'lolv2_real':
                dataset_set = get_lol_v2_training_set(opt.data_train_lolv2_real, size=opt.cropSize if opt.use_crop else 0)
            elif opt.dataset == 'lolv2_syn':
                dataset_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn, size=opt.cropSize if opt.use_crop else 0)
            elif opt.dataset == 'SID':
                dataset_set = get_SID_training_set(opt.data_train_SID, size=opt.cropSize if opt.use_crop else 0)
            elif opt.dataset in ['SICE_mix', 'SICE_grad']:
                dataset_set = get_SICE_training_set(opt.data_train_SICE, size=opt.cropSize if opt.use_crop else 0)
            elif opt.dataset == 'fivek':
                dataset_set = get_fivek_training_set(opt.data_train_fivek, size=opt.cropSize if opt.use_crop else 0)
            else:
                raise ValueError(f"Dataset {opt.dataset} không hợp lệ.")
        else:
            # Tập Validation: Ghép cặp low và high GT từ đường dẫn trong opt
            mapping = {
                'lol_v1': (opt.data_val_lol_v1, opt.data_valgt_lol_v1),
                'lolv2_real': (opt.data_val_lolv2_real, opt.data_valgt_lolv2_real),
                'lolv2_syn': (opt.data_val_lolv2_syn, opt.data_valgt_lolv2_syn),
                'lol_blur': (opt.data_val_lol_blur, opt.data_valgt_lol_blur),
                'SID': (opt.data_val_SID, opt.data_valgt_SID),
                'SICE_mix': (opt.data_val_SICE_mix, opt.data_valgt_SICE_mix),
                'SICE_grad': (opt.data_val_SICE_grad, opt.data_valgt_SICE_grad),
                'fivek': (opt.data_test_fivek, opt.data_valgt_fivek)
            }
            low_path, high_path = mapping[opt.dataset]
            dataset_set = PairedDatasetFromFolder(low_path, high_path, size=opt.cropSize if opt.use_crop else None)

    loader = DataLoader(dataset=dataset_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=False)
    return loader


def evaluate_unweighted_losses(model, data_loader, loss_funcs, opt, device):
    """
    Tính toán và tổng hợp loss không có hệ số cho từng mẫu ảnh trong tập dữ liệu.
    """
    L1_loss, L2_loss, D_loss, E_loss, P_loss, LSGD_loss, EXP_loss = loss_funcs

    # Khởi tạo bộ tích lũy giá trị loss nguyên bản (Thô / Unweighted)
    metrics_sum = {
        'l1_rgb': 0.0, 'l2_rgb': 0.0, 'd_rgb': 0.0, 'p_rgb': 0.0, 'e_rgb': 0.0, 'lsgd_rgb': 0.0,
        'l1_hvi': 0.0, 'l2_hvi': 0.0, 'd_hvi': 0.0, 'p_hvi': 0.0, 'e_hvi': 0.0, 'lsgd_hvi': 0.0, 'exp_hvi': 0.0
    }

    num_samples = 0

    print("===> Đang chạy tính toán Loss nguyên bản (không nhân hệ số)...")
    with torch.no_grad():
        for batch in tqdm(data_loader):
            im1, im2 = batch[0].to(device), batch[1].to(device)

            if opt.gamma:
                gamma_val = opt.start_gamma / 100.0 if hasattr(opt, 'start_gamma') else 1.0
                input_low = im1 ** gamma_val
                input_gt = im2 ** gamma_val
            else:
                input_low = im1
                input_gt = im2

            # Model Forward
            output_rgb = model(input_low)
            gt_rgb = input_gt

            output_hvi = model.HVIT(output_rgb)
            gt_hvi = model.HVIT(gt_rgb)

            # --- Tính Loss cho miền RGB (Không nhân hệ số L1_weight, D_weight, warm_up, v.v...) ---
            l1_rgb = L1_loss(output_rgb, gt_rgb).item()
            l2_rgb = L2_loss(output_rgb, gt_rgb).item()
            d_rgb = D_loss(output_rgb, gt_rgb).item()
            p_rgb = P_loss(output_rgb, gt_rgb)[0].item()
            e_rgb = E_loss(output_rgb, gt_rgb).item()
            lsgd_rgb = LSGD_loss(output_rgb, gt_rgb).item()

            # --- Tính Loss cho miền HVI (Không nhân hệ số HVI_weight) ---
            l1_hvi = L1_loss(output_hvi, gt_hvi).item()
            l2_hvi = L2_loss(output_hvi, gt_hvi).item()
            d_hvi = D_loss(output_hvi, gt_hvi).item()
            p_hvi = P_loss(output_hvi, gt_hvi)[0].item()
            e_hvi = E_loss(output_hvi, gt_hvi).item()
            lsgd_hvi = LSGD_loss(output_hvi, gt_hvi, is_hvi=True).item()
            exp_hvi = EXP_loss(output_hvi[:, 2:3, :, :]).item()

            # Tích lũy
            metrics_sum['l1_rgb'] += l1_rgb
            metrics_sum['l2_rgb'] += l2_rgb
            metrics_sum['d_rgb'] += d_rgb
            metrics_sum['p_rgb'] += p_rgb
            metrics_sum['e_rgb'] += e_rgb
            metrics_sum['lsgd_rgb'] += lsgd_rgb

            metrics_sum['l1_hvi'] += l1_hvi
            metrics_sum['l2_hvi'] += l2_hvi
            metrics_sum['d_hvi'] += d_hvi
            metrics_sum['p_hvi'] += p_hvi
            metrics_sum['e_hvi'] += e_hvi
            metrics_sum['lsgd_hvi'] += lsgd_hvi
            metrics_sum['exp_hvi'] += exp_hvi

            num_samples += 1

    # Tính trung bình trên toàn bộ dataset
    avg_metrics = {k: v / num_samples for k, v in metrics_sum.items()}
    return avg_metrics, num_samples


def main():
    parser = option()
    parser.add_argument('--weights_path', type=str, default='./weights/train/epoch_100.pth',
                        help='Đường dẫn tới file weight .pth đã huấn luyện')
    parser.add_argument('--model_arch', type=str, default='base', choices=['mamba', 'base'],
                        help='Kiến trúc mô hình CIDNet: mamba (CIDNet_Mamba_separable_learning) hoặc base (CIDNet)')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                        help='Tập dữ liệu để đánh giá loss: train hoặc val')
    parser.add_argument('--use_crop', action='store_true', default=False,
                        help='Nếu bật, ảnh sẽ được crop theo cropSize thay vì để nguyên độ phân giải padded')
    parser.add_argument('--low_dir', type=str, default=None,
                        help='Đường dẫn tùy chỉnh cho thư mục ảnh Low (nếu muốn đánh giá folder riêng)')
    parser.add_argument('--high_dir', type=str, default=None,
                        help='Đường dẫn tùy chỉnh cho thư mục ảnh High GT (nếu muốn đánh giá folder riêng)')

    opt = parser.parse_args()

    device = 'cuda' if opt.gpu_mode and torch.cuda.is_available() else 'cpu'

    # Build model & Load weights
    model = build_model(opt.weights_path, opt.model_arch, device)

    # Init loss functions với hệ số = 1.0
    loss_funcs = init_unweighted_losses(device)

    # Load dataset
    data_loader = get_dataset_loader(opt)

    # Run calculation
    avg_loss, num_samples = evaluate_unweighted_losses(model, data_loader, loss_funcs, opt, device)

    # Hiển thị kết quả ra màn hình dạng bảng Markdown
    print("\n" + "="*85)
    print(" BÁO CÁO GIÁ TRỊ LOSS NGUYÊN BẢN (UNWEIGHTED LOSSES - KHÔNG CÓ HỆ SỐ NHÂN TRỌNG SỐ)")
    print("="*85)
    print(f" Weight File : {opt.weights_path}")
    print(f" Model Arch  : CIDNet ({opt.model_arch})")
    print(f" Dataset     : {opt.dataset} | Split: {opt.split} | Total Batches/Images: {num_samples}")
    print("="*85)

    headers = f"{'Thành phần Loss (Raw)':<25} | {'Loss RGB':<18} | {'Loss HVI':<18} | {'Tổng Loss Raw':<18}"
    print(headers)
    print("-" * 85)

    rows = [
        ("1. L1 Loss (MAE)", avg_loss['l1_rgb'], avg_loss['l1_hvi']),
        ("2. L2 Loss (MSE)", avg_loss['l2_rgb'], avg_loss['l2_hvi']),
        ("3. SSIM Loss (1 - SSIM)", avg_loss['d_rgb'], avg_loss['d_hvi']),
        ("4. Perceptual (VGG) Loss", avg_loss['p_rgb'], avg_loss['p_hvi']),
        ("5. Edge Loss (Laplacian)", avg_loss['e_rgb'], avg_loss['e_hvi']),
        ("6. Region LSGD Loss", avg_loss['lsgd_rgb'], avg_loss['lsgd_hvi']),
        ("7. Exposure Loss (EXP)", None, avg_loss['exp_hvi']),
    ]

    total_rgb_raw = 0.0
    total_hvi_raw = 0.0

    for name, val_rgb, val_hvi in rows:
        str_rgb = f"{val_rgb:.6f}" if val_rgb is not None else "N/A"
        str_hvi = f"{val_hvi:.6f}" if val_hvi is not None else "N/A"
        
        sum_val = (val_rgb if val_rgb else 0.0) + (val_hvi if val_hvi else 0.0)
        str_sum = f"{sum_val:.6f}"

        if val_rgb: total_rgb_raw += val_rgb
        if val_hvi: total_hvi_raw += val_hvi

        print(f"{name:<25} | {str_rgb:<18} | {str_hvi:<18} | {str_sum:<18}")

    print("-" * 85)
    total_raw_combined = total_rgb_raw + total_hvi_raw
    print(f"{'TỔNG RAW LOSS (SUM)':<25} | {total_rgb_raw:<18.6f} | {total_hvi_raw:<18.6f} | {total_raw_combined:<18.6f}")
    print("="*85 + "\n")

    # Lưu kết quả ra file markdown trong results/loss_eval/
    os.makedirs("./results/loss_eval", exist_ok=True)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"./results/loss_eval/loss_eval_{opt.dataset}_{opt.split}_{now_str}.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"# Unweighted Loss Evaluation Report\n")
        f.write(f"- **Weight Path**: `{opt.weights_path}`\n")
        f.write(f"- **Model**: CIDNet ({opt.model_arch})\n")
        f.write(f"- **Dataset**: {opt.dataset} ({opt.split} split)\n")
        f.write(f"- **Total Samples**: {num_samples}\n\n")
        f.write("| Thành phần Loss (Raw) | Loss RGB | Loss HVI | Tổng Loss Raw |\n")
        f.write("|-----------------------|----------|----------|---------------|\n")
        for name, val_rgb, val_hvi in rows:
            str_rgb = f"{val_rgb:.6f}" if val_rgb is not None else "N/A"
            str_hvi = f"{val_hvi:.6f}" if val_hvi is not None else "N/A"
            sum_val = (val_rgb if val_rgb else 0.0) + (val_hvi if val_hvi else 0.0)
            f.write(f"| {name} | {str_rgb} | {str_hvi} | {sum_val:.6f} |\n")
        f.write(f"| **TỔNG RAW LOSS** | **{total_rgb_raw:.6f}** | **{total_hvi_raw:.6f}** | **{total_raw_combined:.6f}** |\n")

    print(f"===> Báo cáo chi tiết đã được lưu vào: {out_file}")

if __name__ == '__main__':
    main()
