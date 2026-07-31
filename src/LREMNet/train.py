import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import time
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchvision.utils import save_image
from tqdm import tqdm

import datasets
from models.decom import CTDN


# ===================== Parameters Parsing =====================
def dict2namespace(config_dict):
    """Recursively converts dict to argparse.Namespace (compatible with existing code style)"""
    namespace = argparse.Namespace()
    for key, value in config_dict.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict2namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Stage1: CTDN Retinex Decomposition Training')
    parser.add_argument("--config", default='stage1.yml', type=str,
                        help="Path to config file (under configs/ dir)")
    parser.add_argument('--resume', default='', type=str,
                        help='Path to checkpoint to resume training from')
    parser.add_argument("--image_folder", default='results/stage1/', type=str,
                        help="Validation image save path")
    parser.add_argument('--seed', default=230, type=int,
                        help='Random seed')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    config = dict2namespace(config_dict)
    return args, config


# ===================== Illumination Smooth Loss (TV Loss) =====================
def illumination_smooth_loss(illumination, image):
    """
    Illumination map smooth loss: weighted by original image gradient, allowing illumination changes at edges
    
    Formula: Σ |∇L| * exp(-λ * |∇I|)
    Meaning: Where original image gradient is large (edges), allow illumination to have gradient
             Where original image gradient is small (smooth areas), penalize illumination gradient
    """
    batch, channel, height, width = illumination.shape

    # Gradients of illumination map
    grad_l_x = torch.abs(illumination[:, :, :, 1:] - illumination[:, :, :, :-1])
    grad_l_y = torch.abs(illumination[:, :, 1:, :] - illumination[:, :, :-1, :])

    # Gradients of original image (using mean map as guidance)
    if image.shape[1] == 3:
        gray_image = 0.299 * image[:, 0:1, :, :] + 0.587 * image[:, 1:2, :, :] + 0.114 * image[:, 2:3, :, :]
    else:
        gray_image = image

    grad_i_x = torch.abs(gray_image[:, :, :, 1:] - gray_image[:, :, :, :-1])
    grad_i_y = torch.abs(gray_image[:, :, 1:, :] - gray_image[:, :, :-1, :])

    # Weighting: when original image gradient is large, exp(-10*|∇I|) → 0, allowing illumination changes
    weight_x = torch.exp(-10.0 * grad_i_x)
    weight_y = torch.exp(-10.0 * grad_i_y)

    smooth_loss_x = (grad_l_x * weight_x).mean()
    smooth_loss_y = (grad_l_y * weight_y).mean()

    return smooth_loss_x + smooth_loss_y


# ===================== Main Training Function =====================
def main():
    args, config = parse_args_and_config()

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    print(f"=> Using device: {device}")

    # --- Random Seed ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # --- Dataset ---
    print(f"=> Loading dataset: '{config.data.train_dataset}'")
    DATASET = datasets.__dict__[config.data.type](config)
    train_loader, val_loader = DATASET.get_loaders()
    print(f"   Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- Create Model ---
    print("=> Creating CTDN Model (channels={})...".format(config.model.channels))
    model = CTDN(channels=config.model.channels)
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optim.lr, betas=(0.9, 0.999))

    # --- Learning Rate Scheduler ---
    lr_scheduler_type = config.optim.lr_scheduler if hasattr(config.optim, 'lr_scheduler') else 'cosine'
    if lr_scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=config.training.n_epochs,
                                      eta_min=1e-7)
    elif lr_scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    else:
        scheduler = None

    # --- Loss Functions ---
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()

    # --- Loss Weights ---
    loss_cfg = config.loss
    w_recon = loss_cfg.weight_recon
    w_ref = loss_cfg.weight_reflectance
    w_smooth = loss_cfg.weight_smooth
    w_illum = loss_cfg.weight_illumination

    # --- Resume Training ---
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location='cuda')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('step', 0)
        if scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"=> Resuming from checkpoint: epoch={start_epoch}, step={global_step}")

    # --- Create Save Directory ---
    os.makedirs(config.data.ckpt_dir, exist_ok=True)

    # ===================== Training Loop =====================
    print("\n" + "=" * 60)
    print("Starting Stage 1 Training — CTDN Retinex Decomposition")
    print("=" * 60)
    print(f"  Reconstruction Weight:        {w_recon}")
    print(f"  Reflectance Consist. Weight:  {w_ref}")
    print(f"  Illum Smoothness Weight:      {w_smooth}")
    print(f"  Illum Regularization Weight:  {w_illum}")
    print(f"  Learning Rate:                {config.optim.lr}")
    print(f"  Total Epochs:                 {config.training.n_epochs}")
    print("=" * 60 + "\n")

    for epoch in range(start_epoch, config.training.n_epochs):
        model.train()
        epoch_loss_total = 0.0
        epoch_loss_recon = 0.0
        epoch_loss_ref = 0.0
        epoch_loss_smooth = 0.0
        epoch_loss_illum = 0.0
        data_start = time.time()
        data_time = 0.0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch}/{config.training.n_epochs}]")
        for i, (x, img_ids) in pbar:
            # x: [B, 6, H, W] = [low_img(3ch), high_img(3ch)] concatenated
            # If x dimension is 5 (from some special cases), flatten
            x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
            x = x.to(device)

            # Separate low-light image and normal-light image
            low_img = x[:, :3, :, :]    # [B, 3, H, W]
            high_img = x[:, 3:, :, :]   # [B, 3, H, W]

            # --- Forward Pass: CTDN Decomposition ---
            output = model(x, pred_fea=None)  # pred_fea=None → decomposition mode

            low_R = output["low_R"]       # Low-light reflectance map [B, 3, H/8, W/8]
            low_L = output["low_L"]       # Low-light illumination map [B, 3, H/8, W/8]
            low_fea = output["low_fea"]   # Low-light feature map [B, 3, H/8, W/8] (CTDN's actual reconstruction target)
            high_R = output["high_R"]     # Normal-light reflectance map
            high_L = output["high_L"]     # Normal-light illumination map
            high_fea = output["high_fea"] # Normal-light feature map

            # ============================================
            #  1. Feature Space Reconstruction: R * L ≈ low_fea (Not original image!)
            #     CTDN performs Retinex decomposition in feature space,
            #     low_fea is the 3-channel compressed feature output by channel_down
            # ============================================
            recon_low = l1_loss(low_R * low_L, low_fea)
            recon_high = l1_loss(high_R * high_L, high_fea)
            loss_recon = (recon_low + recon_high) * w_recon

            # ============================================
            #  2. Reflectance Consistency: low_R ≈ high_R
            #     Reflectance is an inherent property of objects, independent of illumination
            # ============================================
            loss_reflectance = l1_loss(low_R, high_R) * w_ref

            # ============================================
            #  3. Illumination Smoothness: Guided by feature map gradient
            # ============================================
            loss_smooth_low = illumination_smooth_loss(low_L, low_fea)
            loss_smooth_high = illumination_smooth_loss(high_L, high_fea)
            loss_smooth = (loss_smooth_low + loss_smooth_high) * w_smooth

            # ============================================
            #  4. Illumination Regularization: L mean ≈ Feature map brightness
            # ============================================
            low_gray = 0.299 * low_fea[:, 0:1] + 0.587 * low_fea[:, 1:2] + 0.114 * low_fea[:, 2:3]
            high_gray = 0.299 * high_fea[:, 0:1] + 0.587 * high_fea[:, 1:2] + 0.114 * high_fea[:, 2:3]

            loss_illum = l2_loss(low_L[:, 0:1].mean(dim=[1, 2, 3]),
                                 low_gray.mean(dim=[1, 2, 3])) * w_illum

            # ============================================
            #  Total Loss
            # ============================================
            loss_total = loss_recon + loss_reflectance + loss_smooth + loss_illum

            # --- Backward Pass ---
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping to prevent NaN
            optimizer.step()

            global_step += 1
            data_time += time.time() - data_start

            # Accumulate epoch statistics
            epoch_loss_total += loss_total.item()
            epoch_loss_recon += loss_recon.item() if isinstance(loss_recon, torch.Tensor) else loss_recon
            epoch_loss_ref += loss_reflectance.item() if isinstance(loss_reflectance, torch.Tensor) else loss_reflectance
            epoch_loss_smooth += loss_smooth.item() if isinstance(loss_smooth, torch.Tensor) else loss_smooth
            epoch_loss_illum += loss_illum.item() if isinstance(loss_illum, torch.Tensor) else loss_illum

            # --- Update Progress Bar ---
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f"{loss_total.item():.4f}",
                'Recon': f"{float(loss_recon):.4f}",
                'LR': f"{current_lr:.2e}"
            })

            data_start = time.time()

            # --- Validation & Save ---
            if hasattr(config.training, 'validation_freq') and global_step % config.training.validation_freq == 0 and global_step != 0:
                print(f"\n=> Validating at step {global_step}...")
                model.eval()
                with torch.no_grad():
                    validate_stage1(model, val_loader, device, args, global_step)

                # Save checkpoint
                save_dict = {
                    'epoch': epoch,
                    'step': global_step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                if scheduler is not None:
                    save_dict['scheduler'] = scheduler.state_dict()

                torch.save(save_dict, os.path.join(config.data.ckpt_dir, 'model_latest.pth'))

                # Also save a copy for Stage 2 fixed loading
                torch.save(save_dict, os.path.join(config.data.ckpt_dir, 'stage1_CTDN.pth'))
                print(f"=> Checkpoint saved to {config.data.ckpt_dir}/\n")

                model.train()

        # --- End of Epoch: Update learning rate ---
        if scheduler is not None:
            scheduler.step()

        # --- Epoch Statistics ---
        num_batches = i + 1
        print(f"\n===== Epoch [{epoch}] Completed =====\n"
              f"  Avg Total:  {epoch_loss_total / num_batches:.5f}\n"
              f"  Avg Recon:  {epoch_loss_recon / num_batches:.5f}\n"
              f"  Avg Ref:    {epoch_loss_ref / num_batches:.5f}\n"
              f"  Avg Smooth: {epoch_loss_smooth / num_batches:.5f}\n"
              f"  Avg Illum:  {epoch_loss_illum / num_batches:.5f}\n"
              f"  LR:         {optimizer.param_groups[0]['lr']:.7f}\n")

        # --- Save once per Epoch ---
        torch.save(
            {'epoch': epoch, 'step': global_step,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict()},
            os.path.join(config.data.ckpt_dir, f'epoch_{epoch}.pth')
        )

    print("\n" + "=" * 60)
    print("Stage 1 Training Completed!")
    print(f"Final model saved at: {config.data.ckpt_dir}/stage1_CTDN.pth")
    print("It can be used directly for Stage 2 training.")
    print("=" * 60)


# ===================== Validation Function =====================
def validate_stage1(model, val_loader, device, args, step):
    """
    Validation: Perform decomposition on validation set images and save results
    Saved items: Original image, Reflectance map R, Illumination map L, Reconstructed map R*L
    """
    save_dir = os.path.join(args.image_folder, f"step_{step}")
    os.makedirs(save_dir, exist_ok=True)

    for i, (x, img_ids) in enumerate(val_loader):
        x = x.to(device)
        low_img = x[:, :3, :, :]
        _, _, h, w = low_img.shape

        output = model(x, pred_fea=None)

        low_R = output["low_R"]        # [B, 3, H/8, W/8]
        low_L = output["low_L"]        # [B, 3, H/8, W/8]
        low_fea = output["low_fea"]    # [B, 3, H/8, W/8] Feature map (CTDN reconstruction target)
        recon_fea = low_R * low_L      # Reconstructed feature map

        # Upsampling to original size just for visualization
        low_R_up = F.interpolate(low_R, size=(h, w), mode='bilinear', align_corners=False)
        low_L_up = F.interpolate(low_L, size=(h, w), mode='bilinear', align_corners=False)
        low_fea_up = F.interpolate(low_fea, size=(h, w), mode='bilinear', align_corners=False)
        recon_fea_up = F.interpolate(recon_fea, size=(h, w), mode='bilinear', align_corners=False)

        # Only save first few samples
        if i >= 2:
            break

        for j in range(min(x.shape[0], 2)):
            img_id = img_ids[j] if isinstance(img_ids, list) else f"{i}_{j}"
            save_image(low_img[j:j+1], os.path.join(save_dir, f"{img_id}_low_input.png"))
            save_image(low_fea_up[j:j+1], os.path.join(save_dir, f"{img_id}_fea.png"))
            save_image(low_R_up[j:j+1], os.path.join(save_dir, f"{img_id}_R.png"))
            save_image(low_L_up[j:j+1].mean(dim=1, keepdim=True).repeat(1,3,1,1),
                       os.path.join(save_dir, f"{img_id}_L.png"))
            save_image(recon_fea_up[j:j+1], os.path.join(save_dir, f"{img_id}_recon_fea.png"))

    print(f"  Validation images saved to {save_dir}")


if __name__ == "__main__":
    main()
