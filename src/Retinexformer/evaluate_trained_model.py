# RetinexFormer custom-folder evaluation.
# This follows the RetinexFormer/BasicSR model creation and checkpoint format.

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from basicsr.models import create_model
from basicsr.utils.options import parse
from Enhancement import utils


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(image_dir):
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Test directory does not exist: {image_dir}")

    image_paths = [
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_paths.sort()
    if not image_paths:
        raise FileNotFoundError(f"No image files found in: {image_dir}")
    return image_paths


def prepare_options(opt_path, device):
    opt = parse(opt_path, is_train=False)
    opt["dist"] = False
    opt["num_gpu"] = 1 if device.type == "cuda" else 0
    opt["path"]["pretrain_network_g"] = None
    return opt


def load_checkpoint(net, weights, device, param_key="params"):
    checkpoint = torch.load(weights, map_location=device)

    if isinstance(checkpoint, dict):
        if param_key and param_key in checkpoint:
            state_dict = checkpoint[param_key]
        elif "params_ema" in checkpoint:
            state_dict = checkpoint["params_ema"]
        elif "params" in checkpoint:
            state_dict = checkpoint["params"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned_state_dict[key] = value

    bare_net = net.module if hasattr(net, "module") else net
    bare_net.load_state_dict(cleaned_state_dict, strict=True)


def self_ensemble(input_tensor, model):
    def forward_transformed(x, hflip, vflip, rotate):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))

        x = model(x)

        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x

    outputs = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rotate in [False, True]:
                outputs.append(forward_transformed(input_tensor, hflip, vflip, rotate))
    return torch.stack(outputs).mean(dim=0)


def pad_to_multiple(input_tensor, factor=4):
    _, _, h, w = input_tensor.shape
    padded_h = ((h + factor) // factor) * factor
    padded_w = ((w + factor) // factor) * factor
    pad_h = padded_h - h if h % factor != 0 else 0
    pad_w = padded_w - w if w % factor != 0 else 0

    if pad_h == 0 and pad_w == 0:
        return input_tensor, h, w
    return F.pad(input_tensor, (0, pad_w, 0, pad_h), mode="reflect"), h, w


def run_model(input_tensor, model, self_ensemble_enabled):
    if self_ensemble_enabled:
        return self_ensemble(input_tensor, model)
    return model(input_tensor)


def enhance_images(model, image_paths, output_dir, device, factor, self_ensemble_enabled):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_outputs = {}

    model.eval()
    with torch.inference_mode():
        for image_path in tqdm(image_paths, desc="Testing"):
            image = np.float32(utils.load_img(str(image_path))) / 255.0
            input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
            input_tensor, original_h, original_w = pad_to_multiple(input_tensor, factor=factor)

            if original_h < 3000 and original_w < 3000:
                restored = run_model(input_tensor, model, self_ensemble_enabled)
            else:
                # Same large-image fallback as RetinexFormer's dataset test script.
                input_odd = input_tensor[:, :, :, 1::2]
                input_even = input_tensor[:, :, :, 0::2]
                restored_odd = run_model(input_odd, model, self_ensemble_enabled)
                restored_even = run_model(input_even, model, self_ensemble_enabled)
                restored = torch.zeros_like(input_tensor)
                restored[:, :, :, 1::2] = restored_odd
                restored[:, :, :, 0::2] = restored_even

            restored = restored[:, :, :original_h, :original_w]
            restored = torch.clamp(restored, 0, 1)
            restored_np = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            metric_outputs[image_path.name] = restored_np

            output_path = output_dir / f"{image_path.stem}.png"
            utils.save_img(str(output_path), np.uint8(np.clip(restored_np * 255.0, 0, 255)))

    return metric_outputs


def calculate_metrics(image_paths, restored_images, gt_dir, gt_mean=False):
    gt_dir = Path(gt_dir)
    psnr_values = []
    ssim_values = []

    for image_path in image_paths:
        target_path = gt_dir / image_path.name
        if not target_path.exists():
            same_stem_matches = [
                path for path in gt_dir.iterdir()
                if path.is_file()
                and path.stem == image_path.stem
                and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if not same_stem_matches:
                raise FileNotFoundError(f"Ground-truth image not found for: {image_path.name}")
            target_path = same_stem_matches[0]

        restored = restored_images[image_path.name]
        target = np.float32(utils.load_img(str(target_path))) / 255.0
        if restored.shape[:2] != target.shape[:2]:
            restored = cv2.resize(restored, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_CUBIC)

        if gt_mean:
            mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_RGB2GRAY).mean()
            mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_RGB2GRAY).mean()
            restored = np.clip(restored * (mean_target / (mean_restored + 1e-8)), 0, 1)

        psnr_values.append(utils.PSNR(target, restored))
        restored_uint8 = np.uint8(np.clip(restored * 255.0, 0, 255))
        target_uint8 = np.uint8(np.clip(target * 255.0, 0, 255))
        ssim_values.append(utils.calculate_ssim(target_uint8, restored_uint8))

    print("PSNR: %f" % np.mean(np.array(psnr_values)))
    print("SSIM: %f" % np.mean(np.array(ssim_values)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a trained RetinexFormer checkpoint and test it on a custom image folder."
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="Options/RetinexFormer_LOL_v1.yml",
        help="Path to RetinexFormer option YAML.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained RetinexFormer .pth checkpoint.",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Folder containing low-light test images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/custom_test",
        help="Folder used to save enhanced images.",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="Optional folder containing ground-truth images with the same names or stems.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="GPU device ids, for example: 0. Ignored when --device cpu is used.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference: cuda or cpu.",
    )
    parser.add_argument(
        "--param_key",
        type=str,
        default="params",
        help="Checkpoint key to load, usually params or params_ema.",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=4,
        help="Pad images to a multiple of this value. RetinexFormer commonly uses 4.",
    )
    parser.add_argument(
        "--GT_mean",
        action="store_true",
        help="Use GT mean correction when calculating PSNR/SSIM.",
    )
    parser.add_argument(
        "--self_ensemble",
        action="store_true",
        help="Use RetinexFormer self-ensemble inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print("export CUDA_VISIBLE_DEVICES=" + args.gpus)

    device = torch.device(args.device)
    opt = prepare_options(args.opt, device)
    model = create_model(opt).net_g
    load_checkpoint(model, args.weights, device, param_key=args.param_key)
    model = model.to(device)

    image_paths = list_images(args.test_dir)
    print(f"Loaded weights: {args.weights}")
    print(f"Found {len(image_paths)} test images in: {args.test_dir}")

    restored_images = enhance_images(
        model=model,
        image_paths=image_paths,
        output_dir=args.output_dir,
        device=device,
        factor=args.factor,
        self_ensemble_enabled=args.self_ensemble,
    )
    print(f"Enhanced images saved to: {args.output_dir}")

    if args.gt_dir:
        calculate_metrics(image_paths, restored_images, args.gt_dir, gt_mean=args.GT_mean)


if __name__ == "__main__":
    main()
