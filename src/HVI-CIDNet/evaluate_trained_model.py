import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from net.CIDNet import CIDNet


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


def load_checkpoint(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    cleaned_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict)
    return model


def pad_to_multiple(tensor, factor=8):
    _, _, h, w = tensor.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h == 0 and pad_w == 0:
        return tensor, h, w
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect"), h, w

def tiled_inference(model, image_tensor, crop_h, crop_w,overlap=32):
    """
    image_tensor: [1, C, H, W]
    return: [1, C, H, W]
    """

    _, c, h, w = image_tensor.shape

    stride_h = crop_h - overlap
    stride_w = crop_w - overlap

    output = torch.zeros_like(image_tensor)
    count_map = torch.zeros_like(image_tensor)

    for y in range(0, h, stride_h):
        for x in range(0, w, stride_w):

            y1 = min(y + crop_h, h)
            x1 = min(x + crop_w, w)

            y0 = max(y1 - crop_h, 0)
            x0 = max(x1 - crop_w, 0)

            patch = image_tensor[:, :, y0:y1, x0:x1]

            patch, ph, pw = pad_to_multiple(patch)

            pred = model(patch)

            pred = pred[:, :, :ph, :pw]

            output[:, :, y0:y1, x0:x1] += pred
            count_map[:, :, y0:y1, x0:x1] += 1

    output = output / count_map.clamp(min=1)

    return output
def enhance_images(
    model,
    image_paths,
    output_dir,
    device,
    gamma=1.0,
    alpha=None,
    use_lol_gate=False,
    use_v2_gate=False,
    crop_h=None,
    crop_w=None
    ):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    if use_lol_gate:
        model.trans.gated = True

    if use_v2_gate:
        model.trans.gated2 = True
        if alpha is not None:
            model.trans.alpha = alpha

    model.eval()

    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="Testing"):

            image = Image.open(image_path).convert("RGB")

            input_tensor = (
                to_tensor(image)
                .unsqueeze(0)
                .to(device)
            )

            original_h, original_w = (
                input_tensor.shape[2],
                input_tensor.shape[3]
            )

            input_tensor = input_tensor ** gamma

            # ---------- inference ----------
            if crop_h is not None and crop_w is not None:

                output = tiled_inference(
                    model=model,
                    image_tensor=input_tensor,
                    crop_h=crop_h,
                    crop_w=crop_w
                )

            else:
                padded_input, _, _ = pad_to_multiple(
                    input_tensor
                )

                output = model(padded_input)

            output = torch.clamp(output, 0, 1)

            output = output[
                :, :, :original_h, :original_w
            ]

            output_image = to_pil(
                output.squeeze(0).cpu()
            )

            output_image.save(
                output_dir / image_path.name
            )

    if use_lol_gate:
        model.trans.gated = False

    if use_v2_gate:
        model.trans.gated2 = False



def run_metrics(output_dir, gt_dir, use_gt_mean):
    from measure import metrics

    output_dir = Path(output_dir)
    gt_dir = Path(gt_dir)
    pattern = str(output_dir / "*")
    label_dir = str(gt_dir) + os.sep
    avg_psnr, avg_ssim, avg_lpips = metrics(pattern, label_dir, use_gt_mean)
    print("===> Avg.PSNR: {:.4f} dB".format(avg_psnr))
    print("===> Avg.SSIM: {:.4f}".format(avg_ssim))
    print("===> Avg.LPIPS: {:.4f}".format(avg_lpips))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a trained CIDNet checkpoint and test it on a custom image folder."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="E:/PythonFile/Project/Low-Light-Image-Enhancement/src/HVI-CIDNet/weights/train/LOLv1/CIDNet_base/epoch_460.pth",
        help="Path to trained .pth checkpoint.",
    )
    parser.add_argument(
        "--crop_h",
        type=int,
        default=1080,
        help="Crop height for testing."
    )

    parser.add_argument(
        "--crop_w",
        type=int,
        default=1920,
        help="Crop width for testing."
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default='E:/PythonFile/Project/Low-Light-Image-Enhancement/mydata/dataset/dataset/Testdata',
        help="Folder containing low-light test images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="E:/PythonFile/Project/Low-Light-Image-Enhancement/mydata/dataset/dataset/Testdata/Result",
        help="Folder used to save enhanced images.",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="Optional folder containing ground-truth images with the same file names.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference, for example: cuda or cpu.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma applied to input before model inference.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Optional alpha value for v2/unpaired gate.",
    )
    parser.add_argument(
        "--lol_gate",
        action="store_true",
        help="Enable LOLv1 gate: model.trans.gated = True.",
    )
    parser.add_argument(
        "--v2_gate",
        action="store_true",
        help="Enable LOLv2/unpaired gate: model.trans.gated2 = True.",
    )
    parser.add_argument(
        "--use_GT_mean",
        action="store_true",
        default=False,
        help="Use GT mean correction when calculating metrics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    test_dir = Path(args.test_dir)
    low_dir = test_dir / "Low"
    high_dir = test_dir / "High"

    # Check folder structure
    if not low_dir.exists():
        raise FileNotFoundError(
            f"Low folder not found: {low_dir}"
        )

    image_paths = list_images(low_dir)

    model = CIDNet().to(device)
    load_checkpoint(model, args.model_path, device)

    print(f"Loaded model: {args.model_path}")
    print(f"Found {len(image_paths)} test images in: {low_dir}")

    enhance_images(
        model=model,
        image_paths=image_paths,
        output_dir=args.output_dir,
        device=device,
        gamma=args.gamma,
        alpha=args.alpha,
        use_lol_gate=args.lol_gate,
        use_v2_gate=args.v2_gate,
        crop_h=args.crop_h,
        crop_w=args.crop_w,
    )

    print(f"Enhanced images saved to: {args.output_dir}")

    # Automatically use High as GT if exists
    if high_dir.exists():
        print(f"Using GT folder: {high_dir}")
        run_metrics(
            args.output_dir,
            str(high_dir),
            args.use_GT_mean
        )
    elif args.gt_dir:
        run_metrics(
            args.output_dir,
            args.gt_dir,
            args.use_GT_mean
        )
    else:
        print("No GT folder found. Skip metrics.")


if __name__ == "__main__":
    main()
