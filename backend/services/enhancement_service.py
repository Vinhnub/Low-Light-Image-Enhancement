import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

from backend.services.model_manager import get_model

def pad_to_multiple(tensor, factor=8):
    """
    Pads a [1, C, H, W] tensor so that H and W are multiples of `factor`.
    """
    _, _, h, w = tensor.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    if pad_h == 0 and pad_w == 0:
        return tensor, h, w
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect"), h, w

def enhance(input_path: str, output_path: str, model_name: str):
    """
    Service layer that bridges the FastAPI backend with the preloaded PyTorch models.
    """
    print(f"Enhancing image {input_path} using model {model_name}")
    
    # Get the preloaded model
    model = get_model(model_name)

    if model is None:
        raise ValueError(f"Model {model_name} is not loaded or unsupported.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name.lower() == "retinexformer":
        # -------------------------------------------------------------
        # Inference for Retinexformer
        # -------------------------------------------------------------
        # Read image using cv2 to match their native `load_img` style
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image {input_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = np.float32(img) / 255.
        img = torch.from_numpy(img).permute(2, 0, 1)
        input_tensor = img.unsqueeze(0).to(device)

        # Padding in case images are not multiples of 4 (Retinexformer typically uses 4)
        factor = 4
        padded_input, h, w = pad_to_multiple(input_tensor, factor=factor)

        with torch.inference_mode():
            restored = model(padded_input)

        # Unpad images to original dimensions
        restored = restored[:, :, :h, :w]

        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        
        # Convert back to uint8
        restored_uint8 = (restored * 255.0).round().astype(np.uint8)
        
        # Save output using cv2
        restored_bgr = cv2.cvtColor(restored_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, restored_bgr)
        
    elif model_name.lower() == "cidnet":
        # -------------------------------------------------------------
        # Inference for HVI-CIDNet
        # -------------------------------------------------------------
        image = Image.open(input_path).convert("RGB")
        to_tensor = transforms.ToTensor()
        to_pil = transforms.ToPILImage()

        input_tensor = to_tensor(image).unsqueeze(0).to(device)
        original_h, original_w = input_tensor.shape[2], input_tensor.shape[3]

        padded_input, _, _ = pad_to_multiple(input_tensor, factor=8)

        with torch.no_grad():
            output = model(padded_input)

        output = torch.clamp(output, 0, 1)
        output = output[:, :, :original_h, :original_w]

        output_image = to_pil(output.squeeze(0).cpu())
        output_image.save(output_path)
    else:
        raise ValueError(f"Model {model_name} is defined but inference logic is missing.")

    return output_path
