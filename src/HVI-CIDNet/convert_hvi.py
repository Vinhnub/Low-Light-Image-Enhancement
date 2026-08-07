import torch
import cv2
import numpy as np
import sys
import os
from PIL import Image
import torchvision.transforms.functional as TF

# Add HVI-CIDNet to path so we can import RGB_HVI
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'HVI-CIDNet'))
from net.HVI_transform import RGB_HVI

def convert_to_hvi_and_save(image_path, output_hv_path, output_i_path):
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Convert to tensor [1, 3, H, W] in range [0, 1]
    img_tensor = TF.to_tensor(img).unsqueeze(0)
    
    # Initialize HVI converter
    converter = RGB_HVI()
    
    # Perform conversion
    with torch.no_grad():
        hvi_tensor = converter.HVIT(img_tensor)
        
    # Extract H, V, I
    H = hvi_tensor[:, 0:1, :, :]
    V = hvi_tensor[:, 1:2, :, :]
    I = hvi_tensor[:, 2:3, :, :]
    
    # H and V are in range [-1, 1], normalize to [0, 1] for visualization
    H_vis = (H + 1.0) / 2.0
    V_vis = (V + 1.0) / 2.0
    
    # Create an RGB image for HV visualization (R=H, G=V, B=0)
    zero_channel = torch.zeros_like(I)
    HV_rgb = torch.cat([H_vis, V_vis, zero_channel], dim=1)
    
    # I is in range [0, 1], save as grayscale (repeat to 3 channels for standard image)
    I_rgb = torch.cat([I, I, I], dim=1)
    
    # Convert back to PIL Image
    HV_img = TF.to_pil_image(HV_rgb.squeeze(0))
    I_img = TF.to_pil_image(I_rgb.squeeze(0))
    
    # Save the images
    HV_img.save(output_hv_path)
    I_img.save(output_i_path)
    
    print(f"Successfully saved HV image to: {output_hv_path}")
    print(f"Successfully saved I image to: {output_i_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert RGB image to HVI color space and output HV and I images.')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_hv', type=str, default='output_hvi_low/output_HV.png', help='Path to output HV image')
    parser.add_argument('--output_i', type=str, default='output_hvi_low/output_I.png', help='Path to output I image')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input image {args.input} does not exist.")
    else:
        convert_to_hvi_and_save(args.input, args.output_hv, args.output_i)
