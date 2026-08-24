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

def convert_to_hvi_and_save(image_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Create grayscale RGB images for H, V, I visualization
    H_rgb = torch.cat([H_vis, H_vis, H_vis], dim=1)
    V_rgb = torch.cat([V_vis, V_vis, V_vis], dim=1)
    I_rgb = torch.cat([I, I, I], dim=1)
    
    # Convert back to PIL Image
    H_img = TF.to_pil_image(H_rgb.squeeze(0))
    V_img = TF.to_pil_image(V_rgb.squeeze(0))
    I_img = TF.to_pil_image(I_rgb.squeeze(0))
    
    # Save the images
    h_path = os.path.join(output_dir, 'H.png')
    v_path = os.path.join(output_dir, 'V.png')
    i_path = os.path.join(output_dir, 'I.png')
    
    H_img.save(h_path)
    V_img.save(v_path)
    I_img.save(i_path)
    
    print(f"Successfully saved H channel to: {h_path}")
    print(f"Successfully saved V channel to: {v_path}")
    print(f"Successfully saved I channel to: {i_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert RGB image to HVI color space and output H, V, and I images in a folder.')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='output_hvi_channels_low', help='Path to output folder')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input image {args.input} does not exist.")
    else:
        convert_to_hvi_and_save(args.input, args.output_dir)
