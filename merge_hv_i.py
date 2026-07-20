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

def merge_hv_and_i(img_hv_path, img_i_path, output_path):
    # Load images
    img1 = Image.open(img_hv_path).convert('RGB')
    img2 = Image.open(img_i_path).convert('RGB')
    
    # Ensure they are the same size
    if img1.size != img2.size:
        print(f"Resize image 2 {img2.size} to match image 1 {img1.size}")
        img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)
    
    # Convert to tensor [1, 3, H, W] in range [0, 1]
    img1_tensor = TF.to_tensor(img1).unsqueeze(0)
    img2_tensor = TF.to_tensor(img2).unsqueeze(0)
    
    # Initialize HVI converter
    converter = RGB_HVI()
    
    # Perform conversion
    with torch.no_grad():
        # Lấy HVI của ảnh 1
        hvi_tensor1 = converter.HVIT(img1_tensor)
        # Lấy HVI của ảnh 2
        hvi_tensor2 = converter.HVIT(img2_tensor)
        
        # Lấy H và V từ ảnh 1
        H1 = hvi_tensor1[:, 0:1, :, :]
        V1 = hvi_tensor1[:, 1:2, :, :]
        
        # Lấy I từ ảnh 2
        I2 = hvi_tensor2[:, 2:3, :, :]
        
        # Tính color_sensitive cho ảnh 1 và ảnh 2
        eps = 1e-8
        pi = 3.141592653589793
        k = converter.density_k.item()
        
        # I của ảnh 1
        I1 = hvi_tensor1[:, 2:3, :, :]
        color_sensitive1 = ((I1 * 0.5 * pi).sin() + eps).pow(k)
        
        # Khôi phục H và V gốc (saturation * cos/sin(hue)) của ảnh 1
        H1_base = H1 / (color_sensitive1 + eps)
        V1_base = V1 / (color_sensitive1 + eps)
        
        # color_sensitive của ảnh 2
        color_sensitive2 = ((I2 * 0.5 * pi).sin() + eps).pow(k)
        
        # Gán lại H và V mới với color_sensitive của ảnh 2
        # Điều này giúp giữ nguyên saturation và hue của ảnh 1 khi đưa qua hàm nghịch đảo PHVIT
        H_new = H1_base * color_sensitive2
        V_new = V1_base * color_sensitive2
        
        # Gộp lại: [H_new, V_new, I2]
        merged_hvi = torch.cat([H_new, V_new, I2], dim=1)
        
        # Chuyển ngược lại về không gian màu RGB
        merged_rgb_tensor = converter.PHVIT(merged_hvi)
    
    # Kẹp giá trị trong khoảng [0, 1] để lưu
    merged_rgb_tensor = torch.clamp(merged_rgb_tensor, 0.0, 1.0)
    
    # Convert back to PIL Image
    merged_img = TF.to_pil_image(merged_rgb_tensor.squeeze(0))
    
    # Save the output image
    merged_img.save(output_path)
    print(f"Successfully saved merged image to: {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Merge HV from image 1 and I from image 2 into a single RGB image.')
    parser.add_argument('--input_hv', type=str, required=True, help='Path to image 1 (to provide HV)')
    parser.add_argument('--input_i', type=str, required=True, help='Path to image 2 (to provide I)')
    parser.add_argument('--output', type=str, default='merged_output.png', help='Path to save output merged image')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_hv):
        print(f"Error: Input image 1 {args.input_hv} does not exist.")
    elif not os.path.exists(args.input_i):
        print(f"Error: Input image 2 {args.input_i} does not exist.")
    else:
        merge_hv_and_i(args.input_hv, args.input_i, args.output)
