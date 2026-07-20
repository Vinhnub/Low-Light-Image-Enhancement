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

def reconstruct_from_hvi(img_hv_path, img_i_path, output_path):
    # Đọc ảnh chứa thông tin HV và ảnh chứa thông tin I
    img_hv = Image.open(img_hv_path).convert('RGB')
    img_i = Image.open(img_i_path).convert('L') # Chuyển I về grayscale 1 kênh
    
    # Đảm bảo cùng kích thước
    if img_hv.size != img_i.size:
        print(f"Resize image I {img_i.size} to match image HV {img_hv.size}")
        img_i = img_i.resize(img_hv.size, Image.Resampling.LANCZOS)
    
    # Chuyển thành tensor [1, C, H, W] với khoảng [0, 1]
    hv_tensor = TF.to_tensor(img_hv).unsqueeze(0)
    i_tensor = TF.to_tensor(img_i).unsqueeze(0)
    
    # Phục hồi lại H và V từ đoạn chuẩn hoá (H_vis = (H + 1) / 2)
    H = hv_tensor[:, 0:1, :, :] * 2.0 - 1.0
    V = hv_tensor[:, 1:2, :, :] * 2.0 - 1.0
    
    # Lấy kênh I
    I = i_tensor[:, 0:1, :, :]
    
    # Gộp lại thành tensor HVI: [H, V, I]
    merged_hvi = torch.cat([H, V, I], dim=1)
    
    # Khởi tạo model và thiết lập k = 0.2 (giá trị mặc định trong model)
    converter = RGB_HVI()
    converter.this_k = converter.density_k.item()
    
    # Thực hiện biến đổi ngược (PHVIT)
    with torch.no_grad():
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
    parser = argparse.ArgumentParser(description='Merge pre-calculated HV image and I image into a single RGB image.')
    parser.add_argument('--input_hv', type=str, required=True, help='Path to HV image')
    parser.add_argument('--input_i', type=str, required=True, help='Path to I image')
    parser.add_argument('--output', type=str, default='merged_output.png', help='Path to save output merged image')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_hv):
        print(f"Error: Input HV image {args.input_hv} does not exist.")
    elif not os.path.exists(args.input_i):
        print(f"Error: Input I image {args.input_i} does not exist.")
    else:
        reconstruct_from_hvi(args.input_hv, args.input_i, args.output)
