import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys

# Đảm bảo import đúng từ thư mục gốc
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import model CIDNet_base_w_edge_tiny đã được chỉnh sửa
from net.CIDNet_base_w_edge_tiny import CIDNet

def visualize_feature_map(feature_map, save_path, title):
    """
    Hàm visualize feature map.
    Đối với các feature map có nhiều kênh, ta lấy trung bình các kênh để visualize.
    """
    if isinstance(feature_map, list) or isinstance(feature_map, tuple):
        for idx, fm in enumerate(feature_map):
            visualize_feature_map(fm, save_path.replace('.png', f'_{idx}.png'), f"{title} part {idx}")
        return

    # Chuyển tensor thành numpy (lấy batch đầu tiên)
    fm = feature_map.detach().cpu().numpy()[0]
    
    # Tính trung bình trên tất cả các kênh (channel) để có 1 bản đồ kích hoạt không gian
    fm_mean = np.mean(fm, axis=0)
    
    # Chuẩn hóa về [0, 1] để hiển thị
    fm_mean = (fm_mean - np.min(fm_mean)) / (np.max(fm_mean) - np.min(fm_mean) + 1e-8)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(fm_mean, cmap='viridis')
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_channels(tensor, save_dir, prefix):
    """
    Hiển thị từng kênh riêng biệt của một tensor đầu vào (ví dụ hvi_edge, i_edge).
    """
    os.makedirs(save_dir, exist_ok=True)
    fm = tensor.detach().cpu().numpy()[0]
    num_channels = fm.shape[0]
    
    for i in range(num_channels):
        channel_data = fm[i]
        # Chuẩn hóa kênh
        channel_data = (channel_data - np.min(channel_data)) / (np.max(channel_data) - np.min(channel_data) + 1e-8)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(channel_data, cmap='gray')
        plt.title(f"{prefix} - Channel {i}")
        plt.axis('off')
        save_path = os.path.join(save_dir, f"{prefix}_channel_{i}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize input channels and feature maps of CIDNet_base_w_edge_tiny")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--weight_path', type=str, default='weights/train/epoch_20.pth', help='Path to model weights (optional)')
    parser.add_argument('--output_dir', type=str, default='results/feature_maps_tiny', help='Directory to save visualizations')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CIDNet().to(device)
    
    if os.path.exists(args.weight_path):
        state_dict = torch.load(args.weight_path, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded weights from {args.weight_path}")
    else:
        print(f"Warning: Weights not found at {args.weight_path}. Using random weights.")

    model.eval()

    # 2. Setup Hooks để lưu Feature Maps
    feature_maps = {}

    def get_hook(name):
        def hook(module, input, output):
            feature_maps[name] = output
        return hook
    
    def get_input_hook(name):
        def hook(module, input, output):
            # Input của module thường là tuple
            feature_maps[name] = input[0]
        return hook

    # Các module cần visualize output
    modules_to_hook_output = [
        'IE_block0', 'HVE_block0',
        'IE_block1', 'HVE_block1',
        'IE_block2', 'HVE_block2',
        'IE_block3', 'HVE_block3',
        'HV_LCA1', 'I_LCA1',
        'HV_LCA2', 'I_LCA2',
        'HV_LCA3', 'I_LCA3',
        'HV_LCA4', 'I_LCA4',
        'HV_LCA5', 'I_LCA5',
        'HV_LCA6', 'I_LCA6',
        'HVD_block3', 'ID_block3',
        'HVD_block2', 'ID_block2',
        'HVD_block1', 'ID_block1',
        'HVD_block0', 'ID_block0'
    ]

    for name, module in model.named_modules():
        if name in modules_to_hook_output:
            module.register_forward_hook(get_hook(name))
            
        # Hook đặc biệt để lấy input đầu vào của IE_block0 và HVE_block0 (i_edge và hvi_edge)
        if name == 'IE_block0':
            module.register_forward_hook(get_input_hook('Input_I_Branch_i_edge'))
        if name == 'HVE_block0':
            module.register_forward_hook(get_input_hook('Input_HV_Branch_hvi_edge'))

    # 3. Đọc và preprocess ảnh
    try:
        img = Image.open(args.image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Resize ảnh để kích thước chia hết cho 16
    w, h = img.size
    new_w = w - w % 16
    new_h = h - h % 16
    if new_w != w or new_h != h:
        img = img.resize((new_w, new_h), Image.BILINEAR)
    
    transform = transforms.ToTensor()
    input_tensor = transform(img).unsqueeze(0).to(device)

    print("Forwarding image through model...")

    # 4. Forward
    with torch.no_grad():
        output = model(input_tensor)
    
    # 5. Lưu ảnh output
    output_img = output.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    output_img = (output_img * 255).astype(np.uint8)
    Image.fromarray(output_img).save(os.path.join(args.output_dir, 'enhanced_output.png'))

    # 6. Hiển thị và lưu Feature Maps
    print("Saving feature maps and input channels...")
    
    # Thư mục lưu các kênh đầu vào riêng
    input_channels_dir = os.path.join(args.output_dir, 'input_channels')
    
    for name, fm in feature_maps.items():
        if 'Input_' in name:
            # Nếu là input tensor của 2 nhánh thì lưu từng kênh ra
            visualize_channels(fm, input_channels_dir, name)
            # Cũng lưu một bản feature map trung bình
            save_path = os.path.join(args.output_dir, f"{name}_mean.png")
            visualize_feature_map(fm, save_path, f"{name} Mean")
        else:
            save_path = os.path.join(args.output_dir, f"{name}.png")
            visualize_feature_map(fm, save_path, name)
            
    print(f"Successfully saved all feature maps to {args.output_dir}")

if __name__ == '__main__':
    main()
