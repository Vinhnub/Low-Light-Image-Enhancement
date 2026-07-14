import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys

# Ensure the root directory of HVI-CIDNet is in path to import its modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from net.CIDNet_Mamba import CIDNet

def visualize_feature_map(feature_map, save_path, title):
    """
    Saves a visualization of the feature map.
    If the feature map is a tuple/list (like output of MMMamba), it splits and saves each.
    """
    if isinstance(feature_map, list) or isinstance(feature_map, tuple):
        for idx, fm in enumerate(feature_map):
            visualize_feature_map(fm, save_path.replace('.png', f'_{idx}.png'), f"{title} part {idx}")
        return

    # Convert to numpy and take the first batch element
    fm = feature_map.detach().cpu().numpy()[0]
    
    # Average across all channels to get a spatial activation map
    fm_mean = np.mean(fm, axis=0)
    
    # Normalize the activation map to [0, 1] for visualization
    fm_mean = (fm_mean - np.min(fm_mean)) / (np.max(fm_mean) - np.min(fm_mean) + 1e-8)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(fm_mean, cmap='viridis')
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Test HVI-CIDNet and visualize feature maps")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--weight_path', type=str, default='weights/best_model.pth', help='Path to the model weights (optional)')
    parser.add_argument('--output_dir', type=str, default='results/feature_maps', help='Directory to save visualizations')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CIDNet().to(device)
    
    if os.path.exists(args.weight_path):
        state_dict = torch.load(args.weight_path, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Remove 'module.' prefix if saved with DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded weights from {args.weight_path}")
    else:
        print(f"Warning: Weights not found at {args.weight_path}. Using random weights for visualization.")

    model.eval()

    # 2. Setup Hooks to capture Feature Maps
    feature_maps = {}

    def get_hook(name):
        def hook(model, input, output):
            feature_maps[name] = output
        return hook

    # List of module names we want to capture
    modules_to_hook = [
        'IE_block0', 'HVE_block0',
        'IE_block1', 'HVE_block1',
        'HV_LCA1', 'I_LCA1', 'MMMamba_1',
        'IE_block2', 'HVE_block2',
        'HV_LCA2', 'I_LCA2', 'MMMamba_2',
        'IE_block3', 'HVE_block3',
        'HV_LCA3', 'I_LCA3', 'MMMamba_3',
        'HV_LCA4', 'I_LCA4', 'MMMamba_4',
        'HVD_block3', 'ID_block3',
        'HV_LCA5', 'I_LCA5', 'MMMamba_5',
        'HVD_block2', 'ID_block2',
        'HV_LCA6', 'I_LCA6', 'MMMamba_6',
        'HVD_block1', 'ID_block1',
        'HVD_block0', 'ID_block0'
    ]

    for name, module in model.named_modules():
        if name in modules_to_hook:
            module.register_forward_hook(get_hook(name))

    # 3. Load and preprocess the image
    try:
        img = Image.open(args.image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Resize image to a multiple of 16 to avoid size mismatch in skip connections
    w, h = img.size
    new_w = w - w % 16
    new_h = h - h % 16
    if new_w != w or new_h != h:
        print(f"Resizing image from {w}x{h} to {new_w}x{new_h} to match downsampling requirements.")
        img = img.resize((new_w, new_h), Image.BILINEAR)
    
    transform = transforms.ToTensor()
    input_tensor = transform(img).unsqueeze(0).to(device)

    print(f"Testing on image of shape {input_tensor.shape}...")

    # 4. Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    # Save the enhanced output image
    output_img = output.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    output_img = (output_img * 255).astype(np.uint8)
    Image.fromarray(output_img).save(os.path.join(args.output_dir, 'enhanced_output.png'))
    print(f"Saved enhanced image to {os.path.join(args.output_dir, 'enhanced_output.png')}")

    # 5. Visualize and save the captured feature maps
    print("Saving feature maps...")
    for name, fm in feature_maps.items():
        save_path = os.path.join(args.output_dir, f"{name}.png")
        visualize_feature_map(fm, save_path, name)
        
    print(f"Successfully saved all feature maps to {args.output_dir}")

if __name__ == '__main__':
    main()
