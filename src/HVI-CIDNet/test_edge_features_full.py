import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from net.CIDNet_base import CIDNet as CIDNet_Base
from net.CIDNet_base_w_edge import CIDNet as CIDNet_Edge

def visualize_full_pipeline():
    img_path = r'E:\PythonFile\Project\Low-Light-Image-Enhancement\mydata\dataset\dataset\LOLv2-real\Test\Input\00787.png'
    if not os.path.exists(img_path):
        return
        
    img = Image.open(img_path).convert('RGB')
    x = T.ToTensor()(img).unsqueeze(0) 
    
    model_base = CIDNet_Base(channels=[36, 36, 72, 144])
    model_edge = CIDNet_Edge(channels=[36, 36, 72, 144])
    
    weight_path_base = os.path.join(current_dir, 'weights', 'epoch_670_best_psnr.pth')
    weight_path_edge = os.path.join(current_dir, 'weights', 'epoch_700.pth')
    
    def load_weights(model, path):
        if os.path.exists(path):
            state_dict = torch.load(path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)

    load_weights(model_base, weight_path_base)
    load_weights(model_edge, weight_path_edge)
    
    model_base.eval()
    model_edge.eval()

    features_base = {}
    features_edge = {}

    def get_hook(name, feature_dict):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            feature_dict[name] = output.detach()
        return hook
        
    target_layers = [
        'HVE_block0', 'HVE_block1', 'HVE_block2', 'HVE_block3',
        'HVD_block3', 'HVD_block2', 'HVD_block1', 'HVD_block0',
        'IE_block0', 'IE_block1', 'IE_block2', 'IE_block3',
        'ID_block3', 'ID_block2', 'ID_block1', 'ID_block0'
    ]
    
    for name in target_layers:
        getattr(model_base, name).register_forward_hook(get_hook(name, features_base))
        getattr(model_edge, name).register_forward_hook(get_hook(name, features_edge))
        
    with torch.no_grad():
        _ = model_base(x)
        _ = model_edge(x)
        
    def to_numpy(tensor):
        t = tensor.squeeze().cpu().numpy()
        if t.ndim != 2:
            return t
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        return t
        
    num_layers = len(target_layers)
    fig, axes = plt.subplots(num_layers, 8, figsize=(28, 4 * num_layers))
    fig.suptitle('Full Pipeline Features: HV & I (BASE vs EDGE)', fontsize=26, y=0.99)
    
    for row, layer_name in enumerate(target_layers):
        feat_base = features_base[layer_name]
        feat_edge = features_edge[layer_name]
        
        mean_base = torch.mean(feat_base, dim=1, keepdim=True)
        axes[row, 0].imshow(to_numpy(mean_base), cmap='jet')
        axes[row, 0].set_ylabel(layer_name, fontsize=18, fontweight='bold')
        if row == 0: axes[row, 0].set_title('BASE: Mean', fontsize=16)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        
        for i in range(3):
            if i < feat_base.shape[1]:
                axes[row, i+1].imshow(to_numpy(feat_base[:, i:i+1, :, :]), cmap='gray')
            else:
                axes[row, i+1].axis('off')
            if row == 0: axes[row, i+1].set_title(f'BASE: Ch{i}', fontsize=16)
            axes[row, i+1].set_xticks([])
            axes[row, i+1].set_yticks([])
            
        mean_edge = torch.mean(feat_edge, dim=1, keepdim=True)
        axes[row, 4].imshow(to_numpy(mean_edge), cmap='jet')
        if row == 0: axes[row, 4].set_title('EDGE M: Mean', fontsize=16)
        axes[row, 4].set_xticks([])
        axes[row, 4].set_yticks([])
        
        for i in range(3):
            if i < feat_edge.shape[1]:
                axes[row, 5+i].imshow(to_numpy(feat_edge[:, i:i+1, :, :]), cmap='gray')
            else:
                axes[row, 5+i].axis('off')
            if row == 0: axes[row, 5+i].set_title(f'EDGE M: Ch{i}', fontsize=16)
            axes[row, 5+i].set_xticks([])
            axes[row, 5+i].set_yticks([])
            
    plt.tight_layout(rect=[0, 0.0, 1, 0.98])
    out_img = 'visualize_full_HVI_pipeline.png'
    plt.savefig(out_img, dpi=200)

if __name__ == '__main__':
    visualize_full_pipeline()
