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

def visualize_features_comparison():
    img_path = r'E:\PythonFile\Project\Low-Light-Image-Enhancement\mydata\dataset\dataset\LOLv1\test\low\778.png'
    if not os.path.exists(img_path):
        print(f"Không tìm thấy ảnh {img_path}.")
        return
        
    img = Image.open(img_path).convert('RGB')
    x = T.ToTensor()(img).unsqueeze(0) 
    
    model_base = CIDNet_Base(channels=[36, 36, 72, 144])
    model_edge = CIDNet_Edge(channels=[36, 36, 72, 144])
    
    weight_path_base = os.path.join(current_dir, 'weights', 'epoch_670_best_psnr.pth')
    weight_path_edge = os.path.join(current_dir, 'weights', 'epoch_700.pth') # Cập nhật tên file này cho khớp
    
    def load_weights(model, path):
        if os.path.exists(path):
            state_dict = torch.load(path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            print(f"Nạp thành công: {path}")
        else:
            print(f"Không tìm thấy: {path}")

    load_weights(model_base, weight_path_base)
    load_weights(model_edge, weight_path_edge)
    
    model_base.eval()
    model_edge.eval()
    
    with torch.no_grad():
        dtypes = x.dtype
        hvi_base = model_base.trans.HVIT(x)
        hv_0_base = model_base.HVE_block0(hvi_base) 
        
    with torch.no_grad():
        hvi_edge_trans = model_edge.trans.HVIT(x)
        hv = hvi_edge_trans[:, 0:2, :, :]
        edge_hv = model_edge.edge_hv_ext(hv, average_channels=True).to(dtypes) 
        hvi_input_edge = torch.cat([hvi_edge_trans, edge_hv], dim=1) 
        hv_0_edge = model_edge.HVE_block0(hvi_input_edge) 
        
    def to_numpy(tensor):
        t = tensor.squeeze().cpu().numpy()
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        return t
        
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('So Sánh Đặc Trưng Sau HVE_block0', fontsize=16)
    
    mean_base = torch.mean(hv_0_base, dim=1, keepdim=True)
    axes[0, 0].imshow(to_numpy(mean_base), cmap='jet')
    axes[0, 0].set_title('BASE: Mean')
    axes[0, 0].axis('off')
    
    for i in range(3):
        axes[0, i+1].imshow(to_numpy(hv_0_base[:, i:i+1, :, :]), cmap='gray')
        axes[0, i+1].set_title(f'BASE: Ch{i}')
        axes[0, i+1].axis('off')
    
    mean_edge = torch.mean(hv_0_edge, dim=1, keepdim=True)
    axes[1, 0].imshow(to_numpy(mean_edge), cmap='jet')
    axes[1, 0].set_title('w/ EDGE: Mean')
    axes[1, 0].axis('off')
    
    for i in range(3):
        axes[1, i+1].imshow(to_numpy(hv_0_edge[:, i:i+1, :, :]), cmap='gray')
        axes[1, i+1].set_title(f'w/ EDGE: Ch{i}')
        axes[1, i+1].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_img = 'visualize_HVE_block0_comparison.png'
    plt.savefig(out_img, dpi=300)
    print(f"Lưu ảnh tại: {out_img}")

if __name__ == '__main__':
    visualize_features_comparison()
