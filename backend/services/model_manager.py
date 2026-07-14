import os
import sys
import torch

loaded_models = {
    "retinexformer": None,
    "cidnet": None
}

def load_all_models():
    """
    Load Retinexformer and CIDNet models into memory.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading models on device: {device}")

    # ---------------------------------------------------------
    # 1. Load Retinexformer
    # ---------------------------------------------------------
    retinexformer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/Retinexformer'))
    if retinexformer_path not in sys.path:
        sys.path.insert(0, retinexformer_path)
    
    try:
        from basicsr.models import create_model
        from basicsr.utils.options import parse
        
        # Paths for Retinexformer config and weights
        # Assuming SDSD_indoor as default config, user can change this
        opt_path = os.path.join(retinexformer_path, 'Options/RetinexFormer_SDSD_indoor.yml')
        weights_path = os.path.join(retinexformer_path, 'pretrained_weights/SDSD_indoor.pth')
        
        if os.path.exists(opt_path) and os.path.exists(weights_path):
            opt = parse(opt_path, is_train=False)
            opt['dist'] = False
            model_restoration = create_model(opt).net_g
            
            checkpoint = torch.load(weights_path, map_location=device)
            try:
                model_restoration.load_state_dict(checkpoint['params'])
            except:
                new_checkpoint = {}
                for k in checkpoint['params']:
                    new_checkpoint['module.' + k] = checkpoint['params'][k]
                model_restoration.load_state_dict(new_checkpoint)
            
            model_restoration.to(device)
            model_restoration.eval()
            loaded_models["retinexformer"] = model_restoration
            print("Retinexformer loaded successfully.")
        else:
            print(f"Retinexformer weights or config not found. Expected: {weights_path} and {opt_path}")
    except Exception as e:
        print(f"Error loading Retinexformer: {e}")

    # ---------------------------------------------------------
    # 2. Load HVI-CIDNet
    # ---------------------------------------------------------
    cidnet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/HVI-CIDNet'))
    if cidnet_path not in sys.path:
        sys.path.insert(0, cidnet_path)

    try:
        from net.CIDNet import CIDNet
        # Assume default weight path
        cidnet_weights_path = os.path.join(cidnet_path, 'weights/train/epoch_460_best_psnr.pth')
        
        if os.path.exists(cidnet_weights_path):
            model_cidnet = CIDNet().to(device)
            
            checkpoint = torch.load(cidnet_weights_path, map_location=device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]

            cleaned_state_dict = {}
            for key, value in checkpoint.items():
                if key.startswith("module."):
                    key = key[len("module."):]
                cleaned_state_dict[key] = value

            model_cidnet.load_state_dict(cleaned_state_dict)
            model_cidnet.eval()
            loaded_models["cidnet"] = model_cidnet
            print("CIDNet loaded successfully.")
        else:
            print(f"CIDNet weights not found. Expected: {cidnet_weights_path}")
    except Exception as e:
        print(f"Error loading CIDNet: {e}")

def get_model(name: str):
    name = name.lower()
    if name == 'cidnet':
        name = 'cidnet'
    return loaded_models.get(name)
