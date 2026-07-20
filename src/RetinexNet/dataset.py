import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from utils import load_images, data_augmentation

class LowLightDataset(Dataset):
    def __init__(self, low_data_names, high_data_names=None, patch_size=48, mode='train'):
        self.low_data_names = low_data_names
        self.high_data_names = high_data_names
        self.patch_size = patch_size
        self.mode = mode
        
        self.train_low_data = []
        self.train_high_data = []
        
        print(f"[*] Loading {len(self.low_data_names)} images...")
        for i in range(len(self.low_data_names)):
            self.train_low_data.append(load_images(self.low_data_names[i]))
            if self.high_data_names is not None:
                self.train_high_data.append(load_images(self.high_data_names[i]))

    def __len__(self):
        return len(self.train_low_data)

    def __getitem__(self, idx):
        low_im = self.train_low_data[idx]
        
        if self.mode == 'train':
            high_im = self.train_high_data[idx]
            h, w, _ = low_im.shape
            x = random.randint(0, h - self.patch_size)
            y = random.randint(0, w - self.patch_size)
            
            rand_mode = random.randint(0, 7)
            
            low_patch = data_augmentation(low_im[x:x+self.patch_size, y:y+self.patch_size, :], rand_mode)
            high_patch = data_augmentation(high_im[x:x+self.patch_size, y:y+self.patch_size, :], rand_mode)
            
            # To tensor: HWC -> CHW
            low_patch = torch.from_numpy(low_patch.transpose((2, 0, 1)).copy()).float()
            high_patch = torch.from_numpy(high_patch.transpose((2, 0, 1)).copy()).float()
            
            return low_patch, high_patch
            
        else: # eval or test
            # To tensor: HWC -> CHW
            low_im = torch.from_numpy(low_im.transpose((2, 0, 1)).copy()).float()
            name = os.path.basename(self.low_data_names[idx])
            if self.high_data_names is not None:
                high_im = self.train_high_data[idx]
                high_im = torch.from_numpy(high_im.transpose((2, 0, 1)).copy()).float()
                return low_im, high_im, name
            return low_im, name
