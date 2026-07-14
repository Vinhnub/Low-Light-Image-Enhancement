
import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from data.util import *
from torchvision import transforms as t


class LOLDatasetFromFolder2Stage(data.Dataset):
    def __init__(self, data_dir, gt_s1_dir, transform=None):
        super(LOLDatasetFromFolder2Stage, self).__init__()
        self.data_dir = data_dir
        self.gt_s1_dir = gt_s1_dir
        self.transform = transform
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):

        folder = self.data_dir+'/low'
        folder2 = self.data_dir+'/high'
        folder3 = self.gt_s1_dir
        
        data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]
        data_filenames3 = [join(folder3, x) for x in listdir(folder3) if is_image_file(x)]
        
        im1 = load_img(data_filenames[index])
        im2 = load_img(data_filenames2[index])
        im3 = load_img(data_filenames3[index])
        
        _, file1 = os.path.split(data_filenames[index])
        _, file2 = os.path.split(data_filenames2[index])
        
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2) 
            
            random.seed(seed)
            torch.manual_seed(seed)         
            im3 = self.transform(im3)
            
        return im1, im2, im3, file1, file2

    def __len__(self):
        return 485

    
class LOLv2DatasetFromFolder2Stage(data.Dataset):
    def __init__(self, data_dir, gt_s1_dir, transform=None):
        super(LOLv2DatasetFromFolder2Stage, self).__init__()
        self.data_dir = data_dir
        self.gt_s1_dir = gt_s1_dir
        self.transform = transform

    def __getitem__(self, index):

        folder = self.data_dir+'/Low'
        folder2 = self.data_dir+'/Normal'
        folder3 = self.gt_s1_dir
        
        data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]
        data_filenames3 = [join(folder3, x) for x in listdir(folder3) if is_image_file(x)]
        
        im1 = load_img(data_filenames[index])
        im2 = load_img(data_filenames2[index])
        im3 = load_img(data_filenames3[index])
        
        _, file1 = os.path.split(data_filenames[index])
        _, file2 = os.path.split(data_filenames2[index])
        
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        
        if self.transform:
            random.seed(seed) # apply this seed to img tranforms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)      
            
            random.seed(seed) # apply this seed to img tranforms
            torch.manual_seed(seed) # needed for torchvision 0.7 
            im2 = self.transform(im2)
            
            random.seed(seed)
            torch.manual_seed(seed)         
            im3 = self.transform(im3)
            
        return im1, im2, im3, file1, file2

    def __len__(self):
        return 685



class LOLv2SynDatasetFromFolder2Stage(data.Dataset):
    def __init__(self, data_dir, gt_s1_dir, transform=None):
        super(LOLv2SynDatasetFromFolder2Stage, self).__init__()
        self.data_dir = data_dir
        self.gt_s1_dir = gt_s1_dir
        self.transform = transform
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):

        folder = self.data_dir+'/Low'
        folder2 = self.data_dir+'/Normal'
        folder3 = self.gt_s1_dir
        
        data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]
        data_filenames3 = [join(folder3, x) for x in listdir(folder3) if is_image_file(x)]

        im1 = load_img(data_filenames[index])
        im2 = load_img(data_filenames2[index])
        im3 = load_img(data_filenames3[index])
        
        _, file1 = os.path.split(data_filenames[index])
        _, file2 = os.path.split(data_filenames2[index])
        
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2)
            
            random.seed(seed)
            torch.manual_seed(seed)         
            im3 = self.transform(im3)
            
        return im1, im2, im3, file1, file2

    def __len__(self):
        return 900
