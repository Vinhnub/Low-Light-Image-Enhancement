import os
import argparse
from glob import glob
import time
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import RetinexNet, RetinexLoss
from dataset import LowLightDataset
from utils import save_images

parser = argparse.ArgumentParser(description='')

parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default="0", help='GPU idx')
parser.add_argument('--gpu_mem', dest='gpu_mem', type=float, default=0.5, help="gpu memory usage (ignored in PyTorch)")
parser.add_argument('--phase', dest='phase', default='train', help='train or test')

parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='number of total epoches')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size')
parser.add_argument('--start_lr', dest='start_lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', type=int, default=10, help='evaluating and saving checkpoints every # epoch')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='directory for checkpoints')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='directory for evaluating outputs')

parser.add_argument('--save_dir', dest='save_dir', default='./test_results', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./data/test/low', help='directory for testing inputs')
parser.add_argument('--decom', dest='decom', type=int, default=0, help='decom flag, 0 for enhanced results only and 1 for decomposition results')

args = parser.parse_args()

def adjust_learning_rate(optimizer, epoch, start_lr):
    lr = start_lr if epoch < 20 else start_lr / 10.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def evaluate(model, eval_loader, epoch, sample_dir, train_phase, device):
    print(f"[*] Evaluating for phase {train_phase} / epoch {epoch}...")
    model.eval()
    with torch.no_grad():
        for idx, (low_im, name) in enumerate(eval_loader):
            low_im = low_im.to(device)
            R_low, L_low, I_delta, S = model(low_im)
            
            if train_phase == "Decom":
                result_1 = R_low.cpu().numpy().transpose(0, 2, 3, 1)
                result_2 = torch.cat([L_low]*3, dim=1).cpu().numpy().transpose(0, 2, 3, 1)
            elif train_phase == "Relight":
                result_1 = S.cpu().numpy().transpose(0, 2, 3, 1)
                result_2 = torch.cat([I_delta]*3, dim=1).cpu().numpy().transpose(0, 2, 3, 1)

            save_images(os.path.join(sample_dir, f'eval_{train_phase}_{idx+1}_{epoch}.png'), result_1[0], result_2[0])

def train(model, train_loader, eval_loader, criterion, optimizer, device, train_phase, start_epoch=0):
    model.train()
    print(f"[*] Start training for phase {train_phase}...")
    num_batches = len(train_loader)
    
    ckpt_path = os.path.join(args.ckpt_dir, train_phase)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    start_time = time.time()

    for epoch in range(start_epoch, args.epoch):
        adjust_learning_rate(optimizer, epoch, args.start_lr)
        
        for batch_id, (batch_low, batch_high) in enumerate(train_loader):
            batch_low = batch_low.to(device)
            batch_high = batch_high.to(device)
            
            optimizer.zero_grad()
            
            if train_phase == "Decom":
                R_low, L_low = model.decom_net(batch_low)
                R_high, L_high = model.decom_net(batch_high)
                loss = criterion(batch_low, batch_high, R_low, L_low, R_high, L_high, None, phase="Decom")
            else: # Relight
                # Frozen DecomNet during Relight training
                with torch.no_grad():
                    R_low, L_low = model.decom_net(batch_low)
                    R_high, L_high = model.decom_net(batch_high)
                
                I_delta = model.relight_net(L_low, R_low)
                loss = criterion(batch_low, batch_high, R_low, L_low, R_high, L_high, I_delta, phase="Relight")
            
            loss.backward()
            optimizer.step()
            
            print(f"{train_phase} Epoch: [{epoch+1}] [{batch_id+1:04d}/{num_batches:04d}] time: {time.time()-start_time:.4f}, loss: {loss.item():.6f}")

        if (epoch + 1) % args.eval_every_epoch == 0:
            evaluate(model, eval_loader, epoch + 1, args.sample_dir, train_phase, device)
            model.train()
            torch.save(model.state_dict(), os.path.join(ckpt_path, f'RetinexNet_{train_phase}_epoch{epoch+1}.pth'))

def lowlight_train(model, device):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    train_low_data_names = glob('E:/PythonFile/Project/Low-Light-Image-Enhancement/mydata/dataset/dataset/LOLv1/train/low/*.png') + glob('./data/syn/low/*.png')
    train_low_data_names.sort()
    train_high_data_names = glob('E:/PythonFile/Project/Low-Light-Image-Enhancement/mydata/dataset/dataset/LOLv1/train/high/*.png') + glob('./data/syn/high/*.png')
    train_high_data_names.sort()
    assert len(train_low_data_names) == len(train_high_data_names)

    eval_low_data_names = glob('E:/PythonFile/Project/Low-Light-Image-Enhancement/mydata/dataset/dataset/LOLv1/test/low/*.*')

    train_dataset = LowLightDataset(train_low_data_names, train_high_data_names, patch_size=args.patch_size, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    eval_dataset = LowLightDataset(eval_low_data_names, mode='eval')
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)

    criterion = RetinexLoss()

    # Train Decom
    optimizer_Decom = optim.Adam(model.decom_net.parameters(), lr=args.start_lr)
    train(model, train_loader, eval_loader, criterion, optimizer_Decom, device, "Decom")

    # Train Relight
    optimizer_Relight = optim.Adam(model.relight_net.parameters(), lr=args.start_lr)
    train(model, train_loader, eval_loader, criterion, optimizer_Relight, device, "Relight")

def lowlight_test(model, device):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    test_low_data_names = glob(os.path.join(args.test_dir) + '/*.*')
    test_dataset = LowLightDataset(test_low_data_names, mode='eval')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("[*] Testing...")
    model.eval()
    with torch.no_grad():
        for idx, (low_im, name) in enumerate(test_loader):
            print(name[0])
            low_im = low_im.to(device)
            R_low, L_low, I_delta, S = model(low_im)

            name_only = os.path.splitext(name[0])[0]
            suffix = os.path.splitext(name[0])[1][1:]

            S_np = S.cpu().numpy().transpose(0, 2, 3, 1)[0]
            
            if args.decom == 1:
                R_low_np = R_low.cpu().numpy().transpose(0, 2, 3, 1)[0]
                L_low_np = torch.cat([L_low]*3, dim=1).cpu().numpy().transpose(0, 2, 3, 1)[0]
                I_delta_np = torch.cat([I_delta]*3, dim=1).cpu().numpy().transpose(0, 2, 3, 1)[0]

                save_images(os.path.join(args.save_dir, f"{name_only}_R_low.{suffix}"), R_low_np)
                save_images(os.path.join(args.save_dir, f"{name_only}_I_low.{suffix}"), L_low_np)
                save_images(os.path.join(args.save_dir, f"{name_only}_I_delta.{suffix}"), I_delta_np)
            
            save_images(os.path.join(args.save_dir, f"{name_only}_S.{suffix}"), S_np)

def main():
    if args.use_gpu and torch.cuda.is_available():
        print(f"[*] GPU {args.gpu_idx}\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)
        device = torch.device("cuda")
    else:
        print("[*] CPU\n")
        device = torch.device("cpu")

    model = RetinexNet().to(device)

    if args.phase == 'train':
        lowlight_train(model, device)
    elif args.phase == 'test':
        # Load weights
        decom_ckpt = os.path.join(args.ckpt_dir, "Decom")
        relight_ckpt = os.path.join(args.ckpt_dir, "Relight")
        decom_files = sorted(glob(os.path.join(decom_ckpt, '*.pth')))
        relight_files = sorted(glob(os.path.join(relight_ckpt, '*.pth')))
        if relight_files:
            print("[*] Load weights successfully...")
            model.load_state_dict(torch.load(relight_files[-1], map_location=device), strict=False)
        elif decom_files:
            print("[*] Load Decom weights only...")
            model.load_state_dict(torch.load(decom_files[-1], map_location=device), strict=False)
        else:
            print("[!] Checkpoints not found, testing with random weights...")

        lowlight_test(model, device)
    else:
        print('[!] Unknown phase')

if __name__ == '__main__':
    main()
