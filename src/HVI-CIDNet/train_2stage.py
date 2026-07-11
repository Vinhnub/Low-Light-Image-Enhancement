import torch
import os
import random
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from net.CIDNet_2Stage import CIDNetTwoStage as CIDNet
from data.options_2stage import option
from measure import metrics
from eval import eval
from data.data_2stage import *
from loss.losses import *
from data.scheduler import *
from tqdm import tqdm
from datetime import datetime

opt = option().parse_args()

def seed_torch():
    seed = random.randint(1, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def train_init():
    seed_torch()
    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
def train(epoch):
    model.train()
    loss_print = 0
    pic_cnt = 0
    loss_last_10 = 0
    loss_s1_print = 0
    loss_s2_print = 0
    pic_last_10 = 0
    train_len = len(training_data_loader)
    iter = 0
    torch.autograd.set_detect_anomaly(opt.grad_detect)
    for batch in tqdm(training_data_loader):
        im1, im2, im3, path1, path2 = batch[0], batch[1], batch[2], batch[3], batch[4]
        im1 = im1.cuda()
        im2 = im2.cuda()
        im3 = im3.cuda()
        
        # use random gamma function (enhancement curve) to improve generalization
        if opt.gamma:
            gamma = (
                random.randint(
                    opt.start_gamma,
                    opt.end_gamma
                ) / 100.0
            )
            input_low = im1 ** gamma
            input_gt = im2 ** gamma
            gt_s1 = im3 ** gamma
        else:
            input_low = im1
            input_gt = im2
            gt_s1 = im3

        # Forward pass returning outputs from both stages
        output_rgb_s1, output_rgb_s2 = model(
            input_low,
            stop_grad=opt.stop_grad,
            return_all=True
        )

        # Calculate Stage 1 Losses (RGB + HVI)
        output_hvi_s1 = model.HVIT(output_rgb_s1)
        gt_hvi_s1 = model.HVIT(gt_s1)
        
        loss_hvi_s1 = s1_L1_loss(output_hvi_s1, gt_hvi_s1) + s1_D_loss(output_hvi_s1, gt_hvi_s1) + s1_E_loss(output_hvi_s1, gt_hvi_s1) + s1_LSGD_loss(output_hvi_s1, gt_hvi_s1, is_hvi=True)
        if opt.s1_P_weight > 0:
            loss_hvi_s1 += opt.s1_P_weight * s1_P_loss(output_hvi_s1, gt_hvi_s1)[0]
            
        loss_rgb_s1 = s1_L1_loss(output_rgb_s1, gt_s1) + s1_D_loss(output_rgb_s1, gt_s1) + s1_E_loss(output_rgb_s1, gt_s1) + s1_LSGD_loss(output_rgb_s1, gt_s1)
        if opt.s1_P_weight > 0:
            loss_rgb_s1 += opt.s1_P_weight * s1_P_loss(output_rgb_s1, gt_s1)[0]
            
        loss_s1 = loss_rgb_s1 + opt.s1_HVI_weight * loss_hvi_s1

        # Calculate Stage 2 Losses (RGB + HVI)
        output_hvi_s2 = model.HVIT(output_rgb_s2)
        gt_hvi_s2 = model.HVIT(input_gt)
        
        loss_hvi_s2 = s2_L1_loss(output_hvi_s2, gt_hvi_s2) + s2_D_loss(output_hvi_s2, gt_hvi_s2) + s2_E_loss(output_hvi_s2, gt_hvi_s2) + s2_LSGD_loss(output_hvi_s2, gt_hvi_s2, is_hvi=True)
        if opt.s2_P_weight > 0:
            loss_hvi_s2 += opt.s2_P_weight * s2_P_loss(output_hvi_s2, gt_hvi_s2)[0]
            
        loss_rgb_s2 = s2_L1_loss(output_rgb_s2, input_gt) + s2_D_loss(output_rgb_s2, input_gt) + s2_E_loss(output_rgb_s2, input_gt) + s2_LSGD_loss(output_rgb_s2, input_gt)
        if opt.s2_P_weight > 0:
            loss_rgb_s2 += opt.s2_P_weight * s2_P_loss(output_rgb_s2, input_gt)[0]
            
        loss_s2 = loss_rgb_s2 + opt.s2_HVI_weight * loss_hvi_s2

        # Combined Loss
        loss = opt.lambda1 * loss_s1 + opt.lambda2 * loss_s2
        iter += 1
        
        if opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_print = loss_print + loss.item()
        loss_last_10 = loss_last_10 + loss.item()
        loss_s1_print += loss_s1.item()
        loss_s2_print += loss_s2.item()
        pic_cnt += 1
        pic_last_10 += 1

        if iter == train_len:
            print("===> Epoch[{}]: Loss: {:.4f} (S1: {:.4f}, S2: {:.4f}) || Learning rate: lr={}.".format(
                epoch,
                loss_last_10/pic_last_10,
                loss_s1_print/pic_last_10,
                loss_s2_print/pic_last_10,
                optimizer.param_groups[0]['lr']
            ))
            loss_last_10 = 0
            loss_s1_print = 0
            loss_s2_print = 0
            pic_last_10 = 0

            # Save sample outputs for visual validation
            output_img_s1 = transforms.ToPILImage()((output_rgb_s1)[0].squeeze(0))
            output_img_s2 = transforms.ToPILImage()((output_rgb_s2)[0].squeeze(0))
            gt_img = transforms.ToPILImage()((im2)[0].squeeze(0))
            if not os.path.exists(opt.val_folder+'training'):          
                os.mkdir(opt.val_folder+'training') 
            output_img_s1.save(opt.val_folder+'training/test_s1.png')
            output_img_s2.save(opt.val_folder+'training/test_s2.png')
            gt_img.save(opt.val_folder+'training/gt.png')
            
    return loss_print, pic_cnt

def checkpoint(epoch):
    if not os.path.exists("./weights"):          
        os.mkdir("./weights") 
    if not os.path.exists("./weights/train"):          
        os.mkdir("./weights/train")  
    model_out_path = "./weights/train/epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path
    
def load_datasets():
    print(f'===> Loading datasets: {opt.dataset}')
    if opt.dataset == 'lol_v1':
        train_set = get_lol_training_set(opt.data_train_lol_v1, opt.data_train_gt_s1_lol_v1, size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lol_v1)
        
    elif opt.dataset == 'lol_blur':
        train_set = get_training_set_blur(opt.data_train_lol_blur,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lol_blur)

    elif opt.dataset == 'lolv2_real':
        train_set = get_lol_v2_training_set(opt.data_train_lolv2_real, opt.data_train_gt_s1_lolv2_real, size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lolv2_real)
        
    elif opt.dataset == 'lolv2_syn':
        train_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn, opt.data_train_gt_s1_lolv2_syn, size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lolv2_syn)
    
    elif opt.dataset == 'SID':
        train_set = get_SID_training_set(opt.data_train_SID,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_SID)
        
    elif opt.dataset == 'SICE_mix':
        train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
        test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
        
    elif opt.dataset == 'SICE_grad':
        train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
        test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
        
    elif opt.dataset == 'fivek':
        train_set = get_fivek_training_set(opt.data_train_fivek,size=opt.cropSize)
        test_set = get_fivek_eval_set(opt.data_val_fivek)
    else:
        raise Exception("should choose a dataset")
    
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    return training_data_loader, testing_data_loader

def build_model():
    print('===> Building model ')
    model = CIDNet().cuda()
    if opt.start_epoch > 0:
        pth = f"./weights/train/epoch_{opt.start_epoch}.pth"
        model.load_state_dict(torch.load(pth, map_location=lambda storage, loc: storage))
    return model

def make_scheduler():
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)      
    if opt.cos_restart_cyclic:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[(opt.nEpochs//4)-opt.warmup_epochs, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[opt.nEpochs//4, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
    elif opt.cos_restart:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.warmup_epochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
    else:
        raise Exception("should choose a scheduler")
    return optimizer,scheduler

def init_loss():
    # Stage 1 Losses
    s1_L1_loss= L1Loss(loss_weight=opt.s1_L1_weight, reduction='mean').cuda()
    s1_D_loss = SSIM(weight=opt.s1_D_weight).cuda()
    s1_E_loss = EdgeLoss(loss_weight=opt.s1_E_weight).cuda()
    s1_P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = opt.s1_P_weight ,criterion='mse').cuda()
    s1_LSGD_loss = RegionLSGDLoss(loss_weight=opt.s1_LSGD_weight).cuda()

    # Stage 2 Losses
    s2_L1_loss= L1Loss(loss_weight=opt.s2_L1_weight, reduction='mean').cuda()
    s2_D_loss = SSIM(weight=opt.s2_D_weight).cuda()
    s2_E_loss = EdgeLoss(loss_weight=opt.s2_E_weight).cuda()
    s2_P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = opt.s2_P_weight ,criterion='mse').cuda()
    s2_LSGD_loss = RegionLSGDLoss(loss_weight=opt.s2_LSGD_weight).cuda()

    return (
        s1_L1_loss, s1_P_loss, s1_E_loss, s1_D_loss, s1_LSGD_loss,
        s2_L1_loss, s2_P_loss, s2_E_loss, s2_D_loss, s2_LSGD_loss
    )

if __name__ == '__main__':  
    
    '''
    preparision
    '''
    train_init()
    training_data_loader, testing_data_loader = load_datasets()
    model = build_model()
    optimizer,scheduler = make_scheduler()
    s1_L1_loss, s1_P_loss, s1_E_loss, s1_D_loss, s1_LSGD_loss, s2_L1_loss, s2_P_loss, s2_E_loss, s2_D_loss, s2_LSGD_loss = init_loss()
    
    '''
    train
    '''
    psnr = []
    ssim = []
    lpips = []
    start_epoch=0
    if opt.start_epoch > 0:
        start_epoch = opt.start_epoch
    if not os.path.exists(opt.val_folder):          
        os.mkdir(opt.val_folder) 
        
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    with open(f"./results/training/metrics_2stage_{now}.md", "w") as f:
        f.write("dataset: "+ opt.dataset + "\n")  
        f.write(f"lr: {opt.lr}\n")  
        f.write(f"batch size: {opt.batchSize}\n")  
        f.write(f"crop size: {opt.cropSize}\n")  
        f.write(f"S1_HVI_weight: {opt.s1_HVI_weight}, S2_HVI_weight: {opt.s2_HVI_weight}\n")  
        f.write(f"S1_L1_weight: {opt.s1_L1_weight}, S2_L1_weight: {opt.s2_L1_weight}\n")  
        f.write(f"S1_D_weight: {opt.s1_D_weight}, S2_D_weight: {opt.s2_D_weight}\n")  
        f.write(f"S1_E_weight: {opt.s1_E_weight}, S2_E_weight: {opt.s2_E_weight}\n")  
        f.write(f"S1_P_weight: {opt.s1_P_weight}, S2_P_weight: {opt.s2_P_weight}\n")  
        f.write(f"S1_LSGD_weight: {opt.s1_LSGD_weight}, S2_LSGD_weight: {opt.s2_LSGD_weight}\n")  
        f.write(f"2Stage alpha: {opt.alpha}\n")  
        f.write(f"2Stage lambda1: {opt.lambda1}\n")  
        f.write(f"2Stage lambda2: {opt.lambda2}\n")  
        f.write(f"2Stage stop_grad: {opt.stop_grad}\n")  
        f.write("| Epochs | PSNR | SSIM | LPIPS |\n")  
        f.write("|----------------------|----------------------|----------------------|----------------------|\n")  
        
    for epoch in range(start_epoch+1, opt.nEpochs + start_epoch + 1):
        epoch_loss, pic_num = train(epoch)
        scheduler.step()
        
        if epoch % opt.snapshots == 0:
            model_out_path = checkpoint(epoch) 
            norm_size = True

            # LOL three subsets
            if opt.dataset == 'lol_v1':
                output_folder = 'LOLv1/'
                label_dir = opt.data_valgt_lol_v1
            if opt.dataset == 'lolv2_real':
                output_folder = 'LOLv2_real/'
                label_dir = opt.data_valgt_lolv2_real
            if opt.dataset == 'lolv2_syn':
                output_folder = 'LOLv2_syn/'
                label_dir = opt.data_valgt_lolv2_syn
            
            # LOL-blur dataset with low_blur and high_sharp_scaled
            if opt.dataset == 'lol_blur':
                output_folder = 'LOL_blur/'
                label_dir = opt.data_valgt_lol_blur
                
            if opt.dataset == 'SID':
                output_folder = 'SID/'
                label_dir = opt.data_valgt_SID
                npy = True
            if opt.dataset == 'SICE_mix':
                output_folder = 'SICE_mix/'
                label_dir = opt.data_valgt_SICE_mix
                norm_size = False
            if opt.dataset == 'SICE_grad':
                output_folder = 'SICE_grad/'
                label_dir = opt.data_valgt_SICE_grad
                norm_size = False
                
            if opt.dataset == 'fivek':
                output_folder = 'fivek/'
                label_dir = opt.data_valgt_fivek
                norm_size = False

            im_dir = opt.val_folder + output_folder + '*.png'
            is_lol_v1 = (opt.dataset == 'lol_v1')
            is_lolv2_real = (opt.dataset == 'lolv2_real')
            eval(model, testing_data_loader, model_out_path, opt.val_folder+output_folder, 
                 norm_size=norm_size, LOL=is_lol_v1, v2=is_lolv2_real, alpha=0.8)
            
            avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=False)
            print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
            print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
            print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))
            psnr.append(avg_psnr)
            ssim.append(avg_ssim)
            lpips.append(avg_lpips)
            print(psnr)
            print(ssim)
            print(lpips)
            with open(f"./results/training/metrics_2stage_{now}.md", "a") as f:
                f.write(f"| {epoch} | { avg_psnr:.4f} | {avg_ssim:.4f} | {avg_lpips:.4f} |\n") 

            # --- Eval with GT Mean
            avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=True)
            print("===> Avg.PSNR (GT): {:.4f} dB ".format(avg_psnr))
            print("===> Avg.SSIM (GT): {:.4f} ".format(avg_ssim))
            print("===> Avg.LPIPS (GT): {:.4f} ".format(avg_lpips))
            with open(f"./results/training/metrics_2stage_{now}.md", "a") as f:
                f.write(f"| {epoch} | { avg_psnr:.4f} | {avg_ssim:.4f} | {avg_lpips:.4f} | GT Mean |\n") 

        torch.cuda.empty_cache()
