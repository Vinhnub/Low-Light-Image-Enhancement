import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from basicsr.models.losses.loss_util import *

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss   #把 l1_loss 作为 weighted_loss 的输入
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss   #把 mse_loss 作为 weighted_loss 的输入
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss
    
class EdgeLoss(nn.Module):
    def __init__(self,loss_weight=1.0, reduction='mean'):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1).cuda()

        self.weight = loss_weight
        
    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = mse_loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss*self.weight
    
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True,weight=1.):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.weight = weight

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return (1. - map_ssim(img1, img2, window, self.window_size, channel, self.size_average)) * self.weight


_reduction_modes = ['none', 'mean', 'sum']
class RegionLSGDLoss(nn.Module):

    def __init__(
        self,
        loss_weight=1.0,
        reduction='mean',
        use_dark_weight=True,
        eps=1e-6,
    ):

        super(RegionLSGDLoss, self).__init__()

        if reduction not in _reduction_modes:
            raise ValueError(
                f'Unsupported reduction mode: '
                f'{reduction}. '
                f'Supported ones are: '
                f'{_reduction_modes}'
            )

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.use_dark_weight = use_dark_weight
        self.eps = eps

    def compute_spatial_gradient(self, img):
        """
        Compute spatial first-order variation

        img:
            [B,C,H,W]

        return:
            grad_x, grad_y
        """

        grad_x = (
            img[:, :, :, 1:]
            - img[:, :, :, :-1]
        )

        grad_y = (
            img[:, :, 1:, :]
            - img[:, :, :-1, :]
        )

        return grad_x, grad_y

    def compute_weight_map(self, gt, is_hvi=False):
        """
        Dark-region weighting.

        For RGB:
            luminance-based

        For HVI:
            use I channel directly
        """

        channels = gt.shape[1]

        # -------------------
        # RGB
        # -------------------
        if channels == 3:

            luminance = (
                0.299 * gt[:, 0:1]
                + 0.587 * gt[:, 1:2]
                + 0.114 * gt[:, 2:3]
            ) if is_hvi else gt[:, 2:3]

        # -------------------
        # grayscale
        # -------------------
        elif channels == 1:

            luminance = gt

        else:
            raise ValueError(
                f'Unsupported channel size: {channels}'
            )

        # darker -> larger weight
        weight = 1.0 - luminance

        return weight.clamp(
            min=self.eps,
            max=1.0
        )

    def reduction_fn(self, x):

        if self.reduction == 'mean':
            return x.mean()

        elif self.reduction == 'sum':
            return x.sum()

        return x

    def forward(self, pred, gt, is_hvi=False):
        """
        pred:
            output_rgb or output_hvi

        gt:
            gt_rgb or gt_hvi
        """
        pred_gx, pred_gy = (
            self.compute_spatial_gradient(pred)
        )

        gt_gx, gt_gy = (
            self.compute_spatial_gradient(gt)
        )

        diff_x = torch.abs(
            pred_gx - gt_gx
        )

        diff_y = torch.abs(
            pred_gy - gt_gy
        )

        # -------------------
        # region weighting
        # -------------------
        if self.use_dark_weight:

            weight = self.compute_weight_map(gt, is_hvi=is_hvi)

            weight_x = weight[:, :, :, 1:]
            weight_y = weight[:, :, 1:, :]

            diff_x = (diff_x * weight_x)

            diff_y = (diff_y * weight_y)

        loss_x = self.reduction_fn(diff_x)
        loss_y = self.reduction_fn(diff_y)

        loss = (loss_x + loss_y)

        return (loss * self.loss_weight)
    


class L1EdgeRegionLoss(nn.Module):
    def __init__(self, l1_weight=1.0, edge_weight=10.0, region_weight=1.0):
        super(L1EdgeRegionLoss, self).__init__()
        self.l1 = L1Loss(loss_weight=l1_weight, reduction='mean')
        self.edge = EdgeLoss(loss_weight=edge_weight, reduction='mean')
        self.region = RegionLSGDLoss(loss_weight=region_weight, reduction='mean')

    def forward(self, pred, target):
        return (
            self.l1(pred, target)
            + self.edge(pred, target)
            + self.region(pred, target)
        )


# def gradient(input_tensor, direction):
#     smooth_kernel_x = torch.reshape(torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32), [2, 2, 1, 1])
#     smooth_kernel_y = torch.transpose(smooth_kernel_x, 0, 1)
#     if direction == "x":
#         kernel = smooth_kernel_x
#     elif direction == "y":
#         kernel = smooth_kernel_y
#     gradient_orig = torch.abs(torch.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
#     grad_min = torch.min(gradient_orig)
#     grad_max = torch.max(gradient_orig)
#     grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
#     return grad_norm

# class SmoothLoss(nn.Moudle):
#     """ illumination smoothness"""

#     def __init__(self, loss_weight=0.15, reduction='mean', eps=1e-2):
#         super(SmoothLoss,self).__init__()
#         self.loss_weight = loss_weight
#         self.eps = eps
#         self.reduction = reduction
    
#     def forward(self, illu, img):
#         # illu: b×c×h×w   illumination map
#         # img:  b×c×h×w   input image
#         illu_gradient_x = gradient(illu, "x")
#         img_gradient_x  = gradient(img, "x")
#         x_loss = torch.abs(torch.div(illu_gradient_x, torch.maximum(img_gradient_x, 0.01)))

#         illu_gradient_y = gradient(illu, "y")
#         img_gradient_y  = gradient(img, "y")
#         y_loss = torch.abs(torch.div(illu_gradient_y, torch.maximum(img_gradient_y, 0.01)))

#         loss = torch.mean(x_loss + y_loss) * self.loss_weight

#         return loss

# class MultualLoss(nn.Moudle):
#     """ Multual Consistency"""

#     def __init__(self, loss_weight=0.20, reduction='mean'):
#         super(MultualLoss,self).__init__()

#         self.loss_weight = loss_weight
#         self.reduction = reduction
    

#     def forward(self, illu):
#         # illu: b x c x h x w
#         gradient_x = gradient(illu,"x")
#         gradient_y = gradient(illu,"y")

#         x_loss = gradient_x * torch.exp(-10*gradient_x)
#         y_loss = gradient_y * torch.exp(-10*gradient_y)

#         loss = torch.mean(x_loss+y_loss) * self.loss_weight
#         return loss




