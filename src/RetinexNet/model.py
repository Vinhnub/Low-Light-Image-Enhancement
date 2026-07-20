import torch
import torch.nn as nn
import torch.nn.functional as F

class DecomNet(nn.Module):
    def __init__(self, layer_num=5, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.layer_num = layer_num
        self.conv0 = nn.Conv2d(4, channel, kernel_size * 3, padding=4)
        
        self.convs = nn.ModuleList([
            nn.Conv2d(channel, channel, kernel_size, padding=1) for _ in range(layer_num)
        ])
        self.recon_layer = nn.Conv2d(channel, 4, kernel_size, padding=1)

    def forward(self, input_im):
        # input_im: (B, 3, H, W)
        input_max, _ = torch.max(input_im, dim=1, keepdim=True)
        x = torch.cat([input_max, input_im], dim=1)
        
        x = self.conv0(x)
        for i in range(self.layer_num):
            x = F.relu(self.convs[i](x))
            
        x = self.recon_layer(x)
        
        R = torch.sigmoid(x[:, 0:3, :, :])
        L = torch.sigmoid(x[:, 3:4, :, :])
        
        return R, L

class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()
        self.conv0 = nn.Conv2d(4, channel, kernel_size, padding=1)
        self.conv1 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1)
        
        self.deconv1 = nn.Conv2d(channel, channel, kernel_size, padding=1)
        self.deconv2 = nn.Conv2d(channel, channel, kernel_size, padding=1)
        self.deconv3 = nn.Conv2d(channel, channel, kernel_size, padding=1)
        
        self.feature_fusion = nn.Conv2d(channel * 3, channel, 1, padding=0)
        self.output_layer = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, input_L, input_R):
        x = torch.cat([input_R, input_L], dim=1)
        
        conv0 = self.conv0(x)
        conv1 = F.relu(self.conv1(conv0))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        
        up1 = F.interpolate(conv3, size=(conv2.size(2), conv2.size(3)), mode='nearest')
        deconv1 = F.relu(self.deconv1(up1)) + conv2
        
        up2 = F.interpolate(deconv1, size=(conv1.size(2), conv1.size(3)), mode='nearest')
        deconv2 = F.relu(self.deconv2(up2)) + conv1
        
        up3 = F.interpolate(deconv2, size=(conv0.size(2), conv0.size(3)), mode='nearest')
        deconv3 = F.relu(self.deconv3(up3)) + conv0
        
        deconv1_resize = F.interpolate(deconv1, size=(deconv3.size(2), deconv3.size(3)), mode='nearest')
        deconv2_resize = F.interpolate(deconv2, size=(deconv3.size(2), deconv3.size(3)), mode='nearest')
        
        feature_gather = torch.cat([deconv1_resize, deconv2_resize, deconv3], dim=1)
        feature_fusion = self.feature_fusion(feature_gather)
        output = self.output_layer(feature_fusion)
        
        return output

class RetinexNet(nn.Module):
    def __init__(self):
        super(RetinexNet, self).__init__()
        self.decom_net = DecomNet()
        self.relight_net = RelightNet()
        
    def forward(self, x_low, x_high=None):
        R_low, L_low = self.decom_net(x_low)
        I_delta = self.relight_net(L_low, R_low)
        S = R_low * torch.cat([I_delta]*3, dim=1)
        
        if x_high is not None:
            R_high, L_high = self.decom_net(x_high)
            return R_low, L_low, R_high, L_high, I_delta, S
        
        return R_low, L_low, I_delta, S

class RetinexLoss(nn.Module):
    def __init__(self):
        super(RetinexLoss, self).__init__()
        self.smooth_kernel_x = torch.FloatTensor([[[[0, 0], [-1, 1]]]]).view(1, 1, 2, 2)
        self.smooth_kernel_y = torch.FloatTensor([[[[0, -1], [0, 1]]]]).view(1, 1, 2, 2)
        
    def gradient(self, input_tensor, direction):
        kernel = self.smooth_kernel_x if direction == "x" else self.smooth_kernel_y
        kernel = kernel.to(input_tensor.device)
        grad = F.conv2d(input_tensor, kernel, padding=1)
        return torch.abs(grad[:, :, :-1, :-1])

    def ave_gradient(self, input_tensor, direction):
        grad = self.gradient(input_tensor, direction)
        return F.avg_pool2d(grad, kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R_gray = 0.2989 * input_R[:, 0:1, :, :] + 0.5870 * input_R[:, 1:2, :, :] + 0.1140 * input_R[:, 2:3, :, :]
        
        grad_I_x = self.gradient(input_I, "x")
        grad_I_y = self.gradient(input_I, "y")
        
        grad_R_x = self.ave_gradient(input_R_gray, "x")
        grad_R_y = self.ave_gradient(input_R_gray, "y")
        
        weight_x = torch.exp(-10 * grad_R_x)
        weight_y = torch.exp(-10 * grad_R_y)
        
        loss_x = grad_I_x * weight_x
        loss_y = grad_I_y * weight_y
        
        return torch.mean(loss_x + loss_y)

    def forward(self, input_low, input_high, R_low, I_low, R_high, I_high, I_delta, phase="Decom"):
        if phase == "Decom":
            I_low_3 = torch.cat([I_low]*3, dim=1)
            I_high_3 = torch.cat([I_high]*3, dim=1)
            
            recon_loss_low = torch.mean(torch.abs(R_low * I_low_3 - input_low))
            recon_loss_high = torch.mean(torch.abs(R_high * I_high_3 - input_high))
            recon_loss_mutal_low = torch.mean(torch.abs(R_high * I_low_3 - input_low))
            recon_loss_mutal_high = torch.mean(torch.abs(R_low * I_high_3 - input_high))
            equal_R_loss = torch.mean(torch.abs(R_low - R_high))
            
            Ismooth_loss_low = self.smooth(I_low, R_low)
            Ismooth_loss_high = self.smooth(I_high, R_high)
            
            loss_Decom = recon_loss_low + recon_loss_high + \
                         0.001 * recon_loss_mutal_low + 0.001 * recon_loss_mutal_high + \
                         0.1 * Ismooth_loss_low + 0.1 * Ismooth_loss_high + \
                         0.01 * equal_R_loss
            return loss_Decom
            
        elif phase == "Relight":
            I_delta_3 = torch.cat([I_delta]*3, dim=1)
            relight_loss = torch.mean(torch.abs(R_low * I_delta_3 - input_high))
            Ismooth_loss_delta = self.smooth(I_delta, R_low)
            
            loss_Relight = relight_loss + 3 * Ismooth_loss_delta
            return loss_Relight
