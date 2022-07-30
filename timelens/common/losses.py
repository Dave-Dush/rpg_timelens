import torch.nn as nn
import torch
import lpips

class FusionLoss(nn.Module):
    def __init__(self, device):
        super(FusionLoss, self).__init__()
        self.device = device

    def lpips_loss(self, img1, img2):
        loss_fn_alex = lpips.LPIPS("net=alex", verbose=False).to(self.device)
        return loss_fn_alex(img1, img2)

    def l1_loss(self, img1, img2):
        l1_loss_fn = nn.L1Loss()
        return l1_loss_fn(img1, img2)

    def forward(self, img1, img2):
        self.perceptual_loss = 0.7 * (self.lpips_loss(img1, img2))

        self.adjusted_l1_loss = (1-0.70) * self.l1_loss(img1, img2)

        self.fusion_loss = torch.mean(self.perceptual_loss) + self.adjusted_l1_loss
        return self.fusion_loss
