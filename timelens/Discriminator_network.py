import torch.nn as nn
import torch as th

def _pack(example, rgb2concat, detach=False):
    ip_tensor =  th.cat([example['before']['voxel_grid'],
                   example['before']['rgb_image_tensor'],
                   example['after']['voxel_grid'],
                   example['after']['rgb_image_tensor'],
                   rgb2concat], dim=1)
    ip_tensor = ip_tensor.detach().to(device="cuda") if detach else ip_tensor.to(device="cuda")
    return ip_tensor

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator_network = nn.Sequential(
            # (2*(3 + 5))+3 comes from
            # 2 * (current_rgb_set=3, current events with 5 bins=5) + real/fake rgb image
            nn.Conv2d( (2*(3 + 5))+3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d( 64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d( 128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d( 256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d( 512, 1, kernel_size=4, stride=1, padding=1),
        )

    def run_discriminator(self, example, rgb2concat, detach):
        return self.discriminator_network(_pack(example, rgb2concat, detach))

    def forward(self, example, rgb2concat, detach):
        return self.run_discriminator(example, rgb2concat, detach)