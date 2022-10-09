import torch.nn as nn
import torch as th

def _pack(example, rgb2concat):
    ip_tensor =  th.cat([example['before']['voxel_grid'],
                   example['before']['rgb_image_tensor'],
                   example['after']['voxel_grid'],
                   example['after']['rgb_image_tensor'],
                   rgb2concat], dim=1)
    return ip_tensor.to(device="cuda")

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator_network = nn.Sequential(
            # (2*(3 + 5))+3 comes from
            # 2 * (current_rgb_set=3, current events with 5 bins=5) + real/fake rgb image
            nn.Conv2d( (2*(3 + 5))+3, 32, 4, 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d( 32, 64, 4, 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d( 64, 64, 4, 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d( 64, 1, (30,54), 2),
            nn.Sigmoid()
        )

    def run_discriminator(self, example, rgb2concat):
        return self.discriminator_network(_pack(example, rgb2concat))

    def forward(self, example, rgb2concat):
        return self.run_discriminator(example, rgb2concat)