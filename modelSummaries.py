from timelens.fusion_network import Fusion as Generator
from timelens.Discriminator_network import Discriminator

from torchsummary import summary
import torch

def get_summaries():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = Generator().to(device)
    netD = Discriminator().to(device)

    summary(netG, (3, 448, 256))
    summary(netD, (3, 448, 256))

get_summaries()