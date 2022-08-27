import os
from tqdm import tqdm
import numpy as np
import math

import torch
#from torchvision import utils
from torch.utils.data import DataLoader
#import torchmetrics
# from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
import lpips
from wandb import wandb

from timelens.common import(
    TimelensVimeoDataset,
)
from timelens.fusion_network import Fusion as Generator
from timelens.Discriminator_network import Discriminator

###################################
# Get Model Graphs
def draw_models(timelens_loader, writer, netD, netG):
    trainable_features, left_ev_tensors, right_ev_tensors, targets = next(iter(timelens_loader))
    writer.add_graph(netD, (trainable_features, targets))
    writer.add_graph(netG, trainable_features)
    writer.close()
###################################


###################################
# Training

#*******
# Prep
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#*******

def train(left_imgs, middle_imgs, right_imgs, event_roots, matching_ts, args, device):

    #writer = SummaryWriter(args.summary_path)
    wandb.init(project="gan_setup", entity="dave-dush")

    wandb.config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "starting_lr": args.lr,
    }

    print("beginning setup")
    #################
    # Models
    netD = Discriminator().to(device)
    #netD = torch.nn.DataParallel(netD, [0,1,2])
    netD.apply(weights_init) # apply normally distributed weights with 0.02 std

    netG = Generator().to(device)
    #netG = torch.nn.DataParallel(netG, [0,1,2])
    netG.apply(weights_init) # apply normally distributed weights with 0.02 stds
    #################

    #####################
    # Loss functions
    l1_loss = torch.nn.L1Loss()
    lpips_loss = lpips.LPIPS("net=alex", verbose=False).to(device)
    bce_loss = torch.nn.BCELoss()
    #####################

    #####################
    # Optimizers
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Scheduler
    #genScheduler = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=30, gamma=0.1)
    #####################


    ####################
    # Data prep
    print("preparing data")
    real_label = 0
    fake_label = 1

    timelens_dset = TimelensVimeoDataset.TimelensVimeoDataset(left_imgs, middle_imgs, right_imgs, event_roots, matching_ts, args)
    timelens_loader = DataLoader(timelens_dset, batch_size=args.batch_size, shuffle=True, num_workers=10)
    ####################

    #draw_models(timelens_loader, writer, netD, netG)

    ####################
    # Loss stats
    G_losses = list()
    D_losses = list()
    ####################

    print("training")
    for epoch in tqdm(range(args.epochs)):
        print(f"{epoch+1}/{args.epochs}")
        for i, (trainable_features, left_ev_tensors, right_ev_tensors, targets) in enumerate(timelens_loader):
            print(f"{i+1}/{len(timelens_loader)}")
            # Train D against 'real' data
            netD.zero_grad()
            b_size = targets.size(0)
            gan_label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            
            output = netD(trainable_features, targets).view(-1)
            errD_real = bce_loss(output, gan_label)
            # calculate gradients for D
            errD_real.backward(retain_graph=True)
            D_x = output.mean().item()


            # Train G & D against 'fake' data
            fake = netG(trainable_features).to("cpu")
            gan_label.fill_(fake_label)
            output = netD(trainable_features, fake).view(-1)
            errD_fake = bce_loss(output, gan_label)

            # calculate gradients for fake batch (summed with errD_real grads)
            errD_fake.backward(retain_graph=True)
            D_G_z1 = output.mean().item()
            errD = errD_fake + errD_real
            # update D
            optimizerD.step()


            # Update G
            netG.zero_grad()
            gan_label.fill_(real_label)
            output = netD(trainable_features, fake).view(-1)
            fake = fake.to(device)
            targets = targets.to(device)

            # Loss for generator bce + l1 + lpips
            _advLoss = bce_loss(output, gan_label)

            rawL1Loss = l1_loss(fake, targets)
            _l1Loss = (args.lambda_l1 * rawL1Loss)

            rawPercepLoss = (lpips_loss(fake, targets).view(-1)).mean()
            _percepLoss = (args.lambda_lpips * rawPercepLoss)

            errG = _advLoss + _l1Loss + _percepLoss
            errG.backward()

            D_G_z2 = output.mean().item()
            optimizerG.step()

            ##########################
            # Logging metrics
            if i%2 == 0:
                print("[%d]/[%d] [%d]/[%d]\tLoss_D: %.4f\tLoss G: %.4f\tD(x): %.4f\tD(G(x)): %.4f / %.4f" %(epoch+1, args.epochs, i+1, len(timelens_loader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                wandb.log({
                    "fake": wandb.Image(fake),
                    "target": wandb.Image(targets),
                })

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            wandb.log({
                "errD": errD.item(),
                "errG": errG.item(),
                "l1 loss": rawL1Loss,
                "lpips loss": rawPercepLoss,
                "G adv Loss": _advLoss,
            })
            #########################
        #genScheduler.step()
###################################



###################################
# Data Utils
def dirs_to_paths(seq_dirs):
    """
    Return path dictionary that contains left, middle and right structure
    """
    max_interpolations = len(seq_dirs) * 3

    event_roots = list()

    left_imgs = list()
    middle_imgs = list()
    right_imgs = list()
    matching_ts = list()

    for m_i in tqdm(range(0, max_interpolations, 3)):
        dir_idx = math.floor(m_i/3)

        _seq_dir = seq_dirs[dir_idx]
        _ts = list()

        with open(os.path.join(_seq_dir, "upsampled/imgs/timestamp.txt"), "r") as _f:
            _temp_ts = [float(line.strip()) for line in _f]
            _ts.extend(_temp_ts)

        t_start = _ts[0]
        t_end = _ts[-1]
        
        _dt = np.linspace(t_start, t_end, 7)

        _event_root = os.path.join(_seq_dir, "events/")
        event_roots.append(_event_root)

        for in_i in range(3):

            _left_t = _dt[(2 * in_i)]
            _middle_t = _dt[(2 * in_i) + 1]
            _right_t = _dt[(2 * in_i) + 2]

            left_match = (np.abs(_ts - _left_t).argmin())
            middle_match = (np.abs(_ts - _middle_t).argmin())
            right_match = (np.abs(_ts - _right_t).argmin()) 

            _ts_dict = {
                "left": _left_t,
                "middle": _middle_t,
                "right": _right_t,
            }
            matching_ts.append(_ts_dict)
            
            left_id = f"{left_match:08d}"
            middle_id = f"{middle_match:08d}"
            right_id = f"{right_match:08d}" 

            left_path = os.path.join(_seq_dir, f"upsampled/imgs/{left_id}.png") 
            left_imgs.append(left_path)

            middle_path = os.path.join(_seq_dir, f"upsampled/imgs/{middle_id}.png") 
            middle_imgs.append(middle_path)

            right_path = os.path.join(_seq_dir, f"upsampled/imgs/{right_id}.png")  
            right_imgs.append(right_path)
    
    return left_imgs, middle_imgs, right_imgs, event_roots, matching_ts
def config_parse():
    import configargparse

    parser = configargparse.ArgumentParser()

    parser.add_argument("--config", is_config_file=True)

    parser.add_argument("--equal_dir_txt", type=str)

    parser.add_argument("--model_path", type=str)

    parser.add_argument("--width", type=int)

    parser.add_argument("--height", type=int)

    parser.add_argument("--epochs", type=int)

    parser.add_argument("--batch_size", type=int)

    parser.add_argument("--lr", type=float)

    parser.add_argument("--dataset_size", type=int)

    parser.add_argument("--lambda_l1", type=float)

    parser.add_argument("--lambda_lpips", type=float)

    args = parser.parse_args()

    return args
###################################

if __name__ == "__main__":
    args = config_parse()
    
    equal_dir_txt = args.equal_dir_txt
    dset_size = args.dataset_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Getting appropriate dirs")

    dir_list = []
    with open(equal_dir_txt, "r") as _eq:
        _dir = [line.strip() for line in _eq]
        dir_list.extend(_dir)
    
    dir_list = dir_list[:dset_size]

    left_imgs, middle_imgs, right_imgs, event_roots, matching_ts = dirs_to_paths(dir_list)

    train(left_imgs, middle_imgs, right_imgs, event_roots, matching_ts, args, device)
