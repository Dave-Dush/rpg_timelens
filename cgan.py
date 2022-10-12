import os
from tqdm import tqdm
import numpy as np
import math

import torch
#from torchvision import utils
from torch.utils.data import DataLoader
import torchmetrics
# from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
import lpips
from wandb import wandb

from timelens.common import(
    TimelensVimeoDataset,
)
from timelens.common.losses import GANLoss
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
    wandb.init(project="gan_setup", entity="dave-dush", dir=args.wandb_path, config={
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "starting_lr": args.lr,
    })

    #wandb_name = wandb.run.name

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
    # gan_criterion = torch.nn.BCEWithLogitsLoss()
    gan_criterion = GANLoss('vanilla').to(device)
    #####################
    psnr = torchmetrics.PeakSignalNoiseRatio().to(device)

    #####################
    # Optimizers
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Scheduler
    genScheduler = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=3, gamma=0.1)
    disScheduler = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=3, gamma=0.1)
    #####################


    ####################
    # Data prep
    print("preparing data")

    timelens_dset = TimelensVimeoDataset.TimelensVimeoDataset(left_imgs, middle_imgs, right_imgs, event_roots, matching_ts, args)
    timelens_loader = DataLoader(timelens_dset, batch_size=args.batch_size, shuffle=False, num_workers=10, pin_memory=True)
    ####################

    #draw_models(timelens_loader, writer, netD, netG)


    wandb.watch(models= netG, criterion=lpips_loss, log= "gradients", log_freq=10)

    print("training")
    for epoch in tqdm(range(args.epochs)):
        for i, (trainable_features, left_ev_tensors, right_ev_tensors, targets) in enumerate(timelens_loader):
            print(f"{i+1}/{len(timelens_loader)}")
            
            # 1 generate fake image from generator
            fake = netG(trainable_features).to("cpu")

            # 2 update D
            # 2.1 enable backprop for D
            for param in netD.parameters():
                param.requires_grad = True
            # 2.2 Flush gradients fo optimD
            optimizerD.zero_grad()
            # 2.3 Discriminator prediction on fake 
            pred_fake = netD(trainable_features, fake, detach=True)
            # 2.4 GAN criterion Discriminator BCE with logit loss with fake data
            loss_D_fake = gan_criterion(pred_fake, False)
            # 2.5 Discriminator prediction on real (targets)
            pred_real = netD(trainable_features, targets, detach=False)
            # 2.6 GAN criterion Discriminator BCE with logit loss with real data
            loss_D_real = gan_criterion(pred_real, True)
            # 2.7 Combined loss is averaged and passed through backward()
            lossD = 0.5 * (loss_D_fake + loss_D_real)
            lossD.backward()
            # 2.8 Optimizer step on D
            optimizerD.step()

            # 3 update G
            # 3.1 disable backprop for D when training G
            for param in netD.parameters():
                param.requires_grad = False

            # 3.2 Flush gradients for optimG
            optimizerG.zero_grad()

            # 3.3 Get fake prediction from updated D
            pred_fake = netD(trainable_features, fake, detach=False)
            # 3.4 GAN loss for G
            loss_G_GAN = gan_criterion(pred_fake, True)
            
            fake = fake.to(device)
            targets = targets.to(device)
            
            # 3.5 L1 loss for G
            loss_G_L1 = l1_loss(fake, targets)
            # 3.5.1 skipping LPIPS loss
            loss_G_Lpips = (lpips_loss(fake, targets).view(-1)).mean()
            # 3.6 Combined G loss 
            lossG = loss_G_GAN + (args.lambda_l1*loss_G_L1) + (args.lambda_lpips*loss_G_Lpips)
            lossG.backward()

            # 3.7 Optimizer step for G
            optimizerG.step()


            ##########################
            # Logging metrics
            if i%30 == 0:
                print("[%d]/[%d] [%d]/[%d]" %(epoch+1, args.epochs, i+1, len(timelens_loader)))

                psnr_score = psnr(fake, targets)
                wandb.log({
                    "fake": wandb.Image(fake),
                    "target": wandb.Image(targets),
                    "psnr": psnr_score.item(),
                })

            wandb.log({
                "lossG": lossG.item(),
                "lossD": lossD.item(),
                "l1_loss": loss_G_L1.item(),
                "lpips": loss_G_Lpips.item(),
            })
            #########################
        wandb.log({
            "Epoch": epoch+1,
        })
        genScheduler.step()
        disScheduler.step()

###################################



###################################
# Data Utils
def dirs_to_paths(seq_dirs):
    """
    Return path dictionary that contains left, middle and right structure
    """
    max_interpolations = len(seq_dirs) * 9 #9 triplets

    event_roots = list()

    left_imgs = list()
    middle_imgs = list()
    right_imgs = list()
    matching_ts = list()

    for m_i in tqdm(range(0, max_interpolations, 9)):
        dir_idx = math.floor(m_i/9)

        _seq_dir = seq_dirs[dir_idx]
        _ts = list()

        with open(os.path.join(_seq_dir, "upsampled/imgs/timestamp.txt"), "r") as _f:
            _temp_ts = [float(line.strip()) for line in _f]
            _ts.extend(_temp_ts)

        t_start = _ts[0]
        t_end = _ts[-1]
        
        _dt = np.linspace(t_start, t_end, 19)

        _event_root = os.path.join(_seq_dir, "events/")
        event_roots.append(_event_root)

        #9 triplets
        for in_i in range(9):

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

    parser.add_argument("--lambda_adv", type=float)

    parser.add_argument("--lambda_l1", type=float)

    parser.add_argument("--lambda_lpips", type=float)

    parser.add_argument("--wandb_path", type=str)

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
    #print(left_imgs, "\n", middle_imgs, "\n", right_imgs, "\n", matching_ts)
    train(left_imgs, middle_imgs, right_imgs, event_roots, matching_ts, args, device)
