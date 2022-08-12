import math
from wandb import wandb
from timelens.common import (
    FlowDataset,
    losses,
)
from timelens.warp_network import Warp
import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import wandb
from tqdm import tqdm
from torchmetrics import PeakSignalNoiseRatio

def train(left_imgs, middle_imgs, right_imgs, event_roots, matching_ts, args, device):

    flow_dset = FlowDataset.FlowDataset(left_imgs, middle_imgs, right_imgs, event_roots, matching_ts, args)

    train_size = math.floor(0.8* len(flow_dset))
    val_size = len(flow_dset) - train_size

    train_set, val_set = random_split(flow_dset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=10)

    wandb.init(project="nsff", entity="dave-dush")

    wandb.config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "starting_lr": args.lr,
    }

    warp_model = Warp()
    warp_model = nn.DataParallel(warp_model)
    warp_model.to(device)

    optimizer = torch.optim.Adam(warp_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
    warp_loss = nn.L1Loss()
    # psnr per epoch
    psnr_ep = list()
    
    #change criterion to warp loss
    wandb.watch(warp_model, log_freq=50)

    for epoch in tqdm(range(args.epochs)):
        running_train_loss = list()
    
        warp_model.train(True)
        for i, (trainable_features, left_ev_tensors, right_ev_tensors, targets) in enumerate(tqdm(train_loader)):        

            targets = targets.to(device)

            optimizer.zero_grad()

            warped_items = warp_model(trainable_features)

            #split into forward and backward loss
            #combine them 
            bwd_loss = warp_loss(warped_items[0], targets)
            fwd_loss = warp_loss(warped_items[1], targets)

            train_loss = fwd_loss + bwd_loss
            train_loss.backward()

            optimizer.step()
            running_train_loss.append(train_loss.item())
            
            wandb.log({"Running Train loss": train_loss})
            if (i % 25) == 0:
                #change logging items
                wandb.log({"targets": wandb.Image(targets), 

                            "backward warp": wandb.Image(warped_items[0]),
                            "forward warp": wandb.Image(warped_items[1]),

                            "backward flow": wandb.Image(warped_items[2]),
                            "forward flow": wandb.Image(warped_items[3]),

                            # "bwd mask": wandb.Image(warped_items[4]),
                            # "fwd mask": wandb.Image(warped_items[5]),

                            "left imgs": wandb.Image(trainable_features["before"]["rgb_image_tensor"]),
                            "right imgs": wandb.Image(trainable_features["after"]["rgb_image_tensor"]),

                            "left evs": wandb.Image(left_ev_tensors),
                            "right evs": wandb.Image(right_ev_tensors)})
     
        avg_training_loss = torch.tensor(running_train_loss).mean()

        #validation loop
        running_val_losses = list()
        running_psnr = list()
        
        warp_model.eval()
        with torch.no_grad():
            for val_i, (val_features, val_left_ev_tensors, val_right_ev_tensors, val_targets) in enumerate(tqdm(val_loader)):

                val_targets = val_targets.to(device)

                val_warped_items = warp_model(val_features)

                val_bwd_loss = warp_loss(val_warped_items[0], val_targets)
                val_fwd_loss = warp_loss(val_warped_items[1], val_targets)

                val_loss = val_bwd_loss + val_fwd_loss
                
                running_val_losses.append(val_loss)
                
                psnr = PeakSignalNoiseRatio().to(device)
                bwd_psnr_score = psnr(val_warped_items[0], val_targets)
                fwd_psnr_score = psnr(val_warped_items[1], val_targets)
                psnr_score = torch.tensor((bwd_psnr_score + fwd_psnr_score)).mean()

                running_psnr.append(psnr_score)
                

                wandb.log({"Running Val loss": val_loss.item()})

                if (val_i % 25) == 0:
                    wandb.log({"val targets": wandb.Image(val_targets), 
                                "val bwd warp": wandb.Image(val_warped_items[0]),
                                "val fwd warp": wandb.Image(val_warped_items[1]),

                                "val bwd flow": wandb.Image(val_warped_items[2]),
                                "val fwd flow": wandb.Image(val_warped_items[3]),

                                # "val bwd mask": wandb.Image(val_warped_items[4]),
                                # "val fwd mask": wandb.Image(val_warped_items[5]),

                                "val left imgs": wandb.Image(val_features["before"]["rgb_image_tensor"]),
                                "val right imgs": wandb.Image(val_features["after"]["rgb_image_tensor"]),
                                
                                "val left evs": wandb.Image(val_left_ev_tensors),
                                "val right evs": wandb.Image(val_right_ev_tensors)})


            avg_val_loss = torch.tensor(running_val_losses).mean()
            #print(f"Train loss: {avg_training_loss}, Val loss: {avg_val_loss}")
            avg_psnr = torch.tensor(running_psnr).mean()
            #tb.add_scalar("PSNR", avg_psnr, epoch)

            psnr_ep.append(avg_psnr)
            print(f"PSNR: {avg_psnr}")

            if avg_psnr >= max(psnr_ep):
                print(f"Model at epoch {epoch + 1} has better PSNR!")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": warp_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dir": scheduler.state_dict(),
                }, args.model_path)
                print(f"Model saved at {args.model_path}")

            # tb.add_scalars("Training vs Val", {
            #     "Training loss": avg_training_loss,
            #     "Validation loss": avg_val_loss
            # }, epoch)

        #print(f"Epoch {epoch + 1}, training loss {avg_training_loss}, val loss {avg_val_loss}, psnr {avg_psnr}")
        wandb.log({"epoch": epoch+1, "training loss": avg_training_loss, "val loss": avg_val_loss, "psnr": avg_psnr})
        scheduler.step()
    # tb.flush()
    # tb.close()

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

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = config_parse()
    
    equal_dir_txt = args.equal_dir_txt
    total_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    dset_size = args.dataset_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Getting appropriate dirs")

    #get dir list from txt file using with and not np
    dir_list = []
    with open(equal_dir_txt, "r") as _eq:
        _dir = [line.strip() for line in _eq]
        dir_list.extend(_dir)
    
    dir_list = dir_list[:dset_size]

    left_imgs, middle_imgs, right_imgs, event_roots, matching_ts = dirs_to_paths(dir_list)
   
    train(left_imgs, middle_imgs, right_imgs, event_roots, matching_ts, args, device)
