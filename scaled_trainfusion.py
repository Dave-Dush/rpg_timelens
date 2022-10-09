import math
from wandb import wandb
from timelens.common import (
    TimelensVimeoDataset,
    losses,
)
from timelens.fusion_network import Fusion
import torchvision
import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import wandb
from tqdm import tqdm
from torchmetrics import PeakSignalNoiseRatio

def train(left_imgs, middle_imgs, right_imgs, event_roots, matching_ts, args, device):

    timelens_dset = TimelensVimeoDataset.TimelensVimeoDataset(left_imgs, middle_imgs, right_imgs, event_roots, matching_ts, args)

    train_size = math.floor(0.8* len(timelens_dset))
    val_size = len(timelens_dset) - train_size

    train_set, val_set = random_split(timelens_dset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=10)


    # #tb = SummaryWriter()

    wandb.init(project="timelens", entity="dave-dush")

    wandb.config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "starting_lr": args.lr,
    }

    # img_height = args.height
    # img_width = args.width

    # timelens_dset = TimelensVimeoDataset.TimelensVimeoDataset(seq_dirs=dir_list, device=device, width=img_width, height=img_height)
    # train_size = math.floor(0.8* len(timelens_dset))
    # val_size = len(timelens_dset) - train_size

    # train_set, val_set = random_split(timelens_dset, [train_size, val_size])

    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=10, pin_memory=True)
    # val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=10, pin_memory=True)

    fusion_model = Fusion()
    fusion_model = nn.DataParallel(fusion_model)
    fusion_model.to(device)

    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
    fusion_loss_fn = losses.FusionLoss(device)
    # psnr per epoch
    psnr_ep = list()
    for epoch in tqdm(range(args.epochs)):
        running_train_loss = list()
    
        fusion_model.train(True)
        for i, (trainable_features, left_ev_tensors, right_ev_tensors, targets) in enumerate(tqdm(train_loader)):        

            targets = targets.to(device)

            optimizer.zero_grad()

            synthesized_img = fusion_model(trainable_features)

            train_loss = fusion_loss_fn(synthesized_img, targets)
            train_loss.backward()

            optimizer.step()
            running_train_loss.append(train_loss)
            
            avg_batch_loss = train_loss.mean() 
            # tb.add_scalars("Training Loss",
            #             {"Training": avg_batch_loss}, epoch*len(train_loader) + i) 

            wandb.log({"Running Train loss": avg_batch_loss})
            if (i % 10) == 0:
                wandb.log({"targets": wandb.Image(targets), 
                            "logits": wandb.Image(synthesized_img),
                            "left imgs": wandb.Image(trainable_features["before"]["rgb_image_tensor"]),
                            "right imgs": wandb.Image(trainable_features["after"]["rgb_image_tensor"]),
                            "left evs": wandb.Image(left_ev_tensors),
                            "right evs": wandb.Image(right_ev_tensors)})
                
                # left_grid = torchvision.utils.make_grid(trainable_features["before"]["rgb_image_tensor"])
                # right_grid = torchvision.utils.make_grid(trainable_features["after"]["rgb_image_tensor"])                
                # tb.add_images("Left and Right images",
                #             {"Left": left_grid, "Right": right_grid},
                #             epoch*len(train_loader) + i)
                #wandb.log({"targets": wandb.Image(targets), "logits": wandb.Image(synthesized_img)})
                #pass
                #log images and events in grid
     
        avg_training_loss = torch.tensor(running_train_loss).mean()

        # validation loop
        #print(f"Validating at epoch {epoch+1}")
        running_val_losses = list()
        running_psnr = list()
        
        fusion_model.eval()
        with torch.no_grad():
            for i, (val_features, val_left_ev_tensors, val_right_ev_tensors, val_targets) in enumerate(tqdm(val_loader)):

                val_targets = val_targets.to(device)

                # left_ev_seqs = event.EventSequence(left_ev_features, 256, 448)
                # right_ev_seqs = event.EventSequence(right_ev_features, 256, 448)

                val_synthesized_img = fusion_model(val_features)

                val_loss = fusion_loss_fn(val_synthesized_img, val_targets)
                running_val_losses.append(val_loss)
                psnr = PeakSignalNoiseRatio().to(device)
                psnr_score = psnr(val_synthesized_img, val_targets)
                running_psnr.append(psnr_score)
                
                avg_val_batch_loss = val_loss.mean()
                # tb.add_scalars("Val Loss",
                #             {"Val": avg_val_batch_loss}, epoch*len(val_loader) + i) 

                wandb.log({"Running Val loss": avg_val_batch_loss})
                  

                if (i % 10) == 0:
                    wandb.log({"val targets": wandb.Image(val_targets), 
                                "val logits": wandb.Image(val_synthesized_img),
                                "val left imgs": wandb.Image(val_features["before"]["rgb_image_tensor"]),
                                "val right imgs": wandb.Image(val_features["after"]["rgb_image_tensor"]),
                                "val left evs": wandb.Image(val_left_ev_tensors),
                                "val right evs": wandb.Image(val_right_ev_tensors)})
                    #wandb.log({"Val targets": wandb.Image(val_targets), "Val logits": wandb.Image(val_synthesized_img)})
                    # val_left_grid = torchvision.utils.make_grid(val_features["before"]["rgb_image_tensor"])
                    # val_right_grid = torchvision.utils.make_grid(val_features["after"]["rgb_image_tensor"])                
                    # tb.add_images("Val Left and Right images",
                    #             {"Left": val_left_grid, "Right": val_right_grid},
                    #             epoch*len(val_loader) + i)
                    # val_tar_grid = torchvision.utils.make_grid(targets)
                    # val_log_grid = torchvision.utils.make_grid(synthesized_img)


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
                    "model_state_dict": fusion_model.state_dict(),
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

            # _left_event = _event_seq.filter_by_timestamp(_left_t, _middle_t)
            # left_evs.append(_left_event)

            # _right_event = _event_seq.filter_by_timestamp(_middle_t, _right_t)
            # right_evs.append(_right_event)

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

            # _id_dict = {
            #     "left": left_id,
            #     "middle": middle_id,
            #     "right": right_id
            # }  

            

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
   
    # print(matching_ts)
    # print(f"\nFirst Dirs like {dir_list[:3]}")
    # print(f"\nLast Dirs like {dir_list[-3:]}")
    
    #print("\nPerforming training...")
    train(left_imgs, middle_imgs, right_imgs, event_roots, matching_ts, args, device)
