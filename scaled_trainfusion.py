import math
import os
from glob import glob
from timelens.common import (
    TimelensVimeoDataset,
    transformers,
    losses,
)
from timelens.fusion_network import Fusion
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from torchmetrics import PeakSignalNoiseRatio
# from ignite.metrics import PSNR
# from ignite.engine import *

def train(train_dirs, val_dirs, args, device):

    wandb.init(project="timelens", entity="dave-dush")

    wandb.config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "starting_lr": args.lr,
    }


    # def eval_step(engine, batch):
    #     return batch

    # default_evaluator = Engine(eval_step)

    transform = transformers.initialize_transformers()

    train_set = TimelensVimeoDataset.TimelensVimeoDataset(seq_dirs= train_dirs, skip_scheme=3, transform=transform, mode="Train")
    val_set = TimelensVimeoDataset.TimelensVimeoDataset(seq_dirs=val_dirs, skip_scheme=3, transform=transform, mode="Val")

    

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=10, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=10, pin_memory=True)


    fusion_model = Fusion()
    fusion_model.to(device)

    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
    fusion_loss_fn = losses.FusionLoss(device)

    # psnr per epoch
    psnr_ep = list()
    for epoch in tqdm(range(args.epochs)):
        running_train_loss = list()
    
        #print(f"Training at epoch {epoch+1}")
        # training loop
        for i, (trainable_features, targets) in enumerate(tqdm(train_loader)):
            #wandb.watch(fusion_model, fusion_loss_fn, log="all", log_freq=5)
            targets = targets.to(device)

            optimizer.zero_grad()

            # to prevent batch collapsing when features batch have only one image
            if trainable_features.shape[0] != 1:
                trainable_features = torch.squeeze(trainable_features)
            else:
                trainable_features = trainable_features.view(1, 16, 256, 448)
            
            trainable_features = trainable_features.to(device)

            synthesized_img = fusion_model(trainable_features)
            synthesized_img = synthesized_img.to(device)

            train_loss = fusion_loss_fn(synthesized_img, targets)
            train_loss.backward()

            optimizer.step()
            running_train_loss.append(train_loss)

        avg_training_loss = torch.tensor(running_train_loss).mean()
        wandb.log({"targets": wandb.Image(targets[:3]), "logits": wandb.Image(synthesized_img[:3])})
        # validation loop

        #print(f"Validating at epoch {epoch+1}")
        running_val_losses = list()
        running_psnr = list()
        with torch.no_grad():
            fusion_model.to(device)
            fusion_model.eval()
            for i, (val_features, val_targets) in enumerate(tqdm(val_loader)):
                
                val_features = val_features.to(device)
                val_targets = val_targets.to(device)

                if val_features.shape[0] != 1:
                    val_features = torch.squeeze(val_features)
                else:
                    val_features = val_features.view(1, 16, 256, 448)

                val_synthesized_img = fusion_model(val_features)
                val_synthesized_img = val_synthesized_img.to(device)

                val_loss = fusion_loss_fn(val_synthesized_img, val_targets)
                running_val_losses.append(val_loss)
                psnr = PeakSignalNoiseRatio().to(device)
                psnr_score = psnr(val_synthesized_img, val_targets)
                # psnr = PSNR(data_range=1.0)
                # psnr.attach(default_evaluator, 'psnr')

                # state = default_evaluator.run([[val_synthesized_img, val_targets]])
                #print(f"PSNR at batch {i}: {state.metrics['psnr']}")
                # running_psnr.append(state.metrics['psnr'])
                running_psnr.append(psnr_score)
        wandb.log({"val_targets": wandb.Image(val_targets[:3]), "val logits": wandb.Image(val_synthesized_img[:3])})

        avg_val_loss = torch.tensor(running_val_losses).mean()
        avg_psnr = torch.tensor(running_psnr).mean()
        psnr_ep.append(avg_psnr)


        if avg_psnr >= max(psnr_ep):
            print(f"Model at epoch {epoch + 1} has better PSNR!")
            torch.save({
                "epoch": epoch,
                "model_state_dict": fusion_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dir": scheduler.state_dict(),
            }, args.model_path)
            print(f"Model saved at {args.model_path}")

        # if epoch > 0:
        #     if psnr_ep[epoch -1] < psnr_ep[epoch]:
        #         print(f"Model at epoch {epoch + 1} has better PSNR!")
        #         torch.save({
        #             "epoch": epoch,
        #             "model_state_dict": fusion_model.state_dict(),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "scheduler_state_dir": scheduler.state_dict(),
        #         }, args.model_path)
        #         print(f"Model saved at {args.model_path}")

        print(f"Epoch {epoch + 1}, training loss {avg_training_loss}, val loss {avg_val_loss}, psnr {avg_psnr}")
        wandb.log({"epoch": epoch+1, "training loss": avg_training_loss, "val loss": avg_val_loss, "psnr": avg_psnr})

        scheduler.step()

        # wandb.log({"epoch": epoch+1, "training loss": avg_training_loss, "val loss": avg_val_loss, \
        #     "targets": wandb.Image(targets[:3]), "logits": wandb.Image(synthesized_img[:3]), \
        #     "val_targets": wandb.Image(val_targets[:3]), "val logits": wandb.Image(val_synthesized_img[:3])})



def config_parse():
    import configargparse

    parser = configargparse.ArgumentParser()

    parser.add_argument("--config", is_config_file=True)

    parser.add_argument("--dataset_root", type=str)

    parser.add_argument("--model_path", type=str)

    parser.add_argument("--epochs", type=int)

    parser.add_argument("--batch_size", type=int)

    parser.add_argument("--lr", type=float)

    parser.add_argument("--dataset_size", type=int)

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = config_parse()
    
    base_dir = args.dataset_root
    total_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    dset_size = args.dataset_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #print("Getting appropriate dirs")
    dir_list = glob(os.path.join(base_dir, "*/*"), recursive=True)
    dir_list.sort()
    dir_list = dir_list[:dset_size]
    #print(f"\nDirs like {dir_list[:3]}")

    train_size = math.floor(0.6*(len(dir_list)))
    val_size = math.floor(0.2*(len(dir_list)))
    test_size = math.floor(0.2*(len(dir_list)))

    train_dirs = dir_list[:train_size]
    val_dirs = dir_list[train_size : (train_size + val_size)]
    test_dirs = dir_list[(train_size + val_size): ]

    
    #print("\nPerforming training...")
    train(train_dirs, val_dirs, args, device)
