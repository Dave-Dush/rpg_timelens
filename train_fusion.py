import os
from timelens.common import (
    HybridDataset,
    hybrid_storage,
    transformers,
    pytorch_tools,
    os_tools,
    losses,
)
import torch
from timelens.fusion_network import Fusion
import math
from torch.utils.data import DataLoader
import torchvision.transforms as pt_transforms
import wandb
import lpips
import cv2
import glob

"""
Convert definitions to arg flags
"""

# for timelens_vimeo_dir in timelens_vimeo_dirs:
#     pass

# root_image_folder = "/usr/stud/dave/storage/user/dave/minivimeo_septuplet/00003_0003/upsampled/imgs/"
# root_event_folder = "/usr/stud/dave/storage/user/dave/minivimeo_septuplet/00003_0003/events_copy/"

# wandb.init(project="timelens", entity="dave-dush")

# transform = transformers.initialize_transformers()

# nb_epochs = 40
# batch_size = 4
# starting_lr = 1e-4

# # Hyperparameters
# wandb.config = {
#     "nb_epochs": nb_epochs,
#     "batch_size": batch_size,
#     "starting_lr": starting_lr,
# }


# ergb_dset = HybridDataset.HybridDataset(root_image_folder,root_event_folder,3, transform)

# train_size = math.floor(0.7*len(ergb_dset))
# val_size = math.ceil(0.2*len(ergb_dset))
# test_size = len(ergb_dset) - train_size - val_size

# train_dset = torch.utils.data.Subset(ergb_dset, range(train_size))
# val_dset = torch.utils.data.Subset(ergb_dset, range(train_size, train_size+val_size))
# test_dset = torch.utils.data.Subset(ergb_dset, range(train_size+val_size, train_size+val_size+test_size))

# train_ergb_loader = DataLoader(dataset=train_dset, batch_size=batch_size, shuffle=False, num_workers=0)
# val_ergb_loader = DataLoader(dataset=val_dset, batch_size=batch_size, shuffle=False, num_workers=0)
# # test_ergb_loader = DataLoader(dataset=test_dset, batch_size=batch_size, shuffle=False, num_workers=0)


# nb_iterations = math.ceil(train_size/batch_size)

# device = torch.device("cuda")
# fusion_model = Fusion()
# fusion_model.to(device)

# optimizer = torch.optim.Adam(fusion_model.parameters(), lr=starting_lr)
# #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# fusion_loss_fn = losses.FusionLoss(device)

# for epoch in range(nb_epochs):
#     losses = list()
#     for i, (trainable_features, targets) in enumerate(train_ergb_loader):
#         fusion_model.train(True)
#         trainable_features = trainable_features.to(device)
#         targets = targets.to(device)

#         optimizer.zero_grad()

#         trainable_features = torch.squeeze(trainable_features)
        

#         synthesized_img = fusion_model(trainable_features)
#         synthesized_img = synthesized_img.to(device)

#         train_loss = fusion_loss_fn(synthesized_img, targets)
#         train_loss.backward()

#         optimizer.step()
#         losses.append(train_loss.item())
        
#     # scheduler.step()
#     # running_lr=optimizer.param_groups[0]["lr"]
#     avg_training_loss=torch.tensor(losses).mean()
#     # print(f"Epoch {epoch+1}/{nb_epochs}, learning rate {running_lr}, training loss {torch.tensor(losses).mean():.5f}")

#     val_losses = list()
#     with torch.no_grad():
#         fusion_model.eval()
#         for i, (val_features, val_targets) in enumerate(val_ergb_loader):
            
#             val_features = val_features.to(device)
#             val_targets = val_targets.to(device)

#             val_features = torch.squeeze(val_features)

#             val_synthesized_img = fusion_model(val_features)
#             val_synthesized_img = val_synthesized_img.to(device)

#             val_loss = fusion_loss_fn(val_synthesized_img, val_targets)
#             val_losses.append(val_loss)
#     avg_val_loss = torch.tensor(val_losses).mean()
    
#     wandb.log({"epoch": epoch+1, "training loss": avg_training_loss, "val loss": avg_val_loss, \
#         "targets": wandb.Image(targets), "logits": wandb.Image(synthesized_img), \
#         "val_targets": wandb.Image(val_targets), "val logits": wandb.Image(val_synthesized_img)})

def config_parser():
    import configargparse

    parser = configargparse.ArgumentParser()

    parser.add_argument("--config", is_config_file=True, help="config file path")

    parser.add_argument("--dataset_root", type=str, help="path to dataset")

    parser.add_argument("--epochs", type=int, help="total epochs to train the model")

    parser.add_argument("--batch_size", type=int, help="keep batch size smaller around 4")

    parser.add_argument("--lr", type=float, help="remains same if not using lr decay mechanism")

    return parser

def train(args, train_dirs, val_dirs):

    base_dir = args.dataset_root
    total_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    transform = transformers.initialize_transformers()

    training_set = list()
    validation_set = list()
    
    for train_dir in train_dirs:
        
        
    # wandb.init(project="timelens", entity="dave-dush")

    # wandb.config = {
    # "total_epochs": total_epochs,
    # "batch_size": batch_size,
    # "lr": lr,
    # }
    
    # transform = transformers.initialize_transformers()

    # device = torch.device("cuda")
    # fusion_model = Fusion()
    # fusion_model.to(device)

    # optimizer = torch.optim.Adam(fusion_model.parameters(), lr=lr)
    # fusion_loss_fn = losses.FusionLoss(device)

    # for epoch in range(total_epochs):
    #     running_train_loss = list()
        
    #     for current_dir in total_dirs:
    #         img_dir = os.path.join(current_dir, "upsampled/imgs/")
    #         eve_dir = os.path.join(current_dir, "events/")

    #         current_dset = HybridDataset.HybridDataset(img_dir, eve_dir, skip_scheme=3, transform=transform)

    #         train_size = math.floor(0.7*len(current_dset))
    #         val_size = math.floor(0.3*len(current_dset))
    #         #test_size = len(current_dset) - train_size - val_size

    #         train_dset = torch.utils.data.Subset(current_dset, range(train_size))
    #         val_dset = torch.utils.data.Subset(current_dset, range(train_size, train_size + val_size))

    #         train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=False, num_workers=0)
    #         val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False, num_workers=0)
            

    #         for idx, (trainable_features, targets) in enumerate(train_loader):
    #             fusion_model.train(True)
    #             trainable_features = trainable_features.to(device)
    #             targets = targets.to(device)

    #             optimizer.zero_grad()

    #             trainable_features = torch.squeeze(trainable_features)
                

    #             synthesized_img = fusion_model(trainable_features)
    #             synthesized_img = synthesized_img.to(device)

    #             train_loss = fusion_loss_fn(synthesized_img, targets)
    #             train_loss.backward()

    #             optimizer.step()
    #             running_train_loss.append(train_loss.item())

    #         avg_training_loss = torch.tensor(running_train_loss).mean()

    #         running_val_losses = list()
    #         with torch.no_grad():
    #             fusion_model.eval()
    #             for i, (val_features, val_targets) in enumerate(val_loader):
                    
    #                 val_features = val_features.to(device)
    #                 val_targets = val_targets.to(device)

    #                 if val_features.shape[0] != 1:
    #                     val_features = torch.squeeze(val_features)
    #                 else:
    #                     val_features = val_features.view(1, 16, 256, 448)

    #                 val_synthesized_img = fusion_model(val_features)
    #                 val_synthesized_img = val_synthesized_img.to(device)

    #                 val_loss = fusion_loss_fn(val_synthesized_img, val_targets)
    #                 running_val_losses.append(val_loss)
    #         avg_val_loss = torch.tensor(running_val_losses).mean()

    #         print(f"Epoch {epoch}, training loss {avg_training_loss}, {avg_val_loss}")
        
    #         # wandb.log({"epoch": epoch+1, "training loss": avg_training_loss, "val loss": avg_val_loss, \
    #         #     "targets": wandb.Image(targets), "logits": wandb.Image(synthesized_img), \
    #         #     "val_targets": wandb.Image(val_targets), "val logits": wandb.Image(val_synthesized_img)})
    #     break



if __name__ == "__main__":

    parser = config_parser()
    args = parser.parse_args()

    # TRAINING SCHEME 1: SPLIT DATA WITHIN DIRECTORIES
    # train(args)

    # TRAINING SCHEME 2: SPLIT DATA UPON ALL DIRECTORIES
    base_dir = args.dataset_root
    total_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    dir_list = glob.glob(os.path.join(base_dir, "*/*"), recursive=True)
    dir_list.sort()

    train_size = math.floor(0.6*(len(dir_list)))
    val_size = math.floor(0.2*(len(dir_list)))
    test_size = math.floor(0.2*(len(dir_list)))

    train_dirs = dir_list[:train_size]
    val_dirs = dir_list[train_size:(train_size + val_size)]
    test_dirs = dir_list[(train_size + val_size): ]

    train(args, train_dirs, val_dirs)



