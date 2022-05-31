#import os
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
"""
Convert definitions to arg flags
"""
root_image_folder = "/usr/stud/dave/storage/user/dave/minivimeo_septuplet/00003_0003/upsampled/imgs/"
root_event_folder = "/usr/stud/dave/storage/user/dave/minivimeo_septuplet/00003_0003/events_copy/"

wandb.init(project="timelens", entity="dave-dush")

transform = transformers.initialize_transformers()

nb_epochs = 40
batch_size = 4
starting_lr = 1e-4

# Hyperparameters
wandb.config = {
    "nb_epochs": nb_epochs,
    "batch_size": batch_size,
    "starting_lr": starting_lr,
}


ergb_dset = HybridDataset.HybridDataset(root_image_folder,root_event_folder,3, transform)

train_size = math.floor(0.7*len(ergb_dset))
val_size = math.ceil(0.2*len(ergb_dset))
test_size = len(ergb_dset) - train_size - val_size

train_dset = torch.utils.data.Subset(ergb_dset, range(train_size))
val_dset = torch.utils.data.Subset(ergb_dset, range(train_size, train_size+val_size))
test_dset = torch.utils.data.Subset(ergb_dset, range(train_size+val_size, train_size+val_size+test_size))

train_ergb_loader = DataLoader(dataset=train_dset, batch_size=batch_size, shuffle=False, num_workers=0)
val_ergb_loader = DataLoader(dataset=val_dset, batch_size=batch_size, shuffle=False, num_workers=0)
# test_ergb_loader = DataLoader(dataset=test_dset, batch_size=batch_size, shuffle=False, num_workers=0)


nb_iterations = math.ceil(train_size/batch_size)

device = torch.device("cuda")
fusion_model = Fusion()
fusion_model.to(device)

optimizer = torch.optim.Adam(fusion_model.parameters(), lr=starting_lr)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
fusion_loss_fn = losses.FusionLoss(device)

for epoch in range(nb_epochs):
    losses = list()
    for i, (trainable_features, targets) in enumerate(train_ergb_loader):
        trainable_features = trainable_features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        trainable_features = torch.squeeze(trainable_features)
        

        synthesized_img = fusion_model(trainable_features)
        synthesized_img = synthesized_img.to(device)

        train_loss = fusion_loss_fn(synthesized_img, targets)
        train_loss.backward()

        optimizer.step()
        losses.append(train_loss.item())
        
    # scheduler.step()
    # running_lr=optimizer.param_groups[0]["lr"]
    training_loss=torch.tensor(losses).mean()
    # print(f"Epoch {epoch+1}/{nb_epochs}, learning rate {running_lr}, training loss {torch.tensor(losses).mean():.5f}")

    val_losses = list()
    for i, (val_features, val_targets) in enumerate(val_ergb_loader):
        val_features = val_features.to(device)
        val_targets = val_targets.to(device)

        val_features = torch.squeeze(val_features)
        
        with torch.no_grad():
            val_synthesized_img = fusion_model(val_features)
            val_synthesized_img = val_synthesized_img.to(device)

        val_loss = fusion_loss_fn(val_synthesized_img, val_targets)
        val_losses.append(val_loss)
    avg_val_loss = torch.tensor(val_losses).mean()
    
    wandb.log({"epoch": epoch+1, "training loss": training_loss, "val loss": avg_val_loss, \
        "targets": wandb.Image(targets), "logits": wandb.Image(synthesized_img), \
        "val_targets": wandb.Image(val_targets), "val logits": wandb.Image(val_synthesized_img)})
