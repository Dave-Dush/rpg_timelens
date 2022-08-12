import os
import numpy as np
from timelens.common import (
    FlowDataset,
    transformers,
    losses,
)
from train_warp import dirs_to_paths
from timelens.warp_network import Warp
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from torchmetrics import PeakSignalNoiseRatio

def test(left_imgs, middle_imgs, right_imgs, event_roots, matching_ts, args):

    infer_path = args.infer_path
    
    flow_dset = FlowDataset.FlowDataset(left_imgs, middle_imgs, right_imgs, event_roots, matching_ts, args)
    test_loader = DataLoader(flow_dset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    warp_model = Warp()

    warp_model_checkpoint = torch.load(args.model_path)
    warp_model.load_state_dict(warp_model_checkpoint["model_state_dict"])
    warp_model.to(device)
    warp_model.eval()

    warp_loss = torch.nn.L1Loss()

    with torch.no_grad():
        for val_i, (val_features, val_left_ev_tensors, val_right_ev_tensors, val_targets) in enumerate(tqdm(test_loader)):

            val_targets = val_targets.to(device)

            val_warped_items = warp_model(val_features)

            val_bwd_loss = warp_loss(val_warped_items[0], val_targets)
            val_fwd_loss = warp_loss(val_warped_items[1], val_targets)

            val_loss = val_bwd_loss + val_fwd_loss

            


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
    
    dir_list = dir_list[-1]

    left_imgs, middle_imgs, right_imgs, event_roots, matching_ts = dirs_to_paths([dir_list])
   
    #train(left_imgs, middle_imgs, right_imgs, event_roots, matching_ts, args, device)