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
from torchvision.utils import save_image
from tqdm import tqdm
from torchmetrics import PeakSignalNoiseRatio

def test(test_dirs, args, device):
    transform = transformers.initialize_transformers()

    infer_path = args.infer_path
    
    test_set = TimelensVimeoDataset.TimelensVimeoDataset(seq_dirs= test_dirs, skip_scheme=3, transform=transform, mode="Test")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = Fusion()

    model_checkpoint = torch.load(args.model_path)
    model.load_state_dict(model_checkpoint["model_state_dict"])
    model.to(device)
    
    with torch.no_grad():
        model.eval()
        fusion_loss_fn = losses.FusionLoss(device)

        running_test_losses = list()
        running_test_psnr = list()
        psnr = PeakSignalNoiseRatio().to(device)

        for outer_i, (test_features, test_targets) in enumerate(tqdm(test_loader)):
                        
            test_features = test_features.to(device)
            test_targets = test_targets.to(device)

            if test_features.shape[0] != 1:
                test_features = torch.squeeze(test_features)
            else:
                test_features = test_features.view(1, 16, 256, 448)

            test_synthesized_img = model(test_features)
            test_synthesized_img = test_synthesized_img.to(device)

            test_loss = fusion_loss_fn(test_synthesized_img, test_targets)
            running_test_losses.append(test_loss)
            
            for batch_i in range(test_synthesized_img.size(0)):
                save_image(test_synthesized_img[batch_i,:,:,:], f"{infer_path}/{outer_i}_{batch_i}.png")
           
            psnr_score = psnr(test_synthesized_img, test_targets)
            running_test_psnr.append(psnr_score)

        avg_test_loss = torch.tensor(running_test_losses).mean()
        avg_test_psnr = torch.tensor(running_test_psnr).mean()

        print(f"Average Test loss: {avg_test_loss}")
        print(f"Average test PSNR {avg_test_psnr}")

def config_parse():
    import configargparse

    parser = configargparse.ArgumentParser()

    parser.add_argument("--config", is_config_file=True)

    parser.add_argument("--dataset_root", type=str)

    parser.add_argument("--model_path", type=str)

    parser.add_argument("--infer_path", type=str)

    parser.add_argument("--batch_size", type=int)

    parser.add_argument("--dataset_size", type=int)

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = config_parse()
    
    base_dir = args.dataset_root
    dset_size = args.dataset_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #print("Getting appropriate dirs")
    dir_list = glob(os.path.join(base_dir, "*/*"), recursive=True)
    dir_list.sort()
    train_val_dir_list = dir_list[:dset_size]

    infer_dirs = list(set(dir_list) - set(train_val_dir_list))
    infer_dirs.sort()

    #print(infer_dirs[-2:-1])
    test(infer_dirs[-6:-5], args, device)