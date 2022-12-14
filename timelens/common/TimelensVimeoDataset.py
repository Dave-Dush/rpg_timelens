from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as torch_transforms
from . import (event, representation)
from PIL import Image
import math


class TimelensVimeoDataset(Dataset):

    def __init__(
        self, left_imgs, middle_imgs, right_imgs, event_roots, matching_ts, args
    ):
        self.left_imgs_pths = left_imgs
        self.middle_imgs_pths = middle_imgs
        self.right_imgs_pths = right_imgs
        self.event_roots = event_roots
        self.matching_ts = matching_ts

        self.transforms = torch_transforms.ToTensor()
        self.to_voxel = representation.to_voxel_grid
        
        self.img_height = args.height
        self.img_width = args.width

        self.triplets = args.triplets

    def __len__(self):
        return len(self.middle_imgs_pths)

    def __getitem__(self, idx):

        ev_dir = math.floor(idx/self.triplets)
        left_img_tensor = self.transforms(Image.open(self.left_imgs_pths[idx]))
        target_img_tensor = self.transforms(Image.open(self.middle_imgs_pths[idx]))
        right_img_tensor = self.transforms(Image.open(self.right_imgs_pths[idx]))

        #ACCESS EVENTS as 00000000.npz to 0000000032.npz
        _ts = self.matching_ts[idx] #ids are dicts

        left_ts = _ts["left"]
        middle_ts = _ts["middle"]
        right_ts = _ts["right"]

        
        ev_seq = event.EventSequence.from_folder(self.event_roots[ev_dir], self.img_height, self.img_width, "*.npz")

        left_ev = ev_seq.filter_by_timestamp(left_ts, middle_ts)
        right_ev = ev_seq.filter_by_timestamp(middle_ts, right_ts)

        left_voxel = self.to_voxel(left_ev)
        right_voxel = self.to_voxel(right_ev)

        return {
        "before": {"rgb_image_tensor": left_img_tensor, "voxel_grid": left_voxel},
        "after": {"rgb_image_tensor": right_img_tensor, "voxel_grid": right_voxel},
        }, target_img_tensor