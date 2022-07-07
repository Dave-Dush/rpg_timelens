from glob import glob
import math
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as torch_transforms
from . import (transformers, pytorch_tools, event)
from PIL import Image


class TimelensVimeoDataset(Dataset):

    def __init__(
        self, seq_dirs, skip_scheme, transform, mode
    ):
        self.seq_dirs = seq_dirs
        self.skip_scheme = skip_scheme
        self.transform = transform
        self.mode = mode

        self.left_imgs_pth = list()
        self.right_imgs_pth = list()
        self.left_evs_pth = list()
        self.right_evs_pth = list()
        self.imgs_to_interpolate_pth = list()

        self.max_interpolations = 0
        count = 0
        print(f"\nInitializing {self.mode} dataset...")
        for seq_dir in seq_dirs:

            #print(f"\nAccumulating {seq_dir}")
            _imgs = glob(os.path.join(seq_dir + "/upsampled/imgs", "*.png"))
            _evs = glob(os.path.join(seq_dir, "events/*.npz"))
            
            _imgs.sort()
            _evs.sort()

            _local_interpolations = math.floor(len(_imgs) / self.skip_scheme)
            
            for idx in range(_local_interpolations):

                _idx_to_interpolate = ((idx+1) * self.skip_scheme) - 1

                # idx limit compared agains events because there can be
                # less events than number of upsampled images (rpg_vid2e effect)
                if(_idx_to_interpolate < (len(_evs)-1) ):

                    self.left_imgs_pth.append(_imgs[_idx_to_interpolate-1])
                    self.left_evs_pth.append(_evs[_idx_to_interpolate-1])

                    self.right_imgs_pth.append(_imgs[_idx_to_interpolate+1])
                    self.right_evs_pth.append(_evs[_idx_to_interpolate+1])

                    self.imgs_to_interpolate_pth.append(_imgs[_idx_to_interpolate])

                    self.max_interpolations += 1


        # print("\nDataset processed")
            
        # print(f"Left events {self.left_evs_pth[-3:]}")
        # print(f"Right events {self.right_evs_pth[-3:]}")

        # print(f"Left imgs {self.left_imgs_pth[-3:]}")
        # print(f"Right imgs {self.right_imgs_pth[-3:]}")

        # print(f"To interpolate {self.imgs_to_interpolate_pth[-3:]}")

        

    def __len__(self):
        return self.max_interpolations

    def __getitem__(self, idx):

        target_img = Image.open(self.imgs_to_interpolate_pth[idx])

        default_tensor_conversion = torch_transforms.ToTensor()
        target_img_tensor = default_tensor_conversion(target_img)

        left_img = Image.open(self.left_imgs_pth[idx])
        right_img = Image.open(self.right_imgs_pth[idx])
        img_width, img_height = left_img.size

        left_eve = event.EventSequence.from_npz_files(
                        self.left_evs_pth[idx],
                        img_height,
                        img_width
                    )

        right_eve = event.EventSequence.from_npz_files(
                        self.right_evs_pth[idx],
                        img_height,
                        img_width
                     )

        right_weight = float(idx + 1.0) / (self.max_interpolations + 1.0)

        example = self._pack_to_example(left_img, right_img, left_eve, right_eve, right_weight)
        example = transformers.apply_transforms(example, self.transform)
        example = transformers.collate([example]),
        example = pytorch_tools.move_tensors_to_cuda(example)

        trainable_features = torch.cat([
            example[0]["before"]["voxel_grid"],
            example[0]["before"]["rgb_image_tensor"],
            example[0]["after"]["voxel_grid"],
            example[0]["before"]["rgb_image_tensor"],
            ], dim=1)

        return trainable_features, target_img_tensor

    def _pack_to_example(self, left_image, right_image, left_events, right_events, right_weight):
        return {
            "before": {"rgb_image": left_image, "events": left_events},
            "middle": {"weight": right_weight},
            "after": {"rgb_image": right_image, "events": right_events},
        }