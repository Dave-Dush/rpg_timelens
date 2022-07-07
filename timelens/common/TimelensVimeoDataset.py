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
        self.skip_scheme = skip_scheme # no. of contiguous frame skips
        self.transform = transform
        self.mode = mode

        self.left_imgs_pth = list()
        self.right_imgs_pth = list()

        self.imgs_to_interpolate_pth = list()
        
        self.event_pth = list()

        self.max_interpolations = 0
        print(f"\nInitializing {self.mode} dataset...")
        for seq_dir in self.seq_dirs:

            #print(f"\nAccumulating {seq_dir}")
            _imgs = glob(os.path.join(seq_dir + "/upsampled/imgs", "*.png"))
            _evs = glob(os.path.join(seq_dir, "events/*.npz"))
            
            _imgs.sort()
            _evs.sort()

            # relying on number of events because sometimes
            # rpg_vid2e generates more frames
            # than number of events
            
            _local_interpolations = math.floor( (len(_evs) - 1) / (self.skip_scheme + 1) )
            
            for idx in range(_local_interpolations):

                left_idx = (idx * (self.skip_scheme + 1))
                right_idx = left_idx + (self.skip_scheme + 1)

                self.left_imgs_pth.append(_imgs[left_idx])
                self.right_imgs_pth.append(_imgs[right_idx])
                
                # list of skipped images' path
                skipped_imgs_pths = [_imgs[skp_idx] for skp_idx in range(left_idx+1, right_idx)]
                self.imgs_to_interpolate_pth.append(skipped_imgs_pths)

                # list of all events occouring at an index
                event_set = [_evs[ev_idx] for ev_idx in range(left_idx, right_idx+1)]
                self.event_pth.append(event_set)

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

        target_imgs_pth = self.imgs_to_interpolate_pth[idx]

        raw_target_imgs = [Image.open(tgt_img) for tgt_img in target_imgs_pth]
        target_imgs_tensor = [torch_transforms.ToTensor(raw_tgt) for raw_tgt in raw_target_imgs]

        left_img = Image.open(self.left_imgs_pth[idx])
        right_img = Image.open(self.right_imgs_pth[idx])
        img_width, img_height = left_img.size # assumption made: all images are of same size

        trainable_features = list()
        evs_set = self.event_pth[idx]

        for skp_idx in range(self.skip_scheme):
            left_eve = event.EventSequence.from_npz_files(
                evs_set[:skp_idx+2],
                img_height,
                img_width
            )
            right_eve = event.EventSequence.from_npz_files(
                evs_set[skp_idx+2:],
                img_height,
                img_width
            )

            right_weight = float(idx + 1.0) / (self.max_interpolations + 1.0)

            example = self._pack_to_example(left_img, right_img, left_eve, right_eve, right_weight)
            example = transformers.apply_transforms(example, self.transform)
            example = transformers.collate([example]),
            example = pytorch_tools.move_tensors_to_cuda(example)

            cat_features = torch.cat([
                example[0]["before"]["voxel_grid"],
                example[0]["before"]["rgb_image_tensor"],
                example[0]["after"]["voxel_grid"],
                example[0]["before"]["rgb_image_tensor"],
                ], dim=1)
            trainable_features.append(cat_features)

        return trainable_features, target_imgs_tensor

    def _pack_to_example(self, left_image, right_image, left_events, right_events, right_weight):
        return {
            "before": {"rgb_image": left_image, "events": left_events},
            "middle": {"weight": right_weight},
            "after": {"rgb_image": right_image, "events": right_events},
        }