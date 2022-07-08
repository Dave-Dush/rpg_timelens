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
        self.torch_tensor = torch_transforms.ToTensor()

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
            _local_interpolation_sets = math.floor( (len(_evs) - 1) / (self.skip_scheme + 1) )

            
            _local_interpolations = self.skip_scheme * _local_interpolation_sets
            self.max_interpolations += _local_interpolations

            for idx in range(_local_interpolation_sets):

                left_idx = (idx * (self.skip_scheme + 1))
                right_idx = left_idx + (self.skip_scheme + 1)

                self.left_imgs_pth.append(_imgs[left_idx])
                self.right_imgs_pth.append(_imgs[right_idx])
                
                # list of skipped images' path
                skipped_imgs_pths = [_imgs[skp_idx] for skp_idx in range(left_idx+1, right_idx)]
                self.imgs_to_interpolate_pth.extend(skipped_imgs_pths)

                # list of all events occouring at an index
                event_set = [_evs[ev_idx] for ev_idx in range(left_idx+1, right_idx+1)]
                self.event_pth.append(event_set)

    def __len__(self):
        return self.max_interpolations

    def __getitem__(self, idx):

        target_img_pth = self.imgs_to_interpolate_pth[idx]

        raw_target = Image.open(target_img_pth)
        target_img_tensor = self.torch_tensor(raw_target)

        current_set = math.floor(idx / self.skip_scheme)

        left_img = Image.open(self.left_imgs_pth[current_set])
        right_img = Image.open(self.right_imgs_pth[current_set])
        img_width, img_height = left_img.size # assumption made: all images are of same size

        evs_set = self.event_pth[current_set]

        eve_slice_idx = (idx % self.skip_scheme) + 1

        left_eve = event.EventSequence.from_npz_files(
            evs_set[:eve_slice_idx],
            img_height,
            img_width
        )

        right_eve = event.EventSequence.from_npz_files(
            evs_set[eve_slice_idx:],
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