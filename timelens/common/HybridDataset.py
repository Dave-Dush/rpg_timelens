from torch.utils.data import Dataset
import torchvision.transforms as pt_transforms
import torch
import os
from PIL import Image
import math
import numpy as np
import glob
from . import (transformers, hybrid_storage, pytorch_tools, event, )


class HybridDataset(Dataset):
    def __init__(
        self,
        root_image_folder,
        root_event_folder,
        skip_scheme,
        transform=None,
        pack_to_example=None,
        number_of_frames_to_interpolate = None,
    ):

        self.root_image_folder = root_image_folder
        self.root_event_folder = root_event_folder
        self.number_of_frames_to_interpolate = number_of_frames_to_interpolate
        self.transform = transform
        self.pack_to_example = pack_to_example
        self.skip_scheme = skip_scheme

        self.img_pths = glob.glob(os.path.join(self.root_image_folder, "*.png"))        
        self.img_pths.sort()

        self.eve_pths = glob.glob(os.path.join(self.root_event_folder, "*.npz"))
        self.eve_pths.sort()

        self.max_interpolations = math.floor(len(self.img_pths) / self.skip_scheme)

    def __len__(self):
        #assert len(self.img_pths) == len(self.eve_pths), "Imbalanced events and images"
        return self.max_interpolations 

    def __getitem__(self, idx):
        assert idx <= self.max_interpolations - 1
        
        default_tensor_conversion = pt_transforms.ToTensor()

        idx_interpolate = ((idx+1) * self.skip_scheme) - 1

        target_img = Image.open(self.img_pths[idx_interpolate])
        target_img_tensor = default_tensor_conversion(target_img)
        left_img = Image.open(self.img_pths[idx_interpolate - 1])
        right_img = Image.open(self.img_pths[idx_interpolate + 1])
        img_width, img_height = left_img.size

        left_eve = event.EventSequence.from_npz_files(
                    self.eve_pths[idx_interpolate],
                    img_height,
                    img_width
                )
        # check timestamps of events and image from timestamps.txt by assertion
        right_eve = event.EventSequence.from_npz_files(
                    self.eve_pths[idx_interpolate+1],
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

        


        

