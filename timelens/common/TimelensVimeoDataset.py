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

    def __len__(self):
        return len(self.middle_imgs_pths)

    def __getitem__(self, idx):

        ev_dir = math.floor(idx/9)
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

        left_ev_tensor = self.transforms(left_ev.to_image())
        right_ev_tensor = self.transforms(right_ev.to_image())

        

        return {
        "before": {"rgb_image_tensor": left_img_tensor, "voxel_grid": left_voxel},
        "after": {"rgb_image_tensor": right_img_tensor, "voxel_grid": right_voxel},
        }, left_ev_tensor, right_ev_tensor, target_img_tensor

        # _dir_idx = math.floor(idx/3)
        # _match_idx = idx % 3

        # _ts = np.loadtxt(os.path.join(self.seq_dirs[_dir_idx], "upsampled/imgs/timestamp.txt")).tolist()

        # _event_root = os.path.join(self.seq_dirs[_dir_idx], "events/")
        # _event_seq = event.EventSequence.from_folder(_event_root, self.height, self.width, "*.npz")

        # t_start = self._dt[(2*_match_idx)]
        # t_middle = self._dt[(2*_match_idx) + 1]
        # t_end = self._dt[(2*_match_idx) + 2]

        # left_event = _event_seq.filter_by_timestamp(t_start, t_middle)
        # left_voxel = representation.to_voxel_grid(left_event)

        # right_event = _event_seq.filter_by_timestamp(t_middle, t_end)
        # right_voxel = representation.to_voxel_grid(right_event)

        # left_match = (np.abs(_ts - self._dt[(2*_match_idx)]).argmin())
        # middle_match = (np.abs(_ts - self._dt[(2*_match_idx) + 1]).argmin())
        # right_match = (np.abs(_ts - self._dt[(2*_match_idx) + 2]).argmin())

        # left_id = f"{left_match:08d}"
        # middle_id = f"{middle_match:08d}"
        # right_id = f"{right_match:08d}"

        # left_path = os.path.join(self.seq_dirs[_dir_idx], f"upsampled/imgs/{left_id}.png")
        # _left_img = Image.open(left_path)
        # _left_img_tensor = self.torch_tensor(_left_img)

        # middle_path = os.path.join(self.seq_dirs[_dir_idx], f"upsampled/imgs/{middle_id}.png")
        # _target_img = Image.open(middle_path)
        # _target_img_tensor = self.torch_tensor(_target_img)

        # right_path = os.path.join(self.seq_dirs[_dir_idx], f"upsampled/imgs/{right_id}.png")
        # _right_img = Image.open(right_path)
        # _right_img_tensor = self.torch_tensor(_right_img)        

        # return {
        # "before": {"rgb_image_tensor": _left_img_tensor, "voxel_grid": left_voxel},
        # "after": {"rgb_image_tensor": _right_img_tensor, "voxel_grid": right_voxel},
        # }, _target_img_tensor