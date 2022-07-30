# #import torch

# from matplotlib import pyplot as plt
# import numpy as np


# def exceptEvery(nth, a):
#     print(type(a.size(0)))
#     m = a.size(0) // nth * nth
#     return torch.cat((a[:m].reshape(-1,nth)[:,:nth-1].reshape(-1), a[m:m+nth-1]))

# def plot_event_hist(ev_arr, path, idx, W_final=448, H_final=256, pol_neg=-1, pol_pos=1, idx_pol=3, chunks=1):
#     """
#     Inputs: 
#     :ev_arr np.array (N_evs, 5)
#     :path = /path/x.png
#     """
#     ev_arr = np.load(ev_arr)
#     ev_arr = np.transpose(np.stack((ev_arr["x"], ev_arr["y"], ev_arr["t"], ev_arr["p"])))
#     N_evs = ev_arr.shape[0]
#     ch_size = int(N_evs/chunks)
#     if not os.path.exists(path):
#         os.makedirs(path)
#     for ch in np.arange(chunks):
#         evs = ev_arr[(ch*ch_size):(ch+1)*ch_size]
#         if len(evs) == 0:
#             print("No more events! (check events_rectified and timestamps file for mismatch)")
#         pos = evs[np.where(evs[:, idx_pol] == pol_pos)]
#         neg = evs[np.where(evs[:, idx_pol] == pol_neg)]
#         plt.gca().invert_yaxis()
#         plt.scatter(pos[:, 0], pos[:, 1], color="blue", s=0.7)
#         plt.scatter(neg[:, 0], neg[:, 1], color="red", s=0.7)
#         plt.xlim(0, W_final)
#         plt.ylim(0, H_final)
#         # have to invert axis after setting limits
#         plt.gca().invert_yaxis()
#         plt.savefig(f"{path}_{idx}.png")
#         plt.close()
# ev_arrs = glob.glob(os.path.join("/usr/stud/dave/storage/user/dave/timelens_vimeo/00001/0001/events", "*.npz"))
# ev_arrs.sort()
# img_pths = "/usr/stud/dave/storage/user/dave/timelens_vimeo/00001/0001/upsampled/vis_evs/vis"

# for idx, ev_arr in enumerate(ev_arrs):
#     plot_event_hist(ev_arr, img_pths, idx)

# get length of dataset with subdirectories

# from glob import glob
# import os
# import torch
# from torchvision import transforms
# import numpy as np
# import math
# from event import EventSequence
# from image_sequence import ImageSequence
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import ImageGrid
# from representation import to_voxel_grid
# import sys

# tensor_trans = transforms.ToTensor()
# pil_trans = transforms.ToPILImage()
# seq_dir = "/usr/stud/dave/storage/user/dave/timelens_vimeo/00001/0038"

# TIMESTAMP_COLUMN = 2
# X_COLUMN = 0
# Y_COLUMN = 1
# POLARITY_COLUMN = 3

# def _pack_to_example(left_image, right_image, left_voxel_grid, right_voxel_grid):
#     return {
#         "before": {"rgb_image_tensor": left_image, "voxel_grid": left_voxel_grid},
#         "after": {"rgb_image_tensor": right_image, "voxel_grid": right_voxel_grid},
#     }

# _timestamp_file = os.path.join(seq_dir, "upsampled/imgs/timestamp.txt")
# _timestamps = np.loadtxt(_timestamp_file, delimiter="\n")

# # t_start and t_end are timestamps of the entire seq_dir
# t_start = _timestamps[0]
# t_end = _timestamps[-1]

# downsampled_timestamps = list()
# upsampling_factor = math.floor(t_end / 6)

# last_idx = 0
# for idx in range(1, 7):
#     _upfactor = upsampling_factor * idx
#     _dt = list()
#     while _timestamps[last_idx] < _upfactor :
#         _dt.append(_timestamps[last_idx])
#         last_idx += 1
#     downsampled_timestamps.append(_dt)

# downsampled_timestamps.append([_timestamps[-1]])

# img_root = os.path.join(seq_dir, "upsampled/imgs/")
# image_seq = ImageSequence.from_folder(img_root, "*.png", "timestamp.txt")

# event_root = os.path.join(seq_dir, "events/")
# event_seq = EventSequence.from_folder(event_root, image_seq._width, image_seq._height, "*.npz")

# event_voxels = list()
# events = list()
# for i in range(0, 6):
#     print(downsampled_timestamps[i][0], downsampled_timestamps[i+1][0])
#     event_block = event_seq.filter_by_timestamp(downsampled_timestamps[i][0], downsampled_timestamps[i+1][0] - downsampled_timestamps[i][0])
#     event_voxels.append(to_voxel_grid(event_block))
#     events.append(event_block)
    


# matching_timestamps = list()
# for idx in range(len(downsampled_timestamps)):
#     matching_timestamps.append(downsampled_timestamps[idx][0])
    
# matching_idx = [image_seq._timestamps.index(x) for x in matching_timestamps]

# downsampled_images = [tensor_trans(image_seq._images[img_idx]) for img_idx in matching_idx]

# for int_set in range(0, 6, 2):
#     left_img = downsampled_images[int_set]
#     target = downsampled_images[int_set + 1]
#     right_img = downsampled_images[int_set + 2]

#     left_eve = event_voxels[int_set]
#     right_eve = event_voxels[int_set+1]

#     left_pil = pil_trans(left_img.squeeze_(0))
#     target_pil = pil_trans(target.squeeze_(0))
#     right_pil = pil_trans(right_img.squeeze_(0))
#     left_eve_img = events[int_set].to_image()
#     right_eve_img = events[int_set+1].to_image()

#     fig = plt.figure(figsize=(8., 8.))
#     grid = ImageGrid(fig, 111,
#                     nrows_ncols=(3,2),
#                     axes_pad=0.1,
#                     )
#     for ax, im in zip(grid, [left_pil, left_eve_img, right_pil, right_eve_img, target_pil]):
#         ax.imshow(im)

#     fig.savefig(f"example_input_{int_set}.png")

#     example = _pack_to_example(left_img, right_img, left_eve, right_eve)
# from PIL import Image
# from losses import FusionLoss
# import torch
# from torchvision import transforms

# img1 = Image.open("/usr/stud/dave/storage/user/dave/timelens_vimeo/00001/0001/upsampled/imgs/00000000.png")
# img2 = Image.open("/usr/stud/dave/storage/user/dave/timelens_vimeo/00001/0001/upsampled/imgs/00000001.png")

# img1 = transforms.ToTensor()(img1)
# img2 = transforms.ToTensor()(img2)
# device = "cpu"
# lpips_loss_fn = FusionLoss(device)
# lpips_loss = lpips_loss_fn.l1_loss(img1, img2)
# print(lpips_loss)

# discard = 0
# for idx, existing_dir in enumerate(existing_dirs):

#     img_len = len(glob(os.path.join(existing_dir, "upsampled/imgs/*.png")))
#     ev_len = len(glob(os.path.join(existing_dir, "*/*.npz")))
#     if( img_len != ev_len ):
#         discard += 1

# print(discard)


# print(len(existing_dirs))

#check psnr scheme with cv
#cv2.PSNR(img1, img2)

# from random import randint, randrange


# def max_psnr():
#     psnr = 42
    
#     rand_scores = [42]

#     if psnr >= max(rand_scores):
#         print(psnr)


# max_psnr()

# import numpy as np
# # def read_events():
# #     ev_file = "/usr/stud/dave/storage/user/dave/timelens_vimeo/00001/0039/events/0000000001.npz"
# #     evs = np.load(ev_file)

# #     print(evs["t"])

# read_events()
import os
equal_dir_txt = "/usr/stud/dave/storage/user/dave/timelens_vimeo/equal_dirs.txt"

dir_list = list()
_ts = list()

with open(equal_dir_txt, "r") as _eq:
    _dir = [line.strip() for line in _eq]
    dir_list.extend(_dir)

with open(os.path.join(dir_list, "upsampled/imgs/timestamp.txt"), "r") as _f:
    _temp_ts = [float(line.strip()) for line in _f]
    start = f"{_temp_ts[0]}"
    end = f"{_temp_ts[0]}"