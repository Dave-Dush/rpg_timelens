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

# import glob
# import os
# timelens_vimeo_dir = "/usr/stud/dave/storage/user/dave/timelens_vimeo"
# existing_dirs = glob.glob(os.path.join(timelens_vimeo_dir, "*/*"), recursive=True)
# existing_dirs.sort()

# print(len(existing_dirs))

#check psnr scheme with cv
#cv2.PSNR(img1, img2)

from random import randint, randrange


def max_psnr():
    psnr = 42
    
    rand_scores = [42]

    if psnr >= max(rand_scores):
        print(psnr)


max_psnr()
