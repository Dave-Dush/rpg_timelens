import numpy as np
import math
from os.path import join

def mul_by_factor(v2e_timestamps, factor=1e9):
    timelens_timestamps = []
    for len in range(v2e_timestamps.size):
        timelens_timestamps.append(int(v2e_timestamps[len]*factor))
    return np.asarray(timelens_timestamps)

def from_v2e_2_timelens(FRAME_INPUT_DIR):
    v2e_timestamps = np.genfromtxt(join(FRAME_INPUT_DIR, "timestamps.txt"))
    timelens_timestamps = mul_by_factor(v2e_timestamps)
    with open(join(FRAME_INPUT_DIR, "new_timestamps.txt"), "a") as f:
        for len in range(timelens_timestamps.size):
            f.write(str(timelens_timestamps[len]) + '\n')

from_v2e_2_timelens("/usr/stud/dave/storage/user/dave/minivimeo_septuplet/00003_0003/upsampled/imgs/")
