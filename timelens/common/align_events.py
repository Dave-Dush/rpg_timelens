from os.path import join
import os
import glob
import numpy as np

def fetch_and_shift(ev_dir):
    npz_lst = glob.glob(join(ev_dir, '*.npz'))
    npz_lst.sort()
    for idx,eve in enumerate(reversed(npz_lst)):
        sub_idx = len(npz_lst) - idx
        shift_idx = str(format(sub_idx, '010d')) + '.npz'
        new_name = ev_dir + shift_idx
        #print(eve, new_name)
        os.rename(eve, new_name)
    
    # save null file for timelens convention
    null_file = ev_dir + str(format(0, '010d') + '.npz')
    # print(null_file)
    x = []
    y = []
    t = []
    p = []
    np.savez(null_file, x=x, y=y, t=t, p=p)


fetch_and_shift("/usr/stud/dave/storage/user/dave/minivimeo_septuplet/00003_0003/events_copy/")
