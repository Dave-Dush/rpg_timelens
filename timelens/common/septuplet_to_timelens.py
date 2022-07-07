import os
import glob
from os.path import join
import shutil
import subprocess
import numpy as np


sequence_root = "/usr/stud/dave/storage/user/dave/vimeo_septuplet/sequences"
train_dir_txt = "/usr/stud/dave/storage/user/dave/vimeo_septuplet/sep_trainlist.txt"
septuplet_fps = 30

timelens_sequence_root = "/usr/stud/dave/storage/user/dave/timelens_vimeo"

upsampling_script = "/usr/stud/dave/rpg_vid2e/upsampling/upsample.py"
event_gen_script = "/usr/stud/dave/rpg_vid2e/esim_torch/scripts/generate_events.py"

train_dirs = list()

with open(train_dir_txt) as tr_file:
    train_dirs = tr_file.read().splitlines()

# running on cpu train_dirs_set1 = train_dirs[:0071]
#train_dirs_set2 = train_dirs[200: 600]

train_dirs_set_leftover = list()
for train_dir in train_dirs[612:]:
    train_dirs_set_leftover.append(train_dir)

### Function definitions
def prepare_for_v2e_upsample(train_dir):
    """
    Adds fps=30 and saves in train_dir/fps.txt
    """
    print(f"Processing {train_dir}")
    og_dir = os.path.join(sequence_root, train_dir)
    og_imgs = glob.glob(os.path.join(og_dir, "*.png"))

    timelens_img_dir = os.path.join(timelens_sequence_root, train_dir, "imgs")
    
    if not os.path.exists(timelens_img_dir):
        os.makedirs(timelens_img_dir)

    for og_img in og_imgs:
        shutil.copy(og_img, timelens_img_dir)
    print("Files copied.")

    with open(os.path.join(timelens_sequence_root, train_dir, "fps.txt"), "a+") as fps:
        fps.write(str(septuplet_fps))
    print(f"FPS set for {train_dir}")

def upsample_images(train_dir):
    # change loop over train_dirs to pre_upsampled_dir after prep_to_upsample
    # or come up with better scheme to begin multi-process
    print(f"Upsamling {train_dir}")
    pre_upsample_dir = os.path.join(timelens_sequence_root, train_dir)
    post_upsample_dir = os.path.join(timelens_sequence_root, train_dir, "upsampled")

    fps_txt = os.path.join(timelens_sequence_root, train_dir, "fps.txt")

    ups_result = subprocess.run(
        ["python", upsampling_script, \
            f"--input_dir={pre_upsample_dir}", \
            f"--output_dir={post_upsample_dir}"], capture_output=True, text=True)


def ups_to_events(train_dir, contrast_threshold=0.2, refractory_period_ns=0):
    print(f"Genetating events for {train_dir}")
    post_upsample_dir = os.path.join(timelens_sequence_root, train_dir, "upsampled/imgs")
    eve_gen_dir = os.path.join(timelens_sequence_root, train_dir, "events")
    eve_result = subprocess.run(
        ["python", event_gen_script, \
            f"--input_dir={post_upsample_dir}", \
            f"--output_dir={eve_gen_dir}", \
            f"--contrast_threshold_neg={contrast_threshold}", \
            f"--contrast_threshold_pos={contrast_threshold}", \
            f"--refractory_period_ns={refractory_period_ns}"]
    )
    return eve_gen_dir

def fetch_and_shift(ev_dir):
    print(f"Shifting indexes for {train_dir}")
    npz_lst = glob.glob(join(ev_dir, '*.npz'))
    npz_lst.sort()
    for idx,eve in enumerate(reversed(npz_lst)):
        sub_idx = len(npz_lst) - idx
        shift_idx = str(format(sub_idx, '010d')) + '.npz'
        new_name = os.path.join(ev_dir, shift_idx)
        #print(eve, new_name)
        os.rename(eve, new_name)
    
    # save null file for timelens convention
    null_idx = str(format(0, '010d') + '.npz')
    null_file = os.path.join(ev_dir, null_idx)
    #print(null_file)
    x = []
    y = []
    t = []
    p = []
    np.savez(null_file, x=x, y=y, t=t, p=p)



"""
Method calling
"""

def convert_fragment(fragment_file):
    print(f"Converting from {fragment_file}")
    frag_file_txt = open(fragment_file, 'r')
    dir_list = frag_file_txt.read().splitlines()
    for train_dir in dir_list:
        prepare_for_v2e_upsample(train_dir)
        upsample_images(train_dir)
        current_ev_dir = ups_to_events(train_dir)
        fetch_and_shift(current_ev_dir)
    

