import os
import glob

timelens_vimeo_dir = "/usr/stud/dave/storage/user/dave/timelens_vimeo"
total_train_dir_txt = "/usr/stud/dave/storage/user/dave/vimeo_septuplet/sep_trainlist.txt"

total_train_dirs = list()

with open(total_train_dir_txt) as tr_file:
    total_train_dirs = tr_file.read().splitlines()

existing_dirs = glob.glob(os.path.join(timelens_vimeo_dir, "*/*"), recursive=True)
existing_dirs.sort()

existing_dirs_basename = list()

for existing_dir in existing_dirs:
    frag = existing_dir.split('/')
    existing_dirs_basename.append(frag[-2] + '/' + frag[-1])

dirs_to_preproc = list(set(total_train_dirs) - set(existing_dirs_basename))
dirs_to_preproc.sort()

def sanity_check():
    for dir_to_preproc in dirs_to_preproc:
        with open(dirs_to_preprocess_txt, 'a+') as tpr:
            tpr.write(dir_to_preproc + '\n')

def fragment_dirs_to_files(dirs_to_preproc):
    dir_counter = 0
    fragment_counter = 0
    print(f'Processing to_preproc {fragment_counter}')
    for dir_to_preproc in dirs_to_preproc:
        if dir_counter == 6400:
            fragment_counter += 1
            print(f'Processing to_preproc {fragment_counter}')
            dir_counter = 0
        fragment_to_preprocess_txt = f"/usr/stud/dave/storage/user/dave/timelens_vimeo/to_preproc{fragment_counter}.txt"
        with open(fragment_to_preprocess_txt, 'a+') as fpr:
            fpr.write(dir_to_preproc + '\n')
        dir_counter += 1
        
fragment_dirs_to_files(dirs_to_preproc)