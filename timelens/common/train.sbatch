#!/bin/bash
#SBATCH --job-name="Upscaling DS"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=32G
#SBATCH --time=80:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/dave/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/dave/logs/slurm-%j.out
srun python conversion_script.py /usr/stud/dave/storage/user/dave/timelens_vimeo/to_preproc10.txt