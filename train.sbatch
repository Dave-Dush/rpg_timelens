#!/bin/bash
#SBATCH --job-name="advArch"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=32G
#SBATCH --time=168:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/stud/dave/storage/slurm/logs/gan_training/slurm-%j.out
#SBATCH --error=/usr/stud/dave/storage/slurm/logs/gan_training/slurm-%j.out
srun python cgan.py --config config/timelens_vimeo.txt