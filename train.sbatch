#!/bin/bash
#SBATCH --job-name="4cgan500"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/stud/dave/storage/slurm/logs/gan_training/slurm-%j.out
#SBATCH --error=/usr/stud/dave/storage/slurm/logs/gan_training/slurm-%j.out
srun python video2frame.py --config config/video2frame.txt