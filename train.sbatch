#!/bin/bash
#SBATCH --job-name="fwbwflo1K"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=64G
#SBATCH --time=168:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/dave/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/dave/logs/slurm-%j.out
srun python train_warp.py --config config/timelens_warp.txt