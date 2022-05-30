#!/bin/bash
#SBATCH --job-name="Fusion Training"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/dave/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/dave/logs/slurm-%j.out
srun python train_fusion.py