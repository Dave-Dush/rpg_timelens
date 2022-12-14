#!/bin/bash
#SBATCH --job-name="longCG"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/usr/stud/dave/storage/slurm/logs/gan_training/slurm-%j.out
#SBATCH --error=/usr/stud/dave/storage/slurm/logs/gan_training/slurm-%j.out
srun python modelSummaries.py