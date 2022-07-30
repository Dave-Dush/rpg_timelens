#!/bin/bash
#SBATCH --job-name="Len"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=32G
#SBATCH --time=80:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/dave/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/dave/logs/slurm-%j.out
srun python sandbox.py