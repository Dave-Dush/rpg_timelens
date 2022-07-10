#!/bin/bash
#SBATCH --job-name="seq3Ds1kBs16"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=32G
#SBATCH --time=168:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/dave/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/dave/logs/slurm-%j.out
srun python scaled_trainfusion.py --config config/timelens_vimeo.txt