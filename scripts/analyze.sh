#!/usr/bin/env bash
#SBATCH --job-name=analyze
#SBATCH --output=analyze-%j.log
#SBATCH --error=analyze-%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

module purge
module load miniforge
source activate vistiq-env

python coincidence.py --input $1 --sigma-low 1.0 --sigma-high 12.0 --threshold 0.1 