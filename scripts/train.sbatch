#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --output=train-%J.log
#SBATCH --error=train-%J.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

input="$1"
output="$2"

module purge
module load miniforge
source activate /standard/vol191/siegristlab/software/vistiq-env # replace with your own environment path

echo Executing from $(pwd)

vistiq --device "cuda" --loglevel "DEBUG" train --input '{"paths":"24h/AkhGal4 x OR Susie/", "include":"*Preprocessed_Dpn.tif", "exclude": "*training_data*"}' --labels '{"paths":"24h/AkhGal4 x OR Susie/", "include":"*Labels_Dpn_curated.tif", "exclude": "*training_data*"}' --output '{"path": "./training_data", "overwrite":true}' --step '{"classname": "MicroSAMTrainer", "patch_shape": [1,512,512]}'
