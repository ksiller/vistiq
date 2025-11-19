#!/usr/bin/env bash
#SBATCH --job-name=analyze
#SBATCH --output=analyze-%A-%a.log
#SBATCH --error=analyze-%A-%a.err
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=1-100%10

set -e

# Check if file list is provided
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <file_list>"
    echo "  file_list:  Text file with one image file path per line"
    echo "              When run as job array, each task processes the file at line #ARRAY_TASK_ID"
    exit 1
fi

file_list="$1"

# Check if file list exists
if [[ ! -f "$file_list" ]]; then
    echo "Error: File list does not exist: $file_list"
    exit 1
fi

# Get array task ID (support both SLURM and PBS)
if [[ -n "$SLURM_ARRAY_TASK_ID" ]]; then
    TASK_ID=$SLURM_ARRAY_TASK_ID
elif [[ -n "$PBS_ARRAYID" ]]; then
    TASK_ID=$PBS_ARRAYID
else
    echo "Warning: No array task ID found. Processing first file in list."
    TASK_ID=1
fi

# Extract the file path for this task ID (1-indexed)
input_file=$(sed -n "${TASK_ID}p" "$file_list")

# Check if we got a valid file
if [[ -z "$input_file" ]]; then
    echo "Error: No file found at line $TASK_ID in $file_list"
    exit 1
fi

# Trim whitespace
input_file=$(echo "$input_file" | xargs)

# Check if the input file exists
if [[ ! -f "$input_file" ]]; then
    echo "Error: Input file does not exist: $input_file"
    exit 1
fi

echo "Task ID: $TASK_ID"
echo "Processing file: $input_file"

module purge
module load miniforge
source activate /standard/vol191/siegristlab/software/vistiq-env

# Run vistiq coincidence on the selected file
export QT_QPA_PLATFORM=offscreen # for napari headless
vistiq coincidence --input "$input_file" --sigma-low 1.0 --sigma-high 12.0 --threshold 0.1 
