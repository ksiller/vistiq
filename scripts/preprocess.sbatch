#!/usr/bin/env bash
#SBATCH --job-name=preprocess
#SBATCH --output=preprocess-%A-%a.log
#SBATCH --error=preprocess-%A-%a.err
#SBATCH --time=1:00:00
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

# Check if file list is provided
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <file_list> [output_dir]"
    echo "  file_list:  Text file with one image file path per line (optionally with output dir as second column)"
    echo "              When run as job array, each task processes the file at line #ARRAY_TASK_ID"
    echo "  output_dir: Optional output directory (default: current directory)"
    echo "              If file list has second column with output dir, it will be used with priority:"
    echo "              - If second column is absolute path, it overrides CLI output_dir"
    echo "              - If second column is relative path, it is joined with CLI output_dir (or PWD)"
    exit 1
fi

file_list="$1"
cli_output_dir="${2:-$PWD}"

# Check if file list exists and is a file (not a directory)
if [[ ! -f "$file_list" ]]; then
    echo "Error: file list not found: $file_list"
    exit 1
fi

# Get array task ID (support both SLURM and PBS)
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

# Extract the file path for this task ID (1-indexed)
input_line=$(sed -n "${TASK_ID}p" "$file_list")

# Check if we got a valid file
if [[ -z "$input_line" ]]; then
    echo "Error: no file at line $TASK_ID in $file_list"
    exit 1
fi

# Parse quoted paths from file list (handles paths with spaces)
# Format: 'path1' 'path2' or path1 path2
# Use awk with single quote as field separator to extract content between quotes
# If line starts with quote, use quoted parsing; otherwise use regular field splitting
if [[ "$input_line" =~ ^\' ]]; then
    # Line starts with quote - extract between quotes
    # Fields when split on ': empty, path1, separator, path2, empty
    input_file=$(echo "$input_line" | awk -F"'" '{print $2}')
    file_output_dir=$(echo "$input_line" | awk -F"'" '{print $4}')
else
    # No quotes - use regular field splitting
    input_file=$(echo "$input_line" | awk '{print $1}')
    file_output_dir=$(echo "$input_line" | awk '{print $2}')
fi

# Determine output directory
# Priority: 1) Second CLI argument, 2) Second column in file list, 3) Current directory
if [[ -n "${file_output_dir:-}" ]]; then
    # File column exists
    if [[ "$file_output_dir" == /* ]]; then
        # Absolute path in file - use it directly
        output_dir="$file_output_dir"
    else
        # Relative path in file - join with cli_output_dir (which defaults to $PWD)
        output_dir="$cli_output_dir/$file_output_dir"
    fi
else
    # No file column - use cli_output_dir (which defaults to $PWD)
    output_dir="$cli_output_dir"
fi

# Check if the input file exists
if [[ ! -f "$input_file" ]]; then
    echo "Error: input file missing: $input_file"
    exit 1
fi

echo "Task ID: $TASK_ID"
echo "Input:   $input_file"
echo "Output:  $output_dir"

module purge
module load miniforge
source activate /standard/vol191/siegristlab/software/vistiq-env # replace with your own environment path

# Run vistiq preprocess on the selected file
vistiq preprocess \
    -i "$input_file" \
    -o "$output_dir" \
    -s '{"classname":"DoG","sigma_low":1.0,"sigma_high":12.0,"normalize":true}' \
    --loglevel DEBUG \
    --processes "${SLURM_CPUS_PER_TASK:-8}"
