#!/usr/bin/env bash

set -e

# Check if input and output paths are provided
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <input_path> <output_path>"
    echo "  input_path:  Directory to search recursively for .lif files"
    echo "  output_path:  File to write absolute paths of found .lif files"
    exit 1
fi

input_path="$1"
output_path="$2"

# Check if input path exists
if [[ ! -e "$input_path" ]]; then
    echo "Error: Input path does not exist: $input_path"
    exit 1
fi

# Find all .lif files recursively and write absolute paths to output
# Convert input_path to absolute path first (ensures find returns absolute paths)
abs_input_path="$(cd "$input_path" && pwd)"

# Find files - since we use absolute path, find will return absolute paths
find "$abs_input_path" -type f -name "*.lif" > "$output_path"

# Check if any files were found
if [[ ! -s "$output_path" ]]; then
    echo "Warning: No .lif files found in $input_path"
    exit 0
fi

echo "Found $(wc -l < "$output_path") .lif file(s) in $input_path"
echo "Absolute paths written to: $output_path"