#!/usr/bin/env bash

set -e

# Check if input and output paths are provided
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <input_path> <output_path>"
    echo "  input_path:  Directory to search recursively for .lif files"
    echo "  output_path:  File to write two columns: absolute_path relative_path"
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

# Find files and write two columns: absolute_path relative_path
# Use a temporary file to process each file
temp_file=$(mktemp)
trap "rm -f '$temp_file'" EXIT

find "$abs_input_path" -type f -name "*.lif" > "$temp_file"

# Process each file to compute relative path
# Initialize output file (truncate if exists)
> "$output_path"

while IFS= read -r file_path || [[ -n "$file_path" ]]; do
    # Skip empty lines
    [[ -z "$file_path" ]] && continue
    
    # Compute relative path from abs_input_path to file_path
    rel_path=$(realpath --relative-to="$abs_input_path" "$file_path" 2>/dev/null || \
               python3 -c "import os; print(os.path.relpath('$file_path', '$abs_input_path'))" 2>/dev/null || \
               echo "${file_path#$abs_input_path/}")
    echo "'$file_path' '$rel_path'"
done < "$temp_file" >> "$output_path"

# Check if any files were found
if [[ ! -s "$output_path" ]]; then
    echo "Warning: No .lif files found in $input_path"
    exit 0
fi

echo "Found $(wc -l < "$output_path") .lif file(s) in $input_path"
echo "Absolute and relative paths written to: $output_path"