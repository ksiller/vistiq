#!/usr/bin/env bash
# Script to rename files and folders recursively based on mapping patterns
# Usage: rename.sh [--dry-run] <folder> <old1:new1> [<old2:new2> ...]

set -e

# Parse arguments
DRY_RUN=false
FOLDER=""
MAPPINGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -*)
            echo "Error: Unknown option: $1" >&2
            exit 1
            ;;
        *)
            if [[ -z "$FOLDER" ]]; then
                FOLDER="$1"
            elif [[ "$1" =~ ^[^:]+:[^:]+$ ]]; then
                MAPPINGS+=("$1")
            else
                echo "Error: Invalid mapping: '$1'. Expected 'old:new'" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate
if [[ -z "$FOLDER" ]] || [[ ! -d "$FOLDER" ]] || [[ ${#MAPPINGS[@]} -eq 0 ]]; then
    echo "Usage: $0 [--dry-run] <folder> <old1:new1> [<old2:new2> ...]" >&2
    exit 1
fi

# Get absolute path
ORIGINAL_FOLDER=$(cd "$FOLDER" && pwd)
CURRENT_FOLDER="$ORIGINAL_FOLDER"

# Build sed script for fast replacements
SED_SCRIPT=""
for mapping in "${MAPPINGS[@]}"; do
    IFS=':' read -r old new <<< "$mapping"
    old_escaped=$(printf '%s\n' "$old" | sed 's/[[\.*^$()+?{|]/\\&/g')
    new_escaped=$(printf '%s\n' "$new" | sed 's/[[\.*^$()+?{|]/\\&/g')
    SED_SCRIPT="${SED_SCRIPT}s/${old_escaped}/${new_escaped}/g;"
done

# Fast mapping function using sed
apply_mappings() {
    echo "$1" | sed "$SED_SCRIPT"
}

# Rename function
rename_item() {
    local old="$1"
    local new="$2"
    [[ "$old" == "$new" ]] && return 0
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "Would rename: $old -> $new"
    else
        if [[ -e "$new" ]]; then
            echo "Warning: $new exists, skipping" >&2
            return 0  # Return 0 to not abort script
        fi
        mv "$old" "$new"
        echo "Renamed: $old -> $new"
    fi
}

# Helper to build new path and find first changed component for directories
build_dir_rename() {
    local dir="$1"
    local base_folder="$2"
    
    # Get relative path from base folder
    local rel="${dir#$base_folder/}"
    [[ "$rel" == "$dir" ]] && echo "$dir|$dir" && return
    
    # Split and apply mappings to each component
    IFS='/'
    read -ra parts <<< "$rel"
    local new_parts=()
    local first_changed_idx=-1
    
    for i in "${!parts[@]}"; do
        local part="${parts[i]}"
        [[ -z "$part" ]] && continue
        
        local new_part=$(apply_mappings "$part")
        new_parts+=("$new_part")
        
        # Track first changed component
        if [[ "$part" != "$new_part" ]] && [[ $first_changed_idx -eq -1 ]]; then
            first_changed_idx=$i
        fi
    done
    
    # If nothing changed, return as-is
    if [[ $first_changed_idx -eq -1 ]]; then
        echo "$dir|$dir"
        return
    fi
    
    # Build old and new paths up to and including first changed component
    local old_prefix_parts=("${parts[@]:0:$first_changed_idx}")
    local new_prefix_parts=("${new_parts[@]:0:$first_changed_idx}")
    
    local old_prefix=$(IFS='/'; echo "${old_prefix_parts[*]}")
    local new_prefix=$(IFS='/'; echo "${new_prefix_parts[*]}")
    
    # Build paths: base + prefix + changed_component
    local old_path="$base_folder"
    [[ -n "$old_prefix" ]] && old_path="$base_folder/$old_prefix"
    old_path="$old_path/${parts[$first_changed_idx]}"
    
    local new_path="$base_folder"
    [[ -n "$new_prefix" ]] && new_path="$base_folder/$new_prefix"
    new_path="$new_path/${new_parts[$first_changed_idx]}"
    
    echo "$old_path|$new_path"
}

# Helper to build full new path for files
build_new_path() {
    local item="$1"
    local base_folder="$2"
    
    local rel="${item#$base_folder/}"
    [[ "$rel" == "$item" ]] && echo "$item" && return
    
    IFS='/'
    read -ra parts <<< "$rel"
    local new_parts=()
    for part in "${parts[@]}"; do
        [[ -n "$part" ]] && new_parts+=("$(apply_mappings "$part")")
    done
    
    local new_rel=$(IFS='/'; echo "${new_parts[*]}")
    echo "$base_folder/$new_rel"
}

# Step 1: Collect and rename all directories depth-first
declare -a dirs
declare -a dir_depths

# Find all directories
while IFS= read -r -d '' item; do
    if [[ "$item" != "$ORIGINAL_FOLDER" ]] && [[ -d "$item" ]]; then
        rel="${item#$ORIGINAL_FOLDER/}"
        if [[ "$rel" != "$item" ]]; then
            depth=$(echo "$rel" | tr -cd '/' | wc -c)
            dirs+=("$item")
            dir_depths+=("$depth")
        fi
    fi
done < <(find "$ORIGINAL_FOLDER" -type d -print0)

# Sort directories by depth (deepest first)
if [[ ${#dirs[@]} -gt 0 ]]; then
    tmp=$(mktemp)
    for i in "${!dirs[@]}"; do
        printf "%d\t%s\n" "${dir_depths[i]}" "${dirs[i]}"
    done | sort -rn -t$'\t' -k1 > "$tmp"
    
    dirs=()
    while IFS=$'\t' read -r depth dir; do
        dirs+=("$dir")
    done < "$tmp"
    rm -f "$tmp"
fi

# Rename directories depth-first (may need multiple passes)
changed=true
pass=0
max_passes=100  # Safety limit

while [[ "$changed" == true ]] && [[ $pass -lt $max_passes ]]; do
    changed=false
    pass=$((pass + 1))
    
    # Re-find directories in current folder (paths may have changed)
    dirs=()
    dir_depths=()
    
    while IFS= read -r -d '' item; do
        if [[ "$item" != "$CURRENT_FOLDER" ]] && [[ -d "$item" ]]; then
            rel="${item#$CURRENT_FOLDER/}"
            if [[ "$rel" != "$item" ]]; then
                depth=$(echo "$rel" | tr -cd '/' | wc -c)
                dirs+=("$item")
                dir_depths+=("$depth")
            fi
        fi
    done < <(find "$CURRENT_FOLDER" -type d -print0)
    
    # Sort by depth (deepest first)
    if [[ ${#dirs[@]} -gt 0 ]]; then
        tmp=$(mktemp)
        for i in "${!dirs[@]}"; do
            printf "%d\t%s\n" "${dir_depths[i]}" "${dirs[i]}"
        done | sort -rn -t$'\t' -k1 > "$tmp"
        
        dirs=()
        while IFS=$'\t' read -r depth dir; do
            dirs+=("$dir")
        done < "$tmp"
        rm -f "$tmp"
    fi
    
    # Process each directory
    for dir in "${dirs[@]}"; do
        result=$(build_dir_rename "$dir" "$CURRENT_FOLDER")
        IFS='|' read -r old_path new_path <<< "$result"
        
        if [[ "$old_path" != "$new_path" ]] && [[ -d "$old_path" ]]; then
            rename_item "$old_path" "$new_path"
            changed=true
            
            # Update CURRENT_FOLDER if it was the root that changed
            if [[ "$old_path" == "$CURRENT_FOLDER" ]]; then
                CURRENT_FOLDER="$new_path"
            fi
        fi
    done
done

# Rename root folder if needed (check at end in case it wasn't processed)
base=$(basename "$CURRENT_FOLDER")
new_base=$(apply_mappings "$base")
if [[ "$base" != "$new_base" ]]; then
    new_root="$(dirname "$CURRENT_FOLDER")/$new_base"
    if [[ "$CURRENT_FOLDER" != "$new_root" ]] && [[ -d "$CURRENT_FOLDER" ]]; then
        rename_item "$CURRENT_FOLDER" "$new_root"
        CURRENT_FOLDER="$new_root"
    fi
fi

# Step 2: Find all files AFTER directory renaming is complete
declare -a files

# Find files in the current (possibly renamed) folder
while IFS= read -r -d '' item; do
    [[ -f "$item" ]] && files+=("$item")
done < <(find "$CURRENT_FOLDER" -type f -print0)

# Rename all files
for file in "${files[@]}"; do
    new_file=$(build_new_path "$file" "$CURRENT_FOLDER")
    [[ "$file" != "$new_file" ]] && rename_item "$file" "$new_file"
done

[[ "$DRY_RUN" == true ]] && echo -e "\nDry run complete." || echo -e "\nRenaming complete."
