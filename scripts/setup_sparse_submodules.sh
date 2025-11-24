#!/bin/bash
# Script to set up submodules with sparse checkout
# Called during pip install process


# Example usage:
# setup_sparse_submodule "submodule-name" "https://github.com/user/repo.git" "path1/" "path2/specific/" "path3/"

set -e

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

setup_sparse_submodule() {
    local submodule_name=$1
    local repo_url=$2
    shift 2
    local sparse_paths=("$@")

    local submodule_path="3rdparty/$submodule_name"

    # Check if submodule exists in .gitmodules
    if ! git config --file .gitmodules --get "submodule.$submodule_path.url" >/dev/null 2>&1; then
        git submodule add "$repo_url" "$submodule_path" 2>/dev/null || {
            # If submodule already exists in the index but not in .gitmodules
            git config --file .gitmodules "submodule.$submodule_path.path" "$submodule_path"
            git config --file .gitmodules "submodule.$submodule_path.url" "$repo_url"
        }
    fi

    # Check if submodule directory exists and has content
    if [ -d "$submodule_path/.git" ] && [ -n "$(ls -A $submodule_path 2>/dev/null)" ]; then
    else
        # Initialize the submodule
        git submodule update --init --depth=1 "$submodule_path"
    fi

    # Enter submodule directory
    cd "$submodule_path"

    # Initialize sparse checkout with cone mode if not already initialized
    if ! git sparse-checkout list >/dev/null 2>&1; then
        git sparse-checkout init --cone
    fi

    # Set the sparse paths
    git sparse-checkout set "${sparse_paths[@]}"

    # Apply sparse checkout (this ensures the working tree matches sparse-checkout config)
    git read-tree -m -u HEAD

    cd "$REPO_ROOT"
}

# Function to setup sparse checkout from .gitmodules info
setup_sparse_from_gitmodules() {
    local submodule_name=$1
    shift
    local sparse_paths=("$@")

    # Get URL from .gitmodules
    local submodule_path="3rdparty/$submodule_name"
    local repo_url=$(git config --file .gitmodules --get "submodule.$submodule_path.url")

    if [ -z "$repo_url" ]; then
        echo "Error: Submodule $submodule_name not found in .gitmodules"
        return 1
    fi

    setup_sparse_submodule "$submodule_name" "$repo_url" "${sparse_paths[@]}"
}
