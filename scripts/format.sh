#!/usr/bin/env bash

# Copyright (c) 2023 by FlashInfer team.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# File is modified from https://github.com/tile-ai/tilelang/blob/main/format.sh


# Usage:
#    # Do work and commit your work.

#    # Format files that differ from origin/main.
#    bash format.sh

#    # Commit changed files with message 'Run yapf and ruff'
#
#
# YAPF + Clang formatter (if installed). This script formats all changed files from the last mergebase.
# You are encouraged to run this locally before pushing changes for review.

# Cause the script to exit if a single command fails
set -eo pipefail

################################################################################
# If certain Python-based tools (yapf, ruff, codespell, clang-format) are not
# found or are not at the required version, install (or reinstall) them using pip.
# We remove references to requirements-dev.txt, as requested.
# Note: clang-tidy is often a system tool, so we do not attempt to pip-install it.
################################################################################

# Helper function to install or update YAPF
install_or_update_yapf() {
    local REQUIRED_VERSION="0.40.2"
    if ! command -v yapf &>/dev/null; then
        echo "yapf not found. Installing yapf==$REQUIRED_VERSION..."
        pip install "yapf==$REQUIRED_VERSION"
    else
        local INSTALLED_VERSION
        # 'yapf --version' outputs "yapf X.Y.Z"
        INSTALLED_VERSION=$(yapf --version | awk '{print $2}')
        if [[ "$INSTALLED_VERSION" != "$REQUIRED_VERSION" ]]; then
            echo "Incorrect yapf version ($INSTALLED_VERSION). Installing yapf==$REQUIRED_VERSION..."
            pip install "yapf==$REQUIRED_VERSION"
        fi
    fi
}

# Helper function to install or update Ruff
install_or_update_ruff() {
    local REQUIRED_VERSION="0.6.5"
    if ! command -v ruff &>/dev/null; then
        echo "ruff not found. Installing ruff==$REQUIRED_VERSION..."
        pip install "ruff==$REQUIRED_VERSION"
    else
        local INSTALLED_VERSION
        # 'ruff --version' outputs "ruff X.Y.Z"
        INSTALLED_VERSION=$(ruff --version | awk '{print $2}')
        if [[ "$INSTALLED_VERSION" != "$REQUIRED_VERSION" ]]; then
            echo "Incorrect ruff version ($INSTALLED_VERSION). Installing ruff==$REQUIRED_VERSION..."
            pip install "ruff==$REQUIRED_VERSION"
        fi
    fi
}

# Helper function to install or update Codespell
install_or_update_codespell() {
    local REQUIRED_VERSION="2.3.0"
    if ! command -v codespell &>/dev/null; then
        echo "codespell not found. Installing codespell==$REQUIRED_VERSION..."
        pip install "codespell==$REQUIRED_VERSION"
    else
        local INSTALLED_VERSION
        # 'codespell --version' outputs "codespell vX.Y.Z"
        INSTALLED_VERSION=$(codespell --version | awk '{print $2}' | sed 's/^v//')
        if [[ "$INSTALLED_VERSION" != "$REQUIRED_VERSION" ]]; then
            echo "Incorrect codespell version ($INSTALLED_VERSION). Installing codespell==$REQUIRED_VERSION..."
            pip install "codespell==$REQUIRED_VERSION"
        fi
    fi
}

# Helper function to install or update clang-format
# Note: This assumes that installing the Python package "clang-format" provides
#       a valid clang-format binary. If you're using system clang-format instead,
#       you may wish to remove or adjust this part.
install_or_update_clang_format() {
    local REQUIRED_VERSION="15.0.7"
    if ! command -v clang-format &>/dev/null; then
        echo "clang-format not found. Installing clang-format==$REQUIRED_VERSION..."
        pip install "clang-format==$REQUIRED_VERSION"
    else
        local INSTALLED_VERSION
        # 'clang-format --version' outputs "clang-format version X.Y.Z"
        # Typically, it's something like: "clang-format version 15.0.7 ..."
        # The third field is often the version number.
        INSTALLED_VERSION=$(clang-format --version | awk '{print $3}')
        if [[ "$INSTALLED_VERSION" != "$REQUIRED_VERSION" ]]; then
            echo "Incorrect clang-format version ($INSTALLED_VERSION). Installing clang-format==$REQUIRED_VERSION..."
            pip install "clang-format==$REQUIRED_VERSION"
        fi
    fi
}

################################################################################
# Install/update the required tools before proceeding
################################################################################

install_or_update_yapf
install_or_update_ruff
install_or_update_codespell
install_or_update_clang_format

################################################################################
# Main lint/format script logic
################################################################################

# Prevent 'git rev-parse' from failing if we run this from inside the .git directory
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

echo 'flashinfer yapf: Check Start'

YAPF_FLAGS=(
    '--recursive'
    '--parallel'
)

YAPF_EXCLUDES=(
    '--exclude' 'build/**'
    '--exclude' '3rdparty/**'
)

# Format specified files
format() {
    yapf --in-place "${YAPF_FLAGS[@]}" "$@"
}

# Format files that differ from main branch
format_changed() {
    if git show-ref --verify --quiet refs/remotes/origin/main; then
        BASE_BRANCH="origin/main"
    else
        BASE_BRANCH="main"
    fi
    MERGEBASE="$(git merge-base "$BASE_BRANCH" HEAD)"

    # Only format *.py and *.pyi files changed since merge-base
    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' \
            | xargs -P 5 yapf --in-place "${YAPF_EXCLUDES[@]}" "${YAPF_FLAGS[@]}"
    fi
}

# Format all Python files
format_all() {
    yapf --in-place "${YAPF_FLAGS[@]}" "${YAPF_EXCLUDES[@]}" .
}

# Decide which formatting approach to use
# if [[ "$1" == '--files' ]]; then
#     format "${@:2}"
# elif [[ "$1" == '--all' ]]; then
#     format_all
# else
#     format_changed
# fi

# echo 'flashinfer yapf: Done'

# TODO: enable yapf format check if we have a commonly recognized configuration
echo 'Skip yapf auto format'

echo 'flashinfer codespell: Check Start'

# Spelling check of specified files
spell_check() {
    codespell "$@"
}

# Check spelling of all (for example, in pyproject.toml or the entire repo)
spell_check_all(){
    codespell --toml pyproject.toml
}

# Check only changed Python files
spell_check_changed() {
    if git show-ref --verify --quiet refs/remotes/origin/main; then
        BASE_BRANCH="origin/main"
    else
        BASE_BRANCH="main"
    fi
    MERGEBASE="$(git merge-base "$BASE_BRANCH" HEAD)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs codespell
    fi
}

# Decide how to run codespell
if [[ "$1" == '--files' ]]; then
    spell_check "${@:2}"
elif [[ "$1" == '--all' ]]; then
    spell_check_all
else
    spell_check_changed
fi

echo 'flashinfer codespell: Done'

echo 'flashinfer ruff: Check Start'

# Lint specified files
lint() {
    ruff check "$@"
}

# Lint only changed Python files
lint_changed() {
    if git show-ref --verify --quiet refs/remotes/origin/main; then
        BASE_BRANCH="origin/main"
    else
        BASE_BRANCH="main"
    fi
    MERGEBASE="$(git merge-base "$BASE_BRANCH" HEAD)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs ruff check
    fi
}

# Decide how to run Ruff
if [[ "$1" == '--files' ]]; then
    lint "${@:2}"
elif [[ "$1" == '--all' ]]; then
    lint flashinfer tests
else
    lint_changed
fi

echo 'flashinfer ruff: Done'

echo 'flashinfer clang-format: Check Start'

# If clang-format is available, apply it
if command -v clang-format &>/dev/null; then

    CLANG_FORMAT_FLAGS=("-i")

    # Format given files
    clang_format() {
        clang-format "${CLANG_FORMAT_FLAGS[@]}" "$@"
    }

    # Format all C/C++ files
    clang_format_all() {
        find . -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) \
            -not -path "./3rdparty/*" \
            -not -path "./build/*" \
            -exec clang-format -i {} +
    }

    # Format changed C/C++ files relative to main branch
    clang_format_changed() {
        if git show-ref --verify --quiet refs/remotes/origin/main; then
            BASE_BRANCH="origin/main"
        else
            BASE_BRANCH="main"
        fi
        MERGEBASE="$(git merge-base "$BASE_BRANCH" HEAD)"

        if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.c' '*.cc' '*.cpp' '*.h' '*.hpp' &>/dev/null; then
            git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.c' '*.cc' '*.cpp' '*.h' '*.hpp' \
                | xargs clang-format -i
        fi
    }

    if [[ "$1" == '--files' ]]; then
        clang_format "${@:2}"
    elif [[ "$1" == '--all' ]]; then
        clang_format_all
    else
        clang_format_changed
    fi
else
    echo "clang-format not found. Skipping C/C++ formatting."
fi

echo 'flashinfer clang-format: Done'

################################################################################
# Check if any files were modified by the above operations
# If so, prompt the user to review and stage them
################################################################################
if ! git diff --quiet &>/dev/null; then
    echo 'Reformatted files. Please review and stage the changes.'
    echo 'Changes not staged for commit:'
    echo
    git --no-pager diff --name-only

    exit 1
fi

echo 'flashinfer: All checks passed'
