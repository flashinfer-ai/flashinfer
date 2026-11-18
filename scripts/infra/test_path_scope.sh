#!/bin/bash
# TEST_PATH intersection for dedicated CI scripts (multi-GPU / multi-node).
# Empty TEST_PATH keeps every file. Tokens are whitespace-separated.

test_path_selects_file() {
    local file="${1%/}"
    local token
    if [ -z "${TEST_PATH:-}" ]; then
        return 0
    fi
    # shellcheck disable=SC2086
    for token in ${TEST_PATH}; do
        token="${token%/}"
        if [ -z "$token" ]; then
            continue
        fi
        if [ "$file" = "$token" ]; then
            return 0
        fi
        case "$file" in
            "$token"/*) return 0 ;;
        esac
    done
    return 1
}

filter_files_by_test_path() {
    local selected=""
    local file
    # shellcheck disable=SC2086
    for file in $1; do
        if test_path_selects_file "$file"; then
            selected="${selected}${selected:+ }${file}"
        fi
    done
    printf '%s' "$selected"
}
