#!/usr/bin/env bash
# Redirect: this script moved to scripts/infra/task_test_jit_cache_package_build_import.sh.
# Kept until the GitLab CI setup uses the new location; then delete.
_redirect_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    # shellcheck disable=SC1091
    source "${_redirect_dir}/infra/task_test_jit_cache_package_build_import.sh" "$@"
else
    exec bash "${_redirect_dir}/infra/task_test_jit_cache_package_build_import.sh" "$@"
fi
