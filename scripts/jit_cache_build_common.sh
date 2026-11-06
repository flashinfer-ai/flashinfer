#!/usr/bin/env bash
# Redirect: this script moved to scripts/infra/jit_cache_build_common.sh.
# Kept until the GitLab CI setup uses the new location; then delete.
_redirect_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    # shellcheck disable=SC1091
    source "${_redirect_dir}/infra/jit_cache_build_common.sh" "$@"
else
    exec bash "${_redirect_dir}/infra/jit_cache_build_common.sh" "$@"
fi
