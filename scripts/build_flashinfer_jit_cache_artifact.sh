#!/usr/bin/env bash
# Redirect: this script moved to scripts/infra/build_flashinfer_jit_cache_artifact.sh.
# Kept until the GitLab CI setup uses the new location; then delete.
_redirect_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    # shellcheck disable=SC1091
    source "${_redirect_dir}/infra/build_flashinfer_jit_cache_artifact.sh" "$@"
else
    exec bash "${_redirect_dir}/infra/build_flashinfer_jit_cache_artifact.sh" "$@"
fi
