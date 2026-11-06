#!/usr/bin/env bash
# Redirect: this script moved to scripts/infra/run_with_test_sccache.sh.
# Kept until the GitLab CI setup uses the new location; then delete.
_redirect_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    # shellcheck disable=SC1091
    source "${_redirect_dir}/infra/run_with_test_sccache.sh" "$@"
else
    exec bash "${_redirect_dir}/infra/run_with_test_sccache.sh" "$@"
fi
