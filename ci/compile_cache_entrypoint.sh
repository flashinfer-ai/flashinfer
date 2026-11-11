#!/usr/bin/env bash
set -euo pipefail

# The seed is read-only; all test-generated artifacts go to the job's private
# bind mount. Preserve timestamps for consumers with mtime-based validation.
seed=$1
destination=$2
shift 2
s3_tree=${COMPILE_CACHE_S3_TREE:-}

# A copy can stop midway, typically on a full disk. A module copied in part
# would load a truncated object, so report the sizes involved, drop
# everything copied and run cold.
copy_failed() {
    echo "::warning title=Compile cache::cannot copy the compile cache into ${destination}; running without it"
    echo "Size of each compile cache source (MB):"
    du -sm "$@" 2>/dev/null || true
    echo "Space at ${destination} (MB):"
    df -m "${destination}" 2>/dev/null || true
    if ! find "${destination}" -mindepth 1 -delete; then
        echo "::error title=Compile cache::cannot empty the half-copied ${destination}"
        exit 1
    fi
}

if [ -n "${s3_tree}" ] && [ -d "${s3_tree}" ]; then
    # The trusted S3 tarball first; the per-runner seed only fills its gaps,
    # never replacing a module whose meta.json differs.
    if ! { cp -a "${s3_tree}/." "${destination}/" \
        && bash "$(dirname "$0")/compile_cache_s3.sh" merge "${seed}" "${destination}"; }; then
        copy_failed "${s3_tree}" "${seed}"
    fi
elif [ -d "${seed}" ]; then
    cp -a "${seed}/." "${destination}/" || copy_failed "${seed}"
fi
exec "$@"
