#!/usr/bin/env bash
set -euo pipefail

# The seed is read-only; all test-generated artifacts go to the job's private
# bind mount. Preserve timestamps for consumers with mtime-based validation.
seed=$1
destination=$2
shift 2
s3_tree=${COMPILE_CACHE_S3_TREE:-}
if [ -n "${s3_tree}" ] && [ -d "${s3_tree}" ]; then
    # The trusted S3 tarball first; the per-runner seed only fills its gaps,
    # never replacing a module whose meta.json differs.
    cp -a "${s3_tree}/." "${destination}/"
    bash "$(dirname "$0")/compile_cache_s3.sh" merge "${seed}" "${destination}"
elif [ -d "${seed}" ]; then
    cp -a "${seed}/." "${destination}/"
fi
exec "$@"
