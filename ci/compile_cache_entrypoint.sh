#!/usr/bin/env bash
set -euo pipefail

# The seed is read-only; all test-generated artifacts go to the job's private
# bind mount. Preserve timestamps for consumers with mtime-based validation.
seed=$1
destination=$2
shift 2
if [ -d "${seed}" ]; then
    cp -a "${seed}/." "${destination}/"
fi
exec "$@"
