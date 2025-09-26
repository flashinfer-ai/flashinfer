#!/bin/bash

set -eo pipefail
set -x

: ${JUNIT_DIR:=$(realpath ./junit)}
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}

pip install -e . -v

EXIT_CODE=0

pytest tests/ || EXIT_CODE=1

exit $EXIT_CODE
