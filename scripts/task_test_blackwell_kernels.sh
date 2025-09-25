#!/bin/bash

set -eo pipefail
set -x

: ${JUNIT_DIR:=$(realpath ./junit)}
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}

pip install -e . -v
pip install --upgrade nvidia-cudnn-cu12
pip install --upgrade cuda-python==12.*

EXIT_CODE=0

pytest tests/ || EXIT_CODE=1

exit $EXIT_CODE
