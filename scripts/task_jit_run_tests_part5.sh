#!/bin/bash

set -eo pipefail
set -x
: ${MAX_JOBS:=$(nproc)}
: ${CUDA_VISIBLE_DEVICES:=0}
: ${SKIP_INSTALL:=0}

if [ "$SKIP_INSTALL" = "0" ]; then
  pip install -e . -v
fi

# Run each test file separately to isolate CUDA memory issues
pytest -s tests/utils/test_logits_processor.py
