#!/bin/bash

set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}

pip install -e . -v

pytest -v tests/test_norm.py
