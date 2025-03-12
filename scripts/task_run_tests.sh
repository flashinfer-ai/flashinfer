#!/bin/bash

set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}

pip install dist/*.whl

pytest -v tests/test_norm.py
