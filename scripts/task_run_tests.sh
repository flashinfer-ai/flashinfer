#!/bin/bash

set -eo pipefail
set -x
: ${NUM_THREADS:=$(nproc)}

pytest -v tests/test_norm.py
