#!/bin/bash

set -eo pipefail
set -x

pytest -s tests/test_mm_fp4.py
pytest -s tests/test_groupwise_scaled_gemm_fp8.py
pytest -s tests/test_groupwise_scaled_gemm_mxfp4.py
