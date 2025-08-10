#!/bin/bash

set -eo pipefail
set -x

pytest -s tests/test_trtllm_gen_fused_moe.py
pytest -s tests/test_trtllm_cutlass_fused_moe.py
