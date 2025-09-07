#!/bin/bash

set -eo pipefail
set -x

pytest -s tests/test_blackwell_fmha.py
pytest -s tests/test_deepseek_mla.py

# trtllm-gen
pytest -s tests/test_trtllm_gen_attention.py
pytest -s tests/test_trtllm_gen_mla.py

# cudnn
pytest -s tests/test_cudnn_decode.py
pytest -s tests/test_cudnn_prefill.py
pytest -s tests/test_cudnn_prefill_deepseek.py
