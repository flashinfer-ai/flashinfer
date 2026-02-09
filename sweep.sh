#!/bin/bash
  # Issue #2356 magnitude-accuracy sweep

  TEST="tests/moe/test_dpsk_fused_moe_fp8.py::test_correctness_dpsk_fp8_fused_moe[NoShuffle_MajorK-False-Qwen3_480B-640-4-0-False]"

  for fraction in 0.1 0.3 0.45 0.7; do
    for magnitude in 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001; do
      echo "========================================"
      echo "fraction=$fraction, magnitude=$magnitude"
      echo "========================================"
      SMALL_SCALE_FRACTION=$fraction SMALL_SCALE_MAGNITUDE=$magnitude \
        pytest "$TEST" -s 2>&1 | grep -E "(small_scale_|Comparison stats)" -A6
    done
  done
