"""Offline knob tuner for the SM90 pull-style W4A16 mega kernel (``--dtype sm90_bf16_nvfp4``).

The W4A16 backend runs the FP8 pull-style kernel with BF16 operands and the
``weight_format="nvfp4"`` mainloop, so the sweep is the FP8 tuner's: CLI
dtype ``sm90_bf16_nvfp4`` maps to the shim kind ``"bf16"`` with NVFP4
weights, candidates start from the W4A16 heuristic rows and are swap-AB only,
and winners are recorded under the knob-cache dtype ``"bf16_nvfp4"``.
"""

from __future__ import annotations

from ..fp8_fp8_bf16_pull_cutedsl.tuner import run_tuning, tune_one

__all__ = ["run_tuning", "tune_one"]
