"""Mega-path compute backends (fused comm + local MoE)."""

from . import bf16_cutedsl, deep_gemm_mega, mxfp8_cutedsl, nvfp4_cutedsl

__all__ = ["deep_gemm_mega", "bf16_cutedsl", "mxfp8_cutedsl", "nvfp4_cutedsl"]
