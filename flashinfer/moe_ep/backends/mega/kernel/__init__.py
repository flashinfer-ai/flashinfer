"""Mega-path compute backends (fused comm + local MoE)."""

from . import deep_gemm_mega, mxfp8_cutedsl, nvfp4_cutedsl

__all__ = ["deep_gemm_mega", "mxfp8_cutedsl", "nvfp4_cutedsl"]
