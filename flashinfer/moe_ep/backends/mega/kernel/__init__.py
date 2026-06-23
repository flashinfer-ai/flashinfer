"""Mega-path compute backends (fused comm + local MoE)."""

from . import deep_gemm_mega, nvfp4_cutedsl

__all__ = ["deep_gemm_mega", "nvfp4_cutedsl"]
