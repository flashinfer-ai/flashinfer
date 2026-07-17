"""Mega-path compute backends (fused comm + local MoE)."""

from . import deep_gemm_mega, mxfp8_cutedsl, nvfp4_cutedsl, sm90_push_fp8

__all__ = ["deep_gemm_mega", "mxfp8_cutedsl", "nvfp4_cutedsl", "sm90_push_fp8"]
