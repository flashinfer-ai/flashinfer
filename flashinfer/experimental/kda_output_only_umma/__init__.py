"""Experimental tcgen05 / UMMA backend of the KDA output-only (frozen-state) decode.

Selected with ``backend="umma"`` on
``flashinfer.kda_kernels.kda_decode_wy_output_only.kda_wy_output_only`` and
``kda_recoverssm_verify`` (see README.md). Kernels are JIT-compiled per process.
"""

from .launch import (
    kda_oo_umma_unavailable_reason,
    run_kda_oo_umma,
)

__all__ = ["kda_oo_umma_unavailable_reason", "run_kda_oo_umma"]
