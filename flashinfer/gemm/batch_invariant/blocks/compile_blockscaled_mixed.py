# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile entry of the mixed MX classes (``gemm.mxfp4w8a.kfold`` / ``kchunk[C]`` on the sm_100 / sm_103 instruction,
``gemm.mxfp4w8a.kfold64`` / ``kchunk64[C]`` on the sm_107 one; ``blocks/blockscaled_mx_mixed.py``): the shared block-scaled
compile closure (``compile_blockscaled.py``) with the mixed plans, in the cache module ``batch_invariant_gemm_blockscaled_mixed``
rooted here, so an edit to the mixed plans leaves the one-format block-scaled module's objects valid (the shared body stays in
both closures). Operands: A (M, K) e4m3 with K contiguous, B (N, K / 2) the byte-packed e2m1 weight, SFA / SFB flat uint8
buffers in the 128x4 swizzled layout, C (M, N) N-major, ``alpha`` a 1-element fp32 tensor; under the swapped decode record
the kernel's A is the packed weight and its B the e4m3 tokens (the API hands them over exchanged).
"""

from __future__ import annotations

import functools

from ..generator import Knobs
from .blockscaled_mx_mixed import plan_for
from .compile_blockscaled import compile_blockscaled

MODULE = "batch_invariant_gemm_blockscaled_mixed"


@functools.cache
def get_compiled_blockscaled_mixed(
    cfg: Knobs,
    chunks: int,
    clusters: int,
    in_fmt: str,
    out_fmt: str = "bf16",
    family: str = "",
):
    """``kernel(a3, b3, sfa, sfb, c3, alpha)`` of the mixed class ``in_fmt`` (the direct stores; the workspace split's
    partials are unscaled and the dense module's fixup applies ``alpha``), with the conventions of
    :func:`compile_blockscaled.get_compiled_blockscaled`."""
    return compile_blockscaled(
        cfg, chunks, clusters, in_fmt, out_fmt, MODULE, plan_for(in_fmt, cfg)
    )
