# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``qsa_topk``: the DSL-free host entry of the block-sparse indexer's decode top-k kernel (``kernels/qsa_topk_decode.py``).

The kernel module imports the CuTe DSL and ``cudnn.frost.tile_dsl`` at module scope, so importing it on a DSL below the FROST floor
fails inside the DSL (a bare ``ModuleNotFoundError`` for ``cutlass.experimental`` on the 4.6.2 wheel the serving stacks pin) before
any gate can name the version.  This module is the entry a caller hits (Rule 7): it imports nothing of the DSL, gates the installed
version FIRST and imports the kernel module lazily, only once the gate passes -- a too-old DSL reads as a typed
``NotImplementedError`` naming the version, never as an error from inside the DSL.  Everything else (the geometry, the device's
target, cc 9.0+ for clusters and PDL, the shared-memory limit, the form checks) is the kernel module's and is documented there.

Contract and knobs: ``kernels/qsa_topk_decode.py`` -- its module docstring carries the pinned input contract (``counts`` is the
VISIBLE count per row, never the list length), the two pre-development tables and the degenerate-input matrix.  The plan-time /
per-execute split an engine binds (``compile_qsa_topk_decode`` / ``run_qsa_topk_decode``) is reached through
:func:`qsa_topk_decode_kernel_module`, so the binding inherits the gate.
"""

from __future__ import annotations

import importlib
from typing import Optional

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_requirement_error

__all__ = ["QSA_TOPK_DECODE_KERNEL_MODULE", "qsa_topk_decode", "qsa_topk_decode_kernel_module"]

QSA_TOPK_DECODE_KERNEL_MODULE = 'flashinfer.experimental.qwen_3_5_block._cudnn_frost.gated_attention_block.kernels.qsa_topk_decode'
"""The kernel module's dotted path -- imported lazily by :func:`qsa_topk_decode_kernel_module`, never at import time here."""


def qsa_topk_decode_kernel_module():
    """The kernel module, imported once the CuTe DSL floor is met (Rule 7: the gate runs HERE, before the import).

    Raises ``NotImplementedError`` naming the installed version when it is below the floor.  The module exposes the plan-time /
    per-execute split (``compile_qsa_topk_decode`` / ``run_qsa_topk_decode``), the knob domains (``CLUSTER_SIZES``,
    ``THREAD_COUNTS``) and the geometry helpers (``cols_cap_for``, ``smem_bytes_for``).
    """
    too_old = cutedsl_requirement_error("qsa_topk_decode")
    if too_old is not None:
        raise NotImplementedError(too_old)
    return importlib.import_module(QSA_TOPK_DECODE_KERNEL_MODULE)


def qsa_topk_decode(
    logits,
    counts,
    block_ids,
    *,
    cluster_ctas: Optional[int] = None,
    threads_per_cta: Optional[int] = None,
    use_pdl: bool = False,
    stream=None,
):
    """Compile (cached) and run once -- the convenience form of the kernel module's ``qsa_topk_decode``, behind the version gate.

    ``logits [T, n_blocks_max]`` fp32, ``counts [T]`` int32 (the VISIBLE count per row, never the list length), ``block_ids
    [T, top_k]`` int32 -- the CALLER's buffer, written in canonical order (ids strictly descending, ``-1`` past
    ``min(top_k, counts[t])``) and returned.  The knobs and the stream are the kernel module's.
    """
    return qsa_topk_decode_kernel_module().qsa_topk_decode(
        logits, counts, block_ids, cluster_ctas=cluster_ctas, threads_per_cta=threads_per_cta, use_pdl=use_pdl, stream=stream
    )
