"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Cake backend: chunked large-vocabulary LM-head projection + loss with a
memory-bounded backward for training on SM100 / SM103
(flashinfer-ai/flashinfer#5680).

Contract
--------
* ``X [T, H]`` BF16 hidden states (row-major; any leading stride ``ld_X >= H``
  whose byte pitch is a multiple of 16 -- otherwise a contiguous copy is made
  and reported), ``W [V, H]`` BF16 output weight (contiguous), ``labels [T]``
  int64 with ``-100`` = ignored.  ``V`` and ``H`` must be multiples of 256
  (the GEMM tiles; GLM-class geometry ``H = 6144``, ``V = 154880``).  ``T`` is
  arbitrary (irregular, not a multiple of the chunk); the token chunk ``C``
  (default 4096) bounds every vocabulary-sized intermediate.
* Precision boundary: BF16 GEMM output ``z = bf16(X @ W^T)``, promoted to FP32
  for max / log-sum-exp / loss arithmetic; ``dlogits`` BF16; ``dW`` accumulated
  in FP32 across chunks and cast ONCE at the output boundary (BF16 by default,
  FP32 on request); ``dX`` accumulated in FP32 and cast once to BF16.
* ``logp_t = z[t, y_t] - logsumexp_v z[t, v]`` for valid rows, 0 for ignored
  rows.  Cross-entropy: ``loss = -sum(logp[valid]) / loss_div`` with a
  caller-supplied positive ``loss_div`` (never a local token mean);
  ``d_t = -1 / loss_div`` on valid rows.  Policy: ``loss = -sum(w_t * min(
  exp(logp_t - infer_logp_t), 2))`` over valid rows (``w`` FP32 signed, already
  masked and normalized); ``d_t = -w_t * ratio_t`` when ``ratio_t <= 2`` and
  ``0`` when ``ratio_t > 2``.  ``dz[t, v] = d_t * (1[v = y_t] - softmax(z_t)_v)``;
  ignored rows: zero loss, zero gradient, ``logp = 0``.  ``dX = dz @ W``,
  ``dW = dz^T @ X`` chunk by chunk.
* Entry (a) ``chunked_lm_head_loss``: the forward produces ``dX_acc`` (FP32
  ``[T, H]``) and ``dW_acc`` (FP32 ``[V, H]``) directly per chunk (three GEMMs,
  no logits recomputation); the backward applies the incoming scalar
  gradient ``g`` and performs the single cast ``dX = bf16(g * dX_acc)``,
  ``dW = cast(g * dW_acc)`` -- never mutating the saved accumulators, so a
  retained graph may run the backward repeatedly.  Entry (b)
  ``chunked_lm_head_logprob``: the forward saves only the FP32 row statistic
  ``lse[T]``; the backward recomputes each chunk's logits (four GEMMs) with the
  incoming ``dlogp`` as ``d_t`` (masked to valid rows).
* Memory rule: no logits / dlogits / probability buffer spans more than ``C``
  tokens (a batch smaller than ``C`` is one chunk).  :func:`memory_report`
  states the peak temporary bytes separately from the model weights, the
  required outputs and the FP32 gradient accumulators.
* Valid-row compaction (``compact_rows``, default on unless
  ``FLASHINFER_CAKE_LM_HEAD_LOSS_COMPACT_ROWS=0``): when some labels are
  ``-100`` the chunk loop runs over the valid rows only -- the row index
  ``idx = (labels >= 0).nonzero()`` is formed once per call (one device
  synchronization for the count, one for the index), each chunk's rows of
  ``X`` are gathered into one reusable BF16 ``[C, H]`` buffer, the row operands
  are compacted to ``[T_v]``, the FP32 ``dX`` accumulator is ``[T_v, H]`` and
  ``logp`` / ``dX`` are scattered back to ``[T]`` with exact zeros on the
  ignored rows.  Ignored rows contribute exactly zero to the loss and both
  gradients, so the result is the same computation over fewer rows: per row
  ``logp`` / ``lse`` are bitwise those of the uncompacted path, ``dX`` rows too
  whenever the chunk's K-slice count agrees; the ``loss`` and ``dW``
  reductions run over different chunk boundaries, hence differ by FP32
  rounding.  All rows valid: the uncompacted path (no gather, no scatter).
* Fused dX finalize (default on unless ``FLASHINFER_CAKE_LM_HEAD_LOSS_DX_FINALIZE=0``):
  the K-slice slabs of a sliced dX GEMM are added into ``dX_acc`` by one
  fixed-order kernel (``slab_sum``) instead of the host's chain of in-place
  ``add_`` launches, and a compacted call's dX output boundary writes the
  ``[T, H]`` BF16 ``dX`` in one pass (``scale_cast_scatter_bf16``: ``bf16(g *
  dX_acc[compact row])`` on the valid rows, exact zeros elsewhere, addressed by
  the inclusive int32 scan of the valid-row mask) instead of the flat cast, the
  zero fill and the ``index_copy_``.  The same FP32 operations in the same
  order: ``dX`` is bitwise the previous path's (``0`` restores that path).
* dW side stream (``FLASHINFER_CAKE_LM_HEAD_LOSS_DW_STREAM``: ``auto`` (default)
  = calls of three or more chunks, ``1`` = every multi-chunk call, ``0`` =
  never): each chunk's weight-gradient accumulate GEMM is launched on a
  per-device side stream forked after the chunk's ``row_grad`` (``dz`` ready)
  and joined before the next chunk reuses the chunk buffer and before the call
  returns, so its CTAs fill the tail wave of the chunk's dX GEMM instead of
  queueing behind it.  Kernels, buffers, launch order per kernel and numerics
  are unchanged (the dX GEMM writes the dX rows / slabs, the accumulate reads
  the same ``dz`` and read-modify-writes ``dW_acc``): ``loss`` / ``logp`` /
  ``dX`` / ``dW`` are bitwise the single-stream path's.  Two-chunk calls pay
  the join without an overlap gain and a one-chunk call defers its only
  accumulate to the backward, hence the ``auto`` rule (:func:`dw_side_stream`).
* ``deterministic=True`` (the only mode this backend serves): fixed sequential
  chunk order, no atomics -- bitwise reproducible ``loss``, ``logp``, ``dX``
  and ``dW`` across runs.

Every allocation happens in :func:`prepare_lm_head_loss`; the returned runner
launches with no CUDA allocation and no host synchronization, so a runner (or
a CUDA graph capturing it) replays for new values written into the bound
tensors.  Kernels are reached through the argument plans of the registry
records in ``cake_jit.MODULES``; the ``abi`` field of a record names the
keyword set its kernels expect (see :data:`ABI_CONTRACT` and
:func:`stage_values`).  ``backend="reference"`` runs the same chunk loop with
PyTorch operators at the same rounding boundaries (a host-layer development
aid that needs no generated program and no CUDA device).

The eager entry points (:func:`forward_loss`, :func:`backward_loss`,
:func:`forward_logprob`, :func:`backward_logprob` and the autograd Functions
behind the public API) validate and bind once per *input binding* and remember
the result in :data:`BINDING_CACHE` (see :class:`BindingCache`): a later call
whose inputs have the same ``(data_ptr, shape, stride, dtype)`` and options
slots the current tensors, freshly allocated outputs and per-call scratch into
the remembered argument plans and launches.  A call without rows (``T == 0``)
returns loss 0, an empty ``logp`` and zero gradients without binding or
launching anything.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import json
import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

import torch

from .cake_jit import (
    BASE_STAGES,
    GEMM_STAGES,
    MODULES,
    STAGES,
    load_cake_lm_head_loss_module,
    registered_stages,
    select_module,
)

IGNORE_INDEX = -100
DEFAULT_CHUNK_SIZE = 4096
RATIO_CLIP = 2.0
OBJECTIVES = ("ce", "policy")
ENTRIES = ("loss", "logprob")
BACKENDS = ("cake", "reference")
GRAD_WEIGHT_DTYPES = (torch.bfloat16, torch.float32)
# ``mode`` scalar of ``row_finalize`` / ``loss_reduce``: which per-row quantities the stage forms.
MODE_CE, MODE_POLICY, MODE_EXTERNAL, MODE_NONE = 0, 1, 2, 3
WORKSPACE_ALIGN = 256
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}

# Host binding profile: the keyword set of the generated kernels (see
# :func:`stage_values`: the host provides, per stage, the kernel's own argument
# names; a kernel's argument plan selects from them).  A record with another
# ``abi`` (a plumbing placeholder program) is not served by this backend.
ABI_CONTRACT = "lm_head_loss_v1"
SUPPORTED_ABIS = (ABI_CONTRACT,)

# Geometry the kernels of a record were built for (record field ``geometry``;
# the defaults are the GLM-class contract).
GEOMETRY_DEFAULTS: dict[str, Any] = dict(
    stats_tile=256,  # vocabulary columns per (max, sum-exp) partial: stats[C, V / stats_tile, 2]
    row_tile=128,  # output rows per CTA of the GEMMs (m_tiles = ceil(rows / row_tile), rounded to the CTA pair)
    cta_group=2,  # CTAs per GEMM cluster (the pair takes adjacent row tiles)
    k_block=64,  # K step of the GEMMs (k_iters of the weight-gradient GEMM = ceil(rows_c / k_block))
    cast_vec=8,  # elements per vector of the scale-cast kernels (num_vecs = numel / cast_vec)
    vocab_multiple=256,  # V % vocab_multiple == 0
    hidden_multiple=256,  # H % hidden_multiple == 0
    ld_multiple=8,  # row stride of X in elements (16 B TMA pitch)
    labels_dtype="int64",  # element type the row kernels read (``int32`` = one host cast per call)
    hidden=None,  # the GEMM instances are specialized to one H (None = any multiple of hidden_multiple)
    vocab=None,  # ... and one V (None = any multiple of vocab_multiple)
    # CTAs per thread-block cluster of each GEMM (the cluster takes adjacent row tiles, so ``m_tiles`` rounds up to it;
    # 2 = the CTA pair, 4 = two pairs sharing the B operand by multicast).  Checked against the module's launch cluster:
    # a module launched with a preferred cluster dimension (two pairs sharing the A operand where the GPC topology allows,
    # a pair elsewhere) declares its required / fallback cluster here, and its grid rule counts the wider work items.
    logits_cluster_ctas=2,
    dx_cluster_ctas=2,
    dw_cluster_ctas=2,
)
LABEL_DTYPES = {"int64": torch.int64, "int32": torch.int32}
_GEOMETRY_OPTIONAL = ("hidden", "vocab")

# Values the host provides to every stage of one chunk (the kernels' own
# argument names; the argument plan of a stage selects from them).  Tensors:
# ``A`` / ``B`` / ``C`` / ``STATS_OUT`` of the GEMMs (per stage: X_c / W / z_c /
# stats; dz_c / W / dX_acc rows; dz_c / X_c / dW_acc), ``stats``, ``z`` (the
# ``[C, V]`` BF16 workspace: ``z_c`` then ``dz_c`` in place), full-length ``[T]``
# vectors ``labels`` / ``lse`` / ``logp`` / ``infer_logp`` / ``loss_weights`` /
# ``d_in`` (indexed by ``row0 + r``), chunk-local ``d`` / ``term`` (``[C]``), the
# ``loss_acc`` / ``loss_out`` cells and, for the casts, ``acc`` / ``g`` / ``out``.
# Scalars: ``M`` / ``m_tiles`` / ``k_iters`` / ``first_chunk`` of the GEMMs,
# ``rows_c`` / ``row0`` / ``V`` / ``num_tiles`` / ``mode`` / ``loss_div`` /
# ``last_chunk`` / ``d_off`` of the row kernels, ``num_vecs`` of the casts, and
# ``T`` / ``H`` / ``C`` for grid rules.
STAGE_TENSORS = {
    **{stage: ("A", "B", "C", "STATS_OUT", "WS") for stage in GEMM_STAGES},
    "row_finalize": (
        "stats",
        "z",
        "labels",
        "infer_logp",
        "loss_weights",
        "d_in",
        "lse",
        "logp",
        "d",
        "term",
    ),
    "loss_reduce": ("term", "loss_acc", "loss_out"),
    "row_grad": ("z", "labels", "lse", "d"),
    "scale_cast_bf16": ("acc", "g", "out"),
    "scale_cast_f32": ("acc", "g", "out"),
    "slab_sum": ("dx", "ws"),
    "scale_cast_scatter_bf16": ("acc", "g", "scan", "idx_lo", "out"),
}
COMMON_TENSORS = ("workspace", "tma_descriptor_workspace")
COMMON_SCALARS = (
    "rows_c",
    "row0",
    "T",
    "H",
    "V",
    "chunk",
    "num_tiles",
    "mode",
    "loss_div",
    "first_chunk",
    "last_chunk",
    "d_off",
    "ws_slab",
)
# K-sliced forms of the dX GEMM: ``gemm_dx_s<S>`` runs ``S`` K-slice work items per output tile (a
# persistent grid with ``S`` x more items fills the last wave); slice 0 writes ``dX_acc``, slice ``s >= 1``
# writes slab ``s - 1`` of the FP32 workspace ``WS [S - 1, rows, H]`` (``ws_slab`` = elements between
# slabs) and the host adds the slabs into ``dX_acc`` in fixed slab order (one RN add per element per
# slab, no atomics).  A record registers a contiguous prefix of these; the slice count of a chunk is
# chosen from its row count and the SM count (:func:`recommended_k_slices`).  ``WS`` of the other
# GEMMs is an unused 16-float dummy (``ws_slab`` 0).
DX_SLICE_STAGES = ("gemm_dx_s2", "gemm_dx_s3", "gemm_dx_s4")
# Fused weight-gradient output: the LAST chunk's ``dz_c^T @ X_c`` GEMM run in the backward (where the upstream scalar
# gradient ``g`` is known) with the scale and the output cast fused into its epilogue, ``C = cast(g * (WS + tile))`` --
# or ``cast(g * tile)`` for a one-chunk plan, which then has no FP32 ``[V, H]`` accumulator at all.  ``WS`` is the FP32
# sum of the chunks before it (read, never written), ``STATS_OUT`` the ``[1]`` FP32 scale cell.  The same FP32
# operations in the same order as ``gemm_dw_acc`` followed by ``scale_cast_*``, so the weight gradient is bitwise
# identical; the flat ``scale_cast`` pass over ``dW`` disappears.  Selected by ``fuse_dw_cast`` (default on).
DW_CAST_STAGES = ("gemm_dw_cast_bf16", "gemm_dw_cast_f32")
# Fused dX finalize (``dx_finalize``, default on): ``slab_sum`` adds the FP32 K-slice slabs of a sliced dX GEMM into
# the chunk's rows of ``dX_acc`` in ascending slab order in one launch (one RN add per element per slab -- the
# evaluation order of the host's ``add_`` chain, ``dX_acc`` read and written once instead of once per slab), and
# ``scale_cast_scatter_bf16`` writes a compacted call's ``[T, H]`` BF16 ``dX`` in one pass: ``bf16(g * dX_acc[scan[r]
# - 1])`` on a valid output row ``r`` (``scan`` = the inclusive int32 count of valid rows, so ``idx[scan[r] - 1] == r``
# exactly on the valid rows), exact zeros elsewhere -- the flat ``scale_cast``, the zero fill and the ``index_copy_``
# of the compact rows in one kernel that writes every output element once.  Bitwise the same ``dX``.
DX_FINALIZE_STAGES = ("slab_sum", "scale_cast_scatter_bf16")
K_SLICE_PENALTY = 0.01  # wave-efficiency score penalty per extra slab (the kernels' fitted per-slab cost share)

# --------------------------------------------------------------------------- per-chunk instance rules
# The GEMM instances of a record come in VARIANTS, one stage name per rule output, selected per chunk by the same
# rules the production launchers of the kernel source apply (same thresholds; an explicit knob wins) -- never by the
# shape itself, so a record built for a geometry carries exactly the variants its rules can reach:
# * raster height (``_g<group_m>``): at H >= RASTER_RULE_MIN_HIDDEN a chunk of >= RASTER_RULE_MIN_ROWS rows runs the
#   logits GEMM with the shorter grouped raster and the weight-gradient GEMMs (accumulate and fused cast) with the
#   taller one (RASTER_WIDE_GROUPS per architecture); below that H, a chunk of >= LOGITS_LONG_RASTER_MIN_ROWS rows (31
#   row tiles of 128) runs the logits GEMM with the shorter raster (LOGITS_LONG_RASTER_GROUPS per architecture, ``_g16``)
#   and a chunk of >= DW_LONG_CHUNK_MIN_ROWS rows runs the weight-gradient GEMMs with the long-chunk raster
#   (DW_LONG_CHUNK_GROUP_M, ``_g16``: the same height on every supported architecture); every other chunk -- the
#   default geometry's chunks of up to 30 row tiles, short tail chunks -- keeps the default raster.  Bitwise: the raster
#   changes the tile order only.
# * dX tile (``_tn256``): the 512-column pair tile unless H % 512 != 0 or the launch's wide work items would not fill
#   the device's SM pairs (:func:`dx_tile_rule`, evaluated after the K-slice count), or -- on an architecture listed in
#   DX_WIDE_MIN_EFF -- a one-slice launch below RASTER_RULE_MIN_HIDDEN whose wide items fill the SM pairs' waves below that
#   floor (the badly quantized chunk tails); the K-slice forms (``_s<k>``) come from :func:`recommended_k_slices` as
#   before.  Bitwise: the same slice boundaries and per-element K order on both tiles.
# * dX operand ring (``_st3``): below RASTER_RULE_MIN_HIDDEN a chunk of >= DW_LONG_CHUNK_MIN_ROWS rows runs the 512-wide dX
#   tile with the 3-deep operand ring (:func:`dx_stages_variant`, DX_LONG_CHUNK_STAGES per architecture); the 256-wide
#   fallback keeps its depth.  Bitwise: the ring depth only changes when a stage is refilled.
# * dW 2-D blocked raster (``_gn12``): below RASTER_RULE_MIN_HIDDEN a chunk of DW_BLOCK_MIN_ROWS .. DW_BLOCK_MAX_ROWS rows
#   runs the weight-gradient accumulate and the fp32 fused cast with column blocks of DW_BLOCK_GROUPS[arch] tiles
#   (:func:`dw_block_variant`; the bf16 cast keeps the 1-D raster).  Bitwise: a permutation of the same tiles.
# ``gemm_tuning`` (:func:`prepare_lm_head_loss`; ``$FLASHINFER_CAKE_LM_HEAD_LOSS_GEMM_TUNING`` as JSON for the eager
# entry points) pins knobs explicitly per GEMM -- ``{"logits": {"group_m": 16}, "dx": {"tile_n": 256, "k_slices": 2},
# "dw": {"group_m": 32, "group_n": 0}}``; a knob given as ``None`` pins the default form -- and wins over the
# rule for the knobs it names.  A variant the record does not register fails closed (``NotImplementedError``).
RASTER_RULE_MIN_HIDDEN = 7168  # first H whose [rows_c, H] BF16 operand panel no longer stays L2-resident at rows_c >= 2049
RASTER_RULE_MIN_ROWS = 2049
RASTER_WIDE_GROUPS = {
    "sm_100a": (16, 32),
    "sm_103a": (16, 32),
}  # (logits group_m, weight-gradient group_m) of the wide geometries, per architecture
DW_LONG_CHUNK_GROUP_M = 16  # weight-gradient group_m of the long chunks (>= DW_LONG_CHUNK_MIN_ROWS rows) below RASTER_RULE_MIN_HIDDEN
DW_LONG_CHUNK_GROUPS = {
    "sm_100a": DW_LONG_CHUNK_GROUP_M,
    "sm_103a": DW_LONG_CHUNK_GROUP_M,
}  # the same height on every supported architecture (the kernel source keeps a per-architecture table; the export mirrors it)
DW_LONG_CHUNK_MIN_ROWS = 4097
LOGITS_LONG_RASTER_GROUPS = {
    "sm_100a": 16,
    "sm_103a": 16,
}  # logits group_m of the GLM-class chunks of >= LOGITS_LONG_RASTER_MIN_ROWS rows below RASTER_RULE_MIN_HIDDEN, per architecture
LOGITS_LONG_RASTER_MIN_ROWS = (
    3841  # 30 row tiles of 128 + 1: the first chunk size with 31 row tiles
)
DX_LONG_CHUNK_STAGES = {
    "sm_100a": 3,
    "sm_103a": 3,
}  # operand-ring depth of the 512-wide dX tile at the long chunks (>= DW_LONG_CHUNK_MIN_ROWS rows) below RASTER_RULE_MIN_HIDDEN
DX_WIDE_TILE_STAGES = 4  # the 512-wide dX tile's table ring depth (an explicit ``stages`` of 4 pins the default form)
DW_BLOCK_GROUPS = {
    "sm_100a": 12,
    "sm_103a": 12,
}  # column tiles per raster block of the weight-gradient accumulate / fp32 fused cast at the C 4096 chunk, per architecture
DW_BLOCK_MIN_ROWS = RASTER_RULE_MIN_ROWS  # 2049
DW_BLOCK_MAX_ROWS = (
    DW_LONG_CHUNK_MIN_ROWS - 1
)  # 4096: above, the long-chunk raster owns the instance
DW_BLOCK_BASES = (
    "gemm_dw_acc",
    "gemm_dw_cast_f32",
)  # the fp32 output streams of the weight-gradient GEMM
DW_ITEM_COLS = 2  # column tiles of the A-sharing weight-gradient work item (a block is a multiple of them)
DX_TILE_WIDE = 512
DX_TILE_NARROW = 256
DX_WIDE_MIN_EFF = {
    "sm_100a": 0.83,
}  # wave-efficiency floor of a one-slice 512-wide dX launch below RASTER_RULE_MIN_HIDDEN, per architecture; below it the
#    256-wide cluster form runs the chunk (absent = the wide tile stays)
DX_K_SLICES_MAX = 4
GEMM_TUNING_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_GEMM_TUNING"
_RASTER_BASES = ("gemm_logits", "gemm_logits_nostats", "gemm_dw_acc") + DW_CAST_STAGES
_TUNING_KNOBS = {
    "logits": ("group_m",),
    "dx": ("k_slices", "tile_n", "stages"),
    "dw": ("group_m", "epi_store", "group_n"),
}
_STAGE_RE = re.compile(
    r"(?P<base>" + "|".join(sorted(BASE_STAGES, key=len, reverse=True)) + r")"
    r"(?:_s(?P<k>[2-9]))?(?:_tn(?P<tile>\d+))?(?:_st(?P<stages>\d+))?(?:_g(?P<group>\d+))?(?:_gn(?P<group_n>\d+))?(?P<tma>_tma)?"
)


def stage_variant(
    base: str,
    *,
    k_slices: int = 1,
    tile_n: Optional[int] = None,
    group_m: Optional[int] = None,
    epi_store: Optional[str] = None,
    stages: Optional[int] = None,
    group_n: Optional[int] = None,
) -> str:
    """Stage name of one instance variant of ``base``: the rule outputs that select it, in the fixed suffix order
    ``_s<k>``, ``_tn256``, ``_st<stages>``, ``_g<group_m>``, ``_gn<group_n>``, ``_tma``; a default output (one slice, the
    wide tile, the table ring depth, no raster override, the 1-D raster, the default epilogue) adds nothing, so
    ``stage_variant(base)`` is ``base`` itself."""
    if base not in BASE_STAGES:
        raise ValueError(f"unknown base stage {base!r}")
    name = base
    if int(k_slices) > 1:
        if base != "gemm_dx" or not 1 <= int(k_slices) <= DX_K_SLICES_MAX:
            raise ValueError(
                f"{base}: K slices ({k_slices}) exist for gemm_dx (1 .. {DX_K_SLICES_MAX}) only"
            )
        name += f"_s{int(k_slices)}"
    if tile_n is not None and int(tile_n) != DX_TILE_WIDE:
        if base != "gemm_dx" or int(tile_n) != DX_TILE_NARROW:
            raise ValueError(
                f"{base}: the only non-default dX tile is {DX_TILE_NARROW}, got {tile_n}"
            )
        name += f"_tn{DX_TILE_NARROW}"
    if stages is not None and int(stages) != DX_WIDE_TILE_STAGES:
        if (
            base != "gemm_dx"
            or int(stages) not in set(DX_LONG_CHUNK_STAGES.values())
            or (tile_n is not None and int(tile_n) != DX_TILE_WIDE)
        ):
            raise ValueError(
                f"{base}: the only ring-depth variant is the 512-wide dX tile's {sorted(set(DX_LONG_CHUNK_STAGES.values()))}-deep ring, got {stages} (tile {tile_n})"
            )
        name += f"_st{int(stages)}"
    if group_m is not None:
        if base not in _RASTER_BASES:
            raise ValueError(f"{base}: no raster-height rule")
        name += f"_g{int(group_m)}"
    if group_n is not None and int(group_n) != 0:
        if base not in DW_BLOCK_BASES or int(group_n) not in set(
            DW_BLOCK_GROUPS.values()
        ):
            raise ValueError(
                f"{base}: the only block-width variant is the weight-gradient accumulate's / fp32 cast's {sorted(set(DW_BLOCK_GROUPS.values()))}, got {group_n}"
            )
        name += f"_gn{int(group_n)}"
    if epi_store is not None and epi_store != "redsm":
        if base != "gemm_dw_acc" or epi_store != "tma":
            raise ValueError(
                f"{base}: the only epilogue variant is the weight-gradient accumulate's TMA reduce-add, got {epi_store!r}"
            )
        name += "_tma"
    return name


def parse_stage(stage: str) -> tuple[str, dict[str, Any]]:
    """``(base, {k_slices, tile_n, group_m, epi_store, stages, group_n})`` of a stage name (the inverse of
    :func:`stage_variant`)."""
    m = _STAGE_RE.fullmatch(stage)
    if m is None:
        raise ValueError(f"not a stage name of this backend: {stage!r}")
    knobs: dict[str, Any] = dict(
        k_slices=int(m["k"]) if m["k"] else 1,
        tile_n=int(m["tile"]) if m["tile"] else None,
        group_m=int(m["group"]) if m["group"] else None,
        epi_store="tma" if m["tma"] else None,
        stages=int(m["stages"]) if m["stages"] else None,
        group_n=int(m["group_n"]) if m["group_n"] else None,
    )
    if stage_variant(m["base"], **knobs) != stage:
        raise ValueError(f"not a stage name of this backend: {stage!r}")
    return m["base"], knobs


def base_stage(stage: str) -> str:
    """The base stage a (variant) stage name instantiates."""
    return parse_stage(stage)[0]


def raster_variant(
    hidden: int, rows_c: int, arch: Optional[str]
) -> Optional[tuple[Optional[int], Optional[int]]]:
    """``(logits group_m, weight-gradient group_m)`` the raster rules select for a chunk -- ``None`` in a slot = that
    GEMM's default raster, ``None`` altogether = the default raster for both.  At ``H >= RASTER_RULE_MIN_HIDDEN`` a chunk
    of ``>= RASTER_RULE_MIN_ROWS`` rows takes ``RASTER_WIDE_GROUPS[arch]``; below that ``H`` a chunk of
    ``>= LOGITS_LONG_RASTER_MIN_ROWS`` rows takes the logits height ``LOGITS_LONG_RASTER_GROUPS[arch]`` and a chunk of
    ``>= DW_LONG_CHUNK_MIN_ROWS`` rows the weight-gradient height ``DW_LONG_CHUNK_GROUP_M``, on every supported
    architecture."""
    if arch is None:
        return None
    if int(hidden) >= RASTER_RULE_MIN_HIDDEN:
        groups = RASTER_WIDE_GROUPS.get(arch)
        if groups is None or int(rows_c) < RASTER_RULE_MIN_ROWS:
            return None
        return groups
    logits_group = LOGITS_LONG_RASTER_GROUPS.get(arch)
    if logits_group is None or int(rows_c) < LOGITS_LONG_RASTER_MIN_ROWS:
        logits_group = None
    dw_group = DW_LONG_CHUNK_GROUPS.get(arch)
    if dw_group is None or int(rows_c) < DW_LONG_CHUNK_MIN_ROWS:
        dw_group = None
    if logits_group is None and dw_group is None:
        return None
    return (logits_group, dw_group)


def _dw_raster_groups(
    arch: Optional[str], hidden: Optional[int]
) -> tuple[Optional[int], ...]:
    """The weight-gradient raster heights the rules can select for a record (``None`` = the default first): the wide
    height at ``hidden >= RASTER_RULE_MIN_HIDDEN``, the long-chunk height below it, the default only for an unpinned
    geometry or an architecture without a rule."""
    if hidden is None or arch is None:
        return (None,)
    if int(hidden) >= RASTER_RULE_MIN_HIDDEN:
        groups = RASTER_WIDE_GROUPS.get(arch)
        return (None,) if groups is None else (None, int(groups[1]))
    dw_group = DW_LONG_CHUNK_GROUPS.get(arch)
    return (None,) if dw_group is None else (None, int(dw_group))


def dx_stages_variant(
    rows_c: int, hidden: int, tile_n: Optional[int], arch: Optional[str]
) -> Optional[int]:
    """The dX operand-ring depth the ring rule selects for a chunk (``None`` = the table depth): ``DX_LONG_CHUNK_STAGES``
    of the architecture for the 512-wide tile at ``>= DW_LONG_CHUNK_MIN_ROWS`` rows below ``RASTER_RULE_MIN_HIDDEN``."""
    st = DX_LONG_CHUNK_STAGES.get(arch) if arch is not None else None
    if (
        st is None
        or int(hidden) >= RASTER_RULE_MIN_HIDDEN
        or int(rows_c) < DW_LONG_CHUNK_MIN_ROWS
    ):
        return None
    if tile_n is not None and int(tile_n) != DX_TILE_WIDE:
        return None
    return int(st)


def _dw_block_group(arch: Optional[str], hidden: Optional[int]) -> Optional[int]:
    """The weight-gradient block width the block rule can select for a record (``None`` = no block form): the
    architecture's ``DW_BLOCK_GROUPS`` entry below ``RASTER_RULE_MIN_HIDDEN`` when it tiles the ``H / 256`` column tiles
    with whole work items and is narrower than the row of items."""
    gn = DW_BLOCK_GROUPS.get(arch) if arch is not None else None
    if gn is None or hidden is None or int(hidden) >= RASTER_RULE_MIN_HIDDEN:
        return None
    n_tiles = int(hidden) // DX_TILE_NARROW
    if gn % DW_ITEM_COLS or n_tiles % gn or gn >= n_tiles:
        return None
    return int(gn)


def dw_block_variant(rows_c: int, hidden: int, arch: Optional[str]) -> Optional[int]:
    """The weight-gradient block width the block rule selects for a chunk (``None`` = the 1-D raster):
    :func:`_dw_block_group` at ``DW_BLOCK_MIN_ROWS .. DW_BLOCK_MAX_ROWS`` rows."""
    gn = _dw_block_group(arch, hidden)
    if gn is None or not (DW_BLOCK_MIN_ROWS <= int(rows_c) <= DW_BLOCK_MAX_ROWS):
        return None
    return gn


def dx_tile_rule(
    rows_c: int,
    hidden: int,
    num_sms: int,
    k_slices: int,
    geometry: "Geometry",
    arch: Optional[str] = None,
) -> int:
    """Column width of the dX GEMM's pair tile for a chunk: ``DX_TILE_WIDE`` unless ``H % 512 != 0``, or the launch's
    wide work items -- ``(row pairs) x (H / 512) x k_slices`` -- are fewer than the device's SM pairs, or (on an
    architecture listed in ``DX_WIDE_MIN_EFF``) a one-slice launch below ``RASTER_RULE_MIN_HIDDEN`` fills the SM pairs'
    waves below that floor -- each takes the ``DX_TILE_NARROW`` form (the K-slice count is decided first, on the narrow
    form's inputs, so ``dX`` is bitwise the same on both tiles).  Deterministic in the shapes, the slice count, the SM
    count and the architecture."""
    if int(hidden) % DX_TILE_WIDE:
        return DX_TILE_NARROW
    ctas = int(geometry.cta_group)
    pairs = geometry.row_tiles(rows_c, ctas) // ctas
    items = pairs * (int(hidden) // DX_TILE_WIDE) * max(1, int(k_slices))
    device_pairs = max(1, int(num_sms) // ctas)
    if items < device_pairs:
        return DX_TILE_NARROW
    floor = DX_WIDE_MIN_EFF.get(arch) if arch is not None else None
    if (
        floor is not None
        and int(k_slices) == 1
        and int(hidden) < RASTER_RULE_MIN_HIDDEN
        and items / (-(-items // device_pairs) * device_pairs) < floor
    ):
        return DX_TILE_NARROW
    return DX_TILE_WIDE


def gemm_tuning_default() -> dict[str, dict[str, Any]]:
    """The explicit GEMM knobs of ``$FLASHINFER_CAKE_LM_HEAD_LOSS_GEMM_TUNING`` (a JSON object; unset / empty = none)."""
    raw = os.environ.get(GEMM_TUNING_ENV, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise ValueError(
            f"${GEMM_TUNING_ENV} must be a JSON object of GEMM knobs: {exc}"
        ) from exc
    return _resolve_gemm_tuning(value)


def _resolve_gemm_tuning(gemm_tuning) -> dict[str, dict[str, Any]]:
    """Validated copy of an explicit GEMM knob set (``None`` = :func:`gemm_tuning_default`)."""
    if gemm_tuning is None:
        return gemm_tuning_default()
    if not isinstance(gemm_tuning, dict):
        raise ValueError("gemm_tuning must be a dict {GEMM: {knob: value}}")
    out: dict[str, dict[str, Any]] = {}
    for op, knobs in gemm_tuning.items():
        if op not in _TUNING_KNOBS:
            raise ValueError(
                f"gemm_tuning: unknown GEMM {op!r}; one of {sorted(_TUNING_KNOBS)}"
            )
        if not isinstance(knobs, dict) or set(knobs) - set(_TUNING_KNOBS[op]):
            raise ValueError(
                f"gemm_tuning[{op!r}] must be a dict over the knobs {_TUNING_KNOBS[op]}, got {knobs!r}"
            )
        out[op] = {
            k: None if v is None else (str(v) if k == "epi_store" else int(v))
            for k, v in knobs.items()
        }
    return out


def _tuning_key(tuning: dict[str, dict[str, Any]]) -> tuple:
    return tuple(
        (op, tuple(sorted(knobs.items()))) for op, knobs in sorted(tuning.items())
    )


@dataclass(frozen=True)
class ChunkVariants:
    """The instance knobs of one chunk's GEMMs as the rules (or the explicit knobs) resolved them; ``None`` = the
    default form of the base stage."""

    logits_group_m: Optional[int] = None
    dx_tile_n: Optional[int] = None  # None = DX_TILE_WIDE
    dx_stages: Optional[int] = None  # None = the table ring depth
    dw_group_m: Optional[int] = None
    dw_epi_store: Optional[str] = None
    dw_group_n: Optional[int] = None  # None = the 1-D raster


def chunk_variants(
    rows_c: int,
    hidden: int,
    num_sms: int,
    k_slices: int,
    geometry: "Geometry",
    arch: Optional[str],
    tuning: Optional[dict[str, dict[str, Any]]] = None,
) -> ChunkVariants:
    """Per-chunk instance knobs: the explicit ``tuning`` knob when present (``None`` pins the default), else the rule
    of a registered program's architecture (``arch``; ``None`` -- the reference path -- plans the base stages)."""
    tuning = tuning or {}
    logits_t, dx_t, dw_t = (tuning.get(op, {}) for op in ("logits", "dx", "dw"))
    raster = raster_variant(hidden, rows_c, arch)
    logits_g = (
        logits_t["group_m"]
        if "group_m" in logits_t
        else (raster[0] if raster else None)
    )
    dw_g = dw_t["group_m"] if "group_m" in dw_t else (raster[1] if raster else None)
    if "tile_n" in dx_t:
        tile = dx_t["tile_n"]
    else:
        tile = (
            dx_tile_rule(rows_c, hidden, num_sms, k_slices, geometry, arch)
            if arch is not None
            else None
        )
    if "stages" in dx_t:
        stages = dx_t["stages"]
    else:
        stages = (
            dx_stages_variant(rows_c, hidden, tile, arch) if arch is not None else None
        )
    epi = dw_t.get(
        "epi_store"
    )  # no rule selects an epilogue form (the SM103 TMA reduce-add rule was retired): explicit pins only
    if "group_n" in dw_t:
        gn = dw_t["group_n"]
    else:
        gn = dw_block_variant(rows_c, hidden, arch) if arch is not None else None
    return ChunkVariants(
        logits_group_m=None if logits_g is None else int(logits_g),
        dx_tile_n=None if tile is None or int(tile) == DX_TILE_WIDE else int(tile),
        dx_stages=None
        if stages is None or int(stages) == DX_WIDE_TILE_STAGES
        else int(stages),
        dw_group_m=None if dw_g is None else int(dw_g),
        dw_epi_store=None if epi in (None, "redsm") else str(epi),
        dw_group_n=None if gn in (None, 0) else int(gn),
    )


def reachable_variants(bases, arch: str, geometry: "Geometry", dx_max: int) -> set[str]:
    """Every variant the per-chunk rules can select for a record (its architecture, pinned geometry and registered
    dX slice depth) from the base GEMM stages ``bases``; a program is complete for an entry when it registers them."""
    wide = (
        geometry.hidden is not None
        and int(geometry.hidden) >= RASTER_RULE_MIN_HIDDEN
        and arch in RASTER_WIDE_GROUPS
    )
    dw_groups = _dw_raster_groups(arch, geometry.hidden)
    glm = geometry.hidden is not None and int(geometry.hidden) < RASTER_RULE_MIN_HIDDEN
    g_logits = (
        RASTER_WIDE_GROUPS.get(arch, (None, None))[0]
        if wide
        else (LOGITS_LONG_RASTER_GROUPS.get(arch) if glm else None)
    )
    st_long = DX_LONG_CHUNK_STAGES.get(arch) if glm else None
    gn = _dw_block_group(arch, geometry.hidden)
    out: set[str] = set()
    for base in bases:
        if base in ("gemm_logits", "gemm_logits_nostats"):
            if g_logits is not None:
                out.add(stage_variant(base, group_m=g_logits))
        elif base == "gemm_dx":
            for k in range(1, int(dx_max) + 1):
                for tile in (DX_TILE_WIDE, DX_TILE_NARROW):
                    out.add(stage_variant(base, k_slices=k, tile_n=tile))
                if st_long is not None:
                    out.add(
                        stage_variant(
                            base, k_slices=k, tile_n=DX_TILE_WIDE, stages=st_long
                        )
                    )
        elif base == "gemm_dw_acc":
            for g in dw_groups:
                out.add(stage_variant(base, group_m=g))
            if gn is not None:
                out.add(stage_variant(base, group_n=gn))
        elif base in DW_CAST_STAGES:
            for g in dw_groups:
                if g is not None:
                    out.add(stage_variant(base, group_m=g))
            if gn is not None and base in DW_BLOCK_BASES:
                out.add(stage_variant(base, group_n=gn))
    return out


# Accepted spellings of the same host value (kernel side -> host side).
CONTRACT_ALIASES = {
    "num_rows": "T",
    "total_rows": "T",
    "hidden": "H",
    "vocab": "V",
    "chunk_size": "chunk",
    "num_vocab_tiles": "num_tiles",
    "vocab_tiles": "num_tiles",
    "objective": "mode",
    "grad_scale": "g",
    "scale": "g",
    "src": "acc",
    "dst": "out",
    "z_c": "z",
    "logits": "z",
    "dz": "z",
    "dz_c": "z",
    "dlogits": "z",
    "dlogp": "d_in",
    "y": "labels",
    "targets": "labels",
    "weights": "loss_weights",
    "w": "loss_weights",
}


@dataclass(frozen=True)
class Geometry:
    stats_tile: int
    row_tile: int
    cta_group: int
    k_block: int
    cast_vec: int
    vocab_multiple: int
    hidden_multiple: int
    ld_multiple: int
    labels_dtype: torch.dtype
    hidden: Optional[int] = None
    vocab: Optional[int] = None
    logits_cluster_ctas: int = 2
    dx_cluster_ctas: int = 2
    dw_cluster_ctas: int = 2

    @classmethod
    def from_record(cls, record: Optional[dict[str, Any]]) -> "Geometry":
        raw = dict(GEOMETRY_DEFAULTS)
        if record is not None:
            declared = record.get("geometry", {})
            unknown = sorted(set(declared) - set(GEOMETRY_DEFAULTS))
            if unknown:
                raise ValueError(f"registry record: unknown geometry fields {unknown}")
            raw.update(declared)
        for name in GEOMETRY_DEFAULTS:
            if name == "labels_dtype" or (
                name in _GEOMETRY_OPTIONAL and raw[name] is None
            ):
                continue
            if isinstance(raw[name], bool) or int(raw[name]) < 1:
                raise ValueError(
                    f"registry record: geometry {name} must be a positive integer"
                )
        if raw["labels_dtype"] not in LABEL_DTYPES:
            raise ValueError(
                f"registry record: geometry labels_dtype must be one of {sorted(LABEL_DTYPES)}"
            )
        return cls(
            **{
                name: int(raw[name])
                for name in GEOMETRY_DEFAULTS
                if name != "labels_dtype" and name not in _GEOMETRY_OPTIONAL
            },
            labels_dtype=LABEL_DTYPES[raw["labels_dtype"]],
            hidden=None if raw["hidden"] is None else int(raw["hidden"]),
            vocab=None if raw["vocab"] is None else int(raw["vocab"]),
        )

    def row_tiles(self, rows: int, cluster_ctas: Optional[int] = None) -> int:
        """``ceil(rows / row_tile)`` rounded up to a multiple of the cluster's CTA count (default ``cta_group``: the
        pair takes adjacent row tiles; a wider cluster takes as many)."""
        ctas = self.cta_group if cluster_ctas is None else int(cluster_ctas)
        tiles = -(-int(rows) // self.row_tile)
        return -(-tiles // ctas) * ctas

    def cluster_ctas_of(self, stage: str) -> Optional[int]:
        """CTAs per cluster of a GEMM stage (``None`` for the row kernels and the casts)."""
        base = base_stage(stage) if stage in GEMM_STAGES else stage
        if base in ("gemm_logits", "gemm_logits_nostats"):
            return self.logits_cluster_ctas
        if base == "gemm_dx":
            return self.dx_cluster_ctas
        if base == "gemm_dw_acc" or base in DW_CAST_STAGES:
            return self.dw_cluster_ctas
        return None

    def k_iters(self, rows_c: int) -> int:
        return -(-int(rows_c) // self.k_block)


DEFAULT_GEOMETRY = Geometry.from_record(None)


# ---------------------------------------------------------------------------
# Device / registry queries
# ---------------------------------------------------------------------------


def arch_for(device: Optional[torch.device] = None) -> Optional[str]:
    """Architecture tag of ``device`` (``None`` when unsupported or without CUDA)."""
    if not torch.cuda.is_available():
        return None
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type != "cuda":
        return None
    return SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))


def record_for(
    device: Optional[torch.device] = None,
    hidden: Optional[int] = None,
    vocab: Optional[int] = None,
) -> tuple[str, dict[str, Any]]:
    """``(module_name, record)`` registered for ``device`` and the ``(hidden, vocab)`` geometry (when given; else the
    architecture's default-geometry record, see :func:`cake_jit.select_module`); raises when absent."""
    arch = arch_for(device)
    if arch is None:
        raise ValueError(
            "the chunked LM-head + loss kernels require compute capability 10.0 or 10.3"
        )
    name = select_module(arch, hidden, vocab)
    return name, MODULES[name]


def record_abi(record: dict[str, Any]) -> str:
    abi = str(record.get("abi", ABI_CONTRACT))
    if abi not in SUPPORTED_ABIS:
        raise NotImplementedError(f"unsupported host binding profile {abi!r}")
    return abi


def dx_stage(
    k_slices: int, tile_n: Optional[int] = None, stages: Optional[int] = None
) -> str:
    """Stage name of the dX GEMM with ``k_slices`` K-slice work items per output tile (``tile_n`` = the pair tile's
    column width, ``None`` / ``DX_TILE_WIDE`` = the wide default; ``stages`` = the operand-ring depth, ``None`` = the
    table depth)."""
    return stage_variant(
        "gemm_dx", k_slices=int(k_slices), tile_n=tile_n, stages=stages
    )


def dx_max_slices(stages) -> int:
    """Largest slice count the registered stages serve (``1`` + the contiguous prefix of ``DX_SLICE_STAGES``)."""
    count = 1
    for stage in DX_SLICE_STAGES:
        if stage not in stages:
            break
        count += 1
    return count


def wave_efficiency(
    rows_c: int,
    hidden: int,
    num_sms: int,
    k_slices: int,
    geometry: "Geometry",
    resident: Optional[int] = None,
) -> float:
    """Fraction of the last persistent wave of the dX GEMM that carries work: ``items / (ceil(items / clusters) * clusters)``
    with ``clusters`` = the co-resident ``dx_cluster_ctas``-wide clusters (``resident``; ``None`` = one per SM group of
    that width, the rule of the two-CTA cluster)."""
    ctas = geometry.dx_cluster_ctas
    clusters = (
        max(1, int(num_sms) // ctas) if resident is None else max(1, int(resident))
    )
    items = (
        (geometry.row_tiles(rows_c, ctas) // ctas)
        * (int(hidden) // geometry.hidden_multiple)
        * int(k_slices)
    )
    return items / (-(-items // clusters) * clusters)


def recommended_k_slices(
    rows_c: int,
    hidden: int,
    num_sms: int,
    max_slices: int,
    geometry: "Geometry",
    resident: Optional[int] = None,
) -> int:
    """Slice count of the dX GEMM for a chunk of ``rows_c`` rows: the ``S`` in ``[1, max_slices]`` with the best
    wave efficiency net of ``K_SLICE_PENALTY`` per extra slab.  Deterministic in the shapes, the SM count and the
    device's co-resident cluster count (:func:`cluster_resident`)."""
    best, best_score = (
        1,
        wave_efficiency(rows_c, hidden, num_sms, 1, geometry, resident),
    )
    for s in range(2, max(1, int(max_slices)) + 1):
        score = wave_efficiency(
            rows_c, hidden, num_sms, s, geometry, resident
        ) - K_SLICE_PENALTY * (s - 1)
        if score > best_score + 1e-9:
            best, best_score = s, score
    return best


# --------------------------------------------------------------------------- co-resident clusters

_PROBE_KERNEL = "cake_lm_head_loss_cluster_probe"
_PROBE_SOURCE = f'extern "C" __global__ void {_PROBE_KERNEL}() {{}}\n'


@functools.lru_cache(maxsize=None)
def _probe_cluster_capacity(device_index: int, cluster_ctas: int) -> int:
    """``cuOccupancyMaxActiveClusters`` of an empty probe kernel launched with the device's maximum opt-in dynamic shared
    memory (one CTA per SM) in ``cluster_ctas``-wide clusters.  The part's GPC topology after floorsweeping decides the
    answer, not ``SMs // cluster_ctas`` (a 148-SM B200 holds 33 four-CTA clusters, a 152-SM GB300 36)."""
    from ...cuda_utils import checkCudaErrors, driver, nvrtc

    with torch.cuda.device(device_index):
        torch.empty(1, device="cuda")  # primary context
        major, minor = torch.cuda.get_device_capability(device_index)
        prog = checkCudaErrors(
            nvrtc.nvrtcCreateProgram(_PROBE_SOURCE.encode(), b"probe.cu", 0, [], [])
        )
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode()]
        checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
        size = checkCudaErrors(nvrtc.nvrtcGetCUBINSize(prog))
        cubin = b" " * size
        checkCudaErrors(nvrtc.nvrtcGetCUBIN(prog, cubin))
        checkCudaErrors(nvrtc.nvrtcDestroyProgram(prog))
        ctx = checkCudaErrors(driver.cuDevicePrimaryCtxRetain(device_index))
        try:
            checkCudaErrors(driver.cuCtxSetCurrent(ctx))
            module = checkCudaErrors(driver.cuModuleLoadData(cubin))
            try:
                func = checkCudaErrors(
                    driver.cuModuleGetFunction(module, _PROBE_KERNEL.encode())
                )
                smem = checkCudaErrors(
                    driver.cuDeviceGetAttribute(
                        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                        device_index,
                    )
                )
                checkCudaErrors(
                    driver.cuFuncSetAttribute(
                        func,
                        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        smem,
                    )
                )
                attr = driver.CUlaunchAttribute()
                attr.id = (
                    driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
                )
                attr.value.clusterDim.x = int(cluster_ctas)
                attr.value.clusterDim.y = 1
                attr.value.clusterDim.z = 1
                config = driver.CUlaunchConfig()
                config.gridDimX = int(cluster_ctas)
                config.gridDimY = 1
                config.gridDimZ = 1
                config.blockDimX = 128
                config.blockDimY = 1
                config.blockDimZ = 1
                config.sharedMemBytes = smem
                config.attrs = [attr]
                config.numAttrs = 1
                count = int(
                    checkCudaErrors(driver.cuOccupancyMaxActiveClusters(func, config))
                )
            finally:
                checkCudaErrors(driver.cuModuleUnload(module))
        finally:
            checkCudaErrors(driver.cuDevicePrimaryCtxRelease(device_index))
    return max(1, count)


def cluster_resident(
    device: Optional[torch.device], cluster_ctas: int, num_sms: int
) -> int:
    """Co-resident clusters of ``cluster_ctas`` CTAs at one CTA per SM on ``device``: the SM pairs for the two-CTA
    cluster (the source launcher's rule), the driver's occupancy answer (:func:`_probe_cluster_capacity`) for wider
    clusters.  The persistent GEMM grids and the dX slice rule take it as ``resident``."""
    cluster_ctas = int(cluster_ctas)
    if cluster_ctas <= 2:
        return max(1, int(num_sms) // max(1, cluster_ctas))
    if device is None or device.type != "cuda":
        raise ValueError(
            f"the co-resident count of {cluster_ctas}-CTA clusters needs a CUDA device"
        )
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _probe_cluster_capacity(int(index), cluster_ctas)


def _dx_reduce(values: dict[str, Any]) -> None:
    """Add the K-slice slabs into the chunk's rows of ``dX_acc`` in fixed slab order (host side, in place)."""
    out, ws, rows_c, k = (
        values["acc"],
        values["ws"],
        int(values["rows_c"]),
        int(values["k_slices"]),
    )
    for slab in range(k - 1):
        out.add_(ws[slab, :rows_c])


# --------------------------------------------------------------------------- valid-row compaction

COMPACT_ROWS_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_COMPACT_ROWS"  # "0" turns the compaction off (the before / after switch)


def compact_rows_default() -> bool:
    """Default of ``compact_rows``: chunk over the valid rows only unless ``$FLASHINFER_CAKE_LM_HEAD_LOSS_COMPACT_ROWS`` is ``0``."""
    return os.environ.get(COMPACT_ROWS_ENV, "1") != "0"


FUSE_DW_CAST_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_FUSE_DW_CAST"  # "0" turns the fused weight-gradient cast off


def fuse_dw_cast_default() -> bool:
    """Default of ``fuse_dw_cast``: the last chunk's weight-gradient GEMM runs in the backward with the upstream scale and
    the output cast fused into its epilogue (:data:`DW_CAST_STAGES`) unless ``$FLASHINFER_CAKE_LM_HEAD_LOSS_FUSE_DW_CAST``
    is ``0`` (then every chunk accumulates into the FP32 ``dW_acc`` and ``scale_cast`` casts it; bitwise the same ``dW``)."""
    return os.environ.get(FUSE_DW_CAST_ENV, "1") != "0"


def _resolve_fuse(fuse_dw_cast) -> bool:
    return fuse_dw_cast_default() if fuse_dw_cast is None else bool(fuse_dw_cast)


DX_FINALIZE_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_DX_FINALIZE"  # "0" turns the fused dX finalize off (the before / after switch)


def dx_finalize_default() -> bool:
    """Default of ``dx_finalize``: the K-slice slabs of a sliced dX GEMM are added by the ``slab_sum`` kernel and a
    compacted call's ``dX`` output is written by the ``scale_cast_scatter_bf16`` kernel (:data:`DX_FINALIZE_STAGES`)
    unless ``$FLASHINFER_CAKE_LM_HEAD_LOSS_DX_FINALIZE`` is ``0`` (then the host adds the slabs with in-place ``add_``
    launches and casts, zero-fills and ``index_copy_``-scatters the compact rows; bitwise the same ``dX``).  Read once
    per forward; the backward follows the forward's choice through the saved valid-row mask."""
    return os.environ.get(DX_FINALIZE_ENV, "1") != "0"


def _resolve_dx_finalize(dx_finalize) -> bool:
    return dx_finalize_default() if dx_finalize is None else bool(dx_finalize)


DW_STREAM_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_DW_STREAM"  # "0" never, "1" every multi-chunk call, "auto" (default)
DW_STREAM_MIN_CHUNKS = 3  # the side stream pays one cross-stream join per chunk and overlaps only the chunks before a deferred last one
_DW_STREAMS: dict[
    int, tuple
] = {}  # device index -> (side stream, fork event); not a tensor, outside every workspace / binding


def dw_stream_mode() -> str:
    """``$FLASHINFER_CAKE_LM_HEAD_LOSS_DW_STREAM`` as ``"0"`` (never), ``"1"`` (every multi-chunk call) or ``"auto"``
    (the default: calls of at least :data:`DW_STREAM_MIN_CHUNKS` chunks, :func:`dw_side_stream`)."""
    value = os.environ.get(DW_STREAM_ENV, "auto").strip().lower()
    if value in ("0", "false", "off", "no"):
        return "0"
    if value in ("1", "true", "on", "yes"):
        return "1"
    return "auto"


def dw_side_stream(num_chunks: int) -> bool:
    """Whether a call of ``num_chunks`` chunks launches each chunk's weight-gradient accumulate GEMM on the per-device
    side stream: forked right after the chunk's ``row_grad`` (``dz`` is ready; the accumulate depends on it alone, not
    on the dX GEMM launched next) and joined before the next chunk touches the chunk buffer and before the call returns,
    so the accumulate's CTAs fill the tail wave of the chunk's dX GEMM instead of queueing behind it (GLM-class 4096-row
    chunks: 768 dX work items on 76 CTA pairs = 11 waves at 0.919 fill).  Measured on the same device in one process
    (ratio of the step's GPU span to the single-stream step): four-chunk calls 0.991 / 0.995 cross-entropy / policy
    (SM103) and 0.998 / 0.995 (SM100), eight-chunk 2048-row calls 0.996 / 0.993; two-chunk calls pay the join without an
    overlap gain (0.999 / 1.002) and a one-chunk call defers its only accumulate to the backward -- hence the ``auto``
    rule of :data:`DW_STREAM_MIN_CHUNKS` chunks.  Kernels, buffers, launch order per kernel and numerics are unchanged
    (the dX GEMM writes the dX rows / slabs, the accumulate reads the same ``dz`` and read-modify-writes ``dW_acc``;
    each kernel keeps its own order): only the launch stream differs, so every output is bitwise the single-stream
    path's.  The per-device stream and fork event are shared by the calls on that device (one call at a time per
    device, as the generated launches themselves)."""
    n = int(num_chunks)
    if n <= 1:
        return False
    mode = dw_stream_mode()
    if mode == "0":
        return False
    if mode == "1":
        return True
    return n >= DW_STREAM_MIN_CHUNKS


def _dw_stream(index: int) -> tuple:
    """``(side stream, fork event)`` of device ``index`` (created on first use; never part of a binding or a workspace)."""
    pair = _DW_STREAMS.get(index)
    if pair is None:
        pair = _DW_STREAMS[index] = (
            torch.cuda.Stream(device=index),
            torch.cuda.Event(),
        )
    return pair


def valid_rows(
    labels: torch.Tensor, *, mask: bool = False
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """``(row_index, row_valid)``.  ``row_index`` is ``None`` when every row is valid (or ``T == 0``: the uncompacted
    path, no gather cost), else the ascending int64 index of the valid rows (``labels >= 0``; ``-100`` is the ignore
    index).  Rows with a negative label contribute exactly zero to the loss, ``dX`` and ``dW``, so skipping them is
    exact.  One device synchronization (the count) plus the ``nonzero`` synchronization when rows are ignored.
    ``mask=True`` (the fused dX finalize of a call that produces ``dX``) also returns the already materialized bool
    valid-row mask ``labels >= 0`` (``[T]``; :func:`finalize_dx` turns it into the inclusive int32 scan behind the
    backward's queued GEMMs, not here in the synchronization shadow of ``nonzero``); ``None`` whenever ``row_index``
    is."""
    T = int(labels.numel())
    if T == 0:
        return None, None
    valid = labels >= 0
    if int(valid.sum().item()) == T:
        return None, None
    return valid.nonzero().squeeze(1), (valid if mask else None)


def valid_row_index(labels: torch.Tensor) -> Optional[torch.Tensor]:
    """The valid-row index of :func:`valid_rows` alone (``None`` when every row is valid)."""
    return valid_rows(labels)[0]


def scatter_rows(
    rows: torch.Tensor, idx: Optional[torch.Tensor], num_rows: int
) -> torch.Tensor:
    """Rows of a compacted ``[T_v, ...]`` tensor back into a fresh ``[num_rows, ...]`` tensor with exact zeros
    elsewhere (``idx=None``: the identity, the tensor itself)."""
    if idx is None:
        return rows
    out = torch.zeros(
        (int(num_rows),) + tuple(rows.shape[1:]), dtype=rows.dtype, device=rows.device
    )
    if idx.numel():
        out.index_copy_(0, idx, rows)
    return out


def _gather_rows(values: dict[str, Any]) -> None:
    """Gather ``src[idx]`` into the preallocated ``out`` (host side; a chunk's ``X`` rows into the reusable ``x_c``
    buffer, or a caller's ``[T]`` operand into its compact ``[T_v]`` form)."""
    torch.index_select(values["src"], 0, values["idx"], out=values["out"])


# Host-side steps of the chunk loop (torch operators over the bound tensors, no kernel of the program, no allocation).
HOST_STAGES: dict[str, Callable[[dict[str, Any]], None]] = {
    "dx_reduce": _dx_reduce,
    "gather_rows": _gather_rows,
}


def stages_for_entry(
    entry: str,
    *,
    need_dx: bool = True,
    need_dw: bool = True,
    grad_weight_dtype=torch.bfloat16,
    fuse_dw_cast: Optional[bool] = None,
    num_chunks: Optional[int] = None,
    dx_cast: bool = True,
) -> tuple[str, ...]:
    """Stages an entry point launches (``need_dx`` / ``need_dw`` drop the GEMMs of frozen inputs).

    ``fuse_dw_cast`` (default :func:`fuse_dw_cast_default`): the weight gradient's last chunk comes out of the fused
    GEMM ``gemm_dw_cast_bf16`` / ``gemm_dw_cast_f32`` instead of the flat ``scale_cast``; ``gemm_dw_acc`` then
    accumulates the chunks before it and is absent when ``num_chunks`` is given as 1 (unknown: kept).
    ``dx_cast=False`` leaves the flat ``dX`` cast out: the caller finalizes ``dx_acc`` itself (the fused dX finalize of
    a compacted log-probability backward scatters it in one pass, :func:`finalize_dx`).  The ``slab_sum`` kernel of a
    K-sliced dX GEMM is a per-chunk instance like the ``_s<k>`` forms (:attr:`Plan.stages`), not listed here.
    """
    if entry not in ENTRIES:
        raise ValueError(f"entry must be one of {ENTRIES}")
    fp32 = entry == "loss" and grad_weight_dtype == torch.float32
    stages = ["gemm_logits", "row_finalize"]
    if entry == "loss":
        stages.append("loss_reduce")
    else:
        stages.append("gemm_logits_nostats")
    if need_dx or need_dw:
        stages.append("row_grad")
    if need_dx:
        stages += ["gemm_dx"] + (["scale_cast_bf16"] if dx_cast else [])
    if need_dw:
        if _resolve_fuse(fuse_dw_cast):
            if num_chunks is None or int(num_chunks) > 1:
                stages.append("gemm_dw_acc")
            stages.append("gemm_dw_cast_f32" if fp32 else "gemm_dw_cast_bf16")
        else:
            stages += ["gemm_dw_acc", "scale_cast_f32" if fp32 else "scale_cast_bf16"]
    return tuple(s for s in STAGES if s in stages)


def generated_program_available(
    device: Optional[torch.device] = None,
    *,
    entry: str = "loss",
    hidden: Optional[int] = None,
    vocab: Optional[int] = None,
) -> bool:
    """True when this checkout registers a contract program for ``device`` (and the ``(hidden, vocab)`` geometry when
    given) with every stage the entry point needs -- the base stages plus every variant the per-chunk instance rules
    can select for the record's geometry and architecture (:func:`reachable_variants`)."""
    arch = arch_for(device)
    if arch is None:
        return False
    try:
        name = select_module(arch, hidden, vocab)
    except (NotImplementedError, ValueError):
        return False
    record = MODULES[name]
    if str(record.get("abi", ABI_CONTRACT)) != ABI_CONTRACT:
        return False
    stages = registered_stages(name)
    needed = set(stages_for_entry(entry)) | set(
        stages_for_entry(entry, grad_weight_dtype=torch.float32)
    )
    needed |= reachable_variants(
        needed, arch, Geometry.from_record(record), dx_max_slices(stages)
    )
    if dx_finalize_default():
        needed |= set(DX_FINALIZE_STAGES)
    return all(stage in stages for stage in needed)


# ---------------------------------------------------------------------------
# Validation and planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    """Validated geometry and options of one call."""

    num_rows: int  # T
    hidden: int  # H
    vocab: int  # V
    chunk: int  # C
    ld_x: int  # leading stride of X as launched (after a copy: H)
    x_copy: bool  # X is materialized contiguously (leading stride not a TMA pitch)
    objective: str
    loss_div: Optional[float]
    grad_weight_dtype: torch.dtype
    entry: str = "loss"

    @property
    def mode(self) -> int:
        """``mode`` of ``row_finalize`` in the forward."""
        if self.entry == "logprob":
            return MODE_NONE
        return MODE_CE if self.objective == "ce" else MODE_POLICY


def _positive_scalar(value, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        if value.is_cuda:
            raise ValueError(
                f"{name} must be a Python number or a CPU scalar (a device value would synchronize)"
            )
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return value


def validate_lm_head_inputs(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    objective: str = "ce",
    loss_div=None,
    infer_logp: Optional[torch.Tensor] = None,
    loss_weights: Optional[torch.Tensor] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    deterministic: bool = True,
    entry: str = "loss",
    geometry: Geometry = DEFAULT_GEOMETRY,
) -> Problem:
    """Shape / dtype / stride validation shared by the entry points.

    Returns the :class:`Problem`.  Device placement is checked separately so
    this runs on host tensors.  ``T == 0`` is accepted (the eager entry points
    return zeros for it without launching).
    """
    if entry not in ENTRIES:
        raise ValueError(f"entry must be one of {ENTRIES}")
    if deterministic is not True:
        raise NotImplementedError(
            "only deterministic=True is available: fixed chunk order, no atomics"
        )
    if X.ndim != 2 or X.dtype != torch.bfloat16:
        raise ValueError("X must be a BF16 [T, H] tensor")
    if W.ndim != 2 or W.dtype != torch.bfloat16:
        raise ValueError("W must be a BF16 [V, H] tensor")
    if not W.is_contiguous():
        raise ValueError("W must be contiguous")
    num_rows, hidden = (int(s) for s in X.shape)
    vocab, w_hidden = (int(s) for s in W.shape)
    if w_hidden != hidden:
        raise ValueError(
            f"W has {w_hidden} columns, X has {hidden}: the hidden sizes differ"
        )
    if hidden < 1 or hidden % geometry.hidden_multiple:
        raise ValueError(
            f"H must be a positive multiple of {geometry.hidden_multiple}, got {hidden}"
        )
    if vocab < 1 or vocab % geometry.vocab_multiple:
        raise ValueError(
            f"V must be a positive multiple of {geometry.vocab_multiple}, got {vocab}"
        )
    if geometry.hidden is not None and hidden != geometry.hidden:
        raise ValueError(
            f"the registered program is specialized to H = {geometry.hidden}, got {hidden}"
        )
    if geometry.vocab is not None and vocab != geometry.vocab:
        raise ValueError(
            f"the registered program is specialized to V = {geometry.vocab}, got {vocab}"
        )
    if X.stride(1) != 1:
        raise ValueError("X rows must be contiguous (stride(1) == 1)")
    ld_x = int(X.stride(0))
    if num_rows > 1 and ld_x < hidden:
        raise ValueError("X must be row-major with a leading stride of at least H")
    # The TMA operands need a 16 B row pitch and base; otherwise X is materialized contiguously (reported).
    x_copy = bool(
        num_rows and (ld_x < hidden or ld_x % geometry.ld_multiple or X.data_ptr() % 16)
    )
    if x_copy or num_rows == 0:
        ld_x = hidden
    if labels.ndim != 1 or int(labels.shape[0]) != num_rows:
        raise ValueError("labels must be a [T] tensor")
    if labels.dtype != torch.int64:
        raise ValueError("labels must be int64 (-100 marks ignored rows)")
    if (
        isinstance(chunk_size, bool)
        or int(chunk_size) != chunk_size
        or int(chunk_size) < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    if int(chunk_size) > 65535:
        raise ValueError(
            "chunk_size must not exceed 65535 (the row kernels index the chunk row by blockIdx.y)"
        )
    if grad_weight_dtype not in GRAD_WEIGHT_DTYPES:
        raise ValueError("grad_weight_dtype must be torch.bfloat16 or torch.float32")
    if entry == "logprob":
        if (
            objective != "ce"
            or loss_div is not None
            or infer_logp is not None
            or loss_weights is not None
        ):
            raise ValueError(
                "the log-probability entry point takes no objective arguments"
            )
        return Problem(
            num_rows,
            hidden,
            vocab,
            int(chunk_size),
            ld_x,
            x_copy,
            "ce",
            None,
            grad_weight_dtype,
            entry,
        )
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    if objective == "ce":
        if loss_div is None:
            raise ValueError(
                "objective='ce' needs loss_div (a caller-supplied positive scalar)"
            )
        if infer_logp is not None or loss_weights is not None:
            raise ValueError("infer_logp and loss_weights belong to objective='policy'")
        loss_div = _positive_scalar(loss_div, "loss_div")
    else:
        if loss_div is not None:
            raise ValueError("loss_div belongs to objective='ce'")
        if infer_logp is None or loss_weights is None:
            raise ValueError("objective='policy' needs infer_logp and loss_weights")
        for name, t in (("infer_logp", infer_logp), ("loss_weights", loss_weights)):
            if t.ndim != 1 or int(t.shape[0]) != num_rows or t.dtype != torch.float32:
                raise ValueError(f"{name} must be an FP32 [T] tensor")
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
    return Problem(
        num_rows,
        hidden,
        vocab,
        int(chunk_size),
        ld_x,
        x_copy,
        objective,
        loss_div,
        grad_weight_dtype,
        entry,
    )


def plan_chunks(num_rows: int, chunk: int) -> tuple[tuple[int, int], ...]:
    """``(row0, rows_c)`` of every chunk in launch order: full chunks first, the
    tail (``T % C``) last, never dropped, never padded; empty for ``T == 0``."""
    if int(chunk) < 1:
        raise ValueError("chunk must be positive")
    return tuple(
        (row0, min(int(chunk), int(num_rows) - row0))
        for row0 in range(0, int(num_rows), int(chunk))
    )


@dataclass(frozen=True)
class Plan:
    """The chunk schedule of one binding."""

    problem: Problem
    chunks: tuple[tuple[int, int], ...]
    need_dx: bool
    need_dw: bool
    geometry: Geometry = DEFAULT_GEOMETRY
    dx_slices: tuple[
        int, ...
    ] = ()  # K-slice count of the dX GEMM per chunk (empty = one slice everywhere)
    valid_rows: Optional[int] = (
        None  # rows of the compacted chunk loop (the valid rows); None = every row of the caller
    )
    fuse_dw_cast: bool = False  # the last chunk's weight-gradient GEMM runs in the backward with the fused scale + cast
    variants: tuple[
        ChunkVariants, ...
    ] = ()  # per chunk: the GEMM instance knobs the rules resolved (empty = defaults)
    dx_finalize: bool = False  # a K-sliced dX GEMM's slabs are added by the ``slab_sum`` kernel (else the host's ``add_`` chain)
    dx_cast: bool = True  # the backward casts ``dx_acc`` into ``dx_out`` (False: the caller finalizes ``dx_acc``, :func:`finalize_dx`)

    def variants_of(self, index: int) -> ChunkVariants:
        return self.variants[index] if self.variants else ChunkVariants()

    def logits_stage(self, index: int, *, stats: bool = True) -> str:
        """The logits GEMM variant of chunk ``index`` (``stats=False``: the log-probability backward's recompute)."""
        return stage_variant(
            "gemm_logits" if stats else "gemm_logits_nostats",
            group_m=self.variants_of(index).logits_group_m,
        )

    def dx_stage_of(self, index: int) -> str:
        """The dX GEMM variant of chunk ``index`` (its K-slice count and tile)."""
        v = self.variants_of(index)
        return dx_stage(self.dx_slices_of(index), v.dx_tile_n, v.dx_stages)

    def dw_acc_stage(self, index: int) -> str:
        """The weight-gradient accumulate variant of chunk ``index``."""
        v = self.variants_of(index)
        return stage_variant(
            "gemm_dw_acc",
            group_m=v.dw_group_m,
            epi_store=v.dw_epi_store,
            group_n=v.dw_group_n,
        )

    @property
    def compact(self) -> bool:
        """The chunk loop runs over the valid rows only (gather / scatter around the kernels)."""
        return self.valid_rows is not None

    @property
    def rows(self) -> int:
        """Rows the chunk loop processes: the valid rows when compacted, else ``T``."""
        return int(self.valid_rows) if self.compact else int(self.problem.num_rows)

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def num_tiles(self) -> int:
        return -(-self.problem.vocab // self.geometry.stats_tile)

    def dx_slices_of(self, index: int) -> int:
        return int(self.dx_slices[index]) if self.dx_slices else 1

    @property
    def dx_ws_slabs(self) -> int:
        """FP32 ``[rows, H]`` slabs of the K-slice workspace the plan needs (largest slice count - 1)."""
        return (
            max((int(k) for k in self.dx_slices), default=1) - 1 if self.need_dx else 0
        )

    @property
    def dw_deferred(self) -> bool:
        """The last chunk's weight-gradient GEMM runs in the backward with the upstream scale and the output cast fused
        into its epilogue (:data:`DW_CAST_STAGES`); the forward accumulates the chunks before it only."""
        return bool(self.need_dw and self.fuse_dw_cast and self.num_chunks > 0)

    @property
    def dw_acc_needed(self) -> bool:
        """The FP32 ``[V, H]`` accumulator exists: every unfused plan, and a fused plan of more than one chunk."""
        return bool(self.need_dw and not (self.dw_deferred and self.num_chunks == 1))

    @property
    def last_chunk(self) -> tuple[int, int]:
        """``(row0, rows_c)`` of the last chunk (``(0, 0)`` without chunks)."""
        return self.chunks[-1] if self.chunks else (0, 0)

    @property
    def stages(self) -> tuple[str, ...]:
        base = stages_for_entry(
            self.problem.entry,
            need_dx=self.need_dx,
            need_dw=self.need_dw,
            grad_weight_dtype=self.problem.grad_weight_dtype,
            fuse_dw_cast=self.fuse_dw_cast,
            num_chunks=self.num_chunks,
            dx_cast=self.dx_cast,
        )
        used = {s for s in base if s not in GEMM_STAGES}
        if self.need_dx and self.dx_finalize and self.dx_ws_slabs:
            used.add(
                "slab_sum"
            )  # some chunk slices its dX GEMM: its slabs are added by the kernel
        for index in range(self.num_chunks):
            if "gemm_logits" in base:
                used.add(self.logits_stage(index))
            if "gemm_logits_nostats" in base:
                used.add(self.logits_stage(index, stats=False))
            if "gemm_dx" in base:
                used.add(self.dx_stage_of(index))
            if "gemm_dw_acc" in base and not (
                self.dw_deferred and index == self.num_chunks - 1
            ):
                used.add(self.dw_acc_stage(index))
        if self.dw_deferred:
            used.add(self.dw_cast_stage)
        return tuple(s for s in STAGES if s in used)

    @property
    def dw_cast_stage(self) -> str:
        """The stage that writes the final ``dW``: the fused GEMM of the deferred last chunk, else the flat cast."""
        fp32 = (
            self.problem.entry == "loss"
            and self.problem.grad_weight_dtype == torch.float32
        )
        if self.dw_deferred:
            v = self.variants_of(self.num_chunks - 1)
            return stage_variant(
                "gemm_dw_cast_f32" if fp32 else "gemm_dw_cast_bf16",
                group_m=v.dw_group_m,
                group_n=v.dw_group_n
                if fp32
                else None,  # the bf16 cast keeps the 1-D raster
            )
        return "scale_cast_f32" if fp32 else "scale_cast_bf16"


def make_plan(
    problem: Problem,
    *,
    need_dx: bool,
    need_dw: bool,
    geometry: Geometry = DEFAULT_GEOMETRY,
    dx_max_slices: int = 1,
    num_sms: int = 1,
    valid_rows: Optional[int] = None,
    dx_resident: Optional[int] = None,
    fuse_dw_cast: bool = False,
    arch: Optional[str] = None,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
    dx_finalize: bool = False,
    dx_cast: bool = True,
) -> Plan:
    """The chunk schedule; ``dx_max_slices`` > 1 (K-sliced dX stages registered) picks each chunk's slice count
    (``dx_resident`` = the device's co-resident dX clusters, :func:`cluster_resident`); ``valid_rows`` (compaction)
    chunks that many rows instead of ``T``; ``fuse_dw_cast`` defers the last chunk's weight-gradient GEMM to the
    backward with the fused scale + cast epilogue.  ``arch`` (a registered program's architecture) applies the
    per-chunk instance rules (:func:`chunk_variants`; ``None`` = the reference path, base stages only) and
    ``gemm_tuning`` pins knobs explicitly (already validated by the caller or :func:`_resolve_gemm_tuning`);
    ``dx_finalize`` adds a sliced dX GEMM's slabs with the ``slab_sum`` kernel and ``dx_cast=False`` leaves ``dx_acc``
    uncast for the caller (:func:`finalize_dx`)."""
    rows = problem.num_rows if valid_rows is None else int(valid_rows)
    chunks = plan_chunks(rows, problem.chunk)
    tuning = gemm_tuning or {}
    dx_tuning = tuning.get("dx", {})
    slices: tuple[int, ...] = ()
    if need_dx and "k_slices" in dx_tuning:
        # an explicit knob wins over the slice rule: a value pins that slice count, ``None`` pins the default form
        # (one slice, the base ``gemm_dx`` stage) -- the same ``None`` semantics as every other pinned knob
        explicit_k = 1 if dx_tuning["k_slices"] is None else int(dx_tuning["k_slices"])
        if not 1 <= explicit_k <= int(dx_max_slices):
            raise ValueError(
                f"gemm_tuning dx k_slices={explicit_k}: the registered program serves 1 .. {int(dx_max_slices)} dX slices"
            )
        slices = tuple(explicit_k for _ in chunks)
    elif need_dx and int(dx_max_slices) > 1:
        slices = tuple(
            recommended_k_slices(
                rows_c, problem.hidden, num_sms, dx_max_slices, geometry, dx_resident
            )
            for _, rows_c in chunks
        )
    variants: tuple[ChunkVariants, ...] = ()
    if arch is not None or tuning:
        variants = tuple(
            chunk_variants(
                rows_c,
                problem.hidden,
                num_sms,
                slices[i] if slices else 1,
                geometry,
                arch,
                tuning,
            )
            for i, (_, rows_c) in enumerate(chunks)
        )
    return Plan(
        problem,
        chunks,
        bool(need_dx),
        bool(need_dw),
        geometry,
        slices,
        None if valid_rows is None else int(valid_rows),
        bool(fuse_dw_cast),
        variants,
        bool(dx_finalize),
        bool(dx_cast),
    )


# ---------------------------------------------------------------------------
# Workspace and memory accounting
# ---------------------------------------------------------------------------


def _align(nbytes: int) -> int:
    return (nbytes + WORKSPACE_ALIGN - 1) // WORKSPACE_ALIGN * WORKSPACE_ALIGN


def workspace_layout(
    num_rows: int,
    vocab: int,
    chunk: int,
    *,
    stats_tile: int = GEOMETRY_DEFAULTS["stats_tile"],
    entry: str = "loss",
    tma_workspace_bytes: int = 0,
    scratch_bytes: int = 0,
    dx_ws_slabs: int = 0,
    hidden: int = 0,
    compact: bool = False,
) -> dict:
    """Byte ``(offset, size)`` of every workspace region plus ``"total"``.

    ``dx_ws_slabs`` > 0 adds the FP32 ``dx_ws [slabs, min(T, C), H]`` slabs of
    the K-sliced dX GEMM (``hidden`` = ``H``); ``compact`` adds the BF16
    ``x_c [min(T, C), H]`` buffer the valid rows of ``X`` are gathered into
    (``num_rows`` is then the valid-row count).

    The workspace holds the temporaries only the launches read: the
    vocabulary buffer ``logits`` (``min(T, C) x V`` BF16, ``z_c`` then ``dz_c``
    in place), its row-statistics partials ``stats``, the chunk-local per-row
    gradient scale ``d`` and loss terms ``term``, the loss accumulator and the
    ``grad_scale`` cell, plus the descriptor storage of pointer-ABI programs
    and kernel-private scratch when a record declares them.  The ``O(T)`` row
    vectors a caller or the autograd graph keeps (``lse``, ``logp``, the
    ``loss`` cell) and the FP32 accumulators ``dX_acc`` / ``dW_acc`` are
    allocated separately so that no returned or saved tensor pins the
    vocabulary buffer.
    """
    rows = min(
        int(chunk), max(int(num_rows), 1)
    )  # the logits workspace never exceeds T rows either
    tiles = -(-int(vocab) // int(stats_tile))
    sizes = [
        ("logits", rows * int(vocab) * 2),
        ("stats", rows * tiles * 2 * 4),
        ("d", rows * 4),
        ("term", rows * 4),
        ("loss_acc", 8),  # FP64 accumulator of the fixed-order loss sum
        ("grad_scale", 4),
    ]
    if dx_ws_slabs:
        if int(hidden) < 1:
            raise ValueError(
                "workspace_layout needs hidden > 0 for the K-slice workspace"
            )
        sizes.append(("dx_ws", int(dx_ws_slabs) * rows * int(hidden) * 4))
    if compact:
        if int(hidden) < 1:
            raise ValueError(
                "workspace_layout needs hidden > 0 for the gather buffer of the compacted rows"
            )
        sizes.append(("x_c", rows * int(hidden) * 2))
    if scratch_bytes:
        sizes.append(("workspace", int(scratch_bytes)))
    if tma_workspace_bytes:
        sizes.append(("tma_descriptor_workspace", int(tma_workspace_bytes)))
    layout: dict = {}
    offset = 0
    for name, nbytes in sizes:
        layout[name] = (offset, nbytes)
        offset += _align(nbytes)
    layout["total"] = offset
    return layout


def _record_tma_bytes(record: dict[str, Any], stages) -> int:
    return max(
        (int(record[s].get("tma_workspace_bytes", 0)) for s in stages), default=0
    )


def _record_scratch_bytes(record: dict[str, Any], stages) -> int:
    return max((int(record[s].get("workspace_bytes", 0)) for s in stages), default=0)


def memory_report(
    num_rows: int,
    hidden: int,
    vocab: int,
    chunk: int,
    *,
    stats_tile: int = GEOMETRY_DEFAULTS["stats_tile"],
    entry: str = "loss",
    need_dx: bool = True,
    need_dw: bool = True,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    return_logp: bool = False,
    x_copy: bool = False,
    tma_workspace_bytes: int = 0,
    scratch_bytes: int = 0,
    dx_ws_slabs: int = 0,
    valid_rows: Optional[int] = None,
    fuse_dw_cast: Optional[bool] = None,
    dx_finalize: Optional[bool] = None,
) -> dict[str, Any]:
    """The reporting buckets of the memory rule (bytes).

    ``temporary``: the workspace regions of :func:`workspace_layout`, the
    ``O(T)`` row vectors (``lse``; ``logp`` unless it is returned; the
    contiguous ``dlogp`` of the log-probability backward) and a contiguous copy
    of ``X`` when its leading stride forced one; ``outputs``: what the caller
    receives (``loss``, ``logp`` when requested, ``dX`` BF16, ``dW`` in
    ``grad_weight_dtype``); ``accumulators``: the FP32 ``dX_acc`` / ``dW_acc``
    the loss entry keeps alive between forward and backward (the
    log-probability entry allocates them in its backward only) and the row
    statistic the log-probability entry saves; ``weights``: the model tensors
    (``W``, ``X``).  ``vocab_rows_max`` is the largest token extent of any
    vocabulary-sized buffer (``min(T, C)``).

    ``valid_rows`` (compaction): the chunk loop, the workspace, ``lse``, the
    ``dX`` accumulator and the saved statistic span the valid rows ``T_v``;
    the temporaries gain the BF16 ``x_c`` gather buffer (``gather_bytes``),
    the int64 row index, the compact ``[T_v]`` ``logp`` / ``dlogp`` before and
    after the scatter and the compact BF16 ``dX`` rows the backward casts
    before scattering them into the ``[T, H]`` output; a contiguous copy of
    ``X`` is never needed (the gather output is contiguous).

    ``fuse_dw_cast`` (default :func:`fuse_dw_cast_default`): the last chunk's
    weight-gradient GEMM runs in the backward with the scale + cast fused, so
    the BF16 chunk buffer holding its ``dz`` rows stays alive until the
    backward -- the saved rows are a view of the whole ``[min(T, C), V]``
    buffer, never a copy, so ``saved_dz_bytes`` counts that whole buffer (part
    of the ``temporary`` chunk buffer; the eager entry points allocate it on
    its own from the first call on, so it is also the only storage the saved
    view keeps alive, while a prepared runner keeps it inside its workspace) --
    and a one-chunk call has no FP32 ``dW_acc`` at all.

    ``dx_finalize`` (default :func:`dx_finalize_default`): a compacted call's
    ``dX`` is written in one pass from the accumulator, so the compact BF16
    ``dX`` rows disappear from the temporaries; the bool valid-row mask and its
    int32 scan (``row_valid`` / ``row_scan``, ``[T]`` each) take their place.
    """
    compact = valid_rows is not None
    rows = (
        int(valid_rows) if compact else int(num_rows)
    )  # rows the chunk loop processes
    chunks = plan_chunks(rows, chunk)
    deferred = bool(need_dw and _resolve_fuse(fuse_dw_cast) and chunks)
    fused_dx = bool(need_dx and _resolve_dx_finalize(dx_finalize))
    layout = workspace_layout(
        rows,
        vocab,
        chunk,
        stats_tile=stats_tile,
        entry=entry,
        tma_workspace_bytes=tma_workspace_bytes,
        scratch_bytes=scratch_bytes,
        dx_ws_slabs=dx_ws_slabs if need_dx else 0,
        hidden=hidden,
        compact=compact,
    )
    temporary = {
        name: size
        for name, (_, size) in ((k, v) for k, v in layout.items() if k != "total")
    }
    temporary["lse"] = rows * 4
    if entry == "loss" and (compact or not return_logp):
        temporary["logp"] = (
            rows * 4
        )  # compacted: the [T_v] rows before the scatter into the returned [T] logp
    if entry == "logprob":
        temporary["dlogp"] = int(num_rows) * 4
    if compact:
        temporary["row_index"] = rows * 8  # int64 valid-row index
        if entry == "logprob":
            temporary["dlogp_compact"] = rows * 4
        if need_dx and fused_dx:
            temporary["row_valid"] = int(
                num_rows
            )  # the bool valid-row mask the fused dX finalize scatters by
            temporary["row_scan"] = (
                int(num_rows) * 4
            )  # its inclusive int32 scan (formed in the backward)
        elif need_dx:
            temporary["dx_compact"] = (
                rows * int(hidden) * 2
            )  # the cast compact rows, transient before the scatter into dX
    if x_copy and not compact:
        temporary["x_copy"] = int(num_rows) * int(hidden) * 2
    outputs = {
        "loss": 4 if entry == "loss" else 0,
        "logp": int(num_rows) * 4 if (return_logp or entry == "logprob") else 0,
    }
    if need_dx:
        outputs["dX"] = int(num_rows) * int(hidden) * 2
    if need_dw:
        outputs["dW"] = (
            int(vocab)
            * int(hidden)
            * (4 if (entry == "loss" and grad_weight_dtype == torch.float32) else 2)
        )
    accumulators = {}
    if need_dx:
        accumulators["dX_acc"] = rows * int(hidden) * 4
    if need_dw and not (deferred and len(chunks) == 1):
        accumulators["dW_acc"] = int(vocab) * int(hidden) * 4
    if entry == "logprob":
        accumulators["saved_lse"] = rows * 4
    weights = {"W": int(vocab) * int(hidden) * 2, "X": int(num_rows) * int(hidden) * 2}
    return dict(
        temporary_bytes=sum(temporary.values()),
        temporary=temporary,
        outputs_bytes=sum(outputs.values()),
        outputs=outputs,
        accumulator_bytes=sum(accumulators.values()),
        accumulators=accumulators,
        weights_bytes=sum(weights.values()),
        weights=weights,
        vocab_rows_max=min(int(chunk), max(rows, 1)),
        chunk=int(chunk),
        num_chunks=len(chunks),
        compact_rows=compact,
        valid_rows=rows,
        gather_bytes=int(layout["x_c"][1]) if compact else 0,
        fuse_dw_cast=deferred,
        saved_dz_bytes=min(int(chunk), rows) * int(vocab) * 2 if deferred else 0,
        dx_finalize=fused_dx,
    )


def lm_head_loss_workspace_size(
    num_rows: int,
    vocab: int,
    chunk: int = DEFAULT_CHUNK_SIZE,
    device: Optional[torch.device] = None,
    *,
    entry: str = "loss",
    backend: str = "cake",
    hidden: Optional[int] = None,
    need_dx: bool = True,
    compact_rows: bool = False,
) -> int:
    """Workspace bytes :func:`prepare_lm_head_loss` needs for ``(T, V, C)`` on ``device`` (``hidden`` = ``H``;
    defaults to the registered program's pinned hidden size).  ``compact_rows`` adds the gather buffer of the
    compacted path; sized for ``T`` rows it bounds every valid-row count -- including the K-slice slabs of a
    compacted plan, whose tail chunk (the valid-row count modulo ``C``) may take more slices than any full-``T``
    chunk (the slab count is taken over every possible tail row count as well)."""
    if compact_rows and not hidden and backend == "reference":
        raise ValueError(
            "lm_head_loss_workspace_size needs hidden= for the compacted path"
        )
    if backend == "reference":
        return int(
            workspace_layout(
                num_rows,
                vocab,
                chunk,
                entry=entry,
                hidden=hidden or 0,
                compact=compact_rows,
            )["total"]
        )
    name, record = record_for(device, hidden, int(vocab))
    stages = registered_stages(name)
    geometry = Geometry.from_record(record)
    hidden = geometry.hidden if hidden is None else int(hidden)
    slabs = 0
    if need_dx and dx_max_slices(stages) > 1:
        if hidden is None:
            raise ValueError(
                "lm_head_loss_workspace_size needs hidden= for a program with K-sliced dX stages"
            )
        num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
        resident = cluster_resident(
            torch.device("cuda", torch.cuda.current_device())
            if device is None
            else torch.device(device),
            geometry.dx_cluster_ctas,
            num_sms,
        )
        rows_candidates = [rows_c for _, rows_c in plan_chunks(num_rows, chunk)]
        if compact_rows:
            # the valid-row count is unknown here: the compacted plan's tail chunk can have any row count up to one
            # chunk; the slice rule depends on rows only through the row-tile count, so one row count per tile suffices
            cap = min(int(num_rows), int(chunk))
            rows_candidates += list(range(1, cap + 1, int(geometry.row_tile))) + [cap]
        slabs = (
            max(
                (
                    recommended_k_slices(
                        rows_c,
                        hidden,
                        num_sms,
                        dx_max_slices(stages),
                        geometry,
                        resident,
                    )
                    for rows_c in rows_candidates
                ),
                default=1,
            )
            - 1
        )
    return int(
        workspace_layout(
            num_rows,
            vocab,
            chunk,
            stats_tile=geometry.stats_tile,
            entry=entry,
            tma_workspace_bytes=_record_tma_bytes(record, stages),
            scratch_bytes=_record_scratch_bytes(record, stages),
            dx_ws_slabs=slabs,
            hidden=hidden or 0,
            compact=bool(compact_rows),
        )["total"]
    )


def _carve(flat: torch.Tensor, layout: dict, name: str, dtype, shape) -> torch.Tensor:
    offset, nbytes = layout[name]
    needed = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if needed > nbytes:
        raise ValueError(
            f"workspace region {name!r} holds {nbytes} bytes, {needed} needed"
        )
    return flat[offset : offset + needed].view(dtype).view(shape)


# ---------------------------------------------------------------------------
# Launch binding
# ---------------------------------------------------------------------------

_GRID_FUNCTIONS = {"min": min, "max": max}


def grid_dims(
    rule, scalars: dict[str, Any], num_sms: int, resident: Optional[int] = None
) -> tuple[int, int, int]:
    """Evaluate a registry grid rule.

    Each of the three entries is an integer or an integer expression over the
    scalar names of the stage's host values (``rows_c``, ``m_tiles``, ``V``,
    ``num_vecs``, ...), ``sms`` (the SM count) and ``resident`` (the co-resident
    clusters of the stage's cluster width, :func:`cluster_resident`; only a
    clustered stage provides it) with ``+``, ``-``, ``*``, ``/`` (rounds up),
    ``//`` (rounds down), parentheses and ``min`` / ``max``: ``"rows_c/8"`` (one
    CTA per eight rows), ``"max(1, min(m_tiles//4*24, resident))*4"`` (a
    persistent four-CTA cluster grid capped by the co-resident clusters),
    ``"max(1, m_tiles//2*605)*2"`` (the whole work-item domain of a dynamically
    scheduled kernel).
    """
    if len(rule) != 3:
        raise ValueError("grid rule must have three entries")
    # Only the integer host values take part; tensors and None never enter the evaluation environment, and the
    # evaluator is a plain module-level function -- a nested recursive closure would form a reference cycle that
    # keeps every captured host value (the accumulators / outputs of a scale_cast call) alive until the cyclic GC.
    names = {
        key: int(value)
        for key, value in scalars.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    names["sms"] = int(num_sms)
    if resident is not None:
        names["resident"] = int(resident)
    x, y, z = (_grid_term(value, names) for value in rule)
    return max(1, x), max(1, y), max(1, z)


def _grid_term(value, names: dict[str, int]) -> int:
    if isinstance(value, bool):
        raise ValueError("grid rule entries must be integers or expressions")
    if isinstance(value, int):
        return int(value)
    try:
        tree = ast.parse(str(value).strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"grid rule entry {value!r} is not an expression") from exc
    return _grid_eval(tree, names)


def _grid_eval(node, names: dict[str, int]) -> int:
    if isinstance(node, ast.Expression):
        return _grid_eval(node.body, names)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError(
                f"grid rule constants must be integers, got {node.value!r}"
            )
        return int(node.value)
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise KeyError(f"grid rule names the unknown scalar {node.id!r}")
        return names[node.id]
    if isinstance(node, ast.BinOp):
        left, right = _grid_eval(node.left, names), _grid_eval(node.right, names)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, (ast.Div, ast.FloorDiv)):
            if right == 0:
                raise ValueError("grid rule divides by zero")
            return -(-left // right) if isinstance(node.op, ast.Div) else left // right
        raise ValueError(f"grid rule operator {type(node.op).__name__} is not allowed")
    if isinstance(node, ast.Call):
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id not in _GRID_FUNCTIONS
            or node.keywords
        ):
            raise ValueError("grid rule calls must be min(...) or max(...)")
        if len(node.args) < 2:
            raise ValueError("grid rule min()/max() take at least two terms")
        return _GRID_FUNCTIONS[node.func.id](
            _grid_eval(arg, names) for arg in node.args
        )
    raise ValueError(f"grid rule syntax {type(node).__name__} is not allowed")


@dataclass(frozen=True)
class _Launch:
    stage: str
    module: str
    entry: Callable[..., Any] = field(repr=False)
    arguments: tuple = field(repr=False)
    grid: tuple[int, int, int]
    # Descriptor-preparation entry of a pointer-ABI module (same arguments).
    prepare: Optional[Callable[..., Any]] = field(default=None, repr=False)
    # ``(argument index, host value name)`` of every tensor argument (re-bound per call by a remembered binding).
    slots: tuple[tuple[int, str], ...] = ()

    def __call__(self) -> None:
        self.entry(*self.arguments)

    def templated(self) -> "_Launch":
        """Copy whose tensor arguments are ``None`` placeholders (holds no tensor)."""
        arguments = list(self.arguments)
        for index, _ in self.slots:
            arguments[index] = None
        return replace(self, arguments=tuple(arguments))

    def arguments_for(self, values: dict[str, Any]) -> list:
        """The argument list with every slot filled from ``values`` (fails closed on a missing one)."""
        arguments = list(self.arguments)
        for index, key in self.slots:
            value = values.get(key)
            if value is None:
                raise RuntimeError(
                    f"stage {self.stage!r} of {self.module!r}: the remembered argument plan needs "
                    f"{key!r} (argument {index}) and the call provides no value for it"
                )
            arguments[index] = value
        return arguments


def bind_stage(
    module_name: str, stage: str, values: dict[str, Any], grid: tuple[int, int, int]
) -> _Launch:
    """Order ``values`` by the generated argument plan of ``stage`` and load its entry.

    Fails closed: a keyword the kernel expects that the host does not provide
    raises ``KeyError`` naming both sides.  The returned launch records which
    argument positions hold tensors (re-bound per call by a remembered binding).
    """
    physical = MODULES[module_name][stage]
    grid_values = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    arguments = []
    slots = []
    for kind, name in physical["arg_plan"]:
        key = name if name in values else CONTRACT_ALIASES.get(name, name)
        if kind == "grid":
            arguments.append(grid_values[name])
        elif (
            key in values and values[key] is not None
        ):  # buffer / tma_buffer / workspace / parameter
            value = values[key]
            if isinstance(value, torch.Tensor):
                slots.append((len(arguments), key))
            arguments.append(value)
        else:
            raise KeyError(
                f"generated module {module_name!r} stage {stage!r} expects argument {name!r} ({kind}); "
                f"the host binding provides {sorted(k for k, v in values.items() if v is not None)}"
            )
    module = load_cake_lm_head_loss_module(module_name, stage)
    prepare_entry = physical.get("tma_prepare_entry")
    prepare = getattr(module, prepare_entry) if prepare_entry else None
    return _Launch(
        stage,
        module_name,
        getattr(module, physical["ffi_entry"]),
        tuple(arguments),
        grid,
        prepare,
        tuple(slots),
    )


_FFI_DEVICES: dict[int, Any] = {}


@functools.lru_cache(maxsize=None)
def _device_constants(index: int) -> dict[str, torch.Tensor]:
    """The constant cells a remembered binding binds on every call, allocated once per device:
    ``grad_scale`` / ``unit_scale`` (``[1] = 1.0``; read-only here -- the loss entry's backward
    casts through :func:`backward_loss`, the log-probability entry's cast scale is 1) and the
    unused fp32 ``WS`` of the unsliced GEMMs.  Per-call ``torch.ones`` / ``torch.zeros`` would add
    three fill launches to every remembered forward and backward."""
    device = torch.device("cuda", int(index))
    return {
        "grad_scale": torch.ones((1,), dtype=torch.float32, device=device),
        "unit_scale": torch.ones((1,), dtype=torch.float32, device=device),
        "f32_dummy": torch.zeros((16,), dtype=torch.float32, device=device),
    }


def _unit_scale(device: torch.device) -> torch.Tensor:
    """The ``[1] = 1.0`` fp32 scale of a cast whose incoming gradient is ``None`` (the log-probability
    entry's dX finalize): the per-device cached cell on CUDA -- a per-call ``torch.ones`` would be one
    fill launch inside the backward -- and a fresh cell elsewhere (the reference backend on the host)."""
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        return _device_constants(int(index))["unit_scale"]
    return torch.ones((1,), dtype=torch.float32, device=device)


def _ffi_stream_context(index: int):
    """tvm-ffi environment-stream context for torch's current stream on device ``index``."""
    import tvm_ffi

    device = _FFI_DEVICES.get(index)
    if device is None:
        device = _FFI_DEVICES[index] = tvm_ffi.device(f"cuda:{index}")
    getter = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    raw = (
        getter(index)
        if getter is not None
        else torch.cuda.current_stream(index).cuda_stream
    )
    return tvm_ffi.use_raw_stream(device, raw)


def side_stream_schedule(plan: Plan, keys: tuple) -> Optional[tuple]:
    """Side-stream placement of :func:`dw_side_stream` over a launch-key sequence: ``None`` when the plan has no ``dW``,
    too few chunks for the rule, or the sequence has no per-chunk accumulate (the loss entry's cast keys); else the
    positions ``(joins, forks, sides)`` -- the first key of every chunk after the first and the first key after the last
    chunk's keys (the caller's stream waits for the side stream: the chunk buffer is about to be reused / read), each
    chunk's ``row_grad`` key (the fork event is recorded after it) and the per-chunk ``dW`` accumulate keys (launched
    on the side stream after it waited for the fork).  A deferred last chunk's cast GEMM and the flat casts stay on
    the caller's stream; the sequence always ends with a join."""
    if not (plan.need_dw and dw_side_stream(plan.num_chunks)):
        return None
    chunk_keys = [
        (pos, stage, index)
        for pos, (stage, index) in enumerate(keys)
        if isinstance(index, int) and not isinstance(index, bool)
    ]
    sides = frozenset(
        pos for pos, stage, index in chunk_keys if stage == plan.dw_acc_stage(index)
    )
    if not sides:
        return None
    joins, forks, seen = set(), set(), set()
    for pos, stage, index in chunk_keys:
        if index not in seen:
            seen.add(index)
            if index > 0:
                joins.add(pos)
        if stage == "row_grad":
            forks.add(pos)
    after = chunk_keys[-1][0] + 1
    if after < len(keys):
        joins.add(after)
    return frozenset(joins), frozenset(forks), sides


def _run_keys(
    plan: Plan, device_index: int, keys: tuple, run_key: Callable[[Any], None]
) -> None:
    """Run the launch keys in order on the caller's stream; with the placement of :func:`side_stream_schedule` the
    per-chunk ``dW`` accumulates go to the side stream.  The generated launches take tvm-ffi's environment stream, so a
    side launch enters its own stream context and the caller's context is re-entered afterwards (no nesting)."""
    schedule = side_stream_schedule(plan, keys)
    if schedule is None:
        with _ffi_stream_context(device_index):
            for key in keys:
                run_key(key)
        return
    joins, forks, sides = schedule
    side, fork = _dw_stream(device_index)
    main = torch.cuda.current_stream(device_index)
    with contextlib.ExitStack() as stack:
        stack.enter_context(_ffi_stream_context(device_index))
        for pos, key in enumerate(keys):
            if pos in joins:
                main.wait_stream(side)
            if pos in sides:
                stack.close()
                side.wait_event(fork)
                with torch.cuda.stream(side), _ffi_stream_context(device_index):
                    run_key(key)
                stack.enter_context(_ffi_stream_context(device_index))
                continue
            run_key(key)
            if pos in forks:
                fork.record(main)
    main.wait_stream(side)


# ---------------------------------------------------------------------------
# Host values of the stages
# ---------------------------------------------------------------------------


def stage_values(
    stage: str, t: dict[str, Any], plan: Plan, index: int, *, order_key: Any = None
) -> dict[str, Any]:
    """Host values of ``stage`` for chunk ``index`` (the kernel's own argument names).

    ``t`` holds the bound tensors: ``X`` (as launched), ``W``, ``labels``,
    ``lse``, ``logp``, ``infer_logp`` / ``loss_weights`` (policy), ``d_in``
    (the ``[T]`` incoming ``dlogp`` of the log-probability backward), the
    workspace regions ``logits`` / ``stats`` / ``d`` / ``term`` / ``loss_acc`` /
    ``grad_scale``, ``loss`` (the finished-loss cell), ``dx_acc`` / ``dw_acc``
    and the cast outputs ``dx_out`` / ``dw_out``.  For the casts ``index`` is
    ``"dx"`` or ``"dw"``.

    Compacted plan: the row-indexed tensors (``labels``, ``lse``, ``logp``,
    ``infer_logp``, ``loss_weights``, ``d_in``, ``dx_acc``, ``dx_out``) hold the
    valid rows only (``T`` is the valid-row count), ``t["row_index"]`` is the
    int64 valid-row index, ``t["x_c"]`` the BF16 ``[C, H]`` gather buffer the
    GEMMs read the chunk's ``X`` rows from, and the host stage ``gather_rows``
    fills them: chunk ``index`` gathers the chunk's rows of ``X``; a string
    ``index`` (``"infer_logp"``, ``"loss_weights"``, ``"d_in"``) gathers the
    caller's ``[T]`` operand ``t[index + "_full"]`` into ``t[index]``.
    """
    p, g = plan.problem, plan.geometry
    values: dict[str, Any] = {name: t.get(name) for name in COMMON_TENSORS}
    values.update(
        T=int(plan.rows),
        H=int(p.hidden),
        V=int(p.vocab),
        chunk=int(p.chunk),
        num_tiles=int(plan.num_tiles),
        mode=int(p.mode),
        loss_div=float(p.loss_div) if p.loss_div is not None else 1.0,
        ws_slab=0,
    )
    if (
        stage == "gather_rows"
    ):  # host-side row gather of the compacted path (no kernel of the program)
        if not plan.compact:
            raise ValueError("gather_rows belongs to a compacted plan")
        if isinstance(index, str):
            values.update(
                src=t[index + "_full"],
                idx=t["row_index"],
                out=t[index],
                rows_c=int(plan.rows),
                row0=0,
                first_chunk=0,
                last_chunk=0,
                d_off=0,
            )
            return values
        row0, rows_c = plan.chunks[index]
        values.update(
            src=t["X"],
            idx=t["row_index"][row0 : row0 + rows_c],
            out=t["x_c"][:rows_c],
            rows_c=int(rows_c),
            row0=int(row0),
            first_chunk=int(index == 0),
            last_chunk=int(index == plan.num_chunks - 1),
            d_off=0,
        )
        return values
    if stage == "scale_cast_scatter_bf16":
        raise ValueError(
            "scale_cast_scatter_bf16 is bound by finalize_dx at the dX output boundary, not by the chunk loop"
        )
    if stage in ("scale_cast_bf16", "scale_cast_f32"):
        acc = t["dx_acc"] if index == "dx" else t["dw_acc"]
        out = t["dx_out"] if index == "dx" else t["dw_out"]
        if acc.numel() != out.numel() or acc.numel() % g.cast_vec:
            raise ValueError(
                "scale-cast operands must match in size and hold whole vectors"
            )
        # loss entry: the backward writes the incoming scalar gradient into the ``grad_scale`` workspace cell;
        # log-probability entry: the scale is the constant 1, held outside the workspace (a caller may poison
        # the workspace between steps; ``dlogp`` carries the gradient)
        scale = t["grad_scale"] if p.entry == "loss" else t["unit_scale"]
        values.update(
            acc=acc.reshape(-1),
            g=scale,
            out=out.reshape(-1),
            num_vecs=int(acc.numel() // g.cast_vec),
            rows_c=0,
            row0=0,
            first_chunk=0,
            last_chunk=0,
            d_off=0,
        )
        return values
    row0, rows_c = plan.chunks[index]
    stop = row0 + rows_c
    first, last = int(index == 0), int(index == plan.num_chunks - 1)
    values.update(
        rows_c=int(rows_c), row0=int(row0), first_chunk=first, last_chunk=last, d_off=0
    )
    x_chunk = (
        t["x_c"][:rows_c] if plan.compact else t["X"][row0:stop]
    )  # compacted: the chunk's gathered valid rows
    if t.get("x_last") is not None and index == plan.num_chunks - 1:
        x_chunk = t[
            "x_last"
        ]  # the eager backward of a deferred plan: the saved view of the last chunk's X rows
    logits = t["logits"]
    dz_chunk = logits[:rows_c]
    base = base_stage(stage) if stage in GEMM_STAGES else stage
    if base in ("gemm_logits", "gemm_logits_nostats"):
        want = plan.logits_stage(index, stats=base == "gemm_logits")
        if stage != want:
            raise ValueError(
                f"chunk {index} plans the logits GEMM {want!r}; stage {stage!r} was requested"
            )
        # ``C`` is the chunk's [rows_c, V] rows of the logits workspace: the TMA-store epilogue's tensor map takes its
        # row extent from it (rows >= rows_c are clipped, never written); a pointer store sees the same base address
        values.update(
            A=x_chunk,
            B=t["W"],
            C=logits[:rows_c],
            STATS_OUT=t["stats"],
            WS=t["f32_dummy"],
            M=int(rows_c),
            m_tiles=g.row_tiles(rows_c, g.logits_cluster_ctas),
            k_iters=1,
            first_chunk=0,
        )
    elif base == "gemm_dx":
        # every token chunk writes its own rows of dX_acc: the first K chunk of the GEMM always stores (first_chunk=1);
        # the K-sliced forms write slices >= 1 into the ``dx_ws`` slabs (added by ``dx_reduce`` afterwards)
        k = plan.dx_slices_of(index)
        want = plan.dx_stage_of(index)
        if stage != want:
            raise ValueError(
                f"chunk {index} plans {k} dX slice(s) as {want!r}; stage {stage!r} was requested"
            )
        ws = t["dx_ws"] if k > 1 else t["f32_dummy"]
        values.update(
            A=dz_chunk,
            B=t["W"],
            C=t["dx_acc"][row0:stop],
            STATS_OUT=t["stats"],
            WS=ws,
            M=int(rows_c),
            m_tiles=g.row_tiles(rows_c, g.dx_cluster_ctas),
            k_iters=1,
            first_chunk=1,
            ws_slab=int(ws.stride(0)) if k > 1 else 0,
        )
    elif stage == "slab_sum":
        # the fused dX finalize's slab reduction of the K-sliced dX GEMM: one kernel over the chunk's rows of dX_acc
        k = plan.dx_slices_of(index)
        if not plan.dx_finalize or k <= 1:
            raise ValueError(
                f"slab_sum belongs to a K-sliced dX GEMM of a plan with dx_finalize (chunk {index}: {k} slice(s))"
            )
        ws = t["dx_ws"]
        values.update(
            dx=t["dx_acc"][row0:stop],
            ws=ws,
            ws_slab=int(ws.stride(0)),
            num_vecs=int(rows_c * p.hidden // g.cast_vec),
            n_slabs=int(k - 1),
        )
    elif (
        stage == "dx_reduce"
    ):  # host-side slab reduction of the K-sliced dX GEMM (no kernel of the program)
        values.update(
            acc=t["dx_acc"][row0:stop], ws=t["dx_ws"], k_slices=plan.dx_slices_of(index)
        )
    elif base == "gemm_dw_acc":
        want = plan.dw_acc_stage(index)
        if stage != want:
            raise ValueError(
                f"chunk {index} plans the weight-gradient accumulate {want!r}; stage {stage!r} was requested"
            )
        values.update(
            A=dz_chunk,
            B=x_chunk,
            C=t["dw_acc"],
            STATS_OUT=t["stats"],
            WS=t["f32_dummy"],
            M=int(p.vocab),
            m_tiles=g.row_tiles(p.vocab, g.dw_cluster_ctas),
            k_iters=g.k_iters(rows_c),
            first_chunk=first,
        )
    elif base in DW_CAST_STAGES:
        # the deferred last chunk (backward): the same GEMM as ``gemm_dw_acc`` with the upstream scale and the output cast
        # fused into its epilogue -- ``C`` is the final ``dW``, ``WS`` the read-only FP32 sum of the chunks before it (a
        # dummy for a one-chunk plan, whose ``first_chunk`` form stores ``cast(g * tile)``), ``STATS_OUT`` the scale cell
        if not plan.dw_deferred or not last:
            raise ValueError(
                f"stage {stage!r} belongs to the last chunk of a plan with fuse_dw_cast (chunk {index} of {plan.num_chunks})"
            )
        if stage != plan.dw_cast_stage:
            raise ValueError(
                f"plan casts its weight gradient through {plan.dw_cast_stage!r}; stage {stage!r} was requested"
            )
        values.update(
            A=dz_chunk,
            B=x_chunk,
            C=t["dw_out"],
            STATS_OUT=t["grad_scale"] if p.entry == "loss" else t["unit_scale"],
            WS=t["dw_acc"] if plan.num_chunks > 1 else t["f32_dummy"],
            M=int(p.vocab),
            m_tiles=g.row_tiles(p.vocab, g.dw_cluster_ctas),
            k_iters=g.k_iters(rows_c),
            first_chunk=int(plan.num_chunks == 1),
        )
    elif stage == "row_finalize":
        d = t["d"]

        def present(
            name,
        ):  # policy / external operands: the chunk-local ``d`` stands in when the mode never reads them
            return d if t.get(name) is None else t[name]

        values.update(
            stats=t["stats"],
            z=logits,
            labels=t["labels"],
            infer_logp=present("infer_logp"),
            loss_weights=present("loss_weights"),
            d_in=present("d_in"),
            lse=t["lse"],
            logp=t["logp"],
            d=d,
            term=t["term"],
        )
    elif stage == "loss_reduce":
        values.update(term=t["term"], loss_acc=t["loss_acc"], loss_out=t["loss"])
    elif stage == "row_grad":
        external = (
            p.entry == "logprob"
        )  # the recompute reads the caller's [T] dlogp at d[row0 + r]
        values.update(
            z=logits,
            labels=t["labels"],
            lse=t["lse"],
            d=t["d_in"] if external else t["d"],
            d_off=int(row0) if external else 0,
        )
    else:
        raise ValueError(f"unknown stage {stage!r}")
    return values


def forward_keys(plan: Plan) -> tuple[tuple[str, Any], ...]:
    """Launch keys ``(stage, chunk index)`` of the forward chunk loop (compacted plan: the host gathers first)."""
    keys: list[tuple[str, Any]] = []
    if (
        plan.compact
        and plan.problem.entry == "loss"
        and plan.problem.objective == "policy"
    ):
        keys += [("gather_rows", "infer_logp"), ("gather_rows", "loss_weights")]
    for index in range(plan.num_chunks):
        if plan.compact:
            keys.append(("gather_rows", index))
        keys += [(plan.logits_stage(index), index), ("row_finalize", index)]
        if plan.problem.entry == "loss":
            keys.append(("loss_reduce", index))
            if plan.need_dx or plan.need_dw:
                keys.append(("row_grad", index))
            if plan.need_dx:
                keys += _dx_keys(plan, index)
            if plan.need_dw and not (plan.dw_deferred and index == plan.num_chunks - 1):
                keys.append(
                    (plan.dw_acc_stage(index), index)
                )  # a deferred last chunk runs in the backward (cast_keys)
    return tuple(keys)


def _dx_keys(plan: Plan, index: int) -> list:
    """The dX GEMM of chunk ``index`` plus, when the plan slices it, the slab reduction: the ``slab_sum`` kernel of the
    fused dX finalize or the host's ``add_`` chain (``dx_reduce``)."""
    k = plan.dx_slices_of(index)
    if k <= 1:
        return [(plan.dx_stage_of(index), index)]
    reduce = "slab_sum" if plan.dx_finalize else "dx_reduce"
    return [(plan.dx_stage_of(index), index), (reduce, index)]


def recompute_keys(plan: Plan) -> tuple[tuple[str, Any], ...]:
    """Launch keys of the log-probability backward (recompute the logits, then the gradient GEMMs)."""
    keys: list[tuple[str, Any]] = []
    if plan.compact:
        keys.append(("gather_rows", "d_in"))
    for index in range(plan.num_chunks):
        if plan.compact:
            keys.append(("gather_rows", index))
        keys += [(plan.logits_stage(index, stats=False), index), ("row_grad", index)]
        if plan.need_dx:
            keys += _dx_keys(plan, index)
        if plan.need_dw:
            if plan.dw_deferred and index == plan.num_chunks - 1:
                keys.append(
                    (plan.dw_cast_stage, index)
                )  # the scale is 1 here: the GEMM writes the final BF16 dW
            else:
                keys.append((plan.dw_acc_stage(index), index))
    return tuple(keys)


def cast_keys(plan: Plan) -> tuple[tuple[str, Any], ...]:
    """Launch keys of the output casts: the flat ``dX`` cast and, for ``dW``, the flat cast of the accumulator or -- when
    the plan defers its last chunk -- that chunk's fused GEMM (its ``X`` rows gathered again first when the rows are
    compacted; the log-probability backward runs it inside :func:`recompute_keys` instead)."""
    keys: list[tuple[str, Any]] = []
    if plan.need_dx and plan.dx_cast:
        keys.append(("scale_cast_bf16", "dx"))
    if plan.need_dw:
        if not plan.dw_deferred:
            keys.append((plan.dw_cast_stage, "dw"))
        elif plan.problem.entry == "loss":
            last = plan.num_chunks - 1
            if plan.compact:
                keys.append(("gather_rows", last))
            keys.append((plan.dw_cast_stage, last))
    return tuple(keys)


# ---------------------------------------------------------------------------
# Reference engine: the stages as PyTorch operators over the same host values
# ---------------------------------------------------------------------------


def _mm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` of BF16 operands with FP32 accumulation and an FP32 result (no intermediate rounding)."""
    if a.is_cuda:
        try:
            return torch.mm(a, b, out_dtype=torch.float32)
        except (TypeError, RuntimeError):
            pass
    return torch.mm(a.float(), b.float())


class ReferenceEngine:
    """PyTorch implementation of every stage at the kernels' rounding boundaries.

    Each method consumes the host values a generated stage is bound to
    (:func:`stage_values`), so the chunk loop, the plans and the autograd
    Functions are exercised without a generated program.  Not
    allocation-free (a development aid).
    """

    def run(self, stage: str, values: dict[str, Any]) -> None:
        getattr(self, stage if hasattr(self, stage) else base_stage(stage))(values)

    @staticmethod
    def _rows(values: dict[str, Any]) -> tuple[int, int]:
        return int(values["row0"]), int(values["rows_c"])

    @staticmethod
    def _valid(values: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        row0, rows_c = ReferenceEngine._rows(values)
        labels = values["labels"][row0 : row0 + rows_c].long()
        return labels, labels >= 0

    @staticmethod
    def gemm_logits(values: dict[str, Any], stats: bool = True) -> None:
        rows = int(values["M"])
        z = values["C"][:rows]
        torch.mm(
            values["A"], values["B"].t(), out=z
        )  # BF16 result of an FP32 accumulation
        if stats:
            tiles = int(values["num_tiles"])
            zf = z.float().view(rows, tiles, -1)
            pmax = zf.amax(-1)
            psum = torch.exp(zf - pmax[..., None]).sum(-1)
            out = values["STATS_OUT"][:rows]
            out[..., 0] = pmax
            out[..., 1] = psum

    @staticmethod
    def gemm_logits_nostats(values: dict[str, Any]) -> None:
        ReferenceEngine.gemm_logits(values, stats=False)

    @staticmethod
    def row_finalize(values: dict[str, Any]) -> None:
        row0, rows_c = ReferenceEngine._rows(values)
        stop = row0 + rows_c
        stats = values["stats"][:rows_c]
        pmax, psum = stats[..., 0], stats[..., 1]
        m = pmax.amax(-1)
        lse = m + torch.log((psum * torch.exp(pmax - m[:, None])).sum(-1))
        labels, valid = ReferenceEngine._valid(values)
        index = torch.where(valid, labels, torch.zeros_like(labels)).clamp_max(
            int(values["V"]) - 1
        )
        zy = values["z"][:rows_c].gather(1, index[:, None]).squeeze(1).float()
        zero = torch.zeros_like(lse)
        logp = torch.where(valid, zy - lse, zero)
        values["lse"][row0:stop] = lse
        values["logp"][row0:stop] = logp
        mode = int(values["mode"])
        d, term = zero, zero
        if mode == MODE_CE:
            d = torch.where(
                valid, torch.full_like(lse, -1.0 / float(values["loss_div"])), zero
            )
            term = logp
        elif mode == MODE_POLICY:
            ratio = torch.exp(logp - values["infer_logp"][row0:stop])
            w = values["loss_weights"][row0:stop]
            d = torch.where(valid & (ratio <= RATIO_CLIP), -w * ratio, zero)
            term = torch.where(valid, w * torch.clamp_max(ratio, RATIO_CLIP), zero)
        elif mode == MODE_EXTERNAL:
            d = torch.where(valid, values["d_in"][row0:stop], zero)
        values["d"][:rows_c] = d
        values["term"][:rows_c] = term

    @staticmethod
    def loss_reduce(values: dict[str, Any]) -> None:
        rows_c = int(values["rows_c"])
        total = (
            values["term"][:rows_c].double().sum().reshape(1)
        )  # FP64 accumulation of the FP32 terms
        acc = values["loss_acc"]
        if int(values["first_chunk"]):
            acc.copy_(total)
        else:
            acc.add_(total)
        if int(values["last_chunk"]):
            neg = -acc
            values["loss_out"].copy_(
                neg / float(values["loss_div"])
                if int(values["mode"]) == MODE_CE
                else neg
            )  # one FP64 -> FP32 rounding

    @staticmethod
    def row_grad(values: dict[str, Any]) -> None:
        row0, rows_c = ReferenceEngine._rows(values)
        z = values["z"][:rows_c]
        labels, valid = ReferenceEngine._valid(values)
        lse = values["lse"][row0 : row0 + rows_c]
        d_off = int(values["d_off"])
        d = torch.where(
            valid, values["d"][d_off : d_off + rows_c], torch.zeros_like(lse)
        )
        p = torch.exp(z.float() - lse[:, None])
        index = torch.where(valid, labels, torch.zeros_like(labels)).clamp_max(
            int(values["V"]) - 1
        )
        onehot = torch.zeros_like(p)
        onehot.scatter_(1, index[:, None], 1.0)
        dz = d[:, None] * (onehot - p)
        dz[~valid] = 0.0
        z.copy_(dz)  # the BF16 dlogits boundary

    @staticmethod
    def gemm_dx(values: dict[str, Any]) -> None:
        values["C"].copy_(_mm_fp32(values["A"], values["B"]))

    gemm_dx_s2 = gemm_dx_s3 = gemm_dx_s4 = (
        gemm_dx  # the reference path never slices K (one product per chunk)
    )

    @staticmethod
    def dx_reduce(values: dict[str, Any]) -> None:
        _dx_reduce(values)

    @staticmethod
    def slab_sum(values: dict[str, Any]) -> None:
        # the fused dX finalize's slab reduction: the same fixed-order FP32 adds as ``dx_reduce``
        dx, ws = values["dx"], values["ws"]
        for slab in range(int(values["n_slabs"])):
            dx.add_(ws[slab, : dx.shape[0]])

    @staticmethod
    def scale_cast_scatter_bf16(values: dict[str, Any]) -> None:
        # ``out[idx] = bf16(g * acc)``, exact zeros elsewhere (``idx_lo`` is the int32 view of the int64 index)
        idx = values["idx_lo"].view(torch.int64)
        out = values["out"]
        out.zero_()
        if idx.numel():
            out.index_copy_(0, idx, (values["acc"] * values["g"]).to(out.dtype))

    @staticmethod
    def gemm_dw_acc(values: dict[str, Any]) -> None:
        product = _mm_fp32(values["A"].t(), values["B"])
        if int(values["first_chunk"]):
            values["C"].copy_(product)
        else:
            values["C"].add_(product)

    @staticmethod
    def gemm_dw_cast_bf16(values: dict[str, Any]) -> None:
        product = _mm_fp32(values["A"].t(), values["B"])
        acc = (
            product if int(values["first_chunk"]) else values["WS"] + product
        )  # the FP32 accumulate of gemm_dw_acc (one RN add per element), WS never written
        values["C"].copy_(
            acc * values["STATS_OUT"]
        )  # scale_cast's FP32 multiply, one rounding at the copy

    gemm_dw_cast_f32 = gemm_dw_cast_bf16

    @staticmethod
    def scale_cast_bf16(values: dict[str, Any]) -> None:
        values["out"].copy_(values["acc"] * values["g"])  # one rounding at the copy

    scale_cast_f32 = scale_cast_bf16


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class LmHeadLossRunner:
    """The prepared launches of one tensor binding.

    Loss entry: ``forward()`` runs the chunk loop (three GEMMs per chunk) and
    returns ``(loss [1], logp [T])`` while filling ``dx_acc`` (FP32 ``[T, H]``)
    and ``dw_acc`` (FP32 ``[V, H]``); ``backward(grad)`` casts ``dX = bf16(g *
    dX_acc)`` into ``dx_out`` and ``dW = cast(g * dW_acc)`` into ``dw_out``
    (``g`` from the ``[1]`` FP32 ``grad_scale`` cell; ``None`` = 1); with
    ``fuse_dw_cast`` the last chunk's weight-gradient GEMM runs there instead
    and writes ``dw_out = cast(g * (dw_acc + dz_c^T @ X_c))`` from its
    epilogue (that chunk's ``dz`` stays in the workspace between the calls).
    Log-probability entry: ``forward()`` returns ``(logp, lse)``;
    ``backward()`` reads ``dlogp`` (the bound FP32 ``[T]`` tensor
    ``tensors["d_in"]``, masked to valid rows by the row kernel), recomputes
    the logits per chunk and produces ``dx_out`` / ``dw_out`` (BF16) through
    the accumulators.  No launch of the ``cake`` backend allocates or
    synchronizes; capture into a CUDA graph belongs to the caller.  Prepare a
    new runner when a shape, dtype, stride, option or tensor binding changes;
    values may change freely.

    Compacted runner (``compact_rows``): the chunk loop covers the valid rows
    of the labels as of preparation (``row_index``, ``valid_rows``); ``logp``,
    ``lse``, ``dx_acc`` and ``dx_out`` are the compact ``[T_v]`` / ``[T_v, H]``
    forms (:meth:`scatter` restores ``[T]``), ``dlogp`` stays the caller's
    ``[T]`` tensor (gathered per step).  Label values decide the compaction,
    so a runner must be re-prepared when the set of ignored rows changes.
    """

    backend: str
    module_name: Optional[str]
    plan: Plan
    tensors: dict[str, Any] = field(repr=False)
    values: dict[Any, dict[str, Any]] = field(repr=False)
    launches: dict[Any, _Launch] = field(repr=False)
    forward_order: tuple = ()
    backward_order: tuple = ()
    device_index: int = 0
    workspace: Optional[torch.Tensor] = field(default=None, repr=False)
    layout: dict = field(default_factory=dict, repr=False)
    memory: dict = field(default_factory=dict, repr=False)
    engine: Optional[ReferenceEngine] = field(default=None, repr=False)
    _tma_prepared: bool = False

    @property
    def problem(self) -> Problem:
        return self.plan.problem

    @property
    def loss(self) -> torch.Tensor:
        return self.tensors["loss"]

    @property
    def logp(self) -> torch.Tensor:
        return self.tensors["logp"]

    @property
    def lse(self) -> torch.Tensor:
        return self.tensors["lse"]

    @property
    def dx_acc(self) -> Optional[torch.Tensor]:
        return self.tensors.get("dx_acc")

    @property
    def dw_acc(self) -> Optional[torch.Tensor]:
        return self.tensors.get("dw_acc")

    @property
    def dx_out(self) -> Optional[torch.Tensor]:
        return self.tensors.get("dx_out")

    @property
    def dw_out(self) -> Optional[torch.Tensor]:
        return self.tensors.get("dw_out")

    @property
    def dlogp(self) -> Optional[torch.Tensor]:
        return self.tensors.get("d_in_full", self.tensors.get("d_in"))

    @property
    def row_index(self) -> Optional[torch.Tensor]:
        """Int64 ``[T_v]`` index of the valid rows the compacted chunk loop processes (``None`` when uncompacted)."""
        return self.tensors.get("row_index")

    @property
    def valid_rows(self) -> int:
        return self.plan.rows

    def scatter(self, rows: torch.Tensor) -> torch.Tensor:
        """A compact ``[T_v, ...]`` result (``logp``, ``lse``, ``dx_acc``, ``dx_out``) as the caller's ``[T, ...]`` tensor
        with exact zeros on the ignored rows (the tensor itself when the runner is not compacted)."""
        return scatter_rows(rows, self.row_index, self.problem.num_rows)

    @property
    def stages(self) -> tuple[str, ...]:
        return self.plan.stages

    def prepare_tma(self) -> None:
        """Encode the descriptors of pointer-ABI stages once (idempotent)."""
        if self._tma_prepared or self.backend != "cake":
            return
        with _ffi_stream_context(self.device_index):
            for launch in self.launches.values():
                if launch.prepare is not None:
                    launch.prepare(*launch.arguments)
        self._tma_prepared = True

    def _run(self, keys: tuple) -> None:
        if self.backend == "cake":
            self.prepare_tma()

            def run_key(key) -> None:
                host = HOST_STAGES.get(key[0])
                if host is not None:
                    host(self.values[key])
                else:
                    self.launches[key]()

            _run_keys(self.plan, self.device_index, keys, run_key)
        else:
            for key in keys:
                host = HOST_STAGES.get(key[0])
                if host is not None:
                    host(self.values[key])
                else:
                    self.engine.run(key[0], self.values[key])

    def forward(self):
        """Loss entry: ``(loss [1], logp [T])``; log-probability entry: ``(logp, lse)``."""
        t = self.tensors
        self._run(self.forward_order)
        return (
            (t["loss"], t["logp"])
            if self.problem.entry == "loss"
            else (t["logp"], t["lse"])
        )

    def backward(self, grad: Optional[torch.Tensor] = None):
        """``(dx_out, dw_out)`` (``None`` for a frozen input).

        Loss entry: ``grad`` is the incoming scalar gradient (``None`` = 1);
        the saved accumulators are read, never written.  Log-probability
        entry: the caller has written ``dlogp`` into the bound ``dlogp``
        tensor; ``grad`` must be ``None`` (the cast scale is 1).
        """
        t = self.tensors
        if self.problem.entry == "logprob":
            if grad is not None:
                raise ValueError(
                    "the log-probability runner takes its gradient from the bound dlogp tensor"
                )
            # the cast scale of this entry is the constant 1 (``unit_scale``, outside the workspace)
        elif grad is None:
            t["grad_scale"].fill_(1.0)
        else:
            t["grad_scale"].copy_(grad.reshape(1).to(torch.float32))
        self._run(self.backward_order)
        dx = (
            t.get("dx_out") if self.plan.dx_cast else t.get("dx_acc")
        )  # dx_cast=False: the caller finalizes the accumulator
        return dx, t.get("dw_out")

    def step(self, grad: Optional[torch.Tensor] = None):
        self.forward()
        return self.backward(grad)

    __call__ = step


def _check_output(t: Optional[torch.Tensor], name: str, shape: tuple, dtype) -> None:
    if t is None:
        return
    if tuple(t.shape) != tuple(shape) or t.dtype != dtype or not t.is_contiguous():
        raise ValueError(
            f"{name} must be a contiguous {dtype} tensor of shape {tuple(shape)}"
        )


def _prepare_x(
    X: torch.Tensor, problem: Problem, compact: bool = False
) -> torch.Tensor:
    return (
        X.contiguous() if (problem.x_copy and not compact) else X
    )  # the per-chunk gather yields contiguous rows


def _prepare_labels(
    labels: torch.Tensor, geometry: Geometry, row_index: Optional[torch.Tensor] = None
) -> torch.Tensor:
    labels = (
        labels.contiguous() if row_index is None else labels.index_select(0, row_index)
    )
    if geometry.labels_dtype != labels.dtype:
        labels = labels.to(
            geometry.labels_dtype
        )  # one O(T) cast per call when the kernels read int32
    return labels


def _bind_all(
    record, module_name, keys, values, device, geometry: Geometry
) -> dict[Any, _Launch]:
    num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    launches: dict[Any, _Launch] = {}
    for key in keys:
        stage = key[0]
        if (
            stage in HOST_STAGES
        ):  # host-side step (slab reduction, row gather), no kernel
            continue
        physical = record[stage]
        cluster = physical.get("launch", {}).get("cluster") or (1, 1, 1)
        cluster_ctas = int(math.prod(int(c) for c in cluster))
        declared = geometry.cluster_ctas_of(stage)
        if declared is not None and declared != cluster_ctas:
            raise ValueError(
                f"stage {stage!r}: the record's geometry declares {declared}-CTA clusters, its module launches {tuple(cluster)}"
            )
        resident = (
            cluster_resident(device, cluster_ctas, num_sms)
            if cluster_ctas > 1
            else None
        )
        grid = grid_dims(
            physical.get("grid", ["rows_c", 1, 1]),
            values[key],
            num_sms,
            resident=resident,
        )
        if cluster_ctas > 1 and any(g % c for g, c in zip(grid, cluster, strict=True)):
            raise ValueError(
                f"stage {stage!r}: grid {grid} is not a multiple of the cluster shape {tuple(cluster)} baked into the module"
            )
        launches[key] = bind_stage(module_name, stage, values[key], grid)
    return launches


def prepare_lm_head_loss(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    objective: str = "ce",
    loss_div=None,
    infer_logp: Optional[torch.Tensor] = None,
    loss_weights: Optional[torch.Tensor] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    need_dx: bool = True,
    need_dw: bool = True,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    deterministic: bool = True,
    entry: str = "loss",
    dlogp: Optional[torch.Tensor] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    dx_acc: Optional[torch.Tensor] = None,
    dw_acc: Optional[torch.Tensor] = None,
    dx_out: Optional[torch.Tensor] = None,
    dw_out: Optional[torch.Tensor] = None,
    logp: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    backend: str = "cake",
    compact_rows=False,
    fuse_dw_cast: Optional[bool] = None,
    gemm_tuning: Optional[dict[str, dict[str, Any]]] = None,
    dx_finalize: Optional[bool] = None,
    dx_cast: bool = True,
) -> LmHeadLossRunner:
    """Validate one binding and prepare its launches.

    Missing outputs, accumulators and the workspace are allocated here (the
    only allocations of the ``cake`` backend).  Pass ``workspace_buffer`` of
    :func:`lm_head_loss_workspace_size` bytes to reuse storage across steps.
    For the log-probability entry ``dlogp`` is the FP32 ``[T]`` tensor the
    backward reads (allocated here when absent) and ``lse`` the statistic the
    forward writes / the backward reads.  ``T == 0`` is rejected here (the
    eager entry points return zeros for it without a runner).
    ``backend="reference"`` builds the same runner over PyTorch operators.

    ``compact_rows=True`` compacts the chunk loop to the valid rows of
    ``labels`` (:func:`valid_row_index`; an int64 index tensor is accepted in
    place of ``True``): ``lse`` / ``logp`` / ``dx_acc`` / ``dx_out`` are then
    bound or allocated as compact ``[T_v]`` / ``[T_v, H]`` tensors, the row
    operands are gathered per step and every row ignored is rejected (the
    eager entry points return zeros for it).

    ``fuse_dw_cast`` (default :func:`fuse_dw_cast_default`): the last chunk's
    weight-gradient GEMM runs in ``backward()`` with the scale + cast fused
    (``dw_acc`` is then the FP32 sum of the chunks before it and absent for a
    one-chunk plan); the chunk buffer must not be overwritten between
    ``forward()`` and ``backward()``.

    The registered program of the ``(H, V)`` geometry is selected
    (:func:`record_for`) and each chunk's GEMM instance variants follow the
    per-chunk rules of its architecture (:func:`chunk_variants`);
    ``gemm_tuning`` (default :func:`gemm_tuning_default`) pins knobs
    explicitly, e.g. ``{"logits": {"group_m": 16}}``.

    ``dx_finalize`` (default :func:`dx_finalize_default`): a K-sliced dX GEMM's
    slabs are added into ``dx_acc`` by the ``slab_sum`` kernel instead of the
    host's ``add_`` chain (bitwise the same accumulator).  ``dx_cast=False``
    leaves the flat ``dX`` cast out of ``backward()``, which then returns
    ``dx_acc`` for the caller to finalize (:func:`finalize_dx`: the fused dX
    finalize of a compacted :func:`backward_logprob`); ``dx_out`` is then
    neither bound nor allocated.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    device = X.device
    record = None
    module_name = None
    tuning = _resolve_gemm_tuning(gemm_tuning)
    if backend == "cake":
        if not X.is_cuda:
            raise ValueError("the cake backend needs CUDA tensors")
        module_name, record = record_for(
            device,
            int(X.shape[-1]) if X.ndim == 2 else None,
            int(W.shape[0]) if W.ndim == 2 else None,
        )
        record_abi(record)
    geometry = Geometry.from_record(record)
    problem = validate_lm_head_inputs(
        X,
        W,
        labels,
        objective=objective,
        loss_div=loss_div,
        infer_logp=infer_logp,
        loss_weights=loss_weights,
        chunk_size=chunk_size,
        grad_weight_dtype=grad_weight_dtype,
        deterministic=deterministic,
        entry=entry,
        geometry=geometry,
    )
    if problem.num_rows == 0:
        raise ValueError(
            "a prepared runner needs at least one row (T > 0); the eager entry points return zeros for T == 0"
        )
    given = [
        t
        for t in (
            W,
            labels,
            infer_logp,
            loss_weights,
            dlogp,
            workspace_buffer,
            dx_acc,
            dw_acc,
            dx_out,
            dw_out,
            logp,
            lse,
        )
        if t is not None
    ]
    if not all(t.device == device for t in given):
        raise ValueError("Expected all tensors on one device")
    stages = registered_stages(module_name) if record is not None else tuple(STAGES)
    num_sms = (
        int(torch.cuda.get_device_properties(device).multi_processor_count)
        if device.type == "cuda"
        else 1
    )
    row_index = None
    if isinstance(compact_rows, torch.Tensor):
        row_index = compact_rows
        if (
            row_index.dtype != torch.int64
            or row_index.ndim != 1
            or row_index.device != device
        ):
            raise ValueError(
                "compact_rows given as a tensor must be the int64 [T_v] valid-row index on X's device"
            )
    elif compact_rows:
        row_index = valid_row_index(labels)
    if row_index is not None and row_index.numel() == 0:
        raise ValueError(
            "a prepared runner needs at least one valid row (every label is ignored); the eager entry points return zeros for it"
        )
    dx_resident = (
        cluster_resident(device, geometry.dx_cluster_ctas, num_sms)
        if record is not None
        else None
    )
    plan = make_plan(
        problem,
        need_dx=need_dx,
        need_dw=need_dw,
        geometry=geometry,
        dx_max_slices=dx_max_slices(stages) if record is not None else 1,
        num_sms=num_sms,
        valid_rows=None if row_index is None else int(row_index.numel()),
        dx_resident=dx_resident,
        fuse_dw_cast=_resolve_fuse(fuse_dw_cast) if need_dw else False,
        arch=record["arch"] if record is not None else None,
        gemm_tuning=tuning,
        dx_finalize=bool(need_dx and _resolve_dx_finalize(dx_finalize)),
        dx_cast=bool(dx_cast),
    )
    missing = [s for s in plan.stages if s not in stages]
    if missing:
        raise NotImplementedError(
            f"the registered program {module_name!r} lacks the stages {missing} (registered: {stages})"
        )
    T, H, V, C = problem.num_rows, problem.hidden, problem.vocab, problem.chunk
    rows_loop = (
        plan.rows
    )  # rows the chunk loop processes (the valid rows when compacted)
    tma_bytes = _record_tma_bytes(record, stages) if record is not None else 0
    scratch_bytes = _record_scratch_bytes(record, stages) if record is not None else 0
    layout = workspace_layout(
        rows_loop,
        V,
        C,
        stats_tile=geometry.stats_tile,
        entry=entry,
        tma_workspace_bytes=tma_bytes,
        scratch_bytes=scratch_bytes,
        dx_ws_slabs=plan.dx_ws_slabs,
        hidden=H,
        compact=plan.compact,
    )
    if workspace_buffer is None:
        workspace_buffer = torch.empty(
            layout["total"], dtype=torch.uint8, device=device
        )
    flat = workspace_buffer.view(-1).view(torch.uint8)
    if flat.numel() < layout["total"]:
        raise ValueError(
            f"workspace_buffer needs {layout['total']} bytes, got {flat.numel()}"
        )

    def output(given, name, shape, dtype):
        _check_output(given, name, shape, dtype)
        return (
            given
            if given is not None
            else torch.empty(shape, dtype=dtype, device=device)
        )

    t: dict[str, Any] = dict(
        X=_prepare_x(X, problem, plan.compact),
        W=W,
        labels=_prepare_labels(labels, geometry, row_index),
    )
    rows = layout["logits"][1] // (V * 2)
    t["logits"] = _carve(flat, layout, "logits", torch.bfloat16, (rows, V))
    t["stats"] = _carve(flat, layout, "stats", torch.float32, (rows, plan.num_tiles, 2))
    t["d"] = _carve(flat, layout, "d", torch.float32, (rows,))
    t["term"] = _carve(flat, layout, "term", torch.float32, (rows,))
    t["loss_acc"] = _carve(flat, layout, "loss_acc", torch.float64, (1,))
    t["grad_scale"] = _carve(flat, layout, "grad_scale", torch.float32, (1,))
    t["unit_scale"] = torch.ones(
        (1,), dtype=torch.float32, device=device
    )  # the log-probability entry's cast scale
    t["f32_dummy"] = torch.zeros(
        (16,), dtype=torch.float32, device=device
    )  # unused ``WS`` of the unsliced GEMMs
    if plan.dx_ws_slabs:
        t["dx_ws"] = _carve(
            flat, layout, "dx_ws", torch.float32, (plan.dx_ws_slabs, rows, H)
        )
    if plan.compact:
        t["row_index"] = row_index
        t["x_c"] = _carve(
            flat, layout, "x_c", torch.bfloat16, (rows, H)
        )  # the chunk's valid X rows, gathered per chunk
    # O(T) row vectors and the loss cell are separate allocations: a caller (or the autograd graph) keeping
    # ``loss`` / ``logp`` / ``lse`` alive must not pin the vocabulary workspace.
    t["lse"] = output(lse, "lse", (rows_loop,), torch.float32)
    t["logp"] = output(logp, "logp", (rows_loop,), torch.float32)
    t["loss"] = torch.empty((1,), dtype=torch.float32, device=device)
    if entry == "loss":
        if infer_logp is not None:
            if (
                plan.compact
            ):  # gathered into their compact [T_v] forms at the start of every step
                t["infer_logp_full"], t["loss_weights_full"] = infer_logp, loss_weights
                t["infer_logp"] = torch.empty(
                    (rows_loop,), dtype=torch.float32, device=device
                )
                t["loss_weights"] = torch.empty(
                    (rows_loop,), dtype=torch.float32, device=device
                )
            else:
                t["infer_logp"], t["loss_weights"] = infer_logp, loss_weights
    else:
        d_in = output(dlogp, "dlogp", (T,), torch.float32)
        if plan.compact:
            t["d_in_full"] = d_in
            t["d_in"] = torch.empty((rows_loop,), dtype=torch.float32, device=device)
        else:
            t["d_in"] = d_in
    if plan.need_dx:
        t["dx_acc"] = output(dx_acc, "dx_acc", (rows_loop, H), torch.float32)
        if plan.dx_cast:
            t["dx_out"] = output(dx_out, "dx_out", (rows_loop, H), torch.bfloat16)
    if plan.need_dw:
        if plan.dw_acc_needed:
            t["dw_acc"] = output(dw_acc, "dw_acc", (V, H), torch.float32)
        t["dw_out"] = output(
            dw_out,
            "dw_out",
            (V, H),
            torch.float32
            if plan.dw_cast_stage in ("scale_cast_f32", "gemm_dw_cast_f32")
            else torch.bfloat16,
        )
    if layout.get("workspace"):
        t["workspace"] = _carve(
            flat, layout, "workspace", torch.uint8, (layout["workspace"][1],)
        )
    if layout.get("tma_descriptor_workspace"):
        t["tma_descriptor_workspace"] = _carve(
            flat,
            layout,
            "tma_descriptor_workspace",
            torch.uint8,
            (layout["tma_descriptor_workspace"][1],),
        )

    fwd = forward_keys(plan)
    bwd = (recompute_keys(plan) if entry == "logprob" else ()) + cast_keys(plan)
    values = {key: stage_values(key[0], t, plan, key[1]) for key in fwd + bwd}
    launches: dict[Any, _Launch] = {}
    engine = None
    if backend == "cake":
        launches = _bind_all(record, module_name, fwd + bwd, values, device, geometry)
    else:
        engine = ReferenceEngine()
    memory = memory_report(
        T,
        H,
        V,
        C,
        stats_tile=geometry.stats_tile,
        entry=entry,
        need_dx=plan.need_dx,
        need_dw=plan.need_dw,
        grad_weight_dtype=problem.grad_weight_dtype,
        return_logp=True,
        x_copy=problem.x_copy and not plan.compact,
        tma_workspace_bytes=tma_bytes,
        scratch_bytes=scratch_bytes,
        dx_ws_slabs=plan.dx_ws_slabs,
        valid_rows=plan.valid_rows,
        fuse_dw_cast=plan.fuse_dw_cast,
        dx_finalize=plan.dx_finalize,
    )
    if device.type == "cuda":
        device_index = int(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    else:
        device_index = 0
    return LmHeadLossRunner(
        backend=backend,
        module_name=module_name,
        plan=plan,
        tensors=t,
        values=values,
        launches=launches,
        forward_order=fwd,
        backward_order=bwd,
        device_index=device_index,
        workspace=flat,
        layout=layout,
        memory=memory,
        engine=engine,
    )


# ---------------------------------------------------------------------------
# Binding cache of the eager entry points (cake backend)
# ---------------------------------------------------------------------------

BINDING_CACHE_ENV = (
    "FLASHINFER_CAKE_LM_HEAD_LOSS_BINDING_CACHE"  # "0" disables the cache at import
)
BINDING_CACHE_CAPACITY_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_BINDING_CACHE_CAPACITY"
BINDING_CACHE_DEFAULT_CAPACITY = 64


def binding_cache_capacity() -> int:
    raw = os.environ.get(BINDING_CACHE_CAPACITY_ENV)
    if raw is None or not raw.strip():
        return BINDING_CACHE_DEFAULT_CAPACITY
    try:
        capacity = int(raw)
    except ValueError:
        capacity = 0
    if capacity < 1:
        raise ValueError(
            f"{BINDING_CACHE_CAPACITY_ENV} must be a positive integer, got {raw!r}"
        )
    return capacity


def _meta(t: Optional[torch.Tensor]):
    return (
        None
        if t is None
        else (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)
    )


def forward_binding_key(
    X,
    W,
    labels,
    *,
    objective,
    loss_div,
    infer_logp,
    loss_weights,
    chunk_size,
    need_dx,
    need_dw,
    grad_weight_dtype,
    entry,
    valid_rows=None,
    fuse_dw_cast=False,
    gemm_tuning=None,
    dx_finalize=False,
) -> tuple:
    """Cache key of a forward binding: ``(data_ptr, shape, stride, dtype)`` of every
    input plus every option that shapes the argument plans; ``valid_rows`` is the
    compacted row count (a label-dependent fact of the plan; ``None`` = uncompacted);
    ``fuse_dw_cast`` the resolved weight-gradient form; ``gemm_tuning`` the explicit
    GEMM knobs (``None`` = the environment's, :func:`gemm_tuning_default`);
    ``dx_finalize`` the resolved dX finalize form (the slab reduction's launches differ)."""
    return (
        "fwd",
        entry,
        _meta(X),
        _meta(W),
        _meta(labels),
        _meta(infer_logp),
        _meta(loss_weights),
        objective,
        None if loss_div is None else float(loss_div),
        int(chunk_size),
        bool(need_dx),
        bool(need_dw),
        grad_weight_dtype,
        None if valid_rows is None else int(valid_rows),
        bool(fuse_dw_cast),
        _tuning_key(_resolve_gemm_tuning(gemm_tuning)),
        bool(dx_finalize),
    )


def logprob_backward_binding_key(
    X,
    W,
    labels,
    lse,
    dlogp,
    *,
    chunk_size,
    need_dx,
    need_dw,
    valid_rows=None,
    fuse_dw_cast=False,
    gemm_tuning=None,
    dx_finalize=False,
) -> tuple:
    return (
        "bwd",
        "logprob",
        _meta(X),
        _meta(W),
        _meta(labels),
        _meta(lse),
        _meta(dlogp),
        int(chunk_size),
        bool(need_dx),
        bool(need_dw),
        None if valid_rows is None else int(valid_rows),
        bool(fuse_dw_cast),
        _tuning_key(_resolve_gemm_tuning(gemm_tuning)),
        bool(dx_finalize),
    )


# Values a remembered binding owns: the descriptor workspace of pointer-ABI stages and kernel-private scratch (a few
# kilobytes).  The vocabulary workspace, the row vectors, the accumulators and the outputs are allocated per call from
# the caching allocator (their sizes are part of the plan; the allocator reuses the blocks).
_OWNED_VALUES = ("workspace", "tma_descriptor_workspace")


@dataclass
class _Binding:
    """One remembered input binding: templated launches (no tensor pinned) plus the plan."""

    plan: Plan
    device: torch.device
    device_index: int
    launches: dict[Any, _Launch] = field(repr=False)
    forward_order: tuple = ()
    backward_order: tuple = ()
    owned: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    owned_bytes: int = 0
    layout: dict = field(default_factory=dict, repr=False)
    memory: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_runner(cls, runner: LmHeadLossRunner) -> "_Binding":
        if runner.backend != "cake":
            raise ValueError("only cake-backend runners are remembered")
        t = runner.tensors
        owned = {name: torch.empty_like(t[name]) for name in _OWNED_VALUES if name in t}
        return cls(
            plan=runner.plan,
            device=t["X"].device,
            device_index=runner.device_index,
            launches={
                key: launch.templated() for key, launch in runner.launches.items()
            },
            forward_order=runner.forward_order,
            backward_order=runner.backward_order,
            owned=owned,
            owned_bytes=sum(v.numel() * v.element_size() for v in owned.values()),
            layout=runner.layout,
            memory=runner.memory,
        )

    def holds_no_tensor(self) -> bool:
        return not any(
            isinstance(a, torch.Tensor)
            for launch in self.launches.values()
            for a in launch.arguments
        )

    def _scratch(self, t: dict[str, Any]) -> None:
        """Per-call temporaries from the caching allocator into ``t`` (the runner's regions, minus the owned ones)."""
        p = self.plan.problem
        rows = self.layout["logits"][1] // (p.vocab * 2)
        t["logits"] = torch.empty(
            (rows, p.vocab), dtype=torch.bfloat16, device=self.device
        )
        t["stats"] = torch.empty(
            (rows, self.plan.num_tiles, 2), dtype=torch.float32, device=self.device
        )
        t["d"] = torch.empty((rows,), dtype=torch.float32, device=self.device)
        t["term"] = torch.empty((rows,), dtype=torch.float32, device=self.device)
        t["loss_acc"] = torch.empty((1,), dtype=torch.float64, device=self.device)
        t["loss"] = torch.empty((1,), dtype=torch.float32, device=self.device)
        t.update(
            _device_constants(self.device_index)
        )  # constant cells: no per-call fill launches
        if self.plan.dx_ws_slabs:
            t["dx_ws"] = torch.empty(
                (self.plan.dx_ws_slabs, rows, p.hidden),
                dtype=torch.float32,
                device=self.device,
            )
        if self.plan.compact:
            t["x_c"] = torch.empty(
                (rows, p.hidden), dtype=torch.bfloat16, device=self.device
            )
        for name, tensor in self.owned.items():
            t[name] = tensor

    def _compact(self, t: dict[str, Any], row_index: Optional[torch.Tensor]) -> None:
        """Check the call's valid-row index against the remembered plan and bind it."""
        plan = self.plan
        if plan.compact != (row_index is not None) or (
            plan.compact and int(row_index.numel()) != plan.rows
        ):
            raise ValueError(
                "the remembered binding was planned for another set of valid rows"
            )
        if plan.compact:
            t["row_index"] = row_index

    def _launch(self, keys: tuple, t: dict[str, Any]) -> None:
        def run_key(key) -> None:
            host = HOST_STAGES.get(key[0])
            if host is not None:
                host(stage_values(key[0], t, self.plan, key[1]))
                return
            launch = self.launches[key]
            arguments = launch.arguments_for(stage_values(key[0], t, self.plan, key[1]))
            if (
                launch.prepare is not None
            ):  # descriptors of a pointer-ABI stage see the fresh tensors
                launch.prepare(*arguments)
            launch.entry(*arguments)

        _run_keys(self.plan, self.device_index, keys, run_key)

    def forward(self, X, W, labels, infer_logp, loss_weights, row_index=None):
        p, plan = self.plan.problem, self.plan
        T, H, V = (
            plan.rows,
            p.hidden,
            p.vocab,
        )  # rows of the chunk loop (the valid rows when compacted)
        t: dict[str, Any] = dict(
            X=_prepare_x(X, p, plan.compact),
            W=W,
            labels=_prepare_labels(labels, plan.geometry, row_index),
        )
        self._compact(t, row_index)
        self._scratch(t)
        t["lse"] = torch.empty((T,), dtype=torch.float32, device=self.device)
        t["logp"] = torch.empty((T,), dtype=torch.float32, device=self.device)
        if p.entry == "loss":
            if infer_logp is not None:
                if plan.compact:
                    t["infer_logp_full"], t["loss_weights_full"] = (
                        infer_logp,
                        loss_weights,
                    )
                    t["infer_logp"] = torch.empty(
                        (T,), dtype=torch.float32, device=self.device
                    )
                    t["loss_weights"] = torch.empty(
                        (T,), dtype=torch.float32, device=self.device
                    )
                else:
                    t["infer_logp"], t["loss_weights"] = infer_logp, loss_weights
            if plan.need_dx:
                t["dx_acc"] = torch.empty(
                    (T, H), dtype=torch.float32, device=self.device
                )
            if plan.dw_acc_needed:
                t["dw_acc"] = torch.empty(
                    (V, H), dtype=torch.float32, device=self.device
                )
        self._launch(self.forward_order, t)
        if p.entry == "loss":
            return (
                t["loss"],
                t["logp"],
                t.get("dx_acc"),
                t.get("dw_acc"),
                _deferred_operands(plan, t),
            )
        return t["logp"], t["lse"]

    def backward_logprob(self, X, W, labels, lse, dlogp, row_index=None):
        p, plan = self.plan.problem, self.plan
        T, H, V = plan.rows, p.hidden, p.vocab
        t: dict[str, Any] = dict(
            X=_prepare_x(X, p, plan.compact),
            W=W,
            labels=_prepare_labels(labels, plan.geometry, row_index),
            lse=lse,
        )
        self._compact(t, row_index)
        if plan.compact:
            t["d_in_full"] = dlogp
            t["d_in"] = torch.empty((T,), dtype=torch.float32, device=self.device)
        else:
            t["d_in"] = dlogp
        self._scratch(t)
        if plan.need_dx:
            t["dx_acc"] = torch.empty((T, H), dtype=torch.float32, device=self.device)
            if plan.dx_cast:
                t["dx_out"] = torch.empty(
                    (T, H), dtype=torch.bfloat16, device=self.device
                )
        if plan.need_dw:
            if plan.dw_acc_needed:
                t["dw_acc"] = torch.empty(
                    (V, H), dtype=torch.float32, device=self.device
                )
            t["dw_out"] = torch.empty((V, H), dtype=torch.bfloat16, device=self.device)
        self._launch(self.backward_order, t)
        dx = (
            t.get("dx_out") if plan.dx_cast else t.get("dx_acc")
        )  # dx_cast=False: the caller finalizes the accumulator
        return dx, t.get("dw_out")


def _deferred_operands(plan: Plan, t: dict[str, Any]) -> tuple:
    """``(dz_last, x_last, x_src, x_idx)`` of a deferred plan after its forward: the last chunk's BF16 ``dz`` rows of the
    chunk buffer and its ``X`` rows -- a view of the launched ``X`` (uncompacted) or the ``X`` plus the chunk's int64
    row index, gathered again in the backward (compacted) -- so no ``[rows, H]`` copy outlives the forward.  The ``dz``
    rows are a view too: the whole ``[min(T, C), V]`` chunk buffer stays alive until the backward (what
    :func:`memory_report` reports as ``saved_dz_bytes``) rather than paying a tail copy per step; the eager entry points
    launch through the templated binding from the first call on, so that buffer is a per-call allocation of its own and
    the view pins nothing else (a prepared runner keeps the chunk's ``dz`` inside its workspace).  All ``None`` for an
    undeferred plan."""
    if not plan.dw_deferred:
        return None, None, None, None
    row0, rows_c = plan.last_chunk
    dz_last = t["logits"][:rows_c]
    if plan.compact:
        return dz_last, None, t["X"], t["row_index"][row0 : row0 + rows_c]
    return dz_last, t["X"][row0 : row0 + rows_c], None, None


class BindingCache:
    """Remembered input bindings of the eager entry points (least recently used, bounded)."""

    def __init__(self, capacity: Optional[int] = None, enabled: bool = True):
        self.capacity = binding_cache_capacity() if capacity is None else int(capacity)
        if self.capacity < 1:
            raise ValueError(
                "BindingCache needs a capacity of at least one binding (use enabled=False to bypass it)"
            )
        self.enabled = bool(enabled)
        self.hits = 0
        self.misses = 0
        self._bindings: OrderedDict[tuple, _Binding] = OrderedDict()

    def __len__(self) -> int:
        return len(self._bindings)

    @property
    def owned_bytes(self) -> int:
        return sum(b.owned_bytes for b in self._bindings.values())

    def clear(self) -> None:
        self._bindings.clear()

    def lookup(self, key: tuple) -> Optional[_Binding]:
        binding = self._bindings.get(key)
        if binding is None:
            self.misses += 1
        else:
            self.hits += 1
            self._bindings.move_to_end(key)
        return binding

    def peek(self, key: tuple) -> Optional[_Binding]:
        return self._bindings.get(key)

    def remember(self, key: tuple, binding: _Binding) -> _Binding:
        self._bindings.pop(key, None)
        self._bindings[key] = binding
        while len(self._bindings) > self.capacity:
            self._bindings.popitem(last=False)
        return binding


BINDING_CACHE = BindingCache(enabled=os.environ.get(BINDING_CACHE_ENV, "1") != "0")


# ---------------------------------------------------------------------------
# Eager entry points (allocate, launch, return)
# ---------------------------------------------------------------------------


@dataclass
class ForwardResult:
    loss: Optional[torch.Tensor]  # FP32 [] (loss entry)
    logp: (
        torch.Tensor
    )  # FP32 [T] (0 on ignored rows; scattered back when the rows were compacted)
    lse: Optional[torch.Tensor] = (
        None  # FP32 (log-probability entry: the saved statistic; compact [T_v] rows of ``row_index`` when compacted)
    )
    dx_acc: Optional[torch.Tensor] = (
        None  # FP32 [T, H] (loss entry, X trainable); compact [T_v, H] rows of ``row_index`` when compacted
    )
    dw_acc: Optional[torch.Tensor] = None  # FP32 [V, H] (loss entry, W trainable)
    memory: Optional[dict] = None
    backend: str = "cake"
    row_index: Optional[torch.Tensor] = (
        None  # int64 [T_v] valid-row index when the chunk loop was compacted (None otherwise)
    )
    # the bool [T] valid-row mask when the compacted forward ran with the fused dX finalize (None otherwise): the
    # backward follows the forward's choice through it and forms the inclusive scan the scatter kernel reads
    row_valid: Optional[torch.Tensor] = None
    num_rows: int = 0  # T, the caller's row count
    # Deferred weight gradient (``fuse_dw_cast``, loss entry): the last chunk's BF16 ``dz`` rows (a view of the chunk
    # buffer) and its ``X`` rows -- ``x_last`` (a view of ``X``) on the uncompacted path, ``x_src`` / ``x_idx`` (``X`` and
    # the chunk's int64 row index, gathered in the backward) when compacted.  ``backward_loss`` finishes ``dW`` from them;
    # ``dw_acc`` then holds the chunks before the last one and is ``None`` for a one-chunk call.
    dz_last: Optional[torch.Tensor] = None
    x_last: Optional[torch.Tensor] = None
    x_src: Optional[torch.Tensor] = None
    x_idx: Optional[torch.Tensor] = None

    def backward(self, grad=None, *, grad_weight_dtype=torch.bfloat16, backend=None):
        """``(dX, dW)`` of this forward through :func:`backward_loss` (``grad`` = the upstream scalar, ``None`` = 1)."""
        if self.loss is None:
            raise ValueError(
                "ForwardResult.backward belongs to the loss entry (use backward_logprob)"
            )
        return backward_loss(
            self.dx_acc,
            self.dw_acc,
            grad,
            grad_weight_dtype=grad_weight_dtype,
            backend=self.backend if backend is None else backend,
            row_index=self.row_index,
            num_rows=self.num_rows,
            dz_last=self.dz_last,
            x_last=self.x_last,
            x_src=self.x_src,
            x_idx=self.x_idx,
            row_valid=self.row_valid,
        )


def _empty_forward(
    problem: Problem,
    X: torch.Tensor,
    *,
    need_dx: bool,
    need_dw: bool,
    row_index: Optional[torch.Tensor] = None,
    row_valid: Optional[torch.Tensor] = None,
) -> ForwardResult:
    """Zeros without binding or launching: ``T == 0``, or every row ignored (``row_index`` empty)."""
    device = X.device
    T = int(problem.num_rows)
    memory = memory_report(
        T,
        problem.hidden,
        problem.vocab,
        problem.chunk,
        entry=problem.entry,
        need_dx=need_dx,
        need_dw=need_dw,
        grad_weight_dtype=problem.grad_weight_dtype,
        return_logp=True,
        valid_rows=None if row_index is None else 0,
    )
    zeros = torch.zeros((T,), dtype=torch.float32, device=device)
    if problem.entry == "loss":
        return ForwardResult(
            loss=torch.zeros((), dtype=torch.float32, device=device),
            logp=zeros,
            dx_acc=torch.zeros((0, problem.hidden), dtype=torch.float32, device=device)
            if need_dx
            else None,
            dw_acc=torch.zeros(
                (problem.vocab, problem.hidden), dtype=torch.float32, device=device
            )
            if need_dw
            else None,
            memory=memory,
            row_index=row_index,
            row_valid=row_valid,
            num_rows=T,
        )
    return ForwardResult(
        loss=None,
        logp=zeros,
        lse=zeros.clone(),
        memory=memory,
        row_index=row_index,
        num_rows=T,
    )


def _resolve_compact(compact_rows) -> bool:
    return compact_rows_default() if compact_rows is None else bool(compact_rows)


def forward_loss(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    objective: str = "ce",
    loss_div=None,
    infer_logp: Optional[torch.Tensor] = None,
    loss_weights: Optional[torch.Tensor] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    need_dx: bool = True,
    need_dw: bool = True,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    deterministic: bool = True,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    fuse_dw_cast: Optional[bool] = None,
) -> ForwardResult:
    """Forward of the loss entry: ``loss`` (FP32 scalar), ``logp`` (FP32 ``[T]``) and the
    FP32 gradient accumulators ``dx_acc`` / ``dw_acc`` of the trainable inputs.

    The first call for an input binding validates and binds through
    :func:`prepare_lm_head_loss` and launches through the templated binding it
    remembers (:data:`BINDING_CACHE`); later calls with the same binding take
    the remembered launches directly.  A call without rows returns
    loss 0, an empty ``logp`` and zero accumulators without binding or launching.
    ``compact_rows`` (default :func:`compact_rows_default`) chunks over the valid
    rows only: ``logp`` comes back scattered to ``[T]``, ``dx_acc`` is the compact
    ``[T_v, H]`` (``row_index`` / ``num_rows`` of the result restore ``[T, H]``
    through :func:`backward_loss`); every row ignored returns zeros without
    binding or launching.  ``fuse_dw_cast`` (default :func:`fuse_dw_cast_default`):
    the last chunk's weight-gradient GEMM is deferred to :func:`backward_loss`
    (``dw_acc`` holds the chunks before it, ``None`` for a one-chunk call; the
    result carries that chunk's operands ``dz_last`` / ``x_last`` / ``x_src`` /
    ``x_idx``).  The fused dX finalize (:func:`dx_finalize_default`, read once
    here) adds a sliced dX GEMM's slabs with the ``slab_sum`` kernel and, on a
    compacted call, keeps the valid-row mask (``row_valid``) for
    :func:`backward_loss`'s one-pass scatter of ``dX``.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    problem = validate_lm_head_inputs(
        X,
        W,
        labels,
        objective=objective,
        loss_div=loss_div,
        infer_logp=infer_logp,
        loss_weights=loss_weights,
        chunk_size=chunk_size,
        grad_weight_dtype=grad_weight_dtype,
        deterministic=deterministic,
        entry="loss",
    )
    T = problem.num_rows
    if T == 0:
        return _empty_forward(problem, X, need_dx=need_dx, need_dw=need_dw)
    fused_dx = bool(
        need_dx and dx_finalize_default()
    )  # read once per call; the backward follows it through row_valid
    row_index, row_valid = (
        valid_rows(labels, mask=fused_dx)
        if _resolve_compact(compact_rows)
        else (None, None)
    )
    if row_index is not None and row_index.numel() == 0:  # every row ignored
        return _empty_forward(
            problem,
            X,
            need_dx=need_dx,
            need_dw=need_dw,
            row_index=row_index,
            row_valid=row_valid,
        )
    valid_count = None if row_index is None else int(row_index.numel())
    fuse = bool(need_dw and _resolve_fuse(fuse_dw_cast))
    common = dict(
        objective=objective,
        loss_div=problem.loss_div,
        infer_logp=infer_logp,
        loss_weights=loss_weights,
        chunk_size=chunk_size,
        need_dx=need_dx,
        need_dw=need_dw,
        grad_weight_dtype=grad_weight_dtype,
    )
    cache = BINDING_CACHE
    key = None
    if backend == "cake" and cache.enabled:
        key = forward_binding_key(
            X,
            W,
            labels,
            entry="loss",
            valid_rows=valid_count,
            fuse_dw_cast=fuse,
            dx_finalize=fused_dx,
            **common,
        )
        binding = cache.lookup(key)
        if binding is not None:
            loss, logp, dx_acc, dw_acc, deferred = binding.forward(
                X, W, labels, infer_logp, loss_weights, row_index
            )
            dz_last, x_last, x_src, x_idx = deferred
            return ForwardResult(
                loss=loss.reshape(()),
                logp=scatter_rows(logp, row_index, T),
                dx_acc=dx_acc,
                dw_acc=dw_acc,
                memory=binding.memory,
                backend=backend,
                row_index=row_index,
                row_valid=row_valid,
                num_rows=T,
                dz_last=dz_last,
                x_last=x_last,
                x_src=x_src,
                x_idx=x_idx,
            )
    runner = prepare_lm_head_loss(
        X,
        W,
        labels,
        deterministic=deterministic,
        entry="loss",
        backend=backend,
        compact_rows=False if row_index is None else row_index,
        fuse_dw_cast=fuse,
        dx_finalize=fused_dx,
        **common,
    )
    if backend == "cake":
        # The first call launches through the templated binding too: its per-call temporaries come from the caching
        # allocator, so the deferred ``dz`` rows keep the chunk buffer alive and nothing else -- a view into the
        # runner's single workspace allocation would pin the whole workspace until the backward.  The runner (and its
        # workspace) is dropped here; only the binding survives.
        binding = _Binding.from_runner(runner)
        del runner
        if key is not None:
            cache.remember(key, binding)
        loss, logp, dx_acc, dw_acc, deferred = binding.forward(
            X, W, labels, infer_logp, loss_weights, row_index
        )
        memory = binding.memory
    else:
        loss, logp = runner.forward()
        dx_acc, dw_acc = runner.dx_acc, runner.dw_acc
        deferred = _deferred_operands(runner.plan, runner.tensors)
        memory = runner.memory
    dz_last, x_last, x_src, x_idx = deferred
    return ForwardResult(
        loss=loss.reshape(()),
        logp=scatter_rows(logp, row_index, T),
        dx_acc=dx_acc,
        dw_acc=dw_acc,
        memory=memory,
        backend=backend,
        row_index=row_index,
        row_valid=row_valid,
        num_rows=T,
        dz_last=dz_last,
        x_last=x_last,
        x_src=x_src,
        x_idx=x_idx,
    )


def scale_cast(
    acc: torch.Tensor,
    grad: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    *,
    backend: str = "cake",
) -> torch.Tensor:
    """``cast(g * acc)`` into a new tensor of ``out_dtype`` (one rounding); ``acc`` is never written.

    The single fp32 -> output cast of the backward, for ``dW`` and ``dX`` alike
    (``g`` = the incoming scalar gradient, ``None`` = 1).
    """
    if out_dtype not in GRAD_WEIGHT_DTYPES:
        raise ValueError("the cast produces bfloat16 or float32")
    if not acc.is_contiguous() or acc.dtype != torch.float32:
        raise ValueError("the accumulator must be a contiguous FP32 tensor")
    out = torch.empty(acc.shape, dtype=out_dtype, device=acc.device)
    if acc.numel() == 0:
        return out
    g = (
        _unit_scale(acc.device)
        if grad is None
        else grad.detach().reshape(1).to(device=acc.device, dtype=torch.float32)
    )
    if backend == "reference":
        out.reshape(-1).copy_(acc.reshape(-1) * g)
        return out
    module_name, record = record_for(acc.device)
    record_abi(record)
    geometry = Geometry.from_record(record)
    stage = "scale_cast_f32" if out_dtype == torch.float32 else "scale_cast_bf16"
    if acc.numel() % geometry.cast_vec:
        raise ValueError(f"the cast needs a multiple of {geometry.cast_vec} elements")
    values: dict[str, Any] = {name: None for name in COMMON_TENSORS}
    values.update({name: 0 for name in COMMON_SCALARS})
    values.update(
        acc=acc.reshape(-1),
        g=g,
        out=out.reshape(-1),
        num_vecs=int(acc.numel() // geometry.cast_vec),
        loss_div=1.0,
    )
    launches = _bind_all(
        record,
        module_name,
        ((stage, "eager"),),
        {(stage, "eager"): values},
        acc.device,
        geometry,
    )
    launch = launches[(stage, "eager")]
    index = (
        acc.device.index
        if acc.device.index is not None
        else torch.cuda.current_device()
    )
    with _ffi_stream_context(int(index)):
        if launch.prepare is not None:
            launch.prepare(*launch.arguments)
        launch()
    return out


def scale_cast_scatter(
    acc: torch.Tensor,
    grad: Optional[torch.Tensor],
    row_index: torch.Tensor,
    scan: torch.Tensor,
    num_rows: int,
    *,
    backend: str = "cake",
) -> torch.Tensor:
    """``scatter_rows(bf16(g * acc), row_index, num_rows)`` as ONE new BF16 ``[num_rows, H]`` tensor written in one
    pass: ``bf16(g * acc[compact row])`` on the valid rows, exact zeros elsewhere.  ``acc`` is the compact FP32
    ``[T_v, H]`` accumulator (never written), ``row_index`` its ascending int64 original rows, ``scan`` the inclusive
    int32 count of valid rows (``cumsum(labels >= 0)``: ``scan[r] - 1`` is the compact row of a valid output row
    ``r``, which the kernel validates through ``row_index[scan[r] - 1] == r``), ``g`` = ``grad`` (``None`` = 1).  The
    same FP32 multiply and BF16 rounding as :func:`scale_cast`, so the valid rows are bitwise the flat cast's; every
    output element is written exactly once.  ``T_v == 0`` (every row ignored) is one zero fill: the kernel reads a
    candidate compact row for every output row and would have none."""
    if acc.dtype != torch.float32 or acc.ndim != 2 or not acc.is_contiguous():
        raise ValueError("the accumulator must be a contiguous FP32 [T_v, H] tensor")
    T_v, H = (int(s) for s in acc.shape)
    num_rows = int(num_rows)
    if (
        row_index.dtype != torch.int64
        or tuple(row_index.shape) != (T_v,)
        or not row_index.is_contiguous()
        or row_index.device != acc.device
    ):
        raise ValueError(
            f"row_index must be a contiguous int64 [{T_v}] tensor on the accumulator's device"
        )
    if (
        scan.dtype != torch.int32
        or tuple(scan.shape) != (num_rows,)
        or not scan.is_contiguous()
        or scan.device != acc.device
    ):
        raise ValueError(
            f"scan must be a contiguous int32 [{num_rows}] tensor on the accumulator's device"
        )
    if T_v > num_rows:
        raise ValueError(
            f"the accumulator has more rows ({T_v}) than the output ({num_rows})"
        )
    out = torch.empty((num_rows, H), dtype=torch.bfloat16, device=acc.device)
    if out.numel() == 0:
        return out
    if T_v == 0:
        out.zero_()
        return out
    g = (
        _unit_scale(acc.device)
        if grad is None
        else grad.detach().reshape(1).to(device=acc.device, dtype=torch.float32)
    )
    if backend == "reference":
        out.zero_()
        out.index_copy_(0, row_index, (acc * g).to(torch.bfloat16))
        return out
    module_name, record = record_for(acc.device)
    record_abi(record)
    geometry = Geometry.from_record(record)
    stage = "scale_cast_scatter_bf16"
    if stage not in record:
        raise NotImplementedError(
            f"the registered program {module_name!r} lacks the stage {stage!r}"
        )
    if H % geometry.cast_vec:
        raise ValueError(
            f"the scatter needs a row width of a multiple of {geometry.cast_vec} elements"
        )
    values: dict[str, Any] = {name: None for name in COMMON_TENSORS}
    values.update({name: 0 for name in COMMON_SCALARS})
    values.update(
        acc=acc,
        g=g,
        scan=scan,
        idx_lo=row_index.view(
            torch.int32
        ),  # the low int32 word of each int64 index (stride 2)
        out=out,
        row_vecs=int(H // geometry.cast_vec),
        num_rows=num_rows,
        loss_div=1.0,
    )
    launches = _bind_all(
        record,
        module_name,
        ((stage, "eager"),),
        {(stage, "eager"): values},
        acc.device,
        geometry,
    )
    launch = launches[(stage, "eager")]
    index = (
        acc.device.index
        if acc.device.index is not None
        else torch.cuda.current_device()
    )
    with _ffi_stream_context(int(index)):
        if launch.prepare is not None:
            launch.prepare(*launch.arguments)
        launch()
    return out


def finalize_dx(
    acc: torch.Tensor,
    grad: Optional[torch.Tensor],
    row_index: Optional[torch.Tensor],
    row_valid: Optional[torch.Tensor],
    num_rows: Optional[int],
    *,
    backend: str = "cake",
) -> torch.Tensor:
    """The dX output boundary: ``bf16(g * acc)`` restored to the caller's ``[num_rows, H]`` rows.  Every row valid
    (``row_index`` None): :func:`scale_cast`, already one pass over every row.  Compacted rows with the forward's
    valid-row mask (``row_valid``: the fused dX finalize): the inclusive int32 scan -- one ``cumsum``, issued here
    behind the backward's queued GEMMs -- and one :func:`scale_cast_scatter` launch.  Compacted rows without the mask
    (the forward ran with ``FLASHINFER_CAKE_LM_HEAD_LOSS_DX_FINALIZE=0``): the flat cast, then :func:`scatter_rows`
    (zero fill + ``index_copy_``).  All three produce the same bits."""
    if row_index is None:
        return scale_cast(acc, grad, torch.bfloat16, backend=backend)
    if num_rows is None:
        raise ValueError("finalize_dx needs num_rows (the caller's T) with row_index")
    if row_valid is None:
        return scatter_rows(
            scale_cast(acc, grad, torch.bfloat16, backend=backend),
            row_index,
            int(num_rows),
        )
    if row_valid.dtype != torch.bool or tuple(row_valid.shape) != (int(num_rows),):
        raise ValueError(f"row_valid must be the bool [{int(num_rows)}] valid-row mask")
    scan = torch.cumsum(row_valid, 0, dtype=torch.int32)
    return scale_cast_scatter(
        acc, grad, row_index, scan, int(num_rows), backend=backend
    )


def dw_cast(
    dz_last: torch.Tensor,
    x_rows: torch.Tensor,
    dw_acc: Optional[torch.Tensor],
    grad: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    *,
    backend: str = "cake",
) -> torch.Tensor:
    """The deferred last chunk's weight gradient as a new tensor: ``cast(g * (dw_acc + dz_last^T @ x_rows))``, or
    ``cast(g * dz_last^T @ x_rows)`` when ``dw_acc`` is ``None`` (a one-chunk call), through the fused GEMM
    (:data:`DW_CAST_STAGES`); ``dw_acc`` is read, never written (repeatable).  ``dz_last`` BF16 ``[rows, V]``
    contiguous, ``x_rows`` BF16 ``[rows, H]`` with a 16-byte row pitch (else copied), ``g`` = ``grad`` (``None`` = 1)."""
    if out_dtype not in GRAD_WEIGHT_DTYPES:
        raise ValueError("the cast produces bfloat16 or float32")
    if (
        dz_last.dtype != torch.bfloat16
        or dz_last.ndim != 2
        or not dz_last.is_contiguous()
    ):
        raise ValueError("dz_last must be a contiguous BF16 [rows, V] tensor")
    rows, V = dz_last.shape
    if (
        x_rows.dtype != torch.bfloat16
        or x_rows.ndim != 2
        or x_rows.shape[0] != rows
        or x_rows.stride(1) != 1
    ):
        raise ValueError("x_rows must be a BF16 [rows, H] tensor with contiguous rows")
    if x_rows.stride(0) % 8 or x_rows.data_ptr() % 16:
        x_rows = x_rows.contiguous()
    H = int(x_rows.shape[1])
    if dw_acc is not None and (
        dw_acc.dtype != torch.float32
        or tuple(dw_acc.shape) != (V, H)
        or not dw_acc.is_contiguous()
    ):
        raise ValueError(
            "dw_acc must be a contiguous FP32 [V, H] tensor (or None for a one-chunk call)"
        )
    device = dz_last.device
    out = torch.empty((V, H), dtype=out_dtype, device=device)
    if rows == 0:
        return out.zero_()
    g = (
        _unit_scale(device)
        if grad is None
        else grad.detach().reshape(1).to(device=device, dtype=torch.float32)
    )
    if backend == "reference":
        product = _mm_fp32(dz_last.t(), x_rows)
        acc = product if dw_acc is None else dw_acc + product
        out.copy_(acc * g)
        return out
    module_name, record = record_for(device, H, int(V))
    record_abi(record)
    # the last chunk's variant: the raster rule of the record's architecture on this chunk's rows, or the explicit knob
    dw_tuning = _resolve_gemm_tuning(None).get("dw", {})
    raster = raster_variant(H, int(rows), record["arch"])
    group_m = (
        dw_tuning["group_m"]
        if "group_m" in dw_tuning
        else (raster[1] if raster else None)
    )
    stage = stage_variant(
        "gemm_dw_cast_f32" if out_dtype == torch.float32 else "gemm_dw_cast_bf16",
        group_m=group_m,
    )
    if stage not in registered_stages(module_name):
        raise NotImplementedError(
            f"the registered program {module_name!r} lacks the stage {stage!r} (fuse_dw_cast)"
        )
    geometry = Geometry.from_record(record)
    index = device.index if device.index is not None else torch.cuda.current_device()
    values: dict[str, Any] = {name: None for name in COMMON_TENSORS}
    values.update({name: 0 for name in COMMON_SCALARS})
    values.update(
        A=dz_last,
        B=x_rows,
        C=out,
        STATS_OUT=g,
        WS=dw_acc if dw_acc is not None else _device_constants(int(index))["f32_dummy"],
        M=int(V),
        m_tiles=geometry.row_tiles(V, geometry.dw_cluster_ctas),
        k_iters=geometry.k_iters(rows),
        first_chunk=int(dw_acc is None),
        rows_c=int(rows),
        T=int(rows),
        H=H,
        V=int(V),
        loss_div=1.0,
    )
    launches = _bind_all(
        record,
        module_name,
        ((stage, "eager"),),
        {(stage, "eager"): values},
        device,
        geometry,
    )
    launch = launches[(stage, "eager")]
    with _ffi_stream_context(int(index)):
        if launch.prepare is not None:
            launch.prepare(*launch.arguments)
        launch()
    return out


def backward_loss(
    dx_acc: Optional[torch.Tensor],
    dw_acc: Optional[torch.Tensor],
    grad: Optional[torch.Tensor] = None,
    *,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    backend: str = "cake",
    row_index: Optional[torch.Tensor] = None,
    num_rows: Optional[int] = None,
    dz_last: Optional[torch.Tensor] = None,
    x_last: Optional[torch.Tensor] = None,
    x_src: Optional[torch.Tensor] = None,
    x_idx: Optional[torch.Tensor] = None,
    row_valid: Optional[torch.Tensor] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Backward of the loss entry from the saved accumulators: ``(dX, dW)`` = ``bf16(g * dX_acc)``,
    ``cast(g * dW_acc)``; the accumulators are read, never written (repeatable).  A compacted forward
    (``ForwardResult.row_index`` / ``num_rows``): the compact rows are cast once, then scattered into a
    zero BF16 ``[T, H]``.  A deferred weight gradient (``fuse_dw_cast``; ``ForwardResult.dz_last`` with
    ``x_last`` or ``x_src`` / ``x_idx``): ``dW = cast(g * (dW_acc + dz_last^T @ X_last))`` through the fused
    GEMM (:func:`dw_cast`), ``dW_acc`` being the chunks before the last one or ``None``;
    :meth:`ForwardResult.backward` passes everything.  ``row_valid`` (``ForwardResult.row_valid``, the
    valid-row mask of a compacted forward that ran with the fused dX finalize) selects the one-pass
    scale-cast-scatter of the compact rows (:func:`finalize_dx`); without it the compact rows are cast,
    then scattered into a zero BF16 ``[T, H]``.  Both give the same bits."""
    dx = None
    if dx_acc is not None:
        if row_index is not None and num_rows is None:
            raise ValueError(
                "backward_loss needs num_rows (the caller's T) with row_index"
            )
        dx = finalize_dx(dx_acc, grad, row_index, row_valid, num_rows, backend=backend)
    if dz_last is not None:
        if x_last is None:
            if x_src is None or x_idx is None:
                raise ValueError(
                    "a deferred weight gradient needs x_last, or x_src with the chunk's row index x_idx"
                )
            x_last = torch.index_select(
                x_src, 0, x_idx
            )  # the compacted chunk's rows, gathered again
        dw = dw_cast(dz_last, x_last, dw_acc, grad, grad_weight_dtype, backend=backend)
    else:
        dw = (
            None
            if dw_acc is None
            else scale_cast(dw_acc, grad, grad_weight_dtype, backend=backend)
        )
    return dx, dw


def forward_logprob(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    deterministic: bool = True,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
) -> ForwardResult:
    """Forward of the log-probability entry: ``logp`` (FP32 ``[T]``, 0 on ignored rows) plus the saved statistic
    ``lse`` (compacted: the ``[T_v]`` rows of ``row_index``, exactly what :func:`backward_logprob` expects back)."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    problem = validate_lm_head_inputs(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        deterministic=deterministic,
        entry="logprob",
    )
    T = problem.num_rows
    if T == 0:
        return _empty_forward(problem, X, need_dx=False, need_dw=False)
    row_index = valid_row_index(labels) if _resolve_compact(compact_rows) else None
    if row_index is not None and row_index.numel() == 0:  # every row ignored
        return _empty_forward(
            problem, X, need_dx=False, need_dw=False, row_index=row_index
        )
    valid_rows = None if row_index is None else int(row_index.numel())
    cache = BINDING_CACHE
    key = None
    if backend == "cake" and cache.enabled:
        key = forward_binding_key(
            X,
            W,
            labels,
            objective="ce",
            loss_div=None,
            infer_logp=None,
            loss_weights=None,
            chunk_size=chunk_size,
            need_dx=False,
            need_dw=False,
            grad_weight_dtype=torch.bfloat16,
            entry="logprob",
            valid_rows=valid_rows,
        )
        binding = cache.lookup(key)
        if binding is not None:
            logp, lse = binding.forward(X, W, labels, None, None, row_index)
            return ForwardResult(
                loss=None,
                logp=scatter_rows(logp, row_index, T),
                lse=lse,
                memory=binding.memory,
                backend=backend,
                row_index=row_index,
                num_rows=T,
            )
    runner = prepare_lm_head_loss(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        need_dx=False,
        need_dw=False,
        deterministic=deterministic,
        entry="logprob",
        backend=backend,
        compact_rows=False if row_index is None else row_index,
    )
    logp, lse = runner.forward()
    if key is not None:
        cache.remember(key, _Binding.from_runner(runner))
    return ForwardResult(
        loss=None,
        logp=scatter_rows(logp, row_index, T),
        lse=lse,
        memory=runner.memory,
        backend=backend,
        row_index=row_index,
        num_rows=T,
    )


def backward_logprob(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    lse: torch.Tensor,
    dlogp: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    need_dx: bool = True,
    need_dw: bool = True,
    deterministic: bool = True,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    fuse_dw_cast: Optional[bool] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Backward of the log-probability entry: recompute each chunk's logits, ``dz = dlogp_t *
    (1[v = y_t] - softmax)`` (ignored rows zero), then ``dX`` (BF16) and ``dW`` (BF16) through
    the FP32 accumulators.  ``(None, None)`` when neither input is trainable.  ``compact_rows``
    must match the forward's: the valid-row index is recomputed from ``labels`` (identical to the
    forward's), ``lse`` is then the forward's compact ``[T_v]`` statistic and ``dX`` comes back
    scattered to ``[T, H]``; every row ignored returns zeros without launching.  ``fuse_dw_cast``
    (default :func:`fuse_dw_cast_default`): the last chunk's weight-gradient GEMM writes the final BF16
    ``dW`` from its epilogue (the scale is 1 here)."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    problem = validate_lm_head_inputs(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        deterministic=deterministic,
        entry="logprob",
    )
    if not (need_dx or need_dw):
        return None, None
    T, H, V = problem.num_rows, problem.hidden, problem.vocab
    fused_dx = bool(need_dx and dx_finalize_default())  # read once per call
    row_index, row_valid = (
        valid_rows(labels, mask=fused_dx)
        if (T and _resolve_compact(compact_rows))
        else (None, None)
    )
    rows = T if row_index is None else int(row_index.numel())
    if rows == 0:  # no rows, or every row ignored
        return (
            torch.zeros((T, H), dtype=torch.bfloat16, device=X.device)
            if need_dx
            else None,
            torch.zeros((V, H), dtype=torch.bfloat16, device=X.device)
            if need_dw
            else None,
        )
    _check_output(lse, "lse", (rows,), torch.float32)
    dlogp = dlogp.detach().reshape(T).to(torch.float32).contiguous()
    cache = BINDING_CACHE
    key = None
    valid_count = None if row_index is None else rows
    fuse = bool(need_dw and _resolve_fuse(fuse_dw_cast))
    # the fused dX finalize of a compacted backward: the runner leaves dx_acc uncast and finalize_dx writes the [T, H]
    # dX in one pass (uncompacted: the in-loop flat cast is already one pass over every row)
    dx_cast = not (fused_dx and row_index is not None)

    def finish_dx(dx):
        if dx is None:
            return None
        if dx_cast:
            return scatter_rows(dx, row_index, T)
        return finalize_dx(dx, None, row_index, row_valid, T, backend=backend)

    if backend == "cake" and cache.enabled:
        key = logprob_backward_binding_key(
            X,
            W,
            labels,
            lse,
            dlogp,
            chunk_size=chunk_size,
            need_dx=need_dx,
            need_dw=need_dw,
            valid_rows=valid_count,
            fuse_dw_cast=fuse,
            dx_finalize=fused_dx,
        )
        binding = cache.lookup(key)
        if binding is not None:
            dx, dw = binding.backward_logprob(X, W, labels, lse, dlogp, row_index)
            return finish_dx(dx), dw
    runner = prepare_lm_head_loss(
        X,
        W,
        labels,
        chunk_size=chunk_size,
        need_dx=need_dx,
        need_dw=need_dw,
        deterministic=deterministic,
        entry="logprob",
        dlogp=dlogp,
        lse=lse,
        backend=backend,
        compact_rows=False if row_index is None else row_index,
        fuse_dw_cast=fuse,
        dx_finalize=fused_dx,
        dx_cast=dx_cast,
    )
    dx, dw = runner.backward()
    if key is not None:
        cache.remember(key, _Binding.from_runner(runner))
    return finish_dx(dx), dw


# ---------------------------------------------------------------------------
# Autograd Functions and the backend-level entry points
# ---------------------------------------------------------------------------


class ChunkedLmHeadLossFunction(torch.autograd.Function):
    """Entry (a): the forward produces the FP32 gradient accumulators; the backward scales and casts them."""

    @staticmethod
    def forward(
        ctx,
        X,
        W,
        labels,
        objective,
        loss_div,
        infer_logp,
        loss_weights,
        chunk_size,
        return_logp,
        grad_weight_dtype,
        deterministic,
        backend,
        compact_rows,
        fuse_dw_cast,
    ):
        need_dx, need_dw = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
        result = forward_loss(
            X,
            W,
            labels,
            objective=objective,
            loss_div=loss_div,
            infer_logp=infer_logp,
            loss_weights=loss_weights,
            chunk_size=chunk_size,
            need_dx=need_dx,
            need_dw=need_dw,
            grad_weight_dtype=grad_weight_dtype,
            deterministic=deterministic,
            backend=backend,
            compact_rows=compact_rows,
            fuse_dw_cast=fuse_dw_cast,
        )
        ctx.set_materialize_grads(False)
        ctx.dx_acc, ctx.dw_acc = (
            result.dx_acc,
            result.dw_acc,
        )  # saved state, never mutated
        # the deferred weight gradient's operands: the last chunk's dz rows and its X rows, the latter a view of the
        # input X (or X itself with the chunk's row index when compacted) -- saved through the autograd context so an
        # in-place write to X between the forward and the backward raises PyTorch's saved-tensor version error
        # instead of yielding a stale dW
        ctx.save_for_backward(result.dz_last, result.x_last, result.x_src, result.x_idx)
        ctx.row_index, ctx.num_rows = (
            result.row_index,
            int(X.shape[0]),
        )  # compacted: dx_acc holds the rows of row_index
        ctx.row_valid = (
            result.row_valid
        )  # the fused dX finalize's valid-row mask (None otherwise)
        ctx.grad_weight_dtype = grad_weight_dtype
        ctx.backend = backend
        ctx.empty = X.shape[0] == 0 or (
            result.row_index is not None and result.row_index.numel() == 0
        )
        ctx.mark_non_differentiable(result.logp)
        return result.loss, result.logp

    @staticmethod
    def backward(ctx, grad_loss, grad_logp=None):
        none = (None,) * 14
        if grad_loss is None:
            return none
        if ctx.empty:  # T == 0 or every row ignored: zero gradients without binding or launching a program
            dx = (
                None
                if ctx.dx_acc is None
                else torch.zeros(
                    (ctx.num_rows, ctx.dx_acc.shape[1]),
                    dtype=torch.bfloat16,
                    device=ctx.dx_acc.device,
                )
            )
            dw = (
                None
                if ctx.dw_acc is None
                else torch.zeros(
                    ctx.dw_acc.shape,
                    dtype=ctx.grad_weight_dtype,
                    device=ctx.dw_acc.device,
                )
            )
            return (dx, dw) + none[2:]
        dz_last, x_last, x_src, x_idx = ctx.saved_tensors
        dx, dw = backward_loss(
            ctx.dx_acc,
            ctx.dw_acc,
            grad_loss,
            grad_weight_dtype=ctx.grad_weight_dtype,
            backend=ctx.backend,
            row_index=ctx.row_index,
            num_rows=ctx.num_rows,
            dz_last=dz_last,
            x_last=x_last,
            x_src=x_src,
            x_idx=x_idx,
            row_valid=ctx.row_valid,
        )
        return (dx, dw) + none[2:]


class ChunkedLmHeadLogprobFunction(torch.autograd.Function):
    """Entry (b): the forward saves the row statistic; the backward recomputes the logits."""

    @staticmethod
    def forward(
        ctx,
        X,
        W,
        labels,
        chunk_size,
        deterministic,
        backend,
        compact_rows,
        fuse_dw_cast,
    ):
        compact = _resolve_compact(
            compact_rows
        )  # resolved once: the backward recomputes the same valid-row index
        result = forward_logprob(
            X,
            W,
            labels,
            chunk_size=chunk_size,
            deterministic=deterministic,
            backend=backend,
            compact_rows=compact,
        )
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(X, W, labels, result.lse)
        ctx.chunk_size, ctx.deterministic, ctx.backend, ctx.compact = (
            chunk_size,
            deterministic,
            backend,
            compact,
        )
        ctx.fuse_dw_cast = _resolve_fuse(fuse_dw_cast)
        return result.logp

    @staticmethod
    def backward(ctx, dlogp):
        if dlogp is None:
            return (None,) * 8
        X, W, labels, lse = ctx.saved_tensors
        need_dx, need_dw = bool(ctx.needs_input_grad[0]), bool(ctx.needs_input_grad[1])
        dx, dw = backward_logprob(
            X,
            W,
            labels,
            lse,
            dlogp,
            chunk_size=ctx.chunk_size,
            need_dx=need_dx,
            need_dw=need_dw,
            deterministic=ctx.deterministic,
            backend=ctx.backend,
            compact_rows=ctx.compact,
            fuse_dw_cast=ctx.fuse_dw_cast,
        )
        return (dx, dw) + (None,) * 6


def chunked_lm_head_loss(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    objective: str = "ce",
    loss_div=None,
    infer_logp: Optional[torch.Tensor] = None,
    loss_weights: Optional[torch.Tensor] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    return_logp: bool = False,
    grad_weight_dtype: torch.dtype = torch.bfloat16,
    deterministic: bool = True,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    fuse_dw_cast: Optional[bool] = None,
):
    """Differentiable chunked LM-head + loss (see the module docstring); returns the FP32 scalar
    ``loss`` and, with ``return_logp``, the detached FP32 ``logp [T]``.

    ``grad_weight_dtype`` must equal ``W.dtype`` here: PyTorch's autograd engine
    casts every gradient to its leaf's dtype, so an FP32 ``dW`` cannot leave
    this entry through ``W.grad`` for a BF16 ``W``.  Use ``forward_loss(...,
    need_dw=True)`` + ``backward_loss(dx_acc, dw_acc, g, grad_weight_dtype=
    torch.float32, row_index=result.row_index, num_rows=result.num_rows)`` for
    an FP32 weight gradient.  ``compact_rows`` (default
    :func:`compact_rows_default`): chunk over the valid rows only; the ``dX``
    rows of ignored tokens are exact zeros either way.  ``fuse_dw_cast`` (default
    :func:`fuse_dw_cast_default`): the last chunk's weight-gradient GEMM runs in
    the backward with the upstream scale and the cast fused into its epilogue
    (bitwise the same ``dW``, one pass over ``dW`` fewer; a one-chunk call
    keeps no FP32 ``dW`` accumulator).  The fused dX finalize
    (``FLASHINFER_CAKE_LM_HEAD_LOSS_DX_FINALIZE``, on unless ``0``) adds a
    sliced dX GEMM's slabs with one kernel and writes a compacted call's
    ``[T, H]`` ``dX`` in one pass (bitwise the same ``dX``).
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    if grad_weight_dtype not in GRAD_WEIGHT_DTYPES:
        raise ValueError("grad_weight_dtype must be torch.bfloat16 or torch.float32")
    if grad_weight_dtype != W.dtype:
        raise ValueError(
            f"grad_weight_dtype={grad_weight_dtype} differs from W.dtype={W.dtype}: the autograd engine casts "
            "every gradient to its leaf's dtype, so this entry cannot return it; use forward_loss(..., need_dw=True) + "
            "backward_loss(dx_acc, dw_acc, g, grad_weight_dtype=torch.float32, row_index=..., num_rows=...) for an FP32 dW"
        )
    loss, logp = ChunkedLmHeadLossFunction.apply(
        X,
        W,
        labels,
        objective,
        loss_div,
        infer_logp,
        loss_weights,
        int(chunk_size),
        bool(return_logp),
        grad_weight_dtype,
        deterministic,
        backend,
        compact_rows,
        fuse_dw_cast,
    )
    return (loss, logp.detach()) if return_logp else loss


def chunked_lm_head_logprob(
    X: torch.Tensor,
    W: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    deterministic: bool = True,
    backend: str = "cake",
    compact_rows: Optional[bool] = None,
    fuse_dw_cast: Optional[bool] = None,
) -> torch.Tensor:
    """Differentiable FP32 ``logp [T]`` (0 on ignored rows) for arbitrary downstream losses;
    ``compact_rows`` / ``fuse_dw_cast`` as in :func:`chunked_lm_head_loss`."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    return ChunkedLmHeadLogprobFunction.apply(
        X,
        W,
        labels,
        int(chunk_size),
        deterministic,
        backend,
        compact_rows,
        fuse_dw_cast,
    )
