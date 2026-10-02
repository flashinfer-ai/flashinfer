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

Cake backend: Kimi-K3 vision tower (MoonViT-3D encoder + PatchMergerV2) on
SM100 / SM103.

The operator is the complete ``kimi_k3_vision_tower(pixel_values, grid_thws,
weights, out)`` call of ``nvidia/Kimi-K3-NVFP4`` (``modeling_kimi_k3.py``:
``MoonViT3dPretrainedModel`` + ``PatchMergerMLPV2``): packed BF16 patch pixels
``[T, 3, 14, 14]`` and the host ``grid_thws`` (one ``(t, h, w)`` per image or
4-frame video group, ``h``/``w`` even) to the caller-owned BF16 ``out
[N, 7168]`` with ``N = sum (h / 2) * (w / 2)``.  Every launch of the route is a
generated Cake program::

    patch_embed          gemm pos_sqxw:       x  = bf16(pixels @ Wpe^T) + pos_rows          [T, 1024]
                                              + handoff: xw = bf16(x * norm0[0]), stats = rowsumsq(x)
    27 x encoder layer:
      layer_norm_qkv_rope gemm norm_qkv_rope: q, k, v = RoPE2D(bf16((xw @ Wqkv^T) * rstd(x))) 3 x [T, 12, 128]
                          (``*_cs`` tiles: cos/sin from the packed f16x2 table ``rope_cs`` of the plan)
      layer_attention     packed-varlen noncausal BF16 attention per grid_thw segment  [T, 12, 128]
                          (``attention:<layout>[:wide][:split]``; a split plan writes the tail-round units'
                          K/V parts as FP32 partials and runs ``layer_attention_merge`` -- the exact
                          fixed-order merge -- right after it)
      layer_out_proj      gemm residual_wo_sqxw: x += bf16(a @ Wo^T); xw = bf16(x * norm1), stats (in place)
      layer_norm_fc0_gelu gemm norm_gelu:      f  = bf16(gelu_tanh(bf16((xw @ Wfc0^T) * rstd(x)))) [T, 4096]
      layer_fc1           gemm residual_fc1_sqxw: x += bf16(f @ Wfc1^T); xw = bf16(x * norm0[l+1]), stats
                          (in place; the last layer runs the plain residual_fc1 form: no handoff)
    final_norm_merge     final RMSNorm + 2x2 spatial / temporal-mean merge               [N, 4096]
    merger_gemm0         gemm gelu_erf:       h  = bf16(gelu_erf(bf16(m @ Wp0^T)))         [N, 4096]
    merger_gemm1         gemm rmsnorm:        y  = bf16(h @ Wp1^T)                         [N, 7168]
    merger_rmsnorm_apply out = bf16(RMSNorm(y, post_norm, eps = 1e-5))                    (in place)

Programmatic dependent launch: the exported programs follow the Cake
production PDL default (``kimi_k3_vision_tower.default_use_pdl``, env
``KIMI_K3_VISION_TOWER_PDL``; on since the round-2 route), so every launch
of the sequence carries the
``cudaLaunchAttributeProgrammaticStreamSerialization`` attribute.  The
generated bindings own it: the attribute is emitted into each binding's
launch (the generated binding's host shim), the kernel runs its prologue
(mbarrier / TMEM setup, descriptor prefetch) while its predecessor drains,
``griddepcontrol.wait``s before its first access to a route buffer and
signals ``launch_dependents`` so the successor's prologue overlaps its tail.
The host only calls the bindings in order; under CUDA-graph capture the
attribute becomes a programmatic dependency edge.  Programs exported with
PDL off are ordinary serial launches.  The numerics are identical either way.
Round 5: each GEMM tile has two PDL binaries -- the production one and the
``PDL_EARLY`` one (``griddepcontrol.wait`` moved into the load / epilogue
roles so the weight stages of the first ring fill stream before the wait,
``launch_dependents`` at entry).  The host picks the early binary per launch
from the form's census window (:func:`pdl_early_on`, Cake
``kimi_k3_vision_gemm._pdl_early_on`` / ``PDL_EARLY_WINDOW``), never on the
stream-K / tail / multicast / ``pos`` tiles (:func:`pdl_early_selected`); the
registry names it ``gemm:<variant>:<tile>:pdle``.  Same numerics, same
kernel parameters.

Host work is split exactly like the Cake production launcher:

* :func:`prepare_kimi_k3_vision_weights` -- once per model: the patch
  projection is zero-padded / shifted for the pixel-row TMA maps and every
  parameter copied contiguously.  The RMSNorm weights are NOT folded into the
  GEMM weights: the residual-stream producers (``patch_embed``,
  ``layer_out_proj``, ``layer_fc1``) also store ``xw = bf16(x * w_next)`` and
  the FP32 ``[T, 16]`` per-64-column row sums of squares of ``x``; the norm
  GEMMs read ``xw`` with the original weight, compute the row ``rstd`` in FP32
  from those statistics and scale the FP32 accumulator before the single BF16
  rounding.
* :func:`build_kimi_k3_vision_plan` -- once per ``grid_thws`` batch: the
  ``cu_seqlens``, the 2-D RoPE ``cos``/``sin`` tables and their packed f16x2
  ``rope_cs`` table (:func:`pack_rope_table`: one ``(cos, sin)`` word per pair,
  read by the ``*_cs`` QKV tiles), the merge table, the attention segment plan
  (unit layout + LPT unit table), the GEMM tile configurations for the token
  counts and every workspace, including the attention kernel's TMA descriptor
  workspace (``attn_tma_desc``, see below).  The positional
  rows depend on the weights and are derived by
  :func:`prepare_kimi_k3_vision_tower` (or passed in by a serving runtime that
  caches them per grid).
* :func:`prepare_kimi_k3_vision_tower` -- binds every launch of the sequence
  to the generated argument plans and prepares the attention descriptor
  workspace (below).  The returned runner's ``launch()`` performs no
  allocation and no host synchronization and is CUDA-graph capturable.

TMA descriptor ABI: the GEMM family and the RMSNorm apply pass take their
tensor maps as ``__grid_constant__`` kernel parameters, exactly as in Cake
production.  The attention kernel is exported in its production *pointer*
ABI: its four ``CUtensorMap``s (Q, K, V, O -- all plan workspaces) live in
device memory owned by the plan (``workspace["attn_tma_desc"]``, 128 B per
map) and the kernel receives their addresses.  The generated attention
binding exports two entries: ``run_prepare_tma`` (same arguments as ``run``;
validates, encodes the descriptors and copies them into the workspace once,
synchronously, outside CUDA-graph capture) and ``run`` (launch-only).
:func:`prepare_kimi_k3_vision_tower` calls ``run_prepare_tma`` once per plan
(the descriptors depend only on plan-owned buffers); ``launch()`` and graph
replays never touch the workspace again.

The host plan reproduces the Cake planner table for table (segment plan,
unit table, merge table, tile selection, launch grids); the generated-program
export checks that parity on every contract row.  See ``README.md`` in this
package and flashinfer-ai/flashinfer#4568 (tracker #4254).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Optional, Sequence, Union

import torch
import tvm_ffi

from ..minimax_h3_varlen_attention.cake_backend import assign_unit_slots
from .cake_jit import (
    ATTENTION_MERGE_KERNEL_KEY,
    MERGE_KERNEL_KEY,
    MODULES,
    RMSNORM_APPLY_KERNEL_KEY,
    attention_kernel_key,
    gemm_kernel_key,
    kernel_module_name,
    load_cake_kimi_k3_vision_tower_module,
    route_available,
)

# ---------------------------------------------------------------------------
# Model constants (vision_config of nvidia/Kimi-K3-NVFP4)
# ---------------------------------------------------------------------------
PATCH = 14
IN_CH = 3
PATCH_DIM = IN_CH * PATCH * PATCH  # 588
HIDDEN = 1024
QKV_HIDDEN = 1536
HEADS = 12
HEAD_DIM = QKV_HIDDEN // HEADS  # 128
QKV_N = 3 * QKV_HIDDEN  # 4608
FFN = 4096
MERGE_KERNEL = (2, 2)
MERGED_DIM = HIDDEN * MERGE_KERNEL[0] * MERGE_KERNEL[1]  # 4096
TEXT_HIDDEN = 7168
POS_EMB_H = 64
POS_EMB_W = 64
POS_EMB_T = 4
ROPE_THETA = 10000.0
ROPE_MAX_HW = 512
ROPE_PAIRS = HEAD_DIM // 2  # 64
NORM_EPS = 2.0**-7  # nn.RMSNorm(eps=None) on BF16 -> torch.finfo(torch.bfloat16).eps
PROJECTOR_EPS = 1.0e-5
SOFTMAX_SCALE = 1.0 / math.sqrt(HEAD_DIM)

# GEMM family geometry (kimi_k3_vision_gemm).
PATCH_DIM_PAD = 640  # 10 x 64
POS_SHIFT = 4  # odd pixel rows start 8 bytes past a 16-byte boundary: TMA reads them 4 elements early
GEMM_BLOCK_M = 128
GEMM_BLOCK_K = 64
STATS_PARTS = 16  # FP32 per-64-column partial sums of squares of a 1024-wide row
RMS_ROWS_PER_CTA = 4  # rmsnorm_apply: one warp per 7168-wide row, four rows per CTA

# Attention (kimi_k3_vision_attention, forked from minimax_h3_varlen_attention).
ATTN_BLOCK_M = 128  # Q rows per tile
ATTN_BLOCK_N = 128  # K/V rows per pipeline block
ATTN_UNIT_OVERHEAD_BLOCKS = 2
ATTN_MAX_HEADS = 1 << 15
ATTN_MAX_SEGMENT_CLUSTERS = 1 << 16
MODE_TWO_TILE = 2
MODE_SPLIT_KV = 1
SPLIT_MARGIN = 0.05
# Round 4 (CAKE-722 lever A): the SPLIT_KV layout is priced per architecture (Cake
# ``kimi_k3_vision_attention.SPLIT_COST_MODELS`` / ``split_cost_model``).  A split unit costs
# ``iter_scale * stage_iters + unit_overhead`` K/V-block equivalents and the layout wins when its
# LPT makespan is below ``(1 - margin)`` x the two-tile makespan (which keeps the H3 block model:
# ``iter_scale`` 1, ``ATTN_UNIT_OVERHEAD_BLOCKS``).  ``"r3"`` is the round-3 model, used by the
# arches without an entry and by plans built without an arch; on sm_100a the ring loop is priced
# at its measured 0.95 blocks per K/V block with a one-block unit boundary and no margin (the
# post-A4 re-fit routes every contract row but img_640x480 to the ring there).
SPLIT_COST_MODELS: dict[str, dict[str, float]] = {
    "r3": {
        "iter_scale": 1.0,
        "unit_overhead": float(ATTN_UNIT_OVERHEAD_BLOCKS),
        "margin": SPLIT_MARGIN,
    },
    "sm_100a": {"iter_scale": 0.95, "unit_overhead": 1.0, "margin": 0.0},
}
# Round 3 (A5): the production SPLIT_KV form is the shared-O three-deep score ring
# (``attention:ring3``) where the arch / longest-segment table selects it (Cake
# ``kimi_k3_vision_attention.RING3_MAX_SEGMENT_TOKENS`` / ``RING3_MIN_SEGMENT_TOKENS``):
# every SPLIT_KV row on sm_100a, segments of 576 .. 10764 tokens on sm_103a.
RING3_MAX_SEGMENT_TOKENS: dict[str, Optional[int]] = {"sm_100a": None, "sm_103a": 10764}
RING3_MIN_SEGMENT_TOKENS: dict[str, int] = {"sm_100a": 0, "sm_103a": 576}
# Round 6 (CAKE-771) per-row attention rules of the Cake plan (``kimi_k3_vision_attention.build_segment_plan``):
# * lever G grid shaping (Cake ``GRID_POLICIES`` / ``GRID_AUTO_MIN_FREED`` / ``_GRID_POLICY`` default "auto"):
#   "min" = the smallest cluster count whose LPT makespan equals the full grid's, applied by "auto" only on the
#   architectures listed and only when it frees at least that many clusters (sm_103a: img_1280x720 74 -> 60 on the
#   unsplit two-tile plan, batch8_448 74 -> 64; sm_100a has no entry: every shaped row lost there).
GRID_POLICIES = ("full", "min", "auto")
GRID_AUTO_MIN_FREED: dict[str, int] = {"sm_103a": 10}
GRID_DEFAULT_POLICY = "auto"
# * split-aware layout + automatic KV tail split (Cake ``SPLIT_POLICIES`` / ``SPLIT_TAIL_RULE`` / ``_SPLIT_POLICY`` default
#   "auto" / ``MAX_KV_PARTS``): the tail-round units of each layout are split into k = min(G // tail, min_blocks // 2,
#   MAX_KV_PARTS) contiguous K/V ranges (the partial-output ``:split`` kernel build + the ``attention_merge`` launch) when
#   the block-model makespan drops by >= min_gain and the row spans >= min_span blocks; the layout is then chosen on the
#   split-aware makespans with the arch's margin.  Per-part fixed cost (blocks) per layout and the merge launch are charged
#   in block units.
SPLIT_POLICIES = ("off", "auto")
SPLIT_TAIL_RULE: dict[str, Any] = {
    "min_gain": 0.05,
    "min_span": 75.0,
    "merge_blocks": 4.0,
    "part_overhead": {MODE_TWO_TILE: 8.0, MODE_SPLIT_KV: 2.0},
}
SPLIT_DEFAULT_POLICY = "auto"
MAX_KV_PARTS = 8  # Cake ``kimi_k3_vision_attention.MAX_KV_PARTS`` (= ``kimi_k3_vision_attention_merge.MAX_KV_PARTS``)
# * the wide unit table + next-unit prefetch build (Cake ``UNIT_PREFETCH_MAX_TOKENS`` / ``unit_prefetch_selected``): on for
#   rows of at most this many tokens per arch (the unit start dominates 1-4-block units), off elsewhere.
UNIT_PREFETCH_MAX_TOKENS: dict[str, int] = {"sm_100a": 1656, "sm_103a": 2552}
# The split merge kernel (Cake ``kimi_k3_vision_attention_merge``): one warp per output row, four rows per CTA; merge
# table rows ``[segment, head << 16 | cluster, first partial slot, parts]``.
MERGE_ROWS_PER_CTA = 4
ATTN_MERGE_WORDS = 4
PROBE_WORDS = 64  # unused diagnostic buffer parameter of the attention kernel (PROBE_UNITS * PROBE_EVENTS)

SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}

STAGE_NAMES = (
    "patch_embed",
    "layer_norm_qkv_rope",
    "layer_attention",
    "layer_attention_merge",  # split-KV rows only (``AttentionPlan.num_merge_units > 0``)
    "layer_out_proj",
    "layer_norm_fc0_gelu",
    "layer_fc1",
    "layer_fc1_last",
    "final_norm_merge",
    "merger_gemm0",
    "merger_gemm1",
    "merger_rmsnorm_apply",
)
# GEMM variant launched by each GEMM stage and the token count it runs on.
# The residual-stream producers run their ``_sqxw`` handoff form: the epilogue
# also stores ``xw = bf16(x * w_next)`` (the next RMSNorm's weight applied on
# the activation side) and the FP32 per-64-column row sums of squares that the
# following norm GEMM consumes; the last layer's FC1 has no consumer and runs
# the plain form.
STAGE_GEMM_VARIANT = {
    "patch_embed": ("pos_sqxw", "T"),
    "layer_norm_qkv_rope": ("norm_qkv_rope", "T"),
    "layer_out_proj": ("residual_wo_sqxw", "T"),
    "layer_norm_fc0_gelu": ("norm_gelu", "T"),
    "layer_fc1": ("residual_fc1_sqxw", "T"),
    "layer_fc1_last": ("residual_fc1", "T"),
    "merger_gemm0": ("gelu_erf", "N"),
    "merger_gemm1": ("rmsnorm", "N"),
}
PRODUCTION_GEMM_VARIANTS = tuple(
    dict.fromkeys(variant for variant, _count in STAGE_GEMM_VARIANT.values())
)

# Exact keyword sets of the generated ``run`` entries (bound by the export's
# argument plans, in the kernels' parameter order); ``grid`` is expanded to
# ``grid_x/y/z``.  The GEMM template takes the packed RoPE table ``CS``
# (bf16 view of the u32 words; defaults to ``B``), the tail split-K workspace
# ``WS`` (f32) / arrival counters ``FLAGS`` (u32) with ``full_tiles`` /
# ``tail_split``; the tiles without a stream-K tail have no tail paths, so the
# host binds the Cake launcher's dummies (``WS`` = f32 zeros[16], ``FLAGS`` =
# u32 zeros[16], ``full_tiles`` = cluster tiles, ``tail_split`` = 1); the
# stream-K twin binds the plan's partial workspace / counters with the
# data-parallel tile count and the k-steps per tail piece.
GEMM_KWARGS = (
    "A",
    "A2",
    "B",
    "C",
    "C2",
    "C3",
    "R",
    "COS",
    "SIN",
    "CS",
    "SQ",
    "XW",
    "WN",
    "WS",
    "FLAGS",
    "RT",
    "CT",
    "XWT",
    "M",
    "m_tiles",
    "full_tiles",
    "tail_split",
    "pf_l2",
    "eps",
    "grid",
)
ATTENTION_KWARGS = (
    "Q",
    "Q_raw",
    "K",
    "V",
    "O",
    "O_tma",
    "seg_begin",
    "seg_len",
    "unit_table",
    "probe",
    "total_tiles",
    "num_heads",
    "softmax_scale_log2",
    "partial_O",
    "partial_ML",
    "tma_descriptor_workspace",
    "grid",
)
ATTENTION_MERGE_KWARGS = (
    "partial_O",
    "partial_ML",
    "merge_table",
    "seg_begin",
    "seg_len",
    "O",
    "num_heads",
    "part_rows",
    "softmax_scale_log2",
    "grid",
)
MERGE_KWARGS = ("x", "norm_weight", "merge_table", "m_out", "eps", "grid")
RMSNORM_APPLY_KWARGS = ("y", "weight", "rowsumsq", "M", "eps", "grid")


# ---------------------------------------------------------------------------
# Token bookkeeping and per-grid tables
# ---------------------------------------------------------------------------


def validate_grid_thws(
    grid_thws: Sequence[Sequence[int]],
) -> list[tuple[int, int, int]]:
    grids = []
    for entry in grid_thws:
        if len(entry) != 3:
            raise ValueError(f"grid_thw entries must be (t, h, w), got {tuple(entry)}")
        t, h, w = (int(v) for v in entry)
        if t < 1 or h < 1 or w < 1:
            raise ValueError(f"grid_thw entries must be positive, got {(t, h, w)}")
        if h % MERGE_KERNEL[0] or w % MERGE_KERNEL[1]:
            raise ValueError(
                f"grid h/w must be multiples of {MERGE_KERNEL}, got {(t, h, w)}"
            )
        if h > ROPE_MAX_HW or w > ROPE_MAX_HW:
            raise ValueError(
                f"grid h/w exceed the 2-D RoPE table {ROPE_MAX_HW}: {(t, h, w)}"
            )
        if t > POS_EMB_T:
            raise ValueError(
                f"t={t} exceeds the temporal position table {POS_EMB_T}: {(t, h, w)}"
            )
        grids.append((t, h, w))
    if not grids:
        raise ValueError("grid_thws must describe at least one segment")
    return grids


def cu_seqlens_of(grid_thws: Sequence[Sequence[int]]) -> tuple[int, ...]:
    bounds = [0]
    for t, h, w in grid_thws:
        bounds.append(bounds[-1] + int(t) * int(h) * int(w))
    return tuple(bounds)


def merged_tokens(grid_thws: Sequence[Sequence[int]]) -> int:
    return sum(
        (int(h) // MERGE_KERNEL[0]) * (int(w) // MERGE_KERNEL[1])
        for _, h, w in grid_thws
    )


def token_positions(grid_thws: Sequence[Sequence[int]]):
    """Per-token ``(t, y, x)`` int32 CPU tensors in packed order (t slow, then y, then x)."""
    ts, ys, xs = [], [], []
    for t, h, w in grid_thws:
        tt = torch.arange(t, dtype=torch.int32).view(t, 1, 1).expand(t, h, w)
        yy = torch.arange(h, dtype=torch.int32).view(1, h, 1).expand(t, h, w)
        xx = torch.arange(w, dtype=torch.int32).view(1, 1, w).expand(t, h, w)
        ts.append(tt.reshape(-1))
        ys.append(yy.reshape(-1))
        xs.append(xx.reshape(-1))
    return torch.cat(ts), torch.cat(ys), torch.cat(xs)


def rope_freqs() -> torch.Tensor:
    """``1 / theta^(4i/128)`` for ``i in [0, 32)`` (``Rope2DPosEmbRepeated._precompute_freqs_cis``)."""
    dim_range = torch.arange(0, HEAD_DIM, 4)[: HEAD_DIM // 4].float()
    return 1.0 / (ROPE_THETA ** (dim_range / HEAD_DIM))


def rope_cos_sin(
    grid_thws: Sequence[Sequence[int]], device: Optional[torch.device] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 ``cos``/``sin`` tables ``[T, 64]`` indexed by the interleaved pair of the 128-wide head.

    Pair ``2i`` rotates by ``x * freq_i`` (width axis), pair ``2i + 1`` by
    ``y * freq_i`` (height axis); the table repeats over ``t``.
    """
    _, ys, xs = token_positions(grid_thws)
    freqs = rope_freqs()
    x_ang = xs.float()[:, None] * freqs[None, :]
    y_ang = ys.float()[:, None] * freqs[None, :]
    ang = torch.stack([x_ang, y_ang], dim=-1).reshape(-1, ROPE_PAIRS)
    cos, sin = torch.cos(ang), torch.sin(ang)
    if device is not None:
        cos, sin = cos.to(device), sin.to(device)
    return cos.contiguous(), sin.contiguous()


def sincos_time_table(dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """``get_1d_sincos_pos_embed(1024, 4)``: ``[4, 1024]`` = ``[sin(t*omega) | cos(t*omega)]``."""
    omega = torch.arange(HIDDEN // 2, dtype=torch.float64) / (HIDDEN / 2.0)
    omega = 1.0 / (10000.0**omega)
    pos = torch.arange(POS_EMB_T, dtype=torch.float64)
    out = pos[:, None] * omega[None, :]
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()
    return emb if dtype is None else emb.to(dtype)


def pos_emb_rows(
    pos_emb_weight: torch.Tensor,
    time_weight: torch.Tensor,
    grid_thws: Sequence[Sequence[int]],
) -> torch.Tensor:
    """``Learnable2DInterpPosEmbDivided_fixed`` rows ``[T, 1024]`` in the weight's dtype.

    Bilinear resize of the ``[64, 64, 1024]`` table to ``(h, w)`` on the stored
    dtype, plus ``time_weight[t]`` for multi-frame grids, in packed token order.
    """
    import torch.nn.functional as F

    rows = []
    for t, h, w in grid_thws:
        if (h, w) == tuple(pos_emb_weight.shape[:2]):
            emb2d = pos_emb_weight.reshape(-1, HIDDEN)
        else:
            emb2d = (
                F.interpolate(
                    pos_emb_weight.permute(2, 0, 1).unsqueeze(0),
                    size=(h, w),
                    mode="bilinear",
                )
                .squeeze(0)
                .permute(1, 2, 0)
                .reshape(-1, HIDDEN)
            )
        if t == 1:
            emb3d = emb2d
        else:
            emb3d = emb2d.unsqueeze(0).repeat(t, 1, 1) + time_weight[:t].unsqueeze(1)
        rows.append(emb3d.reshape(-1, HIDDEN))
    return torch.cat(rows, dim=0).contiguous()


def build_merge_table_host(grid_thws: Sequence[Sequence[int]]) -> list[int]:
    """Flat ``[N * 4]`` int32 rows ``(first_token, row_stride, frame_stride, t)`` in merged-token order."""
    table: list[int] = []
    start = 0
    kh, kw = MERGE_KERNEL
    for t, h, w in grid_thws:
        for ny in range(h // kh):
            for nx in range(w // kw):
                table.extend((start + (ny * kh) * w + nx * kw, w, h * w, t))
        start += t * h * w
    return table


def build_merge_table(
    grid_thws: Sequence[Sequence[int]], device: torch.device
) -> torch.Tensor:
    """Device ``int32 [N, 4]`` merge table (built once per ``grid_thws``)."""
    rows = build_merge_table_host(grid_thws)
    return torch.tensor(rows or [0, 0, 0, 1], dtype=torch.int32, device=device).view(
        -1, 4
    )


# ---------------------------------------------------------------------------
# Attention segment plan (unit layout rule + LPT unit table)
# ---------------------------------------------------------------------------


def attention_grid_clusters(device: torch.device) -> int:
    """Persistent-grid capacity of ``device`` in 2-CTA clusters (``num_SMs / 2``)."""
    return max(1, torch.cuda.get_device_properties(device).multi_processor_count // 2)


def _attention_unit_costs(
    lens: Sequence[int],
    num_heads: int,
    tiles_per_cta: int,
    *,
    iter_scale: float = 1.0,
    unit_overhead: float = ATTN_UNIT_OVERHEAD_BLOCKS,
) -> tuple[list[tuple[int, int, int]], list[float]]:
    """Enumerate (segment, head, cluster) units segment-major (heads slow, clusters fast) with LPT costs
    of ``iter_scale * stage_iters + unit_overhead`` K/V-block equivalents (the defaults are the H3
    block-count model; the split cost model of the arch retunes the SPLIT_KV layout)."""
    rows_per_cluster = 2 * tiles_per_cta * ATTN_BLOCK_M
    units: list[tuple[int, int, int]] = []
    costs: list[float] = []
    for seg, length in enumerate(lens):
        blocks = (length + ATTN_BLOCK_N - 1) // ATTN_BLOCK_N
        stage_iters = blocks if tiles_per_cta == MODE_TWO_TILE else (blocks + 1) // 2
        cost = iter_scale * stage_iters + unit_overhead
        seg_clusters = (length + rows_per_cluster - 1) // rows_per_cluster
        if seg_clusters >= ATTN_MAX_SEGMENT_CLUSTERS:
            raise ValueError(
                f"a segment needs fewer than {ATTN_MAX_SEGMENT_CLUSTERS} clusters"
            )
        for head in range(num_heads):
            for c in range(seg_clusters):
                units.append((seg, head, c))
                costs.append(cost)
    return units, costs


def lpt_makespan(costs: Sequence[float], grid_clusters: int) -> tuple[float, list[int]]:
    """(max per-cluster cost, slot -> unit) of the kernel's LPT slot assignment on ``grid_clusters`` slots."""
    if not costs:
        return 0.0, []
    num_clusters = min(grid_clusters, len(costs))
    slots = assign_unit_slots(list(costs), num_clusters)
    per_cluster = [0.0] * num_clusters
    for slot, unit in enumerate(slots):
        per_cluster[slot % num_clusters] += costs[unit]
    return max(per_cluster), slots


def ring3_selected(arch: str, lens: Sequence[int]) -> bool:
    """Whether the production SPLIT_KV form on ``arch`` is the shared-O ring for these segment lengths
    (mirrors ``kimi_k3_vision_attention.ring3_selected``)."""
    if arch not in RING3_MAX_SEGMENT_TOKENS:
        return False
    limit = RING3_MAX_SEGMENT_TOKENS[arch]
    longest = max((int(n) for n in lens), default=0)
    if longest < RING3_MIN_SEGMENT_TOKENS[arch]:
        return False
    return limit is None or longest <= int(limit)


def unit_prefetch_selected(arch: Optional[str], total_tokens: int) -> bool:
    """Whether the production plan on ``arch`` uses the wide unit table (the ``:wide`` build) for a row of
    ``total_tokens`` (mirrors ``kimi_k3_vision_attention.unit_prefetch_selected``)."""
    limit = UNIT_PREFETCH_MAX_TOKENS.get(arch) if arch is not None else None
    return limit is not None and int(total_tokens) <= int(limit)


def shape_grid(
    costs: Sequence[float], grid_clusters: int, policy: str, arch: Optional[str] = None
) -> int:
    """Cluster count of the persistent grid under ``policy`` (mirrors ``kimi_k3_vision_attention.shape_grid``):
    ``"full"`` = ``min(grid_clusters, items)``; ``"min"`` = the smallest count whose LPT makespan equals the
    full grid's; ``"auto"`` = ``"min"`` when ``arch`` has a ``GRID_AUTO_MIN_FREED`` entry and the shaping frees
    at least that many clusters, otherwise ``"full"``."""
    if policy not in GRID_POLICIES:
        raise ValueError(f"grid policy must be one of {GRID_POLICIES}, got {policy!r}")
    full = min(grid_clusters, max(len(costs), 1))
    if policy == "full" or not costs:
        return full
    if policy == "auto" and arch not in GRID_AUTO_MIN_FREED:
        return full
    span = lpt_makespan(costs, full)[0]
    g = full
    while g > 1 and lpt_makespan(costs, g - 1)[0] <= span:
        g -= 1
    if policy == "auto" and full - g < GRID_AUTO_MIN_FREED[arch]:
        return full
    return g


def split_cost_model(arch: Optional[str]) -> dict[str, float]:
    """The SPLIT_KV cost model of ``arch`` (``SPLIT_COST_MODELS``; arches without an entry and
    ``None`` use the round-3 model ``"r3"``).  Mirrors ``kimi_k3_vision_attention.split_cost_model``."""
    if arch is not None and arch in SPLIT_COST_MODELS:
        return SPLIT_COST_MODELS[arch]
    return SPLIT_COST_MODELS["r3"]


def select_tiles_per_cta(
    lens: Sequence[int],
    num_heads: int,
    grid_clusters: int,
    arch: Optional[str] = None,
) -> dict[str, Any]:
    """Unit layout from the LPT makespans on this device: two tiles per CTA unless
    SPLIT_KV (one tile per CTA, both softmax stages on the K/V parities) wins by the
    margin of ``arch``'s split cost model (:func:`split_cost_model`; the two-tile
    layout is always priced by the H3 block model).  Mirrors the Cake production
    planner's ``_select_tiles_per_cta``."""
    model = split_cost_model(arch)
    makespan: dict[int, float] = {}
    for mode in (MODE_TWO_TILE, MODE_SPLIT_KV):
        if mode == MODE_SPLIT_KV:
            _units, costs = _attention_unit_costs(
                lens,
                num_heads,
                mode,
                iter_scale=model["iter_scale"],
                unit_overhead=model["unit_overhead"],
            )
        else:
            _units, costs = _attention_unit_costs(lens, num_heads, mode)
        makespan[mode], _slots = lpt_makespan(costs, grid_clusters)
    two, split = makespan[MODE_TWO_TILE], makespan[MODE_SPLIT_KV]
    force_split = bool(model.get("force_split", False))
    chosen = (
        MODE_SPLIT_KV
        if force_split or split < two * (1.0 - model["margin"])
        else MODE_TWO_TILE
    )
    return {"tiles_per_cta": chosen, "makespan": makespan, "cost_model": dict(model)}


def _split_aware_makespan(
    lens: Sequence[int],
    num_heads: int,
    grid_clusters: int,
    mode: int,
    costs: Sequence[float],
    rule: dict[str, Any],
    iter_scale: float = 1.0,
) -> tuple[float, int]:
    """(LPT makespan, k) of ``mode`` with its tail-round units split ``k`` ways per the KV-split rule (k = 0:
    unsplit); mirrors ``kimi_k3_vision_attention._split_aware_makespan``."""
    total = len(costs)
    span0 = lpt_makespan(costs, grid_clusters)[0]
    num_clusters = min(grid_clusters, max(total, 1))
    tail = total % num_clusters if total else 0
    if not tail or total <= num_clusters:
        return span0, 0
    units, _c = _attention_unit_costs(lens, num_heads, mode)
    blocks_of = [
        (lens[seg] + ATTN_BLOCK_N - 1) // ATTN_BLOCK_N for seg, _head, _c in units
    ]
    chosen = sorted(range(total), key=lambda u: (costs[u], u))[:tail]
    k = min(num_clusters // tail, min(blocks_of[u] for u in chosen) // 2, MAX_KV_PARTS)
    if k < 2:
        return span0, 0
    part_overhead = float(rule["part_overhead"][mode])
    items: list[float] = []
    chosen_set = set(chosen)
    for u in range(total):
        if u not in chosen_set:
            items.append(costs[u])
            continue
        blocks = blocks_of[u]
        per = (blocks + k - 1) // k
        for lo in range(0, blocks, per):
            count = min(per, blocks - lo)
            stage_iters = count if mode == MODE_TWO_TILE else (count + 1) // 2
            items.append(iter_scale * float(stage_iters) + part_overhead)
    span1 = lpt_makespan(items, grid_clusters)[0] + float(rule["merge_blocks"])
    if span0 < float(rule["min_span"]) or span1 > span0 * (
        1.0 - float(rule["min_gain"])
    ):
        return span0, 0
    return span1, k


def select_layout_and_split(
    lens: Sequence[int],
    num_heads: int,
    grid_clusters: int,
    arch: Optional[str] = None,
    rule: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Split-aware layout selection (split policy "auto"; mirrors
    ``kimi_k3_vision_attention.select_layout_and_split``): per layout the better of the unsplit / tail-split
    makespans, then the two-tile-vs-SPLIT_KV comparison with the arch's margin.  ``kv_split`` is ``0`` (the
    automatic tail split of the chosen layout) or ``None`` (whole units)."""
    rule = SPLIT_TAIL_RULE if rule is None else rule
    model = split_cost_model(arch)
    makespan: dict[int, float] = {}
    split_k: dict[int, int] = {}
    for mode in (MODE_TWO_TILE, MODE_SPLIT_KV):
        if mode == MODE_SPLIT_KV:
            _units, costs = _attention_unit_costs(
                lens,
                num_heads,
                mode,
                iter_scale=model["iter_scale"],
                unit_overhead=model["unit_overhead"],
            )
        else:
            _units, costs = _attention_unit_costs(lens, num_heads, mode)
        makespan[mode], split_k[mode] = _split_aware_makespan(
            lens,
            num_heads,
            grid_clusters,
            mode,
            costs,
            rule,
            iter_scale=model["iter_scale"] if mode == MODE_SPLIT_KV else 1.0,
        )
    two, split = makespan[MODE_TWO_TILE], makespan[MODE_SPLIT_KV]
    force_split = bool(model.get("force_split", False))
    mode = (
        MODE_SPLIT_KV
        if force_split or split < two * (1.0 - model["margin"])
        else MODE_TWO_TILE
    )
    return {
        "tiles_per_cta": mode,
        "kv_split": 0 if split_k[mode] >= 2 else None,
        "makespan": makespan,
        "split_k": split_k,
        "cost_model": dict(model),
        "rule": dict(rule),
    }


def build_unit_table(
    lens: Sequence[int],
    num_heads: int,
    grid_clusters: int,
    tiles_per_cta: int,
    *,
    kv_split: Optional[int] = None,
    wide: bool = False,
    grid_policy: str = "full",
    arch: Optional[str] = None,
) -> dict[str, Any]:
    """Device-free unit table (mirrors ``kimi_k3_vision_attention.build_unit_table`` for the production 2-CTA
    cluster): slot-ordered ``[segment, head << 16 | cluster]`` words, plus ``[doc_begin, doc_len]`` for the wide
    build and ``[kv_block_begin << 16 | kv_blocks, partial_slot]`` for the split build.  ``kv_split`` ``None`` =
    whole units (two- / four-word table), ``0`` = the automatic tail split ``k = min(G // tail, min_blocks // 2,
    MAX_KV_PARTS)`` of the ``units mod G`` cheapest units (no split below 2), ``k >= 2`` = that split, ``1`` = the
    partial format without split units.  Every part is an LPT item of its own block count; ``merge_table`` lists
    ``[segment, head << 16 | cluster, first slot, parts]`` per split unit; the items are laid out on
    ``shape_grid(item_costs, grid_clusters, grid_policy, arch)`` clusters."""
    if tiles_per_cta not in (MODE_TWO_TILE, MODE_SPLIT_KV):
        raise ValueError(
            f"tiles_per_cta must be {MODE_TWO_TILE} or {MODE_SPLIT_KV}, got {tiles_per_cta}"
        )
    if kv_split is not None and not 0 <= int(kv_split) <= MAX_KV_PARTS:
        raise ValueError(
            f"kv_split must be None or in [0, {MAX_KV_PARTS}], got {kv_split}"
        )
    units, costs = _attention_unit_costs(lens, num_heads, tiles_per_cta)
    total_units = len(units)
    num_clusters = min(grid_clusters, max(total_units, 1))
    rows_per_cluster = 2 * tiles_per_cta * ATTN_BLOCK_M
    blocks_of = [
        (lens[seg] + ATTN_BLOCK_N - 1) // ATTN_BLOCK_N for seg, _head, _c in units
    ]
    split_of = [1] * total_units
    k_used = 0
    if kv_split is not None and total_units:
        k = int(kv_split)
        tail = total_units % num_clusters
        chosen = (
            sorted(range(total_units), key=lambda u: (costs[u], u))[:tail]
            if tail
            else []
        )
        if chosen and k == 0:
            k = min(
                num_clusters // tail,
                min(blocks_of[u] for u in chosen) // 2,
                MAX_KV_PARTS,
            )
        if chosen and k >= 2:
            k_used = k
            for u in chosen:
                split_of[u] = max(1, min(k, blocks_of[u]))
    items: list[tuple[int, int, int, int, int, int]] = []
    item_costs: list[float] = []
    merge: list[int] = []
    slots_used = 0
    for u, (seg, head, c) in enumerate(units):
        blocks = blocks_of[u]
        parts = split_of[u]
        if parts <= 1:
            items.append((seg, head, c, 0, blocks, -1))
            item_costs.append(costs[u])
            continue
        per = (blocks + parts - 1) // parts
        chunks = [(lo, min(per, blocks - lo)) for lo in range(0, blocks, per)]
        merge.extend((seg, (head << 16) | c, slots_used, len(chunks)))
        for lo, count in chunks:
            items.append((seg, head, c, lo, count, slots_used))
            stage_iters = count if tiles_per_cta == MODE_TWO_TILE else (count + 1) // 2
            item_costs.append(float(stage_iters + ATTN_UNIT_OVERHEAD_BLOCKS))
            slots_used += 1
    total_tiles = len(items)
    num_clusters = shape_grid(item_costs, grid_clusters, grid_policy, arch)
    makespan, slots = lpt_makespan(item_costs, num_clusters)
    unit_words = 2 + (2 if wide else 0) + (0 if kv_split is None else 2)
    begins = [0]
    for length in lens:
        begins.append(begins[-1] + int(length))
    table: list[int] = []
    for item in slots:
        seg, head, c, kv_lo, kv_blocks, slot = items[item]
        table.extend((seg, (head << 16) | c))
        if wide:
            table.extend((begins[seg], int(lens[seg])))
        if kv_split is not None:
            table.extend(((kv_lo << 16) | kv_blocks, slot))
    return {
        "tiles_per_cta": tiles_per_cta,
        "total_tiles": total_tiles,
        "total_units": total_units,
        "num_clusters": num_clusters,
        "full_clusters": min(grid_clusters, max(total_tiles, 1)),
        "grid_policy": grid_policy,
        "makespan": makespan,
        "total_clusters": sum(
            (length + rows_per_cluster - 1) // rows_per_cluster for length in lens
        ),
        "table": table,
        "unit_words": unit_words,
        "wide": bool(wide),
        "kv_split": None if kv_split is None else int(kv_split),
        "kv_split_used": k_used,
        "split_units": sum(1 for v in split_of if v > 1),
        "merge_table": merge,
        "num_merge_units": len(merge) // ATTN_MERGE_WORDS,
        "num_partial_slots": slots_used,
        "part_rows": rows_per_cluster,
    }


@dataclass(frozen=True)
class AttentionPlan:
    """Host segment plan of the vision attention kernel (non-empty segments only).

    A *unit* is one head x one cluster tile (``2 * tiles_per_cta * 128``
    consecutive Q rows of one segment); the ``total_tiles`` LPT items (units,
    or the K/V-range parts of the split units) run on a persistent grid of
    ``num_clusters`` 2-CTA clusters.  ``unit_table`` holds ``unit_words`` int32
    per persistent-grid slot -- the segment index and ``head << 16 |
    cluster_in_segment``, then ``doc_begin, doc_len`` on the wide build and
    ``kv_block_begin << 16 | kv_blocks, partial_slot`` on the split build -- in
    slot order (slot ``k * G + i`` is the ``k``-th item of cluster ``i``).
    ``kernel_key`` names the attention build the plan runs
    (``attention:<layout>[:wide][:split]``); a plan with ``num_merge_units > 0``
    launches ``attention_merge`` after the attention kernel.
    """

    cu_seqlens: tuple[int, ...]
    num_heads: int
    num_segments: int
    tiles_per_cta: int
    ring3: bool  # SPLIT_KV rows: the shared-O ring form (``attention:ring3``) instead of ``attention:tiles1``
    wide: bool  # the UNIT_PREFETCH wide-table build (``:wide``)
    kv_split: Optional[
        int
    ]  # None = whole units; 0 = the automatic tail split (the ``:split`` build)
    kv_split_used: int
    split_units: int
    unit_words: int
    num_merge_units: int
    num_partial_slots: int
    part_rows: int
    grid_clusters: int
    grid_policy: str
    split_policy: str
    makespan: dict[int, float]
    total_units: int
    total_clusters: int
    total_tiles: int
    num_clusters: int  # the launched (lever-G shaped) cluster count
    full_clusters: int  # ``min(grid_clusters, total_tiles)``: the unshaped grid
    seg_begin: torch.Tensor
    seg_len: torch.Tensor
    unit_table: torch.Tensor
    merge_table: (
        torch.Tensor
    )  # ``[num_merge_units, 4]`` int32 rows (``[0, 0, 0, 0]`` without split units)

    @property
    def kernel_key(self) -> str:
        return attention_kernel_key(
            self.tiles_per_cta,
            self.ring3,
            wide=self.wide,
            split=self.kv_split is not None,
        )


def _table(values: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        list(values) if values else [0], dtype=torch.int32, device=device
    )


def build_attention_plan(
    cu_seqlens: Sequence[int],
    device: torch.device,
    num_heads: int = HEADS,
    *,
    grid_clusters: Optional[int] = None,
    tiles_per_cta: Optional[int] = None,
    arch: Optional[str] = None,
    grid_policy: str = GRID_DEFAULT_POLICY,
    split_policy: str = SPLIT_DEFAULT_POLICY,
    kv_split: Optional[int] = None,
    wide: Optional[bool] = None,
) -> AttentionPlan:
    """Build the attention segment plan (tables on ``device``); mirrors
    ``kimi_k3_vision_attention.build_segment_plan`` step by step.

    ``grid_clusters`` defaults to :func:`attention_grid_clusters` of ``device``
    (which must then be a CUDA device).  ``arch`` selects the split cost model of
    the layout rule (:func:`split_cost_model`; the round-3 model when ``None``),
    the SPLIT_KV kernel form (``ring3`` per :func:`ring3_selected`; the plain form
    when ``None``), the wide-table rows (:func:`unit_prefetch_selected`; never
    when ``None``) and the lever-G shaping (:func:`shape_grid`).  ``split_policy``
    ``"auto"`` applies the split-aware layout rule + automatic tail split
    (:func:`select_layout_and_split`); a request that splits nothing falls back
    to whole units.  ``tiles_per_cta`` forces a unit layout (2 = two-tile, 1 =
    SPLIT_KV; the split rule is then not applied, as in Cake), ``kv_split`` /
    ``wide`` pin the builds (tests).
    """
    cu = tuple(int(v) for v in cu_seqlens)
    if (
        len(cu) < 2
        or cu[0] != 0
        or any(b < a for a, b in zip(cu, cu[1:], strict=False))
    ):
        raise ValueError("cu_seqlens must start at zero and be non-decreasing")
    num_heads = int(num_heads)
    if not 0 < num_heads < ATTN_MAX_HEADS:
        raise ValueError(f"num_heads must be in [1, {ATTN_MAX_HEADS}), got {num_heads}")
    if grid_policy not in GRID_POLICIES:
        raise ValueError(
            f"grid policy must be one of {GRID_POLICIES}, got {grid_policy!r}"
        )
    if split_policy not in SPLIT_POLICIES:
        raise ValueError(
            f"split policy must be one of {SPLIT_POLICIES}, got {split_policy!r}"
        )
    if grid_clusters is None:
        grid_clusters = attention_grid_clusters(device)
    grid_clusters = max(1, int(grid_clusters))
    segments = [(a, b - a) for a, b in zip(cu, cu[1:], strict=False) if b > a]
    begins = [a for a, _ in segments]
    lens = [length for _, length in segments]
    wd = unit_prefetch_selected(arch, cu[-1]) if wide is None else bool(wide)
    kvs = None if kv_split is None else int(kv_split)
    if split_policy == "auto" and tiles_per_cta is None and kv_split is None:
        selection = select_layout_and_split(lens, num_heads, grid_clusters, arch)
        kvs = selection["kv_split"]
    else:
        selection = select_tiles_per_cta(lens, num_heads, grid_clusters, arch)
    mode = (
        int(tiles_per_cta)
        if tiles_per_cta is not None
        else int(selection["tiles_per_cta"])
    )
    if mode not in (MODE_TWO_TILE, MODE_SPLIT_KV):
        raise ValueError(
            f"tiles_per_cta must be {MODE_TWO_TILE} or {MODE_SPLIT_KV}, got {mode}"
        )
    layout = build_unit_table(
        lens,
        num_heads,
        grid_clusters,
        mode,
        kv_split=kvs,
        wide=wd,
        grid_policy=grid_policy,
        arch=arch,
    )
    if kvs == 0 and int(layout["split_units"]) == 0:
        # The automatic rule split nothing: the production (or wide) build on the plain table -- the partial-output
        # build is never launched idle (Cake ``build_segment_plan``).
        kvs = None
        layout = build_unit_table(
            lens,
            num_heads,
            grid_clusters,
            mode,
            kv_split=None,
            wide=wd,
            grid_policy=grid_policy,
            arch=arch,
        )
    return AttentionPlan(
        cu_seqlens=cu,
        num_heads=num_heads,
        num_segments=len(segments),
        tiles_per_cta=mode,
        ring3=bool(
            mode == MODE_SPLIT_KV and arch is not None and ring3_selected(arch, lens)
        ),
        wide=wd,
        kv_split=kvs,
        kv_split_used=int(layout["kv_split_used"]),
        split_units=int(layout["split_units"]),
        unit_words=int(layout["unit_words"]),
        num_merge_units=int(layout["num_merge_units"]),
        num_partial_slots=int(layout["num_partial_slots"]),
        part_rows=int(layout["part_rows"]),
        grid_clusters=grid_clusters,
        grid_policy=grid_policy,
        split_policy=split_policy,
        makespan=dict(selection["makespan"]),
        total_units=int(layout["total_units"]),
        total_clusters=int(layout["total_clusters"]),
        total_tiles=int(layout["total_tiles"]),
        num_clusters=int(layout["num_clusters"]),
        full_clusters=int(layout["full_clusters"]),
        seg_begin=_table(begins, device),
        seg_len=_table(lens, device),
        unit_table=torch.tensor(
            layout["table"] or [0, 0], dtype=torch.int32, device=device
        ),
        merge_table=torch.tensor(
            layout["merge_table"] or [0] * ATTN_MERGE_WORDS,
            dtype=torch.int32,
            device=device,
        ),
    )


# ---------------------------------------------------------------------------
# GEMM tile configurations and launch geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileConfig:
    """One instance of the ``kimi_k3_vision_gemm`` template (its ``TileConfig``)."""

    name: str
    cta_group: int
    acc_n: int
    num_stages: int
    group_m: int
    ksplit: int = 1  # > 1: split-K cluster of single-CTA tiles (non-persistent)
    epi_warps: int = (
        4  # 4 (one warp per TMEM lane quadrant) or 8 (two per quadrant, column halves)
    )
    tail: bool = (
        False  # tail split-K paths compiled in (opt-in in Cake, never selected here)
    )
    packed: bool = False  # packed bf16x2 residual epilogue (EPI_PACKED)
    prefetch: bool = False  # residual / norm-weight rows loaded before the mainloop wait (EPI_PREFETCH)
    rope: bool = False  # RoPE cos/sin from the packed f16x2 table CS (ROPE_PACKED; norm_qkv_rope only)
    smem_res: bool = False  # residual rows staged in SMEM before the mainloop wait (EPI_SMEM_RES; opt-in in Cake)
    tma_epi: bool = False  # TMA-loaded residual tile, in-place BF16 output + xw tiles TMA-stored (EPI_TMA)
    tma_onebuf: bool = False  # tma_epi with one staging tile (EPI_TMA_ONEBUF)
    mcast: int = (
        1  # > 1: cluster of single-CTA tiles sharing the A tile (MCAST; opt-in in Cake)
    )
    sk: bool = False  # stream-K tail: fractional K-ranges over the tail tiles, static ownership, staged fix-up (SK, round 4)
    half: bool = False  # half-N tail: the tiles past the last full round run as half-width items (HALF_TAIL, round 6)

    @property
    def b_rows(self) -> int:
        return self.acc_n // self.cta_group

    @property
    def stage_bytes(self) -> int:
        return GEMM_BLOCK_M * GEMM_BLOCK_K * 2 + self.b_rows * GEMM_BLOCK_K * 2

    @property
    def cluster_x(self) -> int:
        return self.cta_group * self.ksplit * self.mcast


def _cfg(
    name: str, cta_group: int, acc_n: int, num_stages: int, **kw: Any
) -> TileConfig:
    return TileConfig(name, cta_group, acc_n, num_stages, 32, **kw)


# The Cake catalogue (``kimi_k3_vision_gemm.TILE_CONFIGS``): pair tile 256 x 256
# (``l``), single-CTA 128 x 128 / 128 x 64 (``s`` / ``xs``) and their split-K
# clusters, the wave-balance / epilogue-width tiles (``m`` 256 x 128 pair,
# ``w``, ``l_g8``, ``l_e8`` / ``s_e8`` eight epilogue warps) and the twins:
# ``*_t`` tail split-K (opt-in in Cake; never selected by the production
# policy), ``*_p`` / ``*_pf`` packed residual epilogue (+ pre-mainloop
# residual / norm-weight prefetch), ``*_cs`` packed f16x2 RoPE table,
# ``l_sk`` the pieces-first stream-K twin of the pair tile (round 4), and
# (round 5) ``m_sk`` the stream-K twin of the 256 x 128 pair tile for the FC1
# census points and ``m_e8`` / ``m_e8_cs`` the eight-warp 256 x 128 pair
# tile of the K = 1024 norm GEMMs where the 256-wide tile's round count is
# worst.
TILE_CONFIGS: dict[str, TileConfig] = {
    c.name: c
    for c in (
        _cfg("l", 2, 256, 7),
        _cfg("s", 1, 128, 7),
        _cfg("xs", 1, 64, 9),
        _cfg("s_k2", 1, 128, 7, ksplit=2),
        _cfg("s_k4", 1, 128, 7, ksplit=4),
        _cfg("xs_k2", 1, 64, 9, ksplit=2),
        _cfg("xs_k4", 1, 64, 9, ksplit=4),
        _cfg("m", 2, 128, 9),
        _cfg("w", 1, 256, 4),
        TileConfig("l_g8", cta_group=2, acc_n=256, num_stages=7, group_m=8),
        _cfg("l_e8", 2, 256, 7, epi_warps=8),
        _cfg("s_e8", 1, 128, 7, epi_warps=8),
        _cfg("l_t", 2, 256, 7, tail=True),
        _cfg("l_e8_t", 2, 256, 7, epi_warps=8, tail=True),
        _cfg("m_t", 2, 128, 9, tail=True),
        _cfg("s_t", 1, 128, 7, tail=True),
        _cfg("xs_t", 1, 64, 9, tail=True),
        _cfg("s_e8_t", 1, 128, 7, epi_warps=8, tail=True),
        _cfg("m_p", 2, 128, 9, packed=True),
        _cfg("l_e8_p", 2, 256, 7, epi_warps=8, packed=True),
        _cfg("s_e8_p", 1, 128, 7, epi_warps=8, packed=True),
        _cfg("xs_p", 1, 64, 9, packed=True),
        _cfg("xs_k4_p", 1, 64, 9, ksplit=4, packed=True),
        _cfg("m_pf", 2, 128, 9, packed=True, prefetch=True),
        _cfg("l_e8_pf", 2, 256, 7, epi_warps=8, packed=True, prefetch=True),
        _cfg("s_e8_pf", 1, 128, 7, epi_warps=8, packed=True, prefetch=True),
        _cfg("xs_pf", 1, 64, 9, packed=True, prefetch=True),
        _cfg("xs_k4_pf", 1, 64, 9, ksplit=4, packed=True, prefetch=True),
        _cfg("l_e8_cs", 2, 256, 7, epi_warps=8, rope=True),
        _cfg("s_cs", 1, 128, 7, rope=True),
        _cfg("xs_cs", 1, 64, 9, rope=True),
        _cfg("xs_cs_pf", 1, 64, 9, rope=True, prefetch=True),
        # Round 3: the one-buffer TMA-epilogue pair tile of the K = 1536 residual out-proj at M > 6144.
        _cfg("m_tma1", 2, 128, 8, packed=True, tma_epi=True, tma_onebuf=True),
        # Round 4: the pieces-first stream-K twin of the 256 x 256 pair tile (projector GEMMs, census window).
        _cfg("l_sk", 2, 256, 7, sk=True),
        # Round 5 (Cake ``CFG_M_SK``): the stream-K twin of the 256 x 128 pair tile (residual_fc1 at M <= MID_M_LIMIT
        # where the 256 x 256 tile leaves a tiny extra round).
        _cfg("m_sk", 2, 128, 9, sk=True),
        # Round 5 (Cake ``CFG_M_E8`` / ``CFG_M_E8_CS``): the eight-warp 256 x 128 pair tile of norm_gelu /
        # norm_qkv_rope for 1656 < M <= M_E8_MAX_M (bitwise identical to l_e8; paired > 1.00 at 2552).
        _cfg("m_e8", 2, 128, 9, epi_warps=8),
        _cfg("m_e8_cs", 2, 128, 9, epi_warps=8, rope=True),
        # Round 6 (Cake ``CFG_*_H``, lever G1 / G2): the half-N tail twins of the plain persistent tiles -- same
        # mainloop, the tiles past the last full round run as half-width (acc_n / 2) items when they fit one
        # half-round (``half_tail_split``); bitwise identical per output element to the base tile.
        _cfg(
            "s_e8_pf_h", 1, 128, 7, epi_warps=8, packed=True, prefetch=True, half=True
        ),
        _cfg("m_e8_h", 2, 128, 9, epi_warps=8, half=True),
        _cfg("m_e8_cs_h", 2, 128, 9, epi_warps=8, rope=True, half=True),
        _cfg("xs_h", 1, 64, 9, half=True),
        _cfg("xs_cs_pf_h", 1, 64, 9, rope=True, prefetch=True, half=True),
        _cfg("s_h", 1, 128, 7, half=True),
        _cfg("s_cs_h", 1, 128, 7, rope=True, half=True),
        _cfg("m_p_h", 2, 128, 9, packed=True, half=True),
        _cfg("l_e8_h", 2, 256, 7, epi_warps=8, half=True),
        _cfg("l_e8_cs_h", 2, 256, 7, epi_warps=8, rope=True, half=True),
        _cfg(
            "l_e8_pf_h", 2, 256, 7, epi_warps=8, packed=True, prefetch=True, half=True
        ),
        _cfg("l_h", 2, 256, 7, half=True),
        _cfg(
            "m_tma1_h", 2, 128, 8, packed=True, tma_epi=True, tma_onebuf=True, half=True
        ),
    )
}
# Round 6 (Cake ``_HALF_TWIN`` / ``HALF_ROUND_MARGIN`` / ``half_tail_split`` / ``_half_twin``, ``_HALF_POLICY`` default
# "auto" = env ``KIMI_K3_VISION_GEMM_HALF``): the outermost routing step maps a plain persistent tile to its half-N twin
# when the census tail (tiles past the last full round on ``min(tiles, SM // cluster_x)`` clusters) fits one half-round
# with margin: ``0 < 2 * tail <= clusters - HALF_ROUND_MARGIN``.  ``gemm_pos`` (row-pair maps) and the stream-K / tail /
# split-K tiles have no twin.  32 contract points move (B200 11 / 12 census points positive; B300 confirmed).
HALF_TWIN = {
    "s_e8_pf": "s_e8_pf_h",
    "m_e8": "m_e8_h",
    "m_e8_cs": "m_e8_cs_h",
    "xs": "xs_h",
    "xs_cs_pf": "xs_cs_pf_h",
    "s": "s_h",
    "s_cs": "s_cs_h",
    "m_p": "m_p_h",
    "l_e8": "l_e8_h",
    "l_e8_cs": "l_e8_cs_h",
    "l_e8_pf": "l_e8_pf_h",
    "l": "l_h",
    "m_tma1": "m_tma1_h",
}
HALF_ROUND_MARGIN = 4
# Round-6 routing limits from the paired B300 / B200 A/B (Cake ``HALF_MAX_ROUNDS`` / ``HALF_ROUTE_EXCLUDE``): no half twin past
# 24 persistent rounds (the half-round saving is inside the twin's full-item overhead: out-proj 66564) and never at the
# ``xs`` tile: gelu_erf 638 is a tie / loss on both arches, and the norm_gelu / norm_qkv_rope 576 twins, a per-kernel win,
# cost the img_336 tower row ~1.6 % inside the PDL chain on B300 (the tower row is the acceptance unit).
HALF_MAX_ROUNDS = 24
HALF_ROUTE_EXCLUDE = frozenset(
    {("gelu_erf", "xs"), ("norm_gelu", "xs"), ("norm_qkv_rope", "xs_cs_pf")}
)
# Per-config minimum round count (Cake ``HALF_MIN_ROUNDS``): the TMA-epilogue twin m_tma1_h at the out-proj 8192 point
# (4 rounds, tail 34 / 74) wins per kernel but costs the batch8_448 tower row 0.13 % on B200; 12288 (6 rounds) and the
# 16508 .. 43056 points keep it.
HALF_MIN_ROUNDS = {"m_tma1": 6}


def half_tail_split(cluster_tiles: int, clusters: int) -> int:
    """Tail tiles of a persistent grid that run as half-N items (0 when the half round would be (nearly) full);
    mirrors ``kimi_k3_vision_gemm.half_tail_split``."""
    tail = cluster_tiles % clusters if cluster_tiles > clusters else 0
    return tail if 0 < 2 * tail <= clusters - HALF_ROUND_MARGIN else 0


def _half_twin(name: str, n_total: int, M: int, sm_count: int, variant: str) -> str:
    """The half-N twin of ``name`` where the census tail fits one half-round, the grid runs at most
    ``HALF_MAX_ROUNDS`` rounds and ``(variant, tile)`` is not excluded (Cake ``_half_twin``)."""
    if name not in HALF_TWIN or (variant, name) in HALF_ROUTE_EXCLUDE:
        return name
    cfg = TILE_CONFIGS[name]
    m_tiles = (M + GEMM_BLOCK_M - 1) // GEMM_BLOCK_M
    m_tiles += m_tiles % cfg.cta_group
    tiles = (m_tiles // cfg.cta_group) * (n_total // cfg.acc_n)
    clusters = min(tiles, int(sm_count) // cfg.cluster_x)
    rounds = -(-tiles // clusters)
    if rounds > HALF_MAX_ROUNDS or rounds < HALF_MIN_ROUNDS.get(name, 0):
        return name
    return HALF_TWIN[name] if half_tail_split(tiles, clusters) else name


# norm_qkv_rope: the packed f16x2 cos/sin twin of each base tile (Cake
# ``_ROPE_TWIN``; the pre-mainloop table prefetch only on the one-drain-group
# ``xs`` tile) and the FP32-table config the launcher falls back to when no
# table is passed (``_ROPE_BASE``).  The host always passes ``rope_cs``.
ROPE_TWIN = {"l_e8": "l_e8_cs", "s": "s_cs", "xs": "xs_cs_pf", "m_e8": "m_e8_cs"}
ROPE_BASE = {twin: base for base, twin in ROPE_TWIN.items()}
ROPE_BASE["xs_cs"] = "xs"
# residual / pos forms: the packed epilogue twin everywhere, with the residual
# prefetch on the one-drain-group configs and on ``l_e8`` (Cake ``_PACKED_TWIN``).
PACKED_TWIN = {
    "m": "m_p",
    "l_e8": "l_e8_pf",
    "s_e8": "s_e8_pf",
    "xs": "xs_pf",
    "xs_k4": "xs_k4_pf",
}
# Round 3 (G2): the TMA-epilogue twin of the 256 x 128 pair tile, taken by the K = 1536 residual
# out-proj forms only (Cake ``_TMA_EPI_TWIN`` / ``_TMA_EPI_POLICY`` auto; ties ``m_p`` on the K = 4096 FC1).
TMA_EPI_TWIN = {"m": "m_tma1"}
# Round 4 (CAKE-722 lever C): the pieces-first stream-K twin of the 256 x 256 pair tile (Cake ``_SK_TWIN`` /
# ``_SK_POLICY`` auto), taken by the projector GEMMs ``gelu_erf`` / ``rmsnorm`` inside the census window of
# ``_sk_twin``.  The full rounds stay data-parallel; the tail tiles' K-steps are cut into ranges of
# ``tail_split`` (= q) k-steps, one per cluster; the non-owning pieces dump FP32 partials to ``WS`` and the
# owner sums them behind the arrival counters ``FLAGS`` (self-reset by the last arriver).
SK_TWIN = {"l": "l_sk"}
SK_MAX_CONTRIB = 4  # largest number of contributors per tail tile the kernel admits (ceil(K / q) + 1)
SK_TAIL_FRAC_MAX = (
    0.30  # stream-K twin only when the tail holds <= 30 % of the clusters
)
SK_MIN_SAVING = 0.065  # ... and removing that partial round is worth >= 6.5 % of the mainloop rounds
# Round 5 (CAKE-749; Cake ``SK_MIN_SAVING_ANY`` / ``SK_TAIL_FRAC_ANY``): a removed partial round worth >= 9 % of the
# rounds wins whatever the tail fraction up to 60 % (gelu_erf 2691 tail 28 / 74, rmsnorm 2835 tail 40 / 74 paired > 1.00 on
# both arches); the round-4 window above stays as the second admission path.
SK_MIN_SAVING_ANY = 0.09
SK_TAIL_FRAC_ANY = 0.60
DEFAULT_SM_COUNT = (
    148  # B200 / B300: the census device of the tile rule when no plan supplies one
)
# Round 5 (Cake ``M_E8_MAX_M`` / ``QKV_M_E8_MIN_M``, ``_m_e8_twin``, ``_M_E8_POLICY`` auto): the ``l_e8`` branch of the
# K = 1024 norm GEMMs (norm_gelu, norm_qkv_rope) takes the 256 x 128 eight-warp pair tile up to M_E8_MAX_M rows,
# norm_qkv_rope only above QKV_M_E8_MIN_M (2552: m_e8 1.074-1.098x, m_e8_cs 1.026-1.033x; 4784 loses; 1656 / 4144 unmeasured
# stay on l_e8).
M_E8_MAX_M = 4096
QKV_M_E8_MIN_M = 2304
# Round 5 (Cake ``FC1_SK_TINY_TAIL`` / ``FC1_SK_L_MIN_ROUNDS`` / ``FC1_SK_L_MAX_M`` / ``FC1_SK_TWIN_SMALL`` /
# ``FC1_SK_TWIN_LARGE``, ``_fc1_sk_twin``, ``_FC1_SK_POLICY`` auto): residual_fc1 (N = 1024, K = 4096) above
# FC1_S_E8_LIMIT takes a pair stream-K twin at its census points -- ``m_sk`` when M <= MID_M_LIMIT and the 256 x 256 tile
# leaves 0 < tail <= FC1_SK_TINY_TAIL cluster tiles (4784: 1.18-1.20x), ``l_sk`` when M > MID_M_LIMIT, the ideal round
# count is >= FC1_SK_L_MIN_ROUNDS and M <= FC1_SK_L_MAX_M (16576 / 19136 / 43056 / 66564 / 105984 paired > 1.00 in every
# pass; 153088 unmeasured -> m_p).  The census is on the 256 x 256 tile (``tiles_l``) whatever the routed tile.
FC1_SK_TINY_TAIL = 4
FC1_SK_L_MIN_ROUNDS = 3.4
FC1_SK_L_MAX_M = 105984
FC1_SK_TWIN_SMALL = "m_sk"
FC1_SK_TWIN_LARGE = "l_sk"
# Round 5 (Cake ``PDL_EARLY_WINDOW`` / ``PDL_EARLY_DEFAULT_WINDOW`` / ``_pdl_early_on``; policy ``_PDL_EARLY_POLICY``
# default auto): the PDL_EARLY binary of a form is launched only at the M where its paired effect was > 1.00 in every
# pass on both arches (route-level tower A/B for T <= 10764, per-form 4-kernel chains at 10764 / 16576 / 19136 / 43056,
# merger chains at merged N).  Inclusive (lo, hi) ranges per variant on the GEMM's own M (tokens T for the per-layer
# forms, merged N for gelu_erf / rmsnorm); variants without an entry (the last layer's plain ``residual_fc1``, ``pos_sqxw``)
# use the default window.  The stream-K / tail / multicast / pos tiles never take the early binary
# (:func:`pdl_early_selected`; Cake ``gemm_ir``: ``PDL_EARLY = pdl_early and not (cfg.tail or cfg.sk or cfg.mcast > 1
# or POS_SPLIT)``).
PDL_EARLY_WINDOW: dict[str, tuple[tuple[int, int], ...]] = {
    "norm_qkv_rope": ((1, 8192),),
    "norm_gelu": ((1, 19136),),
    "residual_wo_sqxw": ((1, 43056),),
    "residual_fc1_sqxw": ((1, 43056),),
    "gelu_erf": ((144, 638), (1196, 1196)),
    "rmsnorm": ((144, 638), (1196, 1196)),
}
PDL_EARLY_DEFAULT_WINDOW: tuple[tuple[int, int], ...] = ((1, 43056),)
for _name in (
    *ROPE_TWIN,
    *ROPE_TWIN.values(),
    *PACKED_TWIN,
    *PACKED_TWIN.values(),
    *TMA_EPI_TWIN.values(),
    *SK_TWIN,
    *SK_TWIN.values(),
    *HALF_TWIN,
    *HALF_TWIN.values(),
):
    assert _name in TILE_CONFIGS, _name
for _name, _twin in HALF_TWIN.items():
    assert TILE_CONFIGS[_twin].half and not TILE_CONFIGS[_name].half, _name
    assert not (
        TILE_CONFIGS[_twin].sk
        or TILE_CONFIGS[_twin].tail
        or TILE_CONFIGS[_twin].ksplit > 1
    ), _twin

# variant -> (N, K, pos-split pixel-row maps); ``_sq`` / ``_sqxw`` are the
# handoff forms of the residual-stream producers (same GEMM, extra epilogue
# stores), with the base variant's tile rule.
GEMM_VARIANTS: dict[str, tuple[int, int, bool]] = {
    "pos": (HIDDEN, PATCH_DIM_PAD, True),
    "norm_qkv_rope": (QKV_N, HIDDEN, False),
    "residual_wo": (HIDDEN, QKV_HIDDEN, False),
    "residual_fc1": (HIDDEN, FFN, False),
    "norm_gelu": (FFN, HIDDEN, False),
    "gelu_erf": (MERGED_DIM, MERGED_DIM, False),
    "rmsnorm": (TEXT_HIDDEN, MERGED_DIM, False),
}
for _base in ("pos", "residual_wo", "residual_fc1"):
    GEMM_VARIANTS[_base + "_sq"] = GEMM_VARIANTS[_base]
    GEMM_VARIANTS[_base + "_sqxw"] = GEMM_VARIANTS[_base]
# Production tile selection thresholds (kimi_k3_vision_gemm.select_tile_config).
TINY_M_LIMIT = (
    256  # M <= this -> 128 x 64 tiles for every shape but the 7168-wide projector GEMM
)
SMALL_M_LIMIT = (
    1024  # M <= this -> single-CTA tiles (128 x 64 for N = 1024, 128 x 128 otherwise)
)
MID_M_LIMIT = 8192  # residual_fc1: 256 x 256 (8 epilogue warps) up to here, 256 x 128 pair tile above
# Round-3 per-form tile boundaries (Cake ``kimi_k3_vision_gemm`` ``_TILE_BOUNDARY_POLICY`` = "r3").
WO_S_E8_LIMIT = 6144  # residual_wo*: single-CTA 128 x 128 eight-warp tile (s_e8_pf) for M in (SMALL_M_LIMIT, this]
FC1_S_E8_LIMIT = (
    2304  # residual_fc1*: s_e8_pf for M in (SMALL_M_LIMIT, this], l_e8_pf above
)
QKV_XS_LIMIT = 768  # norm_qkv_rope: 128 x 64 (xs_cs_pf) up to here
QKV_S_LIMIT = (
    1536  # norm_qkv_rope: 128 x 128 (s_cs) for M in (QKV_XS_LIMIT, this], l_e8_cs above
)
NG_XS_LOW, NG_XS_HIGH = (
    512,
    768,
)  # norm_gelu / gelu_erf: 128 x 64 (xs) for M in (512, 768] (and M <= TINY_M_LIMIT)
NG_S_LIMIT = 1656  # norm_gelu / gelu_erf: 128 x 128 (s) up to here, the pair tile above


def _rope_twin(name: str, n_total: int) -> str:
    """norm_qkv_rope (N = 4608) takes the packed f16x2 cos/sin twin (Cake ``_ROPE_POLICY`` auto)."""
    return ROPE_TWIN.get(name, name) if n_total == QKV_N else name


def _packed_twin(name: str, tma_ok: bool = True) -> str:
    """Residual / pos forms: the packed bf16x2 epilogue twin (Cake ``_PACKED_POLICY`` auto), or the
    TMA-epilogue twin where the form allows it (Cake ``_TMA_EPI_POLICY`` auto, K = 1536 out-proj only)."""
    if tma_ok and name in TMA_EPI_TWIN:
        return TMA_EPI_TWIN[name]
    return PACKED_TWIN.get(name, name)


def _sk_range(tail_tiles: int, clusters: int, k_iters: int) -> int:
    """Stream-K k-steps per cluster over the tail (Cake ``_sk_range``): ceil(tail x K / clusters), raised to
    ceil(K / (SK_MAX_CONTRIB - 1)) so that no tile has more than SK_MAX_CONTRIB contributors (clusters past
    the tail's end get empty pieces); 0 = no tail."""
    if tail_tiles <= 0:
        return 0
    return max(
        -(-tail_tiles * k_iters // clusters), -(-k_iters // (SK_MAX_CONTRIB - 1))
    )


def _sk_twin(name: str, n_total: int, M: int, sm_count: int) -> str:
    """The stream-K twin of a pair tile when the census for ``(N, M)`` on ``sm_count`` SMs leaves a small
    tail whose removal is worth the fix-up (Cake ``_sk_twin``): ``tail / clusters <= SK_TAIL_FRAC_MAX`` and
    ``(1 - tail / clusters) / rounds >= SK_MIN_SAVING``; tiles without a twin and M <= ``SMALL_M_LIMIT``
    are returned unchanged."""
    if name not in SK_TWIN or M <= SMALL_M_LIMIT:
        return name
    cfg = TILE_CONFIGS[name]
    m_tiles = (M + GEMM_BLOCK_M - 1) // GEMM_BLOCK_M
    m_tiles += m_tiles % cfg.cta_group
    tiles = (m_tiles // cfg.cta_group) * (n_total // cfg.acc_n)
    clusters = min(tiles, int(sm_count) // cfg.cta_group)
    tail = tiles % clusters
    if tail == 0:
        return name
    frac = tail / clusters
    rounds = -(-tiles // clusters)
    saving = (1.0 - frac) / rounds
    # Round 5: the wide window (Cake ``_sk_twin`` auto branch) OR the round-4 window.
    if (saving >= SK_MIN_SAVING_ANY and frac <= SK_TAIL_FRAC_ANY) or (
        frac <= SK_TAIL_FRAC_MAX and saving >= SK_MIN_SAVING
    ):
        return SK_TWIN[name]
    return name


def _m_e8_twin(name: str, n_total: int, M: int) -> str:
    """The ``l_e8`` branch of the K = 1024 norm GEMMs takes ``m_e8`` up to ``M_E8_MAX_M`` rows, norm_qkv_rope only
    above ``QKV_M_E8_MIN_M`` (Cake ``_m_e8_twin``; the RoPE twin ``m_e8_cs`` follows through :func:`_rope_twin`)."""
    if name != "l_e8" or M > M_E8_MAX_M:
        return name
    if n_total == QKV_N and M <= QKV_M_E8_MIN_M:
        return name
    return "m_e8"


def _fc1_sk_twin(name: str, M: int, sm_count: int) -> str:
    """residual_fc1 (N = 1024, K = 4096) above ``FC1_S_E8_LIMIT``: the pair stream-K twin at the census points
    (Cake ``_fc1_sk_twin``): ``m_sk`` when M <= MID_M_LIMIT and the 256 x 256 tile census leaves ``0 < tail <=
    FC1_SK_TINY_TAIL`` cluster tiles, ``l_sk`` when M > MID_M_LIMIT, ``tiles / clusters >= FC1_SK_L_MIN_ROUNDS`` and
    ``M <= FC1_SK_L_MAX_M``; otherwise ``name``."""
    if M <= FC1_S_E8_LIMIT:
        return name
    clusters = int(sm_count) // 2
    m_tiles = (M + GEMM_BLOCK_M - 1) // GEMM_BLOCK_M
    m_tiles += m_tiles % 2
    tiles_l = (m_tiles // 2) * (HIDDEN // 256)
    tail_l = tiles_l % clusters if tiles_l > clusters else 0
    if M <= MID_M_LIMIT:
        return FC1_SK_TWIN_SMALL if 0 < tail_l <= FC1_SK_TINY_TAIL else name
    if tiles_l / clusters >= FC1_SK_L_MIN_ROUNDS and M <= FC1_SK_L_MAX_M:
        return FC1_SK_TWIN_LARGE
    return name


def select_tile_config(
    variant: str, M: int, sm_count: int = DEFAULT_SM_COUNT
) -> TileConfig:
    """Production tile config for ``(variant, M)``; mirrors ``kimi_k3_vision_gemm.select_tile_config``: the
    round-3 .. round-5 routing (:func:`_select_tile_config_base`) and, as the outermost step (round 6), the
    half-N tail twin of the non-pos forms where the census tail fits one half-round (:func:`_half_twin`)."""
    cfg = _select_tile_config_base(variant, M, sm_count)
    n_total, _k_total, pos_split = GEMM_VARIANTS[variant]
    if pos_split:
        return cfg  # gemm_pos streams row pairs: no half-N tail
    return TILE_CONFIGS[_half_twin(cfg.name, n_total, M, sm_count, variant)]


def _select_tile_config_base(
    variant: str, M: int, sm_count: int = DEFAULT_SM_COUNT
) -> TileConfig:
    """Round-3 .. round-5 tile config for ``(variant, M)``; mirrors ``kimi_k3_vision_gemm._select_tile_config_base``
    with its production policies (packed residual twins on, RoPE table twins on, TMA epilogue twin
    for the out-proj on, round-3 tile boundaries, stream-K twin inside its census window on
    ``sm_count`` SMs, tail / SMEM-staged / multicast / L2-prefetch twins off).

    Small M is bound by the per-CTA operand stream, so the smallest tile wins:
    128 x 64 up to ``SMALL_M_LIMIT`` for the N = 1024 shapes and up to
    ``TINY_M_LIMIT`` for the wide ones (the 7168-wide projector GEMM prefers
    128 x 128); split-K only for the K = 4096 residual GEMM at M <= 256.
    Round 3 re-measured every form's boundaries: the single-CTA 128 x 128
    eight-warp tile ``s_e8_pf`` serves the residual GEMMs while the pair grid
    underfills the machine (out-proj to ``WO_S_E8_LIMIT``, FC1 to
    ``FC1_S_E8_LIMIT``), the out-proj pair tile is the TMA-epilogue twin
    ``m_tma1``, and the K = 1024 norm GEMMs / projector GEMMs keep 128 x 64 to
    768 rows and 128 x 128 to 1536 / 1656 rows before the 256-wide pair tiles.
    Round 4 added the stream-K twin ``l_sk`` of the projector GEMMs' pair tile
    where the tile census leaves a small tail (:func:`_sk_twin`).  Round 5
    widened that window (:data:`SK_MIN_SAVING_ANY`), added the FC1 stream-K
    twins ``m_sk`` / ``l_sk`` at the FC1 census points (:func:`_fc1_sk_twin`)
    and the eight-warp 256 x 128 pair tile ``m_e8`` / ``m_e8_cs`` of the norm
    GEMMs up to ``M_E8_MAX_M`` rows (:func:`_m_e8_twin`).
    """
    n_total, k_total, pos_split = GEMM_VARIANTS[variant]
    cfg = TILE_CONFIGS
    if pos_split:
        return cfg[_packed_twin("xs" if M <= SMALL_M_LIMIT else "s_e8", tma_ok=False)]
    if n_total == HIDDEN:  # residual_wo (K = 1536), residual_fc1 (K = 4096)
        if M <= TINY_M_LIMIT and k_total == FFN:
            return cfg[_packed_twin("xs_k4")]
        if M <= SMALL_M_LIMIT:
            return cfg[_packed_twin("xs")]
        s_e8_limit = WO_S_E8_LIMIT if k_total == QKV_HIDDEN else FC1_S_E8_LIMIT
        if s_e8_limit >= M:
            return cfg[_packed_twin("s_e8", tma_ok=False)]
        if k_total == FFN:
            # fc1: l_e8_pf up to MID_M_LIMIT, m_p above, or the stream-K twin at the census points (round 5).
            return cfg[
                _fc1_sk_twin(
                    _packed_twin("l_e8" if M <= MID_M_LIMIT else "m", tma_ok=False),
                    M,
                    sm_count,
                )
            ]
        return cfg[_packed_twin("m", tma_ok=True)]
    if M <= TINY_M_LIMIT:
        return cfg[_rope_twin("s" if n_total == TEXT_HIDDEN else "xs", n_total)]
    if n_total == QKV_N and M <= QKV_XS_LIMIT:
        return cfg[_rope_twin("xs", n_total)]
    if n_total == FFN and NG_XS_LOW < M <= NG_XS_HIGH:
        return cfg["xs"]
    if M <= SMALL_M_LIMIT:
        return cfg[_rope_twin("s", n_total)]
    if n_total == QKV_N and M <= QKV_S_LIMIT:
        return cfg[_rope_twin("s", n_total)]
    if n_total == FFN and M <= NG_S_LIMIT:
        return cfg["s"]
    # The K = 1024 norm GEMMs (statistics handoff on the critical path) take the
    # eight-warp epilogue (``m_e8`` inside its window, round 5); the projector
    # GEMMs the four-warp pair tile (stream-K twin inside the census window).
    wide = _m_e8_twin("l_e8", n_total, M) if k_total == HIDDEN else "l"
    return cfg[_rope_twin(_sk_twin(wide, n_total, M, sm_count), n_total)]


def launch_tile_config(
    variant: str, cfg: TileConfig, *, rope_table: bool = True
) -> TileConfig:
    """The physical config the launcher runs (``kimi_k3_vision_gemm._launch_gemm``): a ``*_cs``
    tile without a packed table falls back to its FP32-table base (the host always passes the
    table); ``pos`` requires single-CTA tiles and a K split of 10 K-steps."""
    if cfg.rope and not rope_table:
        cfg = TILE_CONFIGS[ROPE_BASE[cfg.name]]
    if GEMM_VARIANTS[variant][2]:
        if cfg.cta_group != 1:
            cfg = TILE_CONFIGS["s"]
        if (PATCH_DIM_PAD // GEMM_BLOCK_K) % cfg.ksplit:
            cfg = TILE_CONFIGS["s_k2"] if cfg.acc_n == 128 else TILE_CONFIGS["xs_k2"]
    return cfg


def pdl_early_on(variant: str, M: int) -> bool:
    """Whether ``M`` lies inside the PDL_EARLY census window of ``variant`` (Cake ``_pdl_early_on`` with its
    production policy ``auto``; variants without an entry use ``PDL_EARLY_DEFAULT_WINDOW``)."""
    return any(
        lo <= int(M) <= hi
        for lo, hi in PDL_EARLY_WINDOW.get(variant, PDL_EARLY_DEFAULT_WINDOW)
    )


def pdl_early_selected(variant: str, cfg: TileConfig, M: int) -> bool:
    """Whether the launch of ``variant`` on ``cfg`` at ``M`` runs the PDL_EARLY binary: inside the window AND
    on a plain persistent tile -- the stream-K twins (``sk``), the tail split-K twins, the A-multicast tiles and
    the ``pos`` pixel-row-map forms launch the production PDL binary whatever the window says (Cake ``gemm_ir``:
    ``PDL_EARLY = pdl_early and not (cfg.tail or cfg.sk or cfg.mcast > 1 or POS_SPLIT)``)."""
    if not pdl_early_on(variant, M):
        return False
    return not (cfg.tail or cfg.sk or cfg.mcast > 1 or GEMM_VARIANTS[variant][2])


def gemm_stage_key(variant: str, M: int, sm_count: int = DEFAULT_SM_COUNT) -> str:
    """The logical kernel key the Cake launcher resolves for ``(variant, M)`` on ``sm_count`` SMs: the launched
    tile and its PDL form (``gemm:<variant>:<tile>[:pdle]``; mirrors the export adapter's ``gemm_stage_key``)."""
    cfg = launch_tile_config(variant, select_tile_config(variant, M, sm_count))
    return gemm_kernel_key(variant, cfg.name, pdl_early_selected(variant, cfg, M))


def gemm_kernel_keys_for(
    total_tokens: int, merged: int, sm_count: int = DEFAULT_SM_COUNT
) -> dict[str, str]:
    """Logical GEMM kernel key per launched variant for one ``(T, N)`` on ``sm_count`` SMs."""
    return {
        variant: gemm_stage_key(
            variant,
            total_tokens if variant not in ("gelu_erf", "rmsnorm") else merged,
            sm_count,
        )
        for variant in PRODUCTION_GEMM_VARIANTS
    }


class GemmLaunchGeometry(NamedTuple):
    """Grid and the geometry parameters of one GEMM launch (``_launch_gemm``)."""

    grid: tuple[int, int, int]
    m_tiles: int
    cluster_tiles: int
    full_tiles: int  # cluster tiles of the data-parallel rounds (= cluster_tiles without a stream-K / half-N tail)
    tail_split: int  # stream-K k-steps per cluster over the tail (``_sk_range``); 1 on the other tiles


def gemm_launch_geometry(
    variant: str, cfg: TileConfig, M: int, sm_count: int
) -> GemmLaunchGeometry:
    """Geometry of one GEMM launch; mirrors ``kimi_k3_vision_gemm._launch_gemm``.

    The production tiles carry no tail split-K paths, so ``full_tiles`` is the
    cluster tile count and ``tail_split`` is 1 (the Cake launcher's values for
    non-tail configs); the stream-K twin (``sk``) keeps the full rounds
    data-parallel and cuts the tail tiles' k-steps into ranges of ``tail_split``
    (= q) k-steps per cluster.  The ``*_t`` twins are opt-in in Cake and outside
    the exported plan.
    """
    if cfg.tail:
        raise NotImplementedError(
            f"tile config {cfg.name!r} is a tail split-K twin: opt-in in Cake and not part of "
            "the exported production plan"
        )
    n_total, k_total, pos_split = GEMM_VARIANTS[variant]
    if pos_split:
        m_tiles = 2 * ((M + 2 * GEMM_BLOCK_M - 1) // (2 * GEMM_BLOCK_M))
    else:
        m_tiles = (M + GEMM_BLOCK_M - 1) // GEMM_BLOCK_M
        m_tiles += m_tiles % cfg.cta_group
    n_tiles = n_total // cfg.acc_n
    cluster_tiles = (m_tiles // cfg.cta_group) * (n_tiles // cfg.mcast)
    if cfg.ksplit > 1:
        clusters = cluster_tiles  # non-persistent: one cluster per output tile
    else:
        clusters = min(cluster_tiles, int(sm_count) // cfg.cluster_x)
    full_tiles, tail_split = cluster_tiles, 1
    if cfg.sk:
        # Stream-K: the full rounds stay data-parallel; the tail tiles' k-steps are cut into
        # ranges of q per cluster (``tail_split`` carries q; 0 = no tail, every piece empty).
        full_tiles = (cluster_tiles // clusters) * clusters
        tail_split = _sk_range(
            cluster_tiles - full_tiles, clusters, k_total // GEMM_BLOCK_K
        )
    elif cfg.half:
        # Half-N tail (round 6): the tiles past the last full round run as half items when they fit one half-round
        # (Cake ``_launch_gemm``: ``full_tiles = cluster_tiles - half_tail_split(cluster_tiles, clusters)``).
        full_tiles = cluster_tiles - half_tail_split(cluster_tiles, clusters)
    return GemmLaunchGeometry(
        (clusters * cfg.cluster_x, 1, 1), m_tiles, cluster_tiles, full_tiles, tail_split
    )


def gemm_configs_for(
    total_tokens: int, merged: int, sm_count: int = DEFAULT_SM_COUNT
) -> dict[str, str]:
    """Physical tile config name per launched GEMM variant for one ``(T, N)`` on ``sm_count`` SMs."""
    configs: dict[str, str] = {}
    for variant in PRODUCTION_GEMM_VARIANTS:
        count = total_tokens if variant not in ("gelu_erf", "rmsnorm") else merged
        configs[variant] = launch_tile_config(
            variant, select_tile_config(variant, count, sm_count)
        ).name
    return configs


def stream_k_workspace_ctas(sm_count: int) -> int:
    """CTAs the stream-K partial workspace is sized for (the Cake launcher's ``_tail_buffers``: the
    largest pair grid plus one cluster)."""
    return (int(sm_count) // 2 + 1) * 2


# Stream-K census bound of the merger forms (round 4): above 16 pair-grid rounds the twin's minimum saving
# (SK_MIN_SAVING = 6.5 % of the rounds) can no longer be met, so a per-tile scan up to here is exhaustive for
# ``_sk_twin``.  The other routing windows (``FC1_SK_L_MAX_M``, the PDL_EARLY windows) extend the census below.
# Mirrors the Cake export's ``MERGER_STREAM_K_SCAN_LIMIT`` / ``gemm_census_limit``.
MERGER_STREAM_K_SCAN_LIMIT = 65536
GEMM_CENSUS_LIMIT = (
    max(
        MERGER_STREAM_K_SCAN_LIMIT,
        FC1_SK_L_MAX_M,
        *(hi for ranges in PDL_EARLY_WINDOW.values() for _lo, hi in ranges),
        *(hi for _lo, hi in PDL_EARLY_DEFAULT_WINDOW),
    )
    + GEMM_BLOCK_M
)


def gemm_census_counts(variant: str) -> tuple[int, ...]:
    """Every token count at which the routing of ``variant`` can change (Cake export ``gemm_census_counts``):
    one count per 128-row tile from 1 to ``GEMM_CENSUS_LIMIT`` (tile selection and the stream-K censuses depend
    on the tile count), every token-count boundary of the tile rule and its neighbour, and every edge of the
    form's PDL_EARLY window (the window is on M, not on the tile count)."""
    counts = set(range(1, GEMM_CENSUS_LIMIT + 1, GEMM_BLOCK_M))
    for bound in (
        TINY_M_LIMIT,
        SMALL_M_LIMIT,
        MID_M_LIMIT,
        WO_S_E8_LIMIT,
        FC1_S_E8_LIMIT,
        QKV_XS_LIMIT,
        QKV_S_LIMIT,
        NG_XS_LOW,
        NG_XS_HIGH,
        NG_S_LIMIT,
        M_E8_MAX_M,
        QKV_M_E8_MIN_M,
        FC1_SK_L_MAX_M,
    ):
        counts.update((bound, bound + 1))
    for lo, hi in PDL_EARLY_WINDOW.get(variant, PDL_EARLY_DEFAULT_WINDOW):
        counts.update((lo - 1, lo, hi, hi + 1))
    return tuple(sorted(c for c in counts if 1 <= c <= GEMM_CENSUS_LIMIT))


# Attention census (Cake export ``attention_census_cu_seqlens`` / ``attention_required_keys``): the forms the plan rule
# selects on single segments of every 128-token step up to ATTENTION_CENSUS_LIMIT (past the largest contract segment
# 66564 and every routing threshold) plus the thresholds' edges; the contract rows add their multi-segment shapes.
# :func:`attention_census_keys` recomputes the census (~1 s per arch, tests); the module keeps the result as the
# literal ``ATTENTION_FORM_KEYS`` copied once from the frozen protocol so that importing the package stays cheap.
ATTENTION_CENSUS_LIMIT = 70016
ATTENTION_CENSUS_STEP = 128
CONTRACT_ROW_GRIDS: dict[str, tuple[tuple[int, int, int], ...]] = {
    # the 22 rows of the Cake evaluation contract ``eval_contract_kimi_k3_vision_tower`` (grid_thws per row)
    "img_224": ((1, 16, 16),),
    "img_336": ((1, 24, 24),),
    "img_448": ((1, 32, 32),),
    "img_640x480": ((1, 36, 46),),
    "img_800x600": ((1, 44, 58),),
    "img_1024x768": ((1, 56, 74),),
    "img_1280x720": ((1, 52, 92),),
    "img_1920x1080": ((1, 78, 138),),
    "doc_1240x1754": ((1, 126, 90),),
    "img_2560x1440": ((1, 104, 184),),
    "img_3840x2160": ((1, 156, 276),),
    "img_max_4096sq": ((1, 258, 258),),
    "batch4_1024x768": ((1, 56, 74),) * 4,
    "batch8_448": ((1, 32, 32),) * 8,
    "mixed_1080p_xga_448_336": ((1, 78, 138), (1, 56, 74), (1, 32, 32), (1, 24, 24)),
    "video_720p_4f": ((4, 52, 92),),
    "video_1080p_4f": ((4, 78, 138),),
    "video_720p_32f": ((4, 52, 92),) * 8,
    "video_480p_64f": ((4, 36, 46),) * 16,
    "smoke_2x2": ((1, 2, 2),),
    "smoke_ragged": ((1, 2, 6), (1, 10, 4), (2, 4, 4), (1, 6, 30)),
    "smoke_t3": ((3, 8, 8), (1, 12, 14)),
}
# Export-only coverage rows of the Cake export (``COVERAGE_GRIDS``, tag ``coverage``): the minimal grid_thws set reaching
# every kernel key the 22 contract rows do not (half-N twins at token counts no row has, the plain two-tile form on
# SM100, the plain / split one-tile forms on SM103).  Exported and validated bitwise; not part of the acceptance geomean.
COVERAGE_ROW_GRIDS: dict[str, tuple[tuple[int, int, int], ...]] = {
    "cov_batch11_224": ((1, 16, 16),) * 11,
    "cov_img_476x476": ((1, 34, 34),),
    "cov_img_2436x392": ((1, 28, 174),),
    "cov_img_3948x700": ((1, 50, 282),),
    "cov_img_2324x140": ((1, 10, 166),),
    "cov_img_4060x924": ((1, 66, 290),),
    "cov_img_3108x2716": ((1, 194, 222),),
}


def attention_census_cu_seqlens() -> list[tuple[int, ...]]:
    tokens = set(range(4, ATTENTION_CENSUS_LIMIT + 1, ATTENTION_CENSUS_STEP))
    for limit in UNIT_PREFETCH_MAX_TOKENS.values():
        tokens.update((int(limit), int(limit) + 1))
    for limit in (
        *RING3_MIN_SEGMENT_TOKENS.values(),
        *RING3_MAX_SEGMENT_TOKENS.values(),
    ):
        if limit:
            tokens.update((int(limit) - 1, int(limit), int(limit) + 1))
    census: list[tuple[int, ...]] = [(0, t) for t in sorted(tokens)]
    census.extend(cu_seqlens_of(grids) for grids in CONTRACT_ROW_GRIDS.values())
    census.extend(cu_seqlens_of(grids) for grids in COVERAGE_ROW_GRIDS.values())
    return census


def attention_census_keys(
    arch: str, grid_clusters: int = DEFAULT_SM_COUNT // 2
) -> tuple[str, ...]:
    """The attention forms of the census on ``arch`` (first-seen order) + ``attention_merge`` when any form splits."""
    keys: list[str] = []
    merge = False
    for cu in attention_census_cu_seqlens():
        plan = build_attention_plan(
            cu, torch.device("cpu"), HEADS, grid_clusters=grid_clusters, arch=arch
        )
        if plan.kernel_key not in keys:
            keys.append(plan.kernel_key)
        merge = merge or plan.num_merge_units > 0
    if merge:
        keys.append(ATTENTION_MERGE_KERNEL_KEY)
    return tuple(keys)


# = attention_census_keys(arch) at the round-6 head (copied once from the frozen protocol; the CPU test recomputes it).
ATTENTION_FORM_KEYS: dict[str, tuple[str, ...]] = {
    "sm_100a": (
        "attention:ring3:wide",
        "attention:tiles2:wide",
        "attention:tiles2",
        "attention:ring3",
        "attention:ring3:split",
        ATTENTION_MERGE_KERNEL_KEY,
    ),
    "sm_103a": (
        "attention:tiles2:wide",
        "attention:tiles1:wide",
        "attention:ring3:wide",
        "attention:tiles2",
        "attention:ring3",
        "attention:ring3:split",
        "attention:tiles2:split",
        "attention:tiles1",
        "attention:tiles1:split",
        ATTENTION_MERGE_KERNEL_KEY,
    ),
}


def required_kernel_keys(arch: str) -> tuple[str, ...]:
    """Every logical kernel the production plan can select on ``arch`` (the tile / PDL-form census of every
    GEMM form on the default SM count; the attention forms of the attention census + the split merge; merge;
    rmsnorm apply).  Mirrors the Cake export's ``required_kernel_keys``."""
    keys: list[str] = []
    for variant in PRODUCTION_GEMM_VARIANTS:
        for count in gemm_census_counts(variant):
            key = gemm_stage_key(variant, count)
            if key not in keys:
                keys.append(key)
    keys.extend(ATTENTION_FORM_KEYS[arch])
    keys.extend((MERGE_KERNEL_KEY, RMSNORM_APPLY_KERNEL_KEY))
    return tuple(keys)


REQUIRED_KERNEL_KEYS: dict[str, tuple[str, ...]] = {
    arch: required_kernel_keys(arch) for arch in SUPPORTED_COMPUTE_CAPABILITIES.values()
}
# Reachable keys no exported route exercises (Cake export ``UNCOVERED_KERNEL_KEYS``).  The round-6 routing leaves the
# 22 contract rows short of 12 GEMM half-twin / displaced-PDL keys per arch, ``attention:tiles2`` on sm_100a and
# ``attention:tiles1`` / ``attention:tiles1:split`` on sm_103a; the export's coverage rows (``COVERAGE_ROW_GRIDS``) reach
# exactly those, so the declaration is EMPTY at delivery and every reachable key is registered.  The machinery stays as
# the loud check: a plan resolving to an unregistered key is refused by name (:func:`prepare_kimi_k3_vision_tower`,
# ``NotImplementedError``) -- never served by another binary.
UNCOVERED_KERNEL_KEYS: dict[str, tuple[str, ...]] = {"sm_100a": (), "sm_103a": ()}
for _arch, _uncovered in UNCOVERED_KERNEL_KEYS.items():
    assert set(_uncovered) <= set(REQUIRED_KERNEL_KEYS[_arch]), _arch
# The keys the delivery registers per arch (= every route's keys of the export round).
REGISTERED_KERNEL_KEYS: dict[str, tuple[str, ...]] = {
    arch: tuple(
        k for k in REQUIRED_KERNEL_KEYS[arch] if k not in UNCOVERED_KERNEL_KEYS[arch]
    )
    for arch in REQUIRED_KERNEL_KEYS
}


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

WEIGHT_KEYS = (
    "patch_proj",
    "pos_emb",
    "time_weight",
    "final_norm",
    "merger_proj0",
    "merger_proj1",
    "post_norm",
    "layers",
)
LAYER_WEIGHT_KEYS = ("norm0", "wqkv", "wo", "norm1", "fc0", "fc1")


@dataclass(frozen=True)
class PreparedWeights:
    """Model parameters in the layout the generated kernels consume (BF16, contiguous).

    ``patch_proj`` is ``[2048, 640]``: rows ``[0, 1024)`` the ``[1024, 588]``
    patch projection zero-padded along K to 640, rows ``[1024, 2048)`` the same
    weight shifted right by four elements (odd pixel rows are streamed by TMA
    from a 16-byte-aligned start four elements early).  Per layer the six
    checkpoint tensors ``norm0``, ``wqkv``, ``wo``, ``norm1``, ``fc0``, ``fc1``
    as contiguous copies: the RMSNorm weights are NOT folded into the GEMM
    weights (a folded ``bf16(W * w_norm)`` is a fixed weight perturbation the
    HF chain does not have); they are applied on the activation side by the
    residual epilogues (``xw = bf16(x * w_next)``).
    """

    patch_proj: torch.Tensor
    pos_emb: torch.Tensor
    time_weight: torch.Tensor
    final_norm: torch.Tensor
    merger_proj0: torch.Tensor
    merger_proj1: torch.Tensor
    post_norm: torch.Tensor
    layers: tuple[dict[str, torch.Tensor], ...]

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def device(self) -> torch.device:
        return self.patch_proj.device


def _check_weight(t: torch.Tensor, shape: tuple[int, ...], name: str) -> torch.Tensor:
    if (
        not isinstance(t, torch.Tensor)
        or tuple(t.shape) != shape
        or t.dtype != torch.bfloat16
    ):
        raise ValueError(f"weights[{name!r}] must be a bf16 tensor of shape {shape}")
    return t.contiguous()


def prepare_kimi_k3_vision_weights(weights: dict[str, Any]) -> PreparedWeights:
    """Pad ``patch_proj`` and copy every parameter contiguously (once per model; no folding).

    ``weights`` uses ``nn.Linear`` ``[out, in]`` BF16 tensors without biases:
    ``patch_proj [1024, 588]`` (the 14x14 Conv2d flattened), ``pos_emb [64, 64,
    1024]``, ``time_weight [4, 1024]`` (``sincos_time_table``), ``final_norm
    [1024]``, ``merger_proj0 [4096, 4096]``, ``merger_proj1 [7168, 4096]``,
    ``post_norm [7168]`` and ``layers`` = list of ``{"norm0": [1024], "wqkv":
    [4608, 1024], "wo": [1024, 1536], "norm1": [1024], "fc0": [4096, 1024],
    "fc1": [1024, 4096]}``.  Every output tensor is a fresh contiguous copy.
    """
    missing = [k for k in WEIGHT_KEYS if k not in weights]
    if missing:
        raise ValueError(f"weights lack {missing}")
    w_pe = _check_weight(weights["patch_proj"], (HIDDEN, PATCH_DIM), "patch_proj")
    w_pe_pad = torch.zeros(
        (2 * HIDDEN, PATCH_DIM_PAD), dtype=torch.bfloat16, device=w_pe.device
    )
    w_pe_pad[:HIDDEN, :PATCH_DIM] = w_pe
    w_pe_pad[HIDDEN:, POS_SHIFT : POS_SHIFT + PATCH_DIM] = w_pe

    layer_shapes = {
        "norm0": (HIDDEN,),
        "wqkv": (QKV_N, HIDDEN),
        "wo": (HIDDEN, QKV_HIDDEN),
        "norm1": (HIDDEN,),
        "fc0": (FFN, HIDDEN),
        "fc1": (HIDDEN, FFN),
    }
    layers = []
    for index, lw in enumerate(weights["layers"]):
        missing = [k for k in LAYER_WEIGHT_KEYS if k not in lw]
        if missing:
            raise ValueError(f"weights['layers'][{index}] lacks {missing}")
        layers.append(
            {
                name: _check_weight(lw[name], shape, f"layers[{index}].{name}")
                for name, shape in layer_shapes.items()
            }
        )
    if not layers:
        raise ValueError("weights['layers'] must hold at least one encoder layer")
    return PreparedWeights(
        patch_proj=w_pe_pad,
        pos_emb=_check_weight(
            weights["pos_emb"], (POS_EMB_H, POS_EMB_W, HIDDEN), "pos_emb"
        ),
        time_weight=_check_weight(
            weights["time_weight"], (POS_EMB_T, HIDDEN), "time_weight"
        ),
        final_norm=_check_weight(weights["final_norm"], (HIDDEN,), "final_norm"),
        merger_proj0=_check_weight(
            weights["merger_proj0"], (MERGED_DIM, MERGED_DIM), "merger_proj0"
        ),
        merger_proj1=_check_weight(
            weights["merger_proj1"], (TEXT_HIDDEN, MERGED_DIM), "merger_proj1"
        ),
        post_norm=_check_weight(weights["post_norm"], (TEXT_HIDDEN,), "post_norm"),
        layers=tuple(layers),
    )


# ---------------------------------------------------------------------------
# Plan: everything a serving runtime derives once per grid_thws batch
# ---------------------------------------------------------------------------


def _arch_for(device: torch.device) -> str:
    capability = torch.cuda.get_device_capability(device)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise ValueError(
            "Kimi-K3 vision tower requires compute capability 10.0 or 10.3 "
            f"(got {capability[0]}.{capability[1]})"
        )
    return arch


def generated_program_available(device: torch.device) -> bool:
    """True when this checkout registers every kernel of the export's contract denominator for ``device``
    (``REGISTERED_KERNEL_KEYS``: the reachable keys minus the declared ``UNCOVERED_KERNEL_KEYS``)."""
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))
    return arch is not None and route_available(arch, REGISTERED_KERNEL_KEYS[arch])


def plan_kernel_keys(plan: "VisionTowerPlan") -> tuple[str, ...]:
    """The logical kernels one plan launches: its GEMM keys, its attention build (+ the split merge), merge,
    rmsnorm apply."""
    keys = list(dict.fromkeys(plan.gemm_kernel_keys.values()))
    keys.append(plan.attention.kernel_key)
    if plan.attention.num_merge_units > 0:
        keys.append(ATTENTION_MERGE_KERNEL_KEY)
    keys.extend((MERGE_KERNEL_KEY, RMSNORM_APPLY_KERNEL_KEY))
    return tuple(keys)


@dataclass(frozen=True)
class VisionTowerPlan:
    """Per-``grid_thws`` host plan: tables, attention plan, tile selection and workspaces."""

    grid_thws: tuple[tuple[int, int, int], ...]
    cu_seqlens: tuple[int, ...]
    total_tokens: int
    merged_tokens: int
    num_layers: int
    device: torch.device
    arch: str
    sm_count: int
    attention: AttentionPlan
    merge_table: torch.Tensor
    gemm_configs: dict[str, str]  # variant -> launched tile config name
    gemm_kernel_keys: dict[str, str]  # variant -> logical kernel key (tile + PDL form)
    cos: torch.Tensor
    sin: torch.Tensor
    rope_cs: (
        torch.Tensor
    )  # pack_rope_table(cos, sin): u32 [T, 64] f16x2 (cos, sin) words
    workspace: dict[str, torch.Tensor] = field(repr=False)
    # (module, workspace address) pairs whose TMA descriptor workspace has been
    # prepared (``run_prepare_tma``); the descriptors depend on plan-owned
    # buffers only, so one preparation per plan serves every runner.
    prepared_tma: dict[tuple[str, int], bool] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def tiles_per_cta(self) -> int:
        return self.attention.tiles_per_cta


def attention_tma_workspace_bytes(arch: str, key: str) -> int:
    """Bytes of the caller-owned TMA descriptor workspace of the registered attention module ``key``
    (``attention:<layout>[:wide][:split]``; 0 = by-value ABI / unregistered)."""
    if not route_available(arch, (key,)):
        return 0
    return int(MODULES[kernel_module_name(arch, key)].get("tma_workspace_bytes", 0))


def vision_workspace_shapes(
    total_tokens: int,
    merged: int,
    attention_tma_bytes: int = 0,
    stream_k_ctas: int = 0,
    partial_slots: Optional[int] = None,
    part_rows: int = 0,
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """``partial_slots`` = the attention plan's ``num_partial_slots`` on the split build (``None`` = whole units:
    the two partial pointers bind an 8-element f32 dummy, as the Cake plan)."""
    bf16 = torch.bfloat16
    shapes = {
        "x": ((total_tokens, HIDDEN), bf16),
        # RMSNorm handoff written by the residual epilogues and read by the next
        # norm GEMM: xw = bf16(x * w_norm_next) and the FP32 [T, 16] row sums of
        # squares of x (per 64 columns).
        "xw": ((total_tokens, HIDDEN), bf16),
        "stats": ((total_tokens, STATS_PARTS), torch.float32),
        "q": ((total_tokens, HEADS, HEAD_DIM), bf16),
        "k": ((total_tokens, HEADS, HEAD_DIM), bf16),
        "v": ((total_tokens, HEADS, HEAD_DIM), bf16),
        "attn_out": ((total_tokens, HEADS, HEAD_DIM), bf16),
        "ffn": ((total_tokens, FFN), bf16),
        "m": ((merged, MERGED_DIM), bf16),
        "h": ((merged, MERGED_DIM), bf16),
        "rowsumsq": ((merged,), torch.float32),
        # Bound to unused pointer parameters (never dereferenced for the tiles
        # the kernels run): FP32 dummy for COS/SIN/SQ and the tail workspace
        # WS / u32 zeros for the tail arrival counters FLAGS of the tiles
        # without a stream-K tail, the odd-row pixel map when T == 1 (CS / XW /
        # WN default to B / C / B like the Cake launcher), and the attention
        # kernel's diagnostic probe buffer.
        "f32_dummy": ((16,), torch.float32),
        "u32_dummy": ((16,), torch.uint32),
        "pixel_dummy": ((2, PATCH_DIM), bf16),
        "probe_dummy": ((PROBE_WORDS,), torch.uint64),
        # KV-split partial workspace (round 6; Cake ``build_segment_plan``): the f32 partial O rows ``[slot][row][128]``
        # relative to the part's reference max and the ``(max, sum)`` pairs ``[slot][row][2]`` of every partial slot,
        # read by the split merge; 8-element dummies on the plans without split units.
        "partial_O": (
            (
                (max(int(partial_slots), 1) * int(part_rows) * HEAD_DIM,)
                if partial_slots is not None
                else (8,)
            ),
            torch.float32,
        ),
        "partial_ML": (
            (
                (max(int(partial_slots), 1) * int(part_rows) * 2,)
                if partial_slots is not None
                else (8,)
            ),
            torch.float32,
        ),
    }
    if attention_tma_bytes:
        # Pointer-ABI attention: the plan owns the device bytes of the kernel's
        # CUtensorMaps (prepared once by ``run_prepare_tma``; 128-byte aligned
        # by the caching allocator's 512-byte granularity, verified by the binding).
        shapes["attn_tma_desc"] = ((int(attention_tma_bytes),), torch.uint8)
    if stream_k_ctas:
        # Stream-K GEMM tiles (round 4): the FP32 partial workspace of SK_MAX_CONTRIB - 1
        # pieces per (tail tile, CTA) over every tail tile of the largest grid, 256 columns
        # wide, and the u32 arrival counters (zero; self-reset by the last arriver) -- the
        # Cake launcher's ``_tail_buffers(device, SK_MAX_CONTRIB - 1)``, shared by every
        # stream-K launch of the plan (the launches are stream-ordered).
        shapes["sk_ws"] = (
            (int(stream_k_ctas) * (SK_MAX_CONTRIB - 1) * GEMM_BLOCK_M * 256,),
            torch.float32,
        )
        shapes["sk_flags"] = ((int(stream_k_ctas),), torch.uint32)
    return shapes


def pack_rope_table(cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Pack the FP32 ``[T, 64]`` cos/sin tables into one ``uint32`` ``[T, 64]`` table of f16x2
    words (low half = cos, high half = sin) for the ``*_cs`` QKV tiles: half the table bytes per
    q/k column tile (``kimi_k3_vision_gemm.pack_rope_table``).  f16 rounds cos/sin to 2^-11
    relative; the rotation stays FP32 and the q/k outputs are BF16.  Built once per plan."""
    if cos.shape != sin.shape or cos.dim() != 2 or cos.shape[1] != ROPE_PAIRS:
        raise ValueError(
            f"cos/sin must be [T, {ROPE_PAIRS}] tables, got {tuple(cos.shape)} / {tuple(sin.shape)}"
        )
    packed = torch.stack([cos.float().half(), sin.float().half()], dim=-1).reshape(
        cos.shape[0], 2 * ROPE_PAIRS
    )
    return packed.contiguous().view(torch.uint32)


def build_kimi_k3_vision_plan(
    grid_thws: Sequence[Sequence[int]],
    device: Union[torch.device, str],
    *,
    num_layers: int,
    grid_clusters: Optional[int] = None,
    sm_count: Optional[int] = None,
    arch: Optional[str] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
) -> VisionTowerPlan:
    """Segment plan, merge table, RoPE tables, tile selection and all workspaces for one batch.

    ``sm_count`` / ``grid_clusters`` default to the device's SM count (and its
    half) and ``arch`` to the device's architecture; pass them to build a plan
    for another device (CPU tests; ``arch`` is honoured on CPU devices only).
    ``cos`` / ``sin`` may be supplied by a runtime that caches them per grid;
    the packed f16x2 ``rope_cs`` table the QKV GEMM reads is derived from them
    here.
    """
    device = torch.device(device)
    grids = tuple(validate_grid_thws(grid_thws))
    cu = cu_seqlens_of(grids)
    total = cu[-1]
    merged = merged_tokens(grids)
    if sm_count is None:
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    sm_count = int(sm_count)
    if grid_clusters is None:
        grid_clusters = max(1, sm_count // 2)
    if device.type == "cuda":
        arch = _arch_for(device)
    elif arch is None:
        arch = "cpu"
    if cos is None or sin is None:
        cos, sin = rope_cos_sin(grids, device)
    for name, table in (("cos", cos), ("sin", sin)):
        if (
            tuple(table.shape) != (total, ROPE_PAIRS)
            or table.dtype != torch.float32
            or table.device != device
            or not table.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be a contiguous fp32 [{total}, {ROPE_PAIRS}] tensor on {device}"
            )
    attention = build_attention_plan(
        cu, device, HEADS, grid_clusters=grid_clusters, arch=arch
    )
    tma_bytes = (
        attention_tma_workspace_bytes(arch, attention.kernel_key)
        if device.type == "cuda"
        else 0
    )
    gemm_configs = gemm_configs_for(total, merged, sm_count)
    gemm_kernel_keys = gemm_kernel_keys_for(total, merged, sm_count)
    stream_k_ctas = (
        stream_k_workspace_ctas(sm_count)
        if any(TILE_CONFIGS[name].sk for name in gemm_configs.values())
        else 0
    )
    workspace = {
        name: torch.zeros(shape, dtype=dtype, device=device)
        for name, (shape, dtype) in vision_workspace_shapes(
            total,
            merged,
            attention_tma_bytes=tma_bytes,
            stream_k_ctas=stream_k_ctas,
            partial_slots=attention.num_partial_slots
            if attention.kv_split is not None
            else None,
            part_rows=attention.part_rows,
        ).items()
    }
    return VisionTowerPlan(
        grid_thws=grids,
        cu_seqlens=cu,
        total_tokens=total,
        merged_tokens=merged,
        num_layers=int(num_layers),
        device=device,
        arch=arch,
        sm_count=sm_count,
        attention=attention,
        merge_table=build_merge_table(grids, device),
        gemm_configs=gemm_configs,
        gemm_kernel_keys=gemm_kernel_keys,
        cos=cos,
        sin=sin,
        rope_cs=pack_rope_table(cos, sin),
        workspace=workspace,
    )


# ---------------------------------------------------------------------------
# Binding to the generated argument plans
# ---------------------------------------------------------------------------


def _bind(
    module_name: str, kwargs: dict[str, Any]
) -> tuple[Callable[..., Any], tuple, Optional[Callable[..., Any]]]:
    """Order ``kwargs`` by the generated argument plan of ``module_name`` and load its entries.

    Returns the launch entry, its positional arguments and the module's TMA
    preparation entry (``tma_prepare_entry``; ``None`` for by-value descriptor
    modules), which takes the same positional arguments.
    """
    record = MODULES[module_name]
    grid = dict(zip(("grid_x", "grid_y", "grid_z"), kwargs["grid"], strict=True))
    arguments = []
    for kind, name in record["arg_plan"]:
        if kind == "grid":
            arguments.append(grid[name])
        elif name in kwargs:
            arguments.append(kwargs[name])
        else:
            raise KeyError(
                f"generated module {module_name!r} expects argument {name!r} "
                f"({kind}); host binding provides {sorted(kwargs)}"
            )
    module = load_cake_kimi_k3_vision_tower_module(module_name)
    prepare_entry = record.get("tma_prepare_entry")
    prepare = getattr(module, prepare_entry) if prepare_entry else None
    return getattr(module, record["ffi_entry"]), tuple(arguments), prepare


@dataclass(frozen=True)
class _Launch:
    stage: str
    layer: int  # -1 outside the encoder layers
    module: str
    kwargs: dict[str, Any] = field(repr=False)
    entry: Callable[..., Any] = field(repr=False)
    arguments: tuple = field(repr=False)
    # ``run_prepare_tma`` of a pointer-ABI module (same arguments); None otherwise.
    prepare: Optional[Callable[..., Any]] = field(default=None, repr=False)

    def __call__(self) -> None:
        self.entry(*self.arguments)

    def prepare_tma(self, plan: "VisionTowerPlan") -> bool:
        """Prepare this launch's descriptor workspace once per plan; True if a copy was made."""
        if self.prepare is None:
            return False
        key = (self.module, int(self.kwargs["tma_descriptor_workspace"].data_ptr()))
        if key in plan.prepared_tma:
            return False
        with tvm_ffi.use_torch_stream():
            self.prepare(*self.arguments)
        plan.prepared_tma[key] = True
        return True


@dataclass(frozen=True)
class KimiK3VisionTowerRunner:
    """The prepared launch sequence of one ``(pixel_values, grid_thws, weights, out)`` binding.

    ``launch()`` runs every kernel of the tower on the current torch stream into
    the caller-owned ``out`` with no CUDA allocation and no host
    synchronization and returns ``out``; it is CUDA-graph capturable (capture
    belongs to the caller; the attention descriptor workspace was prepared by
    :func:`prepare_kimi_k3_vision_tower`).  Prepare a new runner when ``grid_thws``, the layer
    count or a tensor binding (``pixel_values``, ``out``, weights) changes;
    values may change freely.  ``stages`` exposes one launch per stage (layer
    stages use layer 0) for per-operator tests and timing.
    """

    plan: VisionTowerPlan
    weights: PreparedWeights
    pixel_values: torch.Tensor
    pos_rows: torch.Tensor
    out: torch.Tensor
    launches: tuple[_Launch, ...] = field(repr=False)

    def launch(self) -> torch.Tensor:
        with tvm_ffi.use_torch_stream():
            for item in self.launches:
                item.entry(*item.arguments)
        return self.out

    __call__ = launch

    @property
    def stages(self) -> dict[str, Callable[[], None]]:
        """One zero-argument launch per stage: encoder stages bound to layer 0,
        ``layer_fc1_last`` to the last layer (its only occurrence).  On the split-KV
        rows ``layer_attention`` runs the attention kernel AND its split merge (the
        complete attention operator, as the Cake stage function); the merge is also
        exposed alone as ``layer_attention_merge``."""
        result: dict[str, Callable[[], None]] = {}
        launches = list(self.launches)
        for index, item in enumerate(launches):
            if item.stage in result:
                continue
            group = [item]
            if (
                item.stage == "layer_attention"
                and index + 1 < len(launches)
                and launches[index + 1].stage == "layer_attention_merge"
            ):
                group.append(launches[index + 1])

            def run(group: list[_Launch] = group) -> None:
                with tvm_ffi.use_torch_stream():
                    for launch in group:
                        launch.entry(*launch.arguments)

            result[item.stage] = run
        return result

    @property
    def stage_modules(self) -> dict[str, str]:
        """Physical generated module per stage (identical for every encoder layer)."""
        result: dict[str, str] = {}
        for item in self.launches:
            result.setdefault(item.stage, item.module)
        return result

    @property
    def launch_count(self) -> int:
        return len(self.launches)

    @property
    def route_metadata(self) -> dict[str, Any]:
        plan = self.plan
        return dict(
            arch=plan.arch,
            layers=plan.num_layers,
            total_tokens=plan.total_tokens,
            merged_tokens=plan.merged_tokens,
            segment_count=plan.attention.num_segments,
            tiles_per_cta=plan.attention.tiles_per_cta,
            ring3=plan.attention.ring3,
            wide=plan.attention.wide,
            kv_split=plan.attention.kv_split,
            kv_split_used=plan.attention.kv_split_used,
            split_units=plan.attention.split_units,
            attention_kernel_key=plan.attention.kernel_key,
            attention_merge=plan.attention.num_merge_units > 0,
            attention_merge_units=plan.attention.num_merge_units,
            attention_items=plan.attention.total_tiles,
            attention_units=plan.attention.total_units,
            attention_clusters=plan.attention.num_clusters,
            attention_full_clusters=plan.attention.full_clusters,
            gemm_configs=dict(plan.gemm_configs),
            gemm_kernel_keys=dict(plan.gemm_kernel_keys),
            gemm_pdl_early={
                variant: key.endswith(":pdle")
                for variant, key in plan.gemm_kernel_keys.items()
            },
            launch_count=len(self.launches),
            sm_count=plan.sm_count,
        )


def _check_2d(
    t: torch.Tensor,
    shape: tuple[int, ...],
    name: str,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    if tuple(t.shape) != tuple(shape) or t.dtype != dtype or not t.is_contiguous():
        raise ValueError(
            f"{name} must be a contiguous {dtype} tensor of shape {tuple(shape)}, got {tuple(t.shape)} {t.dtype}"
        )


def _gemm_launch(
    stage: str,
    layer: int,
    plan: VisionTowerPlan,
    variant: str,
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    C: torch.Tensor,
    C2: Optional[torch.Tensor] = None,
    C3: Optional[torch.Tensor] = None,
    R: Optional[torch.Tensor] = None,
    COS: Optional[torch.Tensor] = None,
    SIN: Optional[torch.Tensor] = None,
    CS: Optional[torch.Tensor] = None,
    SQ: Optional[torch.Tensor] = None,
    XW: Optional[torch.Tensor] = None,
    WN: Optional[torch.Tensor] = None,
    eps: float = 0.0,
) -> _Launch:
    ws = plan.workspace
    M = int(A.shape[0])
    cfg = TILE_CONFIGS[plan.gemm_configs[variant]]
    if cfg.rope and CS is None:
        raise ValueError(f"{variant} on {cfg.name!r} requires the packed RoPE table CS")
    geometry = gemm_launch_geometry(variant, cfg, M, plan.sm_count)
    if GEMM_VARIANTS[variant][2]:
        # Odd-row map over the same pixel tensor; M == 1 has no odd row but the
        # descriptor needs floor(rows / 2) >= 1.
        A2 = A if M >= 2 else ws["pixel_dummy"]
    else:
        A2 = A
    kwargs = dict(
        A=A,
        A2=A2,
        B=B,
        C=C,
        C2=C2 if C2 is not None else C,
        C3=C3 if C3 is not None else C,
        R=R if R is not None else C,
        COS=COS if COS is not None else ws["f32_dummy"],
        SIN=SIN if SIN is not None else ws["f32_dummy"],
        CS=CS if CS is not None else B,
        SQ=SQ if SQ is not None else ws["f32_dummy"],
        XW=XW if XW is not None else C,
        WN=WN if WN is not None else B,
        # Stream-K tiles: the plan's FP32 partial workspace and arrival counters; the other
        # tiles bind the Cake launcher's dummies (never dereferenced).
        WS=ws["sk_ws"] if cfg.sk else ws["f32_dummy"],
        FLAGS=ws["sk_flags"] if cfg.sk else ws["u32_dummy"],
        # TMA-epilogue tensor maps (residual tile in, C and xw tiles out); the register-epilogue
        # tiles bind any 2-D bf16 tensor with a 64-multiple row length (unused), as the Cake launcher.
        RT=(R if R is not None else C) if cfg.tma_epi else B,
        CT=C if cfg.tma_epi else B,
        XWT=(XW if XW is not None else C) if cfg.tma_epi else B,
        M=M,
        m_tiles=int(geometry.m_tiles),
        full_tiles=int(geometry.full_tiles),
        tail_split=int(geometry.tail_split),
        pf_l2=0,  # operand L2 prefetch (Cake ``_L2PF_POLICY``) is off in production
        eps=float(eps),
        grid=geometry.grid,
    )
    assert tuple(kwargs) == GEMM_KWARGS
    # The plan's key carries the tile AND the PDL form (production vs PDL_EARLY binary) of this launch.
    key = plan.gemm_kernel_keys[variant]
    assert key.split(":")[2] == cfg.name, (key, cfg.name)
    module = kernel_module_name(plan.arch, key)
    entry, arguments, prepare = _bind(module, kwargs)
    return _Launch(stage, layer, module, kwargs, entry, arguments, prepare)


def _attention_launch(plan: VisionTowerPlan, layer: int) -> _Launch:
    ws, attn = plan.workspace, plan.attention
    kwargs = dict(
        Q=ws["q"],
        Q_raw=ws["q"],
        K=ws["k"],
        V=ws["v"],
        O=ws["attn_out"],
        O_tma=ws["attn_out"],
        seg_begin=attn.seg_begin,
        seg_len=attn.seg_len,
        unit_table=attn.unit_table,
        probe=ws["probe_dummy"],
        total_tiles=int(attn.total_tiles),
        num_heads=int(attn.num_heads),
        softmax_scale_log2=float(SOFTMAX_SCALE) / math.log(2.0),
        # KV-split partial workspace (the split build writes it, the merge reads it; dummies otherwise).
        partial_O=ws["partial_O"],
        partial_ML=ws["partial_ML"],
        # Pointer-ABI descriptor workspace (plan-owned; ignored by a by-value module).
        tma_descriptor_workspace=ws.get("attn_tma_desc", ws["u32_dummy"]),
        # The lever-G shaped persistent grid (2 CTAs per cluster).
        grid=(2 * int(attn.num_clusters), 1, 1),
    )
    assert tuple(kwargs) == ATTENTION_KWARGS
    module = kernel_module_name(plan.arch, attn.kernel_key)
    entry, arguments, prepare = _bind(module, kwargs)
    return _Launch("layer_attention", layer, module, kwargs, entry, arguments, prepare)


def _attention_merge_launch(plan: VisionTowerPlan, layer: int) -> _Launch:
    """The exact fixed-order merge of the split units (Cake ``launch_split_merge``): one warp per output row,
    ``part_rows // MERGE_ROWS_PER_CTA`` CTAs per merge unit."""
    ws, attn = plan.workspace, plan.attention
    if attn.part_rows % MERGE_ROWS_PER_CTA:
        raise ValueError(
            f"part_rows={attn.part_rows} must be a multiple of {MERGE_ROWS_PER_CTA}"
        )
    kwargs = dict(
        partial_O=ws["partial_O"],
        partial_ML=ws["partial_ML"],
        merge_table=attn.merge_table,
        seg_begin=attn.seg_begin,
        seg_len=attn.seg_len,
        O=ws["attn_out"],
        num_heads=int(attn.num_heads),
        part_rows=int(attn.part_rows),
        softmax_scale_log2=float(SOFTMAX_SCALE) / math.log(2.0),
        grid=(
            int(attn.num_merge_units) * (int(attn.part_rows) // MERGE_ROWS_PER_CTA),
            1,
            1,
        ),
    )
    assert tuple(kwargs) == ATTENTION_MERGE_KWARGS
    module = kernel_module_name(plan.arch, ATTENTION_MERGE_KERNEL_KEY)
    entry, arguments, prepare = _bind(module, kwargs)
    return _Launch(
        "layer_attention_merge", layer, module, kwargs, entry, arguments, prepare
    )


def _merge_launch(plan: VisionTowerPlan, weights: PreparedWeights) -> _Launch:
    ws = plan.workspace
    kwargs = dict(
        x=ws["x"],
        norm_weight=weights.final_norm,
        merge_table=plan.merge_table,
        m_out=ws["m"],
        eps=float(NORM_EPS),
        grid=(int(plan.merged_tokens), 1, 1),
    )
    assert tuple(kwargs) == MERGE_KWARGS
    module = kernel_module_name(plan.arch, MERGE_KERNEL_KEY)
    entry, arguments, prepare = _bind(module, kwargs)
    return _Launch("final_norm_merge", -1, module, kwargs, entry, arguments, prepare)


def _rmsnorm_apply_launch(
    plan: VisionTowerPlan, weights: PreparedWeights, out: torch.Tensor
) -> _Launch:
    N = plan.merged_tokens
    kwargs = dict(
        y=out,
        weight=weights.post_norm,
        rowsumsq=plan.workspace["rowsumsq"],
        M=int(N),
        eps=float(PROJECTOR_EPS),
        grid=((N + RMS_ROWS_PER_CTA - 1) // RMS_ROWS_PER_CTA, 1, 1),
    )
    assert tuple(kwargs) == RMSNORM_APPLY_KWARGS
    module = kernel_module_name(plan.arch, RMSNORM_APPLY_KERNEL_KEY)
    entry, arguments, prepare = _bind(module, kwargs)
    return _Launch(
        "merger_rmsnorm_apply", -1, module, kwargs, entry, arguments, prepare
    )


def _launch_sequence(
    plan: VisionTowerPlan,
    weights: PreparedWeights,
    pixel_values: torch.Tensor,
    pos_rows: torch.Tensor,
    out: torch.Tensor,
) -> tuple[_Launch, ...]:
    ws = plan.workspace
    T = plan.total_tokens
    pixels2d = pixel_values.view(T, PATCH_DIM)
    layers = weights.layers
    stats, xw = ws["stats"], ws["xw"]
    launches = [
        _gemm_launch(
            "patch_embed",
            -1,
            plan,
            "pos_sqxw",
            pixels2d,
            weights.patch_proj,
            C=ws["x"],
            R=pos_rows,
            SQ=stats,
            XW=xw,
            WN=layers[0]["norm0"],
        )
    ]
    for index, lw in enumerate(layers):
        launches.append(
            _gemm_launch(
                "layer_norm_qkv_rope",
                index,
                plan,
                "norm_qkv_rope",
                xw,
                lw["wqkv"],
                C=ws["q"],
                C2=ws["k"],
                C3=ws["v"],
                COS=plan.cos,
                SIN=plan.sin,
                # The kernel's CS pointer is bf16-typed: the u32 words viewed as bf16 pairs.
                CS=plan.rope_cs.view(torch.bfloat16),
                SQ=stats,
                eps=NORM_EPS,
            )
        )
        launches.append(_attention_launch(plan, index))
        if plan.attention.num_merge_units > 0:
            launches.append(_attention_merge_launch(plan, index))
        launches.append(
            _gemm_launch(
                "layer_out_proj",
                index,
                plan,
                "residual_wo_sqxw",
                ws["attn_out"].view(T, QKV_HIDDEN),
                lw["wo"],
                C=ws["x"],
                R=ws["x"],
                SQ=stats,
                XW=xw,
                WN=lw["norm1"],
            )
        )
        launches.append(
            _gemm_launch(
                "layer_norm_fc0_gelu",
                index,
                plan,
                "norm_gelu",
                xw,
                lw["fc0"],
                C=ws["ffn"],
                SQ=stats,
                eps=NORM_EPS,
            )
        )
        if index + 1 < len(layers):
            launches.append(
                _gemm_launch(
                    "layer_fc1",
                    index,
                    plan,
                    "residual_fc1_sqxw",
                    ws["ffn"],
                    lw["fc1"],
                    C=ws["x"],
                    R=ws["x"],
                    SQ=stats,
                    XW=xw,
                    WN=layers[index + 1]["norm0"],
                )
            )
        else:
            # No consumer after the last layer: the merge kernel reads x directly.
            launches.append(
                _gemm_launch(
                    "layer_fc1_last",
                    index,
                    plan,
                    "residual_fc1",
                    ws["ffn"],
                    lw["fc1"],
                    C=ws["x"],
                    R=ws["x"],
                )
            )
    launches.append(_merge_launch(plan, weights))
    launches.append(
        _gemm_launch(
            "merger_gemm0",
            -1,
            plan,
            "gelu_erf",
            ws["m"],
            weights.merger_proj0,
            C=ws["h"],
        )
    )
    launches.append(
        _gemm_launch(
            "merger_gemm1", -1, plan, "rmsnorm", ws["h"], weights.merger_proj1, C=out
        )
    )
    launches.append(_rmsnorm_apply_launch(plan, weights, out))
    return tuple(launches)


# ---------------------------------------------------------------------------
# Preparation and one-shot entry point
# ---------------------------------------------------------------------------


def prepare_kimi_k3_vision_tower(
    pixel_values: torch.Tensor,
    grid_thws: Sequence[Sequence[int]],
    weights: Union[dict[str, Any], PreparedWeights],
    out: Optional[torch.Tensor] = None,
    *,
    plan: Optional[VisionTowerPlan] = None,
    pos_rows: Optional[torch.Tensor] = None,
    backend: str = "cake",
) -> KimiK3VisionTowerRunner:
    """Validate, plan and bind one complete vision-tower call.

    ``pixel_values`` is the contiguous BF16 ``[T, 3, 14, 14]`` patch tensor in
    packed ``grid_thws`` order; ``weights`` the parameter dict of
    :func:`prepare_kimi_k3_vision_weights` or its prepared form (prepare once
    per model); ``out`` the optional caller-owned BF16 ``[N, 7168]`` output.
    ``plan`` (:func:`build_kimi_k3_vision_plan`) and ``pos_rows``
    (:func:`pos_emb_rows`) may be supplied by a runtime that caches them per
    ``grid_thws``; otherwise they are derived here.  Every allocation happens
    in this call; the returned runner launches with none.
    """
    if backend != "cake":
        raise ValueError("Kimi-K3 vision tower supports backend='cake'")
    if not pixel_values.is_cuda:
        raise ValueError("pixel_values must be a CUDA tensor")
    device = pixel_values.device
    if (
        pixel_values.dtype != torch.bfloat16
        or pixel_values.ndim != 4
        or tuple(pixel_values.shape[1:]) != (IN_CH, PATCH, PATCH)
        or not pixel_values.is_contiguous()
    ):
        raise ValueError("pixel_values must be a contiguous bf16 [T, 3, 14, 14] tensor")
    if pixel_values.data_ptr() % 16:
        raise ValueError("pixel_values must start at a 16-byte-aligned address")
    prepared = (
        weights
        if isinstance(weights, PreparedWeights)
        else prepare_kimi_k3_vision_weights(weights)
    )
    if prepared.device != device:
        raise ValueError(f"weights live on {prepared.device}, pixel_values on {device}")
    grids = tuple(validate_grid_thws(grid_thws))
    if plan is None:
        plan = build_kimi_k3_vision_plan(grids, device, num_layers=prepared.num_layers)
    if (
        plan.grid_thws != grids
        or plan.device != device
        or plan.num_layers != prepared.num_layers
    ):
        raise ValueError(
            "plan was built for another grid_thws batch, device or layer count"
        )
    arch = _arch_for(device)
    if plan.arch != arch:
        raise ValueError(f"plan targets {plan.arch}, device is {arch}")
    if int(pixel_values.shape[0]) != plan.total_tokens:
        raise ValueError(
            f"pixel_values has {int(pixel_values.shape[0])} tokens, grid_thws describe {plan.total_tokens}"
        )
    # The plan's own kernels must be registered: the delivery covers the export's contract denominator
    # (``REGISTERED_KERNEL_KEYS``); a plan resolving to one of the declared ``UNCOVERED_KERNEL_KEYS`` (a reachable
    # form no contract row exercises) is refused by name -- never served by another binary.
    needed = plan_kernel_keys(plan)
    if not route_available(arch, needed):
        missing = [key for key in needed if not route_available(arch, (key,))]
        uncovered = [
            key for key in missing if key in UNCOVERED_KERNEL_KEYS.get(arch, ())
        ]
        raise NotImplementedError(
            f"The generated Kimi-K3 vision tower programs for {arch} do not cover this plan in this checkout "
            f"(missing {missing}; declared uncovered by the export denominator: {uncovered}; "
            "see flashinfer-ai/flashinfer#4568)"
        )
    if out is None:
        out = torch.empty(
            (plan.merged_tokens, TEXT_HIDDEN), dtype=torch.bfloat16, device=device
        )
    _check_2d(out, (plan.merged_tokens, TEXT_HIDDEN), "out")
    if out.device != device:
        raise ValueError(f"out must live on {device}")
    if pos_rows is None:
        pos_rows = pos_emb_rows(prepared.pos_emb, prepared.time_weight, grids)
    _check_2d(pos_rows, (plan.total_tokens, HIDDEN), "pos_rows")
    launches = _launch_sequence(plan, prepared, pixel_values, pos_rows, out)
    # Pointer-ABI modules (the attention kernel): encode the CUtensorMaps and
    # copy them into the plan-owned workspace once, synchronously, before any
    # launch or graph capture.  Idempotent per (module, workspace).
    for item in launches:
        item.prepare_tma(plan)
    return KimiK3VisionTowerRunner(
        plan, prepared, pixel_values, pos_rows, out, launches
    )


def kimi_k3_vision_tower(
    pixel_values: torch.Tensor,
    grid_thws: Sequence[Sequence[int]],
    weights: Union[dict[str, Any], PreparedWeights],
    out: Optional[torch.Tensor] = None,
    *,
    plan: Optional[VisionTowerPlan] = None,
    pos_rows: Optional[torch.Tensor] = None,
    backend: str = "cake",
) -> torch.Tensor:
    runner = prepare_kimi_k3_vision_tower(
        pixel_values,
        grid_thws,
        weights,
        out,
        plan=plan,
        pos_rows=pos_rows,
        backend=backend,
    )
    return runner.launch()
