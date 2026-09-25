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

    patch_embed          gemm pos:            x  = bf16(pixels @ Wpe^T) + pos_rows          [T, 1024]
    27 x encoder layer:
      layer_norm_qkv_rope gemm norm_qkv_rope: q, k, v = RoPE2D(bf16(RMSNorm(x) @ Wqkv^T)) 3 x [T, 12, 128]
      layer_attention     packed-varlen noncausal BF16 attention per grid_thw segment  [T, 12, 128]
      layer_out_proj      gemm residual_wo:    x += bf16(a @ Wo^T)                          (in place)
      layer_norm_fc0_gelu gemm norm_gelu:      f  = bf16(gelu_tanh(bf16(RMSNorm(x) @ Wfc0^T))) [T, 4096]
      layer_fc1           gemm residual_fc1:   x += bf16(f @ Wfc1^T)                        (in place)
    final_norm_merge     final RMSNorm + 2x2 spatial / temporal-mean merge               [N, 4096]
    merger_gemm0         gemm gelu_erf:       h  = bf16(gelu_erf(bf16(m @ Wp0^T)))         [N, 4096]
    merger_gemm1         gemm rmsnorm:        y  = bf16(h @ Wp1^T)                         [N, 7168]
    merger_rmsnorm_apply out = bf16(RMSNorm(y, post_norm, eps = 1e-5))                    (in place)

Host work is split exactly like the Cake production launcher:

* :func:`prepare_kimi_k3_vision_weights` -- once per model: the RMSNorm
  weights are folded into ``wqkv`` / ``fc0`` (``bf16(W * norm_w[None, :])``)
  and the patch projection is zero-padded / shifted for the pixel-row TMA
  maps.  The kernels compute the row ``rstd`` in FP32 and scale the FP32
  accumulator before the single BF16 rounding.
* :func:`build_kimi_k3_vision_plan` -- once per ``grid_thws`` batch: the
  ``cu_seqlens``, the 2-D RoPE ``cos``/``sin`` tables, the merge table, the
  attention segment plan (unit layout + LPT unit table), the GEMM tile
  configurations for the token counts and every workspace.  The positional
  rows depend on the weights and are derived by
  :func:`prepare_kimi_k3_vision_tower` (or passed in by a serving runtime that
  caches them per grid).
* :func:`prepare_kimi_k3_vision_tower` -- binds every launch of the sequence
  to the generated argument plans.  The returned runner's ``launch()`` performs
  no allocation and no host synchronization and is CUDA-graph capturable.

The host plan reproduces the Cake planner table for table (segment plan,
unit table, merge table, tile selection, launch grids); the generated-program
export checks that parity on every contract row.  See ``README.md`` in this
package and flashinfer-ai/flashinfer#4568 (tracker #4254).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Union

import torch
import tvm_ffi

from ..minimax_h3_varlen_attention.cake_backend import assign_unit_slots
from .cake_jit import (
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
PROBE_WORDS = 64  # unused diagnostic buffer parameter of the attention kernel (PROBE_UNITS * PROBE_EVENTS)

SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}

STAGE_NAMES = (
    "patch_embed",
    "layer_norm_qkv_rope",
    "layer_attention",
    "layer_out_proj",
    "layer_norm_fc0_gelu",
    "layer_fc1",
    "final_norm_merge",
    "merger_gemm0",
    "merger_gemm1",
    "merger_rmsnorm_apply",
)
# GEMM variant launched by each GEMM stage and the token count it runs on.
STAGE_GEMM_VARIANT = {
    "patch_embed": ("pos", "T"),
    "layer_norm_qkv_rope": ("norm_qkv_rope", "T"),
    "layer_out_proj": ("residual_wo", "T"),
    "layer_norm_fc0_gelu": ("norm_gelu", "T"),
    "layer_fc1": ("residual_fc1", "T"),
    "merger_gemm0": ("gelu_erf", "N"),
    "merger_gemm1": ("rmsnorm", "N"),
}

# Exact keyword sets of the generated ``run`` entries (bound by the export's
# argument plans); ``grid`` is expanded to ``grid_x/y/z``.
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
    "SQ",
    "M",
    "m_tiles",
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
    lens: Sequence[int], num_heads: int, tiles_per_cta: int
) -> tuple[list[tuple[int, int, int]], list[int]]:
    """Enumerate (segment, head, cluster) units segment-major (heads slow, clusters fast) with LPT costs."""
    rows_per_cluster = 2 * tiles_per_cta * ATTN_BLOCK_M
    units: list[tuple[int, int, int]] = []
    costs: list[int] = []
    for seg, length in enumerate(lens):
        blocks = (length + ATTN_BLOCK_N - 1) // ATTN_BLOCK_N
        stage_iters = blocks if tiles_per_cta == MODE_TWO_TILE else (blocks + 1) // 2
        cost = stage_iters + ATTN_UNIT_OVERHEAD_BLOCKS
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


def lpt_makespan(costs: Sequence[int], grid_clusters: int) -> tuple[int, list[int]]:
    """(max per-cluster cost, slot -> unit) of the kernel's LPT slot assignment on ``grid_clusters`` slots."""
    if not costs:
        return 0, []
    num_clusters = min(grid_clusters, len(costs))
    slots = assign_unit_slots(list(costs), num_clusters)
    per_cluster = [0] * num_clusters
    for slot, unit in enumerate(slots):
        per_cluster[slot % num_clusters] += costs[unit]
    return max(per_cluster), slots


def select_tiles_per_cta(
    lens: Sequence[int], num_heads: int, grid_clusters: int
) -> dict[str, Any]:
    """Unit layout from the LPT makespans on this device: two tiles per CTA unless
    SPLIT_KV (one tile per CTA, both softmax stages on the K/V parities) wins by
    ``SPLIT_MARGIN``.  Mirrors the Cake production planner's ``_select_tiles_per_cta``."""
    makespan: dict[int, int] = {}
    for mode in (MODE_TWO_TILE, MODE_SPLIT_KV):
        _units, costs = _attention_unit_costs(lens, num_heads, mode)
        makespan[mode], _slots = lpt_makespan(costs, grid_clusters)
    two, split = makespan[MODE_TWO_TILE], makespan[MODE_SPLIT_KV]
    chosen = MODE_SPLIT_KV if split < two * (1.0 - SPLIT_MARGIN) else MODE_TWO_TILE
    return {"tiles_per_cta": chosen, "makespan": makespan}


@dataclass(frozen=True)
class AttentionPlan:
    """Host segment plan of the vision attention kernel (non-empty segments only).

    A *unit* is one head x one cluster tile (``2 * tiles_per_cta * 128``
    consecutive Q rows of one segment); ``total_tiles`` units run on a
    persistent grid of ``num_clusters`` 2-CTA clusters.  ``unit_table`` holds
    two int32 per persistent-grid slot -- the segment index and
    ``head << 16 | cluster_in_segment`` -- in slot order (slot ``k * G + i`` is
    the ``k``-th unit of cluster ``i``).
    """

    cu_seqlens: tuple[int, ...]
    num_heads: int
    num_segments: int
    tiles_per_cta: int
    grid_clusters: int
    makespan: dict[int, int]
    total_clusters: int
    total_tiles: int
    num_clusters: int
    seg_begin: torch.Tensor
    seg_len: torch.Tensor
    unit_table: torch.Tensor


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
) -> AttentionPlan:
    """Build the attention segment plan (tables on ``device``).

    ``grid_clusters`` defaults to :func:`attention_grid_clusters` of ``device``
    (which must then be a CUDA device); ``tiles_per_cta`` forces a unit layout
    (2 = two-tile, 1 = SPLIT_KV) instead of the runtime makespan rule.
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
    if grid_clusters is None:
        grid_clusters = attention_grid_clusters(device)
    grid_clusters = max(1, int(grid_clusters))
    segments = [(a, b - a) for a, b in zip(cu, cu[1:], strict=False) if b > a]
    begins = [a for a, _ in segments]
    lens = [length for _, length in segments]
    selection = select_tiles_per_cta(lens, num_heads, grid_clusters)
    mode = (
        int(tiles_per_cta)
        if tiles_per_cta is not None
        else int(selection["tiles_per_cta"])
    )
    if mode not in (MODE_TWO_TILE, MODE_SPLIT_KV):
        raise ValueError(
            f"tiles_per_cta must be {MODE_TWO_TILE} or {MODE_SPLIT_KV}, got {mode}"
        )
    units, costs = _attention_unit_costs(lens, num_heads, mode)
    total_tiles = len(units)
    num_clusters = min(grid_clusters, max(total_tiles, 1))
    _makespan, slots = lpt_makespan(costs, num_clusters)
    table: list[int] = []
    for unit in slots:
        seg, head, c = units[unit]
        table.extend((seg, (head << 16) | c))
    rows_per_cluster = 2 * mode * ATTN_BLOCK_M
    return AttentionPlan(
        cu_seqlens=cu,
        num_heads=num_heads,
        num_segments=len(segments),
        tiles_per_cta=mode,
        grid_clusters=grid_clusters,
        makespan=dict(selection["makespan"]),
        total_clusters=sum(
            (length + rows_per_cluster - 1) // rows_per_cluster for length in lens
        ),
        total_tiles=total_tiles,
        num_clusters=num_clusters,
        seg_begin=_table(begins, device),
        seg_len=_table(lens, device),
        unit_table=torch.tensor(table or [0, 0], dtype=torch.int32, device=device),
    )


# ---------------------------------------------------------------------------
# GEMM tile configurations and launch geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileConfig:
    name: str
    cta_group: int
    acc_n: int
    num_stages: int
    group_m: int
    ksplit: int = 1
    epi_warps: int = 4

    @property
    def cluster_x(self) -> int:
        return self.cta_group * self.ksplit


TILE_CONFIGS: dict[str, TileConfig] = {
    c.name: c
    for c in (
        TileConfig("l", cta_group=2, acc_n=256, num_stages=7, group_m=32),
        TileConfig("s", cta_group=1, acc_n=128, num_stages=7, group_m=32),
        TileConfig("xs", cta_group=1, acc_n=64, num_stages=9, group_m=32),
        TileConfig("s_k2", cta_group=1, acc_n=128, num_stages=7, group_m=32, ksplit=2),
        TileConfig("s_k4", cta_group=1, acc_n=128, num_stages=7, group_m=32, ksplit=4),
        TileConfig("xs_k2", cta_group=1, acc_n=64, num_stages=9, group_m=32, ksplit=2),
        TileConfig("xs_k4", cta_group=1, acc_n=64, num_stages=9, group_m=32, ksplit=4),
        TileConfig("m", cta_group=2, acc_n=128, num_stages=9, group_m=32),
        TileConfig("w", cta_group=1, acc_n=256, num_stages=4, group_m=32),
        TileConfig("l_g8", cta_group=2, acc_n=256, num_stages=7, group_m=8),
        TileConfig(
            "l_e8", cta_group=2, acc_n=256, num_stages=7, group_m=32, epi_warps=8
        ),
        TileConfig(
            "s_e8", cta_group=1, acc_n=128, num_stages=7, group_m=32, epi_warps=8
        ),
    )
}

# variant -> (N, K, pos-split pixel-row maps)
GEMM_VARIANTS: dict[str, tuple[int, int, bool]] = {
    "pos": (HIDDEN, PATCH_DIM_PAD, True),
    "norm_qkv_rope": (QKV_N, HIDDEN, False),
    "residual_wo": (HIDDEN, QKV_HIDDEN, False),
    "residual_fc1": (HIDDEN, FFN, False),
    "norm_gelu": (FFN, HIDDEN, False),
    "gelu_erf": (MERGED_DIM, MERGED_DIM, False),
    "rmsnorm": (TEXT_HIDDEN, MERGED_DIM, False),
}
# Production tile selection thresholds (kimi_k3_vision_gemm.select_tile_config).
TINY_M_LIMIT = 256
SMALL_M_LIMIT = 1024
MID_M_LIMIT = 8192


def select_tile_config(variant: str, M: int) -> TileConfig:
    """Production tile config for ``(variant, M)``; mirrors ``kimi_k3_vision_gemm.select_tile_config``."""
    n_total, k_total, pos_split = GEMM_VARIANTS[variant]
    cfg = TILE_CONFIGS
    if pos_split:
        return cfg["xs"] if M <= SMALL_M_LIMIT else cfg["s_e8"]
    if n_total == HIDDEN:  # residual_wo (K = 1536), residual_fc1 (K = 4096)
        if M <= TINY_M_LIMIT and k_total == FFN:
            return cfg["xs_k4"]
        if M <= SMALL_M_LIMIT:
            return cfg["xs"]
        return cfg["l_e8"] if (k_total == FFN and M <= MID_M_LIMIT) else cfg["m"]
    if M <= TINY_M_LIMIT:
        return cfg["s"] if n_total == TEXT_HIDDEN else cfg["xs"]
    if M <= SMALL_M_LIMIT:
        return cfg["s"]
    return cfg["l"]


def launch_tile_config(variant: str, cfg: TileConfig) -> TileConfig:
    """The physical config the launcher runs: ``pos`` requires single-CTA tiles and a K split of 10 K-steps."""
    if GEMM_VARIANTS[variant][2]:
        if cfg.cta_group != 1:
            cfg = TILE_CONFIGS["s"]
        if (PATCH_DIM_PAD // GEMM_BLOCK_K) % cfg.ksplit:
            cfg = TILE_CONFIGS["s_k2"] if cfg.acc_n == 128 else TILE_CONFIGS["xs_k2"]
    return cfg


def gemm_launch_geometry(
    variant: str, cfg: TileConfig, M: int, sm_count: int
) -> tuple[tuple[int, int, int], int]:
    """``(grid, m_tiles)`` of one GEMM launch; mirrors ``kimi_k3_vision_gemm._launch_gemm``."""
    n_total, _k_total, pos_split = GEMM_VARIANTS[variant]
    if pos_split:
        m_tiles = 2 * ((M + 2 * GEMM_BLOCK_M - 1) // (2 * GEMM_BLOCK_M))
    else:
        m_tiles = (M + GEMM_BLOCK_M - 1) // GEMM_BLOCK_M
        m_tiles += m_tiles % cfg.cta_group
    n_tiles = n_total // cfg.acc_n
    cluster_tiles = (m_tiles // cfg.cta_group) * n_tiles
    if cfg.ksplit > 1:
        clusters = cluster_tiles  # non-persistent: one cluster per output tile
    else:
        clusters = min(cluster_tiles, int(sm_count) // cfg.cta_group)
    return (clusters * cfg.cluster_x, 1, 1), m_tiles


def gemm_configs_for(total_tokens: int, merged: int) -> dict[str, str]:
    """Physical tile config name per GEMM variant for one ``(T, N)``."""
    configs: dict[str, str] = {}
    for variant in GEMM_VARIANTS:
        count = total_tokens if variant not in ("gelu_erf", "rmsnorm") else merged
        configs[variant] = launch_tile_config(
            variant, select_tile_config(variant, count)
        ).name
    return configs


def required_kernel_keys() -> tuple[str, ...]:
    """Every logical kernel the production plan can select (all token-count buckets)."""
    keys: list[str] = []
    for variant in GEMM_VARIANTS:
        for count in (1, TINY_M_LIMIT + 1, SMALL_M_LIMIT + 1, MID_M_LIMIT + 1):
            key = gemm_kernel_key(
                variant,
                launch_tile_config(variant, select_tile_config(variant, count)).name,
            )
            if key not in keys:
                keys.append(key)
    keys.extend(
        (
            attention_kernel_key(MODE_TWO_TILE),
            attention_kernel_key(MODE_SPLIT_KV),
            MERGE_KERNEL_KEY,
            RMSNORM_APPLY_KERNEL_KEY,
        )
    )
    return tuple(keys)


REQUIRED_KERNEL_KEYS = required_kernel_keys()


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
    from a 16-byte-aligned start four elements early).  Per layer
    ``wqkv_folded = bf16(wqkv * norm0[None, :])`` and ``fc0_folded = bf16(fc0 *
    norm1[None, :])``; ``wo`` / ``fc1`` are the checkpoint tensors.
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
    """Fold the RMSNorm weights into ``wqkv``/``fc0`` and pad ``patch_proj`` (once per model).

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

    def fold(w: torch.Tensor, norm_w: torch.Tensor) -> torch.Tensor:
        return (w.float() * norm_w.float()[None, :]).to(torch.bfloat16).contiguous()

    layers = []
    for index, lw in enumerate(weights["layers"]):
        missing = [k for k in LAYER_WEIGHT_KEYS if k not in lw]
        if missing:
            raise ValueError(f"weights['layers'][{index}] lacks {missing}")
        layers.append(
            {
                "wqkv_folded": fold(
                    _check_weight(lw["wqkv"], (QKV_N, HIDDEN), f"layers[{index}].wqkv"),
                    _check_weight(lw["norm0"], (HIDDEN,), f"layers[{index}].norm0"),
                ),
                "wo": _check_weight(
                    lw["wo"], (HIDDEN, QKV_HIDDEN), f"layers[{index}].wo"
                ),
                "fc0_folded": fold(
                    _check_weight(lw["fc0"], (FFN, HIDDEN), f"layers[{index}].fc0"),
                    _check_weight(lw["norm1"], (HIDDEN,), f"layers[{index}].norm1"),
                ),
                "fc1": _check_weight(lw["fc1"], (HIDDEN, FFN), f"layers[{index}].fc1"),
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
    """True when this checkout registers every kernel the plan can select for ``device``."""
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))
    return arch is not None and route_available(arch, REQUIRED_KERNEL_KEYS)


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
    gemm_configs: dict[str, str]
    cos: torch.Tensor
    sin: torch.Tensor
    workspace: dict[str, torch.Tensor] = field(repr=False)

    @property
    def tiles_per_cta(self) -> int:
        return self.attention.tiles_per_cta


def vision_workspace_shapes(
    total_tokens: int, merged: int
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    bf16 = torch.bfloat16
    return {
        "x": ((total_tokens, HIDDEN), bf16),
        "q": ((total_tokens, HEADS, HEAD_DIM), bf16),
        "k": ((total_tokens, HEADS, HEAD_DIM), bf16),
        "v": ((total_tokens, HEADS, HEAD_DIM), bf16),
        "attn_out": ((total_tokens, HEADS, HEAD_DIM), bf16),
        "ffn": ((total_tokens, FFN), bf16),
        "m": ((merged, MERGED_DIM), bf16),
        "h": ((merged, MERGED_DIM), bf16),
        "rowsumsq": ((merged,), torch.float32),
        # Bound to unused pointer parameters (never dereferenced for the tiles
        # the kernels run): FP32 dummy for COS/SIN/SQ, the odd-row pixel map
        # when T == 1, and the attention kernel's diagnostic probe buffer.
        "f32_dummy": ((16,), torch.float32),
        "pixel_dummy": ((2, PATCH_DIM), bf16),
        "probe_dummy": ((PROBE_WORDS,), torch.uint64),
    }


def build_kimi_k3_vision_plan(
    grid_thws: Sequence[Sequence[int]],
    device: Union[torch.device, str],
    *,
    num_layers: int,
    grid_clusters: Optional[int] = None,
    sm_count: Optional[int] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
) -> VisionTowerPlan:
    """Segment plan, merge table, RoPE tables, tile selection and all workspaces for one batch.

    ``sm_count`` / ``grid_clusters`` default to the device's SM count (and its
    half); pass them to build a plan for another device (CPU tests).  ``cos`` /
    ``sin`` may be supplied by a runtime that caches them per grid.
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
    arch = _arch_for(device) if device.type == "cuda" else "cpu"
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
    workspace = {
        name: torch.zeros(shape, dtype=dtype, device=device)
        for name, (shape, dtype) in vision_workspace_shapes(total, merged).items()
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
        attention=build_attention_plan(cu, device, HEADS, grid_clusters=grid_clusters),
        merge_table=build_merge_table(grids, device),
        gemm_configs=gemm_configs_for(total, merged),
        cos=cos,
        sin=sin,
        workspace=workspace,
    )


# ---------------------------------------------------------------------------
# Binding to the generated argument plans
# ---------------------------------------------------------------------------


def _bind(module_name: str, kwargs: dict[str, Any]) -> tuple[Callable[..., Any], tuple]:
    """Order ``kwargs`` by the generated argument plan of ``module_name`` and load its entry."""
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
    return getattr(module, record["ffi_entry"]), tuple(arguments)


@dataclass(frozen=True)
class _Launch:
    stage: str
    layer: int  # -1 outside the encoder layers
    module: str
    kwargs: dict[str, Any] = field(repr=False)
    entry: Callable[..., Any] = field(repr=False)
    arguments: tuple = field(repr=False)

    def __call__(self) -> None:
        self.entry(*self.arguments)


@dataclass(frozen=True)
class KimiK3VisionTowerRunner:
    """The prepared launch sequence of one ``(pixel_values, grid_thws, weights, out)`` binding.

    ``launch()`` runs every kernel of the tower on the current torch stream into
    the caller-owned ``out`` with no CUDA allocation and no host
    synchronization and returns ``out``; it is CUDA-graph capturable (capture
    belongs to the caller).  Prepare a new runner when ``grid_thws``, the layer
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
        """One zero-argument launch per stage (encoder stages bound to layer 0)."""
        result: dict[str, Callable[[], None]] = {}
        for item in self.launches:
            if item.stage in result or item.layer > 0:
                continue

            def run(item: _Launch = item) -> None:
                with tvm_ffi.use_torch_stream():
                    item.entry(*item.arguments)

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
            attention_units=plan.attention.total_tiles,
            attention_clusters=plan.attention.num_clusters,
            gemm_configs=dict(plan.gemm_configs),
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
    eps: float = 0.0,
) -> _Launch:
    ws = plan.workspace
    M = int(A.shape[0])
    cfg = TILE_CONFIGS[plan.gemm_configs[variant]]
    grid, m_tiles = gemm_launch_geometry(variant, cfg, M, plan.sm_count)
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
        SQ=ws["f32_dummy"],
        M=M,
        m_tiles=int(m_tiles),
        eps=float(eps),
        grid=grid,
    )
    assert tuple(kwargs) == GEMM_KWARGS
    module = kernel_module_name(plan.arch, gemm_kernel_key(variant, cfg.name))
    entry, arguments = _bind(module, kwargs)
    return _Launch(stage, layer, module, kwargs, entry, arguments)


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
        grid=(2 * int(attn.num_clusters), 1, 1),
    )
    assert tuple(kwargs) == ATTENTION_KWARGS
    module = kernel_module_name(plan.arch, attention_kernel_key(attn.tiles_per_cta))
    entry, arguments = _bind(module, kwargs)
    return _Launch("layer_attention", layer, module, kwargs, entry, arguments)


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
    entry, arguments = _bind(module, kwargs)
    return _Launch("final_norm_merge", -1, module, kwargs, entry, arguments)


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
    entry, arguments = _bind(module, kwargs)
    return _Launch("merger_rmsnorm_apply", -1, module, kwargs, entry, arguments)


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
    launches = [
        _gemm_launch(
            "patch_embed",
            -1,
            plan,
            "pos",
            pixels2d,
            weights.patch_proj,
            C=ws["x"],
            R=pos_rows,
        )
    ]
    for index, lw in enumerate(weights.layers):
        launches.append(
            _gemm_launch(
                "layer_norm_qkv_rope",
                index,
                plan,
                "norm_qkv_rope",
                ws["x"],
                lw["wqkv_folded"],
                C=ws["q"],
                C2=ws["k"],
                C3=ws["v"],
                R=ws["x"],
                COS=plan.cos,
                SIN=plan.sin,
                eps=NORM_EPS,
            )
        )
        launches.append(_attention_launch(plan, index))
        launches.append(
            _gemm_launch(
                "layer_out_proj",
                index,
                plan,
                "residual_wo",
                ws["attn_out"].view(T, QKV_HIDDEN),
                lw["wo"],
                C=ws["x"],
                R=ws["x"],
            )
        )
        launches.append(
            _gemm_launch(
                "layer_norm_fc0_gelu",
                index,
                plan,
                "norm_gelu",
                ws["x"],
                lw["fc0_folded"],
                C=ws["ffn"],
                R=ws["x"],
                eps=NORM_EPS,
            )
        )
        launches.append(
            _gemm_launch(
                "layer_fc1",
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
    if not route_available(arch, REQUIRED_KERNEL_KEYS):
        missing = [
            key for key in REQUIRED_KERNEL_KEYS if not route_available(arch, (key,))
        ]
        raise NotImplementedError(
            f"The generated Kimi-K3 vision tower programs for {arch} are not registered in this "
            f"checkout (missing {missing}; see flashinfer-ai/flashinfer#4568)"
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
