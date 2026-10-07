"""Cake FMHA routing and validation for DCP speculative decode.

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
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from .jit.cake_dcp import DcpSpecTarget

from .utils import (
    _check_workspace_buffer_alignment,
    check_shape_dtype_device,
    get_compute_capability,
    get_device_sm_count,
)

_BLOCK_N = 128
_HEAD_DIM = 128
_D256_HEAD_DIM = 256
_BF16_PAGE_SIZE = 16
_FP8_PAGE_SIZE = 64
_MAX_NUM_SPLIT = 16
_MIN_SPLIT_LOCAL_BLOCKS = 16
_TARGET_PAIRS_PER_SPLIT = 3
_RETAIN_KV_L2_MAX_BLOCKS = 9
_FP8_MAX_NUM_SPLIT = 4
_FP8_D256_MAX_NUM_SPLIT = 16
_FP8_MIN_SPLIT_LOCAL_BLOCKS = 4
_FP8_D256_CP1_MIN_LOCAL_BLOCKS = 128
_FP8_D256_CP4_MIN_LOCAL_BLOCKS = 64
_FP8_RETAIN_KV_L2_MAX_BLOCKS = 18
_BF16_SUPPORTED_Q_LENS = (1, 2, 3, 4, 5, 6, 8)
_FP8_SUPPORTED_Q_LENS = (1, 2, 3, 4, 5, 6, 8)
_FP8_D256_SUPPORTED_Q_LENS = (1, 2, 3, 4, 5, 6, 7, 8)
_SUPPORTED_CP_WORLDS = (1, 2, 4, 8)


def get_dcp_spec_workspace_size_bytes(
    batch_size: int,
    q_len_per_req: int,
    num_qo_heads: int,
    num_split: int = _MAX_NUM_SPLIT,
    *,
    head_dim: int = _HEAD_DIM,
) -> int:
    """Bytes for Cake FMHA Split-KV BF16 partial-O and FP32 partial-LSE scratch."""

    if min(batch_size, q_len_per_req, num_qo_heads) <= 0:
        raise ValueError("batch_size, q_len_per_req, and num_qo_heads must be positive")
    if head_dim not in (_HEAD_DIM, _D256_HEAD_DIM):
        raise ValueError("Cake FMHA workspace head_dim must be 128 or 256")
    if not 2 <= num_split <= _MAX_NUM_SPLIT:
        raise ValueError(f"num_split must be in [2, {_MAX_NUM_SPLIT}]")
    partial_rows = batch_size * q_len_per_req * num_qo_heads * num_split
    return partial_rows * (head_dim * 2 + 4)


def get_dcp_spec_counter_bytes(
    batch_size: int,
    q_len_per_req: int,
    num_kv_heads: int,
) -> int:
    """Bytes for the v4 completion tickets, zeroed once then self-reset."""

    if min(batch_size, q_len_per_req, num_kv_heads) <= 0:
        raise ValueError("batch_size, q_len_per_req, and num_kv_heads must be positive")
    return batch_size * q_len_per_req * num_kv_heads * 4


def _split_workspace_views(
    *,
    workspace_buffer: torch.Tensor,
    completion_buffer: Optional[torch.Tensor],
    device: torch.device,
    batch_size: int,
    q_len_per_req: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_split: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bind caller-owned Split-KV scratch without allocating during launch."""

    required_counter = get_dcp_spec_counter_bytes(
        batch_size, q_len_per_req, num_kv_heads
    )
    if completion_buffer is None:
        raise ValueError(
            "multi_ctas_kv_counter_buffer is required for the DCP Split-KV route; "
            f"pass a zero-initialized reusable CUDA buffer with at least {required_counter} bytes"
        )
    if completion_buffer.device != device or not completion_buffer.is_contiguous():
        raise ValueError(
            "multi_ctas_kv_counter_buffer must be contiguous and on the query device"
        )
    _check_workspace_buffer_alignment(completion_buffer, "multi_ctas_kv_counter_buffer")
    completion_u8 = completion_buffer.view(torch.uint8).reshape(-1)
    if completion_u8.numel() < required_counter:
        raise ValueError(
            "multi_ctas_kv_counter_buffer is too small for DCP Split-KV: "
            f"got {completion_u8.numel()} bytes, need {required_counter}"
        )
    split_completion = completion_u8[:required_counter].view(torch.int32)

    if workspace_buffer.device != device or not workspace_buffer.is_contiguous():
        raise ValueError("workspace_buffer must be contiguous and on the query device")
    _check_workspace_buffer_alignment(workspace_buffer, "workspace_buffer")
    required_workspace = get_dcp_spec_workspace_size_bytes(
        batch_size,
        q_len_per_req,
        num_qo_heads,
        num_split,
        head_dim=head_dim,
    )
    workspace_u8 = workspace_buffer.view(torch.uint8).reshape(-1)
    if workspace_u8.numel() < required_workspace:
        raise ValueError(
            "workspace_buffer is too small for DCP Split-KV: "
            f"got {workspace_u8.numel()} bytes, need {required_workspace}"
        )
    partial_rows = batch_size * q_len_per_req * num_qo_heads * num_split
    partial_o_bytes = partial_rows * head_dim * 2
    partial_lse_bytes = partial_rows * 4
    partial_o = workspace_u8[:partial_o_bytes].view(torch.bfloat16)
    partial_lse = workspace_u8[
        partial_o_bytes : partial_o_bytes + partial_lse_bytes
    ].view(torch.float32)
    return partial_o, partial_lse, split_completion


def _static_split_instance(num_split: int) -> str:
    """Registry instance of the static split-KV programs: ``split1`` (one launch, FP8 only),
    ``split2`` (straight-line two-way merge) or ``splitn`` (``NUM_SPLIT`` >= 3 on the compile line)."""

    if num_split == 1:
        return "split1"
    return "split2" if num_split == 2 else "splitn"


def _static_constants(
    q_len: int,
    cp_world: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_split: Optional[int] = None,
) -> dict[str, int]:
    """Compile-line constants of one static DCP program (``NUM_SPLIT`` only for the split-KV families)."""

    constants = {
        "Q_LEN": int(q_len),
        "CP_WORLD": int(cp_world),
        "NUM_Q_HEADS": int(num_q_heads),
        "NUM_KV_HEADS": int(num_kv_heads),
    }
    if num_split is not None:
        constants["NUM_SPLIT"] = int(num_split)
    return constants


def _select_num_split(
    *,
    logical_tiles: int,
    sm_count: int,
    local_blocks: int,
) -> int:
    if local_blocks < _MIN_SPLIT_LOCAL_BLOCKS:
        return 1
    total_pairs = (local_blocks + 1) // 2
    work_cap = (total_pairs + _TARGET_PAIRS_PER_SPLIT - 1) // _TARGET_PAIRS_PER_SPLIT
    num_split = min(
        _MAX_NUM_SPLIT,
        total_pairs,
        work_cap,
        sm_count // logical_tiles,
    )
    return num_split if num_split >= 2 else 1


def _select_fp8_num_split(
    *,
    logical_tiles: int,
    sm_count: int,
    local_blocks: int,
    cp_world: int,
    head_dim: int = _HEAD_DIM,
) -> int:
    """Fill one SM wave while retaining two FP8 K/V block pairs per CTA."""

    if logical_tiles >= sm_count or local_blocks < _FP8_MIN_SPLIT_LOCAL_BLOCKS:
        return 1
    total_pairs = (local_blocks + 1) // 2
    work_cap = (total_pairs + 1) // 2
    max_num_split = 3 if cp_world > 1 else _FP8_MAX_NUM_SPLIT
    if (
        head_dim == _D256_HEAD_DIM
        and cp_world == 1
        and local_blocks >= _FP8_D256_CP1_MIN_LOCAL_BLOCKS
    ):
        max_num_split = _FP8_D256_MAX_NUM_SPLIT
    elif (
        head_dim == _D256_HEAD_DIM
        and cp_world == 4
        and local_blocks >= _FP8_D256_CP4_MIN_LOCAL_BLOCKS
    ):
        max_num_split = 8
    num_split = min(
        max_num_split,
        total_pairs,
        work_cap,
        sm_count // logical_tiles,
    )
    if head_dim == _D256_HEAD_DIM:
        for supported_split in (16, 8, 4, 3, 2):
            if num_split >= supported_split:
                return supported_split
        return 1
    return num_split if num_split >= 2 else 1


# ---------------------------------------------------------------------------
# On-device load-balanced DCP routes (CAKE-685 round 3)
# ---------------------------------------------------------------------------
#
# Three add-on families of the Cake export serve the DCP profiles with the
# packed-row balanced scheduler: ``dcp_spec_bf16_balanced`` (BF16 / page 16 /
# head_dim 128, GQA-8), ``dcp_spec_bf16_fp8_balanced`` (BF16 Q over an E4M3 /
# page-64 cache, head_dim 128, GQA-8) and ``dcp_spec_bf16_fp8_d256_balanced``
# (head_dim 256, GQA-16).  One persistent CTA per SM plans the split-KV
# schedule on the device from ``causal_seqlens_kv_global``, rank and world;
# batch, heads, lengths, rank and world are runtime kernel arguments, the
# counters self-reset and the workspace bound is shape independent, so a
# prepared launch replays under CUDA Graph capture for any length vector.
# The route decision below mirrors the Cake dispatcher bands
# (``bf16_p16_prefers_balanced`` / ``fp8_p64_prefers_balanced`` /
# ``fp8_d256_prefers_balanced``) and reads host metadata only -- never the
# device lengths -- so it is graph-safe.

DCP_BALANCED_KINDS = ("bf16_p16", "fp8_p64", "fp8_p64_d256")
_DCP_BALANCED_FAMILY = {
    "bf16_p16": "dcp_spec_bf16_balanced",
    "fp8_p64": "dcp_spec_bf16_fp8_balanced",
    "fp8_p64_d256": "dcp_spec_bf16_fp8_d256_balanced",
}
_DCP_BALANCED_GROUP = {"bf16_p16": 8, "fp8_p64": 8, "fp8_p64_d256": 16}
_DCP_BALANCED_PAGE_SIZE = {"bf16_p16": 16, "fp8_p64": 64, "fp8_p64_d256": 64}
_DCP_BALANCED_HEAD_DIM = {
    "bf16_p16": _HEAD_DIM,
    "fp8_p64": _HEAD_DIM,
    "fp8_p64_d256": _D256_HEAD_DIM,
}
# Architecture key of the per-arch band constants for each compile target;
# the SM107 family target maps to no measured arch and takes the defaults.
_DCP_BALANCED_ARCH = {"sm100a": "sm_100a", "sm103a": "sm_103a"}
_DCP_ROUTES = ("auto", "static", "balanced")
DCP_BALANCED_MAX_REQUESTS = 1024  # MAX_REQUEST_GROUPS * REQUEST_GROUP of the planner
DCP_BALANCED_MAX_N_ROWS = 64  # physical packed tile (speculative rows x group)
DCP_BALANCED_MAX_BALANCE_FACTOR = 8  # chunk length >= total work / (k * CTAs)
DCP_BALANCED_STATS_PER_SLOT = 2 * DCP_BALANCED_MAX_N_ROWS  # max[64] then sum[64]
DCP_BALANCED_COUNTERS_PER_TILE = 4  # arrivals, two reduce-queue words, published flag
DCP_BALANCED_QUEUE_COUNTERS = 4
DCP_BALANCED_CHUNK_TOKENS = 256  # one planner chunk pair = two 128-token blocks
DCP_BALANCED_D256_Q_BOX_ROWS = 4  # speculative rows per D256 row tile (64 / 16)
# Band constants (Cake ``trtllm_fmha_forgen_bf16_fp8.py``): a row goes to the
# balanced kernel when the static route needs a second wave of tiles and the
# planner's chunk-pair work bound reaches the items floor, or when one static
# wave streams at least the long-tile block count per CTA.
DCP_BALANCED_BF16_MIN_Q_LEN = 3  # q_len 1-2 stay static (<= 16 live rows per tile)
DCP_BALANCED_BF16_MAX_Q_LEN = 8
DCP_BALANCED_BF16_MIN_ITEMS = 128
DCP_BALANCED_BF16_LONG_TILE_BLOCKS = 16
# unit 87: a single request within the whole-tile bound of the BF16 family runs its whole-tile static program instead of
# one static wave, per architecture and packed-row instance (the 32-row instance measured faster on both GPUs, the 64-row
# instance only on GB300; see the band probe in the design document)
DCP_BALANCED_BF16_WHOLE_TILE_N_ROWS = {"sm_100a": (32,), "sm_103a": (32, 64)}
DCP_BALANCED_FP8_MIN_Q_LEN = 3
DCP_BALANCED_FP8_MAX_Q_LEN = 8
DCP_BALANCED_FP8_MIN_ITEMS = 160
# Round-5 programs, long-tile floor per architecture: the round-3 fit of 24
# left the 17- and 22-block one-wave rows static.  The 22-block row
# (b1/S32768 cp4) wins on both parts (1.15 GB300 / 1.11 B200 vs the static
# route); the 17-block rows win on sm_103a (b1/S24576 cp4 1.055, cp1 b1/S8192
# 1.053) and are a tie band on sm_100a (0.98 / 0.99), so sm_103a admits 17
# blocks and sm_100a keeps that class static (floor 18, inside the unmeasured
# window (17, 22]).  The scalar is the default for unmeasured targets.
DCP_BALANCED_FP8_LONG_TILE_BLOCKS = 18
DCP_BALANCED_FP8_LONG_TILE_BLOCKS_BY_ARCH = {"sm_100a": 18, "sm_103a": 17}
# Two-wave floor per architecture.  Round 3 fitted sm_100a to 384 items (the
# 320-item two-wave row prod_b8_s4096_q4_cp4 ran static 5-9 % faster on B200);
# the round-5 bodies win that row on both parts (1.29 GB300 / 1.17 B200), so
# both floors sit at the family's items floor.  The per-architecture form is
# kept: it is the manifest's and the Cake dispatcher's contract.
DCP_BALANCED_FP8_TWO_WAVE_MIN_ITEMS = {"sm_100a": 160, "sm_103a": 160}
DCP_BALANCED_D256_MIN_Q_LEN = 1
DCP_BALANCED_D256_MAX_Q_LEN = 8
DCP_BALANCED_D256_MIN_ITEMS = 160
DCP_BALANCED_D256_LONG_TILE_BLOCKS = 96
# One-wave row-tile regime of the D256 family (round 5): the static D256 route
# streams a request's KV once per speculative row (one Q16 tile per (request,
# row)), the balanced row tile of up to four rows streams it once per tile, so
# one static wave at q_len >= 3 re-reads every request's KV 2.5-4x and the
# balanced program wins once the static tile streams 22 or more blocks per
# CTA (round-5 band probe, static / forced balanced, GB300 | B200: q4 b12 at
# 22 blocks 1.11 | 1.14, b16 1.22 | 1.28, b32 1.42 | 1.43; q3 b16 at 22
# blocks 1.07 | 1.06; q5 / q8 1.45-1.79).  Below 22 blocks the plan + fold are
# not amortised (q4 b8 at 16 blocks 1.00 | 1.04 tie band, b4 / b1 0.80 /
# 0.90); q_len 1 reads KV once on both routes (0.89-0.97) and q_len 2 gains
# at most 4-6 % (16 blocks: 0.91-0.94), both stay static.  Per-architecture
# floor (the parts agree); the scalar is the default for unmeasured targets.
DCP_BALANCED_D256_ONE_WAVE_MIN_Q_LEN = 3
DCP_BALANCED_D256_ONE_WAVE_LONG_TILE_BLOCKS = 22
DCP_BALANCED_D256_ONE_WAVE_LONG_TILE_BLOCKS_BY_ARCH = {"sm_100a": 22, "sm_103a": 22}
_DCP_BALANCED_Q_LEN_RANGE = {
    "bf16_p16": (DCP_BALANCED_BF16_MIN_Q_LEN, DCP_BALANCED_BF16_MAX_Q_LEN),
    "fp8_p64": (DCP_BALANCED_FP8_MIN_Q_LEN, DCP_BALANCED_FP8_MAX_Q_LEN),
    "fp8_p64_d256": (DCP_BALANCED_D256_MIN_Q_LEN, DCP_BALANCED_D256_MAX_Q_LEN),
}
_DCP_BALANCED_MIN_ITEMS = {
    "bf16_p16": DCP_BALANCED_BF16_MIN_ITEMS,
    "fp8_p64": DCP_BALANCED_FP8_MIN_ITEMS,
    "fp8_p64_d256": DCP_BALANCED_D256_MIN_ITEMS,
}
_DCP_BALANCED_LONG_TILE_BLOCKS = {
    "bf16_p16": DCP_BALANCED_BF16_LONG_TILE_BLOCKS,
    "fp8_p64": DCP_BALANCED_FP8_LONG_TILE_BLOCKS,
    "fp8_p64_d256": DCP_BALANCED_D256_LONG_TILE_BLOCKS,
}


def _check_dcp_balanced_kind(kind: str) -> None:
    if kind not in DCP_BALANCED_KINDS:
        raise ValueError(
            f"balanced DCP kind must be one of {DCP_BALANCED_KINDS}, got {kind!r}"
        )


def get_dcp_spec_balanced_workspace_bytes(
    sm_count: int, head_dim: int = _HEAD_DIM
) -> int:
    """Bytes of ``workspace_buffer`` the balanced DCP routes carve (shape independent).

    Per split item one FP32 partial tile ``[64, head_dim]`` and one statistics
    slot (row maxima then row sums), plus the reserved slot that receives the
    device planner's plan facts; ``2 * 8 * sm_count`` split items bound every
    shape (the Cake planner's ``workspace_bounds``).  May be uninitialized.
    """

    if sm_count <= 0:
        raise ValueError("sm_count must be positive")
    if head_dim not in (_HEAD_DIM, _D256_HEAD_DIM):
        raise ValueError("balanced DCP workspace head_dim must be 128 or 256")
    max_split_items = 2 * DCP_BALANCED_MAX_BALANCE_FACTOR * sm_count
    partial_o_bytes = max_split_items * DCP_BALANCED_MAX_N_ROWS * head_dim * 4
    stats_offset = (partial_o_bytes + 255) // 256 * 256
    return stats_offset + (max_split_items + 1) * DCP_BALANCED_STATS_PER_SLOT * 4


def get_dcp_spec_balanced_counter_bytes(sm_count: int) -> int:
    """Zero-initialized ``multi_ctas_kv_counter_buffer`` bytes of the balanced DCP routes.

    Four words per split tile (chunk arrivals, the two reduce-queue words and
    the two-chunk published flag) for ``8 * sm_count`` tiles, then the four
    16-byte-aligned queue counters.  Every counter the kernel touched reads
    zero again when it exits, so one buffer zeroed at allocation serves every
    launch and is shared with the static split route's completion tickets.
    """

    if sm_count <= 0:
        raise ValueError("sm_count must be positive")
    max_split_tiles = DCP_BALANCED_MAX_BALANCE_FACTOR * sm_count
    tile_bytes = max_split_tiles * DCP_BALANCED_COUNTERS_PER_TILE * 4
    return (tile_bytes + 15) // 16 * 16 + DCP_BALANCED_QUEUE_COUNTERS * 4


def dcp_balanced_q_tiles(kind: str, q_len: int) -> int:
    """Row tiles per request: one on the D128 families, ``ceil(q_len / 4)`` on D256."""

    _check_dcp_balanced_kind(kind)
    if q_len <= 0:
        raise ValueError(f"q_len must be positive, got {q_len}")
    if kind != "fp8_p64_d256":
        return 1
    return -(-q_len // DCP_BALANCED_D256_Q_BOX_ROWS)


def dcp_balanced_n_rows(kind: str, q_len: int) -> int:
    """Packed-row instance (32 or 64 live rows) serving ``q_len`` speculative rows.

    D128 families pack ``q_len * 8`` rows per (request, KV head); the D256
    family packs ``min(q_len, 4) * 16`` rows per row tile.
    """

    _check_dcp_balanced_kind(kind)
    if not 1 <= q_len <= 8:
        raise ValueError(f"balanced DCP q_len must be in [1, 8], got {q_len}")
    rows_per_tile = (
        min(q_len, DCP_BALANCED_D256_Q_BOX_ROWS) if kind == "fp8_p64_d256" else q_len
    )
    return 32 if rows_per_tile * _DCP_BALANCED_GROUP[kind] <= 32 else 64


def dcp_balanced_program(
    kind: str,
    *,
    batch_size: int,
    num_kv_heads: int,
    max_pages_per_seq: int,
    sm_count: int,
    arch: Optional[str] = None,
    n_rows: Optional[int] = None,
) -> Optional[int]:
    """The traced program a balanced family's launch runs, or ``None`` for a one-program family.

    The E4M3 head_dim-128 family ships three programs.  At or above the grid
    (``batch_size * num_kv_heads >= sm_count``: one persistent CTA per
    multiprocessor and every (request, KV head) pair at least one chunk ticket)
    the idle-CTA eight-slice fold of split tiles can never be taken, so the
    launch runs the program without that fold body (``at_or_above_grid``;
    identical plan and fold order).  Below the grid, a launch whose page-table
    width (``max_pages_per_seq``, the ``block_tables`` width) bounds every pair
    to ``n_max >= min_chunks`` chunks of ``chunk_tokens`` and whose
    ``batch_size * num_kv_heads * (n_max + reduce_tickets_per_tile)`` tickets
    fit the grid runs the plan-free static one-wave program
    (``static_one_wave``: ticket = CTA index, no device planner); every other
    launch runs the default planner program (``below_grid``).  The BF16
    head_dim-128 family ships two programs under the regime's whole-tile form
    (``form = whole_tiles``): a single request whose ``block_tables`` width is
    within ``whole_pages_max[sm_count][num_kv_heads]`` runs the whole-tile
    static program (one ticket per (request, KV head) tile, no device planner,
    no partials; the planner itself plans whole tiles for every length within
    that bound, so the output is bitwise the planner program's), every other
    launch the planner program.  Where the regime names a ``swapped`` form, a
    whole-tile launch on one of its architectures (``arch``: the compile
    target's architecture key, ``sm_100a`` / ``sm_103a``) with one of its
    packed-row instances (``n_rows``) runs that program instead: the
    swapped-QK whole-tile program (S^T = K Q^T with the keys on M, lane-per-key
    softmax, two P^T staging buffers; bitwise the whole-tile program's
    output).  An unknown architecture or instance keeps the whole-tile
    program.  Where the regime names an ``early_issue`` block, a whole-tile
    launch on one of its architectures runs the early-issue form of its
    program (``programs[program]``: the loader decodes its own ticket in the
    kernel prologue and issues the first Q / K / V boxes before the
    scheduler's token; bitwise the program's output).  Mirrors the manifest's
    ``program_variants`` rule from host metadata only.
    """

    _check_dcp_balanced_kind(kind)
    from .jit.cake_dcp import dcp_balanced_program_variants

    variants = dcp_balanced_program_variants(_DCP_BALANCED_FAMILY[kind])
    if variants is None:
        return None
    if variants.get("items_lower_bound") != "batch_size * num_kv_heads":
        raise RuntimeError(
            f"unsupported balanced DCP program rule for {kind}: "
            f"{variants.get('items_lower_bound')!r}"
        )
    if int(sm_count) <= 0:
        raise ValueError(f"sm_count must be positive, got {sm_count}")
    if int(batch_size) <= 0 or int(num_kv_heads) <= 0:
        raise ValueError("batch_size and num_kv_heads must be positive")
    if int(max_pages_per_seq) <= 0:
        raise ValueError(f"max_pages_per_seq must be positive, got {max_pages_per_seq}")
    tiles = int(batch_size) * int(num_kv_heads)
    if tiles >= int(sm_count):
        return int(variants["at_or_above_grid"])
    regime = variants["static_one_wave_regime"]
    if regime.get("form", "split_chunks") == "whole_tiles":
        # The whole-tile form (the BF16 head_dim-128 family): one request whose
        # block-table width stays within ``whole_pages_max[sm_count][num_kv_heads]``
        # (the widest bound for which the planner itself plans every tile whole)
        # runs the static program; everything else, including SM counts or KV-head
        # counts outside the table, runs the planner program.
        limit = (
            regime["whole_pages_max"]
            .get(str(int(sm_count)), {})
            .get(str(int(num_kv_heads)))
        )
        if (
            int(batch_size) == 1
            and int(regime.get("one_request", 1)) == 1
            and limit is not None
            and int(max_pages_per_seq) <= int(limit)
        ):
            swapped = regime.get("swapped")
            program = int(variants["static_one_wave"])
            if (
                swapped is not None
                and arch in swapped["arches"]
                and n_rows is not None
                and int(n_rows) in [int(value) for value in swapped["n_rows"]]
            ):
                program = int(swapped["program"])
            early = regime.get("early_issue")
            if early is not None and arch in early["arches"]:
                # the early-issue form of the whole-tile program on its architectures (CAKE-685 unit 85e)
                program = int(early["programs"].get(str(program), program))
            return program
        return int(variants["below_grid"])
    n_max = -(
        -int(max_pages_per_seq)
        * int(regime["page_size"])
        // int(regime["chunk_tokens"])
    )
    if n_max >= int(regime["min_chunks"]) and tiles * (
        n_max + int(regime["reduce_tickets_per_tile"])
    ) <= int(sm_count):
        return int(variants["static_one_wave"])
    return int(variants["below_grid"])


def _dcp_whole_tile_one_wave(
    kind: str, *, num_kv_heads: int, max_pages_per_seq: int, sm_count: int
) -> bool:
    """Does a single request with this page-table width run the family's whole-tile
    static program (manifest regime ``form = whole_tiles``,
    ``whole_pages_max[sm_count][num_kv_heads]``)?  ``False`` for families without
    the whole-tile form, SM counts or KV-head counts outside the table."""

    from .jit.cake_dcp import dcp_balanced_program_variants

    variants = dcp_balanced_program_variants(_DCP_BALANCED_FAMILY[kind])
    if variants is None:
        return False
    regime = variants.get("static_one_wave_regime") or {}
    if regime.get("form", "split_chunks") != "whole_tiles":
        return False
    limit = (
        regime["whole_pages_max"]
        .get(str(int(sm_count)), {})
        .get(str(int(num_kv_heads)))
    )
    return (
        limit is not None
        and int(regime.get("one_request", 1)) == 1
        and 0 < int(max_pages_per_seq) <= int(limit)
    )


def dcp_balanced_items_bound(
    *, batch_size: int, num_kv_heads: int, max_local_seq_len: int
) -> int:
    """Upper bound on the balanced planner's chunk-pair items from host metadata only."""

    pairs = max(
        1,
        (int(max_local_seq_len) + DCP_BALANCED_CHUNK_TOKENS - 1)
        // DCP_BALANCED_CHUNK_TOKENS,
    )
    return int(batch_size) * int(num_kv_heads) * pairs


def dcp_static_shape(
    kind: str,
    *,
    batch_size: int,
    q_len: int,
    num_kv_heads: int,
    max_local_seq_len: int,
    cp_world: int,
    sm_count: int,
) -> tuple[int, int, int]:
    """``(num_split, waves, blocks_per_cta)`` of the static route for the same host metadata.

    The static routes are this module's own ``_select_num_split`` (BF16 v1 /
    v4) and ``_select_fp8_num_split`` (FP8 D128 / D256) specializations; the
    Cake dispatcher's shape helpers evaluate the same rules.
    """

    _check_dcp_balanced_kind(kind)
    if min(batch_size, q_len, num_kv_heads, sm_count) <= 0:
        raise ValueError(
            "batch_size, q_len, num_kv_heads and sm_count must be positive"
        )
    tiles = int(batch_size) * int(q_len) * int(num_kv_heads)
    local_blocks = max(1, (int(max_local_seq_len) + _BLOCK_N - 1) // _BLOCK_N)
    if kind == "bf16_p16":
        num_split = _select_num_split(
            logical_tiles=tiles, sm_count=sm_count, local_blocks=local_blocks
        )
    else:
        num_split = _select_fp8_num_split(
            logical_tiles=tiles,
            sm_count=sm_count,
            local_blocks=local_blocks,
            cp_world=cp_world,
            head_dim=_DCP_BALANCED_HEAD_DIM[kind],
        )
    waves = -(-(tiles * num_split) // int(sm_count))
    blocks_per_cta = -(-local_blocks // num_split)
    return num_split, waves, blocks_per_cta


@dataclass(frozen=True)
class DcpBalancedBand:
    """Route decision of one DCP row with the static geometry it was made from."""

    kind: str
    route: str  # "balanced" or "static"
    reason: str
    num_split: int
    waves: int
    blocks_per_cta: int
    items: int


def dcp_balanced_band(
    kind: str,
    *,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_local_seq_len: int,
    cp_world: int,
    sm_count: int,
    arch: str,
    max_pages_per_seq: Optional[int] = None,
) -> DcpBalancedBand:
    """Host-metadata band of the balanced DCP kernels (mirror of the Cake dispatcher).

    ``balanced`` requires the kernel's contract (head_dim and query heads per
    KV head of the family, its q_len range, at most ``DCP_BALANCED_MAX_REQUESTS``
    requests) and one of the measured regimes: the static route needs a
    second wave of tiles and the chunk-pair work bound reaches the items floor
    (times the row tiles per request on D256), or one static wave streams the
    long-tile block count or more per CTA (per ``arch`` on the FP8 D128
    family), or (D256 only) one static wave whose speculative rows share a
    balanced row tile (``q_len >= DCP_BALANCED_D256_ONE_WAVE_MIN_Q_LEN``)
    streams the ``arch``'s one-wave long-tile block count or more per CTA.  On
    the FP8 D128 family a row at exactly two static waves must reach the
    ``arch``'s two-wave items floor.  On the BF16 family one static wave
    serving a single request whose page-table width (``max_pages_per_seq``,
    the ``block_tables`` width; the narrowest table for ``max_local_seq_len``
    when not given) is within the whole-tile bound runs the whole-tile static
    program when the ``arch`` lists the row's packed-row instance in
    ``DCP_BALANCED_BF16_WHOLE_TILE_N_ROWS`` (``balanced`` /
    ``whole_tile_one_wave``).  ``arch`` is the compile target's
    architecture key (``sm_100a`` / ``sm_103a``); other keys take the scalar
    defaults.
    """

    _check_dcp_balanced_kind(kind)
    if cp_world not in _SUPPORTED_CP_WORLDS:
        raise ValueError(f"cp_world must be one of {_SUPPORTED_CP_WORLDS}")
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    num_split, waves, blocks_per_cta = dcp_static_shape(
        kind,
        batch_size=batch_size,
        q_len=q_len,
        num_kv_heads=num_kv_heads,
        max_local_seq_len=max_local_seq_len,
        cp_world=cp_world,
        sm_count=sm_count,
    )
    items = dcp_balanced_items_bound(
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        max_local_seq_len=max_local_seq_len,
    ) * dcp_balanced_q_tiles(kind, q_len)

    def decide(route: str, reason: str) -> DcpBalancedBand:
        return DcpBalancedBand(
            kind, route, reason, num_split, waves, blocks_per_cta, items
        )

    if (
        head_dim != _DCP_BALANCED_HEAD_DIM[kind]
        or num_q_heads != _DCP_BALANCED_GROUP[kind] * num_kv_heads
    ):
        return decide("static", "group")
    min_q_len, max_q_len = _DCP_BALANCED_Q_LEN_RANGE[kind]
    if not min_q_len <= q_len <= max_q_len:
        return decide("static", "q_len")
    if batch_size > DCP_BALANCED_MAX_REQUESTS:
        return decide("static", "batch")
    long_tile_blocks = _DCP_BALANCED_LONG_TILE_BLOCKS[kind]
    if kind == "fp8_p64":
        long_tile_blocks = DCP_BALANCED_FP8_LONG_TILE_BLOCKS_BY_ARCH.get(
            arch, long_tile_blocks
        )
    if blocks_per_cta >= long_tile_blocks:
        return decide("balanced", "long_tile")
    if waves < 2:
        if (
            kind == "fp8_p64_d256"
            and q_len >= DCP_BALANCED_D256_ONE_WAVE_MIN_Q_LEN
            and blocks_per_cta
            >= DCP_BALANCED_D256_ONE_WAVE_LONG_TILE_BLOCKS_BY_ARCH.get(
                arch, DCP_BALANCED_D256_ONE_WAVE_LONG_TILE_BLOCKS
            )
        ):
            return decide("balanced", "one_wave_row_tiles")
        if (
            kind == "bf16_p16"
            and int(batch_size) == 1
            and dcp_balanced_n_rows(kind, q_len)
            in DCP_BALANCED_BF16_WHOLE_TILE_N_ROWS.get(arch, ())
            and _dcp_whole_tile_one_wave(
                kind,
                num_kv_heads=num_kv_heads,
                max_pages_per_seq=(
                    int(max_pages_per_seq)
                    if max_pages_per_seq is not None
                    else max(
                        1, -(-int(max_local_seq_len) // _DCP_BALANCED_PAGE_SIZE[kind])
                    )
                ),
                sm_count=sm_count,
            )
        ):
            return decide("balanced", "whole_tile_one_wave")
        return decide("static", "one_wave")
    if items < _DCP_BALANCED_MIN_ITEMS[kind]:
        return decide("static", "items")
    if kind == "fp8_p64" and waves == 2:
        floor = DCP_BALANCED_FP8_TWO_WAVE_MIN_ITEMS.get(
            arch, DCP_BALANCED_FP8_MIN_ITEMS
        )
        if items < floor:
            return decide("static", "two_wave_floor")
        return decide("balanced", "two_waves")
    return decide("balanced", "waves")


def dcp_balanced_route(
    kind: str,
    *,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_local_seq_len: int,
    cp_world: int,
    sm_count: int,
    arch: str,
    max_pages_per_seq: Optional[int] = None,
) -> str:
    """``"balanced"`` or ``"static"`` for one DCP row (see :func:`dcp_balanced_band`)."""

    return dcp_balanced_band(
        kind,
        batch_size=batch_size,
        q_len=q_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_local_seq_len=max_local_seq_len,
        cp_world=cp_world,
        sm_count=sm_count,
        arch=arch,
        max_pages_per_seq=max_pages_per_seq,
    ).route


def _dcp_balanced_buffer_problem(
    workspace_buffer: torch.Tensor,
    completion_buffer: Optional[torch.Tensor],
    device: torch.device,
    sm_count: int,
    head_dim: int,
) -> Optional[str]:
    """Why the caller-owned scratch cannot serve the balanced route (``None`` if it can)."""

    required_counter = get_dcp_spec_balanced_counter_bytes(sm_count)
    required_workspace = get_dcp_spec_balanced_workspace_bytes(sm_count, head_dim)
    if completion_buffer is None:
        return (
            "multi_ctas_kv_counter_buffer is required for the balanced DCP route; "
            f"pass a zero-initialized reusable CUDA buffer with at least {required_counter} bytes"
        )
    if completion_buffer.device != device or not completion_buffer.is_contiguous():
        return "multi_ctas_kv_counter_buffer must be contiguous and on the query device"
    counter_bytes = completion_buffer.numel() * completion_buffer.element_size()
    if counter_bytes < required_counter:
        return (
            "multi_ctas_kv_counter_buffer is too small for the balanced DCP route: "
            f"got {counter_bytes} bytes, need {required_counter}"
        )
    if workspace_buffer.device != device or not workspace_buffer.is_contiguous():
        return "workspace_buffer must be contiguous and on the query device"
    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    if workspace_bytes < required_workspace:
        return (
            "workspace_buffer is too small for the balanced DCP route: "
            f"got {workspace_bytes} bytes, need {required_workspace}"
        )
    return None


def _run_dcp_spec_balanced(
    *,
    kind: str,
    target: str,
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    block_tables: torch.Tensor,
    causal_seqlens_kv_global: torch.Tensor,
    workspace_buffer: torch.Tensor,
    completion_buffer: torch.Tensor,
    softmax_scale_log2: float,
    bmm2_scale: float,
    cp_rank: int,
    cp_world: int,
    num_qo_heads: int,
    num_kv_heads: int,
    batch_size: int,
    q_len_per_req: int,
    sm_count: int,
) -> None:
    """Launch one balanced DCP program (the Cake ``_launch`` of the family)."""

    from .jit.cake_dcp import load_dcp_spec_balanced_module

    _check_workspace_buffer_alignment(workspace_buffer, "workspace_buffer")
    _check_workspace_buffer_alignment(completion_buffer, "multi_ctas_kv_counter_buffer")
    n_rows = dcp_balanced_n_rows(kind, q_len_per_req)
    module = load_dcp_spec_balanced_module(
        _DCP_BALANCED_FAMILY[kind],
        target,
        n_rows,
        dcp_balanced_program(
            kind,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            max_pages_per_seq=int(block_tables.shape[1]),
            sm_count=sm_count,
            arch=_DCP_BALANCED_ARCH.get(target, target),
            n_rows=n_rows,
        ),
    )
    if kind == "bf16_p16":
        if float(bmm2_scale) != 1.0:
            raise ValueError("the BF16/page16 DCP profile requires bmm2_scale=1.0")
        module.run(
            query,
            k_cache,
            v_cache,
            out,
            lse,
            block_tables,
            causal_seqlens_kv_global,
            workspace_buffer,
            completion_buffer,
            softmax_scale_log2,
            cp_rank,
            cp_world,
            num_qo_heads,
            num_kv_heads,
            batch_size,
            q_len_per_req,
            sm_count,
        )
        return
    module.run(
        query,
        k_cache.view(torch.uint8),
        v_cache.view(torch.uint8),
        out,
        lse,
        block_tables,
        causal_seqlens_kv_global,
        workspace_buffer,
        completion_buffer,
        softmax_scale_log2,
        float(bmm2_scale),
        cp_rank,
        cp_world,
        num_qo_heads,
        num_kv_heads,
        batch_size,
        q_len_per_req,
        sm_count,
    )


def _is_cuda_version_at_least(version: str) -> bool:
    from .jit.cpp_ext import is_cuda_version_at_least

    return is_cuda_version_at_least(version)


def _select_target(device: torch.device) -> DcpSpecTarget:
    capability = get_compute_capability(device)
    if capability not in ((10, 0), (10, 3), (10, 7)):
        raise RuntimeError(
            "DCP speculative FMHA requires compute capability 10.0 "
            "(B200/GB200), 10.3 (B300/GB300), or 10.7 (Rubin), "
            f"got {capability[0]}.{capability[1]}"
        )
    # Preserve the exact SM100/SM103 targets; SM107 uses the forward-compatible
    # family target for the shared DCP sources.
    if capability == (10, 0):
        if _is_cuda_version_at_least("12.8"):
            return "sm100a"
        raise RuntimeError(
            "DCP speculative FMHA on compute capability 10.0 requires CUDA "
            "12.8 or newer"
        )
    target: DcpSpecTarget = "sm103a" if capability == (10, 3) else "sm100f"
    if _is_cuda_version_at_least("12.9"):
        return target
    raise RuntimeError(
        f"DCP speculative FMHA on compute capability {capability[0]}.{capability[1]} "
        f"requires CUDA 12.9 or newer for the {target} target"
    )


def _validate_core_inputs(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    causal_seqlens_kv_global: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    batch_size: int,
    q_len_per_req: int,
    cp_world: int,
    cp_rank: int,
) -> tuple[int, int, str, int]:
    supported_q_lens: tuple[int, ...]
    if query.dtype != torch.bfloat16:
        raise TypeError("DCP speculative FMHA requires a BF16 query tensor")
    if k_cache.dtype != v_cache.dtype:
        raise TypeError("DCP speculative FMHA key and value dtypes must match")
    if k_cache.dtype == torch.bfloat16:
        profile = "bf16_p16"
        page_size_required = _BF16_PAGE_SIZE
        supported_q_lens = _BF16_SUPPORTED_Q_LENS
    elif k_cache.dtype == torch.float8_e4m3fn:
        profile = "fp8_p64"
        page_size_required = _FP8_PAGE_SIZE
        supported_q_lens = _FP8_SUPPORTED_Q_LENS
    else:
        raise TypeError(
            "DCP speculative FMHA requires BF16 or float8_e4m3fn key/value tensors"
        )
    if query.ndim != 3:
        raise ValueError(
            "DCP speculative FMHA query must have shape "
            "[batch_size * q_len_per_req, num_qo_heads, head_dim]"
        )
    num_tokens, num_qo_heads, head_dim = query.shape
    if num_tokens != batch_size * q_len_per_req:
        raise ValueError(
            "query shape does not match the DCP specialization: "
            f"got {tuple(query.shape)}, expected tokens={batch_size * q_len_per_req}"
        )
    if profile == "bf16_p16" and head_dim != _HEAD_DIM:
        raise ValueError("the BF16/page16 DCP profile requires head_dim=128")
    if profile == "fp8_p64" and head_dim not in (_HEAD_DIM, _D256_HEAD_DIM):
        raise ValueError("the FP8/page64 DCP profile requires head_dim=128 or 256")
    if head_dim == _D256_HEAD_DIM:
        profile = "fp8_p64_d256"
        supported_q_lens = _FP8_D256_SUPPORTED_Q_LENS
    if q_len_per_req not in supported_q_lens:
        raise ValueError(
            f"q_len_per_req must be one of {supported_q_lens} for the {profile} profile"
        )
    if cp_world not in _SUPPORTED_CP_WORLDS:
        raise ValueError(f"cp_world must be one of {_SUPPORTED_CP_WORLDS}")
    if not 0 <= cp_rank < cp_world:
        raise ValueError(f"cp_rank must be in [0, {cp_world}), got {cp_rank}")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError(
            "DCP speculative FMHA HND caches must have shape "
            "[num_pages, num_kv_heads, page_size, head_dim]"
        )
    if k_cache.shape != v_cache.shape:
        raise ValueError("key and value cache shapes must match")
    _, num_kv_heads, page_size, kv_head_dim = k_cache.shape
    if page_size != page_size_required or kv_head_dim != head_dim:
        raise ValueError(
            f"DCP speculative FMHA {profile} requires HND "
            f"page_size={page_size_required} and head_dim={head_dim}, "
            f"got page_size={page_size}, head_dim={kv_head_dim}"
        )
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")
    group_ratio = num_qo_heads // num_kv_heads
    if head_dim == _HEAD_DIM:
        if not 1 <= group_ratio <= 8:
            raise ValueError(
                f"DCP head group ratio must be in [1, 8], got {group_ratio}"
            )
    elif (
        num_qo_heads != 16
        or num_kv_heads != 1
        or group_ratio != 16
        or cp_world not in (1, 4)
    ):
        raise ValueError(
            "DCP D256 production profile requires Hq=16, Hkv=1, "
            "head group ratio 16, and cp_world in {1,4}"
        )

    device = query.device
    for name, tensor in (
        ("k_cache", k_cache),
        ("v_cache", v_cache),
        ("block_tables", block_tables),
        ("seq_lens", seq_lens),
        ("causal_seqlens_kv_global", causal_seqlens_kv_global),
        ("out", out),
        ("lse", lse),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must be on the same CUDA device as query")
    check_shape_dtype_device(out, query.shape, torch.bfloat16, device, "out")
    check_shape_dtype_device(
        lse,
        (num_tokens, num_qo_heads),
        torch.float32,
        device,
        "lse",
    )
    check_shape_dtype_device(
        causal_seqlens_kv_global,
        (batch_size,),
        torch.int32,
        device,
        "causal_seqlens_kv_global",
    )
    if block_tables.ndim != 2 or not block_tables.is_contiguous():
        raise ValueError("block_tables must be a contiguous 2D tensor")
    if block_tables.dtype != torch.int32 or block_tables.shape[0] != batch_size:
        raise ValueError(
            "block_tables must be int32 with shape [batch_size, max_pages_per_seq]"
        )
    if block_tables.shape[1] <= 0:
        raise ValueError("block_tables must contain at least one physical page slot")
    if (
        seq_lens.dtype != torch.int32
        or seq_lens.ndim != 1
        or seq_lens.shape[0] != batch_size
        or not seq_lens.is_contiguous()
    ):
        raise ValueError(
            "seq_lens must be a contiguous int32 tensor with shape [batch_size]"
        )
    if not causal_seqlens_kv_global.is_contiguous():
        raise ValueError("causal_seqlens_kv_global must be contiguous")
    return num_qo_heads, num_kv_heads, profile, page_size_required


def run_dcp_spec_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    causal_seqlens_kv_global: torch.Tensor,
    max_local_seq_len: int,
    bmm1_scale: float,
    bmm2_scale: float,
    cp_world: int,
    cp_rank: int,
    q_len_per_req: int,
    out: torch.Tensor,
    lse: torch.Tensor,
    completion_buffer: Optional[torch.Tensor],
    *,
    route: str = "auto",
) -> None:
    """Run one rank-local Cake FMHA DCP speculative specialization.

    ``route`` selects between the static specializations (``"static"``: v1 /
    v4 for BF16, the split-KV families for FP8) and the on-device load-balanced
    programs (``"balanced"``); ``"auto"`` follows :func:`dcp_balanced_band` and
    launches the balanced program when the row is inside its band and the
    caller-owned ``workspace_buffer`` / ``completion_buffer`` are large enough
    (:func:`get_dcp_spec_balanced_workspace_bytes`,
    :func:`get_dcp_spec_balanced_counter_bytes`); otherwise the static
    specialization serves the row exactly as before.
    """

    if q_len_per_req <= 0 or query.shape[0] % q_len_per_req != 0:
        raise ValueError("query token count must be divisible by q_len_per_req")
    batch_size = query.shape[0] // q_len_per_req
    if batch_size <= 0:
        raise ValueError(
            "DCP speculative FMHA requires a non-empty batch, got "
            f"query.shape[0]={query.shape[0]}"
        )
    num_qo_heads, num_kv_heads, profile, page_size = _validate_core_inputs(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        causal_seqlens_kv_global,
        out,
        lse,
        batch_size=batch_size,
        q_len_per_req=q_len_per_req,
        cp_world=cp_world,
        cp_rank=cp_rank,
    )
    if max_local_seq_len < 0:
        raise ValueError(
            f"max_local_seq_len must be nonnegative, got {max_local_seq_len}"
        )
    local_capacity = block_tables.shape[1] * page_size
    if max_local_seq_len > local_capacity:
        raise ValueError(
            "max_local_seq_len exceeds the rank-local page-table capacity: "
            f"got {max_local_seq_len}, capacity={local_capacity}"
        )

    if not math.isfinite(float(bmm1_scale)) or not math.isfinite(float(bmm2_scale)):
        raise ValueError(
            "DCP speculative FMHA bmm1_scale and bmm2_scale must be finite"
        )

    sm_count = get_device_sm_count(query.device)
    logical_tiles = batch_size * q_len_per_req * num_kv_heads
    target = _select_target(query.device)
    softmax_scale_log2 = float(bmm1_scale) / math.log(2.0)
    max_pages_per_seq = block_tables.shape[1]

    if route not in _DCP_ROUTES:
        raise ValueError(f"route must be one of {_DCP_ROUTES}, got {route!r}")
    if route != "static":
        band = dcp_balanced_band(
            profile,
            batch_size=batch_size,
            q_len=q_len_per_req,
            num_q_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=query.shape[-1],
            max_local_seq_len=max_local_seq_len,
            cp_world=cp_world,
            sm_count=sm_count,
            arch=_DCP_BALANCED_ARCH.get(target, target),
            max_pages_per_seq=max_pages_per_seq,
        )
        if route == "balanced" or band.route == "balanced":
            problem = _dcp_balanced_buffer_problem(
                workspace_buffer,
                completion_buffer,
                query.device,
                sm_count,
                query.shape[-1],
            )
            if problem is None:
                _run_dcp_spec_balanced(
                    kind=profile,
                    target=target,
                    query=query,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    out=out,
                    lse=lse,
                    block_tables=block_tables,
                    causal_seqlens_kv_global=causal_seqlens_kv_global,
                    workspace_buffer=workspace_buffer,
                    completion_buffer=completion_buffer,
                    softmax_scale_log2=softmax_scale_log2,
                    bmm2_scale=float(bmm2_scale),
                    cp_rank=cp_rank,
                    cp_world=cp_world,
                    num_qo_heads=num_qo_heads,
                    num_kv_heads=num_kv_heads,
                    batch_size=batch_size,
                    q_len_per_req=q_len_per_req,
                    sm_count=sm_count,
                )
                return
            if route == "balanced":
                raise ValueError(problem)
            # ``auto``: the caller did not provision the balanced scratch; the
            # static specialization below serves the row exactly as before.

    if profile.startswith("fp8_p64"):
        from .jit.cake_dcp import load_dcp_spec_static_module

        local_blocks = max(1, (max_local_seq_len + _BLOCK_N - 1) // _BLOCK_N)
        num_split = _select_fp8_num_split(
            logical_tiles=logical_tiles,
            sm_count=sm_count,
            local_blocks=local_blocks,
            cp_world=cp_world,
            head_dim=query.shape[-1],
        )
        head_dim = query.shape[-1]
        if head_dim == _D256_HEAD_DIM:
            module = load_dcp_spec_static_module(
                "fp8_d256",
                "split1" if num_split == 1 else "splitn",
                target,
                _static_constants(
                    q_len_per_req, cp_world, num_qo_heads, num_kv_heads, num_split
                ),
            )
        else:
            retain_kv_l2 = int(
                cp_world > 1 and local_blocks <= _FP8_RETAIN_KV_L2_MAX_BLOCKS
            )
            module = load_dcp_spec_static_module(
                "fp8_d128",
                f"{_static_split_instance(num_split)}_retain{retain_kv_l2}",
                target,
                _static_constants(
                    q_len_per_req, cp_world, num_qo_heads, num_kv_heads, num_split
                ),
            )
        if num_split == 1:
            partial_o = out
            partial_lse = lse
            split_completion = seq_lens
        else:
            partial_o, partial_lse, split_completion = _split_workspace_views(
                workspace_buffer=workspace_buffer,
                completion_buffer=completion_buffer,
                device=query.device,
                batch_size=batch_size,
                q_len_per_req=q_len_per_req,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                num_split=num_split,
            )
        grid = min(sm_count, logical_tiles * num_split)
        module.run(
            query,
            k_cache.view(torch.uint8),
            v_cache.view(torch.uint8),
            partial_o,
            partial_lse,
            out,
            lse,
            split_completion,
            block_tables,
            seq_lens,
            causal_seqlens_kv_global,
            max_pages_per_seq,
            max_local_seq_len,
            softmax_scale_log2,
            float(bmm2_scale),
            cp_rank,
            num_qo_heads,
            num_kv_heads,
            batch_size,
            grid,
            1,
            1,
        )
        return

    if float(bmm2_scale) != 1.0:
        raise ValueError("the BF16/page16 DCP profile requires bmm2_scale=1.0")

    local_blocks = max(1, (max_local_seq_len + _BLOCK_N - 1) // _BLOCK_N)
    num_split = _select_num_split(
        logical_tiles=logical_tiles,
        sm_count=sm_count,
        local_blocks=local_blocks,
    )
    from .jit.cake_dcp import load_dcp_spec_static_module

    if num_split == 1:
        retain_kv_l2 = int(local_blocks <= _RETAIN_KV_L2_MAX_BLOCKS)
        module = load_dcp_spec_static_module(
            "bf16_v1",
            f"retain{retain_kv_l2}",
            target,
            _static_constants(q_len_per_req, cp_world, num_qo_heads, num_kv_heads),
        )
        grid = min(sm_count, logical_tiles)
        module.run(
            query,
            k_cache,
            v_cache,
            out,
            lse,
            block_tables,
            causal_seqlens_kv_global,
            max_pages_per_seq,
            max_local_seq_len,
            softmax_scale_log2,
            cp_rank,
            num_qo_heads,
            num_kv_heads,
            batch_size,
            grid,
            1,
            1,
        )
        return

    partial_o, partial_lse, split_completion = _split_workspace_views(
        workspace_buffer=workspace_buffer,
        completion_buffer=completion_buffer,
        device=query.device,
        batch_size=batch_size,
        q_len_per_req=q_len_per_req,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=query.shape[-1],
        num_split=num_split,
    )

    module = load_dcp_spec_static_module(
        "bf16_v4",
        _static_split_instance(num_split),
        target,
        _static_constants(
            q_len_per_req, cp_world, num_qo_heads, num_kv_heads, num_split
        ),
    )
    grid = min(sm_count, logical_tiles * num_split)
    module.run(
        query,
        k_cache,
        v_cache,
        partial_o,
        partial_lse,
        out,
        lse,
        split_completion,
        block_tables,
        causal_seqlens_kv_global,
        max_pages_per_seq,
        max_local_seq_len,
        softmax_scale_log2,
        cp_rank,
        num_qo_heads,
        num_kv_heads,
        batch_size,
        grid,
        1,
        1,
    )


__all__ = [
    "DcpBalancedBand",
    "dcp_balanced_band",
    "dcp_balanced_n_rows",
    "dcp_balanced_program",
    "dcp_balanced_route",
    "dcp_static_shape",
    "get_dcp_spec_balanced_counter_bytes",
    "get_dcp_spec_balanced_workspace_bytes",
    "get_dcp_spec_counter_bytes",
    "get_dcp_spec_workspace_size_bytes",
]
