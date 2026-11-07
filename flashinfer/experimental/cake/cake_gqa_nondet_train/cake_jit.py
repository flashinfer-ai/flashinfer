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

Registry of the Cake packed-varlen GQA training attention kernels, head
dimension 256 (SM100 / SM103 / SM107).  The loader and the launch plan are
:mod:`.cake_backend`.

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit plus a tvm-ffi binding, compiled once per
architecture.  A record is keyed by its stage name (or ``<stage>_<variant>``
for a stage that ships a different program per architecture) and carries the
host plan: the ``stage`` it serves, the architectures it is delivered for and
the ``run`` entry.  The derived leaves (``library``, ``kernel_symbol``,
``launch``, ``arg_plan``) are written by the exporter from the rendered
binding; ``library`` is the program identity (``cake_gqa_train_<20 hex>``),
so a kernel swap renames it.  ``MODULES`` is a literal table (the exporter
and the manifest checker read it with ``ast.literal_eval``).

The stages, in launch order: ``fwd`` (O and the natural-log LSE),
``bwd_pre`` (``dPsum``, ``LSE * log2(e)`` and the zeroed FP32 dQ accumulator
in the pack-2 scratch layout), ``bwd_plan`` (one CTA: the workspace tile
numbering and the dQ work order from the segment boundaries; PyTorch ops
instead past ``PLAN_MAX_PAIRS``), then per workspace chunk ``bwd_main_fp32`` /
``bwd_main_bf16`` (S^T, dP^T, dK and dV of a key block, dS^T stored to the
workspace; dK / dV stored as FP32 window gradients or as BF16) and
``bwd_dq`` (dQ = dS K over the workspace into the accumulator), then
``bwd_fold`` (row-map destinations only: the FP32 window rows added into
their destination rows in ascending window order) and ``bwd_post`` (dQ from
the accumulator).  Every sum has one fixed order, so every output is bitwise
reproducible run to run for one shape, one ``max_seqlen_k`` and one
workspace budget.  Every launch is a plain stream-ordered launch: nothing
cooperative, no PDL.
"""

from __future__ import annotations

from typing import Any, Optional

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
HEAD_DIM = 256
PACKAGE = "flashinfer.experimental.cake.cake_gqa_nondet_train"
LIBRARY_PREFIX = "cake_gqa_train"
MANIFEST = "build.json"

# The stages of one training step, in launch order; every stage is served on
# every architecture of ``ARCHES`` by exactly one record.
STAGES = (
    "fwd",
    "bwd_pre",
    "bwd_plan",
    "bwd_main_fp32",
    "bwd_main_bf16",
    "bwd_dq",
    "bwd_fold",
    "bwd_post",
)

MODULES: dict[str, dict[str, Any]] = {
    "fwd": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "O"],
            ["buffer", "LSE"],
            ["buffer", "cu_seqlens_q"],
            ["buffer", "cu_seqlens_k"],
            ["buffer", "tile_counter"],
            ["parameter", "num_segments"],
            ["parameter", "num_heads_q"],
            ["parameter", "group"],
            ["parameter", "total_q"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_gqa_train_493c4a28dafa5fc6b2f1",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 232064,
            "use_pdl": False,
        },
        "library": "cake_gqa_train_493c4a28dafa5fc6b2f1",
        "stage": "fwd",
    },
    "bwd_pre": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "O"],
            ["buffer", "dO"],
            ["buffer", "LSE"],
            ["buffer", "cu_seqlens_q"],
            ["buffer", "LSE_log2"],
            ["buffer", "dPsum"],
            ["buffer", "dQ_accum"],
            ["parameter", "T_q"],
            ["parameter", "H_q"],
            ["parameter", "num_segments"],
            ["parameter", "rows_pad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_gqa_train_a91752e8a2732bf2e246",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gqa_train_a91752e8a2732bf2e246",
        "stage": "bwd_pre",
    },
    "bwd_plan": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "cu_seqlens_q"],
            ["buffer", "cu_seqlens_k"],
            ["buffer", "tile_base"],
            ["buffer", "pair_base"],
            ["buffer", "pair_order"],
            ["parameter", "num_segments"],
            ["parameter", "key_blocks"],
            ["parameter", "pair_bound"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "kernel_symbol": "",
        "launch": {
            "block": (1024, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 65792,
            "use_pdl": False,
        },
        "library": "",
        "stage": "bwd_plan",
    },
    "bwd_main_fp32": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "dO"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "lse_log2"],
            ["buffer", "dpsum"],
            ["buffer", "dq_accum"],
            ["buffer", "cu_seqlens_q"],
            ["buffer", "cu_seqlens_k"],
            ["buffer", "dk_out"],
            ["buffer", "dv_out"],
            ["parameter", "num_segments"],
            ["parameter", "h_kv"],
            ["parameter", "group"],
            ["parameter", "rows_pad"],
            ["parameter", "softmax_scale"],
            ["parameter", "softmax_scale_log2"],
            ["buffer", "row_map"],
            ["buffer", "ds_ws"],
            ["buffer", "tile_base"],
            ["parameter", "n0"],
            ["parameter", "h0"],
            ["parameter", "tile_stride"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "kernel_symbol": "",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 232448,
            "use_pdl": False,
        },
        "library": "",
        "stage": "bwd_main_fp32",
    },
    "bwd_main_bf16": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "dO"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "lse_log2"],
            ["buffer", "dpsum"],
            ["buffer", "dq_accum"],
            ["buffer", "cu_seqlens_q"],
            ["buffer", "cu_seqlens_k"],
            ["buffer", "dk_out"],
            ["buffer", "dv_out"],
            ["parameter", "num_segments"],
            ["parameter", "h_kv"],
            ["parameter", "group"],
            ["parameter", "rows_pad"],
            ["parameter", "softmax_scale"],
            ["parameter", "softmax_scale_log2"],
            ["buffer", "row_map"],
            ["buffer", "ds_ws"],
            ["buffer", "tile_base"],
            ["parameter", "n0"],
            ["parameter", "h0"],
            ["parameter", "tile_stride"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "kernel_symbol": "",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 232448,
            "use_pdl": False,
        },
        "library": "",
        "stage": "bwd_main_bf16",
    },
    "bwd_dq": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "dsw"],
            ["buffer", "K"],
            ["buffer", "tile_base"],
            ["buffer", "pair_base"],
            ["buffer", "pair_order"],
            ["buffer", "cu_seqlens_q"],
            ["buffer", "cu_seqlens_k"],
            ["buffer", "dq_accum"],
            ["parameter", "num_segments"],
            ["parameter", "group"],
            ["parameter", "rows_pad"],
            ["parameter", "n0"],
            ["parameter", "n1"],
            ["parameter", "h0"],
            ["parameter", "hc"],
            ["parameter", "tile_stride"],
            ["parameter", "zero_tile"],
            ["parameter", "accumulate"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "kernel_symbol": "",
        "launch": {
            "block": (192, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197632,
            "use_pdl": False,
        },
        "library": "",
        "stage": "bwd_dq",
    },
    "bwd_fold": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "dk_win"],
            ["buffer", "dv_win"],
            ["buffer", "perm"],
            ["buffer", "dst_rows"],
            ["buffer", "run_end"],
            ["buffer", "dk_out"],
            ["buffer", "dv_out"],
            ["parameter", "T_k"],
            ["parameter", "h_kv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "kernel_symbol": "",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "",
        "stage": "bwd_fold",
    },
    "bwd_post": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "dQ_accum"],
            ["buffer", "cu_seqlens_q"],
            ["buffer", "dQ"],
            ["parameter", "T_q"],
            ["parameter", "H_q"],
            ["parameter", "num_segments"],
            ["parameter", "rows_pad"],
            ["parameter", "softmax_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_gqa_train_76decc9a5d22dc91ddd1",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_gqa_train_76decc9a5d22dc91ddd1",
        "stage": "bwd_post",
    },
}

# ---------------------------------------------------------------------------
# The host plan: grids, workspace chunks and scratch sizes from host-known
# values only (the tensors' shapes, the segment count ``len(cu_seqlens) - 1``,
# ``max_seqlen_k``, the workspace budget and the SM count); nothing here reads
# a tensor.

# Forward: a CTA pair covers 256 query rows of one (segment, query head); the
# grid is persistent (tickets from a self-resetting counter), one pair per two
# SMs at most, any count >= 1 (no co-residency is assumed).
FWD_TILE_ROWS = 256
FWD_CLUSTER = 2
# Backward: a key block is 128 keys (one CTA pair of the main kernel, which
# loops the key head's query heads two at a time), a Q tile 64 query rows of
# one head pair (128 packed scratch rows).  The main kernel stores dS^T per
# (head pair, key block, Q tile) as one 32 KiB workspace tile (128 keys x 128
# packed rows, BF16); the dQ GEMM reads it as 256 lines of 128 bytes.  Every
# CTA group is a pair (``cta_group::2``).
BWD_TILE_KEYS = 128
BWD_TILE_ROWS = 64
BWD_CLUSTER = 2
DS_TILE_BYTES = 32768
DS_TILE_LINES = 256
# Row-map fold: one warp per (window row, key head), eight per CTA.
FOLD_ITEMS_PER_CTA = 8
# The tile tables (``bwd_plan``): one CTA; it keeps one count per Q-tile pair
# position in shared memory, so a call with more positions than this builds
# the tables with PyTorch ops instead.
PLAN_MAX_PAIRS = 16384
# Pack-2 scratch: scratch head ``s`` holds query heads ``2 s`` and ``2 s + 1``;
# every segment starts on a 128-row tile.  The preprocess runs one CTA per 32
# packed rows of one scratch head, the postprocess one per 64 (each program's
# ``rows_per_cta``; a grid past ``rows_pad`` would read past the scratch).
SCRATCH_TILE_ROWS = 128
PRE_ROWS_PER_CTA = 32
POST_ROWS_PER_CTA = 64
# Byte alignment of every scratch buffer (TMA / bulk copies need 16 B, the
# 256-bit loads and stores of the pre/postprocess 32 B).
SCRATCH_ALIGN = 256
# Default bound of the dS^T workspace of one backward call.  A smaller budget
# splits the call into more chunks (KV heads first, then key-block ranges);
# the key-block split re-reads and rewrites the FP32 dQ accumulator once per
# extra chunk, and moves the dQ rounding (the bits, not the accuracy).
DEFAULT_WORKSPACE_BUDGET = 16 << 30


def rows_pad(total_q: int, num_segments: int) -> int:
    """Packed rows per scratch head: ``round_up(2 T_q + 128 B, 128)``."""
    rows = 2 * total_q + SCRATCH_TILE_ROWS * num_segments
    return -(-rows // SCRATCH_TILE_ROWS) * SCRATCH_TILE_ROWS


def scratch_layout(
    total_q: int, num_heads_q: int, num_segments: int
) -> dict[str, tuple[int, tuple[int, ...]]]:
    """The backward scratch as ``name -> (byte offset, FP32 shape)``: ``lse_log2`` and
    ``dpsum`` ``[H_q / 2, rows_pad]``, ``dq_accum`` ``[H_q / 2, rows_pad, 256]``."""
    heads = num_heads_q // 2
    rows = rows_pad(total_q, num_segments)
    layout: dict[str, tuple[int, tuple[int, ...]]] = {}
    offset = 0
    for name, shape in (
        ("dq_accum", (heads, rows, HEAD_DIM)),
        ("lse_log2", (heads, rows)),
        ("dpsum", (heads, rows)),
    ):
        layout[name] = (offset, shape)
        count = 1
        for extent in shape:
            count *= extent
        offset += -(-4 * count // SCRATCH_ALIGN) * SCRATCH_ALIGN
    return layout


def scratch_bytes(total_q: int, num_heads_q: int, num_segments: int) -> int:
    """Bytes of the backward scratch of one call (:func:`scratch_layout`)."""
    if total_q == 0:
        return 0
    offset, shape = scratch_layout(total_q, num_heads_q, num_segments)["dpsum"]
    return offset + -(-4 * shape[0] * shape[1] // SCRATCH_ALIGN) * SCRATCH_ALIGN


def key_blocks(total_k: int, max_seqlen_k: Optional[int]) -> int:
    """Key blocks of the longest segment, rounded up to even (the chunks split the key blocks in pairs): from
    ``max_seqlen_k`` when the caller gives it, else from ``T_k``, which bounds every segment."""
    longest = total_k if max_seqlen_k is None else min(int(max_seqlen_k), total_k)
    blocks = max(1, -(-longest // BWD_TILE_KEYS))
    return blocks + (blocks & 1)


def pair_bound(total_q: int, num_segments: int) -> int:
    """A bound on the Q-tile pairs (128 query rows of one segment) of a call: ``sum_b ceil(q_b / 128) <= ceil(T_q /
    128) + B``; the length of the dQ GEMM's work order."""
    return -(-total_q // (2 * BWD_TILE_ROWS)) + num_segments


def tiles_per_key_block(total_q: int, num_segments: int) -> int:
    """A bound on one head pair's dS^T tiles of one key block over all segments: ``sum_b ceil(q_b / 64) <=
    (T_q + 63 B) // 64``."""
    return max(1, (total_q + (BWD_TILE_ROWS - 1) * num_segments) // BWD_TILE_ROWS)


def chunk_plan(
    *,
    total_q: int,
    total_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    num_segments: int,
    max_seqlen_k: Optional[int],
    budget_bytes: int,
) -> dict[str, int]:
    """The workspace chunks of one backward call: ``h_chunk`` KV heads times ``n_chunk`` key blocks per chunk,
    ``tile_stride`` tiles per head pair (a bound of :func:`tiles_per_key_block` per key block), the all-zero tile
    ``zero_tile`` after the chunk's tiles and ``ws_tiles`` in all.  As many KV heads as fit with every key block;
    else one KV head and the widest even key-block range that fits; at least one KV head and two key blocks, even
    past the budget."""
    blocks = key_blocks(total_k, max_seqlen_k)
    pairs = num_heads_q // num_heads_kv // 2
    per_block = tiles_per_key_block(total_q, num_segments)
    cap = max(0, int(budget_bytes) // DS_TILE_BYTES - 1)
    per_head = blocks * pairs * per_block
    if per_head * num_heads_kv <= cap:
        h_chunk, n_chunk = num_heads_kv, blocks
    elif per_head <= cap:
        h_chunk, n_chunk = cap // per_head, blocks
    else:
        h_chunk, n_chunk = 1, max(2, (cap // (pairs * per_block)) & ~1)
    stride = n_chunk * per_block
    zero_tile = h_chunk * pairs * stride
    return {
        "key_blocks": blocks,
        "h_chunk": h_chunk,
        "n_chunk": n_chunk,
        "tile_stride": stride,
        "zero_tile": zero_tile,
        "ws_tiles": zero_tile + 1,
    }


def grid(
    stage: str,
    *,
    total_q: int,
    total_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    num_segments: int,
    sm_count: int,
    chunk_blocks: int = 0,
    chunk_heads: int = 0,
) -> tuple[int, int, int]:
    """The launch grid of ``stage``; ``chunk_blocks`` / ``chunk_heads`` are the key blocks and KV heads of the
    workspace chunk a ``bwd_main_*`` / ``bwd_dq`` launch covers.  Tile counts are upper bounds from host values
    (``sum_b ceil(len_b / n) <= T // n + B``); the kernels decode the segments on the device and retire the CTAs (or
    skip the work items) past the last tile."""
    if stage == "fwd":
        tiles = num_heads_q * (total_q // FWD_TILE_ROWS + num_segments)
        pairs = max(1, min(sm_count // FWD_CLUSTER, tiles))
        return (FWD_CLUSTER * pairs, 1, 1)
    if stage == "bwd_plan":
        return (1, 1, 1)
    if stage in ("bwd_pre", "bwd_post"):
        rows = rows_pad(total_q, num_segments)
        per_cta = PRE_ROWS_PER_CTA if stage == "bwd_pre" else POST_ROWS_PER_CTA
        return (rows // per_cta, num_heads_q // 2, 1)
    if stage in ("bwd_main_fp32", "bwd_main_bf16"):
        return (BWD_CLUSTER * chunk_blocks, chunk_heads, num_segments)
    if stage == "bwd_dq":
        items = (
            pair_bound(total_q, num_segments)
            * chunk_heads
            * (num_heads_q // num_heads_kv // 2)
        )
        return (BWD_CLUSTER * max(1, min(sm_count // BWD_CLUSTER, items)), 1, 1)
    if stage == "bwd_fold":
        return (-(-total_k * num_heads_kv // FOLD_ITEMS_PER_CTA), 1, 1)
    raise KeyError(f"no stage {stage!r}; the stages are {STAGES}")


def stage_record(stage: str, arch: str) -> str:
    """The record of ``MODULES`` serving ``stage`` on ``arch``."""
    for name, found in MODULES.items():
        if found["stage"] == stage and arch in found["arches"]:
            return name
    raise KeyError(f"no record serves stage {stage!r} on {arch}")


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "BWD_CLUSTER",
    "BWD_TILE_KEYS",
    "BWD_TILE_ROWS",
    "DEFAULT_WORKSPACE_BUDGET",
    "DS_TILE_BYTES",
    "DS_TILE_LINES",
    "FOLD_ITEMS_PER_CTA",
    "FWD_CLUSTER",
    "FWD_TILE_ROWS",
    "HEAD_DIM",
    "LIBRARY_PREFIX",
    "MANIFEST",
    "MODULES",
    "PACKAGE",
    "PLAN_MAX_PAIRS",
    "POST_ROWS_PER_CTA",
    "PRE_ROWS_PER_CTA",
    "SCRATCH_ALIGN",
    "SCRATCH_TILE_ROWS",
    "STAGES",
    "chunk_plan",
    "grid",
    "key_blocks",
    "pair_bound",
    "rows_pad",
    "scratch_bytes",
    "scratch_layout",
    "stage_record",
    "tiles_per_key_block",
]
