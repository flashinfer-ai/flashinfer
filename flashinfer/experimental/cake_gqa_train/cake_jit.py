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
``bwd_pre`` (``dPsum``, ``LSE * log2(e)`` and
the zeroed FP32 dQ accumulator in the pack-2 scratch layout), the backward
main kernel in its three dK / dV outputs ``bwd_fp32`` (FP32 window
gradients), ``bwd_rowmap`` (FP32 adds into caller-owned destinations through
a row map) and ``bwd_bf16``, and ``bwd_post`` (dQ from the accumulator).
Every launch is a plain stream-ordered launch: nothing cooperative, no PDL.
"""

from __future__ import annotations

from typing import Any, Optional

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
HEAD_DIM = 256
PACKAGE = "flashinfer.experimental.cake_gqa_train"
LIBRARY_PREFIX = "cake_gqa_train"
MANIFEST = "build.json"

# The stages of one training step, in launch order; every stage is served on
# every architecture of ``ARCHES`` by exactly one record.
STAGES = ("fwd", "bwd_pre", "bwd_fp32", "bwd_rowmap", "bwd_bf16", "bwd_post")

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
    "bwd_fp32": {
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
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_gqa_train_6fa13db592acdb66a6c9",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 231424,
            "use_pdl": False,
        },
        "library": "cake_gqa_train_6fa13db592acdb66a6c9",
        "stage": "bwd_fp32",
    },
    "bwd_rowmap": {
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
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_gqa_train_37f4cd45025705f3cfc8",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 231424,
            "use_pdl": False,
        },
        "library": "cake_gqa_train_37f4cd45025705f3cfc8",
        "stage": "bwd_rowmap",
    },
    "bwd_bf16": {
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
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_gqa_train_1afdfdde603ff8806b4d",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 231424,
            "use_pdl": False,
        },
        "library": "cake_gqa_train_1afdfdde603ff8806b4d",
        "stage": "bwd_bf16",
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
# The host plan: grids and scratch sizes from host-known shapes only (the
# tensors' shapes, the segment count ``len(cu_seqlens) - 1`` and the SM
# count); nothing here reads a tensor.

# Forward: a CTA pair covers 256 query rows of one (segment, query head); the
# grid is persistent (tickets from a self-resetting counter), one pair per two
# SMs at most, any count >= 1 (no co-residency is assumed).
FWD_TILE_ROWS = 256
FWD_CLUSTER = 2
# Backward main kernel: a CTA pair covers 128 keys of one (segment, key head)
# and loops the key head's query heads two at a time (``group`` even).  The
# grid is ``(2 N, H_kv, B)``: the pair at ``(2 n, h, b)`` owns key block ``n``
# of segment ``b``; ``N`` is at least the longest segment's key-block count
# (from ``max_seqlen_k`` when the caller gives it, else from ``T_k``, which
# bounds every segment), and a pair past its segment's last block does nothing.
BWD_TILE_KEYS = 128
BWD_CLUSTER = 2
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


def grid(
    stage: str,
    *,
    total_q: int,
    total_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    num_segments: int,
    sm_count: int,
    max_seqlen_k: Optional[int] = None,
) -> tuple[int, int, int]:
    """The launch grid of ``stage``.  Tile counts are upper bounds from host
    values (``sum_b ceil(len_b / n) <= T // n + B``; the longest segment has
    at most ``max_seqlen_k`` or ``T_k`` keys); the kernels decode the segments
    on the device and retire the CTAs past the last tile."""
    if stage == "fwd":
        tiles = num_heads_q * (total_q // FWD_TILE_ROWS + num_segments)
        pairs = max(1, min(sm_count // FWD_CLUSTER, tiles))
        return (FWD_CLUSTER * pairs, 1, 1)
    if stage in ("bwd_pre", "bwd_post"):
        rows = rows_pad(total_q, num_segments)
        per_cta = PRE_ROWS_PER_CTA if stage == "bwd_pre" else POST_ROWS_PER_CTA
        return (rows // per_cta, num_heads_q // 2, 1)
    if stage in ("bwd_fp32", "bwd_rowmap", "bwd_bf16"):
        longest = total_k if max_seqlen_k is None else min(max_seqlen_k, total_k)
        blocks = max(1, -(-longest // BWD_TILE_KEYS))
        return (BWD_CLUSTER * blocks, num_heads_kv, num_segments)
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
    "FWD_CLUSTER",
    "FWD_TILE_ROWS",
    "HEAD_DIM",
    "LIBRARY_PREFIX",
    "MANIFEST",
    "MODULES",
    "PACKAGE",
    "SCRATCH_ALIGN",
    "POST_ROWS_PER_CTA",
    "PRE_ROWS_PER_CTA",
    "SCRATCH_TILE_ROWS",
    "STAGES",
    "grid",
    "rows_pad",
    "scratch_bytes",
    "scratch_layout",
    "stage_record",
]
