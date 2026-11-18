# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The sparse MLA classes ``mla.sparse.<dtype>.d<576|512>``: constants, list contract and fingerprint.

The term is a kernel FlashInfer already ships: the ``prims_ts`` sparse MLA decode kernel on one pinned schedule
(two-CTA family, one 128-row query tile per query row, KV never split, the 512 output columns never split). A query
row's output is a function of that row alone: its query, its selected-key list in list order, the cache bytes at those
indices and the two scales. The row is folded over its own list in whole 128-key tiles in list order; the two CTAs of
its cluster each take 64 of the tile's query rows, and no partial result is rounded and merged, which is what the
planned-extent schedules of the default path do. Prefill, chunked prefill, decode and multi-token (MTP) rows are
placements of the same row term (``lean/BIKernels/MlaSparse.lean``); ``README.md`` gives the term in order.
"""

from __future__ import annotations

import functools
import hashlib
import os
from typing import Dict, List, Tuple

FAMILY = "mla.sparse"
#: SM100, SM103 and SM107 give one set of bits for the same inputs.
MLA_ARCHS = ((10, 0), (10, 3), (10, 7))
LATENT_DIM = 512
ROPE_DIM = 64
#: Query / key widths of the two layouts: 512 latent + 64 RoPE, and latent only.
HEAD_DIMS = (LATENT_DIM + ROPE_DIM, LATENT_DIM)
DTYPES = ("e4m3", "bf16")
MAX_HEADS = 128
#: Keys per online-softmax step: the running maximum is updated once per tile, tiles run in list order.
KEY_TILE = 128
#: Query rows of one work tile (the heads of one query row, padded); a tile never mixes query rows.
QUERY_TILE = 128
#: CTAs that serve one query row (one cluster).
CTAS_PER_ROW = 2
SPLIT_KV = 1
HEAD_DIM_CTAS = 1

#: Pinned ``SparseMlaProfile`` fields per Q / KV dtype; scheduler and loader fields do not enter the arithmetic.
PROFILES: Dict[str, Dict[str, object]] = {
    "e4m3": dict(
        family="2cta",
        tile_size_q=QUERY_TILE,
        split_kv=SPLIT_KV,
        scheduler="static",
        head_dim_ctas=HEAD_DIM_CTAS,
        gather_issue_warps=1,
    ),
    "bf16": dict(
        family="2cta",
        tile_size_q=QUERY_TILE,
        split_kv=SPLIT_KV,
        scheduler="static",
        head_dim_ctas=HEAD_DIM_CTAS,
        gather_issue_warps=8,
        kv_pipeline_stages=7,
    ),
}

#: Static state the planner sets on the kernel object; a device test compares it with the planned kernel.
KERNEL_STATE: Dict[str, object] = dict(
    static_split_kv=SPLIT_KV,
    is_var_split_kv=False,
    is_var_seq=False,
    fixed_seqlen=True,
    softmax_rescale_threshold_log2=0.0,
    fuse_sparse_epilogue=True,
    fuse_sparse_reduction=False,
    direct_sparse=True,
    output_quant=None,
    kv_cache_format="native",
    extra_kv_cache_format="native",
)

#: Term sources under ``flashinfer/attention/prims_ts/kernels``; a directory counts its ``.py`` files.
TERM_SOURCES = (
    "mla_decode/sparse_policy.py",
    "mla_decode/parallel_reduction_topology.py",
    "mla_decode/throughput_2cta",
    "mla_decode/helpers",
    "mla_decode/throughput_latency_1cta/config.py",
    "fmha_decode/fmha_decode_resources/smem_transformed_kv.py",
    "fmha_decode/fmha_decode_resources/tmem_transformed_kv.py",
    "mask.py",
    "separate_reduction.py",
    "stage.py",
    "task_compat.py",
    "tcgen05_compat.py",
    "tensor_map.py",
)


def dtype_name(dtype) -> str:
    """``torch.float8_e4m3fn`` / ``"e4m3"`` / ``"fp8"`` -> ``"e4m3"``; ``torch.bfloat16`` / ``"bf16"`` -> ``"bf16"``."""
    name = str(dtype).removeprefix("torch.")
    names = {"float8_e4m3fn": "e4m3", "e4m3": "e4m3", "fp8": "e4m3"}
    names.update({"bfloat16": "bf16", "bf16": "bf16"})
    if name not in names:
        raise ValueError(
            f"the {FAMILY} classes take Q / KV dtype torch.float8_e4m3fn or torch.bfloat16, not {dtype!r}"
        )
    return names[name]


def class_name(dtype, head_dim: int) -> str:
    """The class of a Q / KV dtype and a query width: ``mla.sparse.e4m3.d576`` (GLM-5.3 RoPE rows, FP8 cache),
    ``mla.sparse.e4m3.d512`` (NoPE rows), and the two ``bf16`` ones."""
    if head_dim not in HEAD_DIMS:
        raise ValueError(
            f"the {FAMILY} classes take head_dim 576 or 512, not {head_dim!r}"
        )
    return f"{FAMILY}.{dtype_name(dtype)}.d{int(head_dim)}"


def class_names() -> List[str]:
    return [class_name(d, h) for d in DTYPES for h in HEAD_DIMS]


def list_is_packed(indices, length: int) -> bool:
    """The list contract of one row: the first ``length`` entries are live (non-negative), ``-1`` only after them."""
    return all(int(i) >= 0 for i in list(indices)[:length])


def key_tiles(length: int) -> int:
    """Whole 128-key tiles the row's fold visits; a zero-length row visits none and returns zeros."""
    return -(-length // KEY_TILE)


def tile_bounds(length: int) -> List[Tuple[int, int]]:
    """``[start, stop)`` list positions of each tile in fold order; positions at or after ``length`` are masked."""
    return [(t * KEY_TILE, (t + 1) * KEY_TILE) for t in range(key_tiles(length))]


def _kernels_dir() -> str:
    package = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.path.join(package, "attention", "prims_ts", "kernels")


def term_source_files() -> List[str]:
    root = _kernels_dir()
    files = []
    for rel in TERM_SOURCES:
        path = os.path.join(root, rel)
        if os.path.isdir(path):
            files += [
                f"{rel}/{n}" for n in sorted(os.listdir(path)) if n.endswith(".py")
            ]
        else:
            files.append(rel)
    return files


@functools.lru_cache(maxsize=None)
def source_digest() -> str:
    h = hashlib.sha256()
    for rel in term_source_files():
        with open(os.path.join(_kernels_dir(), rel), "rb") as f:
            h.update(rel.encode())
            h.update(f.read())
    return h.hexdigest()


def _dsl_version() -> str:
    from importlib import metadata

    try:
        return metadata.version("nvidia-cutlass-dsl")
    except metadata.PackageNotFoundError:
        return "none"


@functools.lru_cache(maxsize=None)
def spec_fingerprint(dtype, head_dim: int) -> str:
    """Identifier of one class: its constants, the pinned schedule, the digest of the kernel sources (hand-written, so
    their source is the dataflow and an edit renames the class) and the installed ``nvidia-cutlass-dsl`` version that
    compiles them. The identifier therefore differs between environments with different DSL builds."""
    name = class_name(dtype, head_dim)
    profile = PROFILES[dtype_name(dtype)]
    parts = [
        f"class={name}",
        f"latent={LATENT_DIM}",
        f"rope={int(head_dim) - LATENT_DIM}",
        f"key_tile={KEY_TILE}",
        f"query_tile={QUERY_TILE}",
        f"ctas_per_row={CTAS_PER_ROW}",
        "profile=" + ",".join(f"{k}={profile[k]}" for k in sorted(profile)),
        "out=bf16",
        "lse=f32.ln",
        "kernel=" + ",".join(f"{k}={KERNEL_STATE[k]}" for k in sorted(KERNEL_STATE)),
        f"cute_dsl={_dsl_version()}",
        f"sources={source_digest()}",
    ]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return f"{name.replace('.', '-')}-{digest}"
