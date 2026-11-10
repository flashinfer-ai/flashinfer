"""Shared-memory geometry for the standalone K3 dense MLA kernels."""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute

K3_QK_DIM = 576
K3_VALUE_DIM = 512
HEADS_PER_TILE = 8
CANDIDATES_PER_CHUNK = 64
KV_STAGES = 2
MATH_WARPS_PER_QUERY = 4


@dataclass(frozen=True)
class SmemLayout:
    query_tile: int
    fp8: bool
    kv_stages: int
    record_bytes: int
    record_stride_bytes: int
    q_stage_bytes: int
    kv_stage_bytes: int
    reduce_group_bytes: int
    reduce_bytes: int
    weight_group_bytes: int
    weight_bytes: int
    weight_scale_group_bytes: int
    weight_scale_bytes: int
    total_bytes: int


def make_smem_layout(
    *, query_tile: int, fp8: bool, qk_dim: int = K3_QK_DIM
) -> SmemLayout:
    query_tile = int(query_tile)
    if query_tile not in (1, 2, 4):
        raise ValueError("dense MLA query_tile must be 1, 2, or 4")
    if query_tile == 4 and not fp8:
        raise ValueError("the four-query dense MLA tile requires E4M3")

    element_bytes = 1 if fp8 else 2
    # Two BF16 64-record stages cannot physically fit in the 101,376-byte
    # opt-in SMEM limit exposed by RTX PRO 6000 Blackwell.  BF16 retains the
    # same producer/consumer transaction protocol with one stage; the primary
    # E4M3 serving path keeps the latency-hiding double buffer.
    qk_dim = int(qk_dim)
    if qk_dim <= 0 or qk_dim % 32:
        raise ValueError("dense MLA qk_dim must be a positive multiple of 32")
    kv_stages = KV_STAGES if fp8 and qk_dim <= K3_QK_DIM else 1
    record_bytes = qk_dim * element_bytes
    # The extra 16 bytes rotate successive rows across banks while making
    # every row a legal cp.async.bulk destination.
    record_stride_bytes = record_bytes + 16
    q_stage_bytes = query_tile * HEADS_PER_TILE * record_stride_bytes
    kv_stage_bytes = kv_stages * CANDIDATES_PER_CHUNK * record_stride_bytes

    # Per query: max and sum for four warps x eight heads.
    reduce_group_bytes = 2 * MATH_WARPS_PER_QUERY * HEADS_PER_TILE * 4
    reduce_bytes = query_tile * reduce_group_bytes

    if fp8:
        # HIGH/LOW E4M3 probability rows: 16 x (64 + 16 pad) bytes.
        weight_group_bytes = 16 * (CANDIDATES_PER_CHUNK + 16)
        # Eight scales plus eight cached reciprocals.
        weight_scale_group_bytes = 16 * 4
    else:
        # BF16 probability rows. The upper eight rows are computation-only
        # padding for m16n8k16; their resulting fragments are discarded.
        weight_group_bytes = 16 * CANDIDATES_PER_CHUNK * 2
        weight_scale_group_bytes = 4
    weight_bytes = query_tile * weight_group_bytes
    weight_scale_bytes = query_tile * weight_scale_group_bytes

    # Typed structs apply the declared alignments; instantiate below and use
    # its exact result as the launch resource contract.
    layout = SmemLayout(
        query_tile=query_tile,
        fp8=bool(fp8),
        kv_stages=kv_stages,
        record_bytes=record_bytes,
        record_stride_bytes=record_stride_bytes,
        q_stage_bytes=q_stage_bytes,
        kv_stage_bytes=kv_stage_bytes,
        reduce_group_bytes=reduce_group_bytes,
        reduce_bytes=reduce_bytes,
        weight_group_bytes=weight_group_bytes,
        weight_bytes=weight_bytes,
        weight_scale_group_bytes=weight_scale_group_bytes,
        weight_scale_bytes=weight_scale_bytes,
        total_bytes=0,
    )
    storage = get_shared_storage_cls(layout)
    return SmemLayout(
        **{
            **layout.__dict__,
            "total_bytes": int(storage.size_in_bytes()),
        }
    )


def get_shared_storage_cls(layout: SmemLayout):
    """Return the typed storage for one dtype/query-tile specialization."""

    class SharedStorage:
        pass

    SharedStorage.__annotations__ = {
        "q": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, layout.q_stage_bytes],
            128,
        ],
        "kv": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, layout.kv_stage_bytes],
            128,
        ],
        # full[kv_stages], empty[kv_stages].
        "mbar": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint64, 2 * layout.kv_stages],
            16,
        ],
        "reduce": cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, layout.reduce_bytes // 4],
            16,
        ],
        "weight_scale": cute.struct.Align[
            cute.struct.MemRange[
                cutlass.Uint8,
                max(layout.weight_scale_bytes, 1),
            ],
            16,
        ],
        "weight": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, layout.weight_bytes],
            128,
        ],
    }
    return cute.struct(SharedStorage)


__all__ = [
    "CANDIDATES_PER_CHUNK",
    "SmemLayout",
    "HEADS_PER_TILE",
    "K3_QK_DIM",
    "K3_VALUE_DIM",
    "KV_STAGES",
    "MATH_WARPS_PER_QUERY",
    "get_shared_storage_cls",
    "make_smem_layout",
]
