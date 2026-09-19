# SPDX-License-Identifier: Apache-2.0
"""Opt-in Ulysses experiments. Stable communicator behavior is unchanged.

These entry points do not select TP/EP/CP groups or automatically fall back.
See ``docs/design_docs/ulysses_architecture_experiments.md`` before use.
"""

from ..api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def pack_ulysses_qkv_fp8(query, key, value, scales, *, world_size, out):
    """Quantize BF16 QKV into caller-owned destination-major E4M3 storage.

    Q/K/V are [S_local,H,D]; scales are three globally agreed [H] FP32
    dequantization factors. Output is [U,S_local,H/U,3D]. No collective
    or scale reduction is performed. Calling this function is the opt-in.
    """
    from ..experimental.ulysses.fp8 import quant_pack

    return quant_pack(query, key, value, scales, world_size=world_size, out=out)


@flashinfer_experimental_api
def prepare_ulysses_producer(
    weight, *, world_size, heads, head_dim, local_seq, schedule
):
    """Prepare coarse QKV GEMMs spanning every destination rank per group.

    Weight is frozen BF16 [3*H*D,K], with Q, K, V planes in that order.
    ``schedule`` partitions H/U, not H. The returned inference-only object
    exposes ``produce(x, group_index)``; output views alias reusable storage.
    The framework owns readiness events, postprocessing and communication.
    """
    from ..experimental.ulysses.producer import GroupedQKVProducer

    return GroupedQKVProducer(
        weight,
        world_size=world_size,
        heads=heads,
        head_dim=head_dim,
        local_seq=local_seq,
        schedule=schedule,
    )


@flashinfer_experimental_api
def prepare_ulysses_k_mean(*, sequence, heads, head_dim, device, dtype):
    """Allocate whole-head scratch for shape-stable per-chunk K mean.

    The returned object's ``update(chunk, head_offset=...)`` uses the same
    full-head reduction geometry for every chunk. This is a Sage integration
    building block, not a new Sage Attention backend or quantization recipe.
    """
    from ..experimental.ulysses.producer import ShapeStableKMean

    return ShapeStableKMean(sequence, heads, head_dim, device=device, dtype=dtype)


@flashinfer_experimental_api
def prepare_ulysses_distributed_fa4(
    *, group, local_seq, used_seqlen, full_nvlink, discard_tail_output
):
    """Collectively construct the pinned BF16 SM100 prefix-attention runner.

    Requires B=1,H=56,D=128, U=2/4/8, full NVLink and a preconfigured
    NVSHMEM symmetric-memory backend. All ranks must call with matching
    metadata. All ranks must have applied the bundled FA4 patch themselves.
    No automatic patching, process-group creation or runtime fallback occurs.
    ``full_nvlink=True`` is the caller's topology attestation, not a probe.
    Returned outputs alias workspace; use it serially on one stream.
    """
    from ..experimental.ulysses.sm100 import prepare

    return prepare(
        group=group,
        local_seq=local_seq,
        used_seqlen=used_seqlen,
        full_nvlink=full_nvlink,
        discard_tail_output=discard_tail_output,
    )
