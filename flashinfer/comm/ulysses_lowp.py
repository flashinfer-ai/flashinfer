"""
Copyright (c) 2026 by FlashInfer team.
Copyright (c) 2026 NVIDIA Corporation.

Low-precision (INT8 Q/K + FP8 V) Ulysses all-to-all payload operations on the
V2-G global quantization grid (payload ABI v3, stats protocol 3 / ALIGN-128).

V2-G preserves SageAttention2's GLOBAL 32/64-token Q/K quantization grids
across rank boundaries.  Under ALIGN-128 every local shard is a whole number
of 128-token blocks, so no quantization group (Q 32 / K 64) can straddle a
rank boundary: each rank's locally computed grouped amax IS the final
per-group scale, the single stats AllGather carries only the K per-channel
sum and V per-channel amax, and no boundary-merge machinery exists.  The
compute side consumes the unpacked tensors on the global grid unchanged.
Attention itself is delegated to an external Sage backend (e.g. the
sageattention package's prequant entry); this module carries quantization
primitives only.

NOTE: this submodule is addressed as ``flashinfer.comm.ulysses_lowp``. Never
re-export a function named exactly ``ulysses_lowp`` from ``flashinfer.comm``:
it would shadow this submodule on the package and break attribute-based
module access (see the ulysses_a2a merge note in ``ulysses.py``).

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

import functools
import math
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, Union

import torch

from ..api_logging import flashinfer_api
from ..jit.comm import gen_ulysses_lowp_module
from ..utils import register_custom_op

ABI_VERSION = 3
# Stats protocol 3 (ALIGN-128): ONE stats AllGather carrying the K per-channel
# sum and V per-channel amax only.  The ALIGN-128 shard guarantee
# (local_sequence % 128 == 0, enforced by the unpack precondition) removes
# every boundary-crossing group, so the locally computed Q/K grouped amax are
# already final and no boundary descriptor/merge machinery exists.  Ranks
# disagreeing on this value must refuse the low-precision path group-wide.
STATS_PROTOCOL = 3
# Both stats protocols share the payload ABI: 3 = ALIGN-128 (no boundary
# machinery, aligned unpack); 2 = 64-aligned global packing (boundary
# descriptor/min-max machinery below + unpack_for_sage(aligned=False)).
SUPPORTED_STATS_PROTOCOLS = (2, 3)
HEAD_DIM = 128
Q_GROUP = 32
K_GROUP = 64
V_SCALE_MAX = 2.25
# Stage-1 chunk width of the two-stage sequence-parallel k_sum/v_amax
# reduction; fixes the fp32 sum association (see k_sum_v_amax).
KSUM_CHUNK_TOKENS = 256


# ---------------------------------------------------------------------------
# Global-grid arithmetic (pure Python, mirrored by the CUDA grid:: helpers)
# ---------------------------------------------------------------------------


def slots(local_sequence: int, group: int) -> int:
    """ceil((L+G-1)/G): groups a length-L interval can touch at ANY offset.

    This is the fixed per-source slot allocation, not the per-rank valid
    count; ranks whose offset happens to be group-aligned touch fewer groups
    and leave the surplus slots deterministically zero.
    """

    return (local_sequence + 2 * group - 2) // group


def group_first(rank: int, local_sequence: int, group: int) -> int:
    return (rank * local_sequence) // group


def group_last(rank: int, local_sequence: int, group: int) -> int:
    return (rank * local_sequence + local_sequence - 1) // group


def touched(rank: int, local_sequence: int, group: int) -> int:
    return (
        group_last(rank, local_sequence, group)
        - group_first(rank, local_sequence, group)
        + 1
    )


def owner(group_id: int, local_sequence: int, group: int) -> int:
    """Canonical owner: the smallest group-rank holding the group's start token."""

    return (group_id * group) // local_sequence


# ---------------------------------------------------------------------------
# JIT module loading
# ---------------------------------------------------------------------------


@functools.cache
def get_ulysses_lowp_module():
    module = gen_ulysses_lowp_module().build_and_load()

    @register_custom_op(
        "flashinfer::ulysses_lowp_k_sum_v_amax",
        mutates_args=["k_sum", "v_amax", "k_partial", "v_partial"],
    )
    def ulysses_lowp_k_sum_v_amax(
        k: torch.Tensor,
        v: torch.Tensor,
        k_sum: torch.Tensor,
        v_amax: torch.Tensor,
        k_partial: torch.Tensor,
        v_partial: torch.Tensor,
    ) -> None:
        module.ulysses_lowp_k_sum_v_amax(k, v, k_sum, v_amax, k_partial, v_partial)

    @register_custom_op(
        "flashinfer::ulysses_lowp_q_grouped_amax", mutates_args=["amax_out"]
    )
    def ulysses_lowp_q_grouped_amax(
        q: torch.Tensor,
        amax_out: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> None:
        module.ulysses_lowp_q_grouped_amax(q, amax_out, rank, world_size)

    @register_custom_op(
        "flashinfer::ulysses_lowp_k_grouped_amax", mutates_args=["amax_out"]
    )
    def ulysses_lowp_k_grouped_amax(
        k: torch.Tensor,
        k_mean: torch.Tensor,
        amax_out: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> None:
        module.ulysses_lowp_k_grouped_amax(k, k_mean, amax_out, rank, world_size)

    @register_custom_op(
        "flashinfer::ulysses_lowp_quant_q_int8_pack", mutates_args=["output"]
    )
    def ulysses_lowp_quant_q_int8_pack(
        q: torch.Tensor,
        q_amax_final: torch.Tensor,
        output: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> None:
        module.ulysses_lowp_quant_q_int8_pack(q, q_amax_final, output, rank, world_size)

    @register_custom_op(
        "flashinfer::ulysses_lowp_quant_kv_int8_fp8_pack", mutates_args=["output"]
    )
    def ulysses_lowp_quant_kv_int8_fp8_pack(
        k: torch.Tensor,
        v: torch.Tensor,
        k_mean: torch.Tensor,
        k_amax_final: torch.Tensor,
        v_scale: torch.Tensor,
        output: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> None:
        module.ulysses_lowp_quant_kv_int8_fp8_pack(
            k, v, k_mean, k_amax_final, v_scale, output, rank, world_size
        )

    @register_custom_op(
        "flashinfer::ulysses_lowp_quant_q_int8_pack_fused", mutates_args=["output"]
    )
    def ulysses_lowp_quant_q_int8_pack_fused(
        q: torch.Tensor,
        output: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> None:
        module.ulysses_lowp_quant_q_int8_pack_fused(q, output, rank, world_size)

    @register_custom_op(
        "flashinfer::ulysses_lowp_quant_kv_int8_fp8_pack_fused", mutates_args=["output"]
    )
    def ulysses_lowp_quant_kv_int8_fp8_pack_fused(
        k: torch.Tensor,
        v: torch.Tensor,
        k_mean: torch.Tensor,
        v_scale: torch.Tensor,
        output: torch.Tensor,
        rank: int,
        world_size: int,
        used_sequence: int,
    ) -> None:
        module.ulysses_lowp_quant_kv_int8_fp8_pack_fused(
            k, v, k_mean, v_scale, output, rank, world_size, used_sequence
        )

    @register_custom_op(
        "flashinfer::ulysses_lowp_unpack_for_sage",
        mutates_args=["q", "k", "v", "q_scale", "k_scale"],
    )
    def ulysses_lowp_unpack_for_sage(
        input: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        local_sequence: int,
        world_size: int,
    ) -> None:
        module.ulysses_lowp_unpack_for_sage(
            input, q, k, v, q_scale, k_scale, local_sequence, world_size
        )

    @register_custom_op(
        "flashinfer::ulysses_lowp_unpack_for_sage_unaligned",
        mutates_args=["q", "k", "v", "q_scale", "k_scale"],
    )
    def ulysses_lowp_unpack_for_sage_unaligned(
        input: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        local_sequence: int,
        world_size: int,
    ) -> None:
        module.ulysses_lowp_unpack_for_sage_unaligned(
            input, q, k, v, q_scale, k_scale, local_sequence, world_size
        )

    @register_custom_op(
        "flashinfer::ulysses_lowp_quant_v_fp8_with_scale", mutates_args=["output"]
    )
    def ulysses_lowp_quant_v_fp8_with_scale(
        input: torch.Tensor,
        scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        module.ulysses_lowp_quant_v_fp8_with_scale(input, scale, output)

    @register_custom_op("flashinfer::ulysses_lowp_abi_version", mutates_args=[])
    def ulysses_lowp_abi_version() -> int:
        return module.ulysses_lowp_abi_version()

    return SimpleNamespace(
        ulysses_lowp_k_sum_v_amax=ulysses_lowp_k_sum_v_amax,
        ulysses_lowp_q_grouped_amax=ulysses_lowp_q_grouped_amax,
        ulysses_lowp_k_grouped_amax=ulysses_lowp_k_grouped_amax,
        ulysses_lowp_quant_q_int8_pack=ulysses_lowp_quant_q_int8_pack,
        ulysses_lowp_quant_kv_int8_fp8_pack=ulysses_lowp_quant_kv_int8_fp8_pack,
        ulysses_lowp_quant_q_int8_pack_fused=ulysses_lowp_quant_q_int8_pack_fused,
        ulysses_lowp_quant_kv_int8_fp8_pack_fused=ulysses_lowp_quant_kv_int8_fp8_pack_fused,
        ulysses_lowp_unpack_for_sage=ulysses_lowp_unpack_for_sage,
        ulysses_lowp_unpack_for_sage_unaligned=ulysses_lowp_unpack_for_sage_unaligned,
        ulysses_lowp_quant_v_fp8_with_scale=ulysses_lowp_quant_v_fp8_with_scale,
        ulysses_lowp_abi_version=ulysses_lowp_abi_version,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _require_sm120(tensor: torch.Tensor) -> None:
    capability = torch.cuda.get_device_capability(tensor.device)
    if capability != (12, 0):
        raise RuntimeError(
            "low-precision Ulysses V2-G operations require CUDA capability "
            f"(12, 0), but device {tensor.device} reports {capability}"
        )


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _world_size(world_size: int) -> int:
    world_size = _positive_int("world_size", world_size)
    if world_size not in (2, 4, 6, 8):
        raise ValueError("V2-G requires world_size in {2,4,6,8}")
    return world_size


def _rank(rank: int, world_size: int) -> int:
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world_size:
        raise ValueError("V2-G requires 0 <= rank < world_size")
    return rank


def _validate_nhd_input(name: str, tensor: torch.Tensor) -> Tuple[int, int, int, int]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"{name} must have dtype torch.bfloat16 or torch.float16")
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [B, L, H, D]")
    batch, local_sequence, num_heads, head_dim = tensor.shape
    if batch <= 0 or local_sequence <= 0:
        raise ValueError("B and L must be non-zero")
    if num_heads <= 0:
        # Head count is parametric: TP shards heads before attention (e.g.
        # 56/TP2 = 28).  All V2-G statistics and grids are per-(b, h, d)
        # independent; divisibility by the Ulysses world size is enforced by
        # the payload spec.  D=128 stays a hard requirement.
        raise ValueError(f"V2-G requires a positive head count, got H={num_heads}")
    if head_dim != HEAD_DIM:
        raise ValueError(f"V2-G requires D={HEAD_DIM}, got D={head_dim}")
    if tensor.stride(-1) != 1:
        raise ValueError(
            f"{name} must be contiguous along head_dim; got strides {tensor.stride()}"
        )
    # Every kernel addresses the source through explicit batch/token/head
    # strides, so only head_dim has to be dense.  This admits the fused
    # QKV-projection views ([B, L, H, 3, D] sliced on the 3-axis: head stride
    # 3*D) without materializing three contiguous copies.  The kernels load
    # 16-byte vectors along head_dim, so the base pointer and the outer
    # strides must keep 16-byte alignment (8 elements of bf16/fp16).
    vec_elems = 16 // tensor.element_size()
    if tensor.data_ptr() % 16 != 0 or any(
        tensor.stride(i) % vec_elems != 0 for i in range(3)
    ):
        raise ValueError(
            f"{name} must keep 16-byte alignment of every head_dim row; "
            f"got data_ptr % 16 = {tensor.data_ptr() % 16}, strides {tensor.stride()}"
        )
    _require_sm120(tensor)
    return batch, local_sequence, num_heads, head_dim


# ---------------------------------------------------------------------------
# Capability and payload spec
# ---------------------------------------------------------------------------


@flashinfer_api
def capability(
    device: Optional[Union[int, str, torch.device]] = None,
) -> Dict[str, Any]:
    """Describe availability of the V2-G payload ABI v3.

    Supported means the compiled kernel module reports
    ``abi_version() == ABI_VERSION`` and the device capability is exactly
    ``(12, 0)`` (SM120).
    """

    device_capability = None
    if torch.cuda.is_available():
        if isinstance(device, int):
            cuda_device = torch.device("cuda", device)
        elif device is None:
            cuda_device = torch.device("cuda", torch.cuda.current_device())
        else:
            cuda_device = torch.device(device)

        if cuda_device.type == "cuda":
            if cuda_device.index is None:
                cuda_device = torch.device("cuda", torch.cuda.current_device())
            device_capability = tuple(torch.cuda.get_device_capability(cuda_device))

    try:
        compiled_abi_version = int(get_ulysses_lowp_module().ulysses_lowp_abi_version())
    except Exception:  # noqa: BLE001 — probe never raises; report ABI 0
        # mirrors the upstream getattr(..., lambda: 0)() probe: a module that
        # cannot be built or loaded reports compiled ABI 0, never raises
        compiled_abi_version = 0
    supported = bool(
        compiled_abi_version == ABI_VERSION and device_capability == (12, 0)
    )
    return {
        "abi_version": ABI_VERSION,
        "compiled_abi_version": compiled_abi_version,
        "device_capability": device_capability,
        "supported": supported,
    }


@flashinfer_api
def abi_version() -> int:
    """Return the payload ABI version compiled into the kernel module."""

    return int(get_ulysses_lowp_module().ulysses_lowp_abi_version())


@flashinfer_api
def payload_spec(
    *,
    batch_size: int,
    local_sequence: int,
    num_heads: int,
    head_dim: int,
    world_size: int,
) -> Dict[str, Union[int, float]]:
    """Return the headerless V2-G destination-chunk layout in bytes."""

    batch_size = _positive_int("batch_size", batch_size)
    local_sequence = _positive_int("local_sequence", local_sequence)
    num_heads = _positive_int("num_heads", num_heads)
    head_dim = _positive_int("head_dim", head_dim)
    world_size = _world_size(world_size)
    if head_dim != HEAD_DIM:
        raise ValueError(f"V2-G requires D={HEAD_DIM}, got D={head_dim}")
    if num_heads % world_size:
        # The only structural head constraint: equal head split across the
        # Ulysses group.  The count itself is parametric (28 under TP2, 56
        # without TP, ...).
        raise ValueError("num_heads must be divisible by world_size")

    local_heads = num_heads // world_size
    q_slots = slots(local_sequence, Q_GROUP)
    k_slots = slots(local_sequence, K_GROUP)
    main_bytes = batch_size * local_sequence * local_heads * head_dim
    q_scale_bytes = batch_size * local_heads * q_slots * 4
    k_scale_bytes = batch_size * local_heads * k_slots * 4
    q_scale_offset = 3 * main_bytes
    k_scale_offset = q_scale_offset + q_scale_bytes
    raw_chunk_bytes = k_scale_offset + k_scale_bytes
    chunk_bytes = (raw_chunk_bytes + 127) // 128 * 128
    payload_bytes = world_size * chunk_bytes
    bf16_payload_bytes = 3 * batch_size * local_sequence * num_heads * head_dim * 2
    logical_sequence = world_size * local_sequence
    return {
        "abi_version": ABI_VERSION,
        "local_heads": local_heads,
        "q_slots_per_source": q_slots,
        "k_slots_per_source": k_slots,
        "logical_sequence": logical_sequence,
        "padded_sequence": (logical_sequence + 63) // 64 * 64,
        "q_scale_alloc": (logical_sequence + 127) // 128 * 4,
        "k_scale_alloc": (logical_sequence + 63) // 64,
        "main_bytes": main_bytes,
        "q_offset": 0,
        "k_offset": main_bytes,
        "v_offset": 2 * main_bytes,
        "q_scale_offset": q_scale_offset,
        "k_scale_offset": k_scale_offset,
        "raw_chunk_bytes": raw_chunk_bytes,
        "chunk_bytes": chunk_bytes,
        "payload_bytes": payload_bytes,
        "bf16_payload_bytes": bf16_payload_bytes,
        "payload_reduction_pct": (1.0 - payload_bytes / bf16_payload_bytes) * 100.0,
    }


# ---------------------------------------------------------------------------
# Local statistics (pre-collective)
# ---------------------------------------------------------------------------


@flashinfer_api
def k_sum_v_amax(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute local FP32 K sum and V absolute maximum over sequence.

    Both inputs use canonical NHD ``[B, L, H, 128]`` storage. The returned
    ``[B, H, 128]`` tensors are local statistics; callers must synchronize
    them across the Ulysses process group before quantizing K or V.

    The reduction is two-stage sequence-parallel: stage 1 reduces fixed
    ``KSUM_CHUNK_TOKENS``-token chunks into fp32 partials — the
    ``[B, H, ceil(L/256), 128]`` fp32 workspaces are allocated here and
    passed to the kernel — and stage 2 combines the chunk partials in FIXED
    ascending chunk order, so results are deterministic (bit-identical run
    to run).  NOTE: the fp32 ``k_sum`` association differs from a single-pass
    sequential sum by ULPs (the one deliberate bit change of the two-stage
    launch-occupancy fix); ``v_amax`` is max-reduced (order-independent) and
    stays byte-identical to the single-pass form.
    """

    batch, local_sequence, num_heads, head_dim = _validate_nhd_input("k", k)
    _validate_nhd_input("v", v)
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shapes")
    if k.dtype != v.dtype:
        raise TypeError("k and v must have the same dtype")
    if k.device != v.device:
        raise ValueError("k and v must be on the same CUDA device")
    if out is None:
        k_sum = torch.empty(
            (batch, num_heads, head_dim), dtype=torch.float32, device=k.device
        )
        v_amax = torch.empty_like(k_sum)
    else:
        if not isinstance(out, tuple) or len(out) != 2:
            raise TypeError("out must be a (k_sum, v_amax) tensor tuple")
        k_sum, v_amax = out
    num_chunks = (local_sequence + KSUM_CHUNK_TOKENS - 1) // KSUM_CHUNK_TOKENS
    k_partial = torch.empty(
        (batch, num_heads, num_chunks, head_dim), dtype=torch.float32, device=k.device
    )
    v_partial = torch.empty_like(k_partial)
    get_ulysses_lowp_module().ulysses_lowp_k_sum_v_amax(
        k, v, k_sum, v_amax, k_partial, v_partial
    )
    return k_sum, v_amax


# ---------------------------------------------------------------------------
# Grouped amax (locally final under ALIGN-128)
# ---------------------------------------------------------------------------


@flashinfer_api
def q_grouped_amax(q: torch.Tensor, *, rank: int, world_size: int) -> torch.Tensor:
    """Per-touched-global-group |Q| partial amax, ``[B, H, slots(L,32)]``.

    Valid slots are ``[0, touched)``; the surplus allocation stays zero.
    """

    batch, local_sequence, num_heads, _ = _validate_nhd_input("q", q)
    world_size = _world_size(world_size)
    rank = _rank(rank, world_size)
    amax = torch.zeros(
        (batch, num_heads, slots(local_sequence, Q_GROUP)),
        dtype=torch.float32,
        device=q.device,
    )
    get_ulysses_lowp_module().ulysses_lowp_q_grouped_amax(q, amax, rank, world_size)
    return amax


@flashinfer_api
def k_grouped_amax(
    k: torch.Tensor,
    k_mean_global: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    used_sequence: Optional[int] = None,
) -> torch.Tensor:
    """Per-touched-global-group |K - global mean| partial amax, ``[B, H, slots(L,64)]``.

    ``used_sequence`` is the live global row count when the caller's packed
    sequence carries zero-filled tail padding in rows ``[used, S)``.  Unlike Q
    and V, whose amax is taken on the raw values (a zero row contributes 0 and
    cannot raise any group's max), the K amax is taken on the MEAN-SUBTRACTED
    ``Kc = K - k_mean_global``: a zero padding row contributes
    ``|0 - k_mean| = |k_mean|``, which under the smooth-k regime (large channel
    means, small deviations -- the very regime smooth_k exists for) dominates
    the live rows' amax and silently inflates the tail group's scale.  The
    CUDA kernel has no notion of ``used``, so when padding is present this
    recomputes the partial live group's slot over the live rows only, with
    the exact pinned math (fp32 convert, subtract, abs, order-independent
    max, 1e-7 floor), so the pack kernel's per-group scales see the corrected
    value.  With ``used_sequence is None`` or ``used == S`` nothing is
    touched, keeping the no-padding path bit-identical to the original kernel
    output.

    ALIGN-128 relaxation: the tail padding may span many trailing K groups.
    Only ONE group can mix live and padding rows -- the partial live group
    ``(used-1)//64`` -- and only there does the zero padding pollute a scale
    that live rows consume.  Fully-padded groups beyond it are never read by
    the compute (rows are sliced to ``used``); their kernel-produced values
    are deterministic don't-cares.
    """

    batch, local_sequence, num_heads, head_dim = _validate_nhd_input("k", k)
    world_size = _world_size(world_size)
    rank = _rank(rank, world_size)
    if not isinstance(k_mean_global, torch.Tensor):
        raise TypeError("k_mean_global must be a torch.Tensor")
    if k_mean_global.dtype != k.dtype:
        raise TypeError("k_mean_global must have the same dtype as k")
    if tuple(k_mean_global.shape) != (batch, num_heads, head_dim):
        raise ValueError(f"k_mean_global must have shape {(batch, num_heads, head_dim)}")
    if not k_mean_global.is_contiguous() or k_mean_global.device != k.device:
        raise ValueError("k_mean_global must be contiguous and on the K device")
    global_sequence = local_sequence * world_size
    if used_sequence is not None and not 0 < int(used_sequence) <= global_sequence:
        raise ValueError("used_sequence must lie in (0, local_sequence * world_size]")
    amax = torch.zeros(
        (batch, num_heads, slots(local_sequence, K_GROUP)),
        dtype=torch.float32,
        device=k.device,
    )
    get_ulysses_lowp_module().ulysses_lowp_k_grouped_amax(
        k, k_mean_global, amax, rank, world_size
    )
    if (
        used_sequence is not None
        and int(used_sequence) < global_sequence
        and int(used_sequence) % K_GROUP
    ):
        used = int(used_sequence)
        # ALIGN-128: the tail padding may span many trailing K groups.  Only
        # ONE group can mix live and padding rows -- the partial live group
        # (used-1)//64 -- and only there does the zero padding pollute a
        # scale that live rows consume.  Fully-padded groups beyond it are
        # never read by the compute (rows are sliced to ``used``); their
        # kernel-produced values are deterministic don't-cares.  When
        # ``used`` is a multiple of the group size there is no partial group
        # and nothing to repair (the branch condition above skips).  Correct
        # only the ranks that touch the partial group; each overwrites its
        # own partial with the live-rows amax (a group straddling ranks --
        # unit-test shapes only under ALIGN-128 -- still works: every
        # touching rank repairs its own slice).
        tail_group = (used - 1) // K_GROUP
        g_first = (rank * local_sequence) // K_GROUP
        g_last = (rank * local_sequence + local_sequence - 1) // K_GROUP
        if g_first <= tail_group <= g_last:
            lo = max(tail_group * K_GROUP, rank * local_sequence)
            hi = min(used, (rank + 1) * local_sequence)
            lo_local = lo - rank * local_sequence
            hi_local = hi - rank * local_sequence
            if hi_local > lo_local:
                kc = k[:, lo_local:hi_local].float() - k_mean_global.float().unsqueeze(1)
                live = kc.abs().amax(dim=(1, 3)).clamp_(min=1e-7)
            else:
                # this rank's slice of the tail group is padding only
                live = torch.full(
                    (batch, num_heads), 1e-7, dtype=torch.float32, device=k.device
                )
            amax[..., tail_group - g_first] = live
    return amax


# ---------------------------------------------------------------------------
# Boundary machinery (stats protocol 2: 64-aligned global packing)
# ---------------------------------------------------------------------------


@flashinfer_api
def boundary_descriptors(
    grouped_amax: torch.Tensor,
    *,
    rank: int,
    local_sequence: int,
    group: int,
    world_size: int,
) -> torch.Tensor:
    """Extract the ``[B, H, 2]`` boundary descriptor for one stats collective.

    slot0 is the first touched group's partial amax and slot1 the last's.
    When the rank touches a single group both slots carry the same value; the
    downstream max merge is idempotent, so no dedup special case is needed.
    """

    world_size = _world_size(world_size)
    rank = _rank(rank, world_size)
    touched_count = touched(rank, local_sequence, group)
    first = grouped_amax[..., 0:1]
    last = grouped_amax[..., touched_count - 1 : touched_count]
    return torch.cat([first, last], dim=-1).contiguous()


@flashinfer_api
def merge_boundary_amax(
    grouped_amax: torch.Tensor,
    gathered_descriptors: torch.Tensor,
    *,
    rank: int,
    local_sequence: int,
    group: int,
    world_size: int,
) -> torch.Tensor:
    """Overwrite this rank's boundary slots with the cross-rank max merge.

    ``gathered_descriptors`` is ``[P, B, H, 2]`` in group-rank order.  For each
    of this rank's (at most two) boundary groups, the final amax is the max of
    the partial amax from every rank whose interval intersects that group.
    The merge uses only ``max`` (exact, order-independent), so the result is
    bit-identical on every participating rank.  Returns ``grouped_amax``
    (modified in place) for convenience.
    """

    world_size = _world_size(world_size)
    rank = _rank(rank, world_size)
    if gathered_descriptors.shape[0] != world_size or gathered_descriptors.shape[-1] != 2:
        raise ValueError("gathered_descriptors must have shape [P, B, H, 2]")
    my_first = group_first(rank, local_sequence, group)
    touched_count = touched(rank, local_sequence, group)
    for boundary_group in {my_first, my_first + touched_count - 1}:
        parts = []
        for other in range(world_size):
            other_first = group_first(other, local_sequence, group)
            other_last = group_last(other, local_sequence, group)
            if not other_first <= boundary_group <= other_last:
                continue
            slot = 0 if boundary_group == other_first else 1
            parts.append(gathered_descriptors[other, ..., slot])
        merged = parts[0]
        for part in parts[1:]:
            merged = torch.maximum(merged, part)
        grouped_amax[..., boundary_group - my_first] = merged
    return grouped_amax


@flashinfer_api
def k_boundary_minmax(
    k: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    used_sequence: Optional[int] = None,
) -> torch.Tensor:
    """Per-channel raw-K min/max of this rank's two K boundary slices.

    Returns ``[B, H, 2, 2, D]`` fp32: dim2 is the boundary slot (0 = first
    touched global 64-group's slice, 1 = last touched group's; when the rank
    touches a single group both slots describe the same slice, and the
    downstream max-combine is idempotent).  dim3 is (min, max).

    Stats-protocol 2 gathers this INSTEAD of the mean-dependent K boundary
    amax: raw-K min/max needs no global mean, so it rides in AllGather #1,
    and after ``k_mean_global`` is derived every rank reconstructs each
    boundary group's exact |K - mean| amax locally (see
    :func:`derive_k_boundary_amax`), eliminating the second collective.

    fp32 transport is exact: every BF16/FP16 value converts to fp32 without
    rounding, and min/max are selections.  ``used_sequence`` applies the
    live-rows rule one stage earlier: rows ``[used, S)`` are zero-filled
    padding whose inclusion would contribute ``min=0/max=0`` and hence
    ``|0 - mean| = |mean|`` after the derive -- exactly the tail-group
    pollution the k_grouped_amax fix removes.  A slice that is entirely
    padding gets the sentinels ``min=+inf, max=-inf``; the derive turns those
    into ``-inf`` before its 1e-7 floor, i.e. the same "this rank contributes
    nothing" value the two-collective path used.

    ``used_sequence`` must satisfy the padding admission condition
    ``ceil(used_sequence/64) == ceil(S/64)`` (all padding inside the single
    last global K group); violations raise :class:`ValueError`.
    """

    batch, local_sequence, num_heads, head_dim = _validate_nhd_input("k", k)
    world_size = _world_size(world_size)
    rank = _rank(rank, world_size)
    global_sequence = local_sequence * world_size
    if used_sequence is not None:
        if not 0 < int(used_sequence) <= global_sequence:
            raise ValueError("used_sequence must lie in (0, local_sequence * world_size]")
        # Enforced admission condition (comment-only precondition upstream):
        # all padding must live in the single last global K group.
        if (int(used_sequence) + K_GROUP - 1) // K_GROUP != (
            global_sequence + K_GROUP - 1
        ) // K_GROUP:
            raise ValueError(
                "used_sequence must satisfy ceil(used_sequence/64) == ceil(S/64): "
                "all tail padding must lie inside the last global K group"
            )
    used = int(used_sequence) if used_sequence is not None else global_sequence

    out = torch.empty(
        (batch, num_heads, 2, 2, head_dim), dtype=torch.float32, device=k.device
    )
    g_first = group_first(rank, local_sequence, K_GROUP)
    g_last = group_last(rank, local_sequence, K_GROUP)
    base = rank * local_sequence
    for slot, group_id in enumerate((g_first, g_last)):
        lo = max(group_id * K_GROUP, base)
        hi = min((group_id + 1) * K_GROUP, base + local_sequence, used)
        if hi > lo:
            # [B, rows, H, D] slice in local coordinates; fp32 convert is
            # exact for BF16/FP16, min/max are selections (no rounding).
            rows = k[:, lo - base : hi - base].float()
            out[:, :, slot, 0] = rows.amin(dim=1)
            out[:, :, slot, 1] = rows.amax(dim=1)
        else:
            out[:, :, slot, 0] = float("inf")
            out[:, :, slot, 1] = float("-inf")
    return out


@flashinfer_api
def derive_k_boundary_amax(
    grouped_amax: torch.Tensor,
    gathered_minmax: torch.Tensor,
    k_mean_global: torch.Tensor,
    *,
    rank: int,
    local_sequence: int,
    world_size: int,
) -> torch.Tensor:
    """Overwrite this rank's K boundary slots with the derived cross-rank amax.

    ``gathered_minmax`` is ``[P, B, H, 2, 2, D]`` fp32 in group-rank order
    (from AllGather #1 under stats protocol 2).  For each of this rank's (at
    most two) boundary groups g and every rank r touching g:

        contrib(r, g) = max_d max( rn(maxK[r,g,d] - m[d]),
                                   rn(m[d]  - minK[r,g,d]) )   floored at 1e-7

    which is bit-equal to r's |K - m| partial amax over its slice of g: fp32
    round-to-nearest subtraction is monotone in its tensor argument, so
    ``max_t rn(k_t - m) = rn(max_t k_t - m)`` and likewise for the min side,
    and ``max_t |rn(k_t - m)| = max(rn(maxK - m), rn(m - minK))`` exactly.
    The final slot value is ``max_r contrib(r, g)`` -- the same values the
    retired AllGather #2 merge produced, computed from identical gathered
    inputs with identical exact ops on every rank, hence bit-identical
    group-wide without a second collective.  A padding-only slice's sentinels
    (min=+inf, max=-inf) yield ``-inf`` before the floor, i.e. contribute
    1e-7 exactly as the two-collective path did.  Non-crossing boundary
    groups (single toucher) reduce to the kernel's own partial, so the
    unconditional overwrite is bit-neutral there.
    """

    world_size = _world_size(world_size)
    rank = _rank(rank, world_size)
    if gathered_minmax.shape[0] != world_size or gathered_minmax.shape[-3:-1] != (2, 2):
        raise ValueError("gathered_minmax must have shape [P, B, H, 2, 2, D]")
    mean32 = k_mean_global.float()
    my_first = group_first(rank, local_sequence, K_GROUP)
    touched_count = touched(rank, local_sequence, K_GROUP)
    for boundary_group in {my_first, my_first + touched_count - 1}:
        merged = None
        for other in range(world_size):
            other_first = group_first(other, local_sequence, K_GROUP)
            other_last = group_last(other, local_sequence, K_GROUP)
            if not other_first <= boundary_group <= other_last:
                continue
            slot = 0 if boundary_group == other_first else 1
            mn = gathered_minmax[other, :, :, slot, 0]
            mx = gathered_minmax[other, :, :, slot, 1]
            contrib = (
                torch.maximum(mx - mean32, mean32 - mn)
                .amax(dim=-1)
                .clamp_(min=1e-7)
            )
            merged = contrib if merged is None else torch.maximum(merged, contrib)
        grouped_amax[..., boundary_group - my_first] = merged
    return grouped_amax



# ---------------------------------------------------------------------------
# Quantize-and-pack into the headerless payload
# ---------------------------------------------------------------------------


def _validate_send(
    send_u8: torch.Tensor,
    *,
    batch_size: int,
    local_sequence: int,
    num_heads: int,
    world_size: int,
) -> Dict[str, Union[int, float]]:
    if not isinstance(send_u8, torch.Tensor):
        raise TypeError("send_u8 must be a torch.Tensor")
    spec = payload_spec(
        batch_size=batch_size,
        local_sequence=local_sequence,
        num_heads=num_heads,
        head_dim=HEAD_DIM,
        world_size=world_size,
    )
    if not send_u8.is_cuda or send_u8.dtype != torch.uint8:
        raise ValueError("send_u8 must be a CUDA uint8 tensor")
    if tuple(send_u8.shape) != (world_size, spec["chunk_bytes"]):
        raise ValueError(
            f"send_u8 must have shape {(world_size, spec['chunk_bytes'])}, "
            f"got {tuple(send_u8.shape)}"
        )
    if not send_u8.is_contiguous():
        raise ValueError("send_u8 must be contiguous")
    return spec


@flashinfer_api
def zero_scale_and_padding(
    send_u8: torch.Tensor, spec: Dict[str, Union[int, float]]
) -> None:
    """Deterministically zero every scale slot and the alignment tail.

    Must run before the first V2-G pack launch: the pack kernels write only
    the touched slots, and the ABI requires every unused byte to be zero.
    """

    send_u8[:, int(spec["q_scale_offset"]) :].zero_()


@flashinfer_api
def quant_q_into_payload(
    q: torch.Tensor,
    q_amax_final: torch.Tensor,
    send_u8: torch.Tensor,
    *,
    rank: int,
    world_size: int,
) -> None:
    """Quantize Q on the global grid directly into the V2-G payload."""

    batch, local_sequence, num_heads, _ = _validate_nhd_input("q", q)
    world_size = _world_size(world_size)
    rank = _rank(rank, world_size)
    _validate_send(
        send_u8,
        batch_size=batch,
        local_sequence=local_sequence,
        num_heads=num_heads,
        world_size=world_size,
    )
    if send_u8.device != q.device or q_amax_final.device != q.device:
        raise ValueError("q, q_amax_final, and send_u8 must share a device")
    get_ulysses_lowp_module().ulysses_lowp_quant_q_int8_pack(
        q, q_amax_final, send_u8, rank, world_size
    )


@flashinfer_api
def quant_kv_into_payload(
    k: torch.Tensor,
    v: torch.Tensor,
    k_mean_global: torch.Tensor,
    k_amax_final: torch.Tensor,
    v_scale_global: torch.Tensor,
    send_u8: torch.Tensor,
    *,
    rank: int,
    world_size: int,
) -> None:
    """Quantize K (global grid) and V (global per-channel) into the payload."""

    batch, local_sequence, num_heads, head_dim = _validate_nhd_input("k", k)
    _validate_nhd_input("v", v)
    world_size = _world_size(world_size)
    rank = _rank(rank, world_size)
    if k.shape != v.shape or k.dtype != v.dtype or k.device != v.device:
        raise ValueError("k and v must have identical shape, dtype, and device")
    if k_mean_global.dtype != k.dtype:
        raise TypeError("k_mean_global must have the same dtype as k")
    if v_scale_global.dtype != torch.float32:
        raise TypeError("v_scale_global must have dtype torch.float32")
    for name, tensor in (("k_mean_global", k_mean_global), ("v_scale_global", v_scale_global)):
        if tuple(tensor.shape) != (batch, num_heads, head_dim):
            raise ValueError(f"{name} must have shape {(batch, num_heads, head_dim)}")
        if not tensor.is_contiguous() or tensor.device != k.device:
            raise ValueError(f"{name} must be contiguous and on the K device")
    _validate_send(
        send_u8,
        batch_size=batch,
        local_sequence=local_sequence,
        num_heads=num_heads,
        world_size=world_size,
    )
    if send_u8.device != k.device:
        raise ValueError("K/V, statistics, and send_u8 must share a device")
    get_ulysses_lowp_module().ulysses_lowp_quant_kv_int8_fp8_pack(
        k, v, k_mean_global, k_amax_final, v_scale_global, send_u8, rank, world_size
    )


@flashinfer_api
def quant_qkv_pack(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_mean_global: torch.Tensor,
    q_amax_final: torch.Tensor,
    k_amax_final: torch.Tensor,
    v_scale_global: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused V2-G quantize-and-pack convenience entry point."""

    batch, local_sequence, num_heads, _ = _validate_nhd_input("q", q)
    if q.shape != k.shape or q.shape != v.shape or q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have identical shape and dtype")
    spec = payload_spec(
        batch_size=batch,
        local_sequence=local_sequence,
        num_heads=num_heads,
        head_dim=HEAD_DIM,
        world_size=world_size,
    )
    if out is None:
        out = torch.empty(
            (world_size, spec["chunk_bytes"]), dtype=torch.uint8, device=q.device
        )
    zero_scale_and_padding(out, spec)
    quant_q_into_payload(q, q_amax_final, out, rank=rank, world_size=world_size)
    quant_kv_into_payload(
        k,
        v,
        k_mean_global,
        k_amax_final,
        v_scale_global,
        out,
        rank=rank,
        world_size=world_size,
    )
    return out


@flashinfer_api
def quant_qkv_pack_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_mean_global: torch.Tensor,
    v_scale_global: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    used_sequence: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused amax+quant V2-G pack (ALIGN-128 fast path).

    Byte-identical to ``q_grouped_amax`` + ``k_grouped_amax`` +
    ``quant_qkv_pack`` but reads Q/K from HBM once: each CTA loads its group,
    reduces the per-group amax in-block, and quantizes from registers.  Legal
    only when the locally computed amax IS the final scale, i.e. under
    ALIGN-128 (``local_sequence % 128 == 0``); protocol 2 keeps the split
    path because a collective sits between amax and quant for its boundary
    groups.  ``used_sequence`` applies the K tail-group repair in-kernel with
    the exact split-path semantics.
    """

    batch, local_sequence, num_heads, _ = _validate_nhd_input("q", q)
    _validate_nhd_input("k", k)
    _validate_nhd_input("v", v)
    if q.shape != k.shape or q.shape != v.shape or q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have identical shape and dtype")
    if local_sequence % 128:
        raise ValueError(
            "quant_qkv_pack_fused is an ALIGN-128 fast path: local_sequence "
            f"must be a whole number of 128-token blocks, got {local_sequence}"
        )
    world_size = _world_size(world_size)
    rank = _rank(rank, world_size)
    global_sequence = local_sequence * world_size
    if used_sequence is not None and not 0 < int(used_sequence) <= global_sequence:
        raise ValueError("used_sequence must lie in (0, local_sequence * world_size]")
    spec = payload_spec(
        batch_size=batch,
        local_sequence=local_sequence,
        num_heads=num_heads,
        head_dim=HEAD_DIM,
        world_size=world_size,
    )
    if out is None:
        out = torch.empty(
            (world_size, spec["chunk_bytes"]), dtype=torch.uint8, device=q.device
        )
    _validate_send(
        out,
        batch_size=batch,
        local_sequence=local_sequence,
        num_heads=num_heads,
        world_size=world_size,
    )
    zero_scale_and_padding(out, spec)
    mod = get_ulysses_lowp_module()
    mod.ulysses_lowp_quant_q_int8_pack_fused(q, out, rank, world_size)
    mod.ulysses_lowp_quant_kv_int8_fp8_pack_fused(
        k, v, k_mean_global, v_scale_global, out, rank, world_size,
        int(used_sequence) if used_sequence is not None else 0,
    )
    return out


# ---------------------------------------------------------------------------
# Receiver unpack
# ---------------------------------------------------------------------------


@flashinfer_api
def unpack_for_sage(
    recv_u8: torch.Tensor,
    *,
    batch_size: int,
    local_sequence: int,
    local_heads: int,
    head_dim: int,
    world_size: int,
    aligned: bool = True,
    out: Optional[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack source chunks into pre-quant compute inputs on the global grid.

    Returns contiguous logical Q/K ``[B,S,h,128]``, globally packed V
    ``[B,128,h,ceil(S/64)*64]`` with a zero global tail, and global-grid Q/K
    scale tensors whose unused tail slots are deterministically zero.
    """

    batch_size = _positive_int("batch_size", batch_size)
    local_sequence = _positive_int("local_sequence", local_sequence)
    local_heads = _positive_int("local_heads", local_heads)
    head_dim = _positive_int("head_dim", head_dim)
    world_size = _world_size(world_size)
    if aligned and local_sequence % 128:
        raise ValueError(
            "ALIGN-128 (stats protocol 3): local_sequence must be a whole "
            f"number of 128-token blocks, got {local_sequence}; pass "
            "aligned=False for the protocol-2 (64-aligned global) receiver"
        )
    spec = payload_spec(
        batch_size=batch_size,
        local_sequence=local_sequence,
        num_heads=local_heads * world_size,
        head_dim=head_dim,
        world_size=world_size,
    )
    if not isinstance(recv_u8, torch.Tensor):
        raise TypeError("recv_u8 must be a torch.Tensor")
    if not recv_u8.is_cuda or recv_u8.dtype != torch.uint8:
        raise ValueError("recv_u8 must be a CUDA uint8 tensor")
    if tuple(recv_u8.shape) != (world_size, spec["chunk_bytes"]):
        raise ValueError(
            f"recv_u8 must have shape {(world_size, spec['chunk_bytes'])}, "
            f"got {tuple(recv_u8.shape)}"
        )
    if not recv_u8.is_contiguous():
        raise ValueError("recv_u8 must be contiguous")
    _require_sm120(recv_u8)

    logical_sequence = int(spec["logical_sequence"])
    padded_sequence = int(spec["padded_sequence"])
    if out is None:
        q_logical = torch.empty(
            (batch_size, logical_sequence, local_heads, head_dim),
            dtype=torch.int8,
            device=recv_u8.device,
        )
        k_logical = torch.empty_like(q_logical)
        v_packed = torch.empty(
            (batch_size, head_dim, local_heads, padded_sequence),
            dtype=torch.float8_e4m3fn,
            device=recv_u8.device,
        )
        q_scale = torch.empty(
            (batch_size, local_heads, int(spec["q_scale_alloc"])),
            dtype=torch.float32,
            device=recv_u8.device,
        )
        k_scale = torch.empty(
            (batch_size, local_heads, int(spec["k_scale_alloc"])),
            dtype=torch.float32,
            device=recv_u8.device,
        )
    else:
        if not isinstance(out, tuple) or len(out) != 5:
            raise TypeError("out must be a five-tensor global-grid Sage tuple")
        q_logical, k_logical, v_packed, q_scale, k_scale = out
    fn = (
        get_ulysses_lowp_module().ulysses_lowp_unpack_for_sage
        if aligned
        else get_ulysses_lowp_module().ulysses_lowp_unpack_for_sage_unaligned
    )
    fn(
        recv_u8,
        q_logical,
        k_logical,
        v_packed,
        q_scale,
        k_scale,
        local_sequence,
        world_size,
    )
    return q_logical, k_logical, v_packed, q_scale, k_scale


@flashinfer_api
def verify_duplicate_scale_slots(
    recv_u8: torch.Tensor,
    *,
    batch_size: int,
    local_sequence: int,
    local_heads: int,
    head_dim: int,
    world_size: int,
) -> bool:
    """Debug/test-only check: every cross-boundary scale slot is bit-identical
    on all sources that carry it.  Runs as a read-only pass after the A2A, so
    there is no concurrent-writer hazard.  Never call on the hot path.
    """

    spec = payload_spec(
        batch_size=batch_size,
        local_sequence=local_sequence,
        num_heads=local_heads * world_size,
        head_dim=head_dim,
        world_size=world_size,
    )
    chunks = recv_u8.view(world_size, -1)
    for group, offset_key, slots_key in (
        (Q_GROUP, "q_scale_offset", "q_slots_per_source"),
        (K_GROUP, "k_scale_offset", "k_slots_per_source"),
    ):
        offset = int(spec[offset_key])
        slot_count = int(spec[slots_key])
        count = batch_size * local_heads * slot_count
        views = [
            chunks[src, offset : offset + count * 4]
            .view(torch.float32)
            .view(batch_size, local_heads, slot_count)
            for src in range(world_size)
        ]
        total_groups = (world_size * local_sequence + group - 1) // group
        for g in range(total_groups):
            holders = [
                src
                for src in range(world_size)
                if group_first(src, local_sequence, group)
                <= g
                <= group_last(src, local_sequence, group)
            ]
            if len(holders) < 2:
                continue
            reference = None
            for src in holders:
                slot = g - group_first(src, local_sequence, group)
                value = views[src][..., slot]
                if reference is None:
                    reference = value
                elif not torch.equal(reference, value):
                    return False
    return True


# ---------------------------------------------------------------------------
# Canonical FP8 V bytes (shared helper, outside the packed payload)
# ---------------------------------------------------------------------------


@flashinfer_api
def quant_v_fp8_with_scale(
    v: torch.Tensor,
    v_scale_global: torch.Tensor,
    scale_max: float = V_SCALE_MAX,
) -> torch.Tensor:
    """Quantize canonical NHD V to canonical E4M3 bit patterns.

    Args:
        v: Contiguous ``[B, S, H, 128]`` BF16/FP16 CUDA tensor.
        v_scale_global: Contiguous ``[B, H, 128]`` FP32 global per-channel
            divisor. The payload contract requires it to be produced as
            ``global_amax / 2.25`` from BF16/FP16 V values; this preserves
            pinned Sage FP8 bits.
        scale_max: Format guard. Only ``2.25`` is supported.

    Returns:
        A contiguous ``[B, S, H, 128]`` uint8 tensor containing raw
        ``torch.float8_e4m3fn`` bit patterns.
    """

    _require_tensor("v", v)
    _require_tensor("v_scale_global", v_scale_global)
    if not v.is_cuda or not v_scale_global.is_cuda:
        raise ValueError("v and v_scale_global must be CUDA tensors")
    if v.device != v_scale_global.device:
        raise ValueError("v and v_scale_global must be on the same CUDA device")
    if v.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("v must have dtype torch.bfloat16 or torch.float16")
    if v_scale_global.dtype != torch.float32:
        raise TypeError("v_scale_global must have dtype torch.float32")
    if v.ndim != 4:
        raise ValueError("v must have shape [B, S, H, D]")
    if v_scale_global.ndim != 3:
        raise ValueError("v_scale_global must have shape [B, H, D]")
    if not v.is_contiguous() or not v_scale_global.is_contiguous():
        raise ValueError("v and v_scale_global must be contiguous")

    batch, sequence, heads, head_dim = v.shape
    if batch <= 0 or sequence <= 0 or heads <= 0:
        raise ValueError("B, S, and H must all be non-zero")
    if head_dim != HEAD_DIM:
        raise ValueError(f"quant_v_fp8_with_scale requires D={HEAD_DIM}, got D={head_dim}")
    if tuple(v_scale_global.shape) != (batch, heads, head_dim):
        raise ValueError(
            "v_scale_global must have shape "
            f"{(batch, heads, head_dim)}, got {tuple(v_scale_global.shape)}"
        )
    if isinstance(scale_max, bool) or not isinstance(scale_max, (int, float)):
        raise TypeError("scale_max must be a finite real number")
    scale_max = float(scale_max)
    if not math.isfinite(scale_max) or scale_max != V_SCALE_MAX:
        raise ValueError(f"quant_v_fp8_with_scale requires scale_max={V_SCALE_MAX}")

    _require_sm120(v)
    output = torch.empty_like(v, dtype=torch.uint8, memory_format=torch.contiguous_format)
    get_ulysses_lowp_module().ulysses_lowp_quant_v_fp8_with_scale(
        v, v_scale_global, output
    )
    return output


__all__ = [
    "ABI_VERSION",
    "HEAD_DIM",
    "KSUM_CHUNK_TOKENS",
    "K_GROUP",
    "Q_GROUP",
    "STATS_PROTOCOL",
    "V_SCALE_MAX",
    "abi_version",
    "capability",
    "gen_ulysses_lowp_module",
    "get_ulysses_lowp_module",
    "group_first",
    "group_last",
    "k_grouped_amax",
    "k_sum_v_amax",
    "owner",
    "payload_spec",
    "q_grouped_amax",
    "boundary_descriptors",
    "derive_k_boundary_amax",
    "k_boundary_minmax",
    "merge_boundary_amax",
    "SUPPORTED_STATS_PROTOCOLS",
    "quant_kv_into_payload",
    "quant_q_into_payload",
    "quant_qkv_pack",
    "quant_qkv_pack_fused",
    "quant_v_fp8_with_scale",
    "slots",
    "touched",
    "unpack_for_sage",
    "verify_duplicate_scale_slots",
    "zero_scale_and_padding",
]
