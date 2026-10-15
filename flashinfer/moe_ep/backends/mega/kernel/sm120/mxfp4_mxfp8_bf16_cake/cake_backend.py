# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FlashInfer MegaKernelBackend for the Cake SM120 MXFP4 x MXFP8 hybrid-transport MegaMoE.

One cooperative persistent kernel per MoE layer call performs dispatch, task
build, W1 + fused SwiGLU/requant, W2 (plus the optional in-kernel FP8 shared
expert), BF16 return, top-k combine and acknowledgement.  Every pair of EP
ranks is served by the transport the per-pair table assigns to it: LSA
load/store over NVLink or PCIe peer mappings (bulk copies by copier CTAs) or
NCCL GIN puts, so one kernel covers a single PCIe host, an NVLink host and
multi-host deployments with LSA inside each host and GIN across hosts.

The device code is one parametric CUDA source.  The deployment geometry — EP world
size, global (physical) expert count, top-k, hidden and intermediate size, per-rank
row capacity and the shared-expert variant — are its compile-time parameters; the
backend derives them from ``FleetParams`` / ``BootstrapConfig`` / the kernel config,
checks them against the generated geometry's guards in ``validate_init`` and JIT
compiles the instances it launches once per parameter set.  EPLB redundant experts
need nothing special: ``num_experts`` is the physical expert count (logical experts
plus replicas) and ``topk_ids`` carry physical expert ids.

The host side owns:

* one transport session per workspace (adopted from the EP process group's
  NCCL communicator): symmetric windows, LSA peer pointers, GIN device comm;
* the transport table (setup probe or a pinned table) and, from it, the
  instance pair to launch: with or without the LSA copier CTAs;
* the two row schedules of that pair (M64 for at most ``m64_max_rows`` tokens
  on every rank, M128 otherwise), chosen per call from the largest per-rank row
  count, exactly as the generator's qualification harness chose them;
* fixed-capacity scratch shared by both schedules (each schedule binds views
  with its own extents), TMA descriptor carriers, and the kernel-ordered
  argument list, rebuilt per call from a static template.

Per call the host needs a rank-uniform bound on the per-rank row counts
(heterogeneous counts and zero-row ranks are supported): it selects the row
schedule, fills the high half of ``active_rows`` and skips a call in which no rank
holds a token.  A caller that already knows the largest row count of the call
across the EP ranks passes it as the token bucket of the invocation
(``MoEEpMegaLayer.stage_inputs(t, compile_tokens_per_rank=...)``, the same value on
every rank, at least the largest row count; the hook #4632 added for CUDA-graph
token buckets).  The call then has no host synchronization and the cooperative
grid is sized from the bound alone (:func:`rows_residency_grid`).  Without it,
one small all-gather of ``[rows, expert histogram]`` gives every rank the row
counts and the per-rank task counts that size the grid exactly as the
generator's qualification harness did (:func:`residency_grid`); that read-back
is a host synchronization.  A bound above the true largest count is correct
(the kernel only requires ``bound >= rows`` on every rank) but spends some
per-call setup on the rows it does not have and may select M128 where M64 would
do; passing the exact largest count reproduces the exchanged schedule.

CUDA graphs: a call with ``compile_tokens_per_rank`` has no host-side dependency on
device data -- the row schedule, grid and ``active_rows`` come from the bound, the
launch arguments are copied by value into the captured kernel node, and the
kernel keeps its epoch in device memory -- so it can be captured once the layer
has run eagerly on its workspace (the weights' TMA descriptors are uploaded on
first use).  A graph replays the bound and the row count it was captured with:
capture one graph per token bucket, and every EP rank must replay graphs captured
with the same bucket, in the same order, like the eager calls.  Capture without a
bound is refused (the routing exchange reads device data back).
"""

from __future__ import annotations

import ctypes
import struct
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from ......config import BootstrapConfig, FleetParams
from ......core.kernel.base import MegaKernelBackend
from ......core.kernel.registry import register_mega_kernel
from ......core.runtime import TORCH_DIST
from ......core.validation.common import (
    MoEEpArchError,
    MoEEpConfigError,
    validate_mega_fleet_params,
)
from ......weights import MoEWeightPack
from .cake_config import (
    TRANSPORT_TABLE_AUTO,
    TRANSPORT_TABLE_TUNED,
    Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig,
)
from .cake_geometry import GeometryError, KernelParams, describe_kernel, kernel_max_rows
from .cake_session import TransportSession
from .cake_staging import stage_inputs, validate_forward_inputs
from .cake_transport import (
    TRANSPORT_GIN,
    TRANSPORT_LSA,
    TransportTable,
    gather_lsa_pair_matrix,
    lsa_copier_count,
    lsa_where_reachable,
    probe_traffic,
    select_transport_table,
    uniform_table,
    validate_table,
)
from .cake_weights import (
    TransformedCakeMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors

KERNEL_NAME = "sm120_mxfp4_mxfp8_bf16_cake"
_TENSOR_MAP_BYTES = 128
_PER_CALL_NAMES = (
    "W1_B",
    "W1_B8",
    "W1_SFB",
    "W2_B",
    "W2_B8",
    "W2_SFB",
    "topk_idx_i32",
    "topk_weights",
    "x_fp8_i32",
    "x_sf_i32",
    "active_rows",
)
_LAYER_DESCRIPTORS = ("W1_B", "W1_B8", "W1_SFB", "W2_B", "W2_B8", "W2_SFB")
_STATIC_DESCRIPTORS = ("W1_A", "W1_SFA", "W1_D", "W2_A", "W2_SFA", "W2_D")
_KERNEL_WINDOWS = (
    "dispatch_header_out",
    "dispatch_payload_out",
    "dispatch_header_inbox",
    "dispatch_payload_inbox",
    "result_out",
    "result_inbox",
    "ack_out",
    "ack_inbox",
)
_TMA_DTYPE_CODE = {
    "u8": 0,
    "i8": 0,
    "u16": 1,
    "u32": 2,
    "i32": 3,
    "u64": 4,
    "i64": 5,
    "bf16": 6,
    "f16": 7,
    "f32": 8,
}
_TMA_ELEM_BYTES = {
    "u8": 1,
    "i8": 1,
    "u16": 2,
    "u32": 4,
    "i32": 4,
    "u64": 8,
    "i64": 8,
    "bf16": 2,
    "f16": 2,
    "f32": 4,
}
_TMA_SWIZZLE_CODE = {
    (0, 0, 0): 0,
    (4, 1, 3): 1,
    (4, 2, 3): 2,
    (4, 3, 3): 3,
    (5, 2, 2): 4,
}
_TMA_L2_CODE = {"none": 0, "l2_64b": 1, "l2_128b": 2, "l2_256b": 3}
_CTYPE_WIDTH = {
    "ptr": 8,
    "c_void_p": 8,
    "c_int8": 1,
    "c_uint8": 1,
    "c_bool": 1,
    "c_int16": 2,
    "c_uint16": 2,
    "c_int32": 4,
    "c_uint32": 4,
    "c_int": 4,
    "c_uint": 4,
    "c_float": 4,
    "c_int64": 8,
    "c_uint64": 8,
    "c_double": 8,
}
_FILL_ZERO, _FILL_MINUS_ONE, _FILL_NONE = 0, -1, None
_GEOMETRY_ITEMSIZE = {
    "torch.uint8": 1,
    "torch.int8": 1,
    "torch.bfloat16": 2,
    "torch.int32": 4,
    "torch.uint32": 4,
    "torch.float32": 4,
    "torch.int64": 8,
    "torch.uint64": 8,
}


def _scalar_bits(abi: str, value: Any) -> int:
    """Raw little-endian integer image of one scalar argument."""

    if abi == "c_float":
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]
    if abi == "c_double":
        return struct.unpack("<Q", struct.pack("<d", float(value)))[0]
    if abi == "c_bool":
        return 1 if value else 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"scalar argument expects an integer, got {type(value).__name__}"
        )
    width = _CTYPE_WIDTH[abi]
    bits = 8 * width
    unsigned = abi.startswith("c_u")
    lower, upper = (
        (0, (1 << bits) - 1)
        if unsigned
        else (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
    )
    if not lower <= value <= upper:
        raise OverflowError(f"scalar {value} outside {abi} range")
    return value & ((1 << bits) - 1) if value < 0 else value


def _as_int64_image(raw: int) -> int:
    """Reinterpret an unsigned 64-bit image as a signed int64 for a torch tensor."""

    return raw - (1 << 64) if raw >= (1 << 63) else raw


def scratch_specs(
    C: dict[str, int], *, task_gate_u64: bool
) -> list[tuple[str, torch.dtype, tuple[int, ...], int | None]]:
    """Every kernel scratch buffer of one instance: (name, dtype, shape, fill).

    Mirrors the generator's qualification harness: the same shapes, dtypes and
    one-time initial values (zero, ``-1`` or left uninitialized); the kernel
    re-initializes its per-launch state itself.
    """

    P, ES, MR, T = C["PHYSICAL_RANKS"], C["EXPERT_SLOTS"], C["MAX_ROWS"], C["MAX_TASKS"]
    PR, DC, RC = C["POOL_ROWS"], C["DISPATCH_CHUNKS"], C["RESULT_CHUNKS"]
    i32, u32, u64 = torch.int32, torch.uint32, torch.uint64
    specs: list[tuple[str, torch.dtype, tuple[int, ...], int | None]] = [
        ("pool_fp8_u32", i32, (PR, C["ACTIVATION_WORDS"]), _FILL_ZERO),
        ("pool_sf_u32", i32, (C["SCALE_WORDS"], PR), _FILL_ZERO),
        ("routing_weight_pool", torch.float32, (PR,), _FILL_ZERO),
    ]
    for name in ("meta_source_rank", "meta_token", "meta_slot", "meta_result_index"):
        specs.append((name, i32, (PR,), _FILL_MINUS_ONE))
    for name, size in (
        ("expert_counts", ES),
        ("owner_expert_route_counts", P * ES),
        ("owner_minexp_record_base", P * ES),
        ("owner_minexp_record_cursor", P * ES),
        ("sorted_record_token", P * MR),
        ("sorted_record_route_base", P * MR),
        ("dispatch_chunk_targets", P * DC),
        ("source_route_sum", P),
        ("source_expert_counts", P * ES),
        ("expert_source_base", ES * P),
        ("expert_source_offsets", ES * P),
        ("source_expert_prefix", ES * P),
        ("task_max_source", T),
        ("source_record_counts", P),
        ("source_route_counts", P),
        ("source_active_rows", P),
        ("expert_row_offsets", ES),
        ("expert_task_base", ES),
        ("task_rows_landed", T),
        ("tb_pub", C["TB_PUB_ALLOC_WORDS"]),
        ("expert_block_task", ES * T),
        ("task_source_slot_base", T * P),
        ("expert_scatter_offsets", ES),
        ("total_valid_routes", 1),
        ("total_padded_rows", 1),
        ("total_m_tasks", 1),
        ("histogram_done", 1),
        ("prefix_done", 1),
        ("requant_groups_done", 1),
        ("w1_warp_done", T * C["W1_N_BLOCKS"]),
        ("w1_tiles_completed", 1),
        ("w2_warp_done", T * C["W2_N_BLOCKS"]),
        ("w2_tiles_completed", 1),
        ("c56_claim_cursor", 1),
        ("combine_claim_cursor", 1),
        ("c56_tile_mailbox", C["C56_MB_ROWS"] * C["C56_MB_STRIDE"]),
        ("owner_record_counts", P),
        ("owner_route_counts", P),
        ("protocol_error", 1),
        ("result_chunk_total", P * RC),
        ("result_chunk_tally", P * RC),
        ("result_ovf_cursor", P * RC),
        ("result_copy_cursor", C["RESULT_COPY_SLOTS"]),
        ("pr_gq", C["PR_GQ_CAP"]),
        ("pr_gq_ctl", C["PR_GQ_CTL_WORDS"]),
    ):
        specs.append((name, i32, (size,), _FILL_ZERO))
    specs.append(
        (
            "task_gate_packed",
            u64 if task_gate_u64 else i32,
            (T * C["TASK_GATE_WORDS"],),
            _FILL_ZERO,
        )
    )
    for name in (
        "task_expert",
        "task_source_rank",
        "task_owner_rank",
        "task_local_expert",
        "task_pool_row",
        "task_m_local",
        "task_valid_m",
    ):
        specs.append((name, i32, (T,), _FILL_MINUS_ONE))
    specs.append(("route_result_index", i32, (MR * C["TOP_K"],), _FILL_MINUS_ONE))
    specs += [
        ("W1_D", torch.bfloat16, (PR, C["W1_N"]), _FILL_NONE),
        ("W2_D", torch.bfloat16, (PR, C["W2_N"]), _FILL_NONE),
        ("intermediate_fp8", torch.uint8, (PR, C["INTERMEDIATE"]), _FILL_NONE),
        (
            "intermediate_sfa_u8",
            torch.uint8,
            (C["REQUANT_PACKED_GROUPS"], PR, 4),
            _FILL_NONE,
        ),
        ("shared_out", torch.bfloat16, (MR, C["HIDDEN"]), _FILL_NONE),
        ("final_output", torch.bfloat16, (MR, C["HIDDEN"]), _FILL_ZERO),
    ]
    for name, size in (
        ("result_signal_base_scratch", 2 * P),
        # index P is the deferred-ack-credit armed flag (0 before the first launch)
        ("ack_signal_base_scratch", P + 1),
        ("dispatch_chunk_signal_base_scratch", P * DC),
        ("phase_timestamps", C["PHASE_TIMESTAMP_COUNT"] + C["DIAG_WORDS"]),
        ("peer_phase_timestamps", C["PEER_PHASE_TIMESTAMP_COUNT"]),
        ("pull_request_scratch", P * DC * 2),
        ("signal_base_scratch", P),
    ):
        specs.append((name, u64, (size,), _FILL_ZERO))
    for name, size in (
        ("w2_task_counter", T),
        ("w1_task_counter", T),
        ("dispatch_chunk_scatter_counter", P * DC),
        ("pull_chunk_arrived", P * DC),
        ("pull_chunk_copied", P * DC),
        ("c24_front_sync", C["C24_SYNC_SLOTS"]),
        ("result_owner_ready", P),
        ("result_owner_progress", P),
        ("result_copy_done", C["RESULT_COPY_SLOTS"]),
        # Device-resident epoch: the kernel reads it (ring-slot parity, dispatch-header epoch)
        # and advances it by one at the end of every launch, so a call needs no host scalar.
        # Shared by both schedules (one storage per name), like every other scratch buffer.
        ("epoch_state", 1),
    ):
        specs.append((name, u32, (size,), _FILL_ZERO))
    return specs


def _check_scratch_extent(
    spec: dict[str, Any], name: str, itemsize: int, shape: tuple[int, ...]
) -> None:
    """Cross-check one host scratch extent against the generated geometry's buffer table."""

    buffers = spec.get("buffers", {})
    if name not in buffers or not isinstance(buffers[name][1], tuple):
        return
    dtype, dims = buffers[name]
    expected = int(torch.Size(dims).numel()) * _GEOMETRY_ITEMSIZE[dtype]
    if int(torch.Size(shape).numel()) * itemsize != expected:
        raise RuntimeError(
            f"{KERNEL_NAME}: scratch {name} has {int(torch.Size(shape).numel()) * itemsize} bytes, "
            f"the kernel geometry expects {expected}"
        )


# Local memory per thread of a spill-free instance: the generated code's own 8-byte stack
# frame (every compiled instance of the shipped test matrix reports stack 8 / local 0).  A
# register spill grows it (one spilled register = 16 bytes), so anything above this limit is
# refused.
LOCAL_BYTES_LIMIT = 8
# Register-budget evidence (generator-side compile grid): all four instances of both variants
# compile without spills at EP 2..16 for DeepSeek-V4-Flash, DeepSeek-V3, Qwen3-235B-A22B,
# Qwen3-30B-A3B, a padded GPT-OSS shape, H512/top-1, top-9 with 96 local experts and
# H7680 with one local expert (registers up to the 168 cap).  The M128 copier-free instance
# runs at the 168-register cap; before the device-resident epoch it spilled one register at
# 32 ranks for DeepSeek-V3, both Qwen3 shapes and H7680, and the shipped source compiles all of
# them (and DeepSeek-V4-Flash, GPT-OSS-padded, 288 physical experts) spill-free at EP 32.
# Above 16 ranks the shipped grid is not exhaustive, so ``validate_init`` still compiles the
# instances a deployment may launch (same JIT cache the workspace uses) and rejects any that
# spill.
REGISTER_BUDGET_VERIFIED_MAX_RANKS = 16


def candidate_instances(transport_table: Any) -> tuple[int, ...]:
    """Instance indices a deployment may launch: a pinned all-GIN table never needs copiers;
    any other table (or the setup probe) may choose either flavor."""

    if transport_table == "all_gin":
        return (2, 3)  # m128_nocopier, m64_nocopier
    return (0, 1, 2, 3)


def check_device_resources(
    params: KernelParams,
    instances: tuple[int, ...],
    *,
    resource_usage: Any = None,
) -> dict[int, dict[str, int]]:
    """Fail closed on parameter sets whose device build spills (no GPU needed).

    Within the verified region (``params.ranks <= REGISTER_BUDGET_VERIFIED_MAX_RANKS``)
    nothing is compiled.  Above it each candidate instance is compiled (cached) and its
    stack + local bytes per thread must not exceed :data:`LOCAL_BYTES_LIMIT`, the same
    limit the workspace enforces when it loads the instance.
    """

    if params.ranks <= REGISTER_BUDGET_VERIFIED_MAX_RANKS:
        return {}
    if resource_usage is None:
        from ......kernel_src.sm120.cake_mxfp4_megamoe import (
            device_cubin_resource_usage,
        )

        resource_usage = device_cubin_resource_usage
    from .cake_geometry import INSTANCE_NAMES

    report: dict[int, dict[str, int]] = {}
    for index in instances:
        usage = dict(resource_usage(params.device_parameters(), index))
        report[index] = usage
        if usage["stack"] + usage["local"] > LOCAL_BYTES_LIMIT:
            raise GeometryError(
                f"the {INSTANCE_NAMES[index]} instance spills registers at this parameter set "
                f"(REG {usage['reg']}, stack {usage['stack']} B, local {usage['local']} B per "
                f"thread; limit {LOCAL_BYTES_LIMIT} B)"
            )
    return report


def residency_grid(
    *,
    rows_by_rank: list[int],
    expert_counts: list[int],
    local_experts: int,
    task_m: int,
    with_shared_expert: bool,
    capacity: int,
    C: dict[str, int],
    resident_peer_fold: bool,
) -> int:
    """Rank-uniform cooperative grid of one launch (the generator harness's rule).

    One coordinator CTA plus workers; workers are bounded by the minimum cooperative
    capacity over the ranks and by the minimum per-rank W1/W2 tile count, so every
    worker owns a tile in both persistent GEMM phases.  A rank without tasks is sized
    like one task (it still hosts its dispatch/scatter/copier roles).  The grid never
    goes below the kernel's structural minimum (``FRONT_EXCLUDED_CTAS + world + 1``):
    with few tiles per task (small hidden / intermediate) and a rank without tasks the
    tile bound can be smaller at large EP; the extra workers claim no tile and exit.
    """

    world = len(rows_by_rank)
    tile_blocks = min(C["W1_N_BLOCKS"], C["W2_N_BLOCKS"])
    min_tiles = None
    for rank in range(world):
        counts = expert_counts[rank * local_experts : (rank + 1) * local_experts]
        tasks = sum((c + task_m - 1) // task_m for c in counts)
        if with_shared_expert:
            tasks += (rows_by_rank[rank] + task_m - 1) // task_m
        tiles = max(tasks, 1) * tile_blocks
        min_tiles = tiles if min_tiles is None else min(min_tiles, tiles)
    assert min_tiles is not None
    copier_ctas = C["PR_COPIER_CTAS_EFFECTIVE"]
    if resident_peer_fold:
        if capacity < copier_ctas + 2:
            raise RuntimeError(
                f"{KERNEL_NAME}: resident capacity {capacity} cannot host the {copier_ctas}-CTA "
                "copier pool plus coordinator and math worker"
            )
        workers = min(capacity - 1, max(copier_ctas + 1, min_tiles + copier_ctas))
    else:
        workers = min(capacity - 1, max(1, min_tiles))
    workers = min(capacity - 1, max(workers, C["FRONT_EXCLUDED_CTAS"] + world))
    grid = workers + 1
    if workers > C["C56_MB_ROWS"]:
        raise RuntimeError(
            f"{KERNEL_NAME}: the tile mailbox holds {C['C56_MB_ROWS']} worker CTAs, grid wants {workers}"
        )
    if grid < C["FRONT_EXCLUDED_CTAS"] + world + 1:
        raise RuntimeError(
            f"{KERNEL_NAME}: residency grid {grid} below the structural minimum "
            f"{C['FRONT_EXCLUDED_CTAS'] + world + 1}"
        )
    if workers < 1 or grid > capacity:
        raise RuntimeError(
            f"{KERNEL_NAME}: invalid rank-uniform grid workers={workers} capacity={capacity}"
        )
    return grid


def static_residency_grid(
    *,
    world: int,
    local_experts: int,
    task_m: int,
    with_shared_expert: bool,
    capacity: int,
    C: dict[str, int],
    resident_peer_fold: bool,
) -> int:
    """Routing-independent grid for a launch whose routing is not known on the host.

    :func:`residency_grid` sizes every rank as at least one task, so the empty-routing grid
    is a lower bound of the grid for any routing and still gives every worker a tile in
    both GEMM phases.  It needs no per-call host data (a prerequisite for CUDA-graph
    capture), at the cost of fewer workers than a routing-sized grid.  Not used by
    :meth:`compute` (calls without caller row counts size the grid from the gathered
    routing, calls with them from the rows).
    """

    return residency_grid(
        rows_by_rank=[0] * world,
        expert_counts=[0] * (world * local_experts),
        local_experts=local_experts,
        task_m=task_m,
        with_shared_expert=with_shared_expert,
        capacity=capacity,
        C=C,
        resident_peer_fold=resident_peer_fold,
    )


def even_expert_counts(
    *, rows_by_rank: list[int], top_k: int, num_experts: int
) -> list[int]:
    """Global expert histogram of the most even routing of ``sum(rows) * top_k`` routes."""

    world = len(rows_by_rank)
    local_experts = num_experts // world
    base, extra = divmod(sum(rows_by_rank) * top_k, num_experts)
    counts = [base] * num_experts
    # The leftover routes (fewer than num_experts) go one per expert, owners in turn.
    for i in range(extra):
        counts[(i % world) * local_experts + i // world] += 1
    return counts


def rows_residency_grid(
    *,
    rows_by_rank: list[int],
    top_k: int,
    num_experts: int,
    task_m: int,
    with_shared_expert: bool,
    capacity: int,
    C: dict[str, int],
    resident_peer_fold: bool,
) -> int:
    """Rank-uniform grid of a launch whose routing histogram is not known on the host.

    The routing-sized grid of the most even routing with the same per-rank row
    counts.  Every grid from :func:`static_residency_grid` up to the cooperative
    capacity is valid for any routing: the persistent GEMM phases claim W1/W2 tiles
    from one device-wide cursor, and a worker whose claim is past the last tile posts
    the end sentinel and exits -- what every worker of a rank without tasks already
    does under :func:`residency_grid`.  A skewed routing only leaves workers idle;
    a tile's math does not depend on the worker that runs it.
    """

    return residency_grid(
        rows_by_rank=rows_by_rank,
        expert_counts=even_expert_counts(
            rows_by_rank=rows_by_rank, top_k=top_k, num_experts=num_experts
        ),
        local_experts=num_experts // len(rows_by_rank),
        task_m=task_m,
        with_shared_expert=with_shared_expert,
        capacity=capacity,
        C=C,
        resident_peer_fold=resident_peer_fold,
    )


@dataclass
class _LayerDescriptors:
    """TMA carriers for one layer's weights (the transformed weights stay caller-owned)."""

    carriers: torch.Tensor  # uint8 [6, 128]
    slots: dict[str, int]  # name -> row of `carriers`
    owner: TransformedCakeMegaWeights


@dataclass
class _Instance:
    """One loaded kernel instance (row schedule) of the selected copier flavor."""

    name: str
    function: int
    smem_bytes: int
    task_m: int
    constants: dict[str, int]
    flags: dict[str, bool]
    capacity: int  # minimum cooperative capacity over the EP ranks
    arg_values: torch.Tensor  # CPU int64 [n] static template, per-call slots rewritten
    buffers: dict[str, torch.Tensor]  # views with this instance's extents
    static_carriers: torch.Tensor


@dataclass
class _CakeWorkspace:
    module: Any
    session: TransportSession
    kernel: dict[str, Any]
    transport_table: TransportTable
    transport_probe_report: dict[str, Any] | None
    copiers_active: bool
    instances: dict[str, _Instance]  # "m64" / "m128"
    plan: list[dict[str, Any]]
    arg_widths: torch.Tensor  # CPU int8 [n]
    slots: dict[str, list[int]]
    storage: dict[str, torch.Tensor]
    inputs: dict[str, torch.Tensor]
    peer_transport: torch.Tensor
    lsa_peer_ptrs: torch.Tensor
    lsa_send_state: torch.Tensor
    device: torch.device
    group: Any
    rank: int
    world_size: int
    threads: int
    variant: str
    max_rows: int
    hidden: int
    top_k: int
    num_experts: int
    local_experts: int
    m64_max_rows: int
    stats: torch.Tensor  # int32 [1 + E] per-call row count + expert histogram
    gathered_stats: torch.Tensor  # int32 [world, 1 + E]
    epoch: int = (
        0  # host count of launches (diagnostics); the kernel keeps its own epoch_state
    )
    staged_rows: int = 0
    # Rank-uniform per-rank row bound of the invocation (``compile_tokens_per_rank``).
    compile_tokens_per_rank: int | None = None
    staged_bound: int | None = None
    last_launch: dict[str, Any] = field(default_factory=dict)
    layers: dict[int, _LayerDescriptors] = field(default_factory=dict)
    destroyed: bool = False

    @property
    def output(self) -> torch.Tensor:
        return (
            self.storage["final_output"]
            .view(torch.bfloat16)
            .view(self.max_rows, self.hidden)
        )

    def destroy(self) -> None:
        if self.destroyed:
            return
        self.destroyed = True
        torch.cuda.synchronize(self.device)
        self.layers.clear()
        self.session.close()
        self.storage.clear()
        self.instances.clear()


@register_mega_kernel(KERNEL_NAME)
class Sm120Mxfp4Mxfp8CakeMegaKernelBackend(MegaKernelBackend):
    supports_output_view = True

    def __init__(self, config: Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig) -> None:
        if not isinstance(config, Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig):
            raise TypeError(
                f"{KERNEL_NAME} config must be Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig, got {type(config).__name__}"
            )
        super().__init__(config)
        self._kernel_config = config
        if config.intermediate_size <= 0 or config.top_k <= 0:
            raise ValueError("intermediate_size and top_k must be positive")
        if not config.fast_math:
            raise NotImplementedError("fast_math=False is not supported")
        self._transformed: TransformedCakeMegaWeights | None = None

    # ------------------------------------------------------------------ setup

    @classmethod
    def kernel_name(cls) -> str:
        return KERNEL_NAME

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        del bootstrap
        # The transport session is created from the EP process group's NCCL
        # communicator; NVSHMEM is not used.
        return frozenset({TORCH_DIST})

    def kernel_params(
        self, world_size: int, fleet_params: FleetParams, lsa_copiers: int = 1
    ) -> KernelParams:
        """Compile-time parameters of the device build serving this deployment.

        EP world size, global (physical) expert count, hidden size and the per-rank token
        capacity come from the fleet; intermediate size, top-k, the shared-expert
        variant, the SwiGLU clamp and the codec overflow capacity from the kernel config;
        ``lsa_copiers`` (copier instances only) from the transport table.  The kernel's
        row capacity is ``max_tokens_per_rank`` rounded up to a multiple of 128.
        """

        config = self._kernel_config
        return KernelParams(
            ranks=world_size,
            experts=fleet_params.num_experts,
            top_k=config.top_k,
            max_rows=kernel_max_rows(fleet_params.max_tokens_per_rank),
            hidden=fleet_params.token_hidden_size,
            intermediate=config.intermediate_size,
            shared_expert=config.variant == "fp8shared",
            swiglu_limit=config.swiglu_limit,
            codec_ovf_blocks=config.codec_overflow_blocks,
            lsa_copiers=lsa_copiers,
        )

    def _describe(
        self, world_size: int, fleet_params: FleetParams, lsa_copiers: int = 1
    ) -> dict[str, Any]:
        try:
            return describe_kernel(
                self.kernel_params(world_size, fleet_params, lsa_copiers)
            )
        except GeometryError as error:
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: unsupported configuration (world_size={world_size}, "
                f"num_experts={fleet_params.num_experts}, hidden={fleet_params.token_hidden_size}, "
                f"intermediate={self._kernel_config.intermediate_size}, top_k={self._kernel_config.top_k}, "
                f"swiglu_limit={self._kernel_config.swiglu_limit}, "
                f"codec_overflow_blocks={self._kernel_config.codec_overflow_blocks}, "
                f"lsa_copiers={lsa_copiers}, "
                f"max_tokens_per_rank={fleet_params.max_tokens_per_rank}): {error}"
            ) from error

    def validate_init(
        self, bootstrap: BootstrapConfig, fleet_params: FleetParams
    ) -> None:
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            # The device instances are built for sm_120a, which does not load on 12.1.
            if capability != (12, 0):
                raise MoEEpArchError(
                    f"{KERNEL_NAME} requires sm_120 (compute capability 12.0); host has "
                    f"sm_{capability[0]}{capability[1]}"
                )
        validate_mega_fleet_params(
            fleet_params,
            bootstrap.world_size,
            intermediate_size=self._kernel_config.intermediate_size,
            top_k=self._kernel_config.top_k,
            alignment=128,
        )
        table = self._kernel_config.transport_table
        world = bootstrap.world_size
        lsa_copiers = 1
        if isinstance(table, tuple):
            if len(table) != world**2:
                raise MoEEpConfigError(
                    f"transport_table has {len(table)} cells, expected {world**2}"
                )
            lsa_copiers = lsa_copier_count(TransportTable(world, table, "override"))
        elif table == "all_lsa":
            lsa_copiers = world - 1
        # Fail closed at construction on any parameter set the device source rejects
        # (the same guards the source checks with static_assert / #error) or whose
        # device build would spill registers (checked by compiling above the verified
        # rank range; the workspace reuses the cached cubins).  Tables resolved at
        # workspace creation ("lsa_where_reachable", "auto") are checked again there
        # with their copier count.
        kernel = self._describe(world, fleet_params, lsa_copiers)
        try:
            check_device_resources(
                kernel["params"],
                candidate_instances(self._kernel_config.transport_table),
            )
        except GeometryError as error:
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: unsupported configuration "
                f"({kernel['key']}, swiglu_limit={self._kernel_config.swiglu_limit}): {error}"
            ) from error

    def preprocess_weights(
        self, weights: MoEWeightPack, fleet_params: FleetParams
    ) -> TransformedCakeMegaWeights:
        transformed = preprocess_mega_weights(
            weights,
            hidden_size=fleet_params.token_hidden_size,
            intermediate_size=self._kernel_config.intermediate_size,
            shared_expert=self._kernel_config.shared_expert,
        )
        self._transformed = transformed
        return transformed

    def validate_transformed_weights(
        self,
        transformed_weights: TransformedCakeMegaWeights,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        validate_transformed_mega_weights(
            transformed_weights,
            hidden_size=fleet_params.token_hidden_size,
            intermediate_size=self._kernel_config.intermediate_size,
            local_experts=fleet_params.num_experts // bootstrap.world_size,
            shared_expert=self._kernel_config.shared_expert,
        )
        self._transformed = transformed_weights

    def _workspace_pool_key(self, fleet_params: FleetParams) -> Any:
        from ......core.kernel.workspace_pool import knobs_pool_key

        config = self._kernel_config
        return (
            KERNEL_NAME,
            torch.cuda.current_device(),
            self.ep_rank,
            self.ep_world_size,
            id(self._ep_comm_group),
            fleet_params.num_experts,
            fleet_params.max_tokens_per_rank,
            fleet_params.token_hidden_size,
            config.intermediate_size,
            config.top_k,
            config.swiglu_limit,
            config.codec_overflow_blocks,
            config.variant,
            config.transport_table,
            knobs_pool_key(config.knobs),
        )

    # -------------------------------------------------------------- workspace

    def _choose_table(
        self,
        module: Any,
        session: TransportSession,
        kernel: dict[str, Any],
        group: Any,
    ) -> tuple[TransportTable, dict[str, Any] | None]:
        spec = self._kernel_config.transport_table
        world = session.world_size
        report = None
        if spec == TRANSPORT_TABLE_AUTO:
            transport = kernel["transport"]
            table, report = select_transport_table(
                module,
                session,
                group=group,
                signal_base=int(transport["dispatch_signal_base"]),
                signal_count=int(transport["signal_count"]),
                signal_stride=int(transport["lsa_signal_stride"]),
                traffic=probe_traffic(kernel),
            )
            pair_ok: Any = report["lsa_pair_matrix"]
        else:
            pair_ok = gather_lsa_pair_matrix(session.lsa_reachable, group)
            if spec == "all_gin":
                table = uniform_table(world, TRANSPORT_GIN)
            elif spec == "all_lsa":
                table = uniform_table(world, TRANSPORT_LSA)
            elif spec == "lsa_where_reachable":
                table = lsa_where_reachable(world, pair_ok)
            elif spec == TRANSPORT_TABLE_TUNED:
                table, report = self._tuned_table(session, kernel, pair_ok, group)
            else:
                table = TransportTable(world, tuple(int(c) for c in spec), "override")
        validate_table(table, pair_ok, session.gin_available)
        return table, report

    def _tuned_table(
        self,
        session: TransportSession,
        kernel: dict[str, Any],
        pair_ok: Any,
        group: Any,
    ) -> tuple[TransportTable, dict[str, Any]]:
        """Knob-cache lookup of the tuned table (rank 0 reads, the EP group agrees)."""

        import torch.distributed as dist

        from .cake_knob_cache import lookup_knobs, topology_digest

        world = session.world_size
        key = {
            "variant": self._kernel_config.variant,
            "world_size": world,
            "hidden": int(kernel["hidden"]),
            "intermediate": int(kernel["intermediate"]),
            "num_experts": world * int(kernel["local_experts"]),
            "topk": int(kernel["top_k"]),
            "topology": topology_digest(pair_ok, session.gin_available),
        }
        payload: list[Any] = [None]
        if dist.get_rank(group) == 0:
            payload = [lookup_knobs(max_tokens=int(kernel["max_rows"]), **key)]
        dist.broadcast_object_list(
            payload, src=dist.get_global_rank(group, 0), group=group
        )
        knobs = payload[0]
        cells = None if knobs is None else knobs.get("transport_table")
        if cells is None:
            table = uniform_table(world, TRANSPORT_GIN)
            source = "default"
        else:
            name = str(knobs.get("table_name", "override"))
            table = TransportTable(world, tuple(int(c) for c in cells), f"tuned:{name}")
            source = "knob_cache"
        return table, {"source": source, "key": key, "knobs": knobs}

    def _allocate_workspace(self, fleet_params: FleetParams) -> _CakeWorkspace:
        import torch.distributed as dist

        from ......kernel_src.sm120.cake_mxfp4_megamoe import (
            device_cubin,
            load_cake_sm120_megamoe_module,
        )

        group = self.ep_comm_group
        rank, world_size = self.ep_rank, self.ep_world_size
        device = torch.device("cuda", torch.cuda.current_device())
        module = load_cake_sm120_megamoe_module(device=device)
        kernel = self._describe(world_size, fleet_params)
        session_windows = list(kernel["windows"])
        abi = kernel["abi"]
        variant = self._kernel_config.variant

        # -- transport session (collective over the EP group) -----------------
        gin = kernel["gin"]
        session = TransportSession(
            module,
            group=group,
            comm_pointer=self._nccl_comm_pointer(group),
            rank=rank,
            world_size=world_size,
            device=device,
            windows=[(str(n), int(b)) for n, b in session_windows],
            gin_context_count=int(gin["context_count"]),
            gin_signal_count=int(gin["signal_count"]),
            gin_world_barrier_count=int(gin["world_barrier_count"]),
            gin_strong_signals_required=bool(gin["strong_signals_required"]),
        )
        try:
            table, report = self._choose_table(module, session, kernel, group)
            copiers = table.count(TRANSPORT_LSA) != 0
            if copiers:
                # The copier instances reserve one copier CTA per LSA destination of a
                # rank (MOE_LSA_COPIERS); the session (windows, signals) does not
                # depend on it.
                kernel = self._describe(
                    world_size, fleet_params, lsa_copier_count(table)
                )
                if kernel["windows"] != session_windows:
                    raise RuntimeError(
                        f"{KERNEL_NAME}: symmetric windows depend on the copier count"
                    )
            workspace = self._build_workspace(
                module,
                session,
                kernel,
                abi,
                table,
                report,
                copiers,
                fleet_params,
                group,
                rank,
                world_size,
                device,
                variant,
                device_cubin,
            )
        except BaseException:
            session.close()
            raise
        torch.cuda.current_stream(device).synchronize()
        dist.barrier(group=group)
        return workspace

    def _build_workspace(
        self,
        module: Any,
        session: TransportSession,
        kernel: dict[str, Any],
        abi: dict[str, Any],
        table: TransportTable,
        report: dict[str, Any] | None,
        copiers: bool,
        fleet_params: FleetParams,
        group: Any,
        rank: int,
        world_size: int,
        device: torch.device,
        variant: str,
        device_cubin: Any,
    ) -> _CakeWorkspace:
        import torch.distributed as dist

        suffix = "_copier" if copiers else "_nocopier"
        specs = {name: kernel["instances"][name + suffix] for name in ("m128", "m64")}
        for name, spec in specs.items():
            if spec["copiers_active"] != copiers:
                raise RuntimeError(
                    f"{KERNEL_NAME}: instance {name}{suffix} copier flag mismatch"
                )
        for desc in _LAYER_DESCRIPTORS:
            if specs["m128"]["descriptors"][desc] != specs["m64"]["descriptors"][desc]:
                raise RuntimeError(
                    f"{KERNEL_NAME}: weight descriptor {desc} differs between schedules"
                )
        C128 = specs["m128"]["constants"]
        threads = int(kernel["threads"])

        # -- shared scratch storage (max extent over both schedules) ----------
        per_instance = {
            name: scratch_specs(
                spec["constants"], task_gate_u64=bool(spec["flags"]["_TASK_GATE_U64"])
            )
            for name, spec in specs.items()
        }
        storage: dict[str, torch.Tensor] = {}
        views: dict[str, dict[str, torch.Tensor]] = {name: {} for name in specs}
        for index, (name, dtype, _, fill) in enumerate(per_instance["m128"]):
            entries = [per_instance[inst][index] for inst in specs]
            if any(e[0] != name or e[1] != dtype for e in entries):
                raise RuntimeError(
                    f"{KERNEL_NAME}: scratch layout differs between schedules at {name}"
                )
            itemsize = torch.empty((), dtype=dtype).element_size()
            for inst, entry in zip(specs, entries, strict=True):
                _check_scratch_extent(specs[inst], name, itemsize, entry[2])
            nbytes = max(itemsize * int(torch.Size(e[2]).numel()) for e in entries)
            if fill is _FILL_ZERO:
                raw = torch.zeros(nbytes, dtype=torch.uint8, device=device)
            elif fill is _FILL_MINUS_ONE:
                raw = torch.full((nbytes,), 0xFF, dtype=torch.uint8, device=device)
            else:
                raw = torch.empty(nbytes, dtype=torch.uint8, device=device)
            storage[name] = raw
            for inst, entry in zip(specs, entries, strict=True):
                count = int(torch.Size(entry[2]).numel())
                views[inst][name] = raw[: count * itemsize].view(dtype).view(entry[2])
        MR, H, TK = C128["MAX_ROWS"], C128["HIDDEN"], C128["TOP_K"]
        inputs = {
            "x_fp8": torch.zeros((MR, H), dtype=torch.uint8, device=device),
            "x_sf": torch.zeros((MR, H // 32), dtype=torch.uint8, device=device),
            "topk_idx": torch.full((MR, TK), -1, dtype=torch.int64, device=device),
            "topk_weights": torch.zeros((MR, TK), dtype=torch.float32, device=device),
        }

        # -- transport bindings ----------------------------------------------
        peer_transport = torch.tensor(table.cells, dtype=torch.uint8, device=device)
        lsa_peer_ptrs = torch.tensor(
            [_as_int64_image(p) for p in session.all_peer_pointers()],
            dtype=torch.int64,
            device=device,
        ).view(torch.uint64)
        lsa_send_state = torch.zeros(
            world_size * int(kernel["transport"]["lsa_channels"]) * 2,
            dtype=torch.uint32,
            device=device,
        )
        transport_bindings: dict[str, Any] = {
            "lsa_peer_ptrs": lsa_peer_ptrs,
            "lsa_send_state": lsa_send_state,
            "lsa_signals": session.window_pointer("lsa_signals"),
            "peer_transport": peer_transport,
            # Both transports are always compiled; without GIN the communicator and the
            # window handles bind to null and the table validator forbids GIN cells.
            "gin_dev_comm": session.dev_comm if session.gin_available else 0,
        }
        for name in _KERNEL_WINDOWS:
            pointer = session.window_pointer(name)
            transport_bindings[name] = pointer
            transport_bindings[f"{name}_gwin"] = pointer
            transport_bindings[f"{name}_gwin_window"] = session.window_handle(name)

        # -- argument template --------------------------------------------------
        plan = list(abi["args"])
        slots: dict[str, list[int]] = {}
        for entry in plan:
            if entry["kind"] not in ("buffer", "parameter"):
                raise RuntimeError(
                    f"{KERNEL_NAME}: unsupported argument kind {entry['kind']!r}"
                )
            slots.setdefault(entry["name"], []).append(int(entry["index"]))
        arg_widths = torch.zeros(len(plan), dtype=torch.int8)
        for entry in plan:
            arg_widths[int(entry["index"])] = _CTYPE_WIDTH[entry["abi"]]

        # -- kernel instances ---------------------------------------------------
        sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
        instances: dict[str, _Instance] = {}
        device_parameters = kernel["params"].device_parameters()
        for inst_name, spec in specs.items():
            cubin = device_cubin(device_parameters, int(spec["instance"]))
            function = int(
                module.module_load_function(
                    torch.frombuffer(bytearray(cubin), dtype=torch.uint8),
                    str(spec["symbol"]),
                )
            )
            if int(module.function_local_bytes(function)) > LOCAL_BYTES_LIMIT:
                raise RuntimeError(
                    f"{KERNEL_NAME}: {inst_name}{suffix} uses local memory"
                )
            smem = int(spec["smem_bytes"])
            blocks = int(module.function_max_active_blocks(function, threads, smem))
            if blocks <= 0:
                raise RuntimeError(
                    f"{KERNEL_NAME}: {inst_name}{suffix} has zero resident CTA capacity"
                )
            cap = torch.tensor([blocks * sm_count], dtype=torch.int64, device=device)
            dist.all_reduce(cap, op=dist.ReduceOp.MIN, group=group)
            C = spec["constants"]
            b = views[inst_name]
            static_sources = {
                "W1_A": b["pool_fp8_u32"]
                .view(torch.uint8)
                .reshape(C["POOL_ROWS"], C["HIDDEN"]),
                "W1_SFA": b["pool_sf_u32"].view(torch.uint32),
                "W1_D": b["W1_D"],
                "W2_A": b["intermediate_fp8"],
                "W2_SFA": b["intermediate_sfa_u8"]
                .view(torch.uint32)
                .reshape(C["REQUANT_PACKED_GROUPS"], C["POOL_ROWS"]),
                "W2_D": b["W2_D"],
            }
            carriers = torch.zeros(
                (len(_STATIC_DESCRIPTORS), _TENSOR_MAP_BYTES),
                dtype=torch.uint8,
                device=device,
            )
            static_bindings: dict[str, Any] = {**transport_bindings, **b}
            for row, name in enumerate(_STATIC_DESCRIPTORS):
                self._encode_descriptor(
                    module,
                    static_sources[name],
                    spec["descriptors"][name],
                    carriers[row],
                    name,
                )
                static_bindings[name] = carriers[row].data_ptr()
            static_bindings["rank"] = rank
            static_bindings["world_size"] = world_size
            missing = (
                set(abi["binding_names"]) - set(static_bindings) - set(_PER_CALL_NAMES)
            )
            if missing:
                raise RuntimeError(
                    f"{KERNEL_NAME}: unbound kernel arguments {sorted(missing)}"
                )
            arg_values = torch.zeros(len(plan), dtype=torch.int64)
            for entry in plan:
                if entry["name"] in _PER_CALL_NAMES:
                    continue
                self._write_arg(arg_values, entry, static_bindings[entry["name"]])
            instances[inst_name] = _Instance(
                name=inst_name + suffix,
                function=function,
                smem_bytes=smem,
                task_m=int(spec["task_m"]),
                constants=C,
                flags=spec["flags"],
                capacity=int(cap.item()),
                arg_values=arg_values,
                buffers=b,
                static_carriers=carriers,
            )

        E = int(kernel["world_size"]) * int(kernel["local_experts"])
        return _CakeWorkspace(
            module=module,
            session=session,
            kernel=kernel,
            transport_table=table,
            transport_probe_report=report,
            copiers_active=copiers,
            instances=instances,
            plan=plan,
            arg_widths=arg_widths,
            slots=slots,
            storage=storage,
            inputs=inputs,
            peer_transport=peer_transport,
            lsa_peer_ptrs=lsa_peer_ptrs,
            lsa_send_state=lsa_send_state,
            device=device,
            group=group,
            rank=rank,
            world_size=world_size,
            threads=threads,
            variant=variant,
            max_rows=MR,
            hidden=H,
            top_k=TK,
            num_experts=E,
            local_experts=int(kernel["local_experts"]),
            m64_max_rows=int(kernel["transport"]["m64_max_rows"]),
            stats=torch.zeros(1 + E, dtype=torch.int32, device=device),
            gathered_stats=torch.zeros(
                (world_size, 1 + E), dtype=torch.int32, device=device
            ),
        )

    @staticmethod
    def _nccl_comm_pointer(group: Any) -> int:
        import torch.distributed as dist

        # Make sure the communicator exists (ProcessGroupNCCL creates it lazily).
        dist.barrier(group=group)
        backend = group._get_backend(torch.device("cuda"))
        if not hasattr(backend, "_comm_ptr"):
            raise RuntimeError(
                f"{KERNEL_NAME}: the EP process group is not a ProcessGroupNCCL exposing _comm_ptr"
            )
        version = tuple(int(v) for v in backend.get_runtime_nccl_version())
        if version != (2, 30, 7):
            raise RuntimeError(
                f"{KERNEL_NAME} requires NCCL 2.30.7, torch loaded {version}"
            )
        pointer = int(backend._comm_ptr())
        if pointer <= 0:
            raise RuntimeError(
                f"{KERNEL_NAME}: ProcessGroupNCCL returned a null communicator pointer"
            )
        return pointer

    @staticmethod
    def _encode_descriptor(
        module: Any,
        source: torch.Tensor,
        spec: dict[str, Any],
        carrier: torch.Tensor,
        name: str,
    ) -> None:
        """Encode one canonical 2-D tiled (or native FP4) descriptor into `carrier`."""

        if source.stride(-1) != 1 or source.dim() < 2:
            raise ValueError(
                f"{name}: TMA source must be rank-2+ with unit inner stride"
            )
        if source.dim() > 2:  # leading dims must fold densely onto the row axis
            expected = source.stride(-2) * source.shape[-2]
            for axis in range(source.dim() - 3, -1, -1):
                if source.shape[axis] > 1 and source.stride(axis) != expected:
                    raise ValueError(f"{name}: TMA source leading dims are not dense")
                expected *= source.shape[axis]
        box0, box1 = (int(v) for v in spec["box"])
        swizzle_key = tuple(int(v) for v in spec["swizzle"])
        if len(swizzle_key) != 3:
            raise ValueError(f"{name}: TMA swizzle spec must have three entries")
        swizzle = _TMA_SWIZZLE_CODE[(swizzle_key[0], swizzle_key[1], swizzle_key[2])]
        l2 = _TMA_L2_CODE[str(spec["l2_promotion"])]
        cols = int(source.shape[-1])
        rows = int(source.numel() // cols)
        if spec.get("native_fp4"):
            if source.dtype != torch.uint8:
                raise TypeError(
                    f"{name}: native FP4 TMA source must be uint8 packed E2M1"
                )
            dtype_code, cols_elems, pitch = 9, 2 * cols, int(source.stride(-2))
        else:
            dtype = str(spec["dtype"])
            if source.element_size() != _TMA_ELEM_BYTES[dtype]:
                raise TypeError(
                    f"{name}: element size {source.element_size()} != descriptor dtype {dtype}"
                )
            dtype_code, cols_elems, pitch = (
                _TMA_DTYPE_CODE[dtype],
                cols,
                int(source.stride(-2)) * source.element_size(),
            )
        module.tma_encode_2d(
            source,
            dtype_code,
            cols_elems,
            rows,
            pitch,
            box0,
            box1,
            swizzle,
            l2,
            carrier,
            carrier.device.index,
            torch.cuda.current_stream(carrier.device).cuda_stream,
        )

    @staticmethod
    def _write_arg(arg_values: torch.Tensor, entry: dict[str, Any], value: Any) -> None:
        abi, index = entry["abi"], int(entry["index"])
        if abi == "ptr":
            if isinstance(value, torch.Tensor):
                if not value.is_cuda:
                    raise ValueError(
                        f"{entry['name']}: pointer binding must be a CUDA tensor"
                    )
                ptr = value.data_ptr()
            elif isinstance(value, ctypes.c_void_p):
                ptr = int(value.value or 0)
            elif isinstance(value, int) and not isinstance(value, bool):
                ptr = value
            else:
                raise TypeError(
                    f"{entry['name']}: pointer binding expects a CUDA tensor or int, got {type(value).__name__}"
                )
            arg_values[index] = _as_int64_image(ptr)
        else:
            arg_values[index] = _as_int64_image(_scalar_bits(abi, value))

    # ------------------------------------------------------------- per-layer

    def _layer_descriptors(
        self, ws: _CakeWorkspace, tw: TransformedCakeMegaWeights
    ) -> _LayerDescriptors:
        cached = ws.layers.get(id(tw))
        if cached is not None and cached.owner is tw:
            return cached
        L = ws.local_experts
        if tw.local_experts != L:
            raise ValueError(
                f"transformed weights carry {tw.local_experts} local experts, kernel family has {L}"
            )
        C = ws.instances["m128"].constants
        sources = {
            "W1_B": tw.w1_fp4.view(torch.uint8).reshape(
                L * C["W1_N"], C["HIDDEN"] // 2
            ),
            "W1_B8": tw.w1_b8,
            "W1_SFB": tw.w1_sfb,
            "W2_B": tw.w2_fp4.view(torch.uint8).reshape(
                L * C["W2_N"], C["INTERMEDIATE"] // 2
            ),
            "W2_B8": tw.w2_b8,
            "W2_SFB": tw.w2_sfb,
        }
        descriptors = ws.kernel["instances"][ws.instances["m128"].name]["descriptors"]
        carriers = torch.zeros(
            (len(sources), _TENSOR_MAP_BYTES), dtype=torch.uint8, device=ws.device
        )
        slots: dict[str, int] = {}
        for row, (name, source) in enumerate(sources.items()):
            self._encode_descriptor(
                ws.module, source, descriptors[name], carriers[row], name
            )
            slots[name] = row
        layer = _LayerDescriptors(carriers=carriers, slots=slots, owner=tw)
        ws.layers[id(tw)] = layer
        return layer

    # --------------------------------------------------------------- forward

    def validate_forward(
        self, t: MoEEpTensors, fleet_params: FleetParams, *, quantize_input: bool
    ) -> None:
        validate_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            quantize_input=quantize_input,
            scales=t.scales,
        )

    def validate_capture_ready(
        self, workspace: Any, transformed_weights: TransformedCakeMegaWeights
    ) -> None:
        ws = self._live(workspace)
        cached = ws.layers.get(id(transformed_weights))
        if cached is None or cached.owner is not transformed_weights:
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: layer is not warmed for CUDA graph capture on this workspace; "
                "run it eagerly first (layer.warmup(...) or one forward)"
            )

    def set_compile_tokens_per_rank(
        self, workspace: Any, compile_tokens_per_rank: int | None
    ) -> None:
        """Rank-uniform bound on the per-rank row counts of the next invocation.

        ``MoEEpMegaLayer`` sets it on every call (``None`` on ``forward``).  With a bound
        the call skips the routing all-gather and can be captured into a CUDA graph.
        """

        ws = self._live(workspace)
        if compile_tokens_per_rank is not None and not (
            0 <= int(compile_tokens_per_rank) <= ws.max_rows
        ):
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: compile_tokens_per_rank={compile_tokens_per_rank} must be in "
                f"[0, {ws.max_rows}] (the workspace row capacity)"
            )
        ws.compile_tokens_per_rank = (
            None if compile_tokens_per_rank is None else int(compile_tokens_per_rank)
        )

    def stage_inputs(
        self, t: MoEEpTensors, workspace: Any, *, quantize_input: bool
    ) -> None:
        ws = self._live(workspace)
        bound = ws.compile_tokens_per_rank
        if torch.cuda.is_current_stream_capturing() and bound is None:
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: CUDA graph capture needs compile_tokens_per_rank "
                "(MoEEpMegaLayer.stage_inputs; the routing all-gather reads device data "
                "back on the host)"
            )
        if bound is not None and t.hidden_states.shape[0] > bound:
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: compile_tokens_per_rank={bound} is below this rank's "
                f"{t.hidden_states.shape[0]} tokens"
            )
        ws.staged_rows = stage_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            x_fp8=ws.inputs["x_fp8"],
            x_sf=ws.inputs["x_sf"],
            topk_idx_out=ws.inputs["topk_idx"],
            topk_weights_out=ws.inputs["topk_weights"],
            quantize_input=quantize_input,
            scales=t.scales,
        )
        ws.staged_bound = bound

    def workspace_output(self, workspace: Any) -> torch.Tensor:
        return self._live(workspace).output

    def _gather_routing(
        self, ws: _CakeWorkspace, rows: int
    ) -> tuple[list[int], list[int]]:
        """Per-rank row counts and the global expert histogram (one all-gather + sync)."""

        import torch.distributed as dist

        E = ws.num_experts
        ids = ws.inputs["topk_idx"][:rows].reshape(-1)
        valid = ids[(ids >= 0) & (ids < E)]
        ws.stats.zero_()
        ws.stats[1:] = torch.bincount(valid, minlength=E)[:E].to(torch.int32)
        ws.stats[0] = rows
        dist.all_gather_into_tensor(ws.gathered_stats, ws.stats, group=ws.group)
        gathered = ws.gathered_stats.cpu()
        rows_by_rank = [int(v) for v in gathered[:, 0].tolist()]
        expert_counts = [int(v) for v in gathered[:, 1:].sum(dim=0).tolist()]
        return rows_by_rank, expert_counts

    def compute(
        self,
        workspace: Any,
        transformed_weights: TransformedCakeMegaWeights,
        *,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        ws = self._live(workspace)
        rows = ws.staged_rows
        if torch.cuda.is_current_stream_capturing() and ws.staged_bound is None:
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: CUDA graph capture needs compile_tokens_per_rank "
                "(MoEEpMegaLayer.stage_inputs; the routing all-gather reads device data "
                "back on the host)"
            )
        stream = torch.cuda.current_stream(ws.device)
        layer = self._layer_descriptors(ws, transformed_weights)
        if ws.staged_bound is None:
            rows_by_rank, expert_counts = self._gather_routing(ws, rows)
        else:
            # Size the grid for the bound on every rank; any grid from the static one
            # up to the cooperative capacity is valid (see :func:`rows_residency_grid`).
            rows_by_rank = [ws.staged_bound] * ws.world_size
            expert_counts = None
        collective_rows = max(rows_by_rank)
        out = ws.output[:rows]
        if collective_rows == 0:
            # No rank holds a token: every rank skips the launch consistently.
            ws.last_launch = {"skipped": True, "rows_by_rank": rows_by_rank}
            if output is not None:
                return output[:0]
            return out
        inst = ws.instances["m64" if collective_rows <= ws.m64_max_rows else "m128"]
        grid_args: dict[str, Any] = dict(
            task_m=inst.task_m,
            with_shared_expert=ws.variant == "fp8shared",
            capacity=inst.capacity,
            C=inst.constants,
            resident_peer_fold=bool(inst.flags["_RESIDENT_PEER_FOLD"]),
        )
        if expert_counts is None:
            grid = rows_residency_grid(
                rows_by_rank=rows_by_rank,
                top_k=ws.top_k,
                num_experts=ws.num_experts,
                **grid_args,
            )
        else:
            grid = residency_grid(
                rows_by_rank=rows_by_rank,
                expert_counts=expert_counts,
                local_experts=ws.local_experts,
                **grid_args,
            )
        x = ws.inputs
        per_call: dict[str, Any] = {
            **{
                name: layer.carriers[layer.slots[name]].data_ptr()
                for name in layer.slots
            },
            "topk_idx_i32": x["topk_idx"],
            "topk_weights": x["topk_weights"],
            "x_fp8_i32": x["x_fp8"],
            "x_sf_i32": x["x_sf"],
            # The high 16 bits carry the largest per-rank row count of this call.
            "active_rows": rows | (collective_rows << 16),
        }
        for entry in ws.plan:
            name = entry["name"]
            if name in per_call:
                self._write_arg(inst.arg_values, entry, per_call[name])
        ws.module.launch_cooperative(
            inst.function,
            grid,
            ws.threads,
            inst.smem_bytes,
            inst.arg_values,
            ws.arg_widths,
            ws.device.index,
            stream.cuda_stream,
        )
        ws.last_launch = {
            "instance": inst.name,
            "grid": grid,
            "epoch": ws.epoch,
            "rows_by_rank": rows_by_rank,
            "collective_rows": collective_rows,
            "grid_rule": "routing" if expert_counts is not None else "bound",
        }
        ws.epoch += 1
        if output is not None:
            output[:rows].copy_(out)
            return output[:rows]
        return out

    def device_epoch(self, workspace: Any) -> int:
        """Synchronous read of the kernel's device-resident epoch (= completed launches)."""

        return (
            int(
                self._live(workspace).storage["epoch_state"].view(torch.int32)[0].item()
            )
            & 0xFFFFFFFF
        )

    def protocol_error(self, workspace: Any) -> int:
        """Synchronous read of the kernel's fail-closed protocol flag (0 = clean)."""

        return int(
            self._live(workspace).storage["protocol_error"].view(torch.int32)[0].item()
        )

    @staticmethod
    def _live(workspace: Any) -> _CakeWorkspace:
        if not isinstance(workspace, _CakeWorkspace):
            raise TypeError(
                f"{KERNEL_NAME} workspace must be created by this backend, got {type(workspace).__name__}"
            )
        if workspace.destroyed:
            raise RuntimeError(f"{KERNEL_NAME} workspace has been destroyed")
        return workspace


__all__ = [
    "KERNEL_NAME",
    "LOCAL_BYTES_LIMIT",
    "REGISTER_BUDGET_VERIFIED_MAX_RANKS",
    "Sm120Mxfp4Mxfp8CakeMegaKernelBackend",
    "candidate_instances",
    "check_device_resources",
    "even_expert_counts",
    "residency_grid",
    "rows_residency_grid",
    "scratch_specs",
    "static_residency_grid",
]
