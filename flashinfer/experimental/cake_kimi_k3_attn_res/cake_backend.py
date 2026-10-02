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
"""

"""Host side of the experimental Cake Kimi-K3 AttnRes backend (SM100 / SM103).

The operator (Kimi-K3 attention-residual mixing, ``_apply_attn_res`` of the
model) on caller-owned BF16 tensors with ``H = 7168`` and ``K = num_blocks``
in ``[0, 8]``::

    prefix[M, H]       += delta[M, H]            (BF16 rounded, when delta is given)
    blocks[:, w, :]     = prefix                 (exact snapshot, when block_write_idx = w >= 0)
    c_i                 = blocks[:, i, :] (i < K), c_K = prefix
    s_i                 = sum_h rmsnorm(c_i, eps)[h] * norm_weight[h] * qk_weight[h]   (FP32)
    p                   = softmax(s)                                                  (FP32)
    mixed               = sum_i p_i * c_i                                             (FP32)
    out                 = BF16(rmsnorm(mixed, output_norm_eps) * output_norm_weight)  (or BF16(mixed))

``plan_route`` selects the generated program from host-known scalars and tensor
metadata only (architecture, SM count, ``M``, ``K``, the PDL launch policy and
the semantic flags), exactly as the Cake production dispatcher does; the export
verifies key, grid and schedule parity for every row of its denominator.  Dense
token-major inputs with the residual add and the fused output norm take the
persistent TMEM path (or one of the installed native ports / the exact K = 0
path); the other semantic variants (no delta, snapshot write, no output norm,
row-padded layouts) take the one-CTA-per-token bootstrap program.

``prepare_kimi_k3_attn_res`` binds one call; its ``launch()`` performs no CUDA
allocation and no host synchronisation and is CUDA-graph capturable (capture
belongs to the caller).  ``prefix`` (and ``blocks`` when a snapshot is written)
are mutated in place on every launch, as in the model.
"""

import functools
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Optional

import torch
import tvm_ffi

from .cake_jit import (
    MODULES,
    kernel_module_name,
    load_cake_kimi_k3_attn_res_module,
    route_available,
)

HIDDEN_SIZE = 7168
MAX_BLOCKS = 8
DIRECT_THREADS = 256
PERSISTENT_THREADS = 288
NATIVE_GRID = 148
PERSISTENT_BALANCED_GRID = 128
PERSISTENT_BALANCED_GRID_MAX_M = 512
PERSISTENT_CHUNK_DEPTH = 2
PERSISTENT_SOURCES_PER_CHUNK = 4
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
ARCHES = tuple(sorted(set(SUPPORTED_COMPUTE_CAPABILITIES.values())))

# ---------------------------------------------------------------------------
# Production route policy (Cake ``kimi_k3_attn_res.select_route``), ported.
# Every table below is the measured per-architecture selection of the Cake
# dispatcher; the export checks this port against it row by row.
# ---------------------------------------------------------------------------
_SM103_EARLY_CONSUMED_RELEASE_CELLS = frozenset(
    {
        (256, 8),
        (512, 4),
        (512, 8),
        (1024, 1),
        (1024, 4),
        (1024, 8),
        (2048, 1),
        (2048, 4),
        (2048, 8),
        (4096, 1),
        (4096, 2),
        (4096, 4),
        (4096, 5),
        (4096, 6),
        (4096, 7),
        (4096, 8),
        (8192, 1),
        (8192, 8),
        (16384, 1),
        (16384, 8),
    }
)
_SM100_K1_NC2_M = frozenset({512, 1024, 2048, 8192})
_SM103_K1_NC2_M = frozenset({256})
_SM103_K5_NC4_DEPTH_M = {4096: 3}
# Cells (M, K) that run three sources per chunk with a depth-3 pipeline.
_NC3_D3_CELLS = {
    "sm_100a": frozenset(
        {(2048, 4), (4096, 4), (8192, 4), (16384, 4), (4096, 3), (4096, 5)}
    ),
    "sm_103a": frozenset({(1024, 4), (2048, 4), (4096, 4)}),
}
# sm_100a dense cells that hold the consumed-stage release through the output stats.
_SM100_HELD_CONSUMED_RELEASE_CELLS = frozenset({(4096, 5)})
# K = 0 TMA route: grid multiple of the SM count on the promoted mid-M cells.
_K0_TMA_GRID_MULTIPLIER_M = {
    "sm_100a": {256: 2, 512: 2, 1024: 2},
    "sm_103a": {256: 2, 512: 3, 1024: 3},
}
_SM100_RELAXED_PRODUCER_WAIT_CELLS: frozenset[tuple[int, int]] = frozenset()
_SM103_RELAXED_PRODUCER_WAIT_CELLS = frozenset({(4096, 5)})
_NATIVE_ROUTES = (
    # name, {arch: routed M}, num_blocks, PDL modes, consumed-release policy
    (
        "k5",
        {"sm_100a": frozenset({1}), "sm_103a": frozenset({1})},
        5,
        (False, True),
        "native_lane0_before_wait_st",
    ),
    (
        "m128",
        {
            "sm_100a": frozenset({1, 2, 4, 8, 16, 32, 64, 128, 1024}),
            "sm_103a": frozenset({1, 2, 4, 8, 16, 32, 64, 128, 256}),
        },
        4,
        (False, True),
        "native_all_reader_before_wait_st",
    ),
    (
        "k6",
        {"sm_100a": frozenset({1}), "sm_103a": frozenset({1})},
        6,
        (False, True),
        "native_all_reader_before_wait_st",
    ),
    (
        "k7",
        {"sm_100a": frozenset({1}), "sm_103a": frozenset({1})},
        7,
        (False,),
        "native_all_reader_nofence_before_wait_st",
    ),
    (
        "k8",
        {
            "sm_100a": frozenset({1, 2, 4, 8, 32, 256}),
            "sm_103a": frozenset({1, 8, 256}),
        },
        8,
        (False, True),
        "native_all_reader_nofence_before_wait_st",
    ),
)
_NATIVE_K8_SM100_PDL_ONLY_M = frozenset({16})


class RoutePlan(NamedTuple):
    """The generated program one call launches and its launch geometry."""

    kind: str  # "direct" | "native" | "k0_tma" | "persistent"
    kernel_key: str
    grid_x: int
    threads: int
    schedule_id: str
    route_id: str
    arch: Optional[str]
    use_pdl: bool


@functools.lru_cache(maxsize=None)
def _device_capability(index: int) -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability(index))


@functools.lru_cache(maxsize=None)
def _device_sm_count(index: int) -> int:
    return int(torch.cuda.get_device_properties(index).multi_processor_count)


def _device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else int(device.index)


def arch_for(device: torch.device) -> str:
    capability = _device_capability(_device_index(device))
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise ValueError(
            "the Kimi-K3 AttnRes kernels require compute capability 10.0 or 10.3 "
            f"(got {capability[0]}.{capability[1]})"
        )
    return arch


def sm_count_for(device: torch.device) -> int:
    return _device_sm_count(_device_index(device))


def _wait_policy(arch: str) -> bool:
    return arch == "sm_100a"


def _schedule(arch: str, M: int, K: int) -> tuple[int, int]:
    """``(sources_per_chunk, chunk_depth)`` of the persistent common path."""
    if (M, K) in _NC3_D3_CELLS[arch]:
        return 3, 3
    if arch == "sm_100a":
        if K == 0 and M in {1, 2, 16, 32, 64, 512, 2048, 4096, 8192, 16384}:
            return 1, PERSISTENT_CHUNK_DEPTH
        if K == 1 and M in _SM100_K1_NC2_M:
            return 2, PERSISTENT_CHUNK_DEPTH
        if K == 4:
            return (
                5 if M in {1, 32, 1024, 2048, 4096, 8192, 16384} else 3
            ), PERSISTENT_CHUNK_DEPTH
    else:
        if K == 0 and M not in {4, 16, 256}:
            return 1, PERSISTENT_CHUNK_DEPTH
        if K == 1 and M in _SM103_K1_NC2_M:
            return 2, PERSISTENT_CHUNK_DEPTH
        if K == 5 and M in _SM103_K5_NC4_DEPTH_M:
            return PERSISTENT_SOURCES_PER_CHUNK, _SM103_K5_NC4_DEPTH_M[M]
        if K == 4:
            return (3 if M <= 512 else 5), PERSISTENT_CHUNK_DEPTH
    if K == 8:
        return (3 if M <= 256 else 5), PERSISTENT_CHUNK_DEPTH
    return PERSISTENT_SOURCES_PER_CHUNK, PERSISTENT_CHUNK_DEPTH


def _grid(M: int, num_sms: int, sources_per_chunk: int) -> tuple[int, str]:
    if (
        sources_per_chunk in {2, 3, PERSISTENT_SOURCES_PER_CHUNK}
        and 256 <= M <= PERSISTENT_BALANCED_GRID_MAX_M
        and num_sms >= PERSISTENT_BALANCED_GRID
    ):
        return PERSISTENT_BALANCED_GRID, "balanced128_m256_m512"
    return min(M, num_sms), "token_or_full_sm"


def _early_consumed_release(arch: str, M: int, K: int, grid_x: int) -> bool:
    if arch == "sm_100a":
        return K > 0 and grid_x < M and (M, K) not in _SM100_HELD_CONSUMED_RELEASE_CELLS
    return K > 0 and grid_x < M and (M, K) in _SM103_EARLY_CONSUMED_RELEASE_CELLS


def _producer_wait_acquire(arch: str, M: int, K: int) -> bool:
    return not (
        (arch == "sm_100a" and (M, K) in _SM100_RELAXED_PRODUCER_WAIT_CELLS)
        or (arch == "sm_103a" and (M, K) in _SM103_RELAXED_PRODUCER_WAIT_CELLS)
    )


def _native_route(arch: str, num_sms: int, M: int, K: int, use_pdl: bool):
    if num_sms != NATIVE_GRID:
        return None
    for name, routed, k, pdl_modes, release in _NATIVE_ROUTES:
        if k != K or use_pdl not in pdl_modes:
            continue
        pdl_only = (
            name == "k8"
            and arch == "sm_100a"
            and use_pdl
            and M in _NATIVE_K8_SM100_PDL_ONLY_M
        )
        if M in routed[arch] or pdl_only:
            return name, release
    return None


def plan_route(
    arch: Optional[str],
    num_sms: int,
    M: int,
    num_blocks: int,
    use_pdl: bool,
    *,
    has_delta: bool = True,
    block_write_idx: int = -1,
    apply_output_norm: bool = True,
    common_path: bool = True,
) -> RoutePlan:
    """Select the generated program for one call from host-known facts only."""
    M = int(M)
    K = int(num_blocks)
    use_pdl = bool(use_pdl)
    if M <= 0:
        raise ValueError("AttnRes requires M > 0")
    if not 0 <= K <= MAX_BLOCKS:
        raise ValueError(f"num_blocks must be in [0, {MAX_BLOCKS}]")
    if not common_path:
        write_block = int(block_write_idx) >= 0
        schedule_id = "direct_cta256_bootstrap"
        route_id = (
            f"{schedule_id}.k{K}.d{int(bool(has_delta))}.w{int(write_block)}."
            f"n{int(bool(apply_output_norm))}.pdl{int(use_pdl)}.m_all"
        )
        key = f"direct:k{K}_d{int(bool(has_delta))}_w{int(write_block)}_n{int(bool(apply_output_norm))}"
        return RoutePlan(
            "direct", key, M, DIRECT_THREADS, schedule_id, route_id, arch, use_pdl
        )
    if arch not in SUPPORTED_COMPUTE_CAPABILITIES.values():
        raise ValueError(f"unsupported Kimi-K3 AttnRes architecture {arch!r}")
    num_sms = int(num_sms)
    if num_sms <= 0:
        raise ValueError(f"invalid persistent launch geometry M={M}, num_sms={num_sms}")
    native = _native_route(arch, num_sms, M, K, use_pdl)
    if native is not None:
        name, _release = native
        schedule_id = f"native_{name}_nc3_d2_ws288_grid148"
        route_id = f"{schedule_id}.{arch}.native_no_hint.k{K}.delta1.write0.norm1.pdl{int(use_pdl)}.native_full_sm"
        return RoutePlan(
            "native",
            f"native_{name}",
            NATIVE_GRID,
            PERSISTENT_THREADS,
            schedule_id,
            route_id,
            arch,
            use_pdl,
        )
    if K == 0:
        multiplier = _K0_TMA_GRID_MULTIPLIER_M[arch].get(M)
        if multiplier is None:
            grid_x, grid_policy = min(M, num_sms), "token_or_full_sm"
        else:
            grid_x, grid_policy = multiplier * num_sms, f"full_sm_x{multiplier}"
        schedule_id = "k0_tma_persistent_ws288_vec128_fp32x2"
        route_id = f"{schedule_id}.{arch}.none_k0.k0.delta1.write0.norm1.pdl{int(use_pdl)}.{grid_policy}"
        return RoutePlan(
            "k0_tma",
            "k0_tma",
            grid_x,
            PERSISTENT_THREADS,
            schedule_id,
            route_id,
            arch,
            use_pdl,
        )
    defer = _wait_policy(arch)
    nc, depth = _schedule(arch, M, K)
    grid_x, grid_policy = _grid(M, num_sms, nc)
    ecr = _early_consumed_release(arch, M, K, grid_x)
    pwa = _producer_wait_acquire(arch, M, K)
    prefix_bf16_add = (
        (
            arch == "sm_100a"
            and (
                K == 2
                or (K in {1, 3, 5, 6, 7} and nc == 4)
                or (K == 4 and nc == 5)
                or (K == 8 and nc == 3)
            )
            and not ecr
        )
        or (
            arch == "sm_103a"
            and (K in {1, 2, 3} or (M == 1 and K == 5))
            and nc == 4
            and not ecr
        )
        or (K == 4 and nc == 3 and not ecr)
        or (arch == "sm_103a" and K == 8 and nc == 3 and not ecr)
    )
    prefix_round_once = K == 8 and nc == 3 and not ecr
    one_token_per_cta = K in {1, 2, 3} and nc == 4 and not ecr and grid_x == M
    prefetch_fifth_stats = (
        (arch == "sm_103a" or (arch == "sm_100a" and M == 32))
        and K == 4
        and nc == 5
        and not ecr
    )
    k4_early_nc5_input_first = (
        arch == "sm_100a" and M == 1024 and K == 4 and nc == 5 and ecr
    )
    k4_held_nc3_input_first = arch == "sm_103a" and K == 4 and M in {64, 128}
    k4_nc5_source0_carry = (
        arch == "sm_103a" and M == 1024 and K == 4 and nc == 5 and not ecr
    )
    k1_on_output_norm_cache = (
        arch == "sm_103a" and use_pdl and K == 1 and M in {1024, 2048, 4096, 8192}
    )
    k1_m1_on_source_barrier_norm_prefetch = (
        arch == "sm_103a" and use_pdl and K == 1 and M == 1
    )
    k4_m4_on_packed_cross_warp = (
        arch == "sm_100a" and K == 4 and not use_pdl and M == 64
    )
    k5_m1_on_terminal_packed_stats = arch == "sm_103a" and use_pdl and K == 5 and M == 1
    k1_output_norm_packed_cache = (
        arch == "sm_103a"
        and not use_pdl
        and K == 1
        and nc == 4
        and not one_token_per_cta
        and not prefix_bf16_add
        and not prefix_round_once
        and ecr
        and pwa
        and not defer
    )
    k8_sm103_nonallocator_qk_prelude = (
        arch == "sm_103a"
        and not use_pdl
        and K == 8
        and nc == 3
        and not ecr
        and prefix_bf16_add
        and prefix_round_once
        and not one_token_per_cta
    )
    flags = (
        ecr,
        pwa,
        prefix_bf16_add,
        prefix_round_once,
        one_token_per_cta,
        prefetch_fifth_stats,
        k4_early_nc5_input_first,
        k4_held_nc3_input_first,
        k4_nc5_source0_carry,
        k1_on_output_norm_cache,
        k1_m1_on_source_barrier_norm_prefetch,
        k4_m4_on_packed_cross_warp,
        k5_m1_on_terminal_packed_stats,
        k1_output_norm_packed_cache,
        k8_sm103_nonallocator_qk_prelude,
    )
    bits = "".join("1" if flag else "0" for flag in flags)
    key = f"persistent:k{K}_nc{nc}_d{depth}_f{bits}"
    schedule_id = f"trtllm_persistent_ws288_nc{nc}_d{depth}_vec128_fp32x2"
    if ecr:
        schedule_id += "_early_consume"
    if not pwa:
        schedule_id += "_relaxed_producer_wait"
    if prefix_bf16_add:
        schedule_id += "_prefix_bf16_add_packed_inputs_prefix_shared_addr"
    if prefix_round_once:
        schedule_id += "_prefix_round_once"
    if one_token_per_cta:
        schedule_id += "_one_token_per_cta"
    wait_policy = "deferred_wait_st" if defer else "control_wait_st"
    route_id = f"{schedule_id}.{arch}.{wait_policy}.k{K}.delta1.write0.norm1.pdl{int(use_pdl)}.{grid_policy}"
    return RoutePlan(
        "persistent",
        key,
        grid_x,
        PERSISTENT_THREADS,
        schedule_id,
        route_id,
        arch,
        use_pdl,
    )


# ---------------------------------------------------------------------------
# Input validation and common-path eligibility (Cake ``_can_use_persistent_common_path``)
# ---------------------------------------------------------------------------


def _dense_vector(tensor: torch.Tensor, row_strides=()) -> bool:
    if tensor.stride(-1) != 1 or tensor.data_ptr() % 16 != 0:
        return False
    elem = tensor.element_size()
    return all(tensor.stride(dim) * elem % 16 == 0 for dim in row_strides)


def validate_kimi_k3_attn_res_inputs(
    prefix: torch.Tensor,
    delta: Optional[torch.Tensor],
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
) -> None:
    tensors = {
        "prefix": prefix,
        "blocks": blocks,
        "norm_weight": norm_weight,
        "qk_weight": qk_weight,
        "out": out,
    }
    if delta is not None:
        tensors["delta"] = delta
    if output_norm_weight is not None:
        tensors["output_norm_weight"] = output_norm_weight
    device = prefix.device
    for name, tensor in tensors.items():
        if not tensor.is_cuda or tensor.device != device:
            raise ValueError(f"{name} must be a CUDA tensor on {device}")
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be bfloat16")
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must have a dense last dimension")
    M = int(prefix.shape[0])
    if prefix.dim() != 2 or prefix.shape[1] != HIDDEN_SIZE or M <= 0:
        raise ValueError(f"prefix must be [M > 0, {HIDDEN_SIZE}]")
    if delta is not None and tuple(delta.shape) != (M, HIDDEN_SIZE):
        raise ValueError(f"delta must be [{M}, {HIDDEN_SIZE}]")
    if tuple(out.shape) != (M, HIDDEN_SIZE) or not out.is_contiguous():
        raise ValueError(f"out must be a contiguous [{M}, {HIDDEN_SIZE}] tensor")
    if blocks.dim() != 3 or blocks.shape[0] != M or blocks.shape[2] != HIDDEN_SIZE:
        raise ValueError(f"blocks must be [{M}, <= {MAX_BLOCKS}, {HIDDEN_SIZE}]")
    if not 0 <= int(num_blocks) <= MAX_BLOCKS or int(num_blocks) > blocks.shape[1]:
        raise ValueError(
            f"num_blocks must be in [0, min({MAX_BLOCKS}, blocks.shape[1])]"
        )
    if int(block_write_idx) != -1 and not 0 <= int(block_write_idx) < blocks.shape[1]:
        raise ValueError("block_write_idx must be -1 or a valid snapshot index")
    if blocks.stride(1) < HIDDEN_SIZE or blocks.stride(0) < blocks.shape[
        1
    ] * blocks.stride(1):
        raise ValueError("blocks rows must not overlap")
    for name in ("norm_weight", "qk_weight", "output_norm_weight"):
        tensor = tensors.get(name)
        if tensor is not None and tuple(tensor.shape) != (HIDDEN_SIZE,):
            raise ValueError(f"{name} must be [{HIDDEN_SIZE}]")
    if float(eps) <= 0.0 or (
        output_norm_weight is not None and float(output_norm_eps) <= 0.0
    ):
        raise ValueError("eps values in use must be positive")


def common_path_eligible(
    prefix: torch.Tensor,
    delta: Optional[torch.Tensor],
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    block_write_idx: int,
) -> bool:
    """True when the call takes the persistent common path: residual add + fused output norm,
    no snapshot write, token-major dense layouts with 16-byte aligned rows and a full
    ``[M, 8, H]`` snapshot bank."""
    if delta is None or output_norm_weight is None or int(block_write_idx) >= 0:
        return False
    if (
        prefix.stride(0) != HIDDEN_SIZE
        or delta.stride(0) != HIDDEN_SIZE
        or out.stride(0) != HIDDEN_SIZE
    ):
        return False
    if blocks.shape[1] < MAX_BLOCKS:
        return False
    return all(
        (
            _dense_vector(prefix, (0,)),
            _dense_vector(delta, (0,)),
            _dense_vector(out, (0,)),
            _dense_vector(blocks, (0, 1)),
            _dense_vector(norm_weight),
            _dense_vector(qk_weight),
            _dense_vector(output_norm_weight),
        )
    )


# ---------------------------------------------------------------------------
# Launch binding
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
                f"generated module {module_name!r} expects argument {name!r} ({kind}); "
                f"host binding provides {sorted(kwargs)}"
            )
    module = load_cake_kimi_k3_attn_res_module(module_name)
    return getattr(module, record["ffi_entry"]), tuple(arguments)


@dataclass(frozen=True)
class KimiK3AttnResRunner:
    """One prepared AttnRes call.  ``launch()`` runs the single generated program on the
    current torch stream, mutates ``prefix`` (and the written snapshot) in place, writes
    ``out`` and returns it; no CUDA allocation, no host synchronisation, CUDA-graph
    capturable (capture belongs to the caller).  The program reads its inputs on device at
    every launch, so the same runner (or a graph capturing it) stays valid when the caller
    writes new values into the bound tensors.  Prepare a new runner when a shape, stride,
    flag, tensor binding or the PDL policy changes."""

    plan: RoutePlan
    module_name: str
    prefix: torch.Tensor
    blocks: torch.Tensor
    out: torch.Tensor
    launches: tuple[tuple[Callable[..., Any], tuple], ...] = field(repr=False)

    def launch(self) -> torch.Tensor:
        with tvm_ffi.use_torch_stream():
            for entry, arguments in self.launches:
                entry(*arguments)
        return self.out

    __call__ = launch

    @property
    def launch_count(self) -> int:
        return len(self.launches)


def generated_program_available(
    device: torch.device,
    M: Optional[int] = None,
    num_blocks: Optional[int] = None,
    *,
    enable_pdl: bool = False,
    common_path: bool = True,
    has_delta: bool = True,
    block_write_idx: int = -1,
    apply_output_norm: bool = True,
) -> bool:
    """True when this checkout registers the program of that call on ``device`` (or any
    program for the device's architecture when ``M`` / ``num_blocks`` are omitted)."""
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(_device_capability(_device_index(device)))
    if arch is None:
        return False
    if M is None and num_blocks is None:
        return bool(MODULES) and route_available(arch)
    if M is None or num_blocks is None:
        raise ValueError("pass both M and num_blocks or neither")
    plan = plan_route(
        arch,
        sm_count_for(device),
        M,
        num_blocks,
        enable_pdl,
        has_delta=has_delta,
        block_write_idx=block_write_idx,
        apply_output_norm=apply_output_norm,
        common_path=common_path,
    )
    return route_available(arch, (plan.kernel_key,))


def prepare_kimi_k3_attn_res(
    prefix: torch.Tensor,
    delta: Optional[torch.Tensor],
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    num_blocks: int,
    block_write_idx: int = -1,
    eps: float = 1e-5,
    output_norm_eps: float = 1e-5,
    enable_pdl: bool = False,
) -> KimiK3AttnResRunner:
    """Validate one call, select its generated program and bind the launch."""
    validate_kimi_k3_attn_res_inputs(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        out,
        num_blocks=num_blocks,
        block_write_idx=block_write_idx,
        eps=eps,
        output_norm_eps=output_norm_eps,
    )
    device = prefix.device
    arch = arch_for(device)
    M = int(prefix.shape[0])
    common = common_path_eligible(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        out,
        block_write_idx=block_write_idx,
    )
    plan = plan_route(
        arch,
        sm_count_for(device),
        M,
        num_blocks,
        enable_pdl,
        has_delta=delta is not None,
        block_write_idx=int(block_write_idx),
        apply_output_norm=output_norm_weight is not None,
        common_path=common,
    )
    module_name = kernel_module_name(arch, plan.kernel_key)
    # The bootstrap program receives valid dummy pointers for absent optional operands
    # and never dereferences them (Cake production substitutes the same operands).
    bound_delta = delta if delta is not None else prefix
    bound_output_norm_weight = (
        output_norm_weight if output_norm_weight is not None else norm_weight
    )
    kwargs: dict[str, Any] = dict(
        grid=(int(plan.grid_x), 1, 1),
        out=out,
        prefix=prefix,
        delta=bound_delta,
        blocks=blocks,
        norm_weight=norm_weight,
        qk_weight=qk_weight,
        output_norm_weight=bound_output_norm_weight,
        out_stride=int(out.stride(0)),
        prefix_stride=int(prefix.stride(0)),
        delta_stride=int(bound_delta.stride(0)),
        blocks_m_stride=int(blocks.stride(0)),
        blocks_k_stride=int(blocks.stride(1)),
        eps=float(eps),
        output_norm_eps=float(output_norm_eps),
        block_write_idx=int(block_write_idx),
        num_blocks=int(num_blocks),
        has_delta=int(delta is not None),
        apply_output_norm=int(output_norm_weight is not None),
        enable_pdl=int(enable_pdl),
        M=M,
    )
    return KimiK3AttnResRunner(
        plan, module_name, prefix, blocks, out, (_bind(module_name, kwargs),)
    )


def kimi_k3_attn_res(
    prefix: torch.Tensor,
    delta: Optional[torch.Tensor],
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    num_blocks: int,
    block_write_idx: int = -1,
    eps: float = 1e-5,
    output_norm_eps: float = 1e-5,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Prepare and launch one call (see :func:`prepare_kimi_k3_attn_res`)."""
    return prepare_kimi_k3_attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        out,
        num_blocks=num_blocks,
        block_write_idx=block_write_idx,
        eps=eps,
        output_norm_eps=output_norm_eps,
        enable_pdl=enable_pdl,
    ).launch()


def reference_kimi_k3_attn_res(
    prefix: torch.Tensor,
    delta: Optional[torch.Tensor],
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    num_blocks: int,
    block_write_idx: int = -1,
    eps: float = 1e-5,
    output_norm_eps: float = 1e-5,
) -> torch.Tensor:
    """Independent FP32 PyTorch reference with the exact BF16 state mutations (tests / benchmark)."""
    if delta is not None:
        prefix.add_(delta)
    if int(block_write_idx) >= 0:
        blocks[:, int(block_write_idx), :].copy_(prefix)
    K = int(num_blocks)
    if K:
        candidates = torch.cat(
            (blocks[:, :K, :].float(), prefix[:, None, :].float()), dim=1
        )
    else:
        candidates = prefix[:, None, :].float()
    inv_rms = torch.rsqrt(candidates.square().mean(dim=-1) + float(eps))
    scores = (
        candidates * inv_rms[..., None] * (norm_weight.float() * qk_weight.float())
    ).sum(dim=-1)
    probabilities = torch.softmax(scores, dim=1)
    mixed = (probabilities[..., None] * candidates).sum(dim=1)
    if output_norm_weight is not None:
        mixed = (
            mixed
            * torch.rsqrt(
                mixed.square().mean(dim=-1, keepdim=True) + float(output_norm_eps)
            )
            * output_norm_weight.float()
        )
    out.copy_(mixed.to(torch.bfloat16))
    return out
