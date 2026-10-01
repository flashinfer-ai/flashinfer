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

JIT loading for the Cake MoE all-reduce union (SM100 and SM103, world sizes 2, 4, 8).

The union is a set of generated CUDA kernels under
``csrc/cake_trtllm_moe_allreduce_union/kernels``.  One kernel serves every rank
of a route (``world_rank`` is a kernel argument), PDL and the cooperative
launch are runtime launch flags, and the SM100/SM103 programs share one source
(the only per-architecture statement is guarded by ``__CUDA_ARCH__``).  Every
kernel is compiled per architecture together with the one launcher
translation unit ``cake_trtllm_moe_allreduce_union_launcher.cu``.

``KERNELS`` lists the kernels; ``ROUTES`` maps one route key
``(arch, world_size, dtype, launch_with_pdl, specialization)`` to the kernel
(or the per-rank kernels of a rank-specialised route) and its residency
parameters; ``_SPECIALIZATION_RULES`` selects a specialization for the
reviewed ``num_experts`` and token ranges, and ``_WIDE_MLP_CLASSES`` names the
classes whose program outside those ranges is ``wide_mlp`` (``generic``
otherwise).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal, NamedTuple, Optional, Sequence, Union

import torch

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

SOURCE_PACKAGE = "cake_trtllm_moe_allreduce_union"
LAUNCHER_SOURCE = f"{SOURCE_PACKAGE}_launcher.cu"
KERNEL_DIR = "kernels"
ARCHES = ("sm_100a", "sm_103a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}
_NVCC_FLAGS_BY_ARCH = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}
WORLD_SIZES = (2, 4, 8)
HIDDEN_DIM = 7168
WORKSPACE_CONTROL_INDEX_FACTOR = 3

SPECIALIZATION_GENERIC = "generic"
SPECIALIZATION_WIDE_MLP = "wide_mlp"

_DTYPE_NAME = {torch.float16: "float16", torch.bfloat16: "bfloat16"}
_DTYPE_CTYPE = {"float16": "__half", "bfloat16": "__nv_bfloat16"}


class Kernel(NamedTuple):
    """One generated kernel translation unit."""

    dtype: str
    world_size: int
    block: int
    cluster: int


class Route(NamedTuple):
    """The kernel(s) and residency of one route.

    ``kernel`` is one kernel name, or one name per rank for the rank-specialised
    routes.  ``persistent_ctas_per_sm`` and ``max_persistent_clusters`` size the
    persistent cluster grid (see :func:`launch_grid_x`); ``cooperative`` launches
    one co-resident cluster per token.
    """

    kernel: Union[str, tuple[str, ...]]
    persistent_ctas_per_sm: int
    max_persistent_clusters: Optional[int]
    cooperative: bool


KERNELS: dict[str, Kernel] = {}

ROUTES: dict[tuple[str, int, str, bool, str], Route] = {}

# (arch, world_size, dtype, launch_with_pdl, num_experts) ->
#     ((token_lo, token_hi, specialization), ...)
# Inclusive token ranges, sorted and non-overlapping: a launch whose token_num
# falls inside a range runs that specialization; every other launch runs the
# class program.  Each range is the token span the specialization was measured
# correct and at least 2 % faster than the class program over.
_SPECIALIZATION_RULES: dict[
    tuple[str, int, str, bool, int], tuple[tuple[int, int, str], ...]
] = {}

# (arch, world_size, dtype, launch_with_pdl) classes whose program is the
# ``wide_mlp`` schedule for every token count outside the rules' ranges.
_WIDE_MLP_CLASSES: tuple[tuple[str, int, str, bool], ...] = ()

_EXPORTED_SCOPES: frozenset[tuple[str, int]] = frozenset(
    (key[0], key[1]) for key in ROUTES
)


def arch_for_capability(device_capability: Sequence[int]) -> Optional[str]:
    return ARCH_BY_CAPABILITY.get(
        (int(device_capability[0]), int(device_capability[1]))
    )


def exported_arches() -> tuple[str, ...]:
    return tuple(sorted({key[0] for key in ROUTES}))


def select_specialization(
    arch: str,
    world_size: int,
    dtype_name: str,
    launch_with_pdl: bool,
    token_num: int,
    active_experts: int,
) -> str:
    """Return the specialization name for one launch configuration.

    The token-range rule of the ``(arch, world_size, dtype, launch_with_pdl,
    active_experts)`` key that holds ``token_num`` wins; outside every range the
    class program runs (``wide_mlp`` for the ``_WIDE_MLP_CLASSES`` classes,
    ``generic`` otherwise).
    """

    token_num = int(token_num)
    rules = _SPECIALIZATION_RULES.get(
        (
            str(arch),
            int(world_size),
            str(dtype_name),
            bool(launch_with_pdl),
            int(active_experts),
        ),
        (),
    )
    for token_lo, token_hi, specialization in rules:
        if token_lo <= token_num <= token_hi:
            return specialization
    if (
        str(arch),
        int(world_size),
        str(dtype_name),
        bool(launch_with_pdl),
    ) in _WIDE_MLP_CLASSES:
        return SPECIALIZATION_WIDE_MLP
    return SPECIALIZATION_GENERIC


def route_applies(*, world_size: int, device_capability: Sequence[int]) -> bool:
    """Whether the union has programs for this (architecture, world size).

    The union owns exactly the (architecture, world size) scopes it has routes
    for, with or without the all-reduce output: ``moe_allreduce_out`` is a
    runtime-optional output of every union kernel, so the caller's output
    selection does not narrow the scope.
    """

    arch = arch_for_capability(device_capability)
    return arch is not None and (arch, int(world_size)) in _EXPORTED_SCOPES


def route_for(
    *,
    arch: str,
    world_size: int,
    dtype_name: str,
    launch_with_pdl: bool,
    token_num: int,
    active_experts: int,
) -> tuple[tuple[str, int, str, bool, str], Route]:
    """Return the route key and record for one launch configuration."""

    key = (
        str(arch),
        int(world_size),
        str(dtype_name),
        bool(launch_with_pdl),
        select_specialization(
            arch, world_size, dtype_name, launch_with_pdl, token_num, active_experts
        ),
    )
    route = ROUTES.get(key)
    if route is None:
        raise ValueError(f"unsupported Cake MoE all-reduce union route: {key}")
    return key, route


def route_kernel(route: Route, world_rank: int, world_size: int) -> str:
    """Return the kernel name a rank launches on one route."""

    if not 0 <= int(world_rank) < int(world_size):
        raise ValueError(f"world_rank must be in [0, {world_size}), got {world_rank}")
    if isinstance(route.kernel, str):
        return route.kernel
    if len(route.kernel) != int(world_size):
        raise RuntimeError("rank-specialised route does not carry one kernel per rank")
    return route.kernel[int(world_rank)]


def launch_grid_x(
    token_num: int,
    cooperative: bool,
    sm_count: int,
    persistent_ctas_per_sm: int,
    cluster_ctas: int,
    max_persistent_clusters: Optional[int] = None,
) -> int:
    """Cooperative routes launch one cluster per token; every other route uses
    the SM-bounded persistent cluster grid that keeps ``persistent_ctas_per_sm``
    CTAs resident on every SM, capped at ``max_persistent_clusters`` clusters
    where the route records a co-residency cap.  ``cluster_ctas`` is the
    kernel's cluster width (four 224-thread CTAs per token, or one 896-thread
    CTA).  Every cluster of the persistent grid must be co-resident (each
    cluster publishes all of its tiles before it polls the peers' same-index
    tiles); the recorded cap is the exporter's ``cuOccupancyMaxActiveClusters``
    answer on the architecture it was measured on and is never raised here."""

    ctas_per_sm = int(persistent_ctas_per_sm)
    if ctas_per_sm < 1:
        raise ValueError("persistent_ctas_per_sm must be a positive integer")
    cluster = int(cluster_ctas)
    if cluster < 1:
        raise ValueError("cluster_ctas must be a positive integer")
    if max_persistent_clusters is not None and int(max_persistent_clusters) < 1:
        raise ValueError("max_persistent_clusters must be a positive integer or None")
    if cooperative:
        if ctas_per_sm != 1:
            raise ValueError("cooperative routes launch one cluster per token")
        if max_persistent_clusters is not None:
            raise ValueError("cooperative routes record no persistent cluster cap")
        grid = int(token_num) * cluster
    else:
        grid = (
            min(int(sm_count) * ctas_per_sm, int(token_num) * cluster) // cluster
        ) * cluster
        if max_persistent_clusters is not None:
            grid = min(grid, int(max_persistent_clusters) * cluster)
    if grid <= 0:
        raise ValueError("the launch requires at least one complete cluster")
    return grid


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / SOURCE_PACKAGE
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / SOURCE_PACKAGE
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "Cake MoE all-reduce union CUDA sources were not found. Checked:\n"
        f"  - {installed}\n  - {checkout}"
    )


def kernel_source(name: str) -> Path:
    return _source_dir() / KERNEL_DIR / f"{name}.cu"


def kernel_symbol(name: str) -> str:
    return f"kernel_{SOURCE_PACKAGE}_{name}"


def kernel_defines(name: str) -> list[str]:
    """Preprocessor definitions that bind the launcher to one kernel."""

    kernel = KERNELS[name]
    return [
        f"-DCAKE_UNION_KERNEL={kernel_symbol(name)}",
        f"-DCAKE_UNION_DTYPE={_DTYPE_CTYPE[kernel.dtype]}",
        f"-DCAKE_UNION_WORLD_SIZE={kernel.world_size}",
        f"-DCAKE_UNION_BLOCK={kernel.block}",
        f"-DCAKE_UNION_CLUSTER={kernel.cluster}",
    ]


def compile_flags(name: str, arch: str) -> list[str]:
    if arch not in ARCHES:
        raise ValueError(f"unsupported architecture {arch!r}")
    return [*_NVCC_FLAGS_BY_ARCH[arch], "--use_fast_math", *kernel_defines(name)]


@functools.cache
def spec(name: str, arch: str) -> JitSpec:
    source_dir = _source_dir()
    return gen_jit_spec(
        name=f"{SOURCE_PACKAGE}_{name}_{arch}",
        sources=[kernel_source(name), source_dir / LAUNCHER_SOURCE],
        extra_cuda_cflags=compile_flags(name, arch),
        extra_include_paths=[source_dir.parent],
    )


@functools.cache
def load(name: str, arch: str):
    return spec(name, arch).build_and_load()


@functools.cache
def _device_arch(device_index: int) -> Optional[str]:
    return arch_for_capability(torch.cuda.get_device_capability(device_index))


@functools.cache
def _sm_count(device_index: int) -> int:
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def run_cake_moe_allreduce_union(
    *,
    backend: Literal["cake"] = "cake",
    world_size: int,
    world_rank: int,
    token_num: int,
    hidden_dim: int,
    workspace_ptrs: torch.Tensor,
    launch_with_pdl: bool,
    residual_in: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    scale_factor: float,
    moe_reduction_device_num_experts: int,
    moe_reduction_scale_input: torch.Tensor,
    moe_reduction_active_experts_token_input: torch.Tensor,
    moe_reduction_token_input: torch.Tensor,
    moe_allreduce_out: Optional[torch.Tensor],
    residual_out: torch.Tensor,
    norm_out: torch.Tensor,
    weight_bias: Optional[float],
) -> None:
    """Select and launch one union kernel for the caller's validated tensors.

    ``moe_allreduce_out`` may be ``None``: the launcher then clears the kernel's
    runtime ``emit_moe_allreduce_out`` flag and no all-reduce output is stored.
    ``scale_factor`` belongs to the public API's quant path; the union kernels
    emit no quant output and take no scale factor.
    """

    if backend != "cake":
        raise ValueError(f"backend must be 'cake', got {backend!r}")
    del scale_factor
    world_size = int(world_size)
    if world_size not in WORLD_SIZES:
        raise ValueError(
            f"the Cake MoE all-reduce union covers world sizes {WORLD_SIZES}"
        )
    if int(hidden_dim) != HIDDEN_DIM:
        raise ValueError(
            f"the Cake MoE all-reduce union requires hidden_dim={HIDDEN_DIM}"
        )
    dtype_name = _DTYPE_NAME.get(moe_reduction_active_experts_token_input.dtype)
    if dtype_name is None:
        raise ValueError(
            "the Cake MoE all-reduce union supports float16 and bfloat16 only"
        )
    needed = WORKSPACE_CONTROL_INDEX_FACTOR * world_size + 1
    if workspace_ptrs.numel() < needed:
        raise ValueError(f"workspace_ptrs must contain at least {needed} pointers")
    device_index = moe_reduction_active_experts_token_input.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    arch = _device_arch(device_index)
    if arch is None or (arch, world_size) not in _EXPORTED_SCOPES:
        capability = torch.cuda.get_device_capability(device_index)
        raise ValueError(
            "the Cake MoE all-reduce union has no programs for "
            f"SM{capability[0]}{capability[1]} at world size {world_size}"
        )
    _key, route = route_for(
        arch=arch,
        world_size=world_size,
        dtype_name=dtype_name,
        launch_with_pdl=bool(launch_with_pdl),
        token_num=int(token_num),
        active_experts=int(moe_reduction_device_num_experts),
    )
    name = route_kernel(route, int(world_rank), world_size)
    kernel = KERNELS[name]
    grid_x = launch_grid_x(
        int(token_num),
        route.cooperative,
        _sm_count(device_index),
        route.persistent_ctas_per_sm,
        kernel.cluster,
        route.max_persistent_clusters,
    )
    load(name, arch).run(
        moe_reduction_active_experts_token_input,
        moe_reduction_scale_input,
        moe_reduction_token_input,
        residual_in,
        rms_gamma,
        moe_allreduce_out,
        residual_out,
        norm_out,
        workspace_ptrs,
        int(world_rank),
        int(token_num),
        int(moe_reduction_device_num_experts),
        float(rms_eps),
        float(0.0 if weight_bias is None else weight_bias),
        grid_x,
        bool(launch_with_pdl),
        bool(route.cooperative),
    )


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "HIDDEN_DIM",
    "KERNELS",
    "ROUTES",
    "WORLD_SIZES",
    "Kernel",
    "Route",
    "arch_for_capability",
    "compile_flags",
    "exported_arches",
    "kernel_defines",
    "kernel_source",
    "kernel_symbol",
    "launch_grid_x",
    "load",
    "route_applies",
    "route_for",
    "route_kernel",
    "run_cake_moe_allreduce_union",
    "select_specialization",
    "spec",
]
