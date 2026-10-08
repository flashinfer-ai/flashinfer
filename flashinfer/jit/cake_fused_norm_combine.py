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

JIT loader of the Cake eight-peer fused norm-combine (SM100 / SM103).

Three generated programs, one per dispatch band: a CUDA device translation
unit plus a tvm-ffi launcher each, compiled together once into one fatbin
that carries a cubin for every supported build target
(``FLASHINFER_CUDA_ARCH_LIST`` when set, otherwise the visible devices). The
persistent program additionally compiles ``OCCUPANCY_SOURCE``, which exports
the kernel's co-resident CTA capacity query. ``VARIANTS``, ``COMPILE_FLAGS``,
``LARGE_MIN_TOKENS`` and ``WIDE_MIN_TOKENS`` are filled by the exporter from
the program bundle.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any, Literal, NamedTuple, Optional

import torch

from ..compilation_context import CompilationContext
from . import env as jit_env
from .core import JitSpec, gen_jit_spec

# Filled mechanically from the program bundle.
VARIANTS: dict[str, dict[str, Any]] = {
    "parallel_lamport": {
        "module": "cake_fused_norm_combine_bf16_84f760a61c7d96e11441",
        "kernel_symbol": "kernel_cake_fused_norm_combine_bf16_84f760a61c7d96e11441",
        "sources": [
            "csrc/cake_fused_norm_combine/cake_fused_norm_combine_bf16_84f760a61c7d96e11441_kernel.cu",
            "csrc/cake_fused_norm_combine/cake_fused_norm_combine_bf16_84f760a61c7d96e11441_binding.cu",
        ],
        "block": 320,
        "dynamic_smem_bytes": 10496,
        "grid_rule": {"kind": "per_token"},
    },
    "owner_lamport": {
        "module": "cake_fused_norm_combine_bf16_2da1ef49a067e64f946f",
        "kernel_symbol": "kernel_cake_fused_norm_combine_bf16_2da1ef49a067e64f946f",
        "sources": [
            "csrc/cake_fused_norm_combine/cake_fused_norm_combine_bf16_2da1ef49a067e64f946f_kernel.cu",
            "csrc/cake_fused_norm_combine/cake_fused_norm_combine_bf16_2da1ef49a067e64f946f_binding.cu",
        ],
        "block": 160,
        "dynamic_smem_bytes": 128,
        "grid_rule": {"kind": "per_token"},
    },
    "owner_lamport_pipelined": {
        "module": "cake_fused_norm_combine_bf16_490c897e3cff3f38e0b8",
        "kernel_symbol": "kernel_cake_fused_norm_combine_bf16_490c897e3cff3f38e0b8",
        "sources": [
            "csrc/cake_fused_norm_combine/cake_fused_norm_combine_bf16_490c897e3cff3f38e0b8_kernel.cu",
            "csrc/cake_fused_norm_combine/cake_fused_norm_combine_bf16_490c897e3cff3f38e0b8_binding.cu",
        ],
        "block": 160,
        "dynamic_smem_bytes": 128,
        "grid_rule": {"kind": "co_resident"},
        "capacity_receipt": {"ctas_per_sm": 4, "sm_count": 148},
    },
}
COMPILE_FLAGS: list[str] = []
LARGE_MIN_TOKENS = 256
WIDE_MIN_TOKENS = 1024

SOURCE_PACKAGE = "cake_fused_norm_combine"
OCCUPANCY_SOURCE = f"csrc/{SOURCE_PACKAGE}/cake_fused_norm_combine_occupancy.cu"
# The kernels use PDL, volatile vector memory operations and warp shuffles
# only; one source serves B200 (10.0) and B300 (10.3), which are the two
# capabilities the export was validated on.
SUPPORTED_MAJOR_VERSIONS: tuple[int, ...] = (10,)
SUPPORTED_CAPABILITIES: tuple[tuple[int, int], ...] = ((10, 0), (10, 3))
WORLD_SIZE = 8
HIDDEN_DIM = 2560
TRACKS = 2
# Token counts below ``LARGE_MIN_TOKENS`` run the one-shot Lamport exchange
# with the concurrent-track local body; from ``LARGE_MIN_TOKENS`` they run the
# fence-free two-round owner reduce (token t is owned by peer t % 8) with one
# CTA per token; from ``WIDE_MIN_TOKENS`` the same owner reduce runs as a
# persistent software-pipelined grid (each CTA publishes token k while it
# reduces its owned token k - 2) whose CTAs must all be co-resident: its grid
# is sized from the capacity the driver reports for the compiled kernel on
# the launching device (``co_resident_capacity``).
VARIANT_ONE_SHOT = "parallel_lamport"
VARIANT_OWNER_REDUCE = "owner_lamport"
VARIANT_PIPELINED = "owner_lamport_pipelined"
# Workspace pointer table: eight data regions, eight flag regions, eight
# three-slot Lamport payload regions, then the local control words.
WORKSPACE_TABLE_ENTRIES = 3 * WORLD_SIZE + 1


class DeviceFacts(NamedTuple):
    capability: tuple[int, int]
    sm_count: int


def select_variant(tokens: int) -> str:
    """Name the kernel variant the dispatch launches for ``tokens`` rows."""

    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        raise ValueError("tokens must be a positive integer")
    if tokens >= WIDE_MIN_TOKENS:
        return VARIANT_PIPELINED
    if tokens >= LARGE_MIN_TOKENS:
        return VARIANT_OWNER_REDUCE
    return VARIANT_ONE_SHOT


def _capability_of(major: Any, minor: Any) -> tuple[int, int]:
    """``(major, minor)`` of a ``CompilationContext.TARGET_CUDA_ARCHS`` entry (``(10, "3a")`` -> ``(10, 3)``)."""

    return int(major), int(re.sub(r"[a-z]+$", "", str(minor)))


@functools.cache
def target_capabilities() -> tuple[tuple[int, int], ...]:
    """Supported capabilities the fatbin is built for in this process, ascending.

    Read once from ``CompilationContext`` (``FLASHINFER_CUDA_ARCH_LIST`` when
    set, else the visible devices) and restricted to
    :data:`SUPPORTED_CAPABILITIES`.
    """

    context = CompilationContext()
    targets = {
        _capability_of(major, minor) for major, minor in context.TARGET_CUDA_ARCHS
    }
    return tuple(sorted(targets & set(SUPPORTED_CAPABILITIES)))


def supported_capability(capability: tuple[int, int]) -> Optional[tuple[int, int]]:
    """Return ``(major, minor)`` when the programs are built for it, else ``None``."""

    key = (int(capability[0]), int(capability[1]))
    return key if key in target_capabilities() else None


def nvcc_flags() -> list[str]:
    """``-gencode`` flags for every supported build target plus FlashInfer's common flags."""

    return CompilationContext().get_nvcc_flags_list(
        supported_major_versions=list(SUPPORTED_MAJOR_VERSIONS)
    )


@functools.cache
def device_facts(device_index: int) -> DeviceFacts:
    """Compute capability and SM count of one device, queried once per process."""

    properties = torch.cuda.get_device_properties(device_index)
    return DeviceFacts(
        capability=(int(properties.major), int(properties.minor)),
        sm_count=int(properties.multi_processor_count),
    )


def route_applies(
    *, world_size: int, device_capability: tuple[int, int], hidden_dim: int
) -> bool:
    """Whether the export owns a fused norm-combine call."""

    return (
        bool(VARIANTS)
        and supported_capability(device_capability) is not None
        and int(world_size) == WORLD_SIZE
        and int(hidden_dim) == HIDDEN_DIM
    )


def persistent_grid(tokens: int, capacity: int) -> int:
    """Grid of the persistent pipelined kernel.

    ``tokens`` CTAs when they are all co-resident, otherwise the largest odd
    CTA count not above ``capacity``: an odd grid keeps the token-to-owner
    assignment (owner = token % 8) rotating across the CTAs.
    """

    if tokens <= 0 or capacity <= 0:
        raise ValueError("tokens and capacity must be positive")
    limit = min(tokens, capacity)
    if limit >= tokens:
        return tokens
    return limit if limit % 2 else limit - 1


@functools.cache
def co_resident_capacity(variant: str, device_index: int) -> int:
    """CTAs of ``variant`` the driver reports co-resident on one device.

    The compiled module answers through its ``max_active_blocks_per_sm`` entry
    (``OCCUPANCY_SOURCE``); the result is cached per (variant, device) for the
    process lifetime, like the other device facts.
    """

    row = VARIANTS[variant]
    if row["grid_rule"]["kind"] != "co_resident":
        raise ValueError(
            f"variant {variant!r} launches one CTA per token; it has no capacity query"
        )
    blocks = int(
        load(variant).max_active_blocks_per_sm(
            int(device_index), int(row["block"]), int(row["dynamic_smem_bytes"])
        )
    )
    if blocks < 1:
        raise RuntimeError(
            f"the driver reports no co-resident CTA for {row['module']} on cuda:{device_index}"
        )
    return blocks * device_facts(int(device_index)).sm_count


def launch_grid_x(
    tokens: int, rule: dict[str, Any], *, capacity: Optional[int] = None
) -> int:
    """Grid width for ``tokens`` under one variant's grid rule.

    ``per_token`` variants launch one CTA per token; the ``co_resident``
    variant launches ``persistent_grid(tokens, capacity)`` CTAs, where
    ``capacity`` is the device's answer from :func:`co_resident_capacity`.
    """

    kind = rule["kind"]
    if kind == "per_token":
        return int(tokens)
    if kind != "co_resident":
        raise ValueError(f"unknown Cake fused norm-combine grid rule: {kind!r}")
    if capacity is None:
        raise ValueError(
            "a co_resident grid rule needs the device's co-resident capacity"
        )
    return persistent_grid(int(tokens), int(capacity))


def resolve_launch(tokens: int, device_index: int) -> tuple[str, int]:
    """Variant and grid width for ``tokens`` on the device at ``device_index``."""

    facts = device_facts(int(device_index))
    if supported_capability(facts.capability) is None:
        raise ValueError(
            "the Cake fused norm-combine export is not built for compute capability "
            f"{facts.capability[0]}.{facts.capability[1]} (build targets: "
            f"{[f'{major}.{minor}' for major, minor in target_capabilities()]}; "
            "set FLASHINFER_CUDA_ARCH_LIST to include it)"
        )
    variant = select_variant(tokens)
    rule = VARIANTS[variant]["grid_rule"]
    capacity = (
        co_resident_capacity(variant, int(device_index))
        if rule["kind"] == "co_resident"
        else None
    )
    return variant, launch_grid_x(tokens, rule, capacity=capacity)


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / SOURCE_PACKAGE
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / SOURCE_PACKAGE
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "Cake fused norm-combine CUDA sources were not found. Checked:\n"
        f"  - {installed}\n  - {checkout}"
    )


def _source_path(relative: str) -> Path:
    parts = Path(relative).parts
    if parts[:2] != ("csrc", SOURCE_PACKAGE) or len(parts) != 3:
        raise ValueError(
            f"exported source path is outside the fused norm-combine package: {relative!r}"
        )
    return _source_dir() / parts[2]


@functools.cache
def spec(variant: str) -> JitSpec:
    row = VARIANTS[variant]
    flags = [*nvcc_flags(), *COMPILE_FLAGS]
    sources = [_source_path(relative) for relative in row["sources"]]
    # Keyed as ``gen_jit_spec`` declares it (``Mapping[str | Path, ...]``);
    # ``Mapping`` is invariant in its key type, so a ``dict[Path, ...]`` would not do.
    by_source: Optional[dict[str | Path, list[str]]] = None
    if row["grid_rule"]["kind"] == "co_resident":
        # The occupancy query is compiled against this program's kernel symbol.
        occupancy = _source_path(OCCUPANCY_SOURCE)
        sources.append(occupancy)
        by_source = {
            occupancy: [
                *flags,
                f"-DCAKE_FUSED_NORM_COMBINE_KERNEL={row['kernel_symbol']}",
            ]
        }
    return gen_jit_spec(
        name=row["module"],
        sources=sources,
        extra_cuda_cflags=flags,
        extra_include_paths=[_source_dir().parent],
        # The source build carries every math flag in ``COMPILE_FLAGS``; the
        # default ``-use_fast_math`` would change rounding against it.
        use_fast_math="--use_fast_math" in COMPILE_FLAGS,
        extra_cuda_cflags_by_source=by_source,
    )


@functools.cache
def load(variant: str):
    return spec(variant).build_and_load()


def run_cake_fused_norm_combine(
    *,
    backend: Literal["cake"] = "cake",
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    norm_out: torch.Tensor,
    residual_out: torch.Tensor,
    collective_out: torch.Tensor,
    workspace_ptrs: torch.Tensor,
    rank: int,
    tokens: int,
    epsilon: float,
) -> None:
    """Launch the variant selected for ``tokens`` on validated tensors.

    ``flashinfer.comm.cake_fused_norm_combine`` validates shapes, dtypes,
    devices and the workspace before calling this entry; the launch grid is one
    CTA per token row on every peer, or the co-resident persistent grid of the
    launching device for the wide route (``resolve_launch``).
    """

    if backend != "cake":
        raise ValueError(f"backend must be 'cake', got {backend!r}")
    if not 0 <= int(rank) < WORLD_SIZE:
        raise ValueError(f"rank must be in [0, {WORLD_SIZE}), got {rank}")
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    variant, grid_x = resolve_launch(int(tokens), device_index)
    load(variant).run(
        x,
        residual,
        weight,
        norm_out,
        residual_out,
        collective_out,
        workspace_ptrs,
        int(rank),
        int(tokens),
        float(epsilon),
        int(grid_x),
        1,
        1,
    )


__all__ = [
    "COMPILE_FLAGS",
    "HIDDEN_DIM",
    "LARGE_MIN_TOKENS",
    "SUPPORTED_CAPABILITIES",
    "SUPPORTED_MAJOR_VERSIONS",
    "TRACKS",
    "VARIANTS",
    "VARIANT_ONE_SHOT",
    "VARIANT_OWNER_REDUCE",
    "VARIANT_PIPELINED",
    "WIDE_MIN_TOKENS",
    "WORKSPACE_TABLE_ENTRIES",
    "WORLD_SIZE",
    "OCCUPANCY_SOURCE",
    "DeviceFacts",
    "co_resident_capacity",
    "device_facts",
    "launch_grid_x",
    "load",
    "nvcc_flags",
    "persistent_grid",
    "resolve_launch",
    "route_applies",
    "run_cake_fused_norm_combine",
    "select_variant",
    "spec",
    "supported_capability",
    "target_capabilities",
]
