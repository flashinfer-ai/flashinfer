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

Verified source-only JIT loading for the Cake SM100 / SM103 eight-peer fused norm-combine.

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding that FlashInfer's JIT compiles
together.  ``MODULES``, ``ROUTES`` and ``LARGE_MIN_TOKENS`` are filled by the
exporter from the complete verified program bundle.  The loader checks every
source file's SHA-256 before its first build, so a modified or partially
delivered source tree fails closed instead of silently running a different
kernel.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import Any, Literal, Sequence

import torch

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

# Filled mechanically from the complete verified program bundle.
MODULES: dict[str, dict[str, Any]] = {
    "cake_fused_norm_combine_bf16_0ac3aea2cc92e53cb058": {
        "arch": "sm_100a",
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "residual"),
            ("buffer", "weight"),
            ("buffer", "norm_out"),
            ("buffer", "residual_out"),
            ("buffer", "collective_out"),
            ("buffer", "workspace"),
            ("parameter", "rank"),
            ("parameter", "tokens"),
            ("parameter", "epsilon"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_fused_norm_combine_bf16_0ac3aea2cc92e53cb058_sm_100a",
        "compile_flags": [],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_fused_norm_combine_bf16_0ac3aea2cc92e53cb058",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 10496,
            "use_pdl": True,
        },
        "source_sha256": {
            "csrc/cake_fused_norm_combine/sm_100a/cake_fused_norm_combine_bf16_0ac3aea2cc92e53cb058_binding.cu": "f3bfd0e43c1ee0d2b3796f8a55987018f8dd2034307a5aceccdb0827e0ad52c0",
            "csrc/cake_fused_norm_combine/sm_100a/cake_fused_norm_combine_bf16_0ac3aea2cc92e53cb058_kernel.cu": "5aea63b35b23f9a62c99b31e222c2448d894a434bc0b9dabe436c7d438af3488",
        },
        "sources": [
            "csrc/cake_fused_norm_combine/sm_100a/cake_fused_norm_combine_bf16_0ac3aea2cc92e53cb058_kernel.cu",
            "csrc/cake_fused_norm_combine/sm_100a/cake_fused_norm_combine_bf16_0ac3aea2cc92e53cb058_binding.cu",
        ],
    },
    "cake_fused_norm_combine_bf16_a4503e9f297ee7549f0d": {
        "arch": "sm_103a",
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "residual"),
            ("buffer", "weight"),
            ("buffer", "norm_out"),
            ("buffer", "residual_out"),
            ("buffer", "collective_out"),
            ("buffer", "workspace"),
            ("parameter", "rank"),
            ("parameter", "tokens"),
            ("parameter", "epsilon"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_fused_norm_combine_bf16_a4503e9f297ee7549f0d_sm_103a",
        "compile_flags": [],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_fused_norm_combine_bf16_a4503e9f297ee7549f0d",
        "launch": {
            "block": (160, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
        "source_sha256": {
            "csrc/cake_fused_norm_combine/sm_103a/cake_fused_norm_combine_bf16_a4503e9f297ee7549f0d_binding.cu": "dd4b44333046149e289582dca4a739c6f6bef1f1b81d8ee23918b2251becff80",
            "csrc/cake_fused_norm_combine/sm_103a/cake_fused_norm_combine_bf16_a4503e9f297ee7549f0d_kernel.cu": "ee7bfcaa3bc2b01b64a99648ea2a99775a7a43c0b6e7c3895ff99844fc4fb2c5",
        },
        "sources": [
            "csrc/cake_fused_norm_combine/sm_103a/cake_fused_norm_combine_bf16_a4503e9f297ee7549f0d_kernel.cu",
            "csrc/cake_fused_norm_combine/sm_103a/cake_fused_norm_combine_bf16_a4503e9f297ee7549f0d_binding.cu",
        ],
    },
    "cake_fused_norm_combine_bf16_d1b159bb3eda5b4f3912": {
        "arch": "sm_100a",
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "residual"),
            ("buffer", "weight"),
            ("buffer", "norm_out"),
            ("buffer", "residual_out"),
            ("buffer", "collective_out"),
            ("buffer", "workspace"),
            ("parameter", "rank"),
            ("parameter", "tokens"),
            ("parameter", "epsilon"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_fused_norm_combine_bf16_d1b159bb3eda5b4f3912_sm_100a",
        "compile_flags": [],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_fused_norm_combine_bf16_d1b159bb3eda5b4f3912",
        "launch": {
            "block": (160, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": True,
        },
        "source_sha256": {
            "csrc/cake_fused_norm_combine/sm_100a/cake_fused_norm_combine_bf16_d1b159bb3eda5b4f3912_binding.cu": "8e729b3fea199c0afbfebce4bc3c7af4be642ce480c995ac272b6b19522a6273",
            "csrc/cake_fused_norm_combine/sm_100a/cake_fused_norm_combine_bf16_d1b159bb3eda5b4f3912_kernel.cu": "751271d84f95e68ad21b9c1118af970e50f429cf8e8ee22be5623577f53f7859",
        },
        "sources": [
            "csrc/cake_fused_norm_combine/sm_100a/cake_fused_norm_combine_bf16_d1b159bb3eda5b4f3912_kernel.cu",
            "csrc/cake_fused_norm_combine/sm_100a/cake_fused_norm_combine_bf16_d1b159bb3eda5b4f3912_binding.cu",
        ],
    },
    "cake_fused_norm_combine_bf16_ee481367764d334ab381": {
        "arch": "sm_103a",
        "arg_plan": [
            ("buffer", "x"),
            ("buffer", "residual"),
            ("buffer", "weight"),
            ("buffer", "norm_out"),
            ("buffer", "residual_out"),
            ("buffer", "collective_out"),
            ("buffer", "workspace"),
            ("parameter", "rank"),
            ("parameter", "tokens"),
            ("parameter", "epsilon"),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ],
        "cache_name": "cake_fused_norm_combine_bf16_ee481367764d334ab381_sm_103a",
        "compile_flags": [],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_fused_norm_combine_bf16_ee481367764d334ab381",
        "launch": {
            "block": (320, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 10496,
            "use_pdl": True,
        },
        "source_sha256": {
            "csrc/cake_fused_norm_combine/sm_103a/cake_fused_norm_combine_bf16_ee481367764d334ab381_binding.cu": "e444156fe651e1649cf414a217f71e67998725f2d968a685666461cc114c60ef",
            "csrc/cake_fused_norm_combine/sm_103a/cake_fused_norm_combine_bf16_ee481367764d334ab381_kernel.cu": "888daa8fe93eb4b164d724148423ad6aed936a121f40065d728a03de864cdae6",
        },
        "sources": [
            "csrc/cake_fused_norm_combine/sm_103a/cake_fused_norm_combine_bf16_ee481367764d334ab381_kernel.cu",
            "csrc/cake_fused_norm_combine/sm_103a/cake_fused_norm_combine_bf16_ee481367764d334ab381_binding.cu",
        ],
    },
}
ROUTES: dict[str, dict[str, str]] = {
    "sm_100a": {
        "owner_lamport": "cake_fused_norm_combine_bf16_d1b159bb3eda5b4f3912",
        "parallel_lamport": "cake_fused_norm_combine_bf16_0ac3aea2cc92e53cb058",
    },
    "sm_103a": {
        "owner_lamport": "cake_fused_norm_combine_bf16_a4503e9f297ee7549f0d",
        "parallel_lamport": "cake_fused_norm_combine_bf16_ee481367764d334ab381",
    },
}
LARGE_MIN_TOKENS = 256

SOURCE_PACKAGE = "cake_fused_norm_combine"
# One verified build per architecture: B200 (``sm_100a``) and B300 (``sm_103a``).
ARCHES = ("sm_100a", "sm_103a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}
_ARCH_NVCC_FLAGS: dict[str, list[str]] = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}
WORLD_SIZE = 8
HIDDEN_DIM = 2560
TRACKS = 2
# Token counts below ``LARGE_MIN_TOKENS`` run the one-shot Lamport exchange
# with the concurrent-track local body; at and above they run the fence-free
# two-round owner reduce (token t is owned by peer t % 8).
VARIANT_ONE_SHOT = "parallel_lamport"
VARIANT_OWNER_REDUCE = "owner_lamport"
# Workspace pointer table: eight data regions, eight flag regions, eight
# three-slot Lamport payload regions, then the local control words.
WORKSPACE_TABLE_ENTRIES = 3 * WORLD_SIZE + 1


def select_variant(tokens: int) -> str:
    """Name the exported kernel variant the dispatch launches for ``tokens`` rows."""

    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
        raise ValueError("tokens must be a positive integer")
    if tokens >= LARGE_MIN_TOKENS:
        return VARIANT_OWNER_REDUCE
    return VARIANT_ONE_SHOT


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""

    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def route_applies(
    *, world_size: int, device_capability: Sequence[int], hidden_dim: int
) -> bool:
    """Whether the verified export owns a fused norm-combine call."""

    arch = arch_for_capability(device_capability)
    return (
        arch is not None
        and bool(ROUTES.get(arch))
        and int(world_size) == WORLD_SIZE
        and int(hidden_dim) == HIDDEN_DIM
    )


def route_module_name(tokens: int, arch: str) -> str:
    """Return the exported module name for one token count on ``arch``."""

    variant = select_variant(tokens)
    name = ROUTES.get(arch, {}).get(variant)
    if name is None:
        raise ValueError(
            f"unsupported Cake fused norm-combine route: {variant!r} on {arch!r}"
        )
    return name


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
    if parts[:2] != ("csrc", SOURCE_PACKAGE) or len(parts) != 4:
        raise ValueError(
            f"exported source path is outside the fused norm-combine package: {relative!r}"
        )
    return _source_dir().joinpath(*parts[2:])


@functools.cache
def verified_sources(name: str) -> tuple[Path, ...]:
    """Resolve one module's sources and verify their recorded SHA-256 digests."""

    record = MODULES[name]
    paths = []
    for relative in record["sources"]:
        path = _source_path(relative)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"Cake fused norm-combine source is missing: {path}"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = record["source_sha256"][relative]
        if digest != expected:
            raise RuntimeError(
                f"Cake fused norm-combine source {relative} does not match its "
                f"exported digest (expected {expected}, found {digest})"
            )
        paths.append(path)
    return tuple(paths)


@functools.cache
def spec(name: str) -> JitSpec:
    record = MODULES[name]
    return gen_jit_spec(
        name=record["cache_name"],
        sources=list(verified_sources(name)),
        extra_cuda_cflags=[*_ARCH_NVCC_FLAGS[record["arch"]], *record["compile_flags"]],
        extra_include_paths=[_source_dir().parent],
        # The source build carries every math flag in ``compile_flags``; the
        # default ``-use_fast_math`` would change rounding against it.
        use_fast_math="--use_fast_math" in record["compile_flags"],
    )


@functools.cache
def load(name: str):
    return spec(name).build_and_load()


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
    """Launch the exported module selected for ``tokens`` on validated tensors.

    ``flashinfer.comm.cake_fused_norm_combine`` validates shapes, dtypes,
    devices and the workspace before calling this entry; the launch grid is one
    CTA per token row on every peer.
    """

    if backend != "cake":
        raise ValueError(f"backend must be 'cake', got {backend!r}")
    if not 0 <= int(rank) < WORLD_SIZE:
        raise ValueError(f"rank must be in [0, {WORLD_SIZE}), got {rank}")
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    arch = arch_for_capability(torch.cuda.get_device_capability(device_index))
    if arch is None:
        raise ValueError(
            "the Cake fused norm-combine export targets SM100 and SM103 only"
        )
    name = route_module_name(int(tokens), arch)
    record = MODULES[name]
    values: dict[str, object] = {
        "x": x,
        "residual": residual,
        "weight": weight,
        "norm_out": norm_out,
        "residual_out": residual_out,
        "collective_out": collective_out,
        "workspace": workspace_ptrs,
        "rank": int(rank),
        "tokens": int(tokens),
        "epsilon": float(epsilon),
        "grid_x": int(tokens),
        "grid_y": 1,
        "grid_z": 1,
    }
    module = load(name)
    getattr(module, record["ffi_entry"])(
        *(values[key] for _kind, key in record["arg_plan"])
    )


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "HIDDEN_DIM",
    "LARGE_MIN_TOKENS",
    "MODULES",
    "ROUTES",
    "TRACKS",
    "VARIANT_ONE_SHOT",
    "VARIANT_OWNER_REDUCE",
    "WORKSPACE_TABLE_ENTRIES",
    "WORLD_SIZE",
    "arch_for_capability",
    "load",
    "route_applies",
    "route_module_name",
    "run_cake_fused_norm_combine",
    "select_variant",
    "spec",
    "verified_sources",
]
