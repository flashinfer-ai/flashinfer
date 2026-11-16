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

Host plan and loader glue of the Cake experimental FP8 1D1D GEMM programs
(SM100 / SM103), behind :func:`.runtime.prepare_fp8_gemm_1d1d`.

Contract of the programs: ``C = A B^T`` over FP8 E4M3 operands ``A`` ``[M, K]``
and ``B`` ``[N, K]`` (``uint8`` bytes) with UE8M0 block scales at K128
granularity, packed four K blocks to a ``uint32`` word and stored MN-major
(``SFA`` ``[ceil(K/512), M]``, ``SFB`` ``[ceil(K/512), N]``;
:func:`.runtime.pack_ue8m0_words`).  The ``forward`` route writes BF16 ``C``;
the ``wgrad`` route accumulates FP32 ``C += A B^T`` in place.  Problem sizes
(``M``, ``N``, ``K``), the tile counts (``grid_m``, ``grid_n``, ``K_tiles``) and
the persistent CTA count are launch scalars, so one program per route serves
every ``M`` and ``N`` and every ``K`` that is a multiple of 128
(:func:`.runtime.launch_geometry` derives them).  Portions of the generated
device code derive from DeepGEMM (MIT licensed; ``DEEPGEMM_NOTICE.txt`` next
to this file).

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit plus a tvm-ffi binding, compiled once per
architecture.  A record carries the host plan only: the program's ``library``
stem (its ``build.json`` entries are ``<library>_<arch>``), the architectures
it is delivered for, the source ``module`` and ``build`` keywords it was
rendered from, its launch record, the argument plan of its ``run`` entry, the
kernel-parameter -> logical-argument map, the grid rule the runtime evaluates
and the workspace rule (``None``: the kernel owns no scratch;
``in_place_accumulate``: the launch reads and updates the named output, so
an evaluation restores the initializer first).  ``KERNELS`` maps each route
to its record.  What to compile -- sources, digests, nvcc flags -- is recorded
once in ``build.json`` next to this file; :mod:`.cake_build` serves the
library from a prebuilt ``aot/`` file when a builder compiled it and
otherwise through FlashInfer's JIT from the same sources and flags.  Every
``build.json`` entry records no nvcc flag of its own (the programs compile
with nvcc's default ``--fmad=true`` alone); the loader passes
``use_fast_math=False`` so that the manifest, not the loader, states the
numerics.

Tensor maps travel by value in the kernel parameters: the binding encodes a
``CUtensorMap`` from each tensor view per call, so a launch is self-contained,
CUDA-graph capturable, and two layers never share descriptor storage.  The
binding launches the program as a cluster of :data:`CLUSTER_CTAS` CTAs.

The derived leaves of a record (``kernel_symbol``, ``launch.block``,
``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
``launch.use_pdl``, ``arg_plan``, ``logical``) are rewritten by the exporter
from the rendered binding; every other field is the host plan.  Every value
is a literal: FlashInfer's manifest checker and the exporter read this file
with ``ast.literal_eval``.

The program name ``cake_deepgemm_fp8_gemm_<20 hex>`` is the first twenty hex
digits of the SHA-256 of the compact JSON ``{"build": <build keywords, sorted,
every default spelled>, "module": <dotted source module>}`` of the kernel the
program was rendered from; a changed keyword is a new program.
"""

from __future__ import annotations

import functools
from typing import Any

import tvm_ffi

from . import cake_build

# Two verified builds: B200 / GB200 (``sm_100a``) and B300 / GB300 (``sm_103a``),
# each compiled with its exact ``-gencode`` flag set (``build.json`` ``arch_flags``).
ARCHES = ("sm_100a", "sm_103a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}

# The CTAs of the cluster every record's binding launches (two adjacent M tiles
# per cluster pair).
CLUSTER_CTAS = 2

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
# ``launch.dynamic_smem_bytes`` is the kernel's computed shared memory at the
# last render.
MODULES: dict[str, dict[str, Any]] = {
    "cake_deepgemm_fp8_gemm_18870a6ac8a0a8be9883": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "C_tma"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "K_tiles"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "accumulate": False,
            "backend": "cuda_cpp",
            "num_sms": 148,
            "runtime_shape": True,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "block_k": 128,
            "block_m": 128,
            "block_n": 224,
            "kind": "deepgemm_persistent_even",
            "raster_group_m": 16,
        },
        "kernel": "cake_deepgemm_fp8_gemm_18870a6ac8a0a8be9883",
        "kernel_symbol": "kernel_cake_deepgemm_fp8_gemm_18870a6ac8a0a8be9883",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 210176,
            "use_pdl": False,
        },
        "library": "cake_deepgemm_fp8_gemm_18870a6ac8a0a8be9883",
        "logical": {
            "A": "A",
            "B": "B",
            "C_tma": "C_tma",
            "K": "K",
            "K_tiles": "K_tiles",
            "M": "M",
            "N": "N",
            "SFA": "SFA",
            "SFB": "SFB",
            "grid_m": "grid_m",
            "grid_n": "grid_n",
        },
        "module": "tart.fp8_gemm_nt_bf16_static_kmn.sm100_sm103.deepgemm_fp8_gemm_1d1d_source_normal",
        "workspace_rule": None,
    },
    "cake_deepgemm_fp8_gemm_3806a6e9116000fd4200": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "C_tma"],
            ["parameter", "M"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "grid_m"],
            ["parameter", "grid_n"],
            ["parameter", "K_tiles"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "accumulate": True,
            "backend": "cuda_cpp",
            "num_sms": 148,
            "runtime_shape": True,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "block_k": 128,
            "block_m": 128,
            "block_n": 224,
            "kind": "deepgemm_persistent_even",
            "raster_group_m": 16,
        },
        "kernel": "cake_deepgemm_fp8_gemm_3806a6e9116000fd4200",
        "kernel_symbol": "kernel_cake_deepgemm_fp8_gemm_3806a6e9116000fd4200",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 226560,
            "use_pdl": False,
        },
        "library": "cake_deepgemm_fp8_gemm_3806a6e9116000fd4200",
        "logical": {
            "A": "A",
            "B": "B",
            "C_tma": "C_tma",
            "K": "K",
            "K_tiles": "K_tiles",
            "M": "M",
            "N": "N",
            "SFA": "SFA",
            "SFB": "SFB",
            "grid_m": "grid_m",
            "grid_n": "grid_n",
        },
        "module": "tart.fp8_gemm_nt_bf16_static_kmn.sm100_sm103.deepgemm_fp8_gemm_1d1d_source_normal",
        "workspace_rule": {
            "kind": "in_place_accumulate",
            "output": "C_tma",
        },
    },
}

# The route (``forward``: BF16 output; ``wgrad``: FP32 accumulate in place) ->
# the record that serves it.
KERNELS: dict[str, str] = {
    "forward": "cake_deepgemm_fp8_gemm_18870a6ac8a0a8be9883",
    "wgrad": "cake_deepgemm_fp8_gemm_3806a6e9116000fd4200",
}


def arch_for_capability(capability: tuple[int, int]) -> str | None:
    """The exported architecture of one device capability, if any."""
    values = tuple(int(value) for value in capability)
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


@functools.cache
def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when the device
    is not an exported target; the capability is immutable per process)."""
    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported "
            f"Cake FP8 1D1D GEMM target; the exported architectures are {ARCHES}"
        )
    return arch


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""
    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake FP8 1D1D GEMM record {name} is not delivered for {arch}; "
            f"its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``: the
    prebuilt ``aot/`` library when a builder compiled it, else FlashInfer's JIT
    build of the same sources and flags (``cake_build.load_library``)."""
    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up through FlashInfer's ``build_jit_specs``)."""
    return cake_build.jit_specs(arch)


def launch_args(
    record: dict[str, Any], values: dict[str, Any], grid: tuple[int, int, int]
) -> tuple[Any, ...]:
    """The positional arguments of the record's ``run`` entry: ``values`` by the
    logical name of each kernel parameter of ``arg_plan`` and the grid axes."""
    logical = record["logical"]
    args: list[Any] = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            args.append(int(grid["xyz".index(key[-1])]))
            continue
        if key not in logical:
            raise ValueError(f"kernel parameter {key!r} has no logical name")
        try:
            args.append(values[logical[key]])
        except KeyError:
            raise ValueError(
                f"the launch needs logical argument {logical[key]!r}"
            ) from None
    return tuple(args)


def run_module(
    name: str,
    values: dict[str, Any],
    *,
    grid: tuple[int, int, int],
    device_index: int,
) -> None:
    """Launch record ``name`` with logical argument ``values`` on ``grid`` (the
    runtime's ``launch_geometry`` grid) on the current torch stream of the device."""
    record = MODULES[name]
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*launch_args(record, values, grid))


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "CLUSTER_CTAS",
    "KERNELS",
    "MODULES",
    "arch_for_capability",
    "device_arch",
    "launch_args",
    "library_name",
    "load",
    "run_module",
    "sm_count",
    "specs_for_arch",
]
