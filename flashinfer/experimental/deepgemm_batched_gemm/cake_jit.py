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

Host plan and loader glue of the Cake experimental batched FP8 projection
programs (SM100a / SM103a) behind
:func:`flashinfer.fp8_batched_gemm.prepare_fp8_batched_gemm`.

Contract of the programs: ``D[T, H, N] = A[T, H, K] . B[H, N, K]`` per head,
with E4M3 operands and positive power-of-two FP32 scales per A token / K128
block and per B N128 / K128 block, packed into UE8M0 words by
:mod:`.batched_gemm`; ``H = 8``, ``K = 4096`` and ``N = 1024`` are compiled in,
``T`` (``M``), the M tile count (``grid_m``), the output-scale stride
(``sfd_stride``) and ``alpha`` are launch arguments.  Six programs serve the
routes of ``batched_gemm.route_config``: the general 128 x 128 schedule with
the runtime-``alpha`` BF16 epilogue, the 128 x 256 two-CTA schedule and the
two swapped 16 x 128 / 64 x 128 schedules with the dynamic FP8 epilogue (E4M3
values and per-32 UE8M0 scale words), and the two swapped BF16 schedules pinned
to ``T = 128`` and ``T = 4`` on 148 / 152 SMs.  ``ROUTES`` maps each schedule
key (``<route>_bm<block_m>_bn<block_n>_s<stages>_<epilogue>``) to its program.

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit plus a tvm-ffi binding, compiled once per
architecture.  A record carries the host plan only: the program's ``library``
stem (its ``build.json`` entries are ``<library>_<arch>``), the architectures
it is delivered for, the source ``module`` and ``build`` keywords it was
rendered from, its launch record, the argument plan of its ``run`` entry, the
kernel-parameter -> logical-argument map, the grid rule :func:`launch_grid`
evaluates and the workspace rule (``None``: the kernel owns no scratch).
What to compile -- sources, digests, nvcc flags -- is recorded once in
``build.json`` next to this file; :mod:`.cake_build` serves the library from a
prebuilt ``aot/`` file when a builder compiled it and otherwise through
FlashInfer's JIT from the same sources and flags.  The programs compile with
nvcc's default numerics (no fast-math flag; every ``build.json`` entry records
an empty flag list), the setting the kernels were validated bit-exactly under.

Two grid rules: ``one_cta_per_sm`` launches one persistent CTA per SM
(``grid_x = sm_count``); ``working_ctas_even`` launches the working CTAs of a
swapped schedule, an even count that keeps its CTA pairs intact (``tiles =
ceil(T / block_m) x (width / block_n) x num_heads``; ``grid_x = min(sm_count,
tiles + tiles % 2)``).  Every rule carries the program's ``block_m``, the tile
height behind the ``grid_m = ceil(T / block_m)`` launch argument.

Tensor maps travel by value in the kernel parameters: the binding encodes a
``CUtensorMap`` from each tensor view per call, so a launch is self-contained,
CUDA-graph capturable, and two plans never share descriptor storage.  The
binding launches the general program as plain CTAs and the five others as
clusters of two CTAs (``launch.cluster``), without programmatic dependent
launch.

The derived leaves of a record (``kernel_symbol``, ``launch.block``,
``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
``launch.use_pdl``, ``arg_plan``, ``logical``) are rewritten by the exporter
from the rendered binding; every other field is the host plan.  Every value
is a literal: FlashInfer's manifest checker and the exporter read this file
with ``ast.literal_eval``.

The program name ``cake_deepgemm_batched_gemm_<20 hex>`` is the first twenty
hex digits of the SHA-256 of the compact JSON ``{"build": <build keywords,
sorted, every default spelled>, "module": <dotted source module>}`` of the
kernel the program was rendered from; a changed keyword is a new program.
"""

from __future__ import annotations

import functools
from typing import Any

import tvm_ffi

from . import cake_build

# Two verified builds: B200 (``sm_100a``) and B300 / GB300 (``sm_103a``), each
# compiled with its exact ``-gencode`` flag set (``build.json`` ``arch_flags``).
ARCHES = ("sm_100a", "sm_103a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}

# The two grid rules of the records (see the module docstring).
ONE_CTA_PER_SM = "one_cta_per_sm"
WORKING_CTAS_EVEN = "working_ctas_even"

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
# ``launch.dynamic_smem_bytes`` is the kernel's computed shared memory at the
# last render.
MODULES: dict[str, dict[str, Any]] = {
    "cake_deepgemm_batched_gemm_1931777936a4c257b75f": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "SFD"],
            ["parameter", "M"],
            ["parameter", "grid_m"],
            ["parameter", "sfd_stride"],
            ["parameter", "alpha"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "epilogue": "alpha",
            "inner": 4096,
            "num_heads": 8,
            "num_stages": 5,
            "width": 1024,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "block_m": 128,
            "kind": "one_cta_per_sm",
        },
        "kernel": "cake_deepgemm_batched_gemm_1931777936a4c257b75f",
        "kernel_symbol": "kernel_cake_deepgemm_batched_gemm_1931777936a4c257b75f",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 201984,
            "use_pdl": False,
        },
        "library": "cake_deepgemm_batched_gemm_1931777936a4c257b75f",
        "logical": {
            "A": "A",
            "B": "B",
            "D": "D",
            "M": "M",
            "SFA": "SFA",
            "SFB": "SFB",
            "SFD": "SFD",
            "alpha": "alpha",
            "grid_m": "grid_m",
            "sfd_stride": "sfd_stride",
        },
        "module": "tart.fp8_einsum_bhr_hdr_bhd_fp8.sm100_sm103.deepgemm_fp8_batched_gemm",
        "workspace_rule": None,
    },
    "cake_deepgemm_batched_gemm_7e76dfdd9b10af2f8359": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "SFD"],
            ["parameter", "M"],
            ["parameter", "grid_m"],
            ["parameter", "sfd_stride"],
            ["parameter", "alpha"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "block_m": 64,
            "inner": 4096,
            "num_heads": 8,
            "num_stages": 10,
            "width": 1024,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "block_m": 64,
            "block_n": 128,
            "kind": "working_ctas_even",
            "num_heads": 8,
            "width": 1024,
        },
        "kernel": "cake_deepgemm_batched_gemm_7e76dfdd9b10af2f8359",
        "kernel_symbol": "kernel_cake_deepgemm_batched_gemm_7e76dfdd9b10af2f8359",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 219520,
            "use_pdl": False,
        },
        "library": "cake_deepgemm_batched_gemm_7e76dfdd9b10af2f8359",
        "logical": {
            "A": "A",
            "B": "B",
            "D": "D",
            "M": "M",
            "SFA": "SFA",
            "SFB": "SFB",
            "SFD": "SFD",
            "alpha": "alpha",
            "grid_m": "grid_m",
            "sfd_stride": "sfd_stride",
        },
        "module": "tart.fp8_einsum_bhr_hdr_bhd_fp8.sm100_sm103.deepgemm_fp8_batched_gemm_swap_ab",
        "workspace_rule": None,
    },
    "cake_deepgemm_batched_gemm_8b30436f526588de8d13": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "SFD"],
            ["parameter", "M"],
            ["parameter", "grid_m"],
            ["parameter", "sfd_stride"],
            ["parameter", "alpha"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {},
        "ffi_entry": "run",
        "grid_rule": {
            "block_m": 128,
            "kind": "one_cta_per_sm",
        },
        "kernel": "cake_deepgemm_batched_gemm_8b30436f526588de8d13",
        "kernel_symbol": "kernel_cake_deepgemm_batched_gemm_8b30436f526588de8d13",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 204544,
            "use_pdl": False,
        },
        "library": "cake_deepgemm_batched_gemm_8b30436f526588de8d13",
        "logical": {
            "A": "A",
            "B": "B",
            "D": "D",
            "M": "M",
            "SFA": "SFA",
            "SFB": "SFB",
            "SFD": "SFD",
            "alpha": "alpha",
            "grid_m": "grid_m",
            "sfd_stride": "sfd_stride",
        },
        "module": "tart.fp8_einsum_bhr_hdr_bhd_fp8.sm100_sm103.deepgemm_fp8_batched_gemm_n256",
        "workspace_rule": None,
    },
    "cake_deepgemm_batched_gemm_94affe3b74cd7c5a6422": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "SFD"],
            ["parameter", "M"],
            ["parameter", "grid_m"],
            ["parameter", "sfd_stride"],
            ["parameter", "alpha"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {},
        "ffi_entry": "run",
        "grid_rule": {
            "block_m": 64,
            "kind": "one_cta_per_sm",
        },
        "kernel": "cake_deepgemm_batched_gemm_94affe3b74cd7c5a6422",
        "kernel_symbol": "kernel_cake_deepgemm_batched_gemm_94affe3b74cd7c5a6422",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 223616,
            "use_pdl": False,
        },
        "library": "cake_deepgemm_batched_gemm_94affe3b74cd7c5a6422",
        "logical": {
            "A": "A",
            "B": "B",
            "D": "D",
            "M": "M",
            "SFA": "SFA",
            "SFB": "SFB",
            "SFD": "SFD",
            "alpha": "alpha",
            "grid_m": "grid_m",
            "sfd_stride": "sfd_stride",
        },
        "module": "tart.fp8_einsum_bhr_hdr_bhd_bf16.sm100_sm103.deepgemm_fp8_batched_gemm_bf16",
        "workspace_rule": None,
    },
    "cake_deepgemm_batched_gemm_d9a9c12ba43e831e0a02": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "SFD"],
            ["parameter", "M"],
            ["parameter", "grid_m"],
            ["parameter", "sfd_stride"],
            ["parameter", "alpha"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {},
        "ffi_entry": "run",
        "grid_rule": {
            "block_m": 16,
            "kind": "one_cta_per_sm",
        },
        "kernel": "cake_deepgemm_batched_gemm_d9a9c12ba43e831e0a02",
        "kernel_symbol": "kernel_cake_deepgemm_batched_gemm_d9a9c12ba43e831e0a02",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 229760,
            "use_pdl": False,
        },
        "library": "cake_deepgemm_batched_gemm_d9a9c12ba43e831e0a02",
        "logical": {
            "A": "A",
            "B": "B",
            "D": "D",
            "M": "M",
            "SFA": "SFA",
            "SFB": "SFB",
            "SFD": "SFD",
            "alpha": "alpha",
            "grid_m": "grid_m",
            "sfd_stride": "sfd_stride",
        },
        "module": "tart.fp8_einsum_bhr_hdr_bhd_bf16.sm100_sm103.deepgemm_fp8_batched_gemm_bf16_t4",
        "workspace_rule": None,
    },
    "cake_deepgemm_batched_gemm_e8aa7f5b9aa0575db281": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "SFA"],
            ["buffer", "SFB"],
            ["buffer", "D"],
            ["buffer", "SFD"],
            ["parameter", "M"],
            ["parameter", "grid_m"],
            ["parameter", "sfd_stride"],
            ["parameter", "alpha"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "block_m": 16,
            "inner": 4096,
            "num_heads": 8,
            "num_stages": 12,
            "width": 1024,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "block_m": 16,
            "block_n": 128,
            "kind": "working_ctas_even",
            "num_heads": 8,
            "width": 1024,
        },
        "kernel": "cake_deepgemm_batched_gemm_e8aa7f5b9aa0575db281",
        "kernel_symbol": "kernel_cake_deepgemm_batched_gemm_e8aa7f5b9aa0575db281",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 225664,
            "use_pdl": False,
        },
        "library": "cake_deepgemm_batched_gemm_e8aa7f5b9aa0575db281",
        "logical": {
            "A": "A",
            "B": "B",
            "D": "D",
            "M": "M",
            "SFA": "SFA",
            "SFB": "SFB",
            "SFD": "SFD",
            "alpha": "alpha",
            "grid_m": "grid_m",
            "sfd_stride": "sfd_stride",
        },
        "module": "tart.fp8_einsum_bhr_hdr_bhd_fp8.sm100_sm103.deepgemm_fp8_batched_gemm_swap_ab",
        "workspace_rule": None,
    },
}

# Schedule key (route, tile, stage count, epilogue; ``batched_gemm.schedule_key``)
# -> program.  Host plan: a route the dispatcher selects that is not a key here
# is not exported (``NotImplementedError`` at preparation).
ROUTES: dict[str, str] = {
    "bf16_t128_bm64_bn128_s10_bf16": "cake_deepgemm_batched_gemm_94affe3b74cd7c5a6422",
    "bf16_t4_bm16_bn128_s12_bf16": "cake_deepgemm_batched_gemm_d9a9c12ba43e831e0a02",
    "general_bm128_bn128_s5_alpha": "cake_deepgemm_batched_gemm_1931777936a4c257b75f",
    "n256_bm128_bn256_s5_fp8": "cake_deepgemm_batched_gemm_8b30436f526588de8d13",
    "swap_ab_bm16_bn128_s12_fp8": "cake_deepgemm_batched_gemm_e8aa7f5b9aa0575db281",
    "swap_ab_bm64_bn128_s10_fp8": "cake_deepgemm_batched_gemm_7e76dfdd9b10af2f8359",
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
            f"Cake batched FP8 projection target; the exported architectures are "
            f"{ARCHES}"
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
            f"Cake batched FP8 projection record {name} is not delivered for {arch}; "
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


def launch_grid(
    record: dict[str, Any], *, tokens: int, sm_count: int
) -> tuple[int, int, int]:
    """Launch grid of one record for ``tokens`` rows on a device of ``sm_count``
    SMs under its grid rule: one persistent CTA per SM (``one_cta_per_sm``), or
    the working CTAs of a swapped schedule, an even count (``working_ctas_even``:
    ``min(sm_count, tiles + tiles % 2)`` over ``ceil(tokens / block_m) x (width /
    block_n) x num_heads`` tiles)."""
    rule = record["grid_rule"]
    kind = rule["kind"]
    if kind not in (ONE_CTA_PER_SM, WORKING_CTAS_EVEN):
        raise ValueError(f"unknown Cake batched FP8 projection grid rule: {kind!r}")
    tokens, sm_count = int(tokens), int(sm_count)
    if tokens <= 0 or sm_count <= 0:
        raise ValueError(
            "the grid rule needs positive tokens and SMs, "
            f"got tokens={tokens}, sm_count={sm_count}"
        )
    if kind == ONE_CTA_PER_SM:
        return (sm_count, 1, 1)
    block_m, block_n = int(rule["block_m"]), int(rule["block_n"])
    tiles = (
        -(-tokens // block_m) * (int(rule["width"]) // block_n) * int(rule["num_heads"])
    )
    return (min(sm_count, tiles + tiles % 2), 1, 1)


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
    record's :func:`launch_grid`) on the current torch stream of the device."""
    record = MODULES[name]
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*launch_args(record, values, grid))


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "MODULES",
    "ONE_CTA_PER_SM",
    "ROUTES",
    "WORKING_CTAS_EVEN",
    "arch_for_capability",
    "device_arch",
    "launch_args",
    "launch_grid",
    "library_name",
    "load",
    "run_module",
    "sm_count",
    "specs_for_arch",
]
