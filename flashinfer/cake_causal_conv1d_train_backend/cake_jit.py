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

Host plan and loader glue of the Cake packed width-4 depthwise causal
convolution + SiLU training kernels (SM100 / SM103 / SM107).

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture.  ``MODULES`` and ``ROUTES`` carry the host plan only: the
kernel's ``library`` stem, the architectures it is delivered for, its launch
record, grid rule, workspace rule and the kernel-parameter -> logical-argument
map.  What to compile -- sources, digests, nvcc flags -- is recorded once in
``build.json`` next to this file; :mod:`.cake_build` serves the library
``<library>_<arch>`` from a prebuilt ``aot/`` file when a builder compiled it
and otherwise through FlashInfer's JIT from the same sources and flags,
verifying every source digest before the first build.

A route is the module record one public call launches: ``fwd`` (``y`` from
``x``, ``weight`` and ``cu_seqlens``) and ``bwd`` (``dx`` and the deterministic
FP32 ``dweight`` in one cooperative launch).  Token and channel counts are
runtime scalars, so a changed problem size never recompiles anything.

The host plan is the three functions :func:`plan` (the kernel scalars),
:func:`launch_grid` (the grid) and :func:`partial_rows` (the backward
workspace rows); everything else derives from them.

The program name ``cake_causal_conv1d_train_<20 hex>`` is the first twenty hex
digits of the SHA-256 of the compact JSON ``{"build": <build keywords, sorted>,
"module": <dotted kernel module>}`` of the kernel the program was rendered
from; the forward is built with its kernel file's default keywords; the
backward with ``vec=8`` (eight channels, 16-byte vectors, per thread) on
sm_100a and sm_103a, and with ``vec=8, min_blocks=2`` (two CTAs per SM) on
sm_107a.
"""

from __future__ import annotations

import functools
from typing import Any, Sequence

import tvm_ffi

from . import cake_build

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
MODULES: dict[str, dict[str, Any]] = {
    "cake_causal_conv1d_train_f193c5247980992f4f81": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        # The grid is the co-resident CTA count, ``ctas_per_sm x SM count``:
        # CTA ``b`` owns channel tile ``b % n_ch_tiles`` and, with the other
        # CTAs of that tile, an equal contiguous share of the token axis.  Any
        # positive grid is correct; ``ctas_per_sm`` is the measured
        # co-residency of the program: 4 CTAs per SM on B200 (sm_100a, 107
        # registers) and GB300 (sm_103a, 109 registers), the register limit
        # at 128 threads (ncu launch__occupancy_limit_registers); sm_107a is
        # taken as the same launch bound.  ``geometry`` is the kernel's build
        # geometry the host plan derives its scalars from.
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_f193c5247980992f4f81",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_f193c5247980992f4f81",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 33792,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_f193c5247980992f4f81",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "workspace_rule": None,
    },
    "cake_causal_conv1d_train_a578fae272376f2d9537": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw"],
            ["buffer", "partial"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        # Cooperative (a grid-wide barrier precedes the fixed-order dweight
        # reduction): the grid is exactly the co-resident CTA count.
        # ``ctas_per_sm``: 3 CTAs per SM (168 registers at 128 threads, the
        # launch bound; ncu launch__occupancy_limit_registers), the occupancy
        # query's answer on B200 and GB300.
        "grid_rule": {
            "cooperative": True,
            "ctas_per_sm": 3,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_a578fae272376f2d9537",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_a578fae272376f2d9537",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 50176,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_a578fae272376f2d9537",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
        },
        "role": "bwd",
        # The FP32 ``partial`` workspace: ``[ceil(grid / n_ch_tiles), C, taps]``
        # rows (row ``b // n_ch_tiles`` belongs to CTA ``b``) the kernel fully
        # writes and reduces in a fixed order; nothing is zeroed by the host.
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 4,
        },
    },
    "cake_causal_conv1d_train_a93b636beb1d573942bf": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw"],
            ["buffer", "partial"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        # Cooperative (a grid-wide barrier precedes the fixed-order dweight
        # reduction): the grid is exactly the co-resident CTA count.
        # ``ctas_per_sm``: 2 CTAs per SM, the program's ``min_blocks`` 2 launch
        # bound (234 registers at 128 threads; ncu
        # launch__occupancy_limit_registers), the occupancy query's answer on
        # R200; the ``min_blocks`` 3 build of the other two architectures
        # measured 10 % slower at C = 128 on its 212 SMs.
        "grid_rule": {
            "cooperative": True,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_a93b636beb1d573942bf",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_a93b636beb1d573942bf",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 50176,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_a93b636beb1d573942bf",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
        },
        "role": "bwd",
        # The FP32 ``partial`` workspace: ``[ceil(grid / n_ch_tiles), C, taps]``
        # rows (row ``b // n_ch_tiles`` belongs to CTA ``b``) the kernel fully
        # writes and reduces in a fixed order; nothing is zeroed by the host.
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 4,
        },
    },
}

ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "bwd": [
            "cake_causal_conv1d_train_a578fae272376f2d9537",
        ],
        "fwd": [
            "cake_causal_conv1d_train_f193c5247980992f4f81",
        ],
    },
    "sm_103a": {
        "bwd": [
            "cake_causal_conv1d_train_a578fae272376f2d9537",
        ],
        "fwd": [
            "cake_causal_conv1d_train_f193c5247980992f4f81",
        ],
    },
    "sm_107a": {
        "bwd": [
            "cake_causal_conv1d_train_a93b636beb1d573942bf",
        ],
        "fwd": [
            "cake_causal_conv1d_train_f193c5247980992f4f81",
        ],
    },
}

# One verified build per architecture: B200 (``sm_100a``), B300 (``sm_103a``)
# and R200 (``sm_107a``).  Every architecture is compiled with its exact
# ``-gencode`` flag set (``build.json`` ``arch_flags``); there is no family
# fallback.
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
KINDS = ("fwd", "bwd")
# The convolution width: ``weight`` and ``dweight`` are ``[C, WIDTH]`` and tap
# ``WIDTH - 1`` multiplies the current token.
WIDTH = 4
# A forward thread and a backward thread each own one 16-byte vector of
# eight BF16 channels, so ``C`` is a multiple of eight; the ``vec`` of every
# record's geometry divides this.
CHANNEL_MULTIPLE = 8
# The kernels take ``S``, ``C`` and the plan scalars as 32-bit integers.
INT32_MAX = 2**31 - 1
# Logical argument names the public API binds; every module record maps its
# kernel parameter names onto this vocabulary (``record["logical"]``).
LOGICAL_TENSORS = (
    "x",
    "w",
    "cu",
    "y",
    "dy",
    "dx",
    "dw",
    "partial",
)
LOGICAL_SCALARS = (
    "S",
    "C",
    "num_docs",
    "search_steps",
    "ct_vec",
    "n_sub",
    "n_ch_tiles",
)


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""

    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def route_applies(*, device_capability: Sequence[int], kind: str) -> bool:
    """Whether the verified export owns one training call on a device."""

    arch = arch_for_capability(device_capability)
    if arch is None:
        return False
    return kind in ROUTES.get(arch, {})


def route_modules(arch: str, kind: str) -> tuple[str, ...]:
    """Ordered module names of one route on ``arch`` (every program the route launches)."""

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    names = ROUTES.get(arch, {}).get(kind)
    if not names:
        raise ValueError(
            f"unsupported Cake causal conv1d training route {kind!r} on {arch!r}; "
            f"exported architectures: {ARCHES}"
        )
    return tuple(names)


def workspace_record(arch: str) -> dict[str, Any]:
    """The backward module record that owns the ``partial`` workspace on ``arch``."""

    owners = [
        MODULES[name]
        for name in route_modules(arch, "bwd")
        if MODULES[name]["workspace_rule"] is not None
    ]
    if len(owners) != 1:
        raise RuntimeError(
            f"the exported backward route on {arch} declares {len(owners)} workspace "
            "owners; expected exactly one"
        )
    return owners[0]


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-int(numerator) // int(denominator))


def channel_split(C: int, *, geometry: dict[str, int]) -> tuple[int, int, int]:
    """``(ct_vec, n_sub, n_ch_tiles)`` of ``C`` channels under a record's build geometry.

    ``ct_vec = min(C / vec, threads)`` channel vectors per channel tile (one
    thread each), ``n_sub = threads // ct_vec`` token sub-tiles per CTA and
    ``n_ch_tiles = ceil(C / vec / ct_vec)`` channel tiles.
    """

    C = int(C)
    vec = int(geometry["vec"])
    threads = int(geometry["threads"])
    if C <= 0 or C % vec:
        raise ValueError(f"C={C} must be a positive multiple of {vec}")
    channel_vectors = C // vec
    ct_vec = min(channel_vectors, threads)
    return ct_vec, threads // ct_vec, _ceil_div(channel_vectors, ct_vec)


def plan(S: int, C: int, num_docs: int, *, geometry: dict[str, int]) -> dict[str, int]:
    """The kernel's launch scalars for one problem under a record's build geometry.

    ``geometry`` is ``record["grid_rule"]["geometry"]``: ``threads`` per CTA
    and ``vec`` BF16 channels per thread.  The scalars are ``S``, ``C``,
    ``num_docs``, :func:`channel_split`'s ``ct_vec`` / ``n_sub`` /
    ``n_ch_tiles`` and ``search_steps``, the number of halvings of the
    device-side binary search over ``cu_seqlens`` (``max(1, ceil(log2
    num_docs))``).  Every value is a pure function of ``(S, C, num_docs)`` and
    the geometry: the host never reads ``cu_seqlens``, and the kernel derives
    its token ranges from ``S`` and the grid.
    """

    S, C, num_docs = int(S), int(C), int(num_docs)
    if S < 0 or num_docs < 0:
        raise ValueError(f"S and num_docs must be non-negative, got {(S, num_docs)}")
    ct_vec, n_sub, n_ch_tiles = channel_split(C, geometry=geometry)
    return {
        "S": S,
        "C": C,
        "num_docs": num_docs,
        "search_steps": max(1, (num_docs - 1).bit_length()),
        "ct_vec": ct_vec,
        "n_sub": n_sub,
        "n_ch_tiles": n_ch_tiles,
    }


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""

    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def launch_grid(record: dict[str, Any], *, device_index: int) -> tuple[int, int, int]:
    """Launch grid of one module on ``device_index`` under its recorded grid rule.

    ``co_resident`` modules launch exactly ``ctas_per_sm x SM count`` CTAs: the
    co-resident CTA count of the program, which the cooperative backward never
    exceeds.  The grid depends on the device only, so the deterministic
    ``dweight`` order is a pure function of ``(S, C, device)`` and one
    workspace serves every token count.
    """

    rule = record["grid_rule"]
    if rule["kind"] != "co_resident":
        raise ValueError(
            f"unknown Cake causal conv1d training grid rule: {rule['kind']!r}"
        )
    return (max(1, int(rule["ctas_per_sm"]) * sm_count(device_index)), 1, 1)


def partial_rows(grid_x: int, n_ch_tiles: int) -> int:
    """Rows of the backward ``partial`` workspace for a grid: ``ceil(grid_x / n_ch_tiles)``.

    CTA ``b`` of the grid owns channel tile ``b % n_ch_tiles`` and partial row
    ``b // n_ch_tiles``; the kernel writes every row it reduces.
    """

    return _ceil_div(grid_x, n_ch_tiles)


def partial_bytes(record: dict[str, Any], C: int, rows: int) -> int:
    """Bytes of ``rows`` FP32 ``[C, taps]`` partial rows of the module's workspace rule."""

    rule = record["workspace_rule"]
    if rule is None or rule["kind"] != "partial_rows":
        raise ValueError(f"module {record['kernel_symbol']} owns no backward workspace")
    return 4 * int(rule["taps"]) * int(C) * int(rows)


def workspace_layout(
    record: dict[str, Any], C: int, *, device_index: int
) -> dict[str, int]:
    """Byte layout of the caller-owned backward workspace of one module for ``C`` channels.

    The workspace is one CUDA ``uint8`` buffer holding the FP32 ``[rows, C, taps]``
    ``dweight`` partials, ``rows = partial_rows(grid, n_ch_tiles)`` for the
    device's launch grid; it depends on ``C`` and the device only, so one
    buffer serves every token count.  The kernel fully writes the rows it
    reduces, so nothing is zeroed between launches or CUDA Graph replays.
    """

    grid_x = launch_grid(record, device_index=device_index)[0]
    _, _, n_ch_tiles = channel_split(C, geometry=record["grid_rule"]["geometry"])
    rows = partial_rows(grid_x, n_ch_tiles)
    return {
        "grid": grid_x,
        "rows": rows,
        "taps": int(record["workspace_rule"]["taps"]),
        "total_bytes": partial_bytes(record, C, rows),
    }


@functools.cache
def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an exported target)."""

    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported Cake causal conv1d training target"
        )
    return arch


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``: ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake causal conv1d training module {name} is not delivered for {arch}; "
            f"its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``.

    The prebuilt ``aot/`` library when a builder compiled it, else FlashInfer's
    JIT build of the same sources and flags (``cake_build.load_library``).
    """

    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up through FlashInfer's ``build_jit_specs``)."""

    return cake_build.jit_specs(arch)


def run_module(name: str, values: dict[str, Any], *, device_index: int) -> None:
    """Launch one exported module with logical argument ``values``.

    ``values`` maps the logical names in ``LOGICAL_TENSORS`` /
    ``LOGICAL_SCALARS`` to tensors and Python scalars; the module record maps
    its kernel parameter names onto that vocabulary.  The launch grid is the
    module's :func:`launch_grid` on ``device_index``, and the kernel runs on
    the current torch stream.
    """

    record = MODULES[name]
    logical = record["logical"]
    grid = launch_grid(record, device_index=device_index)
    grid_values = {"grid_x": grid[0], "grid_y": grid[1], "grid_z": grid[2]}
    args = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            args.append(grid_values[key])
            continue
        logical_name = logical[key]
        if logical_name not in values:
            raise ValueError(
                f"module {name} needs {logical_name!r} (kernel parameter {key!r})"
            )
        args.append(values[logical_name])
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*args)


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "CHANNEL_MULTIPLE",
    "INT32_MAX",
    "KINDS",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "ROUTES",
    "WIDTH",
    "arch_for_capability",
    "channel_split",
    "device_arch",
    "launch_grid",
    "library_name",
    "load",
    "partial_bytes",
    "partial_rows",
    "plan",
    "route_applies",
    "route_modules",
    "run_module",
    "sm_count",
    "specs_for_arch",
    "workspace_layout",
    "workspace_record",
]
