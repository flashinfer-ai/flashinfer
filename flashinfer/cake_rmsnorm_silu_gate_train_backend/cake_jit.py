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

Host plan and loader glue of the Cake SiLU-gated RMSNorm training kernels
(forward and backward; SM100 / SM103 / SM107).

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture.  ``MODULES`` and ``ROUTES`` carry the host plan only: the
kernel's ``library`` stem, the architectures it is delivered for, its role, its
launch record, its grid rule and the kernel-parameter -> logical-argument map.
What to compile -- sources, digests, nvcc flags -- is recorded once in
``build.json`` next to this file; :mod:`.cake_build` serves the library
``<library>_<arch>`` from a prebuilt ``aot/`` file when a builder compiled it
and otherwise through FlashInfer's JIT from the same sources and flags,
verifying every source digest before the first build.

A role is one kernel of a call: ``fwd`` (the forward), ``bwd`` (the
backward's main kernel: ``dx``, ``dz`` and one FP32 weight-gradient partial
row per CTA chunk) and ``dw_bf16`` / ``dw_f32`` (the fixed-order FP64 column
sum of the partials into a BF16 or FP32 weight gradient).  ``ROUTES[arch][role]``
names the one module a role runs on an architecture.

Grids (:func:`launch_grid`): the forward launches one CTA per block of
``rows_per_cta`` rows; the backward one CTA per chunk of ``blocks_per_cta``
such blocks, where :func:`backward_chunks` derives ``blocks_per_cta`` from the
row count alone (so the partial rows, and the bits of the weight gradient,
depend on ``T * H`` only, never on the device); the reduction a fixed grid of
128 CTAs of one column.  No module is persistent or cooperative, and none
waits for another grid inside a kernel, so a kernel resident on another
stream delays a call by its share of the SMs and never blocks it.

The program name ``cake_rmsnorm_silu_gate_train_<20 hex>`` is the first twenty
hex digits of the SHA-256 of the compact JSON ``{"build": <build keywords,
resolved and sorted>, "module": <dotted kernel module>}`` of the kernel the
program was rendered from; each record's comment names its build.
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
    # The forward: 128 threads, 8 lanes per row, 32-byte accesses, one row
    # per thread group, the gate on packed FP32 pairs (build {"lanes": 8,
    # "math": "x2", "pdl": false, "rows_per_thread": 1, "threads": 128,
    # "width": 256}).
    "cake_rmsnorm_silu_gate_train_3c1aaebdec47df0111e4": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "y"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_blocks",
            "rows_per_cta": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_3c1aaebdec47df0111e4",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_3c1aaebdec47df0111e4",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_3c1aaebdec47df0111e4",
        "logical": {
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "y": "y",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "fwd",
    },
    # The backward's main kernel: 128 threads, 16 lanes per row, 16-byte
    # accesses, one row per thread group, the gate on packed FP32 pairs (build
    # {"lanes": 16, "math": "x2", "pdl": false, "rows_per_thread": 1,
    # "threads": 128, "width": 128}).
    "cake_rmsnorm_silu_gate_train_aeceb63b55b0c3368dc9": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "partial"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "blocks_per_cta"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_chunks",
            "rows_per_cta": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_aeceb63b55b0c3368dc9",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_aeceb63b55b0c3368dc9",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4096,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_aeceb63b55b0c3368dc9",
        "logical": {
            "blocks_per_cta": "blocks_per_cta",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "heads": "heads",
            "partial": "partial",
            "rows": "rows",
            "w": "w",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd",
    },
    # The weight-gradient reduction into BF16: one column per CTA, 8 warps,
    # each thread summing every 256th partial row (build {"cols": 1, "out":
    # "bf16", "pdl": false, "slices": 8}).
    "cake_rmsnorm_silu_gate_train_9d6efc19ce4885c5a4d0": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "partial"],
            ["buffer", "dw"],
            ["parameter", "chunks"],
            ["parameter", "accumulate"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "kernel": "cake_rmsnorm_silu_gate_train_9d6efc19ce4885c5a4d0",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_9d6efc19ce4885c5a4d0",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2304,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_9d6efc19ce4885c5a4d0",
        "logical": {
            "accumulate": "accumulate",
            "chunks": "chunks",
            "dw": "dw",
            "partial": "partial",
        },
        "role": "dw_bf16",
    },
    # The weight-gradient reduction into FP32: one column per CTA, 8 warps,
    # each thread summing every 256th partial row (build {"cols": 1, "out":
    # "f32", "pdl": false, "slices": 8}).
    "cake_rmsnorm_silu_gate_train_47d2327da7f419fce725": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "partial"],
            ["buffer", "dw"],
            ["parameter", "chunks"],
            ["parameter", "accumulate"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "kernel": "cake_rmsnorm_silu_gate_train_47d2327da7f419fce725",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_47d2327da7f419fce725",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2304,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_47d2327da7f419fce725",
        "logical": {
            "accumulate": "accumulate",
            "chunks": "chunks",
            "dw": "dw",
            "partial": "partial",
        },
        "role": "dw_f32",
    },
}

ROUTES: dict[str, dict[str, str]] = {
    "sm_100a": {
        "fwd": "cake_rmsnorm_silu_gate_train_3c1aaebdec47df0111e4",
        "bwd": "cake_rmsnorm_silu_gate_train_aeceb63b55b0c3368dc9",
        "dw_bf16": "cake_rmsnorm_silu_gate_train_9d6efc19ce4885c5a4d0",
        "dw_f32": "cake_rmsnorm_silu_gate_train_47d2327da7f419fce725",
    },
    "sm_103a": {
        "fwd": "cake_rmsnorm_silu_gate_train_3c1aaebdec47df0111e4",
        "bwd": "cake_rmsnorm_silu_gate_train_aeceb63b55b0c3368dc9",
        "dw_bf16": "cake_rmsnorm_silu_gate_train_9d6efc19ce4885c5a4d0",
        "dw_f32": "cake_rmsnorm_silu_gate_train_47d2327da7f419fce725",
    },
    "sm_107a": {
        "fwd": "cake_rmsnorm_silu_gate_train_3c1aaebdec47df0111e4",
        "bwd": "cake_rmsnorm_silu_gate_train_aeceb63b55b0c3368dc9",
        "dw_bf16": "cake_rmsnorm_silu_gate_train_9d6efc19ce4885c5a4d0",
        "dw_f32": "cake_rmsnorm_silu_gate_train_47d2327da7f419fce725",
    },
}

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
ROLES = ("fwd", "bwd", "dw_bf16", "dw_f32")
GRID_KINDS = ("row_blocks", "row_chunks", "fixed")
# Elements of one normalized row.
HEAD_DIM = 128
# The forward reads rows with 32-byte vector accesses (the backward with
# 16-byte ones): a tensor is read in place
# when its storage offset and its token and head strides are multiples of this
# many BF16 elements.
ALIGNMENT_ELEMENTS = 16
# The kernels index rows with 32-bit integers.
INT32_MAX = 2**31 - 1
# The backward's chunk rule: one CTA per ``blocks_per_cta`` row blocks, the
# fewest blocks per CTA that keep the chunk count at most ``MAX_CHUNKS`` (the
# partial buffer then stays at most ``MAX_CHUNKS * 512`` bytes).
MAX_CHUNKS = 8192


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""

    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def route_module(arch: str, role: str) -> str:
    """The module of ``role`` on ``arch`` (``ValueError`` when none is exported)."""

    name = ROUTES.get(arch, {}).get(role)
    if name is None:
        raise ValueError(
            f"no exported Cake SiLU-gated RMSNorm {role!r} module on {arch!r}; "
            f"exported architectures: {ARCHES}"
        )
    return name


def backward_chunks(name: str, rows: int) -> tuple[int, int]:
    """``(chunks, blocks_per_cta)`` of the backward module ``name`` for ``rows``
    rows: the CTA chunks (the grid and the partial rows) and the row blocks of
    one chunk, a function of ``rows`` and the module's ``rows_per_cta`` only."""

    rule = MODULES[name]["grid_rule"]
    if rule["kind"] != "row_chunks":
        raise ValueError(f"module {name} is not a backward main module")
    blocks = -(-int(rows) // int(rule["rows_per_cta"]))
    blocks_per_cta = max(1, -(-blocks // MAX_CHUNKS))
    return -(-blocks // blocks_per_cta), blocks_per_cta


def launch_grid(
    record: dict[str, Any], *, rows: int, chunks: int = 0
) -> tuple[int, int, int]:
    """The grid of one module record for ``rows`` rows (``chunks``: the
    backward's chunk count, :func:`backward_chunks`)."""

    rule = record["grid_rule"]
    kind = rule["kind"]
    if kind == "row_blocks":
        return (-(-int(rows) // int(rule["rows_per_cta"])), 1, 1)
    if kind == "row_chunks":
        return (int(chunks), 1, 1)
    if kind == "fixed":
        return (int(rule["ctas"]), 1, 1)
    raise ValueError(
        f"unknown Cake SiLU-gated RMSNorm grid rule: {kind!r} (known: {GRID_KINDS})"
    )


@functools.cache
def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an exported target)."""

    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported "
            "Cake SiLU-gated RMSNorm target"
        )
    return arch


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``: ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake SiLU-gated RMSNorm module {name} is not delivered for {arch}; "
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


def run_module(
    name: str, values: dict[str, Any], *, rows: int, device_index: int
) -> None:
    """Launch one exported module with logical argument ``values``.

    ``values`` maps logical names to tensors and Python scalars; the module
    record maps its kernel parameter names onto that vocabulary.  The launch
    grid is the module's :func:`launch_grid` (the backward reads ``chunks``
    from ``values``), and the kernel runs on the current torch stream.
    """

    record = MODULES[name]
    logical = record["logical"]
    grid = launch_grid(record, rows=rows, chunks=int(values.get("chunks", 0)))
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
    "ALIGNMENT_ELEMENTS",
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "GRID_KINDS",
    "HEAD_DIM",
    "INT32_MAX",
    "MAX_CHUNKS",
    "MODULES",
    "ROLES",
    "ROUTES",
    "arch_for_capability",
    "backward_chunks",
    "device_arch",
    "launch_grid",
    "library_name",
    "load",
    "route_module",
    "run_module",
    "specs_for_arch",
]
