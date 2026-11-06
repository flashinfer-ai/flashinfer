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

Registry and loader glue of the Cake experimental MLA value projection +
sigmoid output gate programs (SM100 / SM103), the ``backend="cake"`` of
:func:`flashinfer.cake_mla_value_gate.prepare_mla_value_gate`.

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit plus a tvm-ffi binding, compiled once per
architecture.  A record carries the host plan: the program's ``library``
stem (its ``build.json`` entries are ``<library>_<arch>``), the architectures
it is delivered for, the source ``module`` and ``build`` keywords it was
rendered from, the sigmoid numerics it implements (``sigmoid``: ``"fp32"``,
the only mode the API offers), the ``ffi_entry`` it exports and the grid rule
:func:`launch_grid` evaluates.  An architecture may have one program per row
tile: the records of one ``(arch, sigmoid)`` differ in ``grid_rule["rows"]``
(16 or 8 token rows per CTA), the key the host selects on
(:func:`tile_rows`).  The derived leaves of a record
(``kernel_symbol``, ``launch.block``, ``launch.cluster``,
``launch.cooperative``, ``launch.dynamic_smem_bytes``, ``launch.use_pdl``,
``arg_plan`` and ``logical``) are rewritten by the exporter from the rendered
binding; ``launch.use_pdl`` is ``"runtime"`` when the entry takes the
``enable_pdl`` launch argument.  ``logical`` maps every buffer and parameter
of ``arg_plan`` to the host vocabulary ``LOGICAL_TENSORS`` /
``LOGICAL_SCALARS``.  Every value is a literal: FlashInfer's manifest checker
and the exporter read this file with ``ast.literal_eval``.

What to compile -- sources, digests, nvcc flags -- is recorded once in
``build.json`` next to this file; :mod:`.cake_build` serves the library from a
prebuilt ``aot/`` file when a builder compiled it and otherwise through
FlashInfer's JIT from the same sources and flags.

The program name ``cake_mla_value_gate_<20 hex>`` is the first twenty hex
digits of the SHA-256 of the compact JSON ``{"build": <build keywords, sorted,
every default spelled>, "module": <dotted source module>}`` of the kernel the
program was rendered from; a changed keyword is a new program.
"""

from __future__ import annotations

import functools
from typing import Any

# The exported architectures and the device capabilities that select them (exact match).
ARCHES = ("sm_100a", "sm_103a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}

# The value columns of one head (the ``N`` of the projection).
VALUE_DIM = 128

# The host vocabulary: the logical names ``cake_backend`` binds.  ``latent`` is the caller's
# contiguous ``[B, H, 512]`` tensor, ``w_uv`` the physical ``[H, 128, 512]`` view of the bound
# weight, ``gate`` the contiguous span of storage from ``gate[0, 0]`` to the last gate element,
# ``valid_lengths`` the ``[B // q]`` int32 lengths and ``out`` the caller's contiguous ``[B, H * 128]``
# output.  ``rows`` is ``B``, ``heads`` is ``H``, ``rows_per_request`` is ``q`` and ``gate_pitch`` is
# the gate's row pitch in BF16 elements.  ``enable_pdl`` (0 or 1) is the launch argument of an
# entry whose ``launch.use_pdl`` is ``"runtime"``.
LOGICAL_TENSORS = ("latent", "w_uv", "gate", "valid_lengths", "out")
LOGICAL_SCALARS = ("rows", "heads", "rows_per_request", "gate_pitch")
LAUNCH_ARGUMENTS = ("enable_pdl",)

# The exported programs, keyed by library.  The exporter inserts the records and rewrites
# their derived leaves; do not edit them by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_mla_value_gate_00ca8bb1c760493fa672": {
        "library": "cake_mla_value_gate_00ca8bb1c760493fa672",
        "kernel": "cake_mla_value_gate_00ca8bb1c760493fa672",
        "arches": ["sm_100a", "sm_103a"],
        "module": "tart.mla_value_proj_sigmoid_gate_bf16.sm100_sm103.mla_value_gate",
        "build": {},
        "sigmoid": "fp32",
        "ffi_entry": "run",
        "grid_rule": {"kind": "value_tiles", "cols": 32, "rows": 16},
        "kernel_symbol": "kernel_cake_mla_value_gate_00ca8bb1c760493fa672",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 17408,
            "use_pdl": "runtime",
        },
        "arg_plan": [
            ["buffer", "latent"],
            ["buffer", "w_uv"],
            ["buffer", "gate"],
            ["buffer", "valid_lengths"],
            ["buffer", "out"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "rows_per_request"],
            ["parameter", "gate_pitch"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "logical": {
            "gate": "gate",
            "gate_pitch": "gate_pitch",
            "heads": "heads",
            "latent": "latent",
            "out": "out",
            "rows": "rows",
            "rows_per_request": "rows_per_request",
            "valid_lengths": "valid_lengths",
            "w_uv": "w_uv",
        },
    },
    "cake_mla_value_gate_0df66a90a1dc776a3834": {
        "library": "cake_mla_value_gate_0df66a90a1dc776a3834",
        "kernel": "cake_mla_value_gate_0df66a90a1dc776a3834",
        "arches": ["sm_100a", "sm_103a"],
        "module": "tart.mla_value_proj_sigmoid_gate_bf16.sm100_sm103.mla_value_gate",
        "build": {"tile_rows": 8},
        "sigmoid": "fp32",
        "ffi_entry": "run",
        "grid_rule": {"kind": "value_tiles", "cols": 32, "rows": 8},
        "kernel_symbol": "kernel_cake_mla_value_gate_0df66a90a1dc776a3834",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8704,
            "use_pdl": "runtime",
        },
        "arg_plan": [
            ["buffer", "latent"],
            ["buffer", "w_uv"],
            ["buffer", "gate"],
            ["buffer", "valid_lengths"],
            ["buffer", "out"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "rows_per_request"],
            ["parameter", "gate_pitch"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "logical": {
            "gate": "gate",
            "gate_pitch": "gate_pitch",
            "heads": "heads",
            "latent": "latent",
            "out": "out",
            "rows": "rows",
            "rows_per_request": "rows_per_request",
            "valid_lengths": "valid_lengths",
            "w_uv": "w_uv",
        },
    },
}


def arch_for_capability(capability: tuple[int, int]) -> str | None:
    """The exported architecture of one device capability, if any."""
    values = tuple(int(value) for value in capability)
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


@functools.cache
def device_arch(device_index: int) -> str | None:
    """Exported architecture of one CUDA device, or ``None`` (the capability is immutable per process)."""
    import torch

    return arch_for_capability(tuple(torch.cuda.get_device_capability(device_index)))


@functools.cache
def device_sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""
    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def records_for(arch: str, sigmoid: str) -> list[str]:
    """Names of the records delivered for ``arch`` that implement ``sigmoid``, in registry order."""
    return [
        name
        for name, record in MODULES.items()
        if arch in record["arches"] and record["sigmoid"] == sigmoid
    ]


def tile_rows(record: dict[str, Any]) -> int:
    """Token rows per CTA of one record (its ``grid_rule["rows"]``): the key that tells the
    programs of one ``(arch, sigmoid)`` apart."""
    rule = record["grid_rule"]
    rows = rule.get("rows")
    if (
        rule.get("kind") != "value_tiles"
        or isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows <= 0
    ):
        raise RuntimeError(
            f"record {record['library']} has no positive value_tiles row tile: {rule}"
        )
    return rows


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake MLA value gate record {name} is not delivered for {arch}; "
            f"its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``: the
    prebuilt ``aot/`` library when a builder compiled it, else FlashInfer's JIT
    build of the same sources and flags (``cake_build.load_library``)."""
    from . import cake_build

    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up through FlashInfer's ``build_jit_specs``)."""
    from . import cake_build

    return cake_build.jit_specs(arch)


def launch_grid(
    record: dict[str, Any], *, rows: int, heads: int
) -> tuple[int, int, int]:
    """Launch grid of one record for ``rows`` token rows of ``heads`` heads under its grid rule:
    ``value_tiles`` -- one CTA per (``cols`` value columns, head, ``rows``-row tile):
    ``(VALUE_DIM // cols, heads, ceil(rows / rule rows))``."""
    rule = record["grid_rule"]
    if rule["kind"] != "value_tiles":
        raise RuntimeError(
            f"unknown Cake MLA value gate grid rule {rule['kind']!r} in record {record['library']}"
        )
    cols, tile_rows = int(rule["cols"]), int(rule["rows"])
    if cols <= 0 or VALUE_DIM % cols or tile_rows <= 0:
        raise RuntimeError(
            f"record {record['library']} has an invalid value_tiles rule {rule}"
        )
    return (VALUE_DIM // cols, int(heads), -(-int(rows) // tile_rows))


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "LAUNCH_ARGUMENTS",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "VALUE_DIM",
    "arch_for_capability",
    "device_arch",
    "device_sm_count",
    "launch_grid",
    "library_name",
    "load",
    "records_for",
    "specs_for_arch",
    "tile_rows",
]
