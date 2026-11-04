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

Host plan and loader glue of the Cake mHC training head: the stream mean of
the four residual streams and its gradient (hidden 4096, 5120 and 7168;
SM100 / SM103 / SM107).

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture.  ``MODULES`` carries the host plan only: the kernel's
``library`` stem, the architectures it is delivered for, its launch and grid
rule and the kernel-parameter -> logical-argument map.  What to compile --
sources, digests, nvcc flags -- is recorded once in ``build.json`` next to this
file; :mod:`.cake_build` serves the library ``<library>_<arch>`` from a
prebuilt ``aot/`` file when a builder compiled it and otherwise through
FlashInfer's JIT from the same sources and flags, verifying every source
digest before the first build.

Each kernel moves one vector of ``vector_elems`` BF16 values per thread on a
plain (non-persistent) grid: ``ceil(T * hidden / vector_elems / threads)``
CTAs, the token and the column of a thread's vector found by one fast
division by ``hv = hidden / vector_elems`` (the kernel parameter ``hv``).  The
hidden size is therefore a launch argument: one program serves every hidden
size of :data:`HIDDEN_SIZES`, and the token count ``T`` is a runtime scalar
(``T == 0`` launches nothing).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

MODULES: dict[str, dict[str, Any]] = {
    "cake_mhc_det_train_head_fwd": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "y"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 256, "vector_elems": 8},
        "kernel": "cake_mhc_det_train_head_fwd",
        "kernel_symbol": "kernel_cake_mhc_det_train_head_fwd",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_det_train_head_fwd",
        "logical": {
            "T": "rows",
            "hv": "hv",
            "x": "x",
            "y": "y",
        },
        "role": "head_fwd",
    },
    "cake_mhc_det_train_head_bwd": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 256, "vector_elems": 16},
        "kernel": "cake_mhc_det_train_head_bwd",
        "kernel_symbol": "kernel_cake_mhc_det_train_head_bwd",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_det_train_head_bwd",
        "logical": {
            "T": "rows",
            "dx": "dx",
            "dy": "dy",
            "hv": "hv",
        },
        "role": "head_bwd",
    },
    "cake_mhc_det_train_head_bwd__acc": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["parameter", "T"],
            ["parameter", "hv"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "vectors", "threads": 256, "vector_elems": 16},
        "kernel": "cake_mhc_det_train_head_bwd__acc",
        "kernel_symbol": "kernel_cake_mhc_det_train_head_bwd__acc",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_mhc_det_train_head_bwd__acc",
        "logical": {
            "T": "rows",
            "dx": "dx",
            "dy": "dy",
            "hv": "hv",
        },
        "role": "head_bwd_acc",
    },
}
# The module record of each public call: the forward, the backward that writes
# ``dX`` and the backward that adds into it (``accumulate=True``).
ROLES: dict[str, str] = {
    "head_fwd": "cake_mhc_det_train_head_fwd",
    "head_bwd": "cake_mhc_det_train_head_bwd",
    "head_bwd_acc": "cake_mhc_det_train_head_bwd__acc",
}

# One verified build per architecture: B200 (``sm_100a``), B300 (``sm_103a``)
# and R200 (``sm_107a``); every architecture is compiled with its exact
# ``-gencode`` flag set (``build.json`` ``arch_flags``).
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
STREAMS = 4
# The hidden sizes each architecture serves (``hidden`` is a launch argument of
# every program).  SM107 serves 4096, as the rest of the package does, until
# its 5120 / 7168 correctness and timing are recorded on R200.
HIDDEN_SIZES: dict[str, tuple[int, ...]] = {
    "sm_100a": (4096, 5120, 7168),
    "sm_103a": (4096, 5120, 7168),
    "sm_107a": (4096,),
}
# The kernels' token bound (``T * hidden / vector_elems`` vectors are indexed in
# 32 bits; the kernels state ``T <= MAX_TOKENS`` as a launch precondition).
MAX_TOKENS = (1 << 22) - 1


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-int(numerator) // int(denominator))


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""

    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def supported_hidden_sizes(arch: str) -> tuple[int, ...]:
    """Hidden sizes the head serves on ``arch`` (empty for an architecture it does not ship)."""

    return HIDDEN_SIZES.get(arch, ())


def vectors_per_token(record: dict[str, Any], hidden: int) -> int:
    """``hv``: the vectors of one token row of ``hidden`` values (the kernels' fast divisor)."""

    elems = int(record["grid_rule"]["vector_elems"])
    if hidden % elems:
        raise ValueError(
            f"hidden {hidden} is not a multiple of the {elems}-value vector"
        )
    return hidden // elems


def launch_grid(record: dict[str, Any], rows: int, hidden: int) -> tuple[int, int, int]:
    """Launch grid of one module for ``rows`` tokens of ``hidden`` values: one
    vector per thread, ``ceil(rows * hv / threads)`` CTAs.  ``rows`` must be
    positive; the public API returns empty outputs for zero rows without
    launching."""

    rows = int(rows)
    if rows <= 0:
        raise ValueError("rows must be positive")
    rule = record["grid_rule"]
    if rule["kind"] != "vectors":
        raise ValueError(f"unknown Cake mHC training head grid rule: {rule['kind']!r}")
    return (
        _ceil_div(rows * vectors_per_token(record, hidden), int(rule["threads"])),
        1,
        1,
    )


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``: ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake mHC training head module {name} is not delivered for {arch}; its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``.

    The prebuilt ``aot/`` library when a builder compiled it, else FlashInfer's
    JIT build of the same sources and flags (``cake_build.load_library``).
    """

    from . import cake_build

    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up through FlashInfer's ``build_jit_specs``)."""

    from . import cake_build

    return cake_build.jit_specs(arch)
