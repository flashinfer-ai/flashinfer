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
launch record, its grid rule, the backward's workspace and the
kernel-parameter -> logical-argument map.  What to compile -- sources,
digests, nvcc flags -- is recorded once in ``build.json`` next to this file;
:mod:`.cake_build` serves the library ``<library>_<arch>`` from a prebuilt
``aot/`` file when a builder compiled it and otherwise through FlashInfer's
JIT from the same sources and flags, verifying every source digest before the
first build.

A role is one kernel of a call: ``fwd`` (the forward) and ``bwd_bf16`` /
``bwd_f32`` (the backward: ``dx``, ``dz`` and the weight gradient in one
kernel, the weight gradient rounded once to FP32, then to BF16 for a BF16
weight gradient).
``ROUTES[arch][role]`` names the one module a role runs on an architecture,
whatever the row count (:func:`module_for_rows`).  An architecture may also
route ``fwd_packed`` (``PACKED_ROLES``): a forward that addresses the rows of
contiguous ``[T, H, 128]`` inputs as one packed array, without the token and
head strides.  :func:`forward_module` runs it for such inputs and ``fwd``
otherwise; both compute a row with the same FP32 operations in the same order,
so ``y`` is the same bits on either route.

Grids (:func:`launch_grid`): the forward launches one CTA per block of
``rows_per_block`` rows below ``persistent_from_rows_per_sm`` rows per SM, and
from there the fewest whole waves of ``ctas_per_sm * SMs`` CTAs (the CTAs per
SM its launch bounds are compiled for) whose CTAs walk at most
``max_blocks_per_cta`` blocks each, and never more CTAs than blocks, CTA ``b``
walking the blocks ``b``, ``b + grid``, ``b + 2 * grid``, ... and loading each
next block while it computes the current one.  The backward launches
``min(blocks, ctas_per_sm * SMs)`` CTAs of ``rows_per_block``-row blocks; past
one block per CTA its CTAs claim the blocks in chunks whose rows are fixed by
the row count and the grid, and add each chunk's weight-gradient terms exactly
into fixed-point digits, so the weight gradient's bits depend on the rows and
the SM count only, not on which CTA runs a chunk.  The backward's workspace
(``workspace`` of its record: ``slots`` accumulator copies of ``limbs``
64-bit digits per column, then one 32-bit flag word per column and the two
32-bit words of the arrival ticket and the claim counter) starts zero and the
kernel leaves it zero.  No module is cooperative, and none waits for another
CTA or grid inside a kernel.  A kernel resident on another stream delays a
call roughly by its share of the SMs; a forward CTA whose slot such a kernel
holds walks all of its blocks once it starts, and a backward CTA that starts
late claims only the chunks left.

The program name ``cake_rmsnorm_silu_gate_det_train_<20 hex>`` is the first twenty
hex digits of the SHA-256 of the compact JSON ``{"build": <build keywords,
resolved and sorted>, "module": <dotted kernel module>}`` of the kernel the
program was rendered from (a forward keyword left at its default is not
spelled, so a program keeps its name when the kernel gains a keyword); each
record's comment names its build.
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
    # The forward: 128 threads, 8 lanes per row (16-row blocks), at most 6
    # CTAs per SM, each loading its next block while it computes the current
    # one, with the L2 prefetch of its next block and of the first block of
    # the CTA a wave later, the gate and rstd by the approximate instructions
    # (build {"min_blocks": 6, "prefetch_ahead": 1, "prefetch_wave": 1,
    # "threads": 128}); whole waves of at most 2 blocks per CTA.
    "cake_rmsnorm_silu_gate_det_train_36610c51a93167071e05": {
        "arches": [
            "sm_100a",
            "sm_103a",
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
            "ctas_per_sm": 6,
            "kind": "persistent_row_blocks",
            "max_blocks_per_cta": 2,
            "persistent_from_rows_per_sm": 8192,
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_36610c51a93167071e05",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_36610c51a93167071e05",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_36610c51a93167071e05",
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
    # The sm_107a forward: the schedule above with the sum of squares as four
    # FMA chains, rsqrt.approx.ftz and w loaded behind the first block's loads
    # (build {"min_blocks": 6, "prefetch_ahead": 1, "prefetch_wave": 1,
    # "rsqrt_ftz": 1, "sum_pairs": 1, "threads": 128, "w_late": 1}); whole
    # waves of at most 2 blocks per CTA.
    "cake_rmsnorm_silu_gate_det_train_5120481dd90c794f29e4": {
        "arches": [
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
            "ctas_per_sm": 6,
            "kind": "persistent_row_blocks",
            "max_blocks_per_cta": 2,
            "persistent_from_rows_per_sm": 8192,
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_5120481dd90c794f29e4",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_5120481dd90c794f29e4",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_5120481dd90c794f29e4",
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
    # The sm_107a forward of packed rows (contiguous ``[T, H, 128]`` x and z,
    # fewer than 2^24 rows): the forward above with each row's offset taken
    # from its index alone and one 32-byte store per lane (build
    # {"min_blocks": 6, "packed_rows": 1, "prefetch_ahead": 1,
    # "prefetch_wave": 1, "rsqrt_ftz": 1, "store_width": 256, "sum_pairs": 1,
    # "threads": 128, "w_late": 1}); whole waves of at most 2 blocks per CTA.
    "cake_rmsnorm_silu_gate_det_train_8a371a49c065b60c88a9": {
        "arches": [
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
            "ctas_per_sm": 6,
            "kind": "persistent_row_blocks",
            "max_blocks_per_cta": 2,
            "persistent_from_rows_per_sm": 8192,
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_8a371a49c065b60c88a9",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_8a371a49c065b60c88a9",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_8a371a49c065b60c88a9",
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
        "role": "fwd_packed",
    },
    # The backward, BF16 weight gradient: 256 threads, 16 lanes per row
    # (16-row blocks), at most 4 CTAs per SM, claimed chunks, the weight
    # gradient summed exactly in fixed point and rounded once (build
    # {"min_blocks": 4, "out": "bf16", "threads": 256}).
    "cake_rmsnorm_silu_gate_det_train_6e19a9f1b7cac40444cb": {
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
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "claimed_row_blocks",
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_6e19a9f1b7cac40444cb",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_6e19a9f1b7cac40444cb",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 26368,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_6e19a9f1b7cac40444cb",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16",
        "workspace": {
            "limbs": 9,
            "slots": 2,
        },
    },
    # The backward, FP32 weight gradient: 256 threads, 16 lanes per row
    # (16-row blocks), at most 4 CTAs per SM, claimed chunks, the weight
    # gradient summed exactly in fixed point and rounded once (build
    # {"min_blocks": 4, "out": "f32", "threads": 256}).
    "cake_rmsnorm_silu_gate_det_train_6b4739305d767297d565": {
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
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "claimed_row_blocks",
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_6b4739305d767297d565",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_6b4739305d767297d565",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 26368,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_6b4739305d767297d565",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_f32",
        "workspace": {
            "limbs": 9,
            "slots": 2,
        },
    },
}

ROUTES: dict[str, dict[str, str]] = {
    "sm_100a": {
        "fwd": "cake_rmsnorm_silu_gate_det_train_36610c51a93167071e05",
        "bwd_bf16": "cake_rmsnorm_silu_gate_det_train_6e19a9f1b7cac40444cb",
        "bwd_f32": "cake_rmsnorm_silu_gate_det_train_6b4739305d767297d565",
    },
    "sm_103a": {
        "fwd": "cake_rmsnorm_silu_gate_det_train_36610c51a93167071e05",
        "bwd_bf16": "cake_rmsnorm_silu_gate_det_train_6e19a9f1b7cac40444cb",
        "bwd_f32": "cake_rmsnorm_silu_gate_det_train_6b4739305d767297d565",
    },
    "sm_107a": {
        "fwd": "cake_rmsnorm_silu_gate_det_train_5120481dd90c794f29e4",
        "fwd_packed": "cake_rmsnorm_silu_gate_det_train_8a371a49c065b60c88a9",
        "bwd_bf16": "cake_rmsnorm_silu_gate_det_train_6e19a9f1b7cac40444cb",
        "bwd_f32": "cake_rmsnorm_silu_gate_det_train_6b4739305d767297d565",
    },
}

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
# The kernels of a call, routed on every architecture.
ROLES = ("fwd", "bwd_bf16", "bwd_f32")
# The forwards of packed rows, routed on some architectures beside ``fwd``.
PACKED_ROLES = ("fwd_packed",)
GRID_KINDS = ("persistent_row_blocks", "claimed_row_blocks")
# Elements of one normalized row.
HEAD_DIM = 128
# The forward reads rows with 32-byte vector accesses (the backward with
# 16-byte ones): a tensor is read in place when its storage offset and its
# token and head strides are multiples of this many BF16 elements.
ALIGNMENT_ELEMENTS = 16
# The kernels index rows with 32-bit integers.
INT32_MAX = 2**31 - 1
# A forward of packed rows takes the offset of an element of row ``n`` as
# ``n * 128 + column`` in 32-bit arithmetic: it serves fewer rows than this.
PACKED_ROWS = 2**31 // HEAD_DIM


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


def module_for_rows(arch: str, role: str, rows: int, sm_count: int) -> str:
    """The module that runs ``role`` (one of ``ROLES``) for ``rows`` rows on
    ``arch`` with ``sm_count`` SMs: :func:`route_module`, one module whatever
    the rows."""

    return route_module(arch, role)


def forward_module(
    arch: str,
    rows: int,
    heads: int,
    sm_count: int,
    x_tok: int,
    x_head: int,
    z_tok: int,
    z_head: int,
) -> str:
    """The forward module for ``rows`` rows of ``heads`` heads on ``arch`` with
    ``sm_count`` SMs, given the token and head strides (in elements) the
    kernel reads ``x`` and ``z`` with: the arch's ``fwd_packed`` module when it
    routes one and the rows are packed (both head strides 128, both token
    strides ``heads * 128``, fewer than ``PACKED_ROWS`` rows), else
    :func:`module_for_rows` of ``fwd``.  Both give the same bits."""

    packed = ROUTES.get(arch, {}).get("fwd_packed")
    row = heads * HEAD_DIM
    if (
        packed is not None
        and 0 < rows < PACKED_ROWS
        and x_head == HEAD_DIM
        and z_head == HEAD_DIM
        and x_tok == row
        and z_tok == row
    ):
        return packed
    return module_for_rows(arch, "fwd", rows, sm_count)


def workspace_elements(name: str) -> int:
    """FP32 elements of the workspace of backward module ``name``: its
    accumulator (``slots * limbs * HEAD_DIM`` 64-bit digits, two FP32 elements
    each), one 32-bit flag word per column, and the arrival ticket and the
    claim counter."""

    shape = MODULES[name]["workspace"]
    return 2 * int(shape["slots"]) * int(shape["limbs"]) * HEAD_DIM + HEAD_DIM + 2


def launch_grid(
    record: dict[str, Any], *, rows: int, sm_count: int
) -> tuple[int, int, int]:
    """The grid of one module record for ``rows`` rows on a device with
    ``sm_count`` SMs.  The forward launches one CTA per block of
    ``rows_per_block`` rows below ``persistent_from_rows_per_sm`` rows per SM,
    and from there the fewest whole waves of ``ctas_per_sm * sm_count`` CTAs
    whose CTAs walk at most ``max_blocks_per_cta`` blocks each, and at most one
    CTA per block; the backward ``min(blocks, ctas_per_sm * sm_count)`` CTAs."""

    rule = record["grid_rule"]
    kind = rule["kind"]
    blocks = -(-int(rows) // int(rule["rows_per_block"]))
    wave = max(1, int(rule["ctas_per_sm"]) * int(sm_count))
    if kind == "persistent_row_blocks":
        if int(rows) < int(rule["persistent_from_rows_per_sm"]) * int(sm_count):
            return (blocks, 1, 1)
        waves = max(1, -(-blocks // (wave * int(rule["max_blocks_per_cta"]))))
        return (min(blocks, waves * wave), 1, 1)
    if kind == "claimed_row_blocks":
        return (min(blocks, wave), 1, 1)
    raise ValueError(
        f"unknown Cake SiLU-gated RMSNorm grid rule: {kind!r} (known: {GRID_KINDS})"
    )


@functools.cache
def device_sm_count(device_index: int) -> int:
    """SM count of one CUDA device, read once per device (no device work)."""

    import torch

    return torch.cuda.get_device_properties(device_index).multi_processor_count


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
    grid is the module's :func:`launch_grid` on the device's SM count, and the
    kernel runs on the current torch stream.
    """

    record = MODULES[name]
    logical = record["logical"]
    grid = launch_grid(record, rows=rows, sm_count=device_sm_count(device_index))
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
    "MODULES",
    "PACKED_ROLES",
    "PACKED_ROWS",
    "ROLES",
    "ROUTES",
    "arch_for_capability",
    "device_arch",
    "device_sm_count",
    "forward_module",
    "launch_grid",
    "library_name",
    "load",
    "module_for_rows",
    "route_module",
    "run_module",
    "specs_for_arch",
    "workspace_elements",
]
