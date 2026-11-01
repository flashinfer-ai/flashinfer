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

Host plan and loader glue of the Cake SiLU-gated RMSNorm training kernels of
the non-deterministic package (forward and backward; SM100 / SM103 / SM107).

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture.  ``MODULES`` and ``ROUTES`` carry the host plan only: the
kernel's ``library`` stem, the architectures it is delivered for, its role, its
launch record, its grid rule, the weight-gradient accumulator slots of the
backward and the kernel-parameter -> logical-argument map.  What to compile --
sources, digests, nvcc flags -- is recorded once in ``build.json`` next to this
file; :mod:`.cake_build` serves the library ``<library>_<arch>`` from a
prebuilt ``aot/`` file when a builder compiled it and otherwise through
FlashInfer's JIT from the same sources and flags, verifying every source digest
before the first build.

A role is one kernel of a call: ``fwd`` (the forward, the deterministic
package's kernel: it has no reduction) and ``bwd_bf16`` / ``bwd_f32`` (the
backward: ``dx``, ``dz`` and the weight gradient in one kernel, each CTA adding
its FP64 column sums to ``dw_slots`` FP64 accumulators with atomics, the last
CTA to arrive rounding the sum once into a BF16 or FP32 weight gradient and
zeroing the accumulators).  ``ROUTES[arch][role]`` names the one module a role
runs on an architecture.  An architecture may also route a role's long role
(``fwd_long``, ``bwd_bf16_long``, ``bwd_f32_long``) and the forward's mid role
(``fwd_mid``), whose module runs in the role's place from ``min_rows_per_sm``
rows per SM of its grid rule (:func:`module_for_rows`; the long role first,
from more rows): on sm_100a and sm_103a a forward that walks several row
blocks per CTA, loading each next block while it computes the current one,
``y`` the same bits (the long forward 32-row blocks on 256-thread CTAs, the mid
forward 16-row blocks on 128-thread CTAs); on sm_107a the backward with
128-thread CTAs, ``dx`` and ``dz`` the same bits.

Grids (:func:`launch_grid`): the forward launches one CTA per block of
``rows_per_cta`` rows.  The long and mid forwards launch the fewest whole waves of
``ctas_per_sm * SMs`` CTAs (the CTAs per SM its launch bounds are compiled
for) whose CTAs walk at most ``max_blocks_per_cta`` blocks of
``rows_per_block`` rows each, and never more CTAs than blocks; CTA ``b``
walks the blocks ``b``, ``b + grid``, ``b + 2 * grid``, ...  The backward
launches :func:`backward_ctas` CTAs, one per block but at most
``ctas_per_sm`` per SM (one wave).  When the grid has
a CTA for every block, CTA ``c`` takes block ``c`` and claims nothing;
otherwise the CTAs claim every block from the claim counter of the
workspace, in chunks of consecutive blocks that shrink to single blocks at
the end, each CTA claiming its next chunk as the last block of its current
one starts, so a CTA holds one chunk at a time and one that starts late
takes fewer blocks.  The kernel folds each thread's FP32 sums into
FP64 after at most 16 of its rows (a CTA of a few blocks adds them across its
warps in FP32 first and converts once), so a term passes through at most 16
FP32 roundings at any grid: within the deterministic backward's error budget
on every input.  No
module is cooperative and no CTA waits for another inside a kernel: a kernel
resident on another stream delays a call roughly in proportion to the SM slots
it holds, and one that holds resources this kernel needs on every SM can still
delay its start.  A long or mid forward CTA whose slot such a kernel holds
walks all of its blocks once it starts.  The long forward runs only from the
``min_rows_per_sm`` at which, with at most ``max_blocks_per_cta`` blocks a
CTA, that delay is no larger than the forward's; the mid forward only from
the one at which such a call stays within the forward stall test's bound and
ends no later than the forward's (FlashInfer issue #7548).

The workspace of the backward is ``dw_slots * 128`` FP64 accumulators and one
8-byte word holding two 32-bit words, the arrival ticket and the claim counter
(:func:`workspace_elements` FP64 elements in all), zero when a launch starts
and zero again when it ends: a zero-filled workspace stays reusable for any
later call on its stream.  Every backward launch takes the ticket, and
claims blocks when the rows have more blocks than one wave, also one that
computes no weight gradient, so every backward launch needs the workspace.

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
    # The forward (the deterministic package's kernel file): 128 threads, 8
    # lanes per row, 32-byte accesses, one row per thread group, the gate on
    # packed FP32 pairs (build {"lanes": 8,
    # "math": "x2", "pdl": false, "rows_per_thread": 1, "threads": 128,
    # "width": 256}).
    "cake_rmsnorm_silu_gate_train_05ffc12001cf81513436": {
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
        "kernel": "cake_rmsnorm_silu_gate_train_05ffc12001cf81513436",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_05ffc12001cf81513436",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_05ffc12001cf81513436",
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
    # The long forward (sm_100a, sm_103a): 256 threads, 8 lanes per row
    # (32-row blocks), 32-byte accesses, at most 3 CTAs per SM, each loading
    # its next block while it computes the current one, the gate on packed
    # FP32 pairs (build {"lanes": 8, "math": "x2", "min_blocks": 3,
    # "threads": 256}); whole waves of at most 6 blocks per CTA, from 2400
    # rows per SM.
    "cake_rmsnorm_silu_gate_train_e0672c312c57421abe21": {
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
            "ctas_per_sm": 3,
            "kind": "persistent_row_blocks",
            "max_blocks_per_cta": 6,
            "min_rows_per_sm": 2400,
            "rows_per_block": 32,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_e0672c312c57421abe21",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_e0672c312c57421abe21",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_e0672c312c57421abe21",
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
        "role": "fwd_long",
    },
    # The mid forward (sm_100a, sm_103a): the long forward's kernel on 128
    # threads, 8 lanes per row (16-row blocks), at most 6 CTAs per SM (build
    # {"lanes": 8, "math": "x2", "min_blocks": 6, "threads": 128}); whole
    # waves of at most 2 blocks per CTA, from 976 rows per SM up to the long
    # forward's.
    "cake_rmsnorm_silu_gate_train_8bd481246a4b97382250": {
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
            "min_rows_per_sm": 976,
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_8bd481246a4b97382250",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_8bd481246a4b97382250",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_8bd481246a4b97382250",
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
        "role": "fwd_mid",
    },
    # The backward into a BF16 weight gradient: 256 threads, 16 lanes per row,
    # 16-byte accesses, one row per thread group, the gate on packed FP32 pairs,
    # four FP64 accumulator slots (build {"lanes": 16, "math": "x2",
    # "out": "bf16", "rows_per_thread": 1, "slots": 4, "threads": 256,
    # "width": 128}).
    "cake_rmsnorm_silu_gate_train_6055175bf363e85d2936": {
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
            ["buffer", "dw_sums"],
            ["buffer", "dw_done"],
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
        "dw_slots": 4,
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "row_claims",
            "rows_per_cta": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_6055175bf363e85d2936",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_6055175bf363e85d2936",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 20736,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_6055175bf363e85d2936",
        "logical": {
            "accumulate": "accumulate",
            "dw": "dw",
            "dw_done": "dw_done",
            "dw_sums": "dw_sums",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
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
    },
    # The long backward into a BF16 weight gradient (sm_107a): 128
    # threads, 16 lanes per row, 16-byte accesses, one row per thread group,
    # the gate on packed FP32 pairs, four FP64 accumulator slots (build
    # {"lanes": 16, "math": "x2", "out": "bf16", "rows_per_thread": 1,
    # "slots": 4, "threads": 128, "width": 128}), from 1024 rows per SM.
    "cake_rmsnorm_silu_gate_train_95cdf33593373841cd45": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "dw_sums"],
            ["buffer", "dw_done"],
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
        "dw_slots": 4,
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 8,
            "kind": "row_claims",
            "min_rows_per_sm": 1024,
            "rows_per_cta": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_95cdf33593373841cd45",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_95cdf33593373841cd45",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 10496,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_95cdf33593373841cd45",
        "logical": {
            "accumulate": "accumulate",
            "dw": "dw",
            "dw_done": "dw_done",
            "dw_sums": "dw_sums",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
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
        "role": "bwd_bf16_long",
    },
    # The backward into a FP32 weight gradient: 256 threads, 16 lanes per row,
    # 16-byte accesses, one row per thread group, the gate on packed FP32 pairs,
    # four FP64 accumulator slots (build {"lanes": 16, "math": "x2",
    # "out": "f32", "rows_per_thread": 1, "slots": 4, "threads": 256,
    # "width": 128}).
    "cake_rmsnorm_silu_gate_train_32301367ce4fa8110271": {
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
            ["buffer", "dw_sums"],
            ["buffer", "dw_done"],
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
        "dw_slots": 4,
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "row_claims",
            "rows_per_cta": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_32301367ce4fa8110271",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_32301367ce4fa8110271",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 20736,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_32301367ce4fa8110271",
        "logical": {
            "accumulate": "accumulate",
            "dw": "dw",
            "dw_done": "dw_done",
            "dw_sums": "dw_sums",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
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
    },
    # The long backward into a FP32 weight gradient (sm_107a): 128
    # threads, 16 lanes per row, 16-byte accesses, one row per thread group,
    # the gate on packed FP32 pairs, four FP64 accumulator slots (build
    # {"lanes": 16, "math": "x2", "out": "f32", "rows_per_thread": 1,
    # "slots": 4, "threads": 128, "width": 128}), from 1024 rows per SM.
    "cake_rmsnorm_silu_gate_train_931f86526ebac3565c37": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "dw_sums"],
            ["buffer", "dw_done"],
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
        "dw_slots": 4,
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 8,
            "kind": "row_claims",
            "min_rows_per_sm": 1024,
            "rows_per_cta": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_931f86526ebac3565c37",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_931f86526ebac3565c37",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 10496,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_931f86526ebac3565c37",
        "logical": {
            "accumulate": "accumulate",
            "dw": "dw",
            "dw_done": "dw_done",
            "dw_sums": "dw_sums",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
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
        "role": "bwd_f32_long",
    },
}

ROUTES: dict[str, dict[str, str]] = {
    "sm_100a": {
        "fwd": "cake_rmsnorm_silu_gate_train_05ffc12001cf81513436",
        "fwd_long": "cake_rmsnorm_silu_gate_train_e0672c312c57421abe21",
        "fwd_mid": "cake_rmsnorm_silu_gate_train_8bd481246a4b97382250",
        "bwd_bf16": "cake_rmsnorm_silu_gate_train_6055175bf363e85d2936",
        "bwd_f32": "cake_rmsnorm_silu_gate_train_32301367ce4fa8110271",
    },
    "sm_103a": {
        "fwd": "cake_rmsnorm_silu_gate_train_05ffc12001cf81513436",
        "fwd_long": "cake_rmsnorm_silu_gate_train_e0672c312c57421abe21",
        "fwd_mid": "cake_rmsnorm_silu_gate_train_8bd481246a4b97382250",
        "bwd_bf16": "cake_rmsnorm_silu_gate_train_6055175bf363e85d2936",
        "bwd_f32": "cake_rmsnorm_silu_gate_train_32301367ce4fa8110271",
    },
    "sm_107a": {
        "fwd": "cake_rmsnorm_silu_gate_train_05ffc12001cf81513436",
        "bwd_bf16": "cake_rmsnorm_silu_gate_train_6055175bf363e85d2936",
        "bwd_bf16_long": "cake_rmsnorm_silu_gate_train_95cdf33593373841cd45",
        "bwd_f32": "cake_rmsnorm_silu_gate_train_32301367ce4fa8110271",
        "bwd_f32_long": "cake_rmsnorm_silu_gate_train_931f86526ebac3565c37",
    },
}

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
# The kernels of a call, routed on every architecture; ``<role>_long`` is a
# role's long role and ``<role>_mid`` its mid role, routed on some.
ROLES = ("fwd", "bwd_bf16", "bwd_f32")
LONG_ROLES = ("fwd_long", "bwd_bf16_long", "bwd_f32_long")
MID_ROLES = ("fwd_mid",)
# The suffixes of a role's long and mid roles in the order
# :func:`module_for_rows` tries them (a long role runs from more rows per SM).
TIERS = ("_long", "_mid")
GRID_KINDS = ("row_blocks", "persistent_row_blocks", "row_claims")
# Elements of one normalized row.
HEAD_DIM = 128
# The forward reads rows with 32-byte vector accesses (the backward with
# 16-byte ones): a tensor is read in place
# when its storage offset and its token and head strides are multiples of this
# many BF16 elements.
ALIGNMENT_ELEMENTS = 16
# The kernels index rows with 32-bit integers.
INT32_MAX = 2**31 - 1


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
    ``arch`` with ``sm_count`` SMs: the first of the role's long and mid
    modules (``TIERS`` order) that ``arch`` routes with ``rows >=
    min_rows_per_sm * sm_count`` of its grid rule, else :func:`route_module`."""

    routes = ROUTES.get(arch, {})
    for tier in TIERS:
        name = routes.get(f"{role}{tier}")
        if name is not None and int(rows) >= int(
            MODULES[name]["grid_rule"]["min_rows_per_sm"]
        ) * int(sm_count):
            return name
    return route_module(arch, role)


def backward_ctas(name: str, rows: int, sm_count: int) -> int:
    """CTAs of the backward module ``name`` for ``rows`` rows on a device with
    ``sm_count`` SMs: one per block of ``rows_per_cta`` rows, at most
    ``ctas_per_sm * sm_count`` (one wave).  Block ``c`` is CTA ``c``'s while
    every block has a CTA; past one wave the kernel's CTAs claim every block
    from the workspace's claim counter, so the count only bounds how many
    CTAs share the rows."""

    rule = MODULES[name]["grid_rule"]
    if rule["kind"] != "row_claims":
        raise ValueError(f"module {name} is not a backward module")
    blocks = -(-int(rows) // int(rule["rows_per_cta"]))
    return min(blocks, max(1, int(rule["ctas_per_sm"]) * int(sm_count)))


def workspace_elements(name: str) -> int:
    """FP64 elements of the backward module ``name``'s workspace: its
    ``dw_slots * 128`` accumulators, then one element holding the two 32-bit
    words of ``dw_done``: its first four bytes the arrival ticket, its last four
    the claim counter."""

    return int(MODULES[name]["dw_slots"]) * HEAD_DIM + 1


def launch_grid(
    record: dict[str, Any], *, rows: int, sm_count: int, ctas: int = 0
) -> tuple[int, int, int]:
    """The grid of one module record for ``rows`` rows on a device with
    ``sm_count`` SMs (``ctas``: the backward's CTA count,
    :func:`backward_ctas`).  The long and mid forwards take the fewest whole
    waves of ``ctas_per_sm * sm_count`` CTAs whose CTAs walk at most
    ``max_blocks_per_cta`` blocks of ``rows_per_block`` rows each, and at most
    one CTA per block."""

    rule = record["grid_rule"]
    kind = rule["kind"]
    if kind == "row_blocks":
        return (-(-int(rows) // int(rule["rows_per_cta"])), 1, 1)
    if kind == "persistent_row_blocks":
        blocks = -(-int(rows) // int(rule["rows_per_block"]))
        wave = max(1, int(rule["ctas_per_sm"]) * int(sm_count))
        waves = max(1, -(-blocks // (wave * int(rule["max_blocks_per_cta"]))))
        return (min(blocks, waves * wave), 1, 1)
    if kind == "row_claims":
        return (int(ctas), 1, 1)
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
    grid is the module's :func:`launch_grid` on the device's SM count (the
    backward reads ``ctas`` from ``values``), and the kernel runs on the
    current torch stream.
    """

    record = MODULES[name]
    logical = record["logical"]
    grid = launch_grid(
        record,
        rows=rows,
        sm_count=device_sm_count(device_index),
        ctas=int(values.get("ctas", 0)),
    )
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
    "LONG_ROLES",
    "MID_ROLES",
    "MODULES",
    "ROLES",
    "ROUTES",
    "TIERS",
    "arch_for_capability",
    "backward_ctas",
    "device_arch",
    "device_sm_count",
    "launch_grid",
    "library_name",
    "load",
    "module_for_rows",
    "route_module",
    "run_module",
    "specs_for_arch",
    "workspace_elements",
]
