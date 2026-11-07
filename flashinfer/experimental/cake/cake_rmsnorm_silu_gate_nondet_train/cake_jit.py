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
rows per SM of its grid rule (:func:`min_rows_per_sm`: one count, or one per
architecture of the record; :func:`module_for_rows`; the long role first, from
more rows), and sm_103a and sm_107a the forward's extra-long role
(``fwd_xlong``), tried before the long role (on sm_107a, which routes no long
forward, in its place): on every architecture a forward that walks several row
blocks per CTA, loading each next block while it computes the current one,
``y`` the same bits (the long forward 32-row blocks on 256-thread CTAs, the mid
forward 16-row blocks on 128-thread CTAs whose thread 0 also warms later row
blocks of packed rows in L2 with one bulk prefetch a tensor; the extra-long
forward is the long forward whose thread 0 also warms the CTA's next row block
of packed rows in L2 with one bulk prefetch a tensor, on sm_103a from 3000 rows
per SM, the long forward keeping the rows from its own start up to there, and
on sm_107a from the long forward's 1740 rows per SM); on sm_100a and sm_103a
from 12000 rows per SM a backward whose thread 0 warms the next row block of
packed rows in L2 with one bulk prefetch a tensor, launched with 5 CTAs per SM,
and on sm_107a from 3200 the backward with 128-thread CTAs, ``dx`` and ``dz``
the same bits.  On sm_100a and sm_103a the backward's mid roles
(``bwd_bf16_mid``, ``bwd_f32_mid``) run from 800 and 700 rows per SM
respectively up to there the same bulk backward whose threads keep their
folded FP32 runs, and whose CTAs add their column sums, as FP32 two-sum pairs
converted once to FP64 (a column whose pair sum is not finite takes the sum
alone), ``dx`` and ``dz`` the same bits; on sm_107a they run from 1024 rows per
SM up to 3200 the 128-thread backward with the same folds.
An architecture may also route the forward's short role (``fwd_short``), whose
module runs in the role's place up to ``max_rows_per_sm`` rows per SM of its
grid rule (:func:`max_rows_per_sm`: one count, or one per architecture of the
record; tried before the extra-long, long and mid roles): on every
architecture the long forward's kernel on 256-thread CTAs with 16 threads to a
row, one 16-row block per CTA, that computes the packed SiLU gate before the
row's sum of squares (the scalar gate after rstd where the range check fails),
``y`` the same bits.  An architecture may also route the backward's short
roles (``bwd_bf16_short``, ``bwd_f32_short``), whose module runs in the role's
place up to ``max_rows`` rows of its grid rule: on every architecture one
thread-block cluster of 16 CTAs of 256 threads with the backward's row
arithmetic, ``dx`` and ``dz`` the same bits, that sums the weight gradient on
chip (CTA ``r`` of the cluster adds every CTA's FP32 partial of its eight
columns in rank order in FP64 and rounds the sum to the weight gradient's
dtype through FP32), with no workspace, atomic or
arrival ticket, so its weight gradient is the same bits on every call.  A
short role's rule bounds the rows per SM (``max_rows_per_sm``: the short
forward, whose grid of one CTA per row block grows with the rows) or the rows
themselves (``max_rows``: the short backward, whose grid is one cluster
whatever the rows and the SMs).

Grids (:func:`launch_grid`): the forward and the short forward launch one
CTA per block of ``rows_per_cta`` rows; the short backward one cluster of
``cluster`` CTAs (``launch.cluster``) whatever the rows, CTA ``r`` taking
the 16-row blocks ``r``, ``r + 16``, ... (at most one up to its ``max_rows``).
The extra-long, long and mid forwards launch the fewest whole waves of
``ctas_per_sm * SMs`` CTAs (the CTAs per SM its launch bounds are compiled
for) whose CTAs walk at most ``max_blocks_per_cta`` blocks of
``rows_per_block`` rows each, and never more CTAs than blocks; CTA ``b``
walks the blocks ``b``, ``b + grid``, ``b + 2 * grid``, ...  The backward
launches :func:`backward_ctas` CTAs, one per block but at most
``ctas_per_sm`` per SM: one wave, or for the long backward on sm_100a and
sm_103a 5, a quarter more than an SM holds at once (an SM starts its fifth
CTA as one of its first four leaves).  When the grid has
a CTA for every block, CTA ``c`` takes block ``c`` and claims nothing;
otherwise the CTAs claim every block from the claim counter of the
workspace, in chunks of consecutive blocks that shrink to single blocks at
the end, each CTA claiming its next chunk as the last block of its current
one starts, so a CTA holds one chunk at a time and one that starts late
takes fewer blocks, none when every chunk was claimed before it started.
The kernel folds each thread's FP32 sums into
FP64 after at most 16 of its rows (a CTA of a few blocks adds them across its
warps in FP32 first and converts once; the mid backward into FP32 two-sum
pairs), so a term passes through at most 16 FP32 roundings at any grid: within
the deterministic backward's error budget while the gate, the terms and the
sums are FP32 normal numbers or zero (FP32 underflows gradually: with subnormal
values both packages' weight gradients can leave that budget; the mid
backward's pair fold has a fold error bound of its own, and its FP32 pair sums
can pass the FP32 range where FP64 sums stay finite; see the package
documentation).  No
module is cooperative, and no CTA waits for another inside a kernel but in the
short backward, whose cluster's CTAs are scheduled together on the SMs of one
GPC and each wait for the other CTAs' partials of their columns.  A kernel
resident on another stream delays a call of the other modules roughly in
proportion to the SM slots it holds, and one that holds resources a kernel
needs on every SM can still delay its start.  A long or mid forward CTA whose
slot such a kernel holds walks all of its blocks once it starts.  On sm_100a
and sm_103a the long forward runs only from the ``min_rows_per_sm`` at which,
with at most ``max_blocks_per_cta`` blocks a CTA, that delay is no larger
than the forward's, and the mid forward only from the one at which such a
call stays within the forward stall test's bound and ends no later than the
forward's.  On sm_107a (probed with 212 SMs) each runs from the
``min_rows_per_sm`` from which, in the medians over the probe runs, such a
call stayed within that bound beside every kernel holding part of the SMs and
took at most 1 % plus the runs' spread longer than the forward beside every
probed kernel; there the long forward's delay over its share of the held
slots is larger than the forward's at every routed row count probed, and the
mid forward ends after the forward at 1152 rows per SM (by up to 2.15 % in
the worst run; in the worst kernel-launch run also at 960).  The short
forward runs only up to the ``max_rows_per_sm`` at which, on sm_100a and
sm_103a, its grid has fewer CTAs than the SMs and such a call still ends no
later than the forward's, and on sm_107a (probed with 212 SMs) its grid fits
in one wave of 4 CTAs per SM on all SMs but 16 and such a call stayed within
the forward stall test's bound beside that test's kernels holding part of the
SMs (FlashInfer issue #7548); the short backward runs only up to the ``max_rows``
at which no CTA of its cluster takes more than one row block.

The workspace of the backward is ``dw_slots * 128`` FP64 accumulators and one
8-byte word holding two 32-bit words, the arrival ticket and the claim counter
(:func:`workspace_elements` FP64 elements in all), zero when a launch starts
and zero again when it ends: a zero-filled workspace stays reusable for any
later call on its stream.  Every launch of the backward and the long
backward takes the ticket, and claims blocks when the rows have more blocks
than its grid has CTAs, also one that computes no weight gradient, so each
needs the workspace; the short backward takes none (:func:`workspace_elements`
is 0).

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
    # The long forward: 256 threads, 8 lanes per row (32-row blocks), 32-byte
    # accesses, at most 3 CTAs per SM, each loading its next block while it
    # computes the current one, the gate on packed FP32 pairs (build {"lanes":
    # 8, "math": "x2", "min_blocks": 3, "prefetch_ahead": 0, "prefetch_wave":
    # 0, "threads": 256}); whole waves of at most 6 blocks per CTA, from 2400
    # rows per SM on sm_100a and sm_103a (on sm_103a up to the extra-long
    # forward's 3000; sm_107a runs the extra-long forward instead).
    "cake_rmsnorm_silu_gate_train_7b7fac905aff1ddaa258": {
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
            "min_rows_per_sm": {
                "sm_100a": 2400,
                "sm_103a": 2400,
            },
            "rows_per_block": 32,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_7b7fac905aff1ddaa258",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_7b7fac905aff1ddaa258",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_7b7fac905aff1ddaa258",
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
    # The extra-long forward (sm_103a and sm_107a): the long forward's program
    # whose thread 0 also warms in L2, with one bulk prefetch a tensor where x
    # and z hold their rows packed, the CTA's next row block (build {"lanes":
    # 8, "math": "x2", "min_blocks": 3, "prefetch_ahead": 1, "prefetch_wave":
    # 0, "threads": 256}); whole waves of at most 6 blocks per CTA, from 3000
    # rows per SM on sm_103a and from 1740 (four whole waves, the long
    # forward's start there) on sm_107a.
    "cake_rmsnorm_silu_gate_train_7f014401d5cc855ab883": {
        "arches": [
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
            "ctas_per_sm": 3,
            "kind": "persistent_row_blocks",
            "max_blocks_per_cta": 6,
            "min_rows_per_sm": {
                "sm_103a": 3000,
                "sm_107a": 1740,
            },
            "rows_per_block": 32,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_7f014401d5cc855ab883",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_7f014401d5cc855ab883",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_7f014401d5cc855ab883",
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
        "role": "fwd_xlong",
    },
    # The mid forward: the long forward's kernel on 128 threads, 8 lanes per
    # row (16-row blocks), at most 6 CTAs per SM, whose thread 0 also warms in
    # L2, with one bulk prefetch a tensor where x and z hold their rows packed,
    # the CTA's next row block and the first row block of the CTA one wave of 6
    # CTAs per SM later (build {"lanes": 8, "math": "x2", "min_blocks": 6,
    # "prefetch_ahead": 1, "prefetch_wave": 1, "threads": 128}); whole waves of
    # at most 2 blocks per CTA, from 976 rows per SM on sm_100a and sm_103a and
    # from 800 on sm_107a, up to the long forward's.
    "cake_rmsnorm_silu_gate_train_87f0173e9583ad4fdd2a": {
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
            "ctas_per_sm": 6,
            "kind": "persistent_row_blocks",
            "max_blocks_per_cta": 2,
            "min_rows_per_sm": {
                "sm_100a": 976,
                "sm_103a": 976,
                "sm_107a": 800,
            },
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_87f0173e9583ad4fdd2a",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_87f0173e9583ad4fdd2a",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_87f0173e9583ad4fdd2a",
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
    # The short forward: the long forward's kernel on 256 threads, 16 lanes
    # per row (16-row blocks), launch bounds for 4 CTAs per SM, the packed gate
    # computed before the sum of squares (build {"lanes": 16, "math":
    # "x2_early", "min_blocks": 4, "prefetch_ahead": 0, "prefetch_wave": 0,
    # "threads": 256}), one block per CTA, up to 12 rows per SM on sm_100a and
    # sm_103a and 59 on sm_107a.
    "cake_rmsnorm_silu_gate_train_de5df3c9a4a5a804fcf2": {
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
            "max_rows_per_sm": {
                "sm_100a": 12,
                "sm_103a": 12,
                "sm_107a": 59,
            },
            "rows_per_cta": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_de5df3c9a4a5a804fcf2",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_de5df3c9a4a5a804fcf2",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_de5df3c9a4a5a804fcf2",
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
        "role": "fwd_short",
    },
    # The backward into a BF16 weight gradient: 256 threads, 16 lanes per row,
    # 16-byte accesses, one row per thread group, the gate on packed FP32 pairs,
    # four FP64 accumulator slots (build {"lanes": 16, "math": "x2",
    # "out": "bf16", "prefetch": "rows", "rows_per_thread": 1, "slots": 4,
    # "threads": 256, "width": 128}).
    "cake_rmsnorm_silu_gate_train_58f1ac96c476c73ec5c8": {
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
        "kernel": "cake_rmsnorm_silu_gate_train_58f1ac96c476c73ec5c8",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_58f1ac96c476c73ec5c8",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 20736,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_58f1ac96c476c73ec5c8",
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
    # The long backward into a BF16 weight gradient (sm_100a and sm_103a): the
    # backward's build with thread 0 prefetching the next row block of packed
    # dy, x and z into L2 with one bulk copy a tensor (build {"lanes": 16,
    # "math": "x2", "out": "bf16", "prefetch": "bulk", "rows_per_thread": 1,
    # "slots": 4, "threads": 256, "width": 128}), 5 CTAs per SM, from 12000
    # rows per SM (below it the mid backward).
    "cake_rmsnorm_silu_gate_train_9f2f4d1f2b85a7469a7d": {
        "arches": [
            "sm_100a",
            "sm_103a",
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
            "ctas_per_sm": 5,
            "kind": "row_claims",
            "min_rows_per_sm": {
                "sm_100a": 12000,
                "sm_103a": 12000,
            },
            "rows_per_cta": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_9f2f4d1f2b85a7469a7d",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_9f2f4d1f2b85a7469a7d",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 20736,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_9f2f4d1f2b85a7469a7d",
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
    # The mid backward into a BF16 weight gradient (sm_100a and sm_103a): the
    # long backward's bulk build whose threads keep their folded FP32 runs, and
    # whose CTAs add their column sums, as FP32 two-sum pairs converted once to
    # FP64 for the atomic add (a column whose pair sum is not finite takes the
    # sum alone) (build {"fold": "ff", "lanes": 16, "math": "x2", "out": "bf16",
    # "prefetch": "bulk", "rows_per_thread": 1, "slots": 4, "threads": 256,
    # "width": 128}), 5 CTAs per SM, from 800 rows per SM on sm_100a and 700 on
    # sm_103a up to the long backward's 12000.
    "cake_rmsnorm_silu_gate_train_f14663d91454a0e833f7": {
        "arches": [
            "sm_100a",
            "sm_103a",
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
            "ctas_per_sm": 5,
            "kind": "row_claims",
            "min_rows_per_sm": {
                "sm_100a": 800,
                "sm_103a": 700,
            },
            "rows_per_cta": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_f14663d91454a0e833f7",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_f14663d91454a0e833f7",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 20736,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_f14663d91454a0e833f7",
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
        "role": "bwd_bf16_mid",
    },
    # The long backward into a BF16 weight gradient (sm_107a): 128
    # threads, 16 lanes per row, 16-byte accesses, one row per thread group,
    # the gate on packed FP32 pairs, four FP64 accumulator slots (build
    # {"lanes": 16, "math": "x2", "out": "bf16", "prefetch": "rows",
    # "rows_per_thread": 1, "slots": 4, "threads": 128, "width": 128}), from
    # 3200 rows per SM (below it the mid backward).
    "cake_rmsnorm_silu_gate_train_8b61372f6a32eeca4d10": {
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
            "min_rows_per_sm": 3200,
            "rows_per_cta": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_8b61372f6a32eeca4d10",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_8b61372f6a32eeca4d10",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 10496,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_8b61372f6a32eeca4d10",
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
    # The mid backward into a BF16 weight gradient (sm_107a): the long
    # backward's 128-thread build whose threads keep their folded FP32 runs, and
    # whose CTAs add their column sums, as FP32 two-sum pairs converted once to
    # FP64 for the atomic add (a column whose pair sum is not finite takes the
    # sum alone) (build {"fold": "ff", "lanes": 16, "math": "x2", "out": "bf16",
    # "prefetch": "rows", "rows_per_thread": 1, "slots": 4, "threads": 128,
    # "width": 128}), from 1024 rows per SM up to the long backward's 3200.
    "cake_rmsnorm_silu_gate_train_195842e606c6dfb8e998": {
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
            "min_rows_per_sm": {
                "sm_107a": 1024,
            },
            "rows_per_cta": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_195842e606c6dfb8e998",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_195842e606c6dfb8e998",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 10496,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_195842e606c6dfb8e998",
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
        "role": "bwd_bf16_mid",
    },
    # The short backward into a BF16 weight gradient:
    # one cluster of 16 CTAs of 256 threads, the backward's row arithmetic
    # (16 lanes per row, the gate on packed FP32 pairs), CTA r of the
    # cluster finishing the weight-gradient columns 8r to 8r + 7 from every
    # CTA's FP32 partials in rank order in FP64, no workspace (build
    # {"cluster": 16, "out": "bf16", "threads": 256}), up to 256 rows.
    "cake_rmsnorm_silu_gate_train_cb597a0bebd6145e2024": {
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
            "cluster": 16,
            "kind": "cluster",
            "max_rows": 256,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_cb597a0bebd6145e2024",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_cb597a0bebd6145e2024",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (16, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 6144,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_cb597a0bebd6145e2024",
        "logical": {
            "accumulate": "accumulate",
            "dw": "dw",
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
        "role": "bwd_bf16_short",
    },
    # The backward into a FP32 weight gradient: 256 threads, 16 lanes per row,
    # 16-byte accesses, one row per thread group, the gate on packed FP32 pairs,
    # four FP64 accumulator slots (build {"lanes": 16, "math": "x2",
    # "out": "f32", "prefetch": "rows", "rows_per_thread": 1, "slots": 4,
    # "threads": 256, "width": 128}).
    "cake_rmsnorm_silu_gate_train_26c869703d0ab9e4a39d": {
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
        "kernel": "cake_rmsnorm_silu_gate_train_26c869703d0ab9e4a39d",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_26c869703d0ab9e4a39d",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 20736,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_26c869703d0ab9e4a39d",
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
    # The long backward into a FP32 weight gradient (sm_100a and sm_103a): the
    # backward's build with thread 0 prefetching the next row block of packed
    # dy, x and z into L2 with one bulk copy a tensor (build {"lanes": 16,
    # "math": "x2", "out": "f32", "prefetch": "bulk", "rows_per_thread": 1,
    # "slots": 4, "threads": 256, "width": 128}), 5 CTAs per SM, from 12000
    # rows per SM (below it the mid backward).
    "cake_rmsnorm_silu_gate_train_e6a05f4e7ec6b374d3f2": {
        "arches": [
            "sm_100a",
            "sm_103a",
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
            "ctas_per_sm": 5,
            "kind": "row_claims",
            "min_rows_per_sm": {
                "sm_100a": 12000,
                "sm_103a": 12000,
            },
            "rows_per_cta": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_e6a05f4e7ec6b374d3f2",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_e6a05f4e7ec6b374d3f2",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 20736,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_e6a05f4e7ec6b374d3f2",
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
    # The mid backward into an FP32 weight gradient (sm_100a and sm_103a): the
    # long backward's bulk build whose threads keep their folded FP32 runs, and
    # whose CTAs add their column sums, as FP32 two-sum pairs converted once to
    # FP64 for the atomic add (a column whose pair sum is not finite takes the
    # sum alone) (build {"fold": "ff", "lanes": 16, "math": "x2", "out": "f32",
    # "prefetch": "bulk", "rows_per_thread": 1, "slots": 4, "threads": 256,
    # "width": 128}), 5 CTAs per SM, from 800 rows per SM on sm_100a and 700 on
    # sm_103a up to the long backward's 12000.
    "cake_rmsnorm_silu_gate_train_69d7df1dfbf3cfd61749": {
        "arches": [
            "sm_100a",
            "sm_103a",
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
            "ctas_per_sm": 5,
            "kind": "row_claims",
            "min_rows_per_sm": {
                "sm_100a": 800,
                "sm_103a": 700,
            },
            "rows_per_cta": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_69d7df1dfbf3cfd61749",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_69d7df1dfbf3cfd61749",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 20736,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_69d7df1dfbf3cfd61749",
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
        "role": "bwd_f32_mid",
    },
    # The long backward into a FP32 weight gradient (sm_107a): 128
    # threads, 16 lanes per row, 16-byte accesses, one row per thread group,
    # the gate on packed FP32 pairs, four FP64 accumulator slots (build
    # {"lanes": 16, "math": "x2", "out": "f32", "prefetch": "rows",
    # "rows_per_thread": 1, "slots": 4, "threads": 128, "width": 128}), from
    # 3200 rows per SM (below it the mid backward).
    "cake_rmsnorm_silu_gate_train_4c6188a9425364b78977": {
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
            "min_rows_per_sm": 3200,
            "rows_per_cta": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_4c6188a9425364b78977",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_4c6188a9425364b78977",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 10496,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_4c6188a9425364b78977",
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
    # The mid backward into an FP32 weight gradient (sm_107a): the long
    # backward's 128-thread build whose threads keep their folded FP32 runs, and
    # whose CTAs add their column sums, as FP32 two-sum pairs converted once to
    # FP64 for the atomic add (a column whose pair sum is not finite takes the
    # sum alone) (build {"fold": "ff", "lanes": 16, "math": "x2", "out": "f32",
    # "prefetch": "rows", "rows_per_thread": 1, "slots": 4, "threads": 128,
    # "width": 128}), from 1024 rows per SM up to the long backward's 3200.
    "cake_rmsnorm_silu_gate_train_d1031b4d8133fbe01754": {
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
            "min_rows_per_sm": {
                "sm_107a": 1024,
            },
            "rows_per_cta": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_d1031b4d8133fbe01754",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_d1031b4d8133fbe01754",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 10496,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_d1031b4d8133fbe01754",
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
        "role": "bwd_f32_mid",
    },
    # The short backward into a FP32 weight gradient:
    # one cluster of 16 CTAs of 256 threads, the backward's row arithmetic
    # (16 lanes per row, the gate on packed FP32 pairs), CTA r of the
    # cluster finishing the weight-gradient columns 8r to 8r + 7 from every
    # CTA's FP32 partials in rank order in FP64, no workspace (build
    # {"cluster": 16, "out": "f32", "threads": 256}), up to 256 rows.
    "cake_rmsnorm_silu_gate_train_c447141e8f19dd798212": {
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
            "cluster": 16,
            "kind": "cluster",
            "max_rows": 256,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_c447141e8f19dd798212",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_c447141e8f19dd798212",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (16, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 6144,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_c447141e8f19dd798212",
        "logical": {
            "accumulate": "accumulate",
            "dw": "dw",
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
        "role": "bwd_f32_short",
    },
}

ROUTES: dict[str, dict[str, str]] = {
    "sm_100a": {
        "fwd": "cake_rmsnorm_silu_gate_train_05ffc12001cf81513436",
        "fwd_long": "cake_rmsnorm_silu_gate_train_7b7fac905aff1ddaa258",
        "fwd_mid": "cake_rmsnorm_silu_gate_train_87f0173e9583ad4fdd2a",
        "fwd_short": "cake_rmsnorm_silu_gate_train_de5df3c9a4a5a804fcf2",
        "bwd_bf16": "cake_rmsnorm_silu_gate_train_58f1ac96c476c73ec5c8",
        "bwd_bf16_long": "cake_rmsnorm_silu_gate_train_9f2f4d1f2b85a7469a7d",
        "bwd_bf16_mid": "cake_rmsnorm_silu_gate_train_f14663d91454a0e833f7",
        "bwd_bf16_short": "cake_rmsnorm_silu_gate_train_cb597a0bebd6145e2024",
        "bwd_f32": "cake_rmsnorm_silu_gate_train_26c869703d0ab9e4a39d",
        "bwd_f32_long": "cake_rmsnorm_silu_gate_train_e6a05f4e7ec6b374d3f2",
        "bwd_f32_mid": "cake_rmsnorm_silu_gate_train_69d7df1dfbf3cfd61749",
        "bwd_f32_short": "cake_rmsnorm_silu_gate_train_c447141e8f19dd798212",
    },
    "sm_103a": {
        "fwd": "cake_rmsnorm_silu_gate_train_05ffc12001cf81513436",
        "fwd_xlong": "cake_rmsnorm_silu_gate_train_7f014401d5cc855ab883",
        "fwd_long": "cake_rmsnorm_silu_gate_train_7b7fac905aff1ddaa258",
        "fwd_mid": "cake_rmsnorm_silu_gate_train_87f0173e9583ad4fdd2a",
        "fwd_short": "cake_rmsnorm_silu_gate_train_de5df3c9a4a5a804fcf2",
        "bwd_bf16": "cake_rmsnorm_silu_gate_train_58f1ac96c476c73ec5c8",
        "bwd_bf16_long": "cake_rmsnorm_silu_gate_train_9f2f4d1f2b85a7469a7d",
        "bwd_bf16_mid": "cake_rmsnorm_silu_gate_train_f14663d91454a0e833f7",
        "bwd_bf16_short": "cake_rmsnorm_silu_gate_train_cb597a0bebd6145e2024",
        "bwd_f32": "cake_rmsnorm_silu_gate_train_26c869703d0ab9e4a39d",
        "bwd_f32_long": "cake_rmsnorm_silu_gate_train_e6a05f4e7ec6b374d3f2",
        "bwd_f32_mid": "cake_rmsnorm_silu_gate_train_69d7df1dfbf3cfd61749",
        "bwd_f32_short": "cake_rmsnorm_silu_gate_train_c447141e8f19dd798212",
    },
    "sm_107a": {
        "fwd": "cake_rmsnorm_silu_gate_train_05ffc12001cf81513436",
        "fwd_xlong": "cake_rmsnorm_silu_gate_train_7f014401d5cc855ab883",
        "fwd_mid": "cake_rmsnorm_silu_gate_train_87f0173e9583ad4fdd2a",
        "fwd_short": "cake_rmsnorm_silu_gate_train_de5df3c9a4a5a804fcf2",
        "bwd_bf16": "cake_rmsnorm_silu_gate_train_58f1ac96c476c73ec5c8",
        "bwd_bf16_long": "cake_rmsnorm_silu_gate_train_8b61372f6a32eeca4d10",
        "bwd_bf16_mid": "cake_rmsnorm_silu_gate_train_195842e606c6dfb8e998",
        "bwd_bf16_short": "cake_rmsnorm_silu_gate_train_cb597a0bebd6145e2024",
        "bwd_f32": "cake_rmsnorm_silu_gate_train_26c869703d0ab9e4a39d",
        "bwd_f32_long": "cake_rmsnorm_silu_gate_train_4c6188a9425364b78977",
        "bwd_f32_mid": "cake_rmsnorm_silu_gate_train_d1031b4d8133fbe01754",
        "bwd_f32_short": "cake_rmsnorm_silu_gate_train_c447141e8f19dd798212",
    },
}

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
# The kernels of a call, routed on every architecture; ``<role>_xlong`` is a
# role's extra-long role, ``<role>_long`` its long role, ``<role>_mid`` its mid
# role and ``<role>_short`` its short role, routed on some.
ROLES = ("fwd", "bwd_bf16", "bwd_f32")
XLONG_ROLES = ("fwd_xlong",)
LONG_ROLES = ("fwd_long", "bwd_bf16_long", "bwd_f32_long")
MID_ROLES = ("fwd_mid", "bwd_bf16_mid", "bwd_f32_mid")
SHORT_ROLES = ("fwd_short", "bwd_bf16_short", "bwd_f32_short")
# The suffixes of a role's extra-long, long and mid roles in the order
# :func:`module_for_rows` tries them (an extra-long role runs from at least as
# many rows per SM as a long one, a long role from more than a mid one).
TIERS = ("_xlong", "_long", "_mid")
GRID_KINDS = ("row_blocks", "persistent_row_blocks", "row_claims", "cluster")
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


def min_rows_per_sm(name: str, arch: str) -> int:
    """The rows per SM from which the long or mid module ``name`` runs on
    ``arch``: its grid rule's ``min_rows_per_sm``, one count for every
    architecture of the record or one count per architecture."""

    start = MODULES[name]["grid_rule"]["min_rows_per_sm"]
    return int(start[arch] if isinstance(start, dict) else start)


def max_rows_per_sm(name: str, arch: str) -> int:
    """The rows per SM up to which the short module ``name`` runs on
    ``arch``: its grid rule's ``max_rows_per_sm``, one count for every
    architecture of the record or one count per architecture."""

    end = MODULES[name]["grid_rule"]["max_rows_per_sm"]
    return int(end[arch] if isinstance(end, dict) else end)


def module_for_rows(arch: str, role: str, rows: int, sm_count: int) -> str:
    """The module that runs ``role`` (one of ``ROLES``) for ``rows`` rows on
    ``arch`` with ``sm_count`` SMs: the role's short module when ``arch``
    routes one and ``rows`` is at most its grid rule's ``max_rows`` (a grid
    that does not grow with the rows: the short backward's one cluster) or
    else ``max_rows_per_sm(module, arch) * sm_count`` (one CTA per row block:
    the short forward), else the first of the role's extra-long, long and
    mid modules (``TIERS`` order) that ``arch`` routes with
    ``rows >= min_rows_per_sm(module, arch) * sm_count``, else
    :func:`route_module`."""

    routes = ROUTES.get(arch, {})
    short = routes.get(f"{role}_short")
    if short is not None:
        rule = MODULES[short]["grid_rule"]
        if "max_rows" in rule:
            most = int(rule["max_rows"])
        else:
            most = max_rows_per_sm(short, arch) * int(sm_count)
        if int(rows) <= most:
            return short
    for tier in TIERS:
        name = routes.get(f"{role}{tier}")
        if name is not None and int(rows) >= min_rows_per_sm(name, arch) * int(
            sm_count
        ):
            return name
    return route_module(arch, role)


def backward_ctas(name: str, rows: int, sm_count: int) -> int:
    """CTAs of the backward module ``name`` for ``rows`` rows on a device with
    ``sm_count`` SMs: one per block of ``rows_per_cta`` rows, at most
    ``ctas_per_sm * sm_count``.  Block ``c`` is CTA ``c``'s while every block
    has a CTA; with more blocks the kernel's CTAs claim every block from the
    workspace's claim counter, so the count only bounds how many CTAs share
    the rows.  The short backward: its cluster's ``cluster`` CTAs whatever
    the rows."""

    rule = MODULES[name]["grid_rule"]
    if rule["kind"] == "cluster":
        return int(rule["cluster"])
    if rule["kind"] != "row_claims":
        raise ValueError(f"module {name} is not a backward module")
    blocks = -(-int(rows) // int(rule["rows_per_cta"]))
    return min(blocks, max(1, int(rule["ctas_per_sm"]) * int(sm_count)))


def workspace_elements(name: str) -> int:
    """FP64 elements of the backward module ``name``'s workspace: its
    ``dw_slots * 128`` accumulators, then one element holding the two 32-bit
    words of ``dw_done``: its first four bytes the arrival ticket, its last four
    the claim counter.  0 for the short backward, which sums the weight
    gradient on chip and takes no workspace."""

    record = MODULES[name]
    if record["grid_rule"]["kind"] == "cluster":
        return 0
    return int(record["dw_slots"]) * HEAD_DIM + 1


def launch_grid(
    record: dict[str, Any], *, rows: int, sm_count: int, ctas: int = 0
) -> tuple[int, int, int]:
    """The grid of one module record for ``rows`` rows on a device with
    ``sm_count`` SMs (``ctas``: the backward's CTA count,
    :func:`backward_ctas`).  The long and mid forwards take the fewest whole
    waves of ``ctas_per_sm * sm_count`` CTAs whose CTAs walk at most
    ``max_blocks_per_cta`` blocks of ``rows_per_block`` rows each, and at most
    one CTA per block.  The short backward takes one cluster of ``cluster``
    CTAs."""

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
    if kind == "cluster":
        return (int(rule["cluster"]), 1, 1)
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
    "SHORT_ROLES",
    "TIERS",
    "XLONG_ROLES",
    "arch_for_capability",
    "backward_ctas",
    "device_arch",
    "device_sm_count",
    "launch_grid",
    "library_name",
    "load",
    "max_rows_per_sm",
    "min_rows_per_sm",
    "module_for_rows",
    "route_module",
    "run_module",
    "specs_for_arch",
    "workspace_elements",
]
