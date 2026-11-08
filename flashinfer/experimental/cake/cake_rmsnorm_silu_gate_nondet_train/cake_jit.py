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
launch record, its grid rule, the workspace buffers of the backward and the
kernel-parameter -> logical-argument map.  What to compile -- sources,
digests, nvcc flags -- is recorded once in ``build.json`` next to this file;
:mod:`.cake_build` serves the library ``<library>_<arch>`` from a prebuilt
``aot/`` file when a builder compiled it and otherwise through FlashInfer's
JIT from the same sources and flags, verifying every source digest before the
first build.

A role is one kernel of a call: ``fwd`` (the forward: ``rstd`` from
``rsqrt.approx`` and the SiLU gate from ``ex2.approx`` and ``rcp.approx``,
every row's arithmetic fixed, so ``y`` is the same bits on every call) and
``bwd_bf16`` / ``bwd_f32`` (the backward: ``dx``, ``dz`` and the weight
gradient in one kernel; ``dx``, ``dz`` and the FP32 partial of each chunk of
rows are the deterministic backward's arithmetic, and the partials are summed
exactly, as integers, into a weight gradient rounded to FP32 from the FP64
value of the exact sum, then to BF16 for a BF16 one, so every output is the
same bits at any grid and in any order of the CTAs).
``ROUTES[arch][role]`` names the one module a role runs on an architecture;
a module may be delivered for more architectures than route it (every
forward module is delivered for every architecture, so a new forward cut is a
host-plan change).  :func:`module_for_rows` picks a call's module from the
rows, the device's SM count and the layout of the inputs: the forward of
packed rows (``x`` and ``z`` hold row ``n`` at element ``128 n``) by
``FORWARD_BANDS``, of other layouts by ``STRIDED_FORWARD_BANDS``, bands of
rows per SM (the tiny forward, which reads packed rows only, the short
forward, the forward and the mid, long and extra-long forwards, each where
the architecture's bands name it); the backward by its
small-row roles (``BAND_ROLES``) in the rows of their ``BACKWARD_BANDS``
bands, past them its mid role (the claimed backward) where the rows have
more chunks than ``more_chunks_than_per_sm`` per SM, else the call role.
The forward's programs: the tiny forward (16 threads to a row, one 8-row
block per 128-thread CTA, the gate before the sum of squares, packed rows
only; the short forward's arithmetic and bits), the short forward (16
threads to a row, one 16-row block per CTA, the gate before the sum of
squares), the forward (64-thread
CTAs, 8 threads to a row), the mid forward (128-thread CTAs walking 16-row
blocks, thread 0 warming the CTA's next block and the first block of the CTA
a wave later in L2), the long forward (256-thread CTAs walking 32-row
blocks) and the extra-long forward (the long forward whose thread 0 also
warms the CTA's next block in L2).  Each program's arithmetic is fixed; the
programs associate a row's sum of squares differently (the tiny and the short
forward: 8-term FMA chains of 16 lanes folded by XOR shuffles 1, 2, 4, 8; the others:
16-term chains of 8 lanes folded by 1, 2, 4), so ``y``'s bits can differ
between the programs, and so between the bands of one architecture, while
each program gives the same bits for the same inputs on the same device.

The backward's call role is the static backward (CTA ``b`` runs the chunks
``b``, ``b + grid``, ... and claims nothing); its mid role is the claimed
backward (the CTAs claim runs of chunks from the
workspace's claim counter at run time, so a CTA slowed beside another kernel
takes fewer), which runs where the rows have more chunks than the static
backward's grid has CTAs.  The small-row backwards (whose BF16 bands a call
with ``compute_grad_weight=False`` also takes) are each one thread-block
cluster of 16 CTAs with the backward's row arithmetic (``dx`` and ``dz`` the
same bits) that sums the weight gradient on chip in a fixed order and rounds
once, with no workspace: the exact tiny backward (``<role>_t512`` and
``<role>_t1024``: 4 or 8 teams of the deterministic backward's 128-thread CTA
a CTA, up to 512 or 1024 rows, its weight gradient the deterministic pair's
bits).  ``BACKWARD_BANDS`` gives each its rows on each architecture.

Grids (:func:`launch_grid`): the forward and the short forward launch one
CTA per block of ``rows_per_cta`` rows; the extra-long, long and mid forwards
the fewest whole waves of ``ctas_per_sm * SMs`` CTAs whose CTAs walk at most
``max_blocks_per_cta`` blocks of ``rows_per_block`` rows each, never more
CTAs than blocks (CTA ``b`` walks the blocks ``b``, ``b + grid``, ...); a
small-row backward one cluster of ``cluster`` CTAs (``launch.cluster``)
whatever the rows, up to its rule's ``max_rows``.  The static and the
claimed backward cut the rows into blocks of ``rows_per_block`` rows and the
blocks into the deterministic backward's
chunks (:func:`backward_chunks`): ``blocks_per_chunk = max(1, ceil(blocks /
chunk_limit))`` consecutive blocks a chunk, ``C = ceil(blocks /
blocks_per_chunk)`` chunks, and launch :func:`backward_ctas` ``G = min(C,
ctas_per_sm * SMs)`` CTAs; the host passes ``blocks_per_chunk`` to the
kernel.  Either backward is correct, with the same bits, at any grid; the cut
between them is a matter of speed.  No module is cooperative.  A kernel
resident on another stream delays a call roughly in proportion to the SM
slots it holds, and one that holds resources a kernel needs on every SM can
still delay its start; a persistent forward CTA or a static backward CTA
whose slot such a kernel holds runs all of its blocks once it starts.

The workspace of the static and the claimed backward is three buffers
(:func:`workspace_layout`, :func:`workspace_bytes`): ``dw_win``, the 64-bit
integer windows of the exact sum (``slots`` copies of 7 windows of 128
columns), ``dw_spec``, 128 FP32 class words for infinite and NaN partials,
and ``dw_done``, two 32-bit words (the arrival ticket and the claim counter).
All of it is zero when a launch starts and zero again when it ends, so a
zero-filled workspace stays reusable for any later call on its stream.  Every
launch of those backwards takes the ticket, also one that computes no weight
gradient, so each needs the workspace; the small-row backwards take none
(:func:`workspace_bytes` is 0).

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
    # The forward: 64 threads, 8 lanes per row (8-row blocks), 16-byte
    # accesses, the gate computed before the sum of squares, approximate FP32
    # intrinsics (build {"early": true, "hints": false, "lanes": 8,
    # "rows_per_thread": 1, "threads": 64, "width": 128}).
    "cake_rmsnorm_silu_gate_train_c7919260b39b69e4e646": {
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
            "rows_per_cta": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_c7919260b39b69e4e646",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_c7919260b39b69e4e646",
        "launch": {
            "block": (64, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_c7919260b39b69e4e646",
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
    # The extra-long forward: the long forward whose thread 0 also warms the
    # CTA's next row block of packed rows in L2 with one bulk prefetch a
    # tensor (build {"early": false, "lanes": 8, "min_blocks": 3,
    # "prefetch_ahead": 1, "prefetch_wave": 0, "threads": 256}); whole waves
    # of at most 6 blocks per CTA; FORWARD_BANDS gives it its rows.
    "cake_rmsnorm_silu_gate_train_83fe9e25ce0d01cc3ab1": {
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
            "ctas_per_sm": 3,
            "kind": "persistent_row_blocks",
            "max_blocks_per_cta": 6,
            "rows_per_block": 32,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_83fe9e25ce0d01cc3ab1",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_83fe9e25ce0d01cc3ab1",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_83fe9e25ce0d01cc3ab1",
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
    # The long forward: 256 threads, 8 lanes per row (32-row blocks), 32-byte
    # accesses, at most 3 CTAs per SM, each loading its next block while it
    # computes the current one (build {"early": false, "lanes": 8,
    # "min_blocks": 3, "prefetch_ahead": 0, "prefetch_wave": 0, "threads":
    # 256}); whole waves of at most 6 blocks per CTA; FORWARD_BANDS gives it
    # its rows (sm_100a).
    "cake_rmsnorm_silu_gate_train_c6151f64f22ae65a48eb": {
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
            "ctas_per_sm": 3,
            "kind": "persistent_row_blocks",
            "max_blocks_per_cta": 6,
            "rows_per_block": 32,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_c6151f64f22ae65a48eb",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_c6151f64f22ae65a48eb",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_c6151f64f22ae65a48eb",
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
    # The mid forward: 128 threads, 8 lanes per row (16-row blocks), at most 6
    # CTAs per SM, thread 0 warming the CTA's next block and the first block
    # of the CTA one wave later in L2 (build {"early": false, "lanes": 8,
    # "min_blocks": 6, "prefetch_ahead": 1, "prefetch_wave": 1, "threads":
    # 128}); whole waves of at most 2 blocks per CTA; FORWARD_BANDS gives it
    # its rows.
    "cake_rmsnorm_silu_gate_train_cf7a87977aaad1a9822f": {
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
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_cf7a87977aaad1a9822f",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_cf7a87977aaad1a9822f",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_cf7a87977aaad1a9822f",
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
    # The short forward: 256 threads, 16 lanes per row (16-row blocks), one
    # block per CTA, the gate before the sum of squares (build {"early": true,
    # "lanes": 16, "min_blocks": 4, "prefetch_ahead": 0, "prefetch_wave": 0,
    # "threads": 256}); FORWARD_BANDS gives it its rows.
    "cake_rmsnorm_silu_gate_train_3cb6aa31e140d9b13052": {
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
        "kernel": "cake_rmsnorm_silu_gate_train_3cb6aa31e140d9b13052",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_3cb6aa31e140d9b13052",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_3cb6aa31e140d9b13052",
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
    # The tiny forward: 128 threads, 16 lanes per row (8-row blocks), one
    # block per CTA, packed rows only (row n at element 128 n), the short
    # forward's bits (build {"threads": 128}); FORWARD_BANDS gives it its
    # rows.
    "cake_rmsnorm_silu_gate_train_04bb67629537ef270bd4": {
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
            ["parameter", "eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_blocks",
            "rows_per_cta": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_04bb67629537ef270bd4",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_04bb67629537ef270bd4",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_04bb67629537ef270bd4",
        "logical": {
            "eps": "eps",
            "rows": "rows",
            "w": "w",
            "x": "x",
            "y": "y",
            "z": "z",
        },
        "role": "fwd_tiny",
    },
    # The static backward into a BF16 weight gradient: 128 threads, 16 lanes
    # per row (8-row blocks), the deterministic backward's chunks, CTA b
    # running the chunks b, b + grid, ... with no claim, the chunk partials
    # summed exactly in integer windows (build {"chunks_per_step": 0,
    # "min_blocks": 8, "out": "bf16", "prefetch_ahead": 0, "schedule":
    # "static", "slots": 4}); where every chunk has a CTA of a grid of 8 CTAs
    # per SM.
    "cake_rmsnorm_silu_gate_train_dbd89240614cc19b630d": {
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
            ["buffer", "dw_win"],
            ["buffer", "dw_spec"],
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
            ["parameter", "blocks_per_chunk"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunk_limit": 8192,
            "ctas_per_sm": 8,
            "kind": "chunks",
            "rows_per_block": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_dbd89240614cc19b630d",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_dbd89240614cc19b630d",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 15616,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_dbd89240614cc19b630d",
        "logical": {
            "accumulate": "accumulate",
            "blocks_per_chunk": "blocks_per_chunk",
            "dw": "dw",
            "dw_done": "dw_done",
            "dw_spec": "dw_spec",
            "dw_win": "dw_win",
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
        "workspace": {
            "dw_done": 2,
            "dw_spec": 128,
            "dw_win": 3584,
        },
    },
    # The claimed backward into a BF16 weight gradient: the static backward's
    # rows and sums, its CTAs claiming runs of chunks from the claim counter,
    # thread 0 warming the rows of the block two ahead in L2 (build
    # {"chunks_per_step": 0, "min_blocks": 8, "out": "bf16", "prefetch_ahead":
    # 2, "schedule": "claimed", "slots": 4}); where the rows have more chunks
    # than 8 per SM.
    "cake_rmsnorm_silu_gate_train_a654fd29fe5e116dee0c": {
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
            ["buffer", "dw_win"],
            ["buffer", "dw_spec"],
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
            ["parameter", "blocks_per_chunk"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunk_limit": 8192,
            "ctas_per_sm": 8,
            "kind": "chunks",
            "more_chunks_than_per_sm": 8,
            "rows_per_block": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_a654fd29fe5e116dee0c",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_a654fd29fe5e116dee0c",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 15616,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_a654fd29fe5e116dee0c",
        "logical": {
            "accumulate": "accumulate",
            "blocks_per_chunk": "blocks_per_chunk",
            "dw": "dw",
            "dw_done": "dw_done",
            "dw_spec": "dw_spec",
            "dw_win": "dw_win",
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
        "workspace": {
            "dw_done": 2,
            "dw_spec": 128,
            "dw_win": 3584,
        },
    },
    # The exact small-row backward into a BF16 weight gradient (or none, with
    # compute_grad_weight=False):
    # one cluster of 16 CTAs of 4 teams of the deterministic backward's
    # 128-thread CTA (team q of CTA r runs the 8-row block r + 16q), dx, dz
    # and the weight gradient the deterministic pair's bits, CTA r summing
    # the columns 8r to 8r + 7 in the pair's reduce order in FP64, no
    # workspace (build {"out": "bf16", "teams": 4}), up to 512 rows;
    # BACKWARD_BANDS gives it its rows.
    "cake_rmsnorm_silu_gate_train_f6a3420d9aaa20a99e8d": {
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
            "max_rows": 512,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_f6a3420d9aaa20a99e8d",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_f6a3420d9aaa20a99e8d",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (16, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 19584,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_f6a3420d9aaa20a99e8d",
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
        "role": "bwd_bf16_t512",
    },
    # The exact small-row backward into a BF16 weight gradient (or none, with
    # compute_grad_weight=False):
    # one cluster of 16 CTAs of 8 teams of the deterministic backward's
    # 128-thread CTA (team q of CTA r runs the 8-row block r + 16q), dx, dz
    # and the weight gradient the deterministic pair's bits, CTA r summing
    # the columns 8r to 8r + 7 in the pair's reduce order in FP64, no
    # workspace (build {"out": "bf16", "teams": 8}), up to 1024 rows;
    # BACKWARD_BANDS gives it its rows.
    "cake_rmsnorm_silu_gate_train_770965c403d4b2946fba": {
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
            "max_rows": 1024,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_770965c403d4b2946fba",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_770965c403d4b2946fba",
        "launch": {
            "block": (1024, 1, 1),
            "cluster": (16, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 38144,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_770965c403d4b2946fba",
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
        "role": "bwd_bf16_t1024",
    },
    # The static backward into a FP32 weight gradient: the BF16 static
    # backward's schedule and sums, the weight gradient rounded once to FP32
    # (build {"chunks_per_step": 0, "min_blocks": 8, "out": "f32",
    # "prefetch_ahead": 0, "schedule": "static", "slots": 4}).
    "cake_rmsnorm_silu_gate_train_b978b545dca0d0ccafbe": {
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
            ["buffer", "dw_win"],
            ["buffer", "dw_spec"],
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
            ["parameter", "blocks_per_chunk"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunk_limit": 8192,
            "ctas_per_sm": 8,
            "kind": "chunks",
            "rows_per_block": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_b978b545dca0d0ccafbe",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_b978b545dca0d0ccafbe",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 15616,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_b978b545dca0d0ccafbe",
        "logical": {
            "accumulate": "accumulate",
            "blocks_per_chunk": "blocks_per_chunk",
            "dw": "dw",
            "dw_done": "dw_done",
            "dw_spec": "dw_spec",
            "dw_win": "dw_win",
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
        "workspace": {
            "dw_done": 2,
            "dw_spec": 128,
            "dw_win": 3584,
        },
    },
    # The claimed backward into a FP32 weight gradient: the BF16 claimed
    # backward's schedule and sums, the weight gradient rounded once to FP32
    # (build {"chunks_per_step": 0, "min_blocks": 8, "out": "f32",
    # "prefetch_ahead": 2, "schedule": "claimed", "slots": 4}).
    "cake_rmsnorm_silu_gate_train_7e2a23cc1b1359f7b776": {
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
            ["buffer", "dw_win"],
            ["buffer", "dw_spec"],
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
            ["parameter", "blocks_per_chunk"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "chunk_limit": 8192,
            "ctas_per_sm": 8,
            "kind": "chunks",
            "more_chunks_than_per_sm": 8,
            "rows_per_block": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_7e2a23cc1b1359f7b776",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_7e2a23cc1b1359f7b776",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 15616,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_7e2a23cc1b1359f7b776",
        "logical": {
            "accumulate": "accumulate",
            "blocks_per_chunk": "blocks_per_chunk",
            "dw": "dw",
            "dw_done": "dw_done",
            "dw_spec": "dw_spec",
            "dw_win": "dw_win",
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
        "workspace": {
            "dw_done": 2,
            "dw_spec": 128,
            "dw_win": 3584,
        },
    },
    # The exact small-row backward into a FP32 weight gradient:
    # one cluster of 16 CTAs of 4 teams of the deterministic backward's
    # 128-thread CTA (team q of CTA r runs the 8-row block r + 16q), dx, dz
    # and the weight gradient the deterministic pair's bits, CTA r summing
    # the columns 8r to 8r + 7 in the pair's reduce order in FP64, no
    # workspace (build {"out": "f32", "teams": 4}), up to 512 rows;
    # BACKWARD_BANDS gives it its rows.
    "cake_rmsnorm_silu_gate_train_348a297be36a61e8f015": {
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
            "max_rows": 512,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_348a297be36a61e8f015",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_348a297be36a61e8f015",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (16, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 19584,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_348a297be36a61e8f015",
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
        "role": "bwd_f32_t512",
    },
    # The exact small-row backward into a FP32 weight gradient:
    # one cluster of 16 CTAs of 8 teams of the deterministic backward's
    # 128-thread CTA (team q of CTA r runs the 8-row block r + 16q), dx, dz
    # and the weight gradient the deterministic pair's bits, CTA r summing
    # the columns 8r to 8r + 7 in the pair's reduce order in FP64, no
    # workspace (build {"out": "f32", "teams": 8}), up to 1024 rows;
    # BACKWARD_BANDS gives it its rows.
    "cake_rmsnorm_silu_gate_train_717a139d2090d6a610e6": {
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
            "max_rows": 1024,
        },
        "kernel": "cake_rmsnorm_silu_gate_train_717a139d2090d6a610e6",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_train_717a139d2090d6a610e6",
        "launch": {
            "block": (1024, 1, 1),
            "cluster": (16, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 38144,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_train_717a139d2090d6a610e6",
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
        "role": "bwd_f32_t1024",
    },
}

ROUTES: dict[str, dict[str, str]] = {
    "sm_100a": {
        "fwd": "cake_rmsnorm_silu_gate_train_c7919260b39b69e4e646",
        "fwd_xlong": "cake_rmsnorm_silu_gate_train_83fe9e25ce0d01cc3ab1",
        "fwd_long": "cake_rmsnorm_silu_gate_train_c6151f64f22ae65a48eb",
        "fwd_mid": "cake_rmsnorm_silu_gate_train_cf7a87977aaad1a9822f",
        "fwd_short": "cake_rmsnorm_silu_gate_train_3cb6aa31e140d9b13052",
        "fwd_tiny": "cake_rmsnorm_silu_gate_train_04bb67629537ef270bd4",
        "bwd_bf16": "cake_rmsnorm_silu_gate_train_dbd89240614cc19b630d",
        "bwd_bf16_mid": "cake_rmsnorm_silu_gate_train_a654fd29fe5e116dee0c",
        "bwd_bf16_t512": "cake_rmsnorm_silu_gate_train_f6a3420d9aaa20a99e8d",
        "bwd_bf16_t1024": "cake_rmsnorm_silu_gate_train_770965c403d4b2946fba",
        "bwd_f32": "cake_rmsnorm_silu_gate_train_b978b545dca0d0ccafbe",
        "bwd_f32_mid": "cake_rmsnorm_silu_gate_train_7e2a23cc1b1359f7b776",
        "bwd_f32_t512": "cake_rmsnorm_silu_gate_train_348a297be36a61e8f015",
        "bwd_f32_t1024": "cake_rmsnorm_silu_gate_train_717a139d2090d6a610e6",
    },
    "sm_103a": {
        "fwd": "cake_rmsnorm_silu_gate_train_c7919260b39b69e4e646",
        "fwd_xlong": "cake_rmsnorm_silu_gate_train_83fe9e25ce0d01cc3ab1",
        "fwd_mid": "cake_rmsnorm_silu_gate_train_cf7a87977aaad1a9822f",
        "fwd_short": "cake_rmsnorm_silu_gate_train_3cb6aa31e140d9b13052",
        "fwd_tiny": "cake_rmsnorm_silu_gate_train_04bb67629537ef270bd4",
        "bwd_bf16": "cake_rmsnorm_silu_gate_train_dbd89240614cc19b630d",
        "bwd_bf16_mid": "cake_rmsnorm_silu_gate_train_a654fd29fe5e116dee0c",
        "bwd_bf16_t512": "cake_rmsnorm_silu_gate_train_f6a3420d9aaa20a99e8d",
        "bwd_bf16_t1024": "cake_rmsnorm_silu_gate_train_770965c403d4b2946fba",
        "bwd_f32": "cake_rmsnorm_silu_gate_train_b978b545dca0d0ccafbe",
        "bwd_f32_mid": "cake_rmsnorm_silu_gate_train_7e2a23cc1b1359f7b776",
        "bwd_f32_t512": "cake_rmsnorm_silu_gate_train_348a297be36a61e8f015",
        "bwd_f32_t1024": "cake_rmsnorm_silu_gate_train_717a139d2090d6a610e6",
    },
    "sm_107a": {
        "fwd": "cake_rmsnorm_silu_gate_train_c7919260b39b69e4e646",
        "fwd_xlong": "cake_rmsnorm_silu_gate_train_83fe9e25ce0d01cc3ab1",
        "fwd_mid": "cake_rmsnorm_silu_gate_train_cf7a87977aaad1a9822f",
        "fwd_short": "cake_rmsnorm_silu_gate_train_3cb6aa31e140d9b13052",
        "fwd_tiny": "cake_rmsnorm_silu_gate_train_04bb67629537ef270bd4",
        "bwd_bf16": "cake_rmsnorm_silu_gate_train_dbd89240614cc19b630d",
        "bwd_bf16_mid": "cake_rmsnorm_silu_gate_train_a654fd29fe5e116dee0c",
        "bwd_bf16_t512": "cake_rmsnorm_silu_gate_train_f6a3420d9aaa20a99e8d",
        "bwd_bf16_t1024": "cake_rmsnorm_silu_gate_train_770965c403d4b2946fba",
        "bwd_f32": "cake_rmsnorm_silu_gate_train_b978b545dca0d0ccafbe",
        "bwd_f32_mid": "cake_rmsnorm_silu_gate_train_7e2a23cc1b1359f7b776",
        "bwd_f32_t512": "cake_rmsnorm_silu_gate_train_348a297be36a61e8f015",
        "bwd_f32_t1024": "cake_rmsnorm_silu_gate_train_717a139d2090d6a610e6",
    },
}

# The forward module of each architecture by the rows per SM of a call of
# packed rows: ``(last, role)`` bands in increasing ``last`` (``None``: no
# end); a call of ``rows`` rows on a device of ``sm_count`` SMs runs the
# module of the first band's routed ``role`` with ``rows <= last *
# sm_count``.  The cuts are the measured fastest program at each rows per SM
# (CUPTI, cold L2): the tiny forward up to 140 rows per SM on sm_100a, 120 on
# sm_103a and 65 on sm_107a, where beside a kernel holding 16 SMs it ends no
# later than the deterministic package's plain forward (one CTA per 8-row
# block, so the free SMs take the CTAs the held ones cannot run).  Host plan
# only: a new cut needs no new program.
FORWARD_BANDS: dict[str, tuple[tuple[int | None, str], ...]] = {
    "sm_100a": (
        (140, "fwd_tiny"),
        (400, "fwd_long"),
        (1000, "fwd_xlong"),
        (None, "fwd_mid"),
    ),
    "sm_103a": (
        (120, "fwd_tiny"),
        (250, "fwd_mid"),
        (550, "fwd_xlong"),
        (None, "fwd_mid"),
    ),
    "sm_107a": (
        (65, "fwd_tiny"),
        (200, "fwd_mid"),
        (520, "fwd_xlong"),
        (None, "fwd_mid"),
    ),
}

# The forward bands of a call whose ``x`` or ``z`` rows are not packed (the
# tiny forward reads packed rows only), as ``FORWARD_BANDS``.  The cuts are
# the measured fastest program at each rows per SM (CUPTI, cold L2), except
# the short forward's end: 12 rows per SM on sm_100a and sm_103a, where its
# grid of one 16-row block a CTA has at most as many CTAs as the SMs less 16
# (from 64 SMs), and 59 on sm_107a, where it fits one wave of 4 CTAs per SM on
# all SMs but 16 (from 205 SMs), so beside a kernel holding 16 SMs it ends no
# later than the plain forward.
STRIDED_FORWARD_BANDS: dict[str, tuple[tuple[int | None, str], ...]] = {
    "sm_100a": (
        (12, "fwd_short"),
        (140, "fwd_mid"),
        (400, "fwd_long"),
        (1000, "fwd_xlong"),
        (None, "fwd_mid"),
    ),
    "sm_103a": (
        (12, "fwd_short"),
        (250, "fwd_mid"),
        (550, "fwd_xlong"),
        (None, "fwd_mid"),
    ),
    "sm_107a": (
        (59, "fwd_short"),
        (200, "fwd_mid"),
        (520, "fwd_xlong"),
        (None, "fwd_mid"),
    ),
}

# The small-row backward programs of each backward role on each architecture,
# by the rows of a call: ``(last, role)`` bands in increasing ``last``; a call
# of ``rows`` rows runs the module of the first band's routed ``role`` with
# ``rows <= last`` (``last`` at most the module's ``max_rows``), and past the
# last band the static or the claimed backward.  The cuts are the measured
# fastest exact tiny backward at each row count (CUPTI, cold L2): 4 teams
# (``_t512``) or 8 (``_t1024``).  Host plan only: a new cut needs no new
# program.
BACKWARD_BANDS: dict[str, dict[str, tuple[tuple[int, str], ...]]] = {
    "sm_100a": {
        "bwd_bf16": ((512, "bwd_bf16_t512"), (1024, "bwd_bf16_t1024")),
        "bwd_f32": ((512, "bwd_f32_t512"), (1024, "bwd_f32_t1024")),
    },
    "sm_103a": {
        "bwd_bf16": ((512, "bwd_bf16_t512"), (620, "bwd_bf16_t1024")),
        "bwd_f32": ((512, "bwd_f32_t512"), (720, "bwd_f32_t1024")),
    },
    "sm_107a": {
        "bwd_bf16": ((512, "bwd_bf16_t512"), (1024, "bwd_bf16_t1024")),
        "bwd_f32": ((512, "bwd_f32_t512"), (900, "bwd_f32_t1024")),
    },
}


ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
# The kernels of a call, routed on every architecture.  The forward's other
# roles (``fwd_short``, ``fwd_mid``, ``fwd_long``, ``fwd_xlong``) run in the
# rows of their ``FORWARD_BANDS`` bands; a backward role's small-row roles
# (``BAND_ROLES``) in those of its ``BACKWARD_BANDS`` bands and its mid role
# (``<role>_mid``, the claimed backward) past them where the rows have more
# chunks than its rule's ``more_chunks_than_per_sm`` per SM.
ROLES = ("fwd", "bwd_bf16", "bwd_f32")
# The backward's small-row roles, which ``BACKWARD_BANDS`` routes by the rows.
BAND_ROLES = (
    "bwd_bf16_t512",
    "bwd_bf16_t1024",
    "bwd_f32_t512",
    "bwd_f32_t1024",
)
GRID_KINDS = ("row_blocks", "persistent_row_blocks", "chunks", "cluster")
# The backward's workspace buffers in their order in one workspace, each with
# the bytes of one element: the u64 windows, the FP32 class words and the u32
# arrival ticket and claim counter.
WORKSPACE_BUFFERS: tuple[tuple[str, int], ...] = (
    ("dw_win", 8),
    ("dw_spec", 4),
    ("dw_done", 4),
)
# Each workspace buffer starts at a multiple of this many bytes.
WORKSPACE_ALIGNMENT = 16
# Elements of one normalized row.
HEAD_DIM = 128
# The persistent forwards read rows with 32-byte vector accesses (the forward
# and the backward with 16-byte ones): a tensor is read in place when its
# storage offset and its token and head strides are multiples of this many
# BF16 elements.
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


def backward_chunks(name: str, rows: int) -> tuple[int, int]:
    """``(blocks_per_chunk, chunks)`` of the static or claimed backward module
    ``name`` for ``rows`` rows: blocks of ``rows_per_block`` rows,
    ``blocks_per_chunk = max(1, ceil(blocks / chunk_limit))`` consecutive
    blocks a chunk (the deterministic backward's chunks) and ``ceil(blocks /
    blocks_per_chunk)`` chunks."""

    rule = MODULES[name]["grid_rule"]
    if rule["kind"] != "chunks":
        raise ValueError(f"module {name} does not cut its rows into chunks")
    blocks = -(-int(rows) // int(rule["rows_per_block"]))
    per_chunk = max(1, -(-blocks // int(rule["chunk_limit"])))
    return per_chunk, -(-blocks // per_chunk)


def _claims(name: str, rows: int, sm_count: int) -> bool:
    """Whether the claimed backward ``name`` runs for ``rows`` rows: where the
    rows have more chunks than its rule's ``more_chunks_than_per_sm`` per
    SM."""

    rule = MODULES[name]["grid_rule"]
    chunks = backward_chunks(name, rows)[1]
    return chunks > int(rule["more_chunks_than_per_sm"]) * int(sm_count)


def module_for_rows(
    arch: str, role: str, rows: int, sm_count: int, *, packed: bool = True
) -> str:
    """The module that runs ``role`` (one of ``ROLES``) for ``rows`` rows on
    ``arch`` with ``sm_count`` SMs: for the forward the module of the first of
    its ``FORWARD_BANDS[arch]`` bands (``STRIDED_FORWARD_BANDS[arch]`` when
    ``packed`` is false: ``x`` or ``z`` does not hold row ``n`` at element
    ``128 n``) with ``rows <= last * sm_count`` (or no ``last``); for a
    backward role the module of the first of its
    ``BACKWARD_BANDS[arch]`` bands with ``rows <= last``, else its mid module
    (the claimed backward) when ``arch`` routes one and the rows have more
    than ``more_chunks_than_per_sm * sm_count`` chunks; else
    :func:`route_module`."""

    if role == "fwd":
        table = FORWARD_BANDS if packed else STRIDED_FORWARD_BANDS
        for last, band in table.get(arch, ()):
            if last is None or int(rows) <= int(last) * int(sm_count):
                return route_module(arch, band)
        return route_module(arch, role)
    for last, band in BACKWARD_BANDS.get(arch, {}).get(role, ()):
        if int(rows) <= int(last):
            return route_module(arch, band)
    claimed = ROUTES.get(arch, {}).get(f"{role}_mid")
    if claimed is not None and _claims(claimed, rows, sm_count):
        return claimed
    return route_module(arch, role)


def backward_ctas(name: str, rows: int, sm_count: int) -> int:
    """CTAs of the backward module ``name`` for ``rows`` rows on a device with
    ``sm_count`` SMs: the static and the claimed backward ``min(C,
    ctas_per_sm * sm_count)`` for ``C`` chunks (:func:`backward_chunks`), so
    every chunk has a CTA while ``C`` fits one wave; a small-row backward its
    cluster's ``cluster`` CTAs whatever the rows."""

    rule = MODULES[name]["grid_rule"]
    if rule["kind"] == "cluster":
        return int(rule["cluster"])
    if rule["kind"] != "chunks":
        raise ValueError(f"module {name} is not a backward module")
    chunks = backward_chunks(name, rows)[1]
    return min(chunks, max(1, int(rule["ctas_per_sm"]) * int(sm_count)))


def workspace_layout(name: str) -> tuple[tuple[str, int, int], ...]:
    """``(buffer, byte offset, elements)`` of each workspace buffer of the
    backward module ``name`` in ``WORKSPACE_BUFFERS`` order, each starting at
    a multiple of ``WORKSPACE_ALIGNMENT`` bytes; empty for a small-row
    backward, which sums the weight gradient on chip and takes no workspace."""

    counts = MODULES[name].get("workspace")
    if not counts:
        return ()
    layout = []
    offset = 0
    for buffer, size in WORKSPACE_BUFFERS:
        offset = -(-offset // WORKSPACE_ALIGNMENT) * WORKSPACE_ALIGNMENT
        layout.append((buffer, offset, int(counts[buffer])))
        offset += int(counts[buffer]) * size
    return tuple(layout)


def workspace_bytes(name: str) -> int:
    """Bytes of the workspace of the backward module ``name``: its buffers
    (:func:`workspace_layout`) end to end; 0 for a small-row backward."""

    sizes = dict(WORKSPACE_BUFFERS)
    return max(
        (
            offset + elements * sizes[buffer]
            for buffer, offset, elements in workspace_layout(name)
        ),
        default=0,
    )


def launch_grid(
    record: dict[str, Any], *, rows: int, sm_count: int, ctas: int = 0
) -> tuple[int, int, int]:
    """The grid of one module record for ``rows`` rows on a device with
    ``sm_count`` SMs (``ctas``: the backward's CTA count,
    :func:`backward_ctas`).  The long and mid forwards take the fewest whole
    waves of ``ctas_per_sm * sm_count`` CTAs whose CTAs walk at most
    ``max_blocks_per_cta`` blocks of ``rows_per_block`` rows each, and at most
    one CTA per block.  A small-row backward takes one cluster of
    ``cluster`` CTAs."""

    rule = record["grid_rule"]
    kind = rule["kind"]
    if kind == "row_blocks":
        return (-(-int(rows) // int(rule["rows_per_cta"])), 1, 1)
    if kind == "persistent_row_blocks":
        blocks = -(-int(rows) // int(rule["rows_per_block"]))
        wave = max(1, int(rule["ctas_per_sm"]) * int(sm_count))
        waves = max(1, -(-blocks // (wave * int(rule["max_blocks_per_cta"]))))
        return (min(blocks, waves * wave), 1, 1)
    if kind == "chunks":
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
    "BACKWARD_BANDS",
    "BAND_ROLES",
    "FORWARD_BANDS",
    "GRID_KINDS",
    "HEAD_DIM",
    "INT32_MAX",
    "MODULES",
    "ROLES",
    "ROUTES",
    "STRIDED_FORWARD_BANDS",
    "WORKSPACE_ALIGNMENT",
    "WORKSPACE_BUFFERS",
    "arch_for_capability",
    "backward_chunks",
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
    "workspace_bytes",
    "workspace_layout",
]
