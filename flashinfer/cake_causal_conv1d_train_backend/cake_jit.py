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

A route lists every program one public call may select.  The forward routes
produce ``y`` from ``x``, ``weight`` and ``cu_seqlens``: ``fwd`` (FP32 taps),
``fwd_dt`` (BF16 taps) and ``fwd_bias`` (a per-channel bias, FP32 or BF16 taps
and bias).  The backward routes produce ``dx`` and the deterministic
``dweight`` (and ``dbias``): ``bwd`` (FP32 taps and
gradient, SiLU' at the BF16-rounded preactivation), ``bwd_dt`` (BF16 taps and
/ or a BF16 ``dweight``), ``bwd_bias``, ``bwd_fp32g`` (SiLU' at the unrounded
FP32 preactivation for the weight gradient) and ``bwd_bias_fp32g``;
:func:`forward_route` / :func:`backward_route` map a call's arguments to its
route and the ``flags`` bit mask (``FLAG_*``) tells a route's programs which
of ``w``, ``dw``, ``b``, ``db`` are BF16.  A route, forward or backward, may
carry several programs of its kernel and selects exactly one per call,
chosen by the channel count (and, for one record, the token count): a
record's ``select_rule`` (``channels_at_least``: ``C >= channels[arch]``;
``channels_tokens_at_least``: that and ``S >= tokens[arch]``) names the
calls it owns, of several owners the one of the highest channel threshold
runs, the record without a rule is the route's fallback, and
:func:`selected_modules` applies that rule (a pure function of the
architecture, the route, ``C`` and ``S``; the host never reads the data).
Token and channel counts are runtime scalars, so a changed problem size never
recompiles anything; building every library of the architecture
(:func:`specs_for_arch`) before a CUDA Graph capture keeps the JIT out of the
capture.

A forward call is one ordinary launch of the program its route selects, a
dynamic token-chunk forward (round 9): the co-resident CTAs of a channel tile
take the tile's token chunks in order from a device counter, so the CTAs that
finish first take the tail instead of an equal share fixed at launch.  Each
forward route ships the program of one eight-token stage per sub-tile and
chunk for every channel count on sm_100a and sm_103a and, on sm_107a, below
a per-route channel count (the fallback); from that count (C = 256 on
``fwd``, 512 on ``fwd_dt``, 1024 on ``fwd_bias``) sm_107a runs the program
of two stages per chunk.  The counters
are the forward workspace ``work`` (``workspace_rule`` ``tile_counters``:
:func:`counter_words`, two int32 words per channel tile), zero before the first
launch; the last CTA of each tile resets them, so the buffer is zero after
every launch and every CUDA Graph replay and serves every later launch of one
stream.  Two launches that may run at once (two streams, two graphs replayed
concurrently) need two buffers.

A backward call is two ordinary launches on one stream.  The selected program
(persistent, ``grid_mult x ctas_per_sm x SM count`` CTAs) writes ``dx`` and
one FP32 partial row of ``dweight`` (and, on the bias routes, of ``dbias``)
per CTA into the workspace and signals its dependents; then the route's reduce
program (``record["reduce"]``), launched programmatically dependent on it
(``launch.use_pdl``: its binding sets the programmatic-stream-serialization
attribute, and the kernel waits for the main grid before its first load), sums
the rows per channel in a fixed order and stores ``dw`` (and ``db``) in the
dtypes ``flags`` names.  Until round 6 the backward was one cooperative launch
whose grid-wide barrier preceded the same reduction in-kernel.  A cooperative
grid cannot start while any kernel is resident on another stream -- NCCL
collectives are, for most of a training backward (FlashInfer issue #7038) --
whereas an ordinary launch whose CTAs find no free slot runs them in a later
wave.  ``grid_mult`` (``grid_rule["grid_mult"][arch]``) over-decomposes the
token axis so that such a CTA trails by ``1 / grid_mult`` of a stream rather
than a whole one; the partial rows grow with it.  The reduction order is
unchanged, so ``dweight`` is deterministic for a given ``(S, C, device)`` and
route and, at ``grid_mult`` 1, bitwise what the cooperative form produced.
Each backward route has its own reduce program: the default one, and builds
of it with ``dtype_flags`` (the ``dw`` store form), ``bias`` (the fifth, dbias,
slab and ``db``) and ``fp64`` (the ``_fp32g`` routes' FP64 sums).

The host plan is the three functions :func:`plan` (the kernel scalars, with
the channel tile rule :func:`channel_split` of the record's role: the forward
and the backward rules differ where no multiple of 32 divides the channel
vectors), :func:`launch_grid` (the grid:
``co_resident`` records launch ``ctas_per_sm x SM count`` CTAs, ``occupancy``
records ``grid_mult x ctas_per_sm x SM count``, less the CTAs
:func:`main_grid` drops to whole channel-tile ranks on the rows it names,
``channels_per_cta`` records ``ceil(C / channels_per_cta)``),
:func:`partial_rows` (the backward
workspace rows) and :func:`counter_words` (the forward's ``work`` words);
everything else derives from them.  The public workspace is
sized by the backward program of the largest ``grid_mult x ctas_per_sm x
taps`` over every backward route (:func:`workspace_record`, the capacity
owner; the bias programs' partials carry a fifth, dbias, slab); each program
writes the leading rows of its own grid and its reduce program sums exactly
those.  The forward workspace is sized by the forward program of the most
counter words over every forward route (:func:`forward_workspace_words`).

The program name ``cake_causal_conv1d_train_<20 hex>`` is the first twenty hex
digits of the SHA-256 of the compact JSON ``{"build": <build keywords, sorted>,
"module": <dotted kernel module>}`` of the kernel the program was rendered
from; the default reduce program is built with its kernel file's default
keywords and the forward programs with ``fast=True, sched="dynamic",
chunk_stages=1, pair_math=True`` (the dynamic token-chunk schedule of one
eight-token stage per sub-tile and chunk, the f32x2 pair math, the
branch-free in-document stage; ``y`` is bitwise the default build's) plus
the route's keywords, the sm_107a programs the channel count selects with
``chunk_stages=2`` instead (two eight-token stages per sub-tile and chunk);
every backward program carries ``revision=6`` (the
build revision of the split that retired the cooperative form; the exporter
hashes it so the split programs replaced the cooperative ones under new names);
the default route's fallback is built with ``vec=8, copy="tma2d",
stage_pair=True`` (eight channels, 16-byte vectors, per thread; 2-D TMA row
copies; each eight-token copy stage consumed as two four-token halves, the
dweight fold buffer in the stage ring) on sm_100a and sm_103a and with
``min_blocks=2`` (two CTAs per SM) beside those on sm_107a, where for
256 <= C < 512 from S = 262144 tokens the build of ``vec=8, min_blocks=2``
(four-token copy stages, no ``stage_pair``) runs instead, and the backwards
the ``channels_at_least`` rules select, builds of ``vec=8, copy="tma2d",
stage_tokens=8, min_blocks=2`` (eight-token stages, two CTAs per SM): as is
on sm_100a from C = 512 and on sm_103a from 768, with ``fold_overlap=True,
fast_roll=True`` (the dweight fold buffer over the stage slots, the stage body
as two rolled passes of four tokens) on sm_103a for 512 <= C < 768 and on
sm_107a for 512 <= C < 1024, as is on sm_107a for 1024 <= C < 2048, and
with ``fast_roll=True, roll_prefetch=True`` (each rolled pass loads the
next pass's rows before it computes) on sm_107a from 2048.  The programs of
the six other routes are the same bodies built with ``dtype_flags=True`` (the
``flags`` scalar; untyped tap / bias pointers) plus ``bias=True`` for the
bias routes and ``grad_pre="fp32"`` for the ``_fp32g`` routes, their backward
fallbacks without ``stage_pair`` (``vec=8, copy="tma2d"`` on sm_100a and
sm_103a, ``vec=8, min_blocks=2`` on sm_107a); each record's comment names its
build.
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
    # Route ``fwd`` (round 9): the forward's dynamic token-chunk program with the
    # f32x2 pair math and the branch-free in-document stage, built with ``fast=True,
    # sched="dynamic", chunk_stages=1, pair_math=True``.  The CTAs of a channel tile
    # take chunks of ``n_sub x 8`` tokens from a device counter instead of equal
    # contiguous shares, so the CTAs of the SMs that draw the most bandwidth take
    # the tail; ``y`` is bitwise the static program's.  The counters are the
    # ``work`` buffer (``workspace_rule``: int32, two words per channel tile: the
    # next-chunk counter and the done-CTA counter), zero before the first launch;
    # the last CTA of each tile resets them, so ``work`` is zero again after every
    # launch and every CUDA Graph replay, and one buffer serves every later launch
    # of one stream (two launches that may overlap need two buffers).  Where a CTA
    # holds several sub-tiles, a chunk's three history rows come from the stage's
    # slot (the halo rows its copy brings), not from global memory.  It is the
    # route's fallback (no ``select_rule``): every channel count on sm_100a and
    # sm_103a, and C < 256 on sm_107a.
    # ``dynamic_smem_bytes`` 0 is the placeholder of a program whose shared memory follows
    # the kernel (the exporter's ``KERNEL_SMEM_PROGRAMS``): the first render writes the
    # kernel's figure, the kernel symbol, the argument plan and the logical names it keeps.
    "cake_causal_conv1d_train_f5497330bfd286211643": {
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
            ["buffer", "work"],
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
        # CTAs of that tile, claims the tile's token chunks in order from
        # ``work`` (a CTA's chunks rise; its first chunk past ``S`` ends it).
        # Every channel tile needs a CTA (the host refuses a ``C`` of more
        # tiles than the grid).  ``ctas_per_sm`` 4 is the co-residency of the
        # program (128 registers at 128 threads, no stack) on B200 (sm_100a),
        # GB300 (sm_103a) and R200 (sm_107a).  ``geometry`` is the kernel's
        # build geometry the host plan derives its scalars from.
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_f5497330bfd286211643",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_f5497330bfd286211643",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 46208,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_f5497330bfd286211643",
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
            "work": "work",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "workspace_rule": {
            "kind": "tile_counters",
            "words_per_tile": 2,
        },
    },
    "cake_causal_conv1d_train_9bfa65707a49df6a48ce": {
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
            ["buffer", "partial"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        # An ordinary launch of ``grid_mult x ctas_per_sm x SM count`` CTAs:
        # ``ctas_per_sm`` is the measured co-residency, ``grid_mult`` (per
        # architecture) splits the token streams that many times finer so a
        # CTA another stream's kernel displaces trails by ``1 / grid_mult`` of
        # a stream instead of a whole one.  Each CTA writes one partial row;
        # ``reduce`` sums them.  The grid is a function of the device alone.
        # The ``stage_pair`` build: each 8-token copy stage is consumed as two
        # 4-token halves, the dweight fold buffer in the stage ring.
        # ``ctas_per_sm``: 3 CTAs per SM (168 registers at 128 threads, the
        # launch bound; ncu launch__occupancy_limit_registers), the occupancy
        # query's answer on B200 and GB300.
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 3,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_9bfa65707a49df6a48ce",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_9bfa65707a49df6a48ce",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 66560,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_9bfa65707a49df6a48ce",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "x_map": "x",
        },
        "reduce": "cake_causal_conv1d_train_86b5ecfbd90aca4b52e5",
        "role": "bwd",
        # The FP32 ``partial`` workspace: ``[ceil(grid / n_ch_tiles), C, taps]``
        # rows (row ``b // n_ch_tiles`` belongs to CTA ``b``) the kernel fully
        # writes and ``reduce`` sums in a fixed order; nothing is zeroed by the
        # host.
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 4,
        },
    },
    "cake_causal_conv1d_train_b4f7ff5a3dd9ddd2f41c": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "partial"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        # An ordinary launch of ``grid_mult x ctas_per_sm x SM count`` CTAs:
        # ``ctas_per_sm`` is the measured co-residency, ``grid_mult`` (per
        # architecture) splits the token streams that many times finer so a
        # CTA another stream's kernel displaces trails by ``1 / grid_mult`` of
        # a stream instead of a whole one.  Each CTA writes one partial row;
        # ``reduce`` sums them.  The grid is a function of the device alone.
        # The ``copy="tma2d", stage_pair`` build of the sm_100a / sm_103a
        # fallback at ``min_blocks`` 2: 2-D TMA stage copies, each 8-token
        # stage consumed as two 4-token halves, the dweight fold buffer in the
        # stage ring.  ``ctas_per_sm``: 2 CTAs per SM, the program's
        # ``min_blocks`` 2 launch bound (ncu launch__occupancy_limit_registers),
        # the occupancy query's answer on R200; the ``min_blocks`` 3 build of
        # the other two architectures measured 10 % slower at C = 128 on its
        # 212 SMs.
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_b4f7ff5a3dd9ddd2f41c",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_b4f7ff5a3dd9ddd2f41c",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 66560,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_b4f7ff5a3dd9ddd2f41c",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "x_map": "x",
        },
        "reduce": "cake_causal_conv1d_train_86b5ecfbd90aca4b52e5",
        "role": "bwd",
        # The FP32 ``partial`` workspace: ``[ceil(grid / n_ch_tiles), C, taps]``
        # rows (row ``b // n_ch_tiles`` belongs to CTA ``b``) the kernel fully
        # writes and ``reduce`` sums in a fixed order; nothing is zeroed by the
        # host.
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 4,
        },
    },
    "cake_causal_conv1d_train_9a07bcaa0223632640ee": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
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
        # An ordinary launch of ``grid_mult x ctas_per_sm x SM count`` CTAs:
        # ``ctas_per_sm`` is the measured co-residency, ``grid_mult`` (per
        # architecture) splits the token streams that many times finer so a
        # CTA another stream's kernel displaces trails by ``1 / grid_mult`` of
        # a stream instead of a whole one.  Each CTA writes one partial row;
        # ``reduce`` sums them.  The grid is a function of the device alone.
        # ``ctas_per_sm``: 2 CTAs per SM, the program's ``min_blocks`` 2 launch
        # bound (234 registers at 128 threads; ncu
        # launch__occupancy_limit_registers), the occupancy query's answer on
        # R200; the ``min_blocks`` 3 build of the other two architectures
        # measured 10 % slower at C = 128 on its 212 SMs.
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_9a07bcaa0223632640ee",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_9a07bcaa0223632640ee",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 50176,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_9a07bcaa0223632640ee",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
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
        "reduce": "cake_causal_conv1d_train_86b5ecfbd90aca4b52e5",
        "role": "bwd",
        # The four-token build of the sm_107a fallback runs for 256 <= C < 512
        # from S = 262144 tokens; the ``stage_pair`` fallback runs every other
        # C < 512 (on those rows its gain was not distinguishable from the
        # spread between identical copies of one build).
        "select_rule": {
            "channels": {
                "sm_107a": 256,
            },
            "kind": "channels_tokens_at_least",
            "tokens": {
                "sm_107a": 262144,
            },
        },
        # The FP32 ``partial`` workspace: ``[ceil(grid / n_ch_tiles), C, taps]``
        # rows (row ``b // n_ch_tiles`` belongs to CTA ``b``) the kernel fully
        # writes and ``reduce`` sums in a fixed order; nothing is zeroed by the
        # host.
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 4,
        },
    },
    # Rendered by the exporter from the family adapter's scaffold: the backward
    # with stage_tokens 8, copy tma2d, min_blocks 2 (2 CTAs per SM); it runs when
    # C >= select_rule.channels[arch] and no routed record of a higher threshold
    # admits C; below every threshold the fallback record (no select_rule) runs.
    "cake_causal_conv1d_train_37e428406a0cc4a8ca1b": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "partial"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_37e428406a0cc4a8ca1b",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_37e428406a0cc4a8ca1b",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 82944,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_37e428406a0cc4a8ca1b",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "x_map": "x",
        },
        "reduce": "cake_causal_conv1d_train_86b5ecfbd90aca4b52e5",
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_100a": 512,
                "sm_103a": 768,
                "sm_107a": 1024,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 4,
        },
    },
    # Rendered by the exporter from the family adapter's scaffold: the backward
    # with vec 8, copy tma2d, stage_tokens 8, min_blocks 2, fold_overlap,
    # fast_roll
    # (2 CTAs per SM); it runs when C >= select_rule.channels[arch] and no
    # routed record of a higher threshold admits C; below every threshold the
    # fallback record (no select_rule) runs.
    "cake_causal_conv1d_train_d46036aa2bc5f90bb0f6": {
        "arches": [
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "partial"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_103a": 1,
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_d46036aa2bc5f90bb0f6",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_d46036aa2bc5f90bb0f6",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 66560,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_d46036aa2bc5f90bb0f6",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "x_map": "x",
        },
        "reduce": "cake_causal_conv1d_train_86b5ecfbd90aca4b52e5",
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_103a": 512,
                "sm_107a": 512,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 4,
        },
    },
    # Rendered by the exporter from the family adapter's scaffold: the backward
    # with vec 8, copy tma2d, stage_tokens 8, min_blocks 2, fast_roll,
    # roll_prefetch
    # (2 CTAs per SM); it runs when C >= select_rule.channels[arch] and no
    # routed record of a higher threshold admits C; below every threshold the
    # fallback record (no select_rule) runs.
    "cake_causal_conv1d_train_aeff6106e2822f313a0c": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "partial"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_aeff6106e2822f313a0c",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_aeff6106e2822f313a0c",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 82944,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_aeff6106e2822f313a0c",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "x_map": "x",
        },
        "reduce": "cake_causal_conv1d_train_86b5ecfbd90aca4b52e5",
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_107a": 2048,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 4,
        },
    },
    # Route ``fwd_dt`` (round 9): the dynamic token-chunk forward with
    # ``dtype_flags`` (the ``flags`` scalar selects the tap load form: bit 0 set
    # reads BF16 taps), built with ``dtype_flags=True, fast=True,
    # sched="dynamic", chunk_stages=1, pair_math=True``; ``work`` as the ``fwd``
    # record's.  The route's fallback: every channel count on sm_100a and
    # sm_103a, and C < 512 on sm_107a.
    # ``dynamic_smem_bytes`` 0 is the placeholder of a program whose shared memory follows
    # the kernel (the exporter's ``KERNEL_SMEM_PROGRAMS``): the first render writes the
    # kernel's figure, the kernel symbol, the argument plan and the logical names it keeps.
    "cake_causal_conv1d_train_9c1e7ee9f37e3102660e": {
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
            ["buffer", "work"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_9c1e7ee9f37e3102660e",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_9c1e7ee9f37e3102660e",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 46208,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_9c1e7ee9f37e3102660e",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "workspace_rule": {
            "kind": "tile_counters",
            "words_per_tile": 2,
        },
    },
    # Route ``fwd_bias`` (round 9): the dynamic token-chunk forward with a
    # per-channel bias ``b`` added after the four taps (FLA's order) and
    # ``dtype_flags`` (bit 0: BF16 taps, bit 2: BF16 bias), built with ``bias=True,
    # dtype_flags=True, fast=True, sched="dynamic", chunk_stages=1,
    # pair_math=True``; ``work`` as the ``fwd`` record's.  The route's fallback:
    # every channel count on sm_100a and sm_103a, and C < 1024 on sm_107a.
    # ``dynamic_smem_bytes`` 0 is the placeholder of a program whose shared memory follows
    # the kernel (the exporter's ``KERNEL_SMEM_PROGRAMS``): the first render writes the
    # kernel's figure, the kernel symbol, the argument plan and the logical names it keeps.
    "cake_causal_conv1d_train_08880ca12e4064bc8cdb": {
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
            ["buffer", "b"],
            ["buffer", "work"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_08880ca12e4064bc8cdb",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_08880ca12e4064bc8cdb",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 46208,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_08880ca12e4064bc8cdb",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "workspace_rule": {
            "kind": "tile_counters",
            "words_per_tile": 2,
        },
    },
    # Route ``fwd`` on sm_107a from C = 256 (round 9): the dynamic token-chunk
    # forward of two eight-token stages per sub-tile and chunk, built with
    # ``fast=True, sched="dynamic", chunk_stages=2, pair_math=True``.  On R200 the
    # one-stage program loses to the shipped forward from C = 2048 (one sub-tile
    # per CTA over several channel tiles) and wins at C = 128, where two stages
    # lose; this program runs from the smallest measured C where it beats the
    # one-stage program.  ``work`` as the ``fwd`` record's.
    # ``dynamic_smem_bytes`` 0 is the placeholder of a program whose shared memory follows
    # the kernel (the exporter's ``KERNEL_SMEM_PROGRAMS``): the first render writes the
    # kernel's figure, the kernel symbol, the argument plan and the logical names it keeps.
    "cake_causal_conv1d_train_8665d0756b8240ecb4d0": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["buffer", "work"],
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
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_8665d0756b8240ecb4d0",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_8665d0756b8240ecb4d0",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 33920,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_8665d0756b8240ecb4d0",
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
            "work": "work",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "select_rule": {
            "channels": {
                "sm_107a": 256,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "kind": "tile_counters",
            "words_per_tile": 2,
        },
    },
    # Route ``fwd_dt`` on sm_107a from C = 512 (round 9): the two-stage dynamic
    # token-chunk forward with ``dtype_flags``, built with ``dtype_flags=True,
    # fast=True, sched="dynamic", chunk_stages=2, pair_math=True``; ``work`` as
    # the ``fwd`` record's.
    # ``dynamic_smem_bytes`` 0 is the placeholder of a program whose shared memory follows
    # the kernel (the exporter's ``KERNEL_SMEM_PROGRAMS``): the first render writes the
    # kernel's figure, the kernel symbol, the argument plan and the logical names it keeps.
    "cake_causal_conv1d_train_176ebb3e14f7ad9c94dc": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["buffer", "work"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_176ebb3e14f7ad9c94dc",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_176ebb3e14f7ad9c94dc",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 33920,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_176ebb3e14f7ad9c94dc",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "select_rule": {
            "channels": {
                "sm_107a": 512,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "kind": "tile_counters",
            "words_per_tile": 2,
        },
    },
    # Route ``fwd_bias`` on sm_107a from C = 1024 (round 9): the two-stage dynamic
    # token-chunk forward with the per-channel bias, built with ``bias=True,
    # dtype_flags=True, fast=True, sched="dynamic", chunk_stages=2,
    # pair_math=True``; ``work`` as the ``fwd`` record's.
    # ``dynamic_smem_bytes`` 0 is the placeholder of a program whose shared memory follows
    # the kernel (the exporter's ``KERNEL_SMEM_PROGRAMS``): the first render writes the
    # kernel's figure, the kernel symbol, the argument plan and the logical names it keeps.
    "cake_causal_conv1d_train_d829c95e1d610bc14b23": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["buffer", "b"],
            ["buffer", "work"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_d829c95e1d610bc14b23",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_d829c95e1d610bc14b23",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 33920,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_d829c95e1d610bc14b23",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "select_rule": {
            "channels": {
                "sm_107a": 1024,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "kind": "tile_counters",
            "words_per_tile": 2,
        },
    },
    # Route ``bwd_dt`` (hand-written fallback, sm_100a / sm_103a): the fallback backward body
    # ``vec=8, copy="tma2d"`` with ``dtype_flags=True`` (bit 0: BF16 taps; bit 1, the BF16
    # ``dw`` store, is read by its reduce program); SiLU' at the BF16-rounded preactivation
    # as the default route.  Like every backward program since the split it is an ordinary
    # launch that writes ``dx`` and its partial rows; ``reduce`` names the program that sums
    # them (the ``bwd_dt`` reduce: ``dtype_flags`` for the ``dw`` store form).
    # ``dynamic_smem_bytes`` 0 is the placeholder of a program whose shared memory follows
    # the kernel (the exporter's ``KERNEL_SMEM_PROGRAMS``): the first render writes the
    # kernel's figure, the kernel symbol, the argument plan and the logical names it keeps.
    "cake_causal_conv1d_train_9c8310a1d3fa9b8ccffa": {
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
            ["buffer", "partial"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 3,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_9c8310a1d3fa9b8ccffa",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_9c8310a1d3fa9b8ccffa",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 50176,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_9c8310a1d3fa9b8ccffa",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "x_map": "x",
        },
        "reduce": "cake_causal_conv1d_train_ea8f91038a9dea452ff2",
        "role": "bwd",
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 4,
        },
    },
    # Route ``bwd_dt`` (hand-written fallback, sm_107a): the sm_107a fallback body ``vec=8,
    # min_blocks=2`` with ``dtype_flags=True``.
    # ``dynamic_smem_bytes`` 0 is the placeholder of a program whose shared memory follows
    # the kernel (the exporter's ``KERNEL_SMEM_PROGRAMS``): the first render writes the
    # kernel's figure, the kernel symbol, the argument plan and the logical names it keeps.
    "cake_causal_conv1d_train_8b9b8db0833fd64a578f": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "partial"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_8b9b8db0833fd64a578f",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_8b9b8db0833fd64a578f",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 50176,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_8b9b8db0833fd64a578f",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dx": "dx",
            "dy": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
        },
        "reduce": "cake_causal_conv1d_train_ea8f91038a9dea452ff2",
        "role": "bwd",
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 4,
        },
    },
    # Route ``bwd_bias`` (hand-written fallback, sm_100a / sm_103a): the fallback body with
    # ``bias=True`` (reads ``b``; the ``partial`` workspace grows a fifth slab, ``taps`` 5,
    # of dbias partials that its reduce program sums into ``db``) and ``dtype_flags=True``
    # (input bits 0 and 2; the store bits 1 and 3 are the reduce program's), ``min_blocks=2``
    # (the bias accumulators do not fit the 168-register cap of three CTAs per SM).
    # ``dynamic_smem_bytes`` 0 is the placeholder of a program whose shared memory follows
    # the kernel (the exporter's ``KERNEL_SMEM_PROGRAMS``): the first render writes the
    # kernel's figure, the kernel symbol, the argument plan and the logical names it keeps.
    "cake_causal_conv1d_train_2d91e5f926238453c78c": {
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
            ["buffer", "partial"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_2d91e5f926238453c78c",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_2d91e5f926238453c78c",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 54272,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_2d91e5f926238453c78c",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "x_map": "x",
        },
        "reduce": "cake_causal_conv1d_train_1c21c181af94e1635851",
        "role": "bwd",
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 5,
        },
    },
    # Route ``bwd_bias`` (hand-written fallback, sm_107a): the sm_107a fallback body with
    # ``bias=True, dtype_flags=True``.
    # ``dynamic_smem_bytes`` 0 is the placeholder of a program whose shared memory follows
    # the kernel (the exporter's ``KERNEL_SMEM_PROGRAMS``): the first render writes the
    # kernel's figure, the kernel symbol, the argument plan and the logical names it keeps.
    "cake_causal_conv1d_train_cc7a117aee9c7ae829c4": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "partial"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_cc7a117aee9c7ae829c4",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_cc7a117aee9c7ae829c4",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 54272,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_cc7a117aee9c7ae829c4",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dx": "dx",
            "dy": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
        },
        "reduce": "cake_causal_conv1d_train_1c21c181af94e1635851",
        "role": "bwd",
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 5,
        },
    },
    # Route ``bwd_fp32g`` (hand-written fallback, sm_100a / sm_103a): the fallback body with
    # ``grad_pre="fp32"`` (SiLU' at the unrounded FP32 preactivation; its reduce program
    # sums the partial rows in FP64 with one rounding), ``fold_every=4, fold_kahan=True``
    # and ``dtype_flags=True``, built ``min_blocks=2``: two CTAs per SM (the FP32-SiLU'
    # body spills at the 168-register cap of three), so ``ctas_per_sm`` 2.
    # ``dynamic_smem_bytes`` 0 is the placeholder of a program whose shared memory follows
    # the kernel (the exporter's ``KERNEL_SMEM_PROGRAMS``): the first render writes the
    # kernel's figure, the kernel symbol, the argument plan and the logical names it keeps.
    "cake_causal_conv1d_train_c8027a64ba3cc38333bf": {
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
            ["buffer", "partial"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_c8027a64ba3cc38333bf",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_c8027a64ba3cc38333bf",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 66560,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_c8027a64ba3cc38333bf",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "x_map": "x",
        },
        "reduce": "cake_causal_conv1d_train_ce49146b9fef24adb47e",
        "role": "bwd",
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 4,
        },
    },
    # Route ``bwd_fp32g`` (hand-written fallback, sm_107a): the sm_107a fallback body with
    # ``grad_pre="fp32", dtype_flags=True``.
    # ``dynamic_smem_bytes`` 0 is the placeholder of a program whose shared memory follows
    # the kernel (the exporter's ``KERNEL_SMEM_PROGRAMS``): the first render writes the
    # kernel's figure, the kernel symbol, the argument plan and the logical names it keeps.
    "cake_causal_conv1d_train_f1545adfd2bb28044419": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "partial"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_f1545adfd2bb28044419",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_f1545adfd2bb28044419",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 66560,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_f1545adfd2bb28044419",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dx": "dx",
            "dy": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
        },
        "reduce": "cake_causal_conv1d_train_ce49146b9fef24adb47e",
        "role": "bwd",
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 4,
        },
    },
    # Route ``bwd_bias_fp32g`` (hand-written fallback, sm_100a / sm_103a): the fallback body
    # with ``bias=True, grad_pre="fp32", dtype_flags=True``.
    # ``dynamic_smem_bytes`` 0 is the placeholder of a program whose shared memory follows
    # the kernel (the exporter's ``KERNEL_SMEM_PROGRAMS``): the first render writes the
    # kernel's figure, the kernel symbol, the argument plan and the logical names it keeps.
    "cake_causal_conv1d_train_149ca6f53934ba630033": {
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
            ["buffer", "partial"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_149ca6f53934ba630033",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_149ca6f53934ba630033",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 74752,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_149ca6f53934ba630033",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "x_map": "x",
        },
        "reduce": "cake_causal_conv1d_train_4d291f4b5c2cce6ef1ea",
        "role": "bwd",
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 5,
        },
    },
    # Route ``bwd_bias_fp32g`` (hand-written fallback, sm_107a): the sm_107a fallback body
    # with ``bias=True, grad_pre="fp32", dtype_flags=True``.
    # ``dynamic_smem_bytes`` 0 is the placeholder of a program whose shared memory follows
    # the kernel (the exporter's ``KERNEL_SMEM_PROGRAMS``): the first render writes the
    # kernel's figure, the kernel symbol, the argument plan and the logical names it keeps.
    "cake_causal_conv1d_train_843c010db78c79107fd3": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "partial"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_843c010db78c79107fd3",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_843c010db78c79107fd3",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 74752,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_843c010db78c79107fd3",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dx": "dx",
            "dy": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
        },
        "reduce": "cake_causal_conv1d_train_4d291f4b5c2cce6ef1ea",
        "role": "bwd",
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 5,
        },
    },
    # Rendered by the exporter from the family adapter's scaffold: the backward
    # with vec 8, copy tma2d, stage_tokens 8, min_blocks 2, dtype_flags,
    # fold_layout thread
    # (2 CTAs per SM); it runs when C >= select_rule.channels[arch] and no
    # routed record of a higher threshold admits C; below every threshold the
    # fallback record (no select_rule) runs.
    "cake_causal_conv1d_train_2675cc1cb4efc0937378": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "partial"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_2675cc1cb4efc0937378",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_2675cc1cb4efc0937378",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 82944,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_2675cc1cb4efc0937378",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "x_map": "x",
        },
        "reduce": "cake_causal_conv1d_train_ea8f91038a9dea452ff2",
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_100a": 512,
                "sm_103a": 512,
                "sm_107a": 1024,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 4,
        },
    },
    # Rendered by the exporter from the family adapter's scaffold: the backward
    # with vec 8, copy tma2d, stage_tokens 8, min_blocks 2, bias, dtype_flags
    # (2 CTAs per SM); it runs when C >= select_rule.channels[arch] and no
    # routed record of a higher threshold admits C; below every threshold the
    # fallback record (no select_rule) runs.
    "cake_causal_conv1d_train_517710d1f0988cf570a9": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "partial"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_517710d1f0988cf570a9",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_517710d1f0988cf570a9",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 87040,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_517710d1f0988cf570a9",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "x_map": "x",
        },
        "reduce": "cake_causal_conv1d_train_1c21c181af94e1635851",
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_100a": 512,
                "sm_103a": 512,
                "sm_107a": 1024,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 5,
        },
    },
    # Rendered by the exporter from the family adapter's scaffold: the backward
    # with vec 8, copy tma2d, stage_tokens 8, min_blocks 2, grad_pre fp32,
    # fold_every 4, fold_kahan, dtype_flags
    # (2 CTAs per SM); it runs when C >= select_rule.channels[arch] and no
    # routed record of a higher threshold admits C; below every threshold the
    # fallback record (no select_rule) runs.
    "cake_causal_conv1d_train_170d89481a616ae977b3": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "partial"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_170d89481a616ae977b3",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_170d89481a616ae977b3",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 99328,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_170d89481a616ae977b3",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "x_map": "x",
        },
        "reduce": "cake_causal_conv1d_train_ce49146b9fef24adb47e",
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_100a": 512,
                "sm_103a": 512,
                "sm_107a": 1024,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 4,
        },
    },
    # Rendered by the exporter from the family adapter's scaffold: the backward
    # with vec 8, copy tma2d, stage_tokens 8, min_blocks 2, bias, grad_pre fp32,
    # fold_every 4, fold_kahan, dtype_flags
    # (2 CTAs per SM); it runs when C >= select_rule.channels[arch] and no
    # routed record of a higher threshold admits C; below every threshold the
    # fallback record (no select_rule) runs.
    "cake_causal_conv1d_train_a681165b3e81204c73ff": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "partial"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_a681165b3e81204c73ff",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_a681165b3e81204c73ff",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 107520,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_a681165b3e81204c73ff",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "x_map": "x",
        },
        "reduce": "cake_causal_conv1d_train_4d291f4b5c2cce6ef1ea",
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_100a": 512,
                "sm_103a": 512,
                "sm_107a": 1024,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 5,
        },
    },
    # The reduce program every backward route shares: the second launch of a
    # backward call.  Launched programmatically dependent on the main program
    # (``use_pdl``: the binding sets the programmatic-stream-serialization
    # launch attribute; the kernel's first statement waits for the main grid,
    # so the launch latency overlaps the main kernel's tail and an ordinary
    # launch stays correct), it sums the FP32 ``[C, 4]`` partial rows of the
    # workspace per channel: channel ``c`` lies in tile ``t = c // (vec x
    # ct_vec)``, whose ``ceil((main_ctas - t) / n_ch_tiles)`` rows the main
    # grid wrote, in the order the cooperative backward used in-kernel -- one
    # warp per channel, lane sums over rows ``lane, lane + 32, ...`` then the
    # xor tree 16, 8, 4, 2, 1 -- so ``dweight`` is bitwise what one launch
    # produced.  Built with its kernel file's default keywords, one program
    # for the three architectures.
    "cake_causal_conv1d_train_86b5ecfbd90aca4b52e5": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "partial"],
            ["buffer", "dw"],
            ["parameter", "C"],
            ["parameter", "main_ctas"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "ct_vec"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        # One warp per channel, four channels per 128-thread CTA: the grid is
        # ``ceil(C / channels_per_cta)`` CTAs, a function of the call's ``C``.
        "grid_rule": {
            "channels_per_cta": 4,
            "cooperative": False,
            "kind": "channels_per_cta",
        },
        "kernel": "cake_causal_conv1d_train_86b5ecfbd90aca4b52e5",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_86b5ecfbd90aca4b52e5",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2048,
            "use_pdl": True,
        },
        "library": "cake_causal_conv1d_train_86b5ecfbd90aca4b52e5",
        "logical": {
            "C": "C",
            "ct_vec": "ct_vec",
            "dw": "dw",
            "main_ctas": "main_ctas",
            "n_ch_tiles": "n_ch_tiles",
            "partial": "partial",
        },
        "role": "reduce",
        "workspace_rule": None,
    },
    # The reduce program of the ``bwd_dt`` route (hand-written; the render fills the name,
    # symbol, argument plan and logical map): the default reduce built with ``dtype_flags=True``; it
    # stores ``dw`` as BF16 when ``flags`` bit 1 says so;
    # otherwise the default reduce's order and CTA shape (one warp per channel).
    "cake_causal_conv1d_train_ea8f91038a9dea452ff2": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "partial"],
            ["buffer", "dw"],
            ["parameter", "C"],
            ["parameter", "main_ctas"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "ct_vec"],
            ["parameter", "flags"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "channels_per_cta": 4,
            "cooperative": False,
            "kind": "channels_per_cta",
        },
        "kernel": "cake_causal_conv1d_train_ea8f91038a9dea452ff2",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_ea8f91038a9dea452ff2",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2048,
            "use_pdl": True,
        },
        "library": "cake_causal_conv1d_train_ea8f91038a9dea452ff2",
        "logical": {
            "C": "C",
            "ct_vec": "ct_vec",
            "dw": "dw",
            "flags": "flags",
            "main_ctas": "main_ctas",
            "n_ch_tiles": "n_ch_tiles",
            "partial": "partial",
        },
        "role": "reduce",
        "workspace_rule": None,
    },
    # The reduce program of the ``bwd_bias`` route (hand-written; the render fills the name,
    # symbol, argument plan and logical map): the default reduce built with ``bias=True, dtype_flags=True``; it
    # sums the fifth, dbias, slab of the partial rows into ``db`` as well;
    # stores ``dw`` / ``db`` as BF16 when ``flags`` bit 1 / 3 says so;
    # otherwise the default reduce's order and CTA shape (one warp per channel).
    "cake_causal_conv1d_train_1c21c181af94e1635851": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "partial"],
            ["buffer", "dw"],
            ["buffer", "db"],
            ["parameter", "C"],
            ["parameter", "main_ctas"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "ct_vec"],
            ["parameter", "flags"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "channels_per_cta": 4,
            "cooperative": False,
            "kind": "channels_per_cta",
        },
        "kernel": "cake_causal_conv1d_train_1c21c181af94e1635851",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_1c21c181af94e1635851",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2560,
            "use_pdl": True,
        },
        "library": "cake_causal_conv1d_train_1c21c181af94e1635851",
        "logical": {
            "C": "C",
            "ct_vec": "ct_vec",
            "db": "db",
            "dw": "dw",
            "flags": "flags",
            "main_ctas": "main_ctas",
            "n_ch_tiles": "n_ch_tiles",
            "partial": "partial",
        },
        "role": "reduce",
        "workspace_rule": None,
    },
    # The reduce program of the ``bwd_fp32g`` route (hand-written; the render fills the name,
    # symbol, argument plan and logical map): the default reduce built with ``fp64=True, dtype_flags=True``; it
    # sums in FP64 (lane sums and the xor tree) with one final rounding;
    # stores ``dw`` as BF16 when ``flags`` bit 1 says so;
    # otherwise the default reduce's order and CTA shape (one warp per channel).
    "cake_causal_conv1d_train_ce49146b9fef24adb47e": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "partial"],
            ["buffer", "dw"],
            ["parameter", "C"],
            ["parameter", "main_ctas"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "ct_vec"],
            ["parameter", "flags"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "channels_per_cta": 4,
            "cooperative": False,
            "kind": "channels_per_cta",
        },
        "kernel": "cake_causal_conv1d_train_ce49146b9fef24adb47e",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_ce49146b9fef24adb47e",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4096,
            "use_pdl": True,
        },
        "library": "cake_causal_conv1d_train_ce49146b9fef24adb47e",
        "logical": {
            "C": "C",
            "ct_vec": "ct_vec",
            "dw": "dw",
            "flags": "flags",
            "main_ctas": "main_ctas",
            "n_ch_tiles": "n_ch_tiles",
            "partial": "partial",
        },
        "role": "reduce",
        "workspace_rule": None,
    },
    # The reduce program of the ``bwd_bias_fp32g`` route (hand-written; the render fills the name,
    # symbol, argument plan and logical map): the default reduce built with ``bias=True, fp64=True, dtype_flags=True``; it
    # sums the fifth, dbias, slab of the partial rows into ``db`` as well;
    # sums in FP64 (lane sums and the xor tree) with one final rounding;
    # stores ``dw`` / ``db`` as BF16 when ``flags`` bit 1 / 3 says so;
    # otherwise the default reduce's order and CTA shape (one warp per channel).
    "cake_causal_conv1d_train_4d291f4b5c2cce6ef1ea": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "partial"],
            ["buffer", "dw"],
            ["buffer", "db"],
            ["parameter", "C"],
            ["parameter", "main_ctas"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "ct_vec"],
            ["parameter", "flags"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "channels_per_cta": 4,
            "cooperative": False,
            "kind": "channels_per_cta",
        },
        "kernel": "cake_causal_conv1d_train_4d291f4b5c2cce6ef1ea",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_4d291f4b5c2cce6ef1ea",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 5120,
            "use_pdl": True,
        },
        "library": "cake_causal_conv1d_train_4d291f4b5c2cce6ef1ea",
        "logical": {
            "C": "C",
            "ct_vec": "ct_vec",
            "db": "db",
            "dw": "dw",
            "flags": "flags",
            "main_ctas": "main_ctas",
            "n_ch_tiles": "n_ch_tiles",
            "partial": "partial",
        },
        "role": "reduce",
        "workspace_rule": None,
    },
}

# Architecture -> route -> the records a call of the route selects from (the
# fallback first).  The default routes ``fwd`` / ``bwd`` are the shipped
# programs; the six round-6 routes each hold their hand-written fallback
# (above) and, once rendered, the routed programs the exporter scaffolds from
# it; the forward routes' sm_107a two-stage programs are hand-written records.
ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "bwd": [
            "cake_causal_conv1d_train_9bfa65707a49df6a48ce",
            "cake_causal_conv1d_train_37e428406a0cc4a8ca1b",
        ],
        "bwd_bias": [
            "cake_causal_conv1d_train_2d91e5f926238453c78c",
            "cake_causal_conv1d_train_517710d1f0988cf570a9",
        ],
        "bwd_bias_fp32g": [
            "cake_causal_conv1d_train_149ca6f53934ba630033",
            "cake_causal_conv1d_train_a681165b3e81204c73ff",
        ],
        "bwd_dt": [
            "cake_causal_conv1d_train_9c8310a1d3fa9b8ccffa",
            "cake_causal_conv1d_train_2675cc1cb4efc0937378",
        ],
        "bwd_fp32g": [
            "cake_causal_conv1d_train_c8027a64ba3cc38333bf",
            "cake_causal_conv1d_train_170d89481a616ae977b3",
        ],
        "fwd": [
            "cake_causal_conv1d_train_f5497330bfd286211643",
        ],
        "fwd_bias": [
            "cake_causal_conv1d_train_08880ca12e4064bc8cdb",
        ],
        "fwd_dt": [
            "cake_causal_conv1d_train_9c1e7ee9f37e3102660e",
        ],
    },
    "sm_103a": {
        "bwd": [
            "cake_causal_conv1d_train_9bfa65707a49df6a48ce",
            "cake_causal_conv1d_train_37e428406a0cc4a8ca1b",
            "cake_causal_conv1d_train_d46036aa2bc5f90bb0f6",
        ],
        "bwd_bias": [
            "cake_causal_conv1d_train_2d91e5f926238453c78c",
            "cake_causal_conv1d_train_517710d1f0988cf570a9",
        ],
        "bwd_bias_fp32g": [
            "cake_causal_conv1d_train_149ca6f53934ba630033",
            "cake_causal_conv1d_train_a681165b3e81204c73ff",
        ],
        "bwd_dt": [
            "cake_causal_conv1d_train_9c8310a1d3fa9b8ccffa",
            "cake_causal_conv1d_train_2675cc1cb4efc0937378",
        ],
        "bwd_fp32g": [
            "cake_causal_conv1d_train_c8027a64ba3cc38333bf",
            "cake_causal_conv1d_train_170d89481a616ae977b3",
        ],
        "fwd": [
            "cake_causal_conv1d_train_f5497330bfd286211643",
        ],
        "fwd_bias": [
            "cake_causal_conv1d_train_08880ca12e4064bc8cdb",
        ],
        "fwd_dt": [
            "cake_causal_conv1d_train_9c1e7ee9f37e3102660e",
        ],
    },
    "sm_107a": {
        "bwd": [
            "cake_causal_conv1d_train_b4f7ff5a3dd9ddd2f41c",
            "cake_causal_conv1d_train_37e428406a0cc4a8ca1b",
            "cake_causal_conv1d_train_d46036aa2bc5f90bb0f6",
            "cake_causal_conv1d_train_aeff6106e2822f313a0c",
            "cake_causal_conv1d_train_9a07bcaa0223632640ee",
        ],
        "bwd_bias": [
            "cake_causal_conv1d_train_cc7a117aee9c7ae829c4",
            "cake_causal_conv1d_train_517710d1f0988cf570a9",
        ],
        "bwd_bias_fp32g": [
            "cake_causal_conv1d_train_843c010db78c79107fd3",
            "cake_causal_conv1d_train_a681165b3e81204c73ff",
        ],
        "bwd_dt": [
            "cake_causal_conv1d_train_8b9b8db0833fd64a578f",
            "cake_causal_conv1d_train_2675cc1cb4efc0937378",
        ],
        "bwd_fp32g": [
            "cake_causal_conv1d_train_f1545adfd2bb28044419",
            "cake_causal_conv1d_train_170d89481a616ae977b3",
        ],
        "fwd": [
            "cake_causal_conv1d_train_f5497330bfd286211643",
            "cake_causal_conv1d_train_8665d0756b8240ecb4d0",
        ],
        "fwd_bias": [
            "cake_causal_conv1d_train_08880ca12e4064bc8cdb",
            "cake_causal_conv1d_train_d829c95e1d610bc14b23",
        ],
        "fwd_dt": [
            "cake_causal_conv1d_train_9c1e7ee9f37e3102660e",
            "cake_causal_conv1d_train_176ebb3e14f7ad9c94dc",
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
# The routes (``ROUTES[arch]`` keys) a public call may resolve to.  ``fwd`` /
# ``bwd`` are the default routes (FP32 taps, no bias, SiLU' at the BF16-rounded
# preactivation); ``_dt`` routes take BF16 taps and / or store a BF16 gradient
# (the ``flags`` scalar says which), ``_bias`` routes add the per-channel bias
# and its gradient, ``_fp32g`` routes evaluate SiLU' at the unrounded FP32
# preactivation for the weight gradient.  :func:`forward_route` and
# :func:`backward_route` are the host's map from a call's arguments to a route.
KINDS = (
    "fwd",
    "fwd_dt",
    "fwd_bias",
    "bwd",
    "bwd_dt",
    "bwd_bias",
    "bwd_fp32g",
    "bwd_bias_fp32g",
)
# The routes every exported architecture ships: the base forward and backward.
# The other routes are optional per architecture; a call that needs one the
# architecture lacks is refused with a ``ValueError`` naming the route.
REQUIRED_KINDS = ("fwd", "bwd")
# The role of a module record: a forward program, a backward program (the
# first launch of a backward call) or the reduce program a backward record
# names in ``record["reduce"]`` (the second launch).
ROLES = ("fwd", "bwd", "reduce")
# The grid-rule vocabulary (``record["grid_rule"]["kind"]``, see
# :func:`launch_grid`): ``co_resident`` launches exactly ``ctas_per_sm x SM
# count`` CTAs, the co-resident count (the forward records and the
# cooperative KCP programs of :mod:`.kcp_jit`, which share this host plan);
# ``occupancy`` launches ``grid_mult[arch] x ctas_per_sm x SM count`` CTAs
# normally (the backward programs since the split; ``grid_mult`` names every
# architecture of its record); ``channels_per_cta`` launches ``ceil(C /
# channels_per_cta)`` CTAs (the reduce programs).
GRID_KINDS = ("co_resident", "occupancy", "channels_per_cta")
# The selection-rule vocabulary of a forward or backward record
# (``record["select_rule"]["kind"]``); of the records whose rule admits a call
# the one of the highest channel threshold runs, and a record without a rule is
# its route's fallback.  ``channels_tokens_at_least`` adds a token floor to the
# channel threshold.  A reduce record never carries one.
SELECT_KINDS = ("channels_at_least", "channels_tokens_at_least")
# The ``flags`` bit mask of the ``dtype_flags`` programs (every program of the
# six round-6 routes): the host derives it from the tap tensor, the ACTUAL
# ``dweight`` out tensor, the bias tensor and the ACTUAL ``dbias`` out tensor;
# the kernel reads it once in its prologue (load form) and once in its epilogue
# (store form).  The default routes' programs take no ``flags``.
FLAG_W_BF16 = 1  # ``w`` holds BF16 taps (else FP32)
FLAG_DW_BF16 = 2  # ``dw`` is stored as BF16 (else FP32)
FLAG_B_BF16 = 4  # ``b`` holds BF16 (else FP32); bias programs
FLAG_DB_BF16 = 8  # ``db`` is stored as BF16 (else FP32); bias programs
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
    "b",
    "db",
    "work",
)
# The workspace-rule vocabulary (``record["workspace_rule"]["kind"]``): a
# backward program's ``partial_rows`` (its FP32 partial rows, :func:`partial_bytes`)
# and a forward program's ``tile_counters`` (its int32 ``work`` counters,
# ``words_per_tile`` per channel tile, :func:`counter_words`); a reduce
# program has none.
WORKSPACE_KINDS = ("partial_rows", "tile_counters")
# The scalars :func:`plan` produces, in kernel order; ``flags`` (the dtype bit
# mask of the ``dtype_flags`` programs) and ``main_ctas`` (the CTA count of the
# selected backward program's launch, ``grid_x``, from which the reduce program
# derives the partial rows of each channel tile, ``ceil((main_ctas - tile) /
# n_ch_tiles)``) are set by the host beside them.
PLAN_SCALARS = (
    "S",
    "C",
    "num_docs",
    "search_steps",
    "ct_vec",
    "n_sub",
    "n_ch_tiles",
)
LOGICAL_SCALARS = (*PLAN_SCALARS, "flags", "main_ctas")


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


def route_role(kind: str) -> str:
    """``"fwd"`` or ``"bwd"``: the kernel a route's programs are built from (``ValueError`` for an unknown route)."""

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return "fwd" if kind.startswith("fwd") else "bwd"


def forward_route(*, bias: bool, bf16_taps: bool) -> str:
    """The forward route of a call: ``fwd_bias`` with a bias, else ``fwd_dt`` for BF16 taps, else ``fwd``.

    A bias program always has a bias (no runtime switch: adding ``+0.0`` would
    flip the sign of ``-0.0`` outputs), so ``bias=None`` never selects one; the
    bias programs read FP32 or BF16 taps and bias through ``flags``.
    """

    if bias:
        return "fwd_bias"
    return "fwd_dt" if bf16_taps else "fwd"


def backward_route(*, bias: bool, fp32_grad: bool, bf16_taps_or_dweight: bool) -> str:
    """The backward route of a call.

    ``bias`` selects ``bwd_bias`` / ``bwd_bias_fp32g`` (by ``fp32_grad``, the
    ``silu_grad_preactivation="fp32"`` policy); without a bias ``fp32_grad``
    selects ``bwd_fp32g``; else BF16 taps or a BF16 ``dweight`` out tensor
    select ``bwd_dt``; else the default ``bwd``.  Every route but ``bwd``
    reads the tap / gradient dtypes from ``flags``.
    """

    if bias:
        return "bwd_bias_fp32g" if fp32_grad else "bwd_bias"
    if fp32_grad:
        return "bwd_fp32g"
    return "bwd_dt" if bf16_taps_or_dweight else "bwd"


def route_modules(arch: str, kind: str) -> tuple[str, ...]:
    """Ordered module names of one route on ``arch`` (every program the route may launch).

    A call launches the one :func:`selected_modules` picks for its channel
    count.
    """

    route_role(kind)
    names = ROUTES.get(arch, {}).get(kind)
    if not names:
        raise ValueError(
            f"unsupported Cake causal conv1d training route {kind!r} on {arch!r}; "
            f"exported architectures: {ARCHES}"
        )
    return tuple(names)


def select_rule(record: dict[str, Any]) -> dict[str, Any] | None:
    """The selection rule of one module record, ``None`` when the record carries none.

    A record without a rule (no ``select_rule`` key, or ``None``) is its
    route's fallback program.  ``{"kind": "channels_at_least", "channels":
    {arch: C_min, ...}}`` owns the calls whose channel count is at least
    ``channels[arch]`` on the architectures it names and never hits on the
    others.  ``{"kind": "channels_tokens_at_least", "channels": {arch: C_min,
    ...}, "tokens": {arch: S_min, ...}}`` owns those calls only when the token
    count is also at least ``tokens[arch]``.  When several records own a call,
    the one of the highest channel threshold runs.
    """

    return record.get("select_rule")


def _threshold(rule: dict[str, Any], *, arch: str) -> int | None:
    """The channel count from which a selection rule owns the calls on ``arch``, ``None`` where it never does
    (``ValueError`` for an unknown kind)."""

    kind = rule["kind"]
    if kind in SELECT_KINDS:
        channels = rule["channels"]
        return int(channels[arch]) if arch in channels else None
    raise ValueError(
        f"unknown Cake causal conv1d training selection rule: {kind!r} "
        f"(known: {SELECT_KINDS})"
    )


def _tokens(rule: dict[str, Any], *, arch: str) -> int | None:
    """The token count from which a selection rule owns the calls on ``arch`` (its token floor), ``None`` when the
    rule has none there (``ValueError`` for an unknown kind)."""

    _threshold(rule, arch=arch)
    if rule["kind"] != "channels_tokens_at_least":
        return None
    tokens = rule["tokens"]
    return int(tokens[arch]) if arch in tokens else None


def _admits(rule: dict[str, Any], *, arch: str, C: int, S: int | None = None) -> bool:
    """Whether a selection rule owns ``C`` channels (and ``S`` tokens) on ``arch``.

    ``ValueError`` for an unknown kind, or when the rule's token floor decides
    and ``S`` is ``None``.
    """

    threshold = _threshold(rule, arch=arch)
    if threshold is None or int(C) < threshold:
        return False
    floor = _tokens(rule, arch=arch)
    if floor is None:
        return True
    if S is None:
        raise ValueError(
            f"the Cake causal conv1d training selection rule {rule!r} on {arch} "
            f"for C={int(C)} runs from S {floor}; the selection needs S"
        )
    return int(S) >= floor


def selected_modules(
    arch: str, kind: str, *, C: int, S: int | None = None
) -> tuple[str, ...]:
    """Ordered module names one public call launches on ``arch`` for ``C`` channels and ``S`` tokens.

    A route may carry several programs of its kernel and launches exactly one:
    of the records whose :func:`select_rule` admits the call on ``arch`` the
    one of the highest channel threshold, else the route's single fallback
    record (the one without a rule).  The choice is a pure function of the
    architecture, the route, ``C`` and ``S`` -- ``S`` enters only through a
    rule's token floor, and the host never reads the data -- so a given ``(S,
    C, device)`` always runs the same program of a route and ``dweight``
    stays bitwise reproducible.  Two records of one channel threshold that
    both reach ``C``, or zero or several fallbacks, are registry errors
    (``RuntimeError``); an unknown rule kind, or ``S`` of ``None`` where a
    token floor decides, is a ``ValueError``.
    """

    names = route_modules(arch, kind)
    candidates: list[tuple[int, str]] = []
    fallback: list[str] = []
    for name in names:
        rule = select_rule(MODULES[name])
        if rule is None:
            fallback.append(name)
            continue
        threshold = _threshold(rule, arch=arch)
        if threshold is not None and int(C) >= threshold:
            candidates.append((threshold, name))
    if len(fallback) != 1:
        raise RuntimeError(
            f"the exported route {kind!r} on {arch} declares {len(fallback)} "
            "fallback programs (records without a select_rule); expected exactly one"
        )
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        tied = [name for threshold, name in candidates if threshold == candidates[0][0]]
        raise RuntimeError(
            f"the exported route {kind!r} on {arch} selects {len(tied)} "
            f"programs for C={int(C)} at one threshold {candidates[0][0]} ({tied}); "
            "thresholds must be distinct"
        )
    for _threshold_c, name in candidates:
        if _admits(select_rule(MODULES[name]), arch=arch, C=C, S=S):
            return (name,)
    return (fallback[0],)


def backward_kinds(arch: str) -> tuple[str, ...]:
    """The backward routes ``arch`` exports, in :data:`KINDS` order."""

    return tuple(
        kind
        for kind in KINDS
        if route_role(kind) == "bwd" and ROUTES.get(arch, {}).get(kind)
    )


def workspace_owners(arch: str) -> tuple[dict[str, Any], ...]:
    """The backward module records of ``arch`` that own a ``partial`` workspace, over every backward route in
    :data:`KINDS` order, each route in launch order.

    Every backward program declares a ``workspace_rule``; an architecture
    without one is a registry error (``RuntimeError``).
    """

    owners = tuple(
        MODULES[name]
        for kind in backward_kinds(arch)
        for name in route_modules(arch, kind)
        if MODULES[name]["workspace_rule"] is not None
    )
    if not owners:
        raise RuntimeError(
            f"the exported backward routes on {arch} declare no workspace owner; "
            "expected at least one"
        )
    return owners


def workspace_record(arch: str) -> dict[str, Any]:
    """The capacity owner of the backward ``partial`` workspace on ``arch``.

    The public workspace serves every backward program of every backward route
    (:func:`workspace_owners`), each of which writes the leading rows of its
    own grid.  The owners share one build geometry, so for every ``C`` their
    rows are ``ceil(grid_mult x ctas_per_sm x SM count / n_ch_tiles)`` with
    one ``n_ch_tiles`` and their bytes ``4 x taps x C x rows``: the record
    with the largest :func:`grid_per_sm` ``x taps`` (the bias programs' fifth
    slab counts; :data:`KINDS` then route order break ties) bounds them all.
    Owners of different geometries are a registry error (``RuntimeError``).
    """

    owners = workspace_owners(arch)
    geometries = {
        tuple(sorted(owner["grid_rule"]["geometry"].items())) for owner in owners
    }
    if len(geometries) != 1:
        raise RuntimeError(
            f"the backward programs on {arch} declare {len(geometries)} build "
            "geometries; one workspace serves them only when they share one"
        )
    return max(
        owners,
        key=lambda owner: (
            grid_per_sm(owner, arch) * int(owner["workspace_rule"]["taps"])
        ),
    )


def grid_mult(record: dict[str, Any], arch: str) -> int:
    """The over-decomposition factor of an ``occupancy`` record on ``arch`` (1 under every other grid rule).

    ``grid_rule["grid_mult"]`` maps each architecture of the record to the
    factor its grid multiplies the co-resident CTA count by; the factor per
    architecture is the measured trade of tail latency behind a resident
    side kernel against partial-row traffic.
    """

    rule = record["grid_rule"]
    if rule["kind"] != "occupancy":
        return 1
    factor = int(rule["grid_mult"][arch])
    if factor < 1:
        raise ValueError(
            f"module {record['kernel']}: grid_mult[{arch!r}] must be positive, got {factor}"
        )
    return factor


def grid_per_sm(record: dict[str, Any], arch: str) -> int:
    """CTAs per SM of a ``co_resident`` or ``occupancy`` record's grid on ``arch``:
    ``grid_mult x ctas_per_sm`` (the grid is this times the SM count)."""

    return grid_mult(record, arch) * int(record["grid_rule"]["ctas_per_sm"])


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-int(numerator) // int(denominator))


# The roles whose programs plan their channel tiles with :func:`channel_split`
# (a reduce program runs on its backward record's plan): each has its own rule.
SPLIT_ROLES = ("fwd", "bwd")


def channel_split(
    C: int,
    *,
    geometry: dict[str, int],
    role: str | None = None,
    legacy: bool = False,
) -> tuple[int, int, int]:
    """``(ct_vec, n_sub, n_ch_tiles)`` of ``C`` channels under a record's build geometry and role.

    ``ct_vec`` channel vectors per channel tile (one thread each), ``n_sub =
    threads // ct_vec`` token sub-tiles per CTA and ``n_ch_tiles = ceil(C / vec
    / ct_vec)`` channel tiles.  ``role`` is the record's, ``"fwd"`` or
    ``"bwd"`` (a ``ValueError`` without one unless ``legacy``): the two rules
    agree except in the fallback below.  With ``cv = C / vec`` channel
    vectors: ``ct_vec = cv`` when ``cv <= threads``; above that, the multiple
    of 32 dividing ``cv`` (at most ``threads``) with the highest lane use
    ``ct_vec x (threads // ct_vec)``, the larger ``ct_vec`` on a tie.  So C =
    1280 (160 vectors) runs five 32-vector tiles with four token sub-tiles
    each, (32, 4, 5), and C = 1536 (64, 2, 3) -- every lane busy -- instead of
    two 128-vector tiles at 62 % / 75 % lane use; C <= 1024 and 2048 / 4096 /
    8192 are unchanged.  When no multiple of 32 divides ``cv``, the role's
    fallback picks the ``ct_vec`` of the highest lane use over the launch,
    ``cv / (n_ch_tiles x ct_vec) x (ct_vec x (threads // ct_vec) / threads)``
    rounded to 9 decimals (the larger on a tie), among every multiple of 32 at
    most ``threads`` (the last tile partial) and, for the forward, every
    divisor of ``cv`` from 40 to ``threads`` (whole tiles).  At 128 x 8 the
    backward runs C = 1160 as (32, 4, 5), 1344 as (64, 2, 3) and 1600 as (32,
    4, 7), the forward 1160 as (32, 4, 5), 1344 as (42, 3, 4) and 1600 as (40,
    3, 5); both run 1920 as (128, 1, 2).  The per element arithmetic does not
    depend on the tiling, so ``y`` and ``dx`` are bitwise the legacy plan's;
    ``dweight``'s fixed summation order follows the tiling, so its bits differ
    from the legacy plan's (both are reproducible).  ``legacy=True`` is the
    rule the package shipped before round 6, ``ct_vec = min(cv, threads)``,
    for either role; the KCP path keeps it (its kernels pick the grouped path
    from ``ct_vec`` on the device and are measured under this rule).  Each
    role's rule is kept in sync with its copies (the kernel generator of
    that role and the test plan of that kind).
    """

    return _channel_split(
        int(C), int(geometry["vec"]), int(geometry["threads"]), role, bool(legacy)
    )


@functools.lru_cache(maxsize=4096)
def _channel_split(
    C: int, vec: int, threads: int, role: str | None, legacy: bool
) -> tuple[int, int, int]:
    """:func:`channel_split` of plain arguments, memoized: a model's channel
    counts are few, so every call after a channel count's first reuses its
    plan (a backward call plans its row up to twice)."""

    if (role is None and not legacy) or (role is not None and role not in SPLIT_ROLES):
        raise ValueError(
            f"role must be one of {SPLIT_ROLES} (or legacy=True), got {role!r}"
        )
    if C <= 0 or C % vec:
        raise ValueError(f"C={C} must be a positive multiple of {vec}")
    channel_vectors = C // vec
    if legacy or channel_vectors <= threads:
        ct_vec = min(channel_vectors, threads)
    else:
        candidates = [
            width
            for width in range(32, threads + 1, 32)
            if channel_vectors % width == 0
        ]
        ct_vec = (
            max(candidates, key=lambda width: (width * (threads // width), width))
            if candidates
            else _fallback_split(channel_vectors, threads, role)
        )
    return ct_vec, threads // ct_vec, _ceil_div(channel_vectors, ct_vec)


def _fallback_split(channel_vectors: int, threads: int, role: str) -> int:
    """``ct_vec`` of :func:`channel_split` above ``threads`` vectors with no 32-multiple divisor.

    The candidates are every multiple of 32 at most ``threads`` (the last tile
    partial: its copies and stores stop at channel ``C``) and, for the
    forward, every divisor of ``channel_vectors`` from 40 to ``threads`` (whole
    tiles); the one of the highest :func:`_lane_use`, rounded to 9 decimals,
    wins, the larger on a tie.  No tile is narrower than a warp's 32 vectors
    and no whole forward tile narrower than 40.
    """

    candidates = set(range(32, threads + 1, 32))
    if role == "fwd":
        candidates |= {
            width for width in range(40, threads + 1) if channel_vectors % width == 0
        }
    return max(
        candidates,
        key=lambda width: (
            round(_lane_use(channel_vectors, width, threads), 9),
            width,
        ),
    )


def _lane_use(channel_vectors: int, width: int, threads: int) -> float:
    """The busy share of a CTA's lanes over ``ceil(channel_vectors / width)`` tiles of ``width``."""

    return (
        channel_vectors
        / (_ceil_div(channel_vectors, width) * width)
        * (width * (threads // width) / threads)
    )


def plan(
    S: int,
    C: int,
    num_docs: int,
    *,
    geometry: dict[str, int],
    role: str | None = None,
    legacy: bool = False,
) -> dict[str, int]:
    """The kernel's launch scalars for one problem under a record's build geometry and role.

    ``geometry`` is ``record["grid_rule"]["geometry"]``: ``threads`` per CTA
    and ``vec`` BF16 channels per thread; ``role`` is ``record["role"]``
    (``"fwd"`` or ``"bwd"``).  The scalars are ``S``, ``C``, ``num_docs``,
    :func:`channel_split`'s ``ct_vec`` / ``n_sub`` / ``n_ch_tiles`` (``role``
    and ``legacy`` are passed on) and ``search_steps``, the number of halvings
    of the device-side binary search over ``cu_seqlens`` (``max(1, ceil(log2
    num_docs))``).  Every value is a pure function of ``(S, C, num_docs)``, the
    geometry and the role: the host never reads ``cu_seqlens``, and the kernel
    derives its token ranges from ``S`` and the grid.  The ``flags`` scalar of
    the ``dtype_flags`` programs is not a plan value: the host derives it from
    the tensors of the call.
    """

    S, C, num_docs = int(S), int(C), int(num_docs)
    if S < 0 or num_docs < 0:
        raise ValueError(f"S and num_docs must be non-negative, got {(S, num_docs)}")
    ct_vec, n_sub, n_ch_tiles = channel_split(
        C, geometry=geometry, role=role, legacy=legacy
    )
    return {
        "S": S,
        "C": C,
        "num_docs": num_docs,
        "search_steps": max(1, (num_docs - 1).bit_length()),
        "ct_vec": ct_vec,
        "n_sub": n_sub,
        "n_ch_tiles": n_ch_tiles,
    }


# Whole-rank trim of the backward's single-wave main grid (:func:`main_grid`).
# A channel tile ``ct`` of a ``grid``-CTA launch gets ``ceil((grid - ct) /
# n_ch_tiles)`` CTAs, so when ``n_ch_tiles`` does not divide the grid two pairs
# of neighbouring tiles stream other token ranges at any instant; a row that is
# not 64-byte aligned shares a cache line (and a 32-byte ``dx`` sector) between
# them, which beyond L2 is then fetched twice.  Dropping ``grid % n_ch_tiles``
# CTAs gives every tile the same CTA count.  ``TILE_ALIGNED_GRID[arch]``: the
# co-resident grid the rule was measured at (GB300's 304 CTAs; 0: none), on the
# routes of ``TILE_ALIGNED_ROUTES``, from ``TILE_ALIGNED_MIN_S`` tokens, with a
# last tile of at least ``TILE_ALIGNED_MIN_LAST[ct_vec]`` vectors and at most
# ``TILE_ALIGNED_MAX_TILES[ct_vec]`` tiles (a ``ct_vec`` without an entry:
# never).  ``TILE_ALIGNED_CLASSES[arch]`` holds the other devices' measured
# classes, ``(co_resident, ct_vec, n_ch_tiles) -> (min S, min last-tile
# vectors)`` on the routes of ``TILE_ALIGNED_CLASS_ROUTES[arch]`` (B200's 296
# CTAs; R200 none).  Every other launch keeps its grid.  ``dx`` does not depend
# on the grid; ``dweight`` / ``dbias`` sum other partial rows, bitwise
# reproducible for the trimmed grid.  The rule is kept in sync with its copy in
# the backward kernel generator.
TILE_ALIGNED_GRID: dict[str, int] = {"sm_100a": 0, "sm_103a": 304, "sm_107a": 0}
TILE_ALIGNED_ROUTES: tuple[str, ...] = ("bwd",)
TILE_ALIGNED_MIN_S = 98304
TILE_ALIGNED_MIN_LAST: dict[int, int] = {32: 15, 64: 49}
TILE_ALIGNED_MAX_TILES: dict[int, int] = {32: 27, 64: 9}
TILE_ALIGNED_CLASSES: dict[str, dict[tuple[int, int, int], tuple[int, int]]] = {
    "sm_100a": {
        (296, 32, 5): (147456, 27),
        (296, 32, 7): (65536, 31),
        (296, 64, 3): (147456, 59),
    },
    "sm_103a": {},
    "sm_107a": {},
}
TILE_ALIGNED_CLASS_ROUTES: dict[str, tuple[str, ...]] = {
    "sm_100a": ("bwd", "bwd_dt", "bwd_fp32g"),
    "sm_103a": (),
    "sm_107a": (),
}


def main_grid(
    co_resident: int,
    *,
    arch: str,
    grid_mult: int = 1,
    C: int | None = None,
    n_ch_tiles: int = 1,
    S: int | None = None,
    ct_vec: int | None = None,
    route: str | None = None,
) -> int:
    """The backward main launch's CTA count: ``grid_mult x co_resident``.

    ``co_resident`` is the program's co-resident CTA count on the device
    (``ctas_per_sm x SM count``).  A single-wave launch (``grid_mult`` 1) of
    a row the caller names (``C``, ``S``, the plan's ``n_ch_tiles`` and
    ``ct_vec``, the ``route``) is trimmed to whole channel-tile ranks:
    :func:`tile_aligned_grid` on a route of ``TILE_ALIGNED_ROUTES`` when
    ``co_resident`` is ``TILE_ALIGNED_GRID[arch]``, else
    :func:`class_aligned_grid` on a route of
    ``TILE_ALIGNED_CLASS_ROUTES[arch]``.  A caller that names no row keeps the
    untrimmed grid, the most CTAs any row launches.
    """

    if type(grid_mult) is not int or grid_mult < 1:
        raise ValueError(f"grid_mult must be a positive int, got {grid_mult!r}")
    grid = int(co_resident) * grid_mult
    named = None not in (C, S, ct_vec)
    if (
        grid_mult == 1
        and co_resident == TILE_ALIGNED_GRID[arch]
        and route in TILE_ALIGNED_ROUTES
        and named
    ):
        grid = tile_aligned_grid(grid, C, n_ch_tiles, S=S, ct_vec=ct_vec)
    elif grid_mult == 1 and route in TILE_ALIGNED_CLASS_ROUTES[arch] and named:
        grid = class_aligned_grid(grid, C, n_ch_tiles, arch=arch, S=S, ct_vec=ct_vec)
    return grid


def tile_aligned_grid(
    grid: int, C: int, n_ch_tiles: int, *, S: int, ct_vec: int, vec: int = 8
) -> int:
    """``grid`` less ``grid % n_ch_tiles`` CTAs when ``C % 32`` (a row of
    ``x`` / ``dy`` / ``dx`` is not 64-byte aligned), ``1 < n_ch_tiles <=
    TILE_ALIGNED_MAX_TILES[ct_vec]``, ``grid % n_ch_tiles``, ``grid >
    n_ch_tiles``, ``S >= TILE_ALIGNED_MIN_S`` and the last channel tile holds
    at least ``TILE_ALIGNED_MIN_LAST[ct_vec]`` vectors (``C / vec -
    (n_ch_tiles - 1) x ct_vec``); else ``grid``."""

    last = C // vec - (n_ch_tiles - 1) * ct_vec
    if (
        1 < n_ch_tiles <= TILE_ALIGNED_MAX_TILES.get(ct_vec, 0)
        and C % 32
        and grid % n_ch_tiles
        and grid > n_ch_tiles
        and S >= TILE_ALIGNED_MIN_S
        and last >= TILE_ALIGNED_MIN_LAST.get(ct_vec, ct_vec + 1)
    ):
        return grid - grid % n_ch_tiles
    return grid


def class_aligned_grid(
    grid: int,
    C: int,
    n_ch_tiles: int,
    *,
    arch: str,
    S: int,
    ct_vec: int,
    vec: int = 8,
) -> int:
    """``grid`` less ``grid % n_ch_tiles`` CTAs when ``C % 32``, ``grid %
    n_ch_tiles`` and ``(grid, ct_vec, n_ch_tiles)`` is a class of
    ``TILE_ALIGNED_CLASSES[arch]`` whose bounds the row meets (``S`` and the
    last tile's ``C / vec - (n_ch_tiles - 1) x ct_vec`` vectors at least the
    class's); else ``grid``."""

    bound = TILE_ALIGNED_CLASSES[arch].get((grid, ct_vec, n_ch_tiles))
    if bound is None or not C % 32 or not grid % n_ch_tiles or bound[0] > S:
        return grid
    if C // vec - (n_ch_tiles - 1) * ct_vec < bound[1]:
        return grid
    return grid - grid % n_ch_tiles


def row_may_trim(
    record: dict[str, Any], arch: str, co_resident: int, factor: int, C: int, S: int
) -> bool:
    """Whether :func:`main_grid` may trim ``record``'s launch for the row.

    Every trim needs a single wave (``factor`` 1), ``C % 32``, the record on a
    route of the rule and ``S`` at least the rule's smallest token bound for
    ``co_resident``; a call failing any of these launches the untrimmed grid
    without planning the row's channel tiles (no host cost beyond the
    untrimmed grid's)."""

    if factor != 1 or not C % 32:
        return False
    library = record["library"]
    routes = ROUTES[arch]
    if (
        co_resident == TILE_ALIGNED_GRID[arch]
        and S >= TILE_ALIGNED_MIN_S
        and any(library in routes.get(kind, ()) for kind in TILE_ALIGNED_ROUTES)
    ):
        return True
    return any(
        grid == co_resident and bound[0] <= S
        for (grid, _, _), bound in TILE_ALIGNED_CLASSES[arch].items()
    ) and any(
        library in routes.get(kind, ()) for kind in TILE_ALIGNED_CLASS_ROUTES[arch]
    )


def record_route(record: dict[str, Any], arch: str) -> str:
    """The route of ``ROUTES[arch]`` that lists ``record`` (``ValueError`` if none)."""

    return _library_route(record["library"], arch)


@functools.cache
def _library_route(library: str, arch: str) -> str:
    for kind, names in ROUTES[arch].items():
        if library in names:
            return kind
    raise ValueError(f"module {library} is on no route of {arch}")


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""

    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def launch_grid(
    record: dict[str, Any],
    *,
    device_index: int,
    C: int | None = None,
    S: int | None = None,
) -> tuple[int, int, int]:
    """Launch grid of one module on ``device_index`` under its recorded grid rule.

    ``co_resident`` modules launch exactly ``ctas_per_sm x SM count`` CTAs,
    the co-resident count (a cooperative KCP backward never exceeds it).
    ``occupancy`` modules launch :func:`main_grid` of that count:
    ``grid_mult[arch] x ctas_per_sm x SM count`` CTAs normally
    (:func:`grid_per_sm`): ``ctas_per_sm`` co-resident CTAs per SM and
    ``grid_mult`` times finer token streams, so a CTA that finds no free slot
    beside another stream's kernel trails by a short stream in a later wave.
    Given the call's ``C`` and ``S``, a single-wave launch of the rows
    :func:`main_grid` names (its route, the record's plan) drops CTAs to whole
    channel-tile ranks (the row is planned only where :func:`row_may_trim`
    holds).  Either grid is a pure function of ``(S, C, device)`` and the
    route, so is the deterministic ``dweight`` order, and the untrimmed grid
    (``C`` or ``S`` not given) is the most CTAs any call launches, so one
    workspace serves every token count.
    ``channels_per_cta`` modules (the reduce programs) launch ``ceil(C /
    channels_per_cta)`` CTAs for the call's ``C`` channels.
    """

    rule = record["grid_rule"]
    kind = rule["kind"]
    if kind in ("co_resident", "occupancy"):
        arch = device_arch(device_index)
        grid_x = grid_per_sm(record, arch) * sm_count(device_index)
        if kind == "occupancy" and C is not None and S is not None and C % 32:
            co_resident = int(rule["ctas_per_sm"]) * sm_count(device_index)
            factor = grid_mult(record, arch)
            if row_may_trim(record, arch, co_resident, factor, int(C), int(S)):
                ct_vec, _, n_ch_tiles = channel_split(
                    C, geometry=rule["geometry"], role=record["role"]
                )
                grid_x = main_grid(
                    co_resident,
                    arch=arch,
                    grid_mult=factor,
                    C=int(C),
                    n_ch_tiles=n_ch_tiles,
                    S=int(S),
                    ct_vec=ct_vec,
                    route=record_route(record, arch),
                )
    elif kind == "channels_per_cta":
        if C is None:
            raise ValueError(
                f"the {kind} grid rule of module {record['kernel']} needs the channel count C"
            )
        grid_x = _ceil_div(C, int(rule["channels_per_cta"]))
    else:
        raise ValueError(
            f"unknown Cake causal conv1d training grid rule: {kind!r} (known: {GRID_KINDS})"
        )
    return (max(1, grid_x), 1, 1)


def partial_rows(grid_x: int, n_ch_tiles: int) -> int:
    """Rows of the backward ``partial`` workspace for a grid: ``ceil(grid_x / n_ch_tiles)``.

    CTA ``b`` of the grid owns channel tile ``b % n_ch_tiles`` and partial row
    ``b // n_ch_tiles``, so tile ``t`` holds ``ceil((grid_x - t) / n_ch_tiles)``
    written rows; the reduce program derives exactly those from ``main_ctas``
    (= ``grid_x``) and ``n_ch_tiles`` and sums nothing else.
    """

    return _ceil_div(grid_x, n_ch_tiles)


def partial_bytes(record: dict[str, Any], C: int, rows: int) -> int:
    """Bytes of ``rows`` FP32 ``[C, taps]`` partial rows of the module's workspace rule.

    ``taps`` is 4 for the dweight partials; a bias program's rule says 5: its
    kernel lays the ``[rows, C, 4]`` tap slab out first and the ``[rows, C]``
    dbias slab after it (the host view is capacity only).
    """

    rule = record["workspace_rule"]
    if rule is None or rule["kind"] != "partial_rows":
        raise ValueError(f"module {record['kernel_symbol']} owns no backward workspace")
    return 4 * int(rule["taps"]) * int(C) * int(rows)


def workspace_layout(
    record: dict[str, Any], C: int, *, device_index: int
) -> dict[str, int]:
    """Byte layout of the caller-owned backward workspace of one module for ``C`` channels.

    The workspace is one CUDA ``uint8`` buffer holding the FP32 ``[rows, C, taps]``
    ``dweight`` (and, ``taps`` 5, dbias) partials, ``rows = partial_rows(grid,
    n_ch_tiles)`` for the device's launch grid (``grid_mult x ctas_per_sm x SM
    count``: the bytes grow with the record's ``grid_mult``); it depends on
    ``C`` and the device only, so one buffer serves every token count.  The
    kernel fully writes the rows its reduce program sums, so nothing is zeroed
    between launches or CUDA Graph replays.
    """

    grid_x = launch_grid(record, device_index=device_index)[0]
    _, _, n_ch_tiles = channel_split(
        C, geometry=record["grid_rule"]["geometry"], role=record["role"]
    )
    rows = partial_rows(grid_x, n_ch_tiles)
    return {
        "grid": grid_x,
        "rows": rows,
        "taps": int(record["workspace_rule"]["taps"]),
        "total_bytes": partial_bytes(record, C, rows),
    }


def counter_words(record: dict[str, Any], n_ch_tiles: int) -> int:
    """Int32 words of a forward record's ``work`` counters for a launch of ``n_ch_tiles`` channel tiles.

    A ``tile_counters`` rule holds ``words_per_tile`` words per channel tile
    (the dynamic forward's next-chunk counter and done-CTA counter); a record
    without a workspace rule takes no ``work`` (0 words).
    """

    rule = record["workspace_rule"]
    if rule is None:
        return 0
    if rule["kind"] != "tile_counters":
        raise ValueError(
            f"module {record['kernel']}: workspace rule {rule['kind']!r} holds no forward counters"
        )
    if n_ch_tiles <= 0:
        raise ValueError(f"n_ch_tiles must be positive, got {n_ch_tiles}")
    return int(rule["words_per_tile"]) * int(n_ch_tiles)


def forward_kinds(arch: str) -> tuple[str, ...]:
    """The forward routes ``arch`` exports, in :data:`KINDS` order."""

    return tuple(
        kind
        for kind in KINDS
        if route_role(kind) == "fwd" and ROUTES.get(arch, {}).get(kind)
    )


def forward_workspace_words(arch: str, C: int) -> int:
    """Int32 words of the forward workspace for ``C`` channels on ``arch``.

    The largest :func:`counter_words` over every forward program of every
    forward route of the architecture (each planned with its own geometry and
    role), so one buffer serves every forward route; it depends on ``C`` only
    (never on ``S`` or the device's SM count).  0 when no forward program of
    the architecture takes ``work``.
    """

    words = 0
    for kind in forward_kinds(arch):
        for name in route_modules(arch, kind):
            record = MODULES[name]
            if record["workspace_rule"] is None:
                continue
            _, _, n_ch_tiles = channel_split(
                C, geometry=record["grid_rule"]["geometry"], role=record["role"]
            )
            words = max(words, counter_words(record, n_ch_tiles))
    return words


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


_GRID_AXES = {"grid_x": 0, "grid_y": 1, "grid_z": 2}


@functools.cache
def _arg_sources(name: str) -> tuple[tuple[bool, Any], ...]:
    """``(grid, source)`` per entry of ``MODULES[name]["arg_plan"]``, in
    order: a grid slot (``source`` is the axis, 0 / 1 / 2 for ``grid_x`` /
    ``grid_y`` / ``grid_z``) or the logical name the kernel parameter binds
    (the record's ``logical`` map); the registry is constant, so
    :func:`run_modules` reads it once per module."""

    record = MODULES[name]
    logical = record["logical"]
    return tuple(
        (True, _GRID_AXES[key]) if kind == "grid" else (False, logical[key])
        for kind, key in record["arg_plan"]
    )


def _missing_argument(name: str, values: dict[str, Any]) -> ValueError:
    """The error of a launch of module ``name`` whose ``values`` lack a
    logical name it binds (the first such name, with its kernel parameter)."""

    record = MODULES[name]
    logical, key = next(
        (record["logical"][key], key)
        for kind, key in record["arg_plan"]
        if kind != "grid" and record["logical"][key] not in values
    )
    return ValueError(f"module {name} needs {logical!r} (kernel parameter {key!r})")


@functools.cache
def _ffi_entry(name: str, device_index: int) -> Any:
    """The FFI entry of module ``name`` loaded for CUDA device
    ``device_index``'s architecture (:func:`load`), looked up once."""

    module = load(name, device_arch(device_index))
    return getattr(module, MODULES[name]["ffi_entry"])


@functools.cache
def _raw_stream_getter() -> Any:
    """``torch._C._cuda_getCurrentRawStream``: the handle of a device's
    current torch stream as an integer, without constructing a ``Stream``
    object; or the ``torch.cuda`` form on a torch build without it.  Bound on
    first use: this module imports torch lazily."""

    import torch

    getter = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if getter is not None:
        return getter

    def fallback(device_index: int) -> int:
        return torch.cuda.current_stream(device_index).cuda_stream

    return fallback


_FFI_DEVICES: dict[int, Any] = {}


def _stream_context(device_index: int) -> Any:
    """The tvm-ffi environment-stream context of torch's current stream on
    CUDA device ``device_index`` (a CUDA graph's capture stream during
    capture): the bindings launch on that stream.  The FFI device handle is
    kept per index, so entering the context costs a fraction of
    ``tvm_ffi.use_torch_stream()``, which builds a ``torch.cuda.Stream`` and
    parses a device string on every entry."""

    device = _FFI_DEVICES.get(device_index)
    if device is None:
        device = _FFI_DEVICES[device_index] = tvm_ffi.device(f"cuda:{device_index}")
    return tvm_ffi.use_raw_stream(device, _raw_stream_getter()(device_index))


def run_modules(
    launches: Sequence[tuple[str, tuple[int, int, int] | None]],
    values: dict[str, Any],
    *,
    device_index: int,
) -> None:
    """Launch exported modules in order, back to back, with one set of logical
    argument ``values``.

    ``launches`` holds ``(name, grid)`` pairs, ``grid`` as in
    :func:`run_module`.  All of them are issued inside one stream context of
    torch's current stream on ``device_index``, each bound (grid and
    arguments) right before it is issued, so a programmatically dependent
    launch (the backward's reduce program) reaches the stream right behind
    the program it depends on, while that one runs, and the first launch
    waits for no later one's binding.  ``ValueError`` names the first logical
    name a module binds that ``values`` lacks.
    """

    with _stream_context(device_index):
        for name, grid in launches:
            if grid is None:
                grid = launch_grid(
                    MODULES[name],
                    device_index=device_index,
                    C=values.get("C"),
                    S=values.get("S"),
                )
            sources = _arg_sources(name)
            try:
                args = [
                    grid[source] if is_grid else values[source]
                    for is_grid, source in sources
                ]
            except KeyError:
                raise _missing_argument(name, values) from None
            _ffi_entry(name, device_index)(*args)


def run_module(
    name: str,
    values: dict[str, Any],
    *,
    device_index: int,
    grid: tuple[int, int, int] | None = None,
) -> None:
    """Launch one exported module with logical argument ``values``.

    ``values`` maps the logical names in ``LOGICAL_TENSORS`` /
    ``LOGICAL_SCALARS`` to tensors and Python scalars; the module record maps
    its kernel parameter names onto that vocabulary.  The launch grid is
    ``grid`` when the caller planned it (the module's :func:`launch_grid` for
    the call, as the public entry points pass it), else the module's
    :func:`launch_grid` on ``device_index`` for the call's ``C`` and ``S``,
    and the kernel runs on the current torch stream; a
    module whose launch record enables programmatic dependent launch
    (``use_pdl``) overlaps its launch with the preceding kernel of that stream
    and waits for it in-kernel.
    """

    run_modules(((name, grid),), values, device_index=device_index)


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "CHANNEL_MULTIPLE",
    "FLAG_B_BF16",
    "FLAG_DB_BF16",
    "FLAG_DW_BF16",
    "FLAG_W_BF16",
    "GRID_KINDS",
    "INT32_MAX",
    "KINDS",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "PLAN_SCALARS",
    "REQUIRED_KINDS",
    "ROLES",
    "ROUTES",
    "SELECT_KINDS",
    "TILE_ALIGNED_CLASSES",
    "TILE_ALIGNED_CLASS_ROUTES",
    "TILE_ALIGNED_GRID",
    "TILE_ALIGNED_MAX_TILES",
    "TILE_ALIGNED_MIN_LAST",
    "TILE_ALIGNED_MIN_S",
    "TILE_ALIGNED_ROUTES",
    "WIDTH",
    "WORKSPACE_KINDS",
    "arch_for_capability",
    "backward_kinds",
    "backward_route",
    "channel_split",
    "class_aligned_grid",
    "counter_words",
    "device_arch",
    "forward_kinds",
    "forward_route",
    "forward_workspace_words",
    "grid_mult",
    "grid_per_sm",
    "launch_grid",
    "library_name",
    "load",
    "main_grid",
    "partial_bytes",
    "partial_rows",
    "plan",
    "record_route",
    "route_applies",
    "route_modules",
    "route_role",
    "row_may_trim",
    "run_module",
    "run_modules",
    "select_rule",
    "selected_modules",
    "sm_count",
    "specs_for_arch",
    "tile_aligned_grid",
    "workspace_layout",
    "workspace_owners",
    "workspace_record",
]
