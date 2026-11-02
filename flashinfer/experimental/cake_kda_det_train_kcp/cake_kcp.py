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

Host backend of the context-parallel (KCP) route of the Cake packed KDA
chunked training kernels (SM100 / SM103 / SM107): the transient
:class:`WorkspaceDetKcp`, the forward and backward launch sequences with their
collectives, the saved set :class:`SavedDetKcp`, the autograd Function and the
entry points ``chunk_kda_det_kcp`` / ``chunk_kda_det_fwd_kcp`` / ``chunk_kda_det_bwd_kcp``.

Every rank holds its own token fragments (``[T_r, H, 128]`` operands, local
``cu_seqlens`` with one sequence per fragment) and only recurrent-state
summaries cross the ranks.  Forward: the crossing-out fragments' affine
summaries ``(B_f, M_f^T)`` (the walk from zero and the FP32-class transfer
walk) go to the merge owner of each head slice (``c1``, one
``all_to_all_single``), the owner composes every document chain in document
order with the exact FP32 affine merge and returns each continuing fragment's
initial state ``S_in`` (``c2``); the local recurrence then runs from ``S_in``.
Backward: the adjoint offsets ``d_f`` (the reverse walk from zero) and, unless
the forward's transfer was kept, ``M_f^T`` again go to the owners (``b1``), the
owner walks each chain backwards (``X <- M_f^T X + d_f``) and returns every
crossing-out fragment's ``dS_out`` (``b2``); the local adjoint recurrence runs
from it.  One compute stream; the collectives are asynchronous on the caller's
process group and waited at their first consumer.  Shipped schedule (adopted in an
earlier revision): ``wait_first`` -- no merge
or tape walk is issued while a collective is in flight, and the tape walks of a
phase are one fused launch from the carry plane; the earlier ``overlap`` form
(the document-start tape walk under ``c1``, the document-last adjoint tape walk
under ``b2``: the two R7 overlaps) and the separate walks stay selectable through
the test knobs ``schedule`` / ``fused_walks``.  Nothing launched between an issue
and its wait reads that collective's buffers, so the bits are independent of the
schedule and a checkpoint replay re-runs the same program order on every rank.
CUDA-graph replay (adopted in an earlier revision; keyword
``graph="replay"`` of :func:`chunk_kda_det_kcp`, :func:`chunk_kda_det_fwd_kcp` and
:func:`chunk_kda_det_bwd_kcp`, shipped default ``GRAPH_DEFAULT = "off"``): the
first call of a validity key runs eagerly, the second is captured into a
``torch.cuda.CUDAGraph`` and every later call of the key replays it -- the same
launches and collectives, bitwise the eager result.  The key is the call's
kind, its ContextDetKcp / workspace / backend objects, every option and the
shape / stride / dtype / data pointer of every operand and saved tensor: a
framework gets the replay only with pointer-stable operands and the same
objects per layer; anything else is a silent eager call (``GRAPH_STATS``,
``LAST_CALL[kind]["graph_event"]``).  A replayed call returns the graph's static
tensors, overwritten by the next replay of the key, so outputs and the saved
set must be consumed before the next call of the same layer.  Each cached key
pins a private memory pool (0.7-9.0 GiB per rank per key on the A/B
rows; at most :data:`GRAPH_CACHE_MAX` keys); :func:`graph_cache_clear` releases
them and must run before ``torch.distributed.destroy_process_group()``.
``exchange="ring"``, ``debug_sequence`` and calls without a caller-owned
workspace are never captured.

Shipped walk-launch geometry (adopted in an earlier revision;
``WALK_GEOMETRY_DEFAULT = "lpt_tape_one_cta_per_sm"``): the offset walk lists of
K1 / K2 / K3 launch in LPT order (longest segment first, ties by segment number;
the tape lists already did) and the tape programs ``fwdh_tape`` / ``dhu_tape``
run at one CTA per SM through the device cache-config preference that brackets
each of their launches (``cudaFuncCachePreferL1`` before, ``cudaFuncCachePreferNone``
after; the offset programs keep two CTAs per SM).  Order and placement never
change a bit: every CTA walks its own (segment, head) chain into its own indexed
tile.  The earlier geometry ``"shipped"`` and the other :data:`WALK_GEOMETRIES`
stay selectable through the test knob ``walk_geometry`` for the A/B record.

Walk form (``WALK_FORM_DEFAULT = "auto"``): each K1 / K2 walk launch runs its
program's full-width record (one CTA per segment and head) or its two-CTA
record (two CTAs per segment and head over 64 value columns each, bitwise
equal), chosen per launch by the registry's rule ``cake_kcp_jit.walk_form``
from the launch's chains, head count and SM count: the two-CTA form when the
launch list-schedules within its longest chain (one wave of co-resident
half-width CTAs, which hide each other's per-step latency), the full-width
form for the multi-wave launches (more work per issued instruction).  A split
tape launch is never capped at one CTA per SM (the cap serves the full-width
tape CTAs, which thrash when co-resident).  The test knob ``walk_form``
(``"full"`` / ``"split"``) forces one form on every launch for the A/B record;
the call's ``walk_form_log`` records ``(program, form, n_walk)`` per launch
(``LAST_CALL[kind]["walk_forms"]`` keeps the copy).
The tape programs' rule is ``split`` (their full-width records are test-only,
under ``walk_form="full"``).  The rule's inputs are static per ``(ctx, H)``, so
the form of each ``(program, walk list)`` is decided once per table set
(``_Tables.walk_form``) and looked up on every later launch: the eager call
carries no per-launch list scheduling.

The bracket is host-side runtime state, so it is never issued inside a CUDA
stream capture (``cudaDeviceSetCacheConfig`` would invalidate the capture,
see below).  Under ``graph="replay"`` the captured call carries the tape
one-CTA-per-SM cap itself (the shipped
``GRAPH_CAP_DEFAULT = "node_attr"``): right after each tape launch is
captured, the host reads the kernel node it just added
(``cudaStreamGetCaptureInfo``; CUDA 12 runtimes export that signature as
``cudaStreamGetCaptureInfo_v2``, CUDA 12.3+ also as ``_v3``) and sets its
preferred shared-memory carveout launch attribute
(``cudaLaunchAttributePreferredSharedMemoryCarveout`` =
``cudaSharedmemCarveoutMaxL1``, ``cudaGraphKernelNodeSetAttribute`` with a
``cudaGraphKernelNodeGetAttribute`` read-back), so every replay runs the tape
programs at one CTA per SM like the eager bracket does; the eager path is
untouched.  The test knob ``graph_cap`` keeps the other forms selectable for
the A/B record: ``"none"`` (the previous form of record: no
device preference, the tape programs replay uncapped), ``"capture_pref"``
(PreferL1 set once before the capture of a key and reset right after it ends:
6.8-18 % slower on every composition row of the gate -- rejected) and
``"node_attr_post"`` (the same attribute set on the kept node handles after
the capture ends and before the graph is instantiated,
``torch.cuda.CUDAGraph(keep_graph=True)``).  Every call records its form as
``LAST_CALL[kind]["cache_config"]`` (``"bracket"`` / ``"node_attr"`` /
``"node_attr_post"`` / ``"capture_pref"`` / ``"none"`` / ``"unavailable"``);
the node forms also record ``cache_config_nodes`` = the number of capped tape
launches of the captured call.  Degrade contract: when the
runtime lacks one of the capture-graph entry points, or the capture-info call
does not report exactly one kernel-node dependency after a tape launch, the
capture continues UNCAPPED (the ``"none"`` behaviour) after one
``RuntimeWarning`` per process and the call records ``cache_config`` =
``"unavailable"`` (``cache_config_degraded`` = the reason) -- never an eager
fallback; an error returned by the set-attribute call keeps the capture-failure
contract (that key runs eagerly, counted in ``GRAPH_STATS["fallback"]``).  When
no CUDA runtime soname loads, the shipped geometry keeps the LPT order,
launches the tape programs without the cap after one ``RuntimeWarning`` and
records ``"unavailable"``.  Host-only ``ctypes`` entry points
of this module:
``cudaDeviceSetCacheConfig``, ``cudaDeviceGetCacheConfig``,
``cudaGetLastError``, ``cudaRuntimeGetVersion``, ``cudaStreamGetCaptureInfo``
(``_v2`` / ``_v3``), ``cudaGraphNodeGetType``,
``cudaGraphKernelNodeSetAttribute``, ``cudaGraphKernelNodeGetAttribute``.

Shipped host path (this revision;
``HOST_TRIM_DEFAULT = 1``): the host work of an eager call that does not depend
on the call is done once and reused.  The entry point validates ``**options``
once and the autograd node carries the :class:`OptionsDetKcp`;
``WorkspaceDetKcp.ensure`` returns the last tables for the same (context, H,
device, direction, knobs) and keeps the ``pool[:n]`` views of an unchanged pool
tensor; the local stages launch from a per-(arena, table set, direction) value
template with a cached argument plan, arch and raw-stream FFI context; the
K1-K4 launches skip the positional ABI that the ``run_program`` provider form
discards and pass the table set's own device tables.  Every cache is keyed on
object identity plus host integers and never reads a tensor's contents:
``_Tables.dev_list`` (the walk / chunk-start list object, device, dtype),
``_Tables.indexed_meta`` (the index and tile list objects, H, device),
``_Tables.device_tiles`` by identity (the tile table object, device),
``Chains.device_tables`` (the chain set, device), ``_Tables.walk_all_p0`` /
``walk_all_rest`` (the P = 1 tape-replay sublists, one object per table set),
the workspace's last ``ensure`` key and its ``(pool name, n)`` views while the
pool tensor is the same object (the views of a replaced pool are dropped), the
local arena's ``begin_call`` templates ((table set, direction) -> the arena
views and plan scalars) and the per-device-index arch / FFI device of this
module and of :mod:`.cake_kcp_jit`.  Owners and bounds: a table
set's identity-keyed caches hold at most ``ID_CACHE_MAX = 256`` entries each
(emptied beyond; a caller passing fresh objects cannot grow them) and die with
the table set; the chain tables die with their chain set; the ``ensure`` key and
the pool views die with the workspace; the templates live on the arena they
describe (``TEMPLATE_CACHE_ATTR``) and reference the table set only weakly (a
dead table set's entry goes on the next miss), so they die with the arena -- the
backend itself holds no workspace, arena or table set (an earlier form
kept them on the backend for the process lifetime); the conv halo module bounds its
per-context cache at
``HOST_CACHE_MAX = 64`` contexts.  The same kernels launch with the same
arguments in the same order on the same stream, so no bit changes (bitwise on
every A/B and confirmation row of the adoption); under ``graph="replay"`` the
trimmed segments are not part of the replayed call.
``_kcp_test_knobs={"host_trim": 0}`` selects the previous host path for the
A/B record; the D1 explicit-record merge form (``cols_rule="by_chains"`` beyond
``COLS_RULE_CHAINS`` state chains) needs the positional ABI and a diagnostics
launcher on the arena keeps ``KdaWorkspace.run``: both keep the previous path
under either value of ``host_trim``.

Saved carry plane (``SAVED_CARRY_DEFAULT = 1``): the forward of a ``P > 1``
rank keeps its whole carry plane in the saved set (``SavedDetKcp.carry``) and the
saved-mode backward replays its tape from it instead of re-walking every
segment (K1 ``_offset``) and re-merging the carries -- the same remaining
launches, bitwise; ``kda_det_saved_bytes_kcp`` counts the plane.
``_kcp_test_knobs={"saved_carry": 0}`` selects the re-walk for the A/B record;
the replay mode (checkpoint recompute) and ``P = 1`` ranks are the same under
both values.

The metadata (partition, context, head ownership, chain tables, split sizes,
segment plan, compact-buffer addressing, byte queries) is
:mod:`.cake_kcp_plan`; this module reads tensor contents nowhere, dispatches
from metadata only, allocates the outputs and the saved set fresh per call and
keeps only transient buffers in the workspace.  The kernels are launched
through a dispatch table (:class:`Backend`): the shipped backend runs the
exported programs of :mod:`.cake_kcp_jit` and the local chunk stages through
:mod:`..cake_kda_det_train.cake_backend`; a test may install another backend (an FP64 emulation of
the same ABI) for the orchestration tests.

Semantics and contracts follow the KCP design note (its sections 3.5, 3.6, 6.4,
6.5, 7.2-7.5, 9, 10 and 16).

Checkpoint recompute without the state exchange (FlashInfer issue #7815; the
GDN twin is #7818): with ``chunk_kda_det_kcp(..., recompute_handle=)``
(:class:`RecomputeHandleDetKcp`, or any object with an ``entries`` dict) the
original forward exchanges as always and stores its incoming states ``S_in``
(the saved set's own tensor, key ``"kda"``; with ``checkpoint_keep="transfer"``
also the owner's c1 receive tensor) in the handle; torch's recompute of that
forward inside the backward binds them and runs the rank-local programs only --
no c1 / c2 (no ring hop), no wait, no owner merge, no summary walk at ``P = 1``
-- so ``out`` and the saved set are bitwise those of the exchanging recompute.
The backward is unchanged (``checkpoint_keep="states"``, the default, runs it
with ``save_transfer=False``: bitwise the default form).

Raw gate mode (FlashInfer issue #7816; ``use_gate_in_kernel=True`` with
``A_log`` / ``dt_bias`` / ``lower_bound`` on the three entry points, the
single-rank keywords): ``g`` is this rank's raw BF16 gate (a strided view under
the single-rank rule, never copied) and the local stages run the plan's raw
records (``cake_backend.plan(..., gate="raw")``: the intra and prep programs
activate the gate ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` in
their prefix passes, the raw GLM intra program writes the BF16 gradient of the
raw gate and per-chunk ``(hi, lo)`` partials of the parameter gradients, one
``param_finalize`` launch reduces them, no gate epilogue runs).  The ``gk`` rows
the K1 / K2 / K3 walks read are the explicit path's bits for the same activated
gate, so every KCP program, collective and its byte count is unchanged; no
full-size FP32 gate or gate gradient exists on the route.  ``dA_log`` /
``ddt_bias`` of a rank are its partial sums (the gradients through its own
tokens, reduced on the rank in the fixed order of its local chunks): the
training framework sums them over ``ctx.group`` like the gradients of any
parameter replicated across the context-parallel ranks, or calls
:func:`reduce_gate_param_grads_det_kcp` (an all-gather, then an FP32 sum in rank
order) when that sum must be bitwise reproducible -- an NCCL all-reduce does not
fix its summation order.  The workspace (:attr:`WorkspaceDetKcp.gate`), the
saved set and a recompute handle's fingerprint record the mode (and the lower
bound); a call in another mode is refused.  A raw-mode workspace runs the replay
local arena (``local_arena="lean"`` is refused in this mode).


P > 1 staging: the K1 ``_offset`` / K2 ``_offset`` / K3 passes of the segmented rank write two
dense planes ``stage_b`` / ``stage_m`` ([NSEG, H, 128, 128] FP32 each, ``tile_stride`` 1); K4 reads them element-wise through
``elem_m`` / ``elem_c`` (map / carry / transposed-map / adjoint-carry families); the pair convention stays
for K4's states (c1 / b1 pairs, ``tile_stride`` 2) and the carry plane; under ``save_transfer=True`` at ``P > 1`` the
``stage_m`` plane itself is the saved ``M_seg`` (no copy; ``kda_det_saved_bytes_kcp`` keeps the ``NSEG * s`` term).
"""

from __future__ import annotations

import ctypes
import dataclasses
import functools
import warnings
import weakref
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist

from ..cake_kda_det_train import cake_backend, cake_jit
from . import cake_kcp_plan

D = cake_backend.D  # head dimension (128)
BT = cake_backend.BT  # tokens per chunk (64)
SCALE = cake_backend.SCALE
SAVE_MODES = cake_backend.SAVE_MODES
TILE = D * D  # elements of one FP32 state tile [128, 128]
SUMMARY_TILE = 2 * TILE  # one c1 summary tile of one head: [2][128][128] = (B_f, M_f^T)
BOUNDARY_MODES = ("saved", "replay")
EXCHANGES = ("auto", "a2a", "ring")
OFFSET_CLASSES = ("fla", "fp32")
RING_MAX_HOPS = 2
TRANSFER_WALKS = (
    "bf16x3_hw64",
    "tf32_hilo",
    "bf16x3",
)  # K3 program family: the shipped BF16x3 half-width walk first (the others only via _kcp_test_knobs)  # exchange="auto" picks the ring when every chain has at most this many hops (SPEC-2)
KCP_SAVED_VERSION = 2  # saved set {tables, S_in [, c1 receive tensor, M_seg, carry plane] -- the optional members travel as flags}; version 1 also held the token-row Aqk / Akk
# Host / plan knobs (test knob class: reachable only through _kcp_test_knobs):
SCHEDULES = (
    "overlap",
    "wait_first",
)  # schedule: "wait_first" (shipped) = merges and tape walks after each wait; "overlap" = the earlier overlapped issue order of the design note §3.5 / §6.5 (test knob)
COLS_RULES = (
    "fixed",
    "by_chains",
)  # D1: K4 cols_per_cta of the 128-column state launches: the backend's, or 8 up to COLS_RULE_CHAINS chains else 16
COLS_RULE_CHAINS = 24  # D1 threshold (DECISIONS.md D1: <= 24 concurrent chains -> c8, otherwise c16; map launches stay c16)
# Walk-launch geometry: the launch geometry of the K1 / K2 / K3 walk kernels, a test
# knob until an earlier revision made "lpt_tape_one_cta_per_sm" the shipped default.  "shipped" =
# the earlier lists and launches (the offset / staging lists in segment order, the runtime's own occupancy); "lpt_chains" =
# every walk list in LPT order (longest segment first, ties by segment number; the tape lists already are); "one_cta_per_sm" =
# the K1 / K2 kernels (115712 B of dynamic shared memory: two CTAs fit an SM only at the full 228 KiB carveout) launched at
# one CTA per SM (the device cache-config preference bracketing the launch, see _FiPrograms; host side, no kernel change);
# "tape_one_cta_per_sm" = the same cap for the tape kernels only (fwdh_tape / dhu_tape: the step-0 probe found co-resident
# tape CTAs thrash while co-resident offset CTAs hide each other's latency); "lpt_*" = the LPT order plus that cap.  Order and
# placement never touch a bit: every CTA walks its own (segment, head) chain (the kernels' map is walk_seg[b // (nv H)], head
# (b // nv) % H, value slice b % nv with nv the record's CTAs per head) into its own indexed tile.
WALK_GEOMETRIES = (
    "shipped",
    "lpt_chains",
    "one_cta_per_sm",
    "lpt_one_cta_per_sm",
    "tape_one_cta_per_sm",
    "lpt_tape_one_cta_per_sm",
)
WALK_GEOMETRY_DEFAULT = "lpt_tape_one_cta_per_sm"  # shipped default: LPT offset lists + one CTA per SM for the tape programs; "shipped" = the earlier geometry, via _kcp_test_knobs
WALK_LPT_GEOMETRIES = ("lpt_chains", "lpt_one_cta_per_sm", "lpt_tape_one_cta_per_sm")
_WALK_K12 = (
    "fwdh_offset",
    "fwdh_tape",
    "dhu_offset",
    "dhu_tape",
)  # K3 (193 KiB) is one CTA per SM by its footprint
_WALK_TAPES = ("fwdh_tape", "dhu_tape")
WALK_CAP_PROGRAMS: dict[
    str, tuple[str, ...]
] = {  # geometry -> the programs launched at one CTA per SM
    "one_cta_per_sm": _WALK_K12,
    "lpt_one_cta_per_sm": _WALK_K12,
    "tape_one_cta_per_sm": _WALK_TAPES,
    "lpt_tape_one_cta_per_sm": _WALK_TAPES,
}
WALK_CAP_GEOMETRIES = tuple(WALK_CAP_PROGRAMS)
GRAPH_MODES = (
    "off",
    "replay",
)  # graph replay: "replay" = keyed CUDA-graph capture / replay of the per-rank call; public keyword graph= since an earlier revision
GRAPH_DEFAULT = "off"  # shipped: the eager path; "replay" is opt-in per call (section "CUDA-graph replay of a call" below)
GRAPH_CAPS = (
    "none",
    "capture_pref",
    "node_attr",
    "node_attr_post",
)  # the cache-config form of a captured call -- "none" (no device preference under replay) | "capture_pref" (PreferL1 set before the capture of a key, PreferNone right after its end) | the kernel-node forms ("node_attr" = the replay form of record): "node_attr" (the carveout launch attribute set on each captured tape kernel node right after its launch, inside the capture) | "node_attr_post" (the same attribute set on the kept node handles after the capture ends, before the graph is instantiated)
GRAPH_CAP_DEFAULT = "node_attr"  # the graph="replay" cache-config form of record -- the tape one-CTA-per-SM cap as a launch attribute of the captured kernel nodes; "none" (the previous form of record), "capture_pref" (rejected: 6.8-18 % slower on every composition row) and "node_attr_post" stay selectable through _kcp_test_knobs for the A/B record; the eager path is untouched
GRAPH_NODE_CAPS = (
    "node_attr",
    "node_attr_post",
)  # the kernel-node forms (the cap as a kernel-node launch attribute); "node_attr" shipped
GRAPH_NODE_CARVEOUT = 0  # cudaSharedmemCarveoutMaxL1 (driver_types.h): prefer the maximum L1 = the smallest shared-memory configuration that fits the kernel -> one 115712-B tape CTA per SM, the per-node form of WALK_CACHE_CONFIG_CAP
_LAUNCH_ATTR_SHMEM_CARVEOUT = 14  # cudaLaunchAttributePreferredSharedMemoryCarveout (driver_types.h, CUDA >= 12.4: valid for graph nodes and launches)
_GRAPH_NODE_TYPE_KERNEL = 0  # cudaGraphNodeTypeKernel
_STREAM_CAPTURE_ACTIVE = 1  # cudaStreamCaptureStatusActive
# Trimmed host path: host-side trimming of the eager call's Python /
# dispatch time (the start-phase host idle of the profile).  Shipped default:
# 1 = the trimmed host path; 0 = the previous host path, selectable through _kcp_test_knobs for the A/B record.  Host side only:
# the same kernels with the same arguments in the same order on the same stream; caches hold host ints keyed on the context /
# tables / options objects (never a tensor's contents); bitwise by construction (module docstring, "Shipped host path").
HOST_TRIMS = (0, 1)
HOST_TRIM_DEFAULT = (
    1  # shipped: the trimmed host path; 0 = the previous path, via _kcp_test_knobs only
)
# Fused carry merges (test knob): "carry" launches the two forward carry
# families carry_start + carry_in of a rank as ONE K4 launch (chain family "carry_both") and the two adjoint families
# dcarry_end + dcarry_cross as one ("dcarry_both") -- the same m / c / y pools, the start (end) chains keep x0_index = -1
# (zero state) beside the S_in (dS_out) tiles of the crossing chains; one launch saved per phase on ranks holding both a
# document-start (document-last) and a crossing-in (crossing-out) fragment at P > 1.  Chain order inside a launch is the
# CTA issue order only (every CTA walks its own chain / column block): bitwise.  Applies under schedule wait_first (the
# overlap schedule issues carry_start / dcarry_end under the collective) and in the saved-mode backward's re-walk merges (saved_carry=0).
MERGE_FUSES = ("none", "carry")
MERGE_FUSE_DEFAULT = "none"  # shipped: the separate launches; "carry" via _kcp_test_knobs (for the A/B record)
# Saved carry plane: 1 (shipped) keeps the forward's whole carry plane (the "carry" pool's worth of a P > 1 rank:
# n_carry x H tiles -- the carries, the S_in copies of the fused walks, the never-written slots) as a fresh tensor in the
# saved set, and the saved-mode backward replays its tape from that plane instead of re-walking every segment with K1
# ``_offset`` and re-merging carry_start + carry_in (those launches recompute the plane the forward produced from the same
# inputs: bitwise).  The adjoint carries still go to the workspace pool (the saved plane is never written); replay mode
# (checkpoint recompute) and P = 1 ranks are unchanged.  0 = the re-walk (the previous form), via _kcp_test_knobs only
# (the A/B record).
SAVED_CARRIES = (0, 1)
SAVED_CARRY_DEFAULT = 1
# Walk form of the K1 / K2 state walks: "auto" (shipped) = the registry's rule per launch (cake_kcp_jit.walk_form: the
# offset programs take their two-CTA ``_vh64`` record when the launch list-schedules within its longest chain, else the
# full-width record; the tape programs take their two-CTA record always, their full-width records being test-only);
# "full" / "split" force one form on every launch, _kcp_test_knobs only (the A/B record).  The forms are bitwise equal:
# the form never changes a bit.  A split tape launch is never capped at one CTA per SM (the cap serves the full-width
# tape CTAs).
WALK_FORMS = ("auto", "full", "split")
WALK_FORM_DEFAULT = "auto"
P2_KNOBS = {  # knob -> admitted values
    "schedule": SCHEDULES,
    "fused_walks": (0, 1),
    "cols_rule": COLS_RULES,
    "walk_geometry": WALK_GEOMETRIES,
    "walk_form": WALK_FORMS,
    "graph": GRAPH_MODES,
    "graph_cap": GRAPH_CAPS,
    "host_trim": HOST_TRIMS,
    "merge_fuse": MERGE_FUSES,
    "saved_carry": SAVED_CARRIES,
}
# Shipped defaults of the schedule / fused-walk pair (adopted together, the keep
# rule): the "wait_first" schedule with the fused tape walks (one launch per phase).  "overlap" / fused_walks=0 stay
# selectable through the test knobs for the A/B record; cols_rule stays "fixed" (the other column rule was in band everywhere: not adopted,
# knob kept).  The schedule and the walk grouping never change a published bit (the merges read c1_recv / the staging
# planes, the tapes read no collective buffer): G3 holds under every combination.
SCHEDULE_DEFAULT = "wait_first"
FUSED_WALKS_DEFAULT = 1

# Program names of the dispatch table (the K1-K4 programs of the design note §14) and the local chunk stages: the
# existing backend's stages at the per-rank shape (launched with the records its plan selects on
# :data:`LOCAL_PATH_FWD` / :data:`LOCAL_PATH_BWD`) and the two KCP-owned records of :data:`KCP_LOCAL_RECORDS`.
KCP_PROGRAMS = (
    "kcp_fwdh_offset",  # K1 ``_offset``: walk from zero, final state -> compact tiles (B_f / b_seg)
    "kcp_fwdh_tape",  # K1 ``_tape``: walk from an indexed initial state, h / v_new tapes
    "kcp_dhu_offset",  # K2 ``_offset``: reverse walk from zero, final adjoint -> compact tiles (d_f / d_seg)
    "kcp_dhu_tape",  # K2 ``_tape``: reverse walk from an indexed final adjoint, dh tape
    "kcp_transfer_walk",  # K3: FP32-class transfer M_f (^T) (or the FP32-class offset with init_mode = 1)
    "kcp_affine_merge",  # K4: exact FP32 affine chain (state / map / transposed modes)
)
LOCAL_STAGES_FWD = ("fwd_intra", "wy_fwd", "fwd_o")
LOCAL_STAGES_BWD = (
    "bwd_prep",
    "fwd_intra",
    "wy",
    "dav",
    "dqkg",
    "glm_intra",
    "gate_epilogue",  # no launch where the plan's GLM intra program fuses it (glm_intra_v14 on sm_103a)
    "param_finalize",  # raw gate mode only: dA_log / ddt_bias from the GLM intra program's per-chunk partials
)
# The paths of the existing backend's plan the local stages take their records from (the forward without a saved
# set, the checkpoint replay for both backward modes: :mod:`.cake_kcp_plan`) and the arena layouts they run on.
LOCAL_PATH_FWD = cake_kcp_plan.LOCAL_PATH_FWD
LOCAL_PATH_BWD = cake_kcp_plan.LOCAL_PATH_BWD
LOCAL_LAYOUT_FWD = "fwd"
LOCAL_LAYOUT_BWD = "replay"
# The lean backward layout of a local arena sized for it (``WorkspaceDetKcp(local_arena="lean")``, opt-in): this
# route's mode of the existing backend's arenas (registered here); the arena's ``layout_passes`` holds its tail passes
# (absent on an arena or plan view that runs the replay layout)
LOCAL_LAYOUT_BWD_LEAN = cake_kcp_plan.LOCAL_LAYOUT_BWD_LEAN
cake_kcp_plan.register_lean_layout()
# the stages of the backward's tail, once per tail range on the lean layout; the whole-sequence buffers they read by
# chunk-relative rows (leading-axis views offset by the range's first chunk; the h / dh tapes likewise)
TAIL_STAGES = ("dqkg", "glm_intra", "gate_epilogue")
TAIL_ROW_BUFFERS = ("gk", "v_new", "dv2", "beta_int", "do_int", "akk_int")
TAIL_COUNTS_ATTR = "_kcp_tail_counts"  # a lean arena's int32 table 0..L-1 (cake_kcp_plan.local_tail_count_words)
# a forward-only local arena's layout (:func:`cake_kcp_plan.local_fwd_layout`): the forward runs on the layout its
# arena was sized for (``KdaWorkspace.fwd_layout``), on a backward arena :data:`LOCAL_LAYOUT_FWD`
LOCAL_LAYOUT_FWD_ONLY = cake_kcp_plan.LOCAL_LAYOUT_FWD_ONLY
# The KCP-owned local records (:mod:`.cake_kcp_jit`), by the stage name the launch sequences use: the output program
# over the K1 ``h`` / ``v_new`` tapes and the ``dA_qk`` / ``dv1`` program whose ``dv1`` K2 reads (the forms the
# existing backend's output and dA / dv programs had before its fused recurrences took them over).
KCP_LOCAL_RECORDS = {"fwd_o": "kcp_fwd_o", "dav": "kcp_dav"}
# The WY records the KCP launches directly where the plan fuses the WY representation into a state recurrence the
# KCP does not run (its recurrences are the K1 / K2 walks): the forward's program and the replay's.
WY_DIRECT_RECORDS = {"wy_fwd": "wy_fwd", "wy": "wy_bwd_replay"}
SCRATCH_TABLES_ATTR = "_kcp_scratch_tables"  # the arena attribute holding the table set the intra record writes
COLLECTIVES = ("c1", "c2", "b1", "b2")


# ----------------------------------------------------------------------------------------------------------------
# Small helpers (metadata only)


def tile_index(h: int, slot: int, hg: int, n: int) -> int:
    """The kernels' head-group-major compact tile address (the design note §7.3): tile ``((h // hg) * n + slot) * hg
    + h % hg`` of a pool whose consecutive head groups each hold ``n`` slots of ``hg`` heads."""
    return ((h // hg) * n + slot) * hg + h % hg


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _alloc(
    shape: Sequence[int], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Every per-call tensor (outputs, saved set, collective outputs) comes from here: ``torch.empty`` for buffers
    every element of which a kernel or a collective of the same call writes."""
    return torch.empty(tuple(shape), dtype=dtype, device=device)


def _zeros(
    shape: Sequence[int], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.zeros(tuple(shape), dtype=dtype, device=device)


def _squeeze_batch(x: torch.Tensor, name: str, rank: int) -> torch.Tensor:
    if x.dim() == rank + 1:
        if x.shape[0] != 1:
            raise ValueError(
                f"{name}: the batch axis must be 1, got shape {tuple(x.shape)}"
            )
        x = x.squeeze(0)
    if x.dim() != rank:
        raise ValueError(
            f"{name}: expected {rank}-D (or a leading batch axis of 1), got shape {tuple(x.shape)}"
        )
    return x


def _check(
    x: torch.Tensor,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tuple(x.shape) != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {tuple(x.shape)}")
    if x.dtype != dtype:
        raise ValueError(f"{name}: expected {dtype}, got {x.dtype}")
    if x.numel() and not x.is_contiguous():
        raise ValueError(
            f"{name}: must be contiguous (strides {x.stride()}); the kernels take no strided operands"
        )
    if x.device != device:
        raise ValueError(f"{name}: on {x.device}, expected {device}")


def _operands(
    q, k, v, g, beta, ctx, *, dtypes: dict[str, torch.dtype], gate: str = "g"
):
    """The operands without their optional batch axis, ``(T_r, H)`` and whether the inputs carried a batch axis.
    ``T_r`` must equal the context's local token count; ``H`` is the operands' head count (``1..64``).  ``gate="raw"``
    (``use_gate_in_kernel=True``): ``g`` is the raw gate, BF16 ``[T_r, H, 128]``, any view the single-rank rule
    accepts (``cake_backend._check_raw``: unit innermost stride, 16-byte outer strides and address, no overlap; the
    only full check, as the strided-pointer bindings check no layout; never copied)."""
    v = _squeeze_batch(v, "v", 3)
    T, H = int(v.shape[0]), int(v.shape[1])
    device = torch.device(ctx.device)
    batched = q.dim() == 4
    q, k, g = (_squeeze_batch(x, n, 3) for x, n in ((q, "q"), (k, "k"), (g, "g")))
    beta = _squeeze_batch(beta, "beta", 2)
    if int(ctx.T_r) != T:
        raise ValueError(
            f"operands carry {T} tokens, the ContextDetKcp of rank {ctx.rank} holds T_r={ctx.T_r}"
        )
    if not 1 <= H <= 64:
        raise ValueError(f"H={H}: the Cake KDA kernels take 1..64 heads")
    _check(q, "q", (T, H, D), dtypes["q"], device)
    _check(k, "k", (T, H, D), dtypes["k"], device)
    _check(v, "v", (T, H, D), dtypes["v"], device)
    if gate == "raw":
        g = cake_backend._check_raw(g, T, H, device)
    else:
        _check(g, "g", (T, H, D), dtypes["g"], device)
    _check(beta, "beta", (T, H), dtypes["beta"], device)
    return q, k, v, g, beta, T, H, batched


def _ranked(x: torch.Tensor, batched: bool) -> torch.Tensor:
    return x.unsqueeze(0) if batched else x


# ----------------------------------------------------------------------------------------------------------------
# Backends: the dispatch table of the KCP programs and the local chunk stages


# The contract's storage dtypes.  A backend may override them (``Backend.dtypes``): the FP64 emulation keeps every
# operand, state and output in FP64 so the orchestration can be checked against the global FP64 reference to ~1e-12.
CONTRACT_DTYPES: dict[str, torch.dtype] = {
    "q": torch.bfloat16,
    "k": torch.bfloat16,
    "v": torch.bfloat16,
    "g": torch.float32,
    "beta": torch.bfloat16,
    "out": torch.bfloat16,
    "dout": torch.bfloat16,
    "dq": torch.bfloat16,
    "dk": torch.bfloat16,
    "dv": torch.bfloat16,
    "dbeta": torch.bfloat16,
    "dg": torch.float32,
    "state": torch.float32,  # the FP32 states, summaries and adjoints of every pool and collective
    # raw gate mode (use_gate_in_kernel=True): the raw gate and its gradient, the gate parameters and theirs
    "g_raw": torch.bfloat16,
    "dg_raw": torch.bfloat16,
    "A_log": torch.float32,
    "dt_bias": torch.float32,
    "dA_log": torch.float32,
    "ddt_bias": torch.float32,
}


class Backend:
    """A dispatch table for the KCP programs and the local chunk stages.

    ``run(name, args)`` launches program ``name`` (one of :data:`KCP_PROGRAMS`) with the logical arguments ``args``
    (the ABI vocabulary of the design note §14: pools, index tables, ``hg`` / ``n`` addressing scalars, modes; plus
    ``_call``, the :class:`CallDetKcp` the launch belongs to).  ``stage(name, call, **extra)`` launches one local chunk
    stage of :mod:`..cake_kda_det_train.cake_backend` at the per-rank shape.  ``dtypes`` are the storage dtypes of the operands, outputs
    and states; ``local_stages`` says whether the local stages exist (the FP64 emulation folds them into its tape
    programs).  The shipped backend is :class:`KernelBackend`; tests install another with :func:`set_backend`.
    """

    name = "abstract"
    dtypes: dict[str, torch.dtype] = dict(CONTRACT_DTYPES)
    local_stages = True
    path = "fi-rendered"  # recorded with every result: fake (emulation), direct (bring-up), fi-rendered (the package)

    def run(
        self, name: str, args: dict[str, Any]
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def stage(
        self, name: str, call: CallDetKcp, **extra: Any
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def begin_call(self, call: CallDetKcp) -> None:
        """Hook at the start of a forward or backward call (the kernel backend builds the local stage values, the
        emulation resets its per-call scratch)."""

    def end_call(self, call: CallDetKcp) -> None:
        """Hook at the end of a call."""

    @staticmethod
    def transpose_tiles(
        *,
        src: torch.Tensor,
        src_tiles: torch.Tensor,
        dst: torch.Tensor,
        dst_tiles: torch.Tensor,
    ) -> None:
        """``dst`` tiles ``dst_tiles[i]`` <- the transposes of the ``src`` tiles ``src_tiles[i]`` (torch; bitwise;
        shared by every backend: the ``M_f`` -> ``M_f^T`` step of the segmented summaries)."""
        dst.view(-1, D, D)[dst_tiles] = src.view(-1, D, D)[src_tiles].transpose(1, 2)


_SPEC3_ROWS = (
    "gk",
    "kg",
    "qg",
    "w",
    "u",
)  # the prep rows SPEC-3 compares (the design note §6.4)
# the entry points' spec3_probe keyword: "off" (the default) skips the probe; "record" records it; "warn" warns on a
# difference; "assert" raises KcpPreconditionError
SPEC3_PROBES = ("off", "record", "warn", "assert")
SPEC3_PROBE_DEFAULT = "off"
_PROGRAM_OF = {
    "kcp_fwdh_offset": "fwdh_offset",
    "kcp_fwdh_tape": "fwdh_tape",
    "kcp_dhu_offset": "dhu_offset",
    "kcp_dhu_tape": "dhu_tape",
    "kcp_transfer_walk": "transfer_walk",
    "kcp_affine_merge": "affine_merge",
}


def _walk_form_of(call: Any, programs: Any, name: str, args: dict[str, Any]) -> str:
    """The record form of walk launch ``name`` (``"full"`` / ``"split"``): the registry's rule
    (``cake_kcp_jit.walk_form``) on the launch's chains (the chunk counts of its listed segments), the call's head
    count and the device's SM count, under the call's ``walk_form`` knob -- decided once per ``(program, walk list)``
    of the call's table set and looked up on every later launch (:meth:`_Tables.walk_form`); ``"full"`` with a
    registry that has no form rule (one record per walk program)."""
    jit = getattr(programs, "jit", None)
    rule = getattr(jit, "walk_form", None)
    if rule is None or name not in getattr(jit, "WALK_FORM_RULES", {}):
        return "full"
    return call.tb.walk_form(
        name,
        args["walk"],
        int(call.H),
        int(jit.sm_count(call.device.index)),
        str(call.walk_form),
        rule,
    )


class KcpPreconditionError(RuntimeError):
    """The SPEC-3 probe found the backward's ``gk`` / ``kg`` / ``qg`` / ``w`` / ``u`` rows different from the forward's
    on a real token row and the call's ``spec3_probe="assert"`` asked for an error (the design note §6.4).  The requirement of
    record is that the two backward modes (saved set, checkpoint replay) produce identical gradients: both run the
    same prep, intra and WY programs on the same operands, which the GPU tests check bitwise.  The forward's and the
    backward's ``gk`` buffers differ by construction on the current program set: the forward's intra program
    (``write_prep = 1``) writes the chunk-final gate row alone (row 63 of every chunk: the chunk's total gate in log2
    units; 0 on pad chunks) and the backward's prep program writes every token row's prefix, whose row 63 is the same
    total; the K1 / K2 / K3 walks read row 63 only.  ``kg`` / ``qg`` / ``w`` / ``u`` agree bitwise."""


@dataclasses.dataclass(frozen=True)
class _Block:
    """One compact indexed state block as a walk kernel addresses it: base pointer, ``index[NSEG]`` (i32 device),
    ``hg``, ``n`` and the ``tile_stride`` of the pool (1 dense plane, 2 interleaved pair)."""

    ptr: torch.Tensor
    index: torch.Tensor
    hg: int
    n: int
    stride: int


TEMPLATE_CACHE_ATTR = "_kcp_local_templates"  # the arena attribute holding the trimmed path's begin_call templates
# {(id(table set), backward): (weakref.ref(table set), template)}; the views belong to that arena, so the cache dies with it


class KernelBackend(Backend):
    """The shipped backend: the KCP programs K1-K4 through a program provider (``launch(program, args, *, grid,
    device_index, precision_mode)`` with the positional ABI of the design note §14) and the local
    chunk stages through :class:`cake_backend.KdaWorkspace` of the per-rank problem.  The default provider is H3's
    registry (:mod:`.cake_kcp_jit`, imported at the first launch so this module imports without it); a bring-up
    provider launching the source builds directly can be passed (``path="direct"``; never a gated row).

    Local stages (:data:`LOCAL_STAGES_FWD` / :data:`LOCAL_STAGES_BWD`): a forward call runs the existing backend's
    intra program (``write_prep = 1``: the GLM prep's rows and the internal ``Aqk`` / ``Akk`` rows; the token-row
    copies are never written, ``write_ext = 0``), its forward WY program and the KCP's output program ``kcp_fwd_o``
    (the output from the K1 ``h`` / ``v_new`` tapes) on the arena's forward layout (``fwd``; a forward-only arena's
    ``fwd_kcp``, :data:`LOCAL_LAYOUT_FWD_ONLY`), with the records the plan selects
    on the forward path without a saved set (:data:`LOCAL_PATH_FWD`).  A backward call runs on the ``replay`` layout
    (on a lean arena the lean layout :data:`LOCAL_LAYOUT_BWD_LEAN`: ``kcp_dav``'s ``dA_qk`` to its sink, the tail
    stages once per chunk range, :func:`_tail_pass_values`)
    with the records of the replay path (:data:`LOCAL_PATH_BWD`) in both modes: the replay prep (the GLM prep's rows
    and the internal ``dout`` copy ``do_int``), the intra program (``write_prep = 0``: the internal ``Aqk`` / ``Akk``
    rows), the replay's WY program, then after the K1 tape replay the KCP's ``kcp_dav`` (``dA_qk`` and the intra-chunk
    ``dv1`` K2 reads) and after K2 the fused gradient, GLM intra and the gate epilogue.  Where the plan fuses the WY
    representation into a state recurrence the KCP does not run, the WY program is launched directly
    (:data:`WY_DIRECT_RECORDS`) on its grid rule.  The intra record that computes the chunk layout from ``cu_seqlens``
    writes its tables into a scratch set on the arena (:data:`SCRATCH_TABLES_ATTR`); every other program reads the
    context's CPU-built tables.  The saved set carries no token-row ``Aqk`` / ``Akk`` (:class:`SavedDetKcp`): the
    backward recomputes the internal rows in both modes, so the two modes run the same programs on the same operands.
    The K1 / K2 / K3 walks read the layout's ``w`` / ``kg`` / ``u`` / ``gk`` (``qg`` / ``do_int`` / ``dv1``) rows and
    write its ``h`` / ``v_new`` (``dh`` / ``dv2``) tapes.

    SPEC-3 (debug probe, once per backend on the first saved-mode backward whose call's ``spec3_probe`` keyword is
    not ``"off"``): before that backward's prep the
    forward's intra program and WY program run on the ``fwd`` layout and their five rows (``gk`` / ``kg`` / ``qg`` /
    ``w`` / ``u``) are kept; after the backward's WY program the ``replay`` layout's rows are compared with them on
    every real token row (pad rows are reported separately).  The requirement of record is that the two backward
    modes produce identical gradients (the GPU tests check saved mode against the replay bitwise).  The probe's
    ``equal`` is ``False`` by construction: the forward's intra program writes ``gk`` at row 63 of every chunk alone
    (the chunk-final gate row, the only ``gk`` row the K1 / K2 / K3 walks read), the backward's prep program writes
    every row's prefix with the same row 63, so ``rows["gk"]["real_rows_differ"]`` is the real rows minus the full
    chunks while ``kg`` / ``qg`` / ``w`` / ``u`` report 0.  The kept rows (``gk`` FP32, the other four BF16: about
    ``R * H * 1536`` bytes) stay allocated through the backward's prep and WY programs and raise that backward's
    peak by up to their size, so the probe is off by default (``spec3_probe="off"``): ``"record"`` runs it and
    keeps the comparison, ``"warn"`` warns on a difference, ``"assert"`` raises :class:`KcpPreconditionError`; the
    result is in ``LAST_CALL["bwd"]["spec3"]`` and :attr:`spec3`.  The keyword travels with the call's options
    (:class:`OptionsDetKcp`); no environment variable is read.  Under ``graph="replay"`` the probing backward runs
    eagerly; a rank without tokens has nothing to probe and does not hold back the capture of its backward.
    """

    name = "kernels"

    def __init__(
        self,
        programs: Any | None = None,
        *,
        cols_per_cta: int = 8,
        path: str = "fi-rendered",
    ) -> None:
        self._programs = programs
        if cols_per_cta not in (8, 16, 32) or D % cols_per_cta:
            raise ValueError(f"cols_per_cta must be 8, 16 or 32, got {cols_per_cta}")
        self.cols_per_cta = int(cols_per_cta)
        self.path = path
        self._dev: dict[tuple[Any, ...], torch.Tensor] = {}
        self._checked_chains: set[int] = set()
        self.spec3: dict[str, Any] | None = None
        # until the first saved-mode backward that runs the probe
        self._spec3_pending = True

    # -- providers
    def programs(self) -> Any:
        if self._programs is None:
            from . import cake_kcp_jit  # noqa: PLC0415 - H3's registry; optional at import time

            self._programs = _FiPrograms(cake_kcp_jit)
        return self._programs

    # -- device tables (small int tables, cached by content)
    def _dev_table(
        self,
        kind: str | tuple[Any, ...],
        values: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (kind, str(device), dtype, tuple(int(v) for v in values))
        t = self._dev.get(key)
        if t is None:
            t = torch.tensor(list(key[3]), dtype=dtype, device=device)
            self._dev[key] = t
        return t

    # -- local stage values
    def begin_call(self, call: CallDetKcp) -> None:
        if call.local is None:
            return
        ws, ctx, p = call.local, call.ctx, call.local.plan
        if (call.T_r, int(ctx.n_frag), call.H) != (p.S, p.N, p.H):
            raise ValueError(
                f"local arena planned for (S, N, H) = {(p.S, p.N, p.H)}, the call is {(call.T_r, ctx.n_frag, call.H)}"
            )
        if (p.NC_pad, p.R) != (int(ctx.NC_pad), int(ctx.R)):
            raise ValueError(
                f"local plan geometry (NC_pad, R) = {(p.NC_pad, p.R)} differs from the KCP context's {(ctx.NC_pad, ctx.R)}"
            )
        views = ws.views(_bwd_layout(ws) if call.backward else ws.fwd_layout)
        tail = _lean_tail(ws) if call.backward else None
        if call.host_trim:  # the call-independent part of the value dict is built once per (arena, tables, direction)
            # the template is cached on the ARENA whose views it holds (it dies with the arena), keyed by the
            # table set's identity with a WEAK reference to it (a hit needs that very table set alive); the backend
            # keeps no reference to an arena or a table set; dead entries are purged on the next miss.
            cache = ws.__dict__.get(TEMPLATE_CACHE_ATTR)
            if cache is None:
                cache = {}
                setattr(ws, TEMPLATE_CACHE_ATTR, cache)
            key = (id(call.tb), call.backward)
            hit = cache.get(key)
            if hit is None or hit[0]() is not call.tb:
                for stale in [k for k, h in cache.items() if h[0]() is None]:
                    del cache[stale]
                hit = (
                    weakref.ref(call.tb),
                    self._local_template(views, ctx, p, call.backward, tail),
                )
                cache[key] = hit
            values: dict[str, Any] = dict(hit[1])
            values.update(call.ops)
            values["scale"] = call.scale
            call.local_values = values
            call.scratch["views"] = views
            return
        values = self._local_template(views, ctx, p, call.backward, tail)
        values.update(call.ops)
        values["scale"] = call.scale
        call.local_values = values
        call.scratch["views"] = views

    @staticmethod
    def _local_template(
        views: Any,
        ctx: Any,
        p: Any,
        backward: bool,
        tail: tuple[int, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """The values of ``begin_call`` that do not depend on the call: the arena views of the layout, the context's
        four chunk tables (the existing backend's vocabulary; the KCP's segment table is a K1-K4 argument, not a
        local one), the plan scalars of the existing backend's logical vocabulary and the ``h2d`` / ``dh2d`` views;
        on the lean layout (``tail`` = its passes and the arena's tail-count table) the per-range values of the
        tail stages (:func:`_tail_pass_values`) under ``"_kcp_tail"``.
        The operands and ``scale`` are added per call (the trimmed host path caches this part per (arena, table set,
        direction))."""
        values: dict[str, Any] = dict(vars(views))
        for name in cake_backend.TABLE_NAMES[:4]:
            values[name] = ctx.tables[name]
        values.update(
            num_seqs=p.N,
            num_chunks_pad=p.NC_pad,
            num_heads=p.H,
            group=1,
            num_segments=p.NSEG,
            segments=p.P,
            seg_arg=p.seg_arg,
            proportional=p.proportional,
            num_pairs=p.R // cake_jit.PAIR_ROWS,
        )
        values["h2d"] = views.h.view(-1, D)
        if backward:
            values["dh2d"] = views.dh.view(-1, D)
            if tail is not None:
                values["_kcp_tail"] = _tail_pass_values(views, ctx, p, *tail)
        return values

    def stage(self, name: str, call: CallDetKcp, **extra: Any) -> None:
        """One local stage of :data:`LOCAL_STAGES_FWD` / :data:`LOCAL_STAGES_BWD`: the plan's record on the call's path
        (or the direct WY record), the KCP-owned record of ``fwd_o`` / ``dav``; nothing for an optional stage the plan
        does not launch on the path (``gate_epilogue`` where ``glm_intra_v14`` writes the gate gradients itself)."""
        values, views = call.local_values, call.scratch["views"]
        local = call.local
        p = local.plan
        path = LOCAL_PATH_BWD if call.backward else LOCAL_PATH_FWD
        if call.out is not None:
            values["out"] = call.out
        if call.grads is not None:
            values.update(call.grads)
        tail_pass = extra.pop("tail_pass", None)
        # one tail range of the lean layout: its values and grids
        if tail_pass is not None:
            overrides, grids = values["_kcp_tail"][tail_pass]
            # nothing where the plan fuses the stage away (the gate epilogue)
            if name in grids:
                self._launch(
                    call,
                    p.record(name, path),
                    grids[name],
                    {**values, **overrides, **extra},
                )
            return
        if name == "bwd_prep":
            # both backward modes run the replay prep (the GLM prep's rows, the internal dout copy K2 and dav read);
            # the saved set carries no token rows: the intra program recomputes the internal rows in both modes
            if (
                extra.pop("saved", None) is not None
                and self._spec3_pending
                and call.opts.spec3_probe != "off"
            ):
                self._spec3_forward_rows(
                    call
                )  # before the backward's prep: the layouts share the arena
        elif name == "fwd_intra":
            # the token-row Aqk / Akk are never written on this route (the saved set carries none); any BF16 tensor
            # serves as the unused pointer (as cake_backend does)
            extra.update(write_ext=0, aqk_ext=views.akk_int, akk_ext=views.akk_int)
            # the internal rows every KCP consumer reads; the GLM prep's rows on the forward, the backward's prep
            # program wrote them already
            extra.setdefault("write_int", 1)
            extra.setdefault("write_prep", 0 if call.backward else 1)
            if cake_kcp_plan.intra_writes_tables(p, path):
                extra.update(
                    cu_seqlens=call.ctx.cu_seqlens_local, **_scratch_tables(local)
                )
        elif name in KCP_LOCAL_RECORDS:
            if name == "dav" and hasattr(views, "dAqk_sink"):
                # the lean layout: the fused gradient stores every dAqk row the tail reads, kcp_dav's go to the sink
                extra["dAqk"] = views.dAqk_sink
            self.programs().run_record(
                KCP_LOCAL_RECORDS[name],
                {**values, **extra},
                grid=kcp_local_grid(name, p),
                device_index=local.device_index,
            )
            return
        if (
            name in cake_jit.OPTIONAL_STAGES
            and name not in WY_DIRECT_RECORDS
            and not p.dispatched(name, path)
        ):
            return  # the plan fuses the stage away on this device (the gate epilogue inside glm_intra_v14): no launch
        record, grid = _local_record(p, name, path)
        self._launch(call, record, grid, {**values, **extra})
        if name == "wy" and call.scratch.get("spec3_ref") is not None:
            self._spec3_compare(call)

    @staticmethod
    def _launch(
        call: CallDetKcp,
        record: str,
        grid: tuple[int, int, int],
        values: dict[str, Any],
    ) -> None:
        """A record of the existing backend's registry on the local arena: the trimmed launch, or the arena's
        launcher (a diagnostics launcher keeps the base path)."""
        local = call.local
        if call.host_trim and local.launcher is cake_jit.run_module:
            _launch_record_trim(local, record, grid, values)
        else:
            local.launcher(record, values, grid=grid, device_index=local.device_index)

    # -- SPEC-3
    def _spec3_forward_rows(self, call: CallDetKcp) -> None:
        """The probe's reference: the forward's intra program (``write_prep = 1``) and WY program on the ``fwd``
        layout, their five rows kept (the backward's layout shares the arena, so this runs before its prep)."""
        self._spec3_pending = False
        local = call.local
        p = local.plan
        a = local.views(getattr(local, "fwd_layout", LOCAL_LAYOUT_FWD))
        vals = {**call.local_values, **vars(a)}
        vals.update(
            h2d=a.h.view(-1, D),
            aqk_ext=a.akk_int,
            akk_ext=a.akk_int,
            write_ext=0,
            write_int=1,
            write_prep=1,
        )
        if cake_kcp_plan.intra_writes_tables(p, LOCAL_PATH_FWD):
            vals.update(cu_seqlens=call.ctx.cu_seqlens_local, **_scratch_tables(local))
        for stage in ("fwd_intra", "wy_fwd"):
            record, grid = _local_record(p, stage, LOCAL_PATH_FWD)
            self._launch(call, record, grid, vals)
        call.scratch["spec3_ref"] = {n: getattr(a, n).clone() for n in _SPEC3_ROWS}

    def _spec3_compare(self, call: CallDetKcp) -> None:
        ref = call.scratch.pop("spec3_ref")
        b, ctx = call.scratch["views"], call.ctx
        R = int(ctx.R)
        rows = torch.arange(R, device=call.device)
        lens = torch.tensor(list(ctx.chunk_len), dtype=torch.int64, device=call.device)
        real = (rows % BT) < lens[rows // BT]
        result: dict[str, Any] = {
            "checked": True,
            "T_r": call.T_r,
            "H": call.H,
            "R": R,
            "rows": {},
            "equal": True,
        }
        for n in _SPEC3_ROWS:
            x, y = getattr(b, n), ref[n]
            it = torch.int32 if x.dtype == torch.float32 else torch.int16
            differ = (x.view(it) != y.view(it)).view(R, -1).any(1)
            n_all, n_real = int(differ.sum()), int((differ & real).sum())
            result["rows"][n] = {"rows_differ": n_all, "real_rows_differ": n_real}
            if n_real:
                result["equal"] = False
        self.spec3 = call.scratch["spec3"] = result
        if not result["equal"]:
            mode = call.opts.spec3_probe
            message = (
                "SPEC-3 probe: the backward's prep + WY rows differ from the forward's intra + WY rows on real "
                f"token rows: {result['rows']}"
            )
            if mode == "assert":
                raise KcpPreconditionError(message)
            if mode == "warn":
                warnings.warn(message, RuntimeWarning, stacklevel=2)

    # -- KCP programs (positional ABI of the design note §14)
    def abi(
        self, name: str, args: dict[str, Any]
    ) -> tuple[str, list[Any], int, dict[str, Any]]:
        """``(program, positional ABI arguments, grid x, variant)`` of one abstract launch ``args`` of program
        ``name`` (the integration contract with the registry: ``program`` in ``fwdh_offset`` / ``fwdh_tape`` /
        ``dhu_offset`` / ``dhu_tape`` / ``transfer_walk`` / ``affine_merge``; ``variant`` = the K3 test knobs)."""
        call: CallDetKcp = args["_call"]
        prog = _PROGRAM_OF[name]
        if prog in ("fwdh_offset", "fwdh_tape"):
            abi, grid = self._fwdh_args(call, args, tape=prog == "fwdh_tape")
        elif prog in ("dhu_offset", "dhu_tape"):
            abi, grid = self._dhu_args(call, args, tape=prog == "dhu_tape")
        elif prog == "transfer_walk":
            abi, grid = self._transfer_args(call, args)
        else:
            abi, grid = self._merge_args(call, args)
        variant = (
            {"precision_mode": call.precision_mode, "transfer_walk": call.transfer_walk}
            if prog == "transfer_walk"
            else {}
        )
        if prog == "affine_merge" and args["cols_per_cta"] is not None:
            variant = {
                "cols_per_cta": int(args["cols_per_cta"])
            }  # D1: an explicit record instead of the registry's choice
        if (
            prog in _WALK_K12
        ):  # the record form of this walk launch (the registry's rule or the knob)
            variant = {"walk_form": _walk_form_of(call, self.programs(), name, args)}
            args["walk_form"] = variant["walk_form"]
        return prog, abi, int(grid), variant

    def run(self, name: str, args: dict[str, Any]) -> None:
        call: CallDetKcp = args["_call"]
        if call.host_trim and self._run_trim(name, args, call):
            return
        prog, abi, grid, variant = self.abi(name, args)
        self.programs().launch(
            prog,
            abi,
            grid=grid,
            device_index=call.device.index,
            name=name,
            args=args,
            **variant,
        )

    def _run_trim(self, name: str, args: dict[str, Any], call: CallDetKcp) -> bool:
        """Trimmed host path: the launch without :meth:`abi` when the provider launches from the abstract ``args`` anyway (the
        registry's ``run_program`` form discards the positional ABI, whose walk / index / chain device tables cost 100-300 us of
        host time per launch).  Keeps the D1 bookkeeping of the merge (alignment check, ``cols_per_cta``, ``merge_log``).
        Returns False when the base path is needed (another provider, an explicit ``_c<cols>`` merge record)."""
        programs = self.programs()
        jit = getattr(programs, "jit", None)
        if (
            not isinstance(programs, _FiPrograms)
            or jit is None
            or hasattr(jit, "launch_kcp")
            or not hasattr(jit, "run_program")
        ):
            return False
        prog = _PROGRAM_OF[name]
        variant: dict[str, Any] = {}
        if prog == "affine_merge":
            ch: Chains = args["chains"]
            cols = merge_cols_per_cta(ch.ncols, ch.n_chains, call.cols_rule)
            if cols is not None:
                return False  # the explicit record form needs the positional ABI (the base path appends the merge_log entry)
            if id(ch) not in self._checked_chains:
                ch.check_alignment()
                self._checked_chains.add(id(ch))
            args["cols_per_cta"] = None
            call.merge_log.append(
                (str(args.get("family")), int(ch.ncols), int(ch.n_chains), None)
            )
        elif prog == "transfer_walk":
            variant = {
                "precision_mode": call.precision_mode,
                "transfer_walk": call.transfer_walk,
            }
        elif (
            prog in _WALK_K12
        ):  # the record form of this walk launch (the registry's rule or the knob)
            variant = {"walk_form": _walk_form_of(call, programs, name, args)}
            args["walk_form"] = variant["walk_form"]
        programs.launch(
            prog,
            [],
            grid=0,
            device_index=call.device.index,
            name=name,
            args=args,
            **variant,
        )
        return True

    def _walk_common(
        self, call: CallDetKcp, args: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, int, Any]:
        walk = args["walk"]
        walk_seg = self._dev_table("walk", walk, call.device, torch.int32)
        return (
            walk_seg,
            call.tb.hp.device_tables["seg_chunk_start"],
            int(call.tb.NSEG),
            call.scratch["views"],
        )

    def _block(self, call: CallDetKcp, ix: Indexed | None) -> _Block:
        """The kernel's view of an :class:`Indexed` pool; ``None`` (or no slot) = the zero state: every index ``-1``
        and a never-read base pointer."""
        nseg = int(call.tb.NSEG)
        if ix is None or ix.n_slots == 0:
            return _Block(
                call.buf("eye"),
                self._dev_table("neg", [-1] * nseg, call.device, torch.int32),
                call.H,
                1,
                1,
            )
        base_tile, hg, n, stride = ix.addressing(call.H)
        if len(ix.index) != nseg:
            raise ValueError(
                f"Indexed: {len(ix.index)} segment entries, the plan has NSEG = {nseg}"
            )
        ptr = ix.pool.view(-1)[base_tile * TILE :] if base_tile else ix.pool.view(-1)
        return _Block(
            ptr,
            self._dev_table("index", ix.index, call.device, torch.int32),
            int(hg),
            int(n),
            int(stride),
        )

    def _fwdh_args(
        self, call: CallDetKcp, args: dict[str, Any], *, tape: bool
    ) -> tuple[list[Any], int]:
        walk_seg, seg_table, nseg, v = self._walk_common(call, args)
        if tape:  # reads h0, writes the tapes; the out block is unused
            h0 = self._block(call, args.get("h0"))
            out = h0
        else:  # walks from zero, writes the indexed final states; the h0 block is unused
            out = self._block(call, args.get("out"))
            h0 = out
        abi = [
            v.w,
            v.kg,
            v.u,
            v.gk,
            call.H,
            seg_table,
            nseg,
            walk_seg,
            h0.ptr,
            h0.index,
            h0.hg,
            h0.n,
            out.ptr,
            out.index,
            out.hg,
            out.n,
            out.stride if not tape else h0.stride,
            v.h,
            v.v_new,
        ]
        return abi, len(args["walk"]) * call.H

    def _dhu_args(
        self, call: CallDetKcp, args: dict[str, Any], *, tape: bool
    ) -> tuple[list[Any], int]:
        walk_seg, seg_table, nseg, v = self._walk_common(call, args)
        if tape:
            dht = self._block(call, args.get("dht"))
            out = dht
        else:
            out = self._block(call, args.get("out"))
            dht = out
        abi = [
            v.kg,
            v.qg,
            v.w,
            v.do_int,
            v.dv1,
            v.gk,
            call.H,
            seg_table,
            nseg,
            float(call.scale),
            walk_seg,
            dht.ptr,
            dht.index,
            dht.hg,
            dht.n,
            out.ptr,
            out.index,
            out.hg,
            out.n,
            out.stride if not tape else dht.stride,
            v.dh,
            v.dv2,
        ]
        return abi, len(args["walk"]) * call.H

    def _transfer_args(
        self, call: CallDetKcp, args: dict[str, Any]
    ) -> tuple[list[Any], int]:
        walk_seg, seg_table, nseg, v = self._walk_common(call, args)
        out = self._block(call, args["out"])
        abi = [
            v.w,
            v.kg,
            v.u,
            v.gk,
            call.H,
            seg_table,
            nseg,
            walk_seg,
            out.ptr,
            out.index,
            out.hg,
            out.n,
            out.stride,
            int(args["init_mode"]),
            int(args["use_u"]),
            int(args["write_transposed"]),
        ]
        return abi, len(args["walk"]) * call.H

    def _merge_args(
        self, call: CallDetKcp, args: dict[str, Any]
    ) -> tuple[list[Any], int]:
        ch: Chains = args["chains"]
        if id(ch) not in self._checked_chains:
            ch.check_alignment()
            self._checked_chains.add(id(ch))
        args["cols_per_cta"] = cols = merge_cols_per_cta(
            ch.ncols, ch.n_chains, call.cols_rule
        )
        call.merge_log.append(
            (str(args.get("family")), int(ch.ncols), int(ch.n_chains), cols)
        )  # D1 evidence: None = registry record
        cols = self.cols_per_cta if cols is None else cols
        if ch.ncols % cols:
            raise ValueError(
                f"ncols {ch.ncols} is not a multiple of cols_per_cta {cols}"
            )
        dev = call.device
        y = args["y"]
        m = args["m"]
        c = (
            args["c"] if args.get("c") is not None else y
        )  # never read when every elem_c is -1
        x0 = (
            args["x0"] if args.get("x0") is not None else y
        )  # never read when every x0_index is -1
        tag = ("chain", id(ch))
        abi = [
            m,
            c,
            x0,
            y,
            self._dev_table(tag + ("start",), ch.chain_start, dev, torch.int32),
            self._dev_table(tag + ("m",), ch.elem_m, dev, torch.int64),
            self._dev_table(tag + ("c",), ch.elem_c, dev, torch.int64),
            self._dev_table(tag + ("y",), ch.elem_y, dev, torch.int64),
            self._dev_table(tag + ("x0",), ch.x0_index, dev, torch.int64),
            self._dev_table(tag + ("yf",), ch.y_final, dev, torch.int64),
            int(ch.ncols),
            int(args["transpose_m"]),
            int(args["write_states"]),
            cols,
        ]
        return abi, ch.n_chains * (ch.ncols // cols)


def merge_cols_per_cta(ncols: int, n_chains: int, rule: str) -> int | None:
    """D1 (the column-rule decision): the K4 ``cols_per_cta`` a launch asks for explicitly, ``None`` for the
    registry's choice of record (``fixed``: the shipped rule, c8 state / c16 map).  ``by_chains`` keeps the 8-column
    program for 128-column state chains up to :data:`COLS_RULE_CHAINS` concurrent chains and takes the 16-column
    program beyond it; 256-column map chains stay with the registry's c16.  Bitwise per layout (K4 REPORT §3)."""
    if rule == "by_chains" and int(ncols) == D and int(n_chains) > COLS_RULE_CHAINS:
        return 16
    return None


# Kernel parameter names of the frozen ABIs (K1 a217181fa, K2 a217181fa, K3 0637ec9f4, K4 eaa97cf5e): the logical
# vocabulary of the registry records when the FI-rendered provider launches by name.
ABI_PARAMS = {
    "fwdh_offset": (
        "w_tma",
        "kg_tma",
        "u",
        "gk",
        "num_heads",
        "seq_chunk_start",
        "num_seqs",
        "walk_seg",
        "h0",
        "h0_index",
        "h0_hg",
        "h0_n",
        "out",
        "out_index",
        "out_hg",
        "out_n",
        "tile_stride",
        "h_tape",
        "v_new",
    ),
    "dhu_offset": (
        "kg_tma",
        "qg_tma",
        "w_tma",
        "do_tma",
        "dv1",
        "gk",
        "num_heads",
        "seq_chunk_start",
        "num_seqs",
        "scale",
        "walk_seg",
        "dht",
        "dht_index",
        "dht_hg",
        "dht_n",
        "out",
        "out_index",
        "out_hg",
        "out_n",
        "tile_stride",
        "dh_tape",
        "dv2",
    ),
    "transfer_walk": (
        "w_tma",
        "kg_tma",
        "u",
        "gk",
        "num_heads",
        "seq_chunk_start",
        "num_seqs",
        "walk_seg",
        "out",
        "out_index",
        "out_hg",
        "out_n",
        "tile_stride",
        "init_mode",
        "use_u",
        "write_transposed",
    ),
    "affine_merge": (
        "m",
        "c",
        "x0",
        "y",
        "chain_start",
        "elem_m",
        "elem_c",
        "elem_y",
        "x0_index",
        "y_final",
        "ncols",
        "transpose_m",
        "write_states",
        "cols_per_cta",
    ),
}
ABI_PARAMS["fwdh_tape"] = ABI_PARAMS["fwdh_offset"]
ABI_PARAMS["dhu_tape"] = ABI_PARAMS["dhu_offset"]


# Walk-launch geometry one_cta_per_sm / tape_one_cta_per_sm: the K1 / K2 walk kernels' occupancy capped from the host.
# Their launch footprint (192 threads, 115712 B of dynamic shared memory) admits two CTAs per SM only at the full 228 KiB
# carveout; the smallest configuration that fits one CTA admits exactly one.  The runtime picks a kernel's configuration at
# launch from its function attribute (unset here) and otherwise from the device-wide preference (cudaDeviceSetCacheConfig):
# cudaFuncCachePreferL1 selects the smallest fitting shared-memory configuration, i.e. one CTA per SM.  The generated
# binding compiles the launch (cudaLaunchKernelExC with the fixed dynamic shared memory) and exports no symbol for the
# kernel's registered host function (its wrapper survives only as the mangled `__device_stub__`, which the runtime does
# not accept), so the per-function attribute cannot be set from the host; the knob brackets each capped launch with the
# device-wide preference instead (PreferL1 before, PreferNone after).  The preference is host-side state read at launch:
# launches already queued keep their configuration and only the bracketed launch takes PreferL1.  Host side only; no
# kernel, launch-configuration or argument change.
WALK_CACHE_CONFIG_CAP = 2  # cudaFuncCachePreferL1 (driver_types.h): the smallest shared-memory configuration that fits
WALK_CACHE_CONFIG_DEFAULT = (
    0  # cudaFuncCachePreferNone: the runtime's own choice (today's launches)
)
_CUDART_SONAMES = ("libcudart.so.13", "libcudart.so.12", "libcudart.so")
_CUDART_CACHE: dict[
    str, Any
] = {}  # "rt": the loaded runtime, or None after a failed load (one RuntimeWarning)
CACHE_CONFIG_STATS: dict[str, int] = {
    "bracket": 0,
    "capture_pref": 0,
    "none": 0,
    "unavailable": 0,
    "capture_skips": 0,
    "node_attr": 0,
    "node_attr_post": 0,
    "node_attr_launches": 0,
    "node_attr_degraded": 0,
}  # calls recorded per cache-config form; capture_skips = capped launches issued bare inside a capture; node_attr_launches = captured tape launches whose kernel node got the carveout attribute (the kernel-node cap)
_CUDART_CAPTURE_INFO_NAMES = (
    "cudaStreamGetCaptureInfo_v3",
    "cudaStreamGetCaptureInfo_v2",
    "cudaStreamGetCaptureInfo",
)  # export names of the stream-capture-info call: _v3 (CUDA 12.3+, 7 arguments incl. edgeData_out), _v2 (CUDA 11.3-12.x, 6 arguments), plain (CUDA 13 (13000+) = the 7-argument form; older runtimes = the pre-11.3 3-argument form, unusable)
_CUDART_GRAPH_ENTRY_POINTS = (
    "cudaGraphNodeGetType",
    "cudaGraphKernelNodeSetAttribute",
    "cudaGraphKernelNodeGetAttribute",
)  # with the capture-info call: the capture-graph entry points of the kernel-node cap forms


def _cudart() -> Any:
    """The CUDA runtime of this process (the instance torch and the JIT libraries link), its cache-config entry points
    typed."""
    for soname in _CUDART_SONAMES:
        try:
            rt = ctypes.CDLL(soname)
        except OSError:
            continue
        rt.cudaDeviceSetCacheConfig.argtypes = [ctypes.c_int]
        rt.cudaDeviceGetCacheConfig.argtypes = [ctypes.c_void_p]
        rt.cudaGetLastError.argtypes = []
        for fn in (
            rt.cudaDeviceSetCacheConfig,
            rt.cudaDeviceGetCacheConfig,
            rt.cudaGetLastError,
        ):
            fn.restype = ctypes.c_int
        return rt
    raise RuntimeError(
        f"walk_geometry one_cta_per_sm: none of {_CUDART_SONAMES} is loadable in this process"
    )


def _cudart_available() -> Any | None:
    """:func:`_cudart`'s runtime, or ``None`` when no soname of :data:`_CUDART_SONAMES` loads in this process (one
    ``RuntimeWarning`` naming the sonames tried; the capped programs then launch without the bracket, the
    LPT order stays, and the call records ``cache_config="unavailable"``)."""
    if "rt" not in _CUDART_CACHE:
        try:
            _CUDART_CACHE["rt"] = _cudart()
        except RuntimeError as exc:
            _CUDART_CACHE["rt"] = None
            warnings.warn(
                f"{exc}: the one-CTA-per-SM programs of walk_geometry launch without the cache-config bracket (cache_config='unavailable')",
                RuntimeWarning,
                stacklevel=2,
            )
    return _CUDART_CACHE["rt"]


def _capture_info_name(rt: Any) -> str | None:
    """The export name of the stream-capture-info call on runtime ``rt`` (:data:`_CUDART_CAPTURE_INFO_NAMES`, first match), or
    ``None``."""
    return next(
        (name for name in _CUDART_CAPTURE_INFO_NAMES if hasattr(rt, name)), None
    )


def _runtime_version(rt: Any) -> int:
    """``cudaRuntimeGetVersion`` of ``rt`` (0 when the call is missing or fails)."""
    fn = getattr(rt, "cudaRuntimeGetVersion", None)
    if fn is None:
        return 0
    ver = ctypes.c_int(0)
    try:
        rc = int(fn(ctypes.byref(ver)))
    except (TypeError, ctypes.ArgumentError):
        return 0
    return int(ver.value) if rc == 0 else 0


def _capture_info_arity(rt: Any, name: str) -> int:
    """Arguments of the stream-capture-info export ``name`` on ``rt``: 7 for ``_v3`` (``..., dependencies_out, edgeData_out,
    numDependencies_out``), 6 for ``_v2``, and for the plain name 7 on a CUDA 13 runtime (13000+; it exports the v3 form) or 0
    (unusable pre-11.3 form) on older runtimes."""
    if name == "cudaStreamGetCaptureInfo_v3":
        return 7
    if name == "cudaStreamGetCaptureInfo_v2":
        return 6
    return 7 if _runtime_version(rt) >= 13000 else 0


def graph_entry_points() -> dict[str, Any]:
    """The ctypes entry points of the ``node_attr`` forms as resolved in this process (the evidence record): the
    capture-info export name (or ``None``), the fixed entry points, and whether the typed runtime is available."""
    rt = _cudart_graph_available()
    return {
        "capture_info": _CUDART_CACHE.get("capture_info_name"),
        "capture_info_arity": _CUDART_CACHE.get("capture_info_arity"),
        "runtime_version": _CUDART_CACHE.get("runtime_version"),
        "entry_points": list(_CUDART_GRAPH_ENTRY_POINTS),
        "available": rt is not None,
        "cache_config_entry_points": [
            "cudaDeviceSetCacheConfig",
            "cudaDeviceGetCacheConfig",
            "cudaGetLastError",
            "cudaRuntimeGetVersion",
        ],
    }


def _cudart_graph() -> Any:
    """The CUDA runtime of :func:`_cudart_available` with the capture-graph entry points of the ``node_attr`` forms typed
    the stream-capture-info call (``cudaStreamGetCaptureInfo`` on CUDA 13, ``..._v2`` on CUDA 12:
    the capture status / graph of a stream and the nodes the next captured operation would depend on), ``cudaGraphNodeGetType``,
    ``cudaGraphKernelNodeSetAttribute`` / ``...GetAttribute`` (a kernel node's launch attribute).  Raises ``RuntimeError`` when
    the runtime or one of the entry points is missing."""
    rt = _cudart_available()
    if rt is None:
        raise RuntimeError(
            "graph_cap node_attr: no CUDA runtime is loadable in this process"
        )
    if hasattr(rt, "cudaRuntimeGetVersion") and isinstance(rt, ctypes.CDLL):
        rt.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        rt.cudaRuntimeGetVersion.restype = ctypes.c_int
    info_name = _capture_info_name(rt)
    missing = (
        [" | ".join(_CUDART_CAPTURE_INFO_NAMES)] if info_name is None else []
    ) + [name for name in _CUDART_GRAPH_ENTRY_POINTS if not hasattr(rt, name)]
    if missing:
        raise RuntimeError(
            f"graph_cap node_attr: the CUDA runtime exports no {missing}"
        )
    arity = _capture_info_arity(rt, info_name)
    if arity == 0:
        raise RuntimeError(
            f"graph_cap node_attr: {info_name} of this runtime (version {_runtime_version(rt)}) has the pre-11.3 form without dependency nodes"
        )
    info = getattr(rt, info_name)
    deps_t = ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
    info.argtypes = (
        [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_void_p),
            deps_t,
        ]
        + ([deps_t] if arity == 7 else [])
        + [ctypes.POINTER(ctypes.c_size_t)]
    )
    rt.cudaGraphNodeGetType.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    rt.cudaGraphKernelNodeSetAttribute.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    rt.cudaGraphKernelNodeGetAttribute.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    for fn in (
        info,
        rt.cudaGraphNodeGetType,
        rt.cudaGraphKernelNodeSetAttribute,
        rt.cudaGraphKernelNodeGetAttribute,
    ):
        fn.restype = ctypes.c_int
    _CUDART_CACHE["capture_info_name"] = info_name
    _CUDART_CACHE["capture_info_arity"] = arity
    _CUDART_CACHE["runtime_version"] = _runtime_version(rt)
    return rt


def _cudart_graph_available() -> Any | None:
    """:func:`_cudart_graph`'s runtime, or ``None`` (one ``RuntimeWarning``) when the runtime or an entry point is missing: the
    ``node_attr`` forms then capture without the cap and the call records ``cache_config="unavailable"``."""
    if "graph" not in _CUDART_CACHE:
        try:
            _CUDART_CACHE["graph"] = _cudart_graph()
        except RuntimeError as exc:
            _CUDART_CACHE["graph"] = None
            warnings.warn(
                f"{exc}: the graph_cap node_attr forms capture the tape programs without the cap (cache_config='unavailable')",
                RuntimeWarning,
                stacklevel=2,
            )
    return _CUDART_CACHE["graph"]


def capture_kernel_node(device_index: int) -> dict[str, Any]:
    """The kernel node just captured on the current stream of ``device_index`` (the kernel-node cap): the stream-capture-info call must
    report an active capture whose next operation depends on exactly one node, and that node must be a kernel node (the launch
    issued right before this call).  Returns the evidence record with the node handle; raises ``RuntimeError`` otherwise (:meth:`_FiPrograms.cap_captured_node`
    catches it: the launch stays uncapped and the capture goes on)."""
    rt = _cudart_graph_available()
    if rt is None:
        raise RuntimeError(
            "capture_kernel_node: the capture-graph entry points are unavailable"
        )
    info_name = _capture_info_name(rt)
    if info_name is None:
        raise RuntimeError(
            f"capture_kernel_node: the runtime exports none of {_CUDART_CAPTURE_INFO_NAMES}"
        )
    arity = _CUDART_CACHE.get("capture_info_arity") or _capture_info_arity(
        rt, info_name
    )
    if arity == 0:
        raise RuntimeError(
            f"capture_kernel_node: {info_name} has the pre-11.3 form without dependency nodes"
        )
    info = getattr(rt, info_name)
    stream = int(torch.cuda.current_stream(device_index).cuda_stream)
    status, capture_id, graph = (
        ctypes.c_int(-1),
        ctypes.c_ulonglong(0),
        ctypes.c_void_p(),
    )
    deps, edges, n_deps = (
        ctypes.POINTER(ctypes.c_void_p)(),
        ctypes.POINTER(ctypes.c_void_p)(),
        ctypes.c_size_t(0),
    )
    args = [
        ctypes.c_void_p(stream),
        ctypes.byref(status),
        ctypes.byref(capture_id),
        ctypes.byref(graph),
        ctypes.byref(deps),
    ]
    if arity == 7:
        args.append(
            ctypes.byref(edges)
        )  # const cudaGraphEdgeData **edgeData_out (unused: plain edges)
    args.append(ctypes.byref(n_deps))
    rc = int(info(*args))
    rec = {
        "device_index": int(device_index),
        "stream": stream,
        "capture_info": info_name,
        "arity": arity,
        "rc_info": rc,
        "status": int(status.value),
        "capture_id": int(capture_id.value),
        "graph": graph.value,
        "n_deps": int(n_deps.value),
    }
    if rc != 0:
        rt.cudaGetLastError()
        raise RuntimeError(f"{info_name} failed with CUDA error {rc}")
    if status.value != _STREAM_CAPTURE_ACTIVE or n_deps.value != 1:
        raise RuntimeError(
            f"capture_kernel_node: expected one dependency node of an active capture, got status {status.value} and {n_deps.value} nodes"
        )
    node = deps[0]
    node_type = ctypes.c_int(-1)
    rc = int(rt.cudaGraphNodeGetType(ctypes.c_void_p(node), ctypes.byref(node_type)))
    if rc != 0:
        rt.cudaGetLastError()
        raise RuntimeError(f"cudaGraphNodeGetType failed with CUDA error {rc}")
    if node_type.value != _GRAPH_NODE_TYPE_KERNEL:
        raise RuntimeError(
            f"capture_kernel_node: the just-captured node has type {node_type.value}, not a kernel node"
        )
    rec.update(node=int(node), node_type=int(node_type.value))
    return rec


def set_kernel_node_carveout(
    node: int, carveout: int = GRAPH_NODE_CARVEOUT
) -> dict[str, Any]:
    """Set the preferred shared-memory carveout launch attribute of kernel node ``node`` (the kernel-node cap) and read it back; returns
    the evidence record (``rc`` of the set, the read-back value).  ``cudaLaunchAttributeValue`` is a 64-byte union whose first
    unsigned int is ``sharedMemCarveout``."""
    rt = _cudart_graph_available()
    if rt is None:
        raise RuntimeError(
            "set_kernel_node_carveout: the capture-graph entry points are unavailable"
        )
    value = (ctypes.c_uint * 16)()
    value[0] = int(carveout) & 0xFFFFFFFF
    rc = int(
        rt.cudaGraphKernelNodeSetAttribute(
            ctypes.c_void_p(int(node)), _LAUNCH_ATTR_SHMEM_CARVEOUT, ctypes.byref(value)
        )
    )
    if rc != 0:
        rt.cudaGetLastError()
        raise RuntimeError(
            f"cudaGraphKernelNodeSetAttribute(PreferredSharedMemoryCarveout={carveout}) failed with CUDA error {rc}"
        )
    got = (ctypes.c_uint * 16)()
    got[0] = 0xFFFFFFFE
    rc_get = int(
        rt.cudaGraphKernelNodeGetAttribute(
            ctypes.c_void_p(int(node)), _LAUNCH_ATTR_SHMEM_CARVEOUT, ctypes.byref(got)
        )
    )
    if rc_get != 0:
        rt.cudaGetLastError()
    return {
        "node": int(node),
        "set": int(carveout),
        "rc": rc,
        "rc_get": rc_get,
        "readback": ctypes.c_int(got[0]).value if rc_get == 0 else None,
    }


def _capturing() -> bool:
    """``True`` inside a CUDA stream capture of the current thread (the bracket is never issued there)."""
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


# -- the trimmed host path's FFI stream context.  tvm_ffi.use_torch_stream() builds a torch.cuda.Stream and
# parses its device string on every launch (~20 us); use_raw_stream with the current raw stream handle of the device sets the
# same environment stream (the kernel runs on the same torch stream: bitwise the untrimmed launch).
_FFI_DEVICES: dict[int, Any] = {}


def _ffi_device(device_index: int) -> Any:
    dev = _FFI_DEVICES.get(int(device_index))
    if dev is None:
        import tvm_ffi  # noqa: PLC0415 - the launch path's dependency, imported at the first trimmed launch

        dev = _FFI_DEVICES[int(device_index)] = tvm_ffi.device(
            f"cuda:{int(device_index)}"
        )
    return dev


def _raw_stream(device_index: int) -> int:
    get = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if get is not None:
        return int(get(int(device_index)))
    return int(torch.cuda.current_stream(int(device_index)).cuda_stream)


def _ffi_stream(device_index: int) -> Any:
    """The FFI stream context of the trimmed launch: the device's current torch stream, as a raw handle."""
    import tvm_ffi  # noqa: PLC0415

    return tvm_ffi.use_raw_stream(_ffi_device(device_index), _raw_stream(device_index))


_ARCH_BY_INDEX: dict[int, str] = {}


def _arch_of(device_index: int) -> str:
    """``cake_jit.device_arch`` cached per device index (a device's capability never changes within a process)."""
    arch = _ARCH_BY_INDEX.get(int(device_index))
    if arch is None:
        arch = _ARCH_BY_INDEX[int(device_index)] = cake_jit.device_arch(
            int(device_index)
        )
    return arch


def _local_record(p: Any, stage: str, path: str) -> tuple[str, tuple[int, int, int]]:
    """``(record, grid)`` of local stage ``stage`` on ``path`` of plan ``p``: the plan's selection, or the direct WY
    record (:data:`WY_DIRECT_RECORDS`) on its own grid rule where the plan fuses the WY representation into a state
    recurrence the KCP does not run."""
    if stage in WY_DIRECT_RECORDS and not p.dispatched(stage, path):
        record = WY_DIRECT_RECORDS[stage]
        grid = cake_jit.launch_grid(
            record,
            num_chunks_pad=p.NC_pad,
            num_seqs=p.N,
            num_heads=p.H,
            sm_count=p.sm_count,
        )
        return record, grid
    return p.record(stage, path), p.grid(stage, path)


def _bwd_layout(local: Any) -> str:
    """The backward layout of a local arena or plan view: the lean one where it is sized for it
    (:func:`_lean_passes`), else :data:`LOCAL_LAYOUT_BWD`."""
    return LOCAL_LAYOUT_BWD_LEAN if _lean_passes(local) else LOCAL_LAYOUT_BWD


def _lean_passes(local: Any) -> int | None:
    """The tail passes of the lean layout on a local arena or plan view (its ``layout_passes`` entry), ``None`` where
    it runs the replay layout."""
    return getattr(local, "layout_passes", {}).get(LOCAL_LAYOUT_BWD_LEAN)


def _lean_tail(local: Any) -> tuple[int, torch.Tensor] | None:
    """``(tail passes, tail-count table)`` of a local arena or plan view on the lean layout, else ``None``."""
    passes = _lean_passes(local)
    if not passes:
        return None
    return int(passes), local.__dict__[TAIL_COUNTS_ATTR]


def _tail_pass_values(
    views: Any, ctx: Any, p: Any, passes: int, counts: torch.Tensor
) -> tuple[tuple[dict[str, Any], dict[str, tuple[int, int, int]]], ...]:
    """Per tail range of the lean layout (``cake_backend._tail_ranges(NC_pad, passes)``): the value overrides of the
    tail stages and their grids.  The whole-sequence rows they read (:data:`TAIL_ROW_BUFFERS`, the ``h`` / ``dh``
    tapes) as leading-axis views offset by the range's first chunk, the context's chunk tables as the range's slices,
    ``seq_chunk_start`` as the one-entry slice of ``counts`` holding the range's real chunk count (``num_seqs = 0``
    makes the fused gradient read it as its item bound; the context's ``NC`` on the host, so no call forms it), the
    per-pass buffers as the layout sizes them (one range) and the token tensors whole (every kernel reaches them
    through the absolute ``chunk_bos``); the grids are the plan's rules at the range's chunk count.  The same
    kernels address the same bytes as in one pass over the replay layout."""
    nc = int(ctx.NC)
    out = []
    for c0, length in cake_backend._tail_ranges(p.NC_pad, passes):
        rows = cake_backend.BT * c0
        real = min(max(nc - c0, 0), length)
        overrides: dict[str, Any] = {
            name: getattr(views, name)[rows:] for name in TAIL_ROW_BUFFERS
        }
        overrides.update(
            h2d=views.h[c0:].view(-1, D),
            dh2d=views.dh[c0:].view(-1, D),
            chunk_bos=ctx.tables["chunk_bos"][c0 : c0 + length],
            chunk_len=ctx.tables["chunk_len"][c0 : c0 + length],
            seq_chunk_start=counts[real : real + 1],
            num_seqs=0,
            num_chunks_pad=length,
        )
        grids = {
            stage: cake_jit.launch_grid(
                p.record(stage, LOCAL_PATH_BWD),
                num_chunks_pad=length,
                num_seqs=p.N,
                num_heads=p.H,
                sm_count=p.sm_count,
                num_segments=p.NSEG,
            )
            for stage in TAIL_STAGES
            if p.dispatched(stage, LOCAL_PATH_BWD)
        }
        out.append((overrides, grids))
    return tuple(out)


def kcp_local_grid(stage: str, p: Any) -> tuple[int, int, int]:
    """The launch grid of the KCP-owned local record of ``stage`` (:data:`KCP_LOCAL_RECORDS`) from the plan's
    scalars: the output program launches one CTA per (chunk, head) item, ``NC_pad * H`` CTAs (the CTAs that start
    take the items of the CTAs that have not started); the ``dA_qk`` / ``dv1`` program launches one CTA per chunk pair
    and head, ``(NC_pad / 2, H, 1)``."""
    if stage == "fwd_o":
        return (max(1, int(p.NC_pad) * int(p.H)), 1, 1)
    if stage == "dav":
        return (int(p.NC_pad) // 2, int(p.H), 1)
    raise ValueError(f"no KCP-owned local record for stage {stage!r}")


VIEW_CACHE_MAX = 64  # plan views of one local arena a WorkspaceDetKcp keeps (the least recently bound goes first)
SLICE_CACHE_MAX = 256  # pool[:n] views a WorkspaceDetKcp keeps (cleared when full)
ENSURE_CACHE_MAX = (
    128  # trimmed-path ensure records a WorkspaceDetKcp keeps (cleared when full)
)


def _plan_view(arena: Any, plan: Any) -> Any:
    """A view of the local arena ``arena`` (``cake_backend.KdaWorkspace``) under the plan of another pack whose
    layouts fit it: the same bytes, ``eye`` / ``flags`` and epoch counter, with its own layout views, its own launch
    templates (:data:`TEMPLATE_CACHE_ATTR`) and its own scratch table set (:data:`SCRATCH_TABLES_ATTR`, sized by its
    plan).  ``KdaWorkspace.views`` checks that each layout fits the arena's bytes."""
    view = arena._view(plan)
    view.__dict__.pop(TEMPLATE_CACHE_ATTR, None)
    view.__dict__.pop(SCRATCH_TABLES_ATTR, None)
    return view


def _weak(obj: Any) -> Any:
    """A callable returning ``obj`` that holds it weakly where its type allows (a duck-typed context may not)."""
    try:
        return weakref.ref(obj)
    except TypeError:
        return lambda: obj


def _table_tensors(tb: Any) -> list[torch.Tensor]:
    """The live device tables a table set held by a workspace (not by its context) has uploaded
    (:meth:`_Tables.upload_specs`)."""
    return list(tb.__dict__.get("_uploaded", ()))


def _context_list(ctxs: Any) -> list[Any]:
    """One context or a sequence of contexts as a list."""
    if isinstance(ctxs, (list, tuple)):
        return list(ctxs)
    return [ctxs]


def _scratch_tables(local: Any) -> dict[str, torch.Tensor]:
    """The table set the intra record that computes the chunk layout from ``cu_seqlens`` writes (the existing
    backend's tables, its block table included), on a scratch buffer held by the arena: the KCP's CPU-built tables
    are never rewritten by a kernel and no KCP program reads this set (:func:`cake_kcp_plan.local_scratch_table_bytes`
    counts it)."""
    tables = local.__dict__.get(SCRATCH_TABLES_ATTR)
    if tables is None:
        p = local.plan
        buffer = torch.zeros((p.table_words(),), dtype=torch.int32, device=local.device)
        tables = {
            name: buffer[off : off + words] for name, off, words in p.table_layout()
        }
        setattr(local, SCRATCH_TABLES_ATTR, tables)
    return tables


def _launch_record_trim(
    local: Any, name: str, grid: tuple[int, int, int], values: dict[str, Any]
) -> None:
    """Trimmed host path: ``cake_jit.run_module(name, values, grid=grid, device_index=local.device_index)`` with the
    launch inlined -- the record's argument plan (cached by cake_jit), the arena's cached arch, the raw-stream FFI
    context.  Same record, same grid, same positional arguments, same stream as ``local.run``."""
    jit = cake_jit
    record = jit.MODULES[name]
    args = []
    for logical_name, axis in jit._argument_plan(name):
        if logical_name is None:
            args.append(int(grid[axis]))
        else:
            try:
                args.append(values[logical_name])
            except KeyError:
                raise ValueError(
                    f"record {name} needs logical argument {logical_name!r}"
                ) from None
    module = jit.load(name, local.arch)
    with _ffi_stream(local.device_index):
        getattr(module, record["ffi_entry"])(*args)


def walk_device_cache_config(device_index: int, config: int | None) -> dict[str, Any]:
    """Set (``config`` not ``None``) and read back the device-wide cache-configuration preference of device
    ``device_index`` through the CUDA runtime; returns the evidence record (``rc`` of the set, the read-back value)."""
    rt = _cudart()
    with torch.cuda.device(device_index):
        rc = None
        if config is not None:
            rc = int(rt.cudaDeviceSetCacheConfig(int(config)))
            if rc != 0:
                raise RuntimeError(
                    f"cudaDeviceSetCacheConfig({config}) failed with CUDA error {rc}"
                )
        got = ctypes.c_int(-1)
        rc_get = int(rt.cudaDeviceGetCacheConfig(ctypes.byref(got)))
        if rc_get != 0:
            raise RuntimeError(
                f"cudaDeviceGetCacheConfig failed with CUDA error {rc_get}"
            )
    return {
        "device_index": int(device_index),
        "set": config,
        "rc": rc,
        "readback": int(got.value),
    }


class _FiPrograms:
    """The FI-rendered program provider: H3's :mod:`.cake_kcp_jit` registry, through whichever entry it exports:
    ``launch_kcp(program, abi, *, grid, device_index, precision_mode, transfer_walk)`` (positional ABI), else
    ``run_program(name, args, device_index=)`` (the abstract launch ``args`` of :class:`CallDetKcp` — H3's 09:59Z form),
    else ``run_module(record, values, grid=(x, 1, 1), device_index=)`` with ``values`` keyed by :data:`ABI_PARAMS`
    (record names ``kcp_<program>``; the merge record by ``ncols``: ``_c8`` / ``_c16``; K3 test variants by suffix).
    Under a ``walk_geometry`` in :data:`WALK_CAP_GEOMETRIES` (the shipped ``lpt_tape_one_cta_per_sm`` since this
    revision) the programs of :data:`WALK_CAP_PROGRAMS` launch at one CTA per SM (their full-width
    record; a ``walk_form="split"`` launch runs its two-CTA record uncapped): the launch
    is bracketed by the device cache-config preference (:func:`walk_device_cache_config`,
    ``WALK_CACHE_CONFIG_CAP`` before, ``WALK_CACHE_CONFIG_DEFAULT`` after); ``carveout_log`` keeps the first brackets'
    evidence per (program, device) and ``carveout_state`` the preference last set per device.  Inside a CUDA stream
    capture the bracket is never issued (``cudaDeviceSetCacheConfig`` invalidates a capture; the
    captured call's form is :func:`_graph_cap_bracket`'s; ``capture_skips`` counts such launches) and without a loadable
    CUDA runtime the capped programs launch bare after one warning (``unavailable`` counts them)."""

    path = "fi-rendered"

    def __init__(self, jit: Any) -> None:
        self.jit = jit
        self.carveout_state: dict[
            int, int
        ] = {}  # device -> the cache-config preference last set (never touched: absent)
        self.carveout_log: dict[
            tuple[str, int], list[dict[str, Any]]
        ] = {}  # (program, device) -> evidence of the first brackets
        self._rt: Any = None
        self.capture_skips = (
            0  # capped launches issued inside a stream capture (no bracket)
        )
        self.unavailable = (
            0  # capped launches issued without a loadable CUDA runtime (no bracket)
        )
        self.node_attrs = (
            0  # captured tape launches whose kernel node got the carveout attribute
        )
        self.node_attr_log: dict[
            tuple[str, int], list[dict[str, Any]]
        ] = {}  # (program, device) -> evidence of the first node attributes

    def run_record(
        self,
        record: str,
        values: dict[str, Any],
        *,
        grid: tuple[int, int, int],
        device_index: int,
    ) -> None:
        """A KCP-owned local record (:data:`KCP_LOCAL_RECORDS`) with logical ``values`` on ``grid``: the registry's
        ``run_module`` (the record maps its kernel parameters onto the logical vocabulary)."""
        self.jit.run_module(record, values, grid=grid, device_index=device_index)

    def set_cache_config(self, program: str, device_index: int, config: int) -> None:
        """One side of a capped launch's bracket.  The first brackets of a (program, device) go through
        :func:`walk_device_cache_config` (set + read-back, kept as evidence); the later ones are the bare runtime call."""
        key = (program, int(device_index))
        log = self.carveout_log.setdefault(key, [])
        if len(log) < 4:
            log.append(walk_device_cache_config(device_index, config))
        else:
            if self._rt is None:
                self._rt = _cudart()
            if torch.cuda.current_device() != int(device_index):
                with torch.cuda.device(device_index):
                    rc = int(self._rt.cudaDeviceSetCacheConfig(int(config)))
            else:
                rc = int(self._rt.cudaDeviceSetCacheConfig(int(config)))
            if rc != 0:
                raise RuntimeError(
                    f"cudaDeviceSetCacheConfig({config}) failed with CUDA error {rc}"
                )
        self.carveout_state[int(device_index)] = int(config)

    def launch(
        self,
        program: str,
        abi: list[Any],
        *,
        grid: int,
        device_index: int,
        precision_mode: int = 0,
        transfer_walk: str = "bf16x3_hw64",
        name: str | None = None,
        args: dict[str, Any] | None = None,
        cols_per_cta: int | None = None,
        walk_form: str = "full",
    ) -> None:
        call = args.get("_call") if args is not None else None
        geometry = (
            getattr(call, "walk_geometry", WALK_GEOMETRY_DEFAULT)
            if call is not None
            else WALK_GEOMETRY_DEFAULT
        )
        if (
            program in WALK_CAP_PROGRAMS.get(geometry, ()) and walk_form != "split"
        ):  # the full-width launch at one CTA per SM (bracketed preference); a split launch runs two CTAs per SM
            scratch = getattr(
                call, "scratch", None
            )  # the call's record of its cache-config form
            if _capturing():  # never inside a stream capture; the capture's preference is _graph_cap_bracket's
                form = getattr(
                    getattr(call, "opts", None), "graph_cap", GRAPH_CAP_DEFAULT
                )
                if (
                    form in GRAPH_NODE_CAPS and _cudart_graph_available() is not None
                ):  # the cap as the node's launch attribute
                    self._launch(
                        program,
                        abi,
                        grid=grid,
                        device_index=device_index,
                        precision_mode=precision_mode,
                        transfer_walk=transfer_walk,
                        name=name,
                        args=args,
                        cols_per_cta=cols_per_cta,
                        walk_form=walk_form,
                    )
                    self.cap_captured_node(
                        program, device_index, call, immediate=(form == "node_attr")
                    )
                    return
                self.capture_skips += 1
                CACHE_CONFIG_STATS["capture_skips"] += 1
                self._launch(
                    program,
                    abi,
                    grid=grid,
                    device_index=device_index,
                    precision_mode=precision_mode,
                    transfer_walk=transfer_walk,
                    name=name,
                    args=args,
                    cols_per_cta=cols_per_cta,
                    walk_form=walk_form,
                )
                return
            if (
                _cudart_available() is None
            ):  # no CUDA runtime soname -> the LPT order stays, the cap is skipped
                self.unavailable += 1
                if scratch is not None:
                    scratch["cache_config"] = "unavailable"
                self._launch(
                    program,
                    abi,
                    grid=grid,
                    device_index=device_index,
                    precision_mode=precision_mode,
                    transfer_walk=transfer_walk,
                    name=name,
                    args=args,
                    cols_per_cta=cols_per_cta,
                    walk_form=walk_form,
                )
                return
            if scratch is not None:
                scratch["cache_config"] = "bracket"
            self.set_cache_config(program, device_index, WALK_CACHE_CONFIG_CAP)
            try:
                self._launch(
                    program,
                    abi,
                    grid=grid,
                    device_index=device_index,
                    precision_mode=precision_mode,
                    transfer_walk=transfer_walk,
                    name=name,
                    args=args,
                    cols_per_cta=cols_per_cta,
                    walk_form=walk_form,
                )
            finally:
                self.set_cache_config(program, device_index, WALK_CACHE_CONFIG_DEFAULT)
            return
        self._launch(
            program,
            abi,
            grid=grid,
            device_index=device_index,
            precision_mode=precision_mode,
            transfer_walk=transfer_walk,
            name=name,
            args=args,
            cols_per_cta=cols_per_cta,
            walk_form=walk_form,
        )

    def cap_captured_node(
        self, program: str, device_index: int, call: Any, *, immediate: bool
    ) -> None:
        """Kernel-node cap: the kernel node just captured for ``program`` gets the one-CTA-per-SM carveout attribute -- now
        (``graph_cap="node_attr"``) or when the capture has ended (``"node_attr_post"``: the handle is kept on the call for
        :meth:`apply_node_caps`).  The first four records per (program, device) are kept as evidence (``node_attr_log``).
        Degrade contract: when :func:`capture_kernel_node` cannot name the node (a missing entry point, a
        capture-info reply without exactly one kernel-node dependency, a non-kernel node), the launch just issued stays uncapped
        and the capture goes on -- counted as a capture skip and in ``CACHE_CONFIG_STATS["node_attr_degraded"]``, one
        ``RuntimeWarning`` per process, the reason recorded as ``call.scratch["node_attr_degraded"]`` (the call then records
        ``cache_config="unavailable"``); never an eager fallback.  A failing set-attribute call keeps raising."""
        try:
            info = capture_kernel_node(device_index)
        except RuntimeError as exc:  # the launch stays uncapped, the capture goes on (never an eager fallback)
            self.capture_skips += 1
            CACHE_CONFIG_STATS["capture_skips"] += 1
            CACHE_CONFIG_STATS["node_attr_degraded"] += 1
            scratch = getattr(call, "scratch", None)
            if scratch is not None:
                scratch["node_attr_degraded"] = str(exc)[:200]
            if not _CUDART_CACHE.get("node_attr_degraded_warned"):
                _CUDART_CACHE["node_attr_degraded_warned"] = True
                warnings.warn(
                    f"graph_cap node_attr: {exc}: the captured tape launch stays uncapped (the call records cache_config='unavailable')",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return
        if immediate:
            info.update(set_kernel_node_carveout(info["node"]))
            self.node_attrs += 1
            CACHE_CONFIG_STATS["node_attr_launches"] += 1
        log = self.node_attr_log.setdefault((program, int(device_index)), [])
        if len(log) < 4:
            log.append({**info, "program": program, "immediate": bool(immediate)})
        scratch = getattr(call, "scratch", None)
        if scratch is not None:
            scratch["node_attr_launches"] = int(
                scratch.get("node_attr_launches", 0)
            ) + (1 if immediate else 0)
            if not immediate:
                scratch.setdefault("node_attr_pending", []).append(
                    (program, int(device_index), int(info["node"]))
                )

    def apply_node_caps(self, call: Any) -> int:
        """``graph_cap="node_attr_post"``: set the carveout attribute on every node kept by :meth:`cap_captured_node` (after the
        capture ended, before the graph is instantiated); returns the number of nodes set."""
        pending = list(call.scratch.pop("node_attr_pending", []))
        for program, device_index, node in pending:
            rec = set_kernel_node_carveout(node)
            self.node_attrs += 1
            CACHE_CONFIG_STATS["node_attr_launches"] += 1
            for entry in self.node_attr_log.get((program, int(device_index)), []):
                if entry.get("node") == node:
                    entry.update(rec)
        call.scratch["node_attr_launches"] = int(
            call.scratch.get("node_attr_launches", 0)
        ) + len(pending)
        return len(pending)

    def _launch(
        self,
        program: str,
        abi: list[Any],
        *,
        grid: int,
        device_index: int,
        precision_mode: int = 0,
        transfer_walk: str = "bf16x3_hw64",
        name: str | None = None,
        args: dict[str, Any] | None = None,
        cols_per_cta: int | None = None,
        walk_form: str = "full",
    ) -> None:
        jit = self.jit
        if (
            program == "affine_merge" and cols_per_cta is not None
        ):  # D1 cols_rule: the explicit `_c<cols>` record
            values = dict(zip(ABI_PARAMS[program], abi, strict=True))
            jit.run_module(
                f"kcp_affine_merge_c{int(cols_per_cta)}",
                values,
                grid=(int(grid), 1, 1),
                device_index=device_index,
            )
            return
        if hasattr(jit, "launch_kcp"):
            jit.launch_kcp(
                program,
                abi,
                grid=grid,
                device_index=device_index,
                precision_mode=precision_mode,
                transfer_walk=transfer_walk,
            )
            return
        if hasattr(jit, "run_program") and name is not None and args is not None:
            jit.run_program(name, args, device_index=device_index)
            return
        record = f"kcp_{program}"
        if (
            program in _WALK_K12
        ):  # the record of this launch's form: a fit program's two-CTA record is its ``_vh64``
            # variant, a fixed-rule program's other form its test-only ``_<form>`` record
            rule = getattr(jit, "WALK_FORM_RULES", {}).get(record, "fit")
            if rule == "fit":
                if walk_form == "split":
                    record += "_vh64"
            elif walk_form in ("full", "split") and walk_form != rule:
                record += f"_{walk_form}"
            if walk_form == "split":  # two CTAs per segment and head
                grid = int(grid) * 2
        if program == "transfer_walk":
            if precision_mode:
                record += {1: "_three_mma", 2: "_plain_tf32"}[int(precision_mode)]
            elif transfer_walk != TRANSFER_WALKS[0]:
                record += f"_{transfer_walk}"
        elif program == "affine_merge":
            record += (
                "_c8" if int(abi[ABI_PARAMS[program].index("ncols")]) == D else "_c16"
            )
        values = dict(zip(ABI_PARAMS[program], abi, strict=True))
        jit.run_module(
            record, values, grid=(int(grid), 1, 1), device_index=device_index
        )


_BACKEND: Backend = KernelBackend()


def set_backend(backend: Backend | None) -> Backend:
    """Install ``backend`` as the dispatch table of every following call (``None``: the shipped
    :class:`KernelBackend`); returns the previous one."""
    global _BACKEND
    previous = _BACKEND
    _BACKEND = KernelBackend() if backend is None else backend
    return previous


def get_backend() -> Backend:
    return _BACKEND


# ----------------------------------------------------------------------------------------------------------------
# Tables of one (ctx, H) problem, derived from the plan module's head plan (``ctx.heads(H)``)


@dataclasses.dataclass(frozen=True)
class _IndexedMeta:
    """Trimmed host path: the precomputed part of an :class:`Indexed` (see :meth:`_Tables.indexed_meta`)."""

    index: tuple[int, ...]
    tiles: tuple[tuple[int, ...], ...]
    H: int
    addr: tuple[int, int, int, int]
    device: str
    dev_index: torch.Tensor


@dataclasses.dataclass(frozen=True)
class Indexed:
    """A compact state pool as a walk kernel reads or writes it: ``pool`` (a flat tensor of ``128 x 128`` tiles),
    ``index[seg]`` (the slot of segment ``seg``; ``-1`` = no tile: the zero state on input, not written on output) and
    ``tiles[slot][h]`` (the tile number of head ``h`` of that slot inside ``pool``).  :meth:`addressing` recovers the
    kernels' two-integer form: ``tiles[slot][h] == base_tile + tile_index(h, slot, hg, n) * stride``."""

    pool: torch.Tensor
    index: tuple[int, ...]
    tiles: tuple[tuple[int, ...], ...]
    meta: Any = None  # the tables' precomputed _IndexedMeta (addressing + device index table) of the trimmed host path

    @property
    def n_slots(self) -> int:
        return len(self.tiles)

    def addressing(self, H: int) -> tuple[int, int, int, int]:
        """``(base_tile, hg, n, stride)`` of the kernels' addressing, ``ValueError`` when the table has no such form."""
        H = int(H)
        if self.meta is not None and self.meta.H == H:
            return self.meta.addr
        n_slots = len(self.tiles)
        if n_slots == 0:
            return 0, int(H), 0, 1
        base = self.tiles[0][0]
        for stride in (1, 2):
            for hg in range(1, H + 1):
                if hg < H:
                    span = self.tiles[0][hg] - base
                    if span <= 0 or span % (hg * stride):
                        continue
                    n = span // (hg * stride)
                    if n < n_slots:
                        continue
                else:
                    n = n_slots
                if all(
                    self.tiles[s][h] == base + tile_index(h, s, hg, n) * stride
                    for s in range(n_slots)
                    for h in range(H)
                ):
                    return base, hg, n, stride
        raise ValueError("Indexed: the tile table has no two-integer (hg, n) form")


class Chains:
    """K4 chain tables of one launch (the design note §14.4): CSR over elements ``(elem_m, elem_c, elem_y)``
    in application order (element offsets into the ``m`` / ``c`` / ``y`` pools; ``elem_c = -1`` no offset; ``elem_y`` =
    where the state ENTERING the element is published when ``write_states``, ``-1`` none), the initial state
    ``x0_index`` (``-1``: zero) and ``y_final`` (the state leaving the chain; an empty chain publishes its initial state).
    ``elem_m`` must be ``>= 0`` (the kernel has no identity element)."""

    def __init__(self, ncols: int = D) -> None:
        self.ncols = int(ncols)
        self.chain_start: list[int] = [0]
        self.elem_m: list[int] = []
        self.elem_c: list[int] = []
        self.elem_y: list[int] = []
        self.x0_index: list[int] = []
        self.y_final: list[int] = []

    @property
    def n_chains(self) -> int:
        return len(self.x0_index)

    @property
    def n_elems(self) -> int:
        return len(self.elem_m)

    def add(
        self, elems: Sequence[tuple[int, int, int]], *, x0: int = -1, y_final: int = -1
    ) -> None:
        for m, c, y in elems:
            if m < 0:
                raise ValueError(
                    "Chains: every element needs an operator tile (elem_m >= 0)"
                )
            self.elem_m.append(int(m))
            self.elem_c.append(int(c))
            self.elem_y.append(int(y))
        self.chain_start.append(len(self.elem_m))
        self.x0_index.append(int(x0))
        self.y_final.append(int(y_final))

    def check_alignment(self) -> None:
        for name in ("elem_m", "elem_c", "elem_y", "x0_index", "y_final"):
            if any(v % 4 for v in getattr(self, name) if v >= 0):
                raise ValueError(
                    f"Chains: {name} holds an offset that is not a multiple of 4 floats"
                )

    def device_tables(
        self, device: str, upload: Any | None = None
    ) -> dict[str, torch.Tensor]:
        """Trimmed host path: the six device tables of the K4 launch (``chain_start`` int32; ``elem_m`` / ``elem_c`` / ``elem_y`` /
        ``x0_index`` / ``y_final`` int64; an empty element table holds one 0 so the kernel has a valid pointer -- the
        same tensors :func:`cake_kcp_jit._chain_tables` builds), cached on the chain set per device.  The tables are
        frozen once the launches start (``_Tables`` builds them once per ``(ctx, H)``).  ``upload`` (the table set's
        :meth:`_Tables.upload_specs`) copies the six in one asynchronous pinned copy; without it one
        ``torch.tensor(..., device=)`` each."""
        cache = self.__dict__.setdefault("_dev_tables", {})
        t = cache.get(device)
        if t is None:
            pad = lambda table: table if table else [0]  # noqa: E731
            i32, i64 = torch.int32, torch.int64
            specs: dict[str, tuple] = {
                "chain_start": (list(self.chain_start), i32, None),
                "elem_m": (pad(list(self.elem_m)), i64, None),
                "elem_c": (pad(list(self.elem_c)), i64, None),
                "elem_y": (pad(list(self.elem_y)), i64, None),
                "x0_index": (list(self.x0_index), i64, None),
                "y_final": (list(self.y_final), i64, None),
            }
            if upload is not None:
                t = cache[device] = upload(specs, device)
            else:
                t = cache[device] = {
                    name: torch.tensor(values, dtype=dtype, device=device)
                    for name, (values, dtype, _shape) in specs.items()
                }
        return t

    @classmethod
    def concat(cls, *parts: Chains) -> Chains:
        """The chains of ``parts`` in order as one launch table (same ``ncols``; element and chain tables appended, the
        CSR starts shifted) -- the fused carry families of ``merge_fuse="carry"``.  Every chain keeps its own elements, initial state and
        final slot, so the launch computes exactly what the separate launches compute."""
        out = cls(parts[0].ncols if parts else D)
        for c in parts:
            if c.ncols != out.ncols:
                raise ValueError(f"Chains.concat: ncols {c.ncols} != {out.ncols}")
            base = len(out.elem_m)
            out.elem_m += c.elem_m
            out.elem_c += c.elem_c
            out.elem_y += c.elem_y
            out.chain_start += [base + s for s in c.chain_start[1:]]
            out.x0_index += c.x0_index
            out.y_final += c.y_final
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "ncols": self.ncols,
            "n_chains": self.n_chains,
            "chain_start": list(self.chain_start),
            "elem_m": list(self.elem_m),
            "elem_c": list(self.elem_c),
            "elem_y": list(self.elem_y),
            "x0_index": list(self.x0_index),
            "y_final": list(self.y_final),
        }


def _prefix(values) -> list[int]:
    out = [0]
    for v in values:
        out.append(out[-1] + int(v))
    return out


class _Tables:
    """Everything the launch sequences read for one ``(ctx, H)``: the segment plan, walk lists, the compact tile
    tables of every pool, the collective splits and the K4 chains, derived once from ``cake_kcp_plan.ContextDetKcp`` /
    ``KcpHeadPlan`` and cached on the context (``ctx._table_sets``, shared by every workspace; the test knob
    ``table_cache="workspace"`` keeps the previous per-workspace cache) (the design note §7.2, §10.3, §14.4).  The set
    holds the context weakly; a table it copies to the device at first use costs one asynchronous pinned copy
    (:meth:`upload_specs`).

    Pools (tile = 16384 floats; tile numbers pool-relative): ``c1_send`` = owner-major ``(out_slot, head)`` pairs
    ``[B_f | M_f^T]`` (B tile ``2 t``, M tile ``2 t + 1`` of ``hp.out_tile[out_slot][h] = t``); ``S_in`` (the c2
    receive, ``hp.h0_tile``), ``b2_recv`` = ``dS_out`` (``hp.dht_tile``), ``b1_send`` (``hp.b1_d_tile_d`` /
    ``hp.b1_d_tile_dm`` / ``hp.b1_m_tile_dm``: the plan module's layout of the ``d`` and ``dm`` forms); at ``P > 1`` the
    staging planes ``stage_b`` (``b_seg`` / ``d_seg``) and ``stage_m`` (``M_seg^T``) ``[NSEG, H]`` and the carry plane
    ``[n_frag (P - 1), H]`` (forward carries, then the adjoint carries); ``eye`` = one identity tile.

    Operator orientation (K4): ``transpose_m = 0`` reads a tile holding ``Mop^T`` (the c1 ``M_f^T`` tiles and the
    ``M_seg^T`` staging under ``Mop = M``; the b1 ``M_f`` tiles under ``Mop = M^T``), ``transpose_m = 1`` a tile holding
    ``Mop`` (the c1 / staging ``M^T`` tiles under ``Mop = M^T``).  The segment compositions of a fragment (``P > 1``)
    therefore split into the forward family (``Mop = M_p``, segments in order, ``transpose_m = 0``: ``B_f``, ``M_f``,
    carries) and the adjoint family (``Mop = M_p^T``, segments in reverse, ``transpose_m = 1``: ``d_f``, ``M_f^T``,
    adjoint carries); each is one launch (the pair orientations are ``[B_f | M_f^T]`` and ``[d_f | M_f]``).
    """

    _ring: _Ring | None = None  # the ring-exchange tables, derived on first use
    # held by the context's table-set cache (WorkspaceDetKcp.tables)
    on_context: bool = False
    # the trimmed path's absent-pool slot table is this set's own copy (absent_index; WorkspaceDetKcp.tables)
    own_absent: bool = False

    def __init__(
        self,
        ctx: Any,
        H: int,
        *,
        fused_walks: int = FUSED_WALKS_DEFAULT,
        walk_geometry: str = WALK_GEOMETRY_DEFAULT,
    ) -> None:
        try:  # the context owns its table sets (ctx._table_sets): no reference cycle, refcount frees both
            self.ctx = weakref.proxy(ctx)
        except TypeError:  # a context type without weak references
            self.ctx = ctx
        self.H = H = int(H)
        self.fused_walks = int(
            fused_walks
        )  # fused walks: one base pointer per tape phase
        if walk_geometry not in WALK_GEOMETRIES:
            raise ValueError(
                f"walk_geometry must be one of {WALK_GEOMETRIES}, got {walk_geometry!r}"
            )
        self.walk_geometry = str(
            walk_geometry
        )  # walk-launch geometry: the walk lists' order (+ the launch-time occupancy cap)
        hp = self.hp = ctx.heads(H)
        self.P, self.NSEG = int(hp.P), int(hp.NSEG)
        self.Hg, self.Hg_r = int(hp.Hg), int(hp.Hg_r)
        self.n_frag, self.T_r = int(ctx.n_frag), int(ctx.T_r)
        self.n_in, self.n_out = int(ctx.n_in), int(ctx.n_out)
        self.max_hops = max(int(ctx.n_d_max) - 1, 0)
        defaults = {
            "fwd_m_off": SUMMARY_TILE,
            "fwd_b_off": SUMMARY_TILE,
            "fwd_out_off": TILE,
            "adj_d_off_d": TILE,
            "adj_d_off_dm": SUMMARY_TILE,
            "adj_m_off_b1": SUMMARY_TILE,
            "adj_m_off_c1": SUMMARY_TILE,
            "adj_out_off": TILE,
        }
        self.hg_stride = {
            **defaults,
            **{k: int(v) for k, v in (getattr(hp, "hg_stride", None) or {}).items()},
        }
        P, NSEG = self.P, self.NSEG
        self.frags = frags = list(ctx.fragments)
        in_slot, out_slot = list(ctx.in_slot), list(ctx.out_slot)
        live = [f for f in frags if f.length > 0]
        starts = [
            f for f in live if not f.crossing_in
        ]  # document-start fragments (single-fragment documents too)
        ins = [f for f in live if f.crossing_in]
        outs = [f for f in live if f.crossing_out]
        ends = [f for f in live if not f.crossing_out]  # document-last fragments

        seg_table = [int(x) for x in hp.seg_chunk_start]

        def segs(fs, ps):
            # LPT order (longest segment first; ties by segment number): the CTA issue order of a walk launch — the
            # kernels are per-CTA independent, so the order changes tail effects only, never a bit of the results
            out = [f.slot * P + p for f in fs for p in ps]
            return sorted(out, key=lambda s: (-(seg_table[s + 1] - seg_table[s]), s))

        # -- walk lists (segments)
        self.walk_offset = [
            int(s) for s in hp.walk_seg_offset
        ]  # K1 / K3 offset walks (crossing-out frags at P = 1, all at P > 1)
        self.walk_all = list(range(NSEG))
        self.walk_start = segs(
            starts, range(P)
        )  # tape under c1: zero (p = 0) / carried (p >= 1) initial states
        self.walk_in_p0 = segs(ins, [0])  # tape after c2 from S_in
        self.walk_in_p1 = segs(ins, range(1, P))  # tape after c2 from the carry plane
        self.walk_end = segs(
            ends, range(P)
        )  # adjoint tape under b2: zero (p = P - 1) / carried final adjoints
        self.walk_cross_last = segs(outs, [P - 1])  # adjoint tape after b2 from dS_out
        self.walk_cross_rest = segs(
            outs, range(P - 1)
        )  # adjoint tape after b2 from the adjoint carries
        self.walk_in = segs(ins, [0]) if P == 1 else self.walk_all  # K2 offset walk
        self.walk_m_b1 = (
            segs([f for f in ins if f.crossing_out], [0]) if P == 1 else self.walk_all
        )  # K3 M-only (b1)
        # fused_walks: one launch per tape phase -- every live segment after wait(c2) / wait(b2) (schedule
        # wait_first), or the crossing-in (crossing-out) fragments' segments when the document-start (document-last)
        # walk stays under the collective (schedule overlap)
        self.walk_live = segs(starts + ins, range(P))
        self.walk_in_fused = segs(ins, range(P))
        self.walk_cross_fused = segs(outs, range(P))
        # walk geometry "lpt_chains": the offset walks too (K1 / K3 / K2 _offset, the P > 1 staging and re-walk lists) in LPT
        # order; the tape lists above already are.  The order is the CTA issue order of a launch only -- never a bit.
        if self.walk_geometry in WALK_LPT_GEOMETRIES:

            def lpt(lst):
                return sorted(
                    (int(s) for s in lst),
                    key=lambda s: (-(seg_table[s + 1] - seg_table[s]), s),
                )

            self.walk_offset, self.walk_all = lpt(self.walk_offset), lpt(self.walk_all)
            self.walk_in, self.walk_m_b1 = lpt(self.walk_in), lpt(self.walk_m_b1)
        # -- per-segment slot tables
        self.h0_index = tuple(int(i) for i in hp.h0_index)
        self.carry_index = tuple(int(i) for i in hp.carry_index)
        self.dht_index = tuple(int(i) for i in hp.dht_index)
        self.dcarry_index = tuple(int(i) for i in hp.dcarry_index)
        self.h0_index_fused = tuple(
            int(i) for i in hp.h0_index_fused
        )  # fused walks: the carry plane (P > 1) / S_in (P = 1)
        self.dht_index_fused = tuple(
            int(i) for i in hp.dht_index_fused
        )  # fused walks: the carry plane (P > 1) / dS_out (P = 1)
        self.out_index = tuple(int(i) for i in hp.out_index)
        self.in_index = tuple(
            in_slot[s // P] if (s % P == 0 and frags[s // P].crossing_in) else -1
            for s in range(NSEG)
        )
        self.m_b1_index = tuple(
            in_slot[s // P]
            if (s % P == 0 and frags[s // P].crossing_in and frags[s // P].crossing_out)
            else -1
            for s in range(NSEG)
        )
        self.seg_index = tuple(range(NSEG))
        # -- tile tables [slot][h]
        self.c1_b_tiles = tuple(tuple(2 * int(t) for t in row) for row in hp.out_tile)
        self.c1_m_tiles = tuple(
            tuple(2 * int(t) + 1 for t in row) for row in hp.out_tile
        )
        self.s_in_tiles = tuple(tuple(int(t) for t in row) for row in hp.h0_tile)
        self.ds_out_tiles = tuple(tuple(int(t) for t in row) for row in hp.dht_tile)
        self.b1_d_tiles = {
            "d": tuple(tuple(int(t) for t in row) for row in hp.b1_d_tile_d),
            "dm": tuple(tuple(int(t) for t in row) for row in hp.b1_d_tile_dm),
        }
        self.b1_m_tiles = tuple(tuple(int(t) for t in row) for row in hp.b1_m_tile_dm)
        self.stage_tiles = tuple(
            tuple(s * H + h for h in range(H)) for s in range(NSEG)
        )
        c1_tiles = int(hp.numel["c1_send"]) // TILE
        self.mf_tiles = (
            tuple(
                tuple(c1_tiles + o * H + h for h in range(H)) for o in range(self.n_out)
            )
            if P > 1
            else ()
        )
        n_carry = self.n_frag * (P - 1) if P > 1 else 0
        if (
            self.fused_walks and P > 1
        ):  # fused walks: + one slot per crossing fragment for the S_in / dS_out copies
            n_carry += max(self.n_in, self.n_out)
        self.carry_tiles = tuple(
            tuple(c * H + h for h in range(H)) for c in range(n_carry)
        )
        # -- splits (elements) and pool sizes (elements)
        self.splits = {k: tuple(int(x) for x in v) for k, v in hp.splits.items()}
        numel = {k: int(v) for k, v in hp.numel.items()}
        self.numel = numel
        # the plan module's pool sizes (the byte queries read the same function)
        self.sizes = cake_kcp_plan.kcp_pool_numel(
            ctx, H=H, fused_walks=self.fused_walks
        )
        assert self.sizes["carry"] == n_carry * H * TILE
        self._check_tiles()
        self._build_chains(starts, ins, outs, ends, in_slot, out_slot)

    # K4 chain families: the pools their offsets index, the operator orientation and the state publication.
    FAMILIES: dict[str, dict[str, Any]] = {
        "fwd": {
            "m": "c1_recv",
            "c": "c1_recv",
            "x0": "c1_recv",
            "y": "c2_send",
            "tm": 0,
            "ws": 1,
        },
        "adj_b1": {
            "m": "b1_recv",
            "c": "b1_recv",
            "x0": "b1_recv",
            "y": "b2_send",
            "tm": 0,
            "ws": 1,
        },
        "adj_c1": {
            "m": "c1_recv",
            "c": "b1_recv",
            "x0": "b1_recv",
            "y": "b2_send",
            "tm": 1,
            "ws": 1,
        },
        "map": {
            "m": "stage_m",
            "c": "stage_b",
            "x0": "eye",
            "y": "c1_send",
            "tm": 0,
            "ws": 0,
        },
        "carry_start": {
            "m": "stage_m",
            "c": "stage_b",
            "x0": None,
            "y": "carry",
            "tm": 0,
            "ws": 1,
        },
        "carry_in": {
            "m": "stage_m",
            "c": "stage_b",
            "x0": "S_in",
            "y": "carry",
            "tm": 0,
            "ws": 1,
        },
        "tmap_d_d": {
            "m": "stage_m",
            "c": "stage_b",
            "x0": None,
            "y": "b1_send",
            "tm": 1,
            "ws": 0,
        },
        "tmap_d_dm": {
            "m": "stage_m",
            "c": "stage_b",
            "x0": None,
            "y": "b1_send",
            "tm": 1,
            "ws": 0,
        },
        "tmap_m": {
            "m": "stage_m",
            "c": None,
            "x0": "eye",
            "y": "b1_send",
            "tm": 0,
            "ws": 0,
        },
        "dcarry_end": {
            "m": "stage_m",
            "c": "stage_b",
            "x0": None,
            "y": "carry",
            "tm": 1,
            "ws": 1,
        },
        "dcarry_cross": {
            "m": "stage_m",
            "c": "stage_b",
            "x0": "b2_recv",
            "y": "carry",
            "tm": 1,
            "ws": 1,
        },
        # merge_fuse="carry": the two carry families of a phase as one launch (chains of carry_start then
        # carry_in; dcarry_end then dcarry_cross); the start / end chains read no x0 (x0_index -1), the others S_in / dS_out
        "carry_both": {
            "m": "stage_m",
            "c": "stage_b",
            "x0": "S_in",
            "y": "carry",
            "tm": 0,
            "ws": 1,
        },
        "dcarry_both": {
            "m": "stage_m",
            "c": "stage_b",
            "x0": "b2_recv",
            "y": "carry",
            "tm": 1,
            "ws": 1,
        },
    }
    # the b1 pool bound of a family depends on the layout form it addresses
    _B1_BOUND = {
        "tmap_d_d": "b1_send_d",
        "tmap_d_dm": "b1_send_dm",
        "tmap_m": "b1_send_dm",
        "adj_b1": "b1_recv_dm",
        "adj_c1": "b1_recv_d",
    }

    def _check_tiles(self) -> None:
        def bound(name: str, tiles, limit_elems: int) -> None:
            top = max((t for row in tiles for t in row), default=-1)
            if (top + 1) * TILE > limit_elems:
                raise ValueError(
                    f"kcp tables: {name} addresses tile {top} beyond its pool of {limit_elems // TILE} tiles"
                )

        n = self.numel
        bound("c1 B tiles", self.c1_b_tiles, n["c1_send"])
        bound("c1 M tiles", self.c1_m_tiles, n["c1_send"])
        bound("S_in tiles", self.s_in_tiles, n["c2_recv"])
        bound("dS_out tiles", self.ds_out_tiles, n["b2_recv"])
        bound("b1 d tiles (d form)", self.b1_d_tiles["d"], n["b1_send_d"])
        bound("b1 d tiles (dm form)", self.b1_d_tiles["dm"], n["b1_send_dm"])
        bound("b1 M tiles", self.b1_m_tiles, n["b1_send_dm"])
        # the fused-walk tables address the fused carry plane (n_frag (P - 1) + max(n_in, n_out) slots; allocated only when
        # fused_walks is set, inert otherwise) at P > 1 and the S_in / dS_out pools at P = 1
        n_fused = self.n_frag * (self.P - 1) + max(self.n_in, self.n_out)
        fused_pool = (
            (n_fused, n_fused)
            if self.P > 1
            else (len(self.s_in_tiles), len(self.ds_out_tiles))
        )
        for name, idx, n_slots in (
            ("h0_index", self.h0_index, len(self.s_in_tiles)),
            ("dht_index", self.dht_index, len(self.ds_out_tiles)),
            ("carry_index", self.carry_index, len(self.carry_tiles)),
            ("dcarry_index", self.dcarry_index, len(self.carry_tiles)),
            ("in_index", self.in_index, len(self.b1_d_tiles["d"])),
            ("m_b1_index", self.m_b1_index, len(self.b1_m_tiles)),
            ("h0_index_fused", self.h0_index_fused, fused_pool[0]),
            ("dht_index_fused", self.dht_index_fused, fused_pool[1]),
        ):
            if len(idx) != self.NSEG or any(i >= n_slots for i in idx):
                raise ValueError(
                    f"kcp tables: {name} is not a valid [NSEG] slot table of its pool"
                )
        if self.P == 1 and any(i >= len(self.c1_b_tiles) for i in self.out_index):
            raise ValueError("kcp tables: out_index exceeds the c1 slots")
        if self.P > 1 and (
            any(i >= self.NSEG for i in self.out_index)
            or len(self.walk_offset) != self.NSEG
        ):
            raise ValueError(
                "kcp tables: at P > 1 the offset walk must visit every segment into its staging slot"
            )

    def _build_chains(self, starts, ins, outs, ends, in_slot, out_slot) -> None:
        H, P = self.H, self.P
        st = self.stage_tiles
        ch: dict[str, Chains] = {name: Chains() for name in self.FAMILIES}
        ch["fwd"] = self._owner_fwd()
        ch["adj_b1"] = self._owner_adj("adj_d_off_dm", "adj_m_off_b1")
        ch["adj_c1"] = self._owner_adj("adj_d_off_d", "adj_m_off_c1")
        if P > 1:

            def mt(s: int, h: int) -> int:
                return st[s][h] * TILE

            def carry(f, i: int, h: int) -> int:
                return self.carry_tiles[f.slot * (P - 1) + i][h] * TILE

            for f in outs:
                S = [f.slot * P + p for p in range(P)]
                for h in range(H):
                    ch["map"].add(
                        [(mt(s, h), mt(s, h), -1) for s in S],
                        y_final=self.c1_b_tiles[out_slot[f.slot]][h] * TILE,
                    )
                    ch["map"].add(
                        [(mt(s, h), -1, -1) for s in S],
                        x0=0,
                        y_final=self.mf_tiles[out_slot[f.slot]][h] * TILE,
                    )
            extra = (
                self.n_frag * (P - 1)
            )  # fused walks: the extra carry slots hold the S_in (dS_out) copies of the fused form

            def copy_slot(
                f, h: int, slot: int
            ) -> int:  # elem_y of the first element = publish the entering state
                return (
                    self.carry_tiles[extra + slot][h] * TILE if self.fused_walks else -1
                )

            for name, fs in (("carry_start", starts), ("carry_in", ins)):
                for f in fs:
                    for h in range(H):
                        elems = [
                            (
                                mt(f.slot * P + p, h),
                                mt(f.slot * P + p, h),
                                carry(f, p - 1, h) if p >= 1 else -1,
                            )
                            for p in range(P - 1)
                        ]
                        x0 = (
                            self.s_in_tiles[in_slot[f.slot]][h] * TILE
                            if name == "carry_in"
                            else -1
                        )
                        if name == "carry_in":
                            elems[0] = (
                                elems[0][0],
                                elems[0][1],
                                copy_slot(f, h, in_slot[f.slot]),
                            )
                        ch[name].add(elems, x0=x0, y_final=carry(f, P - 2, h))
            for form in ("d", "dm"):
                for f in ins:
                    S = [f.slot * P + p for p in range(P)]
                    for h in range(H):
                        ch[f"tmap_d_{form}"].add(
                            [(mt(s, h), mt(s, h), -1) for s in reversed(S)],
                            y_final=self.b1_d_tiles[form][in_slot[f.slot]][h] * TILE,
                        )
            for f in ins:
                if f.crossing_out:
                    S = [f.slot * P + p for p in range(P)]
                    for h in range(H):
                        ch["tmap_m"].add(
                            [(mt(s, h), -1, -1) for s in S],
                            x0=0,
                            y_final=self.b1_m_tiles[in_slot[f.slot]][h] * TILE,
                        )
            for name, fs in (("dcarry_end", ends), ("dcarry_cross", outs)):
                for f in fs:
                    for h in range(H):
                        elems = [
                            (
                                mt(f.slot * P + p, h),
                                mt(f.slot * P + p, h),
                                carry(f, p, h) if p <= P - 2 else -1,
                            )
                            for p in range(P - 1, 0, -1)
                        ]
                        x0 = (
                            self.ds_out_tiles[out_slot[f.slot]][h] * TILE
                            if name == "dcarry_cross"
                            else -1
                        )
                        if name == "dcarry_cross":
                            elems[0] = (
                                elems[0][0],
                                elems[0][1],
                                copy_slot(f, h, out_slot[f.slot]),
                            )
                        ch[name].add(elems, x0=x0, y_final=carry(f, 0, h))
            # merge_fuse="carry": the two carry families of a phase as one launch
            ch["carry_both"] = Chains.concat(ch["carry_start"], ch["carry_in"])
            ch["dcarry_both"] = Chains.concat(ch["dcarry_end"], ch["dcarry_cross"])
        for name, c in ch.items():
            c.check_alignment()
            self._check_chain(name, c)
        self.chains = ch

    def _check_chain(self, name: str, c: Chains) -> None:
        spec = self.FAMILIES[name]
        limits = {
            "c1_send": self.sizes["c1_send"],
            "c1_recv": self.numel["c1_recv"],
            "c2_send": self.numel["c2_send"],
            "S_in": self.numel["c2_recv"],
            "b2_send": self.numel["b2_send"],
            "b2_recv": self.numel["b2_recv"],
            "stage_b": self.sizes["stage_b"],
            "stage_m": self.sizes["stage_m"],
            "carry": self.sizes["carry"],
            "eye": TILE,
        }
        b1 = self._B1_BOUND.get(name)
        if b1 is not None:
            limits["b1_send"] = limits["b1_recv"] = self.numel[b1]
        for key, values in (
            ("m", c.elem_m),
            ("c", c.elem_c),
            ("x0", c.x0_index),
            ("y", c.elem_y + c.y_final),
        ):
            pool = spec[key]
            top = max((v for v in values if v >= 0), default=-1)
            if top < 0:
                continue
            if pool is None or top + TILE > limits[pool]:
                raise ValueError(
                    f"kcp tables: chain family {name}: {key} offset {top} beyond pool {pool}"
                )

    def _owner_fwd(self) -> Chains:
        """The owner's forward chains (the design note §3.5): per document and owned head, from ``X0 = B_{f1}`` (the
        start fragment's offset: ``S_in(f2)``) over the fragments ``f2 .. f(n-1)`` with ``Mop = M_f`` (c1 ``M_f^T``
        tiles, ``transpose_m = 0``); the state entering each element and the final state are the ``S_in`` of the
        element's fragment and of ``fn``, published into the c2 send layout."""
        hp, ch = self.hp, Chains()
        sm, sb, so = (
            self.hg_stride["fwd_m_off"],
            self.hg_stride["fwd_b_off"],
            self.hg_stride["fwd_out_off"],
        )
        starts = [int(x) for x in hp.fwd_chain_start]
        for ci in range(len(starts) - 1):
            e0, e1 = starts[ci], starts[ci + 1]
            for hg in range(self.Hg_r):
                elems = [
                    (
                        int(hp.fwd_m_off[e]) + hg * sm,
                        int(hp.fwd_b_off[e]) + hg * sb,
                        int(hp.fwd_out_off[e - 1]) + hg * so,
                    )
                    for e in range(e0 + 1, e1)
                ]
                ch.add(
                    elems,
                    x0=int(hp.fwd_b_off[e0]) + hg * sb,
                    y_final=int(hp.fwd_out_off[e1 - 1]) + hg * so,
                )
        return ch

    def _owner_adj(self, d_key: str, m_key: str) -> Chains:
        """The owner's adjoint chains (the design note §6.5): per document and owned head, from ``X0 = d_{fn}`` (the
        last fragment's adjoint offset: ``dS_out(f(n-1))``) over ``f(n-1) .. f2`` with ``Mop = M_f^T``; the state
        entering each element and the final state are the ``dS_out`` of the element's predecessor, published into the
        b2 send layout.  ``d_key`` / ``m_key`` name the plan module's offset tables (``adj_d_off_dm`` +
        ``adj_m_off_b1``: the b1 pairs ``[d_f | M_f]``; ``adj_d_off_d`` + ``adj_m_off_c1``: the b1 ``d_f`` plane and the
        c1 ``M_f^T`` tiles); the head strides come from ``hp.hg_stride``."""
        hp, ch = self.hp, Chains()
        d_off, m_off = getattr(hp, d_key), getattr(hp, m_key)
        sd, sm, so = (
            self.hg_stride[d_key],
            self.hg_stride[m_key],
            self.hg_stride["adj_out_off"],
        )
        starts = [int(x) for x in hp.adj_chain_start]
        for ci in range(len(starts) - 1):
            e0, e1 = starts[ci], starts[ci + 1]
            for hg in range(self.Hg_r):
                elems = [
                    (
                        int(m_off[e]) + hg * sm,
                        int(d_off[e]) + hg * sd,
                        int(hp.adj_out_off[e - 1]) + hg * so,
                    )
                    for e in range(e0 + 1, e1)
                ]
                ch.add(
                    elems,
                    x0=int(d_off[e0]) + hg * sd,
                    y_final=int(hp.adj_out_off[e1 - 1]) + hg * so,
                )
        return ch

    def segment_tokens(self, seg: int) -> tuple[int, int]:
        """``[t0, t1)`` local token range of segment ``seg`` (``t0 == t1``: no chunk), from the chunk tables."""
        scs = self.hp.seg_chunk_start
        c0, c1 = int(scs[seg]), int(scs[seg + 1])
        if c1 <= c0:
            return 0, 0
        t0 = int(self.ctx.chunk_bos[c0])
        t1 = int(self.ctx.chunk_bos[c1 - 1]) + int(self.ctx.chunk_len[c1 - 1])
        return t0, t1

    def split_elems(self, coll: str) -> tuple[list[int], list[int]]:
        """``(input_split_sizes, output_split_sizes)`` in elements of ``c1``, ``c2``, ``b1_d``, ``b1_dm`` or ``b2``."""
        if coll.startswith("b1_"):
            form = coll[3:]
            return list(self.splits[f"b1_send_{form}"]), list(
                self.splits[f"b1_recv_{form}"]
            )
        return list(self.splits[f"{coll}_send"]), list(self.splits[f"{coll}_recv"])

    def device_tiles(
        self, tiles, device: torch.device, *, by_identity: bool = False
    ) -> torch.Tensor:
        """The flat int64 device copy of a tile table ``[slot][h]`` (cached per device).  ``by_identity``: the
        cache key is the table object itself (one of this table set's tuples, pinned by the entry) instead of its
        content -- no per-call conversion of the nested rows."""
        if by_identity:
            cache = self._id_cache("_device_tiles_id")
            id_key = (id(tiles), str(device))
            hit = cache.get(id_key)
            if hit is not None and hit[0] is tiles:
                return hit[1]
            t = self.device_tiles(tiles, device)
            cache[id_key] = (tiles, t)
            return t
        key = (tuple(tuple(int(t) for t in row) for row in tiles), str(device))
        cache = self.__dict__.setdefault("_device_tiles", {})
        t = cache.get(key)
        if t is None:
            flat = [int(x) for row in tiles for x in row]
            t = self.upload(flat, torch.int64, device)
            cache[key] = t
        return t

    def upload_specs(
        self, specs: dict[str, tuple], device: torch.device | str
    ) -> dict[str, torch.Tensor]:
        """Device copies of tables of this set (``name -> (values, dtype, shape)``) in one pinned non-blocking copy
        on the current stream, recorded on the context for its ``wait_ready`` (the context's ``upload="blocking"``
        knob keeps the previous pageable per-table copies).  Never a host synchronization: a table-set miss of a
        call costs one asynchronous copy."""
        up = getattr(self.ctx, "upload_tables", None)
        if up is None:  # a duck-typed context: the same copy, unrecorded
            out = cake_kcp_plan._upload_tables(specs, device)[0]
        else:
            # recorded for ctx.wait_ready only when this set lives as long as the context (its own table-set cache);
            # a set of the table_cache="workspace" knob may be rebuilt per call and must not accumulate records
            out = up(specs, device, record=self.on_context)
        if not self.on_context:  # held by a workspace: its stream changes record them (WorkspaceDetKcp._live_storages)
            self.__dict__.setdefault("_uploaded", weakref.WeakSet()).update(
                out.values()
            )
        return out

    def upload(
        self, values: Sequence[int], dtype: torch.dtype, device: torch.device | str
    ) -> torch.Tensor:
        """One flat table of this set on the device (:meth:`upload_specs`)."""
        return self.upload_specs({"t": (values, dtype, None)}, device)["t"]

    def absent_index(self, device: torch.device | str) -> torch.Tensor:
        """Trimmed host path: the slot table of an absent pool (``NSEG`` entries of ``-1``), one device copy per
        table set and device, used when ``own_absent`` (the untrimmed path and the test knobs of the previous path
        keep the process cache of ``cake_kcp_jit``)."""
        lst = self.__dict__.get("_absent")
        if lst is None:
            lst = self._absent = [-1] * self.NSEG
        return self.dev_list(lst, device)

    @functools.cached_property
    def walk_all_p0(self) -> list[int]:
        """Trimmed host path: the document-first segments of ``walk_all`` (``s % P == 0``) as one list object per table set: the
        trimmed backward passes it (not a per-call comprehension) to the identity-keyed device-table cache."""
        return [s for s in self.walk_all if s % self.P == 0]

    @functools.cached_property
    def walk_all_rest(self) -> list[int]:
        """Trimmed host path: the other segments of ``walk_all`` (``s % P != 0``), one list object per table set (see ``walk_all_p0``)."""
        return [s for s in self.walk_all if s % self.P != 0]

    ID_CACHE_MAX = 256  # entries per identity-keyed cache; a caller passing fresh objects cannot grow one without bound

    def _id_cache(self, name: str) -> dict:
        """Trimmed host path: one identity-keyed cache of this table set (``dev_list`` / ``indexed_meta`` / ``device_tiles``), emptied
        when it holds ID_CACHE_MAX entries (the table set's own objects number far fewer; the bound only limits a misuse)."""
        cache = self.__dict__.setdefault(name, {})
        if len(cache) >= self.ID_CACHE_MAX:
            cache.clear()
        return cache

    def dev_list(
        self,
        values: Sequence[int],
        device: torch.device | str,
        dtype: torch.dtype = torch.int32,
    ) -> torch.Tensor:
        """Trimmed host path: the device copy of one of this table set's host int lists (a walk list, ``hp.seg_chunk_start``), cached
        by the list object's identity (the entry pins the list; the table set's lists are never mutated).  A list object
        not owned by the table set misses on every call (a device copy each time): callers pass the set's own lists."""
        cache = self._id_cache("_dev_lists")
        key = (id(values), str(device), dtype)
        hit = cache.get(key)
        if hit is not None and hit[0] is values:
            return hit[1]
        t = self.upload([int(v) for v in values], dtype, device)
        cache[key] = (values, t)
        return t

    def indexed_meta(
        self,
        index: Sequence[int],
        tiles: Sequence[Sequence[int]],
        H: int,
        device: torch.device | str,
    ) -> _IndexedMeta:
        """Trimmed host path: the per-``(index, tiles, H, device)`` part of an :class:`Indexed` pool view -- the integer tuples, the
        two-integer addressing and the int32 device index table -- computed once per table set (the lists are this
        set's; the entry pins them).  ``CallDetKcp.indexed`` builds its ``Indexed`` from it on the trimmed host path."""
        cache = self._id_cache("_indexed_meta")
        key = (id(index), id(tiles), int(H), str(device))
        hit = cache.get(key)
        if hit is not None and hit[0] is index and hit[1] is tiles:
            return hit[2]
        index_t = tuple(int(i) for i in index)
        tiles_t = tuple(tuple(int(t) for t in row) for row in tiles)
        addr = Indexed(torch.empty(0), index_t, tiles_t).addressing(int(H))
        meta = _IndexedMeta(
            index=index_t,
            tiles=tiles_t,
            H=int(H),
            addr=addr,
            device=str(device),
            dev_index=self.upload(list(index_t), torch.int32, device),
        )
        cache[key] = (index, tiles, meta)
        return meta

    def walk_form(
        self,
        name: str,
        walk: Sequence[int],
        H: int,
        sm_count: int,
        knob: str,
        rule: Any,
    ) -> str:
        """The record form (``"full"`` / ``"split"``) of walk launch ``name`` over the segments ``walk``: the registry's
        rule ``rule`` (``cake_kcp_jit.walk_form``) on the launch's chains (the chunk counts of its listed segments), the
        head count ``H``, the device's SM count and the ``walk_form`` knob -- evaluated once per ``(name, walk list, H,
        SM count, knob, rule)`` of this table set and looked up on every later launch.  The rule's inputs are static per
        ``(ctx, H)`` (the walk lists and ``hp.seg_chunk_start`` are this set's, never mutated), so the lookup returns
        what a per-launch evaluation would; the evaluation itself (the ``fit`` rule list-schedules the chains in
        Python) leaves the launch path.  Keyed by value: a fresh equal list finds the same entry.  The first launch
        per key of a context pays that one evaluation (milliseconds of Python for the ``fit`` rule on long chain
        lists: the one-time cost of an eager first call, not a per-launch regression); every later launch pays a
        dictionary lookup."""
        key = (str(name), tuple(walk), int(H), int(sm_count), str(knob), rule)
        cache = self.__dict__.setdefault("_walk_forms", {})
        form = cache.get(key)
        if form is None:
            if len(cache) >= self.ID_CACHE_MAX:
                cache.clear()
            seg = self.hp.seg_chunk_start
            form = cache[key] = str(
                rule(
                    name,
                    [int(seg[s + 1]) - int(seg[s]) for s in walk],
                    int(H),
                    int(sm_count),
                    knob=str(knob),
                )
            )
        return form

    def pool_sizes(self, backward: bool) -> dict[str, int]:
        names = cake_kcp_plan.FWD_POOLS + (cake_kcp_plan.BWD_POOLS if backward else ())
        return {
            n: self.sizes[n] for n in names
        }  # zero-size pools included: every buffer name resolves

    def describe(self) -> dict[str, Any]:
        return {
            "P": self.P,
            "NSEG": self.NSEG,
            "Hg": self.Hg,
            "Hg_r": self.Hg_r,
            "n_frag": self.n_frag,
            "T_r": self.T_r,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "max_hops": self.max_hops,
            "fused_walks": self.fused_walks,
            "walk_geometry": self.walk_geometry,
            "walks": {
                "live": len(self.walk_live),
                "start": len(self.walk_start),
                "in_p0": len(self.walk_in_p0),
                "in_p1": len(self.walk_in_p1),
                "end": len(self.walk_end),
                "cross_last": len(self.walk_cross_last),
                "cross_rest": len(self.walk_cross_rest),
            },
            "chains": {
                k: {"n_chains": c.n_chains, "n_elems": c.n_elems}
                for k, c in self.chains.items()
            },
            "sizes": dict(self.sizes),
        }


# ----------------------------------------------------------------------------------------------------------------
# Workspace (the design note §7.4): transient buffers only; saved-set tensors are fresh per call (SPEC-4)


class WorkspaceDetKcp:
    """The transient buffers of the KCP route on one rank (the design note §7.4) plus the existing backend's local
    arena (``cake_backend.KdaWorkspace``) for the token-side stages.

    Pools (flat, ``state_dtype``): ``c1_send``, ``c1_recv``, ``c2_send``; ``b1_send``, ``b1_recv``, ``b2_send``,
    ``b2_recv`` (``backward``); at ``P > 1`` the staging planes ``stage_b`` / ``stage_m`` and the ``carry`` plane;
    one identity tile ``eye``.  The backward send pools share the storage of the forward ones
    (``cake_kcp_plan.SHARED_POOLS``: ``b1_send`` = ``c1_send``, ``b2_send`` = ``c2_send``, one tensor of the larger
    need; a call waits its c1 / c2 sends before a backward writes either).  The ring exchange's hop pools
    ``ring_out`` / ``ring_in`` (``cake_kcp_plan.LAZY_POOLS``) are allocated by the first call that reads them, and so
    are, on a forward-only workspace, ``c1_recv`` / ``stage_m`` / ``carry`` (``cake_kcp_plan.FWD_LAZY_POOLS``: a
    forward with a saved set writes a fresh carry plane in place of ``carry`` and, under ``save_transfer``, fresh
    tensors in place of the other two, which it reads without it; a forward without one reads all three; their first
    allocation drops the graph-cache entries pinning the workspace, whose keys hold its pools).  A forward-only
    local arena takes the KCP forward layout (:data:`LOCAL_LAYOUT_FWD_ONLY`, 2432 B per internal token-head); a
    backward workspace sizes every pool and the existing backend's layouts (``local_arena="replay"``, the default,
    :data:`cake_kcp_plan.LOCAL_ARENAS`) or, under ``local_arena="lean"`` (opt-in), the KCP forward layout and the
    lean backward layout (:data:`LOCAL_LAYOUT_BWD_LEAN`, 3460 B per internal token-head; the tail stages per chunk
    range) with its tail-count table (:data:`TAIL_COUNTS_ATTR`), the existing backend's layouts on the rows the lean
    layout does not serve (H 9-15).  ``lean_passes`` sizes a lean arena for that many tail passes (``None``:
    :data:`cake_kcp_plan.LEAN_PASSES_DEFAULT`, two; fewer passes launch fewer tail kernels for more bytes;
    :func:`cake_kcp_plan.kda_det_workspace_bytes_kcp` takes the same arguments).  The saved-set tensors (``S_in`` = the c2
    output, the c1 receive tensor under ``save_transfer``, ``M_seg`` under ``save_transfer`` at ``P > 1``) are never
    arena slices: the calls allocate them fresh (SPEC-4).  A workspace is
    reusable across calls and layers of the same ``(ctx, H)``; a call with a larger need grows it, releasing each
    superseded pool and local arena (and the graph-cache entries pinning that arena) before allocating its successor
    (the old and the new storage never coexist, so a backward on a forward workspace peaks at the grown workspace plus
    the call's own peak).  ``bytes`` counts every pool storage once plus the local arena (:func:`kda_det_workspace_bytes_kcp`
    is the plan module's prediction of the same quantity for an a2a workspace, CONSISTENCY-6).

    One workspace serves any number of prepared packs (contexts) whose calls run in stream order: the pools hold
    the largest need of the packs served and the local arena the largest of their layouts; a pack whose layout fits
    the arena runs on a view of it under its own plan (:func:`_plan_view`, at most :data:`VIEW_CACHE_MAX` cached per
    arena; each view has its own launch templates and its own scratch table set where the intra record writes one,
    :func:`cake_kcp_plan.local_scratch_table_bytes`), and the arena is replaced only by a pack that does not fit.
    A forward never replaces a backward arena with a forward-only one.  :meth:`reserve` sizes the workspace for a
    list of contexts up front, so the calls of those packs allocate nothing and every pointer stays put
    (``kda_det_workspace_bytes_kcp([ctx, ...], H=H)`` predicts its ``bytes``).  No content crosses calls, so the results are
    bitwise those of one workspace per pack.  The workspace keeps no context alive.  It records the stream of its
    last call; a call on another stream first orders that stream's queued work before its own (one
    ``wait_stream``, never a host wait) and marks the workspace's live storages as used on its stream (one
    ``record_stream`` each, :meth:`_order_stream`), so a pool or arena a later call replaces, or the storage of a
    dropped workspace, goes back to the caching allocator only after the queued work of every stream that used it;
    calls on one stream do neither.  ``allocate=False`` with a context allocates nothing until the first call
    and keeps ``backward`` as a sizing floor, so a first forward allocates the backward workspace at once instead
    of a forward-only one that its first backward would replace.

    ``use_gate_in_kernel`` fixes the gate mode of the calls the workspace serves (:attr:`gate`: ``"g"`` the activated
    FP32 gate, ``"raw"`` the raw BF16 gate of ``use_gate_in_kernel=True``) for its lifetime: the local arena is
    planned in that mode (``cake_backend.plan(..., gate=)``: the raw records, and on every architecture a backward
    layout without the gate epilogue's ``dg_int`` / ``db_int``); the KCP pools do not depend on it.  A call in the
    other mode is refused with ``ValueError``."""

    def __init__(
        self,
        ctx: Any | None = None,
        H: int | None = None,
        device: torch.device | str | None = None,
        *,
        backward: bool = True,
        state_dtype: torch.dtype = torch.float32,
        local: bool = True,
        save_mode: str = "light",
        allocate: bool = True,
        local_arena: str | None = None,
        lean_passes: int | None = None,
        use_gate_in_kernel: bool = False,
    ) -> None:
        if save_mode not in SAVE_MODES:
            raise ValueError(
                f"WorkspaceDetKcp: save_mode must be one of {SAVE_MODES}, got {save_mode!r}"
            )
        # the backward arena form (cake_kcp_plan.LOCAL_ARENAS): "replay" (the default) or the opt-in "lean"
        self.local_arena = cake_kcp_plan.check_local_arena(local_arena)
        # the tail passes a lean arena is sized for (None: cake_kcp_plan.LEAN_PASSES_DEFAULT); fewer passes launch
        # fewer tail kernels for more bytes
        self.lean_passes = cake_kcp_plan.check_lean_passes(lean_passes)
        if self.lean_passes is not None and self.local_arena != "lean":
            raise ValueError(
                'WorkspaceDetKcp: lean_passes needs local_arena="lean", got '
                f"{self.local_arena!r}"
            )
        # the gate mode of every call on this workspace (and of its local arena's plan)
        self.gate = cake_backend._gate_mode(use_gate_in_kernel)
        if self.gate == "raw" and self.local_arena == "lean":
            raise NotImplementedError(
                'WorkspaceDetKcp: use_gate_in_kernel=True runs on the replay local arena; local_arena="lean" '
                "is not supported in the raw gate mode"
            )
        self.state_dtype = state_dtype
        self.with_local = bool(local)
        self.save_mode = save_mode
        self.device: torch.device | None = None
        self._pools: dict[str, torch.Tensor] = {}
        self._tables: dict[tuple[Any, int, int, str], _Tables] = {}
        self.local: Any | None = None
        # the arena (or its plan view) the last ensured context runs on; None on a rank without tokens
        self.local_view: Any | None = None
        # plan views of the arena per (S, num_docs, H), least recently bound first
        self._views: dict[tuple[int, int, int], Any] = {}
        self._views_scratch = 0  # local_scratch_table_bytes over the cached views
        # a forward sizes the workspace for the backward too (allocate=False with a context, reserve())
        self._backward_floor = False
        self._stream: tuple[int, Any] | None = (
            None  # (raw handle, stream) of the last call
        )
        # the trimmed path's ensure records per (id(ctx), (H, device, backward, fused_walks, walk_geometry)):
        # (weak ctx, weak tables, local view); dropped whenever a pool or the arena is replaced or a view evicted
        self._fast: dict[tuple[Any, ...], tuple[Any, Any, Any]] = {}
        self._slices: dict[
            tuple[str, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}  # pool[:n] views per (name, n), at most SLICE_CACHE_MAX
        if ctx is not None:
            if H is None:
                raise ValueError("WorkspaceDetKcp: H is required with ctx")
            if allocate:
                self.ensure(ctx, H, device=device, backward=backward)
            else:
                self._backward_floor = bool(backward)

    def tables(
        self,
        ctx: Any,
        H: int,
        *,
        fused_walks: int = FUSED_WALKS_DEFAULT,
        walk_geometry: str = WALK_GEOMETRY_DEFAULT,
    ) -> _Tables:
        key: tuple = (int(H), int(fused_walks), str(walk_geometry))
        sets = getattr(ctx, "_table_sets", None)
        if (
            sets is None
            or getattr(ctx, "table_cache", cake_kcp_plan.TABLE_CACHE_DEFAULT)
            != "context"
        ):  # a duck-typed context, or the table_cache="workspace" knob: the previous per-workspace cache
            sets, key = self._tables, (ctx, *key)
        tb = sets.get(key)
        if tb is None:
            tb = _Tables(ctx, H, fused_walks=fused_walks, walk_geometry=walk_geometry)
            tb.on_context = sets is not self._tables
            # the absent-pool slot table: the set's own copy on the shipped path; the upload="blocking" /
            # table_cache="workspace" knobs keep the previous process cache of cake_kcp_jit
            tb.own_absent = (
                getattr(ctx, "upload", cake_kcp_plan.UPLOAD_DEFAULT) == "async"
                and getattr(ctx, "table_cache", cake_kcp_plan.TABLE_CACHE_DEFAULT)
                == "context"
            )
            sets[key] = tb
        return tb

    def ensure(
        self,
        ctx: Any,
        H: int,
        *,
        device: torch.device | str | None = None,
        backward: bool = True,
        fused_walks: int = FUSED_WALKS_DEFAULT,
        walk_geometry: str = WALK_GEOMETRY_DEFAULT,
        host_trim: bool = False,
    ) -> _Tables:
        """Size the pools (and the local arena) for ``(ctx, H)``; returns the tables.  ``host_trim``: a call
        with the same ``(ctx, H, device, backward, fused_walks, walk_geometry)`` as an earlier trimmed ensure returns
        its tables and local view at once while no pool or arena was replaced since (the pools only grow and the
        arena is replaced only by a pack that does not fit, so nothing is to do; one record per context and
        signature, at most :data:`ENSURE_CACHE_MAX`, the contexts and tables held weakly).  Every ensure orders the
        call's stream after the workspace's last one (:meth:`_order_stream`) before it sizes anything."""
        dev: torch.device = (
            torch.device(device) if device is not None else torch.device(ctx.device)
        )
        backward = bool(backward) or self._backward_floor
        sig = (int(H), dev, backward, int(fused_walks), str(walk_geometry))
        if host_trim:
            hit = self._fast.get((id(ctx), sig))
            if hit is not None and hit[0]() is ctx:
                tb = hit[1]()
                if tb is not None:
                    self.local_view = hit[2]
                    self._order_stream(dev)
                    return tb
        # a record is added only once this ensure succeeds, and every pool or arena replacement drops them all
        # first: a failed ensure (an allocation error after a release) never leaves the fast path approving a
        # workspace it has already taken apart
        self._check_device(dev)
        tb = self.tables(ctx, H, fused_walks=fused_walks, walk_geometry=walk_geometry)
        # ordered (and the live storages recorded) before a pool or the arena can be released
        self._order_stream(dev)
        self._size_pools(self._pool_need(tb, backward), dev)
        self.local_view = None
        if self.with_local and ctx.T_r > 0 and dev.type == "cuda":
            self.local_view = self._bind_local(ctx, int(H), dev, backward)
        if host_trim:
            if len(self._fast) >= ENSURE_CACHE_MAX:
                self._fast.clear()
            self._fast[(id(ctx), sig)] = (_weak(ctx), _weak(tb), self.local_view)
        return tb

    def reserve(
        self,
        ctxs: Any,
        H: int,
        *,
        device: torch.device | str | None = None,
        backward: bool = True,
        fused_walks: int = FUSED_WALKS_DEFAULT,
        walk_geometry: str = WALK_GEOMETRY_DEFAULT,
        ring: bool = False,
    ) -> int:
        """Size this workspace for every context of ``ctxs`` (one context or a sequence of them) at ``H`` heads: each
        pool to its largest need, the local arena to the largest layout (replaced only when one does not fit) with a
        plan view per other pack and its scratch table set, all allocated now on the current stream.  With
        ``backward`` (the default) a later forward never re-sizes it, so the calls of the reserved packs allocate
        nothing and every pool and view keeps its pointer; ``bytes`` is then
        ``kda_det_workspace_bytes_kcp(ctxs, H=H)`` for a workspace reserved once.  Host work and allocations only (no
        device wait).  ``ring=True`` also sizes the ring exchange's hop pools (:data:`cake_kcp_plan.LAZY_POOLS`,
        otherwise allocated by the first ``exchange="ring"`` call; ``kcp_arena_layout(...)["ring"]`` bytes per
        context, the largest over the contexts).  Returns :attr:`bytes`."""
        ctxs = _context_list(ctxs)
        if not ctxs:
            raise ValueError("WorkspaceDetKcp.reserve: no context given")
        dev: torch.device = torch.device(ctxs[0].device if device is None else device)
        backward = bool(backward) or self._backward_floor
        self._backward_floor = backward
        self._check_device(dev)
        H = int(H)
        need: dict[str, int] = {}
        for ctx in ctxs:
            tb = self.tables(
                ctx, H, fused_walks=fused_walks, walk_geometry=walk_geometry
            )
            sizes = self._pool_need(tb, backward)
            if ring:
                sizes.update(
                    (name, tb.sizes[name]) for name in cake_kcp_plan.LAZY_POOLS
                )
            for name, n in sizes.items():
                need[name] = max(need.get(name, 0), n)
        self._order_stream(dev)
        self._size_pools(need, dev)
        if self.with_local and dev.type == "cuda":
            local = [ctx for ctx in ctxs if ctx.T_r > 0]
            plans = {
                id(ctx): self._local_plan((int(ctx.T_r), int(ctx.n_frag), H), dev)
                for ctx in local
            }
            # the largest layout first: every other pack then runs on a view of its arena
            local.sort(
                key=lambda c: -cake_kcp_plan.local_arena_bytes(
                    plans[id(c)],
                    backward=backward,
                    save_mode=self.save_mode,
                    local_arena=self.local_arena,
                    lean_passes=self.lean_passes,
                )
            )
            for ctx in local:
                view = self._bind_local(ctx, H, dev, backward)
                for path in (LOCAL_PATH_FWD, LOCAL_PATH_BWD):
                    if cake_kcp_plan.intra_writes_tables(view.plan, path):
                        _scratch_tables(view)
                        break
            self.local_view = None
        return self.bytes

    def _check_device(self, device: torch.device) -> None:
        if self.device is None:
            self.device = device
        elif self.device != device:
            raise ValueError(
                f"WorkspaceDetKcp: lives on {self.device}, the call is on {device}"
            )

    @staticmethod
    def _pool_need(tb: _Tables, backward: bool) -> dict[str, int]:
        """The elements of every pool ``tb`` needs on a ``backward`` / forward-only workspace, the lazily
        allocated pools excluded and each :data:`cake_kcp_plan.SHARED_POOLS` pair at the larger need of the two."""
        # allocated by the first call that reads them (pool()): the ring pools; on a forward-only workspace the pools
        # only a forward without a saved set reads
        lazy = cake_kcp_plan.LAZY_POOLS + (
            () if backward else cake_kcp_plan.FWD_LAZY_POOLS
        )
        need = {
            name: n for name, n in tb.pool_sizes(backward).items() if name not in lazy
        }
        for pair in cake_kcp_plan.SHARED_POOLS:  # one storage of the larger need
            top = max((need[m] for m in pair if m in need), default=0)
            for m in pair:
                if m in need:
                    need[m] = top
        return need

    def _size_pools(self, need: dict[str, int], device: torch.device) -> None:
        replaced = False
        for name, n in need.items():
            replaced |= self._grow(name, n, device)
        if replaced and self._slices:
            self._drop_stale_slices()

    def _order_stream(self, device: torch.device) -> None:
        """Record the stream of this call; when it differs from the last call's, order the last stream's queued work
        (which may still use this workspace) before the work this call enqueues (one ``wait_stream``, no host wait)
        and mark every live device storage of the workspace as used on this stream (``record_stream``,
        :meth:`_live_storages`), so a storage released later (a grown pool, a replaced arena, the dropped workspace)
        returns to the caching allocator only once this stream's work queued by then has run, whichever stream
        allocated it.  Called before any release.  Nothing on the same stream (stream order covers it) or while the
        current stream captures a CUDA graph (the capture's own ordering applies, and its allocations live in the
        graph's private pool)."""
        if device.type != "cuda":
            return
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        raw = _raw_stream(index)
        last = self._stream
        if last is not None and last[0] == raw:
            return
        cur = torch.cuda.current_stream(index)
        if not torch.cuda.is_current_stream_capturing():
            if last is not None:
                cur.wait_stream(last[1])
            for t in self._live_storages():
                t.record_stream(cur)
        self._stream = (raw, cur)

    def _live_storages(self) -> list[torch.Tensor]:
        """One tensor per live device storage of this workspace that a call's kernels can touch: each distinct pool
        (a :data:`cake_kcp_plan.SHARED_POOLS` pair once), the local arena's bytes and its identity tile (``eye`` /
        ``flags``, shared by its views), the scratch table buffer of the arena and of each cached plan view (one
        buffer per set, its tables are slices of it), and the device tables of the ``table_cache="workspace"`` knob's
        table sets (the shipped path's table sets live on the context, whose ``wait_ready`` records them per
        stream)."""
        out: dict[int, torch.Tensor] = {}

        def add(t: Any) -> None:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                ptr = t.untyped_storage().data_ptr()
                if ptr:  # an empty storage has no block
                    out.setdefault(ptr, t)

        for t in self._pools.values():
            add(t)
        local = self.local
        if local is not None:
            add(local.arena)
            add(local._eye_box[0])
            add(local.__dict__.get(TAIL_COUNTS_ATTR))
            for arena in (local, *self._views.values()):
                for t in (arena.__dict__.get(SCRATCH_TABLES_ATTR) or {}).values():
                    add(t)
                    break  # every table of the set is a slice of one buffer
        for tb in self._tables.values():
            for t in _table_tensors(tb):
                add(t)
        return list(out.values())

    def _local_plan(self, key: tuple[int, int, int], device: torch.device) -> Any:
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        return cake_backend.plan(
            *key, cake_jit.sm_count(index), cake_jit.device_arch(index), gate=self.gate
        )

    def _bind_local(
        self, ctx: Any, H: int, device: torch.device, backward: bool
    ) -> Any:
        """The local arena, or its plan view, that ``ctx``'s calls run on: the arena's own plan, a cached view, a
        new view of a plan whose layouts fit the arena (of its kind: a backward arena serves forwards too), or a new
        arena -- of the backward kind when the call, the sizing floor or the superseded arena is one -- after the
        superseded arena, its views and the graph-cache entries pinning them are released (the two never
        coexist)."""
        key = (int(ctx.T_r), int(ctx.n_frag), int(H))
        base = self.local
        if base is not None and (base.backward or not backward):
            p = base.plan
            if key == (p.S, p.N, p.H):
                return base
            view = self._views.pop(key, None)
            if view is None:
                plan = self._local_plan(key, device)
                passes = self._lean_view_passes(plan, base)
                if (
                    passes is not None
                    or plan.arena_bytes(
                        backward=base.backward,
                        save_mode=self.save_mode,
                        fwd_layout=cake_kcp_plan.local_fwd_layout(
                            base.backward, self.save_mode
                        ),
                    )
                    <= base.bytes
                ):
                    view = _plan_view(base, plan)
                    # the lean layout at the fewest passes that fit the arena, else the replay layout (a dict of
                    # its own: the view's attributes start as the base arena's)
                    view.layout_passes = (
                        {} if passes is None else {LOCAL_LAYOUT_BWD_LEAN: passes}
                    )
                    self._views_scratch += cake_kcp_plan.local_scratch_table_bytes(plan)
                    while len(self._views) >= VIEW_CACHE_MAX:
                        self._fast.clear()
                        old = self._views.pop(next(iter(self._views)))
                        self._views_scratch -= cake_kcp_plan.local_scratch_table_bytes(
                            old.plan
                        )
            if view is not None:
                self._views[key] = view  # most recently bound last
                return view
        kind = backward or (base is not None and base.backward)
        if base is not None:
            # no contents carry over: release the superseded arena, its views (and the graph-cache entries pinning
            # them) before its successor is allocated, so the two never coexist -- the first training step then
            # peaks at the retained workspace plus the call's own peak, as every later step does (with the SPEC-3
            # probe off, the default)
            self._fast.clear()
            _graph_release(base)
            self._views.clear()
            self._views_scratch = 0
            self.local_view = self.local = base = None
        passes = None
        if kind and self.local_arena == "lean" and self.save_mode == "light":
            passes = cake_kcp_plan.local_lean_passes(
                self._local_plan(key, device), passes=self.lean_passes
            )
        if passes is None:
            self.local = cake_backend.KdaDetWorkspace.for_problem(
                *key,
                device,
                backward=kind,
                save_mode=self.save_mode,
                fwd_layout=cake_kcp_plan.local_fwd_layout(kind, self.save_mode),
                gate=self.gate,
            )
            return self.local
        # the lean arena: exactly the KCP forward layout and the lean backward layout, plus its tail-count table
        # (allocated with the arena, never by a call)
        self.local = cake_backend.KdaDetWorkspace(
            self._local_plan(key, device),
            device,
            backward=True,
            save_mode=self.save_mode,
            fwd_layout=cake_kcp_plan.LOCAL_LAYOUT_FWD_ONLY,
            layouts=(
                (cake_kcp_plan.LOCAL_LAYOUT_FWD_ONLY, 1),
                (LOCAL_LAYOUT_BWD_LEAN, passes),
            ),
        )
        words = cake_kcp_plan.local_tail_count_words(self.local.bytes, H)
        setattr(
            self.local,
            TAIL_COUNTS_ATTR,
            torch.arange(words, dtype=torch.int32, device=self.local.device),
        )
        return self.local

    def _lean_view_passes(self, plan: Any, base: Any) -> int | None:
        """The tail passes of a plan view of the lean arena ``base`` on the lean layout: the fewest whose arena fits
        ``base`` (:func:`cake_kcp_plan.local_lean_passes`) with every range's chunk count inside its tail-count
        table; ``None`` (the replay layout, if it fits) on a replay arena, under ``local_arena="replay"`` or where
        the lean layout does not serve the plan."""
        counts = base.__dict__.get(TAIL_COUNTS_ATTR)
        if (
            counts is None
            or not _lean_passes(base)
            or self.local_arena != "lean"
            or self.save_mode != "light"
        ):
            return None
        passes = cake_kcp_plan.local_lean_passes(plan, budget=base.bytes)
        if passes is None or -(-int(plan.NC_pad) // passes) >= counts.numel():
            return None
        return passes

    def _grow(self, name: str, n: int, device: torch.device) -> bool:
        """Make pool ``name`` hold ``n`` elements; a :data:`cake_kcp_plan.SHARED_POOLS` pair stays one tensor.
        Returns whether a pool tensor was replaced."""
        cur = self._pools.get(name)
        if cur is not None and cur.numel() >= n:
            return False
        mate = next(
            (
                m
                for pair in cake_kcp_plan.SHARED_POOLS
                if name in pair
                for m in pair
                if m != name
            ),
            None,
        )
        mt = self._pools.get(mate) if mate is not None else None
        self._fast.clear()
        if mt is not None and mt.numel() >= n:
            self._pools[name] = mt  # the pair's storage is already large enough
            return True
        # the pools carry no contents across calls: release the superseded storage (these locals, the dict entries and
        # the cached views) before the successor is allocated, so the two never coexist
        shared = mt is not None
        del cur, mt
        for m in (name, mate):
            if m is not None and self._pools.pop(m, None) is not None:
                for key in [k for k in self._slices if k[0] == m]:
                    del self._slices[key]
        t = self._pools[name] = _alloc((n,), self.state_dtype, device)
        if shared:
            self._pools[mate] = t
        if name == "eye":
            t.view(D, D).copy_(torch.eye(D, dtype=self.state_dtype, device=device))
        return True

    def _drop_stale_slices(self) -> None:
        """A superseded pool must not live on in the cached ``[:n]`` views."""
        for key in [
            k for k, (t, _) in self._slices.items() if self._pools.get(k[0]) is not t
        ]:
            del self._slices[key]

    def pool(
        self, name: str, n: int | None = None, *, cached: bool = False
    ) -> torch.Tensor:
        # the graph-cache keys hold the forward-only lazy pools (not the ring's): once one is allocated or replaced, no
        # later call matches an earlier key of this workspace, so its entries (and their captured memory) go
        fwd_lazy = name in cake_kcp_plan.FWD_LAZY_POOLS
        if (
            (name in cake_kcp_plan.LAZY_POOLS or fwd_lazy)
            and n
            and self.device is not None
        ):
            if self._grow(name, n, self.device):
                if self._slices:
                    self._drop_stale_slices()
                if fwd_lazy:
                    _graph_release(self)
        t = self._pools.get(name)
        if t is None:
            if n not in (None, 0) or self.device is None:
                raise KeyError(
                    f"WorkspaceDetKcp: pool {name!r} was not sized for this call (ensure() first)"
                )
            t = self._pools[name] = _alloc((0,), self.state_dtype, self.device)
            if fwd_lazy:
                _graph_release(self)
        if n is None:
            return t
        if (
            cached
        ):  # the [:n] view of an unchanged pool tensor is the same view call after call
            hit = self._slices.get((name, n))
            if hit is not None and hit[0] is t:
                return hit[1]
            v = t[:n]
            if len(self._slices) >= SLICE_CACHE_MAX:  # bounded: many packs' sizes
                self._slices.clear()
            self._slices[(name, n)] = (t, v)
            return v
        return t[:n]

    def has(self, name: str) -> bool:
        return name in self._pools

    @property
    def bytes(self) -> int:
        # every storage once (the SHARED_POOLS pairs are one tensor)
        kcp = sum(
            t.numel() * t.element_size()
            for t in {id(t): t for t in self._pools.values()}.values()
        )
        local = 0
        if self.local is not None:
            # the arena and the intra record's scratch table set where it writes one, its own and each plan view's
            counts = self.local.__dict__.get(TAIL_COUNTS_ATTR)
            local = (
                int(self.local.bytes)
                + cake_kcp_plan.local_scratch_table_bytes(self.local.plan)
                + self._views_scratch
                + (0 if counts is None else counts.numel() * counts.element_size())
            )
        return kcp + local

    @property
    def pool_bytes(self) -> dict[str, int]:
        """Bytes per pool name; the two names of a :data:`cake_kcp_plan.SHARED_POOLS` pair report the same
        storage (:attr:`bytes` counts it once)."""
        return {k: t.numel() * t.element_size() for k, t in self._pools.items()}


# ----------------------------------------------------------------------------------------------------------------
# Saved set (the design note §16), options (§10.1), exchange scheme (SPEC-2)


@dataclasses.dataclass(frozen=True)
class SavedDetKcp:
    """The saved context of one forward (``boundary_mode="saved"``, the design note §16): ``S_in`` (the fresh c2
    output), the owner's c1 receive tensor (``save_transfer``), ``M_seg`` (``save_transfer`` at ``P > 1``), the
    forward's carry plane ``carry`` (``P > 1``: the backward's tape replay reads it and skips the K1 offset re-walk +
    carry merges; ``None`` at ``P = 1`` and under the test knob ``saved_carry=0``) and the scalars of the call.  The
    int32 tables are rebuilt from ``ctx`` (droppable, §16 item 3).  No token-row ``Aqk`` / ``Akk``: the backward
    recomputes the internal rows it reads in both modes (version 2; version 1 held them).  ``gate`` (``"g"`` or
    ``"raw"``) and ``lower_bound`` (the raw mode's FP32 lower bound as a host float, ``None`` in ``"g"`` mode) are
    the gate mode of the forward that wrote the set: its tensors and version are the same in both modes, and a
    backward in the other mode or with another lower bound is refused."""

    version: int
    S_in: torch.Tensor
    c1_recv: torch.Tensor | None
    M_seg: torch.Tensor | None
    T_r: int
    H: int
    n_frag: int
    P: int
    Hg: int
    scale: float
    save_mode: str
    save_transfer: bool
    offset_class: str
    exchange: str
    precision_mode: int
    ctx: Any
    transfer_walk: str = "bf16x3_hw64"
    carry: torch.Tensor | None = (
        None  # the forward's carry plane (P > 1; None at P = 1 or under saved_carry=0)
    )
    gate: str = "g"  # the forward's gate mode ("raw": use_gate_in_kernel=True)
    # the raw mode's FP32 lower bound (None in "g" mode)
    lower_bound: float | None = None

    _OPTIONAL = ("c1_recv", "M_seg", "carry")

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.S_in,) + tuple(
            t for t in (getattr(self, n) for n in self._OPTIONAL) if t is not None
        )

    def flags(self) -> tuple[bool, ...]:
        return tuple(getattr(self, n) is not None for n in self._OPTIONAL)

    def meta(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "T_r": self.T_r,
            "H": self.H,
            "n_frag": self.n_frag,
            "P": self.P,
            "Hg": self.Hg,
            "scale": self.scale,
            "save_mode": self.save_mode,
            "save_transfer": self.save_transfer,
            "offset_class": self.offset_class,
            "exchange": self.exchange,
            "precision_mode": self.precision_mode,
            "transfer_walk": self.transfer_walk,
            "gate": self.gate,
            "lower_bound": self.lower_bound,
        }

    @classmethod
    def from_tensors(
        cls,
        tensors: Sequence[torch.Tensor],
        flags: Sequence[bool],
        meta: dict[str, Any],
        ctx: Any,
    ) -> SavedDetKcp:
        it = iter(tensors)
        S_in = next(it)
        opt = {
            n: (next(it) if f else None)
            for n, f in zip(cls._OPTIONAL, flags, strict=True)
        }
        return cls(S_in=S_in, ctx=ctx, **opt, **meta)

    @property
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors())


@dataclasses.dataclass
class OptionsDetKcp:
    """The extra keywords of the entry points (the design note §10.1): ``save_transfer`` (keep the owner's c1
    receive tensor / ``M_seg`` for the backward; ``None`` = the default = resolve: :data:`cake_kcp_plan.SAVE_TRANSFER_DEFAULT`
    = ``True`` since an earlier revision for the a2a exchange, ``False`` for ``exchange="ring"``, whose
    path receives no c1 tensor; an explicit ``False`` = the earlier recompute form, which re-sends
    ``M_f^T`` in b1; an explicit ``True`` with the ring is a ``ValueError`` -- there is no ring form of it),
    ``exchange`` (``auto`` | ``a2a`` | ``ring``; ring only when ``n_d_max - 1 <= 2``, SPEC-2; ``auto`` is ``a2a``
    today), ``head_groups`` (``G``; Phase 1 ships ``1``),
    ``offset_class`` (``fla``: K1 / K2 offsets; ``fp32`` is a Phase-2 knob), ``debug_sequence`` (R7: all-gather the
    collective issue order and assert it), ``spec3_probe`` (the SPEC-3 debug probe of the first saved-mode backward,
    :class:`KernelBackend`: ``"off"`` default, ``"record"``, ``"warn"``, ``"assert"``; pass the same value on every
    rank: the probe is a rank-local host check (a rank without tokens skips it), and ``"assert"`` raises on a rank
    whose rows differ while the other ranks enter the backward's collectives -- it serves single-process debugging;
    the backward reads its own call's value, the forward's is not kept in :class:`SavedDetKcp`),
    ``graph`` (``"off"`` shipped | ``"replay"``: keyed CUDA-graph capture / replay of the
    per-rank call -- see "CUDA-graph replay of a call" below for the validity key, the
    pointer-stability requirement, the private pools' memory, the eager fallback and :func:`graph_cache_clear`),
    ``graph_cap`` (the cache-config form of a captured call: ``"node_attr"`` shipped
    -- the tape one-CTA-per-SM cap as a launch attribute of the captured kernel nodes; ``"none"``
    = the previous form of record (no device preference), ``"capture_pref"`` (rejected) and ``"node_attr_post"`` stay selectable;
    ``_kcp_test_knobs`` only), ``host_trim`` (``1`` shipped: the
    trimmed host path of the eager call -- the same launches in the same order, bitwise; ``0`` = the previous host path;
    ``_kcp_test_knobs`` only), ``saved_carry`` (``1`` shipped: the forward of a ``P > 1`` rank keeps its carry plane in
    the saved set and the saved-mode backward reads it; ``0`` = that backward re-walks the forward carries -- the same
    remaining launches, bitwise; ``_kcp_test_knobs`` only), ``precision_mode`` (K3's static program; set only through
    ``_kcp_test_knobs``)."""

    save_transfer: bool | None = (
        None  # None = resolve: SAVE_TRANSFER_DEFAULT (True) for a2a / auto, False for the ring; explicit False = the recompute form
    )
    exchange: str = "auto"
    head_groups: int = 1
    offset_class: str = "fla"
    debug_sequence: bool = False
    # public keyword: the SPEC-3 probe's mode (SPEC3_PROBES)
    spec3_probe: str = SPEC3_PROBE_DEFAULT
    precision_mode: int = 0
    transfer_walk: str = "bf16x3_hw64"  # K3 program family: the shipped BF16x3 half-width walk; tf32_hilo / bf16x3 only through _kcp_test_knobs
    p_rule: str | None = (
        None  # plan P rule named by _kcp_test_knobs (the wave rules); None = the context's (P_RULE_DEFAULT "wave_fill1_p1tie"); must equal ctx.p_rule
    )
    schedule: str = SCHEDULE_DEFAULT  # "wait_first" (shipped) issues the merges and tape walks after each wait; "overlap" via _kcp_test_knobs
    fused_walks: int = FUSED_WALKS_DEFAULT  # shipped 1: one tape launch per phase from the carry plane (+ the S_in / dS_out copies); 0 via _kcp_test_knobs
    cols_rule: str = "fixed"  # D1: "by_chains" = c8 up to COLS_RULE_CHAINS state chains, c16 beyond; _kcp_test_knobs only
    walk_geometry: str = WALK_GEOMETRY_DEFAULT  # walk geometry: shipped "lpt_tape_one_cta_per_sm"; "shipped" (the earlier geometry) and the other WALK_GEOMETRIES via _kcp_test_knobs only
    walk_form: str = WALK_FORM_DEFAULT  # the record form of the K1 / K2 walk launches: "auto" (shipped) = the registry's rule per launch; "full" / "split" via _kcp_test_knobs only
    graph: str = GRAPH_DEFAULT  # graph replay: public keyword; "replay" captures a call once per validity key and replays it (eager on a key miss); also reachable through _kcp_test_knobs for the A/B record
    graph_cap: str = GRAPH_CAP_DEFAULT  # the cache-config form of a captured call ("node_attr" shipped | "none" the previous form of record | "capture_pref" rejected | "node_attr_post"); _kcp_test_knobs only
    host_trim: int = HOST_TRIM_DEFAULT  # shipped 1 = the trimmed host path of the eager call (same launches, bitwise); 0 = the previous path, _kcp_test_knobs only
    merge_fuse: str = MERGE_FUSE_DEFAULT  # "carry" = carry_start + carry_in (dcarry_end + dcarry_cross) in one K4 launch; _kcp_test_knobs only
    saved_carry: int = SAVED_CARRY_DEFAULT  # shipped 1 = the forward's carry plane stays in the saved set and the saved-mode backward skips the P > 1 re-walk + carry merges; 0 = the re-walk, _kcp_test_knobs only

    def check(self) -> OptionsDetKcp:
        if self.exchange not in EXCHANGES:
            raise ValueError(
                f"exchange must be one of {EXCHANGES}, got {self.exchange!r}"
            )
        if self.offset_class not in OFFSET_CLASSES:
            raise ValueError(
                f"offset_class must be one of {OFFSET_CLASSES}, got {self.offset_class!r}"
            )
        if self.offset_class != "fla":
            raise NotImplementedError(
                "offset_class='fp32' (K3's offset program) is a Phase-2 knob"
            )
        if int(self.head_groups) != 1:
            raise NotImplementedError(
                "head_groups > 1 is not supported on the KCP route"
            )
        if (
            self.save_transfer is None
        ):  # the ring scheme has no save_transfer form -> it resolves to False there
            self.save_transfer = (
                False
                if self.exchange == "ring"
                else bool(cake_kcp_plan.SAVE_TRANSFER_DEFAULT)
            )
        elif self.save_transfer and self.exchange == "ring":
            raise ValueError(
                "exchange='ring' receives no c1 tensor: save_transfer=True has no ring form (leave save_transfer unset, or pass False)"
            )
        self.save_transfer = bool(self.save_transfer)
        self.debug_sequence = bool(self.debug_sequence)
        if self.spec3_probe not in SPEC3_PROBES:
            raise ValueError(
                f"spec3_probe must be one of {SPEC3_PROBES}, got {self.spec3_probe!r}"
            )
        self.precision_mode = int(self.precision_mode)
        if self.transfer_walk not in TRANSFER_WALKS:
            raise ValueError(
                f"transfer_walk must be one of {TRANSFER_WALKS}, got {self.transfer_walk!r}"
            )
        if self.precision_mode and self.transfer_walk != "tf32_hilo":
            raise ValueError(
                "precision_mode selects a program of the tf32_hilo family; it cannot combine with a BF16x3 transfer_walk"
            )
        for key, allowed in P2_KNOBS.items():
            if getattr(self, key) not in allowed:
                raise ValueError(
                    f"{key} must be one of {allowed}, got {getattr(self, key)!r}"
                )
        return self


_OPTION_KEYS = (
    "save_transfer",
    "exchange",
    "head_groups",
    "offset_class",
    "debug_sequence",
    "spec3_probe",
    "graph",
)  # spec3_probe, graph: public keywords


def _options(kwargs: dict[str, Any]) -> OptionsDetKcp:
    """Validate the entry points' ``**options`` (unknown keywords are a ``TypeError``); ``_kcp_test_knobs`` goes
    through the plan module's validator (``precision_mode``) and, for the test knob ``transfer_walk`` (K3's
    ``tf32_hilo`` / ``bf16x3`` programs of ``build.test.json``: the TF32 hi/lo walk and the full-width BF16x3 form),
    through the same environment gate here, so the public API can never select a non-zero ``precision_mode`` or a
    non-shipped transfer-walk program."""
    pre = kwargs.get("_kcp_options")
    if (
        pre is not None
    ):  # the options validated by chunk_kda_det_kcp on the caller's frame, passed through the autograd node
        if not isinstance(pre, OptionsDetKcp) or len(kwargs) != 1:
            raise TypeError(
                "_kcp_options is the validated OptionsDetKcp of the entry point and travels alone"
            )
        return pre
    kwargs = dict(kwargs)
    raw = kwargs.pop("_kcp_test_knobs", None)
    transfer_walk: str | None = (
        None  # unset: the shipped walk (the hi/lo walk under precision_mode)
    )
    if isinstance(raw, dict) and "transfer_walk" in raw:
        if not cake_kcp_plan.test_knobs_enabled():
            raise ValueError(
                f"_kcp_test_knobs is a private test hook: it is accepted only after {cake_kcp_plan.ENABLE_TEST_KNOBS}"
            )
        raw = dict(raw)
        transfer_walk = str(raw.pop("transfer_walk"))
        if transfer_walk not in TRANSFER_WALKS:
            raise ValueError(
                f"_kcp_test_knobs: transfer_walk must be one of {TRANSFER_WALKS}, got {transfer_walk!r}"
            )
    p2: dict[str, Any] = {}
    if isinstance(raw, dict) and any(
        k in raw for k in P2_KNOBS
    ):  # the earlier knobs: the same gate
        if not cake_kcp_plan.test_knobs_enabled():
            raise ValueError(
                f"_kcp_test_knobs is a private test hook: it is accepted only after {cake_kcp_plan.ENABLE_TEST_KNOBS}"
            )
        raw = dict(raw)
        for key in P2_KNOBS:
            if key in raw:
                p2[key] = (
                    int(raw.pop(key))
                    if key in ("fused_walks", "host_trim", "saved_carry")
                    else str(raw.pop(key))
                )
    if (
        "graph" in p2
    ):  # graph is a public keyword as well; the knob form (A/B record) may not contradict it
        if "graph" in kwargs and str(kwargs["graph"]) != p2["graph"]:
            raise ValueError(
                f"graph given twice with different values: keyword {kwargs['graph']!r}, _kcp_test_knobs {p2['graph']!r}"
            )
        kwargs["graph"] = p2.pop("graph")
    knobs = cake_kcp_plan.validate_test_knobs(raw)
    if transfer_walk is None:  # the static TF32 programs are forms of the hi/lo walk
        transfer_walk = (
            "tf32_hilo" if int(knobs["precision_mode"]) else TRANSFER_WALKS[0]
        )
    p_rule = (
        knobs["p_rule"] if isinstance(raw, dict) and "p_rule" in raw else None
    )  # plan rule: the context carries the rule
    unknown = sorted(set(kwargs) - set(_OPTION_KEYS))
    if unknown:
        raise TypeError(
            f"unexpected keyword argument(s) {unknown}; known options: {list(_OPTION_KEYS)} and _kcp_test_knobs"
        )
    return OptionsDetKcp(
        **kwargs,
        precision_mode=int(knobs["precision_mode"]),
        transfer_walk=transfer_walk,
        p_rule=p_rule,
        **p2,
    ).check()


def _exchange_scheme(tb: _Tables, opts: OptionsDetKcp) -> str:
    """``a2a`` or ``ring`` (SPEC-2): ``auto`` is ``a2a`` in Phase 1 (the ring path arrives with gate 5); an explicit
    ``ring`` needs every document chain within ``RING_MAX_HOPS`` hops."""
    if opts.exchange == "ring":
        if tb.max_hops > RING_MAX_HOPS:
            raise ValueError(
                f"exchange='ring' needs n_d_max - 1 <= {RING_MAX_HOPS}, this partition has {tb.max_hops}"
            )
        if opts.save_transfer:  # unreachable through _options() (OptionsDetKcp.check resolves / refuses it); kept as the guard of the contract
            raise NotImplementedError(
                "exchange='ring' receives no c1 tensor: save_transfer=True has no ring form"
            )
        return "ring"
    return "a2a"


# ----------------------------------------------------------------------------------------------------------------
# One call (forward or backward): buffers, launches, collectives


class CallDetKcp:
    """The state of one forward or backward call on one rank: tables, buffers (workspace pools and the call's fresh
    tensors), operands, backend, the launch / collective counters and the issue sequence (R7 debug counter)."""

    def __init__(
        self,
        *,
        ctx: Any,
        H: int,
        ws: WorkspaceDetKcp,
        tb: _Tables,
        backend: Backend,
        opts: OptionsDetKcp,
        scale: float,
        ops: dict[str, torch.Tensor],
        backward: bool,
    ) -> None:
        self.ctx, self.H, self.ws, self.tb, self.backend, self.opts = (
            ctx,
            int(H),
            ws,
            tb,
            backend,
            opts,
        )
        self.scale = float(scale)
        self.device = ws.device
        self.rank, self.world, self.group = (
            int(ctx.rank),
            int(ctx.world_size),
            ctx.group,
        )
        self.state_dtype = ws.state_dtype
        self.ops = ops
        self.T_r = int(ctx.T_r)
        self.backward = bool(backward)
        self.counters = {
            "launches": 0,
            "collectives": 0,
            "walks": 0,
            "merges": 0,
            "stages": 0,
        }
        self.sequence: list[str] = []
        self.merge_log: list[
            tuple[str, int, int, int | None]
        ] = []  # (family, ncols, n_chains, explicit cols_per_cta) per K4 launch
        self.scratch: dict[str, Any] = {}
        self.bufs: dict[str, torch.Tensor] = {}
        self.local = (
            ws.local_view
            if (backend.local_stages and ws.local_view is not None)
            else None
        )
        self.local_values: dict[str, Any] | None = None
        self.out: torch.Tensor | None = None
        self.grads: dict[str, torch.Tensor] | None = None
        # the gate mode ("raw": use_gate_in_kernel=True) and the raw mode's checked parameters (set by _prepare)
        self.gate = "g"
        self.gate_params: Any = None
        self.exchange = _exchange_scheme(tb, opts)
        self.precision_mode = int(opts.precision_mode)
        self.transfer_walk = str(opts.transfer_walk)
        if opts.p_rule is not None and opts.p_rule != getattr(
            ctx, "p_rule", cake_kcp_plan.P_RULE_DEFAULT
        ):
            raise ValueError(
                f"_kcp_test_knobs p_rule {opts.p_rule!r} differs from the context's plan rule {getattr(ctx, 'p_rule', cake_kcp_plan.P_RULE_DEFAULT)!r}: give the knob to build_context_det_kcp"
            )
        self.schedule = str(opts.schedule)
        self.cols_rule = str(opts.cols_rule)  # D1
        if int(tb.fused_walks) != int(opts.fused_walks):
            raise ValueError(
                f"tables built with fused_walks={tb.fused_walks}, the call asks for {opts.fused_walks}"
            )
        self.walk_geometry = str(opts.walk_geometry)  # walk geometry
        if tb.walk_geometry != self.walk_geometry:
            raise ValueError(
                f"tables built with walk_geometry={tb.walk_geometry!r}, the call asks for {self.walk_geometry!r}"
            )
        # 1 = the trimmed host path (shipped), 0 = the previous path (same launches, bitwise)
        self.host_trim = int(opts.host_trim)
        self.walk_form = str(
            opts.walk_form
        )  # "auto": the registry's rule per K1 / K2 launch; "full" / "split" forced
        self.walk_form_log: list[
            tuple[str, str, int]
        ] = []  # (program, form, n_walk) per K1 / K2 walk launch
        self.merge_fuse = str(opts.merge_fuse)  # fused carry merges
        self.saved_carry = int(
            opts.saved_carry
        )  # 1 (shipped) = the saved carry plane; 0 = the re-walk (test knob)

    # -- buffers
    def buf(self, name: str) -> torch.Tensor:
        t = self.bufs.get(name)
        return (
            t
            if t is not None
            else self.ws.pool(name, self.tb.sizes[name], cached=self.host_trim == 1)
        )

    def fresh(self, name: str) -> torch.Tensor:
        """A fresh per-call tensor standing in for pool ``name`` (the saved-set tensors of SPEC-4)."""
        t = _alloc((self.tb.sizes[name],), self.state_dtype, self.device)
        self.bufs[name] = t
        return t

    def indexed(
        self, pool: str, index: Sequence[int], tiles: Sequence[Sequence[int]]
    ) -> Indexed:
        if self.host_trim:  # the tables' precomputed tuples / addressing / device index table (same values)
            meta = self.tb.indexed_meta(index, tiles, self.H, self.device)
            return Indexed(self.buf(pool), meta.index, meta.tiles, meta)
        return Indexed(
            self.buf(pool),
            tuple(int(i) for i in index),
            tuple(tuple(int(t) for t in row) for row in tiles),
        )

    # -- launches
    def kernel(self, name: str, **args: Any) -> None:
        self.counters["launches"] += 1
        self.sequence.append(name)
        args["_call"] = self
        self.backend.run(name, args)

    def walk(self, name: str, walk: Sequence[int], **args: Any) -> None:
        """A K1 / K2 / K3 launch over the segments ``walk`` (nothing when the list is empty)."""
        if len(walk) == 0:
            return
        self.counters["walks"] += 1
        # the trimmed path passes the tables' own list (its identity keys the cached device table)
        self.kernel(
            name, walk=walk if self.host_trim else [int(s) for s in walk], **args
        )

    def merge(
        self,
        family: str,
        *,
        chains: Chains | None = None,
        spec: dict[str, Any] | None = None,
    ) -> None:
        """One K4 launch of chain family ``family`` (nothing when it has no chain); ``chains`` / ``spec`` replace the
        table's family (the ring exchange's one-element chains)."""
        ch = self.tb.chains[family] if chains is None else chains
        if ch.n_chains == 0:
            return
        spec = _Tables.FAMILIES[family] if spec is None else spec
        self.counters["merges"] += 1
        self.kernel(
            "kcp_affine_merge",
            family=family,
            m=self.buf(spec["m"]),
            c=self.buf(spec["c"]) if spec["c"] else None,
            x0=self.buf(spec["x0"]) if spec["x0"] else None,
            y=self.buf(spec["y"]),
            chains=ch,
            transpose_m=int(spec["tm"]),
            write_states=int(spec["ws"]),
        )

    def transpose_tiles(self, src: str, src_tiles, dst: str, dst_tiles) -> None:
        """Scatter the transposes of the tiles ``src_tiles[slot][h]`` of pool ``src`` into ``dst_tiles[slot][h]`` of
        pool ``dst`` (``M_f`` -> ``M_f^T``, bitwise); nothing when there is no tile."""
        n = sum(len(row) for row in src_tiles)
        if n == 0:
            return
        self.counters["launches"] += 1
        self.sequence.append("kcp_transpose_tiles")
        by_id = (
            self.host_trim == 1
        )  # the tile tables are this table set's tuples -> identity-keyed cache
        self.backend.transpose_tiles(
            src=self.buf(src),
            src_tiles=self.tb.device_tiles(src_tiles, self.device, by_identity=by_id),
            dst=self.buf(dst),
            dst_tiles=self.tb.device_tiles(dst_tiles, self.device, by_identity=by_id),
        )

    def stage(self, name: str, **extra: Any) -> None:
        """A token-side stage of the existing backend (skipped without a local arena or tokens)."""
        if self.local is None or self.T_r == 0:
            return
        self.counters["launches"] += 1
        self.counters["stages"] += 1
        self.sequence.append(name)
        self.backend.stage(name, self, **extra)

    # -- collectives
    def a2a(
        self,
        tag: str,
        send_name: str,
        recv_name: str,
        send_splits: Sequence[int],
        recv_splits: Sequence[int],
    ):
        """Issue ``all_to_all_single`` (async) from pool ``send_name`` into ``recv_name`` with element splits per
        peer; returns the work handle (``None`` on a single rank)."""
        send_splits, recv_splits = (
            [int(x) for x in send_splits],
            [int(x) for x in recv_splits],
        )
        if len(send_splits) != self.world or len(recv_splits) != self.world:
            raise ValueError(
                f"{tag}: split lists must have one entry per rank ({self.world})"
            )
        if self.host_trim:  # cached [:n] views of the pools (a fresh per-call tensor is sliced as today)
            send = (
                self.ws.pool(send_name, sum(send_splits), cached=True)
                if send_name not in self.bufs
                else self.buf(send_name)[: sum(send_splits)]
            )
            recv = (
                self.ws.pool(recv_name, sum(recv_splits), cached=True)
                if recv_name not in self.bufs
                else self.buf(recv_name)[: sum(recv_splits)]
            )
        else:
            send = self.buf(send_name)[: sum(send_splits)]
            recv = self.buf(recv_name)[: sum(recv_splits)]
        self.counters["collectives"] += 1
        self._sequence(tag)
        if self.world == 1 or not (dist.is_available() and dist.is_initialized()):
            if send.numel() != recv.numel():
                raise ValueError(
                    f"{tag}: a single rank must send exactly what it receives"
                )
            recv.copy_(send)
            return None
        return dist.all_to_all_single(
            recv,
            send,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.group,
            async_op=True,
        )

    @staticmethod
    def wait(work) -> None:
        if work is not None:
            work.wait()

    def _sequence(self, tag: str, kind: str = "a2a") -> None:
        self.sequence.append(f"{kind}:{tag}")
        seq = getattr(self.ctx, "sequence", None)
        n = seq.next(tag) if seq is not None else len(self.sequence)
        if (
            self.opts.debug_sequence
            and self.world > 1
            and dist.is_available()
            and dist.is_initialized()
        ):
            gathered: list[Any] = [None] * self.world
            dist.all_gather_object(gathered, (tag, n), group=self.group)
            if any(g != (tag, n) for g in gathered):
                raise RuntimeError(
                    f"debug_sequence: the collective issue order differs across ranks: {gathered}"
                )


LAST_CALL: dict[str, dict[str, Any]] = {}


def _record(kind: str, call: CallDetKcp) -> None:
    """The launch inventory of the last forward / backward (the design note §10.5)."""
    LAST_CALL[kind] = {
        **{f"n_{k}": v for k, v in call.counters.items()},
        "sequence": list(call.sequence),
        "exchange": call.exchange,
        "backend": call.backend.name,
        "gate": call.gate,  # "g" / "raw" (use_gate_in_kernel=True)
        "P": call.tb.P,
        "NSEG": call.tb.NSEG,
        "Hg": call.tb.Hg,
        "Hg_r": call.tb.Hg_r,
        "n_in": call.tb.n_in,
        "n_out": call.tb.n_out,
        "workspace_bytes": call.ws.bytes,
        "pool_bytes": call.ws.pool_bytes,
        "fresh_bytes": {k: t.numel() * t.element_size() for k, t in call.bufs.items()},
        "precision_mode": call.precision_mode,
        "transfer_walk": call.transfer_walk,
        "schedule": call.schedule,
        "fused_walks": call.tb.fused_walks,
        "cols_rule": call.cols_rule,  # the earlier knobs
        "walk_geometry": call.walk_geometry,  # walk-launch geometry
        "walk_forms": list(
            call.walk_form_log
        ),  # (record, form, n_walk) per K1 / K2 walk launch: the registry's rule or the walk_form knob
        # the host path of this call (1 trimmed = shipped / 0 previous; same launches)
        "host_trim": call.host_trim,
        "merge_fuse": call.merge_fuse,  # the fused carry merges (test knob)
        "saved_carry": call.saved_carry,  # 1 (shipped) = the saved carry plane; 0 = the re-walk (test knob)
        "saved_carry_plane": call.scratch.get(
            "saved_carry_plane"
        ),  # bytes of the carry plane kept (forward) / read (backward) on a P > 1 rank under saved_carry=1, None otherwise
        "merges": list(
            call.merge_log
        ),  # D1 evidence: (family, ncols, n_chains, explicit cols_per_cta or None = registry)
        "spec3": call.scratch.get(
            "spec3"
        ),  # the SPEC-3 debug assertion's result (first saved-mode backward)
        "recompute_handle": call.scratch.get(
            "recompute_handle"
        ),  # None without a handle; "capture" (the original forward stored its entry) / "replay" (it bound one)
        "graph": call.opts.graph,
        "graph_event": "eager",
        "graph_key": None,
        "graph_fallback": None,  # the graph mode and how this call ran
        "cache_config": call.scratch.get(
            "cache_config", "none"
        ),  # "bracket" (eager capped launches) / "node_attr" (the replay form of record) / "node_attr_post" / "capture_pref" / "none" / "unavailable" (no loadable CUDA runtime, or a degraded cap)
        "cache_config_degraded": call.scratch.get(
            "node_attr_degraded"
        ),  # why a captured tape launch stayed uncapped (None otherwise)
        "cache_config_bracket": call.scratch.get(
            "cache_config_bracket"
        ),  # the capture-wide preference's set / reset evidence (arm "capture_pref")
        "cache_config_nodes": call.scratch.get(
            "node_attr_launches"
        ),  # kernel-node cap forms: captured tape launches whose kernel node carries the carveout attribute
    }
    CACHE_CONFIG_STATS[LAST_CALL[kind]["cache_config"]] += 1


# ----------------------------------------------------------------------------------------------------------------
# Launch sequences (the design note §3.5 forward, §6.5 backward; one compute stream, waits at first consumption;
# shipped: schedule wait_first + fused tape walks, the overlapped issue order only under the overlap knob)


# ----------------------------------------------------------------------------------------------------------------
# Ring exchange (SPEC-2; gate 5): per-document chains of point-to-point hops on ``ctx.group`` instead of the owner
# merge + c1 / c2 (b1 / b2).  Hop ``h`` of the forward: every crossing-out fragment at chain position ``h - 1`` sends
# its exit state ``S_out = M_f S_in + B_f`` (hop 1: ``B_f`` itself) to the fragment at position ``h``; the receiver
# writes it into its ``S_in`` tiles, so the tape walks after the hops read exactly what the a2a path's c2 delivers
# (bitwise: the same K4 element step on the same tile bits, or a plain copy).  The backward runs the hops from the
# document end: the fragment at position ``n_d - h`` sends ``dS_in = M_f^T dS_out + d_f`` (hop 1: ``d_f``) to its
# predecessor as that fragment's ``dS_out`` (into the ``b2_recv`` tiles).  One ``batch_isend_irecv`` per hop in
# ``(doc, slot)`` order; the document-start (document-last) tape walk overlaps hop 1 (R7).


class _Ring:
    """Per-hop send / receive lists of one rank (positions from ``ctx.chains``), built once per ``_Tables``."""

    def __init__(self, tb: _Tables) -> None:
        ctx = tb.ctx
        rank = int(ctx.rank)
        self.hops = tb.max_hops
        # the K4 chains of the hops (one Chains per family, built at its first call): their device tables are copied
        # once per table set, not per call
        self.chains: dict[str, Chains] = {}
        pos: dict[
            int, tuple[int, int]
        ] = {}  # slot -> (position in its document chain, chain length)
        for f in tb.frags:
            if f.length <= 0:
                continue
            chain = list(ctx.chains[f.doc])
            pos[f.slot] = (chain.index((rank, f.slot)), len(chain))
        self.pos = pos
        in_slot, out_slot = list(ctx.in_slot), list(ctx.out_slot)
        # forward hop h: senders at position h - 1 (crossing out), receivers at position h (crossing in)
        self.fwd_send = {
            h: [
                f
                for f in tb.frags
                if f.length > 0 and f.crossing_out and pos[f.slot][0] == h - 1
            ]
            for h in range(1, self.hops + 1)
        }
        self.fwd_recv = {
            h: [
                f
                for f in tb.frags
                if f.length > 0 and f.crossing_in and pos[f.slot][0] == h
            ]
            for h in range(1, self.hops + 1)
        }
        # backward hop h: senders at position n_d - h (crossing in), receivers at position n_d - h - 1 (crossing out)
        self.bwd_send = {
            h: [
                f
                for f in tb.frags
                if f.length > 0
                and f.crossing_in
                and pos[f.slot][0] == pos[f.slot][1] - h
            ]
            for h in range(1, self.hops + 1)
        }
        self.bwd_recv = {
            h: [
                f
                for f in tb.frags
                if f.length > 0
                and f.crossing_out
                and pos[f.slot][0] == pos[f.slot][1] - h - 1
            ]
            for h in range(1, self.hops + 1)
        }
        self.in_slot, self.out_slot = in_slot, out_slot

        def peer(f, step):
            chain = ctx.chains[f.doc]
            return int(chain[pos[f.slot][0] + step][0])

        self.next_rank = {
            f.slot: peer(f, 1) for f in tb.frags if f.length > 0 and f.crossing_out
        }
        self.prev_rank = {
            f.slot: peer(f, -1) for f in tb.frags if f.length > 0 and f.crossing_in
        }


def _ring_tables(tb: _Tables) -> _Ring:
    if tb._ring is None:
        tb._ring = _Ring(tb)
    return tb._ring


def _ring_block(pool: torch.Tensor, slot: int, H: int) -> torch.Tensor:
    """The contiguous ``[H, D, D]`` block of ``slot`` in a ring staging pool (tiles ``slot * H + h``)."""
    return pool.view(-1)[slot * H * TILE : (slot + 1) * H * TILE].view(H, D, D)


def _ring_copy_in(
    call: CallDetKcp,
    tb: _Tables,
    src: torch.Tensor,
    src_tiles,
    slot_src: int,
    dst: torch.Tensor,
    slot_dst: int,
) -> None:
    """``dst`` ring block ``slot_dst`` <- the ``H`` tiles ``src_tiles[slot_src][h]`` of pool ``src`` (bitwise copy)."""
    idx = tb.device_tiles([src_tiles[slot_src]], call.device)
    _ring_block(dst, slot_dst, call.H).copy_(src.view(-1, D, D)[idx.view(-1)])


def _ring_scatter(
    call: CallDetKcp,
    tb: _Tables,
    src: torch.Tensor,
    slot_src: int,
    dst: torch.Tensor,
    dst_tiles,
    slot_dst: int,
) -> None:
    """Pool ``dst`` tiles ``dst_tiles[slot_dst][h]`` <- the ring block ``slot_src`` of ``src`` (bitwise copy)."""
    idx = tb.device_tiles([dst_tiles[slot_dst]], call.device)
    dst.view(-1, D, D)[idx.view(-1)] = _ring_block(src, slot_src, call.H)


def _ring_p2p(
    call: CallDetKcp,
    hop: int,
    sends: list[tuple[torch.Tensor, int]],
    recvs: list[tuple[torch.Tensor, int]],
):
    """One ``batch_isend_irecv`` of the hop (sends then receives, each list in ``(doc, slot)`` order); returns the
    works (empty when the rank has nothing to exchange)."""
    call.counters["collectives"] += 1
    call._sequence(f"hop{hop}", kind="ring")
    if call.world == 1 or not dist.is_initialized():
        for _t, _r in sends:  # a one-rank chain cannot exist (no crossing); defensive
            raise RuntimeError("ring exchange on a single rank")
        return []
    ops = [dist.P2POp(dist.isend, t, peer, group=call.group) for t, peer in sends]
    ops += [dist.P2POp(dist.irecv, t, peer, group=call.group) for t, peer in recvs]
    return dist.batch_isend_irecv(ops) if ops else []


def _ring_wait(works) -> None:
    for w in works:
        w.wait()


def _ring_chain(
    tb: _Tables, family: str, elems: list[tuple[int, int, int, int]]
) -> Chains:
    """One-element chains per (fragment, head) from tile numbers ``(x0, m, c, y)`` (K4 state step on one fragment),
    cached per ``family`` on the table set's ring tables: ``elems`` derive from the table set alone, so every call
    of the set reuses the chains and their device tables (a fresh set per call would upload them, and record the
    upload on the context, on every call)."""
    ring = _ring_tables(tb)
    ch = ring.chains.get(family)
    if ch is None:
        ch = Chains(ncols=D)
        # tile numbers -> element offsets (the chain tables address pools in elements)
        for x0, m, c, y in elems:
            ch.add([(m * TILE, c * TILE, -1)], x0=x0 * TILE, y_final=y * TILE)
        ring.chains[family] = ch
    return ch


def _ring_forward(call: CallDetKcp) -> None:
    """Hops of the forward ring; afterwards ``S_in`` holds the crossing-in fragments' initial states (as after c2)."""
    tb, H = call.tb, call.H
    ring = _ring_tables(tb)
    out_pool, in_pool = call.buf("ring_out"), call.buf("ring_in")
    c1 = call.buf("c1_send")
    s_in = call.buf("S_in")
    for h in range(1, ring.hops + 1):
        senders, receivers = ring.fwd_send[h], ring.fwd_recv[h]
        if (
            h == 1
        ):  # S_out of a document-start fragment is its B_f: copy the c1 B tiles into the contiguous block
            for f in senders:
                _ring_copy_in(
                    call,
                    tb,
                    c1,
                    tb.c1_b_tiles,
                    ring.out_slot[f.slot],
                    out_pool,
                    ring.out_slot[f.slot],
                )
        else:  # S_out = M_f S_in + B_f: one K4 element per sender (forward family: c1 tiles hold M_f^T)
            elems = [
                (
                    tb.s_in_tiles[ring.in_slot[f.slot]][hh],
                    tb.c1_m_tiles[ring.out_slot[f.slot]][hh],
                    tb.c1_b_tiles[ring.out_slot[f.slot]][hh],
                    ring.out_slot[f.slot] * H + hh,
                )
                for f in senders
                for hh in range(H)
            ]
            if elems:
                call.merge(
                    f"ring_fwd_{h}",
                    chains=_ring_chain(tb, f"ring_fwd_{h}", elems),
                    spec={
                        "m": "c1_send",
                        "c": "c1_send",
                        "x0": "S_in",
                        "y": "ring_out",
                        "tm": 0,
                        "ws": 0,
                    },
                )
        works = _ring_p2p(
            call,
            h,
            [
                (
                    _ring_block(out_pool, ring.out_slot[f.slot], H),
                    ring.next_rank[f.slot],
                )
                for f in senders
            ],
            [
                (_ring_block(in_pool, ring.in_slot[f.slot], H), ring.prev_rank[f.slot])
                for f in receivers
            ],
        )
        if h == 1 and call.schedule == "overlap":
            _ring_start_walk(call)  # document starts: under hop 1 (R7)
        _ring_wait(works)
        for f in receivers:
            _ring_scatter(
                call,
                tb,
                in_pool,
                ring.in_slot[f.slot],
                s_in,
                tb.s_in_tiles,
                ring.in_slot[f.slot],
            )
    if (
        ring.hops == 0 and call.schedule == "overlap"
    ):  # no document crosses a rank: no hop 1 to hide it under, the walk still writes the document starts' rows
        _ring_start_walk(call)


def _ring_start_walk(call: CallDetKcp) -> None:
    """The overlap schedule's document-start tape walk of the ring forward (under hop 1; alone when there is none)."""
    tb = call.tb
    carry = call.indexed("carry", tb.carry_index, tb.carry_tiles) if tb.P > 1 else None
    call.walk("kcp_fwdh_tape", tb.walk_start, h0=carry, out=call.out)


def _ring_end_walk(call: CallDetKcp) -> None:
    """The overlap schedule's ``dcarry_end`` merge and document-end tape walk of the ring backward (under hop 1;
    alone when there is none)."""
    tb = call.tb
    dcarry = (
        call.indexed("carry", tb.dcarry_index, tb.carry_tiles) if tb.P > 1 else None
    )
    if tb.P > 1:
        call.merge("dcarry_end")
    call.walk("kcp_dhu_tape", tb.walk_end, dht=dcarry)


def _ring_backward(call: CallDetKcp) -> None:
    """Hops of the adjoint ring; afterwards ``b2_recv`` holds the crossing-out fragments' ``dS_out`` (as after b2).
    Requires the b1 ``dm`` form (``d_f`` and ``M_f`` of the crossing-in fragments in ``b1_send``)."""
    tb, H = call.tb, call.H
    ring = _ring_tables(tb)
    out_pool, in_pool = call.buf("ring_out"), call.buf("ring_in")
    b1, b2 = call.buf("b1_send"), call.buf("b2_recv")
    d_tiles, m_tiles = tb.b1_d_tiles["dm"], tb.b1_m_tiles
    for h in range(1, ring.hops + 1):
        senders, receivers = ring.bwd_send[h], ring.bwd_recv[h]
        if h == 1:  # the document-last fragment's dS_in is its d_f
            for f in senders:
                _ring_copy_in(
                    call,
                    tb,
                    b1,
                    d_tiles,
                    ring.in_slot[f.slot],
                    out_pool,
                    ring.in_slot[f.slot],
                )
        else:  # dS_in = M_f^T dS_out + d_f (adjoint step; the b1 M_f tiles hold Mop^T under Mop = M_f^T)
            elems = [
                (
                    tb.ds_out_tiles[ring.out_slot[f.slot]][hh],
                    m_tiles[ring.in_slot[f.slot]][hh],
                    d_tiles[ring.in_slot[f.slot]][hh],
                    ring.in_slot[f.slot] * H + hh,
                )
                for f in senders
                for hh in range(H)
            ]
            if elems:
                call.merge(
                    f"ring_bwd_{h}",
                    chains=_ring_chain(tb, f"ring_bwd_{h}", elems),
                    spec={
                        "m": "b1_send",
                        "c": "b1_send",
                        "x0": "b2_recv",
                        "y": "ring_out",
                        "tm": 0,
                        "ws": 0,
                    },
                )
        works = _ring_p2p(
            call,
            h,
            [
                (_ring_block(out_pool, ring.in_slot[f.slot], H), ring.prev_rank[f.slot])
                for f in senders
            ],
            [
                (_ring_block(in_pool, ring.out_slot[f.slot], H), ring.next_rank[f.slot])
                for f in receivers
            ],
        )
        if h == 1 and call.schedule == "overlap":
            _ring_end_walk(call)  # document ends: under hop 1 (R7)
        _ring_wait(works)
        for f in receivers:
            _ring_scatter(
                call,
                tb,
                in_pool,
                ring.out_slot[f.slot],
                b2,
                tb.ds_out_tiles,
                ring.out_slot[f.slot],
            )
    if ring.hops == 0 and call.schedule == "overlap":  # the mirror of _ring_forward's
        _ring_end_walk(call)


def _summaries(call: CallDetKcp) -> None:
    """The fragment summaries ``[B_f | M_f^T]`` into the c1 send layout: at ``P = 1`` K1 ``_offset`` and K3 write the
    crossing-out fragments' tiles directly; at ``P > 1`` both walk every segment into the staging planes and K4
    composes the crossing-out fragments' segments (forward family -> ``B_f``, adjoint family -> ``M_f^T``)."""
    tb = call.tb
    if tb.P == 1:
        call.walk(
            "kcp_fwdh_offset",
            tb.walk_offset,
            h0=None,
            out=call.indexed("c1_send", tb.out_index, tb.c1_b_tiles),
        )
        call.walk(
            "kcp_transfer_walk",
            tb.walk_offset,
            out=call.indexed("c1_send", tb.out_index, tb.c1_m_tiles),
            init_mode=0,
            use_u=0,
            write_transposed=0,
        )
        return
    _stage_walks(call)
    call.merge(
        "map"
    )  # B_f (from 0) and M_f (from I) of every crossing-out fragment, forward family
    call.transpose_tiles(
        "c1_send", tb.mf_tiles, "c1_send", tb.c1_m_tiles
    )  # M_f^T into the c1 pairs (bitwise)


def _stage_walks(call: CallDetKcp) -> None:
    """``P > 1``: the K1 ``_offset`` / K3 walks of every segment into the staging planes ``stage_b`` / ``stage_m``."""
    tb = call.tb
    call.walk(
        "kcp_fwdh_offset",
        tb.walk_offset,
        h0=None,
        out=call.indexed("stage_b", tb.seg_index, tb.stage_tiles),
    )
    call.walk(
        "kcp_transfer_walk",
        tb.walk_offset,
        out=call.indexed("stage_m", tb.seg_index, tb.stage_tiles),
        init_mode=0,
        use_u=0,
        write_transposed=0,
    )


def _fwd_tapes(
    call: CallDetKcp,
    *,
    out: torch.Tensor | None,
    carry: Indexed | None,
    wait_first: bool,
) -> None:
    """The forward tape walk(s) after ``S_in`` is complete.  ``fused_walks``: one launch from one pool -- the
    carry plane at ``P > 1`` (``h0_index_fused``: carries, the ``S_in`` copies of ``carry_in``, ``-1`` zero states),
    ``S_in`` itself at ``P = 1`` -- over every live segment (``wait_first``) or the crossing-in fragments' segments
    (the document-start walk ran under c1); otherwise the three launches of the design note §3.5 (the document-start
    walk here only under ``wait_first``)."""
    tb = call.tb
    h0_in = call.indexed("S_in", tb.h0_index, tb.s_in_tiles)
    if tb.fused_walks:
        h0 = (
            call.indexed("carry", tb.h0_index_fused, tb.carry_tiles)
            if tb.P > 1
            else h0_in
        )
        call.walk(
            "kcp_fwdh_tape",
            tb.walk_live if wait_first else tb.walk_in_fused,
            h0=h0,
            out=out,
        )
        return
    if wait_first:
        call.walk(
            "kcp_fwdh_tape", tb.walk_start, h0=carry, out=out
        )  # document starts: after wait(c2)
    call.walk("kcp_fwdh_tape", tb.walk_in_p0, h0=h0_in, out=out)
    if tb.P > 1:
        call.walk("kcp_fwdh_tape", tb.walk_in_p1, h0=carry, out=out)


def _adj_tapes(call: CallDetKcp, *, dcarry: Indexed | None, wait_first: bool) -> None:
    """The adjoint tape walk(s) after ``dS_out`` is complete (the mirror of :func:`_fwd_tapes`: the carry plane holds
    the adjoint carries and the ``dS_out`` copies of ``dcarry_cross`` at ``P > 1``; ``b2_recv`` itself at ``P = 1``)."""
    tb = call.tb
    dht_cross = call.indexed("b2_recv", tb.dht_index, tb.ds_out_tiles)
    if tb.fused_walks:
        dht = (
            call.indexed("carry", tb.dht_index_fused, tb.carry_tiles)
            if tb.P > 1
            else dht_cross
        )
        call.walk(
            "kcp_dhu_tape", tb.walk_live if wait_first else tb.walk_cross_fused, dht=dht
        )
        return
    if wait_first:
        call.walk(
            "kcp_dhu_tape", tb.walk_end, dht=dcarry
        )  # document ends: after wait(b2)
    call.walk("kcp_dhu_tape", tb.walk_cross_last, dht=dht_cross)
    if tb.P > 1:
        call.walk("kcp_dhu_tape", tb.walk_cross_rest, dht=dcarry)


def _forward(
    call: CallDetKcp,
    *,
    save: bool,
    replay: bool,
    bound: dict[str, torch.Tensor] | None = None,
) -> None:
    """The design note §3.5.  ``save``: the saved set is kept (fresh ``S_in``; with ``save_transfer`` also the c1 receive
    tensor and ``M_seg``); ``replay``: the forward of a replay-mode backward (no ``fwd_o``, nothing kept).
    ``bound``: the received tensors a recompute handle stored (``S_in``; under ``save_transfer`` also ``c1_recv``):
    the recompute of a checkpointed forward binds them instead of exchanging -- no c1 / c2 / ring hop and no wait, no
    owner merge, and no summary launch whose only reader is the exchange (``P = 1``: none; ``P > 1``: the two stage
    walks that feed the carry merges stay, the map merge and the transpose into ``c1_send`` go); every other launch
    is the same on the same bits.
    ``schedule="wait_first"`` (the shipped default since an earlier revision): ``c1 -> wait -> fwd merge -> c2 -> wait -> carry merges -> tape walk(s)``
    -- no merge or walk is issued while a collective is in flight (the R7 overlap of the document-start walk under c1
    is dropped); the merges read only ``c1_recv`` / the staging planes and the tapes read no collective buffer, so
    every published bit is the same."""
    tb, opts = call.tb, call.opts
    wait_first = call.schedule == "wait_first"
    keep_transfer = opts.save_transfer and save and not replay
    if bound is None:
        call.fresh("S_in")
    else:
        call.bufs["S_in"] = bound["S_in"]
    if keep_transfer:
        if bound is None:
            call.fresh("c1_recv")
        else:
            call.bufs["c1_recv"] = bound["c1_recv"]
        if tb.P > 1:
            call.fresh("stage_m")  # M_seg^T of the segmented fragments (§16 item 5b)
    if (
        save and not replay and tb.P > 1 and call.saved_carry == 1
    ):  # the carry plane as a fresh tensor: the carry merges below write it, the tape walk reads it, the saved set keeps it
        plane = call.fresh("carry")
        call.scratch["saved_carry_plane"] = plane.numel() * plane.element_size()
    if replay:  # the replay's prep (the GLM prep's rows, the internal dout copy), then the internal Aqk / Akk rows
        call.stage("bwd_prep", saved=None)
        call.stage("fwd_intra")
        call.stage("wy")
    else:  # the intra program writes the GLM prep's rows itself (write_prep = 1); no token-row copies are saved
        call.stage("fwd_intra")
        call.stage("wy_fwd")
    if bound is None:
        _summaries(call)
    elif (
        tb.P > 1
    ):  # the stage walks only: the carry merges below read stage_b / stage_m
        _stage_walks(call)
    carry = call.indexed("carry", tb.carry_index, tb.carry_tiles) if tb.P > 1 else None
    if (
        bound is not None
    ):  # S_in is bound: no exchange; the overlap schedule's launches of the exchange window, in its order
        if not wait_first:
            if tb.P > 1:
                call.merge("carry_start")
            call.walk("kcp_fwdh_tape", tb.walk_start, h0=carry, out=call.out)
    elif (
        call.exchange == "ring"
    ):  # SPEC-2: per-document hops; S_in filled as c2 would (bitwise)
        if tb.P > 1 and not wait_first:
            call.merge("carry_start")
        _ring_forward(
            call
        )  # the document-start tape walk runs under hop 1 (overlap schedule only)
    else:
        w1 = call.a2a("c1", "c1_send", "c1_recv", *tb.split_elems("c1"))
        if not wait_first:
            if tb.P > 1:
                call.merge(
                    "carry_start"
                )  # reads the staging planes only (R7: no collective buffer)
            call.walk(
                "kcp_fwdh_tape", tb.walk_start, h0=carry, out=call.out
            )  # document starts: under c1 (R7)
        call.wait(w1)
        call.merge("fwd")  # the owner's chains: S_in slices into the c2 send layout
        w2 = call.a2a("c2", "c2_send", "S_in", *tb.split_elems("c2"))
        call.wait(w2)
    if tb.P > 1:
        if wait_first and call.merge_fuse == "carry":
            call.merge("carry_both")  # carry_start + carry_in in one launch
        else:
            if wait_first:
                call.merge("carry_start")
            call.merge(
                "carry_in"
            )  # with fused_walks also the S_in copies into the extra carry slots (write_states)
    _fwd_tapes(call, out=call.out, carry=carry, wait_first=wait_first)
    if not replay:
        call.stage("fwd_o")


def _backward(call: CallDetKcp, saved: SavedDetKcp | None) -> None:
    """The design note §6.5.  ``saved=None`` is the replay mode (the forward sequence without ``fwd_o`` recomputes
    ``S_in``, the tapes and the owner's c1 receive tensor); otherwise the tapes come from K1 ``_tape`` over the saved
    ``S_in`` and the owner's adjoint reads ``M_f`` from b1 (``save_transfer=False``) or ``M_f^T`` from the saved c1
    receive tensor.  ``schedule="wait_first"`` (the shipped default since an earlier revision): ``b1 -> wait -> adjoint merge -> b2 -> wait -> dcarry merges ->
    adjoint tape walk(s)`` (the R7 overlap of the document-last walk under b2 is dropped)."""
    tb = call.tb
    wait_first = call.schedule == "wait_first"
    replay = saved is None
    if replay:
        _forward(call, save=False, replay=True)
        m_source = "c1"
    else:
        call.bufs["S_in"] = saved.S_in
        m_source = "b1"
        if saved.c1_recv is not None:
            call.bufs["c1_recv"] = saved.c1_recv
            m_source = "c1"
        if saved.M_seg is not None:
            call.bufs["stage_m"] = saved.M_seg
        call.stage("bwd_prep", saved=saved)
        call.stage(
            "fwd_intra"
        )  # the internal Aqk / Akk rows the WY, dav and gradient programs read
        call.stage("wy")
        plane = (
            saved.carry if tb.P > 1 else None
        )  # the forward's carry plane (shipped; None under saved_carry=0): no K1 offset re-walk, no carry merges
        if plane is not None:
            call.scratch["saved_carry_plane"] = plane.numel() * plane.element_size()
        if (
            tb.P > 1
        ):  # the forward carries again: b_seg (and M_seg^T unless saved) -> K4 carries from 0 / S_in
            if plane is None:
                call.walk(
                    "kcp_fwdh_offset",
                    tb.walk_all,
                    h0=None,
                    out=call.indexed("stage_b", tb.seg_index, tb.stage_tiles),
                )
            if (
                saved.M_seg is None
            ):  # stage_m: the carry_in merge below and the owner's map merge (tmap_m) read it
                call.walk(
                    "kcp_transfer_walk",
                    tb.walk_all,
                    out=call.indexed("stage_m", tb.seg_index, tb.stage_tiles),
                    init_mode=0,
                    use_u=0,
                    write_transposed=0,
                )
            if plane is not None:
                pass  # the saved plane holds carry_start's and carry_in's results (and the S_in copies)
            elif call.merge_fuse == "carry":
                call.merge("carry_both")  # one launch for both carry families
            else:
                call.merge("carry_start")
                call.merge("carry_in")
        if plane is not None:
            call.bufs["carry"] = (
                plane  # the tape replay below reads the saved plane; the adjoint carries go to the workspace pool
            )
        if (
            tb.fused_walks and tb.P > 1
        ):  # fused walks: the whole tape replay from the carry plane in one launch
            call.walk(
                "kcp_fwdh_tape",
                tb.walk_live,
                h0=call.indexed("carry", tb.h0_index_fused, tb.carry_tiles),
                out=None,
            )
        else:
            # the trimmed arm passes the table set's own sublists (a fresh comprehension per call would miss the
            # identity-keyed device-table cache on every call: one blocking host-to-device copy per backward)
            trim = bool(call.host_trim)
            p0 = tb.walk_all_p0 if trim else [s for s in tb.walk_all if s % tb.P == 0]
            call.walk(
                "kcp_fwdh_tape",
                p0,
                h0=call.indexed("S_in", tb.h0_index, tb.s_in_tiles),
                out=None,
            )
            if tb.P > 1:
                rest = (
                    tb.walk_all_rest
                    if trim
                    else [s for s in tb.walk_all if s % tb.P != 0]
                )
                call.walk(
                    "kcp_fwdh_tape",
                    rest,
                    h0=call.indexed("carry", tb.carry_index, tb.carry_tiles),
                    out=None,
                )
        if plane is not None:
            del call.bufs[
                "carry"
            ]  # the saved plane is never written: dcarry_* publish into the workspace pool
    call.stage("dav")
    ring = call.exchange == "ring"
    form = (
        "dm" if (m_source == "b1" or ring) else "d"
    )  # the ring's adjoint step needs M_f on the fragment's own rank
    if tb.P == 1:
        call.walk(
            "kcp_dhu_offset",
            tb.walk_in,
            dht=None,
            out=call.indexed("b1_send", tb.in_index, tb.b1_d_tiles[form]),
        )
        if (
            form == "dm"
        ):  # the maps travel with the offsets: M_f (untransposed) beside d_f
            call.walk(
                "kcp_transfer_walk",
                tb.walk_m_b1,
                out=call.indexed("b1_send", tb.m_b1_index, tb.b1_m_tiles),
                init_mode=0,
                use_u=0,
                write_transposed=1,
            )
    else:
        call.walk(
            "kcp_dhu_offset",
            tb.walk_all,
            dht=None,
            out=call.indexed("stage_b", tb.seg_index, tb.stage_tiles),
        )
        call.merge(
            f"tmap_d_{form}"
        )  # d_f of the crossing-in fragments (adjoint family)
        if form == "dm":
            call.merge(
                "tmap_m"
            )  # M_f of the fragments whose map travels (forward family)
    dcarry = (
        call.indexed("carry", tb.dcarry_index, tb.carry_tiles) if tb.P > 1 else None
    )
    if ring:  # SPEC-2: adjoint hops from the document end; b2_recv filled as b2 would (bitwise)
        _ring_backward(
            call
        )  # dcarry_end + the document-end tape walk run under hop 1 (overlap schedule only)
    else:
        wb1 = call.a2a("b1", "b1_send", "b1_recv", *tb.split_elems(f"b1_{form}"))
        call.wait(wb1)
        call.merge(
            "adj_b1" if m_source == "b1" else "adj_c1"
        )  # dS_out slices into the b2 send layout
        wb2 = call.a2a("b2", "b2_send", "b2_recv", *tb.split_elems("b2"))
        if not wait_first:
            if tb.P > 1:
                call.merge("dcarry_end")  # reads the staging planes only (R7)
            call.walk(
                "kcp_dhu_tape", tb.walk_end, dht=dcarry
            )  # document ends: under b2 (R7)
        call.wait(wb2)
    if tb.P > 1:
        if wait_first and call.merge_fuse == "carry":
            call.merge("dcarry_both")  # dcarry_end + dcarry_cross in one launch
        else:
            if wait_first:
                call.merge("dcarry_end")
            call.merge(
                "dcarry_cross"
            )  # with fused_walks also the dS_out copies into the extra carry slots
    _adj_tapes(call, dcarry=dcarry, wait_first=wait_first)
    tail = (call.local_values or {}).get("_kcp_tail") if call.T_r else None
    # raw gate mode: the raw GLM intra program writes d raw and the per-chunk (hi, lo) partials of dA_log / ddt_bias
    # and no gate epilogue runs; one param_finalize launch then reduces this rank's partials in the order of its
    # local chunks (a raw-mode workspace has no lean arena, so its tail runs in one pass)
    tail_stages = ("dqkg", "glm_intra") if call.gate == "raw" else TAIL_STAGES
    if not tail:
        for name in tail_stages:
            call.stage(name)
    else:  # the lean layout: the tail stages once per chunk range, on that range's per-pass buffers
        for i in range(len(tail)):
            for name in tail_stages:
                call.stage(name, tail_pass=i)
    if call.gate == "raw":
        call.stage("param_finalize", seg_cap=call.scratch.get("seg_cap"))


# ----------------------------------------------------------------------------------------------------------------
# CUDA-graph replay of a call (public keyword ``graph="replay"`` of the three entry points since this
# revision; shipped default ``GRAPH_DEFAULT = "off"``)
#
# The first call of a validity key (kind; the context, workspace, backend and local-arena objects; the pool pointers;
# every option; scale; the save flags; and the shape / stride / dtype / data pointer of every operand and saved
# tensor) runs eagerly.  The second call is captured into a ``torch.cuda.CUDAGraph`` (nothing runs during a capture)
# and the graph is replayed once; every later call of the key replays it: the same kernels and collectives with the
# same arguments in the same order (bitwise the eager result) with no host work between them.  Any other key runs
# eagerly (silent fallback) and starts its own entry; ``exchange="ring"``, ``debug_sequence`` and a call without a
# caller-owned workspace are never captured.  A replayed call returns the graph's static tensors (detached, so every
# call hands autograd fresh objects): the next replay of the same key overwrites them, so a framework must keep its
# inputs pointer-stable and consume the outputs and the saved set before the next call of the same layer.
# ``GRAPH_STATS`` counts the events; ``LAST_CALL[kind]["graph_event"]`` says how the last call ran.  Every captured graph
# keeps its fresh tensors in a private pool (0.7-9.0 GiB per rank per key on the A/B rows;
# at most GRAPH_CACHE_MAX keys, the oldest dropped first); ``graph_cache_clear()`` releases them and must run before
# ``torch.distributed.destroy_process_group()`` (a graph holding a NCCL collective blocks ``ncclCommDestroy``).
#
# Cache-config preference under replay (a probe showed cudaDeviceSetCacheConfig inside a capture
# invalidates it): the per-launch bracket of the capped tape programs (_FiPrograms.launch) is never issued inside a stream
# capture.  The knob ``graph_cap`` selects the captured call's form: "node_attr" (the default of record) puts the one-CTA-per-SM
# carveout on each captured tape kernel node as a launch
# attribute right after its launch is captured, so that the replayed graph itself carries the cap; "node_attr_post" sets the
# same attribute on the kept node handles after the capture ends, before the instantiation; "capture_pref" sets PreferL1 once
# before the capture of a key and resets PreferNone right after the capture ends (never around the replays); "none" (the
# previous default of record) touches no preference.  Degrade: a missing entry point or a capture-info
# reply without exactly one kernel-node dependency leaves that launch uncapped (one RuntimeWarning per process; the call records
# "unavailable" + cache_config_degraded), never an eager fallback.  Every call records the form as LAST_CALL[kind]["cache_config"]
# ("bracket" for an eager call under a capped geometry, "node_attr" / "node_attr_post" / "capture_pref" / "none" for a captured
# call and its replays, "unavailable" without a loadable CUDA runtime or degraded); CACHE_CONFIG_STATS counts them.

GRAPH_STATS: dict[str, int] = {"eager": 0, "capture": 0, "replay": 0, "fallback": 0}
GRAPH_CACHE_MAX = 64  # keys kept per process; the oldest entry goes when a new key arrives beyond this


class _GraphEntry:
    """One validity key: its graph (after the capture), the static results and the launch inventory of the capture."""

    __slots__ = (
        "key",
        "pins",
        "seen",
        "graph",
        "result",
        "record",
        "disabled",
        "digest",
    )

    def __init__(self, key: tuple, pins: tuple) -> None:
        self.key, self.pins = (
            key,
            pins,
        )  # the pins keep the keyed objects alive so their ids cannot be reused
        self.seen = 0
        self.graph: torch.cuda.CUDAGraph | None = None
        self.result: tuple | None = None
        self.record: dict[str, Any] | None = None
        self.disabled: str | None = None
        self.digest = f"{key[0]}:{hash(key) & 0xFFFFFFFFFFFF:012x}"


_GRAPHS: dict[tuple, _GraphEntry] = {}


def _sig(t: Any) -> tuple:
    return (
        (t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype), t.device.index)
        if isinstance(t, torch.Tensor)
        else ("none",)
    )


def _graph_entry(
    kind: str,
    ctx: Any,
    ws: WorkspaceDetKcp | None,
    opts: OptionsDetKcp,
    scale: float | None,
    tensors: Sequence[Any],
    flags: tuple,
) -> _GraphEntry | None:
    """The cache entry of this call's validity key, or ``None`` when the call is never captured."""
    if (
        not isinstance(ws, WorkspaceDetKcp)
        or opts.debug_sequence
        or opts.exchange == "ring"
    ):
        return None
    backend = get_backend()
    key = (
        kind,
        id(ctx),
        id(ws),
        id(backend),
        id(ws.local),
        tuple(
            (n, t.data_ptr(), t.numel())
            for n, t in ws._pools.items()
            if n
            not in cake_kcp_plan.LAZY_POOLS  # read by ring calls only, which are never captured
        ),
        tuple(getattr(opts, f.name) for f in dataclasses.fields(opts)),
        None if scale is None else float(scale),
        flags,
        tuple(_sig(t) for t in tensors),
    )
    e = _GRAPHS.get(key)
    if e is None:
        while len(_GRAPHS) >= GRAPH_CACHE_MAX:
            del _GRAPHS[next(iter(_GRAPHS))]
        e = _GRAPHS[key] = _GraphEntry(key, (ctx, ws, backend, ws.local))
    return e


def _graph_cap_bracket(call: CallDetKcp) -> tuple[str, int | None]:
    """The device cache-config preference around a capture of ``call``: ``("capture_pref", WALK_CACHE_CONFIG_CAP)``
    under ``graph_cap="capture_pref"`` with a capped geometry and a loadable CUDA runtime (PreferL1 for the whole capture,
    PreferNone right after it ends), ``("unavailable", None)`` when no libcudart soname loads, ``("none", None)``
    otherwise (``graph_cap="none"``, or a geometry without capped programs); ``("node_attr", None)`` / ``("node_attr_post", None)``
    under the kernel-node forms (``"node_attr"`` shipped: no device preference, the cap is a launch
    attribute of the captured nodes; ``("unavailable", None)`` without the capture-graph entry points)."""
    if not WALK_CAP_PROGRAMS.get(call.walk_geometry):
        return "none", None
    form = call.opts.graph_cap
    if (
        form in GRAPH_NODE_CAPS
    ):  # the carveout attribute on the captured tape kernel nodes; no device preference
        if _cudart_graph_available() is None or (
            form == "node_attr_post"
            and not hasattr(torch.cuda.CUDAGraph, "instantiate")
        ):
            return "unavailable", None
        return form, None
    if form != "capture_pref":
        return "none", None
    if _cudart_available() is None:
        return "unavailable", None
    return "capture_pref", WALK_CACHE_CONFIG_CAP


def _graph_capture(kind: str, entry: _GraphEntry, call: CallDetKcp, body: Any) -> Any:
    """Capture ``body()`` (the call's allocations, launches and collectives) into a CUDA graph on the capture's side
    stream, replay it once and keep it for the key; returns the body's result, or ``None`` when the capture failed
    (recorded in the entry; the key then runs eagerly for the rest of the process).  The cache-config preference of the
    capture is :func:`_graph_cap_bracket`'s: set before the capture begins, reset right after it ends, never inside it;
    the call records the form and the set / reset evidence."""
    cache_config, pref = _graph_cap_bracket(call)
    post = (
        cache_config == "node_attr_post"
    )  # the node attributes go on after the capture, before the instantiation
    g = torch.cuda.CUDAGraph(keep_graph=True) if post else torch.cuda.CUDAGraph()
    dev = (
        call.device.index
        if call.device.index is not None
        else torch.cuda.current_device()
    )
    bracket: list[dict[str, Any]] = []
    try:
        if pref is not None:
            bracket.append(walk_device_cache_config(dev, pref))
        try:
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                result = body()
        finally:
            if pref is not None:
                bracket.append(walk_device_cache_config(dev, WALK_CACHE_CONFIG_DEFAULT))
        if post:
            programs_of = getattr(call.backend, "programs", None)
            programs = programs_of() if callable(programs_of) else None
            call.scratch["node_attr_applied"] = (
                programs.apply_node_caps(call)
                if isinstance(programs, _FiPrograms)
                else 0
            )
            g.instantiate()
    except Exception as exc:  # noqa: BLE001 - the failure is recorded; the caller runs the eager path
        entry.disabled = f"{type(exc).__name__}: {str(exc)[:200]}"
        GRAPH_STATS["fallback"] += 1
        return None
    g.replay()
    if call.scratch.get(
        "node_attr_degraded"
    ):  # a tape launch of this capture stayed uncapped
        cache_config = "unavailable"
    call.scratch["cache_config"] = cache_config
    call.scratch["cache_config_bracket"] = bracket or None
    _record(kind, call)
    LAST_CALL[kind].update(graph_event="capture", graph_key=entry.digest)
    entry.record = dict(LAST_CALL[kind])
    entry.graph = g
    if call.local is not None and call.local is not call.ws.local:
        # a plan view of the arena: the graph holds its scratch table set, so the entry keeps the view alive
        entry.pins += (call.local,)
    entry.seen += 1
    GRAPH_STATS["capture"] += 1
    call.backend.end_call(call)
    return result


def _graph_replay(kind: str, entry: _GraphEntry) -> None:
    entry.graph.replay()
    GRAPH_STATS["replay"] += 1
    LAST_CALL[kind] = {**entry.record, "graph_event": "replay"}


def _graph_eager(kind: str, entry: _GraphEntry | None) -> None:
    """Bookkeeping of an eager call under the knob (the first call of a key, a disabled key, or a call never captured)."""
    GRAPH_STATS["eager"] += 1
    if entry is not None:
        entry.seen += 1
        LAST_CALL[kind].update(graph_key=entry.digest, graph_fallback=entry.disabled)


def graph_cache_clear() -> int:
    """Drop every cached graph with its static buffers and private memory pool; returns the number of entries dropped.

    A graph that captured a NCCL collective must be destroyed before the process group it used is destroyed
    (``ncclCommDestroy`` waits for graph-captured work; ``torch.distributed.destroy_process_group()`` hangs while such a
    graph is alive).  Call this before destroying the process group when the ``graph`` knob was on, and whenever the operands
    of a key are gone for good to release its pool.  The knob setting and ``GRAPH_STATS`` are left as they are."""
    n = len(_GRAPHS)
    _GRAPHS.clear()
    return n


def _graph_release(pinned: Any) -> int:
    """Drop the cached graphs whose key pins ``pinned``: a local arena that :meth:`WorkspaceDetKcp.ensure` supersedes, or
    a workspace whose :meth:`WorkspaceDetKcp.pool` has just allocated a forward-only lazy pool
    (``cake_kcp_plan.FWD_LAZY_POOLS``, part of every key): no later call can match their keys, and their pins (and
    graphs) would keep that storage alive.  The dropped entries also let go of their pins (the entry of the call doing
    the upgrade, created before its ``ensure``, is still referenced by that call): once the arena is freed its id may
    be reused, and a later call that builds an equal key finds no entry and starts a new one.  Returns the number
    dropped.  A pinned plan view of the arena ``pinned`` (:func:`_plan_view`, the same ``arena`` tensor) matches
    too."""
    arena = pinned.arena if isinstance(pinned, cake_backend.KdaWorkspace) else None
    stale = [
        key
        for key, e in _GRAPHS.items()
        if any(
            p is pinned
            or (
                arena is not None
                and isinstance(p, cake_backend.KdaWorkspace)
                and p.arena is arena
            )
            for p in e.pins
        )
    ]
    for key in stale:
        _GRAPHS.pop(key).pins = ()
    return len(stale)


# ----------------------------------------------------------------------------------------------------------------
# Entry points (the KCP contract §3.2 / the design note §10.1)


def _gate_args(
    use_gate_in_kernel: bool,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
) -> dict[str, Any]:
    """The entry points' gate keywords as one dict (passed through to :func:`_prepare`)."""
    return {
        "use_gate_in_kernel": use_gate_in_kernel,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "lower_bound": lower_bound,
    }


def _gate_key_tensors(gate_args: dict[str, Any]) -> tuple[Any, ...]:
    """The raw mode's parameter tensors in a graph validity key (their pointers, shapes, strides and dtypes; nothing
    in ``"g"`` mode, whose key is unchanged)."""
    if gate_args["use_gate_in_kernel"] is not True:
        return ()
    return (gate_args["A_log"], gate_args["dt_bias"])


def _gate_key_flags(gate_args: dict[str, Any]) -> tuple[Any, ...]:
    """The raw mode's host scalars in a graph validity key: the mode and the lower bound (a captured call binds the
    lower bound as a launch argument; nothing in ``"g"`` mode)."""
    if gate_args["use_gate_in_kernel"] is not True:
        return ()
    lb = gate_args["lower_bound"]
    return ("raw", lb if isinstance(lb, (int, float)) else repr(lb))


def _prepare(
    q,
    k,
    v,
    g,
    beta,
    ctx,
    *,
    scale,
    save_mode,
    workspace,
    options,
    backward,
    dout=None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
):
    backend = get_backend()
    opts = options if isinstance(options, OptionsDetKcp) else _options(options)
    if save_mode not in SAVE_MODES:
        raise ValueError(f"save_mode must be one of {SAVE_MODES}, got {save_mode!r}")
    if save_mode != "light":
        raise NotImplementedError(
            "save_mode='tape' is not supported on the KCP route (light only)"
        )
    gate = cake_backend._gate_mode(use_gate_in_kernel)
    if gate == "raw" and not (
        backend.local_stages and backend.dtypes.get("g_raw") == torch.bfloat16
    ):
        raise NotImplementedError(
            f"use_gate_in_kernel=True needs a backend with the local kernel stages (backend {backend.name!r})"
        )
    q, k, v, g, beta, T, H, batched = _operands(
        q, k, v, g, beta, ctx, dtypes=backend.dtypes, gate=gate
    )
    device = q.device
    gp = cake_backend._gate_params(gate, A_log, dt_bias, lower_bound, H, device)
    if dout is not None:
        dout = _squeeze_batch(dout, "dout", 3)
        _check(dout, "dout", (T, H, D), backend.dtypes["dout"], device)
    if workspace is None:
        workspace = WorkspaceDetKcp(
            state_dtype=backend.dtypes["state"],
            local=backend.local_stages,
            save_mode=save_mode,
            use_gate_in_kernel=gate == "raw",
        )
    elif not isinstance(workspace, WorkspaceDetKcp):
        raise TypeError("workspace must be a WorkspaceDetKcp")
    elif workspace.gate != gate:
        raise ValueError(
            f"workspace: allocated for gate mode {workspace.gate!r} "
            f"(use_gate_in_kernel={workspace.gate == 'raw'}), the call runs gate mode {gate!r} "
            f"(use_gate_in_kernel={gate == 'raw'}); allocate the WorkspaceDetKcp with "
            f"use_gate_in_kernel={gate == 'raw'}"
        )
    if workspace.state_dtype != backend.dtypes["state"]:
        raise ValueError(
            f"workspace state dtype {workspace.state_dtype} differs from the backend's {backend.dtypes['state']}"
        )
    wait_ready = getattr(ctx, "wait_ready", None)
    # a context prepared on another stream: order its uploads on this one (enqueue only)
    if wait_ready is not None:
        wait_ready()
    tb = workspace.ensure(
        ctx,
        H,
        device=device,
        backward=backward,
        fused_walks=opts.fused_walks,
        walk_geometry=opts.walk_geometry,
        host_trim=bool(opts.host_trim),
    )
    if gp is None:
        ops = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
    else:  # the raw records' operands: the raw view, its element strides, A_log, dt_bias [H * 128], lower_bound
        ops = {"q": q, "k": k, "v": v, "beta": beta, **cake_backend._raw_values(g, gp)}
    if dout is not None:
        ops["dout"] = dout
    call = CallDetKcp(
        ctx=ctx,
        H=H,
        ws=workspace,
        tb=tb,
        backend=backend,
        opts=opts,
        scale=SCALE if scale is None else float(scale),
        ops=ops,
        backward=backward,
    )
    call.gate, call.gate_params = gate, gp
    backend.begin_call(call)
    return call, batched


def chunk_kda_det_fwd_kcp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    *,
    scale: float | None = None,
    save_for_backward: bool = True,
    save_mode: str = "light",
    boundary_mode: str = "saved",
    workspace: WorkspaceDetKcp | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    **options: Any,
) -> tuple[torch.Tensor, SavedDetKcp | None]:
    """The KCP forward of this rank's fragments (the KCP contract §3.2): ``out`` like ``v`` (fresh) and the saved set
    (``None`` without ``save_for_backward`` or in ``boundary_mode="replay"``).  ``ctx`` is this rank's
    ``ContextDetKcp``; the token operands are its ``T_r`` local tokens in fragment order; ``**options`` are the
    keywords of :class:`OptionsDetKcp` (+ ``_kcp_test_knobs``), among them ``graph="replay"`` (keyed CUDA-graph
    capture / replay of this call; module docstring and :func:`graph_cache_clear`).

    ``use_gate_in_kernel=True`` (raw gate mode, the single-rank keywords of ``cake_backend.chunk_kda_det_fwd``):
    ``g`` is this rank's raw gate BF16 ``[T_r, H, 128]`` (a strided view under the single-rank rule, never copied)
    with ``A_log`` FP32 ``[H]``, ``dt_bias`` FP32 ``[H * 128]`` or ``[H, 128]`` and ``lower_bound`` (``-5 <=
    lower_bound < 0``, required); the local intra program activates the gate in its prefix pass, every other launch
    and collective is the explicit path's.  The workspace must be of the same mode (:attr:`WorkspaceDetKcp.gate`)
    and the saved set records the mode and the lower bound.  With ``use_gate_in_kernel=False`` the three parameters
    must be ``None``."""
    if boundary_mode not in BOUNDARY_MODES:
        raise ValueError(
            f"boundary_mode must be one of {BOUNDARY_MODES}, got {boundary_mode!r}"
        )
    opts = _options(options)
    gate_args = _gate_args(use_gate_in_kernel, A_log, dt_bias, lower_bound)
    save = bool(save_for_backward) and boundary_mode == "saved"
    entry = (
        _graph_entry(
            "fwd",
            ctx,
            workspace,
            opts,
            scale,
            (q, k, v, g, beta, *_gate_key_tensors(gate_args)),
            (save, save_mode, *_gate_key_flags(gate_args)),
        )
        if opts.graph != "off"
        else None
    )
    if entry is not None and entry.graph is not None:
        _graph_replay("fwd", entry)
        out, saved, batched = entry.result
        return _ranked(out.detach(), batched), saved
    call, batched = _prepare(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        scale=scale,
        save_mode=save_mode,
        workspace=workspace,
        options=opts,
        backward=False,
        **gate_args,
    )
    if entry is not None and entry.seen > 0 and entry.disabled is None:
        res = _graph_capture(
            "fwd",
            entry,
            call,
            lambda: _fwd_body(call, save=save, save_mode=save_mode, ctx=ctx),
        )
        if res is not None:
            entry.result = (res[0], res[1], batched)
            return _ranked(res[0].detach(), batched), res[1]
        call, batched = _prepare(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            scale=scale,
            save_mode=save_mode,
            workspace=workspace,
            options=opts,
            backward=False,
            **gate_args,
        )
    out, saved = _fwd_body(call, save=save, save_mode=save_mode, ctx=ctx)
    _record("fwd", call)
    if opts.graph != "off":
        _graph_eager("fwd", entry)
    call.backend.end_call(call)
    return _ranked(out, batched), saved


def _fwd_body(
    call: CallDetKcp,
    *,
    save: bool,
    save_mode: str,
    ctx: Any,
    bound: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, SavedDetKcp | None]:
    """The forward's allocations and launches (eagerly or inside a capture): ``out`` and the saved set.  ``bound``:
    the tensors of a recompute handle's entry (:func:`_forward`); the saved set then holds them."""
    call.out = _alloc((call.T_r, call.H, D), call.backend.dtypes["out"], call.device)
    _forward(call, save=save, replay=False, bound=bound)
    saved = None
    if save:
        tb, opts = call.tb, call.opts
        saved = SavedDetKcp(
            version=KCP_SAVED_VERSION,
            S_in=call.bufs["S_in"],
            c1_recv=call.bufs.get("c1_recv") if opts.save_transfer else None,
            M_seg=call.bufs.get("stage_m")
            if (opts.save_transfer and tb.P > 1)
            else None,
            carry=call.bufs.get("carry")
            if (tb.P > 1 and call.saved_carry == 1)
            else None,
            T_r=call.T_r,
            H=call.H,
            n_frag=tb.n_frag,
            P=tb.P,
            Hg=tb.Hg,
            scale=call.scale,
            save_mode=save_mode,
            save_transfer=opts.save_transfer,
            offset_class=opts.offset_class,
            exchange=call.exchange,
            precision_mode=call.precision_mode,
            ctx=ctx,
            transfer_walk=call.transfer_walk,
            gate=call.gate,
            lower_bound=None
            if call.gate_params is None
            else call.gate_params.lower_bound,
        )
    return call.out, saved


def _check_saved(saved: SavedDetKcp, call: CallDetKcp) -> None:
    tb, opts = call.tb, call.opts
    if saved.version != KCP_SAVED_VERSION:
        raise ValueError(f"saved: version {saved.version} is not {KCP_SAVED_VERSION}")
    if saved.ctx is not call.ctx:
        raise ValueError("saved: the saved set belongs to another ContextDetKcp")
    if (saved.T_r, saved.H, saved.n_frag, saved.P) != (
        call.T_r,
        call.H,
        tb.n_frag,
        tb.P,
    ):
        raise ValueError(
            "saved: the saved set describes another problem (T_r, H, n_frag, P)"
        )
    if saved.scale != call.scale:
        raise ValueError(
            f"saved: scale {saved.scale} differs from the call's {call.scale}"
        )
    # the gate mode and the FP32 bits of the raw mode's lower bound (the single-rank rule)
    cake_backend._check_saved_gate(saved, call.gate_params)
    if (
        saved.save_transfer != opts.save_transfer
        or saved.exchange != call.exchange
        or saved.precision_mode != call.precision_mode
        or saved.transfer_walk != call.transfer_walk
    ):
        raise ValueError(
            "saved: save_transfer / exchange / precision_mode / transfer_walk differ from the forward's"
        )
    _check(saved.S_in, "saved.S_in", (tb.sizes["S_in"],), call.state_dtype, call.device)
    if opts.save_transfer:
        if saved.c1_recv is None:
            raise ValueError(
                "saved: save_transfer needs the forward's c1 receive tensor"
            )
        _check(
            saved.c1_recv,
            "saved.c1_recv",
            (tb.sizes["c1_recv"],),
            call.state_dtype,
            call.device,
        )
        if tb.P > 1:
            if saved.M_seg is None:
                raise ValueError("saved: save_transfer at P > 1 needs M_seg")
            _check(
                saved.M_seg,
                "saved.M_seg",
                (tb.sizes["stage_m"],),
                call.state_dtype,
                call.device,
            )
    if saved.carry is not None:
        if tb.P == 1:
            raise ValueError("saved: a carry plane needs P > 1")
        _check(
            saved.carry,
            "saved.carry",
            (tb.sizes["carry"],),
            call.state_dtype,
            call.device,
        )


def chunk_kda_det_bwd_kcp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    dout: torch.Tensor,
    *,
    saved: SavedDetKcp | None = None,
    scale: float | None = None,
    save_mode: str = "light",
    workspace: WorkspaceDetKcp | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    **options: Any,
) -> tuple[torch.Tensor, ...]:
    """The KCP backward (the KCP contract §3.2): ``(dq, dk, dv, dbeta, dg)``, fresh.  ``saved=None`` replays the forward.

    Raw gate mode (``use_gate_in_kernel=True`` with the forward's ``A_log`` / ``dt_bias`` / ``lower_bound``):
    ``(dq, dk, dv, dbeta, dg, dA_log, ddt_bias)``, ``dg`` the BF16 gradient of the raw gate (contiguous, ranked like
    ``g``), ``dA_log`` FP32 ``[H]`` and ``ddt_bias`` FP32 shaped like ``dt_bias``.  The two parameter gradients are
    this rank's partial sums: the gradients through this rank's own tokens, reduced on the rank in an order fixed by
    its local chunks alone (bitwise repeatable per rank, the same under every schedule and exchange).  The full
    gradient is their sum over the ranks of ``ctx.group``; a training framework sums them as it sums the gradients
    of any parameter replicated across the context-parallel ranks, or calls :func:`reduce_gate_param_grads_det_kcp` for
    a sum that is bitwise reproducible (an NCCL all-reduce does not fix its summation order).  No gate epilogue runs
    and no full-size FP32 gate or gate gradient exists on the route.  A saved set of the other mode, or with another
    ``lower_bound`` (its FP32 value), is refused with ``ValueError``, as is a workspace of the other mode."""
    opts = _options(options)
    gate_args = _gate_args(use_gate_in_kernel, A_log, dt_bias, lower_bound)
    entry = None
    if opts.graph != "off":
        kept = (
            ()
            if saved is None
            else (saved.S_in, saved.c1_recv, saved.M_seg, saved.carry)
        )  # the validity key sees every saved tensor (the carry plane of saved_carry=1 included)
        entry = _graph_entry(
            "bwd",
            ctx,
            workspace,
            opts,
            scale,
            (q, k, v, g, beta, dout, *kept, *_gate_key_tensors(gate_args)),
            (save_mode, saved is not None, *_gate_key_flags(gate_args)),
        )
    if entry is not None and entry.graph is not None:
        _graph_replay("bwd", entry)
        grads, batched = entry.result
        return _shaped_grads(grads, batched, g, gate_args)
    call, batched = _prepare(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        scale=scale,
        save_mode=save_mode,
        workspace=workspace,
        options=opts,
        backward=True,
        dout=dout,
        **gate_args,
    )
    if saved is not None:
        _check_saved(saved, call)
    if (
        entry is not None
        and entry.seen > 0
        and entry.disabled is None
        and not (
            saved is not None
            and opts.spec3_probe != "off"
            and call.local is not None
            and call.T_r != 0
            and getattr(call.backend, "_spec3_pending", False)
        )
    ):  # SPEC-3's host compare stays eager; a rank without tokens runs no local stage (CallDetKcp.stage), so it has
        # nothing to probe and captures with the other ranks
        res = _graph_capture("bwd", entry, call, lambda: _bwd_body(call, saved))
        if res is not None:
            entry.result = (res, batched)
            return _shaped_grads(res, batched, g, gate_args)
        call, batched = _prepare(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            scale=scale,
            save_mode=save_mode,
            workspace=workspace,
            options=opts,
            backward=True,
            dout=dout,
            **gate_args,
        )
        if saved is not None:
            _check_saved(saved, call)
    grads = _bwd_body(call, saved)
    _record("bwd", call)
    if opts.graph != "off":
        _graph_eager("bwd", entry)
    call.backend.end_call(call)
    return _shaped_grads(grads, batched, g, gate_args, detach=False)


def _shaped_grads(
    grads: Sequence[torch.Tensor],
    batched: bool,
    g: torch.Tensor,
    gate_args: dict[str, Any],
    *,
    detach: bool = True,
) -> tuple[torch.Tensor, ...]:
    """The backward's results as the caller receives them: the token gradients ranked like the inputs (the raw
    gate's like ``g``), ``dA_log`` as is and ``ddt_bias`` in ``dt_bias``'s shape (``detach``: a graph's static
    tensors, handed out as fresh objects)."""

    def out(t: torch.Tensor) -> torch.Tensor:
        return t.detach() if detach else t

    if len(grads) == 5:
        return tuple(_ranked(out(t), batched) for t in grads)
    dq, dk, dv, dbeta, d_raw, dA_log, ddt_bias = grads
    return (
        *(_ranked(out(t), batched) for t in (dq, dk, dv, dbeta)),
        _ranked(out(d_raw), g.dim() == 4),
        out(dA_log),
        out(ddt_bias).view(tuple(gate_args["dt_bias"].shape)),
    )


_RAW_GRADS = ("dq", "dk", "dv", "dbeta", "dg_raw", "dA_log", "ddt_bias")


def _bwd_body(call: CallDetKcp, saved: SavedDetKcp | None) -> tuple[torch.Tensor, ...]:
    """The backward's allocations and launches (eagerly or inside a capture): ``(dq, dk, dv, dbeta, dg)``; in raw
    gate mode ``(dq, dk, dv, dbeta, dg_raw, dA_log, ddt_bias)`` with ``dg_raw`` the BF16 gradient of the raw gate
    (contiguous ``[T_r, H, 128]``), ``dA_log`` FP32 ``[H]`` and ``ddt_bias`` FP32 ``[H * 128]``: this rank's partial
    sums (the parameter gradients of its own tokens).  The raw GLM intra program writes the per-chunk ``(hi, lo)``
    partials ``dA_part`` / ``db_part`` (FP32 ``[NC_pad, H, 128, 2]`` each, per call, outside the arena; the real
    chunks only) and ``param_finalize`` reduces them in the fixed order of the local chunks, with its per-call
    scratch ``part2`` / ``ticket`` and ``seg_cap`` from ``cake_backend._raw_finalize_scratch``; a rank without tokens
    launches nothing and returns zero parameter gradients."""
    dt = call.backend.dtypes
    T, H = call.T_r, call.H
    call.grads = {
        "dq": _alloc((T, H, D), dt["dq"], call.device),
        "dk": _alloc((T, H, D), dt["dk"], call.device),
        "dv": _alloc((T, H, D), dt["dv"], call.device),
    }
    if call.gate == "raw":
        call.grads["dg_raw"] = _alloc((T, H, D), dt["dg_raw"], call.device)
        call.grads["dbeta"] = _alloc((T, H), dt["dbeta"], call.device)
        if call.local is not None and T > 0:
            nc_pad = int(call.local.plan.NC_pad)
            for name in ("dA_part", "db_part"):
                call.grads[name] = _alloc((nc_pad, H, D, 2), dt["dA_log"], call.device)
            # param_finalize's per-call scratch: the second-stage partials and the per-head ticket
            fin = cake_backend._raw_finalize_scratch(nc_pad, H, call.device)
            call.grads["part2"] = fin["part2"]
            call.grads["ticket"] = fin["ticket"]
            call.scratch["seg_cap"] = int(fin["seg_cap"])
            call.grads["dA_log"] = _alloc((H,), dt["dA_log"], call.device)
            call.grads["ddt_bias"] = _alloc((H * D,), dt["ddt_bias"], call.device)
        else:  # no local stage runs on this rank: its tokens contribute nothing
            call.grads["dA_log"] = _zeros((H,), dt["dA_log"], call.device)
            call.grads["ddt_bias"] = _zeros((H * D,), dt["ddt_bias"], call.device)
        _backward(call, saved)
        return tuple(call.grads[n] for n in _RAW_GRADS)
    call.grads["dg"] = _alloc((T, H, D), dt["dg"], call.device)
    call.grads["dbeta"] = _alloc((T, H), dt["dbeta"], call.device)
    _backward(call, saved)
    return tuple(call.grads[n] for n in ("dq", "dk", "dv", "dbeta", "dg"))


# ----------------------------------------------------------------------------------------------------------------
# Checkpoint recompute without the state exchange: the recompute handle (``recompute_handle=``; issue #7815, GDN #7818)
#
# The phase comes from ``torch._C._current_graph_task_id()`` (checked once: present and returning an int; without it a
# call with a handle raises): -1 outside the autograd engine (an original forward), the running graph task inside a
# backward (torch's checkpoint recompute, reentrant or not).  Rules of a call with a handle:
#   * no "kda" entry -> an original forward (also one inside an outer checkpoint's recompute, nested regions):
#     exchange as always, then store ``S_in`` (and under ``checkpoint_keep="transfer"`` the c1 receive tensor) --
#     the saved set's own tensors, no copy -- with the call's context and static fingerprint;
#   * an entry inside the backward -> the recompute: check the context and the fingerprint, bind the stored tensors,
#     issue no collective; the entry stays (``retain_graph`` / a second backward binds it again);
#   * an entry outside the backward -> raise (a handle reused across micro-steps or for a second KDA call).
# Graph replay (keyword or knob), a forward inside a CUDA-graph capture, ``debug_sequence``, ``spec3_probe`` and
# ``boundary_mode="replay"`` are refused with a handle, in the autograd Function itself (:func:`_handle_refusals`).
# The ranks are not checked against each other: the forward exchange is a collective, so every rank of the group must
# take the same branch (all with a handle at the same nesting level, or all without); a mismatch hangs or pairs the
# wrong exchanges (documented, no check, no all-gather).

HANDLE_KEY = "kda"  # this package's entry in a recompute handle
CHECKPOINT_KEEPS = cake_kcp_plan.CHECKPOINT_KEEPS  # "states" (default) | "transfer"
CHECKPOINT_KEEP_DEFAULT = "states"

_GRAPH_TASK_ID_BINDING = getattr(torch._C, "_current_graph_task_id", None)
_GRAPH_TASK_ID_USABLE: bool | None = None  # None = not checked yet


class RecomputeHandleDetKcp:
    """The boundary tensors of one checkpointed call, carried from its original forward to torch's recompute of that
    forward (``chunk_kda_det_kcp(..., recompute_handle=handle)``), so the recompute runs without cross-rank
    communication.

    Create one handle per layer and micro-step, outside the outermost checkpoint region, and pass it as an argument
    of the checkpointed function, so the recompute sees the same object::

        def layer(q, k, v, g, beta, handle):
            return chunk_kda_det_kcp(q, k, v, g, beta, ctx, recompute_handle=handle)

        out = torch.utils.checkpoint.checkpoint(layer, q, k, v, g, beta, RecomputeHandleDetKcp(), use_reentrant=False)

    ``entries`` maps an operation key to that operation's entry: ``"kda"`` is this package's (the forward's incoming
    states ``S_in`` -- and with ``checkpoint_keep="transfer"`` the owner's c1 receive tensor --, the very tensors
    the saved set holds, plus the call's context and fingerprint); other packages that accept a handle with an
    ``entries`` dict use their own keys (the causal convolution's history rows: ``"conv"``), so one handle may serve
    the KDA call and one convolution call of the same layer.  A handle holds one entry per key.  A handle created
    inside a checkpoint region is a new, empty one in that region's recompute, which then exchanges again (correct,
    not communication-free).  The entries live as long as the handle; :meth:`nbytes` is the retained size
    (``kda_det_saved_bytes_kcp(ctx, H=..., checkpoint_keep=..., with_conv=False)`` for the KDA entry).

    The forward state exchange is a collective.  Every rank of ``ctx.group`` must pass a handle for the same call, or
    no rank passes one, and every rank creates it at the same checkpoint nesting level.  A mismatch pairs one rank's
    recompute exchange with another rank's backward exchange: the ranks hang or exchange wrong data.  No check is
    made across ranks.  Never share one handle between two KDA calls (also not in nested regions).  Never call
    :meth:`clear` on a handle, or reuse it, while a backward that needs its entry is pending."""

    __slots__ = ("entries",)

    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}

    def clear(self) -> None:
        """Drop every entry (the retained tensors go with the last reference)."""
        self.entries.clear()

    def nbytes(self) -> int:
        """Bytes of the distinct tensors the entries hold."""
        seen: dict[int, int] = {}
        for entry in self.entries.values():
            for t in entry.values():
                for x in t if isinstance(t, (tuple, list)) else (t,):
                    if isinstance(x, torch.Tensor):
                        seen[id(x)] = x.numel() * x.element_size()
        return sum(seen.values())


def _kcp_in_backward() -> bool:
    """Whether this thread runs inside the autograd engine's backward pass (torch's checkpoint recompute runs there,
    an original forward does not).  The one place the recompute handle reads the phase.  Raises without the torch
    binding (checked once)."""
    global _GRAPH_TASK_ID_USABLE
    if _GRAPH_TASK_ID_USABLE is None:
        ok = False
        if _GRAPH_TASK_ID_BINDING is not None:
            try:
                gid = _GRAPH_TASK_ID_BINDING()
                ok = isinstance(gid, int) and not isinstance(gid, bool)
            except Exception:  # noqa: BLE001 - an unusable binding disables the option
                ok = False
        _GRAPH_TASK_ID_USABLE = ok
    if not _GRAPH_TASK_ID_USABLE:
        raise RuntimeError(
            "recompute_handle needs torch._C._current_graph_task_id to tell the recompute from the original "
            "forward; this torch does not provide it"
        )
    return int(_GRAPH_TASK_ID_BINDING()) != -1


def _handle_refusals(opts: OptionsDetKcp, boundary_mode: str, handle: Any) -> None:
    """The refusals of a call with a handle, shared by :func:`chunk_kda_det_kcp` and the autograd Function: graph replay
    (keyword or test knob), a forward inside a CUDA-graph capture, ``debug_sequence``, ``spec3_probe``,
    ``boundary_mode="replay"`` and a handle without an ``entries`` dict."""
    if not isinstance(getattr(handle, "entries", None), dict):
        raise TypeError(
            f"recompute_handle must carry an 'entries' dict (RecomputeHandleDetKcp), got {type(handle).__name__}"
        )
    if opts.graph != "off":
        raise ValueError(
            f"recompute_handle needs graph='off' (keyword and test knob), got {opts.graph!r}"
        )
    if _capturing():
        raise ValueError(
            "recompute_handle is refused inside a CUDA-graph capture (a replay would overwrite the stored states)"
        )
    if boundary_mode != "saved":
        raise ValueError(
            f"recompute_handle needs boundary_mode='saved' (the replay mode keeps no state), got {boundary_mode!r}"
        )
    if opts.debug_sequence:
        raise ValueError(
            "recompute_handle refuses debug_sequence (it gathers the issue order across ranks)"
        )
    if opts.spec3_probe != "off":
        raise ValueError(
            f"recompute_handle refuses spec3_probe (a debug probe), got {opts.spec3_probe!r}"
        )


def _handle_options(
    opts: OptionsDetKcp, checkpoint_keep: str | None, explicit_save_transfer: Any
) -> OptionsDetKcp:
    """The options of a call with a handle: ``checkpoint_keep`` (default ``"states"``) fixes ``save_transfer``
    (``"states"`` -> ``False``, ``"transfer"`` -> ``True``); an explicit ``save_transfer`` that contradicts it, and
    ``"transfer"`` with the ring exchange (no c1 receive tensor), are refused."""
    keep = CHECKPOINT_KEEP_DEFAULT if checkpoint_keep is None else checkpoint_keep
    if keep not in CHECKPOINT_KEEPS:
        raise ValueError(
            f"checkpoint_keep must be one of {CHECKPOINT_KEEPS}, got {checkpoint_keep!r}"
        )
    transfer = keep == "transfer"
    if explicit_save_transfer is not None and bool(explicit_save_transfer) != transfer:
        raise ValueError(
            f"checkpoint_keep={keep!r} runs the call with save_transfer={transfer}; "
            f"save_transfer={explicit_save_transfer!r} contradicts it"
        )
    if transfer and opts.exchange == "ring":
        raise ValueError(
            "checkpoint_keep='transfer' keeps the c1 receive tensor; exchange='ring' receives none (use 'states')"
        )
    if opts.save_transfer == transfer:
        return opts
    return dataclasses.replace(opts, save_transfer=transfer)


def _handle_lookup(handle: Any) -> dict[str, Any] | None:
    """The handle's ``"kda"`` entry when this forward is the recompute of the forward that stored it, ``None`` for an
    original forward (no entry yet: it exchanges and stores one; also a forward inside an outer checkpoint's
    recompute).  An entry found outside the backward raises."""
    in_backward = _kcp_in_backward()
    entry = handle.entries.get(HANDLE_KEY)
    if entry is None:
        return None
    if not in_backward:
        raise RuntimeError(
            "recompute_handle: the handle already holds a 'kda' entry outside the backward -- a handle serves one "
            "KDA call of one micro-step; create a new RecomputeHandleDetKcp per layer and micro-step"
        )
    return entry


def _handle_fingerprint(
    call: CallDetKcp, *, save_mode: str, keep: str, **modes: Any
) -> tuple:
    """The static configuration of a call that a handle entry must match to stand in for the recompute's exchange
    (host data only): the problem (``T_r``, ``H``, the fragment and segment counts, the operands' shapes and dtypes),
    ``scale``, ``save_mode``, the exchange, the tier and every option.  The context is compared by identity beside
    it.  ``modes``: keywords of the entry point that select another form of the recurrence (a gate mode, an in-kernel
    q / k normalization); every such keyword joins the fingerprint here, so a replay never binds states captured
    under another mode."""
    tb = call.tb
    return (
        ("T_r", call.T_r),
        ("H", call.H),
        ("n_frag", tb.n_frag),
        ("P", tb.P),
        ("NSEG", tb.NSEG),
        ("scale", call.scale),
        ("save_mode", save_mode),
        ("exchange", call.exchange),
        ("keep", keep),
        (
            "operands",
            tuple(
                (n, tuple(t.shape), t.dtype)
                for n, t in call.ops.items()
                # tensors only: the raw mode's lower bound goes under "modes" (with the mode); its stride
                # ints are host scalars of the layout only
                if isinstance(t, torch.Tensor)
            ),
        ),
        ("options", tuple(sorted(dataclasses.asdict(call.opts).items()))),
        ("modes", tuple(sorted(modes.items()))),
    )


def _handle_bind(
    entry: dict[str, Any], call: CallDetKcp, fingerprint: tuple, keep: str
) -> dict[str, torch.Tensor]:
    """The stored tensors of ``entry``, checked against this recompute: the same context, the same fingerprint and
    the shapes / dtype / device of the call's pools."""
    if entry.get("ctx") is not call.ctx:
        raise ValueError(
            "recompute_handle: the 'kda' entry belongs to another ContextDetKcp"
        )
    stored = entry.get("fingerprint")
    if stored != fingerprint:
        differ = (
            sorted(
                k
                for k, v in dict(fingerprint).items()
                if dict(stored).get(k, object()) != v
            )
            if isinstance(stored, tuple)
            else ["fingerprint"]
        )
        raise ValueError(
            f"recompute_handle: the 'kda' entry was stored by another call configuration (differs in {differ})"
        )
    names = ("S_in", "c1_recv") if keep == "transfer" else ("S_in",)
    bound = {}
    for name in names:
        t = entry.get(name)
        if not isinstance(t, torch.Tensor):
            raise ValueError(
                f"recompute_handle: the 'kda' entry holds no {name} tensor"
            )
        _check(
            t,
            f"recompute_handle {name}",
            (call.tb.sizes[name],),
            call.state_dtype,
            call.device,
        )
        bound[name] = t
    return bound


def _gate_modes(call: CallDetKcp) -> dict[str, Any]:
    """The gate keywords of a call in a recompute handle's fingerprint: the mode and the raw mode's FP32 lower bound
    (a recompute binds the stored states only under the same gate; ``A_log`` / ``dt_bias`` are operands)."""
    gp = call.gate_params
    return {
        "use_gate_in_kernel": call.gate == "raw",
        "lower_bound": None if gp is None else gp.lower_bound,
    }


def _fwd_handle(
    q,
    k,
    v,
    g,
    beta,
    ctx,
    *,
    scale,
    save_mode,
    workspace,
    opts,
    handle,
    gate_args: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, SavedDetKcp]:
    """The forward of a call with a handle (eager, saved mode): the original forward exchanges and stores the
    handle's entry; the recompute binds it (:func:`_forward`).  ``gate_args``: the entry point's gate keywords."""
    keep = "transfer" if opts.save_transfer else "states"
    entry = _handle_lookup(handle)
    call, batched = _prepare(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        scale=scale,
        save_mode=save_mode,
        workspace=workspace,
        options=opts,
        backward=False,
        **(gate_args or {}),
    )
    fingerprint = _handle_fingerprint(
        call, save_mode=save_mode, keep=keep, **_gate_modes(call)
    )
    bound = None if entry is None else _handle_bind(entry, call, fingerprint, keep)
    out, saved = _fwd_body(call, save=True, save_mode=save_mode, ctx=ctx, bound=bound)
    if entry is None:
        stored: dict[str, Any] = {"S_in": saved.S_in}
        if keep == "transfer":
            stored["c1_recv"] = saved.c1_recv
        stored["ctx"] = ctx
        stored["fingerprint"] = fingerprint
        handle.entries[HANDLE_KEY] = stored
    call.scratch["recompute_handle"] = "capture" if entry is None else "replay"
    _record("fwd", call)
    call.backend.end_call(call)
    return _ranked(out, batched), saved


# ----------------------------------------------------------------------------------------------------------------
# autograd


class ChunkKDADetFunctionKcp(torch.autograd.Function):
    """:func:`chunk_kda_det_kcp` as an autograd node: the forward keeps the inputs and the saved set (saved mode) or
    the inputs only (replay mode); the backward runs :func:`chunk_kda_det_bwd_kcp`.  Under
    ``torch.utils.checkpoint(use_reentrant=False)`` the forward is re-run in the backward: the same program order
    on every rank, so the collectives match (R7).  ``handle``: a recompute handle (:class:`RecomputeHandleDetKcp`):
    the original forward stores its received states in it, the recompute binds them and exchanges nothing; the
    saved set has the same members either way.  The tier follows the options' ``save_transfer`` (``False``:
    ``S_in``; ``True``: also the c1 receive tensor; :func:`chunk_kda_det_kcp` sets it from ``checkpoint_keep``).  A
    handle is refused with graph replay, inside a CUDA-graph capture, with ``debug_sequence``, ``spec3_probe`` and
    ``boundary_mode="replay"``.  Raw gate mode (``use_gate_in_kernel``): the node saves the raw gate (the caller's
    view) and ``A_log`` / ``dt_bias`` beside the KCP saved set, which records the mode and ``lower_bound``, and
    returns the gradients of the raw gate, ``A_log`` and ``dt_bias`` (the parameter gradients: this rank's partial
    sums, :func:`chunk_kda_det_bwd_kcp`)."""

    @staticmethod
    def forward(
        fctx,
        q,
        k,
        v,
        g,
        beta,
        ctx,
        scale,
        save_mode,
        boundary_mode,
        workspace,
        options,
        handle=None,
        use_gate_in_kernel=False,
        A_log=None,
        dt_bias=None,
        lower_bound=None,
    ):
        gate_args = _gate_args(use_gate_in_kernel, A_log, dt_bias, lower_bound)
        if handle is None:
            out, saved = chunk_kda_det_fwd_kcp(
                q,
                k,
                v,
                g,
                beta,
                ctx,
                scale=scale,
                save_for_backward=True,
                save_mode=save_mode,
                boundary_mode=boundary_mode,
                workspace=workspace,
                **gate_args,
                **options,
            )
        else:
            opts = _options(options)
            _handle_refusals(opts, boundary_mode, handle)
            out, saved = _fwd_handle(
                q,
                k,
                v,
                g,
                beta,
                ctx,
                scale=scale,
                save_mode=save_mode,
                workspace=workspace,
                opts=opts,
                handle=handle,
                gate_args=gate_args,
            )
        raw = use_gate_in_kernel is True
        params = (A_log, dt_bias) if raw else ()
        tensors = saved.tensors() if saved is not None else ()
        fctx.save_for_backward(q, k, v, g, beta, *params, *tensors)
        fctx.kcp = (
            ctx,
            scale,
            save_mode,
            workspace,
            dict(options),
            None if saved is None else (saved.flags(), saved.meta()),
            (raw, lower_bound),
        )
        return out

    @staticmethod
    def backward(fctx, dout):
        q, k, v, g, beta, *tensors = fctx.saved_tensors
        ctx, scale, save_mode, workspace, options, meta, (raw, lower_bound) = fctx.kcp
        A_log = dt_bias = None
        if raw:
            A_log, dt_bias, *tensors = tensors
        saved = (
            None
            if meta is None
            else SavedDetKcp.from_tensors(tensors, meta[0], meta[1], ctx)
        )
        if not dout.is_contiguous():
            dout = dout.contiguous()
        grads = chunk_kda_det_bwd_kcp(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            dout,
            saved=saved,
            scale=scale,
            save_mode=save_mode,
            workspace=workspace,
            **_gate_args(raw, A_log, dt_bias, lower_bound),
            **options,
        )
        dq, dk, dv, dbeta, dg = grads[:5]
        dA_log, ddt_bias = grads[5:] if raw else (None, None)
        return (
            dq,
            dk,
            dv,
            dg,
            dbeta,
            # ctx, scale, save_mode, boundary_mode, workspace, options, handle
            *(None,) * 7,
            None,  # use_gate_in_kernel
            dA_log,
            ddt_bias,
            None,  # lower_bound
        )


def chunk_kda_det_kcp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    scale: float | None = None,
    save_mode: str = "light",
    boundary_mode: str = "saved",
    workspace: WorkspaceDetKcp | None = None,
    *,
    recompute_handle: Any = None,
    checkpoint_keep: str | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    **options: Any,
) -> torch.Tensor:
    """The differentiable KCP forward (the KCP contract §3.2 signature): ``out`` like ``v``; gradients for ``q, k, v, g, beta``.

    ``use_gate_in_kernel=True`` (raw gate mode; keyword-only with ``A_log`` / ``dt_bias`` / ``lower_bound``, as
    :func:`chunk_kda_det_fwd_kcp`): ``g`` is the raw gate and the gradients are those of ``q, k, v``, the raw gate
    (BF16), ``beta``, ``A_log`` and ``dt_bias`` (FP32).  The parameter gradients of a rank are its partial sums
    (the gradients through its own tokens); sum them over ``ctx.group`` as for any parameter replicated across the
    context-parallel ranks, or with :func:`reduce_gate_param_grads_det_kcp` for a bitwise reproducible sum.  A recompute
    handle's entry records the gate mode and ``lower_bound``: a recompute in another mode never binds it.

    ``recompute_handle`` (keyword-only; ``None`` = unchanged): a :class:`RecomputeHandleDetKcp` (any object with an
    ``entries`` dict) makes torch's checkpoint recompute of this call run without the forward state exchange -- the
    original forward stores its incoming states under ``"kda"``, the recompute inside the backward binds them
    (bitwise the exchanging recompute); the backward still exchanges.  One handle per layer and micro-step, created
    outside the outermost checkpoint region; a forward outside the backward that finds an entry raises.
    ``checkpoint_keep`` (with a handle only): ``"states"`` (default) keeps ``S_in`` and runs the call with
    ``save_transfer=False`` (bitwise the default form; the backward re-sends the maps in b1); ``"transfer"`` also
    keeps the owner's c1 receive tensor (``save_transfer=True``; not with ``exchange="ring"``).  An explicit
    ``save_transfer`` must agree with the tier.  Refused with ``graph="replay"`` (keyword or knob), inside a
    CUDA-graph capture, with ``debug_sequence``, ``spec3_probe`` and ``boundary_mode="replay"``.  Retains
    ``recompute_handle.nbytes()`` (``kda_det_saved_bytes_kcp(ctx, H=..., checkpoint_keep=..., with_conv=False)``) while
    the handle lives.  The exchange is a collective: every rank of ``ctx.group`` passes a handle for the same call or
    none, at the same checkpoint nesting level; a mismatch hangs or corrupts data (no check across ranks)."""
    opts = _options(options)  # validate on the caller's frame
    if boundary_mode not in BOUNDARY_MODES:
        raise ValueError(
            f"boundary_mode must be one of {BOUNDARY_MODES}, got {boundary_mode!r}"
        )
    if recompute_handle is None:
        if checkpoint_keep is not None:
            raise ValueError(
                "checkpoint_keep selects what a recompute handle keeps: it needs recompute_handle"
            )
        if opts.host_trim:  # the forward and the backward reuse the validated options instead of re-validating the dict
            options = {"_kcp_options": opts}
        return ChunkKDADetFunctionKcp.apply(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            scale,
            save_mode,
            boundary_mode,
            workspace,
            dict(options),
            None,
            use_gate_in_kernel,
            A_log,
            dt_bias,
            lower_bound,
        )
    opts = _handle_options(opts, checkpoint_keep, options.get("save_transfer"))
    _handle_refusals(opts, boundary_mode, recompute_handle)
    return ChunkKDADetFunctionKcp.apply(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        scale,
        save_mode,
        boundary_mode,
        workspace,
        {"_kcp_options": opts},
        recompute_handle,
        use_gate_in_kernel,
        A_log,
        dt_bias,
        lower_bound,
    )


def reduce_gate_param_grads_det_kcp(
    ctx: Any, dA_log: torch.Tensor, ddt_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The raw gate mode's full parameter gradients from every rank's partial sums, bitwise reproducible.

    ``dA_log`` (FP32 ``[H]``) and ``ddt_bias`` (FP32 ``[H * 128]`` or ``[H, 128]``) are this rank's partial sums
    from :func:`chunk_kda_det_bwd_kcp` / :func:`chunk_kda_det_kcp` with ``use_gate_in_kernel=True`` (the gradients through
    the rank's own tokens).  Every rank of ``ctx.group`` must call this with the same shapes: one
    ``all_gather_into_tensor`` of the ``H + H * 128`` FP32 values on ``ctx.group`` brings every rank's partials in
    group-rank order, then each rank adds them in FP32 in rank order ``0, 1, ..., W - 1`` (no summation in the
    collective).  So every rank gets the same bits, and the result depends only on the per-rank partials, which
    depend only on the rank's tokens and the partition (not on the exchange, schedule, graph replay or recompute).
    An NCCL all-reduce of the partials gives the same sum up to FP32 rounding but does not fix its summation order
    (it varies with the algorithm, the topology and the message size): use this helper when the parameter
    gradients must be bitwise reproducible.  Returns fresh ``(dA_log, ddt_bias)`` of the inputs' shapes; with
    ``world_size == 1`` copies of the inputs.  With ``world_size > 1`` the ``torch.distributed`` process group must
    be initialized: without one the inputs are only this rank's partial sums, so the call raises ``RuntimeError``
    instead of returning them.  Asynchronous on the current stream, no host synchronization."""
    if dA_log.dtype != torch.float32 or ddt_bias.dtype != torch.float32:
        raise ValueError(
            f"dA_log / ddt_bias: expected torch.float32, got {dA_log.dtype} / {ddt_bias.dtype}"
        )
    H = int(dA_log.numel())
    if dA_log.dim() != 1 or int(ddt_bias.numel()) != H * D:
        raise ValueError(
            f"dA_log [H] and ddt_bias [H * {D}] (or [H, {D}]) expected, got {tuple(dA_log.shape)} / "
            f"{tuple(ddt_bias.shape)}"
        )
    if dA_log.device != ddt_bias.device:
        raise ValueError(
            f"dA_log on {dA_log.device}, ddt_bias on {ddt_bias.device}: one device expected"
        )
    W = int(ctx.world_size)
    if W == 1:
        return dA_log.clone(), ddt_bias.clone()
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            f"reduce_gate_param_grads_det_kcp: world_size {W} needs an initialized torch.distributed process "
            "group; without one only this rank's partial sums are available"
        )
    n = H + H * D
    mine = torch.cat((dA_log.reshape(-1), ddt_bias.reshape(-1)))
    every = _alloc((W, n), torch.float32, mine.device)
    dist.all_gather_into_tensor(every, mine, group=ctx.group)
    total = every[0].clone()
    for r in range(1, W):  # FP32, in rank order
        total.add_(every[r])
    return total[:H].clone(), total[H:].view(tuple(ddt_bias.shape)).clone()


# the plan module's byte queries, re-exported for the KCP contract §3.2 literal forms
kda_det_saved_bytes_kcp = cake_kcp_plan.kda_det_saved_bytes_kcp
kda_det_comm_bytes_kcp = cake_kcp_plan.kda_det_comm_bytes_kcp
kda_det_workspace_bytes_kcp = cake_kcp_plan.kda_det_workspace_bytes_kcp

__all__ = [
    "ABI_PARAMS",
    "TRANSFER_WALKS",
    "WALK_GEOMETRIES",
    "WALK_GEOMETRY_DEFAULT",
    "HOST_TRIMS",
    "HOST_TRIM_DEFAULT",
    "WALK_CAP_PROGRAMS",
    "WALK_CACHE_CONFIG_CAP",
    "WALK_CACHE_CONFIG_DEFAULT",
    "walk_device_cache_config",
    "SPEC3_PROBES",
    "KcpPreconditionError",
    "BOUNDARY_MODES",
    "COLLECTIVES",
    "CONTRACT_DTYPES",
    "EXCHANGES",
    "GRAPH_MODES",
    "GRAPH_DEFAULT",
    "GRAPH_CAPS",
    "GRAPH_CAP_DEFAULT",
    "GRAPH_NODE_CAPS",
    "GRAPH_NODE_CARVEOUT",
    "capture_kernel_node",
    "graph_entry_points",
    "set_kernel_node_carveout",
    "GRAPH_STATS",
    "CACHE_CONFIG_STATS",
    "CHECKPOINT_KEEPS",
    "HANDLE_KEY",
    "RecomputeHandleDetKcp",
    "graph_cache_clear",
    "KCP_PROGRAMS",
    "KCP_SAVED_VERSION",
    "LAST_CALL",
    "OFFSET_CLASSES",
    "RING_MAX_HOPS",
    "SUMMARY_TILE",
    "TILE",
    "Backend",
    "Chains",
    "ChunkKDADetFunctionKcp",
    "Indexed",
    "CallDetKcp",
    "OptionsDetKcp",
    "SavedDetKcp",
    "WorkspaceDetKcp",
    "KernelBackend",
    "chunk_kda_det_kcp",
    "chunk_kda_det_bwd_kcp",
    "chunk_kda_det_fwd_kcp",
    "get_backend",
    "kda_det_comm_bytes_kcp",
    "kda_det_saved_bytes_kcp",
    "kda_det_workspace_bytes_kcp",
    "reduce_gate_param_grads_det_kcp",
    "set_backend",
    "tile_index",
]

# The names before the det rename, bound to the same objects (not in ``__all__``).
ChunkKDAKcpFunction = ChunkKDADetFunctionKcp
KcpCall = CallDetKcp
KcpOptions = OptionsDetKcp
KcpRecomputeHandle = RecomputeHandleDetKcp
KcpSaved = SavedDetKcp
KcpWorkspace = WorkspaceDetKcp
chunk_kda_kcp = chunk_kda_det_kcp
chunk_kda_kcp_bwd = chunk_kda_det_bwd_kcp
chunk_kda_kcp_fwd = chunk_kda_det_fwd_kcp
kcp_comm_bytes = kda_det_comm_bytes_kcp
kcp_saved_bytes = kda_det_saved_bytes_kcp
kcp_workspace_bytes = kda_det_workspace_bytes_kcp
