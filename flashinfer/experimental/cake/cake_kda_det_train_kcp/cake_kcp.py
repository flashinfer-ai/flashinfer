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
:class:`CakeKDADetKcpWorkspace`, the forward and backward launch sequences with their
collectives, the saved set :class:`CakeKDADetKcpSaved`, the autograd Function and the
entry points ``cake_kda_det_kcp`` / ``cake_kda_det_train_kcp_forward`` / ``cake_kda_det_train_kcp_backward``.

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
from it.  One compute stream (the forward's summary pair may fork onto a side
stream and join back before the next launch, see "Summary walk streams" below);
the collectives are asynchronous on the caller's
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
``graph="replay"`` of :func:`cake_kda_det_kcp`, :func:`cake_kda_det_train_kcp_forward` and
:func:`cake_kda_det_train_kcp_backward`, shipped default ``GRAPH_DEFAULT = "off"``): the
first call of a validity key runs eagerly, the second is captured into a
``torch.cuda.CUDAGraph`` and every later call of the key replays it -- the same
launches and collectives, bitwise the eager result.  The key is the call's
kind, its CakeKDADetKcpContext / workspace / backend objects, every option and the
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
under ``walk_form="full"``).  The fused forward walk's rule is per architecture
(``cake_kcp_jit.FWDH_O_FORM_RULE``, from kernel A/B medians): on sm_100a and
sm_103a the two-CTA form up to 32 value heads and the full-width form above,
on sm_107a the fit rule up to 48 value heads and the full-width form above.
The rule's inputs are static per ``(ctx, H)``, so
the form of each ``(program, walk list)`` is decided once per table set
(``_Tables.walk_form``) and looked up on every later launch: the eager call
carries no per-launch list scheduling.

Summary walk streams (``WALK_STREAMS_DEFAULT = "auto"``): the forward's
summary pair -- K1 ``_offset`` and K3 over the offset walk list, into the c1
send tiles at ``P = 1`` and the staging planes at ``P > 1`` (also in the
replay forward of a replay-mode backward and in a checkpoint recompute's stage
walks) -- reads the same local operands, writes disjoint tiles and feeds the
next launch only, so K3 may run on a per-device side stream beside K1: an
event recorded on the call's stream forks the side stream, an event recorded
on the side stream joins it back before the next launch (:func:`_offset_pair`;
capturable, bitwise the one-stream form).  The rule :func:`walk_streams`
decides per pair from host-known inputs (the chains, the head count and the
SM count, once per table set: ``_Tables.walk_streams``): two streams when K1
alone fills a wave of its ``2 N`` slots (``len(chains) * H >= 2 N``), one
stream below.  The first pair of a table set over its list on a
device runs on one stream (it uploads the launches' device tables); the
previous host path (``host_trim=0``) keeps one stream.  The test knob
``walk_streams`` (``"one"`` / ``"two"``) forces a form for the A/B record;
``LAST_CALL[kind]["walk_streams"]`` records ``(decided form, form run,
n_walk)`` per pair.

Backward tape stream (``BWD_TAPE_STREAM_DEFAULT = "post"``): the saved-set
backward's tape replay (``kcp_fwdh_tape``: ``w`` / ``kg`` / ``u`` / ``gk`` and
the initial states in, the ``h`` / ``v_new`` tapes out) runs on a per-device
side stream (:class:`_TapeSide`), forked after the adjoint offset walk and
issued after the b1 exchange, beside the exchange and the adjoint merge, which
read neither tape and write only the call's pools; the call's stream waits for
it before the adjoint tape walk (its ``dh`` shares bytes with ``u`` in both
backward layouts).  The first replay of a table set
on a device, a backward that launches ``kcp_dav``, the ring exchange, the
``overlap`` schedule, the previous host path and capped (full-width) tape
launches keep the call's stream.  Placement never changes a bit; the test
knob ``bwd_tape_stream`` keeps ``"one"`` and the other forms of
:data:`BWD_TAPE_STREAMS` selectable for the A/B record, and
``LAST_CALL["bwd"]["bwd_tape_stream"]`` records the form run.

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
once and the autograd node carries the :class:`CakeKDADetKcpOptions`;
``CakeKDADetKcpWorkspace.ensure`` returns the last tables for the same (context, H,
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
launcher on the arena keeps ``CakeKDADetWorkspace.run``: both keep the previous path
under either value of ``host_trim``.

Fused forward output (``FWD_O_FUSE_DEFAULT = 1``): the forward that keeps
no tape (every forward but the one a replay-mode backward runs) launches the
fused walk ``kcp_fwdh_o`` wherever it launched the tape walk ``kcp_fwdh_tape``
-- the same segment lists, the same initial states -- and no ``fwd_o``: the
walk writes ``out`` for its segments' token rows and no ``h`` / ``v_new`` tape.
``out`` is the bits of the tape walk followed by ``fwd_o`` (the same two MMAs on
the same BF16 tiles in the same order, one rounding).  The forward's tapes have
no reader but ``fwd_o`` in the same call: the saved set holds no tape, a
backward replays its own tapes into the backward layout before its first tape
reader, and the replay forward of a replay-mode backward keeps the tape walk.
The forward's ``h`` / ``v_new`` views keep their arena bytes (the previous form
and the replay forward use them).  ``_kcp_test_knobs={"fwd_o_fuse": 0}``
selects the tape walk and ``fwd_o`` for the A/B record.

Saved carry plane (``SAVED_CARRY_DEFAULT = 1``): the forward of a ``P > 1``
rank keeps its whole carry plane in the saved set (``CakeKDADetKcpSaved.carry``) and the
saved-mode backward replays its tape from it instead of re-walking every
segment (K1 ``_offset``) and re-merging the carries -- the same remaining
launches, bitwise; ``cake_kda_det_train_kcp_saved_bytes`` counts the plane.
``_kcp_test_knobs={"saved_carry": 0}`` selects the re-walk for the A/B record;
the replay mode (checkpoint recompute) and ``P = 1`` ranks are the same under
both values.

Output backward: the adjoint walks' records are their forms' ``fuse_dv1``
builds, which form ``dv1 = Aqk^T dO`` in FP32 inside the step from the
internal ``Aqk`` rows (``aqk_int``), so ``kcp_dav``'s ``dv1`` has no reader.
The backward launches ``kcp_dav`` only where its ``dAqk`` has one: where the
existing backend's plan dispatches its ``dav`` stage on the replay path
(:func:`cake_kcp_plan.kcp_dav_needed`; the serial route below 16 heads,
H 9-15), whose fused gradient stores no ``dAqk``.  Everywhere else that fused
gradient stores every ``dAqk`` row GLM intra reads, and the ``dav`` stage
launches nothing (counted in the call's launches and ``sequence`` like the
gate epilogue a plan fuses away).  ``LAST_CALL["bwd"]["kcp_dav"]`` records
``"launched"`` or ``"skipped"``; the private test knob ``kcp_dav``
(``"auto"`` shipped, ``"always"``: launch it on every backward of the replay
layout, the coverage record) changes no output or gradient bit.

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
GDN twin is #7818): with ``cake_kda_det_kcp(..., recompute_handle=)``
(:class:`CakeKDADetKcpRecomputeHandle`, or any object with an ``entries`` dict) the
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
:func:`cake_kda_det_train_kcp_reduce_gate_param_grads` (an all-gather, then an FP32 sum in rank
order) when that sum must be bitwise reproducible -- an NCCL all-reduce does not
fix its summation order.  The workspace (:attr:`CakeKDADetKcpWorkspace.gate`), the
saved set and a recompute handle's fingerprint record the mode (and the lower
bound); a call in another mode is refused.  A raw-mode workspace runs the replay
local arena (``local_arena="lean"`` is refused in this mode).

In-kernel Q / K L2 norm (FlashInfer issue #7817; keyword-only
``use_qk_l2norm_in_kernel=False`` on the three entry points, the workspace and
the byte queries, never an :class:`CakeKDADetKcpOptions` field): with ``True`` the
local stages normalize ``q`` and ``k`` per token and per 128-channel head,
``r = 1 / sqrt(sum(x * x) + 1e-6)`` in FP32, ``x_hat = bf16(x * r)`` (``v`` is
not normalized), and the GLM intra programs fuse the FP32 pullback
``dx = r * (dx_hat - y * dot(dx_hat, y))``, ``y = x * r``, into the final BF16
``dq`` / ``dk`` store.  The normalization is token-local: the halo, the
context, the KCP programs, every collective and its byte count are those of
the flag-off route.  The saved set keeps the raw ``q`` / ``k`` (the caller's
tensors, as without the flag) plus the FP32 rstd ``CakeKDADetKcpSaved.qk_rstd``
``[2, T_r, H]`` (plane 0 q, plane 1 k; empty on a rank without tokens); the
normalized q / k are never saved: the backward rebuilds them bitwise from the
raw rows and the rstd into a per-call buffer, and a checkpoint recompute
writes its rstd afresh (a recompute handle's entry does not change).  ``q`` /
``k`` may be strided views (columns of a packed projection or convolution
output: innermost stride 1, head stride 128 elements, token stride a multiple
of 8 elements and at least a token's row, base address 16-byte aligned: the
single-rank rule), never copied; without the flag they stay contiguous-only.
Every rank checks the view's metadata (shape, dtype, device, the three
strides), one without tokens or with one token too, so all ranks refuse a
view alike; the base-address alignment is a property of each rank's own
storage, checked where the rank has tokens.
The workspace (:attr:`CakeKDADetKcpWorkspace.qk_l2norm`), the saved set, the graph
validity key and a recompute handle's fingerprint record the mode only when it
is on: with ``False`` every key, saved tensor, launch and byte is unchanged.
The mode runs on the replay local arena (``local_arena="lean"`` is refused).
The local programs of the mode are the single-rank backend's (its plan mode
``l2n``: ``"fused"`` for the flag; ``"fallback"``, the separate normalization
and pullback programs, through a test workspace); the records select them,
:mod:`.cake_kcp_plan` lists the ones this route needs.  The first q / k reader
of a saved-mode backward, which reads the saved rstd, runs the record's
saved-rstd twin where one ships for the device's architecture
(:func:`cake_kcp_plan.saved_rstd_record`).  Until the single-rank
registry serves them on the device's architecture with an exported library
and its layouts carry the normalized slot views
(:func:`cake_kcp_plan.qk_l2norm_missing`), a call, a workspace or a byte query
with the flag on raises ``NotImplementedError`` before any workspace build,
plan, launch or collective.  The refusal reads rank-independent inputs only
(the device architecture, the registry records and exports, the flags, the
static layout tables), so every rank raises the same error and message.


P > 1 staging: the K1 ``_offset`` / K2 ``_offset`` / K3 passes of the segmented rank write two
dense planes ``stage_b`` / ``stage_m`` ([NSEG, H, 128, 128] FP32 each, ``tile_stride`` 1); K4 reads them element-wise through
``elem_m`` / ``elem_c`` (map / carry / transposed-map / adjoint-carry families); the pair convention stays
for K4's states (c1 / b1 pairs, ``tile_stride`` 2) and the carry plane; under ``save_transfer=True`` at ``P > 1`` the
``stage_m`` plane itself is the saved ``M_seg`` (no copy; ``cake_kda_det_train_kcp_saved_bytes`` keeps the ``NSEG * s`` term).
"""

from __future__ import annotations

import collections
import ctypes
import dataclasses
import functools
import inspect
import operator
import sys
import threading
import types
import warnings
import weakref
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist

from . import _single_rank, cake_kcp_plan

cake_backend = _single_rank.module("cake_backend")
cake_jit = _single_rank.module("cake_jit")

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
# The walk programs whose record form (full-width / two-CTA) the registry's rule picks per launch: K1 / K2 and the
# fused forward walk (its full-width form is one CTA per SM by its footprint; no cap applies to it)
_WALK_FORMED = (*_WALK_K12, "fwdh_o")
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
# Fused forward output: 1 (shipped) launches the fused walk kcp_fwdh_o in place of every forward tape walk launch of a
# forward that keeps no tape (not the replay forward of a replay-mode backward) and no fwd_o -- out bitwise the same,
# no h / v_new tape written; 0 = the tape walk(s) + fwd_o (the previous form), via _kcp_test_knobs only (the A/B
# record).  The replay forward runs the tape walk under both values.
FWD_O_FUSES = (0, 1)
FWD_O_FUSE_DEFAULT = 1
# Intra rows of a checkpoint recompute: "ckpt" (shipped) = a saving forward that runs inside the autograd engine's
# backward pass (torch's checkpoint recompute, with or without a recompute handle; :func:`_kcp_in_backward`) keeps the
# internal Aqk / Akk rows its intra program writes (BF16 [2, R, H, 64], 256 B per internal token-head), and the
# backward from its saved set uses them in place of its own intra launch.  The saved set does not carry them (its
# type, tensors and meta are those of every other forward: torch's checkpoint keeps the original forward's autograd
# node, its saved-tensor count and its meta): the forward leaves them in :data:`_CKPT_ROWS` under its ``S_in``, and
# the backward of a set with that very ``S_in`` takes them.  They add nothing to the backward's peak: they lie in the
# first bytes of the backward's ``dq`` / ``dk`` pair (one BF16 [2, T_r, H, 128] tensor the forward allocates and the
# backward returns as its two halves).  With the in-kernel L2 norm off the backward's programs read them there (only
# the GLM intra program writes the pair, after the rows' last reader, :func:`_rows_carrier_ok`); otherwise the
# backward first copies them into its arena's own ``aqk_int`` / ``akk_int`` rows (the copy replaces the intra launch)
# and then takes the tensor as its pair (with the in-kernel L2 norm, the bytes its ``dv2`` rides on).  Either way the
# backward allocates no pair of its own, so the kept rows leave no block of another size between the forward and the
# backward (a separate rows tensor, freed after the copy, raised the caching allocator's reserved peak: +258 MiB per
# rank on the four-pack CP16 workload with a workspace, +4.1 GiB with workspace=None).  Both forms need
# ``R <= 2 T_r`` (the rows fit the pair).  Every other forward keeps none, so the saved set and
# cake_kda_det_train_kcp_saved_bytes are unchanged.
# "always" = every saving forward keeps them, "never" = none (the previous form); both via _kcp_test_knobs only (the
# A/B record).  Not on a lean arena, under graph replay or a capture, with the SPEC-3 probe, nor where the backward's
# intra program is its first q / k reader (the in-kernel L2 norm's segmented rows): those keep or use none.  Nor on a
# call without a workspace (``workspace=None``): its default workspace frees its storage at the end of every call, and
# a tensor allocated during the recompute and kept to the backward took bytes of the block the backward's default arena
# needs next, so the caching allocator reserved another arena (+3.5 to +12.4 GiB per rank on the four-pack CP16
# workload; the backward launches its intra program instead).  The rows
# are the intra program's on the same operands: without the fused in-kernel L2 norm the same record on the same rows;
# with it the forward record normalizes the raw q / k in the kernel while the backward's runs on the prep's normalized
# rows, so that form keeps them only on the architectures of :data:`INTRA_ROWS_L2N_ARCHS`, where the two were measured
# bitwise equal.  The gradients are those of the backward that launches it (``rows_probe`` = 1, a test knob: that
# backward launches its intra program as well and records the bitwise comparison in ``LAST_CALL``; host sync).
INTRA_ROWS = ("ckpt", "always", "never")
INTRA_ROWS_DEFAULT = "ckpt"
# The kept rows between a forward and the backward of its saved set: id of the set's ``S_in`` -> (weak reference to
# that tensor, the tensor that holds the rows, the rows [2, R, H, 64]).  Taken (removed) by the first backward of a set
# with that ``S_in``, whether it uses them or not; dropped with ``S_in`` (the reference's callback), so an entry never
# outlives the saved set of a forward whose backward does not run.
_CKPT_ROWS: dict[int, tuple[Any, torch.Tensor, torch.Tensor]] = {}


def _ckpt_rows_drop(ref: Any, key: int) -> None:
    """The weak reference callback of a :data:`_CKPT_ROWS` entry: its ``S_in`` died before a backward took the rows."""
    entry = _CKPT_ROWS.get(key)
    if entry is not None and entry[0] is ref:
        del _CKPT_ROWS[key]


# The architectures where a fused in-kernel L2 norm forward keeps its intra rows (INTRA_ROWS): its intra record on the
# raw q / k and the backward's prep + intra record on the normalized rows write bitwise equal Aqk / Akk rows there
# (rows_probe on every real and pad row, gate g and raw, H = 64, CP4: sm_103a GB300, sm_100a GB200, sm_107a R200).
# The equality is a measured property of each architecture's build, not one by construction: an architecture added
# to the build keeps none (the backward launches its intra program) until it is probed and listed here.
INTRA_ROWS_L2N_ARCHS = ("sm_100a", "sm_103a", "sm_107a")
ROWS_PROBES = (0, 1)
# Walk form of the K1 / K2 state walks: "auto" (shipped) = the registry's rule per launch (cake_kcp_jit.walk_form: the
# offset programs take their two-CTA ``_vh64`` record when the launch list-schedules within its longest chain, else the
# full-width record; the tape programs take their two-CTA record always, their full-width records being test-only;
# the fused forward walk takes the form its architecture's rule names, cake_kcp_jit.FWDH_O_FORM_RULE);
# "full" / "split" force one form on every launch, _kcp_test_knobs only (the A/B record).  The forms are bitwise equal:
# the form never changes a bit.  A split tape launch is never capped at one CTA per SM (the cap serves the full-width
# tape CTAs).
WALK_FORMS = ("auto", "full", "split")
WALK_FORM_DEFAULT = "auto"
# Walk streams of the forward's summary pair (K1 ``_offset`` + K3 over the offset walk list, :func:`_offset_pair`): "auto"
# (shipped) = the rule :func:`walk_streams` per pair (two streams once K1 fills a wave, one stream below); "one" / "two" force
# a form, _kcp_test_knobs only (the A/B record).  The two kernels read the same operands and write disjoint tiles, so
# the form never changes a bit.
WALK_STREAMS = ("auto", "one", "two")
WALK_STREAMS_DEFAULT = "auto"
# The backward's kcp_dav launch: "auto" (shipped) = only where cake_kcp_plan.kcp_dav_needed (the existing backend's plan
# dispatches its dav stage: its fused gradient stores no dAqk); "always" = on every backward of the replay layout,
# _kcp_test_knobs only (the coverage record: kcp_dav's outputs have no reader where "auto" skips it, so every output
# and gradient bit is the same).
KCP_DAV_MODES = ("auto", "always")
KCP_DAV_DEFAULT = "auto"
# Stream of the saved-set backward's tape replay (``kcp_fwdh_tape``, :class:`_TapeSide`): "one" = on the call's stream
# right after the WY program (the previous form); the other forms run it on a per-device side stream between a fork
# (an event recorded on the call's stream, after the WY program or, for "post", after the adjoint offset walk) and a
# join before the adjoint tape walk (before the dcarry merges where the replay reads the workspace's carry pool).
# "pre" issues the replay before the adjoint offset walk, "mid" right after it, "post" after the b1 exchange is
# issued; "*_hi" also run the call's launches of the window on a high-priority stream.  No launch of the window reads
# the h / v_new tapes or writes an operand of the replay, so the form never changes a bit; _kcp_test_knobs only (the
# A/B record).
BWD_TAPE_STREAMS = ("one", "pre", "mid", "post", "pre_hi", "mid_hi", "post_hi")
# the A/B record's fastest form (GB200 CP4: ts4 / cust5pad backward)
BWD_TAPE_STREAM_DEFAULT = "post"
P2_KNOBS = {  # knob -> admitted values
    "schedule": SCHEDULES,
    "fused_walks": (0, 1),
    "cols_rule": COLS_RULES,
    "walk_geometry": WALK_GEOMETRIES,
    "walk_form": WALK_FORMS,
    "walk_streams": WALK_STREAMS,
    "graph": GRAPH_MODES,
    "graph_cap": GRAPH_CAPS,
    "host_trim": HOST_TRIMS,
    "merge_fuse": MERGE_FUSES,
    "saved_carry": SAVED_CARRIES,
    "fwd_o_fuse": FWD_O_FUSES,
    "intra_rows": INTRA_ROWS,
    "rows_probe": ROWS_PROBES,
    "kcp_dav": KCP_DAV_MODES,
    "bwd_tape_stream": BWD_TAPE_STREAMS,
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
    "kcp_fwdh_o",  # K1 ``_tape`` with the chunk output fused in: out, no tape (the forward that keeps no tape)
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
# The lean backward layout of a local arena sized for it (``CakeKDADetKcpWorkspace(local_arena="lean")``, opt-in): this
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
# arena was sized for (``CakeKDADetWorkspace.fwd_layout``), on a backward arena :data:`LOCAL_LAYOUT_FWD`
LOCAL_LAYOUT_FWD_ONLY = cake_kcp_plan.LOCAL_LAYOUT_FWD_ONLY
# The KCP-owned local records (:mod:`.cake_kcp_jit`), by the stage name the launch sequences use: the output program
# over the K1 ``h`` / ``v_new`` tapes and the ``dA_qk`` / ``dv1`` program (the forms the existing backend's output and
# dA / dv programs had before its fused recurrences took them over), launched only where its ``dA_qk`` has a reader
# (:func:`cake_kcp_plan.kcp_dav_needed`; K2 forms its own ``dv1``).
KCP_LOCAL_RECORDS = {"fwd_o": "kcp_fwd_o", "dav": "kcp_dav"}
# The WY records the KCP launches directly where the plan fuses the WY representation into a state recurrence the
# KCP does not run (its recurrences are the K1 / K2 walks): the forward's program and the replay's.
WY_DIRECT_RECORDS = {"wy_fwd": "wy_fwd", "wy": "wy_bwd_replay"}
# The in-kernel Q / K L2 norm's fallback stages (issue #7817): the separate normalization before the first q / k
# reader and the pullback after GLM intra.  The fallback form launches them on every problem; a plan that does not
# dispatch one is refused (:meth:`KernelBackend.stage`), never skipped as an optional stage the plan fuses away.
L2N_FALLBACK_STAGES = ("qk_l2norm", "qk_apply", "qk_pullback")
SCRATCH_TABLES_ATTR = "_kcp_scratch_tables"  # the arena attribute holding the table set the intra record writes
# the arena attribute holding a flag-on backward's moved-views copy per arena view (_l2n_bwd_rows)
L2N_MOVED_ATTR = "_kcp_l2n_moved"
L2N_MOVED_MAX = 16  # arena views (and per view, (buffer, T_r) row heads) one such cache keeps (cleared when full)
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


def _check_qk_view(
    x: torch.Tensor, name: str, T: int, H: int, device: torch.device
) -> torch.Tensor:
    """The strided-input rule of the in-kernel Q / K L2 norm (issue #7817, ``use_qk_l2norm_in_kernel=True``) for a raw
    ``q`` / ``k`` of ``T_r`` tokens.  The metadata, checked on every rank (one with zero or one token too, so every
    rank refuses a view alike): BF16 ``[T_r, H, 128]`` on ``device``, innermost stride 1, head stride exactly 128
    elements (ignored for ``H == 1``, the same on every rank), token stride a multiple of 8 elements and at least
    ``H * 128`` (no overlapping token rows).  Then the per-rank part, a property of the rank's own storage: a 16-byte
    aligned base address where the tensor has elements -- through the single-rank rule ``cake_backend._check_qk_raw``
    where the existing backend has it (the one source of the rule), else here, the view of ``H == 1`` coming back with
    its canonical head stride on the same storage (no copy).  Any other view raises ``ValueError``; nothing is
    copied."""
    shape = (T, H, D)
    if tuple(x.shape) != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {tuple(x.shape)}")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"{name}: expected {torch.bfloat16}, got {x.dtype}")
    if x.device != device:
        raise ValueError(f"{name}: on {x.device}, expected {device}")
    token, head, inner = x.stride()
    head = D if H == 1 else head
    if inner != 1:
        raise ValueError(
            f"{name}: use_qk_l2norm_in_kernel=True needs innermost stride 1, got strides {x.stride()}"
        )
    if head != D:
        raise ValueError(
            f"{name}: use_qk_l2norm_in_kernel=True needs head stride {D}, got strides {x.stride()}"
        )
    if token % 8 or token < H * D:
        raise ValueError(
            f"{name}: use_qk_l2norm_in_kernel=True needs a token stride of a multiple of 8 elements and at least "
            f"{H * D} (no overlapping token rows), got strides {x.stride()}"
        )
    single = getattr(cake_backend, "_check_qk_raw", None)
    if single is not None:
        return single(x, name, T, H, device)
    if x.numel() and x.data_ptr() % 16:
        raise ValueError(
            f"{name}: use_qk_l2norm_in_kernel=True needs a 16-byte aligned base address, got {x.data_ptr():#x}"
        )
    if x.numel() and head != x.stride(1):
        x = x.as_strided(shape, (token, head, 1))
    return x


def _operands(
    q,
    k,
    v,
    g,
    beta,
    ctx,
    *,
    dtypes: dict[str, torch.dtype],
    gate: str = "g",
    qk_l2norm: bool = False,
):
    """The operands without their optional batch axis, ``(T_r, H)`` and whether the inputs carried a batch axis.
    ``T_r`` must equal the context's local token count; ``H`` is the operands' head count (``1..64``).  ``gate="raw"``
    (``use_gate_in_kernel=True``): ``g`` is the raw gate, BF16 ``[T_r, H, 128]``, any view the single-rank rule
    accepts (``cake_backend._check_raw``: unit innermost stride, 16-byte outer strides and address, no overlap; the
    only full check, as the strided-pointer bindings check no layout; never copied).  ``qk_l2norm``
    (``use_qk_l2norm_in_kernel=True``): ``q`` / ``k`` are the raw rows, any view of :func:`_check_qk_view` (never
    copied; a length-1 axis re-strided); without it they must be contiguous."""
    v = _squeeze_batch(v, "v", 3)
    T, H = int(v.shape[0]), int(v.shape[1])
    device = torch.device(ctx.device)
    batched = q.dim() == 4
    q, k, g = (_squeeze_batch(x, n, 3) for x, n in ((q, "q"), (k, "k"), (g, "g")))
    beta = _squeeze_batch(beta, "beta", 2)
    if int(ctx.T_r) != T:
        raise ValueError(
            f"operands carry {T} tokens, the CakeKDADetKcpContext of rank {ctx.rank} holds T_r={ctx.T_r}"
        )
    if not 1 <= H <= 64:
        raise ValueError(f"H={H}: the Cake KDA kernels take 1..64 heads")
    if qk_l2norm:
        q = _check_qk_view(q, "q", T, H, device)
        k = _check_qk_view(k, "k", T, H, device)
    else:
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
    # in-kernel Q / K L2 norm (use_qk_l2norm_in_kernel=True): the per-token rstd of q and k [2, T_r, H]
    "qk_rstd": torch.float32,
}


class Backend:
    """A dispatch table for the KCP programs and the local chunk stages.

    ``run(name, args)`` launches program ``name`` (one of :data:`KCP_PROGRAMS`) with the logical arguments ``args``
    (the ABI vocabulary of the design note §14: pools, index tables, ``hg`` / ``n`` addressing scalars, modes; plus
    ``_call``, the :class:`CakeKDADetKcpCall` the launch belongs to).  ``stage(name, call, **extra)`` launches one local chunk
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
        self, name: str, call: CakeKDADetKcpCall, **extra: Any
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def begin_call(self, call: CakeKDADetKcpCall) -> None:
        """Hook at the start of a forward or backward call (the kernel backend builds the local stage values, the
        emulation resets its per-call scratch)."""

    def end_call(self, call: CakeKDADetKcpCall) -> None:
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
    "kcp_fwdh_o": "fwdh_o",
}


def _walk_form_of(call: Any, programs: Any, name: str, args: dict[str, Any]) -> str:
    """The record form of walk launch ``name`` (``"full"`` / ``"split"``): the registry's rule
    (``cake_kcp_jit.walk_form``) on the launch's chains (the chunk counts of its listed segments), the call's head
    count, the device's SM count and, for a per-architecture rule, its architecture, under the call's ``walk_form``
    knob -- decided once per ``(program, walk list)`` of the call's table set and looked up on every later launch
    (:meth:`_Tables.walk_form`); ``"full"`` with a registry that has no form rule (one record per walk program).
    The provider's ``walk_form`` is called as ``walk_form(program, chains, num_heads, sm_count, *, knob)``, plus
    ``arch=`` (the jit's ``device_arch`` of the launch's device) only when ``WALK_FORM_RULES[name]`` is a mapping
    (a per-architecture rule): a provider whose rules are all strings needs no ``arch`` parameter."""
    jit = getattr(programs, "jit", None)
    rule = getattr(jit, "walk_form", None)
    rules = getattr(jit, "WALK_FORM_RULES", {})
    if rule is None or name not in rules:
        return "full"
    return call.tb.walk_form(
        name,
        args["walk"],
        int(call.H),
        int(jit.sm_count(call.device.index)),
        str(call.walk_form),
        rule,
        arch_of=getattr(jit, "device_arch", None)
        if isinstance(rules[name], Mapping)
        else None,
        device_index=call.device.index,
    )


def walk_streams(
    chains: Sequence[int],
    num_heads: int,
    sm_count: int,
    *,
    knob: str = WALK_STREAMS_DEFAULT,
) -> str:
    """The stream form of one summary pair (K1 ``_offset`` and K3 over the same walk list, :func:`_offset_pair`):
    ``"two"`` (K3 on a side stream beside K1) or ``"one"`` (K3 after K1 on the call's stream).  ``chains`` are the
    chunk counts of the listed segments, ``knob`` the ``walk_streams`` test knob (``"one"`` / ``"two"`` force a form).

    Both programs run two CTAs per SM: K1 launches ``c1 = len(chains) * num_heads`` CTAs, K3 ``2 * c1``.  With
    ``N = sm_count``, two streams only when K1 alone fills at least one wave of its ``2 N`` slots (``c1 >= 2 N``),
    one stream below.  The layer A/B (W4, 16384 tokens per rank) sets the bound: two streams on its interior ranks'
    pair at ``c1 = 512`` (8 walks at H 64) make the layer's forward faster on sm_100a, sm_103a and sm_107a, while on
    the pair at ``c1 = 256`` (16 walks at H 16, one pair per forward) they make it slower on sm_100a and sm_103a
    (152 SMs) and no faster on sm_107a (212 SMs).  An empty list is ``"one"`` (no launch)."""
    if knob not in WALK_STREAMS:
        raise ValueError(f"walk_streams must be one of {WALK_STREAMS}, got {knob!r}")
    if knob != "auto":
        return knob
    if not len(chains):
        return "one"
    return "two" if len(chains) * int(num_heads) >= 2 * int(sm_count) else "one"


_PAIR_STREAMS: dict[
    int, tuple[Any, Any, Any]
] = {}  # device index -> (side stream, fork event, join event)


def _pair_streams(device_index: int) -> tuple[Any, Any, Any]:
    """The side stream of the summary pair on device ``device_index`` and its fork / join events: created once per
    device and process (at the first two-stream decision, an eager call) and reused by every later pair."""
    hit = _PAIR_STREAMS.get(device_index)
    if hit is None:
        hit = _PAIR_STREAMS[device_index] = (
            torch.cuda.Stream(device=device_index),
            torch.cuda.Event(),
            torch.cuda.Event(),
        )
    return hit


_TAPE_STREAMS: dict[
    int, tuple[Any, ...]
] = {}  # device index -> (side stream, high-priority stream, fork event, join event, high-priority join event)


def _tape_streams(device_index: int) -> tuple[Any, ...]:
    """The streams and events of the backward's tape replay on device ``device_index`` (:class:`_TapeSide`): created
    once per device and process and reused by every later backward."""
    hit = _TAPE_STREAMS.get(device_index)
    if hit is None:
        hit = _TAPE_STREAMS[device_index] = (
            torch.cuda.Stream(device=device_index),
            torch.cuda.Stream(device=device_index, priority=-1),
            torch.cuda.Event(),
            torch.cuda.Event(),
            torch.cuda.Event(),
        )
    return hit


class _TapeSide:
    """The saved-set backward's tape replay (``kcp_fwdh_tape``) on a side stream (:data:`BWD_TAPE_STREAMS`).

    The replay reads ``w`` / ``kg`` / ``u`` / ``gk`` and its initial states (``S_in``, or the carry plane / pool) and
    writes the ``h`` / ``v_new`` tapes.  The launches between the WY program and the adjoint tape walk -- the adjoint
    offset walk (``w`` / ``kg`` / ``qg`` / ``do_int`` / ``aqk_int`` / ``gk`` in, its pool out), the map transfers, the
    merges and the b1 / b2 exchange -- read neither tape and write only the call's pools, so the replay may run beside
    them: :meth:`point` records the fork on the call's stream (``"pre"`` / ``"mid"``: after the WY program; ``"post"``:
    after the adjoint offset walk) and issues the replay on the side stream, :meth:`join` makes the call's stream wait
    for it before the first launch that writes an arena slot the replay uses (the adjoint tape walk writes ``dh``,
    which shares bytes with ``u`` in both backward layouts) or the carry pool it reads (the dcarry merges, where the
    replay reads the workspace's pool rather than the saved plane).  ``*_hi``: the call's launches between the fork and
    the join run on a high-priority stream, the replay on the side stream at the default priority.  The fork / join
    are stream-ordered event operations (capturable).  Allocator: neither stream allocates; every operand is owned by
    the workspace, the saved set, the call or the table set and outlives the join.  The call's launch record
    (``counters`` / ``sequence``) takes the replay's launches here, at their place in the program (where the one-stream
    form records them), so the record does not depend on the form."""

    def __init__(
        self, call: CakeKDADetKcpCall, launches: list, form: str, index: int
    ) -> None:
        self.call, self.launches, self.form = call, launches, form
        where = form.split("_")[0]
        self.fork_at = "post" if where == "post" else "pre"
        self.issue_at = where
        self.hi = form.endswith("_hi")
        self.side, self.crit, self.fork_ev, self.join_ev, self.crit_ev = _tape_streams(
            index
        )
        self.main = torch.cuda.current_stream(call.device)
        self.forked = self.issued = False
        self._crit_ctx: Any = None
        # CakeKDADetKcpCall.walk's record, at the program's place
        for walk, _h0 in launches:
            if len(walk):
                call.counters["walks"] += 1
                call.counters["launches"] += 1
                call.sequence.append("kcp_fwdh_tape")

    def _launch(self) -> None:
        """The replay's launches on the current stream (recorded by the constructor)."""
        for walk, h0 in self.launches:
            if len(walk):
                self.call.backend.run(
                    "kcp_fwdh_tape",
                    {"walk": walk, "h0": h0, "out": None, "_call": self.call},
                )

    def point(self, where: str) -> None:
        if where == self.fork_at and not self.forked:
            self.fork_ev.record(self.main)
            self.side.wait_event(self.fork_ev)
            if self.hi:
                self.crit.wait_event(self.fork_ev)
                self._crit_ctx = torch.cuda.stream(self.crit)
                self._crit_ctx.__enter__()
            self.forked = True
        if where == self.issue_at and self.forked and not self.issued:
            with torch.cuda.stream(self.side):
                self._launch()
            self.issued = True

    def join(self, *, complete: bool = True) -> None:
        """Back on the call's stream, ordered after both streams; ``complete``: a replay not yet issued runs there."""
        if self._crit_ctx is not None:
            self._crit_ctx.__exit__(None, None, None)
            self._crit_ctx = None
            self.crit_ev.record(self.crit)
            self.main.wait_event(self.crit_ev)
        if self.issued:
            self.join_ev.record(self.side)
            self.main.wait_event(self.join_ev)
        elif complete:
            self._launch()
        self.forked = self.issued = False
        self.launches = []


def _tape_side(call: CakeKDADetKcpCall, launches: list) -> tuple[_TapeSide | None, Any]:
    """The side-stream form of the backward's tape replay ``launches`` (``[(walk list, initial-state view thunk)]``;
    the side form holds the views, built now), or ``None`` for the call's stream, and the table-set key to mark once
    the replay ran on the call's stream (``None``: nothing to mark).  The side stream runs only for a
    :class:`CakeKDADetKcpCall` on the trimmed host path of the kernel backend on a CUDA device, under the a2a exchange
    and the ``wait_first`` schedule, with the tape launches uncapped (no cache-config bracket: the shipped split form)
    and no ``kcp_dav`` launch (it reads ``v_new`` right after the replay); the first replay of a table set on a device
    runs on the call's stream (it uploads the launches' device tables)."""
    # a test's driver call: the launches in order
    if not isinstance(call, CakeKDADetKcpCall):
        return None, None
    form = str(call.opts.bwd_tape_stream)
    if form == "one" or not any(len(w) for w, _ in launches):
        return None, None
    if not (
        call.device.type == "cuda"
        and isinstance(call.backend, KernelBackend)
        and call.host_trim
        and call.exchange != "ring"
        and call.schedule == "wait_first"
        and call.walk_form != "full"
        and not _kcp_dav_launch(call)
    ):
        return None, None
    index = (
        call.device.index
        if call.device.index is not None
        else torch.cuda.current_device()
    )
    seen = call.tb.__dict__.setdefault("_tape_seen", set())
    key = (tuple(id(w) for w, _ in launches), index)
    if key not in seen:
        return None, (seen, key)
    return _TapeSide(call, [(w, h0()) for w, h0 in launches], form, index), None


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
    chunk stages through :class:`cake_backend.CakeKDADetWorkspace` of the per-rank problem.  The default provider is H3's
    registry (:mod:`.cake_kcp_jit`, imported at the first launch so this module imports without it); a bring-up
    provider launching the source builds directly can be passed (``path="direct"``; never a gated row).

    Local stages (:data:`LOCAL_STAGES_FWD` / :data:`LOCAL_STAGES_BWD`): a forward call runs the existing backend's
    intra program (``write_prep = 1``: the GLM prep's rows and the internal ``Aqk`` / ``Akk`` rows; the token-row
    copies are never written, ``write_ext = 0``), its forward WY program and the KCP's output program ``kcp_fwd_o``
    (the output from the K1 ``h`` / ``v_new`` tapes) on the arena's forward layout (``fwd``; a forward-only arena's
    ``fwd_kcp``, :data:`LOCAL_LAYOUT_FWD_ONLY`), with the records the plan selects
    on the forward path without a saved set (:data:`LOCAL_PATH_FWD`).  A backward call runs on the ``replay`` layout
    (on a lean arena the lean layout :data:`LOCAL_LAYOUT_BWD_LEAN`: no ``kcp_dav``, the tail stages once per chunk
    range, :func:`_tail_pass_values`)
    with the records of the replay path (:data:`LOCAL_PATH_BWD`) in both modes: the replay prep (the GLM prep's rows
    and the internal ``dout`` copy ``do_int``), the intra program (``write_prep = 0``: the internal ``Aqk`` / ``Akk``
    rows), the replay's WY program, then after the K1 tape replay the KCP's ``kcp_dav`` (``dA_qk``, only where the
    plan dispatches its ``dav`` stage: :func:`cake_kcp_plan.kcp_dav_needed`; K2 forms ``dv1`` from ``aqk_int``
    itself) and after K2 the fused gradient, GLM intra and the gate epilogue.  Where the plan fuses the WY
    representation into a state recurrence the KCP does not run, the WY program is launched directly
    (:data:`WY_DIRECT_RECORDS`) on its grid rule.  The intra record that computes the chunk layout from ``cu_seqlens``
    writes its tables into a scratch set on the arena (:data:`SCRATCH_TABLES_ATTR`); every other program reads the
    context's CPU-built tables.  The saved set carries no token-row ``Aqk`` / ``Akk`` (:class:`CakeKDADetKcpSaved`): the
    backward recomputes the internal rows in both modes, so the two modes run the same programs on the same operands;
    the one exception is the set of a checkpoint recompute (:data:`INTRA_ROWS`), whose forward keeps the internal rows
    its intra program wrote and whose backward binds them instead of launching its own intra program.
    The K1 / K2 / K3 walks read the layout's ``w`` / ``kg`` / ``u`` / ``gk`` (``qg`` / ``do_int`` / ``aqk_int``) rows and
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
    (:class:`CakeKDADetKcpOptions`); no environment variable is read.  Under ``graph="replay"`` the probing backward runs
    eagerly; a rank without tokens has nothing to probe and does not hold back the capture of its backward.
    """

    name = "kernels"

    def __init__(
        self,
        programs: Any | None = None,
        *,
        cols_per_cta: int | None = None,
        path: str = "fi-rendered",
    ) -> None:
        self._programs = programs
        if cols_per_cta is not None and (
            cols_per_cta not in (8, 16, 32, 64) or D % cols_per_cta
        ):
            raise ValueError(
                f"cols_per_cta must be None (the registry's merge record per launch) or 8, 16, 32 or 64, got {cols_per_cta}"
            )
        # The K4 column block of the positional merge ABI: None = the column block of the registry record the
        # launch's (ncols, n_chains, transpose_m) selects (cake_kcp_jit.merge_record), the same record the
        # registry's run_program launches.
        self.cols_per_cta = None if cols_per_cta is None else int(cols_per_cta)
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
    def begin_call(self, call: CakeKDADetKcpCall) -> None:
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

    def stage(self, name: str, call: CakeKDADetKcpCall, **extra: Any) -> None:
        """One local stage of :data:`LOCAL_STAGES_FWD` / :data:`LOCAL_STAGES_BWD`: the plan's record on the call's path
        (or the direct WY record), the KCP-owned record of ``fwd_o`` / ``dav``; nothing for an optional stage the plan
        does not launch on the path (``gate_epilogue`` where ``glm_intra_v14`` writes the gate gradients itself), but
        ``RuntimeError`` for a stage of :data:`L2N_FALLBACK_STAGES` it does not launch."""
        values, views = call.local_values, call.scratch["views"]
        local = call.local
        p = local.plan
        path = LOCAL_PATH_BWD if call.backward else LOCAL_PATH_FWD
        if call.out is not None:
            values["out"] = call.out
        if call.grads is not None:
            values.update(call.grads)
        l2n = getattr(call, "l2n_values", None)
        if l2n is not None:
            # use_qk_l2norm_in_kernel=True: the mode's values (one dict for every stage: q / k = the normalized rows,
            # the raw rows under q_raw / k_raw, the rstd, its scalars; _qk_l2norm_begin) over the call's values,
            # under the stage's own extras
            extra = {**l2n, **extra}
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
            if name == "dav":
                # kcp_dav only where its dAqk has a reader: the plan dispatches its dav stage, so its fused gradient
                # stores no dAqk (cake_kcp_plan.kcp_dav_needed); the adjoint walks form dv1 themselves.  Elsewhere
                # nothing launches (counted like a stage the plan fuses away; the knob kcp_dav="always" launches it
                # anyway on the replay layout: its outputs have no reader there; the lean layout serves no plan that
                # dispatches dav)
                launch = _kcp_dav_launch(call)
                call.scratch["kcp_dav"] = "launched" if launch else "skipped"
                if not launch:
                    return
            self.programs().run_record(
                KCP_LOCAL_RECORDS[name],
                {**values, **extra},
                grid=kcp_local_grid(name, p),
                device_index=local.device_index,
            )
            return
        if name in L2N_FALLBACK_STAGES and not p.dispatched(name, path):
            # q / k would stay raw or dq / dk unpulled: refuse.  The capability check (rank-independent) requires
            # the stage's record on every problem of its grid; this guards a problem outside it
            raise RuntimeError(
                f"the plan dispatches no {name} on {path!r}; the in-kernel Q/K L2 "
                "norm fallback form needs it on every problem"
            )
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
        call: CakeKDADetKcpCall,
        record: str,
        grid: tuple[int, int, int],
        values: dict[str, Any],
    ) -> None:
        """A record of the existing backend's registry on the local arena: the trimmed launch, or the arena's
        launcher (a diagnostics launcher keeps the base path).  A launch that reads the saved rstd (``read_rstd =
        1``) runs the record's saved-rstd twin where it ships for the arena's architecture
        (:func:`cake_kcp_plan.saved_rstd_record`)."""
        local = call.local
        if values.get("read_rstd") and record in cake_kcp_plan.SAVED_RSTD_TWINS:
            twin = cake_kcp_plan.saved_rstd_record(record, local.arch)
            if twin != record:
                call.scratch["l2n_twin"] = (record, twin)
                record = twin
        if call.host_trim and local.launcher is cake_jit.run_module:
            _launch_record_trim(local, record, grid, values)
        else:
            local.launcher(record, values, grid=grid, device_index=local.device_index)

    # -- SPEC-3
    def _spec3_forward_rows(self, call: CakeKDADetKcpCall) -> None:
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

    def _spec3_compare(self, call: CakeKDADetKcpCall) -> None:
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
        call: CakeKDADetKcpCall = args["_call"]
        prog = _PROGRAM_OF[name]
        if prog in ("fwdh_offset", "fwdh_tape"):
            abi, grid = self._fwdh_args(call, args, tape=prog == "fwdh_tape")
        elif prog == "fwdh_o":
            abi, grid = self._fwdh_o_args(call, args)
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
            prog in _WALK_FORMED
        ):  # the record form of this walk launch (the registry's rule or the knob)
            variant = {"walk_form": _walk_form_of(call, self.programs(), name, args)}
            args["walk_form"] = variant["walk_form"]
        return prog, abi, int(grid), variant

    def run(self, name: str, args: dict[str, Any]) -> None:
        call: CakeKDADetKcpCall = args["_call"]
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

    def _run_trim(
        self, name: str, args: dict[str, Any], call: CakeKDADetKcpCall
    ) -> bool:
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
            prog in _WALK_FORMED
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
        self, call: CakeKDADetKcpCall, args: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, int, Any]:
        walk = args["walk"]
        walk_seg = self._dev_table("walk", walk, call.device, torch.int32)
        return (
            walk_seg,
            call.tb.hp.device_tables["seg_chunk_start"],
            int(call.tb.NSEG),
            call.scratch["views"],
        )

    def _block(self, call: CakeKDADetKcpCall, ix: Indexed | None) -> _Block:
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
        self, call: CakeKDADetKcpCall, args: dict[str, Any], *, tape: bool
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

    def _fwdh_o_args(
        self, call: CakeKDADetKcpCall, args: dict[str, Any]
    ) -> tuple[list[Any], int]:
        """The fused forward walk's positional ABI (:data:`ABI_PARAMS` ``fwdh_o``): the tape walk's operands, segment
        table, walk list and initial-state block, then the output stage's ``qg`` / ``Aqk`` rows, the call's output,
        the chunk tables and ``scale``; no tape and no output pool."""
        walk_seg, seg_table, nseg, v = self._walk_common(call, args)
        h0 = self._block(call, args.get("h0"))
        values = call.local_values
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
            h0.stride,
            v.qg,
            v.aqk_int,
            call.out,
            values["chunk_bos"],
            values["chunk_len"],
            float(call.scale),
        ]
        return abi, len(args["walk"]) * call.H

    def _dhu_args(
        self, call: CakeKDADetKcpCall, args: dict[str, Any], *, tape: bool
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
            v.aqk_int,
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
        self, call: CakeKDADetKcpCall, args: dict[str, Any]
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
        self, call: CakeKDADetKcpCall, args: dict[str, Any]
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
        if cols is None:
            cols = (
                self.cols_per_cta
                if self.cols_per_cta is not None
                else _registry_merge_cols(ch.ncols, ch.n_chains, call.device)
            )
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


def _registry_merge_cols(ncols: int, n_chains: int, device: torch.device) -> int:
    """``cols_per_cta`` of the K4 record the registry selects for the launch on ``device``
    (:func:`.cake_kcp_jit.merge_record`; a device that is not CUDA: the record every architecture's rule names)."""
    from . import cake_kcp_jit  # noqa: PLC0415 - H3's registry; optional at import time

    arch = (
        cake_kcp_jit.device_arch(device.index if device.index is not None else 0)
        if device.type == "cuda"
        else None
    )
    record = cake_kcp_jit.merge_record(ncols, n_chains, arch)
    return int(cake_kcp_jit.MODULES[record]["grid_rule"]["cols_per_cta"])


def merge_cols_per_cta(ncols: int, n_chains: int, rule: str) -> int | None:
    """D1 (the column-rule decision): the K4 ``cols_per_cta`` a launch asks for explicitly, ``None`` for the
    registry's choice of record (``fixed``: the shipped rule, c8 state / c16 map).  ``by_chains`` keeps the 8-column
    program for 128-column state chains up to :data:`COLS_RULE_CHAINS` concurrent chains and takes the 16-column
    program beyond it; 256-column map chains stay with the registry's c16.  Bitwise per layout (K4 REPORT §3)."""
    if rule == "by_chains" and int(ncols) == D and int(n_chains) > COLS_RULE_CHAINS:
        return 16
    return None


# Kernel parameter names of the frozen ABIs (K1 a217181fa, K2 a217181fa with its fuse_dv1 builds' aqk_tma (the
# internal Aqk rows) at the fifth parameter, where the earlier builds read dv1, K3 0637ec9f4, K4 eaa97cf5e): the logical
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
        "aqk_tma",
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
ABI_PARAMS["fwdh_o"] = (
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
    "tile_stride",
    "qg_tma",
    "aqk_tma",
    "o_out",
    "chunk_bos",
    "chunk_len",
    "scale",
)


# Walk-launch geometry one_cta_per_sm / tape_one_cta_per_sm: the K1 / K2 walk kernels' occupancy capped from the host.
# Their launch footprint (192 threads, 320 for the split-role fwdh offset build; 115712 B of dynamic shared memory)
# admits two CTAs per SM only at the full 228 KiB carveout; the smallest configuration that fits one CTA admits exactly
# one.  The runtime picks a kernel's configuration at launch from its function attribute (unset here) and otherwise from
# the device-wide preference (cudaDeviceSetCacheConfig): cudaFuncCachePreferL1 selects the smallest fitting
# shared-memory configuration, i.e. one CTA per SM.  The generated binding compiles the launch (cudaLaunchKernelExC with
# the fixed dynamic shared memory) and exports no symbol for the kernel's registered host function (its wrapper survives
# only as the mangled `__device_stub__`, which the runtime does not accept), so the per-function attribute cannot be set
# from the host; the knob brackets each capped launch with the device-wide preference instead (PreferL1 before,
# PreferNone after).  The preference is host-side state read at launch: launches already queued keep their configuration
# and only the bracketed launch takes PreferL1.  Host side only; no kernel, launch-configuration or argument change.
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


def _kcp_dav_launch(call: Any) -> bool:
    """Whether the backward's ``dav`` stage launches ``kcp_dav`` on this call (the kernel backend's rule): where its
    ``dAqk`` has a reader (:func:`cake_kcp_plan.kcp_dav_needed`) or under the knob ``kcp_dav="always"`` on the replay
    layout; never on a rank without local stages or tokens."""
    local = call.local
    if local is None or call.T_r == 0:
        return False
    return bool(
        cake_kcp_plan.kcp_dav_needed(local.plan)
        or (call.opts.kcp_dav == "always" and not _lean_passes(local))
    )


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


VIEW_CACHE_MAX = 64  # plan views of one local arena a CakeKDADetKcpWorkspace keeps (the least recently bound goes first)
SLICE_CACHE_MAX = (
    256  # pool[:n] views a CakeKDADetKcpWorkspace keeps (cleared when full)
)
ENSURE_CACHE_MAX = 128  # trimmed-path ensure records a CakeKDADetKcpWorkspace keeps (cleared when full)


def _plan_view(arena: Any, plan: Any) -> Any:
    """A view of the local arena ``arena`` (``cake_backend.CakeKDADetWorkspace``) under the plan of another pack whose
    layouts fit it: the same bytes, ``eye`` / ``flags`` and epoch counter, with its own layout views, its own launch
    templates (:data:`TEMPLATE_CACHE_ATTR`) and its own scratch table set (:data:`SCRATCH_TABLES_ATTR`, sized by its
    plan).  ``CakeKDADetWorkspace.views`` checks that each layout fits the arena's bytes."""
    view = arena._view(plan)
    view.__dict__.pop(TEMPLATE_CACHE_ATTR, None)
    view.__dict__.pop(SCRATCH_TABLES_ATTR, None)
    view.__dict__.pop(L2N_MOVED_ATTR, None)
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


# (record, arch) -> [cake_jit.load, cake_jit.MODULES, fetch, grid axes, FFI entry or None] of _launch_record_trim
_LAUNCH_PLANS: dict[tuple[str, str], list[Any]] = {}


def _launch_plan(name: str, arch: str) -> list[Any]:
    """The launch plan of record ``name`` on ``arch`` for :func:`_launch_record_trim`: where every grid argument
    follows the logical ones in the record's argument plan, an ``operator.itemgetter`` of the logical names (in plan
    order) and the grid axes after them (``fetch=None`` keeps the loop otherwise); the FFI entry is filled in by the
    first launch that gets its arguments (``cake_jit.load`` is cached per (record, arch): the same object on every
    launch)."""
    order = cake_jit._argument_plan(name)
    names = [logical for logical, _ in order if logical is not None]
    axes = tuple(axis for logical, axis in order if logical is None)
    trailing = all(logical is None for logical, _ in order[len(names) :])
    fetch = operator.itemgetter(*names) if trailing and len(names) >= 2 else None
    plan = [cake_jit.load, cake_jit.MODULES, fetch, axes, None]
    _LAUNCH_PLANS[(name, arch)] = plan
    return plan


def _launch_record_trim(
    local: Any, name: str, grid: tuple[int, int, int], values: dict[str, Any]
) -> None:
    """Trimmed host path: ``cake_jit.run_module(name, values, grid=grid, device_index=local.device_index)`` with the
    launch inlined -- the record's argument plan (cached by cake_jit), the arena's cached arch, the raw-stream FFI
    context.  Same record, same grid, same positional arguments, same stream as ``local.run``.  The record's launch
    plan (:func:`_launch_plan`: its FFI entry and one fetch of its logical arguments) is built once per (record,
    architecture) while ``cake_jit``'s ``load`` / ``MODULES`` stay the same objects; a missing logical argument
    raises as the loop below does."""
    jit = cake_jit
    plan = _LAUNCH_PLANS.get((name, local.arch))
    if plan is None or plan[0] is not jit.load or plan[1] is not jit.MODULES:
        plan = _launch_plan(name, local.arch)
    if plan[2] is not None:
        try:
            args = [*plan[2](values), *(int(grid[axis]) for axis in plan[3])]
        except KeyError:
            args = None  # the loop below names the missing argument
        if args is not None:
            if plan[4] is None:
                plan[4] = getattr(
                    jit.load(name, local.arch), jit.MODULES[name]["ffi_entry"]
                )
            with _ffi_stream(local.device_index):
                plan[4](*args)
            return
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
    ``run_program(name, args, device_index=)`` (the abstract launch ``args`` of :class:`CakeKDADetKcpCall` — H3's 09:59Z form),
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
            program in _WALK_FORMED
        ):  # the record of this launch's form: a fit program's two-CTA record is its ``_vh64``
            # variant, a fixed-rule program's other form its test-only ``_<form>`` record
            rule = getattr(jit, "WALK_FORM_RULES", {}).get(record, "fit")
            if rule == "fit" or isinstance(
                rule, Mapping
            ):  # both records shipped (a per-architecture rule: the fused forward walk)
                if walk_form == "split":
                    record += "_vh64"
                elif record in getattr(jit, "WALK_HEADS_RULE", {}):
                    # the full-width record the head-count rule names (the ABI carries the head count)
                    record = jit.walk_heads_record(
                        record, int(abi[ABI_PARAMS[program].index("num_heads")])
                    )
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
            # the registry's record of the launch; the positional ABI carries its column block
            params = ABI_PARAMS[program]
            ncols = int(abi[params.index("ncols")])
            cols = int(abi[params.index("cols_per_cta")])
            record = jit.merge_record(
                ncols, int(grid) // (ncols // cols), jit.device_arch(device_index)
            )
            if int(jit.MODULES[record]["grid_rule"]["cols_per_cta"]) != cols:
                raise ValueError(
                    f"the merge ABI carries cols_per_cta {cols}; the registry's record {record} is built for "
                    f"{jit.MODULES[record]['grid_rule']['cols_per_cta']}"
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
    tables of every pool, the collective splits and the K4 chains, derived once from ``cake_kcp_plan.CakeKDADetKcpContext`` /
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
    # held by the context's table-set cache (CakeKDADetKcpWorkspace.tables)
    on_context: bool = False
    # the trimmed path's absent-pool slot table is this set's own copy (absent_index; CakeKDADetKcpWorkspace.tables)
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
        if not self.on_context:  # held by a workspace: its stream changes record them (CakeKDADetKcpWorkspace._live_storages)
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
        set's; the entry pins them).  ``CakeKDADetKcpCall.indexed`` builds its ``Indexed`` from it on the trimmed host path."""
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
        arch_of: Any = None,
        device_index: int | None = None,
    ) -> str:
        """The record form (``"full"`` / ``"split"``) of walk launch ``name`` over the segments ``walk``: the registry's
        rule ``rule`` (``cake_kcp_jit.walk_form``) on the launch's chains (the chunk counts of its listed segments), the
        head count ``H``, the device's SM count, the ``walk_form`` knob and the architecture ``arch_of(device_index)``
        (passed to the rule as ``arch=`` only where ``arch_of`` is given, which :func:`_walk_form_of` does for a
        per-architecture rule alone; read on the first launch of a key only) -- evaluated once per
        ``(name, walk list, H, SM count, knob, rule, device index)`` of this table set and looked up on every later
        launch.  The rule's inputs are static per
        ``(ctx, H)`` (the walk lists and ``hp.seg_chunk_start`` are this set's, never mutated), so the lookup returns
        what a per-launch evaluation would; the evaluation itself (the ``fit`` rule list-schedules the chains in
        Python) leaves the launch path.  Keyed by value: a fresh equal list finds the same entry.  The first launch
        per key of a context pays that one evaluation (milliseconds of Python for the ``fit`` rule on long chain
        lists: the one-time cost of an eager first call, not a per-launch regression); every later launch pays a
        dictionary lookup."""
        key = (
            str(name),
            tuple(walk),
            int(H),
            int(sm_count),
            str(knob),
            rule,
            device_index,
        )
        cache = self.__dict__.setdefault("_walk_forms", {})
        form = cache.get(key)
        if form is None:
            if len(cache) >= self.ID_CACHE_MAX:
                cache.clear()
            seg = self.hp.seg_chunk_start
            extra = {} if arch_of is None else {"arch": str(arch_of(device_index))}
            form = cache[key] = str(
                rule(
                    name,
                    [int(seg[s + 1]) - int(seg[s]) for s in walk],
                    int(H),
                    int(sm_count),
                    knob=str(knob),
                    **extra,
                )
            )
        return form

    def walk_streams(
        self, walk: Sequence[int], H: int, sm_count: int, knob: str
    ) -> str:
        """The stream form (``"one"`` / ``"two"``) of the summary pair over the segments ``walk``
        (:func:`walk_streams` on the chunk counts of its listed segments, the head count ``H``, the device's SM count
        and the ``walk_streams`` knob): evaluated once per ``(walk list, H, SM count, knob)`` of this table set and
        looked up on every later pair (keyed by value, like :meth:`walk_form`)."""
        key = (tuple(walk), int(H), int(sm_count), str(knob))
        cache = self.__dict__.setdefault("_walk_streams", {})
        form = cache.get(key)
        if form is None:
            if len(cache) >= self.ID_CACHE_MAX:
                cache.clear()
            seg = self.hp.seg_chunk_start
            form = cache[key] = walk_streams(
                [int(seg[s + 1]) - int(seg[s]) for s in walk],
                int(H),
                int(sm_count),
                knob=str(knob),
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


class CakeKDADetKcpWorkspace:
    """The transient buffers of the KCP route on one rank (the design note §7.4) plus the existing backend's local
    arena (``cake_backend.CakeKDADetWorkspace``) for the token-side stages.

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
    lean backward layout (:data:`LOCAL_LAYOUT_BWD_LEAN`, 3332 B per internal token-head; the tail stages per chunk
    range) with its tail-count table (:data:`TAIL_COUNTS_ATTR`), the existing backend's layouts on the rows the lean
    layout does not serve (H 9-15).  ``lean_passes`` sizes a lean arena for that many tail passes (``None``:
    :data:`cake_kcp_plan.LEAN_PASSES_DEFAULT`, two; fewer passes launch fewer tail kernels for more bytes;
    :func:`cake_kcp_plan.cake_kda_det_train_kcp_workspace_bytes` takes the same arguments).  The saved-set tensors (``S_in`` = the c2
    output, the c1 receive tensor under ``save_transfer``, ``M_seg`` under ``save_transfer`` at ``P > 1``) are never
    arena slices: the calls allocate them fresh (SPEC-4).  A workspace is
    reusable across calls and layers of the same ``(ctx, H)``; a call with a larger need grows it, releasing each
    superseded pool and local arena (and the graph-cache entries pinning that arena) before allocating its successor
    (the old and the new storage never coexist, so a backward on a forward workspace peaks at the grown workspace plus
    the call's own peak).  ``bytes`` counts every pool storage once plus the local arena (:func:`cake_kda_det_train_kcp_workspace_bytes`
    is the plan module's prediction of the same quantity for an a2a workspace, CONSISTENCY-6).

    One workspace serves any number of prepared packs (contexts) whose calls run in stream order: the pools hold
    the largest need of the packs served and the local arena the largest of their layouts; a pack whose layout fits
    the arena runs on a view of it under its own plan (:func:`_plan_view`, at most :data:`VIEW_CACHE_MAX` cached per
    arena; each view has its own launch templates and its own scratch table set where the intra record writes one,
    :func:`cake_kcp_plan.local_scratch_table_bytes`), and the arena is replaced only by a pack that does not fit.
    A forward never replaces a backward arena with a forward-only one.  :meth:`reserve` sizes the workspace for a
    list of contexts up front, so the calls of those packs allocate nothing and every pointer stays put
    (``cake_kda_det_train_kcp_workspace_bytes([ctx, ...], H=H)`` predicts its ``bytes``).  No content crosses calls, so the results are
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
    other mode is refused with ``ValueError``.

    ``use_qk_l2norm_in_kernel`` (issue #7817) likewise fixes the in-kernel Q / K L2 norm mode of its calls
    (:attr:`qk_l2norm`) and its form (:attr:`l2n`, the existing backend's plan mode: ``"fused"`` for the flag,
    ``"off"`` without it): the local arena is planned in that mode (:func:`cake_kcp_plan.qk_l2norm_plan_kwargs`), the
    KCP pools do not depend on it, a call in the other mode is refused with ``ValueError``, and the mode runs on the
    replay local arena (``local_arena="lean"`` is refused with it; the lean layout follows the single-rank tail-pass
    layout of the mode's programs).  The rstd of a call without a saved set is per call; the backward's normalized
    q / k take the arena bytes of the buffers it moves onto its outputs (:func:`_l2n_row_hosts`), else per-call
    tensors.  ``_qk_l2norm_form="fallback"`` (with the flag; a test knob, as the single-rank explicit fallback
    workspace) runs the single-rank fallback form: the separate normalization and pullback programs around the plain
    records, whose results are bitwise those of the fused form.  With the flag on, construction raises
    ``NotImplementedError`` while the single-rank programs of the form are absent
    (:func:`cake_kcp_plan.require_qk_l2norm`: rank-independent inputs only, before any arena or plan), on every rank
    alike, as do :meth:`ensure` and :meth:`reserve` on another device architecture."""

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
        use_qk_l2norm_in_kernel: bool = False,
        _qk_l2norm_form: str | None = None,
    ) -> None:
        if save_mode not in SAVE_MODES:
            raise ValueError(
                f"CakeKDADetKcpWorkspace: save_mode must be one of {SAVE_MODES}, got {save_mode!r}"
            )
        # the backward arena form (cake_kcp_plan.LOCAL_ARENAS): "replay" (the default) or the opt-in "lean"
        self.local_arena = cake_kcp_plan.check_local_arena(local_arena)
        # the tail passes a lean arena is sized for (None: cake_kcp_plan.LEAN_PASSES_DEFAULT); fewer passes launch
        # fewer tail kernels for more bytes
        self.lean_passes = cake_kcp_plan.check_lean_passes(lean_passes)
        if self.lean_passes is not None and self.local_arena != "lean":
            raise ValueError(
                'CakeKDADetKcpWorkspace: lean_passes needs local_arena="lean", got '
                f"{self.local_arena!r}"
            )
        # the gate mode of every call on this workspace (and of its local arena's plan)
        self.gate = cake_backend._gate_mode(use_gate_in_kernel)
        if self.gate == "raw" and self.local_arena == "lean":
            raise NotImplementedError(
                'CakeKDADetKcpWorkspace: use_gate_in_kernel=True runs on the replay local arena; local_arena="lean" '
                "is not supported in the raw gate mode"
            )
        # the in-kernel Q / K L2 norm mode of every call on this workspace (and of its local arena's plan)
        self.qk_l2norm = cake_kcp_plan.check_qk_l2norm(use_qk_l2norm_in_kernel)
        self.l2n = cake_kcp_plan.qk_l2norm_mode(self.qk_l2norm, _qk_l2norm_form)
        if self.qk_l2norm and self.local_arena == "lean":
            # the lean layout's support of the mode follows the single-rank layout of its programs (issue #7817)
            raise NotImplementedError(
                'CakeKDADetKcpWorkspace: use_qk_l2norm_in_kernel=True runs on the replay local arena; local_arena="lean" '
                "is not supported with the in-kernel Q/K L2 norm yet"
            )
        # the architectures whose check of the mode passed, and the indexed CUDA devices of those (_require_l2n)
        self._l2n_ok: set[str | None] = set()
        self._l2n_dev_ok: set[torch.device] = set()
        self._require_l2n(
            device if device is not None else getattr(ctx, "device", None)
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
        # a pooled default workspace's lazy pools between calls (_default_release): name -> (pool, bytes), storage freed
        self._parked: dict[str, tuple[torch.Tensor, int]] = {}
        if ctx is not None:
            if H is None:
                raise ValueError("CakeKDADetKcpWorkspace: H is required with ctx")
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
        if self.qk_l2norm:
            self._require_l2n(dev)
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
        ``cake_kda_det_train_kcp_workspace_bytes(ctxs, H=H)`` for a workspace reserved once.  Host work and allocations only (no
        device wait).  ``ring=True`` also sizes the ring exchange's hop pools (:data:`cake_kcp_plan.LAZY_POOLS`,
        otherwise allocated by the first ``exchange="ring"`` call; ``kcp_arena_layout(...)["ring"]`` bytes per
        context, the largest over the contexts).  Returns :attr:`bytes`."""
        ctxs = _context_list(ctxs)
        if not ctxs:
            raise ValueError("CakeKDADetKcpWorkspace.reserve: no context given")
        dev: torch.device = torch.device(ctxs[0].device if device is None else device)
        if self.qk_l2norm:
            self._require_l2n(dev)
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

    def _require_l2n(self, device: torch.device | str | None) -> None:
        """With the flag on, refuse the mode where the single-rank programs of the workspace's form cannot run on
        ``device``'s architecture (:func:`cake_kcp_plan.require_qk_l2norm`, rank-independent inputs only, so every
        rank raises alike before any arena, plan or collective); a passed architecture is kept, so later calls
        check nothing again, and so is a passed indexed CUDA device, whose architecture is not looked up again.
        Nothing with the flag off."""
        if not self.qk_l2norm or device in self._l2n_dev_ok:
            return
        arch = cake_kcp_plan.qk_l2norm_arch(device)
        if arch not in self._l2n_ok:
            cake_kcp_plan.require_qk_l2norm(self.l2n, self.gate, arch)
            self._l2n_ok.add(arch)
        if (
            isinstance(device, torch.device)
            and device.type == "cuda"
            and device.index is not None
        ):
            self._l2n_dev_ok.add(device)

    def _check_device(self, device: torch.device) -> None:
        if self.device is None:
            self.device = device
        elif self.device != device:
            raise ValueError(
                f"CakeKDADetKcpWorkspace: lives on {self.device}, the call is on {device}"
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
            *key,
            cake_jit.sm_count(index),
            cake_jit.device_arch(index),
            gate=self.gate,
            **cake_kcp_plan.qk_l2norm_plan_kwargs(self.l2n),
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
            self.local = cake_backend.CakeKDADetWorkspace.for_problem(
                *key,
                device,
                backward=kind,
                save_mode=self.save_mode,
                fwd_layout=cake_kcp_plan.local_fwd_layout(kind, self.save_mode),
                gate=self.gate,
                **cake_kcp_plan.qk_l2norm_plan_kwargs(self.l2n),
            )
            return self.local
        # the lean arena: exactly the KCP forward layout and the lean backward layout, plus its tail-count table
        # (allocated with the arena, never by a call)
        self.local = cake_backend.CakeKDADetWorkspace(
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
            if self._parked:
                # a pooled default workspace's lazy pool: its storage again at this first read, where a fresh workspace
                # allocates it (the pool, its cached views and the ensure records stay as they were)
                held = self._parked.pop(name, None)
                if held is not None:
                    self._pools[name] = held[0]
                    held[0].untyped_storage().resize_(held[1])
            if self._grow(name, n, self.device):
                if self._slices:
                    self._drop_stale_slices()
                if fwd_lazy:
                    _graph_release(self)
        t = self._pools.get(name)
        if t is None:
            if n not in (None, 0) or self.device is None:
                raise KeyError(
                    f"CakeKDADetKcpWorkspace: pool {name!r} was not sized for this call (ensure() first)"
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
class CakeKDADetKcpSaved:
    """The saved context of one forward (``boundary_mode="saved"``, the design note §16): ``S_in`` (the fresh c2
    output), the owner's c1 receive tensor (``save_transfer``), ``M_seg`` (``save_transfer`` at ``P > 1``), the
    forward's carry plane ``carry`` (``P > 1``: the backward's tape replay reads it and skips the K1 offset re-walk +
    carry merges; ``None`` at ``P = 1`` and under the test knob ``saved_carry=0``) and the scalars of the call.  The
    int32 tables are rebuilt from ``ctx`` (droppable, §16 item 3).  No token-row ``Aqk`` / ``Akk``: the backward
    recomputes the internal rows it reads in both modes (version 2; version 1 held them).  ``gate`` (``"g"`` or
    ``"raw"``) and ``lower_bound`` (the raw mode's FP32 lower bound as a host float, ``None`` in ``"g"`` mode) are
    the gate mode of the forward that wrote the set: its tensors and version are the same in both modes, and a
    backward in the other mode or with another lower bound is refused.  ``qk_l2norm`` / ``qk_rstd``: the in-kernel
    Q / K L2 norm of the forward (``use_qk_l2norm_in_kernel=True``, issue #7817) and its FP32 rstd ``[2, T_r, H]``
    (plane 0 q, plane 1 k; empty on a rank without tokens), the last of :meth:`tensors`, recorded in :meth:`meta`
    only in the mode: a flag-off set has exactly the tensors, flags and meta it had before the mode existed, and the
    version stays.  The raw q / k are not members (the autograd node keeps the caller's tensors, as without the
    flag); a backward in the other mode is refused."""

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
    # use_qk_l2norm_in_kernel=True: the forward's FP32 rstd [2, T_r, H] (None with the flag off)
    qk_rstd: torch.Tensor | None = None
    qk_l2norm: bool = False  # the forward's in-kernel Q / K L2 norm mode

    _OPTIONAL = ("c1_recv", "M_seg", "carry")

    def tensors(self) -> tuple[torch.Tensor, ...]:
        out = (self.S_in,) + tuple(
            t for t in (getattr(self, n) for n in self._OPTIONAL) if t is not None
        )
        if self.qk_l2norm:  # the rstd last, in the mode only
            out += (self.qk_rstd,)
        return out

    def flags(self) -> tuple[bool, ...]:
        return tuple(getattr(self, n) is not None for n in self._OPTIONAL)

    def meta(self) -> dict[str, Any]:
        meta = {
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
        if self.qk_l2norm:  # in the mode only: a flag-off set's meta is unchanged
            meta["qk_l2norm"] = True
        return meta

    @classmethod
    def from_tensors(
        cls,
        tensors: Sequence[torch.Tensor],
        flags: Sequence[bool],
        meta: dict[str, Any],
        ctx: Any,
    ) -> CakeKDADetKcpSaved:
        it = iter(tensors)
        S_in = next(it)
        opt = {
            n: (next(it) if f else None)
            for n, f in zip(cls._OPTIONAL, flags, strict=True)
        }
        if meta.get("qk_l2norm"):
            opt["qk_rstd"] = next(it)
        return cls(S_in=S_in, ctx=ctx, **opt, **meta)

    @property
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors())


@dataclasses.dataclass
class CakeKDADetKcpOptions:
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
    the backward reads its own call's value, the forward's is not kept in :class:`CakeKDADetKcpSaved`),
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
    remaining launches, bitwise; ``_kcp_test_knobs`` only), ``kcp_dav`` (``"auto"`` shipped: the backward launches
    ``kcp_dav`` only where :func:`cake_kcp_plan.kcp_dav_needed`; ``"always"`` = on every backward of the replay
    layout, the same bits; ``_kcp_test_knobs`` only), ``bwd_tape_stream`` (the saved-set backward's tape replay:
    ``"post"`` shipped, on a side stream forked after the adjoint offset walk, beside the b1 / b2 exchange; ``"one"`` =
    the call's stream; the other :data:`BWD_TAPE_STREAMS` forms move the fork / issue point or add a high-priority
    stream; the same bits; ``_kcp_test_knobs`` only), ``precision_mode`` (K3's static program; set only through
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
    walk_form: str = WALK_FORM_DEFAULT  # the record form of the K1 / K2 and fused forward walk launches: "auto" (shipped) = the registry's rule per launch; "full" / "split" via _kcp_test_knobs only
    walk_streams: str = WALK_STREAMS_DEFAULT  # the stream form of the forward's summary pair: "auto" (shipped) = the walk_streams rule per pair; "one" / "two" via _kcp_test_knobs only
    graph: str = GRAPH_DEFAULT  # graph replay: public keyword; "replay" captures a call once per validity key and replays it (eager on a key miss); also reachable through _kcp_test_knobs for the A/B record
    graph_cap: str = GRAPH_CAP_DEFAULT  # the cache-config form of a captured call ("node_attr" shipped | "none" the previous form of record | "capture_pref" rejected | "node_attr_post"); _kcp_test_knobs only
    host_trim: int = HOST_TRIM_DEFAULT  # shipped 1 = the trimmed host path of the eager call (same launches, bitwise); 0 = the previous path, _kcp_test_knobs only
    merge_fuse: str = MERGE_FUSE_DEFAULT  # "carry" = carry_start + carry_in (dcarry_end + dcarry_cross) in one K4 launch; _kcp_test_knobs only
    saved_carry: int = SAVED_CARRY_DEFAULT  # shipped 1 = the forward's carry plane stays in the saved set and the saved-mode backward skips the P > 1 re-walk + carry merges; 0 = the re-walk, _kcp_test_knobs only
    fwd_o_fuse: int = FWD_O_FUSE_DEFAULT  # shipped 1 = the forward that keeps no tape runs kcp_fwdh_o (out, no tape) in place of the tape walk(s) + fwd_o; 0 = the tape walk(s) + fwd_o, _kcp_test_knobs only
    intra_rows: str = INTRA_ROWS_DEFAULT  # shipped "ckpt" = a checkpoint recompute keeps its intra rows for its backward; "always" / "never" via _kcp_test_knobs only
    rows_probe: int = 0  # 1 = a backward that binds kept intra rows also launches its intra program and records the bitwise comparison (host sync); _kcp_test_knobs only
    kcp_dav: str = KCP_DAV_DEFAULT  # the backward's kcp_dav: "auto" (shipped) only where cake_kcp_plan.kcp_dav_needed; "always" on every backward of the replay layout (the same bits), via _kcp_test_knobs only
    bwd_tape_stream: str = BWD_TAPE_STREAM_DEFAULT  # the saved-set backward's tape replay: a side-stream form of BWD_TAPE_STREAMS (shipped "post") or "one" (the call's stream; the same bits), via _kcp_test_knobs only

    def check(self) -> CakeKDADetKcpOptions:
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


def _options(kwargs: dict[str, Any]) -> CakeKDADetKcpOptions:
    """Validate the entry points' ``**options`` (unknown keywords are a ``TypeError``); ``_kcp_test_knobs`` goes
    through the plan module's validator (``precision_mode``) and, for the test knob ``transfer_walk`` (K3's
    ``tf32_hilo`` / ``bf16x3`` programs of ``build.test.json``: the TF32 hi/lo walk and the full-width BF16x3 form),
    through the same environment gate here, so the public API can never select a non-zero ``precision_mode`` or a
    non-shipped transfer-walk program."""
    pre = kwargs.get("_kcp_options")
    if (
        pre is not None
    ):  # the options validated by cake_kda_det_kcp on the caller's frame, passed through the autograd node
        if not isinstance(pre, CakeKDADetKcpOptions) or len(kwargs) != 1:
            raise TypeError(
                "_kcp_options is the validated CakeKDADetKcpOptions of the entry point and travels alone"
            )
        return pre
    if not kwargs:
        return _default_options()
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
                    if key
                    in (
                        "fused_walks",
                        "host_trim",
                        "saved_carry",
                        "fwd_o_fuse",
                        "rows_probe",
                    )
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
    return CakeKDADetKcpOptions(
        **kwargs,
        precision_mode=int(knobs["precision_mode"]),
        transfer_walk=transfer_walk,
        p_rule=p_rule,
        **p2,
    ).check()


# The validated options of an empty ``**options``: (the plan module's SAVE_TRANSFER_DEFAULT it resolved, the options'
# field values in ``dataclasses.fields`` order).
_DEFAULT_OPTIONS: list[tuple[Any, tuple[Any, ...]]] = []


def _default_options() -> CakeKDADetKcpOptions:
    """``_options({})``: an entry point called without keyword options validates the defaults, which depend on no
    input (only on this module's defaults and the plan module's ``SAVE_TRANSFER_DEFAULT``, the cache key); the first
    call validates them through the full path and keeps the field values, every call builds a fresh
    :class:`CakeKDADetKcpOptions` from them (equal field by field).  The copy is the dataclass ``__init__`` with the values
    in field order: no ``__post_init__`` (the check stays explicit), and unlike a ``__dict__`` copy it also works for
    a ``slots=True`` dataclass, at about the same host cost per call."""
    key = cake_kcp_plan.SAVE_TRANSFER_DEFAULT
    if not _DEFAULT_OPTIONS or _DEFAULT_OPTIONS[0][0] is not key:
        opts = _options({"_kcp_test_knobs": None})
        values = tuple(
            getattr(opts, f.name) for f in dataclasses.fields(CakeKDADetKcpOptions)
        )
        _DEFAULT_OPTIONS[:] = [(key, values)]
    return CakeKDADetKcpOptions(*_DEFAULT_OPTIONS[0][1])


def _with_option_signature(fn):
    """Give entry point ``fn`` the signature of the keywords it accepts: its own parameters without ``**options``,
    then one keyword-only parameter per :data:`_OPTION_KEYS` with the default of its :class:`CakeKDADetKcpOptions`
    field, then ``_kcp_test_knobs`` (``None``).  ``inspect.signature(fn).bind_partial(deterministic=False)`` then
    raises ``TypeError``, as for every other entry point.  The implementation still takes ``**options`` and
    validates them with :func:`_options` (an unknown keyword is a ``TypeError`` at the call; the validated options
    the autograd node passes on as ``_kcp_options`` are internal).  Returns ``fn`` itself."""
    fields = {f.name: f for f in dataclasses.fields(CakeKDADetKcpOptions)}
    sig = inspect.signature(fn)
    params = [
        p
        for p in sig.parameters.values()
        if p.kind is not inspect.Parameter.VAR_KEYWORD
    ]
    params += [
        inspect.Parameter(
            key,
            inspect.Parameter.KEYWORD_ONLY,
            default=fields[key].default,
            annotation=fields[key].type,
        )
        for key in _OPTION_KEYS
    ]
    params.append(
        inspect.Parameter(
            "_kcp_test_knobs",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="dict[str, Any] | None",
        )
    )
    fn.__signature__ = sig.replace(parameters=params)
    return fn


def _exchange_scheme(tb: _Tables, opts: CakeKDADetKcpOptions) -> str:
    """``a2a`` or ``ring`` (SPEC-2): ``auto`` is ``a2a`` in Phase 1 (the ring path arrives with gate 5); an explicit
    ``ring`` needs every document chain within ``RING_MAX_HOPS`` hops."""
    if opts.exchange == "ring":
        if tb.max_hops > RING_MAX_HOPS:
            raise ValueError(
                f"exchange='ring' needs n_d_max - 1 <= {RING_MAX_HOPS}, this partition has {tb.max_hops}"
            )
        if opts.save_transfer:  # unreachable through _options() (CakeKDADetKcpOptions.check resolves / refuses it); kept as the guard of the contract
            raise NotImplementedError(
                "exchange='ring' receives no c1 tensor: save_transfer=True has no ring form"
            )
        return "ring"
    return "a2a"


# ----------------------------------------------------------------------------------------------------------------
# One call (forward or backward): buffers, launches, collectives


class CakeKDADetKcpCall:
    """The state of one forward or backward call on one rank: tables, buffers (workspace pools and the call's fresh
    tensors), operands, backend, the launch / collective counters and the issue sequence (R7 debug counter)."""

    def __init__(
        self,
        *,
        ctx: Any,
        H: int,
        ws: CakeKDADetKcpWorkspace,
        tb: _Tables,
        backend: Backend,
        opts: CakeKDADetKcpOptions,
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
        # the in-kernel Q / K L2 norm (use_qk_l2norm_in_kernel=True) and its form (the workspace's l2n mode; both
        # set by _prepare), and the mode's values for every local stage (set by the call's body: _qk_l2norm_begin)
        self.l2n = False
        self.l2n_mode = "off"
        self.l2n_values: dict[str, Any] | None = None
        self.exchange = _exchange_scheme(tb, opts)
        self.precision_mode = int(opts.precision_mode)
        self.transfer_walk = str(opts.transfer_walk)
        if opts.p_rule is not None and opts.p_rule != getattr(
            ctx, "p_rule", cake_kcp_plan.P_RULE_DEFAULT
        ):
            raise ValueError(
                f"_kcp_test_knobs p_rule {opts.p_rule!r} differs from the context's plan rule {getattr(ctx, 'p_rule', cake_kcp_plan.P_RULE_DEFAULT)!r}: give the knob to cake_kda_det_train_kcp_build_context"
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
        # "auto": the walk_streams rule per summary pair; "one" / "two" forced
        self.walk_streams = str(opts.walk_streams)
        self.walk_streams_log: list[
            tuple[str, str, int]
        ] = []  # (decided form, form run, n_walk) per summary pair (_offset_pair)
        self.merge_fuse = str(opts.merge_fuse)  # fused carry merges
        self.saved_carry = int(
            opts.saved_carry
        )  # 1 (shipped) = the saved carry plane; 0 = the re-walk (test knob)
        self.fwd_o_fuse = int(
            opts.fwd_o_fuse
        )  # 1 (shipped) = the fused forward walk; 0 = the tape walk(s) + fwd_o (test knob)
        # the forward state walk program of the forward in progress (_forward sets it): kcp_fwdh_o, or kcp_fwdh_tape
        # where the forward keeps its tapes (the replay forward) or the knob asks for the previous form
        self.fwd_walk = "kcp_fwdh_tape"
        # a workspace=None call on a pooled default workspace: (pool key, weak context); None otherwise (_prepare)
        self.default: tuple[tuple[Any, ...], Any] | None = None
        # a call without a workspace (workspace=None: a pooled or a fresh default workspace; _prepare)
        self.ws_default = False

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


def _record(kind: str, call: CakeKDADetKcpCall) -> None:
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
        "walk_streams": list(
            call.walk_streams_log
        ),  # (decided form, form run, n_walk) per summary pair: the walk_streams rule or knob, and the streams it ran on
        # the host path of this call (1 trimmed = shipped / 0 previous; same launches)
        "host_trim": call.host_trim,
        "merge_fuse": call.merge_fuse,  # the fused carry merges (test knob)
        "saved_carry": call.saved_carry,  # 1 (shipped) = the saved carry plane; 0 = the re-walk (test knob)
        "fwd_o_fuse": call.fwd_o_fuse,  # 1 (shipped) = the fused forward walk; 0 = the tape walk(s) + fwd_o (test knob)
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
        "intra_rows": call.scratch.get(
            "intra_rows"
        ),  # INTRA_ROWS: "kept" (forward), "bound" / "ignored" (backward of a set with rows), None otherwise
        "intra_rows_bytes": call.scratch.get(
            "intra_rows_bytes"
        ),  # the bytes of the tensor holding the kept rows (forward: kept; backward: bound), None otherwise
        "intra_rows_form": call.scratch.get(
            "intra_rows_form"
        ),  # a backward that bound kept rows: "carrier" (they lie in its dq / dk) or "copy" (copied into the arena)
        "rows_probe": call.scratch.get(
            "rows_probe"
        ),  # the rows_probe test knob's bitwise comparison (backward), None otherwise
        # a backward's kcp_dav: "launched" / "skipped" (cake_kcp_plan.kcp_dav_needed or the knob kcp_dav="always");
        # None on a forward or a backward without a local stage
        "kcp_dav": call.scratch.get("kcp_dav"),
        # a saved-set backward's tape replay: "one" (the call's stream) or the side-stream form run (BWD_TAPE_STREAMS);
        # None on a forward or a replay-mode backward
        "bwd_tape_stream": call.scratch.get("bwd_tape_stream"),
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
    if call.l2n:  # use_qk_l2norm_in_kernel=True only: a flag-off record keeps its keys
        # the form ("fused" / "fallback") and the (read_rstd, write_rstd) of the mode's first q / k reader (None
        # without local stages of the mode, as on a rank without tokens)
        LAST_CALL[kind]["l2n"] = call.l2n_mode
        LAST_CALL[kind]["l2n_rstd"] = call.scratch.get("l2n_rstd")
        # (record, saved-rstd twin) where this call's launches ran the twin in the record's place, else None
        LAST_CALL[kind]["l2n_twin"] = call.scratch.get("l2n_twin")
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
    call: CakeKDADetKcpCall,
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
    call: CakeKDADetKcpCall,
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
    call: CakeKDADetKcpCall,
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


def _ring_forward(call: CakeKDADetKcpCall) -> None:
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


def _ring_start_walk(call: CakeKDADetKcpCall) -> None:
    """The overlap schedule's document-start tape walk of the ring forward (under hop 1; alone when there is none;
    the forward's walk program, :func:`_forward`)."""
    tb = call.tb
    carry = call.indexed("carry", tb.carry_index, tb.carry_tiles) if tb.P > 1 else None
    call.walk(call.fwd_walk, tb.walk_start, h0=carry, out=call.out)


def _ring_end_walk(call: CakeKDADetKcpCall) -> None:
    """The overlap schedule's ``dcarry_end`` merge and document-end tape walk of the ring backward (under hop 1;
    alone when there is none)."""
    tb = call.tb
    dcarry = (
        call.indexed("carry", tb.dcarry_index, tb.carry_tiles) if tb.P > 1 else None
    )
    if tb.P > 1:
        call.merge("dcarry_end")
    call.walk("kcp_dhu_tape", tb.walk_end, dht=dcarry)


def _ring_backward(call: CakeKDADetKcpCall) -> None:
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


def _summaries(call: CakeKDADetKcpCall) -> None:
    """The fragment summaries ``[B_f | M_f^T]`` into the c1 send layout: at ``P = 1`` K1 ``_offset`` and K3 write the
    crossing-out fragments' tiles directly; at ``P > 1`` both walk every segment into the staging planes and K4
    composes the crossing-out fragments' segments (forward family -> ``B_f``, adjoint family -> ``M_f^T``)."""
    tb = call.tb
    if tb.P == 1:
        _offset_pair(
            call,
            tb.walk_offset,
            call.indexed("c1_send", tb.out_index, tb.c1_b_tiles),
            call.indexed("c1_send", tb.out_index, tb.c1_m_tiles),
        )
        return
    _stage_walks(call)
    call.merge(
        "map"
    )  # B_f (from 0) and M_f (from I) of every crossing-out fragment, forward family
    call.transpose_tiles(
        "c1_send", tb.mf_tiles, "c1_send", tb.c1_m_tiles
    )  # M_f^T into the c1 pairs (bitwise)


def _stage_walks(call: CakeKDADetKcpCall) -> None:
    """``P > 1``: the K1 ``_offset`` / K3 walks of every segment into the staging planes ``stage_b`` / ``stage_m``."""
    tb = call.tb
    _offset_pair(
        call,
        tb.walk_offset,
        call.indexed("stage_b", tb.seg_index, tb.stage_tiles),
        call.indexed("stage_m", tb.seg_index, tb.stage_tiles),
    )


def _offset_pair(
    call: CakeKDADetKcpCall, walk: Sequence[int], out_b: Indexed, out_m: Indexed
) -> None:
    """The summary pair over ``walk``: K1 ``_offset`` (the walk from zero into ``out_b``) and K3 (the transfer into
    ``out_m``), in this issue order.  The two read the same local operands and write disjoint tiles, and the next
    launch reads both, so they may run concurrently: on two streams (:func:`walk_streams`, the call's
    ``walk_streams`` knob) K1 goes on the call's stream and K3 on the device's side stream
    (:func:`_pair_streams`) between a fork (an event recorded on the call's stream, waited by the side stream) and
    a join (an event recorded on the side stream, waited by the call's stream) -- every launch after the pair, the
    ``c1`` collective included, is ordered after both.  The fork / join are stream-ordered event operations, so a
    CUDA-graph capture of the call captures them as graph dependencies.

    Allocator: the side stream allocates nothing.  Its operands -- the arena views, the pools ``out_m`` addresses
    (built here, before the fork, on the call's stream) and the table set's device tables -- are owned by the
    workspace, the call or the table set and stay alive past the join, which the call's stream waits for before any
    later launch; a block allocated on the call's stream is reused by the caching allocator only on that stream, so
    no reuse can overtake the side stream's work.  The device tables of a launch are uploaded on the current stream
    at its table set's first launch (never under a capture), so the first pair of a table set over ``walk`` on a
    device runs on one stream (its K3 would otherwise read a table K1 uploads after the fork); once both launches
    have been issued, their tables are resident and later pairs take the decided form.  Two streams run only on the
    trimmed host path (``host_trim=1``) of the kernel backend on a CUDA device; the previous host path and other
    backends keep one stream.  ``call.walk_streams_log`` records ``(decided form, form run, n_walk)``."""
    if len(walk) == 0:
        return
    k3_args = {"out": out_m, "init_mode": 0, "use_u": 0, "write_transposed": 0}
    if not isinstance(
        call, CakeKDADetKcpCall
    ):  # a test's driver call: the two launches in order
        call.walk("kcp_fwdh_offset", walk, h0=None, out=out_b)
        call.walk("kcp_transfer_walk", walk, **k3_args)
        return
    decided, streams, seen, key = "one", None, None, None
    if call.device.type == "cuda" and isinstance(call.backend, KernelBackend):
        index = (
            call.device.index
            if call.device.index is not None
            else torch.cuda.current_device()
        )
        decided = call.tb.walk_streams(
            walk, call.H, cake_jit.sm_count(index), call.walk_streams
        )
        if decided == "two" and call.host_trim:
            seen = call.tb.__dict__.setdefault("_pair_seen", set())
            key = (id(walk), index)  # the table set's own list object
            pair = _pair_streams(index)  # created in the eager first pair of the key
            streams = pair if key in seen else None
    if streams is None:
        call.walk("kcp_fwdh_offset", walk, h0=None, out=out_b)
        call.walk("kcp_transfer_walk", walk, **k3_args)
        if seen is not None:
            seen.add(key)  # both launches issued: their device tables are resident
    else:
        side, fork, join = streams
        main = torch.cuda.current_stream(call.device)
        fork.record(main)
        side.wait_event(fork)
        try:
            call.walk("kcp_fwdh_offset", walk, h0=None, out=out_b)
            with torch.cuda.stream(side):
                call.walk("kcp_transfer_walk", walk, **k3_args)
        finally:  # the join also on an error: a capture never ends with the side stream unjoined
            join.record(side)
            main.wait_event(join)
    call.walk_streams_log.append(
        (decided, "one" if streams is None else "two", len(walk))
    )


def _fwd_tapes(
    call: CakeKDADetKcpCall,
    *,
    out: torch.Tensor | None,
    carry: Indexed | None,
    wait_first: bool,
) -> None:
    """The forward tape walk(s) after ``S_in`` is complete (the forward's walk program ``call.fwd_walk``: the tape
    walk, or the fused walk that writes ``out`` instead of the tapes, :func:`_forward`).  ``fused_walks``: one launch
    from one pool -- the carry plane at ``P > 1`` (``h0_index_fused``: carries, the ``S_in`` copies of ``carry_in``,
    ``-1`` zero states), ``S_in`` itself at ``P = 1`` -- over every live segment (``wait_first``) or the crossing-in
    fragments' segments (the document-start walk ran under c1); otherwise the three launches of the design note §3.5
    (the document-start walk here only under ``wait_first``)."""
    tb, walk = call.tb, call.fwd_walk
    h0_in = call.indexed("S_in", tb.h0_index, tb.s_in_tiles)
    if tb.fused_walks:
        h0 = (
            call.indexed("carry", tb.h0_index_fused, tb.carry_tiles)
            if tb.P > 1
            else h0_in
        )
        call.walk(
            walk,
            tb.walk_live if wait_first else tb.walk_in_fused,
            h0=h0,
            out=out,
        )
        return
    if wait_first:
        call.walk(
            walk, tb.walk_start, h0=carry, out=out
        )  # document starts: after wait(c2)
    call.walk(walk, tb.walk_in_p0, h0=h0_in, out=out)
    if tb.P > 1:
        call.walk(walk, tb.walk_in_p1, h0=carry, out=out)


def _adj_tapes(
    call: CakeKDADetKcpCall, *, dcarry: Indexed | None, wait_first: bool
) -> None:
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


def _fwd_buffers(
    call: CakeKDADetKcpCall,
    *,
    save: bool,
    replay: bool,
    bound: dict[str, torch.Tensor] | None,
) -> None:
    """The forward's per-call KCP buffers (:func:`_forward`): ``S_in`` (fresh, or the handle's); with
    ``save_transfer`` on a saving forward the c1 receive tensor (fresh, or the handle's) and at ``P > 1`` ``stage_m``;
    the carry plane of a saving ``P > 1`` forward under ``saved_carry``."""
    tb, opts = call.tb, call.opts
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


def _forward(
    call: CakeKDADetKcpCall,
    *,
    save: bool,
    replay: bool,
    bound: dict[str, torch.Tensor] | None = None,
    alloc_out: bool = False,
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
    every published bit is the same.
    The forward's state walk program (``call.fwd_walk``, every forward tape-walk launch site): the fused walk
    ``kcp_fwdh_o`` (``out`` from the walk, no tape, no ``fwd_o`` launch) unless the forward keeps its tapes
    (``replay``: the backward reads them) or the ``fwd_o_fuse`` knob is ``0``; then the tape walk and ``fwd_o``."""
    tb = call.tb
    wait_first = call.schedule == "wait_first"
    fuse_o = not replay and call.fwd_o_fuse == 1
    call.fwd_walk = "kcp_fwdh_o" if fuse_o else "kcp_fwdh_tape"
    if replay:  # the replay's prep (the GLM prep's rows, the internal dout copy), then the internal Aqk / Akk rows
        _fwd_buffers(call, save=save, replay=replay, bound=bound)
        _bwd_rows(call, None)
    else:  # the intra program writes the GLM prep's rows itself (write_prep = 1); no token-row copies are saved
        if call.l2n:
            _qk_l2norm_stage(call)
        if save and _intra_rows_keep(call):
            # a checkpoint recompute (INTRA_ROWS): the intra program writes its internal Aqk / Akk rows into a fresh
            # tensor the backward of this saved set takes (_fwd_body); this forward's WY and output programs read
            # them there
            _keep_intra_rows(call)
        call.stage("fwd_intra")
        call.stage("wy_fwd")
        # the two local stages read neither the KCP buffers nor ``out`` (their records bind no ``out``, and the
        # buffers are K1-K4 / collective operands): allocated after the stages are issued, so the first launch does
        # not wait for them (host order only: the same launches with the same arguments in the same order)
        _fwd_buffers(call, save=save, replay=replay, bound=bound)
        if alloc_out:
            call.out = _alloc(
                (call.T_r, call.H, D), call.backend.dtypes["out"], call.device
            )
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
            call.walk(call.fwd_walk, tb.walk_start, h0=carry, out=call.out)
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
                call.fwd_walk, tb.walk_start, h0=carry, out=call.out
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
    if not replay and not fuse_o:
        call.stage("fwd_o")


def _bwd_rows(call: CakeKDADetKcpCall, saved: CakeKDADetKcpSaved | None) -> None:
    """The backward's local rows before its walks (``saved=None``: the replay mode): the prep (the GLM prep's rows,
    the internal ``dout`` copy), the intra program (the internal ``Aqk`` / ``Akk`` rows the WY, dav and gradient
    programs read) and the WY program.  In-kernel Q / K L2 norm: the fallback form's separate normalization first;
    in the fused form the plan's first q / k reader first (its intra program where the existing backend runs that
    before the prep, :func:`cake_kcp_plan.qk_l2norm_intra_first`: the prep then reads the normalized rows; the two
    programs read none of each other's rows).  The outputs :func:`_bwd_body` deferred are allocated right after the
    first of the two (:func:`_late_grads`).  A backward that took its forward's intra rows (:data:`INTRA_ROWS`,
    :func:`_ckpt_rows_take`): no intra launch, the rows are bound in its place (the ``dq`` / ``dk`` carrier's) or
    already in the arena (:func:`_ckpt_rows_bind`)."""
    if call.l2n:
        _qk_l2norm_stage(call)
    if (
        call.l2n_mode == "fused"
        and call.local is not None
        and cake_kcp_plan.qk_l2norm_intra_first(call.local.plan)
    ):
        call.stage("fwd_intra")
        _late_grads(call)
        call.stage("bwd_prep", saved=saved)
    else:
        call.stage("bwd_prep", saved=saved)
        _late_grads(call)
        if call.scratch.get("intra_rows") != "bound":
            call.stage("fwd_intra")
        else:
            rows = call.scratch.pop("intra_rows_v", None)
            if call.opts.rows_probe:
                call.stage("fwd_intra")  # on the arena's own rows
                _rows_probe(call, rows)
                if call.scratch["intra_rows_form"] == "copy":
                    _copy_intra_rows(call, rows)
            if call.scratch["intra_rows_form"] == "carrier":
                _bind_intra_rows(call, rows)
            del rows
            if cake_kcp_plan.intra_writes_tables(call.local.plan, LOCAL_PATH_BWD):
                # the arena's scratch table set, which the skipped launch would have allocated (no program reads it)
                _scratch_tables(call.local)
    call.stage("wy")


def _intra_rows_keep(call: CakeKDADetKcpCall) -> bool:
    """Whether this saving forward keeps its intra rows (:data:`INTRA_ROWS`): ``"ckpt"`` inside the autograd engine's
    backward pass (a checkpoint recompute), ``"always"`` on every saving forward, ``"never"`` on none; never on a rank
    without local stages or tokens, with more internal rows than twice its tokens, under graph replay or inside a
    capture, on a lean arena (its backward binds none), where the backward's intra program is its first q / k reader
    (it normalizes: the launch stays), with the fused in-kernel L2 norm outside :data:`INTRA_ROWS_L2N_ARCHS`, nor on a
    call without a workspace (its default workspace's storage is freed between the two calls)."""
    mode = getattr(call.opts, "intra_rows", INTRA_ROWS_DEFAULT)
    if mode == "never" or (mode == "ckpt" and not _kcp_phase_in_backward()):
        return False  # the original forward of the shipped rule: one phase query
    if call.local is None or call.T_r == 0 or call.ws_default:
        return False
    if not isinstance(call.backend, KernelBackend) or call.opts.graph != "off":
        return False
    if _capturing() or getattr(call.ws, "local_arena", None) == "lean":
        return False
    if 2 * call.T_r < call.local.plan.R:
        return False  # the rows fit neither the dq / dk pair nor the outputs allocated after their copy
    if call.l2n_mode == "fused" and (
        call.local.arch not in INTRA_ROWS_L2N_ARCHS
        or cake_kcp_plan.qk_l2norm_intra_first(call.local.plan)
    ):
        return False
    return True


def _keep_intra_rows(call: CakeKDADetKcpCall) -> None:
    """The kept rows of a saving forward (:data:`INTRA_ROWS`), bound in place of the arena's ``aqk_int`` / ``akk_int``:
    in the first ``256 R H`` bytes of a fresh BF16 ``[2, T_r, H, 128]`` tensor, the backward's ``dq`` / ``dk`` pair
    (its programs read them there or it copies them into its arena first, :func:`_ckpt_rows_bind`); a fresh
    ``[2, R, H, 64]`` tensor where ``dq`` / ``dk`` are not of the rows' dtype.  :func:`_fwd_body` hands them to
    :data:`_CKPT_ROWS`."""
    views = call.scratch["views"]
    shape, dtype = tuple(views.aqk_int.shape), views.aqk_int.dtype
    dt = call.backend.dtypes
    if dt["dq"] == dt["dk"] == dtype:
        holder = _alloc((2, call.T_r, call.H, D), dtype, call.device)
        rows = holder.view(-1)[: 2 * views.aqk_int.numel()].view(2, *shape)
    else:
        holder = rows = _alloc((2, *shape), dtype, call.device)
    _bind_intra_rows(call, rows)
    call.scratch["intra_rows_t"] = (holder, rows)
    call.scratch["intra_rows"] = "kept"
    call.scratch["intra_rows_bytes"] = holder.numel() * holder.element_size()


# The decision per plan of :func:`_rows_carrier_ok`: {id(plan): (a weak reference to it, the registry it read, ok)}
_ROWS_CARRIER_KEPT: dict[int, tuple[Any, Any, bool]] = {}


def _rows_carrier_ok(p: Any) -> bool:
    """Whether the backward's ``dq`` / ``dk`` pair may carry the kept intra rows (:data:`INTRA_ROWS`): no record of
    its stages before the GLM intra program (:data:`_L2N_LATE_STAGES`) binds ``dq`` or ``dk`` and no record from
    that program on binds ``aqk_int`` or ``akk_int`` (the KCP's own programs bind no output), so the rows are read
    before either output is written.  Kept per plan while the registry is the same object."""
    kept = _ROWS_CARRIER_KEPT.get(id(p))
    if kept is None or kept[0]() is not p or kept[1] is not cake_jit.MODULES:
        for stale in [k for k, v in _ROWS_CARRIER_KEPT.items() if v[0]() is None]:
            del _ROWS_CARRIER_KEPT[stale]
        early, late = _bwd_stage_bindings(p)
        ok = early.isdisjoint(("dq", "dk")) and late.isdisjoint(("aqk_int", "akk_int"))
        kept = (weakref.ref(p), cake_jit.MODULES, ok)
        _ROWS_CARRIER_KEPT[id(p)] = kept
    return kept[2]


def _ckpt_rows_put(S_in: torch.Tensor, kept: tuple[torch.Tensor, torch.Tensor]) -> None:
    """A saving forward's kept rows into :data:`_CKPT_ROWS` under its saved set's ``S_in`` (a fresh tensor of the call
    or the recompute handle's), with a weak reference whose callback drops the entry with ``S_in``."""
    key = id(S_in)
    ref = weakref.ref(S_in, functools.partial(_ckpt_rows_drop, key=key))
    _CKPT_ROWS[key] = (ref, *kept)


def _ckpt_rows_take(call: CakeKDADetKcpCall, saved: CakeKDADetKcpSaved) -> None:
    """The backward of ``saved``: the rows a forward left in :data:`_CKPT_ROWS` under the set's ``S_in`` are taken
    (the entry goes in every case) and, where this backward uses them, put in the call (``scratch["ckpt_rows"]``) for
    :func:`_ckpt_rows_bind`; not used (dropped: the intra program runs) on a rank without local stages or tokens, by
    another backend, on a lean layout, under graph replay or a capture, with the SPEC-3 probe, where the intra
    program is the first q / k reader, or for rows of another shape."""
    entry = _CKPT_ROWS.pop(id(saved.S_in), None)
    if entry is None or entry[0]() is not saved.S_in:
        return
    rows = entry[2]
    if (
        call.local is None
        or call.T_r == 0
        or not isinstance(call.backend, KernelBackend)
        or call.opts.graph != "off"
        or call.opts.spec3_probe != "off"
        or _capturing()
        or _lean_passes(call.local)
        or (
            call.l2n_mode == "fused"
            and cake_kcp_plan.qk_l2norm_intra_first(call.local.plan)
        )
        or tuple(rows.shape) != (2, int(call.ctx.R), call.H, BT)
        or rows.dtype != torch.bfloat16
        or 2 * call.T_r < call.local.plan.R
    ):
        call.scratch["intra_rows"] = "ignored"
        return
    call.scratch["intra_rows"] = "bound"
    call.scratch["intra_rows_bytes"] = entry[1].numel() * entry[1].element_size()
    call.scratch["ckpt_rows"] = entry[1:]


def _ckpt_rows_bind(
    call: CakeKDADetKcpCall, hosts: dict[str, tuple[str, ...]]
) -> torch.Tensor | None:
    """The taken rows (:func:`_ckpt_rows_take`), first in :func:`_bwd_body`.  Returned: the ``dq`` / ``dk`` pair the
    forward allocated (the backward's two outputs; with the in-kernel L2 norm, the bytes its ``dv2`` rides on), or
    ``None`` (the backward allocates its outputs).  The carrier form, where the in-kernel L2 norm's rows do not ride
    on that pair (``hosts``) and the plan allows it (:func:`_rows_carrier_ok`): the backward's intra rows are bound
    on the pair's first bytes (:func:`_bwd_rows`).  Otherwise the copy form: the rows are copied into the arena's
    ``aqk_int`` / ``akk_int`` (no other program writes those bytes before their readers), enqueued on the call's
    stream before any program that writes the pair; the pair is returned all the same (``None`` for a
    ``[2, R, H, 64]`` rows tensor, and under the ``rows_probe`` test knob, which compares the backward's own rows
    with them after its intra launch)."""
    holder, rows = call.scratch.pop("ckpt_rows")
    dt = call.backend.dtypes
    pair = (
        tuple(holder.shape) == (2, call.T_r, call.H, D)
        and holder.dtype == dt["dq"] == dt["dk"]
    )
    if pair and "dv2" not in hosts and _rows_carrier_ok(call.local.plan):
        call.scratch["intra_rows_form"] = "carrier"
        call.scratch["intra_rows_v"] = rows
        return holder
    call.scratch["intra_rows_form"] = "copy"
    _copy_intra_rows(call, rows)
    if call.opts.rows_probe:
        # the probe compares the backward's own rows with them (_bwd_rows)
        call.scratch["intra_rows_v"] = rows
        return None
    return holder if pair else None


def _copy_intra_rows(call: CakeKDADetKcpCall, rows: torch.Tensor) -> None:
    """``rows`` into the arena's ``aqk_int`` / ``akk_int`` views (enqueued on the call's stream)."""
    views = call.scratch["views"]
    views.aqk_int.copy_(rows[0])
    views.akk_int.copy_(rows[1])


def _bind_intra_rows(call: CakeKDADetKcpCall, rows: torch.Tensor) -> None:
    """``aqk_int`` / ``akk_int`` of this call's local stages and KCP programs on ``rows[0]`` / ``rows[1]``: a copy of
    the call's views (the arena's are cached per layout: never rebound) and its value dict."""
    views = types.SimpleNamespace(**vars(call.scratch["views"]))
    views.aqk_int, views.akk_int = rows[0], rows[1]
    call.scratch["views"] = views
    call.local_values["aqk_int"] = rows[0]
    call.local_values["akk_int"] = rows[1]


def _rows_probe(call: CakeKDADetKcpCall, rows: torch.Tensor) -> None:
    """The ``rows_probe`` test knob: this backward's own intra rows (just launched on the arena's views) against the
    kept ``rows``, bit for bit, per internal row (all rows and the real token rows); the result goes to
    ``LAST_CALL["bwd"]["rows_probe"]``.  Synchronizes the host (a debug probe)."""
    views, ctx = call.scratch["views"], call.ctx
    R = int(ctx.R)
    idx = torch.arange(R, device=call.device)
    lens = torch.tensor(list(ctx.chunk_len), dtype=torch.int64, device=call.device)
    real = (idx % BT) < lens[idx // BT]
    result: dict[str, Any] = {
        "R": R,
        "H": call.H,
        "rows": {},
        "equal": True,
        "real_equal": True,
    }
    for i, name in enumerate(("aqk_int", "akk_int")):
        x = getattr(views, name).view(torch.int16).reshape(R, -1)
        y = rows[i].view(torch.int16).reshape(R, -1)
        ne = x != y
        differ = ne.any(1)
        n_all, n_real = int(differ.sum()), int((differ & real).sum())
        result["rows"][name] = {
            "rows_differ": n_all,
            "real_rows_differ": n_real,
            "elements_differ": int(ne.sum()),
        }
        if n_all:
            result["equal"] = False
        if n_real:
            result["real_equal"] = False
    call.scratch["rows_probe"] = result


def _qk_l2norm_stage(call: CakeKDADetKcpCall) -> None:
    """The fallback form's separate normalization before the first q / k reader (nothing in the fused form, whose
    first reader normalizes): ``qk_apply`` from the saved rstd (a saved-mode backward), else ``qk_l2norm``, which
    writes the rstd."""
    if call.l2n_mode == "fallback":
        read = call.scratch.get("l2n_rstd", (0, 0))[0]
        call.stage("qk_apply" if read else "qk_l2norm")


def _backward(call: CakeKDADetKcpCall, saved: CakeKDADetKcpSaved | None) -> None:
    """The design note §6.5.  ``saved=None`` is the replay mode (the forward sequence without ``fwd_o`` recomputes
    ``S_in``, the tapes and the owner's c1 receive tensor); otherwise the tapes come from K1 ``_tape`` over the saved
    ``S_in`` and the owner's adjoint reads ``M_f`` from b1 (``save_transfer=False``) or ``M_f^T`` from the saved c1
    receive tensor.  ``schedule="wait_first"`` (the shipped default since an earlier revision): ``b1 -> wait -> adjoint merge -> b2 -> wait -> dcarry merges ->
    adjoint tape walk(s)`` (the R7 overlap of the document-last walk under b2 is dropped)."""
    tb = call.tb
    wait_first = call.schedule == "wait_first"
    side: _TapeSide | None = None
    join_carry = False
    if saved is None:
        _forward(call, save=False, replay=True)
        m_source = "c1"
        call.scratch["bwd_tape_stream"] = None  # the replay forward wrote the tapes
    else:
        call.bufs["S_in"] = saved.S_in
        m_source = "b1"
        if saved.c1_recv is not None:
            call.bufs["c1_recv"] = saved.c1_recv
            m_source = "c1"
        if saved.M_seg is not None:
            call.bufs["stage_m"] = saved.M_seg
        _bwd_rows(call, saved)
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
        # the tape replay's launches (walk list, initial-state view thunk), issued here on the call's stream or on the
        # side stream (:class:`_TapeSide`, the knob bwd_tape_stream; its views are built here, while the saved plane
        # stands in for the carry pool)
        if (
            tb.fused_walks and tb.P > 1
        ):  # fused walks: the whole tape replay from the carry plane in one launch
            tapes = [
                (
                    tb.walk_live,
                    lambda: call.indexed("carry", tb.h0_index_fused, tb.carry_tiles),
                )
            ]
        else:
            # the trimmed arm passes the table set's own sublists (a fresh comprehension per call would miss the
            # identity-keyed device-table cache on every call: one blocking host-to-device copy per backward)
            trim = bool(call.host_trim)
            p0 = tb.walk_all_p0 if trim else [s for s in tb.walk_all if s % tb.P == 0]
            tapes = [(p0, lambda: call.indexed("S_in", tb.h0_index, tb.s_in_tiles))]
            if tb.P > 1:
                rest = (
                    tb.walk_all_rest
                    if trim
                    else [s for s in tb.walk_all if s % tb.P != 0]
                )
                tapes.append(
                    (
                        rest,
                        lambda: call.indexed("carry", tb.carry_index, tb.carry_tiles),
                    )
                )
        side, mark = _tape_side(call, tapes)
        if side is None:
            for walk, h0 in tapes:
                call.walk("kcp_fwdh_tape", walk, h0=h0(), out=None)
            if mark is not None:
                mark[0].add(mark[1])  # the replay's device tables are resident
        # where the replay reads the workspace's carry pool (no saved plane), the dcarry merges (which write that
        # pool) wait for it
        join_carry = plane is None and tb.P > 1
        call.scratch["bwd_tape_stream"] = "one" if side is None else side.form
        if plane is not None:
            del call.bufs[
                "carry"
            ]  # the saved plane is never written: dcarry_* publish into the workspace pool
    if side is None:
        _backward_window(call, wait_first, m_source, None, False)
        return
    try:
        side.point("pre")
        _backward_window(call, wait_first, m_source, side, join_carry)
    except BaseException:
        side.join(complete=False)
        raise


def _backward_window(
    call: CakeKDADetKcpCall,
    wait_first: bool,
    m_source: str,
    side: _TapeSide | None,
    join_carry: bool,
) -> None:
    """The backward after its tape replay was issued (or forked: ``side``, :class:`_TapeSide`): ``dav``, the adjoint
    offset walk and its exchange, the adjoint carries and tape walks, then the tail stages.  ``side.join()`` runs before
    the dcarry merges (``join_carry``) or else before the adjoint tape walk."""
    tb = call.tb
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
    if side is not None:
        side.point("mid")
    dcarry = (
        call.indexed("carry", tb.dcarry_index, tb.carry_tiles) if tb.P > 1 else None
    )
    if ring:  # SPEC-2: adjoint hops from the document end; b2_recv filled as b2 would (bitwise)
        _ring_backward(
            call
        )  # dcarry_end + the document-end tape walk run under hop 1 (overlap schedule only)
    else:
        wb1 = call.a2a("b1", "b1_send", "b1_recv", *tb.split_elems(f"b1_{form}"))
        if side is not None:
            side.point("post")
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
    if side is not None and join_carry:
        side.join()
    if tb.P > 1:
        if wait_first and call.merge_fuse == "carry":
            call.merge("dcarry_both")  # dcarry_end + dcarry_cross in one launch
        else:
            if wait_first:
                call.merge("dcarry_end")
            call.merge(
                "dcarry_cross"
            )  # with fused_walks also the dS_out copies into the extra carry slots
    if side is not None:
        side.join()  # the adjoint tape walk writes dh over u (both backward layouts)
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
    if call.l2n_mode == "fallback":
        # the fallback form's separate pullback: dq / dk from the GLM intra capture's FP32 dq_hat / dk_hat
        call.stage("qk_pullback")
    if call.gate == "raw":
        call.stage("param_finalize", seg_cap=call.scratch.get("seg_cap"))
    # the normalized q / k die after their last reader (the GLM intra program)
    if call.l2n:
        call.l2n_values = None


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
    ws: CakeKDADetKcpWorkspace | None,
    opts: CakeKDADetKcpOptions,
    scale: float | None,
    tensors: Sequence[Any],
    flags: tuple,
) -> _GraphEntry | None:
    """The cache entry of this call's validity key, or ``None`` when the call is never captured."""
    if (
        not isinstance(ws, CakeKDADetKcpWorkspace)
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


def _graph_cap_bracket(call: CakeKDADetKcpCall) -> tuple[str, int | None]:
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


def _graph_capture(
    kind: str, entry: _GraphEntry, call: CakeKDADetKcpCall, body: Any
) -> Any:
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
    """Drop the cached graphs whose key pins ``pinned``: a local arena that :meth:`CakeKDADetKcpWorkspace.ensure` supersedes, or
    a workspace whose :meth:`CakeKDADetKcpWorkspace.pool` has just allocated a forward-only lazy pool
    (``cake_kcp_plan.FWD_LAZY_POOLS``, part of every key): no later call can match their keys, and their pins (and
    graphs) would keep that storage alive.  The dropped entries also let go of their pins (the entry of the call doing
    the upgrade, created before its ``ensure``, is still referenced by that call): once the arena is freed its id may
    be reused, and a later call that builds an equal key finds no entry and starts a new one.  Returns the number
    dropped.  A pinned plan view of the arena ``pinned`` (:func:`_plan_view`, the same ``arena`` tensor) matches
    too."""
    arena = (
        pinned.arena if isinstance(pinned, cake_backend.CakeKDADetWorkspace) else None
    )
    stale = [
        key
        for key, e in _GRAPHS.items()
        if any(
            p is pinned
            or (
                arena is not None
                and isinstance(p, cake_backend.CakeKDADetWorkspace)
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


def _l2n_key_flags(l2n: bool) -> tuple[Any, ...]:
    """The in-kernel Q / K L2 norm in a graph validity key: ``("qk_l2norm",)`` with the flag on, nothing with it off
    (whose key is unchanged)."""
    return ("qk_l2norm",) if l2n else ()


def _l2n_key_saved(saved: CakeKDADetKcpSaved | None) -> tuple[Any, ...]:
    """A backward's saved ``qk_rstd`` in its graph validity key (nothing for a flag-off saved set)."""
    return (saved.qk_rstd,) if saved is not None and saved.qk_l2norm else ()


# The host module of the non-deterministic package cake_kda_nondet_train_kcp, this package's sibling: its workspace
# type is recognized through sys.modules without importing that package (none of its objects exists before it is
# imported).
_ND_PARENT = __package__.rpartition(".")[0]
_ND_KCP = (
    f"{_ND_PARENT}.cake_kda_nondet_train_kcp.cake_kcp"
    if _ND_PARENT
    else "cake_kda_nondet_train_kcp.cake_kcp"
)


def _refuse_nd_workspace(workspace: Any) -> None:
    """A workspace of the non-deterministic package raises ``ValueError`` naming the mode."""
    nd = sys.modules.get(_ND_KCP)
    if nd is not None and isinstance(
        workspace, getattr(nd, "CakeKDANondetKcpWorkspace", ())
    ):
        raise ValueError(
            f"workspace: a {CakeKDADetKcpWorkspace.__module__}.CakeKDADetKcpWorkspace is required, got "
            f"{type(workspace).__module__}.{type(workspace).__qualname__}; a workspace of the non-deterministic "
            "package cake_kda_nondet_train_kcp serves that package's calls only"
        )


def _refuse_other_mode(saved: Any, workspace: Any) -> None:
    """The backward's mode check, before any graph key, operand or tensor read: a saved set that is not a
    :class:`CakeKDADetKcpSaved` (the non-deterministic package's ``CakeKDANondetKcpSaved`` has the same version and
    members) and a workspace of the non-deterministic package raise ``ValueError`` naming the mode."""
    if saved is not None and not isinstance(saved, CakeKDADetKcpSaved):
        raise ValueError(
            f"saved: a {CakeKDADetKcpSaved.__module__}.CakeKDADetKcpSaved (the deterministic KCP forward) is required, "
            f"got {type(saved).__module__}.{type(saved).__qualname__}; a saved set of the non-deterministic package "
            "cake_kda_nondet_train_kcp is read by that package's backward"
        )
    _refuse_nd_workspace(workspace)


def _call_workspace(
    workspace: Any,
    backend: Any,
    save_mode: str,
    gate: str,
    l2n: bool,
    device: torch.device | str | None,
) -> CakeKDADetKcpWorkspace:
    """The workspace of a call: without one (``None``) a default workspace of the call's modes, the backend's state
    dtype and local stages (with the flag on, its construction checks the mode on ``device``'s architecture);
    otherwise ``workspace``, which must be a :class:`CakeKDADetKcpWorkspace` of the call's gate and in-kernel Q / K L2 norm
    modes and the backend's state dtype (``TypeError`` / ``ValueError``; a workspace of the non-deterministic
    package: ``ValueError`` naming the mode)."""
    if workspace is None:
        workspace = CakeKDADetKcpWorkspace(
            device=device,
            state_dtype=backend.dtypes["state"],
            local=backend.local_stages,
            save_mode=save_mode,
            use_gate_in_kernel=gate == "raw",
            use_qk_l2norm_in_kernel=l2n,
        )
    elif not isinstance(workspace, CakeKDADetKcpWorkspace):
        _refuse_nd_workspace(workspace)
        raise TypeError("workspace must be a CakeKDADetKcpWorkspace")
    elif workspace.gate != gate:
        raise ValueError(
            f"workspace: allocated for gate mode {workspace.gate!r} "
            f"(use_gate_in_kernel={workspace.gate == 'raw'}), the call runs gate mode {gate!r} "
            f"(use_gate_in_kernel={gate == 'raw'}); allocate the CakeKDADetKcpWorkspace with "
            f"use_gate_in_kernel={gate == 'raw'}"
        )
    elif workspace.qk_l2norm != l2n:
        raise ValueError(
            f"workspace: allocated with use_qk_l2norm_in_kernel={workspace.qk_l2norm}, the call runs "
            f"use_qk_l2norm_in_kernel={l2n}; allocate the CakeKDADetKcpWorkspace with use_qk_l2norm_in_kernel={l2n}"
        )
    if workspace.state_dtype != backend.dtypes["state"]:
        raise ValueError(
            f"workspace state dtype {workspace.state_dtype} differs from the backend's {backend.dtypes['state']}"
        )
    return workspace


# The workspaces of the calls made without one (workspace=None), per (context, device, stream, H, direction, modes and
# table signature), as the single-rank package keeps its default workspaces (cake_backend._entry_acquire /
# _entry_release): the first such call builds a default workspace as before; at its end the device storage of every
# pool and of the local arena is freed (untyped_storage().resize_(0), the caching allocator takes the blocks back on
# the call's stream as when a per-call tensor dies) and the workspace -- its plan, layout views, ensure record and
# launch templates -- is kept.  The next call with the key allocates the same bytes again, in the same order, on the
# same stream (the views and templates follow their storage's new address), so between calls only host objects stay
# and a call's device memory is that of a fresh workspace.  What a fresh workspace builds during a call is dropped at
# release and built again by the next call: the identity tile's content, the intra record's scratch table set and the
# existing backend's eye; a lazy pool keeps its tensor and gets its storage again at the next call's first read of it
# (where a fresh workspace allocates it).  A workspace serves one call at a time; another stream or another context
# takes its own.  A call under CUDA-graph capture, with another backend, CPU operands or a context without weak
# references builds a fresh workspace of its own.  The workspaces of a collected context go at the next call.
_DEFAULT_KEYS = 64
_DEFAULT_PER_KEY = 2
# key -> [(workspace, weak context, the storages and their bytes)], least recently released first
_default_pool: collections.OrderedDict[tuple[Any, ...], list[tuple[Any, ...]]] = (
    collections.OrderedDict()
)
_default_lock = threading.Lock()
# the ids of collected contexts (appended by their weak references' callbacks, drained under the lock)
_default_dead: collections.deque[int] = collections.deque()


def _default_drain() -> None:
    """Drop the pooled workspaces of collected contexts (the caller holds :data:`_default_lock`)."""
    while _default_dead:
        dead = _default_dead.popleft()
        for key in [key for key in _default_pool if key[0] == dead]:
            del _default_pool[key]


def _default_workspace(
    ctx: Any,
    q: Any,
    backend: Any,
    save_mode: str,
    gate: str,
    l2n: bool,
    device: torch.device | str | None,
    backward: bool,
    opts: CakeKDADetKcpOptions,
) -> tuple[CakeKDADetKcpWorkspace, tuple[tuple[Any, ...], Any] | None, Any]:
    """``workspace=None``: ``(workspace, (pool key, weak context) or None, (freed storages, eye tile) or None)``.  A pooled
    workspace comes with the storages :func:`_default_release` freed (:func:`_default_unpark` allocates them before
    the call's ``ensure``); otherwise a default workspace is built (:func:`_call_workspace`), pooled at the end of the
    call where the key is not ``None``."""
    key = None
    if (
        isinstance(backend, KernelBackend)
        and isinstance(q, torch.Tensor)
        and q.is_cuda
        and q.dim() >= 2
        and not torch.cuda.is_current_stream_capturing()
    ):
        index = q.device.index
        key = (
            id(ctx),
            index,
            _raw_stream(index),
            int(q.shape[-2]),
            bool(backward),
            save_mode,
            gate,
            bool(l2n),
            backend.dtypes["state"],
            bool(backend.local_stages),
            int(opts.fused_walks),
            str(opts.walk_geometry),
        )
        with _default_lock:
            if _default_dead:
                _default_drain()
            free = _default_pool.get(key)
            entry = free.pop() if free else None
        if entry is not None and entry[1]() is ctx:
            return entry[0], (key, entry[1]), entry[2]
    workspace = _call_workspace(None, backend, save_mode, gate, l2n, device)
    if key is None:
        return workspace, None, None
    dead = key[0]

    def _on_collect(_: Any) -> None:
        _default_dead.append(dead)

    # a context without weak references: a workspace of this call alone
    try:
        ref = weakref.ref(ctx, _on_collect)
    except TypeError:
        return workspace, None, None
    return workspace, (key, ref), None


def _default_unpark(parked: Any) -> None:
    """Allocate the storages of a pooled default workspace (their bytes, in the order a fresh workspace allocates
    them) on the call's stream and rebuild the identity tile's content (the ``torch.eye`` a fresh workspace copies in,
    written in place)."""
    storages, eye = parked
    for storage, nbytes in storages:
        storage.resize_(nbytes)
    if eye is not None:
        torch.eye(D, out=eye)


def _default_release(call: CakeKDADetKcpCall) -> None:
    """The end of a ``workspace=None`` call on a pooled default workspace (nothing otherwise): free its device
    storages and keep the workspace for the next call with its key (at most :data:`_DEFAULT_PER_KEY` per key and
    :data:`_DEFAULT_KEYS` keys, the least recently released going first).  Not reached when the call raises: its
    workspace then goes with it, as a fresh one does."""
    if call.default is None:
        return
    key, ref = call.default
    ws = call.ws
    local = ws.local
    if local is not None and local.__dict__.get(TAIL_COUNTS_ATTR) is not None:
        return  # a lean arena (never a default workspace's): not pooled
    # the lazily allocated pools are parked by name: the next call's first read allocates the storage again
    # (CakeKDADetKcpWorkspace.pool), as a fresh workspace allocates the pool there
    lazy = cake_kcp_plan.LAZY_POOLS + (
        () if call.backward else cake_kcp_plan.FWD_LAZY_POOLS
    )
    freed: list[Any] = []
    for name in lazy:
        t = ws._pools.pop(name, None)
        if t is not None:
            storage = t.untyped_storage()
            ws._parked[name] = (t, storage.nbytes())
            freed.append(storage)
    ws._tables.clear()  # the table_cache="workspace" knob's sets hold their context
    parked: list[tuple[Any, int]] = []
    seen: set[int] = set()
    # each storage once (a SHARED_POOLS pair is one tensor), in allocation order
    for t in ws._pools.values():
        if id(t) not in seen:
            seen.add(id(t))
            storage = t.untyped_storage()
            if storage.nbytes():
                parked.append((storage, storage.nbytes()))
    if local is not None:
        storage = local.arena.untyped_storage()
        parked.append((storage, storage.nbytes()))
        # the next call builds a fresh eye and scratch table set, as a fresh arena does
        local._eye_box[0] = None
        local._epoch[0] = 0
        for arena in (local, *ws._views.values()):
            arena.__dict__.pop(SCRATCH_TABLES_ATTR, None)
    for storage in [*freed, *(s for s, _ in parked)]:
        storage.resize_(0)
    # the identity tile, rebuilt by _default_unpark
    eye = ws._pools.get("eye")
    eye = eye.view(D, D) if eye is not None and eye.numel() else None
    with _default_lock:
        if _default_dead:
            _default_drain()
        if ref() is None:
            return
        free = _default_pool.setdefault(key, [])
        _default_pool.move_to_end(key)
        if len(free) < _DEFAULT_PER_KEY:
            free.append((ws, ref, (parked, eye)))
        while len(_default_pool) > _DEFAULT_KEYS:
            _default_pool.popitem(last=False)


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
    use_qk_l2norm_in_kernel: bool = False,
):
    backend = get_backend()
    opts = options if isinstance(options, CakeKDADetKcpOptions) else _options(options)
    # workspace=None: the pooled default workspace's key and storages (_default_workspace)
    default = parked = None
    ws_default = workspace is None
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
    l2n = cake_kcp_plan.check_qk_l2norm(use_qk_l2norm_in_kernel)
    if l2n and not backend.local_stages:
        raise NotImplementedError(
            f"use_qk_l2norm_in_kernel=True needs a backend with the local kernel stages (backend {backend.name!r})"
        )
    if l2n and opts.spec3_probe != "off":
        # the probe's reference rows rerun the forward's intra program before the backward's first q / k reader
        raise NotImplementedError(
            "spec3_probe is not supported with use_qk_l2norm_in_kernel=True"
        )
    if l2n:
        # the workspace's checks and the capability refusal before the operand checks, on rank-independent inputs
        # only (the context's device model, the workspace's modes): a rank with misaligned or no tokens raises as
        # the others do (the flag-off order is unchanged)
        ctx_dev = getattr(ctx, "device", None)
        if workspace is None:
            workspace, default, parked = _default_workspace(
                ctx, q, backend, save_mode, gate, l2n, ctx_dev, backward, opts
            )
        else:
            workspace = _call_workspace(
                workspace, backend, save_mode, gate, l2n, ctx_dev
            )
        workspace._require_l2n(ctx_dev)
    q, k, v, g, beta, T, H, batched = _operands(
        q, k, v, g, beta, ctx, dtypes=backend.dtypes, gate=gate, qk_l2norm=l2n
    )
    device = q.device
    gp = cake_backend._gate_params(gate, A_log, dt_bias, lower_bound, H, device)
    if dout is not None:
        dout = _squeeze_batch(dout, "dout", 3)
        _check(dout, "dout", (T, H, D), backend.dtypes["dout"], device)
    if not l2n:
        if workspace is None:
            workspace, default, parked = _default_workspace(
                ctx, q, backend, save_mode, gate, l2n, device, backward, opts
            )
        else:
            workspace = _call_workspace(
                workspace, backend, save_mode, gate, l2n, device
            )
    # the in-kernel Q / K L2 norm's capability on this device, before the workspace binds anything: rank-independent
    # inputs only, so a rank without tokens raises as the others do (cake_kcp_plan.require_qk_l2norm; checked above
    # on the context's device, nothing again on the same architecture)
    workspace._require_l2n(device)
    wait_ready = getattr(ctx, "wait_ready", None)
    # a context prepared on another stream: order its uploads on this one (enqueue only)
    if wait_ready is not None:
        wait_ready()
    if parked is not None:  # a pooled default workspace: its storages for this call
        _default_unpark(parked)
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
    call = CakeKDADetKcpCall(
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
    call.l2n, call.l2n_mode = l2n, workspace.l2n
    call.default = default
    call.ws_default = ws_default
    backend.begin_call(call)
    return call, batched


@_with_option_signature
def cake_kda_det_train_kcp_forward(
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
    workspace: CakeKDADetKcpWorkspace | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    **options: Any,
) -> tuple[torch.Tensor, CakeKDADetKcpSaved | None]:
    """The KCP forward of this rank's fragments (the KCP contract §3.2): ``out`` like ``v`` (fresh) and the saved set
    (``None`` without ``save_for_backward`` or in ``boundary_mode="replay"``).  ``ctx`` is this rank's
    ``CakeKDADetKcpContext``; the token operands are its ``T_r`` local tokens in fragment order; ``**options`` are the
    keywords of :class:`CakeKDADetKcpOptions` (+ ``_kcp_test_knobs``), among them ``graph="replay"`` (keyed CUDA-graph
    capture / replay of this call; module docstring and :func:`graph_cache_clear`).

    ``use_gate_in_kernel=True`` (raw gate mode, the single-rank keywords of ``cake_backend.cake_kda_det_train_forward``):
    ``g`` is this rank's raw gate BF16 ``[T_r, H, 128]`` (a strided view under the single-rank rule, never copied)
    with ``A_log`` FP32 ``[H]``, ``dt_bias`` FP32 ``[H * 128]`` or ``[H, 128]`` and ``lower_bound`` (``-5 <=
    lower_bound < 0``, required); the local intra program activates the gate in its prefix pass, every other launch
    and collective is the explicit path's.  The workspace must be of the same mode (:attr:`CakeKDADetKcpWorkspace.gate`)
    and the saved set records the mode and the lower bound.  With ``use_gate_in_kernel=False`` the three parameters
    must be ``None``.

    ``use_qk_l2norm_in_kernel=True`` (keyword-only, issue #7817; module docstring): ``q`` / ``k`` are this rank's raw
    rows (strided views allowed under the rule of :func:`_check_qk_view`, never copied), normalized in the local
    stages per token and 128-channel head with eps ``1e-6`` (``v`` is not); the saved set adds the FP32
    ``qk_rstd`` ``[2, T_r, H]``.  The workspace must be of the same mode (:attr:`CakeKDADetKcpWorkspace.qk_l2norm`).  With
    ``False`` (the default) nothing changes."""
    if boundary_mode not in BOUNDARY_MODES:
        raise ValueError(
            f"boundary_mode must be one of {BOUNDARY_MODES}, got {boundary_mode!r}"
        )
    opts = _options(options)
    gate_args = _gate_args(use_gate_in_kernel, A_log, dt_bias, lower_bound)
    l2n = cake_kcp_plan.check_qk_l2norm(use_qk_l2norm_in_kernel)
    save = bool(save_for_backward) and boundary_mode == "saved"
    entry = (
        _graph_entry(
            "fwd",
            ctx,
            workspace,
            opts,
            scale,
            (q, k, v, g, beta, *_gate_key_tensors(gate_args)),
            (save, save_mode, *_gate_key_flags(gate_args), *_l2n_key_flags(l2n)),
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
        use_qk_l2norm_in_kernel=l2n,
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
            use_qk_l2norm_in_kernel=l2n,
        )
    out, saved = _fwd_body(call, save=save, save_mode=save_mode, ctx=ctx)
    _record("fwd", call)
    if opts.graph != "off":
        _graph_eager("fwd", entry)
    call.backend.end_call(call)
    _default_release(call)
    return _ranked(out, batched), saved


def _fwd_body(
    call: CakeKDADetKcpCall,
    *,
    save: bool,
    save_mode: str,
    ctx: Any,
    bound: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, CakeKDADetKcpSaved | None]:
    """The forward's allocations and launches (eagerly or inside a capture): ``out`` and the saved set.  ``bound``:
    the tensors of a recompute handle's entry (:func:`_forward`); the saved set then holds them.  In-kernel Q / K L2
    norm: the rstd the fused intra program writes, kept by a saving forward (:func:`_qk_l2norm_begin`)."""
    rstd = _qk_l2norm_begin(call, save=save) if call.l2n else None
    _forward(call, save=save, replay=False, bound=bound, alloc_out=True)
    if call.l2n:
        call.l2n_values = None
    saved = None
    if save:
        tb, opts = call.tb, call.opts
        l2n = {"qk_rstd": rstd, "qk_l2norm": True} if call.l2n else {}
        saved = CakeKDADetKcpSaved(
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
            **l2n,
        )
    # INTRA_ROWS: the kept rows go to the backward of this set (they are not a member of it)
    kept = call.scratch.pop("intra_rows_t", None)
    if kept is not None and saved is not None:
        _ckpt_rows_put(saved.S_in, kept)
    return call.out, saved


def _check_saved(saved: CakeKDADetKcpSaved, call: CakeKDADetKcpCall) -> None:
    tb, opts = call.tb, call.opts
    if saved.version != KCP_SAVED_VERSION:
        raise ValueError(f"saved: version {saved.version} is not {KCP_SAVED_VERSION}")
    if saved.ctx is not call.ctx:
        raise ValueError("saved: the saved set belongs to another CakeKDADetKcpContext")
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
    # the in-kernel Q / K L2 norm mode and its FP32 rstd
    if bool(saved.qk_l2norm) != call.l2n:
        raise ValueError(
            f"saved: written with use_qk_l2norm_in_kernel={bool(saved.qk_l2norm)}, the call runs "
            f"use_qk_l2norm_in_kernel={call.l2n}"
        )
    if saved.qk_l2norm:
        if saved.qk_rstd is None:
            raise ValueError(
                "saved: use_qk_l2norm_in_kernel=True needs the forward's qk_rstd"
            )
        _check(
            saved.qk_rstd,
            "saved.qk_rstd",
            (2, call.T_r, call.H),
            call.backend.dtypes["qk_rstd"],
            call.device,
        )
    elif saved.qk_rstd is not None:
        raise ValueError("saved: a qk_rstd without use_qk_l2norm_in_kernel=True")
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


@_with_option_signature
def cake_kda_det_train_kcp_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    dout: torch.Tensor,
    *,
    saved: CakeKDADetKcpSaved | None = None,
    scale: float | None = None,
    save_mode: str = "light",
    workspace: CakeKDADetKcpWorkspace | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    **options: Any,
) -> tuple[torch.Tensor, ...]:
    """The KCP backward (the KCP contract §3.2): ``(dq, dk, dv, dbeta, dg)``, fresh.  ``saved=None`` replays the forward.

    Raw gate mode (``use_gate_in_kernel=True`` with the forward's ``A_log`` / ``dt_bias`` / ``lower_bound``):
    ``(dq, dk, dv, dbeta, dg, dA_log, ddt_bias)``, ``dg`` the BF16 gradient of the raw gate (contiguous, ranked like
    ``g``), ``dA_log`` FP32 ``[H]`` and ``ddt_bias`` FP32 shaped like ``dt_bias``.  The two parameter gradients are
    this rank's partial sums: the gradients through this rank's own tokens, reduced on the rank in an order fixed by
    its local chunks alone (bitwise repeatable per rank, the same under every schedule and exchange).  The full
    gradient is their sum over the ranks of ``ctx.group``; a training framework sums them as it sums the gradients
    of any parameter replicated across the context-parallel ranks, or calls :func:`cake_kda_det_train_kcp_reduce_gate_param_grads` for
    a sum that is bitwise reproducible (an NCCL all-reduce does not fix its summation order).  No gate epilogue runs
    and no full-size FP32 gate or gate gradient exists on the route.  A saved set of the other mode, or with another
    ``lower_bound`` (its FP32 value), is refused with ``ValueError``, as is a workspace of the other mode.

    ``use_qk_l2norm_in_kernel=True`` (the forward's; issue #7817): the same seven / five tensors, ``dq`` / ``dk`` the
    gradients of the raw ``q`` / ``k`` (contiguous, ranked like ``q``), the FP32 pullback fused into their final BF16
    store.  A saved-mode backward reads the saved ``qk_rstd``; a replay-mode backward recomputes it.  A saved set or
    a workspace of the other mode is refused with ``ValueError``.

    ``saved`` must come from this package's forward and ``workspace`` must not be one of the non-deterministic package
    ``cake_kda_nondet_train_kcp``: either raises ``ValueError`` naming the mode before anything is read."""
    _refuse_other_mode(saved, workspace)
    opts = _options(options)
    gate_args = _gate_args(use_gate_in_kernel, A_log, dt_bias, lower_bound)
    l2n = cake_kcp_plan.check_qk_l2norm(use_qk_l2norm_in_kernel)
    entry = None
    if opts.graph != "off":
        kept = (
            ()
            if saved is None
            else (
                saved.S_in,
                saved.c1_recv,
                saved.M_seg,
                saved.carry,
                *_l2n_key_saved(saved),
            )
        )  # the validity key sees every saved tensor (the carry plane of saved_carry=1 and the rstd included)
        entry = _graph_entry(
            "bwd",
            ctx,
            workspace,
            opts,
            scale,
            (q, k, v, g, beta, dout, *kept, *_gate_key_tensors(gate_args)),
            (
                save_mode,
                saved is not None,
                *_gate_key_flags(gate_args),
                *_l2n_key_flags(l2n),
            ),
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
        use_qk_l2norm_in_kernel=l2n,
    )
    if saved is not None:
        _check_saved(saved, call)
        if _CKPT_ROWS:  # INTRA_ROWS: the rows a forward of this set kept
            _ckpt_rows_take(call, saved)
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
    ):  # SPEC-3's host compare stays eager; a rank without tokens runs no local stage (CakeKDADetKcpCall.stage), so it has
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
            use_qk_l2norm_in_kernel=l2n,
        )
        if saved is not None:
            _check_saved(saved, call)
    grads = _bwd_body(call, saved)
    _record("bwd", call)
    if opts.graph != "off":
        _graph_eager("bwd", entry)
    call.backend.end_call(call)
    _default_release(call)
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


def _bwd_body(
    call: CakeKDADetKcpCall, saved: CakeKDADetKcpSaved | None
) -> tuple[torch.Tensor, ...]:
    """The backward's allocations and launches (eagerly or inside a capture): ``(dq, dk, dv, dbeta, dg)``; in raw
    gate mode ``(dq, dk, dv, dbeta, dg_raw, dA_log, ddt_bias)`` with ``dg_raw`` the BF16 gradient of the raw gate
    (contiguous ``[T_r, H, 128]``), ``dA_log`` FP32 ``[H]`` and ``ddt_bias`` FP32 ``[H * 128]``: this rank's partial
    sums (the parameter gradients of its own tokens).  The raw GLM intra program writes the per-chunk ``(hi, lo)``
    partials ``dA_part`` / ``db_part`` (FP32 ``[NC_pad, H, 128, 2]`` each, per call, outside the arena; the real
    chunks only) and ``param_finalize`` reduces them in the fixed order of the local chunks, with its per-call
    scratch ``part2`` / ``ticket`` and ``seg_cap`` from ``cake_backend._raw_finalize_scratch``; a rank without tokens
    launches nothing and returns zero parameter gradients.  In-kernel Q / K L2 norm: the saved set's rstd or, in
    replay mode, a fresh one, and the normalized q / k (:func:`_qk_l2norm_begin`), in the arena where the replay
    layout's ``v_new`` / ``dv2`` ride on the outputs' bytes (:func:`_l2n_row_hosts`; dq / dk are then the two halves
    of one tensor); the outputs are the same.  A flag-on saved-mode backward of the fused form allocates the outputs
    and scratch its first local stage binds none of (:func:`_l2n_late_grads`) once that stage is launched
    (:func:`_late_grads`): the host work before the first launch is not hidden behind queued device work."""
    dt = call.backend.dtypes
    T, H = call.T_r, call.H
    hosts = _l2n_row_hosts(call) if call.l2n else {}
    late = (
        _l2n_late_grads(call, hosts) if call.l2n and saved is not None else frozenset()
    )
    # (name, shape, dtype) of the outputs allocated after the first local stage's launch
    pending: list[tuple[str, tuple[int, ...], torch.dtype]] = []
    # the intra rows of the forward (INTRA_ROWS) before any output: on the dq / dk pair it allocated (the carrier),
    # or copied into the arena (the pair is then this backward's all the same)
    pair = _ckpt_rows_bind(call, hosts) if "ckpt_rows" in call.scratch else None

    def grad(name: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        if name in late:
            pending.append((name, shape, dtype))
        else:
            call.grads[name] = _alloc(shape, dtype, call.device)

    # the moved arena buffer -> the output tensor whose bytes carry it
    moved: dict[str, torch.Tensor] = {}
    if "dv2" in hosts:
        moved["dv2"] = (
            pair if pair is not None else _alloc((2, T, H, D), dt["dq"], call.device)
        )
        dq, dk = moved["dv2"][0], moved["dv2"][1]
    elif pair is not None:
        dq, dk = pair[0], pair[1]
    else:
        dq = _alloc((T, H, D), dt["dq"], call.device)
        dk = _alloc((T, H, D), dt["dk"], call.device)
    call.grads = {"dq": dq, "dk": dk}
    grad("dv", (T, H, D), dt["dv"])
    if call.gate == "raw":
        names: tuple[str, ...] = _RAW_GRADS
        grad("dg_raw", (T, H, D), dt["dg_raw"])
        grad("dbeta", (T, H), dt["dbeta"])
        if call.local is not None and T > 0:
            nc_pad = int(call.local.plan.NC_pad)
            for name in ("dA_part", "db_part"):
                grad(name, (nc_pad, H, D, 2), dt["dA_log"])
            # param_finalize's per-call scratch: the second-stage partials and the per-head ticket
            if "part2" in late and "ticket" in late:
                call.scratch["late_finalize"] = nc_pad
            else:
                _finalize_scratch(call, nc_pad)
            grad("dA_log", (H,), dt["dA_log"])
            grad("ddt_bias", (H * D,), dt["ddt_bias"])
        else:  # no local stage runs on this rank: its tokens contribute nothing
            call.grads["dA_log"] = _zeros((H,), dt["dA_log"], call.device)
            call.grads["ddt_bias"] = _zeros((H * D,), dt["ddt_bias"], call.device)
    else:
        names = ("dq", "dk", "dv", "dbeta", "dg")
        grad("dg", (T, H, D), dt["dg"])
        grad("dbeta", (T, H), dt["dbeta"])
    if pending or "late_finalize" in call.scratch:
        call.scratch["late_grads"] = pending
    if "v_new" in hosts:
        moved["v_new"] = call.grads[hosts["v_new"][0]]
    if call.l2n:
        _qk_l2norm_begin(call, None if saved is None else saved.qk_rstd, hosts=moved)
        try:
            _backward(call, saved)
        finally:
            # the record: made by _late_grads unless the backward raised before it
            _l2n_record(call)
            _l2n_restore_views(call)
    else:
        _backward(call, saved)
    _late_grads(call)  # nothing left once the first local stage ran
    return tuple(call.grads[n] for n in names)


def _finalize_scratch(call: CakeKDADetKcpCall, nc_pad: int) -> None:
    """``param_finalize``'s per-call scratch (``cake_backend._raw_finalize_scratch``) into the call."""
    fin = cake_backend._raw_finalize_scratch(nc_pad, call.H, call.device)
    call.grads["part2"] = fin["part2"]
    call.grads["ticket"] = fin["ticket"]
    call.scratch["seg_cap"] = int(fin["seg_cap"])


def _late_grads(call: CakeKDADetKcpCall) -> None:
    """Allocate the outputs and scratch :func:`_bwd_body` deferred (:func:`_l2n_late_grads`); nothing when none is
    pending.  :func:`_bwd_rows` calls it right after the backward's first local stage."""
    _l2n_record(call)
    pending = call.scratch.pop("late_grads", None)
    if pending is None:
        return
    for name, shape, dtype in pending:
        call.grads[name] = _alloc(shape, dtype, call.device)
    nc_pad = call.scratch.pop("late_finalize", None)
    if nc_pad is not None:
        _finalize_scratch(call, nc_pad)


# The outputs and scratch a flag-on backward may allocate after its first local stage (:func:`_l2n_late_grads`)
_L2N_LATE_GRADS = (
    "dv",
    "dbeta",
    "dg",
    "dg_raw",
    "dA_part",
    "db_part",
    "part2",
    "ticket",
    "dA_log",
    "ddt_bias",
)
# The logical names the first local stage of a fused flag-on backward binds, per plan: {id(plan): (a weak reference
# to it, the registry it read, the names or None, {id(hosts) or None: (hosts or None, the deferred names)})}
_L2N_FIRST_BOUND: dict[int, tuple[Any, Any, frozenset[str] | None, dict[int, Any]]] = {}


def _l2n_late_grads(
    call: CakeKDADetKcpCall, hosts: dict[str, tuple[str, ...]]
) -> frozenset[str]:
    """The outputs and scratch of :data:`_L2N_LATE_GRADS` a flag-on saved-mode backward allocates once its first
    local stage is launched: those the record of that stage (the plan's first q / k reader of the fused form,
    :func:`cake_kcp_plan.qk_l2norm_intra_first`) binds none of, and that carry no moved arena buffer
    (:func:`_l2n_row_hosts`: the first reader writes ``qn`` / ``kn`` into those buffers' arena bytes and may bind the
    buffers).  Nothing deferred for the fallback form, a rank without local stages or tokens, another backend, a
    first stage without a plan record, or the SPEC-3 probe (its forward rows run before the prep)."""
    if (
        call.l2n_mode != "fused"
        or call.local is None
        or call.T_r == 0
        or not isinstance(call.backend, KernelBackend)
        or call.opts.spec3_probe != "off"
    ):
        return frozenset()
    p = call.local.plan
    kept = _L2N_FIRST_BOUND.get(id(p))
    if kept is None or kept[0]() is not p or kept[1] is not cake_jit.MODULES:
        for stale in [k for k, v in _L2N_FIRST_BOUND.items() if v[0]() is None]:
            del _L2N_FIRST_BOUND[stale]
        stage = "fwd_intra" if cake_kcp_plan.qk_l2norm_intra_first(p) else "bwd_prep"
        bound: frozenset[str] | None = None
        if stage not in KCP_LOCAL_RECORDS and p.dispatched(stage, LOCAL_PATH_BWD):
            record = p.record(stage, LOCAL_PATH_BWD)
            bound = frozenset(cake_jit.MODULES[record]["logical"].values())
        kept = (weakref.ref(p), cake_jit.MODULES, bound, {})
        _L2N_FIRST_BOUND[id(p)] = kept
    bound = kept[2]
    if bound is None:
        return frozenset()
    # per hosts dict (_l2n_row_hosts keeps each non-empty one per plan, gate mode and output dtypes; an empty one is
    # fresh per call): the same set every call
    key = id(hosts) if hosts else None
    hit = kept[3].get(key)
    if hit is not None and hit[0] is (hosts if hosts else None):
        return hit[1]
    carriers = {out for outs in hosts.values() for out in outs}
    late = frozenset(
        name for name in _L2N_LATE_GRADS if name not in bound and name not in carriers
    )
    if len(kept[3]) >= L2N_MOVED_MAX:
        kept[3].clear()
    kept[3][key] = (hosts if hosts else None, late)
    return late


# The in-kernel Q / K L2 norm's backward keeps its token-layout normalized rows in the local arena where it can (issue
# #7817): ``qn`` on the bytes of the replay layout's ``v_new``, ``kn`` on those of ``dv2``, each of which then rides on
# the bytes of per-call outputs that only the stages of :data:`_L2N_LATE_STAGES` write, after the last ``v_new`` /
# ``dv2`` read (the fused gradient, step 6): ``dv2`` on dq / dk (the two halves of one BF16 ``[2, T_r, H, 128]``
# tensor), ``v_new`` on dg (FP32 ``[T_r, H, 128]``; raw gate mode the BF16 ``dg_raw``).  Without them the backward
# would hold a per-call ``[T_r, H, 128]`` BF16 pair beyond the flag-off backward's peak (:func:`_l2n_row_hosts`).
_L2N_LATE_STAGES = frozenset(
    {"glm_intra", "gate_epilogue", "qk_pullback", "param_finalize"}
)
# (the moved arena buffer, the normalized rows on its bytes)
_L2N_ROW_HOSTS = (("v_new", "qn"), ("dv2", "kn"))
# The decision per plan (``cake_backend.plan`` returns one object per problem): {id(plan): (a weak reference to it,
# the registry it read, {(gate, output dtypes): hosts})}, so a backward's host prologue does not walk the records again
_L2N_ROW_HOSTS_KEPT: dict[
    int, tuple[Any, Any, dict[tuple[Any, ...], dict[str, tuple[str, ...]]]]
] = {}


def _l2n_row_hosts(call: CakeKDADetKcpCall) -> dict[str, tuple[str, ...]]:
    """The arena buffers a flag-on backward moves onto per-call outputs, each with the outputs whose bytes carry it:
    ``{"v_new": ("dg",), "dv2": ("dq", "dk")}`` (raw gate mode ``("dg_raw",)``); the freed bytes hold ``qn`` /
    ``kn`` (:data:`_L2N_ROW_HOSTS`, :func:`_qk_l2norm_begin`).  A buffer moves where the replay layout holds it alone
    in its slot, its outputs' BF16 rows cover its ``R`` rows (dq / dk and dg ``2 T_r``, dg_raw ``T_r``), no record of
    the stages of :data:`_L2N_LATE_STAGES` binds it and no record of the backward's other stages binds those
    outputs.  The KCP's own programs bind no output (K1 / K2 write the tapes, ``kcp_dav`` reads ``v_new`` at step
    4).  Nothing moves on a forward, on a rank without local stages or tokens, or on the lean layout.  The decision is
    kept per plan, gate mode and output dtypes while the registry is the same object (:data:`_L2N_ROW_HOSTS_KEPT`);
    callers do not modify the returned dict."""
    if (
        not call.backward
        or call.local is None
        or call.T_r == 0
        or _lean_passes(call.local)
    ):
        return {}
    p, dt = call.local.plan, call.backend.dtypes
    dg = "dg_raw" if call.gate == "raw" else "dg"
    kept = _L2N_ROW_HOSTS_KEPT.get(id(p))
    if kept is None or kept[0]() is not p or kept[1] is not cake_jit.MODULES:
        for stale in [k for k, v in _L2N_ROW_HOSTS_KEPT.items() if v[0]() is None]:
            del _L2N_ROW_HOSTS_KEPT[stale]
        kept = (weakref.ref(p), cake_jit.MODULES, {})
        _L2N_ROW_HOSTS_KEPT[id(p)] = kept
    key = (call.gate, call.T_r, dt["dq"], dt["dk"], dt[dg])
    hosts = kept[2].get(key)
    if hosts is None:
        hosts = kept[2][key] = _l2n_row_hosts_of(p, call.gate, call.T_r, dt)
    return hosts


def _bwd_stage_bindings(p: Any) -> tuple[frozenset[str], frozenset[str]]:
    """The logical names the backward's local records of plan ``p`` bind (the replay path; the KCP's own programs
    bind no output and are left out): ``(early, late)``, those of the stages before the GLM intra program and those
    of :data:`_L2N_LATE_STAGES`."""
    path = LOCAL_PATH_BWD
    late: set[str] = set()
    early: set[str] = set()
    for stage in (*L2N_FALLBACK_STAGES, *LOCAL_STAGES_BWD):
        if stage in KCP_LOCAL_RECORDS:
            continue
        if stage in WY_DIRECT_RECORDS and not p.dispatched(stage, path):
            record = WY_DIRECT_RECORDS[stage]
        elif p.dispatched(stage, path):
            record = p.record(stage, path)
        else:
            continue
        (late if stage in _L2N_LATE_STAGES else early).update(
            cake_jit.MODULES[record]["logical"].values()
        )
    return frozenset(early), frozenset(late)


def _l2n_row_hosts_of(
    p: Any, gate: str, T: int, dt: dict[str, torch.dtype]
) -> dict[str, tuple[str, ...]]:
    """The uncached decision of :func:`_l2n_row_hosts` for plan ``p`` of a rank with ``T`` tokens."""
    early, late = _bwd_stage_bindings(p)
    regions = p.layout(LOCAL_LAYOUT_BWD)
    slot = {r.name: r.slot for r in regions}
    dg = "dg_raw" if gate == "raw" else "dg"
    pair = 2 * T if dt["dq"] == dt["dk"] == torch.bfloat16 else 0
    cases = (("v_new", (dg,), T * dt[dg].itemsize // 2), ("dv2", ("dq", "dk"), pair))
    hosts: dict[str, tuple[str, ...]] = {}
    for buf, outs, rows in cases:
        alone = buf in slot and sum(r.slot == slot[buf] for r in regions) == 1
        if alone and rows >= p.R and buf not in late and early.isdisjoint(outs):
            hosts[buf] = outs
    return hosts


def _qk_l2norm_begin(
    call: CakeKDADetKcpCall,
    rstd: torch.Tensor | None = None,
    *,
    save: bool = False,
    hosts: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor | None:
    """The in-kernel Q / K L2 norm part of a call with ``use_qk_l2norm_in_kernel=True`` (issue #7817), first in its
    body; returns the rstd a saving forward keeps (``None`` otherwise).  The FP32 rstd ``qk_rstd`` ``[2, T_r, H]``
    and the first q / k reader's ``(read_rstd, write_rstd)``: a saved-mode backward reads ``rstd``, the saved set's
    ``(1, 0)``; a replay-mode backward, a saving forward and a fallback-form forward write a fresh one ``(0, 1)`` (a
    saving forward keeps it; empty on a rank without tokens); a fused-form forward without a saved set writes none
    ``(0, 0)`` and binds a one-element tensor it never reads.  The normalized q / k: the fused intra program
    normalizes into its CTA-private internal rows ``qn_int`` / ``kn_int``, the arena's ``u`` / ``w`` slot views of
    the forward and replay layouts (both carry them: the capability check refuses a layout without them, so there
    is no stand-in); on a forward it writes no token-layout ``qn`` / ``kn`` (``write_qkn = 0``: no later forward
    program reads them, the slot views stand in; the fallback form's separate normalization writes them into the
    first ``T_r`` rows of the same slots), on a backward the token-layout rows ``[T_r, H, 128]`` BF16
    (``write_qkn = 1``: written by the first q / k reader, read by the later ones up to the GLM intra program) are
    ``qn`` on the arena bytes of ``v_new`` and ``kn`` on those of ``dv2`` where ``hosts`` (:func:`_l2n_row_hosts`,
    from :func:`_bwd_body`) maps that buffer to the per-call output whose bytes then carry it (the call's views and
    values bind the moved buffer there), else per call outside the workspace.  The values: one dict for every local
    stage (``q`` / ``k`` the token-layout normalized rows, ``q_raw`` / ``k_raw`` the raw views and their strides,
    ``qk_rstd``, ``num_tokens``, the dq / dk row strides, the three scalars).  A rank without tokens binds nothing and
    launches no stage of the mode.  The capability refusal comes earlier, in :func:`_prepare`, on every rank alike."""
    T, H, dt = call.T_r, call.H, call.backend.dtypes
    keep = save and not call.backward
    if call.local is None or T == 0:
        return _alloc((2, T, H), dt["qk_rstd"], call.device) if keep else None
    if rstd is not None:  # a saved-mode backward
        rw = (1, 0)
        if rstd.is_cuda:
            # the design note's switch-only record (the caching allocator ignores a record on the allocating stream),
            # made by _l2n_record (from _late_grads) right after the first local launch: host work off the pre-launch path (the set
            # holds the rstd past the call, so the record still precedes its release)
            call.scratch["l2n_record"] = rstd
    elif call.backward or keep or call.l2n_mode == "fallback":
        rw = (0, 1)
        rstd = _alloc((2, T, H), dt["qk_rstd"], call.device)
    else:  # a fused-form forward without a saved set
        rw = (0, 0)
        rstd = _alloc((1,), dt["qk_rstd"], call.device)
    views = call.scratch["views"]
    qn_int, kn_int = views.qn_int, views.kn_int
    write_qkn = int(call.backward)
    if call.backward:
        qn, kn = _l2n_bwd_rows(call, hosts or {})
    elif call.l2n_mode == "fallback":
        qn, kn = qn_int[:T], kn_int[:T]
    else:
        qn, kn = qn_int, kn_int
    q, k = call.ops["q"], call.ops["k"]
    call.scratch["l2n_rstd"] = rw
    call.l2n_values = {
        "q": qn,
        "k": kn,
        "qn": qn,
        "kn": kn,
        "qn_int": qn_int,
        "kn_int": kn_int,
        "write_qkn": write_qkn,
        "q_raw": q,
        "k_raw": k,
        "qk_rstd": rstd,
        "num_tokens": T,
        "q_raw_row_stride": q.stride(0),
        "q_raw_head_stride": D,
        "k_raw_row_stride": k.stride(0),
        "k_raw_head_stride": D,
        "dq_row_stride": H * D,
        "dk_row_stride": H * D,
        "read_rstd": rw[0],
        "write_rstd": rw[1],
    }
    return rstd if keep else None


def _l2n_bwd_rows(
    call: CakeKDADetKcpCall, hosts: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """A flag-on backward's token-layout ``qn`` / ``kn`` ``[T_r, H, 128]`` BF16: for each arena buffer of
    :data:`_L2N_ROW_HOSTS` in ``hosts``, the buffer moves onto the leading bytes of its host output (a copy of the
    call's views and its values bind it there) and its normalized rows take the first ``T_r`` rows of its arena
    bytes; a per-call tensor for the others (one ``[2, T_r, H, 128]`` pair when neither moves)."""
    T, H, dt = call.T_r, call.H, call.backend.dtypes
    rows: dict[str, torch.Tensor] = {}
    if hosts:
        views = call.scratch["views"]
        # the arena's views are cached per layout: never rebind them.  Their copy with the moved buffers is kept per
        # view on the call's arena (L2N_MOVED_ATTR, built once), its moved buffers set here and put back to the
        # arena's by _l2n_restore_views at the call's end, with the first T_r rows of each moved buffer per T_r
        local = call.local
        cache = local.__dict__.get(L2N_MOVED_ATTR)
        if cache is None:
            cache = {}
            setattr(local, L2N_MOVED_ATTR, cache)
        hit = cache.get(id(views))
        if hit is None or hit[0] is not views:
            if len(cache) >= L2N_MOVED_MAX:
                cache.clear()
            hit = (views, types.SimpleNamespace(**vars(views)), {})
            cache[id(views)] = hit
        moved, heads = hit[1], hit[2]
        for buf, out in _L2N_ROW_HOSTS:
            host = hosts.get(buf)
            if host is None:
                continue
            arena = getattr(views, buf)
            head = heads.get((buf, T))
            if head is None:
                if len(heads) >= L2N_MOVED_MAX:
                    heads.clear()
                head = heads[(buf, T)] = (
                    arena[:T],
                    tuple(arena.shape),
                    _contiguous_strides(arena.shape),
                )
            rows[out] = head[0]
            # the buffer's contiguous shape on the host's leading bytes (the host is a fresh contiguous output)
            if host.dtype != arena.dtype:
                host = host.view(arena.dtype)
            there = host.as_strided(head[1], head[2])
            setattr(moved, buf, there)
            call.local_values[buf] = there
        call.scratch["views"] = moved
        call.scratch["l2n_moved"] = hit
    if not rows:
        qk = _alloc((2, T, H, D), dt["q"], call.device)
        return qk[0], qk[1]
    for out in ("qn", "kn"):
        if out not in rows:
            rows[out] = _alloc((T, H, D), dt["q" if out == "qn" else "k"], call.device)
    return rows["qn"], rows["kn"]


def _l2n_record(call: CakeKDADetKcpCall) -> None:
    """The saved rstd's stream record :func:`_qk_l2norm_begin` deferred past the backward's first launch (once)."""
    rstd = call.scratch.pop("l2n_record", None)
    if rstd is not None and not torch.cuda.is_current_stream_capturing():
        rstd.record_stream(torch.cuda.current_stream(rstd.device))


def _l2n_restore_views(call: CakeKDADetKcpCall) -> None:
    """The end of a flag-on backward that moved arena buffers (:func:`_l2n_bwd_rows`): the kept moved-views copy gets
    the arena's buffers back, so it holds no output of the call."""
    hit = call.scratch.pop("l2n_moved", None)
    if hit is not None:
        views, moved = hit[0], hit[1]
        for buf, _ in _L2N_ROW_HOSTS:
            setattr(moved, buf, getattr(views, buf))


@functools.lru_cache(maxsize=64)
def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """The row-major strides of ``shape`` (elements)."""
    strides, step = [], 1
    for n in reversed(shape):
        strides.append(step)
        step *= max(int(n), 1)
    return tuple(reversed(strides))


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


class CakeKDADetKcpRecomputeHandle:
    """The boundary tensors of one checkpointed call, carried from its original forward to torch's recompute of that
    forward (``cake_kda_det_kcp(..., recompute_handle=handle)``), so the recompute runs without cross-rank
    communication.

    Create one handle per layer and micro-step, outside the outermost checkpoint region, and pass it as an argument
    of the checkpointed function, so the recompute sees the same object::

        def layer(q, k, v, g, beta, handle):
            return cake_kda_det_kcp(q, k, v, g, beta, ctx, recompute_handle=handle)

        out = torch.utils.checkpoint.checkpoint(layer, q, k, v, g, beta, CakeKDADetKcpRecomputeHandle(), use_reentrant=False)

    ``entries`` maps an operation key to that operation's entry: ``"kda"`` is this package's (the forward's incoming
    states ``S_in`` -- and with ``checkpoint_keep="transfer"`` the owner's c1 receive tensor --, the very tensors
    the saved set holds, plus the call's context and fingerprint); other packages that accept a handle with an
    ``entries`` dict use their own keys (the causal convolution's history rows: ``"conv"``), so one handle may serve
    the KDA call and one convolution call of the same layer.  A handle holds one entry per key.  A handle created
    inside a checkpoint region is a new, empty one in that region's recompute, which then exchanges again (correct,
    not communication-free).  The entries live as long as the handle; :meth:`nbytes` is the retained size
    (``cake_kda_det_train_kcp_saved_bytes(ctx, H=..., checkpoint_keep=..., with_conv=False)`` for the KDA entry).

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


def _kcp_phase_in_backward() -> bool:
    """:func:`_kcp_in_backward`, ``False`` where torch lacks a usable binding (the intra-rows rule then keeps none)."""
    try:
        return _kcp_in_backward()
    except RuntimeError:
        return False


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


def _handle_refusals(
    opts: CakeKDADetKcpOptions, boundary_mode: str, handle: Any
) -> None:
    """The refusals of a call with a handle, shared by :func:`cake_kda_det_kcp` and the autograd Function: graph replay
    (keyword or test knob), a forward inside a CUDA-graph capture, ``debug_sequence``, ``spec3_probe``,
    ``boundary_mode="replay"`` and a handle without an ``entries`` dict."""
    if not isinstance(getattr(handle, "entries", None), dict):
        raise TypeError(
            f"recompute_handle must carry an 'entries' dict (CakeKDADetKcpRecomputeHandle), got {type(handle).__name__}"
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
    opts: CakeKDADetKcpOptions, checkpoint_keep: str | None, explicit_save_transfer: Any
) -> CakeKDADetKcpOptions:
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
            "KDA call of one micro-step; create a new CakeKDADetKcpRecomputeHandle per layer and micro-step"
        )
    return entry


def _handle_fingerprint(
    call: CakeKDADetKcpCall, *, save_mode: str, keep: str, **modes: Any
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
    entry: dict[str, Any], call: CakeKDADetKcpCall, fingerprint: tuple, keep: str
) -> dict[str, torch.Tensor]:
    """The stored tensors of ``entry``, checked against this recompute: the same context, the same fingerprint and
    the shapes / dtype / device of the call's pools."""
    if entry.get("ctx") is not call.ctx:
        raise ValueError(
            "recompute_handle: the 'kda' entry belongs to another CakeKDADetKcpContext"
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


def _gate_modes(call: CakeKDADetKcpCall) -> dict[str, Any]:
    """The gate keywords of a call in a recompute handle's fingerprint: the mode and the raw mode's FP32 lower bound
    (a recompute binds the stored states only under the same gate; ``A_log`` / ``dt_bias`` are operands)."""
    gp = call.gate_params
    return {
        "use_gate_in_kernel": call.gate == "raw",
        "lower_bound": None if gp is None else gp.lower_bound,
    }


def _l2n_modes(call: CakeKDADetKcpCall) -> dict[str, Any]:
    """The in-kernel Q / K L2 norm in a recompute handle's fingerprint: its form (``l2n="fused"`` /
    ``"fallback"``) with the flag on, nothing with it off (a flag-off fingerprint is unchanged).  The entry stays
    ``S_in`` (and the c1 receive tensor): the recompute writes its rstd afresh, bitwise the original forward's."""
    return {"l2n": call.l2n_mode} if call.l2n else {}


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
    l2n: bool = False,
) -> tuple[torch.Tensor, CakeKDADetKcpSaved]:
    """The forward of a call with a handle (eager, saved mode): the original forward exchanges and stores the
    handle's entry; the recompute binds it (:func:`_forward`).  ``gate_args``: the entry point's gate keywords;
    ``l2n``: its ``use_qk_l2norm_in_kernel``."""
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
        use_qk_l2norm_in_kernel=l2n,
    )
    fingerprint = _handle_fingerprint(
        call, save_mode=save_mode, keep=keep, **_gate_modes(call), **_l2n_modes(call)
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
    _default_release(call)
    return _ranked(out, batched), saved


# ----------------------------------------------------------------------------------------------------------------
# autograd


class CakeKDADetKcpFunction(torch.autograd.Function):
    """:func:`cake_kda_det_kcp` as an autograd node: the forward keeps the inputs and the saved set (saved mode) or
    the inputs only (replay mode); the backward runs :func:`cake_kda_det_train_kcp_backward`.  Under
    ``torch.utils.checkpoint(use_reentrant=False)`` the forward is re-run in the backward: the same program order
    on every rank, so the collectives match (R7).  ``handle``: a recompute handle (:class:`CakeKDADetKcpRecomputeHandle`):
    the original forward stores its received states in it, the recompute binds them and exchanges nothing; the
    saved set has the same members either way.  The tier follows the options' ``save_transfer`` (``False``:
    ``S_in``; ``True``: also the c1 receive tensor; :func:`cake_kda_det_kcp` sets it from ``checkpoint_keep``).  A
    handle is refused with graph replay, inside a CUDA-graph capture, with ``debug_sequence``, ``spec3_probe`` and
    ``boundary_mode="replay"``.  Raw gate mode (``use_gate_in_kernel``): the node saves the raw gate (the caller's
    view) and ``A_log`` / ``dt_bias`` beside the KCP saved set, which records the mode and ``lower_bound``, and
    returns the gradients of the raw gate, ``A_log`` and ``dt_bias`` (the parameter gradients: this rank's partial
    sums, :func:`cake_kda_det_train_kcp_backward`).  In-kernel Q / K L2 norm (``use_qk_l2norm_in_kernel``): the node keeps the
    caller's raw ``q`` / ``k`` (strided views stay views) and the saved set with its ``qk_rstd``, and returns the
    gradients of the raw ``q`` / ``k``."""

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
        use_qk_l2norm_in_kernel=False,
    ):
        gate_args = _gate_args(use_gate_in_kernel, A_log, dt_bias, lower_bound)
        if handle is None:
            out, saved = cake_kda_det_train_kcp_forward(
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
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
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
                l2n=use_qk_l2norm_in_kernel,
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
            use_qk_l2norm_in_kernel,
        )
        return out

    @staticmethod
    def backward(fctx, dout):
        q, k, v, g, beta, *tensors = fctx.saved_tensors
        ctx, scale, save_mode, workspace, options, meta, (raw, lower_bound), l2n = (
            fctx.kcp
        )
        A_log = dt_bias = None
        if raw:
            A_log, dt_bias, *tensors = tensors
        saved = (
            None
            if meta is None
            else CakeKDADetKcpSaved.from_tensors(tensors, meta[0], meta[1], ctx)
        )
        if not dout.is_contiguous():
            dout = dout.contiguous()
        grads = cake_kda_det_train_kcp_backward(
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
            use_qk_l2norm_in_kernel=l2n,
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
            None,  # use_qk_l2norm_in_kernel
        )


@_with_option_signature
def cake_kda_det_kcp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    scale: float | None = None,
    save_mode: str = "light",
    boundary_mode: str = "saved",
    workspace: CakeKDADetKcpWorkspace | None = None,
    *,
    recompute_handle: Any = None,
    checkpoint_keep: str | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    **options: Any,
) -> torch.Tensor:
    """The differentiable KCP forward (the KCP contract §3.2 signature): ``out`` like ``v``; gradients for ``q, k, v, g, beta``.

    ``use_gate_in_kernel=True`` (raw gate mode; keyword-only with ``A_log`` / ``dt_bias`` / ``lower_bound``, as
    :func:`cake_kda_det_train_kcp_forward`): ``g`` is the raw gate and the gradients are those of ``q, k, v``, the raw gate
    (BF16), ``beta``, ``A_log`` and ``dt_bias`` (FP32).  The parameter gradients of a rank are its partial sums
    (the gradients through its own tokens); sum them over ``ctx.group`` as for any parameter replicated across the
    context-parallel ranks, or with :func:`cake_kda_det_train_kcp_reduce_gate_param_grads` for a bitwise reproducible sum.  A recompute
    handle's entry records the gate mode and ``lower_bound``: a recompute in another mode never binds it.

    ``use_qk_l2norm_in_kernel=True`` (keyword-only, issue #7817, as :func:`cake_kda_det_train_kcp_forward`): ``q`` / ``k`` are
    the raw rows (strided views allowed, never copied) and the gradients are those of the raw ``q`` / ``k``; the
    node keeps the raw rows and the saved set with its FP32 ``qk_rstd``.  A recompute handle's fingerprint records
    the mode (its entry does not change).  With ``False`` (the default) nothing changes.

    ``recompute_handle`` (keyword-only; ``None`` = unchanged): a :class:`CakeKDADetKcpRecomputeHandle` (any object with an
    ``entries`` dict) makes torch's checkpoint recompute of this call run without the forward state exchange -- the
    original forward stores its incoming states under ``"kda"``, the recompute inside the backward binds them
    (bitwise the exchanging recompute); the backward still exchanges.  One handle per layer and micro-step, created
    outside the outermost checkpoint region; a forward outside the backward that finds an entry raises.
    ``checkpoint_keep`` (with a handle only): ``"states"`` (default) keeps ``S_in`` and runs the call with
    ``save_transfer=False`` (bitwise the default form; the backward re-sends the maps in b1); ``"transfer"`` also
    keeps the owner's c1 receive tensor (``save_transfer=True``; not with ``exchange="ring"``).  An explicit
    ``save_transfer`` must agree with the tier.  Refused with ``graph="replay"`` (keyword or knob), inside a
    CUDA-graph capture, with ``debug_sequence``, ``spec3_probe`` and ``boundary_mode="replay"``.  Retains
    ``recompute_handle.nbytes()`` (``cake_kda_det_train_kcp_saved_bytes(ctx, H=..., checkpoint_keep=..., with_conv=False)``) while
    the handle lives.  The exchange is a collective: every rank of ``ctx.group`` passes a handle for the same call or
    none, at the same checkpoint nesting level; a mismatch hangs or corrupts data (no check across ranks)."""
    opts = _options(options)  # validate on the caller's frame
    if boundary_mode not in BOUNDARY_MODES:
        raise ValueError(
            f"boundary_mode must be one of {BOUNDARY_MODES}, got {boundary_mode!r}"
        )
    l2n = cake_kcp_plan.check_qk_l2norm(use_qk_l2norm_in_kernel)
    if recompute_handle is None:
        if checkpoint_keep is not None:
            raise ValueError(
                "checkpoint_keep selects what a recompute handle keeps: it needs recompute_handle"
            )
        if opts.host_trim:  # the forward and the backward reuse the validated options instead of re-validating the dict
            options = {"_kcp_options": opts}
        return CakeKDADetKcpFunction.apply(
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
            l2n,
        )
    opts = _handle_options(opts, checkpoint_keep, options.get("save_transfer"))
    _handle_refusals(opts, boundary_mode, recompute_handle)
    return CakeKDADetKcpFunction.apply(
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
        l2n,
    )


def cake_kda_det_train_kcp_reduce_gate_param_grads(
    ctx: Any, dA_log: torch.Tensor, ddt_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The raw gate mode's full parameter gradients from every rank's partial sums, bitwise reproducible.

    ``dA_log`` (FP32 ``[H]``) and ``ddt_bias`` (FP32 ``[H * 128]`` or ``[H, 128]``) are this rank's partial sums
    from :func:`cake_kda_det_train_kcp_backward` / :func:`cake_kda_det_kcp` with ``use_gate_in_kernel=True`` (the gradients through
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
            f"cake_kda_det_train_kcp_reduce_gate_param_grads: world_size {W} needs an initialized torch.distributed process "
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
cake_kda_det_train_kcp_saved_bytes = cake_kcp_plan.cake_kda_det_train_kcp_saved_bytes
cake_kda_det_train_kcp_comm_bytes = cake_kcp_plan.cake_kda_det_train_kcp_comm_bytes
cake_kda_det_train_kcp_workspace_bytes = (
    cake_kcp_plan.cake_kda_det_train_kcp_workspace_bytes
)

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
    "CakeKDADetKcpRecomputeHandle",
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
    "CakeKDADetKcpFunction",
    "Indexed",
    "CakeKDADetKcpCall",
    "CakeKDADetKcpOptions",
    "CakeKDADetKcpSaved",
    "CakeKDADetKcpWorkspace",
    "KernelBackend",
    "cake_kda_det_kcp",
    "cake_kda_det_train_kcp_backward",
    "cake_kda_det_train_kcp_forward",
    "get_backend",
    "cake_kda_det_train_kcp_comm_bytes",
    "cake_kda_det_train_kcp_saved_bytes",
    "cake_kda_det_train_kcp_workspace_bytes",
    "cake_kda_det_train_kcp_reduce_gate_param_grads",
    "set_backend",
    "tile_index",
]
