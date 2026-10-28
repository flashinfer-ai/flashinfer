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

Host orchestration of the context-parallel (KCP) packed GDN training route
(SM100 / SM103 / SM107): the transient :class:`KcpWorkspace`, the forward and
backward launch sequences with their collectives, the saved set
:class:`KcpSaved`, the autograd Function and the entry points
``chunk_gdn_kcp`` / ``chunk_gdn_kcp_fwd`` / ``chunk_gdn_kcp_bwd``.

Provenance: adapted from ``flashinfer/experimental/cake_kda_train/cake_kcp.py``
at FlashInfer ``13d7fcf4041635f5b7f573bb0dbab23570ae1034`` (the KDA KCP host
module).  Verbatim parts: the compact pools and chains (``Indexed``,
``Chains``), the exchange tables (``_Tables``), the per-call state
(``KcpCall``), the ring exchange, the launch-sequence structure and the
CUDA-graph replay.  Changed for GDN: two head counts (``H_qk`` Q/K heads,
``HV = r * H_qk`` value heads; every state, summary and exchange term is per
VALUE head), the scalar gate ``g`` FP32 ``[T, HV]`` and FP32 ``beta``, strided
packed-QKV operand views, the nine-program ABI (``fwd_prep``, the walks, the
affine merge, ``fwd_o``, ``bwd_prep``, ``bwd_dqkg``, ``gate_epilogue``), the
tape set and the saved set of the design (``save_mode`` ``"light"`` /
``"full"``), the output order ``(dq, dk, dv, dbeta, dg)``, keyword-only options
(:mod:`.cake_options`) with no environment variable anywhere, and a dispatch
table with a ``backend="fake"`` switch for the torch emulation of the ABI.

Every rank holds its own token fragments (``q, k`` ``[T_r, H_qk, 128]``,
``v`` ``[T_r, HV, 128]`` BF16, ``g``, ``beta`` ``[T_r, HV]`` FP32; local
``cu_seqlens`` with one sequence per fragment) and only recurrent-state
summaries cross the ranks.  Forward: ``fwd_prep`` (#1) writes the chunk-local
tapes; the crossing-out fragments' affine summaries ``(B_f, M_f^T)`` (the
offset walk from zero, #2 ``program=offset``, and the TF32-hi/lo transfer walk,
#3) go to the merge owner of each head slice (``c1``, one ``all_to_all_single``
on the caller's group); the owner composes every document chain in document
order with the exact FP32 affine merge (#4) and returns each continuing
fragment's initial state ``S_in`` (``c2``); the tape walk (#2 ``program=tape``)
then runs from ``S_in`` and ``fwd_o`` (#5) writes the output.  Backward:
``bwd_prep`` (#6); the adjoint offsets ``d_f`` (#7 ``program=offset``, the
reverse walk from zero) and, unless the forward's transfer was kept, ``M_f``
again go to the owners (``b1``); the owner walks each chain backwards
(``X <- M_f^T X + d_f``, #4 transposed) and returns every crossing-out
fragment's ``dS_out`` (``b2``); the adjoint tape walk (#7 ``program=tape``)
runs from it; ``bwd_dqkg`` (#8, the FP32 group reduction of the ``r`` value
heads' ``dq`` / ``dk`` fused) and ``gate_epilogue`` (#9) finish the gradients.
One compute stream; the collectives are asynchronous on the caller's process
group and waited at their first consumer; the issue order is the same on
every rank, so a checkpoint replay re-runs the same program order.

Saved set (``boundary_mode="saved"``): ``S_in`` (fresh), the owner's c1
receive tensor (``save_transfer=True``), ``M_seg`` (``P > 1``) and, with
``save_mode="full"``, the forward tapes (``h``, ``v_new``, ``w``, ``u``,
``Aqk``, ``Akk``, the gate rows); ``save_mode="light"`` (default) keeps the
boundary tensors only and the backward re-runs #1 and the #2 tape walk from
the saved ``S_in`` -- no exchange, no transfer walk, no merge, no #5.
``boundary_mode="replay"`` keeps nothing and the backward re-runs the whole
forward exchange.  Every saved tensor travels through
``ctx.save_for_backward`` only, so ``torch.utils.checkpoint(use_reentrant=False)``
retains nothing after a checkpointed forward (#7039).  The three modes are
bitwise identical by construction (the same programs on the same bits).
With ``reuse_recompute_tapes`` the light backward of such a checkpoint binds
the forward tapes of torch's recompute (the forward re-run inside
``.backward()``) instead of replaying #1 and the tape walk itself (the kept
tapes, below ``_graph_task_id`` / ``_KeptTapes``).  With
``recompute_output="auto"`` (default) that recompute leaves its #5 out when
torch's checkpoint frame proves its output dead (below
``_recompute_output_dead``).

Tapes (padded rows ``R = 64 * NC_pad``; token ``t`` of chunk ``c`` sits in row
``64 c + (t - chunk_bos[c])``): ``w, u, v_new, do_g, dv_intra`` BF16
``[R, HV, 128]`` (``dv_new`` is stored in place of ``dv_intra``); ``aqk, akk`` BF16 ``[R, HV, 64]``; ``g_cum, rowscale,
qscale, dg_part`` FP32 ``[R, HV]``; ``gamma, gamma_dot`` FP32 ``[NC_pad, HV]``;
``h_tape, dh_tape`` BF16 ``[NC_pad, HV, 128, 128]``.  Head mapping of a value
head ``hv`` to its Q/K head: ``hv // r``.

This module reads tensor contents nowhere (no ``.item()``, no ``.tolist()``
of a device tensor, no device synchronisation), dispatches from metadata and
the SM count only, allocates the outputs and the saved set fresh per call and
keeps only transient buffers in the workspace.  The metadata (partition,
context, head ownership, chain tables, split sizes, segment plan, byte
queries) is :mod:`.cake_kcp_plan` / :mod:`.cake_bytes`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist

from . import cake_kcp_plan
from . import cake_options as _opt
from .cake_bytes import TICKET_SITES, TICKET_SLOT_BYTES
from .cake_options import KcpOptions, make_options, validate_heads, validate_ring

D = int(cake_kcp_plan.HEAD_DIM)  # head dimension (128)
BT = int(cake_kcp_plan.CHUNK)  # tokens per chunk (64)
SCALE = D**-0.5
TILE = D * D  # elements of one FP32 state tile [128, 128]
TICKET_SLOT_WORDS = (
    TICKET_SLOT_BYTES // 4
)  # u32 words of one persistent launch site's ticket slot (128 B)
SUMMARY_TILE = 2 * TILE  # one c1 summary tile of one head: [2][128][128] = (B_f, M_f^T)
BOUNDARY_MODES = _opt.BOUNDARY_MODES
SAVE_MODES = _opt.SAVE_MODES
EXCHANGES = _opt.EXCHANGES
OFFSET_CLASSES = _opt.OFFSET_CLASSES
RING_MAX_HOPS = _opt.RING_MAX_HOPS
GRAPH_MODES = _opt.GRAPH_MODES
GRAPH_DEFAULT = _opt.GRAPH_DEFAULT
KCP_SAVED_VERSION = 1  # saved set {S_in [, c1 receive tensor, M_seg] [, forward tapes]}
COLLECTIVES = ("c1", "c2", "b1", "b2")
COLS_RULE_CHAINS = 24  # by_chains: <= 24 concurrent 128-column state chains -> c8, otherwise c16; map launches stay c16

# Program names of the dispatch table and the device program each one launches.
PROGRAMS = (
    "fwd_prep",  # #1 chunk-local prep: gcum, gamma, A^-1 (Akk), w, u, Aqk, rowscale, qscale
    "kcp_fwdh_offset",  # #2 program=offset: walk from zero, final state -> compact tiles (B_f / b_seg); no tapes
    "kcp_fwdh_tape",  # #2 program=tape: walk from an indexed initial state, h / v_new tapes
    "kcp_transfer_walk",  # #3: FP32-class transfer M_f^T (TF32 hi/lo; init_mode = 1 / use_u = 1: the FP32-class offset)
    "kcp_affine_merge",  # #4: exact FP32 affine chains (state / map / transposed modes), reused from the KDA package
    "fwd_o",  # #5: o = scale e^G q h_c + (Aqk o mask) v_new
    "bwd_prep",  # #6: do_g, dv_intra
    "kcp_dhu_offset",  # #7 program=offset: reverse walk from zero, final adjoint -> compact tiles (d_f / d_seg)
    "kcp_dhu_tape",  # #7 program=tape: reverse walk from an indexed final adjoint, dh / dv_new tapes
    "bwd_dqkg",  # #8: dq, dk (grouped FP32 reduction), dv, dbeta, dg partials
    "gate_epilogue",  # #9: dg from the partials (reverse cumsum + chunk term), dbeta pass-through
)
LOCAL_PROGRAMS = ("fwd_prep", "fwd_o", "bwd_prep", "bwd_dqkg", "gate_epilogue")
WALK_PROGRAMS = (
    "kcp_fwdh_offset",
    "kcp_fwdh_tape",
    "kcp_transfer_walk",
    "kcp_dhu_offset",
    "kcp_dhu_tape",
)
TART_MODULE_OF = {
    "fwd_prep": "gdn_chunk_train_fwd_prep_v1",
    "kcp_fwdh_offset": "gdn_chunk_kcp_fwdh_v1",
    "kcp_fwdh_tape": "gdn_chunk_kcp_fwdh_v1",
    "kcp_transfer_walk": "gdn_chunk_kcp_transfer_walk_v1",
    "kcp_affine_merge": "kda_chunk_kcp_affine_merge_fp32_v1",
    "fwd_o": "gdn_chunk_train_fwd_o_v1",
    "bwd_prep": "gdn_chunk_train_bwd_prep_v1",
    "kcp_dhu_offset": "gdn_chunk_kcp_dhu_v1",
    "kcp_dhu_tape": "gdn_chunk_kcp_dhu_v1",
    "bwd_dqkg": "gdn_chunk_train_bwd_dqkg_v1",
    "gate_epilogue": "gdn_chunk_train_gate_epilogue_v1",
}
# Registry record names (``cake_jit.MODULES`` / ``TEST_MODULES``): ``<program>[_<variant>]`` -- the merge records
# ``kcp_affine_merge_c8`` / ``_c16`` by ``ncols`` / ``cols_per_cta``; the transfer walk's test-only programs
# (``build.test.json``, loaded through ``cake_backend.load_test``, never selected by default):
# ``precision_mode=1`` the ``_three_mma`` twin and ``precision_mode=2`` the ``_plain_tf32`` negative control (names
# of the registry).  No BF16x3 module exists in this package: ``transfer_walk="bf16x3"`` stays in
# the option vocabulary and the validator refuses it.  ``precision_mode`` / ``transfer_walk`` are STATIC build knobs
# (record selection), never kernel arguments.
RECORD_OF = {name: name for name in PROGRAMS}
TEST_RECORDS = {
    ("kcp_transfer_walk", 1): "kcp_transfer_walk_three_mma",
    ("kcp_transfer_walk", 2): "kcp_transfer_walk_plain_tf32",
}


def test_record(name: str, precision_mode: int, transfer_walk: str) -> str | None:
    """The ``build.test.json`` record of program ``name`` under the test-only knobs, ``None`` for the shipped record
    (``precision_mode=0``, ``transfer_walk="tf32_hilo"``).  Only the transfer walk has test twins."""
    if name != "kcp_transfer_walk":
        return None
    if transfer_walk != _opt.TRANSFER_WALK_DEFAULT:
        raise ValueError(
            f"transfer_walk={transfer_walk!r} is not built in this package"
        )
    if int(precision_mode) != 0:
        return TEST_RECORDS[(name, int(precision_mode))]
    return None


# Logical argument vocabulary of every program: the registry's logical names (``cake_jit.LOGICAL_TENSORS`` /
# ``LOGICAL_SCALARS``; TMA descriptors ``*_tma`` are the plain operand names, the walks' segment table is
# ``seg_chunk_start``, #5's output is ``o``, the backward's incoming gradient is ``do``).  ``program`` /
# ``precision_mode`` / ``transfer_walk`` are STATIC build knobs (record selection), never values.  The kernels'
# positional order is the registry record's ``arg_plan``; the host passes a dict.  :data:`HOST_ONLY_ARGS` are the
# host-level extras of the ``group_reduce="host"`` fallback, ignored by a record that lacks them.
_LOCAL_TABLE_ARGS = ("chunk_bos", "chunk_len", "seq_chunk_start", "num_chunks")
_WALK_COMMON_ARGS = (
    "num_heads",
    "num_qk_heads",
    "seg_chunk_start",
    "num_seqs",
    "walk_seg",
    "chunk_bos",
    "chunk_len",
)
_POOL = ("out", "out_index", "out_hg", "out_n")
PROGRAM_ARGS: dict[str, tuple[str, ...]] = {
    "fwd_prep": (
        "q",
        "k",
        "v",
        "g",
        "beta",
        "num_qk_heads",
        "num_heads",
        "scale",
        *_LOCAL_TABLE_ARGS,
        "w",
        "u",
        "aqk",
        "akk",
        "g_cum",
        "rowscale",
        "qscale",
        "gamma",
    ),
    "kcp_fwdh_offset": (
        "k",
        "w",
        "u",
        "rowscale",
        "gamma",
        *_WALK_COMMON_ARGS,
        "h0",
        "h0_index",
        "h0_hg",
        "h0_n",
        *_POOL,
        "tile_stride",
        "h_tape",
        "v_new",
    ),
    "kcp_transfer_walk": (
        "k",
        "w",
        "u",
        "rowscale",
        "gamma",
        *_WALK_COMMON_ARGS,
        *_POOL,
        "tile_stride",
        "init_mode",
        "use_u",
        "write_transposed",
    ),
    "kcp_affine_merge": (
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
    "fwd_o": (
        "q",
        "h_tape",
        "v_new",
        "aqk",
        "qscale",
        "num_heads",
        "num_qk_heads",
        *_LOCAL_TABLE_ARGS,
        "num_seqs",  # persistent #5 reads seq_chunk_start[num_seqs] as the real chunk count (K3 ABI)
        "o",
        "ticket",  # the persistent programs' self-resetting ticket word (the workspace's slot of this site)
    ),
    # #6 read set (g_cum + scale instead of qscale; do_tstride; num_seqs)
    "bwd_prep": (
        "do",
        "aqk",
        "qscale",
        "g_cum",
        "scale",
        "num_heads",
        *_LOCAL_TABLE_ARGS,
        "num_seqs",
        "do_tstride",
        "do_g",
        "dv_intra",
        "ticket",
    ),
    "kcp_dhu_offset": (
        "q",
        "k",
        "w",
        "do_g",
        "dv_intra",
        "rowscale",
        "gamma",
        *_WALK_COMMON_ARGS,
        "dht",
        "dht_index",
        "dht_hg",
        "dht_n",
        *_POOL,
        "tile_stride",
        "dh_tape",
        "dv_new",
    ),
    # #8 read set (the registry record's logical names)
    "bwd_dqkg": (
        "q",
        "k",
        "v",
        "do",
        "h_tape",
        "dh_tape",
        "v_new",
        "dv_new",
        "w",
        "u",
        "aqk",
        "akk",
        "do_g",
        "beta",
        "g_cum",
        "rowscale",
        "qscale",
        "gamma",
        "scale",
        "num_qk_heads",
        "num_heads",
        *_LOCAL_TABLE_ARGS,
        "dq",
        "dk",
        "dv",
        "dbeta",
        "dg_part",
        "gamma_dot",
        "group_reduce",
        "dq_part",
        "dk_part",
        "num_seqs",
        "dqk_tstride",
        "dv_tstride",
        "ticket",
    ),
    "gate_epilogue": (
        "dg_part",
        "gamma_dot",
        "chunk_bos",
        "chunk_len",
        "num_chunks",
        "num_heads",
        "dg",
    ),
}
PROGRAM_ARGS["kcp_fwdh_tape"] = PROGRAM_ARGS["kcp_fwdh_offset"]
PROGRAM_ARGS["kcp_dhu_tape"] = PROGRAM_ARGS["kcp_dhu_offset"]
# Every host argument is a registry logical name since the registry follows the frozen kernel ABIs (K4 ABI: #8 takes
# ``group_reduce`` / ``dq_part`` / ``dk_part`` / the token strides; #6 ``do_tstride``; the persistent programs
# ``num_seqs``).  ``PROVISIONAL_ARGS`` names the values a registry may not bind yet: the persistent programs'
# ``ticket`` (the dynamic-ticket forms of #5 / #6 / #8) is passed to every registry; a record without that kernel
# parameter leaves the value unbound (the launch glue binds the record's plan only), one with it receives the slot.
HOST_ONLY_ARGS: tuple[str, ...] = ()
PROVISIONAL_ARGS: tuple[str, ...] = ("ticket",)
# #2 and #7 ship two forms per program: ``vh = 64`` (two CTAs per (segment, head), grid x2: CTA b walks item
# b // 2 over the value columns (b % 2) * 64 .. + 64) and ``vh = 128`` (one CTA per (segment, head)); both write
# bitwise-identical tapes / pools, so the host picks per launch on plan integers.  The vh64 records are
# ``<record>_vh64`` in the registry (selected by ``ctas_per_head = 2``; the grid rule carries it).
FWDH_PROGRAMS = ("kcp_fwdh_offset", "kcp_fwdh_tape")
DHU_PROGRAMS = ("kcp_dhu_offset", "kcp_dhu_tape")
VH64_SUFFIX = "_vh64"


def fwdh_vh(
    n_walk: int, num_heads: int, sm_count: int, forced: int | None = None
) -> int:
    """The #2 form of one launch: ``64`` when ``n_walk * HV <= sm_count`` (the KCP one-wave regime), else ``128``;
    ``forced`` (test knob ``fwdh_vh``) overrides the rule."""
    if forced is not None:
        return int(forced)
    return 64 if int(n_walk) * int(num_heads) <= int(sm_count) else 128


def dhu_vh(
    n_walk: int, num_heads: int, sm_count: int, forced: int | None = None
) -> int:
    """The #7 form of one launch: ``64`` when the two-CTA grid ``n_walk * HV * 2`` fits two waves at two CTAs per
    SM (``<= 4 * sm_count``; the half-width form measured never slower up to there), else ``128``; ``forced`` (test
    knob ``dhu_vh``) overrides the rule."""
    if forced is not None:
        return int(forced)
    return 64 if int(n_walk) * int(num_heads) * 2 <= 4 * int(sm_count) else 128


# The tape set (padded rows R = 64 * NC_pad; the frozen tape set of the design).
FWD_TAPES = (
    "w",
    "u",
    "aqk",
    "akk",
    "g_cum",
    "rowscale",
    "qscale",
    "gamma",
    "h_tape",
    "v_new",
)
BWD_TAPES = ("do_g", "dv_intra", "dh_tape", "dv_new", "dg_part", "gamma_dot")
# Tapes that share one buffer (host-side aliasing, no kernel change): #7 stores ``dv_new`` in place of ``dv_intra``.
# The walk visits every token row once and reads the row's ``dv_intra`` before it stores the row's ``dv_new``; the
# offset program only reads; #8 reads ``dv_new`` only.  The two names resolve to one ``[R, HV, 128]`` BF16 buffer
# (identical base pointer and layout, never a partial overlap); the pad rows come out as exact zeros as before.
TAPE_ALIASES: dict[str, str] = {"dv_new": "dv_intra"}
# Forward tapes #5 does not read (its read set: q, h_tape, v_new, aqk, qscale): dead after the walks.  An eager forward
# drops their arena buffers before ``out`` is allocated, so the output's bytes replace theirs at the forward's peak
# (``graph="replay"`` keeps them: a captured graph bakes the arena addresses, and the capture key tracks them).
FWD_DEAD_BEFORE_O: tuple[str, ...] = ("w", "u", "akk", "g_cum", "rowscale", "gamma")
# Tapes #1 writes for the backward only: no forward program after ``fwd_prep`` reads them (the walks read k / w / u /
# rowscale / gamma, #5 reads q / h_tape / v_new / aqk / qscale), so an eager forward that does not save them drops
# them right after #1 -- they would otherwise sit at the forward's peak (the tape walk) for nothing.  The light
# backward's replay of #1 keeps them: #6 and #8 read them there.
FWD_DEAD_AFTER_PREP: tuple[str, ...] = ("akk", "g_cum")
# Pools dead at the end of an eager forward (the merges and walks are done; the saved tensors are fresh, never pools)
# and the pools dead in the backward once the adjoint exchange is merged, before the adjoint tape walk (``carry`` and
# ``b2_recv`` feed that walk and go right before #8).  Dropped pools are sized again by the next call's ``ensure``.
FWD_DEAD_POOLS: tuple[str, ...] = (
    "c1_send",
    "c1_recv",
    "c2_send",
    "stage_b",
    "stage_m",
    "carry",
    "ring_out",
    "ring_in",
)
BWD_DEAD_BEFORE_TAPES: tuple[str, ...] = (
    "c1_send",
    "c1_recv",
    "c2_send",
    "b1_send",
    "b1_recv",
    "b2_send",
    "stage_b",
    "stage_m",
    "ring_out",
    "ring_in",
)
BWD_DEAD_BEFORE_DQKG: tuple[str, ...] = ("carry", "b2_recv")
# The ring-exchange pools: sized only under ``exchange="ring"`` (the a2a forms never touch them; R2-L1 1a).
RING_POOLS: tuple[str, ...] = ("ring_out", "ring_in")
# The pools a call replaces with fresh (forward) or saved (backward) tensors under ``save_transfer``: the c1 receive
# tensor and the ``M_seg^T`` staging plane are saved-set tensors, so the workspace pools of those names are dead for
# that call and are not sized (``_prepare`` -> ``KcpWorkspace.ensure(skip=...)``).
SAVE_TRANSFER_FRESH_POOLS: tuple[str, ...] = ("c1_recv", "stage_m")
# The saved-set tensor bound under each per-call buffer name (``KcpCall.drop_bound`` forgets both at the last reader).
SAVED_TENSOR_OF_BUFFER: dict[str, str] = {
    "S_in": "S_in",
    "c1_recv": "c1_recv",
    "stage_m": "M_seg",
}
# Pools dead once the adjoint exchange is merged (a2a form): the forward exchange pools (the backward never exchanges
# through them) and b1 (sent and merged).  They go before the document-end adjoint walk that the overlap schedule
# issues under b2 -- the backward's peak -- while ``b2_send`` / ``b2_recv`` are in flight.  The staging planes stay:
# the dcarry merges read them (``m`` = stage_m, ``c`` = stage_b) up to the adjoint tape walk.
# The #7 offset walk's never-dereferenced tape pointer: the forward tape of the same shape and dtype, live by then
# (saved or replayed), so ``dh_tape`` is first allocated by the first tape walk -- after the adjoint merge.
DHU_OFFSET_TAPE_STANDIN = "h_tape"
BWD_DEAD_AFTER_ADJ_MERGE: tuple[str, ...] = (
    "c1_send",
    "c1_recv",
    "c2_send",
    "b1_send",
    "b1_recv",
    "ring_out",
    "ring_in",
)
HOST_REDUCE_TAPES = (
    "dq_part",
    "dk_part",
)  # group_reduce="host": FP32 per-value-head partials of dq / dk


# Forward tapes kept in the saved set per ``save_mode`` (the saved-set table): ``"light"`` keeps no tape -- its backward
# re-runs #1 (which writes ``gamma`` / ``g_cum`` / ``akk`` again; #1 has no input for them) and the #2 tape walk from the
# saved ``S_in`` / ``c1_recv`` / ``M_seg``; ``"full"`` keeps every forward tape and skips #1 and the walk.  A saved copy
# of the #1 outputs in ``"light"`` would be dead at the backward's peak (measured: 6,339 B/token at HV = 48 live twice),
# so ``LIGHT_TAPES`` is empty; a non-empty tuple here saves those tapes for a backward that reads them instead.
LIGHT_TAPES: tuple[str, ...] = ()


def saved_tapes(save_mode: str) -> tuple[str, ...]:
    """The tape names a saved set of ``save_mode`` carries (in :data:`FWD_TAPES` order)."""
    return FWD_TAPES if save_mode == "full" else LIGHT_TAPES


# Forward tapes an eager forward leaves bound in the workspace arena (#5 reads them; the others go at
# FWD_DEAD_AFTER_PREP / FWD_DEAD_BEFORE_O): the arena state a forward leaves, and the state a kept backward restores.
FWD_TAPES_ARENA_AFTER_FWD: tuple[str, ...] = tuple(
    n for n in FWD_TAPES if n not in FWD_DEAD_BEFORE_O and n not in FWD_DEAD_AFTER_PREP
)


# ----------------------------------------------------------------------------------------------------------------
# Checkpoint-aware light backward (keyword ``reuse_recompute_tapes``: ``True`` = the shipped default, ``False`` under
# graph="replay"; ``"auto"`` = kept tapes only when the node closes the checkpointed region, see _fwd_body)
#
# ``torch.utils.checkpoint(use_reentrant=False)`` re-runs the checkpointed forward inside ``.backward()``: in the
# unpack hook of the first saved tensor a backward node reads, synchronously, under the autograd engine's graph task
# of that backward (``torch/utils/checkpoint.py``: ``_checkpoint_hook.unpack_hook`` keys the recompute on
# ``torch._C._current_graph_task_id()`` and runs ``frame.recompute_fn`` under ``_recomputation_hook``).  That
# recompute is an ordinary light forward: it writes the ten forward tapes and, without this option, drops them again,
# and the light backward then re-runs #1 and the #2 tape walk (``_replay_tapes``) to rebuild the same bits.  With
# ``reuse_recompute_tapes`` a forward that runs inside a graph task (``_graph_task_id() != -1``; a plain forward, the
# direct autograd forward and the checkpointed forward itself run outside any task and see ``-1``) keeps its ten
# tapes as fresh tensors unbound from the arena -- plus the ``stage_m`` plane when the saved set carries no ``M_seg``
# (``save_transfer=False`` at ``P > 1``) -- in the registry below, keyed by the graph task id and the identities
# (data pointers) of the call's operands and saved ``S_in``; the backward node of the same task takes the entry and
# runs the ``save_mode="full"`` form of the backward on those tapes, so nothing is launched before #6.  The kept
# tapes are produced with the replay's own tape-walk form (``_replay_tape_walk``), so they are the replay's launch
# list by construction.  Correctness never depends on the registry: a backward that finds no entry runs the replay.
# Lifetime: ONE live entry.  A hand-off forward evicts the registry before its first launch (the evicted set is freed
# before the new one is materialised) and puts its own set after #5; the backward of the same task takes it and
# releases it after #8 (the last reader).  In a checkpoint segment holding several GDN layers torch recomputes every
# layer before the first backward, so only the LAST layer's backward finds its set and the earlier layers run the
# replay (same bits): the peak never exceeds the switch-off path's.  An entry no backward node takes (the node was
# excluded from the backward) goes at the task's end through the engine's final callback; a backward that raised
# leaves its entry (the engine runs no final callback after an exception) until the next hand-off forward evicts it.
# The detection is the private binding
# ``torch._C._current_graph_task_id`` (checked once: present, returns an int); without it the option is a no-op
# (today's path).  No tensor content is read anywhere: the key holds the task id and data pointers only.

_GRAPH_TASK_ID_BINDING = getattr(torch._C, "_current_graph_task_id", None)
_GRAPH_TASK_ID_USABLE: bool | None = None


def _graph_task_id() -> int:
    """The autograd engine's graph task id on this thread (``-1`` outside a backward pass); ``-1`` as well when the
    binding is absent or does not return an int (checked once, then remembered)."""
    global _GRAPH_TASK_ID_USABLE
    if _GRAPH_TASK_ID_USABLE is False or _GRAPH_TASK_ID_BINDING is None:
        return -1
    try:
        gid = _GRAPH_TASK_ID_BINDING()
    except Exception:  # noqa: BLE001 - a binding that refuses the call disables the detection
        _GRAPH_TASK_ID_USABLE = False
        return -1
    if isinstance(gid, bool) or not isinstance(gid, int):
        _GRAPH_TASK_ID_USABLE = False
        return -1
    _GRAPH_TASK_ID_USABLE = True
    return int(gid)


# ----------------------------------------------------------------------------------------------------------------
# The recompute's dead output (keyword ``recompute_output``; shipped default ``"auto"``)
#
# ``torch.utils.checkpoint(use_reentrant=False)`` re-runs the checkpointed function inside ``.backward()`` for one
# purpose: to re-save the tensors the original forward saved (``_recomputation_hook.pack_hook`` matches every tensor
# the re-run saves to the original forward's holder).  The re-run's return value is discarded (``recompute_fn`` calls
# the function and returns nothing), and with torch's early stop (the default) the re-run ends by
# ``_StopRecomputationError`` as soon as the last holder is matched.  The backward of this module's autograd node
# reads ``dout``, the saved set (:meth:`KcpSaved.tensors`: ``S_in``, ``c1_recv``, ``M_seg``, the saved tapes -- never
# ``out``) and the kept tapes; so when the node is the LAST saver of the checkpointed region, the recompute's #5
# (``fwd_o``, the output program) writes a tensor nothing ever reads.  ``recompute_output="auto"`` leaves that one
# launch out; the output tensor is still allocated (the same allocation sequence; torch sees the same shape).  The
# proof is read from torch's own frame and is strict:
#   0. the node is executable -- grad mode on at ``apply`` and an operand that requires grad (read by
#      :func:`chunk_gdn_kcp` right before ``apply``: inside ``forward`` grad mode is always off).  Only then does
#      autograd create the node and pack its saved tensors; a non-executable ``apply`` packs nothing, so the holders
#      still waiting would be a later op's and the count below could match by coincidence;
#   1. the torch release (:func:`torch_release`: ``torch.__version__`` without its local ``+<build>`` tag, so
#      ``2.14.1+cu130`` and ``2.14.1+cu132`` are one release) is one of :data:`RECOMPUTE_OUTPUT_TORCH_VERSIONS`
#      (the frame read below is that release's ``torch/utils/checkpoint.py``, the same file in every CUDA build of
#      the release; one constant, no environment variable);
#   2. the top saved-tensors hook pair is a checkpoint-internal one (``_checkpoint_internal``) and its pack hook,
#      unwrapped from ``torch._dynamo.disable``, is the recomputation hook: closure ``target_frame_ref`` (a weakref to
#      the ``_CheckpointFrame``) and ``gid`` (the graph task id or ``GraphExecGroup`` torch keys the recompute on);
#   3. ``frame.early_stop`` is ``True`` (the re-run ends right after this node's tensors are matched, so no later op
#      of the region runs on the unwritten output); and
#   4. ``len(frame.weak_holders) - frame.recomp_counter[gid]`` (the holders still waiting for a recomputed tensor)
#      equals exactly what this node's forward saves next: the five operands plus ``len(saved.tensors())``.
# Any other state launches #5 as before: a non-executable node, no hooks or a user's hooks on top, the original
# forward's hook, an unknown hook or frame shape, more holders waiting (a later op of the region saves a tensor -- e.g. the next layer of a
# multi-layer segment saves this layer's output as its ``v``), fewer (a mismatch torch itself reports), early stop
# off, another torch.  The decision never rests on ``_graph_task_id() != -1`` alone: a graph task also hosts the
# reentrant checkpoint's re-run and forwards inside backward hooks, whose outputs are live.  Only the autograd node's
# forward (:class:`ChunkGDNKcpFunction`, the one that saves exactly those tensors right after) may skip;
# :func:`chunk_gdn_kcp_fwd` and a captured graph (``graph="replay"``) never do.  No tensor content is read.

# torch RELEASES the frame proof was verified on, compared through torch_release(): the local build tag is not part
# of the identity (the proof reads torch/utils/checkpoint.py, the same file in every CUDA build of a release)
RECOMPUTE_OUTPUT_TORCH_VERSIONS: tuple[str, ...] = ("2.14.1",)


def torch_release(version: str | None = None) -> str:
    """The release of a torch version string (default: the running torch): ``"2.14.1+cu132"`` -> ``"2.14.1"``.
    The local ``+<build>`` tag is dropped; a pre-release tag (``2.15.0.dev20260901``) is kept -- that source is not
    a release's."""
    return str(torch.__version__ if version is None else version).split("+", 1)[0]


def torch_release_verified(version: str | None = None) -> bool:
    """Whether ``version`` (default: the running torch) is a release of :data:`RECOMPUTE_OUTPUT_TORCH_VERSIONS`."""
    return torch_release(version) in RECOMPUTE_OUTPUT_TORCH_VERSIONS


# q, k, v, g, beta: what ChunkGDNKcpFunction.forward saves besides the saved set
_SAVED_OPERANDS = 5
RECOMPUTE_OUTPUT_STATS: dict[str, int] = {
    "skipped": 0,  # recompute forwards whose #5 was left out (the proof held)
    "launched": 0,  # recompute forwards (frame reached) that launched #5: holders remain, a mismatch or early stop off
}
_RECOMPUTE_FRAME_REASONS = ("dead", "holders_remain", "mismatch", "no_early_stop")


def _recompute_output_dead(n_saved: int) -> tuple[bool, str]:
    """Whether the forward running now is torch's non-reentrant checkpoint recompute of a region this node closes
    (its ``n_saved`` tensors fill the frame's remaining holders) -- then nothing reads the recompute's output.
    ``(dead, reason)``; every unreadable or unexpected state is ``(False, reason)``."""
    if not torch_release_verified():
        return False, "torch_version"
    try:
        top = torch._C._autograd._top_saved_tensors_default_hooks(False)
    except Exception:  # noqa: BLE001 - no hook-stack binding: launch
        return False, "no_hook_stack"
    if top is None:
        return False, "no_hooks"
    pack = top[0]
    if not getattr(pack, "_checkpoint_internal", False):
        return False, "user_hooks"
    fn = (
        getattr(pack, "_torchdynamo_orig_callable", None)
        or getattr(pack, "__wrapped__", None)
        or pack
    )
    code = getattr(fn, "__code__", None)
    cells = getattr(fn, "__closure__", None)
    if code is None or not cells:
        return False, "hook_shape"
    names = tuple(code.co_freevars)
    if "target_frame_ref" not in names or "gid" not in names:
        # the original forward's hook (closure ``frame``) or an unknown shape
        return False, "checkpoint_forward"
    try:
        frame = cells[names.index("target_frame_ref")].cell_contents()
        gid = cells[names.index("gid")].cell_contents
        holders = frame.weak_holders
        counter = frame.recomp_counter
        early_stop = frame.early_stop
        done = counter.get(gid, 0) if isinstance(counter, dict) else None
    except Exception:  # noqa: BLE001 - a dead weakref or a frame without these fields: launch
        return False, "frame_shape"
    if (
        not isinstance(holders, list)
        or not isinstance(done, int)
        or isinstance(done, bool)
    ):
        return False, "frame_shape"
    if early_stop is not True:
        return False, "no_early_stop"
    remaining = len(holders) - done
    if remaining == n_saved:
        return True, "dead"
    return False, "holders_remain" if remaining > n_saved else "mismatch"


class _KeptTapes:
    """The recompute forward's tapes on their way to the backward of the same graph task (``reuse_recompute_tapes``):
    the ten :data:`FWD_TAPES` as fresh tensors, the ``stage_m`` plane when the saved set carries no ``M_seg``, the
    arena geometry they were produced at and the graph task id."""

    __slots__ = ("tapes", "stage_m", "geometry", "gid")

    def __init__(
        self,
        tapes: dict[str, torch.Tensor],
        stage_m: torch.Tensor | None,
        geometry: tuple[int, int, int, int] | None,
        gid: int,
    ) -> None:
        self.tapes = tapes
        self.stage_m = stage_m
        self.geometry = geometry
        self.gid = int(gid)

    def clear(self) -> None:
        self.tapes = {}
        self.stage_m = None


_KEPT: dict[tuple, _KeptTapes] = {}
KEPT_STATS: dict[str, int] = {
    "put": 0,  # recompute forwards that kept their tapes
    "take": 0,  # backwards that bound a kept entry
    "miss": 0,  # backwards with the option on that found no entry (they ran the replay)
    "released": 0,  # kept entries dropped after #8
    "callback": 0,  # entries dropped at a graph task's end (no backward node took them)
    "evicted": 0,  # entries dropped by a later hand-off forward (one live entry: an earlier layer of the segment, or another task)
}


def kept_registry_size() -> int:
    """Entries of the kept-tape registry (``0`` after every backward that took, or never needed, its entry)."""
    return len(_KEPT)


def _kept_key(
    gid: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    S_in: torch.Tensor,
    T_r: int,
    H_qk: int,
    HV: int,
) -> tuple:
    """The registry key of one autograd node: the graph task and the identities (data pointers, no contents) of the
    node's operands and saved ``S_in`` -- the recompute's ``save_for_backward`` tensors are what the backward unpacks,
    so both ends see the same pointers."""
    return (
        int(gid),
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        g.data_ptr(),
        beta.data_ptr(),
        S_in.data_ptr(),
        int(T_r),
        int(H_qk),
        int(HV),
    )


def _kept_evict_all() -> None:
    """One live entry: every entry goes -- an earlier layer of the same checkpoint segment (its backward runs the
    replay, same bits; the "miss" count says so), a finished task's leftover, or a failed task's (the engine runs no
    final callback after an exception).  Called before a hand-off forward's first launch, so the evicted set is freed
    before the new one is materialised, and again at the put."""
    for k in list(_KEPT):
        _KEPT.pop(k).clear()
        KEPT_STATS["evicted"] += 1


def _kept_put(key: tuple, entry: _KeptTapes) -> None:
    _kept_evict_all()
    _KEPT[key] = entry
    KEPT_STATS["put"] += 1
    _queue_task_end(key)


def _queue_task_end(key: tuple) -> bool:
    """Drop ``key`` at the current graph task's end when no backward node took it.  Outside a graph task the engine
    refuses the callback (``False``); the eviction in :func:`_kept_put` bounds the registry then."""
    try:
        torch.autograd.Variable._execution_engine.queue_callback(
            lambda: _kept_task_end(key)
        )
    except Exception:
        return False
    return True


def _kept_task_end(key: tuple) -> None:
    entry = _KEPT.pop(key, None)
    if entry is not None:
        entry.clear()
        KEPT_STATS["callback"] += 1


def _kept_take(key: tuple) -> _KeptTapes | None:
    entry = _KEPT.pop(key, None)
    KEPT_STATS["take" if entry is not None else "miss"] += 1
    return entry


def tape_spec(
    name: str, *, R: int, NC_pad: int, HV: int, T: int
) -> tuple[tuple[int, ...], torch.dtype]:
    """``(shape, dtype)`` of tape ``name`` for the padded geometry."""
    bf, f32 = torch.bfloat16, torch.float32
    if name in ("w", "u", "v_new", "do_g", "dv_intra", "dv_new"):
        return (R, HV, D), bf
    if name in ("aqk", "akk"):
        return (R, HV, BT), bf
    if name in ("g_cum", "rowscale", "qscale", "dg_part"):
        return (R, HV), f32
    if name in ("gamma", "gamma_dot"):
        return (NC_pad, HV), f32
    if name in ("h_tape", "dh_tape"):
        return (NC_pad, HV, D, D), bf
    if name in HOST_REDUCE_TAPES:
        return (T, HV, D), f32
    raise KeyError(name)


# ----------------------------------------------------------------------------------------------------------------
# Small helpers (metadata only)


def tile_index(h: int, slot: int, hg: int, n: int) -> int:
    """The kernels' head-group-major compact tile address: tile ``((h // hg) * n + slot) * hg + h % hg`` of a pool
    whose consecutive head groups each hold ``n`` slots of ``hg`` heads."""
    return ((h // hg) * n + slot) * hg + h % hg


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _alloc(
    shape: Sequence[int], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Every per-call tensor (outputs, saved set, collective outputs, tapes) comes from here: ``torch.empty`` for
    buffers every element of which a kernel or a collective of the same call writes."""
    return torch.empty(tuple(shape), dtype=dtype, device=device)


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
    *,
    token_strided: bool = False,
) -> None:
    """Shape / dtype / device of an operand.  ``token_strided``: the last two axes ``[heads, 128]`` must be dense and
    the token stride may be anything not smaller than one token's row (a packed-QKV view with token stride
    ``(2 H_qk + HV) * 128`` passes through as a TMA row stride); otherwise contiguous."""
    if tuple(x.shape) != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {tuple(x.shape)}")
    if x.dtype != dtype:
        raise ValueError(f"{name}: expected {dtype}, got {x.dtype}")
    if x.device != device:
        raise ValueError(f"{name}: on {x.device}, expected {device}")
    if not x.numel():
        return
    if token_strided:
        if x.stride(-1) != 1 or (x.dim() >= 3 and x.stride(-2) != shape[-1]):
            raise ValueError(
                f"{name}: the head and channel axes must be dense ([heads, 128] rows; strides {x.stride()})"
            )
        row = 1
        for s in shape[1:]:
            row *= s
        if shape[0] > 1 and x.stride(0) < row:
            raise ValueError(
                f"{name}: token stride {x.stride(0)} overlaps the {row}-element token row"
            )
    elif not x.is_contiguous():
        raise ValueError(f"{name}: must be contiguous (strides {x.stride()})")


def _operands(q, k, v, g, beta, ctx, *, dtypes: dict[str, torch.dtype], layout: str):
    """The operands without their optional batch axis, ``(T_r, H_qk, HV)`` and whether the inputs carried a batch
    axis.  ``T_r`` must equal the context's local token count; ``HV % H_qk == 0``.  ``layout="copy"`` copies strided
    ``q`` / ``k`` / ``v`` views to contiguous operands here (part of the call)."""
    v = _squeeze_batch(v, "v", 3)
    T, HV = int(v.shape[0]), int(v.shape[1])
    device = torch.device(ctx.device)
    batched = q.dim() == 4
    q, k = _squeeze_batch(q, "q", 3), _squeeze_batch(k, "k", 3)
    g, beta = _squeeze_batch(g, "g", 2), _squeeze_batch(beta, "beta", 2)
    H_qk = int(q.shape[1])
    validate_heads(H_qk, HV)
    if int(ctx.T_r) != T:
        raise ValueError(
            f"operands carry {T} tokens, the KcpContext of rank {ctx.rank} holds T_r={ctx.T_r}"
        )
    _check(q, "q", (T, H_qk, D), dtypes["q"], device, token_strided=True)
    _check(k, "k", (T, H_qk, D), dtypes["k"], device, token_strided=True)
    _check(v, "v", (T, HV, D), dtypes["v"], device, token_strided=True)
    _check(g, "g", (T, HV), dtypes["g"], device)
    _check(beta, "beta", (T, HV), dtypes["beta"], device)
    if layout == "copy":
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    return q, k, v, g, beta, T, H_qk, HV, batched


def _ranked(x: torch.Tensor, batched: bool) -> torch.Tensor:
    return x.unsqueeze(0) if batched else x


# ----------------------------------------------------------------------------------------------------------------
# Backends: the dispatch table of the eleven program launches


# The contract's storage dtypes.  A backend may override them (``Backend.dtypes``).
CONTRACT_DTYPES: dict[str, torch.dtype] = {
    "q": torch.bfloat16,
    "k": torch.bfloat16,
    "v": torch.bfloat16,
    "g": torch.float32,
    "beta": torch.float32,
    "out": torch.bfloat16,
    "dout": torch.bfloat16,
    "dq": torch.bfloat16,
    "dk": torch.bfloat16,
    "dv": torch.bfloat16,
    "dbeta": torch.float32,
    "dg": torch.float32,
    "state": torch.float32,  # the FP32 states, summaries and adjoints of every pool and collective
}


class Backend:
    """A dispatch table for the programs of :data:`PROGRAMS`.

    ``run(name, args)`` launches program ``name`` with the logical arguments ``args`` keyed by
    :data:`PROGRAM_ARGS` (device tensors, device int tables and Python ints; every pool is already resolved to its
    kernel form: base pointer, ``[NSEG]`` int32 index table, ``hg`` / ``n`` addressing scalars, ``tile_stride``) plus
    the private keys ``_call`` (the :class:`KcpCall`) and ``_grid`` (the launch's grid x).  ``dtypes`` are the
    storage dtypes of the operands, outputs and states.  The shipped backend is :class:`KernelBackend`
    (``backend="kernels"``); :class:`FakeBackend` (``backend="fake"``) runs the torch emulation of the same ABI.
    """

    name = "abstract"
    dtypes: dict[str, torch.dtype] = dict(CONTRACT_DTYPES)
    path = "fi-rendered"  # recorded with every result: fake (emulation), direct (bring-up), fi-rendered (the package)

    def run(
        self, name: str, args: dict[str, Any]
    ) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def begin_call(self, call: KcpCall) -> None:
        """Hook at the start of a forward or backward call."""

    def end_call(self, call: KcpCall) -> None:
        """Hook at the end of a call."""

    @staticmethod
    def transpose_tiles(
        *,
        src: torch.Tensor,
        src_tiles: torch.Tensor,
        dst: torch.Tensor,
        dst_tiles: torch.Tensor,
        scratch: torch.Tensor | None = None,
    ) -> None:
        """``dst`` tiles ``dst_tiles[i]`` <- the transposes of the ``src`` tiles ``src_tiles[i]`` (torch; bitwise;
        shared by every backend: the ``M_f`` -> ``M_f^T`` step of the segmented summaries).  ``scratch`` (flat,
        >= ``len(src_tiles)`` tiles, nothing live in it): the gathered tiles are written there (``index_select``
        with ``out=``) instead of into a fresh temporary; the values and the scatter are the same either way."""
        tiles = src.view(-1, D, D)
        if scratch is None:
            gathered = tiles[src_tiles]
        else:
            n = int(src_tiles.numel())
            gathered = scratch.view(-1)[: n * TILE].view(n, D, D)
            torch.index_select(tiles, 0, src_tiles, out=gathered)
        dst.view(-1, D, D)[dst_tiles] = gathered.transpose(1, 2)


def _public_args(args: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in args.items() if not k.startswith("_")}


def _registry() -> Any | None:
    """The package registry (:mod:`.cake_jit`: records, ``select_record``, ``launch_grid``), ``None`` without it."""
    try:
        from . import cake_jit  # noqa: PLC0415 - the registry; optional at import time
    except ImportError:
        return None
    return cake_jit


def resolve_launch(
    name: str, args: dict[str, Any], *, cols_per_cta: int = 8
) -> tuple[str, dict[str, Any], tuple[int, int, int], bool]:
    """``(record, values, grid, is_test)`` of one launch of program ``name``: the registry record its scalars select
    (``cake_jit.select_record``: the merge variant by ``ncols``, the transfer walk's test record by the static knobs),
    the public values (the merge's ``cols_per_cta`` filled from the record's grid rule), the grid under the record's
    rule (``cake_jit.launch_grid``) and whether the record is a ``build.test.json`` one.  An explicit ``cols_per_cta``
    (test knob ``cols_rule="by_chains"``) names the merge record directly; without the registry the module's own map
    and the caller's grid x apply."""
    values = _public_args(args)
    pm = int(args.get("_precision_mode", _opt.PRECISION_MODE_DEFAULT))
    tw = str(args.get("_transfer_walk", _opt.TRANSFER_WALK_DEFAULT))
    grid_args = dict(args.get("_grid_args") or {})
    is_test = test_record(name, pm, tw) is not None
    explicit = name == "kcp_affine_merge" and values.get("cols_per_cta") is not None
    vh = None
    if name in FWDH_PROGRAMS:
        vh = fwdh_vh(
            grid_args.get("n_walk", 0),
            grid_args.get("num_heads", 0),
            args.get("_sm_count", 0),
            args.get("_fwdh_vh"),
        )
    elif name in DHU_PROGRAMS:
        vh = dhu_vh(
            grid_args.get("n_walk", 0),
            grid_args.get("num_heads", 0),
            args.get("_sm_count", 0),
            args.get("_dhu_vh"),
        )
    call = args.get("_call")
    jit = _registry()
    if jit is not None and not explicit:
        per_head = 2 if vh == 64 else 1
        select = dict(ncols=values.get("ncols"), precision_mode=pm, transfer_walk=tw)
        try:
            record = str(jit.select_record(name, ctas_per_head=per_head, **select))
        except TypeError:  # a registry without the ctas_per_head select rule: the suffixed record, if any
            record = str(jit.select_record(name, **select))
            if per_head == 2 and (record + VH64_SUFFIX) in jit.MODULES:
                record = record + VH64_SUFFIX
            else:
                per_head = 1
        except ValueError:
            if per_head != 2:
                raise
            per_head = 1  # no vh = 64 record admits the launch: the vh = 128 form
            record = str(jit.select_record(name, **select))
        rec = jit.MODULES.get(record) or jit.TEST_MODULES.get(record) or {}
        if per_head == 2:
            # The form launched is the selected record's: a registry whose records of the program carry no
            # half-width form (selection ``always``) answers the vh = 64 request with the full-head record.
            per_head = int(
                rec.get("grid_rule", {}).get(
                    "ctas_per_head", 2 if record.endswith(VH64_SUFFIX) else 1
                )
            )
        if per_head == 2:
            grid_args["ctas_per_head"] = (
                2  # two CTAs per (segment, head): cake_jit.launch_grid's vh = 64 form
            )
        if name == "kcp_affine_merge":
            values["cols_per_cta"] = int(
                rec.get("grid_rule", {}).get("cols_per_cta", cols_per_cta)
            )
        if grid_args:
            gx, gy, gz = (int(x) for x in jit.launch_grid(record, **grid_args))
            grid = (gx, gy, gz)
        else:
            grid = (int(args["_grid"]), 1, 1)
        if call is not None:
            call.records.append((record, grid))
        return record, values, grid, is_test
    grid_x = int(args["_grid"])
    if is_test:
        record = test_record(name, pm, tw)
    elif name == "kcp_affine_merge":
        values["cols_per_cta"] = int(values.get("cols_per_cta") or cols_per_cta)
        record = f"kcp_affine_merge_c{values['cols_per_cta']}"
    else:
        record = RECORD_OF[name]
        if vh == 64:
            record, grid_x = record + VH64_SUFFIX, grid_x * 2
    grid = (grid_x, 1, 1)
    if call is not None:
        call.records.append((str(record), grid))
    return str(record), values, grid, is_test


class KernelBackend(Backend):
    """The shipped backend: the exported programs through the package's launch glue (:mod:`.cake_backend`
    ``run_module(record, values, *, grid, device_index, module=None)``, imported at the first launch so this module
    imports without it; ``values`` keyed by the registry's logical names = :data:`PROGRAM_ARGS`).  Records:
    :data:`RECORD_OF` (``cake_jit.select_record(program, ncols=)`` when the registry provides it; the merge by
    ``cols_per_cta``: ``kcp_affine_merge_c8`` / ``_c16``); the transfer walk's test-only records (:func:`test_record`:
    ``precision_mode`` 1 / 2; no BF16x3 module exists) are loaded through ``cake_backend.load_test(record,
    arch)`` -- keywords of the call named in ``test_knobs``, never an environment variable."""

    name = "kernels"

    def __init__(
        self,
        launcher: Any | None = None,
        *,
        cols_per_cta: int = 8,
        path: str = "fi-rendered",
    ) -> None:
        self._launcher = launcher
        if cols_per_cta not in (8, 16, 32) or D % cols_per_cta:
            raise ValueError(f"cols_per_cta must be 8, 16 or 32, got {cols_per_cta}")
        self.cols_per_cta = int(cols_per_cta)
        self.path = path
        self._test_modules: dict[tuple[str, int], Any] = {}

    def launcher(self) -> Any:
        """The launch glue: ``cake_backend`` (the exporter's loader module) or, until it lands, the registry's own
        ``run_module`` / ``load`` / ``device_arch`` (``cake_jit``)."""
        if self._launcher is None:
            try:
                from . import cake_backend  # noqa: PLC0415 - the launch glue; optional at import time

                self._launcher = cake_backend
            except ImportError:
                from . import cake_jit  # noqa: PLC0415

                self._launcher = cake_jit
        return self._launcher

    def run(self, name: str, args: dict[str, Any]) -> None:
        call: KcpCall = args["_call"]
        be = self.launcher()
        device_index = (
            call.device.index
            if call.device.index is not None
            else torch.cuda.current_device()
        )
        grid_args = args.get("_grid_args")
        if grid_args and "sm_count" in grid_args:
            # Persistent grids (#5 / #6 / #8): the record's per-architecture CTA factor
            # reads the architecture this launch loads the library for.
            args = {
                **args,
                "_grid_args": {**grid_args, "arch": be.device_arch(device_index)},
            }
        record, values, grid, is_test = resolve_launch(
            name, args, cols_per_cta=self.cols_per_cta
        )
        module = None
        if is_test:  # a build.test.json record: loaded through load_test, never selected by default
            key = (record, int(device_index))
            module = self._test_modules.get(key)
            if module is None:
                load_test = getattr(be, "load_test", None) or be.load
                module = self._test_modules[key] = load_test(
                    record, be.device_arch(device_index)
                )
        be.run_module(
            record, values, grid=grid, device_index=device_index, module=module
        )


class FakeBackend(Backend):
    """The torch emulation of the program ABI (``backend="fake"``; the package's :mod:`.cake_fake_backend`, imported at
    the first launch): the same logical arguments, the same tapes and tables, the design's rounding points, so the
    orchestration (exchange, saved set, replay, #7039) is testable before the kernels exist.  The module exposes
    ``run_module(record, values, *, grid, device_index, module=None)`` (the kernels' call) or one callable per
    program name taking the logical arguments as keywords."""

    name = "fake"
    path = "fake"

    def __init__(self, module: Any | None = None) -> None:
        self._module = module

    def module(self) -> Any:
        if self._module is None:
            from . import cake_fake_backend  # noqa: PLC0415 - the emulation; optional at import time

            self._module = cake_fake_backend
        return self._module

    def run(self, name: str, args: dict[str, Any]) -> None:
        mod = self.module()
        call: KcpCall = args["_call"]
        record, values, grid, _is_test = resolve_launch(name, args)
        run_module = getattr(mod, "run_module", None)
        if (
            run_module is not None
        ):  # the package emulation: the kernels' call on the record name
            index = call.device.index if call.device.type == "cuda" else None
            run_module(record, values, grid=grid, device_index=index)
            return
        getattr(mod, name)(**values)


_BACKENDS: dict[str, Backend] = {}


def get_backend(name: str = _opt.BACKEND_DEFAULT) -> Backend:
    """The backend instance of ``name`` (``"kernels"`` | ``"fake"``), created on first use."""
    b = _BACKENDS.get(name)
    if b is None:
        if name == "kernels":
            b = KernelBackend()
        elif name == "fake":
            b = FakeBackend()
        else:
            raise ValueError(f"backend must be one of {_opt.BACKENDS}, got {name!r}")
        _BACKENDS[name] = b
    return b


def set_backend(name: str, backend: Backend | None) -> Backend | None:
    """Install ``backend`` under ``name`` (``None``: forget the instance so the next call recreates the default);
    returns the previous instance."""
    previous = _BACKENDS.pop(name, None)
    if backend is not None:
        _BACKENDS[name] = backend
    return previous


def merge_cols_per_cta(ncols: int, n_chains: int, rule: str) -> int | None:
    """The merge's ``cols_per_cta`` a launch asks for explicitly, ``None`` for the registry's choice of record
    (``fixed``: c8 state / c16 map).  ``by_chains`` keeps the 8-column program for 128-column state chains up to
    :data:`COLS_RULE_CHAINS` concurrent chains and takes the 16-column program beyond it."""
    if rule == "by_chains" and int(ncols) == D and int(n_chains) > COLS_RULE_CHAINS:
        return 16
    return None


# ----------------------------------------------------------------------------------------------------------------
# Compact pools and chains (verbatim from the KDA host module)


@dataclasses.dataclass(frozen=True)
class Indexed:
    """A compact state pool as a walk kernel reads or writes it: ``pool`` (a flat tensor of ``128 x 128`` tiles),
    ``index[seg]`` (the slot of segment ``seg``; ``-1`` = no tile: the zero state on input, not written on output) and
    ``tiles[slot][h]`` (the tile number of head ``h`` of that slot inside ``pool``).  :meth:`addressing` recovers the
    kernels' two-integer form: ``tiles[slot][h] == base_tile + tile_index(h, slot, hg, n) * stride``."""

    pool: torch.Tensor
    index: tuple[int, ...]
    tiles: tuple[tuple[int, ...], ...]

    @property
    def n_slots(self) -> int:
        return len(self.tiles)

    def addressing(self, H: int) -> tuple[int, int, int, int]:
        """``(base_tile, hg, n, stride)`` of the kernels' addressing, ``ValueError`` when the table has no such form."""
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


@dataclasses.dataclass(frozen=True)
class _Block:
    """One compact indexed state block as a walk kernel addresses it: base pointer, ``index[NSEG]`` (i32 device),
    ``hg``, ``n`` and the ``tile_stride`` of the pool (1 dense plane, 2 interleaved pair)."""

    ptr: torch.Tensor
    index: torch.Tensor
    hg: int
    n: int
    stride: int


class Chains:
    """Merge chain tables of one launch: CSR over elements ``(elem_m, elem_c, elem_y)`` in application order
    (element offsets into the ``m`` / ``c`` / ``y`` pools; ``elem_c = -1`` no offset; ``elem_y`` = where the state
    ENTERING the element is published when ``write_states``, ``-1`` none), the initial state ``x0_index`` (``-1``:
    zero) and ``y_final`` (the state leaving the chain; an empty chain publishes its initial state).  ``elem_m`` must
    be ``>= 0`` (the kernel has no identity element)."""

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

    def content(self) -> tuple:
        """A hashable view of the tables (the device-table cache key: by content, never by object identity)."""
        return (
            self.ncols,
            tuple(self.chain_start),
            tuple(self.elem_m),
            tuple(self.elem_c),
            tuple(self.elem_y),
            tuple(self.x0_index),
            tuple(self.y_final),
        )

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


# ----------------------------------------------------------------------------------------------------------------
# Tables of one (ctx, HV) problem, derived from the plan module's head plan (``ctx.heads(HV)``)


class _Tables:
    """Everything the launch sequences read for one ``(ctx, HV)``: the segment plan, walk lists, the compact tile
    tables of every pool, the collective splits and the merge chains, derived once from ``cake_kcp_plan.KcpContext``
    / ``KcpHeadPlan`` (``H := HV``) and cached by the workspace.  The int device tables of this problem live on the
    head plan (``hp.device_tables``, device-resident on ``ctx`` keyed by ``HV`` / content, built once at first use).

    Pools (tile = 16384 floats; tile numbers pool-relative): ``c1_send`` = owner-major ``(out_slot, head)`` pairs
    ``[B_f | M_f^T]`` (B tile ``2 t``, M tile ``2 t + 1`` of ``hp.out_tile[out_slot][h] = t``); ``S_in`` (the c2
    receive, ``hp.h0_tile``), ``b2_recv`` = ``dS_out`` (``hp.dht_tile``), ``b1_send`` (``hp.b1_d_tile_d`` /
    ``hp.b1_d_tile_dm`` / ``hp.b1_m_tile_dm``: the plan module's layout of the ``d`` and ``dm`` forms); at ``P > 1`` the
    staging planes ``stage_b`` (``b_seg`` / ``d_seg``) and ``stage_m`` (``M_seg^T``) ``[NSEG, HV]`` and the carry
    plane ``[n_frag (P - 1), HV]`` (forward carries, then the adjoint carries); ``eye`` = one identity tile.

    Operator orientation (merge): ``transpose_m = 0`` reads a tile holding ``Mop^T`` (the c1 ``M_f^T`` tiles and the
    ``M_seg^T`` staging under ``Mop = M``; the b1 ``M_f`` tiles under ``Mop = M^T``), ``transpose_m = 1`` a tile holding
    ``Mop`` (the c1 / staging ``M^T`` tiles under ``Mop = M^T``).  Walk lists are in LPT order (longest segment first,
    ties by segment number): the CTA issue order of a launch only -- never a bit.
    """

    _ring: _Ring | None = None  # the ring-exchange tables, derived on first use

    def __init__(
        self, ctx: Any, HV: int, *, fused_walks: int = 1, P: int | None = None
    ) -> None:
        self.ctx = ctx
        self.H = H = int(HV)  # every pool, split and chain is per VALUE head
        self.fused_walks = int(
            fused_walks
        )  # one tape launch per phase from the carry plane
        hp = self.hp = (
            ctx.heads(H) if P is None else ctx.heads(H, P=int(P))
        )  # the segments override
        self.P, self.NSEG = int(hp.P), int(hp.NSEG)
        self.Hg, self.Hg_r = int(hp.Hg), int(hp.Hg_r)
        self.n_frag, self.T_r = int(ctx.n_frag), int(ctx.T_r)
        self.n_in, self.n_out = int(ctx.n_in), int(ctx.n_out)
        self.N_in, self.N_out = (
            int(ctx.N_in),
            int(ctx.N_out),
        )  # group totals (every rank agrees: global metadata)
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

        def lpt(lst):
            return sorted(
                (int(s) for s in lst),
                key=lambda s: (-(seg_table[s + 1] - seg_table[s]), s),
            )

        def segs(fs, ps):
            return lpt(f.slot * P + p for f in fs for p in ps)

        # -- walk lists (segments)
        self.walk_offset = lpt(
            hp.walk_seg_offset
        )  # offset walks (crossing-out frags at P = 1, all at P > 1)
        self.walk_all = lpt(range(NSEG))
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
        self.walk_in = (
            segs(ins, [0]) if P == 1 else self.walk_all
        )  # adjoint offset walk
        self.walk_m_b1 = (
            segs([f for f in ins if f.crossing_out], [0]) if P == 1 else self.walk_all
        )  # M-only (b1)
        self.walk_live = segs(starts + ins, range(P))
        self.walk_in_fused = segs(ins, range(P))
        self.walk_cross_fused = segs(outs, range(P))
        # -- per-segment slot tables
        self.h0_index = tuple(int(i) for i in hp.h0_index)
        self.carry_index = tuple(int(i) for i in hp.carry_index)
        self.dht_index = tuple(int(i) for i in hp.dht_index)
        self.dcarry_index = tuple(int(i) for i in hp.dcarry_index)
        self.h0_index_fused = tuple(
            int(i) for i in hp.h0_index_fused
        )  # the carry plane (P > 1) / S_in (P = 1)
        self.dht_index_fused = tuple(
            int(i) for i in hp.dht_index_fused
        )  # the carry plane (P > 1) / dS_out (P = 1)
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
        # the head plan's own device tables (uploaded once at first use) by content, so dev() never re-uploads them
        self._named: dict[tuple[torch.dtype, tuple[int, ...]], str] = {}
        for name in (
            "seg_chunk_start",
            "walk_seg_offset",
            "walk_seg_tape",
            "h0_index",
            "out_index",
            "dht_index",
            "carry_index",
            "dcarry_index",
            "h0_index_fused",
            "dht_index_fused",
        ):
            vals = getattr(hp, name, None)
            if vals is not None:
                self._named.setdefault((torch.int32, tuple(int(v) for v in vals)), name)
        # tape coverage: #2 / #7 ``program=tape`` write h / v_new (dh / dv_new) for the walked
        # segments' chunks only; the local programs read chunks c < NC only, every such chunk belongs to exactly one
        # live segment, and every tape phase below walks every live segment -- no pre-zeroing of the tapes.
        live_segs: set[int] = {
            s
            for s in range(NSEG)
            if int(hp.seg_chunk_start[s + 1]) > int(hp.seg_chunk_start[s])
        }
        for label, lists in (
            ("fwd fused wait_first", (self.walk_live,)),
            ("fwd fused overlap", (self.walk_start, self.walk_in_fused)),
            ("fwd split", (self.walk_start, self.walk_in_p0, self.walk_in_p1)),
            ("adj fused wait_first", (self.walk_live,)),
            ("adj fused overlap", (self.walk_end, self.walk_cross_fused)),
            (
                "adj split",
                (self.walk_end, self.walk_cross_last, self.walk_cross_rest),
            ),
        ):
            covered: set[int] = set()
            for lst in lists:
                covered.update(int(x) for x in lst)
            if not live_segs <= covered:
                raise RuntimeError(
                    f"tape walks ({label}) miss live segments {sorted(live_segs - covered)}: the tapes would hold "
                    f"unwritten rows"
                )
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
        ):  # + one slot per crossing fragment for the S_in / dS_out copies
            n_carry += max(self.n_in, self.n_out)
        self.carry_tiles = tuple(
            tuple(c * H + h for h in range(H)) for c in range(n_carry)
        )
        # -- splits (elements) and pool sizes (elements)
        self.splits = {k: tuple(int(x) for x in v) for k, v in hp.splits.items()}
        numel = {k: int(v) for k, v in hp.numel.items()}
        self.numel = numel
        self.sizes = {
            "c1_send": numel["c1_send"] + (self.n_out * H * TILE if P > 1 else 0),
            "c1_recv": numel["c1_recv"],
            "c2_send": numel["c2_send"],
            "S_in": numel["c2_recv"],
            "b1_send": max(numel["b1_send_d"], numel["b1_send_dm"]),
            "b1_recv": max(numel["b1_recv_d"], numel["b1_recv_dm"]),
            "b2_send": numel["b2_send"],
            "b2_recv": numel["b2_recv"],
            "stage_b": NSEG * H * TILE if P > 1 else 0,
            "stage_m": NSEG * H * TILE if P > 1 else 0,
            "carry": n_carry * H * TILE,
            "eye": TILE,
            "ring_out": max(self.n_in, self.n_out) * H * TILE,
            "ring_in": max(self.n_in, self.n_out) * H * TILE,
        }
        self._check_tiles()
        self._build_chains(starts, ins, outs, ends, in_slot, out_slot)

    # merge chain families: the pools their offsets index, the operator orientation and the state publication.
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
            extra = self.n_frag * (
                P - 1
            )  # the extra carry slots hold the S_in (dS_out) copies of the fused form

            def copy_slot(f, h: int, slot: int) -> int:
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
        """The owner's forward chains: per document and owned head, from ``X0 = B_{f1}`` over the fragments
        ``f2 .. f(n-1)`` with ``Mop = M_f`` (c1 ``M_f^T`` tiles, ``transpose_m = 0``); the state entering each element
        and the final state are the ``S_in`` of the element's fragment and of ``fn``, published into the c2 send layout."""
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
        """The owner's adjoint chains: per document and owned head, from ``X0 = d_{fn}`` over ``f(n-1) .. f2`` with
        ``Mop = M_f^T``; the state entering each element and the final state are the ``dS_out`` of the element's
        predecessor, published into the b2 send layout.  ``d_key`` / ``m_key`` name the plan module's offset tables."""
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

    # -- device tables (built on the CPU once at first use, device-resident on the head plan, cached by content)
    def dev(
        self,
        key: Any,
        values: Sequence[int],
        device: torch.device,
        dtype: torch.dtype = torch.int32,
    ) -> torch.Tensor:
        cache = self.hp.device_tables
        content = tuple(int(v) for v in values)
        named = self._named.get((dtype, content))
        if named is not None:  # the head plan's upload of this very table
            t = cache.get(named)
            if t is not None and t.device == device and t.dtype == dtype:
                return t
        k = ("kcp", key, str(device), dtype, content)
        t = cache.get(k)
        if t is None:
            # one asynchronous pinned copy (no synchronizing call; cake_kcp_plan.upload_tables)
            t = cake_kcp_plan.upload_tables({"t": (k[4], dtype, None)}, device)["t"]
            cache[k] = t
        return t

    def seg_table(self, device: torch.device) -> torch.Tensor:
        """``seg_chunk_start`` int32 ``[NSEG + 1]`` on the device (the head plan's copy)."""
        t = self.hp.device_tables.get("seg_chunk_start")
        if t is None or t.device != device:
            t = self.dev("seg_chunk_start", self.hp.seg_chunk_start, device)
        return t

    def device_tiles(self, tiles, device: torch.device) -> torch.Tensor:
        """The flat int64 device copy of a tile table ``[slot][h]``."""
        return self.dev(
            "tiles", [int(x) for row in tiles for x in row], device, torch.int64
        )

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

    def pool_sizes(self, backward: bool) -> dict[str, int]:
        names = [
            "c1_send",
            "c1_recv",
            "c2_send",
            "stage_b",
            "stage_m",
            "carry",
            "eye",
            "ring_out",
            "ring_in",
        ]
        if backward:
            names += ["b1_send", "b1_recv", "b2_send", "b2_recv"]
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
# Workspace: transient exchange pools and the tape arena; saved-set tensors are fresh per call


def _identity_tile(
    ctx: Any, n: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """The ``eye`` pool (``n`` = one tile; the ``D x D`` identity), built once per (context, device, dtype) and shared
    by every workspace of the context: no program writes it (the zero block's never-read base, the merge's identity
    operand, the stand-in of unused tape arguments)."""
    caches = getattr(ctx, "_caches", None)
    key = ("kcp_eye", int(n), str(dtype), str(device))
    t = caches.get(key) if isinstance(caches, dict) else None
    if t is None:
        t = _alloc((int(n),), dtype, device)
        t.view(D, D).copy_(torch.eye(D, dtype=dtype, device=device))
        if isinstance(caches, dict):
            caches[key] = t
    return t


def _context_tables(
    ctx: Any, HV: int, *, fused_walks: int = 1, P: int | None = None
) -> _Tables:
    """The launch tables of ``(ctx, HV, fused_walks, P)``, derived once per context (kept in the context's cache
    beside its head plans) so a per-call workspace does not rebuild them: they are plan data, never written by a
    call."""
    caches = getattr(ctx, "_caches", None)
    key = ("kcp_tables", int(HV), int(fused_walks), None if P is None else int(P))
    tb = caches.get(key) if isinstance(caches, dict) else None
    if tb is None:
        tb = _Tables(ctx, HV, fused_walks=fused_walks, P=P)
        if isinstance(caches, dict):
            caches[key] = tb
    return tb


class KcpWorkspace:
    """The transient buffers of the KCP route on one rank: the exchange pools (flat, ``state_dtype``: ``c1_send``,
    ``c1_recv``, ``c2_send``; ``b1_send``, ``b1_recv``, ``b2_send``, ``b2_recv`` for the backward; at ``P > 1`` the
    staging planes ``stage_b`` / ``stage_m`` and the ``carry`` plane; one identity tile ``eye``) and the tape arena
    (:data:`FWD_TAPES` / :data:`BWD_TAPES` of the padded geometry ``(R, NC_pad, HV)``, allocated on first use;
    ``dv_new`` is the ``dv_intra`` buffer, :data:`TAPE_ALIASES`).  The
    saved-set tensors (``S_in`` = the c2 output, the c1 receive tensor under ``save_transfer``, ``M_seg`` under
    ``save_transfer`` at ``P > 1``, the forward tapes under ``save_mode="full"``) are never workspace slices: the
    calls allocate them fresh.  The persistent programs' dynamic-ticket words live here too: one 128-byte u32 slot
    per launch site (:data:`TICKET_SITES`), zeroed once when the workspace is first sized, never released, never
    read by the host; every launch leaves its word at zero (the kernels' self-resetting ticket), so a captured graph
    replays on the same slots.  A workspace is reusable across calls and layers of the same ``(ctx, H_qk, HV)``; a
    call with a larger need grows it.  ``bytes`` counts the pools, the tapes and the ticket slots."""

    def __init__(
        self,
        ctx: Any | None = None,
        H_qk: int | None = None,
        HV: int | None = None,
        device: torch.device | str | None = None,
        *,
        backward: bool = True,
        state_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.state_dtype = state_dtype
        self.device: torch.device | None = None
        self._pools: dict[str, torch.Tensor] = {}
        self._tables: dict[tuple[Any, int, int, int | None], _Tables] = {}
        self._tapes: dict[str, torch.Tensor] = {}
        self._tickets: torch.Tensor | None = (
            None  # u32 [len(TICKET_SITES) * 32]: one 128-byte slot per site
        )
        self.geometry: tuple[int, int, int, int] | None = None  # (R, NC_pad, HV, T)
        if ctx is not None:
            if H_qk is None or HV is None:
                raise ValueError("KcpWorkspace: H_qk and HV are required with ctx")
            self.ensure(ctx, H_qk, HV, device=device, backward=backward)

    def tables(
        self, ctx: Any, HV: int, *, fused_walks: int = 1, P: int | None = None
    ) -> _Tables:
        key = (ctx, int(HV), int(fused_walks), None if P is None else int(P))
        tb = self._tables.get(key)
        if tb is None:
            tb = _context_tables(ctx, HV, fused_walks=fused_walks, P=P)
            self._tables[key] = tb
        return tb

    def ensure(
        self,
        ctx: Any,
        H_qk: int,
        HV: int,
        *,
        device: torch.device | str | None = None,
        backward: bool = True,
        fused_walks: int = 1,
        P: int | None = None,
        skip: Sequence[str] = (),
    ) -> _Tables:
        """Size the pools for ``(ctx, HV)`` and bind the tape geometry; returns the tables (``P``: the segments
        override of the plan).  ``skip``: pool names this call never touches (:func:`_prepare` passes the ring
        pools under ``exchange="a2a"`` and the pools the call replaces with fresh / saved tensors, ``c1_recv`` and
        ``stage_m`` under ``save_transfer``); a skipped pool is not allocated, a held one is left as it is, and a
        later ``ensure()`` without the name sizes it.  A workspace built from ``(ctx, H_qk, HV)`` sizes every pool."""
        device = (
            torch.device(device) if device is not None else torch.device(ctx.device)
        )
        if self.device is None:
            self.device = device
        elif self.device != device:
            raise ValueError(
                f"KcpWorkspace: lives on {self.device}, the call is on {device}"
            )
        validate_heads(H_qk, HV)
        tb = self.tables(ctx, HV, fused_walks=fused_walks, P=P)
        for name, n in tb.pool_sizes(backward).items():
            if name in skip:
                continue
            cur = self._pools.get(name)
            if cur is None or cur.numel() < n:
                self._pools[name] = (
                    _identity_tile(ctx, n, self.state_dtype, device)
                    if name == "eye"
                    else _alloc((n,), self.state_dtype, device)
                )
        if (
            self._tickets is None
        ):  # once per workspace: the kernels keep every word at zero afterwards
            # one slot of slack, then the 128-byte-aligned view: the CUDA caching allocator hands out 512-byte-aligned
            # blocks (offset 0), the CPU allocator only 64-byte-aligned ones (offset 0 or 16 words)
            words = len(TICKET_SITES) * TICKET_SLOT_WORDS
            base = torch.zeros(
                (words + TICKET_SLOT_WORDS,), dtype=torch.uint32, device=device
            )
            skew = (-base.data_ptr() % TICKET_SLOT_BYTES) // base.element_size()
            self._tickets = base[skew : skew + words]
        geometry = (int(ctx.R), int(ctx.NC_pad), int(HV), int(ctx.T_r))
        if self.geometry != geometry:
            self._tapes.clear()  # another padded geometry: the tapes are reallocated on first use
            self.geometry = geometry
        return tb

    def ticket(self, site: str) -> torch.Tensor:
        """The 128-byte u32 ticket slot of persistent launch site ``site`` (one of :data:`TICKET_SITES`): the
        kernel's ``ticket`` argument, zero before and after every launch, never read on the host."""
        if site not in TICKET_SITES:
            raise KeyError(
                f"KcpWorkspace: {site!r} is not a persistent launch site {TICKET_SITES}"
            )
        if self._tickets is None:
            raise KeyError("KcpWorkspace: no ticket slots yet (ensure() first)")
        i = TICKET_SITES.index(site) * TICKET_SLOT_WORDS
        return self._tickets[i : i + TICKET_SLOT_WORDS]

    def pool(self, name: str, n: int | None = None) -> torch.Tensor:
        t = self._pools.get(name)
        if t is None:
            if n not in (None, 0) or self.device is None:
                raise KeyError(
                    f"KcpWorkspace: pool {name!r} was not sized for this call (ensure() first)"
                )
            t = self._pools[name] = _alloc((0,), self.state_dtype, self.device)
        return t if n is None else t[:n]

    def has(self, name: str) -> bool:
        return name in self._pools

    def release_pool(self, name: str) -> bool:
        """Drop pool ``name`` (sized again by the next ``ensure``); ``True`` when it was held."""
        return self._pools.pop(name, None) is not None

    def tape(self, name: str) -> torch.Tensor:
        """Tape ``name`` of the bound geometry (allocated on first use; an alias of :data:`TAPE_ALIASES` resolves to
        its target's buffer)."""
        name = TAPE_ALIASES.get(name, name)
        t = self._tapes.get(name)
        if t is None:
            if self.geometry is None or self.device is None:
                raise KeyError("KcpWorkspace: no tape geometry bound (ensure() first)")
            R, NC_pad, HV, T = self.geometry
            shape, dtype = tape_spec(name, R=R, NC_pad=NC_pad, HV=HV, T=T)
            t = self._tapes[name] = _alloc(shape, dtype, self.device)
        return t

    def release_tape(self, name: str) -> bool:
        """Drop arena tape ``name`` (re-allocated on its next use); ``True`` when it was bound."""
        return self._tapes.pop(TAPE_ALIASES.get(name, name), None) is not None

    def adopt_tape(self, name: str, t: torch.Tensor) -> bool:
        """Bind ``t`` as arena tape ``name`` when none is bound and ``t`` has the bound geometry's shape and dtype on
        this device (a consumed kept tape returning to the arena); ``True`` when bound."""
        name = TAPE_ALIASES.get(name, name)
        if (
            name in self._tapes
            or self.geometry is None
            or self.device is None
            or t.device != self.device
        ):
            return False
        R, NC_pad, HV, T = self.geometry
        shape, dtype = tape_spec(name, R=R, NC_pad=NC_pad, HV=HV, T=T)
        if tuple(t.shape) != tuple(shape) or t.dtype != dtype:
            return False
        self._tapes[name] = t
        return True

    @property
    def bytes(self) -> int:
        return (
            sum(t.numel() * t.element_size() for t in self._pools.values())
            + self.tape_bytes
            + self.ticket_bytes
        )

    @property
    def ticket_bytes(self) -> int:
        t = self._tickets
        return 0 if t is None else t.numel() * t.element_size()

    @property
    def tape_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._tapes.values())

    @property
    def pool_bytes(self) -> dict[str, int]:
        return {k: t.numel() * t.element_size() for k, t in self._pools.items()}


# ----------------------------------------------------------------------------------------------------------------
# Saved set and exchange scheme


@dataclasses.dataclass(frozen=True)
class KcpSaved:
    """The saved context of one forward (``boundary_mode="saved"``): ``S_in`` (the fresh c2 output), the owner's c1
    receive tensor (``save_transfer``), ``M_seg`` (``save_transfer`` at ``P > 1``), the saved tapes of the
    ``save_mode`` (:func:`saved_tapes`: none in ``"light"``, every forward tape in ``"full"``) and the scalars of the
    call.  Every
    tensor is fresh per call and travels through ``save_for_backward`` only (#7039).  ``kept`` (never saved, never
    compared: not in ``tensors()`` / ``flags()`` / ``meta()``) is the recompute forward's tape hand-off of
    ``reuse_recompute_tapes`` on its way through the autograd Function's backward; ``None`` everywhere else."""

    version: int
    S_in: torch.Tensor
    c1_recv: torch.Tensor | None
    M_seg: torch.Tensor | None
    tapes: tuple[torch.Tensor, ...] | None
    T_r: int
    H_qk: int
    HV: int
    n_frag: int
    P: int
    Hg: int
    scale: float
    save_mode: str
    save_transfer: bool
    offset_class: str
    exchange: str
    precision_mode: int
    transfer_walk: str
    ctx: Any
    kept: Any = dataclasses.field(default=None, compare=False, repr=False)
    # The backward may forget ``S_in`` / ``c1_recv`` / ``M_seg`` at their last reader (:meth:`forget`): set by the
    # autograd Function on the set it rebuilds from ``save_for_backward`` (nothing else references those tensors
    # there); a caller-owned set keeps every tensor.  Never saved, never compared.
    release_in_backward: bool = dataclasses.field(
        default=False, compare=False, repr=False
    )

    def tensors(self) -> tuple[torch.Tensor, ...]:
        out = [self.S_in]
        if self.c1_recv is not None:
            out.append(self.c1_recv)
        if self.M_seg is not None:
            out.append(self.M_seg)
        if self.tapes is not None:
            out.extend(self.tapes)
        return tuple(out)

    def flags(self) -> tuple[bool, ...]:
        return (
            self.c1_recv is not None,
            self.M_seg is not None,
            self.tapes is not None,
        )

    def meta(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "T_r": self.T_r,
            "H_qk": self.H_qk,
            "HV": self.HV,
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
        }

    @classmethod
    def from_tensors(
        cls,
        tensors: Sequence[torch.Tensor],
        flags: Sequence[bool],
        meta: dict[str, Any],
        ctx: Any,
    ) -> KcpSaved:
        it = iter(tensors)
        S_in = next(it)
        has_c1, has_m, has_tapes = flags
        c1_recv = next(it) if has_c1 else None
        M_seg = next(it) if has_m else None
        tapes = (
            tuple(next(it) for _ in saved_tapes(meta["save_mode"]))
            if has_tapes
            else None
        )
        return cls(
            S_in=S_in, c1_recv=c1_recv, M_seg=M_seg, tapes=tapes, ctx=ctx, **meta
        )

    def tape(self, name: str) -> torch.Tensor:
        names = saved_tapes(self.save_mode)
        if self.tapes is None or name not in names:
            raise KeyError(
                f"saved: this saved set (save_mode={self.save_mode!r}) carries no tape {name!r}; it holds {names}"
            )
        return self.tapes[names.index(name)]

    def forget(self, name: str) -> None:
        """Drop this set's reference to ``name`` (``S_in`` / ``c1_recv`` / ``M_seg``) once the backward's last reader
        of it has run (``release_in_backward``): the bytes return to the allocator before the backward's next
        allocation.  Frozen dataclass, hence ``object.__setattr__``."""
        if name not in ("S_in", "c1_recv", "M_seg"):
            raise KeyError(f"saved: {name!r} is not a saved-set tensor")
        object.__setattr__(self, name, None)

    @property
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors())


def _exchange_scheme(tb: _Tables, opts: KcpOptions) -> str:
    """``a2a`` or ``ring``: an explicit ``ring`` needs every document chain within :data:`RING_MAX_HOPS` hops and
    ``save_transfer=False`` (both enforced by the options validator and here)."""
    if opts.exchange == "ring":
        validate_ring(tb.max_hops + 1)
        if opts.save_transfer:
            raise ValueError(
                "exchange='ring' receives no c1 tensor: save_transfer=True has no ring form"
            )
        return "ring"
    return "a2a"


# ----------------------------------------------------------------------------------------------------------------
# One call (forward or backward): buffers, launches, collectives


class KcpCall:
    """The state of one forward or backward call on one rank: tables, buffers (workspace pools and the call's fresh
    tensors), tapes, operands, backend, the launch / collective counters and the issue sequence."""

    def __init__(
        self,
        *,
        ctx: Any,
        H_qk: int,
        HV: int,
        ws: KcpWorkspace,
        tb: _Tables,
        backend: Backend,
        opts: KcpOptions,
        scale: float,
        ops: dict[str, torch.Tensor],
        backward: bool,
    ) -> None:
        self.ctx, self.ws, self.tb, self.backend, self.opts = ctx, ws, tb, backend, opts
        self.H_qk, self.HV = int(H_qk), int(HV)
        self.r = validate_heads(self.H_qk, self.HV)
        self.H = self.HV  # the exchange / tile head count
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
            "locals": 0,
        }
        self.sequence: list[str] = []
        self.records: list[
            tuple[str, tuple[int, int, int]]
        ] = []  # (registry record, grid) per launch (evidence)
        self.merge_log: list[
            tuple[str, int, int, int | None]
        ] = []  # (family, ncols, n_chains, explicit cols)
        self.scratch: dict[str, Any] = {}
        self.bufs: dict[str, torch.Tensor] = {}
        self.tapes: dict[
            str, torch.Tensor
        ] = {}  # tape name -> tensor (fresh or arena), bound as the call proceeds
        self.out: torch.Tensor | None = None
        self.grads: dict[str, torch.Tensor] | None = None
        self.exchange = _exchange_scheme(tb, opts)
        self.precision_mode = int(opts.precision_mode)
        self.transfer_walk = str(opts.transfer_walk)
        self.schedule = str(opts.knob("schedule"))
        self.cols_rule = str(opts.knob("cols_rule"))
        self.debug_sequence = bool(opts.knob("debug_sequence"))
        if int(tb.fused_walks) != int(opts.knob("fused_walks")):
            raise ValueError(
                f"tables built with fused_walks={tb.fused_walks}, the call asks for {opts.knob('fused_walks')}"
            )
        p_rule = opts.knob("p_rule")
        if p_rule is not None and p_rule != getattr(ctx, "p_rule", None):
            raise ValueError(
                f"test_knobs p_rule {p_rule!r} differs from the context's plan rule {getattr(ctx, 'p_rule', None)!r}: "
                "give the rule to build_kcp_context"
            )
        if opts.segments is not None and int(opts.segments) != tb.P:
            raise ValueError(
                f"segments={opts.segments}: the plan built P={tb.P} segments per fragment (the override is bounded by "
                "the fragments' chunk counts)"
            )

    # -- buffers
    def buf(self, name: str) -> torch.Tensor:
        t = self.bufs.get(name)
        return t if t is not None else self.ws.pool(name, self.tb.sizes[name])

    def fresh(self, name: str) -> torch.Tensor:
        """A fresh per-call tensor standing in for pool ``name`` (the saved-set tensors)."""
        t = _alloc((self.tb.sizes[name],), self.state_dtype, self.device)
        self.bufs[name] = t
        return t

    def indexed(
        self, pool: str, index: Sequence[int], tiles: Sequence[Sequence[int]]
    ) -> Indexed:
        return Indexed(
            self.buf(pool),
            tuple(int(i) for i in index),
            tuple(tuple(int(t) for t in row) for row in tiles),
        )

    def tape(self, name: str) -> torch.Tensor:
        """Tape ``name`` of this call: a bound fresh / saved tensor, else the workspace arena's."""
        t = self.tapes.get(name)
        if t is None:
            t = self.tapes[name] = self.ws.tape(name)
        return t

    def release_tapes(self, names: Sequence[str], *, keep: Sequence[str] = ()) -> None:
        """Unbind the arena tapes ``names`` (never one of ``keep``: the saved tapes stay bound) and drop them from
        the arena, returning their memory to the allocator before this call's next allocation."""
        for name in names:
            if name in keep:
                continue
            self.tapes.pop(name, None)
            self.ws.release_tape(name)

    def release_pools(self, names: Sequence[str]) -> None:
        """Drop the workspace pools ``names`` (dead for the rest of this call); the fresh per-call tensors stay."""
        for name in names:
            self.ws.release_pool(name)

    def drop_bound(self, name: str, saved: KcpSaved | None = None) -> None:
        """Forget the per-call tensor bound as ``name`` (fresh or saved) once its last reader has run: the ``bufs``
        entry goes, and a saved set the backward owns (``saved.release_in_backward``) forgets its tensor too, so the
        bytes return to the allocator before the next allocation.  Under ``torch.utils.checkpoint`` (non-reentrant)
        torch's unpack hook has already handed the recomputed tensor over, so nothing else holds it; in a direct
        autograd backward the engine's saved variable keeps the data until the node returns (no effect, no harm).
        A pool of that name (a caller-held workspace) is untouched: pools go through :meth:`release_pools`."""
        self.bufs.pop(name, None)
        if (
            saved is not None
            and saved.release_in_backward
            and name in SAVED_TENSOR_OF_BUFFER
        ):
            saved.forget(SAVED_TENSOR_OF_BUFFER[name])

    def fresh_tapes(self, names: Sequence[str]) -> None:
        """Bind fresh tensors for the tapes ``names`` (the saved tapes of ``save_mode="full"``)."""
        R, NC_pad, HV, T = int(self.ctx.R), int(self.ctx.NC_pad), self.HV, self.T_r
        for name in names:
            shape, dtype = tape_spec(name, R=R, NC_pad=NC_pad, HV=HV, T=T)
            self.tapes[name] = _alloc(shape, dtype, self.device)

    def dev(
        self, key: Any, values: Sequence[int], dtype: torch.dtype = torch.int32
    ) -> torch.Tensor:
        return self.tb.dev(key, values, self.device, dtype)

    def block(self, ix: Indexed | None) -> _Block:
        """The kernel's view of an :class:`Indexed` pool; ``None`` (or no slot) = the zero state: every index ``-1``
        and a never-read base pointer."""
        nseg = int(self.tb.NSEG)
        if ix is None or ix.n_slots == 0:
            return _Block(self.buf("eye"), self.dev("neg", [-1] * nseg), self.H, 1, 1)
        base_tile, hg, n, stride = ix.addressing(self.H)
        if len(ix.index) != nseg:
            raise ValueError(
                f"Indexed: {len(ix.index)} segment entries, the plan has NSEG = {nseg}"
            )
        ptr = ix.pool.view(-1)[base_tile * TILE :] if base_tile else ix.pool.view(-1)
        return _Block(ptr, self.dev("index", ix.index), int(hg), int(n), int(stride))

    # -- launches
    def kernel(self, name: str, grid: int, **args: Any) -> None:
        if name not in PROGRAMS:
            raise ValueError(f"unknown program {name!r}")
        self.counters["launches"] += 1
        self.sequence.append(name)
        args["_call"] = self
        args["_grid"] = int(grid)
        self.backend.run(name, args)

    def _walk_common(self, walk: Sequence[int]) -> dict[str, Any]:
        tb = self.tb
        return {
            "num_heads": self.HV,
            "num_qk_heads": self.H_qk,
            "seg_chunk_start": tb.seg_table(self.device),
            "num_seqs": int(tb.NSEG),
            "walk_seg": self.dev("walk", walk),
            "chunk_bos": self.ctx.tables["chunk_bos"],
            "chunk_len": self.ctx.tables["chunk_len"],
        }

    @staticmethod
    def _block_args(prefix: str, b: _Block) -> dict[str, Any]:
        return {
            prefix: b.ptr,
            f"{prefix}_index": b.index,
            f"{prefix}_hg": b.hg,
            f"{prefix}_n": b.n,
        }

    def walk(self, name: str, walk: Sequence[int], **kw: Any) -> None:
        """A walk launch (#2 / #3 / #7) over the segments ``walk`` (nothing when the list is empty); ``kw`` carries
        the pools (``h0`` / ``out`` / ``dht`` as :class:`Indexed` or ``None``) and the program's mode scalars."""
        if len(walk) == 0:
            return
        walk = [int(s) for s in walk]
        self.counters["walks"] += 1
        args = self._walk_common(walk)
        if name in ("kcp_fwdh_offset", "kcp_fwdh_tape"):
            tape = name == "kcp_fwdh_tape"
            # The unused pool of a walk is the ZERO block (every index -1, a never-read base pointer): the tape
            # walk reads h0 and writes the tapes only; the offset walk starts from zero and writes the indexed final
            # states only.  Aliasing one block for both would start the offsets from stale staging tiles and let the
            # tape walk overwrite the carries it reads.
            h0 = self.block(kw.get("h0") if tape else None)
            out = self.block(None if tape else kw.get("out"))
            args.update(
                k=self.ops["k"],
                w=self.tape("w"),
                u=self.tape("u"),
                rowscale=self.tape("rowscale"),
                gamma=self.tape("gamma"),
                **self._block_args("h0", h0),
                **self._block_args("out", out),
                tile_stride=h0.stride if tape else out.stride,
                h_tape=self.tape("h_tape"),
                v_new=self.tape("v_new"),
            )
        elif name == "kcp_transfer_walk":
            out = self.block(kw["out"])
            args.update(
                k=self.ops["k"],
                w=self.tape("w"),
                u=self.tape("u"),
                rowscale=self.tape("rowscale"),
                gamma=self.tape("gamma"),
                **self._block_args("out", out),
                tile_stride=out.stride,
                init_mode=int(kw.get("init_mode", 0)),
                use_u=int(kw.get("use_u", 0)),
                write_transposed=int(kw.get("write_transposed", 0)),
                _precision_mode=self.precision_mode,  # static build knobs: record selection only (never kernel arguments)
                _transfer_walk=self.transfer_walk,
            )
        elif name in ("kcp_dhu_offset", "kcp_dhu_tape"):
            tape = name == "kcp_dhu_tape"
            dht = self.block(
                kw.get("dht") if tape else None
            )  # the unused pool is the zero block (see above)
            out = self.block(None if tape else kw.get("out"))
            args.update(
                q=self.ops["q"],
                k=self.ops["k"],
                w=self.tape("w"),
                do_g=self.tape("do_g"),
                dv_intra=self.tape("dv_intra"),
                rowscale=self.tape("rowscale"),
                gamma=self.tape("gamma"),
                **self._block_args("dht", dht),
                **self._block_args("out", out),
                tile_stride=dht.stride if tape else out.stride,
                # The offset program never dereferences the tape pointers (its build holds no store to them), so it
                # gets ``h_tape`` as ``dh_tape``: a valid pointer of the tape's own shape and dtype that costs
                # nothing.  Allocating ``dh_tape`` here would put it at the backward's peak -- the adjoint merge,
                # before the merged pools are dropped -- for a walk that never writes it; the first tape walk, which
                # runs after that release, allocates it (``DHU_OFFSET_TAPE_STANDIN``).
                dh_tape=self.tape("dh_tape")
                if tape
                else self.tape(DHU_OFFSET_TAPE_STANDIN),
                dv_new=self.tape("dv_new"),
            )
        else:
            raise ValueError(f"{name} is not a walk program")
        self.kernel(
            name,
            len(walk) * self.HV,
            _grid_args={"n_walk": len(walk), "num_heads": self.HV},
            _sm_count=int(getattr(self.ctx, "sm_count", 0) or 0),
            _fwdh_vh=self.opts.knob("fwdh_vh"),
            _dhu_vh=self.opts.knob("dhu_vh"),
            **args,
        )

    def merge(
        self,
        family: str,
        *,
        chains: Chains | None = None,
        spec: dict[str, Any] | None = None,
    ) -> None:
        """One merge launch of chain family ``family`` (nothing when it has no chain); ``chains`` / ``spec`` replace
        the table's family (the ring exchange's one-element chains)."""
        ch = self.tb.chains[family] if chains is None else chains
        if ch.n_chains == 0:
            return
        spec = _Tables.FAMILIES[family] if spec is None else spec
        cols = merge_cols_per_cta(ch.ncols, ch.n_chains, self.cols_rule)
        self.merge_log.append((str(family), int(ch.ncols), int(ch.n_chains), cols))
        cols_eff = (
            cols if cols is not None else getattr(self.backend, "cols_per_cta", 8)
        )
        if ch.ncols % cols_eff:
            raise ValueError(
                f"ncols {ch.ncols} is not a multiple of cols_per_cta {cols_eff}"
            )
        self.counters["merges"] += 1
        y = self.buf(spec["y"])
        m = self.buf(spec["m"])
        c = (
            self.buf(spec["c"]) if spec["c"] else y
        )  # never read when every elem_c is -1
        x0 = (
            self.buf(spec["x0"]) if spec["x0"] else y
        )  # never read when every x0_index is -1
        tag = ("chain", ch.content())
        i64 = torch.int64
        self.kernel(
            "kcp_affine_merge",
            ch.n_chains * (ch.ncols // cols_eff),
            _grid_args={"n_chains": int(ch.n_chains), "ncols": int(ch.ncols)},
            _family=str(family),
            m=m,
            c=c,
            x0=x0,
            y=y,
            chain_start=self.dev(tag + ("start",), ch.chain_start),
            elem_m=self.dev(tag + ("m",), ch.elem_m, i64),
            elem_c=self.dev(tag + ("c",), ch.elem_c, i64),
            elem_y=self.dev(tag + ("y",), ch.elem_y, i64),
            x0_index=self.dev(tag + ("x0",), ch.x0_index, i64),
            y_final=self.dev(tag + ("yf",), ch.y_final, i64),
            ncols=int(ch.ncols),
            transpose_m=int(spec["tm"]),
            write_states=int(spec["ws"]),
            cols_per_cta=cols,
        )

    def transpose_tiles(
        self, src: str, src_tiles, dst: str, dst_tiles, *, scratch: str | None = None
    ) -> None:
        """Scatter the transposes of the tiles ``src_tiles[slot][h]`` of pool ``src`` into ``dst_tiles[slot][h]`` of
        pool ``dst`` (``M_f`` -> ``M_f^T``, bitwise); nothing when there is no tile.  ``scratch``: a pool that holds
        nothing live at this moment and has room for the gathered tiles (the carry plane during the summaries); the
        gather lands there instead of in a fresh temporary of ``n`` tiles (R2-L1 1a: the forward's peak moment)."""
        n = sum(len(row) for row in src_tiles)
        if n == 0:
            return
        self.counters["launches"] += 1
        self.sequence.append("kcp_transpose_tiles")
        scratch_t = None
        if scratch is not None and self.tb.sizes.get(scratch, 0) >= n * TILE:
            scratch_t = self.buf(scratch)[: n * TILE]
        self.backend.transpose_tiles(
            src=self.buf(src),
            src_tiles=self.tb.device_tiles(src_tiles, self.device),
            dst=self.buf(dst),
            dst_tiles=self.tb.device_tiles(dst_tiles, self.device),
            scratch=scratch_t,
        )

    def _local_common(self) -> dict[str, Any]:
        ctx = self.ctx
        return {
            "num_heads": self.HV,
            "num_qk_heads": self.H_qk,
            "chunk_bos": ctx.tables["chunk_bos"],
            "chunk_len": ctx.tables["chunk_len"],
            "seq_chunk_start": ctx.tables["seq_chunk_start"],
            "num_chunks": int(ctx.NC),
            "num_seqs": int(
                ctx.n_frag
            ),  # K4: seq_chunk_start[num_seqs] = the real chunk count on the device
        }

    def local(self, name: str) -> None:
        """A chunk-local program (#1, #5, #6, #8, #9) over this rank's tokens (skipped without tokens)."""
        if self.T_r == 0:
            return
        if name not in LOCAL_PROGRAMS:
            raise ValueError(f"{name} is not a local program")
        self.counters["locals"] += 1
        o, NC = self.ops, int(self.ctx.NC)
        args = self._local_common()
        if name == "fwd_prep":
            args.update(
                q=o["q"],
                k=o["k"],
                v=o["v"],
                g=o["g"],
                beta=o["beta"],
                scale=self.scale,
                w=self.tape("w"),
                u=self.tape("u"),
                aqk=self.tape("aqk"),
                akk=self.tape("akk"),
                g_cum=self.tape("g_cum"),
                rowscale=self.tape("rowscale"),
                qscale=self.tape("qscale"),
                gamma=self.tape("gamma"),
            )
            grid = NC * self.H_qk
        elif name == "fwd_o":
            args.update(
                q=o["q"],
                h_tape=self.tape("h_tape"),
                v_new=self.tape("v_new"),
                aqk=self.tape("aqk"),
                qscale=self.tape("qscale"),
                o=self.out,
                ticket=self.ws.ticket(
                    name
                ),  # the persistent program's self-resetting ticket slot
            )
            grid = NC * self.HV
        elif name == "bwd_prep":
            args.pop("num_qk_heads")
            args.update(
                do=o["dout"],
                aqk=self.tape("aqk"),
                qscale=self.tape("qscale"),
                g_cum=self.tape(
                    "g_cum"
                ),  # K4 provisional #6: qscale = scale * e^{G_t} recomputed from g_cum
                scale=self.scale,
                do_tstride=int(o["dout"].stride(0)),
                do_g=self.tape("do_g"),
                dv_intra=self.tape("dv_intra"),
                ticket=self.ws.ticket(name),
            )
            grid = NC * self.HV
        elif name == "bwd_dqkg":
            host = self.opts.group_reduce == "host"
            args.update(
                q=o["q"],
                k=o["k"],
                v=o["v"],
                do=o["dout"],
                h_tape=self.tape("h_tape"),
                dh_tape=self.tape("dh_tape"),
                v_new=self.tape("v_new"),
                dv_new=self.tape("dv_new"),
                w=self.tape("w"),  # the #8 read set of the dispatch table
                u=self.tape("u"),
                aqk=self.tape("aqk"),
                akk=self.tape("akk"),
                do_g=self.tape("do_g"),
                beta=o["beta"],
                g_cum=self.tape("g_cum"),
                rowscale=self.tape("rowscale"),
                qscale=self.tape("qscale"),
                gamma=self.tape("gamma"),
                scale=self.scale,
                dq=self.grads["dq"],
                dk=self.grads["dk"],
                dv=self.grads["dv"],
                dbeta=self.grads["dbeta"],
                dg_part=self.tape("dg_part"),
                gamma_dot=self.tape("gamma_dot"),
                group_reduce=0
                if host
                else 1,  # K4: i32, 1 = fused bf16 dq / dk, 0 = FP32 per-head partials
                dq_part=self.tape("dq_part")
                if host
                else self.buf(
                    "eye"
                ),  # K4 ABI: written iff group_reduce == 0; a never-written pointer otherwise
                dk_part=self.tape("dk_part") if host else self.buf("eye"),
                dqk_tstride=int(
                    self.grads["dq"].stride(0)
                ),  # K4: output token strides in elements
                dv_tstride=int(self.grads["dv"].stride(0)),
                ticket=self.ws.ticket(name),
            )
            grid = NC * self.H_qk
        else:  # gate_epilogue
            args.pop("num_qk_heads")
            args.pop("seq_chunk_start")
            args.update(
                dg_part=self.tape("dg_part"),
                gamma_dot=self.tape("gamma_dot"),
                dg=self.grads["dg"],
            )
            grid = NC * self.HV
        self.kernel(
            name,
            grid,
            _grid_args={
                "num_chunks": NC,
                "num_heads": self.HV,
                "num_qk_heads": self.H_qk,
                "sm_count": int(
                    getattr(self.ctx, "sm_count", 0) or 0
                ),  # persistent grids (#5 / #6 / #8): min(tiles, SMs)
            },
            **args,
        )

    def host_group_reduce(self) -> None:
        """``group_reduce="host"``: the ``r`` value heads' FP32 ``dq`` / ``dk`` partials summed in a fixed order
        (head ``0``, then ``+1``, ...) and rounded once to BF16 into the outputs."""
        if self.T_r == 0:
            return
        self.counters["launches"] += 1
        self.sequence.append("host_group_reduce")
        for part, name in (("dq_part", "dq"), ("dk_part", "dk")):
            p = self.tape(part).view(self.T_r, self.H_qk, self.r, D)
            acc = p[:, :, 0]
            for j in range(1, self.r):
                acc = acc + p[:, :, j]
            self.grads[name].copy_(acc.to(self.grads[name].dtype))

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
        peer on the caller's group; returns the work handle (``None`` on a single rank)."""
        send_splits, recv_splits = (
            [int(x) for x in send_splits],
            [int(x) for x in recv_splits],
        )
        if len(send_splits) != self.world or len(recv_splits) != self.world:
            raise ValueError(
                f"{tag}: split lists must have one entry per rank ({self.world})"
            )
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
            self.debug_sequence
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


def gather_states(pool: torch.Tensor, tiles: Sequence[Sequence[int]]) -> torch.Tensor:
    """FP32 ``[n_slots, HV, 128, 128]`` copy of the tiles ``tiles[slot][h]`` of a flat state pool (``S_in`` of the
    saved set with ``ctx.heads(HV).h0_tile``: slot = ``in_slot`` of the crossing-in fragment in local order; ``dS_out``
    with ``dht_tile``: slot = ``out_slot``).  A device-side gather; the host reads nothing."""
    n = len(tiles)
    if n == 0:
        return pool.new_zeros((0, 1, D, D))
    H = len(tiles[0])
    idx = torch.tensor(
        [int(t) for row in tiles for t in row], dtype=torch.int64, device=pool.device
    )
    return pool.view(-1, D, D)[idx].view(n, H, D, D).clone()


def saved_states(saved: KcpSaved) -> torch.Tensor:
    """The saved set's ``S_in`` as FP32 ``[n_in, HV, 128, 128]`` in ``in_slot`` order (the oracle form)."""
    return gather_states(saved.S_in, saved.ctx.heads(saved.HV).h0_tile)


def _record(kind: str, call: KcpCall) -> None:
    """The launch inventory of the last forward / backward (evidence)."""
    LAST_CALL[kind] = {
        **{f"n_{k}": v for k, v in call.counters.items()},
        "sequence": list(call.sequence),
        "exchange": call.exchange,
        "backend": call.backend.name,
        "backend_path": call.backend.path,
        "P": call.tb.P,
        "NSEG": call.tb.NSEG,
        "Hg": call.tb.Hg,
        "Hg_r": call.tb.Hg_r,
        "n_in": call.tb.n_in,
        "n_out": call.tb.n_out,
        "H_qk": call.H_qk,
        "HV": call.HV,
        "T_r": call.T_r,
        "workspace_bytes": call.ws.bytes,
        "pool_bytes": call.ws.pool_bytes,
        "tape_bytes": call.ws.tape_bytes,
        "fresh_bytes": {k: t.numel() * t.element_size() for k, t in call.bufs.items()},
        "precision_mode": call.precision_mode,
        "transfer_walk": call.transfer_walk,
        "offset_class": call.opts.offset_class,
        "save_mode": call.opts.save_mode,
        "save_transfer": call.opts.save_transfer,
        "group_reduce": call.opts.group_reduce,
        "layout": call.opts.layout,
        "schedule": call.schedule,
        "fused_walks": call.tb.fused_walks,
        "cols_rule": call.cols_rule,
        "merges": list(call.merge_log),
        "graph": call.opts.graph,
        "graph_event": "eager",
        "graph_key": None,
        "graph_fallback": None,
        "states": call.scratch.get(
            "states"
        ),  # test knob record_states: {"S_in" | "dS_out": [n, HV, 128, 128]}
        "records": [
            (r, list(g)) for r, g in call.records
        ],  # (registry record, grid) per launch; #2's vh form shows here
        "tapes_source": call.scratch.get(
            "tapes_source"
        ),  # backward: "saved" | "kept" | "replay" | "recompute" -- where the forward tapes came from
        "hand_off": bool(
            call.scratch.get("hand_off", False)
        ),  # forward: its tapes were kept for the backward of this graph task
        "fwd_o": call.scratch.get(
            "fwd_o"
        ),  # forward: "launched" | "skipped" (recompute_output="auto": the recompute's dead #5)
        "fwd_o_reason": call.scratch.get(
            "fwd_o_reason"
        ),  # forward: "always" | "dead" | "not_executable" | why the frame proof did not hold (the decision, recorded on every rank)
        "hand_off_reason": call.scratch.get(
            "hand_off_reason"
        ),  # forward: "always" (True) | "off" | "not_executable" | the frame reason under "auto" ("dead" = kept)
    }


# ----------------------------------------------------------------------------------------------------------------
# Ring exchange (``exchange="ring"``): per-document chains of point-to-point hops on ``ctx.group`` instead of the
# owner merge + c1 / c2 (b1 / b2).  Hop ``h`` of the forward: every crossing-out fragment at chain position ``h - 1``
# sends its exit state ``S_out = M_f S_in + B_f`` (hop 1: ``B_f`` itself) to the fragment at position ``h``; the
# receiver writes it into its ``S_in`` tiles, so the tape walks after the hops read exactly what the a2a path's c2
# delivers.  The backward runs the hops from the document end: the fragment at position ``n_d - h`` sends
# ``dS_in = M_f^T dS_out + d_f`` (hop 1: ``d_f``) to its predecessor as that fragment's ``dS_out`` (into the
# ``b2_recv`` tiles).  One ``batch_isend_irecv`` per hop in ``(doc, slot)`` order.


class _Ring:
    """Per-hop send / receive lists of one rank (positions from ``ctx.chains``), built once per ``_Tables``."""

    def __init__(self, tb: _Tables) -> None:
        ctx = tb.ctx
        rank = int(ctx.rank)
        self.hops = tb.max_hops
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
    call: KcpCall, tb: _Tables, src, src_tiles, slot_src: int, dst, slot_dst: int
) -> None:
    """``dst`` ring block ``slot_dst`` <- the ``H`` tiles ``src_tiles[slot_src][h]`` of pool ``src`` (bitwise copy)."""
    idx = tb.device_tiles([src_tiles[slot_src]], call.device)
    _ring_block(dst, slot_dst, call.H).copy_(src.view(-1, D, D)[idx.view(-1)])


def _ring_scatter(
    call: KcpCall, tb: _Tables, src, slot_src: int, dst, dst_tiles, slot_dst: int
) -> None:
    """Pool ``dst`` tiles ``dst_tiles[slot_dst][h]`` <- the ring block ``slot_src`` of ``src`` (bitwise copy)."""
    idx = tb.device_tiles([dst_tiles[slot_dst]], call.device)
    dst.view(-1, D, D)[idx.view(-1)] = _ring_block(src, slot_src, call.H)


def _ring_p2p(call: KcpCall, hop: int, sends, recvs):
    """One ``batch_isend_irecv`` of the hop (sends then receives, each list in ``(doc, slot)`` order); returns the
    works (empty when the rank has nothing to exchange)."""
    call.counters["collectives"] += 1
    call._sequence(f"hop{hop}", kind="ring")
    if call.world == 1 or not dist.is_initialized():
        if sends:
            raise RuntimeError("ring exchange on a single rank")
        return []
    ops = [dist.P2POp(dist.isend, t, peer, group=call.group) for t, peer in sends]
    ops += [dist.P2POp(dist.irecv, t, peer, group=call.group) for t, peer in recvs]
    return dist.batch_isend_irecv(ops) if ops else []


def _ring_wait(works) -> None:
    for w in works:
        w.wait()


def _ring_chain(elems: list[tuple[int, int, int, int]]) -> Chains:
    """One-element chains per (fragment, head) from tile numbers ``(x0, m, c, y)`` (one state step on one fragment)."""
    ch = Chains(ncols=D)
    for x0, m, c, y in elems:
        ch.add([(m * TILE, c * TILE, -1)], x0=x0 * TILE, y_final=y * TILE)
    return ch


def _ring_forward(call: KcpCall) -> None:
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
        else:  # S_out = M_f S_in + B_f: one merge element per sender (forward family: c1 tiles hold M_f^T)
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
                    chains=_ring_chain(elems),
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
            carry = (
                call.indexed("carry", tb.carry_index, tb.carry_tiles)
                if tb.P > 1
                else None
            )
            call.walk(
                "kcp_fwdh_tape", tb.walk_start, h0=carry
            )  # document starts: under hop 1
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


def _ring_backward(call: KcpCall) -> None:
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
                    chains=_ring_chain(elems),
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
            dcarry = (
                call.indexed("carry", tb.dcarry_index, tb.carry_tiles)
                if tb.P > 1
                else None
            )
            if tb.P > 1:
                call.merge("dcarry_end")
            call.walk(
                "kcp_dhu_tape", tb.walk_end, dht=dcarry
            )  # document ends: under hop 1
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


# ----------------------------------------------------------------------------------------------------------------
# Launch sequences (one compute stream; waits at first consumption; shipped schedule wait_first + fused tape walks)


def _offset_walk(call: KcpCall, walk: Sequence[int], out: Indexed) -> None:
    """The offset ``B_f`` / ``b_seg`` of the segments ``walk`` from the zero state into ``out``: the #2 offset walk
    from zero with the local rounding points (``offset_class="fla"``, the default) or the #3 transfer walk
    with ``init_mode = 1`` / ``use_u = 1`` (``offset_class="fp32"``, the FP32-class knob, ``B_f`` only)."""
    if call.opts.offset_class == "fp32":
        call.walk(
            "kcp_transfer_walk",
            walk,
            out=out,
            init_mode=1,
            use_u=1,
            write_transposed=1,  # the offset is a STATE tile (B as is); write_transposed=0 would store B^T like a map
        )
    else:
        call.walk("kcp_fwdh_offset", walk, h0=None, out=out)


def _summaries(call: KcpCall) -> None:
    """The fragment summaries ``[B_f | M_f^T]`` into the c1 send layout: at ``P = 1`` the offset walk and the
    transfer walk write the crossing-out fragments' tiles directly; at ``P > 1`` both walk every segment into the
    staging planes and the merge composes the crossing-out fragments' segments (forward family -> ``B_f``, map
    family -> ``M_f``, transposed into the pairs)."""
    tb = call.tb
    if tb.P == 1:
        _offset_walk(
            call, tb.walk_offset, call.indexed("c1_send", tb.out_index, tb.c1_b_tiles)
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
    _offset_walk(
        call, tb.walk_offset, call.indexed("stage_b", tb.seg_index, tb.stage_tiles)
    )
    call.walk(
        "kcp_transfer_walk",
        tb.walk_offset,
        out=call.indexed("stage_m", tb.seg_index, tb.stage_tiles),
        init_mode=0,
        use_u=0,
        write_transposed=0,
    )
    call.merge(
        "map"
    )  # B_f (from 0) and M_f (from I) of every crossing-out fragment, forward family
    call.transpose_tiles(
        "c1_send", tb.mf_tiles, "c1_send", tb.c1_m_tiles, scratch="carry"
    )  # M_f^T into the c1 pairs (bitwise); the gather lands in the carry plane, which nothing has written yet


def _fwd_tapes(call: KcpCall, *, carry: Indexed | None, wait_first: bool) -> None:
    """The forward tape walk(s) after ``S_in`` is complete.  ``fused_walks``: one launch from one pool -- the carry
    plane at ``P > 1`` (``h0_index_fused``: carries, the ``S_in`` copies of ``carry_in``, ``-1`` zero states),
    ``S_in`` itself at ``P = 1`` -- over every live segment (``wait_first``) or the crossing-in fragments' segments
    (the document-start walk ran under c1); otherwise the three separate launches."""
    tb = call.tb
    h0_in = call.indexed("S_in", tb.h0_index, tb.s_in_tiles)
    if tb.fused_walks:
        h0 = (
            call.indexed("carry", tb.h0_index_fused, tb.carry_tiles)
            if tb.P > 1
            else h0_in
        )
        call.walk(
            "kcp_fwdh_tape", tb.walk_live if wait_first else tb.walk_in_fused, h0=h0
        )
        return
    if wait_first:
        call.walk(
            "kcp_fwdh_tape", tb.walk_start, h0=carry
        )  # document starts: after wait(c2)
    call.walk("kcp_fwdh_tape", tb.walk_in_p0, h0=h0_in)
    if tb.P > 1:
        call.walk("kcp_fwdh_tape", tb.walk_in_p1, h0=carry)


def _adj_tapes(call: KcpCall, *, dcarry: Indexed | None, wait_first: bool) -> None:
    """The adjoint tape walk(s) after ``dS_out`` is complete (the mirror of :func:`_fwd_tapes`)."""
    tb = call.tb
    if tb.fused_walks:
        # at P > 1 the fused walk reads the carry plane only (b2_recv was released before this walk, R2-L1 1b):
        # the b2_recv view is built only where the walk reads it
        dht = (
            call.indexed("carry", tb.dht_index_fused, tb.carry_tiles)
            if tb.P > 1
            else call.indexed("b2_recv", tb.dht_index, tb.ds_out_tiles)
        )
        call.walk(
            "kcp_dhu_tape", tb.walk_live if wait_first else tb.walk_cross_fused, dht=dht
        )
        return
    dht_cross = call.indexed("b2_recv", tb.dht_index, tb.ds_out_tiles)
    if wait_first:
        call.walk(
            "kcp_dhu_tape", tb.walk_end, dht=dcarry
        )  # document ends: after wait(b2)
    call.walk("kcp_dhu_tape", tb.walk_cross_last, dht=dht_cross)
    if tb.P > 1:
        call.walk("kcp_dhu_tape", tb.walk_cross_rest, dht=dcarry)


def _forward(
    call: KcpCall, *, save: bool, replay: bool, hand_off: bool = False
) -> None:
    """The forward sequence.  ``save``: the saved set is kept (fresh ``S_in``; with ``save_transfer`` also the c1
    receive tensor and ``M_seg``; with ``save_mode="full"`` fresh forward tapes); ``replay``: the forward of a
    replay-mode backward (no ``fwd_o``, nothing kept); ``hand_off``: the forward tapes go to the backward of this
    graph task (``reuse_recompute_tapes``): no tape is dropped and the tape walk takes the light replay's form
    (:func:`_replay_tape_walk`).  ``schedule="wait_first"``: ``c1 -> wait -> fwd merge -> c2 ->
    wait -> carry merges -> tape walk(s)`` -- no merge or walk is issued while a collective is in flight; the merges
    read only ``c1_recv`` / the staging planes and the tapes read no collective buffer, so every published bit is the
    same under either schedule."""
    tb, opts = call.tb, call.opts
    # before the first launch: the backend record reads it
    call.scratch["hand_off"] = bool(hand_off)
    wait_first = call.schedule == "wait_first"
    started_early = False  # the document-start tape walk issued under c1 (overlap schedule, a2a exchange only)
    merged_start = (
        False  # the carry_start merge issued under c1 / the ring (overlap schedule)
    )
    keep = save and not replay
    keep_transfer = opts.save_transfer and keep
    if keep_transfer and tb.P > 1:
        call.fresh(
            "stage_m"
        )  # M_seg^T of the segmented fragments: the summaries write it
    if (
        keep
    ):  # the saved tapes are fresh per call (light: none; full: every forward tape)
        call.fresh_tapes(saved_tapes(opts.save_mode))
    call.local("fwd_prep")
    if not replay and opts.graph == "off" and not hand_off:
        # #1's backward-only outputs leave the arena now (FWD_DEAD_AFTER_PREP); a saved copy (save_mode="full") is a
        # fresh tensor bound in call.tapes and stays.  A replay keeps them for #6 / #8; so does a hand-off forward.
        call.release_tapes(
            FWD_DEAD_AFTER_PREP, keep=saved_tapes(opts.save_mode) if keep else ()
        )
    _summaries(call)
    # the c1 receive tensor and S_in (the c2 receive buffer) are first written by their collectives: allocated after
    # the summaries' walks (the forward's peak moment), not at the top of the sequence where they would sit unwritten
    # through the peak (a D-layout rank at CP8 / CP16 holds 100+ MiB of them); S_in goes with its branch so that the
    # a2a form allocates it only once the c1 send pool is released (the c1 exchange is the next peak moment)
    if keep_transfer:
        call.fresh("c1_recv")
    carry = call.indexed("carry", tb.carry_index, tb.carry_tiles) if tb.P > 1 else None
    if (
        tb.N_out == 0
    ):  # no crossing fragment anywhere in the group (global metadata): nothing to exchange or merge
        call.fresh("S_in")
    elif (
        call.exchange == "ring"
    ):  # per-document hops; S_in filled as c2 would (bitwise)
        call.fresh("S_in")
        if tb.P > 1 and not wait_first:
            call.merge("carry_start")
            merged_start = True
        _ring_forward(call)
    else:
        w1 = call.a2a("c1", "c1_send", "c1_recv", *tb.split_elems("c1"))
        if not wait_first:
            if tb.P > 1:
                call.merge(
                    "carry_start"
                )  # reads the staging planes only (no collective buffer)
                merged_start = True
            # a hand-off forward walks every tape in the replay's single form below
            if not hand_off:
                call.walk(
                    "kcp_fwdh_tape", tb.walk_start, h0=carry
                )  # document starts: under c1
                started_early = True
        call.wait(w1)
        w1 = None
        if (
            opts.graph == "off"
        ):  # c1 is delivered; nothing reads c1_send after it (the fwd merge reads c1_recv)
            call.release_pools(("c1_send",))
        # the c2 receive buffer: first written by wait(c2); allocated after the c1 send pool is gone (eager), so the
        # two never overlap
        call.fresh("S_in")
        if not call.ws.has("c2_send"):
            # the eager a2a forward leaves c2_send unsized (``_prepare``): the fwd merge is its first writer, so the
            # plane exists from here to wait(c2) only, not across the whole forward
            call.fresh("c2_send")
        call.merge("fwd")  # the owner's chains: S_in slices into the c2 send layout
        w2 = call.a2a("c2", "c2_send", "S_in", *tb.split_elems("c2"))
        call.wait(w2)
        w2 = None
        # sent: a fresh plane goes now (a workspace pool stays until FWD_DEAD_POOLS)
        call.drop_bound("c2_send")
    if tb.P > 1:
        if not merged_start:
            call.merge("carry_start")
        call.merge(
            "carry_in"
        )  # with fused_walks also the S_in copies into the extra carry slots (write_states)
    if hand_off:
        # the tapes handed to the backward are produced with the light replay's own tape-walk form (one launch list,
        # one h0 table), so the kept tapes are the replay's by construction: at P > 1 under fused walks that is the
        # forward's own fused walk over every live segment from the carry plane; at P = 1 the replay's walk over
        # every segment from S_in (the segments of zero-length fragments write nothing)
        _replay_tape_walk(call)
    else:
        _fwd_tapes(
            call, carry=carry, wait_first=not started_early
        )  # no exchange (N_out = 0) or a ring: every walk runs here, whatever the schedule
    if call.opts.knob("record_states"):
        call.scratch["states"] = {
            "S_in": gather_states(call.buf("S_in"), tb.s_in_tiles)
        }
    # #5 is launched by _fwd_body (never in a replay; not by a recompute whose output is provably dead) once this
    # frame and its views of the pools are gone


def _replay_tapes(call: KcpCall, saved: KcpSaved) -> None:
    """``save_mode="light"`` backward: #1 again and the #2 tape walk from the saved ``S_in`` (segment-start states
    from ``M_seg`` / the recomputed offsets with ``(P - 1)`` state-mode steps); no exchange, no transfer walk over the
    crossing-out fragments, no merge of the owners, no #5."""
    tb = call.tb
    call.local("fwd_prep")
    if (
        tb.P > 1
    ):  # the forward carries again: b_seg (and M_seg^T unless saved) -> carries from 0 / S_in
        _offset_walk(
            call, tb.walk_all, call.indexed("stage_b", tb.seg_index, tb.stage_tiles)
        )
        if saved.M_seg is None:
            call.walk(
                "kcp_transfer_walk",
                tb.walk_all,
                out=call.indexed("stage_m", tb.seg_index, tb.stage_tiles),
                init_mode=0,
                use_u=0,
                write_transposed=0,
            )
        call.merge("carry_start")
        call.merge("carry_in")
    _replay_tape_walk(call)


def _replay_tape_walk(call: KcpCall) -> None:
    """The #2 tape walk(s) of the light replay once ``S_in`` (and at ``P > 1`` the carry plane) is in place: one
    fused launch over every live segment from the carry plane at ``P > 1``, else the segment-0 walk over every segment
    from ``S_in`` (plus the ``p > 0`` segments from the carries without fused walks).  Also the tape walk of a
    hand-off forward (``reuse_recompute_tapes``), so the kept tapes are this launch list."""
    tb = call.tb
    if (
        tb.fused_walks and tb.P > 1
    ):  # the whole tape replay from the carry plane in one launch
        call.walk(
            "kcp_fwdh_tape",
            tb.walk_live,
            h0=call.indexed("carry", tb.h0_index_fused, tb.carry_tiles),
        )
    else:
        p0 = [s for s in tb.walk_all if s % tb.P == 0]
        call.walk(
            "kcp_fwdh_tape", p0, h0=call.indexed("S_in", tb.h0_index, tb.s_in_tiles)
        )
        if tb.P > 1:
            call.walk(
                "kcp_fwdh_tape",
                [s for s in tb.walk_all if s % tb.P != 0],
                h0=call.indexed("carry", tb.carry_index, tb.carry_tiles),
            )


def _kept_release(call: KcpCall, kept: _KeptTapes) -> None:
    """The kept tapes leave the backward after #8, their last reader: unbound from the call; the tapes an eager
    forward leaves in the arena (:data:`FWD_TAPES_ARENA_AFTER_FWD`) return to this call's workspace when it holds
    none of that name at the same geometry -- a caller-held workspace ends a kept backward as a forward leaves it,
    so the next checkpointed forward allocates nothing but its output (#7039, caller-workspace form) -- and the
    other six are freed here, with the handed-over ``stage_m`` plane."""
    for name in FWD_TAPES:
        t = kept.tapes.get(name)
        call.tapes.pop(name, None)
        if (
            t is not None
            and name in FWD_TAPES_ARENA_AFTER_FWD
            and call.ws.geometry == kept.geometry
        ):
            call.ws.adopt_tape(name, t)
    kept.clear()
    KEPT_STATS["released"] += 1


def _backward(call: KcpCall, saved: KcpSaved | None) -> None:
    """The backward sequence.  ``saved=None`` is the replay mode (the forward sequence without ``fwd_o`` recomputes
    ``S_in``, the tapes and the owner's c1 receive tensor); otherwise the tapes come from the saved set
    (``save_mode="full"``) or from #1 + the #2 tape walk over the saved ``S_in`` (``"light"``), and the owner's
    adjoint reads ``M_f`` from b1 (``save_transfer=False``) or ``M_f^T`` from the saved c1 receive tensor.
    ``schedule="wait_first"``: ``b1 -> wait -> adjoint merge -> b2 -> wait -> dcarry merges -> adjoint tape walk(s)``.
    Eager path: the saved-set tensors are forgotten at their last reader (``c1_recv`` after the adjoint merge,
    ``M_seg`` before the adjoint tape walk, ``S_in`` before the gradient outputs) when the backward owns the set
    (:class:`KcpSaved` ``release_in_backward``); #8 reads the tapes only, so none of them reaches the backward's peak."""
    tb = call.tb
    wait_first = call.schedule == "wait_first"
    ended_early = False  # the document-end adjoint walk issued under b2 (overlap schedule, a2a exchange only)
    merged_end = False  # the dcarry_end merge issued under b2 (overlap schedule)
    replay = saved is None
    if replay:
        _forward(call, save=False, replay=True)
        m_source = "c1"
        call.scratch["tapes_source"] = "recompute"
    else:
        call.bufs["S_in"] = saved.S_in
        m_source = "b1"
        if saved.c1_recv is not None:
            call.bufs["c1_recv"] = saved.c1_recv
            m_source = "c1"
        if saved.M_seg is not None:
            call.bufs["stage_m"] = saved.M_seg
        kept: _KeptTapes | None = saved.kept
        if (
            saved.save_mode == "full"
        ):  # every forward tape as saved (None only at T_r = 0)
            if saved.tapes is not None:
                for name, t in zip(FWD_TAPES, saved.tapes, strict=True):
                    call.tapes[name] = t
            call.scratch["tapes_source"] = "saved"
        elif kept is not None:
            # the recompute forward's tapes of this graph task (reuse_recompute_tapes): the full-mode form on them;
            # the handed-over stage_m plane stands in for the saved M_seg^T when the saved set carries none
            for name in FWD_TAPES:
                call.tapes[name] = kept.tapes[name]
            if kept.stage_m is not None:
                call.bufs["stage_m"] = kept.stage_m
            call.scratch["tapes_source"] = "kept"
        else:  # light replay: #1 again (gamma / g_cum / akk included) + the #2 tape walk from the saved S_in
            _replay_tapes(call, saved)
            call.scratch["tapes_source"] = "replay"
    call.local("bwd_prep")
    ring = call.exchange == "ring"
    form = (
        "dm" if (m_source == "b1" or ring) else "d"
    )  # the ring's adjoint step needs M_f on the fragment's rank
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
    wb1 = wb2 = (
        None  # the collective handles: dropped after their wait so the pools can go
    )
    eager = call.opts.graph == "off"
    if tb.N_out == 0:  # no crossing fragment anywhere in the group: no adjoint exchange
        pass
    elif (
        ring
    ):  # adjoint hops from the document end; b2_recv filled as b2 would (bitwise)
        _ring_backward(call)
    else:
        wb1 = call.a2a("b1", "b1_send", "b1_recv", *tb.split_elems(f"b1_{form}"))
        call.wait(wb1)
        wb1 = None
        call.merge(
            "adj_b1" if m_source == "b1" else "adj_c1"
        )  # dS_out slices into the b2 send layout
        if eager:  # the backward's peak follows (the early document-end walk under b2): the merged pools go first
            call.release_pools(BWD_DEAD_AFTER_ADJ_MERGE)
            # the saved c1 receive tensor: the adjoint merge was its last reader
            call.drop_bound("c1_recv", saved)
        wb2 = call.a2a("b2", "b2_send", "b2_recv", *tb.split_elems("b2"))
        if not wait_first:
            if tb.P > 1:
                call.merge("dcarry_end")  # reads the staging planes only
                merged_end = True
            call.walk(
                "kcp_dhu_tape", tb.walk_end, dht=dcarry
            )  # document ends: under b2
            ended_early = True
        call.wait(wb2)
        wb2 = None
    if tb.P > 1:
        if not merged_end:
            call.merge("dcarry_end")
        call.merge(
            "dcarry_cross"
        )  # with fused_walks also the dS_out copies into the extra carry slots
    if eager:  # b2_send is sent; whatever the a2a branch did not drop (ring / no exchange) goes before the tape walk
        call.release_pools(BWD_DEAD_BEFORE_TAPES)
        # ring / no-exchange forms: nothing after this point reads the saved c1 receive tensor; the saved M_seg^T
        # plane: the dcarry merges were its last readers
        call.drop_bound("c1_recv", saved)
        call.drop_bound("stage_m", saved)
        if tb.fused_walks and tb.P > 1:
            # the fused adjoint walk reads the carry plane only (dcarry_cross copied dS_out into its extra slots):
            # b2_recv is dead before the walk that allocates dh_tape -- the backward's peak moment (R2-L1 1b)
            call.release_pools(("b2_recv",))
    _adj_tapes(
        call, dcarry=dcarry, wait_first=not ended_early
    )  # no exchange (N_out = 0) or a ring: every walk runs here, whatever the schedule
    if call.opts.knob("record_states"):
        states = call.scratch.setdefault("states", {})
        states["dS_out"] = gather_states(call.buf("b2_recv"), tb.ds_out_tiles)
        states["S_in"] = gather_states(call.buf("S_in"), tb.s_in_tiles)
    if eager:  # the walk is done: its adjoint sources go before #8 (which only reads the tapes)
        dcarry = None
        call.release_pools(BWD_DEAD_BEFORE_DQKG)
        # the tape walks (and record_states) were S_in's last readers; #8 reads h_tape
        call.drop_bound("S_in", saved)
    _alloc_grads(
        call
    )  # the outputs appear only now: #8 / #9 are their first and only writers
    call.local("bwd_dqkg")
    if call.opts.group_reduce == "host":
        call.host_group_reduce()
    if saved is not None and saved.kept is not None:
        _kept_release(call, saved.kept)  # #8 was the last reader of the forward tapes
    call.local("gate_epilogue")


# ----------------------------------------------------------------------------------------------------------------
# CUDA-graph replay of a call (keyword ``graph="replay"``; shipped default ``"off"``)
#
# The first call of a validity key (kind; the context, workspace and backend objects; every option; scale; the save
# flags; and the shape / stride / dtype / data pointer of every operand and saved tensor) runs eagerly.  The second
# call is captured into a ``torch.cuda.CUDAGraph`` and the graph is replayed once; every later call of the key replays
# it: the same kernels and collectives with the same arguments in the same order (bitwise the eager result) with no
# host work between them.  The graph bakes the addresses of the workspace arena (pools and tapes), so the capture
# records the arena's signature (name, pointer, size of every buffer) and a replay requires the same signature: an
# arena buffer dropped or re-allocated since (an eager ``graph="off"`` call releases its dead pools and tapes) resets
# the entry, which then runs eagerly once and is captured again.  Any other key runs eagerly (silent fallback) and
# starts its own entry; ``exchange="ring"``, ``debug_sequence`` and a call without a caller-owned workspace are never
# captured.  A replayed call returns the graph's static tensors (detached): the next replay of the same key overwrites
# them, so a framework must keep its inputs pointer-stable and consume the outputs and the saved set before the next
# call of the same layer.  ``graph_cache_clear()`` releases the graphs and must run before
# ``torch.distributed.destroy_process_group()``.

GRAPH_STATS: dict[str, int] = {
    "eager": 0,
    "capture": 0,
    "replay": 0,
    "fallback": 0,
    "reset": 0,
}
GRAPH_CACHE_MAX = 64  # keys kept per process; the oldest entry goes when a new key arrives beyond this


class _GraphEntry:
    """One validity key: its graph (after the capture), the static results, the launch inventory of the capture and
    the arena signature the graph baked."""

    __slots__ = (
        "key",
        "pins",
        "seen",
        "graph",
        "result",
        "record",
        "disabled",
        "digest",
        "arena",
        "reset",
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
        self.arena: tuple | None = None  # the workspace arena signature at the capture
        self.reset: str | None = (
            None  # why the last replay candidate ran eagerly instead
        )

    def replayable(self, ws: Any) -> bool:
        """The graph exists and the arena it baked is intact; otherwise the entry is reset (an eager call follows,
        then a new capture)."""
        if self.graph is None:
            return False
        if self.arena == _arena_sig(ws):
            return True
        self.graph = self.result = self.record = self.arena = None
        self.seen = 0
        self.reset = "arena"
        GRAPH_STATS["reset"] += 1
        return False


_GRAPHS: dict[tuple, _GraphEntry] = {}


def _sig(t: Any) -> tuple:
    return (
        (t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype), t.device.index)
        if isinstance(t, torch.Tensor)
        else ("none",)
    )


def _arena_sig(ws: Any) -> tuple:
    """(name, pointer, size) of every pool and tape buffer of the workspace arena, in name order, and of the ticket
    slots (one buffer; a capture bakes their addresses like the pools')."""
    tickets = getattr(ws, "_tickets", None)
    return (
        tuple(sorted((n, t.data_ptr(), t.numel()) for n, t in ws._pools.items())),
        tuple(sorted((n, t.data_ptr(), t.numel()) for n, t in ws._tapes.items())),
        () if tickets is None else (("tickets", tickets.data_ptr(), tickets.numel()),),
    )


def _graph_entry(
    kind: str,
    ctx: Any,
    ws: Any,
    backend: Backend,
    opts: KcpOptions,
    scale,
    tensors,
    flags: tuple,
) -> _GraphEntry | None:
    """The cache entry of this call's validity key, or ``None`` when the call is never captured."""
    if (
        not isinstance(ws, KcpWorkspace)
        or opts.knob("debug_sequence")
        or opts.exchange == "ring"
    ):
        return None
    key = (
        kind,
        id(ctx),
        id(ws),
        id(backend),
        opts.key(),
        None if scale is None else float(scale),
        flags,
        tuple(_sig(t) for t in tensors),
    )
    e = _GRAPHS.get(key)
    if e is None:
        while len(_GRAPHS) >= GRAPH_CACHE_MAX:
            del _GRAPHS[next(iter(_GRAPHS))]
        e = _GRAPHS[key] = _GraphEntry(key, (ctx, ws, backend))
    return e


def _graph_capture(kind: str, entry: _GraphEntry, call: KcpCall, body: Any) -> Any:
    """Capture ``body()`` (the call's allocations, launches and collectives) into a CUDA graph, replay it once and keep
    it for the key; returns the body's result, or ``None`` when the capture failed (recorded in the entry; the key
    then runs eagerly for the rest of the process)."""
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            result = body()
    except Exception as exc:  # noqa: BLE001 - the failure is recorded; the caller runs the eager path
        entry.disabled = f"{type(exc).__name__}: {str(exc)[:200]}"
        GRAPH_STATS["fallback"] += 1
        return None
    g.replay()
    _record(kind, call)
    LAST_CALL[kind].update(graph_event="capture", graph_key=entry.digest)
    entry.record = dict(LAST_CALL[kind])
    entry.graph = g
    entry.arena = _arena_sig(call.ws)
    entry.seen += 1
    GRAPH_STATS["capture"] += 1
    call.backend.end_call(call)
    return result


def _graph_replay(kind: str, entry: _GraphEntry) -> None:
    entry.graph.replay()
    GRAPH_STATS["replay"] += 1
    LAST_CALL[kind] = {**entry.record, "graph_event": "replay"}


def _graph_eager(kind: str, entry: _GraphEntry | None) -> None:
    GRAPH_STATS["eager"] += 1
    if entry is not None:
        entry.seen += 1
        LAST_CALL[kind].update(
            graph_key=entry.digest,
            graph_fallback=entry.disabled,
            graph_reset=entry.reset,
        )
        entry.reset = None


def graph_cache_clear() -> int:
    """Drop every cached graph with its static buffers and private memory pool; returns the number of entries
    dropped.  A graph that captured a NCCL collective must be destroyed before its process group
    (``torch.distributed.destroy_process_group()`` hangs while such a graph is alive)."""
    n = len(_GRAPHS)
    _GRAPHS.clear()
    return n


# ----------------------------------------------------------------------------------------------------------------
# Entry points (public signatures)


def _prepare(
    q,
    k,
    v,
    g,
    beta,
    ctx,
    *,
    scale,
    opts: KcpOptions,
    workspace,
    backward: bool,
    dout=None,
    fresh_pools: Sequence[str] = (),
):
    """``fresh_pools``: the pools this call binds fresh / saved tensors for (never read from the workspace)."""
    backend = get_backend(opts.backend)
    q, k, v, g, beta, T, H_qk, HV, batched = _operands(
        q, k, v, g, beta, ctx, dtypes=backend.dtypes, layout=opts.layout
    )
    if opts.exchange == "ring":
        validate_ring(ctx.n_d_max)
    device = torch.device(ctx.device)
    if dout is not None:
        dout = _squeeze_batch(dout, "dout", 3)
        if not dout.is_contiguous():
            dout = dout.contiguous()
        _check(dout, "dout", (T, HV, D), backend.dtypes["dout"], device)
    if workspace is None:
        workspace = KcpWorkspace(state_dtype=backend.dtypes["state"])
    elif not isinstance(workspace, KcpWorkspace):
        raise TypeError("workspace must be a KcpWorkspace")
    if workspace.state_dtype != backend.dtypes["state"]:
        raise ValueError(
            f"workspace state dtype {workspace.state_dtype} differs from the backend's {backend.dtypes['state']}"
        )
    P = None if opts.segments is None else int(opts.segments)
    prepare = getattr(
        ctx, "prepare", None
    )  # the plan's device tables per (HV, sm_count, P) key, uploaded at first use
    if callable(prepare):
        prepare(H_qk, HV, P=P)
    # Dead pools are not allocated (R2-L1 1a): the ring pools under the a2a exchange and the pools the call binds
    # fresh / saved tensors for (``c1_recv`` / ``stage_m`` under ``save_transfer``) never hold a byte of this call.
    skip = list(fresh_pools)
    if opts.exchange != "ring":
        skip += list(RING_POOLS)
        if not backward and opts.graph == "off":
            # the eager a2a forward binds a fresh c2_send plane at its first writer (the fwd merge) and drops it after
            # wait(c2); a captured forward keeps the pool (its address is baked into the graph)
            skip.append("c2_send")
    tb = workspace.ensure(
        ctx,
        H_qk,
        HV,
        device=device,
        backward=backward,
        fused_walks=int(opts.knob("fused_walks")),
        P=P,
        skip=skip,
    )
    ops = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
    if dout is not None:
        ops["dout"] = dout
    call = KcpCall(
        ctx=ctx,
        H_qk=H_qk,
        HV=HV,
        ws=workspace,
        tb=tb,
        backend=backend,
        opts=opts,
        scale=SCALE if scale is None else float(scale),
        ops=ops,
        backward=backward,
    )
    backend.begin_call(call)
    return call, batched


def chunk_gdn_kcp_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    *,
    scale: float | None = None,
    save_for_backward: bool = True,
    workspace: KcpWorkspace | None = None,
    options: KcpOptions | dict[str, Any] | None = None,
    **knobs: Any,
) -> tuple[torch.Tensor, KcpSaved | None]:
    """The KCP forward of this rank's fragments: ``out`` like ``v`` (fresh) and the saved set (``None``
    without ``save_for_backward`` or with ``boundary_mode="replay"``).  ``ctx`` is this rank's ``KcpContext``; the
    token operands are its ``T_r`` local tokens in fragment order; ``options`` / ``**knobs`` are the keyword-only
    fields of :class:`KcpOptions` (``save_mode``, ``boundary_mode``, ``graph``, ...)."""
    return _fwd_entry(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        scale=scale,
        save_for_backward=save_for_backward,
        workspace=workspace,
        opts=make_options(options, **knobs),
        hand_off=False,
    )


def _fwd_entry(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    *,
    scale: float | None,
    save_for_backward: bool,
    workspace: KcpWorkspace | None,
    opts: KcpOptions,
    hand_off: bool,
    autograd_node: bool = False,
    executable: bool = True,
) -> tuple[torch.Tensor, KcpSaved | None]:
    """:func:`chunk_gdn_kcp_fwd` with validated options.  ``hand_off``: the autograd Function's forward runs inside
    a graph task with ``reuse_recompute_tapes`` -- the light saved set then carries the forward tapes (``saved.kept``)
    for the backward of that task (never with a captured graph, a replay boundary, ``save_mode="full"`` or
    ``T_r = 0``).  ``autograd_node``: the caller is :class:`ChunkGDNKcpFunction` (the one forward whose recompute may
    leave #5 out under ``recompute_output="auto"``); ``executable``: that Function's ``apply`` creates a node and
    packs its saved tensors (grad mode on and an operand requiring grad) -- ``False`` launches #5 unconditionally."""
    save = bool(save_for_backward) and opts.boundary_mode == "saved"
    hand_off = (
        bool(hand_off) and save and opts.save_mode == "light" and opts.graph == "off"
    )
    entry = None
    if opts.graph != "off":
        entry = _graph_entry(
            "fwd",
            ctx,
            workspace,
            get_backend(opts.backend),
            opts,
            scale,
            (q, k, v, g, beta),
            (save,),
        )
    if entry is not None and entry.replayable(workspace):
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
        opts=opts,
        workspace=workspace,
        backward=False,
        fresh_pools=SAVE_TRANSFER_FRESH_POOLS if (save and opts.save_transfer) else (),
    )
    if entry is not None and entry.seen > 0 and entry.disabled is None:
        res = _graph_capture(
            "fwd",
            entry,
            call,
            lambda: _fwd_body(
                call, save=save, autograd_node=autograd_node, executable=executable
            ),
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
            opts=opts,
            workspace=workspace,
            backward=False,
            fresh_pools=SAVE_TRANSFER_FRESH_POOLS
            if (save and opts.save_transfer)
            else (),
        )
    out, saved = _fwd_body(
        call,
        save=save,
        hand_off=hand_off and call.T_r > 0,
        autograd_node=autograd_node,
        executable=executable,
    )
    _record("fwd", call)
    if opts.graph != "off":
        _graph_eager("fwd", entry)
    call.backend.end_call(call)
    return _ranked(out, batched), saved


def _saved_count(call: KcpCall, save: bool) -> int:
    """What :class:`ChunkGDNKcpFunction` saves for this forward: the five operands and the saved set's tensors
    (:meth:`KcpSaved.tensors`: ``S_in``; the c1 receive tensor under ``save_transfer``; ``M_seg`` under
    ``save_transfer`` at ``P > 1``; the saved tapes of the ``save_mode`` when the rank holds tokens) -- known before
    the forward runs, so the frame proof can decide a hand-off (``reuse_recompute_tapes="auto"``)."""
    if not save:
        return _SAVED_OPERANDS
    opts, tb = call.opts, call.tb
    n = (
        1
        + (1 if opts.save_transfer else 0)
        + (1 if (opts.save_transfer and tb.P > 1) else 0)
    )
    if call.T_r > 0:
        n += len(saved_tapes(opts.save_mode))
    return _SAVED_OPERANDS + n


def _fwd_body(
    call: KcpCall,
    *,
    save: bool,
    hand_off: bool = False,
    autograd_node: bool = False,
    executable: bool = True,
) -> tuple[torch.Tensor, KcpSaved | None]:
    """The forward's allocations and launches (eagerly or inside a capture): the sequence, ``out`` and #5, the saved
    set.  ``hand_off``: the ten forward tapes (and the ``stage_m`` plane when the saved set carries no ``M_seg``)
    leave the arena as the saved set's ``kept`` hand-off instead of being dropped (``reuse_recompute_tapes``).
    ``autograd_node`` with ``recompute_output="auto"`` (eager only): #5 is left out when the forward is torch's
    checkpoint recompute whose output nothing reads (:func:`_recompute_output_dead`; ``out`` is allocated as ever,
    never written) -- only for an ``executable`` node (the one that packs the saved tensors the proof counts); a
    non-executable one launches with the static reason ``not_executable``."""
    proof: dict[
        int, tuple[bool, str]
    ] = {}  # the checkpoint-frame proof, read once per saved count

    def frame_proof(n_saved: int) -> tuple[bool, str]:
        if n_saved not in proof:
            proof[n_saved] = _recompute_output_dead(n_saved)
        return proof[n_saved]

    hand_off_reason = "always" if hand_off else "off"
    if hand_off and call.opts.reuse_recompute_tapes == "auto":
        # keep the tapes only when this node closes the checkpointed region (the frame proof of
        # recompute_output="auto"): then its backward is the region's first and the kept tapes die before any other
        # backward runs.  With a later saver in the region (an output projection, an MLP) the kept set would live
        # through those operations' backward and raise the region's backward peak by the full tape set, so the
        # replay form is taken there; every unreadable state (another torch release, early stop off, user hooks, a
        # non-executable node) is the replay form as well.
        if not (autograd_node and executable and call.opts.graph == "off"):
            hand_off, hand_off_reason = False, "not_executable"
        else:
            hand_off, hand_off_reason = frame_proof(_saved_count(call, save))
    call.scratch["hand_off_reason"] = hand_off_reason
    if hand_off:
        _kept_evict_all()  # one live entry: the previous set is freed before this forward materialises its own
    _forward(call, save=save, replay=False, hand_off=hand_off)
    stage_m = None
    if hand_off and call.tb.P > 1 and not call.opts.save_transfer:
        # the transfer walk's M_seg^T plane: the backward's merges read it and no saved M_seg replaces it; the view
        # keeps the pool's storage alive after the release below (a fresh pool is sized by the next ensure)
        stage_m = call.buf("stage_m")
    if call.opts.graph == "off":
        # the forward's peak: the dead arena tapes and pools go first, then the output (FWD_DEAD_BEFORE_O /
        # FWD_DEAD_POOLS); _forward's frame is gone, so no view keeps a pool alive.  A captured graph keeps every
        # arena buffer: it bakes their addresses and the capture key tracks them.  A hand-off forward drops no tape.
        if not hand_off:
            call.release_tapes(
                FWD_DEAD_BEFORE_O, keep=saved_tapes(call.opts.save_mode) if save else ()
            )
        call.release_pools(FWD_DEAD_POOLS)
    call.out = _alloc((call.T_r, call.HV, D), call.backend.dtypes["out"], call.device)
    saved = _saved_set(call) if save else None
    launch, reason = True, "always"
    if (
        autograd_node
        and call.opts.recompute_output == "auto"
        and call.opts.graph == "off"
    ):
        if not executable:
            # autograd creates no node and packs nothing: the holders still waiting are a later op's
            reason = "not_executable"
        else:
            # the Function saves the five operands and saved.tensors() right after this forward returns
            dead, reason = frame_proof(
                _SAVED_OPERANDS + (len(saved.tensors()) if saved is not None else 0)
            )
            launch = not dead
            if reason in _RECOMPUTE_FRAME_REASONS:
                RECOMPUTE_OUTPUT_STATS["skipped" if dead else "launched"] += 1
    call.scratch["fwd_o"] = "launched" if launch else "skipped"
    call.scratch["fwd_o_reason"] = reason
    if launch:
        call.local("fwd_o")
    kept = None
    if hand_off:
        kept = _KeptTapes(
            {n: call.tape(n) for n in FWD_TAPES},
            stage_m,
            call.ws.geometry,
            _graph_task_id(),
        )
        for n in FWD_TAPES:  # unbound: a later forward on this workspace cannot overwrite the kept tensors
            call.ws.release_tape(n)
    if saved is not None and kept is not None:
        saved = dataclasses.replace(saved, kept=kept)
    return call.out, saved


def _saved_set(call: KcpCall) -> KcpSaved:
    """The saved set of this forward (``boundary_mode="saved"``, ``kept=None``): the fresh ``S_in``, with
    ``save_transfer`` the c1 receive tensor and ``M_seg`` (``P > 1``), the saved tapes of the ``save_mode`` (none in
    ``"light"``), the scalars of the call.  Built before #5 so the forward knows what the Function saves."""
    tb, opts = call.tb, call.opts
    return KcpSaved(
        version=KCP_SAVED_VERSION,
        S_in=call.bufs["S_in"],
        c1_recv=call.bufs.get("c1_recv") if opts.save_transfer else None,
        M_seg=call.bufs.get("stage_m") if (opts.save_transfer and tb.P > 1) else None,
        tapes=tuple(call.tapes[n] for n in saved_tapes(opts.save_mode))
        if call.T_r > 0
        else None,
        T_r=call.T_r,
        H_qk=call.H_qk,
        HV=call.HV,
        n_frag=tb.n_frag,
        P=tb.P,
        Hg=tb.Hg,
        scale=call.scale,
        save_mode=opts.save_mode,
        save_transfer=opts.save_transfer,
        offset_class=opts.offset_class,
        exchange=call.exchange,
        precision_mode=call.precision_mode,
        transfer_walk=call.transfer_walk,
        ctx=call.ctx,
        kept=None,
    )


def _check_saved(saved: KcpSaved, call: KcpCall) -> None:
    tb, opts = call.tb, call.opts
    if saved.version != KCP_SAVED_VERSION:
        raise ValueError(f"saved: version {saved.version} is not {KCP_SAVED_VERSION}")
    if saved.ctx is not call.ctx:
        raise ValueError("saved: the saved set belongs to another KcpContext")
    if (saved.T_r, saved.H_qk, saved.HV, saved.n_frag, saved.P) != (
        call.T_r,
        call.H_qk,
        call.HV,
        tb.n_frag,
        tb.P,
    ):
        raise ValueError(
            "saved: the saved set describes another problem (T_r, H_qk, HV, n_frag, P)"
        )
    if saved.scale != call.scale:
        raise ValueError(
            f"saved: scale {saved.scale} differs from the call's {call.scale}"
        )
    if (
        saved.save_transfer != opts.save_transfer
        or saved.exchange != call.exchange
        or saved.precision_mode != call.precision_mode
        or saved.offset_class != opts.offset_class
        or saved.transfer_walk != opts.transfer_walk
        or saved.save_mode != opts.save_mode
    ):
        raise ValueError(
            "saved: save_mode / save_transfer / exchange / precision_mode / transfer_walk / offset_class differ from "
            "the forward's"
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
    if call.T_r > 0:
        names = saved_tapes(opts.save_mode)
        if saved.tapes is None or len(saved.tapes) != len(names):
            raise ValueError(
                f"saved: save_mode={opts.save_mode!r} needs the saved tapes {names}"
            )
        R, NC_pad = int(call.ctx.R), int(call.ctx.NC_pad)
        for name, t in zip(names, saved.tapes, strict=True):
            shape, dtype = tape_spec(name, R=R, NC_pad=NC_pad, HV=call.HV, T=call.T_r)
            _check(t, f"saved.{name}", shape, dtype, call.device)
    kept: _KeptTapes | None = saved.kept
    # the recompute forward's ten tapes of this geometry (+ the stage_m plane without a saved M_seg)
    if kept is not None:
        if saved.save_mode != "light" or opts.graph != "off" or call.T_r == 0:
            raise ValueError(
                "saved: kept recompute tapes apply to the light saved-context backward with graph='off' and T_r > 0"
            )
        if kept.geometry != call.ws.geometry:
            raise ValueError(
                "saved: the kept recompute tapes describe another tape geometry (R, NC_pad, HV, T_r)"
            )
        if set(kept.tapes) != set(FWD_TAPES):
            raise ValueError(
                "saved: the kept recompute tapes are not the ten forward tapes"
            )
        R, NC_pad = int(call.ctx.R), int(call.ctx.NC_pad)
        for name in FWD_TAPES:
            shape, dtype = tape_spec(name, R=R, NC_pad=NC_pad, HV=call.HV, T=call.T_r)
            _check(kept.tapes[name], f"kept.{name}", shape, dtype, call.device)
        if tb.P > 1 and not opts.save_transfer:
            if kept.stage_m is None:
                raise ValueError(
                    "saved: kept recompute tapes at P > 1 without save_transfer need the forward's stage_m plane"
                )
            _check(
                kept.stage_m,
                "kept.stage_m",
                (tb.sizes["stage_m"],),
                call.state_dtype,
                call.device,
            )


def _saved_fresh_pools(saved: KcpSaved | None) -> tuple[str, ...]:
    """The workspace pools a saved-context backward replaces with saved tensors (``c1_recv`` under
    ``save_transfer``, ``stage_m`` = the saved ``M_seg^T`` at ``P > 1``, or the kept recompute forward's plane);
    a replay binds none."""
    if saved is None:
        return ()
    names = []
    if saved.c1_recv is not None:
        names.append("c1_recv")
    if saved.M_seg is not None or (
        saved.kept is not None and saved.kept.stage_m is not None
    ):
        names.append("stage_m")
    return tuple(names)


def chunk_gdn_kcp_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    dout: torch.Tensor,
    *,
    saved: KcpSaved | None = None,
    scale: float | None = None,
    workspace: KcpWorkspace | None = None,
    options: KcpOptions | dict[str, Any] | None = None,
    **knobs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The KCP backward: ``(dq, dk, dv, dbeta, dg)``, fresh.  ``saved=None`` replays the forward."""
    opts = make_options(options, **knobs)
    entry = None
    if opts.graph != "off":
        kept = () if saved is None else saved.tensors()
        entry = _graph_entry(
            "bwd",
            ctx,
            workspace,
            get_backend(opts.backend),
            opts,
            scale,
            (q, k, v, g, beta, dout, *kept),
            (saved is not None,),
        )
    if entry is not None and entry.replayable(workspace):
        _graph_replay("bwd", entry)
        grads, batched = entry.result
        return tuple(_ranked(t.detach(), batched) for t in grads)
    call, batched = _prepare(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        scale=scale,
        opts=opts,
        workspace=workspace,
        backward=True,
        dout=dout,
        fresh_pools=_saved_fresh_pools(saved),
    )
    if saved is not None:
        _check_saved(saved, call)
    if entry is not None and entry.seen > 0 and entry.disabled is None:
        res = _graph_capture("bwd", entry, call, lambda: _bwd_body(call, saved))
        if res is not None:
            entry.result = (res, batched)
            return tuple(_ranked(t.detach(), batched) for t in res)
        call, batched = _prepare(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            scale=scale,
            opts=opts,
            workspace=workspace,
            backward=True,
            dout=dout,
            fresh_pools=_saved_fresh_pools(saved),
        )
        if saved is not None:
            _check_saved(saved, call)
    grads = _bwd_body(call, saved)
    _record("bwd", call)
    if opts.graph != "off":
        _graph_eager("bwd", entry)
    call.backend.end_call(call)
    return tuple(_ranked(t, batched) for t in grads)


def _alloc_grads(call: KcpCall) -> None:
    """The five gradient outputs, fresh per call.  Allocated right before #8 (:func:`_backward` calls this after the
    adjoint tape walk and the ``BWD_DEAD_BEFORE_DQKG`` release): only #8 / #9 write them, and allocating them at the
    top of the backward put 2 x (dq + dk) + dv + dbeta + dg at the backward's peak moment -- the adjoint tape walk --
    for nothing (R2-L1 1b)."""
    dt = call.backend.dtypes
    T, H_qk, HV = call.T_r, call.H_qk, call.HV
    call.grads = {
        "dq": _alloc((T, H_qk, D), dt["dq"], call.device),
        "dk": _alloc((T, H_qk, D), dt["dk"], call.device),
        "dv": _alloc((T, HV, D), dt["dv"], call.device),
        "dbeta": _alloc((T, HV), dt["dbeta"], call.device),
        "dg": _alloc((T, HV), dt["dg"], call.device),
    }


def _bwd_body(call: KcpCall, saved: KcpSaved | None) -> tuple[torch.Tensor, ...]:
    """The backward's allocations and launches (eagerly or inside a capture): ``(dq, dk, dv, dbeta, dg)``."""
    _backward(call, saved)
    return tuple(call.grads[n] for n in ("dq", "dk", "dv", "dbeta", "dg"))


# ----------------------------------------------------------------------------------------------------------------
# autograd


class ChunkGDNKcpFunction(torch.autograd.Function):
    """:func:`chunk_gdn_kcp` as an autograd node: the forward keeps the inputs and the saved set (saved mode) or the
    inputs only (replay mode) through ``save_for_backward`` -- nothing else holds a tensor (#7039); the backward runs
    :func:`chunk_gdn_kcp_bwd`.  Under ``torch.utils.checkpoint(use_reentrant=False)`` the forward is re-run in the
    backward: the same program order on every rank, so the collectives match.  With ``reuse_recompute_tapes`` that
    re-run (a forward inside the backward's graph task) hands its forward tapes to this node's backward through the
    module's kept-tape registry (never through ``save_for_backward``: the recompute must save exactly what the
    original forward saved), and the backward binds them instead of replaying #1 and the tape walk.  With
    ``recompute_output="auto"`` the recompute of a region this node closes (the last saver) leaves its output program
    out (the frame proof of :func:`_recompute_output_dead`); its output is read by nothing.  ``executable`` (read by
    :func:`chunk_gdn_kcp` before ``apply``, where grad mode is still the caller's) says whether autograd will create
    this node and pack its saved tensors at all; a non-executable node never skips."""

    @staticmethod
    def forward(fctx, q, k, v, g, beta, ctx, scale, workspace, opts, executable):
        gid = _graph_task_id() if opts.reuse_recompute_tapes else -1
        out, saved = _fwd_entry(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            scale=scale,
            save_for_backward=True,
            workspace=workspace,
            opts=opts,
            hand_off=gid != -1,
            autograd_node=True,
            executable=bool(executable),
        )
        if saved is not None and saved.kept is not None:
            key = _kept_key(
                gid, q, k, v, g, beta, saved.S_in, saved.T_r, saved.H_qk, saved.HV
            )
            _kept_put(key, saved.kept)
        tensors = saved.tensors() if saved is not None else ()
        fctx.save_for_backward(q, k, v, g, beta, *tensors)
        fctx.kcp = (
            ctx,
            scale,
            workspace,
            opts,
            None if saved is None else (saved.flags(), saved.meta()),
        )
        return out

    @staticmethod
    def backward(fctx, dout):
        # under a non-reentrant checkpoint this unpack runs torch's recompute of the forward
        q, k, v, g, beta, *tensors = fctx.saved_tensors
        ctx, scale, workspace, opts, meta = fctx.kcp
        saved = (
            None
            if meta is None
            else dataclasses.replace(
                KcpSaved.from_tensors(tensors, meta[0], meta[1], ctx),
                # this node's own set: forgotten tensor by tensor at the last reader
                release_in_backward=True,
            )
        )
        # the set above holds the only Python references to the saved tensors
        del tensors
        if (
            saved is not None
            and opts.reuse_recompute_tapes
            and saved.save_mode == "light"
            and opts.graph == "off"
        ):
            gid = _graph_task_id()
            kept = None
            if gid != -1:
                key = _kept_key(
                    gid, q, k, v, g, beta, saved.S_in, saved.T_r, saved.H_qk, saved.HV
                )
                # under "auto" a forward that did not hand off leaves no entry by design: no miss is counted
                kept = (
                    _kept_take(key)
                    if (opts.reuse_recompute_tapes is True or key in _KEPT)
                    else None
                )
            if kept is not None:
                saved = dataclasses.replace(saved, kept=kept)
        if not dout.is_contiguous():
            dout = dout.contiguous()
        dq, dk, dv, dbeta, dg = chunk_gdn_kcp_bwd(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            dout,
            saved=saved,
            scale=scale,
            workspace=workspace,
            options=opts,
        )
        return dq, dk, dv, dg, dbeta, None, None, None, None, None


def chunk_gdn_kcp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    *,
    scale: float | None = None,
    save_mode: str = _opt.SAVE_MODE_DEFAULT,
    boundary_mode: str = _opt.BOUNDARY_MODE_DEFAULT,
    workspace: KcpWorkspace | None = None,
    graph: str = _opt.GRAPH_DEFAULT,
    options: KcpOptions | dict[str, Any] | None = None,
    **knobs: Any,
) -> torch.Tensor:
    """The differentiable KCP forward: ``out`` like ``v``; gradients for ``q, k, v, g, beta``
    in the backward's order ``(dq, dk, dv, dbeta, dg)``.  With an ``options`` instance, the three literal keywords
    override it only when they differ from their documented defaults."""
    over: dict[str, Any] = {}
    for name, value, default in (
        ("save_mode", save_mode, _opt.SAVE_MODE_DEFAULT),
        ("boundary_mode", boundary_mode, _opt.BOUNDARY_MODE_DEFAULT),
        ("graph", graph, _opt.GRAPH_DEFAULT),
    ):
        if options is None or value != default:
            over[name] = value
    opts = make_options(options, **over, **knobs)
    # autograd's own executability test for the node (grad mode and an operand requiring grad, flags only): read
    # here because inside Function.forward grad mode is always off
    executable = torch.is_grad_enabled() and any(
        t.requires_grad for t in (q, k, v, g, beta)
    )
    return ChunkGDNKcpFunction.apply(
        q, k, v, g, beta, ctx, scale, workspace, opts, executable
    )


# ----------------------------------------------------------------------------------------------------------------
# Byte queries (literal public forms; the pure-CPU formulas live in the package's byte module)


def gdn_kcp_comm_bytes(ctx: Any, **kw: Any) -> dict:
    """``gdn_kcp_comm_bytes(ctx, *, H_qk, HV, C, ...)``: bytes this rank moves per collective and per boundary."""
    from . import cake_bytes  # noqa: PLC0415

    return cake_bytes.gdn_kcp_comm_bytes(ctx, **kw)


def gdn_kcp_saved_bytes(ctx: Any, **kw: Any) -> int:
    """``gdn_kcp_saved_bytes(ctx, *, H_qk, HV, save_mode, save_transfer, ...)``: saved-set bytes of one layer."""
    from . import cake_bytes  # noqa: PLC0415

    return cake_bytes.gdn_kcp_saved_bytes(ctx, **kw)


def gdn_kcp_workspace_bytes(ctx: Any, **kw: Any) -> int:
    """``gdn_kcp_workspace_bytes(ctx, *, H_qk, HV, ...)``: caller-held workspace bytes of one rank."""
    from . import cake_bytes  # noqa: PLC0415

    return cake_bytes.gdn_kcp_workspace_bytes(ctx, **kw)


__all__ = [
    "BOUNDARY_MODES",
    "BWD_TAPES",
    "COLLECTIVES",
    "CONTRACT_DTYPES",
    "EXCHANGES",
    "FWDH_PROGRAMS",
    "FWD_TAPES",
    "GRAPH_DEFAULT",
    "GRAPH_MODES",
    "GRAPH_STATS",
    "HOST_ONLY_ARGS",
    "HOST_REDUCE_TAPES",
    "KCP_SAVED_VERSION",
    "LAST_CALL",
    "BWD_DEAD_AFTER_ADJ_MERGE",
    "DHU_OFFSET_TAPE_STANDIN",
    "DHU_PROGRAMS",
    "BWD_DEAD_BEFORE_DQKG",
    "BWD_DEAD_BEFORE_TAPES",
    "RING_POOLS",
    "SAVE_TRANSFER_FRESH_POOLS",
    "FWD_DEAD_AFTER_PREP",
    "FWD_DEAD_BEFORE_O",
    "FWD_DEAD_POOLS",
    "FWD_TAPES_ARENA_AFTER_FWD",
    "KEPT_STATS",
    "LIGHT_TAPES",
    "TAPE_ALIASES",
    "LOCAL_PROGRAMS",
    "OFFSET_CLASSES",
    "PROGRAMS",
    "PROGRAM_ARGS",
    "PROVISIONAL_ARGS",
    "RECOMPUTE_OUTPUT_STATS",
    "RECOMPUTE_OUTPUT_TORCH_VERSIONS",
    "RECORD_OF",
    "RING_MAX_HOPS",
    "SAVE_MODES",
    "SCALE",
    "SUMMARY_TILE",
    "VH64_SUFFIX",
    "TART_MODULE_OF",
    "TICKET_SITES",
    "TICKET_SLOT_BYTES",
    "TICKET_SLOT_WORDS",
    "TEST_RECORDS",
    "TILE",
    "WALK_PROGRAMS",
    "Backend",
    "Chains",
    "ChunkGDNKcpFunction",
    "FakeBackend",
    "Indexed",
    "KcpCall",
    "KcpOptions",
    "KcpSaved",
    "KcpWorkspace",
    "KernelBackend",
    "chunk_gdn_kcp",
    "chunk_gdn_kcp_bwd",
    "chunk_gdn_kcp_fwd",
    "dhu_vh",
    "fwdh_vh",
    "gather_states",
    "gdn_kcp_comm_bytes",
    "gdn_kcp_saved_bytes",
    "gdn_kcp_workspace_bytes",
    "get_backend",
    "graph_cache_clear",
    "kept_registry_size",
    "make_options",
    "merge_cols_per_cta",
    "resolve_launch",
    "saved_states",
    "saved_tapes",
    "set_backend",
    "tape_spec",
    "test_record",
    "tile_index",
    "torch_release",
    "torch_release_verified",
]
