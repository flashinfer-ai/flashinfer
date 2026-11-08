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
(SM100 / SM103 / SM107): the transient :class:`CakeGDNNondetKcpWorkspace`, the forward and
backward launch sequences with their collectives, the saved set
:class:`CakeGDNNondetKcpSaved`, the autograd Function and the entry points
``cake_gdn_nondet_kcp`` / ``cake_gdn_nondet_train_kcp_forward`` / ``cake_gdn_nondet_train_kcp_backward``.

Provenance: adapted from ``flashinfer/experimental/cake_kda_train/cake_kcp.py`` (now
``flashinfer/experimental/cake/cake_kda_det_train_kcp/cake_kcp.py``)
at FlashInfer ``13d7fcf4041635f5b7f573bb0dbab23570ae1034`` (the KDA KCP host
module).  Verbatim parts: the compact pools and chains (``Indexed``,
``Chains``), the exchange tables (``_Tables``), the per-call state
(``CakeGDNNondetKcpCall``), the ring exchange, the launch-sequence structure and the
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
then runs from ``S_in`` and ``fwd_o`` (#5) writes the output -- or, in a
forward whose ``h`` / ``v_new`` tapes have no reader (test knob ``fused_o``,
:func:`_fuse_o`), the output walk ``kcp_fwdh_out`` (#2 with #5 folded into
each chunk step, no tape written) does both; in a ``save_mode="full"``
forward that keeps its saved set and launches #5 (test knob ``fused_o_tape``,
:func:`_fuse_o_tape`), the tape-and-output walk ``kcp_fwdh_out_tape`` (the
output walk that also writes the saved ``h`` / ``v_new`` tapes) does both.
Backward:
``bwd_prep`` (#6); the adjoint offsets ``d_f`` (#7 ``program=offset``, the
reverse walk from zero) and, unless the forward's transfer was kept, ``M_f``
again go to the owners (``b1``); the owner walks each chain backwards
(``X <- M_f^T X + d_f``, #4 transposed) and returns every crossing-out
fragment's ``dS_out`` (``b2``); the adjoint tape walk (#7 ``program=tape``)
runs from it; ``bwd_dqkg`` (#8, the FP32 group reduction of the ``r`` value
heads' ``dq`` / ``dk`` fused) and ``gate_epilogue`` (#9) finish the gradients.
One compute stream (on ``sm_100a`` / ``sm_103a`` with a zigzag / zigzag-per-document partition of at most 8 ranks: the
summaries' offset walk on a side stream beside the transfer walk, fork / join with CUDA events, :func:`_walk_pair_enabled`);
the collectives are asynchronous on the caller's process
group and waited at their first consumer; the issue order is the same on
every rank, so a checkpoint replay re-runs the same program order.  With a recompute handle
(``recompute_handle``, :class:`CakeGDNNondetKcpRecomputeHandle`) the original forward stores its ``S_in`` in the handle and torch's
recompute of the forward reads it back: no collective, no summary walk over the crossing-out fragments, no owner
merge -- the light replay's launches (at ``P > 1`` its local segment walks and carry merges).

Saved set (``boundary_mode="saved"``): ``S_in`` (fresh), the owner's c1
receive tensor (``save_transfer=True``), ``M_seg`` (``P > 1``, with
``save_transfer`` or ``save_mode="full"``) and, with
``save_mode="full"``, the forward tapes (``h``, ``v_new``, ``w``, ``u``,
``Aqk``, ``Akk``, the gate rows); ``save_mode="light"`` (the default without
the in-kernel norm) keeps the boundary tensors only and the backward re-runs
#1 and the #2 tape walk from the saved ``S_in`` -- no exchange, no transfer
walk, no merge, no #5.
``boundary_mode="replay"`` keeps nothing and the backward re-runs the whole
forward exchange.  Every saved tensor travels through
``ctx.save_for_backward`` only, so ``torch.utils.checkpoint(use_reentrant=False)``
retains nothing after a checkpointed forward (#7039).
``save_mode="light_akk"`` (``use_qk_l2norm_in_kernel`` only; the default
there: a call that leaves ``save_mode`` unset runs it) is ``"light"``
plus #1's BF16 ``Akk`` tape: its light replay runs ``fwd_prep_replay_l2n``,
#1_l2n reading that tape instead of forming it (no ``K K^T`` Gram, no block
inverse), every other output bit for bit #1_l2n's.  Under
``torch.utils.checkpoint(use_reentrant=False)`` it saves as ``"light"``
(:func:`_checkpoint_opts`).  The modes are bitwise identical by construction
(the same programs on the same bits).
With ``reuse_recompute_tapes`` the light backward of such a checkpoint binds
the forward tapes of torch's recompute (the forward re-run inside
``.backward()``) instead of replaying #1 and the tape walk itself (the kept
tapes, below ``_graph_task_id`` / ``_KeptTapes``).  With
``recompute_output="auto"`` (default) that recompute leaves its #5 out when
torch's checkpoint frame proves its output dead (below
``_recompute_output_dead``).  Under ``graph="replay"`` both decisions are part
of the recompute's CUDA-graph key and the kept tapes are the workspace arena's
own (below ``_kept_stale``), so a captured checkpoint step launches the eager
step's programs -- from ``GRAPH_RECOMPUTE_MIN_ROWS`` token x value-head rows per
rank under the default ``"auto"``; a smaller recompute decides nothing and
replays the original forward's graph, #5 included (``fwd_o_reason`` ``"small"``;
the size rule, below ``_fwd_entry``).  The kept tapes add no memory there; the
recompute's hand-off graph holds its own static results like every graph key
(a saved set and, when #5 launches in it -- ``recompute_output="always"``, or
``True`` with a later saver -- a full output per checkpointed layer).

Tapes (padded rows ``R = 64 * NC_pad``; token ``t`` of chunk ``c`` sits in row
``64 c + (t - chunk_bos[c])``): ``w, u, v_new, do_g, dv_intra`` BF16
``[R, HV, 128]`` (``dv_new`` is stored in place of ``dv_intra``); ``aqk, akk`` BF16 ``[R, HV, 64]``; ``g_cum, rowscale,
qscale, dg_part`` FP32 ``[R, HV]``; ``gamma, gamma_dot`` FP32 ``[NC_pad, HV]``;
``h_tape, dh_tape`` BF16 ``[NC_pad, HV, 128, 128]``.  Head mapping of a value
head ``hv`` to its Q/K head: ``hv // r``.

In-kernel Q / K L2 norm (``use_qk_l2norm_in_kernel=True``, default ``False``): ``q`` / ``k`` stay the raw operands
(the convolution's output views); #1 runs as ``fwd_prep_l2n``, which writes the normalized rows ``qn`` / ``kn``
(BF16 ``[T_r, H_qk, 128]``, transient arena tapes) and the FP32 ``r = 1 / sqrt(sum q^2 + 1e-6)`` of q and k
(``qk_rstd`` ``[2, T_r, H_qk]``) before its own body; every other q / k reader (#2, #3, #5, #7, #8) is the unchanged
program bound to ``qn`` / ``kn``; #8 runs as ``bwd_dqkg_l2n``, which applies the norm's pullback (from the raw rows
and the saved ``r``) before the one BF16 rounding of ``dq`` / ``dk`` (``group_reduce="host"``: the unchanged #8 and the
pullback of :mod:`.cake_l2n` on the host, chunked by token blocks).  The saved set gains ``qk_rstd`` only
(``KCP_SAVED_VERSION`` 2); ``qn`` / ``kn`` never outlive a call: the forward drops ``kn`` before its output and
``qn`` after #5 (a hand-off forward hands both over with its tapes instead); the backward gets them from #1_l2n
again (light replay, replay mode), from the recompute forward's #1_l2n through the kept hand-off (the kept recompute
tapes) or from ``qk_apply`` (``qn = bf16(q * r)`` from the saved ``r``: ``save_mode="full"`` only) and drops them
after #8.
The numeric orders are :mod:`.cake_l2n`'s (FTZ off), bitwise the device programs'.

This module reads tensor contents nowhere (no ``.item()``, no ``.tolist()``
of a device tensor, no device synchronisation), dispatches from metadata and
the SM count only, allocates the outputs and the saved set fresh per call and
keeps only transient buffers in the workspace.  The metadata (partition,
context, head ownership, chain tables, split sizes, segment plan, byte
queries) is :mod:`.cake_kcp_plan` / :mod:`.cake_bytes`.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.distributed as dist

from . import cake_bytes, cake_kcp_plan, cake_l2n
from . import cake_options as _opt
from .cake_bytes import TICKET_SITES, TICKET_SLOT_BYTES
from .cake_options import (
    CakeGDNNondetKcpOptions,
    make_options,
    validate_heads,
    validate_ring,
)

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
KCP_SAVED_VERSION = (
    2  # saved set {S_in [, c1 receive tensor, M_seg] [, forward tapes] [, qk_rstd]}
)
COLLECTIVES = ("c1", "c2", "b1", "b2")
COLS_RULE_CHAINS = 24  # by_chains: <= 24 concurrent 128-column state chains -> c8, otherwise c16; map launches stay c16

# Program names of the dispatch table and the device program each one launches.
PROGRAMS = (
    "fwd_prep",  # #1 chunk-local prep: gcum, gamma, A^-1 (Akk), w, u, Aqk, rowscale, qscale
    "kcp_fwdh_offset",  # #2 program=offset: walk from zero, final state -> compact tiles (B_f / b_seg); no tapes
    "kcp_fwdh_tape",  # #2 program=tape: walk from an indexed initial state, h / v_new tapes
    "kcp_fwdh_out",  # #2 program=out: the tape walk with #5 folded into each chunk step, o instead of the tapes
    "kcp_fwdh_out_tape",  # #2 program=out_tape: the output walk that also writes the h / v_new tapes
    "kcp_transfer_walk",  # #3: FP32-class transfer M_f^T (TF32 hi/lo; init_mode = 1 / use_u = 1: the FP32-class offset)
    "kcp_affine_merge",  # #4: exact FP32 affine chains (state / map / transposed modes), reused from the KDA package
    "fwd_o",  # #5: o = scale e^G q h_c + (Aqk o mask) v_new
    "bwd_prep",  # #6: do_g, dv_intra
    "kcp_dhu_offset",  # #7 cut walk: reverse walk from zero over chunks c_start .. 0, final adjoint -> d_f / d_seg
    "kcp_dhu_tape",  # #7 program=tape: reverse walk from an indexed final adjoint, dh / dv_new tapes
    "kcp_dhu_cut_start",  # #7 cut table: c_start of each walked (segment, value head) (exact-zero rule)
    "bwd_dqkg",  # #8: dq, dk (grouped FP32 reduction), dv, dbeta, dg partials
    "gate_epilogue",  # #9: dg from the partials (reverse cumsum + chunk term), dbeta pass-through
    "fwd_prep_l2n",  # #1 with the in-kernel Q / K L2 norm: qn / kn / rstd first, then #1's body on qn / kn
    "bwd_dqkg_l2n",  # #8 on qn / kn with the norm's pullback before the BF16 rounding of dq / dk (group_reduce = 1)
    "qk_apply",  # qn = bf16(q * r), kn likewise, from the saved rstd (backward of save_mode="full")
    "fwd_prep_replay_l2n",  # #1_l2n reading the forward's saved Akk tape (light replay of save_mode="light_akk")
)
# The in-kernel Q / K L2 norm programs (``use_qk_l2norm_in_kernel``): #1 and #8 run as their ``_l2n`` siblings, and
# ``qk_apply`` rebuilds ``qn`` / ``kn`` where the backward has no #1 replay and no kept hand-off (save_mode="full").
L2N_PROGRAMS = ("fwd_prep_l2n", "bwd_dqkg_l2n", "qk_apply")
# The solve-free light replay of #1_l2n (``save_mode="light_akk"``): this package's own local program (no det twin; a
# chunk-local #1 form, so its library carries no ``_kcp``), launched in place of ``fwd_prep_l2n`` by the light replay
# of a saved set that keeps the forward's Akk tape (:data:`AKK_REPLAY_OF`).
AKK_REPLAY_PROGRAMS = ("fwd_prep_replay_l2n",)
LOCAL_PROGRAMS = (
    "fwd_prep",
    "fwd_o",
    "bwd_prep",
    "bwd_dqkg",
    "gate_epilogue",
    *L2N_PROGRAMS,
    *AKK_REPLAY_PROGRAMS,
)
WALK_PROGRAMS = (
    "kcp_fwdh_offset",
    "kcp_fwdh_tape",
    "kcp_fwdh_out",
    "kcp_fwdh_out_tape",
    "kcp_transfer_walk",
    "kcp_dhu_offset",
    "kcp_dhu_tape",
    "kcp_dhu_cut_start",
)
TART_MODULE_OF = {
    "fwd_prep": "gdn_chunk_det_train_fwd_prep_v1",
    "kcp_fwdh_offset": "gdn_chunk_det_fwdh_v1_kcp",
    "kcp_fwdh_tape": "gdn_chunk_det_fwdh_v1_kcp",
    "kcp_fwdh_out": "gdn_chunk_fwdh_out_v2_kcp",  # non-deterministic only: no det twin ships it
    "kcp_fwdh_out_tape": "gdn_chunk_fwdh_out_tape_v2_kcp",  # non-deterministic only: no det twin ships it
    "kcp_transfer_walk": "gdn_chunk_det_transfer_walk_v1_kcp",
    "kcp_affine_merge": "kda_chunk_kcp_affine_merge_fp32_v1",
    "fwd_o": "gdn_chunk_det_train_fwd_o_v1",
    "bwd_prep": "gdn_chunk_det_train_bwd_prep_v1",
    "kcp_dhu_offset": "gdn_chunk_dhu_cut_v1_kcp",  # the cut walk: non-deterministic only, no det twin ships it
    "kcp_dhu_tape": "gdn_chunk_det_dhu_v1_kcp",
    "kcp_dhu_cut_start": "gdn_chunk_dhu_cut_v1_kcp",  # the cut walk's table program (same module)
    "bwd_dqkg": "gdn_chunk_det_train_bwd_dqkg_v1",
    "gate_epilogue": "gdn_chunk_det_train_gate_epilogue_v1",
    "fwd_prep_l2n": "gdn_chunk_det_train_fwd_prep_l2n_v1",
    "bwd_dqkg_l2n": "gdn_chunk_det_train_bwd_dqkg_l2n_v1",
    "qk_apply": "gdn_chunk_det_train_qk_apply_v1",
    "fwd_prep_replay_l2n": "gdn_chunk_train_fwd_prep_replay_l2n_v1",  # this package's own (no det twin)
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
        "c_start",
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
# the output walk: the tape walk's inputs (no final-state pool, no tape) and #5's (q, aqk, qscale -> o)
PROGRAM_ARGS["kcp_fwdh_out"] = (
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
    "tile_stride",
    "q",
    "aqk",
    "qscale",
    "o",
)
# the tape-and-output walk: the output walk's arguments and the tape walk's h / v_new tapes
PROGRAM_ARGS["kcp_fwdh_out_tape"] = PROGRAM_ARGS["kcp_fwdh_out"] + ("h_tape", "v_new")
# the tape walk reads no cut table (it walks every chunk); the cut table program reads three of the walk's tapes
PROGRAM_ARGS["kcp_dhu_tape"] = PROGRAM_ARGS["kcp_dhu_offset"][:-1]
PROGRAM_ARGS["kcp_dhu_cut_start"] = (
    "gamma",
    "rowscale",
    "w",
    "chunk_len",
    "num_heads",
    "seg_chunk_start",
    "num_seqs",
    "walk_seg",
    "c_start",
)
# The in-kernel Q / K L2 norm: the extra logical names of the two siblings over #1 / #8 and the arguments of
# ``qk_apply``.  #1_l2n reads the RAW ``q`` / ``k`` (TMA, as #1) and writes ``qn`` / ``kn`` (contiguous
# ``[T_r, H_qk, 128]``) and ``q_rstd`` / ``k_rstd`` (the two planes of ``qk_rstd``, token stride ``rstd_tstride =
# H_qk``); #8_l2n reads ``q`` / ``k`` = ``qn`` / ``kn`` through #8's TMA and the raw rows ``q_raw`` / ``k_raw``
# (token strides ``q_raw_tstride`` / ``k_raw_tstride`` in elements: packed-QKV views pass through) with the rstd
# planes.  Norm-off launches pass exactly the v1 values.
PROGRAM_ARGS_L2N: dict[str, tuple[str, ...]] = {
    "fwd_prep_l2n": ("qn", "kn", "q_rstd", "k_rstd", "rstd_tstride"),
    "bwd_dqkg_l2n": (
        "q_raw",
        "k_raw",
        "q_raw_tstride",
        "k_raw_tstride",
        "rstd_tstride",
        "q_rstd",
        "k_rstd",
    ),
}
PROGRAM_ARGS["fwd_prep_l2n"] = (
    PROGRAM_ARGS["fwd_prep"] + PROGRAM_ARGS_L2N["fwd_prep_l2n"]
)
PROGRAM_ARGS["bwd_dqkg_l2n"] = (
    PROGRAM_ARGS["bwd_dqkg"][:-1]
    + PROGRAM_ARGS_L2N["bwd_dqkg_l2n"]
    + (
        "ticket",
    )  # the persistent #8's ticket stays last (the bwd_dqkg launch site's slot)
)
PROGRAM_ARGS["qk_apply"] = (
    "q_raw",
    "k_raw",
    "q_raw_tstride",
    "k_raw_tstride",
    "q_rstd",
    "k_rstd",
    "rstd_tstride",
    "num_qk_heads",
    *_LOCAL_TABLE_ARGS,
    "qn",
    "kn",
)
# the norm-off program each sibling replaces (the launch site: its ticket slot, its place in the sequence)
L2N_SIBLING_OF = {"fwd_prep": "fwd_prep_l2n", "bwd_dqkg": "bwd_dqkg_l2n"}
# The solve-free replay: #1_l2n's arguments, ``akk`` read (the saved tape) instead of written
PROGRAM_ARGS["fwd_prep_replay_l2n"] = PROGRAM_ARGS["fwd_prep_l2n"]
# the program the light replay of an AKK_SAVE_MODES set launches in place of #1_l2n
AKK_REPLAY_OF = {"fwd_prep_l2n": "fwd_prep_replay_l2n"}
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
FWDH_PROGRAMS = (
    "kcp_fwdh_offset",
    "kcp_fwdh_tape",
    "kcp_fwdh_out",
    "kcp_fwdh_out_tape",
)
DHU_PROGRAMS = ("kcp_dhu_offset", "kcp_dhu_tape")
VH64_SUFFIX = "_vh64"


# The architectures whose #7 tape walk takes the vh = 64 form at every load (measured faster per launch key there);
# elsewhere (sm_107a) the tape walk keeps the two-wave rule of :func:`dhu_vh`.
DHU_TAPE_VH64_ARCHES = ("sm_100a", "sm_103a")

# The architectures where a long-chain #7 walk (:func:`walk_dhu_long_chain`: a segment of at least
# ``DHU_LONG_CHAIN_MIN_CHUNKS`` chunks, and a two-CTA grid that ends later than the one-CTA grid) keeps the two-wave
# rule of :func:`dhu_vh` for both forms (measured slower in the vh = 64 form there).
DHU_LONG_CHAIN_ARCHES = ("sm_100a",)
DHU_LONG_CHAIN_MIN_CHUNKS = 400


def walk_vh64_fits(
    chains: Sequence[tuple[int, int]], num_heads: int, sm_count: int
) -> bool:
    """Whether the two-CTA grid of a walk launch finishes within its longest chain: the LPT makespan
    (:func:`cake_kcp_plan.lpt_makespan`) of ``2 * num_heads`` CTAs per segment over ``2 * sm_count`` slots (two
    CTAs per SM) is at most the longest segment's chunk count.  ``chains`` = the launch's ``(chunks, count)``
    multiset; plan integers only."""
    chains = [(int(c), int(n)) for c, n in chains if int(c) > 0 and int(n) > 0]
    if not chains or int(sm_count) <= 0:
        return False
    two = 2 * int(num_heads)
    makespan = cake_kcp_plan.lpt_makespan(
        [(c, two * n) for c, n in chains], 2 * int(sm_count)
    )
    return makespan <= max(c for c, _ in chains)


def walk_dhu_long_chain(
    chains: Sequence[tuple[int, int]], num_heads: int, sm_count: int
) -> bool:
    """Whether a #7 walk launch is a long-chain walk of :func:`dhu_vh`: its longest segment has at least
    :data:`DHU_LONG_CHAIN_MIN_CHUNKS` chunks and the LPT makespan (:func:`cake_kcp_plan.lpt_makespan`) of its
    two-CTA grid (``2 * num_heads`` CTAs per segment) over ``2 * sm_count`` slots is larger than the LPT makespan
    of its one-CTA grid (``num_heads`` CTAs per segment) over the same slots.  ``chains`` = the launch's
    ``(chunks, count)`` multiset; plan integers only."""
    chains = [(int(c), int(n)) for c, n in chains if int(c) > 0 and int(n) > 0]
    if not chains or int(sm_count) <= 0:
        return False
    if max(c for c, _ in chains) < DHU_LONG_CHAIN_MIN_CHUNKS:
        return False
    heads, slots = int(num_heads), 2 * int(sm_count)
    two = cake_kcp_plan.lpt_makespan([(c, 2 * heads * n) for c, n in chains], slots)
    one = cake_kcp_plan.lpt_makespan([(c, heads * n) for c, n in chains], slots)
    return two > one


def fwdh_vh(
    n_walk: int,
    num_heads: int,
    sm_count: int,
    forced: int | None = None,
    fits: bool = False,
) -> int:
    """The #2 form of one launch: ``64`` when ``n_walk * HV <= sm_count`` (the KCP one-wave regime) or when ``fits``
    (:func:`walk_vh64_fits` of the launch's chains: the two-CTA grid finishes within its longest chain), else
    ``128``; ``forced`` (test knob ``fwdh_vh``) overrides the rule."""
    if forced is not None:
        return int(forced)
    return 64 if int(n_walk) * int(num_heads) <= int(sm_count) or fits else 128


def dhu_vh(
    n_walk: int,
    num_heads: int,
    sm_count: int,
    forced: int | None = None,
    *,
    tape: bool = True,
    arch: str | None = None,
    long_chain: bool = False,
) -> int:
    """The #7 form of one launch: ``64`` for the offset walk (``tape=False``) and for the tape walk on the
    architectures of :data:`DHU_TAPE_VH64_ARCHES` (``arch`` ``None``, the fake backend, counts as one of them);
    elsewhere the tape walk runs vh = 64 when its two-CTA grid ``n_walk * HV * 2`` fits two waves at two CTAs per
    SM (``<= 4 * sm_count``), else ``128``.  On the architectures of :data:`DHU_LONG_CHAIN_ARCHES`, a long-chain
    walk (``long_chain``: :func:`walk_dhu_long_chain` of the launch's chains) of either form takes that two-wave
    rule too.  ``forced`` (test knob ``dhu_vh``) overrides the rule."""
    if forced is not None:
        return int(forced)
    two_wave = 64 if int(n_walk) * int(num_heads) * 2 <= 4 * int(sm_count) else 128
    if long_chain and arch in DHU_LONG_CHAIN_ARCHES:
        return two_wave
    if not tape or arch is None or arch in DHU_TAPE_VH64_ARCHES:
        return 64
    return two_wave


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
# In-kernel Q / K L2 norm: the normalized rows (BF16 ``[T_r, H_qk, 128]``, contiguous) and the FP32 ``r`` of q and k
# (``[2, T_r, H_qk]``: plane 0 = q, plane 1 = k).  Arena tapes of the call that writes them (never saved): the forward
# drops ``kn`` before its output (the tape walks were its last readers) and ``qn`` after #5 -- a hand-off forward
# (``reuse_recompute_tapes``) unbinds both instead and hands them to the backward with its tapes (``_KeptTapes.qk``);
# the backward drops both with the arena ``qk_rstd`` after #8 (or the host reduction).  A forward that keeps a saved set writes a fresh
# ``qk_rstd`` (the saved set's tensor, bound in ``CakeGDNNondetKcpCall.bufs``); the light replay's #1_l2n writes the arena one.
QK_TAPES: tuple[str, ...] = ("qn", "kn")
QK_RSTD = "qk_rstd"
# group_reduce="host" with the norm: the FP32 group sum and the pullback run per block of this many tokens (the same
# per-token order), so the host temporaries stay at a few block-sized FP32 planes instead of full [T_r, H_qk, 128] ones
HOST_L2N_BLOCK_TOKENS = 2048
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
# The forward exchange's pools a saved-context backward never touches (it reads the saved ``S_in`` and, under
# ``save_transfer``, the saved c1 receive tensor; its owner merge reads b1): not sized in the per-call workspace of such a
# backward (a caller-held workspace keeps every pool of its layout).
SAVED_BWD_DEAD_POOLS: tuple[str, ...] = ("c1_send", "c1_recv", "c2_send")
# The ring-exchange pools: sized only under ``exchange="ring"`` (the a2a forms never touch them; R2-L1 1a).
RING_POOLS: tuple[str, ...] = ("ring_out", "ring_in")
# The pools a call replaces with fresh (forward) or saved (backward) tensors under ``save_transfer``: the c1 receive
# tensor and the ``M_seg^T`` staging plane are saved-set tensors, so the workspace pools of those names are dead for
# that call and are not sized (``_prepare`` -> ``CakeGDNNondetKcpWorkspace.ensure(skip=...)``).  ``save_mode="full"`` saves the
# ``M_seg^T`` plane in either ``save_transfer`` form (:func:`_saves_m_seg`).
SAVE_TRANSFER_FRESH_POOLS: tuple[str, ...] = ("c1_recv", "stage_m")
# save_mode="full" without save_transfer: the saved M_seg^T plane only (its backward runs no transfer walk).
_FULL_FRESH_POOLS: tuple[str, ...] = ("stage_m",)


def _saves_m_seg(save_mode: str, save_transfer: bool, P: int) -> bool:
    """Whether the saved set carries ``M_seg`` (the forward's ``M_seg^T`` staging plane): at ``P > 1`` under
    ``save_transfer`` and under ``save_mode="full"``.  The backward's merges (``tmap_d_*``, ``tmap_m``,
    ``dcarry_*``) read that plane; the light backward rebuilds it in its replay (transfer walk), the kept hand-off
    carries the recompute forward's, and the ``"full"`` backward runs no transfer walk, so its saved set holds it."""
    return int(P) > 1 and (bool(save_transfer) or save_mode == "full")


def _fwd_fresh_pools(opts: CakeGDNNondetKcpOptions, save: bool) -> tuple[str, ...]:
    """The pools a forward replaces with fresh saved-set tensors (not sized for the call)."""
    if not save:
        return ()
    if opts.save_transfer:
        return SAVE_TRANSFER_FRESH_POOLS
    return _FULL_FRESH_POOLS if opts.save_mode == "full" else ()


# The saved-set tensor bound under each per-call buffer name (``CakeGDNNondetKcpCall.drop_bound`` forgets both at the last reader).
SAVED_TENSOR_OF_BUFFER: dict[str, str] = {
    "S_in": "S_in",
    "c1_recv": "c1_recv",
    "stage_m": "M_seg",
    QK_RSTD: QK_RSTD,
}
# Pools dead once the adjoint exchange is merged (a2a form): the forward exchange pools (the backward never exchanges
# through them) and b1 (sent and merged).  They go before the document-end adjoint walk that the overlap schedule
# issues under b2 -- the backward's peak -- while ``b2_send`` / ``b2_recv`` are in flight.  The staging planes stay:
# the dcarry merges read them (``m`` = stage_m, ``c`` = stage_b) up to the adjoint tape walk.
# The #7 offset walk's never-dereferenced tape pointer: the forward tape of the same shape and dtype, live by then
# (saved or replayed), so ``dh_tape`` is first allocated by the first tape walk -- after the adjoint merge.
DHU_OFFSET_TAPE_STANDIN = "h_tape"
# The #2 offset walk's never-dereferenced tape pointers ``h_tape`` / ``v_new``: the ``u`` tape (BF16 and contiguous like
# both, written by #1 before any walk), so the summaries bind neither and the first tape walk allocates both.
FWDH_OFFSET_TAPE_STANDIN = "u"
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
# ``save_mode="light_akk"`` (:data:`AKK_SAVE_MODES`, ``use_qk_l2norm_in_kernel`` only): the light set plus #1's BF16
# ``Akk`` tape (``[R, HV, 64]``, R * HV * 128 bytes).  The forward writes it into a fresh saved tensor (it leaves the
# arena with FWD_DEAD_AFTER_PREP otherwise); the light replay binds it and runs ``fwd_prep_replay_l2n`` -- #1_l2n
# without the K K^T Gram, the block inverse and the Akk stores, reading the tape into the WY operand tile -- whose
# other outputs are #1_l2n's bit for bit.  Opt-in only: the tape is live from the forward to the backward.
AKK_SAVE_MODES: tuple[str, ...] = _opt.SAVE_MODES_AKK
AKK_TAPES: tuple[str, ...] = ("akk",)
# The save modes whose forward keeps no h / v_new tape (the output walk, :func:`_fuse_o_enabled`)
LIGHT_FORWARD_SAVE_MODES: tuple[str, ...] = ("light", *AKK_SAVE_MODES)


def saved_tapes(save_mode: str) -> tuple[str, ...]:
    """The tape names a saved set of ``save_mode`` carries (in :data:`FWD_TAPES` order)."""
    if save_mode == "full":
        return FWD_TAPES
    return AKK_TAPES if save_mode in AKK_SAVE_MODES else LIGHT_TAPES


# Forward tapes an eager forward leaves bound in the workspace arena (#5 reads them; the others go at
# FWD_DEAD_AFTER_PREP / FWD_DEAD_BEFORE_O): the arena state a forward leaves, and the state a kept backward restores.
FWD_TAPES_ARENA_AFTER_FWD: tuple[str, ...] = tuple(
    n for n in FWD_TAPES if n not in FWD_DEAD_BEFORE_O and n not in FWD_DEAD_AFTER_PREP
)
# The tapes a fused forward (``kcp_fwdh_out``: #5 inside the walks) binds and leaves: no ``h`` / ``v_new`` tape.
FWD_TAPES_ARENA_AFTER_FUSED: tuple[str, ...] = tuple(
    n for n in FWD_TAPES_ARENA_AFTER_FWD if n not in ("h_tape", "v_new")
)


# ----------------------------------------------------------------------------------------------------------------
# Checkpoint-aware light backward (keyword ``reuse_recompute_tapes``: ``"auto"`` = the shipped default, kept tapes
# only when the node closes the checkpointed region, see _fwd_decision; ``True`` = always; ``False`` = the replay form)
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
# runs the ``save_mode="full"`` form of the backward on those tapes, so nothing is launched before #6 (with
# ``use_qk_l2norm_in_kernel`` the recompute's ``qn`` / ``kn`` go along, so no ``qk_apply`` either).  The kept
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
# Under ``graph="replay"`` (the arena hand-off) nothing is unbound: a captured graph bakes the arena's addresses and
# an unbound tape would reset every captured key of the workspace.  The kept tapes are the arena's own tensors (the
# workspace keeps them alive; the entry holds references only), and the entry is stamped with the workspace and its
# call count (``CakeGDNNondetKcpWorkspace.calls``: every forward / backward entry on a caller-held workspace adds one -- eager,
# captured or replayed) and the launch stream right after the hand-off forward.  A backward binds the entry only
# while that workspace has made no call since, still binds those very tensors and the backward runs on the same stream
# (:func:`_kept_stale`); anything in between -- another layer's forward or backward, an eager call on the same
# workspace, a user's ``ensure`` -- makes it stale and the backward runs the replay (same bits).  The recompute's
# hand-off and #5 decisions are part of its CUDA-graph key (:func:`_fwd_entry`), so the recompute captures its own
# graph (the hand-off form) beside the original forward's, and the kept backward its own graph beside the replay
# form's (the hand-off's layout is part of the backward's key, :func:`_kept_key_part`).

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


def _kcp_in_backward() -> bool:
    """Whether this thread runs inside the autograd engine's backward pass.  Torch's checkpoint recompute runs there
    (``use_reentrant=True`` and ``False`` alike), an original forward does not.  The one place the recompute handle
    reads the phase, so the detection can follow torch."""
    return _graph_task_id() != -1


class CakeGDNNondetKcpRecomputeHandle:
    """The boundary states of one checkpointed layer, carried from its original forward to torch's recompute of that
    forward (``cake_gdn_nondet_kcp(..., recompute_handle=handle)``), so the recompute runs without cross-rank
    communication.

    Create one handle per layer per micro-step (gradient accumulation: a new handle for every micro-step) and pass
    it as an argument of the checkpointed function, so the recompute sees the same object::

        def layer(x, handle):
            ...  # conv, split, gates
            return cake_gdn_nondet_kcp(q, k, v, g, beta, ctx, recompute_handle=handle)

        y = torch.utils.checkpoint.checkpoint(layer, x, CakeGDNNondetKcpRecomputeHandle(), use_reentrant=False)

    ``entries`` maps an operation key to that operation's boundary entry: ``"gdn"`` is this package's (the forward's
    incoming states ``S_in``, the very tensor the saved set holds, no copy); other packages that accept a handle
    with an ``entries`` dict (the causal convolution's halo rows, key ``"conv"``) use their own keys.  A handle
    holds one entry per key: it serves one GDN call and at most one convolution call; a layer with several
    convolution calls (separate Q / K / V convolutions) uses one handle per convolution call, and the GDN call
    shares one of them.  A forward that finds no entry is an original forward: it exchanges as always and stores
    its entry.  The recompute (inside the backward) finds the entry, checks it against the call and exchanges
    nothing; the entry stays for a second backward (``retain_graph``).  A forward outside the backward that finds an
    entry raises (a handle reused across micro-steps).  With nested checkpoint regions create the handles outside
    the outermost region: a handle created inside a region is a new, empty one in that region's recompute, whose
    forwards then exchange again (correct, but not communication-free).  The entries live as long as the handle --
    with the checkpoint frame that holds it as an argument, until the graph is freed; :meth:`nbytes` is the retained
    size (for GDN, ``cake_gdn_nondet_train_kcp_saved_bytes(ctx, ..., save_mode="light", save_transfer=False, with_conv=False)``)."""

    __slots__ = ("entries",)

    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}

    def clear(self) -> None:
        """Drop every entry (the retained states go with the last reference)."""
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


HANDLE_KEY = "gdn"  # this package's entry in a recompute handle


def _handle_key(
    T_r,
    H_qk,
    HV,
    n_frag,
    P,
    scale,
    exchange,
    precision_mode,
    offset_class,
    transfer_walk,
    qk_l2norm,
) -> tuple:
    """What a handle entry must match to stand in for the recompute's exchange: the problem and the programs.
    ``qk_l2norm`` (``use_qk_l2norm_in_kernel``): ``S_in`` is a function of the keys the walks read (``kn`` with the
    norm), so an entry of the other form is refused."""
    return (
        T_r,
        H_qk,
        HV,
        n_frag,
        P,
        scale,
        exchange,
        precision_mode,
        offset_class,
        transfer_walk,
        bool(qk_l2norm),
    )


def _handle_lookup(handle: Any) -> dict[str, Any] | None:
    """The handle's GDN entry when this forward is the recompute of the forward that stored it, ``None`` for an
    original forward (no entry yet: it exchanges and stores one).  An entry found outside the backward is a handle
    reused across micro-steps and raises; so does a torch without the phase binding.  A forward inside the backward
    that finds no entry is an original forward too -- one that runs inside an outer checkpoint's recompute (nested
    regions) -- and stores its entry as usual."""
    entries = getattr(handle, "entries", None)
    if not isinstance(entries, dict):
        raise TypeError(
            f"recompute_handle must carry an 'entries' dict (CakeGDNNondetKcpRecomputeHandle), got {type(handle).__name__}"
        )
    in_backward = _kcp_in_backward()
    if _GRAPH_TASK_ID_BINDING is None or _GRAPH_TASK_ID_USABLE is False:
        raise RuntimeError(
            "recompute_handle needs torch._C._current_graph_task_id to tell the recompute from the original forward; "
            "this torch does not provide it"
        )
    entry = entries.get(HANDLE_KEY)
    if entry is None:
        return None
    if not in_backward:
        raise RuntimeError(
            "recompute_handle: the handle already holds a 'gdn' entry outside the backward -- a handle serves one "
            "layer of one micro-step; create a new CakeGDNNondetKcpRecomputeHandle per micro-step"
        )
    return entry


# ----------------------------------------------------------------------------------------------------------------
# The recompute's dead output (keyword ``recompute_output``; shipped default ``"auto"``)
#
# ``torch.utils.checkpoint(use_reentrant=False)`` re-runs the checkpointed function inside ``.backward()`` for one
# purpose: to re-save the tensors the original forward saved (``_recomputation_hook.pack_hook`` matches every tensor
# the re-run saves to the original forward's holder).  The re-run's return value is discarded (``recompute_fn`` calls
# the function and returns nothing), and with torch's early stop (the default) the re-run ends by
# ``_StopRecomputationError`` as soon as the last holder is matched.  The backward of this module's autograd node
# reads ``dout``, the saved set (:meth:`CakeGDNNondetKcpSaved.tensors`: ``S_in``, ``c1_recv``, ``M_seg``, the saved tapes -- never
# ``out``) and the kept tapes; so when the node is the LAST saver of the checkpointed region, the recompute's #5
# (``fwd_o``, the output program) writes a tensor nothing ever reads.  ``recompute_output="auto"`` leaves that one
# launch out; the output tensor is still allocated (the same allocation sequence; torch sees the same shape).  The
# proof is read from torch's own frame and is strict:
#   0. the node is executable -- grad mode on at ``apply`` and an operand that requires grad (read by
#      :func:`cake_gdn_nondet_kcp` right before ``apply``: inside ``forward`` grad mode is always off).  Only then does
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
# forward (:class:`CakeGDNNondetKcpFunction`, the one that saves exactly those tensors right after) may skip;
# :func:`cake_gdn_nondet_train_kcp_forward` never does, and under ``graph="replay"`` the proof is read only inside a backward pass
# (``_graph_task_id() != -1``), before the CUDA-graph lookup: the decision is part of the recompute's graph key, so a
# graph that left #5 out is replayed only by a forward whose proof holds again.  No tensor content is read.

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


# q, k, v, g, beta: what CakeGDNNondetKcpFunction.forward saves besides the saved set
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
    arena geometry they were produced at and the graph task id.  ``qk``: with ``use_qk_l2norm_in_kernel`` the
    normalized rows ``qn`` / ``kn`` the recompute's #1_l2n wrote (:data:`QK_TAPES`, unbound from the arena like the
    tapes), so the backward runs no ``qk_apply``; empty without the norm.  ``ws`` / ``calls`` / ``stream``: ``None``
    for the eager hand-off (fresh tensors the entry owns); under ``graph="replay"`` (the arena hand-off) the workspace
    whose arena holds the tapes, its call count and the launch stream right after the hand-off forward
    (:func:`_kept_stale`)."""

    __slots__ = ("tapes", "stage_m", "geometry", "gid", "qk", "ws", "calls", "stream")

    def __init__(
        self,
        tapes: dict[str, torch.Tensor],
        stage_m: torch.Tensor | None,
        geometry: tuple[int, int, int, int] | None,
        gid: int,
        qk: dict[str, torch.Tensor] | None = None,
        ws: Any = None,
    ) -> None:
        self.tapes = tapes
        self.stage_m = stage_m
        self.geometry = geometry
        self.gid = int(gid)
        self.qk = {} if qk is None else qk
        self.ws = ws
        self.calls: int | None = None
        self.stream: int | None = None

    def clear(self) -> None:
        self.tapes = {}
        self.stage_m = None
        self.qk = {}
        self.ws = None


def _launch_stream(device: torch.device | None) -> int:
    """The launch stream of ``device`` (torch's current stream of the current CUDA device, the one the launch glue
    binds; ``0`` off CUDA): the arena hand-off's stamp."""
    if device is None or device.type != "cuda":
        return 0
    index = torch.cuda.current_device()
    return (
        int(_RAW_STREAM(index))
        if _RAW_STREAM is not None
        else int(torch.cuda.current_stream(index).cuda_stream)
    )


def _kept_stamped(kept: _KeptTapes) -> _KeptTapes:
    """A fresh arena hand-off entry on the tensors of ``kept`` (the hand-off forward's own entry), stamped with this
    backward pass's task id, the workspace's call count and the launch stream now (right after the hand-off forward of
    this call)."""
    out = _KeptTapes(
        dict(kept.tapes),
        kept.stage_m,
        kept.geometry,
        _graph_task_id(),
        dict(kept.qk),
        ws=kept.ws,
    )
    out.calls = int(kept.ws.calls)
    out.stream = _launch_stream(kept.ws.device)
    return out


class _KeptLayout:
    """An arena hand-off as a captured hand-off forward's cached result keeps it (the template of every replay's
    entry): the workspace, the arena names of the tapes and the ``qn`` / ``kn`` rows, the ``stage_m`` plane's length
    and the geometry -- no tensor, so the graph cache pins no arena buffer after the workspace drops it.
    :func:`_kept_rebound` binds the arena's tensors again at each replay (which found the arena signature of the
    capture: the very buffers the captured graph wrote)."""

    __slots__ = ("names", "qk", "stage_m", "geometry", "ws")

    def __init__(self, kept: _KeptTapes) -> None:
        self.names = tuple(kept.tapes)
        self.qk = tuple(kept.qk)
        self.stage_m = None if kept.stage_m is None else int(kept.stage_m.numel())
        self.geometry = kept.geometry
        self.ws = kept.ws


def _kept_rebound(layout: _KeptLayout) -> _KeptTapes | None:
    """A fresh arena hand-off entry of ``layout`` on the arena's current tensors, stamped as :func:`_kept_stamped`
    stamps; ``None`` (no hand-off: the backward runs the replay form) when the arena binds a name no longer."""
    ws = layout.ws
    arena = ws._tapes
    tapes = {n: arena.get(TAPE_ALIASES.get(n, n)) for n in layout.names}
    qk = {n: arena.get(TAPE_ALIASES.get(n, n)) for n in layout.qk}
    stage_m = None
    if layout.stage_m is not None:
        pool = ws._pools.get("stage_m")
        if pool is None or pool.numel() < layout.stage_m:
            return None
        stage_m = pool if pool.numel() == layout.stage_m else pool[: layout.stage_m]
    if any(t is None for t in (*tapes.values(), *qk.values())):
        return None
    out = _KeptTapes(tapes, stage_m, layout.geometry, _graph_task_id(), qk, ws=ws)
    out.calls = int(ws.calls)
    out.stream = _launch_stream(ws.device)
    return out


KEPT_STALE_STATS: dict[str, int] = {
    "calls": 0,  # arena hand-offs refused: the workspace made a call between the hand-off forward and the backward
    "arena": 0,  # ... the arena no longer binds the kept tensors (re-grown, released or re-bound)
    "stream": 0,  # ... the backward runs on another stream than the hand-off forward
}


def _kept_stale(entry: _KeptTapes) -> str | None:
    """Why an arena hand-off (``graph="replay"``) can no longer stand in for the replay, ``None`` while it can: the
    workspace that holds the tapes made no call since the hand-off forward (its call count is the stamp), its arena
    still binds the very kept tensors (and the ``stage_m`` pool its plane), and the backward launches on the stream
    the forward did.  An eager hand-off (``ws`` ``None``: fresh tensors the entry owns) is never stale.  Host
    metadata only (identities, counters, the stream handle)."""
    ws = entry.ws
    if ws is None:
        return None
    if entry.calls is None or ws.calls != entry.calls:
        return "calls"
    tapes = ws._tapes
    for name, t in (*entry.tapes.items(), *entry.qk.items()):
        if tapes.get(TAPE_ALIASES.get(name, name)) is not t:
            return "arena"
    if entry.stage_m is not None:
        pool = ws._pools.get("stage_m")
        if pool is None or pool.data_ptr() != entry.stage_m.data_ptr():
            return "arena"
    if entry.stream != _launch_stream(ws.device):
        return "stream"
    return None


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
    """The entry of ``key`` for the backward that binds it (``None``: the replay runs).  A stale arena hand-off
    (:func:`_kept_stale`) is dropped and counted as a miss (``KEPT_STALE_STATS`` says why)."""
    entry = _KEPT.pop(key, None)
    if entry is not None:
        stale = _kept_stale(entry)
        if stale is not None:
            entry.clear()
            KEPT_STALE_STATS[stale] += 1
            entry = None
    KEPT_STATS["take" if entry is not None else "miss"] += 1
    return entry


def tape_spec(
    name: str, *, R: int, NC_pad: int, HV: int, T: int, H_qk: int | None = None
) -> tuple[tuple[int, ...], torch.dtype]:
    """``(shape, dtype)`` of tape ``name`` for the padded geometry (``H_qk``: the Q / K head count of the in-kernel
    L2 norm tapes :data:`QK_TAPES` / :data:`QK_RSTD`, token rows ``T``)."""
    bf, f32 = torch.bfloat16, torch.float32
    if name in QK_TAPES or name == QK_RSTD:
        if H_qk is None:
            raise KeyError(f"tape {name!r} needs H_qk")
        return ((T, int(H_qk), D), bf) if name in QK_TAPES else ((2, T, int(H_qk)), f32)
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
            f"operands carry {T} tokens, the CakeGDNNondetKcpContext of rank {ctx.rank} holds T_r={ctx.T_r}"
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


def _dense_strides(shape: Sequence[int]) -> tuple[int, ...]:
    out, acc = [], 1
    for n in reversed(tuple(shape)):
        out.append(acc)
        acc *= max(int(n), 1)
    return tuple(reversed(out))


def _packed_rows(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, H_qk: int, HV: int
) -> torch.Tensor | None:
    """The packed ``[..., T, (2 H_qk + HV) * 128]`` row ``q`` / ``k`` / ``v`` are the head views of, or ``None``.

    Packed: one dtype, device and storage; ``k`` starts ``H_qk * 128`` elements after ``q`` and ``v`` ``H_qk * 128``
    after ``k``; each is ``[..., T, heads, 128]`` with dense ``[heads, 128]`` rows and token stride
    ``(2 H_qk + HV) * 128``.  Pointers and strides only: no tensor content is read.  The returned tensor is a dense
    view of the whole row over ``q``'s storage."""
    C = (2 * H_qk + HV) * D
    if not (q.dtype == k.dtype == v.dtype and q.device == k.device == v.device):
        return None
    if not (q.shape[:-2] == k.shape[:-2] == v.shape[:-2]) or q.dim() < 3:
        return None
    if (q.shape[-2], k.shape[-2], v.shape[-2]) != (H_qk, H_qk, HV) or q.shape[-1] != D:
        return None
    if q.numel() == 0:
        return None
    lead = tuple(q.shape[:-2])
    shape = (*lead, C)
    want = _dense_strides(shape)
    for x in (q, k, v):
        if x.dim() != len(lead) + 2 or x.shape[-1] != D:
            return None
        for axis, n in enumerate(lead):
            if n != 1 and x.stride(axis) != want[axis]:
                return None
        if x.stride(-1) != 1 or (x.shape[-2] != 1 and x.stride(-2) != D):
            return None
    st = q.untyped_storage().data_ptr()
    if not (
        k.untyped_storage().data_ptr() == st and v.untyped_storage().data_ptr() == st
    ):
        return None
    if (k.storage_offset(), v.storage_offset()) != (
        q.storage_offset() + H_qk * D,
        q.storage_offset() + 2 * H_qk * D,
    ):
        return None
    return q.as_strided(shape, want, q.storage_offset())


def _row_span(x: torch.Tensor) -> torch.Tensor:
    """A ``[T, heads, 128]`` gradient output as #8 receives it: itself when contiguous, else (a head view of the
    packed ``dQKV`` row) the contiguous 1-D span from its first to its last element.  The #8 binding requires a
    contiguous ``dq`` / ``dk`` / ``dv``; the kernel addresses them through the ``dqk_tstride`` / ``dv_tstride``
    scalars, so it writes exactly the view's elements."""
    if x.is_contiguous() or x.numel() == 0:
        return x
    n = (x.shape[0] - 1) * x.stride(0) + x.shape[1] * x.shape[2]
    return x.as_strided((n,), (1,), x.storage_offset())


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
    the private keys ``_call`` (the :class:`CakeGDNNondetKcpCall`) and ``_grid`` (the launch's grid x).  ``dtypes`` are the
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

    def begin_call(self, call: CakeGDNNondetKcpCall) -> None:
        """Hook at the start of a forward or backward call."""

    def end_call(self, call: CakeGDNNondetKcpCall) -> None:
        """Hook at the end of a call."""

    def launch_scope(self, call: CakeGDNNondetKcpCall) -> Any:
        """The context manager the forward / backward body of ``call`` runs in (its launches, eagerly or inside a
        capture), or ``None`` (the default): no scope."""
        return None

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
    """``args`` without its private keys (a leading ``_``; ``k[:1]`` is ``k.startswith("_")`` without a method call)."""
    return {k: v for k, v in args.items() if k[:1] != "_"}


def _registry() -> Any | None:
    """The package registry (:mod:`.cake_jit`: records, ``select_record``, ``launch_grid``), ``None`` without it."""
    try:
        from . import cake_jit  # noqa: PLC0415 - the registry; optional at import time
    except ImportError:
        return None
    return cake_jit


# The registry answers of l2n_records_shipped / fused_o_shipped, derived once per registry identity: they are read on
# every call and launch, and the registry is the package's generated record table (data, never changed at run time).
# The key is the registry module, its record table and its records_of_program function (a registry or a table put in
# their place -- a test's stand-in -- is another key); the entry holds the keyed objects, so their ids stay theirs.
_REGISTRY_ANSWERS: dict[tuple, tuple[bool, tuple]] = {}


def _registry_answer(question: tuple, jit: Any, derive: Callable[[], bool]) -> bool:
    modules = getattr(jit, "MODULES", None)
    lookup = getattr(jit, "records_of_program", None)
    try:
        size = len(modules)
    except TypeError:  # no record table: nothing to key on, derived per call
        return derive()
    key = (*question, id(jit), id(modules), size, id(lookup))
    hit = _REGISTRY_ANSWERS.get(key)
    if hit is None:
        hit = _REGISTRY_ANSWERS[key] = (bool(derive()), (jit, modules, lookup))
    return hit[0]


def l2n_records_shipped() -> bool:
    """Whether the registry ships a record of every in-kernel Q / K L2 norm program (:data:`L2N_PROGRAMS`): the
    kernel backend runs ``use_qk_l2norm_in_kernel=True`` only then; the fake backend emulates them either way."""
    jit = _registry()
    if jit is None:
        return False

    def derive() -> bool:
        try:
            return all(jit.records_of_program(p) for p in L2N_PROGRAMS)
        except ValueError:  # a registry that does not know the programs
            return False

    return _registry_answer(("l2n",), jit, derive)


def fused_o_shipped(arch: str | None = None) -> bool:
    """Whether the registry ships the output walk ``kcp_fwdh_out`` in both forms (the ``vh = 128`` record and the
    ``_vh64`` one) for ``arch`` (``None``: for any architecture, the fake backend's emulation): a forward folds #5
    into its tape walks only then (:func:`_fuse_o`), so a registry without them keeps every forward split."""
    return _walk_forms_shipped("kcp_fwdh_out", arch)


def fused_o_tape_shipped(arch: str | None = None) -> bool:
    """Whether the registry ships the tape-and-output walk ``kcp_fwdh_out_tape`` in both forms (the ``vh = 128``
    record and the ``_vh64`` one) for ``arch`` (``None``: for any architecture, the fake backend's emulation): a
    ``save_mode="full"`` forward folds #5 into its tape walks only then (:func:`_fuse_o_tape`), so a registry without
    them keeps the tape walk and #5."""
    return _walk_forms_shipped("kcp_fwdh_out_tape", arch)


def _walk_forms_shipped(program: str, arch: str | None) -> bool:
    """Whether the registry ships ``program`` in the one-CTA and the two-CTA form (``ctas_per_head`` 1 and 2) for
    ``arch`` (``None``: any architecture); ``False`` without a registry or for a program it does not know."""
    jit = _registry()
    if jit is None:
        return False

    def derive() -> bool:
        try:
            names = jit.records_of_program(program)
        except ValueError:  # a registry that does not know the program
            return False
        forms = {
            int(jit.MODULES[n].get("grid_rule", {}).get("ctas_per_head", 1))
            for n in names
            if arch is None or arch in jit.MODULES[n].get("arches", ())
        }
        return forms >= {1, 2}

    return _registry_answer(("walk_forms", program, arch), jit, derive)


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
            bool(args.get("_vh64_fits")),
        )
    elif name in DHU_PROGRAMS:
        vh = dhu_vh(
            grid_args.get("n_walk", 0),
            grid_args.get("num_heads", 0),
            args.get("_sm_count", 0),
            args.get("_dhu_vh"),
            tape=name == "kcp_dhu_tape",
            arch=args.get("_arch"),
            long_chain=bool(args.get("_dhu_long_chain")),
        )
    call = args.get("_call")
    jit = _registry()
    if name in L2N_PROGRAMS and not l2n_records_shipped():
        # an in-kernel L2 norm program the registry does not render yet: the record is the program name (the fake
        # backend's emulation; the kernel backend refuses it before any launch), the grid its host plan's rule
        record = RECORD_OF[name]
        grid = (int(args["_grid"]), 1, 1)
        if (
            jit is not None
            and grid_args
            and record in getattr(jit, "L2N_RECORD_PLANS", {})
        ):
            gx, gy, gz = (int(x) for x in jit.launch_grid(record, **grid_args))
            grid = (gx, gy, gz)
        if call is not None:
            call.records.append((record, grid))
        return record, values, grid, False
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


def _launch_key(name: str, args: dict[str, Any], cols_per_cta: int) -> tuple:
    """Every input :func:`resolve_launch` reads besides the launch's tensors: the program, the static knobs, the grid
    rule's arguments (in the call site's order), the walk-form inputs, the merge's columns and the caller's grid x.

    The launch cache is valid only because :func:`resolve_launch` derives the record, the grid, the test flag and the
    merge's ``cols_per_cta`` from these scalars alone and reads nothing from the launch's tensors: an input added to
    :func:`resolve_launch` (a new grid-rule argument, for example) must be added to this key as well."""
    grid_args = args.get("_grid_args")
    get = args.get
    return (
        name,
        get("_precision_mode"),
        get("_transfer_walk"),
        tuple(grid_args.items()) if grid_args else None,
        *map(get, _LAUNCH_KEY_ARGS),
        int(cols_per_cta),
    )


# the launch key's arguments after the grid rule's (in key order): see _launch_key
_LAUNCH_KEY_ARGS = (
    "_sm_count",
    "_fwdh_vh",
    "_dhu_vh",
    "_vh64_fits",
    "_dhu_long_chain",
    "_arch",
    "ncols",
    "cols_per_cta",
    "_grid",
)


def _resolve_cached(
    name: str, args: dict[str, Any], *, cols_per_cta: int = 8
) -> tuple[str, dict[str, Any], tuple[int, int, int], bool]:
    """:func:`resolve_launch` through the launch cache of the call's tables (``_Tables.launches``, one per context
    plan): the record, the grid, the test flag and the merge's ``cols_per_cta`` of a launch key are resolved once;
    every launch rebuilds its public values and appends ``(record, grid)`` to the call's records as
    :func:`resolve_launch` does."""
    call = args.get("_call")
    cache = getattr(getattr(call, "tb", None), "launches", None)
    if cache is None:
        return resolve_launch(name, args, cols_per_cta=cols_per_cta)
    key = _launch_key(name, args, cols_per_cta)
    hit = cache.get(key)
    if hit is None:
        record, values, grid, is_test = resolve_launch(
            name, args, cols_per_cta=cols_per_cta
        )
        cache[key] = (record, grid, is_test, values.get("cols_per_cta"))
        return record, values, grid, is_test
    record, grid, is_test, cols = hit
    values = _public_args(args)
    if cols is not None:
        values["cols_per_cta"] = cols
    call.records.append((record, grid))
    return record, values, grid, is_test


_L2N_NOT_SHIPPED = (
    "use_qk_l2norm_in_kernel=True needs the in-kernel Q / K L2 norm programs (fwd_prep_l2n, bwd_dqkg_l2n, qk_apply), "
    "which this package's registry does not ship yet; use backend='fake' (the torch emulation) or the norm-off route"
)


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
        call: CakeGDNNondetKcpCall = args["_call"]
        if name in L2N_PROGRAMS and not l2n_records_shipped():
            raise RuntimeError(_L2N_NOT_SHIPPED)
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
        elif name in DHU_PROGRAMS:
            # the #7 width rule reads the architecture (dhu_vh)
            args = {**args, "_arch": be.device_arch(device_index)}
        record, values, grid, is_test = _resolve_cached(
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

    def launch_scope(self, call: CakeGDNNondetKcpCall) -> Any:
        """The launch glue's stream scope (``cake_backend.launch_scope``: one stream binding -- the stream
        ``tvm_ffi.use_torch_stream()`` binds -- for the body's launches instead of one per launch); ``None`` when the
        glue has no scope."""
        scope = getattr(self.launcher(), "launch_scope", None)
        return None if scope is None else scope()


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
        call: CakeGDNNondetKcpCall = args["_call"]
        record, values, grid, _is_test = _resolve_cached(name, args)
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


MEMO_MAX = 4096  # entries of a plan's launch memo (_Tables.memo) before it is cleared and rebuilt on use


class _IndexedSpec:
    """The static part of the :class:`Indexed` views a plan builds from one pair of its tables: the normalised
    ``index`` / ``tiles`` tuples (the views' fields), the kernels' addressing of the tiles (derived at the first
    :meth:`CakeGDNNondetKcpCall.block`, by :meth:`Indexed.addressing`) and the device index table per device.  ``src`` holds the
    keyed source tuples, so their ids cannot be reused while the entry lives."""

    __slots__ = ("src", "index", "tiles", "addr", "dev")

    def __init__(self, index: tuple, tiles: tuple) -> None:
        self.src = (index, tiles)
        self.index = tuple(int(i) for i in index)
        self.tiles = tuple(tuple(int(t) for t in row) for row in tiles)
        self.addr: tuple[int, int, int, int] | None = None
        self.dev: dict[torch.device, torch.Tensor] = {}


def _chain_device_tables(
    tb: _Tables, ch: Chains, device: torch.device
) -> tuple[torch.Tensor, ...]:
    """The device tables of the merge chains ``ch`` (``chain_start`` int32; ``elem_m``, ``elem_c``, ``elem_y``,
    ``x0_index``, ``y_final`` int64), cached by content on the head plan (:meth:`_Tables.dev`)."""
    tag = ("chain", ch.content())
    i64 = torch.int64
    return (
        tb.dev(tag + ("start",), ch.chain_start, device),
        tb.dev(tag + ("m",), ch.elem_m, device, i64),
        tb.dev(tag + ("c",), ch.elem_c, device, i64),
        tb.dev(tag + ("y",), ch.elem_y, device, i64),
        tb.dev(tag + ("x0",), ch.x0_index, device, i64),
        tb.dev(tag + ("yf",), ch.y_final, device, i64),
    )


def _prefix(values) -> list[int]:
    out = [0]
    for v in values:
        out.append(out[-1] + int(v))
    return out


# ----------------------------------------------------------------------------------------------------------------
# Tables of one (ctx, HV) problem, derived from the plan module's head plan (``ctx.heads(HV)``)


class _Tables:
    """Everything the launch sequences read for one ``(ctx, HV)``: the segment plan, walk lists, the compact tile
    tables of every pool, the collective splits and the merge chains, derived once from ``cake_kcp_plan.CakeGDNNondetKcpContext``
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
        # launch key -> (record, grid, is_test, merge cols_per_cta): the launch cache of _resolve_cached (host data of
        # this plan, filled at the first launch of each key)
        self.launches: dict[tuple, tuple] = {}
        # the launches' static pieces of this plan, derived at their first use (host data, never a kernel argument
        # value of its own): the normalised pool tables of CakeGDNNondetKcpCall.indexed with their kernel addressing and device
        # index tables (_IndexedSpec), the walk lists' and the merge chains' device tables, the zero block's index
        # table and the collectives' split lists.  Every device table is the one ``dev`` returns (cached by content
        # on the head plan), so a launch receives the very tensors it received before.
        self.memo: dict[tuple, Any] = {}
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

    def _memo_put(self, key: tuple, value: Any) -> Any:
        # bound for callers that pass fresh tables per call; the plan's own are few
        if len(self.memo) >= MEMO_MAX:
            self.memo.clear()
        self.memo[key] = value
        return value

    def indexed_spec(self, index: Any, tiles: Any) -> _IndexedSpec | None:
        """The memo entry of an :class:`Indexed` view over the tables ``index`` / ``tiles`` -- keyed by the identity
        of the two tuples (immutable; the entry holds them, so their ids stay theirs); ``None`` for any other type
        (a list may change after its first use: such a view is derived per call as before)."""
        if type(index) is not tuple or type(tiles) is not tuple:
            return None
        spec = self.memo.get(("ix", id(index), id(tiles)))
        if spec is None:
            spec = _IndexedSpec(index, tiles)
            self._memo_put(("ix", id(index), id(tiles)), spec)
            self._memo_put(("ixn", id(spec.index), id(spec.tiles)), spec)
        return spec

    def walk_table(
        self, walk: Sequence[int], device: torch.device
    ) -> tuple[int, torch.Tensor]:
        """``(len, int32 device table)`` of the walk list ``walk`` (keyed by its content: the lists are short)."""
        key = ("walk", tuple(walk), device)
        hit = self.memo.get(key)
        if hit is None:
            values = [int(s) for s in walk]
            hit = self._memo_put(key, (len(values), self.dev("walk", values, device)))
        return hit

    def walk_fits(self, walk: Sequence[int], sm_count: int) -> bool:
        """:func:`walk_vh64_fits` of the walk list ``walk`` (its segments' chunk counts; keyed by its content like
        :meth:`walk_table`)."""
        key = ("fits", tuple(walk), int(sm_count))
        hit = self.memo.get(key)
        if hit is None:
            hit = self._memo_put(
                key, walk_vh64_fits(self._walk_chains(walk), self.H, sm_count)
            )
        return hit

    def walk_long_chain(self, walk: Sequence[int], sm_count: int) -> bool:
        """:func:`walk_dhu_long_chain` of the walk list ``walk`` (keyed by its content like :meth:`walk_fits`)."""
        key = ("long_chain", tuple(walk), int(sm_count))
        hit = self.memo.get(key)
        if hit is None:
            hit = self._memo_put(
                key, walk_dhu_long_chain(self._walk_chains(walk), self.H, sm_count)
            )
        return hit

    def _walk_chains(self, walk: Sequence[int]) -> list[tuple[int, int]]:
        """The ``(chunks, count)`` multiset of the walk list's segments (host plan tuple ``seg_chunk_start``)."""
        seg = self.hp.seg_chunk_start
        chains: dict[int, int] = {}
        for s in walk:
            c = int(seg[int(s) + 1]) - int(seg[int(s)])
            chains[c] = chains.get(c, 0) + 1
        return list(chains.items())

    def neg_index(self, device: torch.device) -> torch.Tensor:
        """The zero block's ``[NSEG]`` index table (every entry ``-1``) on the device."""
        key = ("neg", device)
        t = self.memo.get(key)
        if t is None:
            t = self._memo_put(key, self.dev("neg", [-1] * int(self.NSEG), device))
        return t

    def chain_tables(
        self, family: str, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        """The six device tables of the plan's merge chain family ``family`` (``chain_start`` int32; ``elem_m``,
        ``elem_c``, ``elem_y``, ``x0_index``, ``y_final`` int64), uploaded in this order at first use."""
        key = ("chain", family, device)
        hit = self.memo.get(key)
        if hit is None:
            hit = self._memo_put(
                key, _chain_device_tables(self, self.chains[family], device)
            )
        return hit

    def split_lists(self, coll: str) -> tuple[list[int], list[int]]:
        """:meth:`split_elems` of ``coll``, derived once (the caller must not mutate the lists)."""
        key = ("split", coll)
        hit = self.memo.get(key)
        if hit is None:
            hit = self._memo_put(key, self.split_elems(coll))
        return hit

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


def _ticket_slots(device: torch.device) -> torch.Tensor:
    """Zeroed u32 ticket words, one 128-byte slot per persistent launch site (:data:`TICKET_SITES`): one slot of
    slack, then the 128-byte-aligned view (the CUDA caching allocator hands out 512-byte-aligned blocks, offset 0;
    the CPU allocator only 64-byte-aligned ones, offset 0 or 16 words)."""
    words = len(TICKET_SITES) * TICKET_SLOT_WORDS
    base = torch.zeros((words + TICKET_SLOT_WORDS,), dtype=torch.uint32, device=device)
    skew = (-base.data_ptr() % TICKET_SLOT_BYTES) // base.element_size()
    return base[skew : skew + words]


_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)
_IS_CAPTURING = getattr(torch._C, "_cuda_isCurrentStreamCapturing", None)


def _call_tickets(
    ctx: Any, device: torch.device
) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]] | None:
    """The ticket slots of a call without a caller-owned workspace: one zeroed set per ``(context, device, launch
    stream)`` -- the stream the launch glue binds, torch's current stream of the current device -- kept in the
    context's cache beside its identity tile with the slot views the calls hand to the kernels, and shared by every
    such call on that stream.  Every persistent launch leaves its word at zero and the launches of one stream run in
    issue order, so a later launch never meets a word in use; calls on another stream get their own set.  ``None``
    (fresh slots per call, as before) under a CUDA stream capture -- a captured graph keeps the addresses it baked --
    or without a context cache.

    The sharing relies on every persistent launch of such a call staying on the caller's stream (the stream in this
    key): a launch moved to a side stream would no longer be ordered with the other calls' launches and must get its
    own slots (or be ordered with this stream by an event) before it shares this set."""
    caches = getattr(ctx, "_caches", None)
    if not isinstance(caches, dict):
        return None
    if device.type == "cuda":
        capturing = (
            _IS_CAPTURING()
            if _IS_CAPTURING is not None
            else torch.cuda.is_current_stream_capturing()
        )
        if capturing:
            return None
        index = torch.cuda.current_device()
        stream = (
            int(_RAW_STREAM(index))
            if _RAW_STREAM is not None
            else int(torch.cuda.current_stream(index).cuda_stream)
        )
    else:
        index, stream = -1, 0  # the host: one ordered queue
    key = ("kcp_call_tickets", str(device), index, stream)
    hit = caches.get(key)
    if hit is None:
        hit = caches[key] = (_ticket_slots(device), {})
    return hit


def _cut_table(ctx: Any, n: int, device: torch.device) -> torch.Tensor:
    """The int32 cut table ``c_start`` of the #7 offset walk (``kcp_dhu_cut_start`` writes the entries of the walked
    (segment, value head) pairs, the walk then reads them; no other entry is read, none is zeroed, the host reads
    none): one buffer per ``(context, device, launch stream)`` in the context's cache, grown to the largest ``n`` asked
    and shared by the calls on that stream.  A start launch and its walk are issued back to back by
    :meth:`CakeGDNNondetKcpCall.walk` (deferred together on one side stream, if ever), so a later launch never meets an
    entry in use.  Fresh per call under a CUDA stream capture (a captured graph keeps the address it baked) or
    without a context cache, as :func:`_call_tickets`."""
    n = max(int(n), 1)
    caches = getattr(ctx, "_caches", None)
    if not isinstance(caches, dict):
        return _alloc((n,), torch.int32, device)
    if device.type == "cuda":
        capturing = (
            _IS_CAPTURING()
            if _IS_CAPTURING is not None
            else torch.cuda.is_current_stream_capturing()
        )
        if capturing:
            return _alloc((n,), torch.int32, device)
        index = torch.cuda.current_device()
        stream = (
            int(_RAW_STREAM(index))
            if _RAW_STREAM is not None
            else int(torch.cuda.current_stream(index).cuda_stream)
        )
    else:
        index, stream = -1, 0  # the host: one ordered queue
    key = ("kcp_cut_table", str(device), index, stream)
    table = caches.get(key)
    if table is None or table.numel() < n:
        table = caches[key] = _alloc((n,), torch.int32, device)
    return table if table.numel() == n else table[:n]


def _chunk_tokens(ctx: Any, chunk_len: Sequence[int]) -> int:
    """The tokens the context's chunk table ``chunk_len`` covers (its sum; the in-kernel L2 norm's coverage check).
    A context's table is an immutable tuple: its sum is derived once and kept in the context's cache with the table
    it was taken over (a context without a cache, or a table of another type, sums per call)."""
    caches = getattr(ctx, "_caches", None)
    if not isinstance(caches, dict) or type(chunk_len) is not tuple:
        return sum(int(n) for n in chunk_len)
    hit = caches.get(("kcp_chunk_tokens",))
    if hit is None or hit[0] is not chunk_len:
        hit = caches[("kcp_chunk_tokens",)] = (
            chunk_len,
            sum(int(n) for n in chunk_len),
        )
    return hit[1]


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


class CakeGDNNondetKcpWorkspace:
    """The transient buffers of the KCP route on one rank: the exchange pools (flat, ``state_dtype``: ``c1_send``,
    ``c1_recv``, ``c2_send``; ``b1_send``, ``b1_recv``, ``b2_send``, ``b2_recv`` for the backward; at ``P > 1`` the
    staging planes ``stage_b`` / ``stage_m`` and the ``carry`` plane; one identity tile ``eye``) and the tape arena
    (:data:`FWD_TAPES` / :data:`BWD_TAPES` of the padded geometry ``(R, NC_pad, HV)``, allocated on first use;
    ``dv_new`` is the ``dv_intra`` buffer, :data:`TAPE_ALIASES`).  The
    saved-set tensors (``S_in`` = the c2 output, the c1 receive tensor under ``save_transfer``, ``M_seg`` under
    ``save_transfer`` or ``save_mode="full"`` at ``P > 1``, the forward tapes under ``save_mode="full"``) are never
    workspace slices: the
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
        # site -> (the ticket tensor, its slot view): the views of the slots in use, made once per ticket tensor
        self._ticket_views: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.geometry: tuple[int, int, int, int] | None = None  # (R, NC_pad, HV, T)
        # forward / backward entries made on this caller-held workspace (eager, captured or replayed): the stamp of an
        # arena hand-off under graph="replay" (:func:`_kept_stale`)
        self.calls = 0
        # the Q / K head count of the in-kernel L2 norm tapes (QK_TAPES / QK_RSTD: [T, H_qk, ...]); the last ensure()'s
        self.qk_heads: int | None = None
        if ctx is not None:
            if H_qk is None or HV is None:
                raise ValueError(
                    "CakeGDNNondetKcpWorkspace: H_qk and HV are required with ctx"
                )
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
        skip_empty: bool = False,
    ) -> _Tables:
        """Size the pools for ``(ctx, HV)`` and bind the tape geometry; returns the tables (``P``: the segments
        override of the plan).  ``skip``: pool names this call never touches (:func:`_prepare` passes the ring
        pools under ``exchange="a2a"`` and the pools the call replaces with fresh / saved tensors, ``c1_recv`` under
        ``save_transfer`` and ``stage_m`` when the saved set carries ``M_seg``); a skipped pool is not allocated, a
        held one is left as it is, and a later ``ensure()`` without the name sizes it.  ``skip_empty``: a pool of
        size 0 (e.g. the staging and carry planes at ``P = 1``) is not allocated either -- :meth:`pool` makes the
        empty tensor if a call ever asks for it.  A workspace built from ``(ctx, H_qk, HV)`` sizes every pool."""
        device = (
            torch.device(device) if device is not None else torch.device(ctx.device)
        )
        if self.device is None:
            self.device = device
        elif self.device != device:
            raise ValueError(
                f"CakeGDNNondetKcpWorkspace: lives on {self.device}, the call is on {device}"
            )
        validate_heads(H_qk, HV)
        tb = self.tables(ctx, HV, fused_walks=fused_walks, P=P)
        sizes = tb.memo.get(("pool_sizes", bool(backward)))
        if sizes is None:
            sizes = tb._memo_put(
                ("pool_sizes", bool(backward)), tuple(tb.pool_sizes(backward).items())
            )
        for name, n in sizes:
            if name in skip:
                continue
            cur = self._pools.get(name)
            if cur is None and n == 0 and skip_empty:
                continue
            if cur is None or cur.numel() < n:
                self._pools[name] = (
                    _identity_tile(ctx, n, self.state_dtype, device)
                    if name == "eye"
                    else _alloc((n,), self.state_dtype, device)
                )
        if (
            self._tickets is None
        ):  # once per workspace: the kernels keep every word at zero afterwards
            self._tickets = _ticket_slots(device)
        geometry = (int(ctx.R), int(ctx.NC_pad), int(HV), int(ctx.T_r))
        if self.geometry != geometry:
            self._tapes.clear()  # another padded geometry: the tapes are reallocated on first use
            self.geometry = geometry
        if self.qk_heads != int(H_qk):
            for name in (*QK_TAPES, QK_RSTD):  # the L2 norm tapes are [T, H_qk, ...]
                self._tapes.pop(name, None)
            self.qk_heads = int(H_qk)
        return tb

    def ticket(self, site: str) -> torch.Tensor:
        """The 128-byte u32 ticket slot of persistent launch site ``site`` (one of :data:`TICKET_SITES`): the
        kernel's ``ticket`` argument, zero before and after every launch, never read on the host."""
        if site not in TICKET_SITES:
            raise KeyError(
                f"CakeGDNNondetKcpWorkspace: {site!r} is not a persistent launch site {TICKET_SITES}"
            )
        tickets = self._tickets
        if tickets is None:
            raise KeyError(
                "CakeGDNNondetKcpWorkspace: no ticket slots yet (ensure() first)"
            )
        hit = self._ticket_views.get(site)
        if hit is not None and hit[0] is tickets:
            return hit[1]
        i = TICKET_SITES.index(site) * TICKET_SLOT_WORDS
        view = tickets[i : i + TICKET_SLOT_WORDS]
        self._ticket_views[site] = (tickets, view)
        return view

    def pool(self, name: str, n: int | None = None) -> torch.Tensor:
        t = self._pools.get(name)
        if t is None:
            if n not in (None, 0) or self.device is None:
                raise KeyError(
                    f"CakeGDNNondetKcpWorkspace: pool {name!r} was not sized for this call (ensure() first)"
                )
            t = self._pools[name] = _alloc((0,), self.state_dtype, self.device)
        # a pool of exactly ``n`` elements is its own first-``n`` view (same storage, offset, shape and stride)
        return t if n is None or t.numel() == n else t[:n]

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
                raise KeyError(
                    "CakeGDNNondetKcpWorkspace: no tape geometry bound (ensure() first)"
                )
            R, NC_pad, HV, T = self.geometry
            shape, dtype = tape_spec(
                name, R=R, NC_pad=NC_pad, HV=HV, T=T, H_qk=self.qk_heads
            )
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
        shape, dtype = tape_spec(
            name, R=R, NC_pad=NC_pad, HV=HV, T=T, H_qk=self.qk_heads
        )
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
class CakeGDNNondetKcpSaved:
    """The saved context of one forward (``boundary_mode="saved"``): ``S_in`` (the fresh c2 output), the owner's c1
    receive tensor (``save_transfer``), ``M_seg`` (``save_transfer`` or ``save_mode="full"`` at ``P > 1``), the saved
    tapes of the ``save_mode`` (:func:`saved_tapes`: none in ``"light"``, every forward tape in ``"full"``) and the
    scalars of the call.  With ``use_qk_l2norm_in_kernel`` (``qk_l2norm``) it also carries ``qk_rstd``, the FP32
    ``r`` of q and k ``[2, T_r, H_qk]`` (after the tapes in :meth:`tensors`; its presence follows
    ``meta()["qk_l2norm"]``, so :meth:`flags` keeps its three entries).  Every
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
    qk_rstd: torch.Tensor | None = None
    qk_l2norm: bool = False
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
        if self.qk_l2norm:
            out.append(self.qk_rstd)
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
            "qk_l2norm": self.qk_l2norm,
        }

    @classmethod
    def from_tensors(
        cls,
        tensors: Sequence[torch.Tensor],
        flags: Sequence[bool],
        meta: dict[str, Any],
        ctx: Any,
    ) -> CakeGDNNondetKcpSaved:
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
        qk_rstd = next(it) if meta.get("qk_l2norm") else None
        return cls(
            S_in=S_in,
            c1_recv=c1_recv,
            M_seg=M_seg,
            tapes=tapes,
            ctx=ctx,
            qk_rstd=qk_rstd,
            **meta,
        )

    def tape(self, name: str) -> torch.Tensor:
        names = saved_tapes(self.save_mode)
        if self.tapes is None or name not in names:
            raise KeyError(
                f"saved: this saved set (save_mode={self.save_mode!r}) carries no tape {name!r}; it holds {names}"
            )
        return self.tapes[names.index(name)]

    def forget(self, name: str) -> None:
        """Drop this set's reference to ``name`` (``S_in`` / ``c1_recv`` / ``M_seg`` / ``qk_rstd``) once the backward's last reader
        of it has run (``release_in_backward``): the bytes return to the allocator before the backward's next
        allocation.  Frozen dataclass, hence ``object.__setattr__``."""
        if name not in ("S_in", "c1_recv", "M_seg", QK_RSTD):
            raise KeyError(f"saved: {name!r} is not a saved-set tensor")
        object.__setattr__(self, name, None)

    @property
    def bytes(self) -> int:
        return sum(
            t.numel() * t.element_size() for t in self.tensors() if t is not None
        )


def _exchange_scheme(tb: _Tables, opts: CakeGDNNondetKcpOptions) -> str:
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


class CakeGDNNondetKcpCall:
    """The state of one forward or backward call on one rank: tables, buffers (workspace pools and the call's fresh
    tensors), tapes, operands, backend, the launch / collective counters and the issue sequence."""

    def __init__(
        self,
        *,
        ctx: Any,
        H_qk: int,
        HV: int,
        ws: CakeGDNNondetKcpWorkspace,
        tb: _Tables,
        backend: Backend,
        opts: CakeGDNNondetKcpOptions,
        scale: float,
        ops: dict[str, torch.Tensor],
        backward: bool,
        packed_grads: bool = True,
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
        self.grad_layout: str | None = (
            None  # "packed" | "separate" (the backward's dq / dk / dv)
        )
        # whether dq / dk / dv may be the head views of one packed dQKV row (:func:`_alloc_grads`)
        self.packed_grads = bool(packed_grads)
        self.exchange = _exchange_scheme(tb, opts)
        # use_qk_l2norm_in_kernel: the q / k readers bind qn / kn (:meth:`qk`), #1 / #8 run as their _l2n siblings
        self.l2n = bool(opts.use_qk_l2norm_in_kernel)
        self.precision_mode = int(opts.precision_mode)
        self.transfer_walk = str(opts.transfer_walk)
        self.schedule = str(opts.knob("schedule"))
        self.cols_rule = str(opts.knob("cols_rule"))
        self.debug_sequence = bool(opts.knob("debug_sequence"))
        self.walk_pair = _walk_pair_enabled(opts.knob("walk_pair"), ctx, self.device)
        # (name, args) of the launches resolved while _walk_pair defers them (issued on its side stream); else None
        self._deferred: list[tuple[str, dict[str, Any]]] | None = None
        if int(tb.fused_walks) != int(opts.knob("fused_walks")):
            raise ValueError(
                f"tables built with fused_walks={tb.fused_walks}, the call asks for {opts.knob('fused_walks')}"
            )
        p_rule = opts.knob("p_rule")
        if p_rule is not None and p_rule != getattr(ctx, "p_rule", None):
            raise ValueError(
                f"test_knobs p_rule {p_rule!r} differs from the context's plan rule {getattr(ctx, 'p_rule', None)!r}: "
                "give the rule to cake_gdn_nondet_train_kcp_build_context"
            )
        if opts.segments is not None and int(opts.segments) != tb.P:
            raise ValueError(
                f"segments={opts.segments}: the plan built P={tb.P} segments per fragment (the override is bounded by "
                "the fragments' chunk counts)"
            )

    # -- buffers
    def output(self) -> torch.Tensor:
        """The forward's output ``o`` ``[T_r, HV, 128]``, allocated at its first writer: the first output walk
        (``kcp_fwdh_out`` / ``kcp_fwdh_out_tape``) of a fused forward (after the summaries and the exchange's send
        planes), else :func:`_fwd_body` before #5."""
        if self.out is None:
            self.out = _alloc(
                (self.T_r, self.HV, D), self.backend.dtypes["out"], self.device
            )
        return self.out

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
        spec = self.tb.indexed_spec(index, tiles)
        if spec is not None:  # the plan's own tables: normalised once
            return Indexed(self.buf(pool), spec.index, spec.tiles)
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

    def qk(self, name: str) -> torch.Tensor:
        """The ``q`` / ``k`` a reader program binds: the raw operand, or with ``use_qk_l2norm_in_kernel`` the
        normalized rows ``qn`` / ``kn`` (arena tapes #1_l2n / ``qk_apply`` wrote).  ``ops["q"]`` / ``ops["k"]`` stay
        the raw operands (packed-gradient detection, the kept-tape key, #1_l2n's and #8_l2n's raw inputs)."""
        if not self.l2n:
            return self.ops[name]
        return self.tape("qn" if name == "q" else "kn")

    def rstd_out(self) -> torch.Tensor:
        """Where #1_l2n writes ``r``: a forward's fresh saved-set tensor (bound in ``bufs`` when it keeps a saved
        set), else the arena tape (a forward without a saved set, the replay-mode forward, the light replay -- a
        backward never writes the saved tensor)."""
        t = None if self.backward else self.bufs.get(QK_RSTD)
        return t if t is not None else self.tape(QK_RSTD)

    def rstd_in(self) -> torch.Tensor:
        """The ``r`` the pullback and ``qk_apply`` read: the saved set's (bound in ``bufs``), else the arena's (the
        replay mode: its own #1_l2n wrote it)."""
        t = self.bufs.get(QK_RSTD)
        return t if t is not None else self.tape(QK_RSTD)

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

    def drop_bound(self, name: str, saved: CakeGDNNondetKcpSaved | None = None) -> None:
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
            shape, dtype = tape_spec(
                name, R=R, NC_pad=NC_pad, HV=HV, T=T, H_qk=self.H_qk
            )
            self.tapes[name] = _alloc(shape, dtype, self.device)

    def dev(
        self, key: Any, values: Sequence[int], dtype: torch.dtype = torch.int32
    ) -> torch.Tensor:
        return self.tb.dev(key, values, self.device, dtype)

    def block(self, ix: Indexed | None) -> _Block:
        """The kernel's view of an :class:`Indexed` pool; ``None`` (or no slot) = the zero state: every index ``-1``
        and a never-read base pointer."""
        tb = self.tb
        nseg = int(tb.NSEG)
        if ix is None or ix.n_slots == 0:
            return _Block(self.buf("eye"), tb.neg_index(self.device), self.H, 1, 1)
        # a view from indexed() over the plan's own tables: its addressing and index table are derived once
        spec = tb.memo.get(("ixn", id(ix.index), id(ix.tiles)))
        if spec is None:
            base_tile, hg, n, stride = ix.addressing(self.H)
        else:
            if spec.addr is None:
                spec.addr = ix.addressing(self.H)
            base_tile, hg, n, stride = spec.addr
        if len(ix.index) != nseg:
            raise ValueError(
                f"Indexed: {len(ix.index)} segment entries, the plan has NSEG = {nseg}"
            )
        ptr = ix.pool.view(-1)[base_tile * TILE :] if base_tile else ix.pool.view(-1)
        if spec is None:
            index_t = self.dev("index", ix.index)
        else:
            index_t = spec.dev.get(self.device)
            if index_t is None:
                index_t = spec.dev[self.device] = self.dev("index", ix.index)
        return _Block(ptr, index_t, int(hg), int(n), int(stride))

    # -- launches
    def kernel(self, name: str, grid: int, **args: Any) -> None:
        if name not in PROGRAMS:
            raise ValueError(f"unknown program {name!r}")
        self.counters["launches"] += 1
        self.sequence.append(name)
        args["_call"] = self
        args["_grid"] = int(grid)
        if self._deferred is not None:
            self._deferred.append((name, args))
            return
        self.backend.run(name, args)

    def _walk_common(self, walk: Sequence[int]) -> tuple[int, dict[str, Any]]:
        """``(len(walk), the walk programs' common arguments)``; the walk list's device table comes from the plan's
        memo (:meth:`_Tables.walk_table`)."""
        tb = self.tb
        seg = tb.seg_table(self.device)
        n_walk, walk_seg = tb.walk_table(walk, self.device)
        return n_walk, {
            "num_heads": self.HV,
            "num_qk_heads": self.H_qk,
            "seg_chunk_start": seg,
            "num_seqs": int(tb.NSEG),
            "walk_seg": walk_seg,
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
        self.counters["walks"] += 1
        n_walk, args = self._walk_common(walk)
        if name in ("kcp_fwdh_out", "kcp_fwdh_out_tape"):
            # the tape walk's inputs from the indexed initial states and #5's operands: o rows of the walked chunks,
            # no final-state pool; the tape-and-output walk also writes the h / v_new tapes, the output walk neither
            h0 = self.block(kw.get("h0"))
            args.update(
                k=self.qk("k"),
                w=self.tape("w"),
                u=self.tape("u"),
                rowscale=self.tape("rowscale"),
                gamma=self.tape("gamma"),
                **self._block_args("h0", h0),
                tile_stride=h0.stride,
                q=self.qk("q"),
                aqk=self.tape("aqk"),
                qscale=self.tape("qscale"),
                o=self.output(),
            )
            if name == "kcp_fwdh_out_tape":
                args.update(h_tape=self.tape("h_tape"), v_new=self.tape("v_new"))
        elif name in ("kcp_fwdh_offset", "kcp_fwdh_tape"):
            tape = name == "kcp_fwdh_tape"
            # The unused pool of a walk is the ZERO block (every index -1, a never-read base pointer): the tape
            # walk reads h0 and writes the tapes only; the offset walk starts from zero and writes the indexed final
            # states only.  Aliasing one block for both would start the offsets from stale staging tiles and let the
            # tape walk overwrite the carries it reads.
            h0 = self.block(kw.get("h0") if tape else None)
            out = self.block(None if tape else kw.get("out"))
            args.update(
                k=self.qk("k"),
                w=self.tape("w"),
                u=self.tape("u"),
                rowscale=self.tape("rowscale"),
                gamma=self.tape("gamma"),
                **self._block_args("h0", h0),
                **self._block_args("out", out),
                tile_stride=h0.stride if tape else out.stride,
                # The offset program never dereferences the tape pointers (its build holds no load or store of them),
                # so it gets ``u`` for both (``FWDH_OFFSET_TAPE_STANDIN``): binding ``h_tape`` / ``v_new`` here would
                # allocate them at the summaries, while every forward pool is live, for a walk that never writes them.
                h_tape=self.tape("h_tape" if tape else FWDH_OFFSET_TAPE_STANDIN),
                v_new=self.tape("v_new" if tape else FWDH_OFFSET_TAPE_STANDIN),
            )
        elif name == "kcp_transfer_walk":
            out = self.block(kw["out"])
            args.update(
                k=self.qk("k"),
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
                q=self.qk("q"),
                k=self.qk("k"),
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
            if not tape:
                # The cut table of this walk: ``kcp_dhu_cut_start`` writes ``c_start`` of every listed (segment, value
                # head) from the tapes the walk reads, then the walk covers chunks ``c_start .. 0`` of each chain (the
                # full walk's final state, up to the sign of a zero).  Same stream, same walk list, no host read.
                c_start = _cut_table(self.ctx, int(self.tb.NSEG) * self.HV, self.device)
                self.kernel(
                    "kcp_dhu_cut_start",
                    n_walk * self.HV,
                    _grid_args={"n_walk": n_walk, "num_heads": self.HV},
                    gamma=args["gamma"],
                    rowscale=args["rowscale"],
                    w=args["w"],
                    chunk_len=args["chunk_len"],
                    num_heads=self.HV,
                    seg_chunk_start=args["seg_chunk_start"],
                    num_seqs=args["num_seqs"],
                    walk_seg=args["walk_seg"],
                    c_start=c_start,
                )
                args["c_start"] = c_start
        else:
            raise ValueError(f"{name} is not a walk program")
        sm_count = int(getattr(self.ctx, "sm_count", 0) or 0)
        fits = (
            name in FWDH_PROGRAMS
            and n_walk * self.HV > sm_count
            and self.tb.walk_fits(walk, sm_count)
        )  # the chain clause of fwdh_vh (only where the one-wave clause does not already pick vh = 64)
        long_chain = name in DHU_PROGRAMS and self.tb.walk_long_chain(walk, sm_count)
        self.kernel(
            name,
            n_walk * self.HV,
            _grid_args={"n_walk": n_walk, "num_heads": self.HV},
            _sm_count=sm_count,
            _fwdh_vh=self.opts.knob("fwdh_vh"),
            _dhu_vh=self.opts.knob("dhu_vh"),
            _vh64_fits=fits,
            _dhu_long_chain=long_chain,
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
        # the plan's own family: its tables from the plan's memo; explicit chains (the ring's): derived per call
        start, elem_m, elem_c, elem_y, x0_index, y_final = (
            self.tb.chain_tables(family, self.device)
            if chains is None
            else _chain_device_tables(self.tb, ch, self.device)
        )
        self.kernel(
            "kcp_affine_merge",
            ch.n_chains * (ch.ncols // cols_eff),
            _grid_args={"n_chains": int(ch.n_chains), "ncols": int(ch.ncols)},
            _family=str(family),
            m=m,
            c=c,
            x0=x0,
            y=y,
            chain_start=start,
            elem_m=elem_m,
            elem_c=elem_c,
            elem_y=elem_y,
            x0_index=x0_index,
            y_final=y_final,
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
        """A chunk-local program (#1, #5, #6, #8, #9, ``qk_apply``) over this rank's tokens (skipped without
        tokens).  With ``use_qk_l2norm_in_kernel`` ``fwd_prep`` launches ``fwd_prep_l2n`` and the fused
        ``bwd_dqkg`` launches ``bwd_dqkg_l2n`` (:data:`L2N_SIBLING_OF`); ``group_reduce="host"`` keeps #8 itself
        (on ``qn`` / ``kn``) and applies the pullback in :meth:`host_group_reduce`."""
        if self.T_r == 0:
            return
        if (
            name not in LOCAL_PROGRAMS
            or name in L2N_SIBLING_OF.values()
            or name in AKK_REPLAY_OF.values()
        ):
            raise ValueError(f"{name} is not a local program")
        if name == "qk_apply" and not self.l2n:
            raise ValueError("qk_apply runs with use_qk_l2norm_in_kernel only")
        program = name
        if self.l2n and name in L2N_SIBLING_OF:
            if name == "fwd_prep" or self.opts.group_reduce != "host":
                program = L2N_SIBLING_OF[name]
        if program in AKK_REPLAY_OF and self.scratch.get("akk_source") == "saved":
            # the light replay of an AKK_SAVE_MODES set: #1_l2n without the block inverse, reading the forward's Akk
            # tape (bound in self.tapes by _backward) instead of writing it -- every other output bit for bit #1_l2n's
            program = AKK_REPLAY_OF[program]
        self.counters["locals"] += 1
        o, NC = self.ops, int(self.ctx.NC)
        args = self._local_common()
        if name == "fwd_prep":
            args.update(
                q=o["q"],  # #1 / #1_l2n read the raw rows
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
            if program == "fwd_prep_l2n" or program in AKK_REPLAY_PROGRAMS:
                rstd = self.rstd_out()
                args.update(
                    qn=self.tape("qn"),
                    kn=self.tape("kn"),
                    q_rstd=rstd[0],
                    k_rstd=rstd[1],
                    rstd_tstride=self.H_qk,
                )
            grid = NC * self.H_qk
        elif name == "qk_apply":
            args.pop("num_heads")
            args.pop("num_seqs")
            rstd = self.rstd_in()
            args.update(
                q_raw=o["q"],
                k_raw=o["k"],
                q_raw_tstride=int(o["q"].stride(0)),
                k_raw_tstride=int(o["k"].stride(0)),
                q_rstd=rstd[0],
                k_rstd=rstd[1],
                rstd_tstride=self.H_qk,
                qn=self.tape("qn"),
                kn=self.tape("kn"),
            )
            grid = NC * self.H_qk
        elif name == "fwd_o":
            args.update(
                q=self.qk("q"),
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
                q=self.qk("q"),
                k=self.qk("k"),
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
                dq=_row_span(self.grads["dq"]),
                dk=_row_span(self.grads["dk"]),
                dv=_row_span(self.grads["dv"]),
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
            if program == "bwd_dqkg_l2n":
                rstd = self.rstd_in()
                args.update(
                    q_raw=o["q"],
                    k_raw=o["k"],
                    q_raw_tstride=int(o["q"].stride(0)),
                    k_raw_tstride=int(o["k"].stride(0)),
                    rstd_tstride=self.H_qk,
                    q_rstd=rstd[0],
                    k_rstd=rstd[1],
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
            program,
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
        (head ``0``, then ``+1``, ...) and rounded once to BF16 into the outputs.  With ``use_qk_l2norm_in_kernel``
        the FP32 sum goes through the norm's pullback (:func:`.cake_l2n.l2n_pullback`: the raw rows and ``r``) before
        the rounding, per block of :data:`HOST_L2N_BLOCK_TOKENS` tokens (the same per-token order)."""
        if self.T_r == 0:
            return
        self.counters["launches"] += 1
        self.sequence.append("host_group_reduce")
        if self.l2n:
            rstd = self.rstd_in()
            for plane, (part, name) in enumerate(
                (("dq_part", "dq"), ("dk_part", "dk"))
            ):
                p = self.tape(part).view(self.T_r, self.H_qk, self.r, D)
                x, r_t, out = self.ops[name[1]], rstd[plane], self.grads[name]
                for t0 in range(0, self.T_r, HOST_L2N_BLOCK_TOKENS):
                    t1 = min(self.T_r, t0 + HOST_L2N_BLOCK_TOKENS)
                    acc = p[t0:t1, :, 0]
                    for j in range(1, self.r):
                        acc = acc + p[t0:t1, :, j]
                    dx = cake_l2n.l2n_pullback(x[t0:t1], r_t[t0:t1], acc)
                    out[t0:t1].copy_(dx.to(out.dtype))
            return
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
        send, recv = self.buf(send_name), self.buf(recv_name)
        n_send, n_recv = sum(send_splits), sum(recv_splits)
        # a buffer of exactly the exchanged size is its own leading view (same storage, offset, shape, stride)
        if send.numel() != n_send:
            send = send[:n_send]
        if recv.numel() != n_recv:
            recv = recv[:n_recv]
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


def saved_states(saved: CakeGDNNondetKcpSaved) -> torch.Tensor:
    """The saved set's ``S_in`` as FP32 ``[n_in, HV, 128, 128]`` in ``in_slot`` order (the oracle form)."""
    return gather_states(saved.S_in, saved.ctx.heads(saved.HV).h0_tile)


def _record(kind: str, call: CakeGDNNondetKcpCall) -> None:
    """The launch inventory of the last forward / backward (evidence)."""
    ws = call.ws
    pool_bytes = ws.pool_bytes
    tape_bytes = ws.tape_bytes
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
        "workspace_bytes": sum(pool_bytes.values())
        + tape_bytes
        + ws.ticket_bytes,  # = ws.bytes
        "pool_bytes": pool_bytes,
        "tape_bytes": tape_bytes,
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
        "grad_layout": call.grad_layout,
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
        ),  # forward: "launched" | "skipped" (recompute_output="auto": the recompute's dead #5) | "fused" (kcp_fwdh_out)
        # | "fused_tape" (kcp_fwdh_out_tape: save_mode="full", the saved tapes and the output in the same walks)
        "fwd_o_reason": call.scratch.get(
            "fwd_o_reason"
        ),  # forward: "always" | "dead" | "not_executable" | why the frame proof did not hold (the decision, recorded on every rank) | "small" (graph="replay", the size rule)
        "hand_off_reason": call.scratch.get(
            "hand_off_reason"
        ),  # forward: "always" (True) | "off" | "not_executable" | the frame reason under "auto" ("dead" = kept) | "small" (graph="replay": the recompute below GRAPH_RECOMPUTE_MIN_ROWS, no decision)
        "recompute_handle": call.scratch.get(
            "recompute_handle"
        ),  # forward: None | "capture" (stored S_in in the handle) | "replay" (S_in from the handle, no exchange)
        "qk_l2norm": call.l2n,  # use_qk_l2norm_in_kernel
        # where qn / kn came from: forward "fwd_prep" (#1_l2n); backward "recompute" (the replay mode's forward),
        # "replay" (the light replay's #1_l2n), "kept" (the recompute forward's #1_l2n, handed over with the kept
        # tapes) or "qk_apply" (save_mode="full"); None without the norm
        "qk_source": call.scratch.get("qk_source"),
        # backward: "saved" when the light replay read the forward's Akk tape (save_mode="light_akk",
        # fwd_prep_replay_l2n); None otherwise (#1 formed it, or no #1 ran)
        "akk_source": call.scratch.get("akk_source"),
        # the r #1_l2n wrote (forward: "fresh" = the saved set's tensor, "arena") / the pullback read (backward:
        # "saved", "arena" in the replay mode); None without the norm
        "rstd_source": call.scratch.get("rstd_source"),
        "arena_tapes": sorted(
            call.ws._tapes
        ),  # the tapes bound in the workspace arena when the call ended
        "qk": call.scratch.get(
            "qk"
        ),  # test knob record_qk: {"qn", "kn", "rstd"[, "rstd_regen", "rstd_regen_equal"]} copies
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
    call: CakeGDNNondetKcpCall,
    tb: _Tables,
    src,
    src_tiles,
    slot_src: int,
    dst,
    slot_dst: int,
) -> None:
    """``dst`` ring block ``slot_dst`` <- the ``H`` tiles ``src_tiles[slot_src][h]`` of pool ``src`` (bitwise copy)."""
    idx = tb.device_tiles([src_tiles[slot_src]], call.device)
    _ring_block(dst, slot_dst, call.H).copy_(src.view(-1, D, D)[idx.view(-1)])


def _ring_scatter(
    call: CakeGDNNondetKcpCall,
    tb: _Tables,
    src,
    slot_src: int,
    dst,
    dst_tiles,
    slot_dst: int,
) -> None:
    """Pool ``dst`` tiles ``dst_tiles[slot_dst][h]`` <- the ring block ``slot_src`` of ``src`` (bitwise copy)."""
    idx = tb.device_tiles([dst_tiles[slot_dst]], call.device)
    dst.view(-1, D, D)[idx.view(-1)] = _ring_block(src, slot_src, call.H)


def _ring_p2p(call: CakeGDNNondetKcpCall, hop: int, sends, recvs):
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


def _ring_forward(
    call: CakeGDNNondetKcpCall,
    *,
    walk_starts: bool = False,
    program: str = "kcp_fwdh_tape",
) -> None:
    """Hops of the forward ring; afterwards ``S_in`` holds the crossing-in fragments' initial states (as after c2).
    ``walk_starts``: the document-start tape walk (``program``: the tape walk or the output walk) runs under hop 1
    (overlap schedule); the caller then walks only the crossing-in segments."""
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
        if h == 1 and walk_starts:
            carry = (
                call.indexed("carry", tb.carry_index, tb.carry_tiles)
                if tb.P > 1
                else None
            )
            call.walk(program, tb.walk_start, h0=carry)  # document starts: under hop 1
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


def _ring_backward(call: CakeGDNNondetKcpCall, *, walk_ends: bool = False) -> None:
    """Hops of the adjoint ring; afterwards ``b2_recv`` holds the crossing-out fragments' ``dS_out`` (as after b2).
    Requires the b1 ``dm`` form (``d_f`` and ``M_f`` of the crossing-in fragments in ``b1_send``).  ``walk_ends``:
    the ``dcarry_end`` merge (``P > 1``) and the document-end adjoint walk run under hop 1 (overlap schedule); the
    caller then neither merges ``dcarry_end`` again nor walks those segments again (the walk stores ``dv_new`` in
    place of ``dv_intra``, so a second walk of a segment would apply its update twice)."""
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
        if h == 1 and walk_ends:
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


# Walk pair (test knob ``walk_pair``, default ``None`` = auto): the summaries' offset walk and transfer walk read the
# same tapes (k, w, u, rowscale, gamma) and the tables only and write disjoint tiles, so the offset walk may run beside
# the transfer walk.
# Its launch is resolved on the current stream (pools, tapes and device tables are bound and uploaded there: nothing
# is allocated on the side stream) and issued on a per-device side stream after a fork event recorded on the current
# stream; the transfer walk follows on the current stream, which then waits on the side stream's join event before
# anything else is enqueued.  The deferred launch arguments stay referenced until that wait is enqueued, so no tensor
# the side stream reads returns to the allocator (and to the current stream's reuse) before the join; every later use
# or free is ordered after it.  No host synchronization, no device-to-host copy; the launches, their arguments and
# outputs and the call's sequence / records are those of walk_pair=0.  Under graph="replay" the side stream joins and
# leaves the capture through the same two events.
# The pair is enabled on the zigzag partitions only (WALK_PAIR_LAYOUTS) of at most WALK_PAIR_MAX_WORLD ranks; a
# contiguous or explicit partition, or a larger world, keeps the walk_pair=0 path (GB300 CP8: the contiguous rows, three
# summaries per rank, ran 1-3 % slower paired; GB300 CP16: one summary per rank, no measured gain).  The default (auto)
# pairs only on the architectures in WALK_PAIR_AUTO_CAPABILITIES: sm_100a (GB200) and sm_103a (GB300), where no zigzag
# row at CP2 / CP4 / CP8 ran more than 1 % slower in both timing sessions and the CP4 / CP8 rows gained up to 6-8 %.
# sm_107a keeps one stream: on a 212-SM R200 the CP4 zigzag rows ran 1.2-2.5 % slower paired in both sessions, and
# there is no 216-SM measurement.  An explicit walk_pair=0 / 1 forces either path (still on the zigzag partitions only).
WALK_PAIR_LAYOUTS = ("zigzag", "zigzag_per_document")
WALK_PAIR_MAX_WORLD = 8
WALK_PAIR_AUTO_CAPABILITIES = ((10, 0), (10, 3))  # sm_100a, sm_103a
_SIDE_STREAMS: dict[int, torch.cuda.Stream] = {}
_CAPABILITIES: dict[int, tuple[int, int]] = {}
WALK_PAIR_STATS: dict[str, int] = {
    "paired": 0,  # walk pairs issued as fork / join
    "single": 0,  # walk pairs with the knob on that had one side empty (issued on the current stream only)
}


def _device_capability(device: torch.device | None) -> tuple[int, int] | None:
    """The compute capability of a CUDA ``device`` (read once per device index, host only); ``None`` off CUDA."""
    if device is None or device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    capability = _CAPABILITIES.get(index)
    if capability is None:
        capability = _CAPABILITIES[index] = tuple(
            torch.cuda.get_device_capability(index)
        )
    return capability


def _walk_pair_enabled(
    knob: int | None, ctx: Any, device: torch.device | None = None
) -> bool:
    """The walk pair as applied to the call: only for a context whose partition layout is in
    :data:`WALK_PAIR_LAYOUTS` and whose world has at most :data:`WALK_PAIR_MAX_WORLD` ranks; there ``walk_pair=0`` /
    ``1`` forces it off / on, and the default ``None`` (auto) turns it on when ``device`` is a CUDA device whose
    capability is in :data:`WALK_PAIR_AUTO_CAPABILITIES` (``sm_100a``, ``sm_103a``)."""
    part = getattr(ctx, "partition", None)
    layout, world = getattr(part, "layout", None), getattr(part, "world_size", None)
    if layout not in WALK_PAIR_LAYOUTS or world is None or world > WALK_PAIR_MAX_WORLD:
        return False
    if knob is None:
        return _device_capability(device) in WALK_PAIR_AUTO_CAPABILITIES
    return bool(knob)


def _fuse_o_enabled(call: CakeGDNNondetKcpCall) -> bool:
    """Whether this call's forwards fold #5 into their tape walks where the tapes have no reader: test knob
    ``fused_o`` = 1 (shipped), a save mode whose forward keeps no h / v_new tape (:data:`LIGHT_FORWARD_SAVE_MODES`:
    ``"light"``, ``"light_akk"``) and ``boundary_mode="saved"`` (no saved h / v_new tape, no replayed forward
    exchange), and the registry ships the output walk in both forms for the device's architecture
    (:func:`fused_o_shipped`; the fake backend emulates any shipped record)."""
    opts = call.opts
    if int(opts.knob("fused_o")) != 1:
        return False
    if opts.save_mode not in LIGHT_FORWARD_SAVE_MODES or opts.boundary_mode != "saved":
        return False
    return fused_o_shipped(_call_arch(call))


def _call_arch(call: CakeGDNNondetKcpCall) -> str | None:
    """The architecture the kernel backend loads this call's libraries for; ``None`` for the fake backend or off
    CUDA (the registry checks then accept any architecture)."""
    if not (isinstance(call.backend, KernelBackend) and call.device.type == "cuda"):
        return None
    index = (
        call.device.index
        if call.device.index is not None
        else torch.cuda.current_device()
    )
    return call.backend.launcher().device_arch(index)


def _fuse_o(call: CakeGDNNondetKcpCall, *, launch: bool, hand_off: bool) -> bool:
    """Whether this forward runs the output walk ``kcp_fwdh_out`` on every tape-walk launch list in place of the tape
    walk and #5: #5 launches, the tapes go to no backward (not a hand-off forward), the rank holds tokens and
    :func:`_fuse_o_enabled`."""
    return bool(launch and not hand_off and call.T_r > 0 and _fuse_o_enabled(call))


def _fuse_o_tape_enabled(call: CakeGDNNondetKcpCall) -> bool:
    """Whether this call's forwards fold #5 into the tape walks that write the saved tapes: test knob
    ``fused_o_tape`` = 1 (shipped), ``save_mode="full"`` and ``boundary_mode="saved"`` (the ``h`` / ``v_new`` tapes
    are saved-set tensors), and the registry ships the tape-and-output walk in both forms for the device's
    architecture (:func:`fused_o_tape_shipped`; the fake backend emulates any shipped record).  A light forward is
    :func:`_fuse_o`'s or split; its hand-off forward keeps the tape walk and #5."""
    opts = call.opts
    if int(opts.knob("fused_o_tape")) != 1:
        return False
    if opts.save_mode != "full" or opts.boundary_mode != "saved":
        return False
    return fused_o_tape_shipped(_call_arch(call))


def _fuse_o_tape(call: CakeGDNNondetKcpCall, *, launch: bool, save: bool) -> bool:
    """Whether this forward runs the tape-and-output walk ``kcp_fwdh_out_tape`` on every tape-walk launch list in
    place of the tape walk and #5: #5 launches, the forward keeps its saved set (``save``: the saved ``h`` /
    ``v_new`` tapes are the walk's), the rank holds tokens and :func:`_fuse_o_tape_enabled`.  A full forward without a
    saved set (``save_for_backward=False``) and a recompute that leaves #5 out keep the tape walk."""
    return bool(launch and save and call.T_r > 0 and _fuse_o_tape_enabled(call))


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = _SIDE_STREAMS.get(index)
    if stream is None:
        # torch draws every torch.cuda.Stream from a per-device, per-priority pool of 32 (round robin): a default-priority
        # stream would also be every 32nd later default-priority stream of the process (a caller's side stream, the
        # process group's streams), and the offset walk would queue behind that stream's work.  The greatest priority
        # level's pool is not drawn by default-priority users.
        greatest = torch.cuda.Stream.priority_range()[1]
        stream = _SIDE_STREAMS[index] = torch.cuda.Stream(
            device=index, priority=greatest
        )
    return stream


def _walk_pair(
    call: CakeGDNNondetKcpCall, side: Callable[[], None], main: Callable[[], None]
) -> None:
    """``side()`` then ``main()``: in this order on the current stream (walk pair off, a CPU device), or with the
    launches of ``side()`` on the side stream concurrently with ``main()`` (walk pair on, see above)."""
    if not call.walk_pair or call.device.type != "cuda":
        side()
        main()
        return
    pending: list[tuple[str, dict[str, Any]]] = []
    call._deferred = pending
    try:
        side()
    finally:
        call._deferred = None
    if not pending:
        WALK_PAIR_STATS["single"] += 1
        main()
        return
    current = torch.cuda.current_stream(call.device)
    stream = _side_stream(call.device)
    fork = torch.cuda.Event()
    fork.record(current)
    stream.wait_event(fork)
    with torch.cuda.stream(stream):
        for name, args in pending:
            call.backend.run(name, args)
    join = torch.cuda.Event()
    join.record(stream)
    n_main = call.counters["launches"]
    main()
    current.wait_event(join)
    WALK_PAIR_STATS["paired" if call.counters["launches"] > n_main else "single"] += 1
    pending.clear()  # the side launches' arguments are released only after the join is enqueued


def _offset_walk(call: CakeGDNNondetKcpCall, walk: Sequence[int], out: Indexed) -> None:
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


def _summaries(call: CakeGDNNondetKcpCall) -> None:
    """The fragment summaries ``[B_f | M_f^T]`` into the c1 send layout: at ``P = 1`` the offset walk and the
    transfer walk write the crossing-out fragments' tiles directly; at ``P > 1`` both walk every segment into the
    staging planes and the merge composes the crossing-out fragments' segments (forward family -> ``B_f``, map
    family -> ``M_f``, transposed into the pairs)."""
    tb = call.tb
    if tb.P == 1:
        _walk_pair(
            call,
            lambda: _offset_walk(
                call,
                tb.walk_offset,
                call.indexed("c1_send", tb.out_index, tb.c1_b_tiles),
            ),
            lambda: call.walk(
                "kcp_transfer_walk",
                tb.walk_offset,
                out=call.indexed("c1_send", tb.out_index, tb.c1_m_tiles),
                init_mode=0,
                use_u=0,
                write_transposed=0,
            ),
        )
        return
    _walk_pair(
        call,
        lambda: _offset_walk(
            call,
            tb.walk_offset,
            call.indexed("stage_b", tb.seg_index, tb.stage_tiles),
        ),
        lambda: call.walk(
            "kcp_transfer_walk",
            tb.walk_offset,
            out=call.indexed("stage_m", tb.seg_index, tb.stage_tiles),
            init_mode=0,
            use_u=0,
            write_transposed=0,
        ),
    )
    call.merge(
        "map"
    )  # B_f (from 0) and M_f (from I) of every crossing-out fragment, forward family
    call.transpose_tiles(
        "c1_send", tb.mf_tiles, "c1_send", tb.c1_m_tiles, scratch="carry"
    )  # M_f^T into the c1 pairs (bitwise); the gather lands in the carry plane, which nothing has written yet


def _fwd_tapes(
    call: CakeGDNNondetKcpCall,
    *,
    carry: Indexed | None,
    wait_first: bool,
    program: str = "kcp_fwdh_tape",
) -> None:
    """The forward tape walk(s) after ``S_in`` is complete (``program``: the tape walk, or the output walk of a fused
    forward on the same launch list).  ``fused_walks``: one launch from one pool -- the carry
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
        call.walk(program, tb.walk_live if wait_first else tb.walk_in_fused, h0=h0)
        return
    if wait_first:
        call.walk(program, tb.walk_start, h0=carry)  # document starts: after wait(c2)
    call.walk(program, tb.walk_in_p0, h0=h0_in)
    if tb.P > 1:
        call.walk(program, tb.walk_in_p1, h0=carry)


def _adj_tapes(
    call: CakeGDNNondetKcpCall, *, dcarry: Indexed | None, wait_first: bool
) -> None:
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
    call: CakeGDNNondetKcpCall,
    *,
    save: bool,
    replay: bool,
    hand_off: bool = False,
    s_in: torch.Tensor | None = None,
    fuse_o: bool = False,
    fuse_o_tape: bool = False,
) -> None:
    """The forward sequence.  ``fuse_o``: every tape-walk launch runs the output walk ``kcp_fwdh_out`` on its launch
    list instead (``o`` written in the walk, no ``h`` / ``v_new`` tape; :func:`_fuse_o`).  ``fuse_o_tape``: every
    tape-walk launch runs the tape-and-output walk ``kcp_fwdh_out_tape`` on its launch list instead (``o`` and the
    ``h`` / ``v_new`` tapes written in the walk; :func:`_fuse_o_tape`).  ``save``: the saved set is
    kept (fresh ``S_in``; with ``save_transfer`` also the c1 receive tensor; ``M_seg`` at ``P > 1`` with
    ``save_transfer`` or ``save_mode="full"``; with ``save_mode="full"`` fresh forward tapes); ``replay``: the forward
    of a replay-mode backward (no ``fwd_o``, nothing kept); ``hand_off``: the forward tapes go to the backward of this
    graph task (``reuse_recompute_tapes``): no tape is dropped and the tape walk takes the light replay's form
    (:func:`_replay_tape_walk`).  ``s_in``: the recompute of a forward that stored its incoming states in a recompute
    handle -- no collective, no summary walk over the crossing-out fragments, no merge of the owners: #1, at
    ``P > 1`` the local segment walks and carry merges from ``S_in``, and the light replay's tape walk
    (:func:`_replay_from_states`), the very launches of the light backward's replay.  ``schedule="wait_first"``: ``c1 -> wait -> fwd merge -> c2 ->
    wait -> carry merges -> tape walk(s)`` -- no merge or walk is issued while a collective is in flight; the merges
    read only ``c1_recv`` / the staging planes and the tapes read no collective buffer, so every published bit is the
    same under either schedule."""
    tb, opts = call.tb, call.opts
    if fuse_o and (hand_off or replay):
        raise ValueError(
            "fuse_o: a hand-off or replay forward writes the tapes its backward reads"
        )
    if fuse_o_tape and (fuse_o or replay or hand_off):
        raise ValueError(
            "fuse_o_tape: one output walk per forward, a replay forward launches no #5 and a hand-off forward "
            "keeps the tape walk"
        )
    # the walk every tape-walk launch site of this forward runs
    walk_program = (
        "kcp_fwdh_out"
        if fuse_o
        else ("kcp_fwdh_out_tape" if fuse_o_tape else "kcp_fwdh_tape")
    )
    if fuse_o and opts.graph == "off":
        # a caller-held arena may still bind the h / v_new tapes a split forward or a kept backward left
        # (FWD_TAPES_ARENA_AFTER_FWD): this forward writes neither, so they go before #1
        call.release_tapes(("h_tape", "v_new"))
    # before the first launch: the backend record reads it
    call.scratch["hand_off"] = bool(hand_off)
    wait_first = call.schedule == "wait_first"
    started_early = False  # the document-start tape walk issued under c1 / hop 1 of the ring (overlap schedule)
    merged_start = (
        False  # the carry_start merge issued under c1 / the ring (overlap schedule)
    )
    keep = save and not replay
    keep_transfer = opts.save_transfer and keep
    if keep and _saves_m_seg(opts.save_mode, opts.save_transfer, tb.P):
        call.fresh(
            "stage_m"
        )  # M_seg^T of the segmented fragments: the summaries (or the recompute's transfer walk) write it
    if (
        keep
    ):  # the saved tapes are fresh per call (light: none; full: every forward tape)
        call.fresh_tapes(saved_tapes(opts.save_mode))
    if call.l2n:
        # the saved set's r: fresh, written by #1_l2n directly (the arena r of a forward without a saved set and of
        # the replay-mode forward is dropped after #1 / kept for the pullback)
        if keep:
            call.bufs[QK_RSTD] = _alloc(
                (2, call.T_r, call.H_qk), torch.float32, call.device
            )
        call.scratch["qk_source"] = "recompute" if replay else "fwd_prep"
        if call.backward:
            call.scratch["rstd_source"] = "arena"
        else:
            call.scratch["rstd_source"] = "fresh" if keep else "arena"
    call.local("fwd_prep")
    if call.l2n and call.opts.knob("record_qk") and not call.backward:
        _record_qk(call)
    if call.l2n and not replay and opts.graph == "off":
        # no forward program reads r: the arena copy goes now (the saved set's fresh one is not in the arena)
        call.release_tapes((QK_RSTD,))
    if not replay and opts.graph == "off" and not hand_off:
        # #1's backward-only outputs leave the arena now (FWD_DEAD_AFTER_PREP); a saved copy (save_mode="full") is a
        # fresh tensor bound in call.tapes and stays.  A replay keeps them for #6 / #8; so does a hand-off forward.
        call.release_tapes(
            FWD_DEAD_AFTER_PREP, keep=saved_tapes(opts.save_mode) if keep else ()
        )
    if s_in is not None:
        call.bufs["S_in"] = s_in
        _replay_from_states(call, have_m=False, program=walk_program)
        if call.opts.knob("record_states"):
            call.scratch["states"] = {"S_in": gather_states(s_in, tb.s_in_tiles)}
        return
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
        # a hand-off forward walks every tape in the replay's single form below
        started_early = not wait_first and not hand_off
        _ring_forward(call, walk_starts=started_early, program=walk_program)
    else:
        w1 = call.a2a("c1", "c1_send", "c1_recv", *tb.split_lists("c1"))
        if not wait_first:
            if tb.P > 1:
                call.merge(
                    "carry_start"
                )  # reads the staging planes only (no collective buffer)
                merged_start = True
            # a hand-off forward walks every tape in the replay's single form below
            if not hand_off:
                call.walk(
                    walk_program, tb.walk_start, h0=carry
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
        w2 = call.a2a("c2", "c2_send", "S_in", *tb.split_lists("c2"))
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
            call, carry=carry, wait_first=not started_early, program=walk_program
        )  # no exchange (N_out = 0) or the wait_first schedule: every walk runs here
    if call.opts.knob("record_states"):
        call.scratch["states"] = {
            "S_in": gather_states(call.buf("S_in"), tb.s_in_tiles)
        }
    # #5 is launched by _fwd_body (never in a replay; not by a recompute whose output is provably dead) once this
    # frame and its views of the pools are gone


def _record_qk(call: CakeGDNNondetKcpCall, regen: bool = False) -> None:
    """Test knob ``record_qk``: copies of ``qn`` / ``kn`` and of the ``r`` the call's programs read (forward: the one
    #1_l2n wrote); ``regen`` (the light backward): also the arena ``r`` its #1_l2n replay wrote and whether it equals
    the saved one bitwise (a content read: test knob only)."""
    if call.T_r == 0:
        return
    rec = {
        "qn": call.tape("qn").clone(),
        "kn": call.tape("kn").clone(),
        "rstd": (call.rstd_out() if not call.backward else call.rstd_in()).clone(),
    }
    if regen:
        rec["rstd_regen"] = call.tape(QK_RSTD).clone()
        rec["rstd_regen_equal"] = bool(torch.equal(rec["rstd_regen"], rec["rstd"]))
    call.scratch["qk"] = rec


def _replay_tapes(call: CakeGDNNondetKcpCall, saved: CakeGDNNondetKcpSaved) -> None:
    """``save_mode="light"`` backward: #1 again and the #2 tape walk from the saved ``S_in`` (segment-start states
    from ``M_seg`` / the recomputed offsets with ``(P - 1)`` state-mode steps); no exchange, no transfer walk over the
    crossing-out fragments, no merge of the owners, no #5."""
    call.local("fwd_prep")
    _replay_from_states(call, have_m=saved.M_seg is not None)


def _replay_from_states(
    call: CakeGDNNondetKcpCall, *, have_m: bool, program: str = "kcp_fwdh_tape"
) -> None:
    """The light replay after #1, once ``S_in`` is bound: at ``P > 1`` the carries again (``b_seg``, and ``M_seg^T``
    unless ``have_m``, from zero; the carry merges from ``S_in``), then the tape walk(s) (``program``: the output walk
    or the tape-and-output walk in a fused forward).  Shared by the light backward and the recompute of a forward
    whose states a recompute handle carries (no collective either way)."""
    tb = call.tb
    if (
        tb.P > 1
    ):  # the forward carries again: b_seg (and M_seg^T unless saved) -> carries from 0 / S_in

        def offset() -> None:
            _offset_walk(
                call,
                tb.walk_all,
                call.indexed("stage_b", tb.seg_index, tb.stage_tiles),
            )

        if have_m:
            offset()
        else:
            _walk_pair(
                call,
                offset,
                lambda: call.walk(
                    "kcp_transfer_walk",
                    tb.walk_all,
                    out=call.indexed("stage_m", tb.seg_index, tb.stage_tiles),
                    init_mode=0,
                    use_u=0,
                    write_transposed=0,
                ),
            )
        call.merge("carry_start")
        call.merge("carry_in")
    _replay_tape_walk(call, program=program)


def _replay_tape_walk(
    call: CakeGDNNondetKcpCall, *, program: str = "kcp_fwdh_tape"
) -> None:
    """The #2 tape walk(s) of the light replay once ``S_in`` (and at ``P > 1`` the carry plane) is in place: one
    fused launch over every live segment from the carry plane at ``P > 1``, else the segment-0 walk over every segment
    from ``S_in`` (plus the ``p > 0`` segments from the carries without fused walks).  Also the tape walk of a
    hand-off forward (``reuse_recompute_tapes``), so the kept tapes are this launch list; ``program``: the output
    walk or the tape-and-output walk of a fused recompute from a recompute handle (same launch list)."""
    tb = call.tb
    if (
        tb.fused_walks and tb.P > 1
    ):  # the whole tape replay from the carry plane in one launch
        call.walk(
            program,
            tb.walk_live,
            h0=call.indexed("carry", tb.h0_index_fused, tb.carry_tiles),
        )
    else:
        p0 = [s for s in tb.walk_all if s % tb.P == 0]
        call.walk(program, p0, h0=call.indexed("S_in", tb.h0_index, tb.s_in_tiles))
        if tb.P > 1:
            call.walk(
                program,
                [s for s in tb.walk_all if s % tb.P != 0],
                h0=call.indexed("carry", tb.carry_index, tb.carry_tiles),
            )


def _kept_release(call: CakeGDNNondetKcpCall, kept: _KeptTapes) -> None:
    """The kept tapes leave the backward after #8, their last reader: unbound from the call; the tapes an eager
    forward leaves in the arena (:data:`FWD_TAPES_ARENA_AFTER_FWD`, or :data:`FWD_TAPES_ARENA_AFTER_FUSED` when this
    call's forwards fuse #5: the next forward binds no ``h`` / ``v_new``) return to this call's workspace when it holds
    none of that name at the same geometry -- a caller-held workspace ends a kept backward as a forward leaves it,
    so the next checkpointed forward allocates nothing but its output (#7039, caller-workspace form) -- and the
    others are freed here, with the handed-over ``stage_m`` plane and the handed-over ``qn`` / ``kn`` (already
    unbound from the call after #8 with the norm's other buffers; a forward leaves no norm row in the arena).  An arena
    hand-off (``graph="replay"``) changes no binding: its tapes never left the arena (``adopt_tape`` finds them bound)
    and only the entry's references go."""
    leaves = (
        FWD_TAPES_ARENA_AFTER_FUSED
        if _fuse_o_enabled(call)
        else FWD_TAPES_ARENA_AFTER_FWD
    )
    for name in FWD_TAPES:
        t = kept.tapes.get(name)
        call.tapes.pop(name, None)
        if t is not None and name in leaves and call.ws.geometry == kept.geometry:
            call.ws.adopt_tape(name, t)
    kept.clear()
    KEPT_STATS["released"] += 1


def _backward(call: CakeGDNNondetKcpCall, saved: CakeGDNNondetKcpSaved | None) -> None:
    """The backward sequence.  ``saved=None`` is the replay mode (the forward sequence without ``fwd_o`` recomputes
    ``S_in``, the tapes and the owner's c1 receive tensor); otherwise the tapes come from the saved set
    (``save_mode="full"``) or from #1 + the #2 tape walk over the saved ``S_in`` (``"light"``), and the owner's
    adjoint reads ``M_f`` from b1 (``save_transfer=False``) or ``M_f^T`` from the saved c1 receive tensor.
    ``schedule="wait_first"``: ``b1 -> wait -> adjoint merge -> b2 -> wait -> dcarry merges -> adjoint tape walk(s)``.
    Eager path: the saved-set tensors are forgotten at their last reader (``c1_recv`` after the adjoint merge,
    ``M_seg`` before the adjoint tape walk, ``S_in`` before the gradient outputs) when the backward owns the set
    (:class:`CakeGDNNondetKcpSaved` ``release_in_backward``); #8 reads the tapes only, so none of them reaches the backward's peak."""
    tb = call.tb
    wait_first = call.schedule == "wait_first"
    ended_early = False  # the document-end adjoint walk issued under b2 / hop 1 of the ring (overlap schedule)
    merged_end = False  # the dcarry_end merge issued under b2 / hop 1 of the ring (overlap schedule)
    replay = saved is None
    if replay:
        _forward(call, save=False, replay=True)
        m_source = "c1"
        call.scratch["tapes_source"] = "recompute"
    else:
        if call.l2n:
            # the forward's r: read by the pullback (and qk_apply), never written here (the light replay's #1_l2n
            # writes the arena r)
            call.bufs[QK_RSTD] = saved.qk_rstd
            call.scratch["rstd_source"] = "saved"
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
            if call.l2n:
                # the recompute forward's qn / kn (its #1_l2n wrote them from the raw rows, the r of the saved set
                # is its own): bound as they are, so no program re-reads the full raw Q / K to normalize them
                for name in QK_TAPES:
                    call.tapes[name] = kept.qk[name]
                call.scratch["qk_source"] = "kept"
        else:  # light replay: #1 again (gamma / g_cum / akk included) + the #2 tape walk from the saved S_in
            if saved.save_mode in AKK_SAVE_MODES and saved.tapes is not None:
                # "light_akk": the forward's Akk tape, bound before the replay, so #1_l2n runs as the solve-free
                # fwd_prep_replay_l2n (CakeGDNNondetKcpCall.local) and reads it (None only at T_r = 0)
                for name in AKK_TAPES:
                    call.tapes[name] = saved.tape(name)
                call.scratch["akk_source"] = "saved"
            _replay_tapes(call, saved)
            call.scratch["tapes_source"] = "replay"
            if call.l2n:
                call.scratch["qk_source"] = "replay"
    call.local("bwd_prep")
    if call.l2n and call.scratch.get("tapes_source") == "saved":
        # save_mode="full": no #1 replay and no hand-off: qn / kn from the raw rows and the saved r (bitwise
        # #1_l2n's), before the first q / k reader
        call.local("qk_apply")
        call.scratch["qk_source"] = "qk_apply"
    if call.l2n and call.opts.knob("record_qk"):
        _record_qk(call, regen=call.scratch.get("tapes_source") == "replay")
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
        ended_early = not wait_first
        merged_end = ended_early and tb.P > 1
        _ring_backward(call, walk_ends=ended_early)
    else:
        wb1 = call.a2a("b1", "b1_send", "b1_recv", *tb.split_lists(f"b1_{form}"))
        call.wait(wb1)
        wb1 = None
        call.merge(
            "adj_b1" if m_source == "b1" else "adj_c1"
        )  # dS_out slices into the b2 send layout
        if eager:  # the backward's peak follows (the early document-end walk under b2): the merged pools go first
            call.release_pools(BWD_DEAD_AFTER_ADJ_MERGE)
            # the saved c1 receive tensor: the adjoint merge was its last reader
            call.drop_bound("c1_recv", saved)
        wb2 = call.a2a("b2", "b2_send", "b2_recv", *tb.split_lists("b2"))
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
    )  # no exchange (N_out = 0) or the wait_first schedule: every walk runs here
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
    if call.l2n and eager:
        # #8 (or the host reduction) was the last reader of qn / kn and r: the arena copies go, the saved r with them
        call.release_tapes((*QK_TAPES, QK_RSTD))
        call.drop_bound(QK_RSTD, saved)
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
# Keys kept per process, least recently used first out when a new key arrives beyond this.  A checkpointed layer holds
# three keys (the original forward, torch's recompute in its hand-off form, the kept backward; two below the size rule
# GRAPH_RECOMPUTE_MIN_ROWS, where the recompute replays the original forward's graph), so 256 keeps 85 such
# layers; each captured graph holds its own memory pool with its static results (out, the saved set, the gradients)
# besides the workspace arena, released with the entry (or by ``graph_cache_clear()``).
GRAPH_CACHE_MAX = 256


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
        "used",
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
        # the key's last use (the cache drops the least recently used first)
        self.used = next(_GRAPH_USES)

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
# the use order of the cached keys (one hash of the key per lookup: a hit stamps its entry instead of re-inserting)
_GRAPH_USES = iter(range((1 << 63) - 1))


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
    opts: CakeGDNNondetKcpOptions,
    scale,
    tensors,
    flags: tuple,
) -> _GraphEntry | None:
    """The cache entry of this call's validity key, or ``None`` when the call is never captured."""
    if (
        not isinstance(ws, CakeGDNNondetKcpWorkspace)
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
            # the least recently used key first out
            del _GRAPHS[min(_GRAPHS.values(), key=_graph_use).key]
        e = _GRAPHS[key] = _GraphEntry(key, (ctx, ws, backend))
    else:
        e.used = next(_GRAPH_USES)  # a hit makes the key the most recently used
    return e


def _graph_use(e: _GraphEntry) -> int:
    return e.used


def _graph_capture(
    kind: str, entry: _GraphEntry, call: CakeGDNNondetKcpCall, body: Any
) -> Any:
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
    opts: CakeGDNNondetKcpOptions,
    workspace,
    backward: bool,
    dout=None,
    fresh_pools: Sequence[str] = (),
    saved_backward: bool = False,
    packed_grads: bool = True,
):
    """``fresh_pools``: the pools this call binds fresh / saved tensors for (never read from the workspace);
    ``saved_backward``: a backward from a saved set (it never exchanges through the forward's pools);
    ``packed_grads``: a backward may write ``dq`` / ``dk`` / ``dv`` as views of one packed row (:func:`_alloc_grads`)."""
    backend = get_backend(opts.backend)
    if (
        opts.use_qk_l2norm_in_kernel
        and isinstance(backend, KernelBackend)
        and not l2n_records_shipped()
    ):
        raise RuntimeError(_L2N_NOT_SHIPPED)
    q, k, v, g, beta, T, H_qk, HV, batched = _operands(
        q, k, v, g, beta, ctx, dtypes=backend.dtypes, layout=opts.layout
    )
    if opts.use_qk_l2norm_in_kernel:
        # #1_l2n writes qn / kn / r for the tokens of the real chunks only: they must cover [0, T_r) (metadata)
        chunk_len = getattr(ctx, "chunk_len", None)
        if chunk_len is not None:
            covered = _chunk_tokens(ctx, chunk_len)
            if covered != int(T):
                raise ValueError(
                    f"use_qk_l2norm_in_kernel: the context's chunks cover {covered} of {T} tokens"
                )
    if opts.exchange == "ring":
        validate_ring(ctx.n_d_max)
    device = torch.device(ctx.device)
    if dout is not None:
        dout = _squeeze_batch(dout, "dout", 3)
        if not dout.is_contiguous():
            dout = dout.contiguous()
        _check(dout, "dout", (T, HV, D), backend.dtypes["dout"], device)
    per_call = workspace is None
    if per_call:
        workspace = CakeGDNNondetKcpWorkspace(state_dtype=backend.dtypes["state"])
        # the per-call workspace's ticket slots: the context's zeroed set of this stream (fresh under a capture)
        shared = _call_tickets(ctx, device)
        if shared is not None:
            workspace._tickets, workspace._ticket_views = shared
    elif not isinstance(workspace, CakeGDNNondetKcpWorkspace):
        raise TypeError("workspace must be a CakeGDNNondetKcpWorkspace")
    if workspace.state_dtype != backend.dtypes["state"]:
        raise ValueError(
            f"workspace state dtype {workspace.state_dtype} differs from the backend's {backend.dtypes['state']}"
        )
    P = None if opts.segments is None else int(opts.segments)
    prepare = getattr(
        ctx, "prepare", None
    )  # the plan's device tables per (HV, sm_count, P) key, uploaded at first use
    if callable(prepare):
        caches = getattr(ctx, "_caches", None)
        # the bundle is built once per key: later calls skip the lookup
        done = ("kcp_prepared", H_qk, HV, P)
        if not isinstance(caches, dict) or done not in caches:
            prepare(H_qk, HV, P=P)
            if isinstance(caches, dict):
                caches[done] = True
    # Dead pools are not allocated (R2-L1 1a): the ring pools under the a2a exchange and the pools the call binds
    # fresh / saved tensors for (``c1_recv`` under ``save_transfer``, ``stage_m`` when the saved set carries ``M_seg``)
    # never hold a byte of this call.
    skip = list(fresh_pools)
    if opts.exchange != "ring":
        skip += list(RING_POOLS)
        if not backward and opts.graph == "off":
            # the eager a2a forward binds a fresh c2_send plane at its first writer (the fwd merge) and drops it after
            # wait(c2); a captured forward keeps the pool (its address is baked into the graph)
            skip.append("c2_send")
    if per_call and saved_backward:
        # the per-call workspace of a saved-context backward: the forward exchange's pools are never touched (the
        # saved c1 receive tensor, when there is one, is bound instead of c1_recv); a caller-held workspace keeps
        # its documented layout (cake_gdn_nondet_train_kcp_workspace_layout)
        skip += list(SAVED_BWD_DEAD_POOLS)
    tb = workspace.ensure(
        ctx,
        H_qk,
        HV,
        device=device,
        backward=backward,
        fused_walks=int(opts.knob("fused_walks")),
        P=P,
        skip=skip,
        skip_empty=per_call,  # the per-call workspace: no empty pools (made on demand by pool())
    )
    ops = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
    if dout is not None:
        ops["dout"] = dout
    call = CakeGDNNondetKcpCall(
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
        packed_grads=packed_grads,
    )
    call.scratch["workspace"] = "per_call" if per_call else "caller"
    backend.begin_call(call)
    return call, batched


# The numerical mode is in the name: this is the non-deterministic package.  It keeps the deterministic package's
# contract and accuracy budgets, and its results need not equal that package's bit for bit (its fixed-order levers,
# e.g. the ``fill_small`` segment plan, reassociate FP32 sums).  Every launch of this version is fixed-order today, so
# no entry raises PyTorch's alert for an operation without a deterministic implementation and
# ``torch.use_deterministic_algorithms`` changes nothing here; the contract does not promise run-to-run bits.  No entry
# takes a ``deterministic`` keyword (a stray one is a ``TypeError`` of the options validator) and nothing falls back to
# the deterministic package (``flashinfer.cake_gdn_det_train_kcp``).  The mode is never stored in the context; the
# autograd node keeps a private tag (:data:`_MODE`) so that its backward reads only a forward of this package.
_MODE = "nondeterministic"


def _check_saved_mode(saved: Any) -> None:
    """A saved set this package's backward can read: its own :class:`CakeGDNNondetKcpSaved` (a deterministic twin's saved set, or
    any other object, is refused)."""
    if saved is not None and not isinstance(saved, CakeGDNNondetKcpSaved):
        raise ValueError(
            "saved: not a saved set of flashinfer.experimental.cake.cake_gdn_nondet_train_kcp (got "
            f"{type(saved).__module__}.{type(saved).__name__}); a deterministic forward's saved set is read by the "
            "deterministic backward only"
        )


def _option_keywords(fn: Callable[..., Any]) -> Callable[..., Any]:
    """``fn`` with the signature of its call: ``fn``'s parameters without ``**knobs``, then one keyword-only
    parameter per :class:`CakeGDNNondetKcpOptions` field that ``fn`` does not name itself (the field's default).  The
    body still takes those fields through ``**knobs`` and refuses any other name (``TypeError``), so a keyword that
    is no option, ``deterministic`` included, fails ``inspect.signature(fn).bind_partial`` as it fails the call."""
    sig = inspect.signature(fn)
    params = [
        p
        for p in sig.parameters.values()
        if p.kind is not inspect.Parameter.VAR_KEYWORD
    ]
    named = {p.name for p in params}
    params += [
        inspect.Parameter(
            f.name,
            inspect.Parameter.KEYWORD_ONLY,
            default=f.default,
            annotation=f.type,
        )
        for f in dataclasses.fields(CakeGDNNondetKcpOptions)
        if f.name not in named
    ]
    fn.__signature__ = sig.replace(parameters=params)  # type: ignore[attr-defined]
    return fn


def _literal_overrides(options: Any, *pairs: tuple[str, Any, Any]) -> dict[str, Any]:
    """The literal keywords of an entry point as option fields: each ``(name, value, default)`` overrides an
    ``options`` instance / dict only when it differs from its documented default (without ``options`` every value
    is passed, so the validator sees it)."""
    return {
        name: value
        for name, value, default in pairs
        if options is None or value != default
    }


@_option_keywords
def cake_gdn_nondet_train_kcp_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    *,
    scale: float | None = None,
    save_for_backward: bool = True,
    workspace: CakeGDNNondetKcpWorkspace | None = None,
    options: CakeGDNNondetKcpOptions | dict[str, Any] | None = None,
    use_qk_l2norm_in_kernel: bool = _opt.USE_QK_L2NORM_DEFAULT,
    **knobs: Any,
) -> tuple[torch.Tensor, CakeGDNNondetKcpSaved | None]:
    """The KCP forward of this rank's fragments: ``out`` like ``v`` (fresh) and the saved set (``None``
    without ``save_for_backward`` or with ``boundary_mode="replay"``).  ``ctx`` is this rank's ``CakeGDNNondetKcpContext``; the
    token operands are its ``T_r`` local tokens in fragment order; ``options`` / ``**knobs`` are the keyword-only
    fields of :class:`CakeGDNNondetKcpOptions` (``save_mode``, ``boundary_mode``, ``graph``, ...); ``use_qk_l2norm_in_kernel``
    (the one literal keyword; overrides ``options`` only when ``True``): the in-kernel per-head L2 norm of q / k
    (module docstring)."""
    over = _literal_overrides(
        options,
        (
            "use_qk_l2norm_in_kernel",
            use_qk_l2norm_in_kernel,
            _opt.USE_QK_L2NORM_DEFAULT,
        ),
    )
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
        opts=make_options(options, **over, **knobs),
        hand_off=False,
    )


def _in_launch_scope(
    call: CakeGDNNondetKcpCall, body: Any, *args: Any, **kw: Any
) -> Any:
    """``body(*args, **kw)`` -- a forward or backward body, eagerly or inside a capture -- in the backend's launch
    scope of ``call`` (:meth:`Backend.launch_scope`), entered where the body's launches run (inside the capture
    when there is one), so the scope binds their stream; without a scope the body runs as it is."""
    scope = call.backend.launch_scope(call)
    if scope is None:
        return body(*args, **kw)
    with scope:
        return body(*args, **kw)


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
    workspace: CakeGDNNondetKcpWorkspace | None,
    opts: CakeGDNNondetKcpOptions,
    hand_off: bool,
    autograd_node: bool = False,
    executable: bool = True,
    small: bool = False,
    boundary: dict[str, Any] | None = None,
    handle_mode: str | None = None,
) -> tuple[torch.Tensor, CakeGDNNondetKcpSaved | None]:
    """:func:`cake_gdn_nondet_train_kcp_forward` with validated options.  ``boundary``: the recompute handle's entry of this
    layer (the recompute of a forward that stored it): ``S_in`` comes from it and nothing is exchanged.  ``hand_off``: the autograd Function's forward runs inside
    a graph task with ``reuse_recompute_tapes`` -- the light saved set then carries the forward tapes (``saved.kept``)
    for the backward of that task (never with a replay boundary, ``save_mode="full"`` or ``T_r = 0``; under
    ``graph="replay"`` the arena hand-off, :func:`_kept_stale`).  ``autograd_node``: the caller is
    :class:`CakeGDNNondetKcpFunction` (the one forward whose recompute may leave #5 out under ``recompute_output="auto"``);
    ``executable``: that Function's ``apply`` creates a node and packs its saved tensors (grad mode on and an operand
    requiring grad) -- ``False`` launches #5 unconditionally.  Under ``graph="replay"`` the autograd node's forward
    inside a backward pass (torch's recompute) decides the hand-off and #5 before the CUDA-graph lookup
    (:func:`_fwd_decision`): a decision other than (no hand-off, #5 launched) is part of the key, so the hand-off form
    is captured and replayed as its own graph.  Under the size rule of ``reuse_recompute_tapes="auto"``
    (:data:`GRAPH_RECOMPUTE_MIN_ROWS`, a caller-held workspace; ``small``, decided by the Function's forward) no
    forward of the node decides: torch's recompute looks up the original forward's key and runs its form (no hand-off,
    #5 launched; the reasons read ``"small"``)."""
    # with reuse_recompute_tapes the request is the Function's graph-task test (read once, there)
    in_task = bool(hand_off) if opts.reuse_recompute_tapes else None
    save = bool(save_for_backward) and opts.boundary_mode == "saved"
    hand_off = bool(hand_off) and save and opts.save_mode == "light"
    entry = None
    decision = None
    # the size rule of reuse_recompute_tapes="auto" (GRAPH_RECOMPUTE_MIN_ROWS), decided once by the Function's
    # forward: below it no forward of the node decides -- torch's recompute keys and replays the original forward's
    # graph (no frame read, no hand-off)
    small = bool(small) and autograd_node
    if opts.graph != "off":
        if isinstance(workspace, CakeGDNNondetKcpWorkspace):
            workspace.calls += 1
            if (
                autograd_node
                and not small
                and (in_task if in_task is not None else _graph_task_id() != -1)
            ):
                T_r = int(ctx.T_r)
                HV = int(v.shape[-2])
                P = workspace.tables(
                    ctx,
                    HV,
                    fused_walks=int(opts.knob("fused_walks")),
                    P=None if opts.segments is None else int(opts.segments),
                ).P
                decision = _fwd_decision(
                    _saved_count_of(
                        opts, P, T_r, bool(opts.use_qk_l2norm_in_kernel), save
                    ),
                    hand_off=hand_off and T_r > 0,
                    autograd_node=autograd_node,
                    executable=executable,
                    opts=opts,
                )
        flags = (
            (save,)
            if decision is None or (not decision[0] and decision[2])
            else (save, decision[0], decision[2])
        )
        entry = _graph_entry(
            "fwd",
            ctx,
            workspace,
            get_backend(opts.backend),
            opts,
            scale,
            (q, k, v, g, beta),
            flags,
        )
    elif isinstance(workspace, CakeGDNNondetKcpWorkspace):
        workspace.calls += 1
    if entry is not None and entry.replayable(workspace):
        _graph_replay("fwd", entry)
        out, saved, batched = entry.result
        if decision is not None:
            # the reasons of this call's decision (the key fixes the decision, the capture's record its reasons)
            LAST_CALL["fwd"].update(
                hand_off_reason=decision[1], fwd_o_reason=decision[3]
            )
        elif small:
            LAST_CALL["fwd"].update(hand_off_reason="small", fwd_o_reason="small")
        if saved is None or saved.kept is None:
            return _ranked(out.detach(), batched), saved
        return _ranked(out.detach(), batched), _with_stamped_kept(saved)
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
        fresh_pools=_fwd_fresh_pools(opts, save),
    )
    if small:
        # the size rule's form for the eager run and the capture under the original forward's key: no hand-off, #5
        # launched, no frame read
        decision = (False, "small", True, "small", _saved_count(call, save))
    if entry is not None and entry.seen > 0 and entry.disabled is None:
        res = _graph_capture(
            "fwd",
            entry,
            call,
            lambda: _in_launch_scope(
                call,
                _fwd_body,
                call,
                save=save,
                hand_off=hand_off and call.T_r > 0,
                autograd_node=autograd_node,
                executable=executable,
                decision=decision,
            ),
        )
        if res is not None:
            # a hand-off form keeps its kept set's layout as the template of every replay's stamped entry
            entry.result = (res[0], _kept_template(res[1]), batched)
            return _ranked(res[0].detach(), batched), _with_stamped_kept(res[1])
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
            fresh_pools=_fwd_fresh_pools(opts, save),
        )
    s_in = None
    if boundary is not None:
        tb = call.tb
        if boundary.get("ctx") is not call.ctx:
            raise ValueError(
                "recompute_handle: the 'gdn' entry belongs to another CakeGDNNondetKcpContext"
            )
        if boundary.get("key") != _handle_key(
            call.T_r,
            call.H_qk,
            call.HV,
            tb.n_frag,
            tb.P,
            call.scale,
            call.exchange,
            call.precision_mode,
            opts.offset_class,
            opts.transfer_walk,
            call.l2n,
        ):
            raise ValueError(
                "recompute_handle: the 'gdn' entry describes another problem or programs than this recompute"
            )
        s_in = boundary["S_in"]
        _check(
            s_in,
            "recompute_handle S_in",
            (tb.sizes["S_in"],),
            call.state_dtype,
            call.device,
        )
    call.scratch["recompute_handle"] = handle_mode
    out, saved = _in_launch_scope(
        call,
        _fwd_body,
        call,
        save=save,
        hand_off=hand_off and call.T_r > 0,
        autograd_node=autograd_node,
        executable=executable,
        s_in=s_in,
        decision=decision,
    )
    _record("fwd", call)
    if opts.graph != "off":
        _graph_eager("fwd", entry)
    call.backend.end_call(call)
    return _ranked(out, batched), _with_stamped_kept(saved)


def _with_stamped_kept(
    saved: CakeGDNNondetKcpSaved | None,
) -> CakeGDNNondetKcpSaved | None:
    """``saved`` with a freshly stamped arena hand-off when it carries one -- :func:`_kept_stamped` of the hand-off
    forward's entry, or :func:`_kept_rebound` of a replayed graph's template layout (:func:`_kept_template`) --; else
    ``saved`` as it is (no hand-off, or the eager hand-off whose fresh tensors the entry owns).  A shallow copy with the
    one field set (frozen dataclass, hence ``object.__setattr__``): the template stays as it is."""
    kept = None if saved is None else saved.kept
    if kept is None or kept.ws is None:
        return saved
    out = object.__new__(CakeGDNNondetKcpSaved)
    out.__dict__.update(saved.__dict__)
    object.__setattr__(
        out,
        "kept",
        _kept_rebound(kept) if type(kept) is _KeptLayout else _kept_stamped(kept),
    )
    return out


def _kept_template(saved: CakeGDNNondetKcpSaved | None) -> CakeGDNNondetKcpSaved | None:
    """The saved set a captured forward caches as the template of its replays: an arena hand-off's entry becomes its
    :class:`_KeptLayout` (no tensor of the arena stays referenced by the graph cache); else ``saved`` as it is."""
    kept = None if saved is None else saved.kept
    if kept is None or kept.ws is None:
        return saved
    out = object.__new__(CakeGDNNondetKcpSaved)
    out.__dict__.update(saved.__dict__)
    object.__setattr__(out, "kept", _KeptLayout(kept))
    return out


def _saved_count(call: CakeGDNNondetKcpCall, save: bool) -> int:
    """What :class:`CakeGDNNondetKcpFunction` saves for this forward: the five operands and the saved set's tensors
    (:meth:`CakeGDNNondetKcpSaved.tensors`: ``S_in``; the c1 receive tensor under ``save_transfer``; ``M_seg`` under
    ``save_transfer`` or ``save_mode="full"`` at ``P > 1``; the saved tapes of the ``save_mode`` when the rank holds
    tokens; ``qk_rstd`` with ``use_qk_l2norm_in_kernel``) -- known before
    the forward runs, so the frame proof can decide a hand-off (``reuse_recompute_tapes="auto"``)."""
    return _saved_count_of(call.opts, call.tb.P, call.T_r, call.l2n, save)


def _saved_count_of(
    opts: CakeGDNNondetKcpOptions, P: int, T_r: int, l2n: bool, save: bool
) -> int:
    """:func:`_saved_count` from the options, the plan's ``P``, the rank's ``T_r`` and the norm flag (before a call
    exists: the CUDA-graph lookup of a recompute, :func:`_fwd_entry`)."""
    if not save:
        return _SAVED_OPERANDS
    n = (
        1
        + (1 if opts.save_transfer else 0)
        + (1 if _saves_m_seg(opts.save_mode, opts.save_transfer, P) else 0)
    )
    if T_r > 0:
        n += len(saved_tapes(opts.save_mode))
    if l2n:  # qk_rstd, at T_r = 0 too (an empty [2, 0, H_qk] tensor: the flags agree across ranks)
        n += 1
    return _SAVED_OPERANDS + n


# The size rule of ``reuse_recompute_tapes="auto"`` under ``graph="replay"`` (a caller-held workspace): a checkpoint
# recompute decides the hand-off and #5 only when the rank holds at least this many token x value-head rows
# (``T_r * HV``), per architecture of the device.  The node's forward reads the rule once, and only inside a backward
# pass (where a hand-off can follow).  Below it no forward of the node decides -- no checkpoint-frame read, no arena
# hand-off to stamp and check -- the recompute replays the original forward's graph
# (the output walk, #5 inside) and its backward replays #1 and the tape walk.  On a small rank the checkpointed graph
# step is host-bound: the hand-off form launches less GPU work (one B200, 8192 tokens, (2, 6) heads: 356 instead of
# 451 us of kernels) but adds host time per layer and step (the frame read, the stamp, the backward's checks: about
# 48 us there), and the step ends later.  Crossover of one checkpointed layer (CUDA events per step, both forms in one
# process, five rounds): one B200 previous / hand-off 0.953 at 49152 rows, 1.016 / 1.039 / 1.09 at 61440 / 73728 /
# 86016 rows; one GB300 1.013 at 49152 rows, 1.10 at 73728 rows, 1.07-1.13 at 98304 rows; one R200 (208 SMs) 1.07 at
# 49152 rows (the benchmark's checkpointed graph step, L2 flushed).  ``True`` / ``False`` and ``graph="off"`` decide
# at every size.
GRAPH_RECOMPUTE_MIN_ROWS: dict[str, int] = {
    "sm_100a": 61440,
    "sm_103a": 0,
    "sm_107a": 0,
}
# any other device (the fake backend's CPU emulation included)
GRAPH_RECOMPUTE_MIN_ROWS_DEFAULT = 61440
# the architecture of each (backend name, device index) the size rule has read (index -1: a CPU tensor)
_GRAPH_RECOMPUTE_ARCH: dict[tuple[str, int], str | None] = {}


def _graph_recompute_arch(backend: Any, device: torch.device) -> str | None:
    """The architecture the size rule reads: the kernel backend's exported architecture of the CUDA device, ``None``
    for any other backend or device."""
    if not (isinstance(backend, KernelBackend) and device.type == "cuda"):
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    return backend.launcher().device_arch(index)


def _graph_recompute_small(
    opts: CakeGDNNondetKcpOptions, v: torch.Tensor, T_r: int, HV: int
) -> bool:
    """Whether a forward of ``v``'s device under ``graph="replay"`` falls under the size rule:
    ``reuse_recompute_tapes="auto"`` and fewer ``T_r * HV`` rows than :data:`GRAPH_RECOMPUTE_MIN_ROWS` of the
    device's architecture (read once per backend name and device index: the check runs in every graph-mode
    recompute)."""
    if opts.reuse_recompute_tapes != "auto":
        return False
    key = (opts.backend, v.get_device())
    try:
        arch = _GRAPH_RECOMPUTE_ARCH[key]
    except KeyError:
        arch = _GRAPH_RECOMPUTE_ARCH[key] = _graph_recompute_arch(
            get_backend(opts.backend), v.device
        )
    return T_r * HV < GRAPH_RECOMPUTE_MIN_ROWS.get(
        arch, GRAPH_RECOMPUTE_MIN_ROWS_DEFAULT
    )


def _reads_frame(opts: CakeGDNNondetKcpOptions) -> bool:
    """Whether a forward reads torch's checkpoint frame for its hand-off / #5 decision: always with ``graph="off"``;
    with ``graph="replay"`` only inside a backward pass (torch's recompute runs there; the decision is part of the
    recompute's CUDA-graph key)."""
    return opts.graph == "off" or _graph_task_id() != -1


def _fwd_decision(
    n_saved: int,
    *,
    hand_off: bool,
    autograd_node: bool,
    executable: bool,
    opts: CakeGDNNondetKcpOptions,
) -> tuple[bool, str, bool, str, int]:
    """``(hand_off, hand_off_reason, launch, fwd_o_reason, n_saved)`` of a forward whose Function saves ``n_saved``
    tensors: ``hand_off`` (the requested hand-off, ``reuse_recompute_tapes``) and the launch of #5
    (``recompute_output``), from one read of torch's checkpoint frame per saved count (:func:`_recompute_output_dead`);
    counts the #5 decision in ``RECOMPUTE_OUTPUT_STATS``."""
    proof: dict[
        int, tuple[bool, str]
    ] = {}  # the checkpoint-frame proof, read once per saved count

    def frame_proof(n: int) -> tuple[bool, str]:
        if n not in proof:
            proof[n] = _recompute_output_dead(n)
        return proof[n]

    reads = _reads_frame(opts)
    hand_off_reason = "always" if hand_off else "off"
    if hand_off and opts.reuse_recompute_tapes == "auto":
        # keep the tapes only when this node closes the checkpointed region (the frame proof of
        # recompute_output="auto"): then its backward is the region's first and the kept tapes die before any other
        # backward runs.  With a later saver in the region (an output projection, an MLP) the kept set would live
        # through those operations' backward and raise the region's backward peak by the full tape set, so the
        # replay form is taken there; every unreadable state (another torch release, early stop off, user hooks, a
        # non-executable node) is the replay form as well.
        if not (autograd_node and executable and reads):
            hand_off, hand_off_reason = False, "not_executable"
        else:
            hand_off, hand_off_reason = frame_proof(n_saved)
    launch, reason = True, "always"
    if autograd_node and opts.recompute_output == "auto" and reads:
        if not executable:
            # autograd creates no node and packs nothing: the holders still waiting are a later op's
            reason = "not_executable"
        else:
            # the Function saves the five operands and saved.tensors() right after this forward returns
            dead, reason = frame_proof(n_saved)
            launch = not dead
            if reason in _RECOMPUTE_FRAME_REASONS:
                RECOMPUTE_OUTPUT_STATS["skipped" if dead else "launched"] += 1
    return hand_off, hand_off_reason, launch, reason, int(n_saved)


def _fwd_body(
    call: CakeGDNNondetKcpCall,
    *,
    save: bool,
    hand_off: bool = False,
    autograd_node: bool = False,
    executable: bool = True,
    s_in: torch.Tensor | None = None,
    decision: tuple[bool, str, bool, str, int] | None = None,
) -> tuple[torch.Tensor, CakeGDNNondetKcpSaved | None]:
    """The forward's allocations and launches (eagerly or inside a capture): the sequence, ``out`` and #5, the saved
    set.  ``hand_off``: the ten forward tapes (and the ``stage_m`` plane when the saved set carries no ``M_seg``;
    with the norm also ``qn`` / ``kn``) leave the arena as the saved set's ``kept`` hand-off instead of being dropped
    (``reuse_recompute_tapes``); under ``graph="replay"`` they stay bound in the arena and the hand-off references
    them (the arena hand-off, :func:`_kept_stale`).
    ``autograd_node`` with ``recompute_output="auto"``: #5 is left out when the forward is torch's
    checkpoint recompute whose output nothing reads (:func:`_recompute_output_dead`; ``out`` is allocated as ever,
    never written) -- only for an ``executable`` node (the one that packs the saved tensors the proof counts); a
    non-executable one launches with the static reason ``not_executable``.  ``decision``: the
    :func:`_fwd_decision` the CUDA-graph lookup already made for this call (``graph="replay"``); else made here."""
    # #5's decision comes before the forward, which folds #5 into its tape walks when it launches (``fuse_o`` /
    # ``fuse_o_tape``): the frame proof reads the count the Function saves after this forward (memoised per count;
    # this forward packs nothing), asserted against the saved set below
    n_saved = _saved_count(call, save)
    if decision is None:
        decision = _fwd_decision(
            n_saved,
            hand_off=hand_off,
            autograd_node=autograd_node,
            executable=executable,
            opts=call.opts,
        )
    hand_off, hand_off_reason, launch, reason, decided = decision
    if decided != n_saved:
        raise RuntimeError(
            f"the CUDA-graph lookup decided for {decided} saved tensors, the call saves {n_saved}"
        )
    call.scratch["hand_off_reason"] = hand_off_reason
    if hand_off:
        _kept_evict_all()  # one live entry: the previous set is freed before this forward materialises its own
    fuse_o = _fuse_o(call, launch=launch, hand_off=hand_off)
    # a save_mode="full" forward with its saved set writes the saved h / v_new tapes and o in the same walks instead
    fuse_o_tape = _fuse_o_tape(call, launch=launch, save=save)
    _forward(
        call,
        save=save,
        replay=False,
        hand_off=hand_off,
        s_in=s_in,
        fuse_o=fuse_o,
        fuse_o_tape=fuse_o_tape,
    )
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
        if call.l2n and not hand_off:
            # the tape walks were kn's last readers (a hand-off forward hands kn over with its tapes)
            call.release_tapes(("kn",))
        call.release_pools(FWD_DEAD_POOLS)
    if not launch and not fuse_o and call.opts.graph != "off" and call.out is None:
        # graph="replay": the recompute's dead output (no program writes it, torch discards it) would be a static
        # buffer of the captured graph for as long as the graph lives -- [T_r, HV, 128] BF16 beside the original
        # forward's -- so it is a broadcast view of one element instead (same shape, dtype and device)
        call.out = _alloc((1, 1, 1), call.backend.dtypes["out"], call.device).expand(
            call.T_r, call.HV, D
        )
    call.output()  # a fused forward's output walks wrote it; else allocated here, after the releases, for #5
    saved = _saved_set(call) if save else None
    if _SAVED_OPERANDS + (len(saved.tensors()) if saved is not None else 0) != n_saved:
        raise RuntimeError(
            f"_saved_count {n_saved} differs from the saved set the Function packs "
            f"({_SAVED_OPERANDS} + {len(saved.tensors()) if saved is not None else 0})"
        )
    call.scratch["fwd_o"] = (
        "fused"
        if fuse_o
        else "fused_tape"
        if fuse_o_tape
        else ("launched" if launch else "skipped")
    )
    call.scratch["fwd_o_reason"] = reason
    if launch and not (fuse_o or fuse_o_tape):
        call.local("fwd_o")
    if call.l2n and call.opts.graph == "off" and not hand_off:
        call.release_tapes(("qn",))  # #5 was qn's last reader (or #5 is left out)
    kept = None
    if hand_off:
        # with the norm the rows #1_l2n wrote go along: the backward binds them instead of running qk_apply
        qk = {n: call.tape(n) for n in QK_TAPES} if call.l2n else {}
        # the arena hand-off: graph="replay" on a caller-held workspace (the one a captured graph bakes); a per-call
        # workspace is never captured and dies with the call, so its tapes are handed over as in an eager call
        arena = call.opts.graph != "off" and call.scratch.get("workspace") == "caller"
        kept = _KeptTapes(
            {n: call.tape(n) for n in FWD_TAPES},
            stage_m,
            call.ws.geometry,
            _graph_task_id(),
            qk,
            ws=call.ws if arena else None,
        )
        if not arena:
            # unbound: a later forward on this workspace cannot overwrite the kept tensors (the arena hand-off of
            # graph="replay" keeps them bound -- a captured graph bakes their addresses -- and is stamped instead)
            for n in (*FWD_TAPES, *qk):
                call.ws.release_tape(n)
    if saved is not None and kept is not None:
        saved = dataclasses.replace(saved, kept=kept)
    return call.out, saved


def _saved_set(call: CakeGDNNondetKcpCall) -> CakeGDNNondetKcpSaved:
    """The saved set of this forward (``boundary_mode="saved"``, ``kept=None``): the fresh ``S_in``, with
    ``save_transfer`` the c1 receive tensor, ``M_seg`` (``P > 1``, also under ``save_mode="full"``), the saved tapes
    of the ``save_mode`` (none in ``"light"``), the scalars of the call.  Built before #5 so the forward knows what
    the Function saves."""
    tb, opts = call.tb, call.opts
    return CakeGDNNondetKcpSaved(
        version=KCP_SAVED_VERSION,
        S_in=call.bufs["S_in"],
        c1_recv=call.bufs.get("c1_recv") if opts.save_transfer else None,
        M_seg=call.bufs.get("stage_m")
        if _saves_m_seg(opts.save_mode, opts.save_transfer, tb.P)
        else None,
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
        qk_rstd=call.bufs.get(QK_RSTD) if call.l2n else None,
        qk_l2norm=call.l2n,
        kept=None,
    )


def _check_saved(saved: CakeGDNNondetKcpSaved, call: CakeGDNNondetKcpCall) -> None:
    tb, opts = call.tb, call.opts
    if saved.version != KCP_SAVED_VERSION:
        raise ValueError(f"saved: version {saved.version} is not {KCP_SAVED_VERSION}")
    if saved.ctx is not call.ctx:
        raise ValueError(
            "saved: the saved set belongs to another CakeGDNNondetKcpContext"
        )
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
    # the norm flag first: a call that leaves save_mode unset takes its default from it ("light_akk" with the norm,
    # "light" without), so a flag that differs from the forward's also changes that default
    if bool(saved.qk_l2norm) != call.l2n:
        raise ValueError(
            f"saved: use_qk_l2norm_in_kernel={call.l2n} differs from the forward's ({bool(saved.qk_l2norm)})"
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
    if call.l2n:
        if saved.qk_rstd is None:
            raise ValueError(
                "saved: use_qk_l2norm_in_kernel needs the forward's qk_rstd"
            )
        _check(
            saved.qk_rstd,
            "saved.qk_rstd",
            (2, call.T_r, call.H_qk),
            torch.float32,
            call.device,
        )
    elif saved.qk_rstd is not None:
        raise ValueError("saved: a qk_rstd without use_qk_l2norm_in_kernel")
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
    if _saves_m_seg(opts.save_mode, opts.save_transfer, tb.P):
        if saved.M_seg is None:
            raise ValueError(
                'saved: save_transfer or save_mode="full" at P > 1 needs M_seg'
            )
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
        if saved.save_mode != "light" or call.T_r == 0:
            raise ValueError(
                "saved: kept recompute tapes apply to the light saved-context backward with T_r > 0"
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
        # the normalized rows travel with the tapes exactly when the call runs the norm
        if set(kept.qk) != (set(QK_TAPES) if call.l2n else set()):
            raise ValueError(
                "saved: the kept recompute tapes carry qn / kn exactly when use_qk_l2norm_in_kernel is on"
            )
        for name in kept.qk:
            shape, dtype = tape_spec(
                name, R=R, NC_pad=NC_pad, HV=call.HV, T=call.T_r, H_qk=call.H_qk
            )
            _check(kept.qk[name], f"kept.{name}", shape, dtype, call.device)
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


def _saved_fresh_pools(saved: CakeGDNNondetKcpSaved | None) -> tuple[str, ...]:
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


@_option_keywords
def cake_gdn_nondet_train_kcp_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    dout: torch.Tensor,
    *,
    saved: CakeGDNNondetKcpSaved | None = None,
    scale: float | None = None,
    workspace: CakeGDNNondetKcpWorkspace | None = None,
    options: CakeGDNNondetKcpOptions | dict[str, Any] | None = None,
    use_qk_l2norm_in_kernel: bool = _opt.USE_QK_L2NORM_DEFAULT,
    **knobs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The KCP backward: ``(dq, dk, dv, dbeta, dg)``, fresh.  ``saved=None`` replays the forward; a ``saved`` set
    must come from this package's forward.  ``use_qk_l2norm_in_kernel`` must be the forward's (a saved set of the
    other form is refused)."""
    _check_saved_mode(saved)
    over = _literal_overrides(
        options,
        (
            "use_qk_l2norm_in_kernel",
            use_qk_l2norm_in_kernel,
            _opt.USE_QK_L2NORM_DEFAULT,
        ),
    )
    return _bwd_entry(
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
        opts=make_options(options, **over, **knobs),
    )


def _bwd_entry(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    dout: torch.Tensor,
    *,
    saved: CakeGDNNondetKcpSaved | None,
    scale: float | None,
    workspace: CakeGDNNondetKcpWorkspace | None,
    opts: CakeGDNNondetKcpOptions,
    packed_grads: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`cake_gdn_nondet_train_kcp_backward` with validated options and mode.  ``packed_grads``: ``dq`` / ``dk`` / ``dv`` may be
    the head views of one packed row when ``q`` / ``k`` / ``v`` are (:func:`_alloc_grads`); part of the graph key."""
    entry = None
    if isinstance(workspace, CakeGDNNondetKcpWorkspace):
        workspace.calls += 1
    if opts.graph != "off":
        held = () if saved is None else saved.tensors()
        # the kept recompute tapes are part of the key: their backward is its own graph (no #1, no tape walk), and a
        # replay binds the very tensors the capture did (:func:`_kept_key_part`)
        handed, hand = (
            ((), ())
            if saved is None or saved.kept is None
            else _kept_key_part(saved.kept, workspace)
        )
        entry = _graph_entry(
            "bwd",
            ctx,
            workspace,
            get_backend(opts.backend),
            opts,
            scale,
            (q, k, v, g, beta, dout, *held, *handed),
            (saved is not None, packed_grads) + hand,
        )
    if entry is not None and entry.replayable(workspace):
        _graph_replay("bwd", entry)
        if saved is not None and saved.kept is not None:
            # the captured backward read the kept tapes (#8 was their last reader): the entry is consumed
            saved.kept.clear()
            KEPT_STATS["released"] += 1
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
        saved_backward=saved is not None,
        packed_grads=packed_grads,
    )
    if saved is not None:
        _check_saved(saved, call)
    if entry is not None and entry.seen > 0 and entry.disabled is None:
        kept_fields = _kept_fields(None if saved is None else saved.kept)
        res = _graph_capture(
            "bwd", entry, call, lambda: _in_launch_scope(call, _bwd_body, call, saved)
        )
        if res is not None:
            entry.result = (res, batched)
            return tuple(_ranked(t.detach(), batched) for t in res)
        # a capture that failed after the body's #8 released the kept set: the eager form binds it again
        _kept_restore(saved.kept if saved is not None else None, kept_fields)
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
            saved_backward=saved is not None,
            packed_grads=packed_grads,
        )
        if saved is not None:
            _check_saved(saved, call)
    grads = _in_launch_scope(call, _bwd_body, call, saved)
    _record("bwd", call)
    if opts.graph != "off":
        _graph_eager("bwd", entry)
    call.backend.end_call(call)
    return tuple(_ranked(t, batched) for t in grads)


def _kept_fields(kept: _KeptTapes | None) -> tuple | None:
    """The fields of ``kept`` that its release (:func:`_kept_release`) clears, and the release count, before a
    backward capture (:func:`_kept_restore`); ``None`` without a kept set."""
    if kept is None:
        return None
    return (
        dict(kept.tapes),
        kept.stage_m,
        dict(kept.qk),
        kept.ws,
        KEPT_STATS["released"],
    )


def _kept_restore(kept: _KeptTapes | None, fields: tuple | None) -> None:
    """``kept`` as :func:`_kept_fields` found it, after a capture that ran the body (its #8 released the kept set)
    and failed: the eager form that follows binds and releases it once."""
    if kept is None or fields is None:
        return
    kept.tapes, kept.stage_m, kept.qk, kept.ws, KEPT_STATS["released"] = fields


def _kept_tensors(kept: _KeptTapes | None) -> tuple[torch.Tensor, ...]:
    """The tensors of a kept hand-off in a fixed order (the ten :data:`FWD_TAPES`, the ``stage_m`` plane, ``qn`` /
    ``kn``): the backward's CUDA-graph key part of a kept set other than the arena hand-off just taken
    (:func:`_kept_key_part`); ``()`` without one."""
    if kept is None:
        return ()
    out = [kept.tapes[n] for n in FWD_TAPES if n in kept.tapes]
    if kept.stage_m is not None:
        out.append(kept.stage_m)
    out.extend(kept.qk[n] for n in QK_TAPES if n in kept.qk)
    return tuple(out)


def _kept_layout(kept: _KeptTapes) -> tuple:
    """The backward's CUDA-graph key flags of an arena hand-off (:func:`_kept_stale`): the hand-off flag, the arena
    geometry the tapes were produced at and which optional tensors it carries (the ``stage_m`` plane, the ``qn`` /
    ``kn`` rows) -- the tensors themselves are the arena's, pinned by the arena signature of the capture."""
    return (True, kept.geometry, kept.stage_m is not None, tuple(kept.qk))


def _kept_key_part(
    kept: _KeptTapes | None, workspace: Any
) -> tuple[tuple[torch.Tensor, ...], tuple]:
    """``(tensors, flags)`` a kept set adds to the backward's CUDA-graph key.  The arena hand-off this backward just
    took on ``workspace`` (stamped at the workspace's call before the backward's own, which :func:`_bwd_entry` has
    counted) adds its layout only (:func:`_kept_layout`): the take checked that its tensors are the arena's own
    (:func:`_kept_stale`) and a replay requires the arena signature of the capture (:meth:`_GraphEntry.replayable`).
    Any other kept set adds its tensors (:func:`_kept_tensors`) and the flag ``True``; none adds nothing."""
    if (
        kept is not None
        and kept.ws is not None
        and kept.ws is workspace
        and kept.calls == workspace.calls - 1
    ):
        return (), _kept_layout(kept)
    handed = _kept_tensors(kept)
    return handed, ((True,) if handed else ())


def _alloc_grads(call: CakeGDNNondetKcpCall) -> None:
    """The five gradient outputs, fresh per call.  Allocated right before #8 (:func:`_backward` calls this after the
    adjoint tape walk and the ``BWD_DEAD_BEFORE_DQKG`` release): only #8 / #9 write them, and allocating them at the
    top of the backward put 2 x (dq + dk) + dv + dbeta + dg at the backward's peak moment -- the adjoint tape walk --
    for nothing (R2-L1 1b).

    ``dq`` / ``dk`` / ``dv`` are the head views of one packed dQKV row when ``q`` / ``k`` / ``v`` are those of one
    packed row and ``call.packed_grads`` (autograd: the operands are the outputs of :func:`cake_gdn_nondet_train_kcp_packed_qkv_views`'s
    split node, which hands that row on; the explicit backward: always), and three contiguous tensors otherwise."""
    dt = call.backend.dtypes
    T, H_qk, HV = call.T_r, call.H_qk, call.HV
    o = call.ops
    packed = (
        call.packed_grads
        and dt["dq"] == dt["dk"] == dt["dv"] == o["q"].dtype
        and _packed_rows(o["q"], o["k"], o["v"], H_qk, HV) is not None
    )
    if packed:
        # q / k / v are head views of one packed row: dq / dk / dv are the same views of one fresh dQKV row, so the
        # layer below (the convolution's backward) reads one contiguous [T, (2 H_qk + HV) * 128] gradient with no
        # concatenation (:func:`cake_gdn_nondet_train_kcp_packed_qkv_views`).  Any other consumer of the gradients (torch's split, view or
        # slice backward) concatenates or copies them, which is faster from three contiguous tensors than from three
        # column views of one row: those get separate gradients
        qkv = _alloc((T, (2 * H_qk + HV) * D), dt["dq"], call.device)
        a, b = H_qk * D, 2 * H_qk * D
        dq = qkv[:, :a].view(T, H_qk, D)
        dk = qkv[:, a:b].view(T, H_qk, D)
        dv = qkv[:, b:].view(T, HV, D)
    else:
        dq = _alloc((T, H_qk, D), dt["dq"], call.device)
        dk = _alloc((T, H_qk, D), dt["dk"], call.device)
        dv = _alloc((T, HV, D), dt["dv"], call.device)
    call.grad_layout = "packed" if packed else "separate"
    call.grads = {
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "dbeta": _alloc((T, HV), dt["dbeta"], call.device),
        "dg": _alloc((T, HV), dt["dg"], call.device),
    }


def _bwd_body(
    call: CakeGDNNondetKcpCall, saved: CakeGDNNondetKcpSaved | None
) -> tuple[torch.Tensor, ...]:
    """The backward's allocations and launches (eagerly or inside a capture): ``(dq, dk, dv, dbeta, dg)``."""
    _backward(call, saved)
    return tuple(call.grads[n] for n in ("dq", "dk", "dv", "dbeta", "dg"))


# ----------------------------------------------------------------------------------------------------------------
# autograd


# torch.utils.checkpoint's two pack hooks (the original forward's, the recompute's): their module and qualified names,
# the same in torch 2.13 and 2.14; torch 2.14 also marks both ``_checkpoint_internal``, torch 2.13 has no marker
_CHECKPOINT_HOOK_MODULE = "torch.utils.checkpoint"
_CHECKPOINT_PACK_HOOKS = (
    "_checkpoint_hook.__init__.<locals>.pack_hook",
    "_recomputation_hook.__init__.<locals>.pack_hook",
)


def _checkpoint_pack(pack: Any) -> bool:
    """Whether ``pack`` is one of torch's non-reentrant checkpoint pack hooks: marked ``_checkpoint_internal``
    (torch >= 2.14), or -- unwrapped from ``torch._dynamo.disable`` -- named like one (torch 2.13, no marker)."""
    if getattr(pack, "_checkpoint_internal", False):
        return True
    fn = (
        getattr(pack, "_torchdynamo_orig_callable", None)
        or getattr(pack, "__wrapped__", None)
        or pack
    )
    return (
        getattr(fn, "__module__", None) == _CHECKPOINT_HOOK_MODULE
        and getattr(fn, "__qualname__", None) in _CHECKPOINT_PACK_HOOKS
    )


def _in_checkpoint() -> bool:
    """Whether this forward runs under ``torch.utils.checkpoint(use_reentrant=False)``: the top saved-tensors hook pair
    is a checkpoint-internal one (:func:`_checkpoint_pack`) -- the original forward's pack hook and the recompute's
    alike (the hook stack torch reads for the tensors this node saves).  ``False`` when the binding is absent (then
    both forwards see ``False``)."""
    try:
        top = torch._C._autograd._top_saved_tensors_default_hooks(False)
    except Exception:  # noqa: BLE001 - no hook-stack binding: an ordinary forward
        return False
    return top is not None and _checkpoint_pack(top[0])


# (id(opts), save mode) -> (opts, its twin with that save mode): the checkpoint form of a call's options, made once
_CHECKPOINT_TWINS: dict[
    tuple[int, str], tuple[CakeGDNNondetKcpOptions, CakeGDNNondetKcpOptions]
] = {}
CHECKPOINT_TWINS_MAX = 64


def _checkpoint_opts(opts: CakeGDNNondetKcpOptions) -> CakeGDNNondetKcpOptions:
    """The options a forward of :class:`CakeGDNNondetKcpFunction` saves under: ``opts`` itself, except under
    ``torch.utils.checkpoint(use_reentrant=False)`` (:func:`_in_checkpoint`), where an opt-in light save mode
    (:data:`AKK_SAVE_MODES`) saves as ``"light"``: the checkpoint keeps no tape from the original forward, its
    recompute runs inside the backward (where the kept hand-off of ``reuse_recompute_tapes`` applies to ``"light"``),
    and the original forward and the recompute decide alike (they must save the same tensors).  ``"full"`` saves as it
    is."""
    if opts.save_mode not in AKK_SAVE_MODES or not _in_checkpoint():
        return opts
    hit = _CHECKPOINT_TWINS.get((id(opts), "light"))
    if hit is not None and hit[0] is opts:
        return hit[1]
    if len(_CHECKPOINT_TWINS) >= CHECKPOINT_TWINS_MAX:
        _CHECKPOINT_TWINS.clear()
    twin = dataclasses.replace(opts, save_mode="light")
    _CHECKPOINT_TWINS[(id(opts), "light")] = (opts, twin)
    return twin


class CakeGDNNondetKcpFunction(torch.autograd.Function):
    """:func:`cake_gdn_nondet_kcp` as an autograd node: the forward keeps the inputs and the saved set (saved mode) or the
    inputs only (replay mode) through ``save_for_backward`` -- nothing else holds a tensor (#7039); the backward runs
    :func:`cake_gdn_nondet_train_kcp_backward`.  Under ``torch.utils.checkpoint(use_reentrant=False)`` the forward is re-run in the
    backward: the same program order on every rank, so the collectives match.  With ``reuse_recompute_tapes`` that
    re-run (a forward inside the backward's graph task) hands its forward tapes to this node's backward through the
    module's kept-tape registry (never through ``save_for_backward``: the recompute must save exactly what the
    original forward saved), and the backward binds them instead of replaying #1 and the tape walk.  With
    ``recompute_output="auto"`` the recompute of a region this node closes (the last saver) leaves its output program
    out (the frame proof of :func:`_recompute_output_dead`); its output is read by nothing.  ``executable`` (read by
    :func:`cake_gdn_nondet_kcp` before ``apply``, where grad mode is still the caller's) says whether autograd will create
    this node and pack its saved tensors at all; a non-executable node never skips.  ``packed_grads`` (also read by
    :func:`cake_gdn_nondet_kcp` before ``apply``, from the operands' autograd nodes) says whether the backward may write
    ``dq`` / ``dk`` / ``dv`` as views of one packed row: only for the outputs of :func:`cake_gdn_nondet_train_kcp_packed_qkv_views`'s split
    node, which hands that row on (:func:`_feeds_packed_split`).  It is decided in the original forward and kept on
    the node, never read from the operands in the backward (under a non-reentrant checkpoint those are the
    recompute's detached tensors)."""

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
        workspace,
        opts,
        executable,
        handle,
        packed_grads=True,
    ):
        boundary = None if handle is None else _handle_lookup(handle)
        # under a non-reentrant checkpoint the opt-in light save modes save as "light" (the original forward and the
        # recompute alike; _checkpoint_opts); the backward reads the options of the set it gets (fctx.kcp)
        opts = _checkpoint_opts(opts)
        # the graph task (the hand-off's condition), not read where no hand-off can follow (reuse_recompute_tapes=False)
        gid = _graph_task_id() if opts.reuse_recompute_tapes else -1
        # the size rule of "auto" (graph="replay" on a caller-held workspace, the CUDA-graph path), decided once and
        # only where a hand-off could follow (torch's recompute, inside a backward pass); below it no hand-off
        small = (
            gid != -1
            and opts.graph != "off"
            and isinstance(workspace, CakeGDNNondetKcpWorkspace)
            and _graph_recompute_small(opts, v, int(ctx.T_r), v.size(-2))
        )
        if small:
            gid = -1
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
            small=small,
            boundary=boundary,
            handle_mode=None
            if handle is None
            else ("capture" if boundary is None else "replay"),
        )
        if handle is not None and boundary is None:
            # the original forward: its incoming states (the saved set's own tensor) for the recompute
            handle.entries[HANDLE_KEY] = {
                "S_in": saved.S_in,
                "ctx": ctx,
                "key": _handle_key(
                    saved.T_r,
                    saved.H_qk,
                    saved.HV,
                    saved.n_frag,
                    saved.P,
                    saved.scale,
                    saved.exchange,
                    saved.precision_mode,
                    saved.offset_class,
                    saved.transfer_walk,
                    saved.qk_l2norm,
                ),
            }
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
        # this package's tag: the backward reads only a forward of this package
        fctx.kcp_mode = _MODE
        fctx.kcp_packed_grads = bool(packed_grads)
        return out

    @staticmethod
    def backward(fctx, dout):
        # under a non-reentrant checkpoint this unpack runs torch's recompute of the forward
        q, k, v, g, beta, *tensors = fctx.saved_tensors
        ctx, scale, workspace, opts, meta = fctx.kcp
        if getattr(fctx, "kcp_mode", None) != _MODE:
            raise ValueError(
                f"CakeGDNNondetKcpFunction.backward: the forward ran in mode {getattr(fctx, 'kcp_mode', None)!r}, not {_MODE!r}"
            )
        saved = None
        if meta is not None:
            saved = CakeGDNNondetKcpSaved.from_tensors(tensors, meta[0], meta[1], ctx)
            # this node's own set (just built, referenced nowhere else): forgotten tensor by tensor at the last
            # reader; a field outside the comparison, set on the fresh frozen instance (as CakeGDNNondetKcpSaved.forget does)
            object.__setattr__(saved, "release_in_backward", True)
        # the set above holds the only Python references to the saved tensors
        del tensors
        # under "auto" an empty registry has no entry for this node (no graph task to read, no key to build: a direct
        # backward, or a recompute that did not hand off)
        if (
            saved is not None
            and opts.reuse_recompute_tapes
            and saved.save_mode == "light"
            and (opts.reuse_recompute_tapes is True or _KEPT)
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
                # this backward's own saved set (built above): set in place, frozen dataclass
                object.__setattr__(saved, "kept", kept)
        if not dout.is_contiguous():
            dout = dout.contiguous()
        dq, dk, dv, dbeta, dg = _bwd_entry(
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
            opts=opts,
            packed_grads=fctx.kcp_packed_grads,
        )
        return dq, dk, dv, dg, dbeta, None, None, None, None, None, None, None


# cake_gdn_nondet_kcp's validated options per keyword set (no ``options`` object, or an ``options`` dict): an CakeGDNNondetKcpOptions
# built and checked once per distinct keyword values -- and dict entries, taken by value at each call -- (with their
# types, so ``1`` never stands in for ``True``) and shared by the calls -- never handed to the caller, never mutated by
# the package (a recompute handle's form is a dataclasses.replace copy).  An CakeGDNNondetKcpOptions instance is checked per call.
_KEYWORD_OPTIONS: dict[tuple, CakeGDNNondetKcpOptions] = {}
KEYWORD_OPTIONS_MAX = 64
_DICT_OPTIONS = "options"  # the leading mark of a key with an options dict (the keyword-only keys start otherwise)


def _keyword_options(
    over: dict[str, Any],
    knobs: dict[str, Any],
    options: dict[str, Any] | None = None,
) -> CakeGDNNondetKcpOptions:
    try:
        key: tuple[Any, ...]
        key = tuple((k, type(v), v) for k, v in sorted(over.items())) + (
            None,
            *((k, type(v), v) for k, v in sorted(knobs.items())),
        )
        if options is not None:
            key = (
                _DICT_OPTIONS,
                *((k, type(v), v) for k, v in sorted(options.items())),
                None,
                *key,
            )
        opts = _KEYWORD_OPTIONS.get(key)
    except TypeError:  # an unhashable value (a test_knobs dict) or a key that is no str: validated per call
        return make_options(options, **over, **knobs)
    if opts is None:
        opts = make_options(options, **over, **knobs)
        if len(_KEYWORD_OPTIONS) >= KEYWORD_OPTIONS_MAX:
            _KEYWORD_OPTIONS.clear()
        _KEYWORD_OPTIONS[key] = opts
    return opts


# the deterministic package's packed Q/K/V split, by module and class name (never imported here)
_DET_PACKED_SPLIT = (
    "flashinfer.experimental.cake.cake_gdn_det_train_kcp.cake_kcp",
    "CakeGDNDetKcpPackedQKVSplitFunction",
)
# the last two parts of that module: a build that vendors the deterministic package under another parent package
# (or as a top-level package) imports it under another prefix
_DET_PACKED_SPLIT_MODULE_TAIL = "cake_gdn_det_train_kcp.cake_kcp"


def _is_det_packed_split(cls) -> bool:
    """Whether ``cls`` is the deterministic package's packed Q/K/V split class, wherever that package is imported from."""
    module = getattr(cls, "__module__", None) or ""
    return getattr(cls, "__qualname__", None) == _DET_PACKED_SPLIT[1] and (
        module == _DET_PACKED_SPLIT[0]
        or module == _DET_PACKED_SPLIT_MODULE_TAIL
        or module.endswith("." + _DET_PACKED_SPLIT_MODULE_TAIL)
    )


def _feeds_packed_split(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Whether ``q`` / ``k`` / ``v`` are the three outputs, in order, of one :func:`cake_gdn_nondet_train_kcp_packed_qkv_views` split node
    (or of the deterministic package's ``cake_gdn_det_train_kcp_packed_qkv_views`` node): the consumers that hand a packed dQKV row
    on as ``qkv``'s gradient without a copy.  Autograd metadata only (no tensor content read)."""
    node = q.grad_fn
    if node is None or k.grad_fn is not node or v.grad_fn is not node:
        return False
    if (q.output_nr, k.output_nr, v.output_nr) != (0, 1, 2):
        return False
    cls = getattr(type(node), "_forward_cls", None)
    if cls is CakeGDNNondetKcpPackedQKVSplitFunction:
        return True
    return cls is not None and _is_det_packed_split(cls)


PACKED_SPLIT_STATS: dict[str, int] = {
    "aliased": 0,  # split backwards that returned the incoming packed dQKV row itself
    "concatenated": 0,  # split backwards that concatenated (the gradients were not one packed row)
}


def _qkv_views(
    qkv: torch.Tensor, H_qk: int, HV: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lead = tuple(qkv.shape[:-1])
    a, b = H_qk * D, 2 * H_qk * D
    return (
        qkv[..., :a].view(*lead, H_qk, D),
        qkv[..., a:b].view(*lead, H_qk, D),
        qkv[..., b:].view(*lead, HV, D),
    )


class CakeGDNNondetKcpPackedQKVSplitFunction(torch.autograd.Function):
    """:func:`cake_gdn_nondet_train_kcp_packed_qkv_views` as an autograd node.  The backward returns the packed gradient row itself when the
    three incoming gradients are the head views of one packed row (what :func:`cake_gdn_nondet_kcp`'s backward writes for
    this node's outputs when they are packed operands: :func:`_feeds_packed_split`) and concatenates them otherwise
    (a missing gradient counts as zeros)."""

    @staticmethod
    def forward(fctx, qkv, H_qk, HV):
        fctx.meta = (tuple(qkv.shape), qkv.dtype, qkv.device, int(H_qk), int(HV))
        return _qkv_views(qkv, int(H_qk), int(HV))

    @staticmethod
    def backward(fctx, dq, dk, dv):
        shape, dtype, device, H_qk, HV = fctx.meta
        lead = shape[:-1]
        grads = []
        for g, heads in ((dq, H_qk), (dk, H_qk), (dv, HV)):
            if g is None:
                g = torch.zeros((*lead, heads, D), dtype=dtype, device=device)
            grads.append(g)
        row = _packed_rows(*grads, H_qk, HV)
        if row is not None and row.dtype == dtype and tuple(row.shape) == shape:
            PACKED_SPLIT_STATS["aliased"] += 1
            return row, None, None
        PACKED_SPLIT_STATS["concatenated"] += 1
        return torch.cat([g.flatten(-2) for g in grads], dim=-1).to(dtype), None, None


def cake_gdn_nondet_train_kcp_packed_qkv_views(
    qkv: torch.Tensor, *, H_qk: int, HV: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``q`` / ``k`` / ``v`` as head views ``[..., T, H_qk | H_qk | HV, 128]`` of a packed
    ``[..., T, (2 H_qk + HV) * 128]`` row (the convolution's output), with no copy.

    :func:`cake_gdn_nondet_kcp` takes the views as they are (token stride ``(2 H_qk + HV) * 128``) and its backward writes
    ``dq`` / ``dk`` / ``dv`` as the same views of one fresh packed row; this split's backward hands that row on as
    the gradient of ``qkv`` without a concatenation.  Any other incoming gradient is concatenated.

    Needs a dense channel axis (``qkv.stride(-1) == 1``); the backward skips the concatenation only when ``qkv`` is
    dense over ``[..., T, (2 H_qk + HV) * 128]`` (token stride exactly ``(2 H_qk + HV) * 128``) and the GDN call keeps
    ``layout="tma"``.  The views are the outputs of one custom autograd node:
    hooks and ``retain_grad`` on them fire as on any autograd output; a missing gradient counts as zeros and a
    hook that returns a new gradient makes the backward concatenate (``PACKED_SPLIT_STATS``); autograd refuses an
    in-place change to the views while ``qkv`` requires grad."""
    H_qk, HV = int(H_qk), int(HV)
    validate_heads(H_qk, HV)
    C = (2 * H_qk + HV) * D
    if qkv.dim() < 2 or qkv.shape[-1] != C:
        raise ValueError(
            f"qkv: expected [..., T, {C}] for H_qk={H_qk}, HV={HV}, got shape {tuple(qkv.shape)}"
        )
    if qkv.stride(-1) != 1:
        raise ValueError(
            f"qkv: the channel axis must be dense (strides {qkv.stride()})"
        )
    return CakeGDNNondetKcpPackedQKVSplitFunction.apply(qkv, H_qk, HV)


@_option_keywords
def cake_gdn_nondet_kcp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ctx: Any,
    *,
    scale: float | None = None,
    save_mode: str | None = None,
    boundary_mode: str = _opt.BOUNDARY_MODE_DEFAULT,
    workspace: CakeGDNNondetKcpWorkspace | None = None,
    graph: str = _opt.GRAPH_DEFAULT,
    options: CakeGDNNondetKcpOptions | dict[str, Any] | None = None,
    recompute_handle: Any = None,
    use_qk_l2norm_in_kernel: bool = _opt.USE_QK_L2NORM_DEFAULT,
    **knobs: Any,
) -> torch.Tensor:
    """The differentiable KCP forward: ``out`` like ``v``; gradients for ``q, k, v, g, beta``
    in the backward's order ``(dq, dk, dv, dbeta, dg)``.  With an ``options`` instance, the four literal keywords
    (``save_mode``, ``boundary_mode``, ``graph``, ``use_qk_l2norm_in_kernel``) override it only when they differ from
    their documented defaults.  ``save_mode`` left unset (``None``) runs ``"light_akk"`` with
    ``use_qk_l2norm_in_kernel=True`` and ``"light"`` without it (:class:`CakeGDNNondetKcpOptions`; under
    ``torch.utils.checkpoint(use_reentrant=False)`` ``"light_akk"`` saves as ``"light"``).
    ``use_qk_l2norm_in_kernel=True``: the in-kernel per-head L2 norm of q / k (the
    gradient is the exact derivative of ``bf16(q * r)``'s FP32 form at the stored point; module docstring).  ``recompute_handle``
    (:class:`CakeGDNNondetKcpRecomputeHandle`): the checkpoint recompute of this forward exchanges nothing -- it reads the
    incoming states the original forward stored in the handle; needs ``boundary_mode="saved"`` and ``graph="off"``,
    implies ``save_transfer=False`` (a ``save_transfer=True`` keyword or ``options`` dict entry is refused; a
    :class:`CakeGDNNondetKcpOptions` instance's value is replaced by ``False``) and refuses ``debug_sequence``.

    ``q`` / ``k`` / ``v`` that are head views of one packed ``[..., T, (2 H_qk + HV) * 128]`` row get the full backward
    speed as :func:`cake_gdn_nondet_train_kcp_packed_qkv_views` views (conditions: README, "Packed Q/K/V").  Plain ``split`` / ``view``
    views of the row run the same programs on the same operands and give the same results, but the backward writes
    three contiguous ``dq`` / ``dk`` / ``dv`` for them and torch's split backward concatenates them into a new row:
    about 0.2 ms (6-7 %) more backward per call at ``(16, 48)`` and 32,768 tokens on B200 and GB300 and 0.17 ms
    (7-8 %) on R200, with the in-kernel norm off or on (README, "Packed Q/K/V")."""
    over = _literal_overrides(
        options,
        ("save_mode", save_mode, None),
        ("boundary_mode", boundary_mode, _opt.BOUNDARY_MODE_DEFAULT),
        ("graph", graph, _opt.GRAPH_DEFAULT),
        (
            "use_qk_l2norm_in_kernel",
            use_qk_l2norm_in_kernel,
            _opt.USE_QK_L2NORM_DEFAULT,
        ),
    )
    if recompute_handle is not None and (
        knobs.get("save_transfer") is True
        or (isinstance(options, dict) and options.get("save_transfer") is True)
    ):
        raise ValueError(
            "recompute_handle keeps the incoming states only: it needs save_transfer=False"
        )
    opts = (
        _keyword_options(over, knobs, options)
        if options is None or type(options) is dict
        else make_options(options, **over, **knobs)
    )
    if recompute_handle is not None:
        if opts.boundary_mode != "saved" or opts.graph != "off":
            raise ValueError(
                "recompute_handle needs boundary_mode='saved' and graph='off', got "
                f"boundary_mode={opts.boundary_mode!r}, graph={opts.graph!r}"
            )
        if opts.knob("debug_sequence"):
            raise ValueError(
                "recompute_handle refuses debug_sequence (it gathers the issue order across ranks)"
            )
        if opts.save_transfer:
            opts = dataclasses.replace(opts, save_transfer=False)
    # autograd's own executability test for the node (grad mode and an operand requiring grad, flags only): read
    # here because inside Function.forward grad mode is always off
    executable = torch.is_grad_enabled() and any(
        t.requires_grad for t in (q, k, v, g, beta)
    )
    return CakeGDNNondetKcpFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        scale,
        workspace,
        opts,
        executable,
        recompute_handle,
        executable and _feeds_packed_split(q, k, v),
    )


# ----------------------------------------------------------------------------------------------------------------
# Byte queries (literal public forms; the pure-CPU formulas live in the package's byte module)


# The public byte queries are the byte module's functions themselves: their keyword-only parameters are the call's.
cake_gdn_nondet_train_kcp_comm_bytes = cake_bytes.cake_gdn_nondet_train_kcp_comm_bytes
cake_gdn_nondet_train_kcp_saved_bytes = cake_bytes.cake_gdn_nondet_train_kcp_saved_bytes
cake_gdn_nondet_train_kcp_workspace_bytes = (
    cake_bytes.cake_gdn_nondet_train_kcp_workspace_bytes
)


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
    "HOST_L2N_BLOCK_TOKENS",
    "HOST_ONLY_ARGS",
    "HOST_REDUCE_TAPES",
    "KCP_SAVED_VERSION",
    "LAST_CALL",
    "BWD_DEAD_AFTER_ADJ_MERGE",
    "DHU_OFFSET_TAPE_STANDIN",
    "FWDH_OFFSET_TAPE_STANDIN",
    "DHU_PROGRAMS",
    "BWD_DEAD_BEFORE_DQKG",
    "BWD_DEAD_BEFORE_TAPES",
    "RING_POOLS",
    "SAVE_TRANSFER_FRESH_POOLS",
    "FWD_DEAD_AFTER_PREP",
    "FWD_DEAD_BEFORE_O",
    "FWD_DEAD_POOLS",
    "FWD_TAPES_ARENA_AFTER_FWD",
    "FWD_TAPES_ARENA_AFTER_FUSED",
    "KEPT_STALE_STATS",
    "KEPT_STATS",
    "L2N_PROGRAMS",
    "L2N_SIBLING_OF",
    "LIGHT_TAPES",
    "AKK_REPLAY_OF",
    "AKK_REPLAY_PROGRAMS",
    "AKK_SAVE_MODES",
    "AKK_TAPES",
    "LIGHT_FORWARD_SAVE_MODES",
    "HANDLE_KEY",
    "TAPE_ALIASES",
    "LOCAL_PROGRAMS",
    "OFFSET_CLASSES",
    "PACKED_SPLIT_STATS",
    "PROGRAMS",
    "PROGRAM_ARGS",
    "PROGRAM_ARGS_L2N",
    "PROVISIONAL_ARGS",
    "QK_RSTD",
    "QK_TAPES",
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
    "CakeGDNNondetKcpFunction",
    "FakeBackend",
    "Indexed",
    "CakeGDNNondetKcpCall",
    "CakeGDNNondetKcpOptions",
    "CakeGDNNondetKcpRecomputeHandle",
    "CakeGDNNondetKcpSaved",
    "CakeGDNNondetKcpWorkspace",
    "KernelBackend",
    "CakeGDNNondetKcpPackedQKVSplitFunction",
    "cake_gdn_nondet_kcp",
    "cake_gdn_nondet_train_kcp_backward",
    "cake_gdn_nondet_train_kcp_forward",
    "dhu_vh",
    "fwdh_vh",
    "gather_states",
    "cake_gdn_nondet_train_kcp_comm_bytes",
    "cake_gdn_nondet_train_kcp_saved_bytes",
    "cake_gdn_nondet_train_kcp_workspace_bytes",
    "get_backend",
    "graph_cache_clear",
    "fused_o_shipped",
    "fused_o_tape_shipped",
    "kept_registry_size",
    "l2n_records_shipped",
    "make_options",
    "merge_cols_per_cta",
    "cake_gdn_nondet_train_kcp_packed_qkv_views",
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
