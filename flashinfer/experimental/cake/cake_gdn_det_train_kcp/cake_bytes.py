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

Byte queries of the GDN KCP route (pure CPU, from the context and the head
pair; keyword-only beyond ``ctx``).  ``s = HV * 128 * 128 * 4`` is one FP32
state of all value heads (3 MiB at ``HV = 48``); ``n_in`` / ``n_out`` are this
rank's crossing-in / crossing-out fragments (a fragment with a predecessor /
with a successor in its document chain), ``N_in`` / ``N_out`` the group totals,
``n_io`` / ``N_io`` the fragments with both (``hp.n_io_by_rank``: the maps that
travel on b1), ``Hg = ceil(HV / W)`` the heads per owner.

* ``cake_gdn_det_train_kcp_comm_bytes(ctx, *, H_qk, HV, C)``: bytes this rank moves per
  collective and per boundary.  Per crossing-out fragment c1 sends ``2 s``
  (``[B_f | M_f^T]`` pair tiles) and b2 receives ``s`` (``dS_out``); per
  crossing-in fragment c2 receives ``s`` (``S_in``) and b1 sends ``s`` (``d_f``);
  per fragment crossing in and out b1 also sends ``s`` (the re-sent ``M_f^T``,
  which the owner's adjoint chain reads; a document's last fragment has none)
  when the maps are not kept on the owner (``boundary_mode="saved"`` without
  ``save_transfer``), i.e. ``b1_send = s n_in + s n_io [not save_transfer]``
  (reported as ``b1_design``).  Owner side: c1 receives ``2 s_owner`` per crossing-out
  fragment of the group; b1 receives ``s_owner`` per crossing-in fragment (the
  ``d`` plane, ``b1_d``) plus ``s_owner`` per fragment crossing in and out when the
  maps are re-sent (the ``m`` plane, ``b1_m``); c2 / b2 send ``s_owner`` per
  crossing-in / crossing-out fragment of the group, with
  ``s_owner = Hg_r * 128 * 128 * 4``.  ``b1`` reports the bytes of the plan's
  b1 split tables (``hp.b1_splits``: what the host allocates and the collective
  moves): ``d_f`` plane + map plane; ``b1_design`` is the same total built
  from the fragment counts (the padded owner slots included, so ``b1`` ==
  ``b1_design``).  Halo: ``rows * C * 2`` forward (BF16 ``x`` history rows), ``rows * C * 4`` backward (FP32 ``dpre`` rows).  ``B1`` =
  forward (c1 + c2 + halo_fwd), ``B2`` = backward from the saved context
  (b1 + b2 + halo_bwd), ``B3`` = the replayed forward + the backward with b1 in
  its offsets-only form.  ``H_qk`` enters the exchange terms only through the
  default channel count ``C = (2 H_qk + HV) * 128`` of the packed QKV
  convolution (10,240 at (16, 48)); the state terms depend on ``HV`` alone.
  The padded owner (``HV % Hg != 0``) carries ``pad_tiles`` never-read tiles
  per slot, reported under ``padding`` and included in the totals (the buffers
  the ranks actually move).
* ``cake_gdn_det_train_kcp_saved_bytes(ctx, *, H_qk, HV, save_mode, boundary_mode, ...)``: the
  saved set of one layer on this rank (design table 4.1): ``S_in`` (``n_in s_pad``:
  the receive buffer, padded to whole owners, ``s_pad = n_owners Hg 128 * 128 * 4``;
  ``n_in s`` when ``HV % Hg == 0``), the owner's c1 receive tensor with
  ``save_transfer`` (``2 N_out s_owner``), ``M_seg`` at ``P > 1``
  (``P n_frag s``, the whole ``M_seg^T`` plane; see below) and, with ``with_conv``
  (the default), the layer convolution's history rows (``n_hist_rows * C * 2``:
  the module's convolution saves them; this package's entry points do not, so
  ``with_conv=False`` is the saved set of a GDN call);
  ``save_mode="light"`` keeps no tape (its backward re-runs #1 and the #2 tape
  walk); ``save_mode="full"`` adds every forward tape: the ``h`` tape
  ``[NC_pad, HV, 128, 128]``, ``v_new`` / ``w`` / ``u`` ``[R, HV, 128]``, ``Aqk`` and
  ``Akk`` ``[R, HV, 64]`` (BF16), ``gamma [NC_pad, HV]`` and the FP32 row factors
  ``gcum`` / ``rowscale`` / ``qscale`` ``[R, HV]`` (the host skips #1 and the walk
  entirely).  ``M_seg`` is retained with
  ``save_transfer`` and with ``save_mode="full"`` (whose backward runs no transfer
  walk); the light ``save_transfer=False`` form re-runs the transfer walks
  (KDA semantics).  ``boundary_mode="replay"`` keeps no ``S_in`` /
  c1 tensor / ``M_seg``, no tape and no ``qk_rstd`` in either ``save_mode`` (the
  backward re-runs the whole forward: offset and transfer walks, exchange, merge
  and the tapes; ``save_mode="full"`` there runs the same backward as
  ``"light"``).  Inputs are references (0 extra bytes); the int32
  tables live on the context (not per layer).
* ``cake_gdn_det_train_kcp_workspace_bytes(ctx, *, H_qk, HV, ...)``: the caller-held
  workspace of one rank: the exchange pools (the send pools, the c1 / b1 / b2
  receive pools a call does not replace with a saved tensor, the ``P > 1``
  staging and carry planes and ``M_f^T`` tiles, the halo send rows of the
  layer's convolution with ``with_conv``), plus the local transients the programs
  write and the backward consumes (the tapes a mode does not save, the
  ``rowscale`` / ``qscale`` rows, the adjoint tapes ``dh`` / ``do_g`` /
  ``dv_intra`` (``dv_new`` is stored in place of ``dv_intra``: one buffer,
  counted once), the ``dg`` partials and, with the host form of the
  grouped ``dq`` / ``dk`` reduction, two FP32 ``[T_r, HV, 128]`` partial buffers)
  and the persistent programs' dynamic-ticket slots (one 128-byte u32 word per
  launch site, zeroed once with the workspace, never read by the host).
  ``forward`` / ``backward`` are the two lifetimes, the workspace their maximum,
  for the calls :func:`cake_gdn_det_train_kcp_workspace_layout` names as covered (``exchange="a2a"``
  under the ``wait_first`` schedule; ``graph`` ``"off"`` / ``"replay"``; with or
  without a saved set).  The c2 receive buffer ``S_in`` is never a workspace
  term: every call allocates it fresh (the saved set's with a saved set); a call
  without one also holds that fresh ``n_in s_pad`` plane from the c1 release
  to the end of its forward, or to #8 of a replay backward (under
  ``graph="replay"`` to the end of the backward).  The local terms are
  the design's tape sizes; the kernels' own scratch is added when the programs
  exist (every term is listed by :func:`cake_gdn_det_train_kcp_workspace_layout` so the host
  can account for it).

* ``use_qk_l2norm_in_kernel=True`` (the in-kernel Q / K L2 norm): the saved
  set gains ``qk_rstd`` (FP32 ``[2, T_r, H_qk]``: ``8 T_r H_qk`` bytes, 4 MiB
  at ``T_r = 32768``, ``H_qk = 16``; ``boundary_mode="saved"`` only) and, with
  ``save_mode="full"``, the normalized rows ``qn`` / ``kn`` (BF16
  ``[T_r, H_qk, 128]`` each, ``2 * 256 T_r H_qk`` bytes together, 256 MiB at
  the same shape: the full backward binds them, so it rebuilds nothing); the
  workspace gains those rows as transient arena tapes in the forward's tape walk
  and in every backward phase of the replay form when no saved set keeps them
  (and in the kept form's recompute forward, which hands them to its backward
  with the tapes: there they are hand-off tensors, ``bwd.kept_tapes``, not
  workspace), plus the arena
  ``qk_rstd`` in the backward forms that re-run #1
  (the light replay of a saved set and the replay of a call without one; the
  ``save_mode="full"`` and kept-tape backwards read the saved one) and at #1 of
  a forward that keeps no saved set (``boundary_mode="replay"``,
  ``save_for_backward=False``).  The eager forward drops ``kn`` before its
  output, ``qn`` after #5 and the arena ``r`` after #1, so nothing of the norm
  is left in the arena after a forward.  With the norm, a forward whose
  ``akk`` / ``gcum`` go to no backward (every forward but a hand-off recompute
  and a ``save_mode="full"`` save) runs the forward form of #1_l2n where the
  registry ships it (``cake_kcp.prep_form_shipped``): its #1 binds neither
  tape, so its arena never holds them.

All sizes are exact integers of the shapes the host allocates (``R = 64
NC_pad`` padded token rows, ``NC_pad`` the even chunk bound), not estimates;
the workspace lifetimes hold for the calls :func:`cake_gdn_det_train_kcp_workspace_layout`
covers.
"""

from __future__ import annotations

from . import cake_jit
from .cake_kcp_plan import (
    CHUNK,
    HEAD_DIM,
    MAX_SEGMENTS,
    SAVE_TRANSFER_DEFAULT,
    STATE_TILE_BYTES,
    CakeGDNDetKcpContext,
)
from .cake_options import REUSE_RECOMPUTE_TAPES_DEFAULT

SAVE_MODES = ("light", "full")
BOUNDARY_MODES = ("saved", "replay")
GROUP_REDUCE = ("fused", "host")
GRAPH_MODES = ("off", "replay")
# Bytes per (padded token row, value head) of the token-row tapes.
ROW_BF16_128 = HEAD_DIM * 2  # w, u, v_new, do_g, dv_intra (= dv_new): bf16 [R, HV, 128]
ROW_BF16_64 = 64 * 2  # aqk, akk: bf16 [R, HV, 64]
ROW_F32 = 4  # g_cum, rowscale, qscale, dg_part: f32 [R, HV]
CHUNK_F32 = 4  # gamma, gamma_dot: f32 [NC_pad, HV]
TAPE_BF16_TILE = HEAD_DIM * HEAD_DIM * 2  # h, dh: bf16 [NC_pad, HV, 128, 128]
# The persistent programs' dynamic-ticket words: one 128-byte u32 slot per persistent launch site (#5 ``fwd_o``,
# #6 ``bwd_prep``, #8 ``bwd_dqkg``) in the caller-held workspace.  The slots are zeroed once when the workspace is
# (re)allocated, every launch leaves its word at zero (the kernels' self-resetting ticket) and the host never reads
# them; the workspace allocates every site's slot together, so both lifetimes count all of them.
TICKET_SITES = ("fwd_o", "bwd_prep", "bwd_dqkg")
TICKET_SLOT_BYTES = 128
TICKET_BYTES = len(TICKET_SITES) * TICKET_SLOT_BYTES
# The identity tile pool ``eye`` (one FP32 [128, 128] state; the zero block's never-read base, the merge's identity
# operand): built once per context and bound by every workspace of it.
EYE_BYTES = STATE_TILE_BYTES
# Tape lifetimes of the arena (this module's spellings; mirrors of cake_kcp.FWD_DEAD_AFTER_PREP / FWD_TAPES_ARENA_AFTER_FWD,
# asserted equal by the tests): #1's backward-only outputs leave an eager forward's arena right after #1; the four
# tapes #5 reads are what an eager forward leaves bound, and what a kept-tape backward re-adopts after #8.
FWD_TAPES_DEAD_AFTER_PREP = ("akk", "gcum")
FWD_TAPES_ARENA_AFTER_FWD = ("aqk", "qscale", "h_tape", "v_new")
# A fused forward (the output walk ``kcp_fwdh_out``: #5 inside the tape walks) binds no h / v_new tape: it leaves the two
# tapes #5 reads besides them (cake_kcp.FWD_TAPES_ARENA_AFTER_FUSED), and its peak is #1's (akk / gcum still bound).
FWD_TAPES_ARENA_AFTER_FUSED = ("aqk", "qscale")
FUSED_FREE_TAPES = ("h_tape", "v_new")
TAPE_NAME_OF = {"gcum": "g_cum"}  # this module's spelling -> cake_kcp's
# The in-kernel Q / K L2 norm (``use_qk_l2norm_in_kernel``): bytes per (token, Q/K head) of one normalized row tape
# (``qn`` / ``kn``: BF16 [T_r, H_qk, 128]) and of the FP32 ``r`` of q and k together (``qk_rstd`` [2, T_r, H_qk]).
QK_ROW_BF16 = HEAD_DIM * 2
QK_RSTD_BYTES = 2 * 4


def _flag(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _kept_form(reuse_recompute_tapes: bool | str | None, graph: str = "off") -> bool:
    """Whether the workspace terms are the kept-tape backward's (``True``; ``None`` = the package default) or the
    replay form's (``False``; ``"auto"`` = the bound of both forms, the replay's).  Under ``graph="replay"`` the
    default is ``False`` and ``True`` / ``"auto"`` are refused, as the call's options do."""
    if graph != "off":
        if reuse_recompute_tapes is None or reuse_recompute_tapes is False:
            return False
        raise ValueError(
            f"reuse_recompute_tapes={reuse_recompute_tapes!r} needs graph='off' (a captured graph keeps no tapes)"
        )
    value = (
        REUSE_RECOMPUTE_TAPES_DEFAULT
        if reuse_recompute_tapes is None
        else reuse_recompute_tapes
    )
    if value is True:
        return True
    if value is False or value == "auto":
        return False
    raise ValueError(
        f"reuse_recompute_tapes must be True, False, 'auto' or None, got {reuse_recompute_tapes!r}"
    )


def _check(H_qk: int, HV: int) -> tuple[int, int]:
    H_qk, HV = int(H_qk), int(HV)
    if H_qk < 1 or HV < 1:
        raise ValueError(f"H_qk={H_qk} and HV={HV} must be >= 1")
    if HV % H_qk:
        raise ValueError(f"HV={HV} must be a multiple of H_qk={H_qk}")
    return H_qk, HV


def _channels(H_qk: int, HV: int, C) -> int:
    """The convolution channel count: the caller's ``C`` or the packed QKV width ``(2 H_qk + HV) * 128``."""
    if C is None:
        return (2 * H_qk + HV) * HEAD_DIM
    C = int(C)
    if C < 1:
        raise ValueError(f"C={C} must be >= 1")
    return C


def _mode(value: str, allowed: tuple[str, ...], name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    return value


def state_bytes(HV: int) -> int:
    """``s``: one FP32 ``[HV, 128, 128]`` state."""
    return int(HV) * STATE_TILE_BYTES


def _segments(ctx: CakeGDNDetKcpContext, plan_P: int, P: int | None) -> int:
    """``P`` segments per fragment of a byte query: the plan's (``plan_P``) when ``P`` is ``None``, else the caller's
    override (the ``segments`` option of the call) in the range the call accepts, ``[1, min(MAX_SEGMENTS, chunks of
    the longest fragment of this rank)]`` (:meth:`CakeGDNDetKcpContext.heads`); read from the fragments, no plan is built."""
    if P is None:
        return plan_P
    P = int(P)
    nc_max_frag = max((-(-f.length // CHUNK) for f in ctx.fragments), default=0)
    p_max = min(MAX_SEGMENTS, max(1, nc_max_frag))
    if not 1 <= P <= p_max:
        raise ValueError(
            f"P={P} override must be in [1, {p_max}] (MAX_SEGMENTS, chunks of the longest fragment)"
        )
    return P


def cake_gdn_det_train_kcp_comm_bytes(
    ctx: CakeGDNDetKcpContext,
    *,
    H_qk: int,
    HV: int,
    C: int | None = None,
    save_transfer: bool = SAVE_TRANSFER_DEFAULT,
    boundary_mode: str = "saved",
) -> dict:
    """Bytes this rank moves per collective (``send`` / ``recv``) and per boundary (``B1`` / ``B2`` / ``B3``), with
    the crossing-fragment counts, the per-fragment state bytes and the padding (module docstring)."""
    H_qk, HV = _check(H_qk, HV)
    C = _channels(H_qk, HV, C)
    _mode(boundary_mode, BOUNDARY_MODES, "boundary_mode")
    hp = ctx.heads(HV)
    sp, halo = hp.splits, ctx.halo
    with_maps = boundary_mode == "saved" and not save_transfer
    s_pad = (
        hp.n_owners * hp.Hg * STATE_TILE_BYTES
    )  # one state over the (padded) owner slots; = s without padding
    s_owner = (
        hp.Hg_pad_by_rank[ctx.rank] * STATE_TILE_BYTES
    )  # this rank's owner slot (0 for a non-owner)
    b1_d = {
        "send": ctx.n_in * s_pad,
        "recv": ctx.N_in * s_owner,
    }  # d_f of every crossing-in fragment
    n_io, N_io = hp.n_io_by_rank[ctx.rank], sum(hp.n_io_by_rank)
    b1_m = {  # M_f^T re-sent by every fragment crossing in and out unless the owner kept the c1 pairs
        "send": n_io * s_pad if with_maps else 0,
        "recv": N_io * s_owner if with_maps else 0,
    }
    b1_buf_send, b1_buf_recv = hp.b1_splits(
        with_maps
    )  # the plan's split tables (allocated buffers)
    b1_design = {  # d_f plane + re-sent maps from the counts, padding included (= b1)
        "send": b1_d["send"] + b1_m["send"],
        "recv": b1_d["recv"] + b1_m["recv"],
    }
    coll = {
        "c1": {"send": sum(sp["c1_send"]) * 4, "recv": sum(sp["c1_recv"]) * 4},
        "c2": {"send": sum(sp["c2_send"]) * 4, "recv": sum(sp["c2_recv"]) * 4},
        "b1": {  # the allocated split tables = what the collective moves
            "send": sum(b1_buf_send) * 4,
            "recv": sum(b1_buf_recv) * 4,
        },
        "b2": {"send": sum(sp["b2_send"]) * 4, "recv": sum(sp["b2_recv"]) * 4},
        "halo_fwd": {
            "send": sum(halo.fwd_send_rows) * C * 2,
            "recv": sum(halo.fwd_recv_rows) * C * 2,
        },
        "halo_bwd": {
            "send": sum(halo.bwd_send_rows) * C * 4,
            "recv": sum(halo.bwd_recv_rows) * C * 4,
        },
    }
    b1_replay = dict(b1_d)  # the replay re-runs the transfer walks: offsets only on b1

    def total(*parts):
        return {
            "send": sum(p["send"] for p in parts),
            "recv": sum(p["recv"] for p in parts),
        }

    fwd = total(coll["c1"], coll["c2"], coll["halo_fwd"])
    bwd = total(coll["b1"], coll["b2"], coll["halo_bwd"])
    b3 = total(fwd, b1_replay, coll["b2"], coll["halo_bwd"])
    s = state_bytes(HV)
    pn = hp.pad_numel
    pad_slot = (
        hp.pad_tiles * STATE_TILE_BYTES
    )  # never-read tiles per fragment slot over the owners
    pad_self = (
        hp.Hg_pad_by_rank[ctx.rank] - hp.Hg_r
    ) * STATE_TILE_BYTES  # this owner's padding per slot
    n_b1_send = ctx.n_in + (n_io if with_maps else 0)
    n_b1_recv = ctx.N_in + (N_io if with_maps else 0)
    padding = {
        "c1": {"send": pn["c1_send"] * 4, "recv": pn["c1_recv"] * 4},
        "c2": {"send": pn["c2_send"] * 4, "recv": pn["c2_recv"] * 4},
        "b1": {"send": n_b1_send * pad_slot, "recv": n_b1_recv * pad_self},
        "b2": {"send": pn["b2_send"] * 4, "recv": pn["b2_recv"] * 4},
        "s_in_saved": pn["c2_recv"] * 4,
        "padded_owner": hp.padded_owner,
        "pad_tiles_per_slot": hp.pad_tiles,
    }
    padding["B1"] = total(padding["c1"], padding["c2"])
    padding["B2"] = total(padding["b1"], padding["b2"])
    return {
        **coll,
        "b1_d": b1_d,
        "b1_m": b1_m,
        "b1_design": b1_design,
        "b1_replay": b1_replay,
        "padding": padding,
        "B1": fwd,
        "B2": bwd,
        "B3": b3,
        "crossing_fragments_in": ctx.n_in,
        "crossing_fragments_out": ctx.n_out,
        "N_in": ctx.N_in,
        "N_out": ctx.N_out,
        "state_bytes": s,
        "fwd_state_bytes": ctx.n_in * s,
        "fwd_summary_bytes": ctx.n_out * 2 * s,
        "bwd_adjoint_bytes": ctx.n_out * s,
        "H_qk": H_qk,
        "HV": HV,
        "C": C,
        "save_transfer": bool(save_transfer),
        "boundary_mode": boundary_mode,
    }


def cake_gdn_det_train_kcp_saved_layout(
    ctx: CakeGDNDetKcpContext,
    *,
    H_qk: int,
    HV: int,
    save_mode: str = "light",
    boundary_mode: str = "saved",
    save_transfer: bool = SAVE_TRANSFER_DEFAULT,
    with_conv: bool = True,
    C: int | None = None,
    P: int | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> dict[str, int]:
    """The saved set of one layer on this rank, term by term in bytes (design table 4.1; module docstring).
    ``P`` overrides the plan's segment count (the call's ``segments`` option; the range the call accepts) for the
    ``M_seg`` term; ``use_qk_l2norm_in_kernel``: the ``qk_rstd`` term (``8 T_r H_qk``, saved boundary only) and the
    ``qn`` / ``kn`` terms (``256 T_r H_qk`` each, ``save_mode="full"`` with the saved boundary only).
    ``with_conv`` (default ``True``) adds ``hist``, the layer convolution's saved history rows: the module's
    convolution keeps them, the entry points of this package do not, so ``with_conv=False`` is what a GDN call
    saves."""
    H_qk, HV = _check(H_qk, HV)
    l2n = _flag(use_qk_l2norm_in_kernel, "use_qk_l2norm_in_kernel")
    C = _channels(H_qk, HV, C)
    _mode(save_mode, SAVE_MODES, "save_mode")
    _mode(boundary_mode, BOUNDARY_MODES, "boundary_mode")
    hp = ctx.heads(HV)
    P = _segments(ctx, hp.P, P)
    s = state_bytes(HV)
    s_pad = hp.n_owners * hp.Hg * STATE_TILE_BYTES  # = s without padding
    saved = boundary_mode == "saved"
    terms = {
        "S_in": ctx.n_in * s_pad if saved else 0,
        "c1_recv": (ctx.N_out * 2 * hp.Hg_pad_by_rank[ctx.rank] * STATE_TILE_BYTES)
        if (saved and save_transfer)
        else 0,
        "M_seg": (P * ctx.n_frag * s)
        if (saved and (save_transfer or save_mode == "full") and P > 1)
        else 0,
        "hist": ctx.halo.n_hist_rows * C * 2 if with_conv else 0,
    }
    # the replay boundary keeps nothing: a "full" forward under boundary_mode="replay" saves no tape and its backward
    # is the "light" one (cake_kcp: no saved set, the replayed forward writes the tapes into the arena)
    full = save_mode == "full" and saved
    terms.update(  # the #1 outputs: saved in "full" only (the light backward re-runs #1)
        {
            "gamma": ctx.NC_pad * HV * CHUNK_F32 if full else 0,
            "gcum": ctx.R * HV * ROW_F32 if full else 0,
            "akk": ctx.R * HV * ROW_BF16_64 if full else 0,
        }
    )
    terms.update(
        {
            "h_tape": ctx.NC_pad * HV * TAPE_BF16_TILE if full else 0,
            "v_new": ctx.R * HV * ROW_BF16_128 if full else 0,
            "w": ctx.R * HV * ROW_BF16_128 if full else 0,
            "u": ctx.R * HV * ROW_BF16_128 if full else 0,
            "aqk": ctx.R * HV * ROW_BF16_64 if full else 0,
            # the host skips #1 entirely in "full": its FP32 row factors are retained too (+ 8 HV B per padded row;
            # design table 4.1 lists the bf16 tapes only)
            "rowscale": ctx.R * HV * ROW_F32 if full else 0,
            "qscale": ctx.R * HV * ROW_F32 if full else 0,
        }
    )
    # the in-kernel L2 norm's r of q and k (the light backward recomputes qn / kn from the raw rows and it) and, in
    # "full", the forward's normalized rows (the full backward binds them: cake_kcp._saves_qk)
    if l2n:  # (no term without the norm: the norm-off layout is unchanged)
        terms["qk_rstd"] = ctx.T_r * H_qk * QK_RSTD_BYTES if saved else 0
        terms["qn"] = ctx.T_r * H_qk * QK_ROW_BF16 if full else 0
        terms["kn"] = ctx.T_r * H_qk * QK_ROW_BF16 if full else 0
    return terms


def cake_gdn_det_train_kcp_saved_bytes(
    ctx: CakeGDNDetKcpContext,
    *,
    H_qk: int,
    HV: int,
    save_mode: str = "light",
    boundary_mode: str = "saved",
    save_transfer: bool = SAVE_TRANSFER_DEFAULT,
    with_conv: bool = True,
    C: int | None = None,
    P: int | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> int:
    """Saved-set bytes of one layer on this rank (the sum of :func:`cake_gdn_det_train_kcp_saved_layout`)."""
    return sum(
        cake_gdn_det_train_kcp_saved_layout(
            ctx,
            H_qk=H_qk,
            HV=HV,
            save_mode=save_mode,
            boundary_mode=boundary_mode,
            save_transfer=save_transfer,
            with_conv=with_conv,
            C=C,
            P=P,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        ).values()
    )


def _fwdh_out_shipped() -> bool:
    """Whether the registry ships the output walk ``kcp_fwdh_out`` in both forms (``cake_kcp.fused_o_shipped``; each
    record covers every architecture of :data:`cake_jit.ARCHES`): the shipped forward then fuses #5 into its walks."""
    try:
        names = cake_jit.records_of_program("kcp_fwdh_out")
    except ValueError:  # a registry that does not know the program
        return False
    forms = {
        int(cake_jit.MODULES[n].get("grid_rule", {}).get("ctas_per_head", 1))
        for n in names
    }
    return forms >= {1, 2}


def _prep_form_shipped() -> bool:
    """Whether the registry ships the forward form of #1_l2n (``cake_kcp.prep_form_shipped``: a ``fwd_prep_l2n``
    record selected by ``prep_form`` with ``backward_tapes`` False; it covers every architecture of the full
    record): a forward whose ``akk`` / ``gcum`` go to no backward then binds neither tape at #1."""
    try:
        names = cake_jit.records_of_program("fwd_prep_l2n")
    except ValueError:  # a registry that does not know the program
        return False
    return any(
        cake_jit.MODULES[n].get("select_rule", {}).get("kind") == "prep_form"
        and cake_jit.MODULES[n]["select_rule"].get("backward_tapes") is False
        for n in names
    )


def cake_gdn_det_train_kcp_workspace_layout(
    ctx: CakeGDNDetKcpContext,
    *,
    H_qk: int,
    HV: int,
    save_mode: str = "light",
    boundary_mode: str = "saved",
    save_transfer: bool = SAVE_TRANSFER_DEFAULT,
    backward: bool = True,
    with_conv: bool = True,
    C: int | None = None,
    P: int | None = None,
    group_reduce: str = "fused",
    reuse_recompute_tapes: bool | str | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    fused_walks: int = 1,
    graph: str = "off",
    save_for_backward: bool = True,
) -> dict[str, int]:
    """The caller-held workspace of one rank, term by term (``fwd.*`` / ``bwd.*`` in bytes), with the lifetime
    maxima ``forward`` / ``backward`` and their maximum ``workspace`` (module docstring).  Saved-set tensors are
    counted by :func:`cake_gdn_det_train_kcp_saved_layout`, never here.

    Covered calls: ``exchange="a2a"`` under the shipped ``schedule="wait_first"``, ``graph`` ``"off"`` or
    ``"replay"``, with or without a saved set, on a caller-held workspace created empty (``CakeGDNDetKcpWorkspace()``): its
    first forward (an arena that binds no tape yet) and the backward that follows on the arena that forward left.
    ``save_for_backward=False`` is a forward that keeps no saved set: with ``backward=False`` a forward-only call;
    with ``backward=True`` its backward is the replay a ``saved=None`` backward runs (the ``boundary_mode="replay"``
    backward).  A backward leaves its tapes bound, so every later backward on the same workspace (another layer of
    the step, or the next step) binds the ``graph="replay"`` figure of its options (measured for the replay form; the
    kept form's later backwards are not measured) -- at most that figure right after the fused forward of a
    ``save_for_backward=False`` call (light, saved boundary: that forward released ``h`` / ``v_new``, which the
    replayed forward binds again only after its c1 release; less on a rank whose fragment crosses out) -- and
    every later forward binds more than ``forward`` (the backward's tapes stay bound; under ``graph="replay"`` its
    b1 / b2 pools too) and at most that figure (equal to it under ``graph="replay"``).
    ``CakeGDNDetKcpWorkspace(ctx, H_qk, HV, device)`` sizes every pool up front at the plan's
    ``P`` with the fused walks, and a call leaves a held pool bound: the ring pools (``2 max(n_in, n_out)`` states),
    the c2 send pool, the ``c1_recv`` / ``stage_m`` pools a saving call replaces with fresh tensors, with its default
    ``backward=True`` the b1 / b2 pools and, for a call with a smaller ``segments`` override or ``fused_walks=0``,
    the plan's larger staging, carry and ``M_f^T`` pools, all from #1 on -- up to that much above these figures.
    Not covered: ``exchange="ring"`` (its ``ring_out`` / ``ring_in`` pools, ``max(n_in, n_out)`` states each, live
    through the forward and the backward's adjoint exchange, and its forward -- also the forward a replay backward
    re-runs -- keeps the c2 send pool and the c1 send pool, ``M_f^T`` tiles included, through its tape walk: up to
    ``fwd.c1_send + fwd.c1_send_mf + bwd.c2_send`` above ``fwd.phase.walk`` (``bwd.c2_send`` above
    ``fwd.phase.prep``) and up to ``fwd.c1_send + fwd.c1_send_mf`` above ``bwd.phase.exchange``); the
    ``schedule="overlap"`` test knob (its document-start tape walk runs under c1 and its document-end adjoint walk
    under b2: a split forward binds up to ``fwd.c1_send + fwd.c1_send_mf`` more, the backward up to ``bwd.b2_send``,
    the staging planes and, with the fused walks at ``P > 1``, ``bwd.b2_recv`` more); the kept form's recompute
    under a ``CakeGDNDetKcpRecomputeHandle`` (a handle forces ``save_transfer=False``; with the query given
    ``save_transfer=False`` too: the recompute exchanges nothing, so the c1 send pool, ``M_f^T`` tiles included,
    stays bound through its tape walk, up to ``fwd.c1_send + fwd.c1_send_mf`` above ``bwd.phase.recompute_forward``;
    the replay form's figure, ``reuse_recompute_tapes=False`` or ``"auto"``, bounds it).  These figures plan a
    workspace; no call checks a workspace against them (a caller-held ``CakeGDNDetKcpWorkspace`` grows any pool or tape a call
    needs).

    ``P`` overrides the plan's segment count as the call's ``segments`` option does (the range the call accepts):
    the staging planes (``n_frag P`` states each), the carry plane (``n_frag (P - 1)`` states, plus
    ``max(n_in, n_out)`` slots with the fused walks) and the ``M_f^T`` tiles the c1 send pool holds at ``P > 1``
    (``c1_send_mf``, ``n_out`` states; ``c1_send`` stays the collective's bytes) follow it; the split terms do not
    depend on ``P``.  ``fused_walks`` is the call's test knob (1, the shipped form).
    The b1 pools are counted at the larger of their two split forms, as the host sizes them.  A backward on a
    caller-held workspace sizes the forward exchange pools too (``c1_send`` / ``c1_send_mf``, ``c2_send`` and
    ``c1_recv`` unless the saved set holds it; a call without ``workspace=`` skips them after a saved forward).
    ``with_conv`` adds the layer's causal-convolution terms (``halo_send``, ``dpre_head`` / ``dpre_tail``: the halo
    rows of the module's convolution, which the entry points of this package do not allocate): ``with_conv=False``
    is what a call of this package binds.  ``boundary_mode="replay"`` keeps no tape and hands none over, so
    ``save_mode="full"`` and ``reuse_recompute_tapes=True`` there give the light replay form's figures; the backward
    of ``save_for_backward=False`` is that replay too (after the fused forward of a saved boundary its replayed
    forward binds no ``h`` / ``v_new`` before the c1 release).

    The terms follow the arena's lifetimes; under ``graph="off"`` the host drops a buffer at its last reader, so a
    lifetime's figure is the most the workspace binds at one moment of it, not the sum of its terms:

    * ``forward``: the larger of ``fwd.phase.prep`` (#1: the pools, the eight tapes #1 writes and, with the norm,
      ``qn`` / ``kn`` and -- when the forward keeps no saved set -- the arena ``qk_rstd``, dropped after #1) and
      ``fwd.phase.walk`` (the tape walk: ``akk`` / ``gcum`` are gone, the split form binds ``h`` / ``v_new``, and
      when a fragment of the group crosses a rank the c1 send pool, ``M_f^T`` tiles included, is released after
      wait(c1)).  The shipped light / saved forward is fused (the output walk ``kcp_fwdh_out`` where the registry
      ships it: no ``h`` / ``v_new`` tape), so its peak is #1; ``forward_split`` is the split form's (the
      ``fused_o=0`` test knob).  The eager forward binds its c2 send plane as a fresh tensor (``fwd.c2_send`` 0).
      ``save_mode="full"`` with a saved set writes its tapes -- with the norm also ``qn`` / ``kn`` -- into fresh
      saved tensors (no tape or row term); without one they are arena tapes of the split form.  With the norm every
      other forward runs the forward form of #1_l2n where the registry ships it: #1 writes six tapes (no ``akk`` /
      ``gcum`` term in either moment); the kept form's recompute forward (``bwd.phase.recompute_forward``) runs the
      full #1.
    * ``backward``: the largest of its phases -- the adjoint exchange (the backward pools, without the c1 send pool
      that a replayed forward has already released, plus the forward tapes of the form and the tapes #6 wrote), the
      adjoint tape walk (``carry``, ``b2_recv`` unless the fused walks at ``P > 1`` dropped it, and ``dh``) and the
      end state (the tapes #8 / #9 read; of the pools only ``eye``, the ticket slots and, with ``with_conv``, the
      ``dpre`` rows stay); in a backward without a saved set (``boundary_mode="replay"``,
      ``save_for_backward=False``) also the replayed forward before its c1 release (``bwd.phase.replay_forward``:
      every backward pool, the eight tapes #1 re-writes and, after a split forward, the ``h`` / ``v_new`` it left).
      The forward tapes of the form: the replay form (``reuse_recompute_tapes=False``) re-creates the ten light tapes
      in the arena; the kept-tape form (``True``; the ``None`` default resolves to the package default; light
      ``save_mode`` with a saved set only) binds none of them during the backward (the ten tapes are the recompute
      forward's hand-off tensors, outside the workspace: ``bwd.kept_tapes``; at ``P > 1`` without ``save_transfer``
      the recompute forward's ``M_seg^T`` staging plane is handed over too, so the backward's workspace sizes one
      staging plane and ``bwd.kept_tapes`` holds the other until #8) and re-adopts the tapes an eager forward
      leaves bound (``aqk`` / ``qscale``, with ``h`` / ``v_new`` when the forward is split) after #8, so its end
      state is the arena a forward leaves plus the backward tapes; that form's recompute forward keeps every tape in
      the arena until the hand-off (``bwd.phase.recompute_forward``) and counts as a phase too.
      ``"auto"`` reports the replay form (the bound the workspace may reach when a later operation of the
      checkpointed region saves a tensor; a workspace bound: the kept form's hand-off tensors are outside it).  ``save_mode="full"`` with a saved set binds the saved tapes (with
      the norm also the saved ``qn`` / ``kn``), none of which is workspace.  ``backward_transient`` = the backward's phase maximum with the hand-off tensors added
      (the memory a backward needs besides the saved set): the same figure in both forms, except that the kept form
      holds the handed-over ``M_seg^T`` plane to #8 (the replay form drops its staging planes before the adjoint
      tape walk) and, with the norm, needs no arena ``qk_rstd`` (``8 T_r H_qk`` less).
    * ``graph="replay"``: neither call releases a buffer (a captured graph bakes the arena).  ``forward`` is every
      forward pool (``c2_send`` included) and every tape the forward writes (none with ``save_mode="full"`` and a
      saved set) with, under the norm, ``qn`` / ``kn`` and the arena ``qk_rstd`` of a forward without a saved set;
      ``backward`` (``bwd.phase.graph``) is every backward pool, the ten light tapes (none with ``save_mode="full"``
      and a saved set), every backward tape and, under the norm, ``qn`` / ``kn`` and the arena ``qk_rstd`` where
      the backward re-runs #1.
    * ``use_qk_l2norm_in_kernel``: ``qn`` / ``kn`` in both forward moments and every backward phase of the replay
      form (#1_l2n writes them there) and in the kept form's recompute forward, which hands them over with the ten
      tapes: in the kept-tape backward they are hand-off tensors (``bwd.kept_tapes``, ``bwd.qn`` / ``bwd.kn`` 0);
      ``save_mode="full"`` with a saved set keeps them in the saved set (``fwd.qn`` / ``fwd.kn`` / ``bwd.qn`` /
      ``bwd.kn`` 0: :func:`cake_gdn_det_train_kcp_saved_layout`); ``fwd.qk_rstd`` (the arena ``r``) at #1 of a forward without a saved set;
      ``bwd.qk_rstd`` where the backward re-runs #1 (the light replay of a saved set, a backward without one).
    """
    H_qk, HV = _check(H_qk, HV)
    l2n = _flag(use_qk_l2norm_in_kernel, "use_qk_l2norm_in_kernel")
    with_set = _flag(save_for_backward, "save_for_backward")
    C = _channels(H_qk, HV, C)
    _mode(save_mode, SAVE_MODES, "save_mode")
    _mode(boundary_mode, BOUNDARY_MODES, "boundary_mode")
    _mode(group_reduce, GROUP_REDUCE, "group_reduce")
    _mode(graph, GRAPH_MODES, "graph")
    if fused_walks not in (0, 1):
        raise ValueError(f"fused_walks must be 0 or 1, got {fused_walks!r}")
    eager = graph == "off"
    kept_form = _kept_form(reuse_recompute_tapes, graph)
    # the plan's split tables (they do not depend on P) and its segment count
    hp = ctx.heads(HV)
    P = _segments(ctx, hp.P, P)
    n, halo = hp.numel, ctx.halo
    R, NC_pad, T_r = ctx.R, ctx.NC_pad, ctx.T_r
    s = state_bytes(HV)
    saved = boundary_mode == "saved"
    # the forward keeps a saved set (cake_kcp._fwd_body: save); without one the backward is the replay a saved=None
    # backward runs (cake_kcp._backward), whatever the boundary mode
    keep = saved and with_set
    # the backward reads the saved tapes; a call without a saved set keeps none, so "full" there is the light replay
    full = save_mode == "full" and keep
    # the forward writes its tapes into fresh saved tensors (not arena tapes)
    fresh_tapes = full
    # the kept-tape backward exists for the light form with a saved set only (the runtime hands off only from a
    # forward that saves: cake_kcp._fwd_entry / _fwd_body)
    kept = kept_form and keep and not full
    # the shipped forward folds #5 into its tape walks (cake_kcp._fuse_o_enabled with test knob fused_o = 1)
    fused = save_mode == "light" and saved and _fwdh_out_shipped()
    # a2a collectives run when a fragment of the group crosses a rank (cake_kcp._forward: tb.N_out)
    exchanges = ctx.N_out > 0
    # staging planes at P > 1: stage_b always; stage_m only when the call does not bind a saved M_seg^T for it (the
    # forward saves it with save_transfer or save_mode="full" when it keeps a saved set; the backward reads it from
    # that set: cake_kcp SAVE_TRANSFER_FRESH_POOLS / _FULL_FRESH_POOLS / _saved_fresh_pools).  One plane is NSEG =
    # n_frag P states at the segment count of the call (the override P too: the call sizes its pools from the
    # overridden plan; equal to the plan's NSEG when P is None)
    plane = (ctx.n_frag * P * HV * STATE_TILE_BYTES) if P > 1 else 0
    fwd_staging = plane + (0 if (keep and (save_transfer or full)) else plane)
    # the kept form's recompute forward hands its stage_m plane over without save_transfer (cake_kcp._fwd_body,
    # _saved_fresh_pools): that backward sizes stage_b only, the plane is a hand-off tensor until #8 (bwd.kept_tapes)
    bwd_staging = plane + (0 if (keep and (save_transfer or full or kept)) else plane)
    kept_stage_m = plane if (kept and not save_transfer) else 0
    # P > 1: the carry plane holds n_frag (P - 1) segment states and, with the fused walks (the shipped form; the
    # fused_walks test knob of the call), max(n_in, n_out) slots for the S_in / dS_out copies (one index range shared
    # by the crossing-in and crossing-out slots); the c1 send pool also holds the M_f^T tiles of the crossing-out
    # fragments (cake_kcp _Tables: mf_tiles, carry_tiles)
    carry = (
        (ctx.n_frag * (P - 1) + (max(ctx.n_in, ctx.n_out) if fused_walks else 0)) * s
        if P > 1
        else 0
    )
    c1_send_mf = ctx.n_out * s if P > 1 else 0
    # the forward tapes (#1 outputs + the #2 tape walk); in "light" they are workspace in the forward and, replayed,
    # in the backward; in "full" with a saved set they are saved-set bytes (cake_gdn_det_train_kcp_saved_layout), not workspace
    light_tapes = {
        "w": R * HV * ROW_BF16_128,
        "u": R * HV * ROW_BF16_128,
        "aqk": R * HV * ROW_BF16_64,
        "akk": R * HV * ROW_BF16_64,
        "h_tape": NC_pad * HV * TAPE_BF16_TILE,
        "v_new": R * HV * ROW_BF16_128,
        "gamma": NC_pad * HV * CHUNK_F32,
        "gcum": R * HV * ROW_F32,
        "rowscale": R * HV * ROW_F32,
        "qscale": R * HV * ROW_F32,
    }
    light_sum = sum(light_tapes.values())
    fwd_pools = {
        "c1_send": n["c1_send"] * 4,
        "c1_send_mf": c1_send_mf,
        "c1_recv": 0 if (keep and save_transfer) else n["c1_recv"] * 4,
        # the eager a2a forward binds a fresh c2 send plane at its first writer (the forward merge) and drops it after
        # wait(c2); a captured forward keeps the pool (cake_kcp._prepare)
        "c2_send": 0 if eager else n["c2_send"] * 4,
        "staging": fwd_staging,
        "carry": carry,
        "halo_send": sum(halo.fwd_send_rows) * C * 2 if with_conv else 0,
        "eye": EYE_BYTES,  # the identity tile pool (one FP32 state), every workspace of the context
        "tickets": TICKET_BYTES,  # the persistent programs' ticket slots (every site, allocated together)
    }
    fwd_pool_sum = sum(fwd_pools.values())
    # the in-kernel L2 norm's normalized rows: live through the tape walk (kn goes before the output, qn after #5);
    # "full" with a saved set writes them into fresh saved tensors (cake_kcp._saves_qk: no arena row); the arena r:
    # written by #1_l2n when the forward keeps no saved set (else the saved set's fresh r), dropped after #1
    # (cake_kcp._forward; kept under graph="replay")
    qk_row = 0 if full else T_r * H_qk * QK_ROW_BF16
    qk_rows = {"qn": qk_row, "kn": qk_row} if l2n else {}
    qk_sum = sum(qk_rows.values())
    fwd_rstd = T_r * H_qk * QK_RSTD_BYTES if (l2n and not keep) else 0
    # the c1 send pool (M_f^T tiles included) is released after wait(c1), before any tape walk of the wait_first
    # schedule (eager a2a, a crossing fragment in the group)
    c1_gone = (fwd_pools["c1_send"] + c1_send_mf) if (eager and exchanges) else 0
    # the tapes #1 writes (an arena that binds no tape yet: h / v_new come with the first tape walk, after the offset
    # walks bound u in their place); #1's backward-only outputs leave an eager arena right after #1
    prep_tapes = (
        {}
        if fresh_tapes
        else {k: v for k, v in light_tapes.items() if k not in FUSED_FREE_TAPES}
    )
    # the forward form of #1_l2n (cake_kcp._prep_backward_tapes): with the norm, a forward whose akk / gcum go to no
    # backward -- here every first forward but a "full" save with a saved set (fresh saved tapes, no arena term) --
    # binds neither tape; the kept form's recompute forward hands them over and runs the full #1
    prep_fwd_form = l2n and T_r > 0 and not fresh_tapes and _prep_form_shipped()

    def forward_moments(
        fuse: bool, drop_prep: bool, fwd_form: bool = False
    ) -> tuple[dict[str, int], dict[str, int]]:
        prep = {
            k: v
            for k, v in prep_tapes.items()
            if not (fwd_form and k in FWD_TAPES_DEAD_AFTER_PREP)
        }
        walk_tapes = {
            k: v
            for k, v in prep.items()
            if not (drop_prep and k in FWD_TAPES_DEAD_AFTER_PREP)
        }
        if not (fuse or fresh_tapes):
            walk_tapes.update({k: light_tapes[k] for k in FUSED_FREE_TAPES})
        moments = {
            "prep": fwd_pool_sum + sum(prep.values()) + qk_sum + fwd_rstd,
            "walk": fwd_pool_sum
            - c1_gone
            + sum(walk_tapes.values())
            + qk_sum
            + (0 if eager else fwd_rstd),
        }
        return {**prep, **walk_tapes}, moments

    fwd_tapes, fwd_moments = forward_moments(fused, eager, prep_fwd_form)
    _, split_moments = forward_moments(False, eager, prep_fwd_form)
    fwd = {
        **fwd_pools,
        **fwd_tapes,
        **qk_rows,
        **({"qk_rstd": fwd_rstd} if l2n else {}),
    }
    # the b1 pools hold either split form (d only, or d + the re-sent maps; their map half pairs with the crossing-in
    # slots), so the host sizes the larger of the two (cake_kcp _Tables.sizes); the design's per-rank b1 QUERY is in
    # cake_gdn_det_train_kcp_comm_bytes -- the workspace counts what the host allocates.  A caller-held workspace's backward sizes
    # the forward exchange pools as well, in either boundary mode (the replay writes them again; only the per-call
    # workspace of a saved backward skips them, cake_kcp SAVED_BWD_DEAD_POOLS); S_in is a fresh tensor, never a pool
    bwd_pools = {
        "b1_send": max(n["b1_send_d"], n["b1_send_dm"]) * 4,
        "b1_recv": max(n["b1_recv_d"], n["b1_recv_dm"]) * 4,
        "b2_send": n["b2_send"] * 4,
        "b2_recv": n["b2_recv"] * 4,
        "staging": bwd_staging,
        "carry": carry,
        "dpre_head": halo.n_head_rows * C * 4 if with_conv else 0,
        "dpre_tail": halo.n_tail_rows * C * 4 if with_conv else 0,
        "eye": EYE_BYTES,
        "tickets": TICKET_BYTES,
        "c1_send": n["c1_send"] * 4,
        "c1_send_mf": c1_send_mf,
        "c1_recv": 0 if (keep and save_transfer) else n["c1_recv"] * 4,
        "c2_send": n["c2_send"] * 4,
    }
    # the forward tapes the backward's arena holds: the replay re-creates all ten; the kept form none (the hand-off
    # tensors) until #8, then the four an eager forward leaves bound return to the arena; "full" binds saved tensors
    # (without a saved set the backward replays the ten into the arena)
    bwd_fwd_tapes: dict[str, int]
    bwd_fwd_tapes_end: dict[str, int]
    if full:
        bwd_fwd_tapes, bwd_fwd_tapes_end = {}, {}
    elif kept:
        bwd_fwd_tapes = {}
        bwd_fwd_tapes_end = {
            k: light_tapes[k]
            for k in (
                FWD_TAPES_ARENA_AFTER_FUSED if fused else FWD_TAPES_ARENA_AFTER_FWD
            )
        }
    else:
        bwd_fwd_tapes = bwd_fwd_tapes_end = light_tapes
    bwd_tapes = {
        "do_g": R * HV * ROW_BF16_128,  # #6
        "dv_intra": R * HV * ROW_BF16_128,  # #6
        "dh_tape": NC_pad * HV * TAPE_BF16_TILE,  # the adjoint tape walk
        "dv_new": 0,  # stored in place of dv_intra (one buffer: cake_kcp.TAPE_ALIASES)
        "dg_part": R * HV * ROW_F32,  # #8
        "gamma_dot": NC_pad * HV * CHUNK_F32,  # #8
        "group_partials": (2 * T_r * HV * HEAD_DIM * 4)
        if group_reduce == "host"
        else 0,  # #8, the host form of the grouped reduction
    }
    # the norm in the backward: qn / kn from #1_l2n (light replay, replay mode) until #8 -- the kept form binds the
    # recompute forward's hand-off rows (outside the workspace, bwd.kept_tapes), the full form the saved rows (in the
    # saved set: qk_rows are 0 there) -- and the arena r where #1 is re-run (the full / kept forms read the saved r)
    replays_prep = not (full or kept)
    kept_qk = qk_sum if kept else 0
    bwd_l2n = (
        {
            **{k: 0 if kept else v for k, v in qk_rows.items()},
            "qk_rstd": T_r * H_qk * QK_RSTD_BYTES if replays_prep else 0,
        }
        if l2n
        else {}
    )
    l2n_sum = sum(bwd_l2n.values())
    bwd = {**bwd_pools, **bwd_fwd_tapes_end, **bwd_tapes, **bwd_l2n}
    pools_all = sum(bwd_pools.values())
    fwd_arena = sum(bwd_fwd_tapes.values())
    if eager:
        # the backward's phases (the host drops the exchange pools after the adjoint merge, the staging planes -- and,
        # with the fused walks at P > 1, b2_recv -- before the adjoint tape walk, carry / b2_recv before #8; the kept
        # entry is re-adopted after #8); a replayed forward has released the c1 send pool before #6
        replay_c1 = (
            (bwd_pools["c1_send"] + c1_send_mf) if (not keep and exchanges) else 0
        )
        pools_walk = (
            bwd_pools["carry"]
            + (0 if (fused_walks and P > 1) else bwd_pools["b2_recv"])
            + bwd_pools["eye"]
            + bwd_pools["tickets"]
            + bwd_pools["dpre_head"]
            + bwd_pools["dpre_tail"]
        )
        pools_end = (
            bwd_pools["eye"]
            + bwd_pools["tickets"]
            + bwd_pools["dpre_head"]
            + bwd_pools["dpre_tail"]
        )
        phase_exchange = (
            pools_all
            - replay_c1
            + fwd_arena
            + bwd_tapes["do_g"]
            + bwd_tapes["dv_intra"]
            + l2n_sum
        )
        phase_walk = (
            pools_walk
            + fwd_arena
            + bwd_tapes["do_g"]
            + bwd_tapes["dv_intra"]
            + bwd_tapes["dh_tape"]
            + l2n_sum
        )
        phase_end = (
            pools_end
            + sum(bwd_fwd_tapes_end.values())
            + sum(bwd_tapes.values())
            + l2n_sum
        )
        phases = {"exchange": phase_exchange, "walk": phase_walk, "end": phase_end}
        if not keep:
            # the replayed forward before wait(c1): every pool the backward sized, the eight tapes #1 re-wrote, the
            # h / v_new a split forward left (a fused forward leaves none: its tape walk binds them after the c1
            # release) and the norm's rows
            phases["replay_forward"] = (
                pools_all
                + light_sum
                - (sum(light_tapes[k] for k in FUSED_FREE_TAPES) if fused else 0)
                + l2n_sum
            )
        if kept:
            # the recompute forward of the kept form drops no tape before handing all ten over (its qn / kn are live
            # through its tape walk like any forward's, and handed over too): its #1 or its tape walk
            _, rec = forward_moments(False, False)
            phases["recompute_forward"] = max(rec["prep"], rec["walk"])
    else:
        # graph="replay": nothing is released in either call
        phases = {"graph": pools_all + fwd_arena + sum(bwd_tapes.values()) + l2n_sum}
    out = {f"fwd.{k}": v for k, v in fwd.items()}
    out.update({f"fwd.phase.{k}": v for k, v in fwd_moments.items()})
    if backward:
        out.update({f"bwd.{k}": v for k, v in bwd.items()})
        # the hand-off tensors: the ten forward tapes, with the norm also qn / kn, at P > 1 without save_transfer the
        # recompute forward's M_seg^T plane
        out["bwd.kept_tapes"] = light_sum + kept_qk + kept_stage_m if kept else 0
        out.update({f"bwd.phase.{k}": v for k, v in phases.items()})
    out["forward"] = max(fwd_moments.values())
    out["forward_split"] = max(split_moments.values())
    out["backward"] = max(phases.values()) if backward else 0
    # the hand-off tensors added to the phases they live through (the kept form's exchange, tape walk and #8, before
    # the re-adoption): the replay form's figure, plus the M_seg^T plane it drops before its tape walk, less its
    # arena qk_rstd
    out["backward_transient"] = (
        (
            max(
                phases["exchange"],
                phases["walk"],
                phases["end"] - sum(bwd_fwd_tapes_end.values()),
            )
            + out["bwd.kept_tapes"]
        )
        if (backward and kept)
        else out["backward"]
    )
    out["workspace"] = max(out["forward"], out["backward"])
    return out


def cake_gdn_det_train_kcp_workspace_bytes(
    ctx: CakeGDNDetKcpContext,
    *,
    H_qk: int,
    HV: int,
    save_mode: str = "light",
    boundary_mode: str = "saved",
    save_transfer: bool = SAVE_TRANSFER_DEFAULT,
    backward: bool = True,
    with_conv: bool = True,
    C: int | None = None,
    P: int | None = None,
    group_reduce: str = "fused",
    reuse_recompute_tapes: bool | str | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    fused_walks: int = 1,
    graph: str = "off",
    save_for_backward: bool = True,
) -> int:
    """Caller-held workspace bytes of one rank (the ``workspace`` entry of :func:`cake_gdn_det_train_kcp_workspace_layout`): the
    most the workspace binds in the forward or the backward of these options (the calls that query covers)."""
    return cake_gdn_det_train_kcp_workspace_layout(
        ctx,
        H_qk=H_qk,
        HV=HV,
        save_mode=save_mode,
        boundary_mode=boundary_mode,
        save_transfer=save_transfer,
        backward=backward,
        with_conv=with_conv,
        C=C,
        P=P,
        group_reduce=group_reduce,
        reuse_recompute_tapes=reuse_recompute_tapes,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        fused_walks=fused_walks,
        graph=graph,
        save_for_backward=save_for_backward,
    )["workspace"]


__all__ = [
    "BOUNDARY_MODES",
    "GRAPH_MODES",
    "GROUP_REDUCE",
    "SAVE_MODES",
    "TICKET_BYTES",
    "TICKET_SITES",
    "TICKET_SLOT_BYTES",
    "EYE_BYTES",
    "QK_ROW_BF16",
    "QK_RSTD_BYTES",
    "FWD_TAPES_ARENA_AFTER_FUSED",
    "FWD_TAPES_ARENA_AFTER_FWD",
    "FWD_TAPES_DEAD_AFTER_PREP",
    "TAPE_NAME_OF",
    "cake_gdn_det_train_kcp_comm_bytes",
    "cake_gdn_det_train_kcp_saved_bytes",
    "cake_gdn_det_train_kcp_saved_layout",
    "cake_gdn_det_train_kcp_workspace_bytes",
    "cake_gdn_det_train_kcp_workspace_layout",
    "state_bytes",
]
