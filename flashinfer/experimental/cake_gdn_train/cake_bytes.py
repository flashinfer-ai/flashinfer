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
``Hg = ceil(HV / W)`` the heads per owner.

* ``gdn_kcp_comm_bytes(ctx, *, H_qk, HV, C)``: bytes this rank moves per
  collective and per boundary.  Per crossing-out fragment c1 sends ``2 s``
  (``[B_f | M_f^T]`` pair tiles) and b2 receives ``s`` (``dS_out``); per
  crossing-in fragment c2 receives ``s`` (``S_in``) and b1 sends ``s`` (``d_f``);
  per crossing-out fragment b1 also sends ``s`` (the re-sent ``M_f^T``) when the
  maps are not kept on the owner (``boundary_mode="saved"`` without
  ``save_transfer``), i.e. ``b1_send = s n_in + s n_out [not save_transfer]``
  (the per-rank lower-bound form; reported as ``b1_design``).  Owner side: c1 receives ``2 s_owner`` per crossing-out
  fragment of the group; b1 receives ``s_owner`` per crossing-in fragment (the
  ``d`` half, ``b1_d``) plus ``s_owner`` per crossing-out fragment when the maps
  are re-sent (the ``m`` half, ``b1_m``); c2 / b2 send ``s_owner`` per
  crossing-in / crossing-out fragment of the group, with
  ``s_owner = Hg_r * 128 * 128 * 4``.  ``b1`` reports the bytes of the plan's
  b1 split tables (``hp.b1_splits``: what the host allocates and the collective
  moves): their map half is paired with the crossing-IN slots
  (``2 s`` per crossing-in fragment).  ``b1_design`` is the design form above
  (a per-rank lower bound): equal to ``b1`` in the d-only form, different per
  rank when the maps are re-sent on a rank with ``n_in != n_out`` (the sums over
  the ranks agree).  Halo: ``rows * C * 2`` forward (BF16 ``x``
  history rows), ``rows * C * 4`` backward (FP32 ``dpre`` rows).  ``B1`` =
  forward (c1 + c2 + halo_fwd), ``B2`` = backward from the saved context
  (b1 + b2 + halo_bwd), ``B3`` = the replayed forward + the backward with b1 in
  its offsets-only form.  ``H_qk`` enters the exchange terms only through the
  default channel count ``C = (2 H_qk + HV) * 128`` of the packed QKV
  convolution (10,240 at (16, 48)); the state terms depend on ``HV`` alone.
  The padded owner (``HV % Hg != 0``) carries ``pad_tiles`` never-read tiles
  per slot, reported under ``padding`` and included in the totals (the buffers
  the ranks actually move).
* ``gdn_kcp_saved_bytes(ctx, *, H_qk, HV, save_mode, boundary_mode, ...)``: the
  saved set of one layer on this rank (design table 4.1): ``S_in`` (``n_in s``;
  the padded receive buffer), the owner's c1 receive tensor with
  ``save_transfer`` (``2 N_out s_owner``), ``M_seg`` at ``P > 1``
  (``(P - 1) n_frag s``) and the conv history rows (``n_hist_rows * C * 2``);
  ``save_mode="light"`` keeps no tape (its backward re-runs #1 and the #2 tape
  walk); ``save_mode="full"`` adds every forward tape: the ``h`` tape
  ``[NC_pad, HV, 128, 128]``, ``v_new`` / ``w`` / ``u`` ``[R, HV, 128]``, ``Aqk`` and
  ``Akk`` ``[R, HV, 64]`` (BF16), ``gamma [NC_pad, HV]`` and the FP32 row factors
  ``gcum`` / ``rowscale`` / ``qscale`` ``[R, HV]`` (the host skips #1 and the walk
  entirely).  ``M_seg`` is retained with
  ``save_transfer`` only (the ``save_transfer=False`` form re-runs the transfer
  walks, KDA semantics).  ``boundary_mode="replay"`` keeps no ``S_in`` /
  c1 tensor / ``M_seg`` (the backward re-runs the offset and transfer walks, the
  exchange and the merge).  Inputs are references (0 extra bytes); the int32
  tables live on the context (not per layer).
* ``gdn_kcp_workspace_bytes(ctx, *, H_qk, HV, ...)``: the caller-held
  workspace of one rank: the exchange buffers of the live boundary (send
  buffers, the receive buffers that are not retained, the ``P > 1`` staging and
  carry planes, the halo send rows), plus the local transients the programs
  write and the backward consumes (the tapes a mode does not save, the
  ``rowscale`` / ``qscale`` rows, the adjoint tapes ``dh`` / ``do_g`` /
  ``dv_intra`` (``dv_new`` is stored in place of ``dv_intra``: one buffer,
  counted once), the ``dg`` partials and, with the host form of the
  grouped ``dq`` / ``dk`` reduction, two FP32 ``[T_r, HV, 128]`` partial buffers)
  and the persistent programs' dynamic-ticket slots (one 128-byte u32 word per
  launch site, zeroed once with the workspace, never read by the host).
  ``forward`` / ``backward`` are the two lifetimes, the workspace their maximum.
  The local terms are the design's tape sizes; the kernels' own scratch is
  added when the programs exist (every term is listed by
  :func:`gdn_kcp_workspace_layout` so the host can account for it).

All sizes are exact integers of the shapes the host allocates (``R = 64
NC_pad`` padded token rows, ``NC_pad`` the even chunk bound), not estimates.
"""

from __future__ import annotations

from .cake_kcp_plan import (
    HEAD_DIM,
    SAVE_TRANSFER_DEFAULT,
    STATE_TILE_BYTES,
    KcpContext,
)

SAVE_MODES = ("light", "full")
BOUNDARY_MODES = ("saved", "replay")
GROUP_REDUCE = ("fused", "host")
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


def gdn_kcp_comm_bytes(
    ctx: KcpContext,
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
    b1_m = {  # M_f^T re-sent by every crossing-out fragment unless the owner kept the c1 pairs
        "send": ctx.n_out * s_pad if with_maps else 0,
        "recv": ctx.N_out * s_owner if with_maps else 0,
    }
    b1_buf_send, b1_buf_recv = hp.b1_splits(
        with_maps
    )  # the plan's split tables (allocated buffers)
    b1_design = {  # per-rank lower bound of the b1 traffic (d halves + re-sent maps)
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
    n_b1_send = ctx.n_in + (ctx.n_out if with_maps else 0)
    n_b1_recv = ctx.N_in + (ctx.N_out if with_maps else 0)
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


def gdn_kcp_saved_layout(
    ctx: KcpContext,
    *,
    H_qk: int,
    HV: int,
    save_mode: str = "light",
    boundary_mode: str = "saved",
    save_transfer: bool = SAVE_TRANSFER_DEFAULT,
    with_conv: bool = True,
    C: int | None = None,
    P: int | None = None,
) -> dict[str, int]:
    """The saved set of one layer on this rank, term by term in bytes (design table 4.1; module docstring).
    ``P`` overrides the plan's segment count (the ``walk_seg`` override knob) for the ``M_seg`` term."""
    H_qk, HV = _check(H_qk, HV)
    C = _channels(H_qk, HV, C)
    _mode(save_mode, SAVE_MODES, "save_mode")
    _mode(boundary_mode, BOUNDARY_MODES, "boundary_mode")
    hp = ctx.heads(HV)
    P = hp.P if P is None else int(P)
    if P < 1:
        raise ValueError(f"P={P} must be >= 1")
    s = state_bytes(HV)
    s_pad = hp.n_owners * hp.Hg * STATE_TILE_BYTES  # = s without padding
    saved = boundary_mode == "saved"
    terms = {
        "S_in": ctx.n_in * s_pad if saved else 0,
        "c1_recv": (ctx.N_out * 2 * hp.Hg_pad_by_rank[ctx.rank] * STATE_TILE_BYTES)
        if (saved and save_transfer)
        else 0,
        "M_seg": ((P - 1) * ctx.n_frag * s)
        if (saved and save_transfer and P > 1)
        else 0,
        "hist": ctx.halo.n_hist_rows * C * 2 if with_conv else 0,
    }
    full = save_mode == "full"
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
    return terms


def gdn_kcp_saved_bytes(
    ctx: KcpContext,
    *,
    H_qk: int,
    HV: int,
    save_mode: str = "light",
    boundary_mode: str = "saved",
    save_transfer: bool = SAVE_TRANSFER_DEFAULT,
    with_conv: bool = True,
    C: int | None = None,
    P: int | None = None,
) -> int:
    """Saved-set bytes of one layer on this rank (the sum of :func:`gdn_kcp_saved_layout`)."""
    return sum(
        gdn_kcp_saved_layout(
            ctx,
            H_qk=H_qk,
            HV=HV,
            save_mode=save_mode,
            boundary_mode=boundary_mode,
            save_transfer=save_transfer,
            with_conv=with_conv,
            C=C,
            P=P,
        ).values()
    )


def gdn_kcp_workspace_layout(
    ctx: KcpContext,
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
) -> dict[str, int]:
    """The caller-held workspace of one rank, term by term (``fwd.*`` / ``bwd.*`` in bytes), with the lifetime sums
    ``forward`` / ``backward`` and their maximum ``workspace`` (module docstring).  Saved-set tensors are counted by
    :func:`gdn_kcp_saved_layout`, never here."""
    H_qk, HV = _check(H_qk, HV)
    C = _channels(H_qk, HV, C)
    _mode(save_mode, SAVE_MODES, "save_mode")
    _mode(boundary_mode, BOUNDARY_MODES, "boundary_mode")
    _mode(group_reduce, GROUP_REDUCE, "group_reduce")
    hp = ctx.heads(HV)
    P = hp.P if P is None else int(P)
    if P < 1:
        raise ValueError(f"P={P} must be >= 1")
    n, halo = hp.numel, ctx.halo
    R, NC_pad, T_r = ctx.R, ctx.NC_pad, ctx.T_r
    s = state_bytes(HV)
    saved = boundary_mode == "saved"
    full = save_mode == "full"
    # staging planes at P > 1: stage_b always; stage_m only when the call does not bind the saved M_seg^T for it
    # (save_transfer under boundary_mode="saved" saves M_seg: the pool of that name is not sized, cake_kcp
    # SAVE_TRANSFER_FRESH_POOLS); the ring pools are sized only under exchange="ring" and are not workspace here
    plane = (hp.NSEG * HV * STATE_TILE_BYTES) if P > 1 else 0
    staging = plane + (0 if (boundary_mode == "saved" and save_transfer) else plane)
    carry = (ctx.n_frag * (P - 1) * s) if P > 1 else 0
    # tapes the forward writes (#1 outputs + the #2 tape walk); in "light" they are workspace in the forward and again
    # in the backward (the replay re-runs #1 and the walk: every one of them is allocated anew), in "full" they are
    # saved-set bytes (counted by gdn_kcp_saved_layout), not workspace.  Under a non-reentrant checkpoint with
    # reuse_recompute_tapes the backward binds the recompute forward's tapes instead of replaying: the same bytes,
    # live from that forward's end to #8 instead of from the replay to #8 (the backward's count below is unchanged)
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
    fwd = {
        "c1_send": n["c1_send"] * 4,
        "c1_recv": 0 if (saved and save_transfer) else n["c1_recv"] * 4,
        "c2_send": n["c2_send"] * 4,
        "staging": staging,
        "carry": carry,
        "halo_send": sum(halo.fwd_send_rows) * C * 2 if with_conv else 0,
        "tickets": TICKET_BYTES,  # the persistent programs' ticket slots (every site, allocated together)
        **({} if full else light_tapes),
    }
    # the b1 buffers are the plan's split tables (their map half pairs with the crossing-in slots; the design's
    # per-rank b1 QUERY is in gdn_kcp_comm_bytes -- the workspace counts what the host allocates)
    b1_send, b1_recv = hp.b1_splits(saved and not save_transfer)
    bwd = {
        "b1_send": sum(b1_send) * 4,
        "b1_recv": sum(b1_recv) * 4,
        "b2_send": n["b2_send"] * 4,
        "b2_recv": n["b2_recv"] * 4,
        "staging": staging,
        "carry": carry,
        "dpre_head": halo.n_head_rows * C * 4 if with_conv else 0,
        "dpre_tail": halo.n_tail_rows * C * 4 if with_conv else 0,
        "tickets": TICKET_BYTES,
        **({} if full else light_tapes),  # the replay of #1 + the #2 tape walk
        "do_g": R * HV * ROW_BF16_128,
        "dv_intra": R * HV * ROW_BF16_128,
        "dh_tape": NC_pad * HV * TAPE_BF16_TILE,
        "dv_new": 0,  # stored in place of dv_intra (one buffer: cake_kcp.TAPE_ALIASES)
        "dg_part": R * HV * ROW_F32,
        "gamma_dot": NC_pad * HV * CHUNK_F32,
        "group_partials": (2 * T_r * HV * HEAD_DIM * 4)
        if group_reduce == "host"
        else 0,
    }
    if not saved:  # replay of the whole boundary: the forward exchange buffers live again in the backward
        bwd.update(
            {
                "c1_send": n["c1_send"] * 4,
                "c1_recv": n["c1_recv"] * 4,
                "c2_send": n["c2_send"] * 4,
                "c2_recv": n["c2_recv"] * 4,
            }
        )
    out = {f"fwd.{k}": v for k, v in fwd.items()}
    if backward:
        out.update({f"bwd.{k}": v for k, v in bwd.items()})
    out["forward"] = sum(fwd.values())
    out["backward"] = sum(bwd.values()) if backward else 0
    out["workspace"] = max(out["forward"], out["backward"])
    return out


def gdn_kcp_workspace_bytes(
    ctx: KcpContext,
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
) -> int:
    """Caller-held workspace bytes of one rank (the ``workspace`` entry of :func:`gdn_kcp_workspace_layout`)."""
    return gdn_kcp_workspace_layout(
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
    )["workspace"]


__all__ = [
    "BOUNDARY_MODES",
    "GROUP_REDUCE",
    "SAVE_MODES",
    "TICKET_BYTES",
    "TICKET_SITES",
    "TICKET_SLOT_BYTES",
    "gdn_kcp_comm_bytes",
    "gdn_kcp_saved_bytes",
    "gdn_kcp_saved_layout",
    "gdn_kcp_workspace_bytes",
    "gdn_kcp_workspace_layout",
    "state_bytes",
]
