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

Checkpoint hand-off of the non-deterministic context-parallel (KCP) KDA training route
(``chunk_kda_kcp(..., checkpoint_handoff=)``): under ``torch.utils.checkpoint`` the forward keeps its tape set and
the backward reads it instead of re-running the intra program, WY and the K1 tape walk.

Multi-GPU rows (``HANDOFF_ROWS``: the checkpoint rows ``CKPT_ROWS`` of ``test_cake_kda_train_kcp_dist.py``, the
same launch forms, and two rows at H8 and H32): ``out``, the five gradients and ``dw`` (the checkpointed function
returns ``out * w``) with the hand-off on are bitwise those with it off and those of the direct pass, for
``use_reentrant`` False and True, with a recompute handle (both tiers), on a lean arena and a per-call workspace,
with the ring exchange (where the partition allows it) and the overlap schedule, and over two backwards
(``retain_graph``); a forward that keeps the tape set runs the tape walk and ``fwd_o`` where the light forward
runs the fused walk ``kcp_fwdh_o``, its other launches and every collective unchanged; the backward launches no intra
program, WY or K1 tape walk; the saved set grows by the byte query's difference; a NaN-filled tape set, poisoned free
blocks and a poisoned local arena (between the recompute and the backward) change no bit; ranks that decide
differently (``"on"`` on some, ``"off"`` on others) change no bit; the refusals (``"off"``, no checkpoint, a blocked
state such as a CUDA-graph capture, a selective-checkpoint policy) keep the light set in both passes and change no
bit; a Python dispatch mode in one pass only follows the original forward's decision in both; ``"auto"`` follows the
admission rule: the shipped work bound refuses these small rows, with the bound at 0 it engages on a verified device
at an admitted head count and refuses as ``"unverified"`` with the verified set emptied (each refusal counted once);
the facade ``flashinfer.cake_kda_train_kcp`` forwards the mode under ``deterministic=False`` and never engages under
``deterministic=True``; a checkpointed forward with the hand-off retains nothing.  A check whose light path does not
run under its dispatch mode or policy has nothing to compare: the row lists it under ``not_run``.
Single-process rows: the static rule, the admission table, the phase probe, the once-per-call decision (a refusal by
the call's own arguments records nothing in either pass), the device key, the host values kept across calls (default
options, pool needs, scratch table bytes), the per-call workspaces' shared identity tile, the saved set's tape members
and the byte queries of the package and the facade.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import json
import os
import tempfile
import traceback
import types

import pytest
import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode

import flashinfer.cake_kda_train_kcp as kcp_api
from tests.experimental.test_cake_kda_train_kcp_dist import (
    CKPT_ROWS,
    _bound_rank_threads,
    _init_group,
    _local_slices,
    _partition,
    _poison,
    _programs_exported,
    _requested,
    _row_id,
    _skip_unless_runnable,
    _start_ranks,
    cake_kcp,
    cake_kcp_plan,
)
from tests.test_helpers.cake_kda_det_train_reference import make_inputs

TENSORS = ("out", "dq", "dk", "dv", "dg", "dbeta", "dw")
# the launches the hand-off's backward drops (the forward's tape set stands in for their outputs)
SKIPPED = ("fwd_intra", "wy", "kcp_fwdh_tape")
# the shipped admission rule of "auto" (cake_kcp_plan.kcp_handoff_auto_refusal)
VERIFIED_KEYS = frozenset(
    {
        ("sm_100a", 148, "13.4"),
        ("sm_103a", 148, "13.4"),
        ("sm_103a", 152, "13.4"),
        ("sm_107a", 212, "13.4"),
    }
)
AUTO_HEADS = frozenset({4, 8, 16, 32, 64})
MIN_HEAD_TOKENS = 393216
# the hand-off rows: the checkpoint rows (H1, H4, H16, H64) and the H64 row's packing at the other head counts of
# AUTO_HEADS
HANDOFF_ROWS = CKPT_ROWS + [
    ((2048, 5, 3000, 1), heads, 4, "zigzag_per_document") for heads in (8, 32)
]


def _bits(t: torch.Tensor) -> torch.Tensor:
    # bit view: -0.0 vs 0.0 and NaN payloads count as differences
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def _same(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(
        a[n].dtype == b[n].dtype
        and a[n].shape == b[n].shape
        and torch.equal(_bits(a[n]), _bits(b[n]))
        for n in a
    )


def _collectives(seq) -> list:
    return [x for x in seq if x.startswith(("a2a:", "ring:"))]


class _PassThroughMode(TorchDispatchMode):
    """A Python dispatch mode that runs every op unchanged (a stand-in for a debug or flop-counting mode)."""

    def __torch_dispatch__(self, func, _types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


class _ArenaPoison:
    """The recompute's ``context_fn`` exit: fill the caller workspace's local arena with all-ones bytes (NaN in BF16 /
    FP32) after the recompute, before the node's backward runs."""

    def __init__(self, ws) -> None:
        self.ws = ws

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        local = getattr(self.ws, "local", None)
        arena = getattr(local, "arena", None)
        if arena is not None:
            arena.view(torch.uint8).fill_(0xFF)
        return False


def _run_rank_handoff(
    rank: int, W: int, row, rdzv: str | None, result_dir: str | None, local_id: int
) -> dict:
    """One rank of a hand-off row (see the module docstring)."""
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        checkpoint,
        create_selective_checkpoint_contexts,
    )

    lengths, H, _w, layout = row
    device = torch.device("cuda", local_id % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        _init_group(rank, W, device, rdzv)
    group = dist.group.WORLD
    part, cu = _partition(lengths, W, layout)
    ctx = cake_kcp_plan.build_context_kcp(part, group, rank, device)
    glob = make_inputs(lengths, H, device)
    ops = _local_slices(part, rank, cu, glob)
    q, k, v, g, beta, dout = (ops[n] for n in ("q", "k", "v", "g", "beta", "dout"))
    ws = cake_kcp.WorkspaceKcp()
    ws_lean = cake_kcp.WorkspaceKcp(local_arena="lean")
    T_r = int(q.shape[0])
    index = device.index if device.index is not None else torch.cuda.current_device()
    arch = cake_kcp.cake_jit.device_arch(index)
    sms = int(cake_kcp.cake_jit.sm_count(index))
    toolkit = cake_kcp._handoff_toolkit(arch)
    key = (arch, sms, toolkit)
    info: dict[str, object] = {
        "T_r": T_r,
        "n_frag": int(ctx.n_frag),
        "n_in": int(ctx.n_in),
        "P": int(ctx.heads(H).P),
        "key": list(key),
        "torch": torch.__version__,
    }
    flags: dict[str, bool] = {}
    # the flags that passed without a comparison (the light path does not run under the check's mode or policy)
    not_run: list[str] = []

    def leaves():
        return [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)]

    def run(
        mode: str,
        *,
        ckpt: bool = True,
        reentrant: bool = False,
        workspace=ws,
        handle=None,
        keep=None,
        context_fn=None,
        poison_free: bool = False,
        poison_tape: bool = False,
        fwd_mode: bool = False,
        bwd_mode: bool = False,
        twice: bool = False,
        **opts,
    ):
        """One checkpointed (or direct) step of ``out * w``: the results and the LAST_CALL records of the original
        forward, the recompute (None without one) and the backward.  ``fwd_mode`` / ``bwd_mode``: a pass-through
        Python dispatch mode around the forward / the backward (the recompute) only.  ``twice``: two backwards
        (``retain_graph`` on the first; the gradients accumulate, the records are the second's).  ``opts``: options
        of the call (``exchange``, ``_kcp_test_knobs``)."""
        if poison_free:
            _poison(device)
        lv = leaves()
        wl = torch.ones_like(dout).requires_grad_(True)
        kw = dict(opts)
        if handle is not None:
            kw["recompute_handle"] = handle
        if keep is not None:
            kw["checkpoint_keep"] = keep

        def top(q_, k_, v_, g_, b_, w_):
            y = cake_kcp.chunk_kda_kcp(
                q_,
                k_,
                v_,
                g_,
                b_,
                ctx,
                workspace=workspace,
                checkpoint_handoff=mode,
                **kw,
                deterministic=False,
            )
            return y * w_

        cake_kcp.LAST_CALL.pop("fwd", None)
        cake_kcp.LAST_CALL.pop("bwd", None)
        prev = cake_kcp._HANDOFF_POISON
        cake_kcp._HANDOFF_POISON = bool(poison_tape)
        try:
            with _PassThroughMode() if fwd_mode else contextlib.nullcontext():
                if not ckpt:
                    out = top(*lv, wl)
                elif reentrant:
                    out = checkpoint(top, *lv, wl, use_reentrant=True)
                elif context_fn is not None:
                    out = checkpoint(
                        top, *lv, wl, use_reentrant=False, context_fn=context_fn
                    )
                else:
                    out = checkpoint(top, *lv, wl, use_reentrant=False)
            rec_orig = dict(cake_kcp.LAST_CALL["fwd"])
            if ckpt:
                cake_kcp.LAST_CALL.pop("fwd")
            with _PassThroughMode() if bwd_mode else contextlib.nullcontext():
                for i in range(2 if twice else 1):
                    out.backward(dout, retain_graph=twice and i == 0)
        finally:
            cake_kcp._HANDOFF_POISON = prev
        rec_fwd = cake_kcp.LAST_CALL.get("fwd")
        rec_bwd = dict(cake_kcp.LAST_CALL["bwd"])
        res = {"out": out.detach(), "dw": wl.grad}
        res.update(
            {
                f"d{n}": lf.grad
                for n, lf in zip(("q", "k", "v", "g", "beta"), lv, strict=True)
            }
        )
        return res, {"orig": rec_orig, "fwd": rec_fwd, "bwd": rec_bwd}

    def engaged(recs) -> bool:
        """The original forward and the recompute kept a tape set and the backward read it (a rank with tokens), or
        none of them did (a rank without)."""
        want = ("save", "save", "bound") if T_r > 0 else (None, None, None)
        got = (
            recs["orig"].get("checkpoint_handoff"),
            (recs["fwd"] or {}).get("checkpoint_handoff"),
            recs["bwd"].get("checkpoint_handoff"),
        )
        return got == want

    def light(recs) -> bool:
        return all(
            (r or {}).get("checkpoint_handoff") is None
            for r in (recs["orig"], recs["fwd"], recs["bwd"])
        )

    def light_form(rec) -> list:
        """A forward's launches in the light path's form: a forward that kept the tape set runs the tape walk and
        ``fwd_o`` where the light forward runs the fused walk ``kcp_fwdh_o`` (its backward reads the tapes)."""
        seq = rec["sequence"]
        if rec.get("checkpoint_handoff") != "save":
            return list(seq)
        if seq.count("fwd_o") != 1 or "kcp_fwdh_o" in seq:
            return ["not the tape form", *seq]  # never the light path's sequence
        return [
            ("kcp_fwdh_o" if x == "kcp_fwdh_tape" else x) for x in seq if x != "fwd_o"
        ]

    def skips(recs_on, recs_off) -> bool:
        """The hand-off's backward launches none of SKIPPED; the forward's launches (in the light path's form,
        :func:`light_form`) and the backward's collectives are the light path's."""
        on_b, off_b = recs_on["bwd"]["sequence"], recs_off["bwd"]["sequence"]
        ok = light_form(recs_on["orig"]) == recs_off["orig"]["sequence"]
        ok = ok and light_form(recs_on["fwd"]) == recs_off["fwd"]["sequence"]
        ok = ok and _collectives(on_b) == _collectives(off_b)
        ok = ok and recs_on["bwd"]["n_collectives"] == recs_off["bwd"]["n_collectives"]
        if T_r > 0:
            ok = ok and not any(x in SKIPPED for x in on_b)
            ok = ok and all(x in off_b for x in ("fwd_intra", "wy", "kcp_fwdh_tape"))
        return ok

    direct, recs_direct = run("off", ckpt=False)
    base, recs_base = run("off")
    flags["off_light"] = light(recs_base) and light(recs_direct)
    flags["off_equals_direct"] = _same(base, direct)
    # the hand-off on: non-reentrant and reentrant
    on_nr, recs_on = run("on")
    flags["on_nonreentrant_bitwise"] = _same(on_nr, base)
    flags["on_nonreentrant_engaged"] = engaged(recs_on)
    flags["on_launches"] = skips(recs_on, recs_base)
    want_bytes = cake_kcp.kda_handoff_bytes_kcp(ctx, H=H) if T_r > 0 else None
    flags["on_bytes"] = (
        recs_on["fwd"].get("checkpoint_handoff_bytes") == want_bytes
        and recs_on["bwd"].get("checkpoint_handoff_bytes") == want_bytes
    )
    # the saved set: the hand-off forward's minus the light forward's is the byte query's difference (the tape set
    # in, the carry plane out); the absolute figures are recorded (the query also counts the int32 tables)
    st = recs_on["fwd"]["save_transfer"]
    query = {
        flag: cake_kcp_plan.kda_saved_bytes_kcp(
            ctx, H=H, with_conv=False, save_transfer=st, checkpoint_handoff=flag
        )
        for flag in (False, True)
    }
    got = {"off": recs_base["fwd"]["saved_bytes"], "on": recs_on["fwd"]["saved_bytes"]}
    info["saved_bytes"] = {
        "got": got,
        "query": query,
        "tables": cake_kcp_plan.tables_bytes(ctx.NC_pad, ctx.n_frag, ctx.heads(H).NSEG),
    }
    flags["on_saved_bytes"] = got["on"] - got["off"] == (
        query[True] - query[False] if T_r > 0 else 0
    )
    off_r, recs_off_r = run("off", reentrant=True)
    on_r, recs_on_r = run("on", reentrant=True)
    flags["on_reentrant_bitwise"] = _same(on_r, base) and _same(off_r, base)
    flags["on_reentrant_engaged"] = (
        recs_on_r["fwd"] is not None
        and recs_on_r["fwd"].get("checkpoint_handoff") == ("save" if T_r > 0 else None)
        and recs_on_r["bwd"].get("checkpoint_handoff") == ("bound" if T_r > 0 else None)
        # the reentrant first pass runs under no_grad: no request
        and recs_on_r["orig"].get("checkpoint_handoff") is None
        and light(recs_off_r)
    )
    # with a recompute handle (both tiers: "states" runs save_transfer=False, so a P > 1 backward re-walks K3)
    for keep in ("states", "transfer"):
        _, recs_h_off = run("off", handle=cake_kcp.RecomputeHandleKcp(), keep=keep)
        res_h, recs_h = run("on", handle=cake_kcp.RecomputeHandleKcp(), keep=keep)
        flags[f"handle_{keep}_bitwise"] = _same(res_h, base)
        flags[f"handle_{keep}_engaged"] = engaged(recs_h)
        flags[f"handle_{keep}_launches"] = skips(recs_h, recs_h_off)
    # a lean arena (the tail's per-range views of the bound buffers) and a per-call workspace
    res_lean_off, _ = run("off", workspace=ws_lean)
    res_lean, recs_lean = run("on", workspace=ws_lean)
    flags["lean_bitwise"] = _same(res_lean, base) and _same(res_lean_off, base)
    flags["lean_engaged"] = engaged(recs_lean)
    res_pc, recs_pc = run("on", workspace=None)
    flags["per_call_bitwise"] = _same(res_pc, base)
    flags["per_call_engaged"] = engaged(recs_pc)
    # NaN poison: a NaN-filled tape set and poisoned free blocks (per-call workspace: its arena comes from them), and
    # the caller arena poisoned between the recompute and the backward (both modes)
    res_p, recs_p = run("on", workspace=None, poison_free=True, poison_tape=True)
    flags["poison_tape_free_bitwise"] = _same(res_p, base)
    flags["poison_tape_free_engaged"] = engaged(recs_p)
    arena_fn = lambda: (contextlib.nullcontext(), _ArenaPoison(ws))  # noqa: E731
    res_pa_off, _ = run("off", context_fn=arena_fn)
    res_pa_on, recs_pa = run("on", context_fn=arena_fn, poison_tape=True)
    flags["poison_arena_off_bitwise"] = _same(res_pa_off, base)
    flags["poison_arena_on_bitwise"] = _same(res_pa_on, base)
    flags["poison_arena_engaged"] = engaged(recs_pa)
    # ranks that decide differently ("on" on the even ranks, "off" on the odd ones): the hand-off moves no
    # collective, so nothing hangs and no bit changes
    mixed = "on" if rank % 2 == 0 else "off"
    res_mx, recs_mx = run(mixed)
    flags["mixed_ranks_bitwise"] = _same(res_mx, base)
    flags["mixed_ranks_engaged"] = engaged(recs_mx) if mixed == "on" else light(recs_mx)
    # the ring exchange (where the partition allows it) and the overlap schedule (test knob; with a2a and the ring),
    # from a poisoned allocator: bitwise "off", whose launches the hand-off's are minus SKIPPED
    ring_ok = int(ctx.n_d_max) - 1 <= cake_kcp.RING_MAX_HOPS
    info["ring_eligible"] = ring_ok
    overlap = {"_kcp_test_knobs": {"schedule": "overlap"}}
    forms = {"overlap": overlap}
    if ring_ok:
        forms["ring"] = {"exchange": "ring"}
        forms["ring_overlap"] = {"exchange": "ring", **overlap}
    previous_knobs = cake_kcp_plan.enable_test_knobs(True)
    try:
        for form, opts in forms.items():
            res_f_off, recs_f_off = run("off", poison_free=True, **opts)
            res_f_on, recs_f_on = run("on", poison_free=True, **opts)
            flags[f"{form}_bitwise"] = _same(res_f_off, base) and _same(res_f_on, base)
            flags[f"{form}_engaged"] = engaged(recs_f_on) and light(recs_f_off)
            flags[f"{form}_launches"] = skips(recs_f_on, recs_f_off)
            if "exchange" in opts:
                flags[f"{form}_exchange"] = (
                    recs_f_on["orig"]["exchange"] == opts["exchange"]
                    and recs_f_off["orig"]["exchange"] == opts["exchange"]
                )
    finally:
        cake_kcp_plan.enable_test_knobs(previous_knobs)
    # retain_graph: each backward replays the original forward's decision; exactly twice the gradients
    res_tw, recs_tw = run("on", twice=True)
    flags["twice_bitwise"] = _same(
        res_tw, {n: base[n] if n == "out" else base[n] + base[n] for n in base}
    )
    flags["twice_engaged"] = engaged(recs_tw)
    # refusals: no checkpoint (no phase), a blocked state (CUDA-graph capture / torch.compile tracing, simulated in both
    # passes), a selective-checkpoint policy (a dispatch mode in both passes), "auto" off the verified set
    res_nc, recs_nc = run("on", ckpt=False)
    flags["refuse_no_checkpoint"] = _same(res_nc, direct) and light(recs_nc)
    blocked = cake_kcp._handoff_blocked
    cake_kcp._handoff_blocked = lambda: True
    try:
        res_bl, recs_bl = run("on")
    finally:
        cake_kcp._handoff_blocked = blocked
    flags["refuse_blocked"] = _same(res_bl, base) and light(recs_bl)
    sac = lambda: create_selective_checkpoint_contexts(  # noqa: E731
        lambda c, op, *a, **kw: CheckpointPolicy.PREFER_RECOMPUTE
    )
    try:
        res_sac_off, _ = run("off", context_fn=sac)
    except Exception:  # noqa: BLE001 - the light path itself does not run under the policy: nothing to compare
        info["sac_unsupported"] = traceback.format_exc()[-1500:]
        res_sac_off = None
    if res_sac_off is None:
        flags["refuse_sac"] = True
        not_run.append("refuse_sac")
    else:
        try:
            res_sac, recs_sac = run("on", context_fn=sac)
            flags["refuse_sac"] = (
                _same(res_sac, res_sac_off) and _same(res_sac, base) and light(recs_sac)
            )
        except Exception:  # noqa: BLE001 - recorded; the flag fails
            info["sac_error"] = traceback.format_exc()[-1500:]
            flags["refuse_sac"] = False
    # a Python dispatch mode in one pass only: the original forward decides (refused under the mode, engaged without
    # it) and the recompute replays that decision, so the two passes save the same tensors (each side on its own)
    for side in ("fwd", "bwd"):
        mode_kw = {f"{side}_mode": True}
        try:
            res_m_off, _ = run("off", **mode_kw)
        except Exception:  # noqa: BLE001 - the light path itself does not run under the mode: nothing to compare
            info[f"mode_{side}_only_unsupported"] = traceback.format_exc()[-1500:]
            flags[f"mode_{side}_only"] = True
            not_run.append(f"mode_{side}_only")
            continue
        try:
            res_m, recs_m = run("on", **mode_kw)
            flags[f"mode_{side}_only"] = (
                _same(res_m, res_m_off)
                and _same(res_m, base)
                and (light(recs_m) if side == "fwd" else engaged(recs_m))
            )
        except Exception:  # noqa: BLE001 - recorded; the flag fails
            info[f"mode_{side}_only_error"] = traceback.format_exc()[-1500:]
            flags[f"mode_{side}_only"] = False
    # "auto": the shipped rule (on these small rows the work bound refuses, or the head count at H1), then the work
    # bound at 0 (cake_kcp_plan's constants are read at call time): engaged on a verified device at an admitted head
    # count, refused as "unverified" with the verified set emptied.  Decided once per checkpointed call: a refusal is
    # counted once (in the original forward), an engaged call saves twice and binds once
    total = ctx.partition.total
    shipped = (
        cake_kcp_plan.KCP_HANDOFF_MIN_HEAD_TOKENS,
        cake_kcp_plan.KCP_HANDOFF_VERIFIED,
    )
    kept = {"save": 2, "bound": 1} if T_r > 0 else {}

    @contextlib.contextmanager
    def auto_rule(min_head_tokens: int, verified):
        """``"auto"``'s admission constants for the block; yields the rule's reason for this row (``None``:
        admitted)."""
        cake_kcp_plan.KCP_HANDOFF_MIN_HEAD_TOKENS = min_head_tokens
        cake_kcp_plan.KCP_HANDOFF_VERIFIED = verified
        try:
            yield cake_kcp_plan.kcp_handoff_auto_refusal(key, H, total, W)
        finally:
            (
                cake_kcp_plan.KCP_HANDOFF_MIN_HEAD_TOKENS,
                cake_kcp_plan.KCP_HANDOFF_VERIFIED,
            ) = shipped

    reasons = {}
    for case, rule in (
        ("shipped", shipped),
        ("bound0", (0, shipped[1])),
        ("bound0_unverified", (0, frozenset())),
    ):
        with auto_rule(*rule) as reason:
            before = cake_kcp.kcp_handoff_stats()
            res_a, recs_a = run("auto")
            after = cake_kcp.kcp_handoff_stats()
        reasons[case] = reason
        delta = {n: after[n] - before[n] for n in after if after[n] != before[n]}
        if reason is None:
            flags[f"auto_{case}"] = (
                _same(res_a, base) and engaged(recs_a) and delta == kept
            )
        else:
            flags[f"auto_{case}"] = (
                _same(res_a, base)
                and light(recs_a)
                and delta == {"refused_" + reason: 1}
            )
    info["auto_reasons"] = reasons

    # the facade: deterministic=False forwards the mode (engaged as the package decides), deterministic=True runs
    # the deterministic package, which has no hand-off (the counters do not move; every mode gives the same bits)
    def run_facade(mode: str, det: bool):
        lv = leaves()
        wl = torch.ones_like(dout).requires_grad_(True)

        def top(q_, k_, v_, g_, b_, w_):
            y = kcp_api.chunk_kda_kcp(
                q_,
                k_,
                v_,
                g_,
                b_,
                ctx,
                workspace=None if det else ws,
                checkpoint_handoff=mode,
                deterministic=det,
            )
            return y * w_

        before = kcp_api.kcp_handoff_stats()
        out = checkpoint(top, *lv, wl, use_reentrant=False)
        out.backward(dout)
        after = kcp_api.kcp_handoff_stats()
        res = {"out": out.detach(), "dw": wl.grad}
        res.update(
            {
                f"d{n}": lf.grad
                for n, lf in zip(("q", "k", "v", "g", "beta"), lv, strict=True)
            }
        )
        return res, {k: after[k] - before[k] for k in after if after[k] != before[k]}

    det_res = {}
    for mode in ("auto", "on", "off"):
        # the work bound at 0: "auto" engages as on a production-size batch of a verified device
        with auto_rule(0, shipped[1]) as reason:
            res_f, delta = run_facade(mode, False)
        want = kept if (mode == "on" or (mode == "auto" and reason is None)) else {}
        flags[f"facade_nd_{mode}_bitwise"] = _same(res_f, base)
        flags[f"facade_nd_{mode}_engaged"] = {
            k: n for k, n in delta.items() if k in ("save", "bound")
        } == want
        det_res[mode], delta = run_facade(mode, True)
        flags[f"facade_det_{mode}_never_engages"] = delta == {}
    flags["facade_det_modes_bitwise"] = _same(det_res["on"], det_res["off"]) and _same(
        det_res["auto"], det_res["off"]
    )
    # #7039: a checkpointed forward with the hand-off retains nothing (its original forward's set goes at return),
    # after one warm-up step of the same form: per call, on this row's workspace and on a fresh caller-held workspace
    # (its first backward binds a new backward arena that no intra program of the hand-off backward touches)
    retained = {}
    for form, wsp in (
        ("per_call", None),
        ("caller_ws", ws),
        ("fresh_ws", cake_kcp.WorkspaceKcp()),
    ):

        def fn0(*a, _w=wsp):
            return cake_kcp.chunk_kda_kcp(
                *a, ctx, workspace=_w, checkpoint_handoff="on", deterministic=False
            )

        for step in ("warm", "measure"):
            lv = leaves()
            torch.cuda.synchronize(device)
            gc.collect()
            before = _requested(device)
            out = checkpoint(fn0, *lv, use_reentrant=False)
            torch.cuda.synchronize(device)
            if step == "measure":
                retained[form] = (
                    _requested(device) - before - out.numel() * out.element_size()
                )
            out.backward(dout)
            del out, lv
    info["retained_bytes"] = retained
    flags["retained_zero"] = all(r == 0 for r in retained.values())
    info["not_run"] = not_run

    gathered: list = [None] * W
    dist.all_gather_object(gathered, (flags, info))
    flag_list = [f for f, _ in gathered]
    info_list = [i for _, i in gathered]
    report = {
        "row": list(map(str, row)),
        "flags": {key_: all(f[key_] for f in flag_list) for key_ in flags},
        "failing_ranks": {
            key_: [r for r, f in enumerate(flag_list) if not f[key_]] for key_ in flags
        },
        "not_run": sorted({n for i in info_list for n in i["not_run"]}),
        "info_by_rank": info_list,
    }
    report["pass"] = all(report["flags"].values()) and all(
        set(f) == set(flag_list[0]) for f in flag_list
    )
    dist.barrier()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(
            os.path.join(result_dir, "handoff-" + _row_id(row) + ".json"), "w"
        ) as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _spawn_entry_handoff(
    rank: int, W: int, row, rdzv: str | None, result_dir: str
) -> None:
    try:
        _bound_rank_threads(W)
        report = _run_rank_handoff(rank, W, row, rdzv, result_dir, rank)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("row", HANDOFF_ROWS, ids=_row_id)
def test_checkpoint_handoff_rows(row) -> None:
    """The hand-off rows of ``_run_rank_handoff`` on every rank of the row."""
    W = row[2]
    ntasks = int(os.environ.get("SLURM_NTASKS", "1") or 1)
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"a {ntasks}-task Slurm step runs the W={ntasks} rows only")
        if not _programs_exported():
            pytest.skip(
                "the KCP programs are not exported in the package's build.json yet"
            )
        report = _run_rank_handoff(
            int(os.environ["SLURM_PROCID"]),
            W,
            row,
            None,
            os.environ.get("KCP_TEST_RESULT_DIR"),
            int(os.environ.get("SLURM_LOCALID", "0")),
        )
        assert report["pass"], report
        return
    _skip_unless_runnable(W)
    with tempfile.TemporaryDirectory(prefix="cake_kcp_handoff_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        _start_ranks(_spawn_entry_handoff, W, row, tmp, result_dir)
        with open(os.path.join(result_dir, "handoff-" + _row_id(row) + ".json")) as fh:
            report = json.load(fh)
    assert report["pass"], report


# ---------------------------------------------------------------------------------------------------------------
# single process: the static rule, the admission table, the phase probe, the once-per-call decision, the saved set,
# the byte queries


def test_handoff_static_rule_and_modes() -> None:
    """The modes, the shipped admission constants, the refusal reasons of the static rule and the mode check of the
    entry."""
    assert cake_kcp.KCP_HANDOFF_MODES == ("auto", "on", "off")
    assert cake_kcp.KCP_HANDOFF_DEFAULT == "auto"
    assert cake_kcp.KCP_HANDOFF_VERIFIED is cake_kcp_plan.KCP_HANDOFF_VERIFIED
    assert cake_kcp.KCP_HANDOFF_HEADS is cake_kcp_plan.KCP_HANDOFF_HEADS
    assert (
        cake_kcp.KCP_HANDOFF_MIN_HEAD_TOKENS
        == cake_kcp_plan.KCP_HANDOFF_MIN_HEAD_TOKENS
        == MIN_HEAD_TOKENS
    )
    assert cake_kcp_plan.KCP_HANDOFF_VERIFIED == VERIFIED_KEYS
    assert cake_kcp_plan.KCP_HANDOFF_HEADS == AUTO_HEADS
    assert cake_kcp_plan.KCP_HANDOFF_BUFFERS == cake_kcp.SavedKcp._TAPE
    with pytest.raises(ValueError, match="checkpoint_handoff"):
        cake_kcp._handoff_mode("yes")
    opts = cake_kcp._options({})
    v_cpu = torch.zeros(8, 4, 128)
    assert cake_kcp._handoff_refusal("on", opts, "replay", v_cpu, None) == "boundary"
    assert (
        cake_kcp._handoff_refusal(
            "on", cake_kcp._options({"graph": "replay"}), "saved", v_cpu, None
        )
        == "graph"
    )
    assert (
        cake_kcp._handoff_refusal(
            "on", cake_kcp._options({"spec3_probe": "record"}), "saved", v_cpu, None
        )
        == "spec3"
    )
    assert cake_kcp._handoff_refusal("on", opts, "saved", v_cpu, None) == "device"
    # outside any checkpoint pass there is no request
    x = torch.zeros(2, requires_grad=True)
    assert cake_kcp._handoff_phase(x) is None
    with torch.no_grad():
        assert cake_kcp._handoff_phase(x) is None


def _auto_want(key, H: int, total: int, W: int) -> str | None:
    if H not in AUTO_HEADS:
        return "heads"
    if H * (total // W) < MIN_HEAD_TOKENS:
        return "work"
    return None if key in VERIFIED_KEYS else "unverified"


def test_handoff_auto_admission_table() -> None:
    """``"auto"``'s admission rule over device keys x head counts x world sizes x batch sizes, and the measured rows
    it was set from (T = 261392 packed tokens): H4 engages at CP2 (H * T // W = 522784) and not at CP4 (261392, the
    GB300 loss) or CP8; H16 and H64 engage at CP2, CP4 and CP8 (H16 CP8: 522784)."""
    other_keys = [
        ("sm_100a", 132, "13.4"),
        ("sm_103a", 152, "13.3"),
        ("sm_103a", 152, None),
        ("sm_107a", 216, "13.4"),
        ("sm_90a", 132, "13.4"),
    ]
    T = 261392
    for key in sorted(VERIFIED_KEYS) + other_keys:
        for H in (1, 2, 4, 8, 12, 16, 32, 48, 64, 128):
            for W in (1, 2, 4, 8, 16):
                for total in (T, 65536, 98304, 1 << 20):
                    got = cake_kcp_plan.kcp_handoff_auto_refusal(key, H, total, W)
                    assert got == _auto_want(key, H, total, W), (key, H, W, total)
    for key in VERIFIED_KEYS:
        assert cake_kcp_plan.kcp_handoff_auto_refusal(key, 4, T, 2) is None
        assert cake_kcp_plan.kcp_handoff_auto_refusal(key, 4, T, 4) == "work"
        assert cake_kcp_plan.kcp_handoff_auto_refusal(key, 4, T, 8) == "work"
        for H in (16, 64):
            for W in (2, 4, 8):
                assert cake_kcp_plan.kcp_handoff_auto_refusal(key, H, T, W) is None
        # the bound is inclusive
        assert cake_kcp_plan.kcp_handoff_auto_refusal(key, 4, 98304, 1) is None
        assert cake_kcp_plan.kcp_handoff_auto_refusal(key, 4, 98303, 1) == "work"


def test_handoff_auto_decides_alike_on_every_rank(monkeypatch) -> None:
    """``"auto"`` reads the partition's total and world size, not the rank's share: every rank of an uneven
    partition (ranks with different token counts, some without tokens) refuses or admits alike."""
    backend = cake_kcp.KernelBackend()
    monkeypatch.setattr(cake_kcp, "get_backend", lambda: backend)
    opts = cake_kcp._options({})
    lengths = (200000, 100, 61292)  # T = 261392
    layouts = {
        1: "ragged:261392",
        2: "ragged:200000,61392",
        4: "ragged:200000,0,40000,21392",
        8: "ragged:1,1,1,3,0,2,200000,61384",
    }
    contexts = {}
    for W, layout in layouts.items():
        part, _cu = _partition(lengths, W, layout)
        contexts[W] = [
            cake_kcp_plan.build_context_kcp(part, None, r, "cpu", 4, sm_count=148)
            for r in range(W)
        ]
    for key in (("sm_103a", 152, "13.4"), ("sm_90a", 132, "13.4")):
        monkeypatch.setattr(cake_kcp, "_handoff_key", lambda device, _k=key: _k)
        for W, ctxs in contexts.items():
            shares = [ctx.T_r for ctx in ctxs]
            assert W == 1 or len(set(shares)) > 1, shares
            assert W < 4 or min(shares) == 0, shares
            for H in (4, 8, 16, 64, 128):
                got = set()
                for ctx in ctxs:
                    v = types.SimpleNamespace(
                        device=torch.device("cuda", 0), shape=(ctx.T_r, H, 128)
                    )
                    got.add(cake_kcp._handoff_refusal("auto", opts, "saved", v, ctx))
                assert got == {_auto_want(key, H, 261392, W)}, (key, W, H, got)


def test_handoff_phase_probe_reads_the_checkpoint_passes() -> None:
    """``"original"`` / ``"recompute"`` in the two passes of a non-reentrant checkpoint, nothing in the reentrant
    first pass (no_grad) and ``"reentrant"`` in its recompute; under a saved-tensor hook of the caller, nothing.  The
    function saves a tensor (``x * x``): a non-reentrant checkpoint recomputes only to unpack one."""
    from torch.utils.checkpoint import checkpoint

    seen: list = []

    def fn(x):
        seen.append(cake_kcp._handoff_phase(x))
        return x * x

    x = torch.ones(4, requires_grad=True)
    checkpoint(fn, x, use_reentrant=False).sum().backward()
    assert seen == ["original", "recompute"], seen
    seen.clear()
    x = torch.ones(4, requires_grad=True)
    checkpoint(fn, x, use_reentrant=True).sum().backward()
    assert seen == [None, "reentrant"], seen
    seen.clear()
    with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
        fn(torch.ones(4, requires_grad=True))
    assert seen == [None], seen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_handoff_is_blocked_inside_a_cuda_graph_capture() -> None:
    """A CUDA-graph capture blocks the request (a replay would hand a dropped set to a later backward)."""
    assert not cake_kcp._handoff_blocked()
    graph = torch.cuda.CUDAGraph()
    side = torch.zeros(1, device="cuda")
    with torch.cuda.graph(graph):
        inside = cake_kcp._handoff_blocked()
        side.add_(1.0)
    assert inside


def test_handoff_decision_is_made_once_per_checkpointed_call(monkeypatch) -> None:
    """A non-reentrant checkpoint decides the hand-off in its original forward and every recompute of the region
    replays those decisions call by call, whatever capture, compile or dispatch-mode state it runs under, so the two
    passes save the same tensors (torch raises CheckpointError otherwise): a dispatch mode around the forward only
    refuses in both passes, one around the backward only engages in both, a state blocked in the recompute only or in
    one call of the original forward changes nothing in the recompute, each backward of a ``retain_graph`` step
    replays from the first call.  A frame the hooks do not expose keeps the light set in both passes; a reentrant
    checkpoint decides in its recompute (its first pass runs under no_grad); the decisions go with their frame."""
    from torch.utils.checkpoint import checkpoint

    assert cake_kcp._handoff_frames_readable()
    # every request admitted by its arguments (these CPU tensors would be refused as "device") and by the rest
    monkeypatch.setattr(cake_kcp, "_handoff_static", lambda *a: None)
    monkeypatch.setattr(cake_kcp, "_handoff_refusal", lambda *a: None)
    # no capture or tracing (the capture read itself is test_handoff_is_blocked_inside_a_cuda_graph_capture's): the
    # decisions below also run on a host without CUDA, where that read raises and so blocks
    monkeypatch.setattr(cake_kcp, "_handoff_blocked", lambda: False)
    opts = cake_kcp._options({})
    seen: list = []
    gc.collect()
    frames_before = len(cake_kcp._HANDOFF_DECISIONS)

    def fn(x):
        for _ in range(2):  # two calls per region: replayed in call order
            seen.append(
                cake_kcp._handoff_decide("on", opts, "saved", None, x, x, x, x, x)
            )
        return x * x  # saves x: the backward unpacks it and recomputes the region

    def step(fwd_mode=False, bwd_mode=False, backwards=1, reentrant=False) -> list:
        seen.clear()
        x = torch.ones(4, requires_grad=True)
        with _PassThroughMode() if fwd_mode else contextlib.nullcontext():
            y = checkpoint(fn, x, use_reentrant=reentrant).sum()
        for i in range(backwards):
            with _PassThroughMode() if bwd_mode else contextlib.nullcontext():
                y.backward(retain_graph=i + 1 < backwards)
        return list(seen)

    assert step() == [True] * 4
    assert step(fwd_mode=True) == [False] * 4
    assert step(bwd_mode=True) == [True] * 4
    assert step(backwards=2) == [True] * 6
    assert step(reentrant=True) == [False, False, True, True]
    blocked = cake_kcp._handoff_blocked

    def blocked_in_backward() -> bool:
        return cake_kcp._handoff_task() not in (None, -1)

    monkeypatch.setattr(cake_kcp, "_handoff_blocked", blocked_in_backward)
    assert step() == [True] * 4
    calls = []

    def second_call_blocked() -> bool:
        calls.append(None)
        return len(calls) == 2

    monkeypatch.setattr(cake_kcp, "_handoff_blocked", second_call_blocked)
    assert step() == [True, False, True, False]
    monkeypatch.setattr(cake_kcp, "_handoff_blocked", blocked)
    frame = cake_kcp._handoff_frame
    monkeypatch.setattr(cake_kcp, "_handoff_frame", lambda phase: None)
    assert step() == [False] * 4
    monkeypatch.setattr(cake_kcp, "_handoff_frame", frame)
    gc.collect()
    assert len(cake_kcp._HANDOFF_DECISIONS) == frames_before


def test_handoff_argument_refusals_record_nothing(monkeypatch) -> None:
    """A request the call's own arguments refuse (``_handoff_static``: here ``"auto"``'s work bound and
    ``boundary_mode="replay"``) is refused in both passes alike: it records no decision on the frame and takes no
    recompute cursor (the frame holds the admitted calls' decisions only and each recompute advances its cursor once
    per admitted call), so the admitted calls around it replay their own decisions in call order (the first one blocked
    in the original forward, the last one engaged); it reads no device key, and it is counted once per checkpointed
    call -- in the original forward, or in the reentrant recompute -- under ``_handoff_refusal``'s order of reasons
    (the backend before the device and the "auto" rule).  It is not counted under a capture or tracing block, under a
    dispatch mode, or in an original forward whose frames torch does not expose (the reentrant recompute, which reads
    no frame, still counts there)."""
    from torch.utils.checkpoint import checkpoint

    assert cake_kcp._handoff_frames_readable()

    def no_key(device):
        raise AssertionError("a request was decided from the device key")

    monkeypatch.setattr(cake_kcp, "_handoff_key", no_key)
    opts = cake_kcp._options({})
    # H4 at CP8 of T = 261392: H * (T // W) = 130696, below the bound ("work"); "on" reads no rule
    ctx = types.SimpleNamespace(
        partition=types.SimpleNamespace(total=261392), world_size=8
    )
    v = types.SimpleNamespace(device=torch.device("cuda", 0), shape=(32674, 4, 128))
    assert cake_kcp._handoff_static("auto", opts, "saved", v, ctx) == "work"
    assert cake_kcp._handoff_static("on", opts, "replay", v, ctx) == "boundary"
    assert cake_kcp._handoff_static("on", opts, "saved", v, ctx) is None
    blocked: list = []

    # the first dynamic read of a step blocks: that call records False
    def first_blocked() -> bool:
        blocked.append(None)
        return len(blocked) == 1

    monkeypatch.setattr(cake_kcp, "_handoff_blocked", first_blocked)
    seen: list = []
    frames: list = []
    mixed = [("on", "saved"), ("auto", "saved"), ("on", "replay"), ("on", "saved")]
    refused = [("auto", "saved"), ("on", "replay")]
    calls = list(mixed)

    def fn(x):
        for mode, boundary in calls:
            seen.append(
                cake_kcp._handoff_decide(mode, opts, boundary, ctx, x, x, v, x, x)
            )
        # the frame's record at the end of this pass: its decisions and its recompute cursors
        phase = cake_kcp._handoff_phase(x)
        found = (
            cake_kcp._handoff_frame(phase)
            if phase in ("original", "recompute")
            else None
        )
        entry = None if found is None else cake_kcp._HANDOFF_DECISIONS.get(found[0])
        if entry is not None:
            frames.append((phase, list(entry[0]), sorted(entry[1].values())))
        return x * x

    def step(reentrant: bool, fwd_mode: bool = False) -> tuple[list, dict, list]:
        seen.clear()
        frames.clear()
        blocked.clear()
        before = cake_kcp.kcp_handoff_stats()
        x = torch.ones(4, requires_grad=True)
        with _PassThroughMode() if fwd_mode else contextlib.nullcontext():
            y = checkpoint(fn, x, use_reentrant=reentrant).sum()
        y.backward()
        after = cake_kcp.kcp_handoff_stats()
        delta = {n: after[n] - before[n] for n in after if after[n] != before[n]}
        return list(seen), delta, list(frames)

    counted = {"refused_work": 1, "refused_boundary": 1}
    # the frame holds the two admitted calls' decisions; the recompute's cursor advanced twice
    recorded = [("original", [False, True], []), ("recompute", [False, True], [2])]
    assert step(False) == ([False, False, False, True] * 2, counted, recorded)
    assert step(True) == ([False] * 4 + [False, False, False, True], counted, [])
    # where an argument refusal is counted: refused calls only, so no frame record exists in any pass; counted with
    # nothing blocking, not under a capture or tracing block, a dispatch mode or frames torch does not expose
    calls[:] = refused
    monkeypatch.setattr(cake_kcp, "_handoff_blocked", lambda: False)
    assert step(False) == ([False] * 4, counted, [])
    assert step(True) == ([False] * 4, counted, [])
    monkeypatch.setattr(cake_kcp, "_handoff_blocked", lambda: True)
    assert step(False) == ([False] * 4, {}, [])
    assert step(True) == ([False] * 4, {}, [])
    monkeypatch.setattr(cake_kcp, "_handoff_blocked", lambda: False)
    assert step(False, fwd_mode=True) == ([False] * 4, {}, [])
    modes = cake_kcp._handoff_dispatch_modes
    monkeypatch.setattr(cake_kcp, "_handoff_dispatch_modes", lambda: True)
    assert step(True) == ([False] * 4, {}, [])
    monkeypatch.setattr(cake_kcp, "_handoff_dispatch_modes", modes)
    readable = cake_kcp._handoff_frames_readable
    monkeypatch.setattr(cake_kcp, "_handoff_frames_readable", lambda: False)
    assert step(False) == ([False] * 4, {}, [])
    assert step(True) == ([False] * 4, counted, [])
    monkeypatch.setattr(cake_kcp, "_handoff_frames_readable", readable)
    calls[:] = mixed
    monkeypatch.setattr(cake_kcp, "_handoff_blocked", first_blocked)
    # the reasons' order: the backend before the device and "auto"'s rule, after the option keys
    monkeypatch.setattr(cake_kcp, "get_backend", lambda: object())
    recorded = [("original", [False, False], []), ("recompute", [False, False], [2])]
    assert step(False) == (
        [False] * 8,
        {"refused_backend": 2, "refused_boundary": 1},
        recorded,
    )
    assert cake_kcp._handoff_refusal("auto", opts, "saved", v, ctx) == "backend"
    assert cake_kcp._handoff_refusal("on", opts, "replay", v, ctx) == "boundary"


def test_handoff_device_key_is_kept_per_index(monkeypatch) -> None:
    """``_handoff_key`` reads a device's architecture, SM count and toolkit once per device index; a device without an
    exported architecture raises on every call (nothing is kept for it)."""
    calls: list = []

    def arch_of(index: int) -> str:
        calls.append(index)
        if index == 1:
            raise RuntimeError("not an exported Cake KDA training target")
        return "sm_100a"

    monkeypatch.setattr(cake_kcp, "_HANDOFF_KEYS", {})
    monkeypatch.setattr(cake_kcp, "_arch_of", arch_of)
    monkeypatch.setattr(cake_kcp.cake_jit, "sm_count", lambda index: 148)
    monkeypatch.setattr(cake_kcp, "_handoff_toolkit", lambda arch: "13.4")
    for _ in range(3):
        key = cake_kcp._handoff_key(torch.device("cuda", 0))
        assert key == ("sm_100a", 148, "13.4")
    assert calls == [0]
    for _ in range(2):
        with pytest.raises(RuntimeError, match="exported"):
            cake_kcp._handoff_key(torch.device("cuda", 1))
    assert calls == [0, 1, 1]


def test_host_values_kept_across_calls_are_exact() -> None:
    """The host values the calls keep instead of recomputing are the recomputed ones: the validated options of a call
    without keywords (one shared object, never changed by a call that needs other options), a table set's pool needs
    (a copy per call) and a plan's scratch table bytes."""
    shared = cake_kcp._options({})
    assert cake_kcp._options({}) is shared
    fresh = cake_kcp._validate_options({})
    assert fresh is not shared
    assert dataclasses.asdict(fresh) == dataclasses.asdict(shared)
    other = cake_kcp._handle_options(shared, "states", None)
    assert other is not shared and other.save_transfer is False
    assert dataclasses.asdict(shared) == dataclasses.asdict(fresh)
    assert cake_kcp._options({"graph": "off"}) is not shared
    with pytest.raises(TypeError, match="unexpected keyword"):
        cake_kcp._options({"no_such_option": 1})
    cu = torch.tensor([0, 1000, 1700, 3000], dtype=torch.int64)
    part = cake_kcp_plan.partition_kcp(cu, 2, layout="zigzag_per_document")
    ctx = cake_kcp_plan.build_context_kcp(part, None, 0, "cpu", 4, sm_count=148)
    tb = cake_kcp.WorkspaceKcp(local=False).tables(ctx, 4)
    for backward in (True, False):
        lazy = cake_kcp_plan.LAZY_POOLS + (
            () if backward else cake_kcp_plan.FWD_LAZY_POOLS
        )
        want = {n: s for n, s in tb.pool_sizes(backward).items() if n not in lazy}
        for pair in cake_kcp_plan.SHARED_POOLS:
            top = max((want[m] for m in pair if m in want), default=0)
            want.update((m, top) for m in pair if m in want)
        first = cake_kcp.WorkspaceKcp._pool_need(tb, backward)
        assert first == want
        first[next(iter(first))] = -1  # a caller's change stays its own
        again = cake_kcp.WorkspaceKcp._pool_need(tb, backward)
        assert again == want and again is not first
    for S, N, H in ((3000, 3, 4), (1300, 2, 64)):
        plan = cake_kcp.cake_backend.plan(S, N, H, 148, "sm_100a")
        want_bytes = cake_kcp_plan.local_scratch_table_bytes(plan)
        for _ in range(2):
            assert cake_kcp._scratch_bytes(plan) == want_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_per_call_workspaces_share_one_identity_tile() -> None:
    """The per-call workspace of a ``workspace=None`` call binds its device's shared identity tile as its ``"eye"``
    pool (the same storage call after call and on another stream, counted in ``bytes`` as a tile of its own); a
    workspace a caller holds fills a tile of its own; inside a CUDA-graph capture no shared tile is handed out and a
    per-call workspace grown there fills a tile of its own (an identity after a replay); another device has a tile of
    its own."""
    device = torch.device("cuda", torch.cuda.current_device())
    eye = torch.eye(cake_kcp.D, dtype=torch.float32, device=device)

    def grown(shared: bool):
        ws = cake_kcp.WorkspaceKcp(local=False)
        ws._shared_eye = shared
        assert ws._grow("eye", cake_kcp.TILE, device)
        return ws

    a, b = grown(True), grown(True)
    side = torch.cuda.Stream(device)
    with torch.cuda.stream(side):
        c = grown(True)
    own = grown(False)
    tiles = [w.pool("eye") for w in (a, b, c, own)]
    torch.cuda.synchronize(device)
    assert tiles[0].data_ptr() == tiles[1].data_ptr() == tiles[2].data_ptr()
    assert tiles[3].data_ptr() != tiles[0].data_ptr()
    for w, t in zip((a, b, c, own), tiles, strict=True):
        assert torch.equal(t.view(cake_kcp.D, cake_kcp.D), eye)
        assert w.bytes == cake_kcp.TILE * 4 and w.pool_bytes["eye"] == cake_kcp.TILE * 4
    graph = torch.cuda.CUDAGraph()
    out = torch.zeros(1, device=device)
    with torch.cuda.graph(graph):
        inside = cake_kcp._identity_tile(device, torch.float32)
        captured = grown(True)
        out.add_(1.0)
    assert inside is None
    graph.replay()
    torch.cuda.synchronize(device)
    tile = captured.pool("eye")
    assert tile.data_ptr() != tiles[0].data_ptr()
    assert torch.equal(tile.view(cake_kcp.D, cake_kcp.D), eye)
    assert captured.bytes == cake_kcp.TILE * 4
    if torch.cuda.device_count() > 1:
        other = torch.device("cuda", (device.index + 1) % torch.cuda.device_count())
        with torch.cuda.device(device):  # the current device stays the first one
            ws = cake_kcp.WorkspaceKcp(local=False)
            ws._shared_eye = True
            assert ws._grow("eye", cake_kcp.TILE, other)
            far = ws.pool("eye")
            assert far is cake_kcp._identity_tile(other, torch.float32)
        torch.cuda.synchronize(other)
        assert far.device == other and far.data_ptr() != tiles[0].data_ptr()
        assert torch.equal(far.view(cake_kcp.D, cake_kcp.D).cpu(), eye.cpu())


def test_handoff_byte_queries_and_counters_are_public() -> None:
    """The package root and the facade export ``kda_handoff_bytes_kcp`` and ``kcp_handoff_stats``; the facade's
    ``kda_saved_bytes_kcp`` takes ``checkpoint_handoff`` (default ``False``, the light set), forwards it under
    ``deterministic=False`` and ignores it under ``deterministic=True`` (the deterministic package's figure); the
    facade's ``kda_handoff_bytes_kcp`` is the package's under ``deterministic=False`` and ``0`` under ``True``.  The
    byte queries' ``checkpoint_handoff`` is the decision (a ``bool``): the entry's modes raise ``TypeError`` (in both
    modes of the facade), as does a non-bool ``deterministic`` of ``kda_handoff_bytes_kcp``."""
    import importlib
    import inspect

    package = importlib.import_module("flashinfer.experimental.cake_kda_train_kcp")
    assert package.kda_handoff_bytes_kcp is cake_kcp_plan.kda_handoff_bytes_kcp
    assert package.kcp_handoff_stats is cake_kcp.kcp_handoff_stats
    for name in ("kda_handoff_bytes_kcp", "kcp_handoff_stats"):
        assert name in package.__all__ and name in kcp_api.__all__, name
    p = inspect.signature(kcp_api.kda_saved_bytes_kcp).parameters["checkpoint_handoff"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is False
    p = inspect.signature(kcp_api.kda_handoff_bytes_kcp).parameters["deterministic"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is True
    stats = kcp_api.kcp_handoff_stats()
    assert stats == cake_kcp.kcp_handoff_stats() and stats is not cake_kcp.HANDOFF_STATS
    assert set(stats) == set(cake_kcp.HANDOFF_STATS)
    det_plan = importlib.import_module(
        "flashinfer.experimental.cake_kda_det_train_kcp.cake_kcp_plan"
    )
    cu = [0, 1000, 1700, 3000]
    for W in (1, 2, 4):
        nd_part = kcp_api.partition_kcp(
            cu, W, "zigzag_per_document", deterministic=False
        )
        det_part = kcp_api.partition_kcp(cu, W, "zigzag_per_document")
        for rank in range(W):
            nd_ctx = kcp_api.build_context_kcp(
                nd_part, None, rank, "cpu", sm_count=148, deterministic=False
            )
            det_ctx = kcp_api.build_context_kcp(
                det_part, None, rank, "cpu", sm_count=148
            )
            for H in (4, 64):
                hb = cake_kcp_plan.kda_handoff_bytes_kcp(nd_ctx, H=H)
                assert (
                    kcp_api.kda_handoff_bytes_kcp(nd_ctx, H=H, deterministic=False)
                    == hb
                )
                assert kcp_api.kda_handoff_bytes_kcp(det_ctx, H=H) == 0
                for handoff in (False, True):
                    assert kcp_api.kda_saved_bytes_kcp(
                        nd_ctx, H=H, checkpoint_handoff=handoff, deterministic=False
                    ) == cake_kcp_plan.kda_saved_bytes_kcp(
                        nd_ctx, H=H, save_transfer=False, checkpoint_handoff=handoff
                    )
                    assert kcp_api.kda_saved_bytes_kcp(
                        det_ctx, H=H, checkpoint_handoff=handoff
                    ) == det_plan.kda_det_saved_bytes_kcp(
                        det_ctx, H=H, save_transfer=False
                    )
                assert kcp_api.kda_saved_bytes_kcp(
                    nd_ctx, H=H, checkpoint_handoff=False, deterministic=False
                ) == kcp_api.kda_saved_bytes_kcp(det_ctx, H=H)
            for bad in ("auto", "on", "off", 1, None):
                with pytest.raises(TypeError, match="checkpoint_handoff"):
                    cake_kcp_plan.kda_saved_bytes_kcp(nd_ctx, checkpoint_handoff=bad)
                for det, ctx in ((False, nd_ctx), (True, det_ctx)):
                    with pytest.raises(TypeError, match="checkpoint_handoff"):
                        kcp_api.kda_saved_bytes_kcp(
                            ctx, checkpoint_handoff=bad, deterministic=det
                        )
            with pytest.raises(TypeError, match="deterministic"):
                kcp_api.kda_handoff_bytes_kcp(nd_ctx, deterministic=1)


def test_facade_checkpoint_handoff_routing(monkeypatch) -> None:
    """The facade's ``chunk_kda_kcp`` refuses a mode outside ``auto`` / ``on`` / ``off`` under both values of
    ``deterministic`` before any work, forwards the mode to this package under ``deterministic=False`` and never
    passes it to the deterministic package."""
    import importlib

    det_kcp = importlib.import_module(
        "flashinfer.experimental.cake_kda_det_train_kcp.cake_kcp"
    )
    seen: list = []

    def recorder(tag):
        def fn(*args, **kwargs):
            seen.append((tag, kwargs))
            return tag

        return fn

    monkeypatch.setattr(cake_kcp, "chunk_kda_kcp", recorder("nd"))
    monkeypatch.setattr(det_kcp, "chunk_kda_det_kcp", recorder("det"))
    t = torch.zeros(1)
    for det in (True, False):
        for bad in ("yes", None, True, "ON"):
            with pytest.raises(ValueError, match="checkpoint_handoff"):
                kcp_api.chunk_kda_kcp(
                    t, t, t, t, t, object(), checkpoint_handoff=bad, deterministic=det
                )
    assert seen == []
    for mode in ("auto", "on", "off"):
        assert (
            kcp_api.chunk_kda_kcp(
                t, t, t, t, t, object(), checkpoint_handoff=mode, deterministic=False
            )
            == "nd"
        )
        assert seen[-1][1]["checkpoint_handoff"] == mode
        assert (
            kcp_api.chunk_kda_kcp(
                t, t, t, t, t, object(), checkpoint_handoff=mode, deterministic=True
            )
            == "det"
        )
        assert "checkpoint_handoff" not in seen[-1][1]


def test_saved_set_tape_members_round_trip() -> None:
    """The tape set travels as one more flag and five more tensors after the optional members; partial sets and
    flag lists of another length are refused."""
    meta = dict(
        version=cake_kcp.KCP_SAVED_VERSION,
        T_r=8,
        H=4,
        n_frag=1,
        P=1,
        Hg=4,
        scale=0.5,
        save_mode="light",
        save_transfer=True,
        offset_class="fla",
        exchange="a2a",
        precision_mode=0,
        transfer_walk="bf16x3_hw64",
    )
    s_in, c1 = (torch.zeros(n) for n in (3, 5))
    tape = {n: torch.zeros(i + 1) for i, n in enumerate(cake_kcp.SavedKcp._TAPE)}
    light = cake_kcp.SavedKcp(S_in=s_in, c1_recv=c1, M_seg=None, ctx=None, **meta)
    kept = cake_kcp.SavedKcp(
        S_in=s_in, c1_recv=c1, M_seg=None, ctx=None, **tape, **meta
    )
    assert light.tape is None and light.flags() == (True, False, False)
    assert kept.flags() == (True, False, False, True)
    assert [t.numel() for t in kept.tensors()] == [3, 5, 1, 2, 3, 4, 5]
    assert kept.bytes - light.bytes == sum(t.numel() * 4 for t in tape.values())
    for saved in (light, kept):
        back = cake_kcp.SavedKcp.from_tensors(
            saved.tensors(), saved.flags(), saved.meta(), None
        )
        assert back.flags() == saved.flags() and back.meta() == saved.meta()
        assert (back.tape is None) == (saved.tape is None)
        if saved.tape is not None:
            assert all(back.tape[n] is saved.tape[n] for n in saved.tape)
    with pytest.raises(ValueError, match="flags"):
        cake_kcp.SavedKcp.from_tensors(light.tensors(), (True,), light.meta(), None)


def test_handoff_bytes_query() -> None:
    """``kda_handoff_bytes_kcp`` = ``NC_pad * H * 32768 + R * H * 768``; the hand-off's saved set adds it and drops
    the carry plane."""
    cu = torch.tensor([0, 1000, 1700, 3000], dtype=torch.int64)
    for W in (1, 2, 4):
        part = cake_kcp_plan.partition_kcp(cu, W, layout="zigzag_per_document")
        for rank in range(W):
            ctx = cake_kcp_plan.build_context_kcp(
                part, None, rank, "cpu", 4, sm_count=148
            )
            for H in (4, 64):
                hb = cake_kcp_plan.kda_handoff_bytes_kcp(ctx, H=H)
                assert hb == ctx.NC_pad * H * 32768 + ctx.R * H * 768
                assert (
                    cake_kcp_plan.kda_saved_bytes_kcp(ctx, H=H, checkpoint_handoff=True)
                    == cake_kcp_plan.kda_saved_bytes_kcp(ctx, H=H, saved_carry=False)
                    + hb
                )
