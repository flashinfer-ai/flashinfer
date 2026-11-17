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
(``cake_kda_nondet_kcp(..., checkpoint_handoff=)``): under ``torch.utils.checkpoint`` the forward keeps its tape set and
the backward reads it instead of re-running the intra program, WY and the K1 tape walk.

Multi-GPU rows (``HANDOFF_ROWS``: the checkpoint rows ``CKPT_ROWS`` of ``test_cake_kda_nondet_train_kcp_dist.py``, the
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
the facade ``flashinfer.cake_kda_nondet_train_kcp`` forwards the mode to this package; a checkpointed forward with the
hand-off retains nothing.  A check whose light path does not
run under its dispatch mode or policy has nothing to compare: the row lists it under ``not_run``.
Mode rows (``HANDOFF_MODE_ROWS``: raw gate mode, the in-kernel Q / K L2 norm and both, at W = 2 and W = 4, on the
segmented and the serial route and with a rank without tokens): every output and gradient, in raw gate mode the
``A_log`` / ``dt_bias`` partials of each rank and their cross-rank sums too, bitwise ``"off"`` and the direct pass in
the same forms (non-reentrant, reentrant, both handle tiers, a per-call workspace, NaN tape set and poisoned free
blocks, a poisoned caller arena with and without a handle, mixed ranks, ring and overlap, the previous host path
(``host_trim`` 0), ``save_transfer=False`` without a handle, two backwards, the facade),
the hand-off engaged with the launches it skips (with the norm the separate normalization exactly where the intra
program normalized first), the saved bytes, the fallback form, ``"auto"`` under the mode's verified set and nothing
retained (:func:`test_checkpoint_handoff_mode_rows`).
Rows of ``saved_rows`` (``SAVED_ROWS_GPU_ROWS``: W = 1, 2, 4, H 12 to 48, both gate modes, the norm off and on): the
``"intra"`` / ``"wy"`` rows sets of a plain node and of the hand-off's rows form within the light path's FP64 error
budget, with their launches, bytes, counters, refusals, poisons, handles and graph replay
(:func:`test_saved_rows_gpu_rows`).
Single-process rows: the static rule, the admission table (also per mode), the L2 norm's refusal without its direct
normalization record, the phase probe, the once-per-call decision (a refusal by
the call's own arguments records nothing in either pass), the device key, the host values kept across calls (default
options, pool needs, scratch table bytes), the per-call workspaces' shared identity tile, the saved set's tape members
and the byte queries of the package and the facade.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import hashlib
import itertools
import json
import math
import os
import tempfile
import traceback
import types

import pytest
import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode

import flashinfer.cake_kda_nondet_train_kcp as kcp_api
from tests.experimental.test_cake_kda_nondet_train_kcp_dist import (
    ATOL,
    CKPT_ROWS,
    GRAPH_EVENTS,
    RTOL,
    SCALE,
    _bound_rank_threads,
    _bwd_sequence_first_main,
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
from tests.experimental.test_helpers.cake_kda_det_train_reference import (
    LOWER_BOUND,
    kda_exact_raw_with_grads,
    kda_exact_with_grads,
    make_inputs,
    make_raw_inputs,
)
from tests.experimental.test_helpers.cake_kda_det_train_reference_l2n import (
    kda_exact_l2n_with_grads,
    kda_exact_raw_l2n_with_grads,
)

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
        ("sm_100a", 152, "13.4"),
    }
)
GB200 = ("sm_100a", 152, "13.4")
# the devices verified in raw gate mode and with the in-kernel Q / K L2 norm (cake_kcp_plan.KCP_HANDOFF_VERIFIED_MODES):
# GB300, R200 and GB200, where the mode rows of this file and the rule's rows ran
_MODE_KEYS = frozenset({("sm_103a", 152, "13.4"), ("sm_107a", 212, "13.4"), GB200})
VERIFIED_MODE_KEYS = {
    ("raw", False): _MODE_KEYS,
    ("g", True): _MODE_KEYS,
    ("raw", True): _MODE_KEYS,
}
AUTO_HEADS = frozenset({4, 8, 16, 32, 64})
MIN_HEAD_TOKENS = 393216
# the keys with their own, higher work bound (cake_kcp_plan.KCP_HANDOFF_MIN_HEAD_TOKENS_BY_KEY)
MIN_HEAD_TOKENS_BY_KEY = {GB200: 786432}
# the hand-off rows: the checkpoint rows (H1, H4, H16, H64) and the H64 row's packing at the other head counts of
# AUTO_HEADS
HANDOFF_ROWS = CKPT_ROWS + [
    ((2048, 5, 3000, 1), heads, 4, "zigzag_per_document") for heads in (8, 32)
]
# the rows of the raw gate mode, the in-kernel Q / K L2 norm and both (gate mode, use_qk_l2norm_in_kernel), at W = 2 and
# W = 4: H4 runs the segmented route (the norm's fused form normalizes in the replay's intra program first), H64 the
# serial route (in the prep first), H16 the segmented route on sm_103a / sm_107a and the serial one on sm_100a; the
# ragged row has a rank without tokens
HANDOFF_MODE_KINDS = (("raw", False), ("g", True), ("raw", True))
HANDOFF_MODE_SHAPES = [
    ((2048, 5, 3000, 1), 4, 2, "zigzag_per_document"),
    ((2048, 5, 3000, 1), 64, 2, "zigzag_per_document"),
    ((2048, 5, 3000, 1), 4, 4, "zigzag_per_document"),
    ((2048, 5, 3000, 1), 64, 4, "zigzag_per_document"),
    ((100, 640, 77, 1000, 129), 16, 4, "zigzag_per_document"),
    ((1200 + 900 + 900,), 4, 4, "ragged:1200,0,900,900"),
]
HANDOFF_MODE_ROWS = [
    (*shape, gate, l2n)
    for gate, l2n in HANDOFF_MODE_KINDS
    for shape in HANDOFF_MODE_SHAPES
]

# the rows sets of ``saved_rows`` (cake_kcp_plan.KCP_SAVED_ROWS) and their bytes per internal token-head
SAVED_ROWS_SETS = {
    "intra": (("aqk_int", "akk_int"), 256),
    "wy": (("aqk_int", "akk_int", "w", "u"), 768),
}
# the GPU rows of saved_rows (lengths, H, W, layout, gate mode, use_qk_l2norm_in_kernel): W = 1 (in the pytest
# process), 2 and 4; H 12, 16, 32 and 48 (12 and 48 are outside the hand-off's admitted head counts); every gate mode
# with the norm off and on at two world sizes; the ragged row has a rank without tokens
SAVED_ROWS_GPU_ROWS = [
    ((100, 640, 77, 1000, 129), 12, 1, "contiguous", "g", False),
    ((700, 1, 1300, 999), 48, 1, "contiguous", "raw", True),
    ((100, 640, 77, 1000, 129), 16, 2, "zigzag_per_document", "g", True),
    ((3000,), 32, 2, "zigzag_per_document", "raw", False),
    ((2048, 5, 3000, 1), 12, 4, "zigzag_per_document", "raw", True),
    ((100, 640, 77, 1000, 129), 16, 4, "zigzag_per_document", "g", False),
    ((3000,), 48, 4, "zigzag", "g", True),
    ((1200 + 900 + 900,), 32, 4, "ragged:1200,0,900,900", "raw", False),
]
# a saved_rows run against the light path of the same row: its FP64 error (rel_rms and max_abs, every output and
# gradient) at most ACC_GATE times the light path's, as the raw gate and L2N rows bound KCP against the single rank
ACC_GATE = 1.10


def _handoff_delta(
    before: dict,
    ctx,
    H: int,
    gate: str,
    l2n: bool,
    T_r: int,
    *,
    auto_calls: int,
    on_calls: int = 1,
) -> bool:
    """Whether the hand-off counters moved from ``before`` (a :func:`cake_kcp.cake_kda_nondet_train_kcp_handoff_stats`
    copy) exactly as ``auto_calls`` checkpointed ``"auto"`` calls and ``on_calls`` checkpointed ``"on"`` calls of this
    rank's row in the mode (``gate``, ``l2n``) decide: each ``"auto"`` call refused once by the mode's rule
    (:func:`cake_kcp_plan.kcp_handoff_auto_refusal` of this device's key) or engaged, an engaged call kept by both
    passes and read by the backward on a rank with tokens."""
    key = cake_kcp._handoff_key(torch.device(ctx.device))
    reason = cake_kcp_plan.kcp_handoff_auto_refusal(
        key, H, ctx.partition.total, ctx.world_size, gate=gate, l2n=l2n
    )
    want: dict[str, int] = {}
    engaged_calls = on_calls + (auto_calls if reason is None else 0)
    if reason is not None and auto_calls:
        want["refused_" + reason] = auto_calls
    if T_r > 0 and engaged_calls:
        want["save"] = 2 * engaged_calls
        want["bound"] = engaged_calls
    after = cake_kcp.cake_kda_nondet_train_kcp_handoff_stats()
    delta = {n: after[n] - before[n] for n in after if after[n] != before[n]}
    return delta == want


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
    ctx = cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(
        part, group, rank, device
    )
    glob = make_inputs(lengths, H, device)
    ops = _local_slices(part, rank, cu, glob)
    q, k, v, g, beta, dout = (ops[n] for n in ("q", "k", "v", "g", "beta", "dout"))
    ws = cake_kcp.CakeKDANondetKcpWorkspace()
    ws_lean = cake_kcp.CakeKDANondetKcpWorkspace(local_arena="lean")
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
            y = cake_kcp.cake_kda_nondet_kcp(
                q_,
                k_,
                v_,
                g_,
                b_,
                ctx,
                workspace=workspace,
                checkpoint_handoff=mode,
                saved_rows="tape",  # the tape set (saved_rows="auto" keeps the wy rows form)
                **kw,
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
    want_bytes = (
        cake_kcp.cake_kda_nondet_train_kcp_handoff_bytes(ctx, H=H, saved_rows="tape")
        if T_r > 0
        else None
    )
    flags["on_bytes"] = (
        recs_on["fwd"].get("checkpoint_handoff_bytes") == want_bytes
        and recs_on["bwd"].get("checkpoint_handoff_bytes") == want_bytes
    )
    # the saved set: the hand-off forward's minus the light forward's is the byte query's difference (the tape set
    # in, the carry plane out); the absolute figures are recorded (the query also counts the int32 tables)
    st = recs_on["fwd"]["save_transfer"]
    query = {
        flag: cake_kcp_plan.cake_kda_nondet_train_kcp_saved_bytes(
            ctx,
            H=H,
            with_conv=False,
            save_transfer=st,
            checkpoint_handoff=flag,
            saved_rows="tape",
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
        _, recs_h_off = run(
            "off", handle=cake_kcp.CakeKDANondetKcpRecomputeHandle(), keep=keep
        )
        res_h, recs_h = run(
            "on", handle=cake_kcp.CakeKDANondetKcpRecomputeHandle(), keep=keep
        )
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
        cake_kcp_plan.KCP_HANDOFF_MIN_HEAD_TOKENS_BY_KEY,
    )
    kept = {"save": 2, "bound": 1} if T_r > 0 else {}

    @contextlib.contextmanager
    def auto_rule(min_head_tokens: int, verified, by_key):
        """``"auto"``'s admission constants for the block; yields the rule's reason for this row (``None``:
        admitted)."""
        cake_kcp_plan.KCP_HANDOFF_MIN_HEAD_TOKENS = min_head_tokens
        cake_kcp_plan.KCP_HANDOFF_VERIFIED = verified
        cake_kcp_plan.KCP_HANDOFF_MIN_HEAD_TOKENS_BY_KEY = by_key
        try:
            yield cake_kcp_plan.kcp_handoff_auto_refusal(key, H, total, W)
        finally:
            (
                cake_kcp_plan.KCP_HANDOFF_MIN_HEAD_TOKENS,
                cake_kcp_plan.KCP_HANDOFF_VERIFIED,
                cake_kcp_plan.KCP_HANDOFF_MIN_HEAD_TOKENS_BY_KEY,
            ) = shipped

    reasons = {}
    for case, rule in (
        ("shipped", shipped),
        ("bound0", (0, shipped[1], {})),
        ("bound0_unverified", (0, frozenset(), {})),
    ):
        with auto_rule(*rule) as reason:
            before = cake_kcp.cake_kda_nondet_train_kcp_handoff_stats()
            res_a, recs_a = run("auto")
            after = cake_kcp.cake_kda_nondet_train_kcp_handoff_stats()
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

    # the facade forwards the mode to this package (engaged as the package decides)
    def run_facade(mode: str):
        lv = leaves()
        wl = torch.ones_like(dout).requires_grad_(True)

        def top(q_, k_, v_, g_, b_, w_):
            y = kcp_api.cake_kda_nondet_kcp(
                q_,
                k_,
                v_,
                g_,
                b_,
                ctx,
                workspace=ws,
                checkpoint_handoff=mode,
                saved_rows="tape",
            )
            return y * w_

        before = kcp_api.cake_kda_nondet_train_kcp_handoff_stats()
        out = checkpoint(top, *lv, wl, use_reentrant=False)
        out.backward(dout)
        after = kcp_api.cake_kda_nondet_train_kcp_handoff_stats()
        res = {"out": out.detach(), "dw": wl.grad}
        res.update(
            {
                f"d{n}": lf.grad
                for n, lf in zip(("q", "k", "v", "g", "beta"), lv, strict=True)
            }
        )
        return res, {k: after[k] - before[k] for k in after if after[k] != before[k]}

    for mode in ("auto", "on", "off"):
        # the work bound at 0: "auto" engages as on a production-size batch of a verified device
        with auto_rule(0, shipped[1], {}) as reason:
            res_f, delta = run_facade(mode)
        want = kept if (mode == "on" or (mode == "auto" and reason is None)) else {}
        flags[f"facade_{mode}_bitwise"] = _same(res_f, base)
        flags[f"facade_{mode}_engaged"] = {
            k: n for k, n in delta.items() if k in ("save", "bound")
        } == want
    # #7039: a checkpointed forward with the hand-off retains nothing (its original forward's set goes at return),
    # after one warm-up step of the same form: per call, on this row's workspace and on a fresh caller-held workspace
    # (its first backward binds a new backward arena that no intra program of the hand-off backward touches)
    retained = {}
    for form, wsp in (
        ("per_call", None),
        ("caller_ws", ws),
        ("fresh_ws", cake_kcp.CakeKDANondetKcpWorkspace()),
    ):

        def fn0(*a, _w=wsp):
            return cake_kcp.cake_kda_nondet_kcp(
                *a, ctx, workspace=_w, checkpoint_handoff="on", saved_rows="tape"
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


def _mode_row_id(row) -> str:
    lengths, H, W, layout, gate, l2n = row
    return _row_id((lengths, H, W, layout)) + f"-{gate}" + ("-l2n" if l2n else "")


def _mode_inputs(lengths, H: int, device, gate: str, l2n: bool) -> dict:
    """The row's operands: the in-kernel norm's raw q / k rows (zero and subnormal rows among them) with ``l2n``,
    the raw gate with ``A_log`` / ``dt_bias`` in raw gate mode."""
    if l2n:
        from tests.experimental.test_cake_kda_nondet_train_kcp_l2n import _l2n_inputs

        return _l2n_inputs(lengths, H, device, raw_gate=gate == "raw")
    return make_raw_inputs(lengths, H, device)


def _run_rank_handoff_mode(
    rank: int, W: int, row, rdzv: str | None, result_dir: str | None, local_id: int
) -> dict:
    """One rank of a hand-off row in raw gate mode, with the in-kernel Q / K L2 norm or both (see
    :func:`test_checkpoint_handoff_mode_rows`)."""
    from torch.utils.checkpoint import checkpoint

    lengths, H, _w, layout, gate, l2n = row
    raw = gate == "raw"
    device = torch.device("cuda", local_id % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        _init_group(rank, W, device, rdzv)
    group = dist.group.WORLD
    part, cu = _partition(lengths, W, layout)
    ctx = cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(
        part, group, rank, device
    )
    glob = _mode_inputs(lengths, H, device, gate, l2n)
    names = ("q", "k", "v", "g", "beta", "dout")
    ops = _local_slices(part, rank, cu, {n: glob[n] for n in names})
    q, k, v, g, beta, dout = (ops[n] for n in names)
    params = (glob["A_log"], glob["dt_bias"]) if raw else ()
    mode_kw = {"use_qk_l2norm_in_kernel": l2n}
    ws_kw = {"use_gate_in_kernel": raw, "use_qk_l2norm_in_kernel": l2n}
    ws = cake_kcp.CakeKDANondetKcpWorkspace(**ws_kw)
    T_r = int(q.shape[0])
    index = device.index if device.index is not None else torch.cuda.current_device()
    arch = cake_kcp.cake_jit.device_arch(index)
    sms = int(cake_kcp.cake_jit.sm_count(index))
    key = (arch, sms, cake_kcp._handoff_toolkit(arch))
    # the fused form's backward normalizes in its intra program first on this rank's plan: the hand-off then runs the
    # separate normalization from the saved rstd before its prep (cake_kcp._handoff_l2n_rows)
    intra_first = bool(
        l2n
        and T_r > 0
        and cake_kcp_plan.qk_l2norm_intra_first(
            cake_kcp.cake_backend.plan(
                T_r, int(ctx.n_frag), H, sms, arch, gate=gate, l2n="fused"
            )
        )
    )
    info: dict[str, object] = {
        "T_r": T_r,
        "n_frag": int(ctx.n_frag),
        "P": int(ctx.heads(H).P),
        "key": list(key),
        "intra_first": intra_first,
        "torch": torch.__version__,
    }
    flags: dict[str, bool] = {}
    grad_names = ("dq", "dk", "dv", "dg", "dbeta") + (
        ("dA_log", "ddt_bias") if raw else ()
    )

    def leaves():
        return [
            x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta, *params)
        ]

    def gate_kw(lv) -> dict:
        if not raw:
            return {}
        return {
            "use_gate_in_kernel": True,
            "A_log": lv[5],
            "dt_bias": lv[6],
            "lower_bound": LOWER_BOUND,
        }

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
        twice: bool = False,
        entry=None,
        **opts,
    ):
        """One checkpointed (or direct) step of ``out * w``: every output and gradient (in raw gate mode the
        parameter partials of this rank and their cross-rank sums) and the LAST_CALL records of the original forward,
        the recompute (None without one) and the backward."""
        if poison_free:
            _poison(device)
        lv = leaves()
        wl = torch.ones_like(dout).requires_grad_(True)
        kw = dict(opts)
        if handle is not None:
            kw["recompute_handle"] = handle
        if keep is not None:
            kw["checkpoint_keep"] = keep
        call = entry or cake_kcp.cake_kda_nondet_kcp

        def top(q_, k_, v_, g_, b_, w_, *p_):
            y = call(
                q_,
                k_,
                v_,
                g_,
                b_,
                ctx,
                workspace=workspace,
                checkpoint_handoff=mode,
                saved_rows="tape",  # the tape set (saved_rows="auto" keeps the wy rows form)
                **gate_kw([q_, k_, v_, g_, b_, *p_]),
                **mode_kw,
                **kw,
            )
            return y * w_

        cake_kcp.LAST_CALL.pop("fwd", None)
        cake_kcp.LAST_CALL.pop("bwd", None)
        prev = cake_kcp._HANDOFF_POISON
        cake_kcp._HANDOFF_POISON = bool(poison_tape)
        args = (*lv[:5], wl, *lv[5:])
        try:
            if not ckpt:
                out = top(*args)
            elif context_fn is not None:
                out = checkpoint(top, *args, use_reentrant=False, context_fn=context_fn)
            else:
                out = checkpoint(top, *args, use_reentrant=reentrant)
            rec_orig = dict(cake_kcp.LAST_CALL["fwd"])
            if ckpt:
                cake_kcp.LAST_CALL.pop("fwd")
            for i in range(2 if twice else 1):
                out.backward(dout, retain_graph=twice and i == 0)
        finally:
            cake_kcp._HANDOFF_POISON = prev
        rec_fwd = cake_kcp.LAST_CALL.get("fwd")
        rec_bwd = dict(cake_kcp.LAST_CALL["bwd"])
        res = {"out": out.detach(), "dw": wl.grad}
        res.update({n: lf.grad for n, lf in zip(grad_names, lv, strict=True)})
        if raw and not twice:
            # the cross-rank sums of the partials (a collective: every rank of every run)
            sums = cake_kcp.cake_kda_nondet_train_kcp_reduce_gate_param_grads(
                ctx, res["dA_log"], res["ddt_bias"]
            )
            res["dA_log_sum"], res["ddt_bias_sum"] = sums
        return res, {"orig": rec_orig, "fwd": rec_fwd, "bwd": rec_bwd}

    def engaged(recs) -> bool:
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
        seq = rec["sequence"]
        if rec.get("checkpoint_handoff") != "save":
            return list(seq)
        if seq.count("fwd_o") != 1 or "kcp_fwdh_o" in seq:
            return ["not the tape form", *seq]
        return [
            ("kcp_fwdh_o" if x == "kcp_fwdh_tape" else x) for x in seq if x != "fwd_o"
        ]

    def skips(recs_on, recs_off) -> bool:
        """The forwards' launches in the light form, the backward's collectives the light path's; with tokens no
        launch of SKIPPED in the hand-off's backward, and the separate normalization exactly where the skipped intra
        program normalized (the fused form's intra-first plans)."""
        on_b, off_b = recs_on["bwd"]["sequence"], recs_off["bwd"]["sequence"]
        ok = light_form(recs_on["orig"]) == recs_off["orig"]["sequence"]
        ok = ok and light_form(recs_on["fwd"]) == recs_off["fwd"]["sequence"]
        ok = ok and _collectives(on_b) == _collectives(off_b)
        ok = ok and recs_on["bwd"]["n_collectives"] == recs_off["bwd"]["n_collectives"]
        if T_r > 0:
            ok = ok and not any(x in SKIPPED for x in on_b)
            ok = ok and all(x in off_b for x in SKIPPED)
            ok = ok and on_b.count("qk_apply") == int(intra_first)
            ok = ok and "qk_apply" not in off_b
        return ok

    direct, recs_direct = run("off", ckpt=False)
    base, recs_base = run("off")
    flags["off_light"] = light(recs_base) and light(recs_direct)
    flags["off_equals_direct"] = _same(base, direct)
    # the hand-off on, non-reentrant: every output, gradient, partial and sum bitwise; the launches skipped
    on_nr, recs_on = run("on")
    flags["on_nonreentrant_bitwise"] = _same(on_nr, base)
    flags["on_nonreentrant_engaged"] = engaged(recs_on)
    flags["on_launches"] = skips(recs_on, recs_base)
    want_bytes = (
        cake_kcp.cake_kda_nondet_train_kcp_handoff_bytes(ctx, H=H, saved_rows="tape")
        if T_r > 0
        else None
    )
    flags["on_bytes"] = (
        recs_on["fwd"].get("checkpoint_handoff_bytes") == want_bytes
        and recs_on["bwd"].get("checkpoint_handoff_bytes") == want_bytes
    )
    st = recs_on["fwd"]["save_transfer"]
    query = {
        flag: cake_kcp_plan.cake_kda_nondet_train_kcp_saved_bytes(
            ctx,
            H=H,
            with_conv=False,
            save_transfer=st,
            checkpoint_handoff=flag,
            saved_rows="tape",
            **ws_kw,
        )
        for flag in (False, True)
    }
    got = {"off": recs_base["fwd"]["saved_bytes"], "on": recs_on["fwd"]["saved_bytes"]}
    info["saved_bytes"] = {"got": got, "query": query}
    flags["on_saved_bytes"] = got["on"] - got["off"] == (
        query[True] - query[False] if T_r > 0 else 0
    )
    # reentrant: the recompute decides and keeps the set
    on_r, recs_on_r = run("on", reentrant=True)
    flags["on_reentrant_bitwise"] = _same(on_r, base)
    flags["on_reentrant_engaged"] = (
        recs_on_r["fwd"] is not None
        and recs_on_r["fwd"].get("checkpoint_handoff") == ("save" if T_r > 0 else None)
        and recs_on_r["bwd"].get("checkpoint_handoff") == ("bound" if T_r > 0 else None)
        and recs_on_r["orig"].get("checkpoint_handoff") is None
    )
    # with a recompute handle, both tiers
    for keep in ("states", "transfer"):
        _, recs_h_off = run(
            "off", handle=cake_kcp.CakeKDANondetKcpRecomputeHandle(), keep=keep
        )
        res_h, recs_h = run(
            "on", handle=cake_kcp.CakeKDANondetKcpRecomputeHandle(), keep=keep
        )
        flags[f"handle_{keep}_bitwise"] = _same(res_h, base)
        flags[f"handle_{keep}_engaged"] = engaged(recs_h)
        flags[f"handle_{keep}_launches"] = skips(recs_h, recs_h_off)
    # a per-call workspace; a NaN-filled tape set from poisoned free blocks; the caller arena poisoned between the
    # recompute and the backward (a byte the hand-off's backward read from the recompute's arena would show)
    res_pc, recs_pc = run("on", workspace=None)
    flags["per_call_bitwise"] = _same(res_pc, base)
    flags["per_call_engaged"] = engaged(recs_pc)
    res_p, recs_p = run("on", workspace=None, poison_free=True, poison_tape=True)
    flags["poison_tape_free_bitwise"] = _same(res_p, base)
    flags["poison_tape_free_engaged"] = engaged(recs_p)
    arena_fn = lambda: (contextlib.nullcontext(), _ArenaPoison(ws))  # noqa: E731
    res_pa_off, _ = run("off", context_fn=arena_fn)
    res_pa_on, recs_pa = run("on", context_fn=arena_fn, poison_tape=True)
    flags["poison_arena_off_bitwise"] = _same(res_pa_off, base)
    flags["poison_arena_on_bitwise"] = _same(res_pa_on, base)
    flags["poison_arena_engaged"] = engaged(recs_pa)
    for keep in ("states", "transfer"):
        res_hp, recs_hp = run(
            "on",
            handle=cake_kcp.CakeKDANondetKcpRecomputeHandle(),
            keep=keep,
            context_fn=arena_fn,
            poison_tape=True,
        )
        flags[f"poison_arena_handle_{keep}_bitwise"] = _same(res_hp, base)
        flags[f"poison_arena_handle_{keep}_engaged"] = engaged(recs_hp)
    # ranks that decide differently
    mixed = "on" if rank % 2 == 0 else "off"
    res_mx, recs_mx = run(mixed)
    flags["mixed_ranks_bitwise"] = _same(res_mx, base)
    flags["mixed_ranks_engaged"] = engaged(recs_mx) if mixed == "on" else light(recs_mx)
    # the ring exchange (where the partition allows it) and the overlap schedule
    ring_ok = int(ctx.n_d_max) - 1 <= cake_kcp.RING_MAX_HOPS
    info["ring_eligible"] = ring_ok
    overlap = {"_kcp_test_knobs": {"schedule": "overlap"}}
    # also the previous host path (every launch, the separate normalization included, through the arena's
    # launcher) and save_transfer=False without a handle
    forms = {
        "overlap": overlap,
        "host_trim0": {"_kcp_test_knobs": {"host_trim": 0}},
        "save_transfer_off": {"save_transfer": False},
    }
    if ring_ok:
        forms["ring"] = {"exchange": "ring"}
    previous_knobs = cake_kcp_plan.enable_test_knobs(True)
    try:
        for form, opts in forms.items():
            res_f_off, recs_f_off = run("off", **opts)
            res_f_on, recs_f_on = run("on", **opts)
            flags[f"{form}_bitwise"] = _same(res_f_off, base) and _same(res_f_on, base)
            flags[f"{form}_engaged"] = engaged(recs_f_on) and light(recs_f_off)
            flags[f"{form}_launches"] = skips(recs_f_on, recs_f_off)
        ws_fb = None
        if l2n:
            # the fallback form (the workspace's test knob): the separate normalization of the light backward
            try:
                ws_fb = cake_kcp.CakeKDANondetKcpWorkspace(
                    **ws_kw, _qk_l2norm_form="fallback"
                )
            except NotImplementedError as exc:
                # the form's programs are not delivered: reported, not run
                info["fallback_pending"] = str(exc)
                # only the form's capability refusal (rank-independent) skips it, never another error
                flags["fallback_pending_is_capability"] = str(exc).startswith(
                    cake_kcp_plan.QK_L2NORM_PENDING
                )
            info["fallback_form"] = "pending" if ws_fb is None else "ran"
        if ws_fb is not None:
            res_fb_off, recs_fb_off = run("off", workspace=ws_fb)
            res_fb_on, recs_fb_on = run("on", workspace=ws_fb)
            flags["fallback_bitwise"] = _same(res_fb_on, res_fb_off)
            flags["fallback_engaged"] = engaged(recs_fb_on) and light(recs_fb_off)
            info["fallback_eq_fused"] = _same(res_fb_off, base)
            on_b = recs_fb_on["bwd"]["sequence"]
            flags["fallback_launches"] = T_r == 0 or (
                not any(x in SKIPPED for x in on_b)
                and on_b.count("qk_apply") == 1
                and _collectives(on_b) == _collectives(recs_fb_off["bwd"]["sequence"])
            )
    finally:
        cake_kcp_plan.enable_test_knobs(previous_knobs)
    # retain_graph: twice the gradients (the partials too)
    res_tw, recs_tw = run("on", twice=True)
    base_tw = {n: t for n, t in base.items() if not n.endswith("_sum")}
    flags["twice_bitwise"] = _same(
        res_tw, {n: t if n == "out" else t + t for n, t in base_tw.items()}
    )
    flags["twice_engaged"] = engaged(recs_tw)
    # "auto": the shipped rule, then the work bound at 0 with this device verified in the call's mode (engaged), with
    # the mode's set emptied while the activated gate's still lists the device ("unverified"), each decided once
    total = ctx.partition.total
    shipped = (
        cake_kcp_plan.KCP_HANDOFF_MIN_HEAD_TOKENS,
        cake_kcp_plan.KCP_HANDOFF_VERIFIED,
        cake_kcp_plan.KCP_HANDOFF_VERIFIED_MODES,
        cake_kcp_plan.KCP_HANDOFF_MIN_HEAD_TOKENS_BY_KEY,
    )
    kind = (gate, l2n)
    kept = {"save": 2, "bound": 1} if T_r > 0 else {}

    @contextlib.contextmanager
    def auto_rule(min_head_tokens: int, verified, modes, by_key):
        (
            cake_kcp_plan.KCP_HANDOFF_MIN_HEAD_TOKENS,
            cake_kcp_plan.KCP_HANDOFF_VERIFIED,
            cake_kcp_plan.KCP_HANDOFF_VERIFIED_MODES,
            cake_kcp_plan.KCP_HANDOFF_MIN_HEAD_TOKENS_BY_KEY,
        ) = (min_head_tokens, verified, modes, by_key)
        try:
            yield cake_kcp_plan.kcp_handoff_auto_refusal(
                key, H, total, W, gate=gate, l2n=l2n
            )
        finally:
            (
                cake_kcp_plan.KCP_HANDOFF_MIN_HEAD_TOKENS,
                cake_kcp_plan.KCP_HANDOFF_VERIFIED,
                cake_kcp_plan.KCP_HANDOFF_VERIFIED_MODES,
                cake_kcp_plan.KCP_HANDOFF_MIN_HEAD_TOKENS_BY_KEY,
            ) = shipped

    reasons = {}
    for case, rule in (
        ("shipped", shipped),
        (
            "bound0_verified",
            (0, shipped[1], {**shipped[2], kind: frozenset({key})}, {}),
        ),
        (
            "bound0_unverified",
            (0, frozenset({key}), {**shipped[2], kind: frozenset()}, {}),
        ),
    ):
        with auto_rule(*rule) as reason:
            before = cake_kcp.cake_kda_nondet_train_kcp_handoff_stats()
            res_a, recs_a = run("auto")
            after = cake_kcp.cake_kda_nondet_train_kcp_handoff_stats()
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
    flags["auto_rule_reasons"] = (
        reasons["bound0_verified"] is None
        and reasons["bound0_unverified"] == "unverified"
    )
    # the facade forwards the mode keywords and the hand-off mode
    for mode in ("on", "off"):
        res_f, recs_f = run(mode, entry=kcp_api.cake_kda_nondet_kcp)
        flags[f"facade_{mode}_bitwise"] = _same(res_f, base)
        flags[f"facade_{mode}_engaged"] = (
            engaged(recs_f) if mode == "on" else light(recs_f)
        )
    # #7039: a checkpointed forward with the hand-off retains nothing, after one warm-up step of the same form
    retained = {}
    for form, wsp in (("per_call", None), ("caller_ws", ws)):

        def fn0(*a, _w=wsp):
            return cake_kcp.cake_kda_nondet_kcp(
                *a[:5],
                ctx,
                workspace=_w,
                checkpoint_handoff="on",
                saved_rows="tape",
                **gate_kw(a),
                **mode_kw,
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
        "info_by_rank": info_list,
    }
    report["pass"] = all(report["flags"].values()) and all(
        set(f) == set(flag_list[0]) for f in flag_list
    )
    dist.barrier()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(
            os.path.join(result_dir, "handoff-" + _mode_row_id(row) + ".json"), "w"
        ) as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _spawn_entry_handoff_mode(
    rank: int, W: int, row, rdzv: str | None, result_dir: str
) -> None:
    try:
        _bound_rank_threads(W)
        report = _run_rank_handoff_mode(rank, W, row, rdzv, result_dir, rank)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("row", HANDOFF_MODE_ROWS, ids=_mode_row_id)
def test_checkpoint_handoff_mode_rows(row) -> None:
    """The hand-off in raw gate mode, with the in-kernel Q / K L2 norm and with both (``HANDOFF_MODE_ROWS``), on every
    rank of the row: every output and gradient -- the raw gate's, the raw ``q`` / ``k``'s, the ``A_log`` /
    ``dt_bias`` partials of each rank and their cross-rank sums (:func:`cake_kcp.cake_kda_nondet_train_kcp_reduce_gate_param_grads`)
    -- with the hand-off on is bitwise that with it off and that of the direct pass, non-reentrant and reentrant, with
    a recompute handle of both tiers, on a per-call workspace, from a NaN-filled tape set and poisoned free blocks, from
    a caller arena poisoned between the recompute and the backward (with and without a handle), with ranks that decide
    differently, with the ring exchange and the overlap schedule, on the previous host path (``host_trim`` 0), with
    ``save_transfer=False`` without a handle, over two backwards and through the facade; the
    hand-off engages (the original forward and the recompute keep the set, the backward reads it), the forwards run
    the tape form of the light launches, the backward launches no intra program, WY or K1 tape walk and, with the
    norm, the separate normalization from the saved rstd exactly where the plan's intra program is the replay's first
    q / k reader; the saved set grows by the byte query's difference in the mode; with the norm the fallback form
    (the workspace's test knob) is bitwise its own light path; ``"auto"`` follows the rule of the mode (this device
    verified in the mode: engaged; the mode's set emptied: ``"unverified"`` even with the device in
    ``KCP_HANDOFF_VERIFIED``); a checkpointed forward with the hand-off retains nothing."""
    W = row[2]
    ntasks = int(os.environ.get("SLURM_NTASKS", "1") or 1)
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"a {ntasks}-task Slurm step runs the W={ntasks} rows only")
        if not _programs_exported():
            pytest.skip(
                "the KCP programs are not exported in the package's build.json yet"
            )
        report = _run_rank_handoff_mode(
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
    with tempfile.TemporaryDirectory(prefix="cake_kcp_handoff_mode_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        _start_ranks(_spawn_entry_handoff_mode, W, row, tmp, result_dir)
        with open(
            os.path.join(result_dir, "handoff-" + _mode_row_id(row) + ".json")
        ) as fh:
            report = json.load(fh)
    assert report["pass"], report


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
# saved_rows on GPU: the rows forms of a plain autograd node and of the checkpoint hand-off


def _rows_stats(res: dict, refs: dict) -> dict:
    """Per tensor of ``res`` with a reference in ``refs`` (this rank's rows of the FP64 reference; the parameter
    gradients' sums on rank 0 only): the sum of squared errors, the reference's sum of squares, the largest absolute
    error, whether every element is finite and the elements outside ``ATOL + RTOL * |ref|``."""
    out = {}
    for n, ref in refs.items():
        if n not in res:
            continue
        c = res[n].detach().to(torch.float64)
        err = c - ref
        out[n] = (
            float(err.pow(2).sum()),
            float(ref.pow(2).sum()),
            float(err.abs().max()) if err.numel() else 0.0,
            bool(torch.isfinite(c).all()),
            int(((err.abs() - (ATOL + RTOL * ref.abs())) > 0).sum()),
        )
    return out


def _rows_combine(per_rank: list[dict]) -> dict:
    """One run's metrics over the ranks from their :func:`_rows_stats`: ``rel_rms`` (the root of the summed squared
    errors over the summed squared reference), ``max_abs``, ``finite`` and ``mismatch``."""
    tot: dict[str, list] = {}
    for st in per_rank:
        for n, (se, sr, mx, fin, mism) in st.items():
            a = tot.setdefault(n, [0.0, 0.0, 0.0, True, 0])
            a[0] += se
            a[1] += sr
            a[2] = max(a[2], mx)
            a[3] = a[3] and fin
            a[4] += mism
    return {
        n: {
            "rel_rms": math.sqrt(se / sr) if sr > 0 else (math.inf if se > 0 else 0.0),
            "max_abs": mx,
            "finite": fin,
            "mismatch": mism,
        }
        for n, (se, sr, mx, fin, mism) in tot.items()
    }


def _rows_within(m: dict, light: dict) -> bool:
    """A run's tensor against the light path's of the same row: finite, ``rel_rms`` and ``max_abs`` at most
    ACC_GATE times the light path's (an exact zero on both sides passes) and no more elements outside the
    ``ATOL`` / ``RTOL`` budget than the light path."""

    def ratio(a: float, b: float) -> float:
        return a / b if b > 0 else (1.0 if a == 0 else math.inf)

    return (
        m["finite"]
        and ratio(m["rel_rms"], light["rel_rms"]) <= ACC_GATE
        and ratio(m["max_abs"], light["max_abs"]) <= ACC_GATE
        and m["mismatch"] <= light["mismatch"]
    )


def _run_rank_rows(
    rank: int, W: int, row, rdzv: str | None, result_dir: str | None, local_id: int
) -> dict:
    """One rank of a saved_rows row (see :func:`test_saved_rows_gpu_rows`)."""
    from torch.autograd.graph import save_on_cpu
    from torch.utils.checkpoint import checkpoint

    lengths, H, _w, layout, gate, l2n = row
    raw = gate == "raw"
    device = torch.device("cuda", local_id % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    if W > 1 and not dist.is_initialized():
        _init_group(rank, W, device, rdzv)
    group = dist.group.WORLD if W > 1 else None
    part, cu = _partition(lengths, W, layout)
    ctx = cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(
        part, group, rank, device
    )
    if raw or l2n:
        glob = _mode_inputs(lengths, H, device, gate, l2n)
    else:
        glob = make_inputs(lengths, H, device)
    names = ("q", "k", "v", "g", "beta", "dout")
    ops = _local_slices(part, rank, cu, {n: glob[n] for n in names})
    q, k, v, g, beta, dout = (ops[n] for n in names)
    params = (glob["A_log"], glob["dt_bias"]) if raw else ()
    ws_kw = {"use_gate_in_kernel": raw, "use_qk_l2norm_in_kernel": l2n}
    ws = cake_kcp.CakeKDANondetKcpWorkspace(**ws_kw)
    # a lean arena: the activated gate without the norm only (raw gate mode and the norm refuse local_arena="lean")
    ws_lean = (
        None if raw or l2n else cake_kcp.CakeKDANondetKcpWorkspace(local_arena="lean")
    )
    T_r = int(q.shape[0])
    R = int(ctx.R)
    index = device.index if device.index is not None else torch.cuda.current_device()
    arch = cake_kcp.cake_jit.device_arch(index)
    sms = int(cake_kcp.cake_jit.sm_count(index))
    key = (arch, sms, cake_kcp._handoff_toolkit(arch))
    # the fused form's backward normalizes in its intra program first on this rank's plan: a rows backward then runs
    # the separate normalization from the saved rstd in that program's place (cake_kcp._handoff_l2n_rows)
    intra_first = bool(
        l2n
        and T_r > 0
        and cake_kcp_plan.qk_l2norm_intra_first(
            cake_kcp.cake_backend.plan(
                T_r, int(ctx.n_frag), H, sms, arch, gate=gate, l2n="fused"
            )
        )
    )
    info: dict[str, object] = {
        "T_r": T_r,
        "R": R,
        "n_frag": int(ctx.n_frag),
        "P": int(ctx.heads(H).P),
        "key": list(key),
        "intra_first": intra_first,
        "torch": torch.__version__,
    }
    flags: dict[str, bool] = {}
    # per run: this rank's _rows_stats; bitwise equality with the light path (reported, not gated)
    acc: dict[str, dict] = {}
    bitwise: dict[str, bool] = {}
    not_run: list[str] = []
    grad_names = ("dq", "dk", "dv", "dg", "dbeta") + (
        ("dA_log", "ddt_bias") if raw else ()
    )
    # the FP64 reference of the whole pack on this rank's device, then this rank's rows of it; the parameter
    # gradients (raw gate mode) are compared as cross-rank sums, counted on rank 0
    cu_t = torch.tensor(cu, dtype=torch.int32, device=device)
    ref_in = [glob[n] for n in ("q", "k", "v", "g", "beta")]
    if raw:
        exact = kda_exact_raw_l2n_with_grads if l2n else kda_exact_raw_with_grads
        ref_out, ref_g = exact(*ref_in, *params, glob["dout"], cu_t, SCALE, LOWER_BOUND)
    else:
        exact = kda_exact_l2n_with_grads if l2n else kda_exact_with_grads
        ref_out, ref_g = exact(*ref_in, glob["dout"], cu_t, SCALE)
    token_refs = {"out": ref_out, **{n: ref_g[n] for n in grad_names[:5]}}
    refs = _local_slices(part, rank, cu, token_refs)
    if raw and rank == 0:
        refs["dA_log_sum"], refs["ddt_bias_sum"] = ref_g["dA_log"], ref_g["ddt_bias"]
    del ref_out, ref_g, token_refs

    def leaves():
        return [
            x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta, *params)
        ]

    def gate_kw(lv) -> dict:
        if not raw:
            return {}
        return {
            "use_gate_in_kernel": True,
            "A_log": lv[5],
            "dt_bias": lv[6],
            "lower_bound": LOWER_BOUND,
        }

    def sums(res: dict) -> dict:
        """In raw gate mode the cross-rank sums of the parameter partials (a collective: every rank, every set)."""
        if raw:
            res["dA_log_sum"], res["ddt_bias_sum"] = (
                cake_kcp.cake_kda_nondet_train_kcp_reduce_gate_param_grads(
                    ctx, res["dA_log"], res["ddt_bias"]
                )
            )
        return res

    def delta(before: dict, after: dict) -> dict:
        return {n: after[n] - before[n] for n in after if after[n] != before[n]}

    def run(
        rows: str = "none",
        *,
        mode: str = "auto",
        ckpt: bool = False,
        workspace=ws,
        handle=None,
        keep=None,
        context_fn=None,
        poison_free: bool = False,
        poison_rows: bool = False,
        poison_arena: bool = False,
        hooks=None,
        grad: bool = True,
        twice: bool = False,
        entry=None,
        knobs=None,
    ):
        """One step of the autograd entry with ``saved_rows=rows`` and ``checkpoint_handoff=mode``, direct or under a
        non-reentrant checkpoint: the outputs and gradients of each backward (``twice``: two, the first with
        ``retain_graph``; in raw gate mode with the cross-rank parameter sums) and the LAST_CALL records of the
        original forward, the recompute (None without a checkpoint) and the backward (None without one), with the
        moves of SAVED_ROWS_STATS and HANDOFF_STATS.  ``poison_rows``: NaN-filled kept rows before the forward's
        launches (``_HANDOFF_POISON``); ``poison_arena``: the caller workspace's local arena NaN-filled between the
        forward and the backward (direct); ``hooks``: a saved-tensor hooks context around the forward; ``grad=False``:
        the forward under no_grad and no backward; ``knobs``: the call's ``_kcp_test_knobs``."""
        if poison_free:
            _poison(device)
        lv = leaves()
        kw = {}
        if knobs is not None:
            kw["_kcp_test_knobs"] = knobs
        if handle is not None:
            kw["recompute_handle"] = handle
        if keep is not None:
            kw["checkpoint_keep"] = keep
        call = entry or cake_kcp.cake_kda_nondet_kcp

        def top(q_, k_, v_, g_, b_, *p_):
            return call(
                q_,
                k_,
                v_,
                g_,
                b_,
                ctx,
                workspace=workspace,
                checkpoint_handoff=mode,
                saved_rows=rows,
                **gate_kw([q_, k_, v_, g_, b_, *p_]),
                use_qk_l2norm_in_kernel=l2n,
                **kw,
            )

        cake_kcp.LAST_CALL.pop("fwd", None)
        cake_kcp.LAST_CALL.pop("bwd", None)
        rows0 = dict(cake_kcp.SAVED_ROWS_STATS)
        hand0 = cake_kcp.cake_kda_nondet_train_kcp_handoff_stats()
        prev = cake_kcp._HANDOFF_POISON
        cake_kcp._HANDOFF_POISON = bool(poison_rows)
        sets = []
        try:
            hook_cm = hooks() if hooks is not None else contextlib.nullcontext()
            grad_cm = contextlib.nullcontext() if grad else torch.no_grad()
            with hook_cm, grad_cm:
                if not ckpt:
                    out = top(*lv)
                elif context_fn is not None:
                    out = checkpoint(
                        top, *lv, use_reentrant=False, context_fn=context_fn
                    )
                else:
                    out = checkpoint(top, *lv, use_reentrant=False)
            rec_orig = dict(cake_kcp.LAST_CALL["fwd"])
            if ckpt:
                cake_kcp.LAST_CALL.pop("fwd")
            if poison_arena and not ckpt:
                _ArenaPoison(workspace).__exit__(None, None, None)
            if grad:
                for i in range(2 if twice else 1):
                    sets.append(
                        torch.autograd.grad(
                            out, lv, dout, retain_graph=twice and i == 0
                        )
                    )
        finally:
            cake_kcp._HANDOFF_POISON = prev
        rec_fwd = cake_kcp.LAST_CALL.get("fwd") if ckpt else None
        rec_bwd = cake_kcp.LAST_CALL.get("bwd")
        recs = {
            "orig": rec_orig,
            "fwd": None if rec_fwd is None else dict(rec_fwd),
            "bwd": None if rec_bwd is None else dict(rec_bwd),
            "rows": delta(rows0, dict(cake_kcp.SAVED_ROWS_STATS)),
            "handoff": delta(hand0, cake_kcp.cake_kda_nondet_train_kcp_handoff_stats()),
        }
        res = [
            sums({"out": out.detach(), **dict(zip(grad_names, s, strict=True))})
            for s in sets
        ]
        return res or [{"out": out.detach()}], recs

    def kept(recs, form: str | None, passes=("orig", "bwd")) -> bool:
        """The passes ``passes`` kept (forward) / read (backward) the rows set ``form`` -- ``None``: no rows set; none
        on a rank without tokens -- with the byte query's bytes, and no pass took the hand-off's tape form."""
        want = form if T_r > 0 else None
        nbytes = (
            None
            if want is None
            else cake_kcp.cake_kda_nondet_train_kcp_handoff_bytes(
                ctx, H=H, saved_rows=want
            )
        )
        ok = all(
            recs[p] is not None
            and recs[p].get("saved_rows") == want
            and recs[p].get("saved_rows_bytes") == nbytes
            for p in passes
        )
        return ok and all(
            (recs[p] or {}).get("checkpoint_handoff") is None
            for p in ("orig", "fwd", "bwd")
        )

    def rows_stats(form: str | None, saves: int, binds: int) -> dict:
        """The SAVED_ROWS_STATS moves of a run that kept the rows set ``form`` in ``saves`` forwards and read it in
        ``binds`` backwards (nothing on a rank without tokens)."""
        if form is None or T_r == 0:
            return {}
        return {f"save_{form}": saves, f"bound_{form}": binds}

    def launches(recs, light_recs, form: str | None, first: bool = intra_first) -> bool:
        """The forwards' launches are the light path's; with tokens and a rows set the backward's are the light
        backward's without the intra program (the separate normalization in its place where the plan's intra program
        normalizes first) and, with ``"wy"``, without the WY program, the K1 tape replay kept; the collective counts
        are the light path's in every pass.  The two backwards' launches compare in the ``"first"`` / ``"main"``
        order (:func:`_bwd_sequence_first_main`): each run decides its K1 tape replay's place and its K3's stream
        itself (a kept ``u`` admits the replay under b1 on a lean arena, and the table set's first K3 runs on the
        call's stream), so the two may differ there."""
        passes = ("orig", "bwd") if recs["fwd"] is None else ("orig", "fwd", "bwd")
        ok = all(light_recs[p] is not None for p in passes)
        ok = ok and recs["orig"]["sequence"] == light_recs["orig"]["sequence"]
        if recs["fwd"] is not None:
            ok = ok and recs["fwd"]["sequence"] == light_recs["fwd"]["sequence"]
        on_b = _bwd_sequence_first_main(recs["bwd"])
        off_b = _bwd_sequence_first_main(light_recs["bwd"])
        if T_r > 0 and form is not None:
            want = []
            for x in off_b:
                if x == "fwd_intra":
                    if first:
                        want.append("qk_apply")
                elif not (x == "wy" and form == "wy"):
                    want.append(x)
            ok = ok and on_b == want and off_b.count("fwd_intra") == 1
            ok = ok and ("wy" in on_b) == (form == "intra") and "kcp_fwdh_tape" in on_b
        else:
            ok = ok and on_b == off_b
        for p in passes:
            ok = ok and recs[p]["n_collectives"] == light_recs[p]["n_collectives"]
        return ok

    def measure(name: str, res: dict, base: dict | None = None) -> None:
        acc[name] = _rows_stats(res, refs)
        if base is not None:
            bitwise[name] = _same(res, base)

    # the light path (no rows) and "auto", the shipped default (outside a checkpoint the light set, as before)
    (light,), recs_light = run("none")
    measure("light", light)
    flags["light_kept_nothing"] = kept(recs_light, None) and recs_light["rows"] == {}
    (res_a,), recs_auto = run("auto")
    measure("auto", res_a, light)
    flags["auto_is_light"] = (
        kept(recs_auto, None)
        and recs_auto["rows"] == {}
        and launches(recs_auto, recs_light, None)
    )
    st = recs_light["orig"]["save_transfer"]
    query = {
        form: cake_kcp_plan.cake_kda_nondet_train_kcp_saved_bytes(
            ctx, H=H, with_conv=False, save_transfer=st, saved_rows=form, **ws_kw
        )
        for form in ("none", *SAVED_ROWS_SETS)
    }
    got_bytes = {"none": recs_light["orig"]["saved_bytes"]}
    info["saved_bytes"] = {"query": query, "got": got_bytes}
    # "auto" outside a checkpoint is the light path: the same bits and the same saved bytes
    flags["auto_bitwise_light"] = (
        _same(res_a, light) and recs_auto["orig"]["saved_bytes"] == got_bytes["none"]
    )
    # (light arena poisoned between the forward and the backward: the control of the rows runs below; the backward
    # reads nothing of the arena the forward left, so the bits do not move)
    (res_p,), _ = run("none", poison_arena=True)
    measure("light_poison_arena", res_p, light)
    flags["light_poison_arena_bitwise"] = _same(res_p, light)
    if ws_lean is not None:  # the same control on the lean arena
        (res_ll,), _ = run("none", workspace=ws_lean)
        measure("light_lean", res_ll, light)
        (res_p,), _ = run("none", workspace=ws_lean, poison_arena=True)
        measure("light_lean_poison_arena", res_p, light)
        flags["light_lean_poison_arena_bitwise"] = _same(res_p, res_ll)
    for form in SAVED_ROWS_SETS:
        # a plain node: the rows kept, the launches, the counters, the saved bytes
        (res_f,), recs = run(form)
        measure(form, res_f, light)
        flags[f"{form}_kept"] = kept(recs, form)
        # outside a checkpoint the rows stay in the caching allocator's default pool
        flags[f"{form}_default_pool"] = (
            T_r == 0 or recs["orig"].get("kept_pool") is False
        )
        flags[f"{form}_launches"] = launches(recs, recs_light, form)
        flags[f"{form}_stats"] = (
            recs["rows"] == rows_stats(form, 1, 1)
            and recs["handoff"] == recs_auto["handoff"]
        )
        got_bytes[form] = recs["orig"]["saved_bytes"]
        flags[f"{form}_saved_bytes"] = got_bytes[form] - got_bytes["none"] == (
            query[form] - query["none"] if T_r > 0 else 0
        )
        # a per-call workspace and a lean arena (the tail's row views rebuilt over the kept rows); on the lean arena
        # also the arena NaN-filled between the forward and the backward and two backwards (its kept members' arena
        # slots are shared with other buffers there)
        res_ws = {}
        for tag, wsp in (("per_call", None), ("lean", ws_lean)):
            if tag == "lean" and wsp is None:
                continue
            (res_w,), recs_w = run(form, workspace=wsp)
            res_ws[tag] = res_w
            measure(f"{form}_{tag}", res_w, light)
            flags[f"{form}_{tag}_kept"] = kept(recs_w, form)
        if ws_lean is not None:
            (res_lp,), recs_lp = run(form, workspace=ws_lean, poison_arena=True)
            measure(f"{form}_lean_poison_arena", res_lp, light)
            flags[f"{form}_lean_poison_arena_kept"] = kept(recs_lp, form)
            flags[f"{form}_lean_poison_arena_bitwise"] = _same(res_lp, res_ws["lean"])
            (res_l1, res_l2), recs_lt = run(form, workspace=ws_lean, twice=True)
            measure(f"{form}_lean_twice_1", res_l1, light)
            measure(f"{form}_lean_twice_2", res_l2, res_l1)
            rows_ok = recs_lt["rows"] == rows_stats(form, 1, 2)
            flags[f"{form}_lean_twice_kept"] = kept(recs_lt, form) and rows_ok
            flags[f"{form}_lean_twice_bitwise"] = _same(res_l2, res_l1)
        # NaN-filled rows from poisoned free blocks (per-call workspace); the caller arena NaN-filled between the
        # forward and the backward.  Neither moves a bit: the forward writes every kept row and the backward reads
        # nothing of the arena the forward left
        (res_p,), recs_p = run(form, workspace=None, poison_free=True, poison_rows=True)
        measure(f"{form}_poison_rows", res_p, light)
        flags[f"{form}_poison_rows_kept"] = kept(recs_p, form)
        flags[f"{form}_poison_rows_bitwise"] = _same(res_p, res_ws["per_call"])
        (res_p,), recs_p = run(form, poison_arena=True)
        measure(f"{form}_poison_arena", res_p, light)
        flags[f"{form}_poison_arena_kept"] = kept(recs_p, form)
        flags[f"{form}_poison_arena_bitwise"] = _same(res_p, res_f)
        # two backwards (retain_graph): each reads the kept rows, and the first writes none of them (the same bits)
        (res_1, res_2), recs_t = run(form, twice=True)
        measure(f"{form}_twice_1", res_1, light)
        measure(f"{form}_twice_2", res_2, res_1)
        rows_ok = recs_t["rows"] == rows_stats(form, 1, 2)
        flags[f"{form}_twice_kept"] = kept(recs_t, form) and rows_ok
        flags[f"{form}_twice_bitwise"] = _same(res_2, res_1)

    # the norm's fallback form (the workspace's test knob: the separate normalization of the light backward first),
    # against its own light path; where the form's programs are not delivered the row lists it under not_run and fails
    ws_fb = None
    if l2n:
        try:
            ws_fb = cake_kcp.CakeKDANondetKcpWorkspace(
                **ws_kw, _qk_l2norm_form="fallback"
            )
        except NotImplementedError as exc:
            info["fallback_pending"] = str(exc)
            not_run.append("fallback")  # fails the row, as the L2N rows' protocol
    if ws_fb is not None:
        (light_fb,), recs_light_fb = run("none", workspace=ws_fb)
        measure("light_fallback", light_fb, light)
        for form in SAVED_ROWS_SETS:
            (res_fb,), recs_fb = run(form, workspace=ws_fb)
            measure(f"{form}_fallback", res_fb, light_fb)
            flags[f"{form}_fallback_kept"] = kept(recs_fb, form)
            flags[f"{form}_fallback_launches"] = launches(
                recs_fb, recs_light_fb, form, first=False
            )
    # under a checkpoint: the hand-off's rows form ("on"), with and without a recompute handle (both tiers)
    handle_type = cake_kcp.CakeKDANondetKcpRecomputeHandle
    every = ("orig", "fwd", "bwd")
    (res_c,), recs_light_ck = run("none", mode="off", ckpt=True)
    measure("light_ckpt", res_c, light)
    flags["light_ckpt_kept_nothing"] = (
        kept(recs_light_ck, None, every) and recs_light_ck["rows"] == {}
    )
    recs_light_h = {}
    for keep in ("states", "transfer"):
        _, recs_light_h[keep] = run(
            "none", mode="off", ckpt=True, handle=handle_type(), keep=keep
        )
    for form in SAVED_ROWS_SETS:
        (res_c,), recs = run(form, mode="on", ckpt=True)
        measure(f"{form}_handoff", res_c, light)
        flags[f"{form}_handoff_kept"] = (
            kept(recs, form, every)
            and recs["rows"] == rows_stats(form, 2, 1)
            and recs["handoff"] == {}
        )
        flags[f"{form}_handoff_launches"] = launches(recs, recs_light_ck, form)
        # a caller-held workspace: both passes keep the set in the default pool (its arena is not freed per call);
        # workspace=None: both passes keep it in the package-private pool of the device (the knob kept_pool, shipped
        # 1), and with kept_pool=0 in the default pool, bitwise the same
        flags[f"{form}_handoff_held_default_pool"] = T_r == 0 or (
            recs["orig"].get("kept_pool") is False
            and recs["fwd"].get("kept_pool") is False
        )
        (res_n,), recs_n = run(form, mode="on", ckpt=True, workspace=None)
        measure(f"{form}_handoff_per_call", res_n, light)
        flags[f"{form}_handoff_per_call_kept"] = kept(recs_n, form, every)
        flags[f"{form}_handoff_kept_pool"] = T_r == 0 or (
            recs_n["orig"].get("kept_pool") is True
            and recs_n["fwd"].get("kept_pool") is True
            and device.index in cake_kcp._KEPT_POOLS
        )
        previous_knobs = cake_kcp_plan.enable_test_knobs(True)
        try:
            (res_k,), recs_k = run(
                form, mode="on", ckpt=True, workspace=None, knobs={"kept_pool": 0}
            )
        finally:
            cake_kcp_plan.enable_test_knobs(previous_knobs)
        flags[f"{form}_handoff_kept_pool0_bitwise"] = _same(res_k, res_n) and (
            T_r == 0
            or (
                recs_k["orig"].get("kept_pool") is False
                and recs_k["fwd"].get("kept_pool") is False
            )
        )
        for keep in ("states", "transfer"):
            (res_h,), recs_h = run(
                form, mode="on", ckpt=True, handle=handle_type(), keep=keep
            )
            measure(f"{form}_handle_{keep}", res_h, light)
            rows_ok = recs_h["rows"] == rows_stats(form, 2, 1)
            flags[f"{form}_handle_{keep}_kept"] = kept(recs_h, form, every) and rows_ok
            # the recompute binds the handle's states: no collective
            flags[f"{form}_handle_{keep}_recompute"] = (
                recs_h["fwd"].get("recompute_handle") == "replay"
                and recs_h["fwd"]["n_collectives"] == 0
            )
            flags[f"{form}_handle_{keep}_launches"] = launches(
                recs_h, recs_light_h[keep], form
            )
    # the default saved_rows="auto" where the hand-off engages: the wy rows form (KCP_HANDOFF_AUTO_ROWS), not the
    # tape set
    (res_c,), recs = run("auto", mode="on", ckpt=True)
    measure("auto_handoff", res_c, light)
    flags["auto_handoff_keeps_wy"] = (
        kept(recs, cake_kcp_plan.KCP_HANDOFF_AUTO_ROWS, every)
        and recs["rows"] == rows_stats("wy", 2, 1)
        and recs["handoff"] == {}
    )
    flags["auto_handoff_launches"] = launches(recs, recs_light_ck, "wy")
    # NaN-filled rows and the caller arena NaN-filled between the recompute and the backward, with and without a
    # handle
    arena_fn = lambda: (contextlib.nullcontext(), _ArenaPoison(ws))  # noqa: E731
    for tag, hkw in (
        ("handoff", {}),
        ("handle", {"handle": handle_type(), "keep": "states"}),
    ):
        (res_p,), recs_p = run(
            "wy", mode="on", ckpt=True, context_fn=arena_fn, poison_rows=True, **hkw
        )
        measure(f"wy_{tag}_poison", res_p, light)
        flags[f"wy_{tag}_poison_kept"] = kept(recs_p, "wy", every)
    # "off" in a checkpoint pass: the light set in both passes; "auto": the hand-off's shipped rule decides
    (res_c,), recs = run("wy", mode="off", ckpt=True)
    measure("wy_ckpt_off", res_c, light)
    flags["wy_ckpt_off_light"] = (
        kept(recs, None, every) and recs["rows"] == {} and recs["handoff"] == {}
    )
    reason = cake_kcp_plan.kcp_handoff_auto_refusal(
        key, H, ctx.partition.total, W, gate=gate, l2n=l2n
    )
    info["auto_reason"] = reason
    (res_c,), recs = run("wy", mode="auto", ckpt=True)
    measure("wy_ckpt_auto", res_c, light)
    if reason is None:
        flags["wy_ckpt_auto"] = (
            kept(recs, "wy", every)
            and recs["rows"] == rows_stats("wy", 2, 1)
            and recs["handoff"] == {}
        )
    else:
        flags["wy_ckpt_auto"] = (
            kept(recs, None, every)
            and recs["rows"] == {}
            and recs["handoff"] == {"refused_" + reason: 1}
        )
    # a caller's saved-tensor hooks (save_on_cpu): refused once, the light set
    cpu_hooks = lambda: save_on_cpu(pin_memory=True)  # noqa: E731
    try:
        run("none", hooks=cpu_hooks)
        hooks_ok = True
    except Exception:  # noqa: BLE001 - the light path itself does not run under the hooks: nothing to compare
        info["save_on_cpu_unsupported"] = traceback.format_exc()[-1500:]
        hooks_ok = False
    if hooks_ok:
        (res_k,), recs = run("wy", hooks=cpu_hooks)
        measure("wy_hooks", res_k, light)
        rows_ok = recs["rows"] == {"refused_hooks": 1}
        flags["wy_hooks_refused"] = kept(recs, None) and rows_ok
    else:
        # not run: the refusal under hooks is unchecked, so the row fails (the traceback is in the report)
        flags["wy_hooks_refused"] = False
        not_run.append("wy_hooks_refused")
    # no grad: no node, nothing kept
    (res_n,), recs = run("wy", grad=False)
    measure("wy_no_grad", res_n)
    flags["wy_no_grad"] = (
        recs["orig"].get("saved_rows") is None
        and recs["rows"] == {}
        and recs["bwd"] is None
    )
    # ranks that keep different sets (even ranks "wy", odd ranks none): the rows move no collective
    mixed = "wy" if rank % 2 == 0 else "none"
    mixed_form = None if mixed == "none" else mixed
    (res_m,), recs = run(mixed)
    measure("mixed", res_m, light)
    flags["mixed_kept"] = kept(recs, mixed_form)
    (res_m,), recs = run(mixed, mode="on" if mixed == "wy" else "off", ckpt=True)
    measure("mixed_ckpt", res_m, light)
    flags["mixed_ckpt_kept"] = kept(recs, mixed_form, every)
    # the facade forwards saved_rows
    (res_fc,), recs = run("wy", entry=kcp_api.cake_kda_nondet_kcp)
    measure("wy_facade", res_fc, light)
    flags["wy_facade_kept"] = kept(recs, "wy")

    # the functional entries: the saved set's members, dtypes, shapes and bytes; graph="replay"
    fgate = gate_kw([q, k, v, g, beta, *params])
    fnames = ("dq", "dk", "dv", "dbeta", "dg") + (("dA_log", "ddt_bias") if raw else ())

    def functional(form: str, **opts):
        out, saved = cake_kcp.cake_kda_nondet_train_kcp_forward(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            workspace=ws,
            saved_rows=form,
            use_qk_l2norm_in_kernel=l2n,
            **fgate,
            **opts,
        )
        rec_f = dict(cake_kcp.LAST_CALL["fwd"])
        grads = cake_kcp.cake_kda_nondet_train_kcp_backward(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            dout,
            saved=saved,
            workspace=ws,
            use_qk_l2norm_in_kernel=l2n,
            **fgate,
            **opts,
        )
        res = {"out": out.clone()}
        res.update({n: t.clone() for n, t in zip(fnames, grads, strict=True)})
        return sums(res), saved, (rec_f, dict(cake_kcp.LAST_CALL["bwd"]))

    _, saved_none, _ = functional("none")
    bytes_none, carry_none = saved_none.bytes, saved_none.carry is None
    del saved_none
    for form, (members, per) in SAVED_ROWS_SETS.items():
        res_f, saved, (rec_f, rec_b) = functional(form)
        measure(f"{form}_functional", res_f, light)
        if T_r > 0:
            kept_rows = saved.rows or {}
            ok = (
                saved.saved_rows == form
                and tuple(kept_rows) == members
                and saved.tape is None
                and (saved.carry is None) == carry_none
                and saved.bytes - bytes_none == R * H * per
            )
            # BF16 [R, H, 64] (the chunk's Aqk / Akk columns) and [R, H, 128] (w / u)
            ok = ok and all(
                t.dtype == torch.bfloat16
                and tuple(t.shape) == (R, H, 64 if n.endswith("_int") else 128)
                for n, t in kept_rows.items()
            )
        else:
            ok = (
                saved.saved_rows is None
                and saved.rows is None
                and saved.bytes == bytes_none
            )
        want = form if T_r > 0 else None
        flags[f"{form}_functional_saved_set"] = (
            ok and rec_f.get("saved_rows") == want and rec_b.get("saved_rows") == want
        )
        del saved
    # graph="replay" on every rank (a rank without tokens too): eager, capture, then replays of the rows form
    cake_kcp.graph_cache_clear()
    events = []
    graph_rows = True
    try:
        for _ in range(len(GRAPH_EVENTS)):
            res_g, saved_g, (rec_f, rec_b) = functional("wy", graph="replay")
            events.append(
                [str(rec_f.get("graph_event")), str(rec_b.get("graph_event"))]
            )
            want = "wy" if T_r > 0 else None
            graph_rows = graph_rows and (
                rec_f.get("saved_rows") == want and rec_b.get("saved_rows") == want
            )
            del saved_g
        torch.cuda.synchronize(device)
    except BaseException as exc:
        # the failed call's frames still reference its cache entry and graph: release them before the cleanup
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        # a cached graph holding a NCCL collective blocks the process group's destruction
        cake_kcp.graph_cache_clear()
    measure("wy_graph", res_g, light)
    info["graph_events"] = events
    flags["wy_graph_replay"] = events == GRAPH_EVENTS and graph_rows
    info["not_run"] = not_run
    info["bitwise_light"] = bitwise

    payload = (flags, info, acc)
    if W > 1:
        gathered: list = [None] * W
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]
    flag_list = [f for f, _, _ in gathered]
    runs = sorted({n for _, _, a in gathered for n in a})
    metrics = {n: _rows_combine([a[n] for _, _, a in gathered if n in a]) for n in runs}
    # every run against the light path of its form: the norm's fallback form against its own light path
    accuracy = {
        n: {
            t: _rows_within(
                m[t], metrics["light_fallback" if "fallback" in n else "light"][t]
            )
            for t in m
        }
        for n, m in metrics.items()
        if n not in ("light", "light_fallback")
    }
    report = {
        "row": list(map(str, row)),
        "flags": {key_: all(f[key_] for f in flag_list) for key_ in flags},
        "failing_ranks": {
            key_: [r for r, f in enumerate(flag_list) if not f[key_]] for key_ in flags
        },
        "light_finite": all(m["finite"] for m in metrics["light"].values()),
        # the activated gate without the norm: the light path within ATOL / RTOL everywhere (the absolute gate of
        # the dist rows), so every run is too (_rows_within: no more misses than the light path); the raw gate and
        # the norm keep their protocols' relative gate (against the light path here)
        "light_budget": gate != "g"
        or l2n
        or all(m["mismatch"] == 0 for m in metrics["light"].values()),
        "accuracy": {n: all(v.values()) for n, v in accuracy.items()},
        "accuracy_failing": {
            n: [t for t, ok in v.items() if not ok]
            for n, v in accuracy.items()
            if not all(v.values())
        },
        "metrics": metrics,
        "not_run": sorted({n for _, i, _ in gathered for n in i["not_run"]}),
        "info_by_rank": [i for _, i, _ in gathered],
    }
    report["pass"] = (
        all(report["flags"].values())
        and report["light_finite"]
        and report["light_budget"]
        and all(report["accuracy"].values())
        and all(set(f) == set(flag_list[0]) for f in flag_list)
        and not report["not_run"]
    )
    if W > 1:
        dist.barrier()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(
            os.path.join(result_dir, "rows-" + _mode_row_id(row) + ".json"), "w"
        ) as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _spawn_entry_rows(
    rank: int, W: int, row, rdzv: str | None, result_dir: str
) -> None:
    try:
        _bound_rank_threads(W)
        report = _run_rank_rows(rank, W, row, rdzv, result_dir, rank)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            cake_kcp.graph_cache_clear()
            dist.destroy_process_group()


@pytest.mark.parametrize("row", SAVED_ROWS_GPU_ROWS, ids=_mode_row_id)
def test_saved_rows_gpu_rows(row) -> None:
    """``saved_rows="intra"`` / ``"wy"`` of :func:`cake_kcp.cake_kda_nondet_kcp` on every rank of the row
    (``SAVED_ROWS_GPU_ROWS``: W = 1, 2, 4; H 12, 16, 32, 48; the activated and the raw gate; the in-kernel Q / K L2
    norm off and on): every output and gradient (in raw gate mode the raw gate's and the cross-rank sums of the
    ``A_log`` / ``dt_bias`` partials) against the FP64 reference within ACC_GATE times the light path's error
    (``rel_rms`` and ``max_abs``), finite and with no more elements outside ``ATOL`` / ``RTOL`` (activated gate
    without the norm: the light path, so every run, with none outside) -- for ``"auto"`` (the light set outside a
    checkpoint: bitwise the light path, the same saved bytes), each rows set on a caller workspace, a per-call
    workspace and (activated gate, no norm) a lean arena, with the norm's fallback form (against its own light path;
    a form that cannot run fails the row), from NaN-filled rows and poisoned free blocks, with the caller arena
    NaN-filled between the forward and the backward (and the light path so poisoned; each poisoned run bitwise its
    unpoisoned twin), over two backwards (``retain_graph``, the second bitwise the first; also on the lean arena),
    as the checkpoint hand-off's rows form (``checkpoint_handoff="on"``) with and without a recompute handle of both
    tiers (the recompute issues no collective; on a caller workspace both passes keep the rows in the default pool,
    on a per-call workspace in the package-private pool of the device, and with the test knob ``kept_pool=0`` in the
    default pool, bitwise the same; a plain node keeps them in the default pool) and with a poisoned arena, with
    ``"off"`` and ``"auto"`` in a
    checkpoint pass, under ``save_on_cpu`` (refused, the light set; a run that cannot run fails the row), with ranks
    that keep different sets, through the facade, through the functional entries and under ``graph="replay"``
    (eager, capture, replays); the rows kept by the forwards and read by the backward with the byte query's bytes
    (none on a rank without tokens, never as the hand-off's tape form),
    SAVED_ROWS_STATS counting them and HANDOFF_STATS untouched by them, the saved set growing by the query's
    difference (members, BF16 shapes and bytes of the functional saved set), the forwards' launches and every
    collective count the light path's, the backward without the intra program (the separate normalization in its
    place where the plan's intra program normalizes first) and with ``"wy"`` without the WY program; under no_grad
    nothing is kept.  Bitwise equality of a rows run with the light path is reported, not required."""
    W = row[2]
    ntasks = int(os.environ.get("SLURM_NTASKS", "1") or 1)
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"a {ntasks}-task Slurm step runs the W={ntasks} rows only")
        if not _programs_exported():
            pytest.skip(
                "the KCP programs are not exported in the package's build.json yet"
            )
        report = _run_rank_rows(
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
    if W == 1:
        report = _run_rank_rows(
            0,
            1,
            row,
            None,
            os.environ.get("KCP_TEST_RESULT_DIR"),
            torch.cuda.current_device(),
        )
        assert report["pass"], report
        return
    with tempfile.TemporaryDirectory(prefix="cake_kcp_rows_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        _start_ranks(_spawn_entry_rows, W, row, tmp, result_dir)
        with open(
            os.path.join(result_dir, "rows-" + _mode_row_id(row) + ".json")
        ) as fh:
            report = json.load(fh)
    assert report["pass"], report


# ---------------------------------------------------------------------------------------------------------------
# the contract of the rows sets with their producers (``saved_rows``): a saved set's rows stand in for the
# rows the backward's replay would write, so each member a forward keeps must be bitwise the member the light
# backward's replay writes, on every real token row.  The producers are the single-rank package's local programs (the
# deterministic package's records): the forward's intra and WY programs and the replay's.

SAVED_ROWS_PRODUCER_STAGES = ("fwd_intra", "wy_fwd", "wy")
# sha256 (16 hex digits) of the producers of the rows members in the single-rank registry (name, stage, library,
# arches, select rule, the members each binds) with KCP_SAVED_VERSION, the sets, their bytes and the direct WY records
# (_saved_rows_contract_key).  A det head bump that touches a producer changes it: rerun
# test_saved_rows_forward_rows_are_the_replay_rows on sm_100a, sm_103a and sm_107a, then pin the new key here
SAVED_ROWS_CONTRACT_KEY = "d7f3675838835582"
# the contract's GPU rows (lengths, H, gate mode, use_qk_l2norm_in_kernel), one rank: H 4 (below 16 heads) to 64,
# both gate modes with the norm off and on
SAVED_ROWS_CONTRACT_ROWS = [
    ((100, 640, 77, 1000, 129), 4, "g", False),
    ((100, 640, 77, 1000, 129), 12, "g", True),
    ((700, 1, 1300, 999), 16, "raw", False),
    ((2048, 5, 3000, 1), 32, "raw", True),
    ((3000,), 48, "g", True),
    ((1200, 900, 900), 64, "raw", False),
]


def _contract_row_id(row) -> str:
    lengths, H, gate, l2n = row
    return f"s{'_'.join(str(n) for n in lengths)}-h{H}-{gate}" + ("-l2n" if l2n else "")


def _saved_rows_contract_key() -> tuple[str, list]:
    """``(key, producers)``: the registry's records of SAVED_ROWS_PRODUCER_STAGES that bind a member of the ``"wy"``
    set (``[name, stage, library, arches, select rule, (argument, member) pairs]``), with the saved-set version, the
    sets, their bytes and the direct WY records, hashed as sorted JSON (SAVED_ROWS_CONTRACT_KEY)."""
    members = set(cake_kcp_plan.KCP_SAVED_ROWS["wy"])
    producers = []
    for name, m in sorted(cake_kcp.cake_jit.MODULES.items()):
        bound = sorted((k, v) for k, v in m["logical"].items() if v in members)
        if m["stage"] in SAVED_ROWS_PRODUCER_STAGES and bound:
            producers.append(
                [
                    name,
                    m["stage"],
                    m["library"],
                    sorted(m["arches"]),
                    m["select_rule"],
                    bound,
                ]
            )
    blob = {
        "version": cake_kcp.KCP_SAVED_VERSION,
        "sets": {n: list(cake_kcp_plan.KCP_SAVED_ROWS[n]) for n in ("intra", "wy")},
        "bytes": dict(cake_kcp_plan.KCP_SAVED_ROWS_BYTES),
        "wy_direct": dict(cake_kcp.WY_DIRECT_RECORDS),
        "producers": producers,
    }
    text = json.dumps(blob, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16], producers


def test_saved_rows_contract_key() -> None:
    """The rows sets' layout and producers are the ones test_saved_rows_forward_rows_are_the_replay_rows measured
    (SAVED_ROWS_CONTRACT_KEY): the saved-set version, the members of each set, BF16 Aqk / Akk of 64 columns and w / u
    of 128 (their bytes per internal token-head), the direct WY records and every producer record of a member in the
    single-rank registry.  A changed producer (a det head bump) fails here until the GPU contract reruns on every
    architecture and the key is pinned again."""
    assert cake_kcp.KCP_SAVED_VERSION == 2
    assert cake_kcp_plan.KCP_SAVED_ROWS["intra"] == ("aqk_int", "akk_int")
    assert cake_kcp_plan.KCP_SAVED_ROWS["wy"] == ("aqk_int", "akk_int", "w", "u")
    # BF16 (2 bytes): 64 Aqk and 64 Akk columns, then 128 w and 128 u columns
    assert cake_kcp_plan.KCP_SAVED_ROWS_BYTES == {
        "intra": 2 * (64 + 64),
        "wy": 2 * (64 + 64 + 128 + 128),
    }
    assert cake_kcp.WY_DIRECT_RECORDS == {"wy_fwd": "wy_fwd", "wy": "wy_bwd_replay"}
    key, producers = _saved_rows_contract_key()
    stages = {p[1] for p in producers}
    assert stages == set(SAVED_ROWS_PRODUCER_STAGES), stages
    assert key == SAVED_ROWS_CONTRACT_KEY, (
        f"the producers of the saved rows changed (key {key}, pinned {SAVED_ROWS_CONTRACT_KEY}): rerun "
        "test_saved_rows_forward_rows_are_the_replay_rows on sm_100a, sm_103a and sm_107a, then pin the key; "
        f"producers: {[(p[0], p[2]) for p in producers]}"
    )


@pytest.mark.parametrize("row", SAVED_ROWS_CONTRACT_ROWS, ids=_contract_row_id)
def test_saved_rows_forward_rows_are_the_replay_rows(row, monkeypatch) -> None:
    """The contract of the rows sets with their producers (``saved_rows``), one rank: every member a forward keeps
    under ``saved_rows="wy"`` (``aqk_int`` / ``akk_int`` of its intra program, ``w`` / ``u`` of its WY program) is bitwise
    the member the light backward's replay writes (its intra program, its WY program) on every real token row, in the
    same layout (BF16 ``[R, H, 64]`` / ``[R, H, 128]``, the replay's view of the same element count, compared row by
    row); the replay writes every member.  Reported, not gated: the pad rows' differences (as the SPEC-3 probe), the
    forward output's bitwise equality with the light forward's, the producer records each pass launched and the
    contract key (SAVED_ROWS_CONTRACT_KEY)."""
    _skip_unless_runnable(1)
    lengths, H, gate, l2n = row
    raw = gate == "raw"
    device = torch.device("cuda", torch.cuda.current_device())
    part, cu = _partition(lengths, 1, "contiguous")
    ctx = cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(part, None, 0, device)
    if raw or l2n:
        glob = _mode_inputs(lengths, H, device, gate, l2n)
    else:
        glob = make_inputs(lengths, H, device)
    names = ("q", "k", "v", "g", "beta", "dout")
    ops = _local_slices(part, 0, cu, {n: glob[n] for n in names})
    q, k, v, g, beta, dout = (ops[n] for n in names)
    gate_kw = (
        {
            "use_gate_in_kernel": True,
            "A_log": glob["A_log"],
            "dt_bias": glob["dt_bias"],
            "lower_bound": LOWER_BOUND,
        }
        if raw
        else {}
    )
    ws = cake_kcp.CakeKDANondetKcpWorkspace(
        use_gate_in_kernel=raw, use_qk_l2norm_in_kernel=l2n
    )
    common = {"workspace": ws, "use_qk_l2norm_in_kernel": l2n, **gate_kw}
    members = cake_kcp_plan.KCP_SAVED_ROWS["wy"]
    # the members each producer stage writes (the replay's intra program with write_int = 1)
    writes = {
        "fwd_intra": ("aqk_int", "akk_int"),
        "wy_fwd": ("w", "u"),
        "wy": ("w", "u"),
    }
    launches: list[tuple[bool, str, str]] = []
    replay: dict[str, torch.Tensor] = {}
    launch = cake_kcp.KernelBackend._launch

    def logged(call, record, grid, values) -> None:
        launch(call, record, grid, values)
        module = cake_kcp.cake_jit.MODULES.get(record, {})
        stage = module.get("stage")
        if stage not in writes:
            return
        launches.append((bool(call.backward), stage, record))
        if call.backward:
            bound = set(module["logical"].values())
            for n in writes[stage]:
                t = values.get(n)
                if n in bound and isinstance(t, torch.Tensor):
                    # on the launch's stream, right after the program wrote it
                    replay[n] = t.clone()

    monkeypatch.setattr(cake_kcp.KernelBackend, "_launch", staticmethod(logged))
    out_w, saved_w = cake_kcp.cake_kda_nondet_train_kcp_forward(
        q, k, v, g, beta, ctx, saved_rows="wy", **common
    )
    kept = {n: t.clone() for n, t in (saved_w.rows or {}).items()}
    out_w = out_w.clone()
    fwd_launches = list(launches)
    del saved_w
    out_n, saved_n = cake_kcp.cake_kda_nondet_train_kcp_forward(
        q, k, v, g, beta, ctx, saved_rows="none", **common
    )
    out_n = out_n.clone()
    launches.clear()
    cake_kcp.cake_kda_nondet_train_kcp_backward(
        q, k, v, g, beta, ctx, dout, saved=saved_n, **common
    )
    torch.cuda.synchronize(device)
    bwd_launches = [x for x in launches if x[0]]
    R, BT = int(ctx.R), int(cake_kcp.BT)
    rows = torch.arange(R, device=device)
    lens = torch.tensor(list(ctx.chunk_len), dtype=torch.int64, device=device)
    real = (rows % BT) < lens[rows // BT]
    key, _producers = _saved_rows_contract_key()
    report: dict = {
        "row": _contract_row_id(row),
        "key": key,
        "R": R,
        "real_rows": int(real.sum()),
        "kept": list(kept),
        "fwd_producers": sorted({(s, r) for _b, s, r in fwd_launches}),
        "bwd_producers": sorted({(s, r) for _b, s, r in bwd_launches}),
        "out_bitwise_light": _same({"out": out_w}, {"out": out_n}),
        "fields": {},
    }
    ok = tuple(kept) == members
    for n in members:
        a, b = kept.get(n), replay.get(n)
        field: dict = {
            "kept": None if a is None else [str(a.dtype), list(a.shape)],
            "replay": None if b is None else [str(b.dtype), list(b.shape)],
        }
        cols = 64 if n.endswith("_int") else 128
        good = (
            a is not None
            and b is not None
            and a.dtype == torch.bfloat16
            and tuple(a.shape) == (R, H, cols)
            and b.dtype == a.dtype
            and b.numel() == a.numel()
        )
        if good:
            differ = (_bits(a).reshape(R, -1) != _bits(b).reshape(R, -1)).any(1)
            field["real_rows_differ"] = int((differ & real).sum())
            field["pad_rows_differ"] = int((differ & ~real).sum())
            good = field["real_rows_differ"] == 0
        field["ok"] = bool(good)
        report["fields"][n] = field
        ok = ok and good
    report["pass"] = bool(ok)
    result_dir = os.environ.get("KCP_TEST_RESULT_DIR")
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(
            os.path.join(result_dir, f"contract-{_contract_row_id(row)}.json"), "w"
        ) as fh:
            json.dump(report, fh, indent=1, default=str)
    assert ok, report


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
    assert (
        cake_kcp.KCP_HANDOFF_VERIFIED_MODES is cake_kcp_plan.KCP_HANDOFF_VERIFIED_MODES
    )
    assert cake_kcp_plan.KCP_HANDOFF_VERIFIED_MODES == VERIFIED_MODE_KEYS
    assert (
        cake_kcp.KCP_HANDOFF_MIN_HEAD_TOKENS_BY_KEY
        is cake_kcp_plan.KCP_HANDOFF_MIN_HEAD_TOKENS_BY_KEY
    )
    assert cake_kcp_plan.KCP_HANDOFF_MIN_HEAD_TOKENS_BY_KEY == MIN_HEAD_TOKENS_BY_KEY
    assert cake_kcp_plan.KCP_HANDOFF_HEADS == AUTO_HEADS
    assert cake_kcp_plan.KCP_HANDOFF_BUFFERS == cake_kcp.CakeKDANondetKcpSaved._TAPE
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


def _auto_want(
    key, H: int, total: int, W: int, kind: tuple[str, bool] = ("g", False)
) -> str | None:
    if H not in AUTO_HEADS:
        return "heads"
    if H * (total // W) < MIN_HEAD_TOKENS:
        return "work"
    if H * (total // W) < MIN_HEAD_TOKENS_BY_KEY.get(key, 0):
        return "work"
    verified = VERIFIED_KEYS if kind == ("g", False) else VERIFIED_MODE_KEYS[kind]
    return None if key in verified else "unverified"


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
    # raw gate mode and the in-kernel Q / K L2 norm: the same bounds, the devices verified in the mode (a key of
    # KCP_HANDOFF_VERIFIED alone is "unverified" there)
    for kind in VERIFIED_MODE_KEYS:
        gate, l2n = kind
        for key in sorted(VERIFIED_KEYS) + other_keys:
            for H in (1, 4, 16, 64, 128):
                for W in (1, 2, 4, 8):
                    for total in (T, 65536, 1 << 20):
                        got = cake_kcp_plan.kcp_handoff_auto_refusal(
                            key, H, total, W, gate=gate, l2n=l2n
                        )
                        assert got == _auto_want(key, H, total, W, kind), (
                            kind,
                            key,
                            H,
                            W,
                            total,
                        )
        for key in VERIFIED_KEYS - VERIFIED_MODE_KEYS[kind]:
            assert (
                cake_kcp_plan.kcp_handoff_auto_refusal(
                    key, 64, T, 2, gate=gate, l2n=l2n
                )
                == "unverified"
            )
    for key in VERIFIED_KEYS - set(MIN_HEAD_TOKENS_BY_KEY):
        assert cake_kcp_plan.kcp_handoff_auto_refusal(key, 4, T, 2) is None
        assert cake_kcp_plan.kcp_handoff_auto_refusal(key, 4, T, 4) == "work"
        assert cake_kcp_plan.kcp_handoff_auto_refusal(key, 4, T, 8) == "work"
        for H in (16, 64):
            for W in (2, 4, 8):
                assert cake_kcp_plan.kcp_handoff_auto_refusal(key, H, T, W) is None
        # the bound is inclusive
        assert cake_kcp_plan.kcp_handoff_auto_refusal(key, 4, 98304, 1) is None
        assert cake_kcp_plan.kcp_handoff_auto_refusal(key, 4, 98303, 1) == "work"


def test_handoff_auto_per_key_work_bound() -> None:
    """GB200's own work bound (``KCP_HANDOFF_MIN_HEAD_TOKENS_BY_KEY``), from the rule's rows on GB200 (T = 261392):
    ``H * T // W`` = 522784 (H8 at CP4, which does not gain reliably there, and H4 at CP2) is refused as ``"work"`` in
    every gate / L2 norm mode, 1045568 and above (H8 at CP2, H16 and H64 at CP2 and CP4) is admitted, as is the H64 step
    of 16384 tokens per rank; the bound is inclusive, comes after the shared bounds and before the key's verification,
    and leaves the other keys at the shared bound."""
    T = 261392
    assert MIN_HEAD_TOKENS_BY_KEY[GB200] > MIN_HEAD_TOKENS
    assert set(MIN_HEAD_TOKENS_BY_KEY) <= set(VERIFIED_KEYS)
    for gate, l2n in (("g", False), *VERIFIED_MODE_KEYS):

        def refusal(key, H: int, total: int, W: int, _g=gate, _n=l2n) -> str | None:
            return cake_kcp_plan.kcp_handoff_auto_refusal(
                key, H, total, W, gate=_g, l2n=_n
            )

        for H, W in ((4, 2), (8, 4), (4, 4), (4, 8), (16, 8)):
            assert refusal(GB200, H, T, W) == "work", (gate, l2n, H, W)
        for H, W in ((8, 2), (16, 2), (16, 4), (32, 4), (64, 2), (64, 4), (64, 8)):
            assert refusal(GB200, H, T, W) is None, (gate, l2n, H, W)
        assert refusal(GB200, 64, 4 * 16384, 4) is None
        # inclusive: H * (total // W) == 786432 is admitted, one token per rank less is not
        assert refusal(GB200, 4, 196608, 1) is None
        assert refusal(GB200, 4, 196607, 1) == "work"
        assert refusal(GB200, 8, 2 * 98304, 2) is None
        assert refusal(GB200, 8, 2 * 98304 - 2, 2) == "work"
        # the shared bounds first ("heads" before the key's "work"), the key's verification after it
        assert refusal(GB200, 12, T, 2) == "heads"
        assert refusal(("sm_100a", 152, "13.3"), 4, T, 2) == "unverified"
        # the other verified keys keep the shared bound: 522784 is admitted there
        for key in sorted(VERIFIED_KEYS - {GB200}):
            verified = (
                VERIFIED_KEYS
                if (gate, l2n) == ("g", False)
                else VERIFIED_MODE_KEYS[(gate, l2n)]
            )
            assert refusal(key, 4, T, 2) == (None if key in verified else "unverified")
    # the bound is read at call time like the shared constants
    saved = cake_kcp_plan.KCP_HANDOFF_MIN_HEAD_TOKENS_BY_KEY
    try:
        cake_kcp_plan.KCP_HANDOFF_MIN_HEAD_TOKENS_BY_KEY = {}
        assert cake_kcp_plan.kcp_handoff_auto_refusal(GB200, 8, T, 4) is None
    finally:
        cake_kcp_plan.KCP_HANDOFF_MIN_HEAD_TOKENS_BY_KEY = saved


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
            cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(
                part, None, r, "cpu", 4, sm_count=148
            )
            for r in range(W)
        ]
    for key in (("sm_103a", 152, "13.4"), ("sm_90a", 132, "13.4")):
        monkeypatch.setattr(cake_kcp, "_handoff_key", lambda device, _k=key: _k)
        for W, ctxs in contexts.items():
            shares = [ctx.T_r for ctx in ctxs]
            assert W == 1 or len(set(shares)) > 1, shares
            assert W < 4 or min(shares) == 0, shares
            for H in (4, 8, 16, 64, 128):
                for kind in (("g", False), *VERIFIED_MODE_KEYS):
                    got = set()
                    for ctx in ctxs:
                        v = types.SimpleNamespace(
                            device=torch.device("cuda", 0), shape=(ctx.T_r, H, 128)
                        )
                        got.add(
                            cake_kcp._handoff_refusal(
                                "auto", opts, "saved", v, ctx, kind
                            )
                        )
                    want = _auto_want(key, H, 261392, W, kind)
                    assert got == {want}, (key, W, H, kind, got)


def test_handoff_l2n_refused_without_the_direct_record(monkeypatch) -> None:
    """With the in-kernel Q / K L2 norm, a call that would keep the set keeps the light set (``"backend"``, last, in
    both modes) where the device's architecture lacks a record of ``L2N_DIRECT_RECORDS``, decided from the registry
    and ``build.json`` alone (the same answer on every rank of one device kind); without the norm the record is not
    read.  The shipped architectures have it."""
    rule = cake_kcp._handoff_l2n_direct.__wrapped__
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        assert rule(arch), arch
    assert not rule("sm_90a")
    name = cake_kcp.L2N_DIRECT_RECORDS["qk_apply"]
    entry = cake_kcp.cake_jit.MODULES[name]
    with monkeypatch.context() as m:
        m.setitem(
            cake_kcp.cake_jit.MODULES,
            name,
            {**entry, "arches": [a for a in entry["arches"] if a != "sm_103a"]},
        )
        assert not rule("sm_103a")
    assert rule("sm_103a")
    backend = cake_kcp.KernelBackend()
    monkeypatch.setattr(cake_kcp, "get_backend", lambda: backend)
    key = ("sm_103a", 152, "13.4")
    monkeypatch.setattr(cake_kcp, "_handoff_key", lambda device: key)
    opts = cake_kcp._options({})
    lengths = (200000, 100, 61292)  # T = 261392
    part, _cu = _partition(lengths, 2, "ragged:200000,61392")
    ctxs = [
        cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(
            part, None, r, "cpu", 4, sm_count=152
        )
        for r in range(2)
    ]
    for has_record in (True, False):
        monkeypatch.setattr(
            cake_kcp, "_handoff_l2n_direct", lambda arch, _h=has_record: _h
        )
        for ctx in ctxs:
            for H in (4, 64):
                v = types.SimpleNamespace(
                    device=torch.device("cuda", 0), shape=(ctx.T_r, H, 128)
                )
                for kind in VERIFIED_MODE_KEYS:
                    for mode in ("auto", "on"):
                        want = None
                        if mode == "auto":
                            want = _auto_want(key, H, 261392, 2, kind)
                        if want is None and kind[1] and not has_record:
                            want = "backend"
                        got = cake_kcp._handoff_refusal(
                            mode, opts, "saved", v, ctx, kind
                        )
                        assert got == want, (has_record, H, kind, mode, got)
                # without the norm the record is not read
                assert cake_kcp._handoff_refusal("on", opts, "saved", v, ctx) is None
                # the argument refusals come first
                assert (
                    cake_kcp._handoff_refusal(
                        "on", opts, "replay", v, ctx, ("raw", True)
                    )
                    == "boundary"
                )


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


def test_kept_pool_knob_and_rule() -> None:
    """``kept_pool`` is a private test knob: refused without :func:`cake_kcp_plan.enable_test_knobs`, never a public
    keyword, validated against :data:`cake_kcp.KEPT_POOLS`, shipped 1.  :func:`cake_kcp._kept_pool` picks the
    package-private pool of the device in both passes of a non-reentrant checkpoint for a ``workspace=None`` call
    with the knob at 1, and the default pool (``None``) outside a checkpoint pass, for a caller-held workspace, with
    the knob at 0 and off CUDA.  :func:`cake_kcp._kept_extent` rounds up to 16 steps per power of two."""
    from torch.utils.checkpoint import checkpoint

    got = [cake_kcp._kept_extent(n) for n in (0, 1, 31, 32, 33, 64, 65)]
    assert got == [0, 1, 31, 32, 34, 64, 68], got
    for n in (257, 16384, 16385, 16448, 17000, 2**20 + 1):
        e = cake_kcp._kept_extent(n)
        step = 1 << max(n.bit_length() - 5, 0)
        assert n <= e < n + step and e % step == 0 and 16 * (e - n) < n, (n, e)
    assert cake_kcp._kept_extent(16448) == 17408
    assert cake_kcp.KEPT_POOL_DEFAULT == 1 and cake_kcp.KEPT_POOLS == (0, 1)
    assert cake_kcp.P2_KNOBS["kept_pool"] == cake_kcp.KEPT_POOLS
    on = cake_kcp._options({})
    assert on.kept_pool == 1
    previous = cake_kcp_plan.enable_test_knobs(False)
    try:
        with pytest.raises(ValueError, match="enable_test_knobs"):
            cake_kcp._options({"_kcp_test_knobs": {"kept_pool": 0}})
        with pytest.raises(TypeError):
            cake_kcp._options({"kept_pool": 0})
        cake_kcp_plan.enable_test_knobs(True)
        off = cake_kcp._options({"_kcp_test_knobs": {"kept_pool": 0}})
        assert off.kept_pool == 0
        for bad in (2, -1):
            with pytest.raises(ValueError):
                cake_kcp._options({"_kcp_test_knobs": {"kept_pool": bad}})
    finally:
        cake_kcp_plan.enable_test_knobs(previous)
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda", torch.cuda.current_device()))
    for device in devices:
        # scratch["default_ws"]: "hit" / "miss" / "bypass" for a workspace=None call, None for a caller-held one
        call_on = types.SimpleNamespace(
            opts=on, device=device, scratch={"default_ws": "hit"}
        )
        call_off = types.SimpleNamespace(
            opts=off, device=device, scratch={"default_ws": "hit"}
        )
        call_held = types.SimpleNamespace(
            opts=on, device=device, scratch={"default_ws": None}
        )
        seen: list = []

        def fn(x, calls=(call_on, call_off, call_held), seen=seen):
            seen.append(tuple(cake_kcp._kept_pool(c) for c in calls))
            return x * x

        assert cake_kcp._kept_pool(call_on) is None  # outside a checkpoint pass
        x = torch.ones(4, device=device, requires_grad=True)
        checkpoint(fn, x, use_reentrant=False).sum().backward()
        if device.type != "cuda":
            assert seen == [(None, None, None), (None, None, None)], seen
            continue
        pool = cake_kcp._KEPT_POOLS[device.index]
        assert seen == [(pool, None, None), (pool, None, None)], seen


@pytest.mark.skipif(
    not torch.cuda.is_available() or cake_kcp._MEM_POOL is None,
    reason="CUDA device and torch.cuda.MemPool required",
)
def test_default_workspace_cache_clear_drops_the_kept_pools() -> None:
    """A package-private kept-set pool keeps a freed tensor's blocks reserved, and ``torch.cuda.empty_cache()`` does
    not release them while the pool lives; :func:`cake_kcp.default_workspace_cache_clear` drops the pools (not
    counted in its result), after which ``empty_cache`` releases them.  A tensor of a dropped pool stays valid and
    its blocks go back once it is freed."""
    device = torch.device("cuda", torch.cuda.current_device())
    cake_kcp.default_workspace_cache_clear()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_reserved(device)
    nbytes = 64 << 20
    pool = cake_kcp._KEPT_POOLS.setdefault(device.index, cake_kcp._MEM_POOL())
    with cake_kcp._USE_MEM_POOL(pool, device):
        freed = torch.empty(nbytes, dtype=torch.uint8, device=device)
        alive = torch.full((nbytes,), 7, dtype=torch.uint8, device=device)
    del freed
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    held = torch.cuda.memory_reserved(device)
    # the live pool keeps the freed tensor's block
    assert held >= base + 2 * nbytes, (base, held)
    assert cake_kcp.default_workspace_cache_clear() == 0
    assert not cake_kcp._KEPT_POOLS
    del pool
    torch.cuda.empty_cache()
    dropped = torch.cuda.memory_reserved(device)
    assert held - dropped >= nbytes, (held, dropped)
    assert int(alive[0]) == 7 and int(alive[-1]) == 7
    del alive
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    assert dropped - torch.cuda.memory_reserved(device) >= nbytes


@pytest.mark.skipif(
    not torch.cuda.is_available() or cake_kcp._MEM_POOL is None,
    reason="CUDA device and torch.cuda.MemPool required",
)
def test_kept_tensors_round_once_then_take_the_exact_shape() -> None:
    """:func:`cake_kcp._kept_tensors` rounds a key's first tensor (:func:`cake_kcp._kept_extent`, resized in place
    to its rows), serves a later tensor of no more rows with its exact shape from the freed block (no new reserved
    memory), rounds again while the key's previous tensor is alive and above the largest rounded extent, and keys
    on the trailing shape; :func:`cake_kcp.default_workspace_cache_clear` forgets the extents."""
    device = torch.device("cuda", torch.cuda.current_device())
    cake_kcp.default_workspace_cache_clear()
    pool = cake_kcp._MEM_POOL()
    row = 64 * 128 * 2  # bytes per row: blocks above 10 MiB, one segment each

    def kept(rows: int, heads: int = 64) -> torch.Tensor:
        with cake_kcp._USE_MEM_POOL(pool, device):
            t = cake_kcp._kept_tensors(
                device.index, ("w",), {"w": (rows, heads, 128)}, device
            )["w"]
        assert t.shape == (rows, heads, 128) and t.is_contiguous() and t._base is None
        return t

    first = kept(1000)
    assert first.untyped_storage().nbytes() == cake_kcp._kept_extent(1000) * row
    ptr = first.data_ptr()
    del first
    torch.cuda.synchronize(device)
    reserved = torch.cuda.memory_reserved(device)
    exact = kept(990)
    assert exact.untyped_storage().nbytes() == 990 * row and exact.data_ptr() == ptr
    assert torch.cuda.memory_reserved(device) == reserved
    # the previous tensor is alive: rounded, in another block
    other = kept(1010)
    assert other.untyped_storage().nbytes() == cake_kcp._kept_extent(1010) * row
    assert other.data_ptr() != ptr
    del exact, other
    # above the largest rounded extent: rounded again
    above = kept(1100)
    assert above.untyped_storage().nbytes() == cake_kcp._kept_extent(1100) * row
    del above
    # another trailing shape is another key: its first tensor is rounded
    heads = kept(990, heads=32)
    assert heads.untyped_storage().nbytes() == cake_kcp._kept_extent(990) * row // 2
    del heads
    assert len(cake_kcp._KEPT_EXTENTS) == 2
    cake_kcp.default_workspace_cache_clear()
    assert not cake_kcp._KEPT_EXTENTS and not cake_kcp._KEPT_LAST
    pool = None  # dropped: empty_cache releases its blocks
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


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
        before = cake_kcp.cake_kda_nondet_train_kcp_handoff_stats()
        x = torch.ones(4, requires_grad=True)
        with _PassThroughMode() if fwd_mode else contextlib.nullcontext():
            y = checkpoint(fn, x, use_reentrant=reentrant).sum()
        y.backward()
        after = cake_kcp.cake_kda_nondet_train_kcp_handoff_stats()
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
    part = cake_kcp_plan.cake_kda_nondet_train_kcp_partition(
        cu, 2, layout="zigzag_per_document"
    )
    ctx = cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(
        part, None, 0, "cpu", 4, sm_count=148
    )
    tb = cake_kcp.CakeKDANondetKcpWorkspace(local=False).tables(ctx, 4)
    for backward in (True, False):
        lazy = cake_kcp_plan.LAZY_POOLS + (
            () if backward else cake_kcp_plan.FWD_LAZY_POOLS
        )
        want = {n: s for n, s in tb.pool_sizes(backward).items() if n not in lazy}
        for pair in cake_kcp_plan.SHARED_POOLS:
            top = max((want[m] for m in pair if m in want), default=0)
            want.update((m, top) for m in pair if m in want)
        first = cake_kcp.CakeKDANondetKcpWorkspace._pool_need(tb, backward)
        assert first == want
        first[next(iter(first))] = -1  # a caller's change stays its own
        again = cake_kcp.CakeKDANondetKcpWorkspace._pool_need(tb, backward)
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
        ws = cake_kcp.CakeKDANondetKcpWorkspace(local=False)
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
            ws = cake_kcp.CakeKDANondetKcpWorkspace(local=False)
            ws._shared_eye = True
            assert ws._grow("eye", cake_kcp.TILE, other)
            far = ws.pool("eye")
            assert far is cake_kcp._identity_tile(other, torch.float32)
        torch.cuda.synchronize(other)
        assert far.device == other and far.data_ptr() != tiles[0].data_ptr()
        assert torch.equal(far.view(cake_kcp.D, cake_kcp.D).cpu(), eye.cpu())


def test_default_workspace_bypassed_under_capture() -> None:
    """A ``workspace=None`` call issued while its stream captures a CUDA graph takes a new workspace of its own
    (``cake_kcp._default_acquire``: ``"bypass"``, no key; its allocations belong to the graph's private pool) and
    neither takes nor adds a workspace the package keeps; outside the capture the same call takes a workspace the
    package keeps (``"miss"``, then ``"hit"`` after its release)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    device = torch.device("cuda", torch.cuda.current_device())
    part = cake_kcp_plan.cake_kda_nondet_train_kcp_partition([0, 1000, 1577], 1)
    ctx = cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(part, None, 0, device)
    args = (ctx, 4, device, cake_kcp.get_backend(), "light", "g", False, False)
    opts = cake_kcp._options({})
    cake_kcp.default_workspace_cache_clear()

    def release(ws) -> None:
        cake_kcp._default_release(types.SimpleNamespace(ws=ws, unwaited=0))

    def captured():
        side = torch.cuda.Stream(device)
        graph = torch.cuda.CUDAGraph()
        out = torch.zeros(1, device=device)
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph, stream=side):
            key = cake_kcp._default_key(*args, opts)
            ws, found = cake_kcp._default_acquire(*args, opts)
            out.add_(1.0)
        return key, ws, found

    bypass = cake_kcp.DEFAULT_WS_STATS["bypass"]
    key, ws, found = captured()
    assert key is None and found == "bypass" and ws._default_key is None
    assert not cake_kcp._DEFAULT_WS
    assert cake_kcp.DEFAULT_WS_STATS["bypass"] == bypass + 1
    release(ws)  # a bypassed call's workspace is never kept
    assert not cake_kcp._DEFAULT_WS
    kept, found = cake_kcp._default_acquire(*args, opts)
    assert found == "miss" and kept._default_key is not None
    release(kept)
    assert cake_kcp._DEFAULT_WS[kept._default_key] == [kept]
    key, ws, found = captured()
    assert key is None and found == "bypass" and ws is not kept
    assert cake_kcp._DEFAULT_WS[kept._default_key] == [kept]
    again, found = cake_kcp._default_acquire(*args, opts)
    assert found == "hit" and again is kept
    cake_kcp.default_workspace_cache_clear()


def test_handoff_byte_queries_and_counters_are_public() -> None:
    """The package root and the facade export ``cake_kda_nondet_train_kcp_handoff_bytes`` and
    ``cake_kda_nondet_train_kcp_handoff_stats``; the facade's ``cake_kda_nondet_train_kcp_saved_bytes`` takes
    ``checkpoint_handoff`` (keyword-only, default ``False``: the light set) and forwards it to this package, and the
    facade's ``cake_kda_nondet_train_kcp_handoff_bytes`` is the package's.  The light set has the deterministic
    package's bytes on the same partition.  The byte queries' ``checkpoint_handoff`` is the decision (a ``bool``):
    the entry's modes raise ``TypeError`` on the package and on the facade."""
    import importlib
    import inspect

    package = importlib.import_module(
        "flashinfer.experimental.cake.cake_kda_nondet_train_kcp"
    )
    assert (
        package.cake_kda_nondet_train_kcp_handoff_bytes
        is cake_kcp_plan.cake_kda_nondet_train_kcp_handoff_bytes
    )
    assert (
        package.cake_kda_nondet_train_kcp_handoff_stats
        is cake_kcp.cake_kda_nondet_train_kcp_handoff_stats
    )
    for name in (
        "cake_kda_nondet_train_kcp_handoff_bytes",
        "cake_kda_nondet_train_kcp_handoff_stats",
    ):
        assert name in package.__all__ and name in kcp_api.__all__, name
    p = inspect.signature(kcp_api.cake_kda_nondet_train_kcp_saved_bytes).parameters[
        "checkpoint_handoff"
    ]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is False
    params = inspect.signature(
        kcp_api.cake_kda_nondet_train_kcp_handoff_bytes
    ).parameters
    assert list(params) == ["ctx", "H", "saved_rows"]
    assert params["H"].kind is inspect.Parameter.KEYWORD_ONLY
    for fn in (
        kcp_api.cake_kda_nondet_train_kcp_handoff_bytes,
        kcp_api.cake_kda_nondet_train_kcp_saved_bytes,
    ):
        p = inspect.signature(fn).parameters["saved_rows"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default == "auto"
    stats = kcp_api.cake_kda_nondet_train_kcp_handoff_stats()
    assert stats == cake_kcp.cake_kda_nondet_train_kcp_handoff_stats()
    assert stats is not cake_kcp.HANDOFF_STATS
    assert set(stats) == set(cake_kcp.HANDOFF_STATS)
    det_plan = importlib.import_module(
        "flashinfer.experimental.cake.cake_kda_det_train_kcp.cake_kcp_plan"
    )
    cu = [0, 1000, 1700, 3000]
    for W in (1, 2, 4):
        part = kcp_api.cake_kda_nondet_train_kcp_partition(cu, W, "zigzag_per_document")
        det_part = det_plan.cake_kda_det_train_kcp_partition(
            cu, W, "zigzag_per_document"
        )
        for rank in range(W):
            ctx = kcp_api.cake_kda_nondet_train_kcp_build_context(
                part, None, rank, "cpu", sm_count=148
            )
            det_ctx = det_plan.cake_kda_det_train_kcp_build_context(
                det_part, None, rank, "cpu", sm_count=148
            )
            for H in (4, 64):
                hb = cake_kcp_plan.cake_kda_nondet_train_kcp_handoff_bytes(ctx, H=H)
                assert kcp_api.cake_kda_nondet_train_kcp_handoff_bytes(ctx, H=H) == hb
                for handoff in (False, True):
                    assert kcp_api.cake_kda_nondet_train_kcp_saved_bytes(
                        ctx, H=H, checkpoint_handoff=handoff
                    ) == cake_kcp_plan.cake_kda_nondet_train_kcp_saved_bytes(
                        ctx, H=H, save_transfer=False, checkpoint_handoff=handoff
                    )
                assert kcp_api.cake_kda_nondet_train_kcp_saved_bytes(
                    ctx, H=H
                ) == det_plan.cake_kda_det_train_kcp_saved_bytes(
                    det_ctx, H=H, save_transfer=False
                )
            for bad in ("auto", "on", "off", 1, None):
                with pytest.raises(TypeError, match="checkpoint_handoff"):
                    cake_kcp_plan.cake_kda_nondet_train_kcp_saved_bytes(
                        ctx, checkpoint_handoff=bad
                    )
                with pytest.raises(TypeError, match="checkpoint_handoff"):
                    kcp_api.cake_kda_nondet_train_kcp_saved_bytes(
                        ctx, checkpoint_handoff=bad
                    )


def test_facade_forwards_checkpoint_handoff(monkeypatch) -> None:
    """The facade's ``cake_kda_nondet_kcp`` takes ``checkpoint_handoff`` (keyword-only, default ``"auto"``) and
    forwards it to this package unchanged; a mode outside ``auto`` / ``on`` / ``off`` raises ``ValueError`` before
    any work (the stand-in operands are never read)."""
    import inspect

    p = inspect.signature(kcp_api.cake_kda_nondet_kcp).parameters["checkpoint_handoff"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default == "auto"
    t = torch.zeros(1)
    for bad in ("yes", None, True, "ON"):
        with pytest.raises(ValueError, match="checkpoint_handoff"):
            kcp_api.cake_kda_nondet_kcp(t, t, t, t, t, object(), checkpoint_handoff=bad)
    seen: list = []

    def recorder(*args, **kwargs):
        seen.append(kwargs)
        return "nd"

    monkeypatch.setattr(cake_kcp, "cake_kda_nondet_kcp", recorder)
    for mode in ("auto", "on", "off"):
        assert (
            kcp_api.cake_kda_nondet_kcp(
                t, t, t, t, t, object(), checkpoint_handoff=mode
            )
            == "nd"
        )
        assert seen[-1]["checkpoint_handoff"] == mode
    assert kcp_api.cake_kda_nondet_kcp(t, t, t, t, t, object()) == "nd"
    assert seen[-1]["checkpoint_handoff"] == "auto"


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
    tape = {
        n: torch.zeros(i + 1)
        for i, n in enumerate(cake_kcp.CakeKDANondetKcpSaved._TAPE)
    }
    light = cake_kcp.CakeKDANondetKcpSaved(
        S_in=s_in, c1_recv=c1, M_seg=None, ctx=None, **meta
    )
    kept = cake_kcp.CakeKDANondetKcpSaved(
        S_in=s_in, c1_recv=c1, M_seg=None, ctx=None, **tape, **meta
    )
    assert light.tape is None and light.flags() == (True, False, False)
    assert kept.flags() == (True, False, False, True)
    assert [t.numel() for t in kept.tensors()] == [3, 5, 1, 2, 3, 4, 5]
    assert kept.bytes - light.bytes == sum(t.numel() * 4 for t in tape.values())
    for saved in (light, kept):
        back = cake_kcp.CakeKDANondetKcpSaved.from_tensors(
            saved.tensors(), saved.flags(), saved.meta(), None
        )
        assert back.flags() == saved.flags() and back.meta() == saved.meta()
        assert (back.tape is None) == (saved.tape is None)
        if saved.tape is not None:
            assert all(back.tape[n] is saved.tape[n] for n in saved.tape)
    with pytest.raises(ValueError, match="flags"):
        cake_kcp.CakeKDANondetKcpSaved.from_tensors(
            light.tensors(), (True,), light.meta(), None
        )


def test_saved_set_rows_members_round_trip() -> None:
    """A kept rows set (``saved_rows`` "intra" / "wy") travels after the optional members (the carry plane
    included) in :data:`cake_kcp_plan.KCP_SAVED_ROWS` order, before the rstd, with no extra flag and one extra meta
    key ``saved_rows``; :attr:`rows` names it, :attr:`tape` stays ``None``.  A set without rows has exactly the
    tensors, flags and meta of before (no ``saved_rows`` key, ``u`` ``None``).  ``from_tensors`` reads every form
    back (the autograd node's round trip) and refuses a meta ``saved_rows`` outside ``"intra"`` / ``"wy"``."""
    meta = dict(
        version=cake_kcp.KCP_SAVED_VERSION,
        T_r=8,
        H=4,
        n_frag=1,
        P=2,
        Hg=4,
        scale=0.5,
        save_mode="light",
        save_transfer=True,
        offset_class="fla",
        exchange="a2a",
        precision_mode=0,
        transfer_walk="bf16x3_hw64",
    )
    s_in, c1, carry = (torch.zeros(n) for n in (3, 5, 7))
    members = {
        n: torch.zeros(i + 11) for i, n in enumerate(("aqk_int", "akk_int", "w", "u"))
    }
    light = cake_kcp.CakeKDANondetKcpSaved(
        S_in=s_in, c1_recv=c1, M_seg=None, carry=carry, ctx=None, **meta
    )
    assert light.u is None and light.saved_rows is None and light.rows is None
    assert "saved_rows" not in light.meta()
    assert [t.numel() for t in light.tensors()] == [3, 5, 7]
    for name, (names, _) in SAVED_ROWS_SETS.items():
        kept = dataclasses.replace(
            light, saved_rows=name, **{n: members[n] for n in names}
        )
        assert kept.tape is None and list(kept.rows) == list(names)
        assert all(kept.rows[n] is members[n] for n in names)
        assert kept.flags() == light.flags() == (True, False, True)
        assert kept.meta() == {**light.meta(), "saved_rows": name}
        assert all(
            a is b for a, b in zip(kept.tensors()[:3], light.tensors(), strict=True)
        )
        assert list(kept.tensors()[3:]) == [members[n] for n in names]
        assert kept.bytes - light.bytes == sum(members[n].nbytes for n in names)
        back = cake_kcp.CakeKDANondetKcpSaved.from_tensors(
            kept.tensors(), kept.flags(), kept.meta(), None
        )
        assert back.saved_rows == name and back.meta() == kept.meta()
        assert all(back.rows[n] is members[n] for n in names)
        assert all(a is b for a, b in zip(back.tensors(), kept.tensors(), strict=True))
        # the in-kernel Q / K L2 norm: the rstd stays the last tensor
        rstd = torch.zeros(2, 8, 4)
        l2n = dataclasses.replace(kept, qk_rstd=rstd, qk_l2norm=True)
        assert l2n.tensors()[-1] is rstd
        assert all(
            a is b for a, b in zip(l2n.tensors()[:-1], kept.tensors(), strict=True)
        )
        back = cake_kcp.CakeKDANondetKcpSaved.from_tensors(
            l2n.tensors(), l2n.flags(), l2n.meta(), None
        )
        assert back.qk_rstd is rstd and all(back.rows[n] is members[n] for n in names)
    for bad in ("tape", "none", "auto", "INTRA"):
        with pytest.raises(ValueError, match="saved_rows"):
            cake_kcp.CakeKDANondetKcpSaved.from_tensors(
                light.tensors() + tuple(members.values()),
                light.flags(),
                {**light.meta(), "saved_rows": bad},
                None,
            )


def test_handoff_bytes_query() -> None:
    """``cake_kda_nondet_train_kcp_handoff_bytes(..., saved_rows="tape")`` = ``NC_pad * H * 32768 + R * H * 768``; the
    hand-off's saved set with the tape set adds it and drops the carry plane."""
    cu = torch.tensor([0, 1000, 1700, 3000], dtype=torch.int64)
    for W in (1, 2, 4):
        part = cake_kcp_plan.cake_kda_nondet_train_kcp_partition(
            cu, W, layout="zigzag_per_document"
        )
        for rank in range(W):
            ctx = cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(
                part, None, rank, "cpu", 4, sm_count=148
            )
            for H in (4, 64):
                hb = cake_kcp_plan.cake_kda_nondet_train_kcp_handoff_bytes(
                    ctx, H=H, saved_rows="tape"
                )
                assert hb == ctx.NC_pad * H * 32768 + ctx.R * H * 768
                assert (
                    cake_kcp_plan.cake_kda_nondet_train_kcp_saved_bytes(
                        ctx, H=H, checkpoint_handoff=True, saved_rows="tape"
                    )
                    == cake_kcp_plan.cake_kda_nondet_train_kcp_saved_bytes(
                        ctx, H=H, saved_carry=False
                    )
                    + hb
                )


def test_saved_rows_bytes_query() -> None:
    """``saved_rows`` of the byte queries: ``"auto"`` (the default) is the query without it -- the light set
    without the hand-off, the ``"wy"`` set (KCP_HANDOFF_AUTO_ROWS) with it; ``"tape"`` the light set without the
    hand-off, the tape set's saved set with it; ``"none"`` is the light set for either; ``"intra"`` / ``"wy"`` are
    the light set (the carry plane kept at ``P > 1``) plus ``R * H * 256`` / ``R * H * 768`` for either; the
    hand-off query is the wy rows' bytes (``"auto"``), the tape set, 0, or the rows' bytes.  The facade forwards the
    keyword to the package.  Another value raises ``ValueError`` on both."""
    rows = cake_kcp_plan.KCP_SAVED_ROWS
    assert rows["tape"] == cake_kcp_plan.KCP_HANDOFF_BUFFERS and rows["none"] == ()
    assert {n: rows[n] for n in SAVED_ROWS_SETS} == {
        n: names for n, (names, _) in SAVED_ROWS_SETS.items()
    }
    want_bytes = {n: b for n, (_, b) in SAVED_ROWS_SETS.items()}
    assert want_bytes == cake_kcp_plan.KCP_SAVED_ROWS_BYTES
    # bytes per token-head: BF16 [R, H, 64] Aqk / Akk rows (2 * 64 * 2), BF16 [R, H, 128] w / u rows (2 * 128 * 2)
    assert cake_kcp_plan.KCP_SAVED_ROWS_MODES == ("auto", "none", "intra", "wy", "tape")
    assert cake_kcp_plan.KCP_SAVED_ROWS_DEFAULT == "auto"
    assert cake_kcp_plan.KCP_HANDOFF_AUTO_ROWS == "wy"
    for name in (
        "KCP_SAVED_ROWS",
        "KCP_SAVED_ROWS_BYTES",
        "KCP_SAVED_ROWS_DEFAULT",
        "KCP_SAVED_ROWS_MODES",
        "check_saved_rows",
        "kcp_saved_rows_form",
    ):
        assert name in cake_kcp_plan.__all__, name
    form = cake_kcp_plan.kcp_saved_rows_form
    got = [
        form(m, h)
        for m in ("auto", "none", "intra", "wy", "tape")
        for h in (False, True)
    ]
    assert got == [
        *("none", "wy"),
        *("none", "none"),
        *("intra", "intra"),
        *("wy", "wy"),
        *("none", "tape"),
    ]
    saved_bytes = cake_kcp_plan.cake_kda_nondet_train_kcp_saved_bytes
    handoff_bytes = cake_kcp_plan.cake_kda_nondet_train_kcp_handoff_bytes
    cu = torch.tensor([0, 1000, 1700, 3000], dtype=torch.int64)
    for W in (1, 2, 4):
        part = cake_kcp_plan.cake_kda_nondet_train_kcp_partition(
            cu, W, layout="zigzag_per_document"
        )
        for rank in range(W):
            ctx = cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(
                part, None, rank, "cpu", 4, sm_count=148
            )
            for H in (4, 12, 16, 32, 48, 64):
                light = saved_bytes(ctx, H=H)
                tape = saved_bytes(ctx, H=H, checkpoint_handoff=True, saved_rows="tape")
                hb = handoff_bytes(ctx, H=H, saved_rows="tape")
                assert handoff_bytes(ctx, H=H) == ctx.R * H * 768
                assert handoff_bytes(ctx, H=H, saved_rows="auto") == ctx.R * H * 768
                assert handoff_bytes(ctx, H=H, saved_rows="none") == 0
                for handoff in (False, True):
                    assert saved_bytes(
                        ctx, H=H, checkpoint_handoff=handoff, saved_rows="auto"
                    ) == (light + ctx.R * H * 768 if handoff else light)
                    assert saved_bytes(ctx, H=H, checkpoint_handoff=handoff) == (
                        saved_bytes(
                            ctx, H=H, checkpoint_handoff=handoff, saved_rows="auto"
                        )
                    )
                    assert saved_bytes(
                        ctx, H=H, checkpoint_handoff=handoff, saved_rows="tape"
                    ) == (tape if handoff else light)
                    assert (
                        saved_bytes(
                            ctx, H=H, checkpoint_handoff=handoff, saved_rows="none"
                        )
                        == light
                    )
                    for name, (_, per) in SAVED_ROWS_SETS.items():
                        assert (
                            saved_bytes(
                                ctx, H=H, checkpoint_handoff=handoff, saved_rows=name
                            )
                            == light + ctx.R * H * per
                        ), (W, rank, H, handoff, name)
                        assert kcp_api.cake_kda_nondet_train_kcp_saved_bytes(
                            ctx, H=H, checkpoint_handoff=handoff, saved_rows=name
                        ) == saved_bytes(
                            ctx,
                            H=H,
                            save_transfer=False,
                            checkpoint_handoff=handoff,
                            saved_rows=name,
                        )
                for name, (_, per) in SAVED_ROWS_SETS.items():
                    assert handoff_bytes(ctx, H=H, saved_rows=name) == ctx.R * H * per
                    assert (
                        kcp_api.cake_kda_nondet_train_kcp_handoff_bytes(
                            ctx, H=H, saved_rows=name
                        )
                        == ctx.R * H * per
                    )
                # the tape set drops the carry plane, the rows keep it (in ``light`` above)
                carry = light - saved_bytes(ctx, H=H, saved_carry=False)
                assert tape == light - carry + hb
            for bad in ("on", "off", "INTRA", "TAPE", None, 1, True):
                for query in (
                    saved_bytes,
                    handoff_bytes,
                    kcp_api.cake_kda_nondet_train_kcp_saved_bytes,
                    kcp_api.cake_kda_nondet_train_kcp_handoff_bytes,
                ):
                    with pytest.raises(ValueError, match="saved_rows"):
                        query(ctx, saved_rows=bad)


def test_saved_rows_forward_records_write_the_rows() -> None:
    """Every record a KCP forward launches for the rows sets binds the rows it keeps (a record that dropped a write
    would leave a kept row as unwritten memory and the backward would read it): the intra record of the forward's
    path (``cake_kcp.LOCAL_PATH_FWD``, launched with ``write_int = 1``) binds ``aqk_int`` / ``akk_int`` and the WY
    record (the plan's, or the direct record where the plan dispatches none, as ``cake_kcp._local_record``) binds
    ``w`` / ``u``.  Over every architecture, SM count and problem of the norm's capability grid
    (``cake_kcp_plan.QK_L2NORM_SMS`` x ``QK_L2NORM_GRID``), both gate modes and every norm mode the registry
    serves there (a refused mode has no forward to keep rows from).  CPU only: the registry and its select rules."""
    cake_jit = cake_kcp.cake_jit
    want = {
        "fwd_intra": set(cake_kcp_plan.KCP_SAVED_ROWS["intra"]),
        "wy_fwd": {"w", "u"},
    }
    problems = [
        (arch, sm, problem)
        for arch in cake_jit.ARCHES
        for sm in cake_kcp_plan.QK_L2NORM_SMS.get(arch, cake_kcp_plan.QK_L2NORM_SMS_ANY)
        for problem in cake_kcp_plan.QK_L2NORM_GRID
    ]
    modes = [
        (gate, l2n)
        for gate in ("g", "raw")
        for l2n in ("off", *cake_kcp_plan.QK_L2NORM_FORMS)
    ]
    seen: dict[tuple[str, str], set[str]] = {}
    for (arch, sm, (nc, docs, heads)), (gate, l2n), stage in itertools.product(
        problems, modes, want
    ):
        try:
            record = cake_jit.select_record(
                stage,
                num_chunks_pad=nc,
                num_seqs=docs,
                num_heads=heads,
                sm_count=sm,
                arch=arch,
                path=cake_kcp.LOCAL_PATH_FWD,
                gate=gate,
                **({} if l2n == "off" else {"l2n": l2n}),
            )
        except ValueError:
            continue  # the mode is not served on this problem: no forward runs it
        if record is None:
            record = cake_kcp.WY_DIRECT_RECORDS.get(stage)
        if record is None:
            continue  # no launch: the plan refuses the forward
        bound = set(cake_jit.MODULES[record]["logical"].values())
        where = (stage, record, arch, sm, nc, docs, heads, gate, l2n)
        assert want[stage] <= bound, where
        seen.setdefault((stage, arch), set()).add(record)
    # every architecture selected records of both stages (the check is not vacuous)
    assert set(seen) == {(st, a) for st in want for a in cake_jit.ARCHES}, seen


def test_saved_rows_gpu_rows_reach_p_above_one() -> None:
    """The GPU rows of ``saved_rows`` (``SAVED_ROWS_GPU_ROWS``) run the rows beside the carry plane: on every SM
    count of every target architecture (``cake_kcp_plan.QK_L2NORM_SMS``) some rank with tokens of some row plans
    ``P > 1`` (the saved set then keeps the rows and the carry plane, and the backward replays the K1 tape from the
    plane).  The GPU rows record ``P`` per rank; this test fails before a GPU run when the matrix lost that case.
    CPU only: the context's plan reads no device."""
    for arch, sms in cake_kcp_plan.QK_L2NORM_SMS.items():
        for sm in sms:
            segmented = []
            for row in SAVED_ROWS_GPU_ROWS:
                lengths, H, W, layout = row[:4]
                part, _ = _partition(lengths, W, layout)
                for rank in range(W):
                    ctx = cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(
                        part, None, rank, "cpu", 4, sm_count=sm
                    )
                    if int(ctx.T_r) > 0 and ctx.heads(H).P > 1:
                        segmented.append((_mode_row_id(row), rank))
            assert segmented, (arch, sm)


def test_saved_rows_keyword_of_the_functional_forward(monkeypatch) -> None:
    """The functional forward of the package and of the facade takes ``saved_rows`` (keyword-only, default ``"auto"``,
    the same parameter on both); a value outside ``auto`` / ``none`` / ``intra`` / ``wy`` / ``tape`` raises ``ValueError``
    before any work (the stand-in operands are never read), and so does ``"intra"`` / ``"wy"`` with the SPEC-3
    probe; the facade forwards the keyword unchanged."""
    import inspect

    for fn in (
        cake_kcp.cake_kda_nondet_train_kcp_forward,
        kcp_api.cake_kda_nondet_train_kcp_forward,
    ):
        p = inspect.signature(fn).parameters["saved_rows"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default == "auto"
    t = torch.zeros(1)
    ops = (t, t, t, t, t, object())
    for fwd in (
        cake_kcp.cake_kda_nondet_train_kcp_forward,
        kcp_api.cake_kda_nondet_train_kcp_forward,
    ):
        for bad in ("TAPE", "on", "WY", None, True):
            with pytest.raises(ValueError, match="saved_rows"):
                fwd(*ops, saved_rows=bad)
        for rows in ("intra", "wy"):
            for probe in ("record", "warn", "assert"):
                with pytest.raises(ValueError, match="spec3_probe"):
                    fwd(*ops, saved_rows=rows, spec3_probe=probe)
    seen: list = []

    def recorder(*_a, **kw):
        seen.append(kw)
        return t, None

    monkeypatch.setattr(cake_kcp, "cake_kda_nondet_train_kcp_forward", recorder)
    for mode in cake_kcp.KCP_SAVED_ROWS_MODES:
        kcp_api.cake_kda_nondet_train_kcp_forward(*ops, saved_rows=mode)
        assert seen[-1]["saved_rows"] == mode
    kcp_api.cake_kda_nondet_train_kcp_forward(*ops)
    assert seen[-1]["saved_rows"] == "auto"


def test_saved_rows_plain_reads_the_call_state() -> None:
    """The plain rule of ``saved_rows`` "intra" / "wy" (a call the checkpoint hand-off did not engage on): the rows
    with grad mode on, an input that requires grad (any of them; ``None`` inputs skipped), no saved-tensor hooks and
    outside a backward; nothing under no_grad and inference mode, without an input that requires grad, in both
    passes of a non-reentrant checkpoint and of a reentrant one (the hand-off's decision rules there), and under a
    caller's saved-tensor hooks (``saved_tensors_hooks``, ``save_on_cpu``), which alone count as ``refused_hooks``."""
    from torch.utils.checkpoint import checkpoint

    plain = cake_kcp._saved_rows_plain
    before = dict(cake_kcp.SAVED_ROWS_STATS)
    x = torch.ones(4, requires_grad=True)
    assert plain("intra", x) == "intra" and plain("wy", torch.ones(4), x) == "wy"
    # an absent input (A_log / dt_bias of the activated gate) is skipped; a later input alone decides
    assert plain("wy", torch.ones(4), None, None, x) == "wy"
    assert plain("wy", torch.ones(4), None, None) is None
    with torch.no_grad():
        assert plain("wy", x) is None
    with torch.inference_mode():
        assert plain("wy", x) is None
    seen: list = []

    def fn(t):
        seen.append(plain("wy", t))
        return t * t

    checkpoint(
        fn, torch.ones(4, requires_grad=True), use_reentrant=False
    ).sum().backward()
    assert seen == [None, None], seen
    seen.clear()
    checkpoint(
        fn, torch.ones(4, requires_grad=True), use_reentrant=True
    ).sum().backward()
    assert seen == [None, None], seen
    assert before == cake_kcp.SAVED_ROWS_STATS
    with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
        assert plain("wy", x) is None
    with torch.autograd.graph.save_on_cpu():
        assert plain("intra", x) is None
    want = {**before, "refused_hooks": before["refused_hooks"] + 2}
    assert want == cake_kcp.SAVED_ROWS_STATS


# the positions of the hand-off decision and the rows decision among CakeKDANondetKcpFunction.apply's arguments (q, k,
# v, g, beta, ctx, scale, save_mode, boundary_mode, workspace, options, handle, handoff, use_gate_in_kernel, A_log,
# dt_bias, lower_bound, use_qk_l2norm_in_kernel, saved_rows)
APPLY_HANDOFF, APPLY_ROWS, APPLY_ARGS = 12, 18, 19


def test_autograd_entry_resolves_saved_rows(monkeypatch) -> None:
    """``cake_kda_nondet_kcp`` decides the rows in the caller's frame by the table of its docstring and
    passes the result through ``CakeKDANondetKcpFunction.apply``: ``"auto"`` asks the hand-off and takes its
    engagement as the ``"wy"`` rows form (KCP_HANDOFF_AUTO_ROWS, no tape set), else keeps no rows; ``"tape"`` asks
    the hand-off and keeps its tape set where it engages, else no rows (the earlier ``"auto"``); ``"none"`` asks
    nothing and keeps nothing (``ValueError`` with ``checkpoint_handoff="on"``);
    ``"intra"`` / ``"wy"`` ask the hand-off, take its engagement as the rows form (no tape set) and otherwise the
    plain rule -- with and without a recompute handle, under every ``checkpoint_handoff``.  A bad value, and
    ``"intra"`` / ``"wy"`` with the SPEC-3 probe, raise ``ValueError`` before the decision."""
    t = torch.zeros(1)
    ops = (t, t, t, t, t, object())
    applied: list = []
    decided: list = []
    plain: list = []
    state = {"handoff": False, "plain": True}

    def decide(mode, *_a, kind=None):
        decided.append(mode)
        return state["handoff"]

    def plain_rule(rows, *inputs):
        # q, k, v, g, beta, A_log, dt_bias: the call's inputs that may require grad
        assert len(inputs) == 7
        plain.append(rows)
        return rows if state["plain"] else None

    def apply(*args):
        applied.append(args)
        return "out"

    monkeypatch.setattr(cake_kcp, "_handoff_decide", decide)
    monkeypatch.setattr(cake_kcp, "_saved_rows_plain", plain_rule)
    monkeypatch.setattr(cake_kcp.CakeKDANondetKcpFunction, "apply", apply)
    for engaged in (False, True):
        for plain_keeps in (False, True):
            state.update(handoff=engaged, plain=plain_keeps)
            for handle in (None, cake_kcp.CakeKDANondetKcpRecomputeHandle()):
                for mode in ("auto", "on", "off"):
                    for rows in cake_kcp.KCP_SAVED_ROWS_MODES:
                        applied.clear()
                        decided.clear()
                        plain.clear()
                        kw = dict(
                            recompute_handle=handle,
                            checkpoint_handoff=mode,
                            saved_rows=rows,
                        )
                        if rows == "none" and mode == "on":
                            with pytest.raises(ValueError, match="saved_rows='none'"):
                                cake_kcp.cake_kda_nondet_kcp(*ops, **kw)
                            assert not applied and not decided
                            continue
                        assert cake_kcp.cake_kda_nondet_kcp(*ops, **kw) == "out"
                        (args,) = applied
                        assert len(args) == APPLY_ARGS and args[11] is handle
                        case = (engaged, plain_keeps, handle, mode, rows)
                        if rows == "none":
                            assert decided == [] and plain == [], case
                            assert args[APPLY_HANDOFF] is False, case
                            assert args[APPLY_ROWS] is None, case
                        elif rows == "auto":
                            assert decided == [mode] and plain == [], case
                            assert args[APPLY_HANDOFF] is False, case
                            want = (
                                cake_kcp_plan.KCP_HANDOFF_AUTO_ROWS if engaged else None
                            )
                            assert args[APPLY_ROWS] == want, case
                        elif rows == "tape":
                            assert decided == [mode] and plain == [], case
                            assert args[APPLY_HANDOFF] is engaged, case
                            assert args[APPLY_ROWS] is None, case
                        else:
                            assert decided == [mode], case
                            assert args[APPLY_HANDOFF] is False, case
                            if engaged:
                                assert plain == [] and args[APPLY_ROWS] == rows, case
                            else:
                                assert plain == [rows], case
                                want = rows if plain_keeps else None
                                assert args[APPLY_ROWS] == want, case
    applied.clear()
    decided.clear()
    for bad in ("TAPE", None, "WY", 1):
        with pytest.raises(ValueError, match="saved_rows"):
            cake_kcp.cake_kda_nondet_kcp(*ops, saved_rows=bad)
    for rows in ("intra", "wy"):
        with pytest.raises(ValueError, match="spec3_probe"):
            cake_kcp.cake_kda_nondet_kcp(*ops, saved_rows=rows, spec3_probe="record")
    assert not applied and not decided


def test_autograd_node_threads_saved_rows(monkeypatch) -> None:
    """``CakeKDANondetKcpFunction`` takes the rows decision as its last argument: without a handle the forward runs
    the functional forward with ``saved_rows`` (no keyword when ``None``: the call is unchanged), with one
    :func:`cake_kcp._fwd_handle` with ``rows`` and no tape set; the backward returns one gradient slot per forward
    argument (``saved_rows``'s ``None`` included)."""
    import inspect

    class _Stop(Exception):
        pass

    reached: list = []

    def stop(name):
        def record(*_a, **kw):
            reached.append((name, kw))
            raise _Stop

        return record

    for name in ("_fwd_tape", "_fwd_handle", "cake_kda_nondet_train_kcp_forward"):
        monkeypatch.setattr(cake_kcp, name, stop(name))
    t = torch.zeros(1)
    args = (t, t, t, t, t, object())
    # scale, save_mode, boundary_mode, workspace, options
    head = (None, "light", "saved", None, {})
    gate = (False, None, None, None)
    for rows in (None, "intra", "wy"):
        for handle in (None, cake_kcp.CakeKDANondetKcpRecomputeHandle()):
            reached.clear()
            with pytest.raises(_Stop):
                cake_kcp.CakeKDANondetKcpFunction.apply(
                    *args, *head, handle, False, *gate, False, rows
                )
            ((name, kw),) = reached
            if handle is None:
                assert name == "cake_kda_nondet_train_kcp_forward", name
                if rows is None:
                    assert "saved_rows" not in kw, kw
                else:
                    assert kw["saved_rows"] == rows, kw
            else:
                assert name == "_fwd_handle", name
                assert kw["rows"] == rows and kw["tape"] is False, kw
    params = inspect.signature(cake_kcp.CakeKDANondetKcpFunction.forward).parameters
    assert list(params)[-1] == "saved_rows" and len(params) - 1 == APPLY_ARGS
    grads = tuple(torch.full((1,), float(i)) for i in range(5))
    monkeypatch.setattr(
        cake_kcp, "cake_kda_nondet_train_kcp_backward", lambda *a, **k: grads
    )
    fctx = types.SimpleNamespace(
        saved_tensors=(t,) * 5,
        kcp=(object(), None, "light", None, {}, None, (False, None), False),
    )
    back = cake_kcp.CakeKDANondetKcpFunction.backward(fctx, t)
    assert len(back) == APPLY_ARGS
    dq, dk, dv, dbeta, dg = grads
    assert back[:5] == (dq, dk, dv, dg, dbeta)
    assert all(b is None for b in back[5:])


def test_facade_forwards_saved_rows(monkeypatch) -> None:
    """The facade's ``cake_kda_nondet_kcp`` takes ``saved_rows`` (keyword-only, default ``"auto"``, the package
    entry's parameter) and forwards it unchanged; a bad value and ``"none"`` with ``checkpoint_handoff="on"`` raise
    ``ValueError`` before any work (the stand-in operands are never read)."""
    import inspect

    for fn in (kcp_api.cake_kda_nondet_kcp, cake_kcp.cake_kda_nondet_kcp):
        p = inspect.signature(fn).parameters["saved_rows"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default == "auto"
    t = torch.zeros(1)
    ops = (t, t, t, t, t, object())
    for bad in ("TAPE", "on", None, True):
        with pytest.raises(ValueError, match="saved_rows"):
            kcp_api.cake_kda_nondet_kcp(*ops, saved_rows=bad)
    with pytest.raises(ValueError, match="saved_rows='none'"):
        kcp_api.cake_kda_nondet_kcp(*ops, saved_rows="none", checkpoint_handoff="on")
    seen: list = []

    def recorder(*_a, **kwargs):
        seen.append(kwargs)
        return "nd"

    monkeypatch.setattr(cake_kcp, "cake_kda_nondet_kcp", recorder)
    for rows in cake_kcp.KCP_SAVED_ROWS_MODES:
        assert kcp_api.cake_kda_nondet_kcp(*ops, saved_rows=rows) == "nd"
        assert seen[-1]["saved_rows"] == rows
    assert kcp_api.cake_kda_nondet_kcp(*ops) == "nd"
    assert seen[-1]["saved_rows"] == "auto"
