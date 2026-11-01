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

Multi-GPU rows (the checkpoint rows ``CKPT_ROWS`` of ``test_cake_kda_train_kcp_dist.py``, the same launch forms):
``out``, the five gradients and ``dw`` (the checkpointed function returns ``out * w``) with the hand-off on are
bitwise those with it off and those of the direct pass, for ``use_reentrant`` False and True, with a recompute handle
(both tiers), on a lean arena and a per-call workspace; the forward's launches and every collective are unchanged;
the backward launches no intra program, WY or K1 tape walk; a NaN-filled tape set, poisoned free blocks and a
poisoned local arena (between the recompute and the backward) change no bit; the refusals (``"off"``, no checkpoint,
a blocked state such as a CUDA-graph capture, a selective-checkpoint policy, ``"auto"`` off the verified devices)
keep the light set in both passes and change no bit; a checkpointed forward with the hand-off retains nothing.
Single-process rows: the static rule, the phase probe and the saved set's tape members.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import tempfile
import traceback

import pytest
import torch
import torch.distributed as dist

from tests.experimental.test_cake_kda_train_kcp_dist import (
    CKPT_ROWS,
    _bound_rank_threads,
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
    rank: int, W: int, row, port: int, result_dir: str | None, local_id: int
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
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))
        dist.init_process_group("nccl", rank=rank, world_size=W, device_id=device)
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
    ):
        """One checkpointed (or direct) step of ``out * w``: the results and the LAST_CALL records of the original
        forward, the recompute (None without one) and the backward."""
        if poison_free:
            _poison(device)
        lv = leaves()
        wl = torch.ones_like(dout).requires_grad_(True)
        kw = {}
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
            out.backward(dout)
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

    def skips(recs_on, recs_off) -> bool:
        """The hand-off's backward launches none of SKIPPED; the forward's launches and the backward's collectives
        are the light path's."""
        on_b, off_b = recs_on["bwd"]["sequence"], recs_off["bwd"]["sequence"]
        ok = recs_on["orig"]["sequence"] == recs_off["orig"]["sequence"]
        ok = ok and recs_on["fwd"]["sequence"] == recs_off["fwd"]["sequence"]
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
    else:
        try:
            res_sac, recs_sac = run("on", context_fn=sac)
            flags["refuse_sac"] = (
                _same(res_sac, res_sac_off) and _same(res_sac, base) and light(recs_sac)
            )
        except Exception:  # noqa: BLE001 - recorded; the flag fails
            info["sac_error"] = traceback.format_exc()[-1500:]
            flags["refuse_sac"] = False
    verified = cake_kcp.KCP_HANDOFF_VERIFIED
    stats0 = cake_kcp.kcp_handoff_stats()
    cake_kcp.KCP_HANDOFF_VERIFIED = frozenset()
    try:
        res_un, recs_un = run("auto")
    finally:
        cake_kcp.KCP_HANDOFF_VERIFIED = verified
    stats1 = cake_kcp.kcp_handoff_stats()
    want_refusal = (
        "refused_heads" if H not in cake_kcp.KCP_HANDOFF_HEADS else "refused_unverified"
    )
    flags["refuse_auto_unverified"] = (
        _same(res_un, base)
        and light(recs_un)
        and stats1[want_refusal] - stats0[want_refusal] == 2
    )
    # "auto" on the shipped set: engaged exactly where the key and the head count are verified
    res_auto, recs_auto = run("auto")
    auto_on = key in cake_kcp.KCP_HANDOFF_VERIFIED and H in cake_kcp.KCP_HANDOFF_HEADS
    info["auto_engages"] = auto_on
    flags["auto_bitwise"] = _same(res_auto, base)
    flags["auto_rule"] = engaged(recs_auto) if auto_on else light(recs_auto)
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
            os.path.join(result_dir, "handoff-" + _row_id(row) + ".json"), "w"
        ) as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _spawn_entry_handoff(rank: int, W: int, row, port: int, result_dir: str) -> None:
    try:
        _bound_rank_threads(W)
        report = _run_rank_handoff(rank, W, row, port, result_dir, rank)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("row", CKPT_ROWS, ids=_row_id)
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
            int(os.environ.get("MASTER_PORT", "29500")),
            os.environ.get("KCP_TEST_RESULT_DIR"),
            int(os.environ.get("SLURM_LOCALID", "0")),
        )
        assert report["pass"], report
        return
    _skip_unless_runnable(W)
    with tempfile.TemporaryDirectory(prefix="cake_kcp_handoff_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        _start_ranks(_spawn_entry_handoff, W, row, result_dir)
        with open(os.path.join(result_dir, "handoff-" + _row_id(row) + ".json")) as fh:
            report = json.load(fh)
    assert report["pass"], report


# ---------------------------------------------------------------------------------------------------------------
# single process: the static rule, the phase probe, the saved set


def test_handoff_static_rule_and_modes() -> None:
    """The modes, the shipped keys, the refusal reasons of the static rule and the mode check of the entry."""
    assert cake_kcp.KCP_HANDOFF_MODES == ("auto", "on", "off")
    assert cake_kcp.KCP_HANDOFF_DEFAULT == "auto"
    assert cake_kcp.KCP_HANDOFF_VERIFIED is cake_kcp_plan.KCP_HANDOFF_VERIFIED
    assert all(
        isinstance(a, str) and isinstance(s, int) and isinstance(t, str)
        for a, s, t in cake_kcp.KCP_HANDOFF_VERIFIED
    )
    assert cake_kcp_plan.KCP_HANDOFF_BUFFERS == cake_kcp.SavedKcp._TAPE
    with pytest.raises(ValueError, match="checkpoint_handoff"):
        cake_kcp._handoff_mode("yes")
    opts = cake_kcp._options({})
    v_cpu = torch.zeros(8, 4, 128)
    assert cake_kcp._handoff_refusal("on", opts, "replay", v_cpu) == "boundary"
    assert (
        cake_kcp._handoff_refusal(
            "on", cake_kcp._options({"graph": "replay"}), "saved", v_cpu
        )
        == "graph"
    )
    assert (
        cake_kcp._handoff_refusal(
            "on", cake_kcp._options({"spec3_probe": "record"}), "saved", v_cpu
        )
        == "spec3"
    )
    assert cake_kcp._handoff_refusal("on", opts, "saved", v_cpu) == "device"
    # outside any checkpoint pass there is no request
    x = torch.zeros(2, requires_grad=True)
    assert cake_kcp._handoff_phase(x) is None
    with torch.no_grad():
        assert cake_kcp._handoff_phase(x) is None


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
