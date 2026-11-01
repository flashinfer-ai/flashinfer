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

Multi-GPU rows of the non-deterministic context-parallel (KCP) KDA training route (``cake_kda_train_kcp.cake_kcp``,
``deterministic=False``): ``W`` ranks over NCCL, each holding the fragments of one partition of a packed batch, compared with the exact FP64 per-document
recurrence of the same operands (``tests/test_helpers/cake_kda_det_train_reference``) and with one another (saved ==
replay, repeat, autograd == explicit, checkpoint == direct, document-alone == slice, graph replay == eager with
every rank capturing the same calls) and with the deterministic package ``cake_kda_det_train_kcp`` on the same operands
and context (``out``, ``dq``, ``dk``, ``dv``, ``dbeta`` bitwise; ``dg`` within the same budget and at most 1.1 times
the deterministic error against the FP64 reference).

Two launch forms: ``torch.multiprocessing.spawn`` (``pytest``: skipped when fewer than ``W`` CUDA devices are
visible or the KCP programs are not exported yet) and one process per Slurm task (``SLURM_PROCID`` /
``SLURM_NTASKS`` set: the test body runs in the calling process, the rendezvous from ``MASTER_ADDR`` /
``MASTER_PORT``).
"""

from __future__ import annotations

import gc
import json
import os
import socket
import tempfile
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.test_helpers.cake_kda_train_kcp_saved import (
    carry_slots,
    poison_unwritten_carry,
    written_tensors,
)
from tests.test_helpers.cake_kda_det_train_reference import (
    GRAD_NAMES,
    kda_exact_with_grads,
    make_inputs,
)
from tests.test_helpers.cake_package import cake_package

cake_kcp = cake_package("cake_kda_train_kcp", "cake_kcp")
cake_kcp_plan = cake_package("cake_kda_train_kcp", "cake_kcp_plan")
cake_jit = cake_package("cake.cake_kda_train", "cake_jit")
cake_kcp_jit = cake_package("cake_kda_train_kcp", "cake_kcp_jit")
# the deterministic package: the baseline of the comparison rows (imported by the test, never by the package)
det_kcp = cake_package("cake_kda_det_train_kcp", "cake_kcp")

PACKAGE_DIR = Path(cake_kcp_jit.__file__).resolve().parent
TENSORS = ("out",) + GRAD_NAMES
# The outputs the non-deterministic package gives bitwise equal to the deterministic one; ``dg`` (the S1 pair moves
# its ``q * dq`` term into the GLM intra program) is held to the same budget against the FP64 reference and to at most
# DG_ERROR_RATIO times the deterministic package's error.
BITWISE_DET = tuple(n for n in TENSORS if n != "dg")
DG_ERROR_RATIO = 1.1
# graph="replay" over four forward + backward calls of one key: the backward's key sees the saved tensors, which
# change once (the eager forward's -> the captured forward's static ones), so it captures one call after the forward
GRAPH_EVENTS = [
    ["eager", "eager"],
    ["capture", "eager"],
    ["replay", "capture"],
    ["replay", "replay"],
]
ATOL = RTOL = 1e-2
SCALE = cake_kcp.SCALE

# (lengths, H, W, layout): small packings with boundaries inside chunks, a document crossing >= 3 ranks (zigzag
# and per-document zigzag), empty / one-token fragments and an explicit ragged table with a zero-token rank.
ROWS = [
    ((100, 640, 77, 1000, 129), 4, 2, "contiguous"),
    ((100, 640, 77, 1000, 129), 4, 4, "contiguous"),
    ((3000,), 4, 4, "zigzag"),
    ((3000,), 4, 4, "zigzag_per_document"),
    ((2048, 5, 3000, 1), 4, 4, "zigzag_per_document"),
    ((1200 + 900 + 900,), 4, 4, "ragged:1200,0,900,900"),
    ((5000, 1003), 1, 4, "ragged:1,2,4997,1003"),
    # 16 heads: the S1 pair on sm_100a / sm_107a, the deterministic pair on sm_103a (the routing table's boundary)
    ((100, 640, 77, 1000, 129), 16, 2, "zigzag_per_document"),
]


def _row_id(row) -> str:
    lengths, H, W, layout = row
    return "s" + "_".join(map(str, lengths)) + f"-h{H}-w{W}-{layout.split(':')[0]}"


def _programs_exported() -> bool:
    """Every shipped KCP record of ``cake_kcp_jit.MODULES`` has a library in the
    package's ``build.json`` -- else the kernel path cannot run."""
    manifest = PACKAGE_DIR / "build.json"
    if not manifest.exists():
        return False
    try:
        with open(manifest) as fh:
            libraries = {lib["name"] for lib in json.load(fh)["libraries"]}
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return all(
        any(
            cake_kcp_jit.library_name(name, arch) in libraries
            for arch in record["arches"]
        )
        for name, record in cake_kcp_jit.MODULES.items()
    )


def _partition(lengths, W: int, layout: str):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    if layout.startswith("ragged:"):
        t_r = [int(x) for x in layout.split(":")[1].split(",")]
        assert len(t_r) == W and sum(t_r) == cu[-1]
        # contiguous ragged split of the global buffer, cut at document boundaries, in local order
        nested = [[] for _ in range(W)]
        pos = 0
        for r, n in enumerate(t_r):
            end = pos + n
            d = 0
            while d < len(cu) - 1 and cu[d + 1] <= pos:
                d += 1
            p = pos
            while p < end and d < len(cu) - 1:
                e = min(end, cu[d + 1])
                if e > p:
                    nested[r].append((d, p - cu[d], e - p))
                p = e
                d += 1
            pos = end
        return cake_kcp_plan.partition_kcp(cu, W, fragments=nested), cu
    return cake_kcp_plan.partition_kcp(cu, W, layout=layout), cu


def _local_slices(part, rank: int, cu: list[int], tensors: dict) -> dict:
    """This rank's local buffer (fragments in slot order) of every global operand."""
    out = {}
    for name, t in tensors.items():
        if name == "cu_seqlens":
            continue
        parts = [
            t[cu[f.doc] + f.off : cu[f.doc] + f.off + f.length]
            for f in part.fragments[rank]
        ]
        out[name] = torch.cat(parts, 0) if parts else t[:0].clone()
    return out


def _to_global(
    part, rank_tensors: list[torch.Tensor], cu: list[int], total: int
) -> torch.Tensor:
    ref = rank_tensors[0]
    g = torch.empty((total,) + tuple(ref.shape[1:]), dtype=ref.dtype, device=ref.device)
    for r, t in enumerate(rank_tensors):
        pos = 0
        for f in part.fragments[r]:
            if f.length:
                g[cu[f.doc] + f.off : cu[f.doc] + f.off + f.length] = t[
                    pos : pos + f.length
                ]
            pos += f.length
    return g


def _run_rank(rank: int, W: int, row, port: int, result_dir: str | None) -> dict:
    """The body of one rank: the KCP forward / backward in every mode, gathered to rank 0 and compared there."""
    lengths, H, _w, layout = row
    device = torch.device("cuda", rank % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    if W > 1 and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))
        dist.init_process_group("nccl", rank=rank, world_size=W, device_id=device)
    group = dist.group.WORLD if W > 1 else None
    part, cu = _partition(lengths, W, layout)
    ctx = cake_kcp_plan.build_context_kcp(part, group, rank, device)
    total = cu[-1]
    glob = make_inputs(
        lengths, H, device
    )  # the same seeded operands on every rank; each takes its slices
    ops = _local_slices(part, rank, cu, glob)
    q, k, v, g, beta, dout = (ops[n] for n in ("q", "k", "v", "g", "beta", "dout"))
    ws = cake_kcp.WorkspaceKcp()

    def saved_pass(**opts):
        out, saved = cake_kcp.chunk_kda_fwd_kcp(
            q, k, v, g, beta, ctx, workspace=ws, **opts, deterministic=False
        )
        # ``boundary_mode`` is a forward keyword: the backward replays the forward whenever ``saved`` is None
        bwd_opts = {key: val for key, val in opts.items() if key != "boundary_mode"}
        dq, dk, dv, dbeta, dg = cake_kcp.chunk_kda_bwd_kcp(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            dout,
            saved=saved,
            workspace=ws,
            **bwd_opts,
            deterministic=False,
        )
        return {"out": out, "dq": dq, "dk": dk, "dv": dv, "dg": dg, "dbeta": dbeta}

    def autograd_pass(ckpt: bool):
        leaves = [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)]

        def fn(*a):
            return cake_kcp.chunk_kda_kcp(
                *a, ctx, None, "light", "saved", ws, deterministic=False
            )

        if ckpt:
            from torch.utils.checkpoint import checkpoint

            out = checkpoint(fn, *leaves, use_reentrant=False)
        else:
            out = fn(*leaves)
        out.backward(dout)
        return {
            "out": out.detach(),
            **{
                f"d{n}": lf.grad
                for n, lf in zip(("q", "k", "v", "g", "beta"), leaves, strict=True)
            },
        }

    # the probe runs once per backend, on request (spec3_probe="record"; "off" is the default): a row after an
    # earlier probing backward in this process (the Slurm per-task form runs every row in one process) finds it
    # spent; the later passes run the default and must match this one bitwise
    pending = bool(getattr(cake_kcp.get_backend(), "_spec3_pending", False))
    res = saved_pass(spec3_probe="record")
    spec3 = (cake_kcp.LAST_CALL.get("bwd") or {}).get(
        "spec3"
    )  # SPEC-3 debug assertion of the first saved-mode backward
    backend = cake_kcp.get_backend()
    res_replay = saved_pass(boundary_mode="replay")
    res_repeat = saved_pass()
    res_auto = autograd_pass(False)
    res_ckpt = autograd_pass(True)
    # the deterministic package on the same operands and the same context (the context holds no mode and serves
    # both packages): out, dq, dk, dv and dbeta bitwise; dg compared with the FP64 reference on rank 0
    ws_det = det_kcp.WorkspaceDetKcp()
    out_d, saved_d = det_kcp.chunk_kda_det_fwd_kcp(
        q, k, v, g, beta, ctx, workspace=ws_det
    )
    dq_d, dk_d, dv_d, dbeta_d, dg_d = det_kcp.chunk_kda_det_bwd_kcp(
        q, k, v, g, beta, ctx, dout, saved=saved_d, workspace=ws_det
    )
    res_det = {
        "out": out_d,
        "dq": dq_d,
        "dk": dk_d,
        "dv": dv_d,
        "dg": dg_d,
        "dbeta": dbeta_d,
    }
    del ws_det, saved_d
    # the shipped default keeps the owner's c1 receive tensor for the backward (save_transfer=True); the recompute
    # form (save_transfer=False) re-sends it and must be bitwise the same
    res_recompute = saved_pass(save_transfer=False)
    # the forward-only workspace (the KCP forward layout, the lazy pools) against this backward workspace: a saving
    # forward on an empty workspace holds the predicted bytes and none of the lazy pools, a forward without a saved
    # set adds their predicted bytes, both give this workspace's out (and the written bytes of its saved set)
    # bitwise, and the backward that upgrades the forward-only workspace gives the same gradients; workspace=None (a
    # forward-only arena per call) through autograd too
    out_b, saved_b = cake_kcp.chunk_kda_fwd_kcp(
        q, k, v, g, beta, ctx, workspace=ws, deterministic=False
    )
    ws_f, ws_n = cake_kcp.WorkspaceKcp(), cake_kcp.WorkspaceKcp()
    out_f, saved_f = cake_kcp.chunk_kda_fwd_kcp(
        q, k, v, g, beta, ctx, workspace=ws_f, deterministic=False
    )
    out_n, _ = cake_kcp.chunk_kda_fwd_kcp(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        workspace=ws_n,
        save_for_backward=False,
        deterministic=False,
    )
    pred_f = cake_kcp_plan.kda_workspace_bytes_kcp(ctx, H=H, backward=False)
    lazy_f = cake_kcp_plan.kcp_arena_layout(ctx, H=H, backward=False)["lazy"]
    fwd_only_bytes = (
        ws_f.bytes == pred_f
        and not any(ws_f.has(n) for n in cake_kcp_plan.FWD_LAZY_POOLS)
        and ws_n.bytes == pred_f + lazy_f
        and (ws_f.local is None or ws_f.local.fwd_layout == "fwd_kcp")
    )
    # without save_transfer a forward with a saved set reads the c1 receive and transfer staging pools (that option's
    # first state) and a forward without a saved set then adds the carry plane; the ring exchange (checked below)
    # never allocates the c1 receive pool
    ws_t = cake_kcp.WorkspaceKcp()
    out_t, _ = cake_kcp.chunk_kda_fwd_kcp(
        q, k, v, g, beta, ctx, workspace=ws_t, save_transfer=False, deterministic=False
    )
    pred_t = cake_kcp_plan.kda_workspace_bytes_kcp(
        ctx, H=H, backward=False, save_transfer=False
    )
    lay_t = cake_kcp_plan.kcp_arena_layout(
        ctx, H=H, backward=False, save_transfer=False
    )
    fwd_only_bytes = fwd_only_bytes and ws_t.bytes == pred_t and not ws_t.has("carry")
    cake_kcp.chunk_kda_fwd_kcp(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        workspace=ws_t,
        save_for_backward=False,
        deterministic=False,
    )
    fwd_only_bytes = (
        fwd_only_bytes and ws_t.bytes == pred_t + lay_t["lazy"] == pred_f + lazy_f
    )
    # the carry plane slots the forward never writes (the extra slots beyond n_in when n_out > n_in, the slots of a
    # zero-length fragment) hold allocator bytes: the saved sets compare on the written bytes, and NaN in the others
    # must not reach a gradient.  Each rank's NaN bytes go to the report (0 where the plan leaves no slot unwritten).
    # The slot invariants differ per rank: a rank that breaks one fails this flag with the reason, instead of raising
    # before the backward's collectives and leaving its peers waiting
    try:
        carry_slots(saved_b)
        carry_poisoned = poison_unwritten_carry(saved_f)
    except AssertionError as exc:
        carry_poisoned = f"carry_slots: {exc!r}"
    dq, dk, dv, dbeta, dg = cake_kcp.chunk_kda_bwd_kcp(
        q, k, v, g, beta, ctx, dout, saved=saved_f, workspace=ws_f, deterministic=False
    )
    res_fwd_only = {
        "out": out_f,
        "dq": dq,
        "dk": dk,
        "dv": dv,
        "dg": dg,
        "dbeta": dbeta,
    }
    fwd_only_eq = (
        isinstance(carry_poisoned, int)
        and torch.equal(out_b, res["out"])
        and torch.equal(out_n, res["out"])
        and torch.equal(out_t, res["out"])
        and saved_f.flags() == saved_b.flags()
        and all(
            torch.equal(a, b)
            for a, b in zip(
                written_tensors(saved_f), written_tensors(saved_b), strict=True
            )
        )
        and all(torch.equal(res[n], res_fwd_only[n]) for n in TENSORS)
    )
    del ws_f, ws_n, ws_t, out_b, saved_b, out_f, saved_f, out_n, out_t, res_fwd_only
    leaves = [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)]
    out_0 = cake_kcp.chunk_kda_kcp(*leaves, ctx, deterministic=False)
    out_0.backward(dout)
    ws_none_eq = torch.equal(out_0.detach(), res["out"]) and all(
        torch.equal(lf.grad, res[f"d{n}"])
        for n, lf in zip(("q", "k", "v", "g", "beta"), leaves, strict=True)
    )
    del leaves, out_0
    ring_ok = True
    if (
        W > 1 and int(ctx.n_d_max) - 1 <= cake_kcp.RING_MAX_HOPS
    ):  # gate 5: the ring path is bitwise the a2a path; the ring receives no c1 tensor, so recompute form only
        res_ring = saved_pass(exchange="ring", save_transfer=False)
        ring_ok = all(torch.equal(res_recompute[n], res_ring[n]) for n in TENSORS)
        # a forward-only ring workspace: no c1 receive pool, the hop pools on first use
        ws_r = cake_kcp.WorkspaceKcp()
        out_r, _ = cake_kcp.chunk_kda_fwd_kcp(
            q, k, v, g, beta, ctx, workspace=ws_r, exchange="ring", deterministic=False
        )
        ring_ok = ring_ok and torch.equal(out_r, res["out"])
        fwd_only_bytes = (
            fwd_only_bytes
            and not ws_r.has("c1_recv")
            and ws_r.bytes == pred_t - lay_t["pool.c1_recv"] + lay_t["ring"]
        )
        del ws_r, out_r
    # every rank (a rank without tokens included) captures and replays the same calls: a rank that stayed eager
    # while the others captured would run its collectives outside the graphs of its peers
    cake_kcp.graph_cache_clear()
    events = []
    try:
        for _ in range(len(GRAPH_EVENTS)):
            res_graph = {n: t.clone() for n, t in saved_pass(graph="replay").items()}
            events.append(
                [
                    str(cake_kcp.LAST_CALL["fwd"].get("graph_event")),
                    str(cake_kcp.LAST_CALL["bwd"].get("graph_event")),
                ]
            )
        torch.cuda.synchronize(device)
    except BaseException as exc:
        # the failed call's frames still reference its cache entry and graph: release them before the cleanup
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        # a cached graph holding a NCCL collective blocks the process group's destruction
        cake_kcp.graph_cache_clear()
    flags = {
        "saved_eq_replay": all(torch.equal(res[n], res_replay[n]) for n in TENSORS),
        "repeat": all(torch.equal(res[n], res_repeat[n]) for n in TENSORS),
        "autograd_eq_saved": all(torch.equal(res[n], res_auto[n]) for n in TENSORS),
        "ckpt_eq_direct": all(torch.equal(res_auto[n], res_ckpt[n]) for n in TENSORS),
        "save_transfer_eq": all(torch.equal(res[n], res_recompute[n]) for n in TENSORS),
        "det_eq_but_dg": all(torch.equal(res[n], res_det[n]) for n in BITWISE_DET),
        "fwd_only_eq": fwd_only_eq,
        "fwd_only_bytes": fwd_only_bytes,
        "ws_none_eq": ws_none_eq,
        "ring_eq_a2a": ring_ok,
        "graph_replay": events == GRAPH_EVENTS
        and all(torch.equal(res[n], res_graph[n]) for n in TENSORS),
        # a backend with local stages whose probe was pending must have run it (spec3_probe="record" on the
        # first pass); a rank without tokens launches no local stage, so it has nothing to probe.  The probe's row
        # comparison (the forward's intra prep against the backward's prep program) is reported in spec3_by_rank,
        # not gated: the requirement of record is saved_eq_replay (both backward modes run the same local programs)
        "spec3": (not backend.local_stages)
        or not pending
        or (spec3 is None and q.shape[0] == 0)
        or bool(spec3 and spec3.get("checked")),
    }
    gathered = {}
    for n in TENSORS:
        if W > 1:
            objs = [None] * W
            dist.all_gather_object(objs, res[n].cpu())
        else:
            objs = [res[n].cpu()]
        gathered[n] = objs
    if W > 1:
        det_dg = [None] * W
        dist.all_gather_object(det_dg, res_det["dg"].cpu())
    else:
        det_dg = [res_det["dg"].cpu()]
    flag_list = [None] * W
    spec3_list = [None] * W
    events_list = [None] * W
    carry_list = [None] * W
    if W > 1:
        dist.all_gather_object(flag_list, flags)
        dist.all_gather_object(spec3_list, spec3)
        dist.all_gather_object(events_list, events)
        dist.all_gather_object(carry_list, carry_poisoned)
    else:
        flag_list = [flags]
        spec3_list = [spec3]
        events_list = [events]
        carry_list = [carry_poisoned]
    report = {
        "rank": rank,
        "world": W,
        "pass": True,
        "flags": {},
        "metrics": {},
        "backend": backend.name,
        "backend_path": getattr(backend, "path", None),
        "spec3_by_rank": spec3_list,
        "graph_events_by_rank": events_list,
        "carry_poisoned_bytes_by_rank": carry_list,
        "metrics_det": {},
    }
    for key in flags:
        report["flags"][key] = all(f[key] for f in flag_list)
        report["pass"] &= report["flags"][key]
    if rank == 0:
        cu_t = torch.tensor(cu, dtype=torch.int32)
        ref_out, ref_grads = kda_exact_with_grads(
            glob["q"].cpu(),
            glob["k"].cpu(),
            glob["v"].cpu(),
            glob["g"].cpu(),
            glob["beta"].cpu(),
            glob["dout"].cpu(),
            cu_t,
            SCALE,
        )
        refs = {"out": ref_out, **ref_grads}
        for n in TENSORS:
            cand = _to_global(part, gathered[n], cu, total).to(torch.float64)
            ref = refs[n]
            excess = (cand - ref).abs() - (ATOL + RTOL * ref.abs())
            mism = int((excess > 0).sum())
            err = cand - ref
            rel = float(
                err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-300)
            )
            report["metrics"][n] = {
                "mismatch": mism,
                "rel_rms": rel,
                "max_abs": float(err.abs().max()),
            }
            report["pass"] &= mism == 0 and bool(torch.isfinite(cand).all())
        # dg of the deterministic package against the same reference: the non-deterministic error is at most
        # DG_ERROR_RATIO times it (rel_rms and max_abs; an exact zero error on both sides passes)
        ref = refs["dg"]
        cand_d = _to_global(part, det_dg, cu, total).to(torch.float64)
        err_d = cand_d - ref
        det_m = {
            "rel_rms": float(
                err_d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-300)
            ),
            "max_abs": float(err_d.abs().max()),
        }
        report["metrics_det"]["dg"] = det_m
        nd_m = report["metrics"]["dg"]
        report["flags"]["dg_error_vs_det"] = all(
            nd_m[key] <= DG_ERROR_RATIO * det_m[key] for key in ("rel_rms", "max_abs")
        )
        report["pass"] &= report["flags"]["dg_error_vs_det"]
    if W > 1:
        verdict = [None] * W
        dist.all_gather_object(verdict, report["pass"])
        report["pass"] = all(verdict)
        dist.barrier()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, _row_id(row) + ".json"), "w") as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _spawn_entry(rank: int, W: int, row, port: int, result_dir: str) -> None:
    """One spawned rank: the verdict travels through rank 0's report file (no synchronization primitive is
    pickled into the children); a failing rank raises so the parent sees its report."""
    try:
        report = _run_rank(rank, W, row, port, result_dir)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            cake_kcp.graph_cache_clear()
            dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_ranks(entry, W: int, row, result_dir: str) -> None:
    """``entry`` on ``W`` spawned ranks with a fresh rendezvous port; a port another process takes between
    :func:`_free_port` and rank 0's bind (``EADDRINUSE``) is retried on a new port, three tries in all."""
    for attempt in range(3):
        try:
            mp.start_processes(
                entry,
                args=(W, row, _free_port(), result_dir),
                nprocs=W,
                join=True,
                start_method="spawn",
            )
            return
        except mp.ProcessRaisedException as error:
            if "EADDRINUSE" not in str(error) or attempt == 2:
                raise


def _skip_unless_runnable(W: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.device_count() < W:
        pytest.skip(f"{W} CUDA devices required, {torch.cuda.device_count()} visible")
    if not _programs_exported():
        pytest.skip("the KCP programs are not exported in the package's build.json yet")
    if cake_jit.device_arch(torch.cuda.current_device()) not in cake_jit.ARCHES:
        pytest.skip("the current device's architecture is not covered by the package")


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_multi_gpu_rows_match_the_exact_reference(row) -> None:
    lengths, H, W, layout = row
    if (
        os.environ.get("SLURM_PROCID") is not None
        and int(os.environ.get("SLURM_NTASKS", "1")) == W
    ):
        # one process per Slurm task: the body runs here (rendezvous from MASTER_ADDR / MASTER_PORT)
        if not _programs_exported():
            pytest.skip(
                "the KCP programs are not exported in the package's build.json yet"
            )
        report = _run_rank(
            int(os.environ["SLURM_PROCID"]),
            W,
            row,
            int(os.environ.get("MASTER_PORT", "29500")),
            os.environ.get("KCP_TEST_RESULT_DIR"),
        )
        assert report["pass"], report
        return
    _skip_unless_runnable(W)
    with tempfile.TemporaryDirectory(prefix="cake_kcp_dist_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        _start_ranks(_spawn_entry, W, row, result_dir)
        with open(os.path.join(result_dir, _row_id(row) + ".json")) as fh:
            report = json.load(fh)
    assert report["pass"], report


# (lengths, H, W, layout) of the host-synchronization rows: CP2 / CP4 / CP8, every layout, H 4 / 64, one-token
# documents and a document spanning >= 3 ranks; (2000, 1000) at CP4 is a ring row with a three-fragment chain (a hop-2
# chain launch)
UPLOAD_ROWS = [
    ((100, 640, 77, 1000, 129), 4, 2, "contiguous"),
    ((2000, 1000), 4, 4, "contiguous"),
    ((3000,), 64, 4, "zigzag"),
    ((2048, 5, 3000, 1), 4, 4, "zigzag_per_document"),
    ((700, 1, 1300, 999, 3000), 64, 8, "zigzag_per_document"),
    ((5000, 1003), 4, 8, "zigzag"),
    ((1200 + 900 + 900,), 4, 8, "contiguous"),
]
# the previous copies (A/B, negative control)
OLD_PATH = {"upload": "blocking", "table_cache": "workspace"}


DIST_SLEEP_MS = 2000.0  # the queue probe of the dist census windows (the host must return within a quarter of it)


def _run_upload_rank(rank: int, W: int, row, port: int, result_dir: str | None) -> dict:
    """Host synchronization on every rank of a CP row, each window a census of ``tests/test_helpers/cake_host_sync.py``
    (upload counter, profiler copies where traced, sync-debug warnings, an event-based queue probe and a host-time
    bound): the new path is bitwise the previous one (KDA saved / replay / checkpointed passes with and without a
    workspace, conv halo forward / backward); the first call of a fresh context (with the reused workspace and with
    ``workspace=None``) issues only pinned copies (at least one: its table sets) and does not wait; steady-state calls
    issue no copy and do not wait; the previous path's calls and preparation copy from pageable memory and wait; a
    pack prepared behind queued work (``heads=H``) issues one pinned copy, does not wait and runs bitwise; a pack
    prepared from another thread on a held-back side stream does the same; repeated ``exchange="ring"`` calls record
    no further upload on the context (ring rows); the plan's ``P`` of each rank is reported."""
    from flashinfer.cake_causal_conv1d_train_backend import kcp as conv
    from tests.test_helpers.cake_host_sync import (
        census,
        counts_agree,
        nonblocking,
        sleep_cycles,
    )

    lengths, H, _w, layout = row
    device = torch.device("cuda", rank % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    if W > 1 and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))
        dist.init_process_group("nccl", rank=rank, world_size=W, device_id=device)
    group = dist.group.WORLD if W > 1 else None
    part, cu = _partition(lengths, W, layout)
    glob = make_inputs(lengths, H, device)
    ops = _local_slices(part, rank, cu, glob)
    q, k, v, g, beta, dout = (ops[n] for n in ("q", "k", "v", "g", "beta", "dout"))
    gen = torch.Generator(device="cpu").manual_seed(0x7815)
    C = 256
    xg = torch.randn(cu[-1], C, generator=gen).to(torch.bfloat16).to(device)
    dyg = torch.randn(cu[-1], C, generator=gen).to(torch.bfloat16).to(device)
    w = (0.5 * torch.randn(C, 4, generator=gen)).to(device).contiguous()
    loc = _local_slices(part, rank, cu, {"x": xg, "dy": dyg})
    x, dy = loc["x"].contiguous(), loc["dy"].contiguous()

    def run(c, ws, **opts) -> list:
        out, saved = cake_kcp.chunk_kda_fwd_kcp(
            q, k, v, g, beta, c, workspace=ws, **opts, deterministic=False
        )
        # ``boundary_mode`` is a forward keyword
        bwd_opts = {key: val for key, val in opts.items() if key != "boundary_mode"}
        grads = cake_kcp.chunk_kda_bwd_kcp(
            q,
            k,
            v,
            g,
            beta,
            c,
            dout,
            saved=saved,
            workspace=ws,
            **bwd_opts,
            deterministic=False,
        )
        return [out, *grads]

    def ckpt(c, ws) -> list:
        from torch.utils.checkpoint import checkpoint

        leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)]
        out = checkpoint(
            lambda *a: cake_kcp.chunk_kda_kcp(
                *a, c, None, "light", "saved", ws, deterministic=False
            ),
            *leaves,
            use_reentrant=False,
        )
        out.backward(dout)
        return [out.detach()] + [lf.grad for lf in leaves]

    def halo(c) -> list:
        y, hist = conv.cake_causal_conv1d_kcp_forward(x, w, c)
        return [y, *conv.cake_causal_conv1d_kcp_backward(x, w, c, dy, hist)]

    def eq(a, b) -> bool:
        return len(a) == len(b) and all(
            torch.equal(s, t) for s, t in zip(a, b, strict=True)
        )

    def build(**kw):
        return cake_kcp_plan.build_context_kcp(part, group, rank, device, **kw)

    def window(fn, **kw) -> dict:
        return census(fn, device, sleep_ms=DIST_SLEEP_MS, **kw)

    def brief(c: dict) -> dict:
        keys = ("pageable", "pinned", "prof_pageable", "prof_pinned", "sync_warn")
        return {
            **{key: c[key] for key in keys},
            "busy": c["busy"],
            "host_s": round(c["host_s"], 4),
        }

    def fresh(c):
        return c, not c.ready.query()

    prev = cake_kcp_plan.enable_test_knobs(True)
    try:
        sleep_cycles(device, DIST_SLEEP_MS)  # calibrated before any window
        old = build(_kcp_test_knobs=dict(OLD_PATH))
        ctx = build(heads=H)
        ws, ws_old = cake_kcp.WorkspaceKcp(), cake_kcp.WorkspaceKcp()
        P = cake_kcp.WorkspaceKcp(local=False).tables(ctx, H).P
        ref = (
            run(old, None)
            + run(old, ws_old)
            + run(old, None, boundary_mode="replay")
            + ckpt(old, None)
            + halo(old)
        )
        new = (
            run(ctx, None)
            + run(ctx, ws)
            + run(ctx, ws, boundary_mode="replay")
            + ckpt(ctx, ws)
            + halo(ctx)
        )
        for _ in range(2):
            run(ctx, ws)
            run(ctx, None)
        steady_ws = window(lambda: run(ctx, ws))
        steady_none = window(lambda: run(ctx, None) + halo(ctx))
        old_calls = window(lambda: run(old, None))
        # the first call of a fresh context: its table sets are uploaded by the call
        ctx_f = build(heads=H)
        first_ws = window(lambda: run(ctx_f, ws) + halo(ctx_f))
        ctx_g = build(heads=H)
        first_none = window(lambda: run(ctx_g, None) + halo(ctx_g))
        # the next pack prepared behind queued work, and the previous path's preparation
        prep = window(lambda: fresh(build(heads=H)))
        ctx_b, prep_pending = prep["out"]
        prep_old = window(lambda: build(heads=H, _kcp_test_knobs=dict(OLD_PATH)))
        res_b = run(ctx_b, ws) + halo(ctx_b)
        # prepared from another thread on a side stream held back by ~0.5 s of its own work
        side = torch.cuda.Stream(device)

        def hold() -> None:
            with torch.cuda.stream(side):
                torch.cuda._sleep(sleep_cycles(device, 500.0))

        thread_prep = window(
            lambda: fresh(build(heads=H, stream=side)), in_thread=True, setup=hold
        )
        ctx_t, held = thread_prep["out"]
        res_t = run(ctx_t, None) + halo(ctx_t)
        # the ring exchange: its hop chains are uploaded once per table set, not per call
        ring_uploads, ring_chains = None, None
        if W > 1 and int(ctx.n_d_max) - 1 <= cake_kcp.RING_MAX_HOPS:
            ws_r = cake_kcp.WorkspaceKcp()
            run(ctx, ws_r, exchange="ring", save_transfer=False)
            n1 = len(ctx._uploads)
            for _ in range(3):
                run(ctx, ws_r, exchange="ring", save_transfer=False)
            ring_uploads = (n1, len(ctx._uploads))
            ring_chains = sorted(
                cake_kcp._ring_tables(
                    cake_kcp.WorkspaceKcp(local=False).tables(ctx, H)
                ).chains
            )
        torch.cuda.synchronize(device)
    finally:
        cake_kcp_plan.enable_test_knobs(prev)
    halo_new = new[24:27]
    flags = {
        "new_eq_previous_path": eq(new, ref),
        "first_call_reused_ws_nonblocking": nonblocking(first_ws, DIST_SLEEP_MS)
        and first_ws["pinned"] >= 1
        and eq(first_ws["out"], new[6:12] + halo_new),
        "first_call_no_ws_nonblocking": nonblocking(first_none, DIST_SLEEP_MS)
        and first_none["pinned"] >= 1
        and eq(first_none["out"], new[0:6] + halo_new),
        "steady_reused_ws_nonblocking": nonblocking(steady_ws, DIST_SLEEP_MS)
        and steady_ws["pinned"] == 0
        and eq(steady_ws["out"], new[6:12]),
        "steady_no_ws_nonblocking": nonblocking(steady_none, DIST_SLEEP_MS)
        and steady_none["pinned"] == 0
        and eq(steady_none["out"], new[0:6] + halo_new),
        "previous_path_calls_block": old_calls["pageable"] > 0
        and counts_agree(old_calls)
        and old_calls["sync_warn"] > 0
        and not old_calls["busy"],
        "prepare_nonblocking": nonblocking(prep, DIST_SLEEP_MS)
        and prep["pinned"] == 1
        and prep_pending,
        "previous_path_prepare_blocks": prep_old["pageable"] > 0
        and counts_agree(prep_old)
        and prep_old["sync_warn"] > 0
        and not prep_old["busy"],
        "prepared_pack_eq": eq(res_b, new[6:12] + halo_new),
        "thread_prepare_nonblocking": nonblocking(thread_prep, DIST_SLEEP_MS)
        and thread_prep["pinned"] == 1
        and held,
        "thread_prepared_pack_eq": eq(res_t, new[0:6] + halo_new),
        "ring_uploads_bounded": ring_uploads is None
        or ring_uploads[0] == ring_uploads[1],
    }
    counts = {
        "P": P,
        "ring_uploads": ring_uploads,
        "ring_chains": ring_chains,
        "first_ws": brief(first_ws),
        "first_none": brief(first_none),
        "steady_ws": brief(steady_ws),
        "steady_none": brief(steady_none),
        "old_calls": brief(old_calls),
        "prep": brief(prep),
        "prep_old": brief(prep_old),
        "thread_prep": brief(thread_prep),
    }
    if W > 1:
        flag_list, count_list = [None] * W, [None] * W
        dist.all_gather_object(flag_list, flags)
        dist.all_gather_object(count_list, counts)
    else:
        flag_list, count_list = [flags], [counts]
    report = {
        "rank": rank,
        "world": W,
        "row": _row_id(row),
        "flags": {},
        "counts_by_rank": count_list,
    }
    for key in flags:
        report["flags"][key] = all(f[key] for f in flag_list)
    report["pass"] = all(report["flags"].values())
    if W > 1:
        dist.barrier()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(
            os.path.join(result_dir, "upload-" + _row_id(row) + ".json"), "w"
        ) as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _spawn_upload_entry(rank: int, W: int, row, port: int, result_dir: str) -> None:
    try:
        report = _run_upload_rank(rank, W, row, port, result_dir)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            cake_kcp.graph_cache_clear()
            dist.destroy_process_group()


@pytest.mark.parametrize("row", UPLOAD_ROWS, ids=_row_id)
def test_multi_gpu_upload_rows_are_bitwise_and_nonblocking(row) -> None:
    """The rows of :func:`_run_upload_rank`: one process per GPU (``spawn``, rows up to the visible GPU count) or
    one process per Slurm task (a step with ``SLURM_NTASKS`` > 1 runs the rows of its own world size and skips the
    others)."""
    lengths, H, W, layout = row
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"this Slurm step runs the W = {ntasks} rows")
        if not _programs_exported():
            pytest.skip(
                "the KCP programs are not exported in the package's build.json yet"
            )
        report = _run_upload_rank(
            int(os.environ["SLURM_PROCID"]),
            W,
            row,
            int(os.environ.get("MASTER_PORT", "29500")),
            os.environ.get("KCP_TEST_RESULT_DIR"),
        )
        assert report["pass"], report
        return
    _skip_unless_runnable(W)
    with tempfile.TemporaryDirectory(prefix="cake_kcp_upload_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        _start_ranks(_spawn_upload_entry, W, row, result_dir)
        with open(os.path.join(result_dir, "upload-" + _row_id(row) + ".json")) as fh:
            report = json.load(fh)
    assert report["pass"], report


def test_public_api_cannot_select_a_precision_mode(monkeypatch) -> None:
    """The private test hook is refused without the environment variable; the public options never reach K3's
    non-shipped programs (SPEC-11)."""
    monkeypatch.setattr(cake_kcp_plan, "_TEST_KNOBS_ENABLED", False)
    with pytest.raises(ValueError):
        cake_kcp._options({"_kcp_test_knobs": {"precision_mode": 2}})
    with pytest.raises(TypeError):
        cake_kcp._options({"precision_mode": 1})
    assert cake_kcp._options({}).precision_mode == 0
    monkeypatch.setattr(cake_kcp_plan, "_TEST_KNOBS_ENABLED", True)
    assert (
        cake_kcp._options({"_kcp_test_knobs": {"precision_mode": 1}}).precision_mode
        == 1
    )
    with pytest.raises(ValueError):
        cake_kcp._options({"_kcp_test_knobs": {"precision_mode": 7}})


def test_public_api_cannot_select_a_transfer_walk(monkeypatch) -> None:
    """The ``transfer_walk`` knob (K3's ``tf32_hilo`` / ``bf16x3`` programs of ``build.test.json``) is reachable only through
    the gated test hook; the public options never select a non-shipped transfer-walk program (SPEC-11)."""
    monkeypatch.setattr(cake_kcp_plan, "_TEST_KNOBS_ENABLED", False)
    with pytest.raises(TypeError):
        cake_kcp._options({"transfer_walk": "bf16x3"})
    with pytest.raises(ValueError):
        cake_kcp._options({"_kcp_test_knobs": {"transfer_walk": "bf16x3"}})
    assert cake_kcp._options({}).transfer_walk == "bf16x3_hw64"
    monkeypatch.setattr(cake_kcp_plan, "_TEST_KNOBS_ENABLED", True)
    assert (
        cake_kcp._options(
            {"_kcp_test_knobs": {"transfer_walk": "bf16x3"}}
        ).transfer_walk
        == "bf16x3"
    )
    assert (
        cake_kcp._options(
            {"_kcp_test_knobs": {"transfer_walk": "bf16x3_hw64"}}
        ).transfer_walk
        == "bf16x3_hw64"
    )
    assert (
        cake_kcp._options(
            {"_kcp_test_knobs": {"transfer_walk": "tf32_hilo"}}
        ).transfer_walk
        == "tf32_hilo"
    )
    assert (
        cake_kcp._options({"_kcp_test_knobs": {"precision_mode": 1}}).transfer_walk
        == "tf32_hilo"
    )
    assert (
        cake_kcp._options(
            {"_kcp_test_knobs": {"precision_mode": 1, "transfer_walk": "tf32_hilo"}}
        ).transfer_walk
        == "tf32_hilo"
    )
    with pytest.raises(ValueError):
        cake_kcp._options({"_kcp_test_knobs": {"transfer_walk": "fp64"}})
    with pytest.raises(
        ValueError
    ):  # a K3 precision_mode program belongs to the tf32_hilo family
        cake_kcp._options(
            {"_kcp_test_knobs": {"precision_mode": 1, "transfer_walk": "bf16x3"}}
        )
    with pytest.raises(
        ValueError
    ):  # the shipped BF16x3 walk named beside it is refused too
        cake_kcp._options(
            {"_kcp_test_knobs": {"precision_mode": 2, "transfer_walk": "bf16x3_hw64"}}
        )


def test_module_imports_only_torch_and_its_package() -> None:
    """The design note §11.3: the host module imports torch, torch.distributed, tvm_ffi and its own package
    (plus ``ctypes`` for the CUDA runtime's device cache-config preference around the tape launches,
    ``cake_kcp.walk_device_cache_config``; the same ``ctypes`` import also reaches the capture-graph entry points of the
    replay cap of record
    ``graph_cap="node_attr"`` -- ``cudaStreamGetCaptureInfo`` (``_v2`` / ``_v3``), ``cudaGraphNodeGetType``,
    ``cudaGraphKernelNodeSetAttribute``, ``cudaGraphKernelNodeGetAttribute``, ``cudaRuntimeGetVersion``,
    ``cudaGetLastError``; no new module; the trimmed host path:
    ``functools.cached_property`` on the host table set (``walk_all_p0`` / ``walk_all_rest``, host ints only) -- stdlib,
    no new module either; ``weakref`` for the trimmed
    path's begin_call template cache, owned by the local arena and weakly keyed by the table set -- stdlib, no new
    module)."""
    import ast

    src = Path(cake_kcp.__file__).read_text()
    tree = ast.parse(src)
    allowed = {
        "__future__",
        "dataclasses",
        "math",
        "os",
        "types",
        "typing",
        "collections",
        "collections.abc",
        "torch",
        "torch.distributed",
        "tvm_ffi",
        "ctypes",
        "warnings",
        "functools",
        "weakref",
    }  # weakref: the arena-owned template cache of the trimmed host path; functools: cached_property of the trimmed host path; ctypes: the cache-config bracket of the walk-launch geometry and the capture-graph entry points of the replay cap; warnings: the one RuntimeWarning without a loadable CUDA runtime or for a degraded node cap; no other use
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name in allowed, a.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            assert node.module in allowed, node.module


# ---------------------------------------------------------------------------------------------------------------
# Checkpoint recompute (issue #7815): ``chunk_kda_kcp`` under ``torch.utils.checkpoint``.  Rows: ``ROWS`` plus one
# H = 64 row (``P = 1`` ranks) and one row where no document crosses a rank; under a Slurm step of 8 or 16 tasks also the rows of that width (``CKPT_WIDE``; a
# one-node run has no such rows, so they never show up as skips there).

CKPT_WIDE = {
    8: [
        ((3000,), 4, 8, "zigzag"),
        ((2048, 5, 3000, 1), 4, 8, "zigzag_per_document"),
        ((6, 1, 2, 3, 4000), 4, 8, "ragged:1,1,1,3,0,2,4000,4"),
    ],
    16: [
        ((3000,), 4, 16, "zigzag"),
        ((2048, 5, 3000, 1), 4, 16, "zigzag_per_document"),
    ],
}
# no document crosses a rank (one document per rank): no rank binds a state, and the ring has no hop
CKPT_NONCROSSING = ((1200, 900, 900, 600), 4, 4, "ragged:1200,900,900,600")
CKPT_ROWS = (
    ROWS
    + [((2048, 5, 3000, 1), 64, 4, "zigzag_per_document"), CKPT_NONCROSSING]
    + CKPT_WIDE.get(int(os.environ.get("SLURM_NTASKS", "0") or 0), [])
)


def _requested(device) -> int:
    return int(torch.cuda.memory_stats(device).get("requested_bytes.all.current", 0))


def _poison(device) -> None:
    """Fill the caching allocator's free blocks (large and small pool) with all-ones bits (NaN in BF16 / FP32): rows
    of a fresh ``torch.empty`` tensor that no kernel writes then read as NaN, not as a stale earlier result."""
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    big = torch.full((1 << 27,), -1, dtype=torch.int32, device=device)  # 512 MiB
    small = [
        torch.full((1 << 17,), -1, dtype=torch.int32, device=device) for _ in range(256)
    ]  # 512 KiB each: the small pool's 2 MiB segments
    torch.cuda.synchronize(device)
    del big, small


# the launches a bound recompute may drop: the exchange (c1 / c2 or the ring hops) and its feeders (the owner merge,
# the P = 1 summaries, the P > 1 map merge and transpose)
DROPPABLE = {
    "a2a:c1",
    "a2a:c2",
    "kcp_affine_merge",
    "kcp_fwdh_offset",
    "kcp_transfer_walk",
    "kcp_transpose_tiles",
}


def _removed_launches(orig: list, replay: list) -> tuple[bool, list]:
    """``(ok, removed)``: the original forward's launches the replay drops; ``ok`` when the replay is the original
    sequence with launches deleted (same order, nothing added) and every dropped one an exchange launch or a feeder
    of it."""
    removed, i = [], 0
    for x in orig:
        if i < len(replay) and replay[i] == x:
            i += 1
        else:
            removed.append(x)
    ok = i == len(replay) and all(
        x in DROPPABLE or x.startswith("ring:") for x in removed
    )
    return ok, removed


class _Counter:
    """Counts the ``torch.distributed`` collectives issued, the waits on their work handles and the bytes of
    ``all_to_all_single`` (input / output tensors) by phase: ``orig``, ``recompute`` (set by the checkpoint's
    ``context_fn``, re-entered per recompute) and ``bwd``.  ``isend`` / ``irecv`` stay unshimmed: the ring exchange
    passes them to ``P2POp``, which checks their identity; its hops are counted as ``batch_isend_irecv`` calls."""

    NAMES = (
        "all_to_all_single",
        "batch_isend_irecv",
        "all_gather_object",
        "all_gather",
        "all_gather_into_tensor",
        "all_reduce",
        "broadcast",
        "reduce_scatter_tensor",
        "send",
        "recv",
    )

    def __init__(self) -> None:
        self.phase = "orig"
        self.calls: dict[str, int] = {}
        self.waits: dict[str, int] = {}
        self.send: dict[str, int] = {}
        self.recv: dict[str, int] = {}
        self._saved: dict[str, object] = {}
        self._stack: list[str] = []

    def _bump(self, table: dict[str, int], n: int = 1) -> None:
        table[self.phase] = table.get(self.phase, 0) + n

    def _work(self, w):
        counter = self

        class _Work:
            def __init__(self, inner) -> None:
                self._inner = inner

            def wait(self, *a, **kw):
                counter._bump(counter.waits)
                return self._inner.wait(*a, **kw)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        return _Work(w) if hasattr(w, "wait") else w

    def __enter__(self) -> "_Counter":
        for name in self.NAMES:
            fn = getattr(dist, name, None)
            if fn is None:
                continue
            self._saved[name] = fn

            def shim(*a, _fn=fn, _name=name, **kw):
                self._bump(self.calls)
                if _name == "all_to_all_single":
                    out_t, in_t = a[0], a[1]
                    self._bump(self.recv, out_t.numel() * out_t.element_size())
                    self._bump(self.send, in_t.numel() * in_t.element_size())
                r = _fn(*a, **kw)
                if isinstance(r, list):
                    return [self._work(x) for x in r]
                return self._work(r)

            setattr(dist, name, shim)
        return self

    def __exit__(self, *exc) -> None:
        for name, fn in self._saved.items():
            setattr(dist, name, fn)
        self._saved.clear()

    def recompute_cm(self):
        counter = self

        class _Phase:
            def __enter__(self):
                counter._stack.append(counter.phase)
                counter.phase = "recompute"
                return self

            def __exit__(self, *exc):
                counter.phase = counter._stack.pop()
                return False

        return _Phase()

    def record(self) -> dict:
        return {
            t: {p: getattr(self, t).get(p, 0) for p in ("orig", "recompute", "bwd")}
            for t in ("calls", "waits", "send", "recv")
        }


def _run_rank_ckpt(
    rank: int, W: int, row, port: int, result_dir: str | None, local_id: int
) -> dict:
    """One rank of a checkpoint row (issue #7815):

    - #7039: without a handle a checkpointed forward retains nothing (per-call and caller-held workspace, after a
      warm-up step of the same form);
    - ``out``, the five gradients and ``dw`` (the checkpointed function returns ``out * w``: the product's backward
      reads the recompute's ``out``) of the checkpoint with a recompute handle (both tiers, ``use_reentrant`` False
      and True, the ring exchange where the partition allows it, the overlap schedule with a2a and ring,
      ``retain_graph`` + a second backward, early stop off, nested regions with the handle outside / inside the
      outer region) are bitwise the plain checkpoint's and the direct pass's; every recompute's ``out`` is bitwise
      the original forward's; the recompute's launches are the original's in order minus the exchange and its
      feeders;
    - binding: a rank with crossing-in fragments stores a non-empty ``S_in``, and perturbing it before the backward
      changes the recompute's ``out`` and ``dw`` (positive control);
    - collectives, waits and all-to-all bytes per phase (``orig`` / ``recompute`` / ``bwd``): the plain checkpoint
      exchanges in every phase (positive control), the handle's recompute issues none, its original forward and
      backward match the byte queries of the tier (``kda_comm_bytes_kcp(checkpoint_keep=...)``);
    - retained bytes after a checkpointed forward with a handle == ``handle.nbytes()`` ==
      ``kda_saved_bytes_kcp(checkpoint_keep=..., with_conv=False)``;
    - a symmetric fingerprint mismatch raises on every rank (no collective is issued -- counted --, no hang);
    - the layer core: three KCP convolutions (with ``"conv"`` handles, one of them shared with KDA) + q / k
      normalization + KDA under one checkpoint: the recompute issues no collective; every gradient bitwise."""
    import contextlib

    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

    conv_kcp = cake_package("cake_causal_conv1d_train", "kcp")
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
    flags: dict[str, bool] = {}
    info: dict[str, object] = {
        "T_r": int(q.shape[0]),
        "n_frag": int(ctx.n_frag),
        "n_in": int(ctx.n_in),
        "n_out": int(ctx.n_out),
        "P": int(ctx.heads(H).P),
        "torch": torch.__version__,
    }

    def leaves():
        return [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)]

    def bits(t: torch.Tensor) -> torch.Tensor:
        # bit view: -0.0 vs 0.0 and NaN payloads count as differences
        return t.contiguous().view(
            torch.int16 if t.element_size() == 2 else torch.int32
        )

    def same(a: torch.Tensor, b: torch.Tensor) -> bool:
        return (
            a.dtype == b.dtype and a.shape == b.shape and torch.equal(bits(a), bits(b))
        )

    def eq(a: dict, b: dict) -> bool:
        return set(a) == set(b) and all(same(a[n], b[n]) for n in a)

    # #7039: retained bytes after a checkpointed forward without a handle (the output alone)
    retained = {}
    for form, wsp in (("per_call", None), ("caller_ws", ws)):

        def fn0(*a, _w=wsp):
            return cake_kcp.chunk_kda_kcp(
                *a, ctx, None, "light", "saved", _w, deterministic=False
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
    flags["retained_zero"] = all(v_ == 0 for v_ in retained.values())

    def run(
        mode: str,
        handle=None,
        *,
        keep=None,
        twice=False,
        reentrant=False,
        early_stop=True,
        poison=False,
        tamper=None,
        **opts,
    ):
        """mode: direct | ckpt | nested_out (handle outside both regions) | nested_in (handle made in the outer).
        The checkpointed function returns ``out * w`` (``w`` a leaf of ones: no bit of ``out`` or of its gradient
        changes, but the product's backward reads the ``out`` of the forward it belongs to -- the recompute's under
        a checkpoint; ``dw`` = ``dout * out``).  Returns the results, the per-phase counts and ``aux``: the
        LAST_CALL record of the original forward and of each backward's recompute (popped before each backward,
        ``None`` when none ran) and whether every recompute's ``out`` is bitwise the original's.  ``poison``:
        :func:`_poison` first; ``tamper(handle)`` runs between the forward and the backward."""
        if poison:
            _poison(device)
        lv = leaves()
        wl = torch.ones_like(dout).requires_grad_(True)
        kw = dict(opts)
        if keep is not None:
            kw["checkpoint_keep"] = keep
        outs = []

        def fn(q_, k_, v_, g_, b_, h_):
            y = cake_kcp.chunk_kda_kcp(
                q_,
                k_,
                v_,
                g_,
                b_,
                ctx,
                workspace=ws,
                recompute_handle=h_,
                **kw,
                deterministic=False,
            )
            outs.append(y.detach().clone())
            return y

        def top(q_, k_, v_, g_, b_, h_, w_):
            return fn(q_, k_, v_, g_, b_, h_) * w_

        recs = []
        with _Counter() as cc, set_checkpoint_early_stop(early_stop):
            ctx_fn = lambda: (contextlib.nullcontext(), cc.recompute_cm())  # noqa: E731
            if mode == "direct":
                out = top(*lv, handle, wl)
            elif mode == "ckpt":
                if reentrant:
                    out = checkpoint(top, *lv, handle, wl, use_reentrant=True)
                else:
                    out = checkpoint(
                        top, *lv, handle, wl, use_reentrant=False, context_fn=ctx_fn
                    )
            else:

                def outer(q_, k_, v_, g_, b_, h_, w_):
                    # the product saves the outer region's yi (the outer recompute runs and its yi is read)
                    hh = cake_kcp.RecomputeHandleKcp() if mode == "nested_in" else h_
                    yi = checkpoint(fn, q_, k_, v_, g_, b_, hh, use_reentrant=False)
                    return yi * w_

                out = checkpoint(
                    outer, *lv, handle, wl, use_reentrant=False, context_fn=ctx_fn
                )
            recs.append(cake_kcp.LAST_CALL.pop("fwd"))
            if tamper is not None:
                tamper(handle)
            cc.phase = "bwd"
            for i in range(2 if twice else 1):
                out.backward(dout, retain_graph=twice and i == 0)
                recs.append(cake_kcp.LAST_CALL.pop("fwd", None))
        res = {"out": out.detach(), "dw": wl.grad}
        res.update(
            {
                f"d{n}": lf.grad
                for n, lf in zip(("q", "k", "v", "g", "beta"), lv, strict=True)
            }
        )
        modes = [None if r is None else r.get("recompute_handle") for r in recs]
        aux = {
            "recs": recs,
            "n_outs": len(outs),
            "outs_ok": all(same(o, outs[0]) for o in outs[1:]),
        }
        return res, {**cc.record(), "modes": modes}, aux

    def replay_ok(aux, exchange: str) -> bool:
        """The recompute's out is the original's, and its launches are the original's in order minus the exchange
        (every c1 / c2 or every ring hop) and its feeders (:func:`_removed_launches`)."""
        orig, rep = aux["recs"][0], aux["recs"][1]
        if rep is None or not aux["outs_ok"] or aux["n_outs"] < 2:
            return False
        ok, removed = _removed_launches(orig["sequence"], rep["sequence"])
        if exchange == "ring":
            want = [x for x in orig["sequence"] if x.startswith("ring:")]
            exch = [x for x in removed if x.startswith(("ring:", "a2a:"))]
        else:
            want, exch = (
                ["a2a:c1", "a2a:c2"],
                [x for x in removed if x.startswith("a2a:")],
            )
        return ok and exch == want and rep["n_collectives"] == 0

    def plain_ok(aux) -> bool:
        """The plain checkpoint's recompute: the original's launches and out."""
        orig, rep = aux["recs"][0], aux["recs"][1]
        return (
            rep is not None
            and aux["outs_ok"]
            and aux["n_outs"] == 2
            and rep["sequence"] == orig["sequence"]
        )

    counts: dict[str, dict] = {}
    direct, counts["direct"], _ = run("direct")
    plain, counts["plain"], aux = run("ckpt")
    flags["plain_recompute"] = plain_ok(aux)
    plain_re, counts["plain_reentrant"], aux = run("ckpt", reentrant=True)
    flags["plain_reentrant_recompute"] = plain_ok(aux)
    flags["plain_eq_direct"] = eq(plain, direct)
    flags["plain_reentrant_eq_direct"] = eq(plain_re, direct)
    pc = counts["plain"]
    # positive control: the plain checkpoint exchanges (c1 + c2) in the original forward and in its recompute and
    # (b1 + b2) in the backward on every rank
    flags["positive_control"] = all(
        pc[t]["orig"] > 0 and pc[t]["recompute"] == pc[t]["orig"] and pc[t]["bwd"] > 0
        for t in ("calls", "waits")
    )
    n_in = int(ctx.n_in)
    handles = {}
    for keep in ("states", "transfer"):
        for reentrant in (False, True):
            tag = f"handle_{keep}" + ("_reentrant" if reentrant else "")
            hd = cake_kcp.RecomputeHandleKcp()
            res, rec, aux = run("ckpt", hd, keep=keep, reentrant=reentrant)
            counts[tag] = rec
            handles[tag] = hd
            flags[f"{tag}_eq_plain"] = eq(res, plain)
            flags[f"{tag}_eq_direct"] = eq(res, direct)
            flags[f"{tag}_recompute"] = replay_ok(aux, "a2a")
            ref = counts["plain_reentrant" if reentrant else "plain"]
            if reentrant:  # the recompute runs inside the backward (no context_fn): plain = bwd + recompute
                want = {
                    t: {
                        "orig": ref[t]["orig"],
                        "recompute": 0,
                        "bwd": ref[t]["bwd"] - ref[t]["orig"],
                    }
                    for t in ("calls", "waits")
                }
            else:
                want = {
                    t: {"orig": ref[t]["orig"], "recompute": 0, "bwd": ref[t]["bwd"]}
                    for t in ("calls", "waits")
                }
            flags[f"{tag}_counts"] = all(rec[t] == want[t] for t in ("calls", "waits"))
            flags[f"{tag}_modes"] = rec["modes"] == ["capture", "replay"]
            # bytes the collectives moved == the byte query of the tier (KDA alone: c1 + c2, b1 + b2; none in the
            # recompute)
            cb = cake_kcp_plan.kda_comm_bytes_kcp(ctx, H=H, checkpoint_keep=keep)
            want_b = {
                "send": {
                    "orig": cb["c1"]["send"] + cb["c2"]["send"],
                    "recompute": cb["recompute"]["send"],
                    "bwd": cb["b1"]["send"] + cb["b2"]["send"],
                },
                "recv": {
                    "orig": cb["c1"]["recv"] + cb["c2"]["recv"],
                    "recompute": cb["recompute"]["recv"],
                    "bwd": cb["b1"]["recv"] + cb["b2"]["recv"],
                },
            }
            if not reentrant:
                flags[f"{tag}_bytes_eq_query"] = (
                    rec["send"] == want_b["send"] and rec["recv"] == want_b["recv"]
                )
            query = cake_kcp_plan.kda_saved_bytes_kcp(
                ctx, H=H, checkpoint_keep=keep, with_conv=False
            )
            flags[f"{tag}_nbytes_eq_query"] = hd.nbytes() == query
            # a rank with crossing-in fragments binds a non-empty S_in (the checks above are not vacuous there)
            stored = hd.entries["kda"]
            flags[f"{tag}_binding_nonempty"] = n_in == 0 or (
                stored["S_in"].numel() > 0 and hd.nbytes() > 0
            )
            info.setdefault("handle_nbytes", {})[tag] = hd.nbytes()

    # binding positive control: perturbed stored states change the recompute's out (dw) and the gradients on every
    # rank with crossing-in fragments; a recompute that exchanged instead of binding would leave them unchanged
    def perturb(h) -> None:
        e = h.entries["kda"]
        e["S_in"] = e["S_in"] + 1.0

    pert, _, pert_aux = run("ckpt", cake_kcp.RecomputeHandleKcp(), tamper=perturb)
    flags["binding_control"] = n_in == 0 or (
        not pert_aux["outs_ok"] and not same(pert["dw"], plain["dw"])
    )
    # the ring exchange (where the partition allows it): its recompute skips the hops as well
    ring_ok = int(ctx.n_d_max) - 1 <= cake_kcp.RING_MAX_HOPS
    info["ring_eligible"] = ring_ok
    info["max_hops"] = max(int(ctx.n_d_max) - 1, 0)
    if ring_ok:
        plain_ring, counts["plain_ring"], aux_p = run("ckpt", exchange="ring")
        hd_ring, counts["handle_ring"], aux_h = run(
            "ckpt", cake_kcp.RecomputeHandleKcp(), exchange="ring"
        )
        pr, hr = counts["plain_ring"], counts["handle_ring"]
        flags["ring_bitwise"] = eq(plain_ring, direct) and eq(hd_ring, plain_ring)
        flags["ring_counts"] = all(
            hr[t]["recompute"] == 0
            and hr[t]["orig"] == pr[t]["orig"]
            and hr[t]["bwd"] == pr[t]["bwd"]
            and pr[t]["recompute"] == pr[t]["orig"]
            for t in ("calls", "waits")
        ) and hr["modes"] == ["capture", "replay"]
        flags["ring_recompute"] = plain_ok(aux_p) and replay_ok(aux_h, "ring")
        flags["ring_exchange"] = (
            aux_h["recs"][0]["exchange"] == "ring"
            and aux_p["recs"][0]["exchange"] == "ring"
        )
    # the overlap schedule (test knob): the bound branch issues the exchange window's launches without the
    # exchange; with the ring also where no document crosses a rank (no hop: the document-start / -end walks run
    # alone).  Every pass starts from a poisoned allocator, so an unwritten row reads as NaN.
    previous_knobs = cake_kcp_plan.enable_test_knobs(True)
    try:
        ov = {"_kcp_test_knobs": {"schedule": "overlap"}}
        plain_ov, counts["plain_overlap"], aux = run("ckpt", poison=True, **ov)
        flags["overlap_plain"] = eq(plain_ov, direct) and plain_ok(aux)
        po = counts["plain_overlap"]
        for keep in ("states", "transfer"):
            tag = f"handle_overlap_{keep}"
            res, rec, aux = run(
                "ckpt", cake_kcp.RecomputeHandleKcp(), keep=keep, poison=True, **ov
            )
            counts[tag] = rec
            flags[f"{tag}_bitwise"] = eq(res, plain_ov) and eq(res, direct)
            flags[f"{tag}_recompute"] = replay_ok(aux, "a2a")
            flags[f"{tag}_counts"] = all(
                rec[t]["recompute"] == 0
                and rec[t]["orig"] == po[t]["orig"]
                and rec[t]["bwd"] == po[t]["bwd"]
                for t in ("calls", "waits")
            ) and rec["modes"] == ["capture", "replay"]
        if ring_ok:
            plain_rov, counts["plain_ring_overlap"], aux_p = run(
                "ckpt", poison=True, exchange="ring", **ov
            )
            res, counts["handle_ring_overlap"], aux_h = run(
                "ckpt",
                cake_kcp.RecomputeHandleKcp(),
                poison=True,
                exchange="ring",
                **ov,
            )
            pro, hro = counts["plain_ring_overlap"], counts["handle_ring_overlap"]
            flags["ring_overlap_bitwise"] = eq(plain_rov, direct) and eq(res, plain_rov)
            flags["ring_overlap_recompute"] = plain_ok(aux_p) and replay_ok(
                aux_h, "ring"
            )
            flags["ring_overlap_counts"] = all(
                hro[t]["recompute"] == 0
                and hro[t]["orig"] == pro[t]["orig"]
                and hro[t]["bwd"] == pro[t]["bwd"]
                for t in ("calls", "waits")
            ) and hro["modes"] == ["capture", "replay"]
    finally:
        cake_kcp_plan.enable_test_knobs(previous_knobs)
    # retain_graph: each backward's recompute binds the entry (no collective); exactly twice the gradients
    twice, counts["handle_twice"], aux = run(
        "ckpt", cake_kcp.RecomputeHandleKcp(), twice=True
    )
    tw = counts["handle_twice"]
    flags["twice_accumulates"] = (
        all(
            same(twice[n], direct[n] + direct[n])
            for n in (*TENSORS, "dw")
            if n != "out"
        )
        and tw["calls"]["recompute"] == 0
        and tw["calls"]["bwd"] == 2 * pc["calls"]["bwd"]
        and tw["modes"] == ["capture", "replay", "replay"]
        and aux["n_outs"] == 3
        and aux["outs_ok"]
        and all(r is not None and r["n_collectives"] == 0 for r in aux["recs"][1:])
    )
    # early stop off
    late, counts["handle_no_early_stop"], aux = run(
        "ckpt", cake_kcp.RecomputeHandleKcp(), early_stop=False
    )
    flags["no_early_stop"] = (
        eq(late, direct)
        and counts["handle_no_early_stop"]["calls"]["recompute"] == 0
        and replay_ok(aux, "a2a")
    )
    # nested checkpoints
    nest_out, counts["nested_out"], aux_o = run(
        "nested_out", cake_kcp.RecomputeHandleKcp()
    )
    nest_in, counts["nested_in"], aux_i = run(
        "nested_in", cake_kcp.RecomputeHandleKcp()
    )
    flags["nested_out_bitwise"] = eq(nest_out, direct) and aux_o["outs_ok"]
    flags["nested_in_bitwise"] = eq(nest_in, direct) and aux_i["outs_ok"]
    # handle made outside both regions: the outer recompute (recompute phase) and the inner recompute (counted in
    # the backward phase, outside the outer's context) bind the entry; only the backward exchanges
    flags["nested_out_counts"] = counts["nested_out"]["calls"] == {
        "orig": pc["calls"]["orig"],
        "recompute": 0,
        "bwd": pc["calls"]["bwd"],
    }
    # handle made inside the outer region: the outer recompute's inner forward is an original forward (exchanges)
    flags["nested_in_counts"] = counts["nested_in"]["calls"] == pc["calls"]
    # retained bytes after a checkpointed forward with a handle (after a warm-up of the same form): the handle's
    # entry, i.e. kda_saved_bytes_kcp(checkpoint_keep=..., with_conv=False)
    retained_h = {}
    for keep in ("states", "transfer"):
        for form, wsp in (("per_call", None), ("caller_ws", ws)):

            def fn1(q_, k_, v_, g_, b_, h_, _w=wsp, _k=keep):
                return cake_kcp.chunk_kda_kcp(
                    q_,
                    k_,
                    v_,
                    g_,
                    b_,
                    ctx,
                    workspace=_w,
                    recompute_handle=h_,
                    checkpoint_keep=_k,
                    deterministic=False,
                )

            for step in ("warm", "measure"):
                lv = leaves()
                hd = cake_kcp.RecomputeHandleKcp()
                torch.cuda.synchronize(device)
                gc.collect()
                before = _requested(device)
                out = checkpoint(fn1, *lv, hd, use_reentrant=False)
                torch.cuda.synchronize(device)
                if step == "measure":
                    retained_h[f"{keep}/{form}"] = (
                        _requested(device) - before - out.numel() * out.element_size()
                    )
                    query = cake_kcp_plan.kda_saved_bytes_kcp(
                        ctx, H=H, checkpoint_keep=keep, with_conv=False
                    )
                    flags[f"retained_{keep}_{form}"] = (
                        retained_h[f"{keep}/{form}"] == hd.nbytes() == query
                    )
                out.backward(dout)
                del out, lv, hd
    info["retained_bytes_handle"] = retained_h
    info["saved_bytes_query"] = {
        keep: cake_kcp_plan.kda_saved_bytes_kcp(
            ctx, H=H, checkpoint_keep=keep, with_conv=False
        )
        for keep in ("states", "transfer")
    }
    # a symmetric mismatch (every rank's entry describes another mode) raises on every rank before any collective
    # (counted: none in the failing backward)
    hd = cake_kcp.RecomputeHandleKcp()
    lv = leaves()
    out = checkpoint(
        lambda *a: cake_kcp.chunk_kda_kcp(
            *a[:5], ctx, recompute_handle=a[5], deterministic=False
        ),
        *lv,
        hd,
        use_reentrant=False,
    )
    fp = dict(hd.entries["kda"]["fingerprint"])
    fp["modes"] = (("use_qk_l2norm_in_kernel", True),)
    hd.entries["kda"]["fingerprint"] = tuple(fp.items())
    with _Counter() as mc:
        mc.phase = "bwd"
        try:
            out.backward(dout)
            flags["mismatch_raises"] = False
        except ValueError as exc:
            flags["mismatch_raises"] = "differs in ['modes']" in str(exc)
    counts["mismatch"] = mc.record()
    flags["mismatch_no_collective"] = all(
        v_ == 0 for t in ("calls", "waits") for v_ in counts["mismatch"][t].values()
    )
    del out, lv, hd

    # the layer core: three convolutions + q / k normalization + KDA in one region
    C = H * 128
    gen = torch.Generator(device="cpu").manual_seed(78150 + W)
    T_all = cu[-1]
    x_all = [torch.randn(T_all, C, generator=gen).to(torch.bfloat16) for _ in range(3)]
    w_all = [(torch.randn(C, 4, generator=gen) * 0.5).contiguous() for _ in range(3)]
    xs = [
        _local_slices(part, rank, cu, {"x": x})["x"].to(device).contiguous()
        for x in x_all
    ]
    wts = [w.to(device) for w in w_all]

    def l2(x):
        xf = x.float()
        return (xf * torch.rsqrt((xf * xf).sum(-1, keepdim=True) + 1e-6)).to(
            torch.bfloat16
        )

    l_outs: list[torch.Tensor] = []

    def layer(xq, xk, xv, wq, wk, wv, g_, b_, w_, hq, hk, hv):
        T = xq.shape[0]
        qq = conv_kcp.cake_causal_conv1d_kcp(xq, wq, ctx, recompute_handle=hq)
        kk = conv_kcp.cake_causal_conv1d_kcp(xk, wk, ctx, recompute_handle=hk)
        vv = conv_kcp.cake_causal_conv1d_kcp(xv, wv, ctx, recompute_handle=hv)
        y = cake_kcp.chunk_kda_kcp(
            l2(qq.view(T, H, 128)),
            l2(kk.view(T, H, 128)),
            vv.view(T, H, 128),
            g_,
            b_,
            ctx,
            workspace=ws,
            recompute_handle=hq,
            deterministic=False,
        )
        l_outs.append(y.detach().clone())
        return y * w_  # the product's backward reads the (recomputed) KDA out

    def run_layer(mode: str):
        l_outs.clear()
        lv = [
            x.detach().clone().requires_grad_(True)
            for x in (*xs, *wts, g, beta, torch.ones_like(dout))
        ]
        hs = (
            (
                cake_kcp.RecomputeHandleKcp(),  # shared by the q convolution and KDA
                conv_kcp.KcpRecomputeHandle(),
                conv_kcp.KcpRecomputeHandle(),
            )
            if mode == "handle"
            else (None, None, None)
        )
        with _Counter() as cc:
            if mode == "direct":
                out = layer(*lv, *hs)
            else:
                out = checkpoint(
                    layer,
                    *lv,
                    *hs,
                    use_reentrant=False,
                    context_fn=lambda: (contextlib.nullcontext(), cc.recompute_cm()),
                )
            cc.phase = "bwd"
            out.backward(dout)
        names = ("xq", "xk", "xv", "wq", "wk", "wv", "g", "beta", "w")
        res = {
            "out": out.detach(),
            **{f"d{n}": t.grad for n, t in zip(names, lv, strict=True)},
        }
        nb = sum(h.nbytes() for h in hs if h is not None)
        info.setdefault("layer_n_outs", {})[mode] = len(l_outs)
        flags[f"layer_{mode}_out"] = len(l_outs) == (
            1 if mode == "direct" else 2
        ) and all(same(o, l_outs[0]) for o in l_outs[1:])
        return res, cc.record(), nb, hs

    l_direct, counts["layer_direct"], _, _ = run_layer("direct")
    l_plain, counts["layer_plain"], _, _ = run_layer("plain")
    l_handle, counts["layer_handle"], l_nbytes, l_hs = run_layer("handle")
    lp, lh = counts["layer_plain"], counts["layer_handle"]
    flags["layer_bitwise"] = eq(l_plain, l_direct) and eq(l_handle, l_plain)
    flags["layer_positive_control"] = all(
        lp[t]["orig"] > 0 and lp[t]["recompute"] == lp[t]["orig"] and lp[t]["bwd"] > 0
        for t in ("calls", "waits")
    )
    flags["layer_recompute_no_comm"] = all(
        lh[t]["recompute"] == 0
        and lh[t]["orig"] == lp[t]["orig"]
        and lh[t]["bwd"] == lp[t]["bwd"]
        for t in ("calls", "waits")
    )
    # the shared handle holds "conv" + "kda"; the three handles together = KDA states + three hist pools
    flags["layer_nbytes"] = sorted(l_hs[0].entries) == ["conv", "kda"] and l_nbytes == (
        cake_kcp_plan.kda_saved_bytes_kcp(ctx, H=H, checkpoint_keep="states", C=C)
        + 2 * int(ctx.halo.n_hist_rows) * C * 2
    )

    # the same layer with cake_causal_conv1d_kcp_multi (the q / k / v convolutions in one call, one halo exchange
    # per direction) and ONE handle for the convolutions and KDA: bitwise the three single calls, two halo
    # collectives (and waits) fewer in every phase that exchanges halos, the same retained bytes
    def layer_multi(xq, xk, xv, wq, wk, wv, g_, b_, w_, h):
        T = xq.shape[0]
        qq, kk, vv = conv_kcp.cake_causal_conv1d_kcp_multi(
            (xq, xk, xv), (wq, wk, wv), ctx, recompute_handle=h
        )
        y = cake_kcp.chunk_kda_kcp(
            l2(qq.view(T, H, 128)),
            l2(kk.view(T, H, 128)),
            vv.view(T, H, 128),
            g_,
            b_,
            ctx,
            workspace=ws,
            recompute_handle=h,
            deterministic=False,
        )
        l_outs.append(y.detach().clone())
        return y * w_

    def run_layer_multi(mode: str):
        l_outs.clear()
        lv = [
            x.detach().clone().requires_grad_(True)
            for x in (*xs, *wts, g, beta, torch.ones_like(dout))
        ]
        h = cake_kcp.RecomputeHandleKcp() if mode == "handle" else None
        with _Counter() as cc:
            if mode == "direct":
                out = layer_multi(*lv, h)
            else:
                out = checkpoint(
                    layer_multi,
                    *lv,
                    h,
                    use_reentrant=False,
                    context_fn=lambda: (contextlib.nullcontext(), cc.recompute_cm()),
                )
            cc.phase = "bwd"
            out.backward(dout)
        names = ("xq", "xk", "xv", "wq", "wk", "wv", "g", "beta", "w")
        res = {
            "out": out.detach(),
            **{f"d{n}": t.grad for n, t in zip(names, lv, strict=True)},
        }
        flags[f"layer_multi_{mode}_out"] = len(l_outs) == (
            1 if mode == "direct" else 2
        ) and all(same(o, l_outs[0]) for o in l_outs[1:])
        return res, cc.record(), (0 if h is None else h.nbytes()), h

    lm_direct, counts["layer_multi_direct"], _, _ = run_layer_multi("direct")
    lm_plain, counts["layer_multi_plain"], _, _ = run_layer_multi("plain")
    lm_handle, counts["layer_multi_handle"], lm_nbytes, lm_h = run_layer_multi("handle")
    lmd, lmp, lmh = (
        counts["layer_multi_direct"],
        counts["layer_multi_plain"],
        counts["layer_multi_handle"],
    )
    ld = counts["layer_direct"]
    flags["layer_multi_bitwise"] = (
        eq(lm_direct, l_direct) and eq(lm_plain, l_direct) and eq(lm_handle, l_direct)
    )
    flags["layer_multi_counts"] = all(
        lmd[t]["orig"] == ld[t]["orig"] - 2
        and lmp[t]["orig"] == lp[t]["orig"] - 2
        and lmp[t]["recompute"] == lp[t]["recompute"] - 2
        and lmp[t]["bwd"] == lp[t]["bwd"] - 2
        and lmh[t]["orig"] == lp[t]["orig"] - 2
        and lmh[t]["recompute"] == 0
        and lmh[t]["bwd"] == lp[t]["bwd"] - 2
        for t in ("calls", "waits")
    )
    flags["layer_multi_nbytes"] = (
        sorted(lm_h.entries) == ["conv", "kda"] and lm_nbytes == l_nbytes
    )
    info["counts"] = counts

    flag_list = [None] * W
    info_list = [None] * W
    dist.all_gather_object(flag_list, flags)
    dist.all_gather_object(info_list, info)
    report = {
        "world": W,
        "row": _row_id(row),
        "torch": torch.__version__,
        "flags": {key: all(f[key] for f in flag_list) for key in flags},
        "failing_ranks": {
            key: [r for r, f in enumerate(flag_list) if not f[key]] for key in flags
        },
        "info_by_rank": info_list,
    }
    # flags present on every rank only (ring_* where eligible: the eligibility is a property of the partition)
    # every row but the non-crossing one binds a non-empty S_in on some rank (the binding checks are not vacuous)
    report["n_in_by_rank"] = [i["n_in"] for i in info_list]
    report["pass"] = (
        all(report["flags"].values())
        and all(set(f) == set(flag_list[0]) for f in flag_list)
        and (tuple(row) == CKPT_NONCROSSING) == (max(report["n_in_by_rank"]) == 0)
    )
    dist.barrier()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(
            os.path.join(result_dir, "ckpt-" + _row_id(row) + ".json"), "w"
        ) as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _spawn_entry_ckpt(rank: int, W: int, row, port: int, result_dir: str) -> None:
    try:
        report = _run_rank_ckpt(rank, W, row, port, result_dir, rank)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("row", CKPT_ROWS, ids=_row_id)
def test_checkpoint_recompute_rows(row) -> None:
    """The checkpoint rows of ``_run_rank_ckpt`` on every rank of the row."""
    W = row[2]
    ntasks = int(os.environ.get("SLURM_NTASKS", "1") or 1)
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"a {ntasks}-task Slurm step runs the W={ntasks} rows only")
        if not _programs_exported():
            pytest.skip(
                "the KCP programs are not exported in the package's build.json yet"
            )
        report = _run_rank_ckpt(
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
    with tempfile.TemporaryDirectory(prefix="cake_kcp_ckpt_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        _start_ranks(_spawn_entry_ckpt, W, row, result_dir)
        with open(os.path.join(result_dir, "ckpt-" + _row_id(row) + ".json")) as fh:
            report = json.load(fh)
    assert report["pass"], report
