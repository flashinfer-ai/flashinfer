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

Multi-GPU rows of the context-parallel (KCP) KDA training route (``cake_kda_train.cake_kcp``): ``W`` ranks over
NCCL, each holding the fragments of one partition of a packed batch, compared with the exact FP64 per-document
recurrence of the same operands (``tests/test_helpers/cake_kda_train_reference``) and with one another (saved ==
replay, repeat, autograd == explicit, checkpoint == direct, document-alone == slice, graph replay == eager with
every rank capturing the same calls).

Two launch forms: ``torch.multiprocessing.spawn`` (``pytest``: skipped when fewer than ``W`` CUDA devices are
visible or the KCP programs are not exported yet) and one process per Slurm task (``SLURM_PROCID`` /
``SLURM_NTASKS`` set: the test body runs in the calling process, the rendezvous from ``MASTER_ADDR`` /
``MASTER_PORT``).
"""

from __future__ import annotations

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

from tests.test_helpers.cake_kda_train_reference import (
    GRAD_NAMES,
    kda_exact_with_grads,
    make_inputs,
)
from tests.test_helpers.cake_package import cake_package

cake_kcp = cake_package("cake_kda_train", "cake_kcp")
cake_kcp_plan = cake_package("cake_kda_train", "cake_kcp_plan")
cake_jit = cake_package("cake_kda_train", "cake_jit")
cake_kcp_jit = cake_package("cake_kda_train", "cake_kcp_jit")

PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
TENSORS = ("out",) + GRAD_NAMES
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
        return cake_kcp_plan.kcp_partition(cu, W, fragments=nested), cu
    return cake_kcp_plan.kcp_partition(cu, W, layout=layout), cu


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
    ctx = cake_kcp_plan.build_kcp_context(part, group, rank, device)
    total = cu[-1]
    glob = make_inputs(
        lengths, H, device
    )  # the same seeded operands on every rank; each takes its slices
    ops = _local_slices(part, rank, cu, glob)
    q, k, v, g, beta, dout = (ops[n] for n in ("q", "k", "v", "g", "beta", "dout"))
    ws = cake_kcp.KcpWorkspace()

    def saved_pass(**opts):
        out, saved = cake_kcp.chunk_kda_kcp_fwd(
            q, k, v, g, beta, ctx, workspace=ws, **opts
        )
        # ``boundary_mode`` is a forward keyword: the backward replays the forward whenever ``saved`` is None
        bwd_opts = {key: val for key, val in opts.items() if key != "boundary_mode"}
        dq, dk, dv, dbeta, dg = cake_kcp.chunk_kda_kcp_bwd(
            q, k, v, g, beta, ctx, dout, saved=saved, workspace=ws, **bwd_opts
        )
        return {"out": out, "dq": dq, "dk": dk, "dv": dv, "dg": dg, "dbeta": dbeta}

    def autograd_pass(ckpt: bool):
        leaves = [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)]

        def fn(*a):
            return cake_kcp.chunk_kda_kcp(*a, ctx, None, "light", "saved", ws)

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
    # the shipped default keeps the owner's c1 receive tensor for the backward (save_transfer=True); the recompute
    # form (save_transfer=False) re-sends it and must be bitwise the same
    res_recompute = saved_pass(save_transfer=False)
    ring_ok = True
    if (
        W > 1 and int(ctx.n_d_max) - 1 <= cake_kcp.RING_MAX_HOPS
    ):  # gate 5: the ring path is bitwise the a2a path; the ring receives no c1 tensor, so recompute form only
        res_ring = saved_pass(exchange="ring", save_transfer=False)
        ring_ok = all(torch.equal(res_recompute[n], res_ring[n]) for n in TENSORS)
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
    flag_list = [None] * W
    spec3_list = [None] * W
    events_list = [None] * W
    if W > 1:
        dist.all_gather_object(flag_list, flags)
        dist.all_gather_object(spec3_list, spec3)
        dist.all_gather_object(events_list, events)
    else:
        flag_list = [flags]
        spec3_list = [spec3]
        events_list = [events]
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
        mp.start_processes(
            _spawn_entry,
            args=(W, row, _free_port(), result_dir),
            nprocs=W,
            join=True,
            start_method="spawn",
        )
        with open(os.path.join(result_dir, _row_id(row) + ".json")) as fh:
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
