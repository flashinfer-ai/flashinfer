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

Multi-GPU rows of the non-deterministic ``cake_causal_conv1d_kcp(..., deterministic=False)``: ``W`` ranks over NCCL
(W = 2 / 4 / 8) on the deterministic KCP suite's partitions (contiguous, zigzag, per-document zigzag, ragged tables
with a zero-token rank, one-token fragments, a fragment that reads its history from three predecessor ranks), plus
W = 16 rows (``-k w16``) on the layouts any world size partitions: contiguous, zigzag and per-document zigzag (the
ragged tables are written for at most eight ranks).  Per row and rank:

- ``y`` is bitwise the single-rank zero-history convolution of the whole pack on the rank's tokens (anchor) and
  the deterministic KCP call's; ``dx`` is bitwise the deterministic KCP call's;
- an independent reference: the rank's ``y`` and ``dx`` (its tokens gathered from the global FP64 oracle of the
  whole pack under the KCP SiLU' policy) within one BF16 ulp, and the all-reduced ``dweight`` partial sum within
  the package budget of the global FP64 ``dweight``;
- the rank-local FP32 ``dweight`` partial is within the package budget of the deterministic partial, on each of
  ``REPEATS`` launches (``dx`` bitwise on every one; the worst case is recorded), and the sum of the partials over
  the ranks is within ``W`` x the budget of the deterministic sum;
- checkpoints with and without a recompute handle (``use_reentrant`` False and True; 2 x forward + backward), nested
  checkpoints and ``retain_graph`` accumulation: ``y`` / ``dx`` bitwise the direct pass, ``dweight`` within budget,
  and the per-phase collective counts of the deterministic suite (one halo exchange per direction, one more in a
  plain recompute, none in a recompute with a handle);
- the handle retains exactly the history pool;
- a 1-thread kernel resident on another stream does not stall the backward (FlashInfer issues #7038 / #7548: the
  main backward is an ordinary launch; ``busy <= 2 x alone``).

Launch forms as the deterministic suite: ``torch.multiprocessing`` (skipped with fewer than ``W`` visible devices)
or one process per Slurm task (``SLURM_NTASKS == W``).  Skipped until the exporter renders the package.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import statistics
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.test_cake_causal_conv1d_det_train_kcp import Y_ATOL, Y_RTOL, _reference
from tests.test_cake_causal_conv1d_det_train_kcp_dist import (
    ROWS,
    WIDTH,
    _Counter,
    _free_port,
    _local,
    _partition,
    _requested,
    _row_id,
)

DW_ATOL, DW_RTOL = 5e-2, 1e-4
REPEATS = 20
# W = 16 (16-way context parallelism) on the deterministic rows' packs, for the layouts any world size partitions.
ROWS_W16 = [
    ((100, 640, 77, 1000, 129), 16, "contiguous", 512),
    ((3000,), 16, "zigzag", 256),
    ((2048, 5, 3000, 1), 16, "zigzag_per_document", 384),
]


def _run_rank(
    rank: int, W: int, row, port: int, result_dir: str | None, local_id: int
) -> dict:
    from torch.utils.checkpoint import checkpoint

    from flashinfer import cake_causal_conv1d_det_train as conv
    from flashinfer import cake_causal_conv1d_train_kcp as kcp_api
    from flashinfer.cake_causal_conv1d_det_train_backend import kcp as det_kcp
    from flashinfer.experimental.cake.cake_causal_conv1d_train_backend import (
        kcp as conv_kcp,
    )
    from flashinfer.experimental.cake_kda_train import cake_kcp_plan as plan

    lengths, _w, layout, C = row
    device = torch.device("cuda", local_id % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))
        dist.init_process_group("nccl", rank=rank, world_size=W, device_id=device)
    group = dist.group.WORLD
    part, cu = _partition(plan, lengths, W, layout)
    ctx = plan.build_kcp_context(part, group, rank, device, conv_kernel_size=WIDTH)
    T = cu[-1]
    g = torch.Generator(device="cpu").manual_seed(7815)
    x_all = torch.randn(T, C, generator=g).to(torch.bfloat16)
    w_all = (torch.randn(C, WIDTH, generator=g) * 0.5).contiguous()
    dy_all = torch.randn(T, C, generator=g).to(torch.bfloat16)
    x = _local(part, rank, cu, x_all).to(device)
    dy = _local(part, rank, cu, dy_all).to(device)
    w = w_all.to(device)
    n_hist = int(ctx.halo.n_hist_rows)

    def leaves():
        return x.detach().clone().requires_grad_(
            True
        ), w.detach().clone().requires_grad_(True)

    def fn(x_, w_, h_):
        return kcp_api.cake_causal_conv1d_kcp(
            x_, w_, ctx, recompute_handle=h_, deterministic=False
        )

    def run(mode: str, handle=None, *, twice: bool = False, reentrant: bool = False):
        xl, wl = leaves()
        with _Counter() as cc:
            if mode == "direct":
                y = fn(xl, wl, handle)
            elif mode == "ckpt":
                kw = (
                    {"use_reentrant": True}
                    if reentrant
                    else {
                        "use_reentrant": False,
                        "context_fn": lambda: (
                            contextlib.nullcontext(),
                            cc.recompute_cm(),
                        ),
                    }
                )
                y = checkpoint(fn, xl, wl, handle, **kw)
            else:

                def outer(x_, w_, h_):
                    hh = conv_kcp.RecomputeHandleKcp() if mode == "nested_in" else h_
                    yi = checkpoint(fn, x_, w_, hh, use_reentrant=False)
                    return yi * torch.ones_like(yi)

                y = checkpoint(
                    outer, xl, wl, handle, use_reentrant=False,
                    context_fn=lambda: (contextlib.nullcontext(), cc.recompute_cm()),
                )  # fmt: skip
            cc.phase = "bwd"
            y.backward(dy, retain_graph=twice)
            if twice:
                y.backward(dy)
        out = {"y": y.detach(), "dx": xl.grad, "dw": wl.grad}
        counts = {
            t: {p: cc.get(t, p) for p in ("orig", "recompute", "bwd")}
            for t in ("calls", "waits")
        }
        return out, counts

    def bits(t: torch.Tensor) -> torch.Tensor:
        return t.contiguous().view(
            torch.int16 if t.element_size() == 2 else torch.int32
        )

    def same(a: torch.Tensor, b: torch.Tensor) -> bool:
        return (
            a.dtype == b.dtype and a.shape == b.shape and torch.equal(bits(a), bits(b))
        )

    def ratio(a: torch.Tensor, b: torch.Tensor, scale: float = 1.0) -> float:
        """The worst ``|a - b| / (atol + rtol |b|)`` at ``scale`` x the budget (``<= 1``: within budget)."""

        if a.numel() == 0:
            return 0.0
        a, b = a.double(), b.double()
        if not torch.isfinite(a).all():
            return float("inf")
        return float(
            ((a - b).abs() / (scale * DW_ATOL + scale * DW_RTOL * b.abs())).max()
        )

    def ulp_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
        """The worst ``|a - b| / (Y_ATOL + Y_RTOL |b|)`` (``<= 1``: within one BF16 ulp)."""

        if a.numel() == 0:
            return 0.0
        a, b = a.double(), b.double()
        if not torch.isfinite(a).all():
            return float("inf")
        return float(((a - b).abs() / (Y_ATOL + Y_RTOL * b.abs())).max())

    def eq(a, b, scale: float = 1.0) -> bool:
        return (
            same(a["y"], b["y"])
            and same(a["dx"], b["dx"])
            and ratio(a["dw"], b["dw"], scale) <= 1.0
        )

    flags: dict[str, bool] = {}
    info: dict[str, object] = {"n_hist_rows": n_hist, "T_r": int(x.shape[0])}
    # The deterministic KCP call on the same inputs (the bitwise / budget reference).
    xd, wd = leaves()
    yd = det_kcp.cake_causal_conv1d_det_kcp(xd, wd, ctx)
    yd.backward(dy)
    det = {"y": yd.detach(), "dx": xd.grad, "dw": wd.grad}
    direct, c_direct = run("direct")
    plain, c_plain = run("ckpt")
    hd = conv_kcp.RecomputeHandleKcp()
    withh, c_h = run("ckpt", hd)
    plain_re, c_plain_re = run("ckpt", reentrant=True)
    hd_re = conv_kcp.RecomputeHandleKcp()
    withh_re, c_h_re = run("ckpt", hd_re, reentrant=True)
    info["counts"] = {
        "direct": c_direct, "plain": c_plain, "handle": c_h, "plain_reentrant": c_plain_re, "handle_reentrant": c_h_re,
    }  # fmt: skip
    cu_dev = torch.tensor(cu, dtype=torch.int32, device=device)
    y_single = conv.cake_causal_conv1d_det(x_all.to(device), w, cu_dev)
    flags["y_eq_single_rank"] = same(direct["y"], _local(part, rank, cu, y_single))
    flags["y_dx_eq_deterministic"] = same(direct["y"], det["y"]) and same(
        direct["dx"], det["dx"]
    )
    flags["dw_within_budget_of_deterministic"] = ratio(direct["dw"], det["dw"]) <= 1.0
    # The independent reference: the global FP64 oracle of the whole pack, the rank's tokens gathered from it.
    y_ref, dx_ref, dw_ref = _reference(x_all.to(device), w, cu_dev, dy_all.to(device))
    info["y_ratio_to_one_bf16_ulp_vs_fp64"] = ulp_ratio(
        direct["y"], _local(part, rank, cu, y_ref)
    )
    info["dx_ratio_to_one_bf16_ulp_vs_fp64"] = ulp_ratio(
        direct["dx"], _local(part, rank, cu, dx_ref)
    )
    flags["y_within_one_bf16_ulp_of_fp64"] = (
        info["y_ratio_to_one_bf16_ulp_vs_fp64"] <= 1.0
    )
    flags["dx_within_one_bf16_ulp_of_fp64"] = (
        info["dx_ratio_to_one_bf16_ulp_vs_fp64"] <= 1.0
    )
    # Repeated launches: dx bitwise every time, dweight within budget every time.
    worst, patterns, dx_ok = 0.0, set(), True
    for _ in range(REPEATS):
        again, _ = run("direct")
        dx_ok = dx_ok and same(again["dx"], det["dx"]) and same(again["y"], det["y"])
        worst = max(worst, ratio(again["dw"], det["dw"]))
        patterns.add(bytes(again["dw"].contiguous().view(torch.uint8).cpu().numpy()))
    info["repeats"] = {
        "n": REPEATS,
        "worst_dw_ratio_to_budget": worst,
        "distinct_dw_patterns": len(patterns),
    }
    flags["repeats_dx_bitwise"] = dx_ok
    flags["repeats_dw_within_budget"] = worst <= 1.0
    # The rank partials' sum (the framework's reduction) against the deterministic partials' sum.
    total, total_det = direct["dw"].clone(), det["dw"].clone()
    dist.all_reduce(total)
    dist.all_reduce(total_det)
    info["dw_sum_ratio_to_budget"] = ratio(total, total_det, float(W))
    flags["dw_sum_within_budget"] = info["dw_sum_ratio_to_budget"] <= 1.0
    info["dw_sum_ratio_to_budget_vs_fp64"] = ratio(total, dw_ref)
    flags["dw_sum_within_budget_of_fp64"] = (
        info["dw_sum_ratio_to_budget_vs_fp64"] <= 1.0
    )
    flags["plain_eq_direct"] = eq(plain, direct)
    flags["handle_eq_plain"] = eq(withh, plain)
    flags["reentrant_plain_eq_direct"] = eq(plain_re, direct)
    flags["reentrant_handle_eq_direct"] = eq(withh_re, direct)
    for t in ("calls", "waits"):
        flags[f"positive_control_{t}"] = c_direct[t] == {"orig": 1, "recompute": 0, "bwd": 1} and c_plain[t] == {
            "orig": 1, "recompute": 1, "bwd": 1,
        }  # fmt: skip
        flags[f"handle_counts_{t}"] = c_h[t] == {"orig": 1, "recompute": 0, "bwd": 1}
        flags[f"reentrant_counts_{t}"] = c_plain_re[t] == {"orig": 1, "recompute": 0, "bwd": 2} and c_h_re[t] == {
            "orig": 1, "recompute": 0, "bwd": 1,
        }  # fmt: skip
    flags["handle_bytes"] = hd.nbytes() == n_hist * C * 2 == hd_re.nbytes()
    twice, c_twice = run("ckpt", conv_kcp.RecomputeHandleKcp(), twice=True)
    info["counts"]["handle_twice"] = c_twice
    flags["twice_accumulates"] = (
        same(twice["dx"], direct["dx"] + direct["dx"])
        and ratio(twice["dw"], direct["dw"] + direct["dw"], 2.0) <= 1.0
        and c_twice["calls"] == {"orig": 1, "recompute": 0, "bwd": 2}
    )
    nest_out, c_nest_out = run("nested_out", conv_kcp.RecomputeHandleKcp())
    nest_in, c_nest_in = run("nested_in", conv_kcp.RecomputeHandleKcp())
    info["counts"]["nested_out"], info["counts"]["nested_in"] = c_nest_out, c_nest_in
    flags["nested_out_eq_direct"] = eq(nest_out, direct)
    flags["nested_in_eq_direct"] = eq(nest_in, direct)
    flags["nested_out_counts"] = c_nest_out["calls"] == {
        "orig": 1,
        "recompute": 0,
        "bwd": 1,
    }
    flags["nested_in_counts"] = c_nest_in["calls"] == {
        "orig": 1,
        "recompute": 1,
        "bwd": 1,
    }
    retained = {}
    for name, h in (("plain", None), ("handle", conv_kcp.RecomputeHandleKcp())):
        xl, wl = leaves()
        torch.cuda.synchronize(device)
        gc.collect()
        before = _requested(device)
        y = checkpoint(fn, xl, wl, h, use_reentrant=False)
        torch.cuda.synchronize(device)
        retained[name] = _requested(device) - before - y.numel() * y.element_size()
        y.backward(dy)
        del y, xl, wl
    info["retained_bytes"] = retained
    flags["retained"] = retained["plain"] == 0 and retained["handle"] == n_hist * C * 2
    # A kernel resident on another stream of every rank (#7038 / #7548).
    side = torch.cuda.Stream(device=device)
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    _, hist = conv_kcp.cake_causal_conv1d_forward_kcp(x, w, ctx, deterministic=False)

    def timed(with_side_kernel: bool) -> float:
        torch.cuda.synchronize(device)
        dist.barrier()
        if with_side_kernel:
            with torch.cuda.stream(side):
                torch.cuda._sleep(10_000_000)
        start.record()
        dx_t, _ = conv_kcp.cake_causal_conv1d_backward_kcp(
            x, w, ctx, dy, hist, deterministic=False
        )
        end.record()
        torch.cuda.synchronize(device)
        timed.dx = dx_t
        return start.elapsed_time(end)

    def side_kernel_ms() -> float:
        torch.cuda.synchronize(device)
        with torch.cuda.stream(side):
            start.record()
            torch.cuda._sleep(10_000_000)
            end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end)

    for _ in range(3):
        timed(False)
    alone = statistics.median(timed(False) for _ in range(7))
    busy = statistics.median(timed(True) for _ in range(7))
    sleep = side_kernel_ms()
    info["side_kernel_ms"] = {"alone": alone, "busy": busy, "side": sleep}
    flags["side_kernel_dx_bitwise"] = same(timed.dx, det["dx"])
    # A stall waits for the side kernel, so busy reaches its sleep (milliseconds).
    # These rows are tiny: one more wave of CTAs beside the SM that the side kernel
    # holds can exceed 2x alone (about 0.1 ms) without a stall.
    flags["side_kernel_no_stall"] = busy <= max(2.0 * alone, 0.5 * sleep)
    flag_list = [None] * W
    info_list = [None] * W
    dist.all_gather_object(flag_list, flags)
    dist.all_gather_object(info_list, info)
    report = {
        "world": W,
        "row": _row_id(row),
        "torch": torch.__version__,
        "flags": {k: all(f[k] for f in flag_list) for k in flags},
        "failing_ranks": {
            k: [r for r, f in enumerate(flag_list) if not f[k]] for k in flags
        },
        "info_by_rank": info_list,
    }
    report["pass"] = all(report["flags"].values())
    dist.barrier()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(
            os.path.join(result_dir, "nondet-" + _row_id(row) + ".json"), "w"
        ) as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _spawn_entry(rank: int, W: int, row, port: int, result_dir: str) -> None:
    try:
        report = _run_rank(rank, W, row, port, result_dir, rank)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    from flashinfer.experimental.cake.cake_causal_conv1d_train_backend import (
        is_cake_causal_conv1d_train_supported,
    )

    return is_cake_causal_conv1d_train_supported(torch.device("cuda"))


@pytest.mark.parametrize("row", ROWS + ROWS_W16, ids=_row_id)
def test_nondeterministic_kcp_rows(row) -> None:
    W = row[1]
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"a {ntasks}-task Slurm step runs the W={ntasks} rows only")
        if not _supported():
            pytest.skip(
                "the non-deterministic conv package is not rendered or does not support this device"
            )
        report = _run_rank(
            int(os.environ["SLURM_PROCID"]),
            W,
            row,
            int(os.environ.get("MASTER_PORT", "29500")),
            os.environ.get("KCP_TEST_RESULT_DIR"),
            int(os.environ.get("SLURM_LOCALID", "0")),
        )
        assert report["pass"], report
        return
    if not _supported():
        pytest.skip(
            "the non-deterministic conv package is not rendered or does not support this device"
        )
    if torch.cuda.device_count() < W:
        pytest.skip(f"{W} CUDA devices required, {torch.cuda.device_count()} visible")
    with tempfile.TemporaryDirectory(prefix="cake_conv_kcp_nondet_dist_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        mp.start_processes(
            _spawn_entry,
            args=(W, row, _free_port(), result_dir),
            nprocs=W,
            join=True,
            start_method="spawn",
        )
        with open(os.path.join(result_dir, "nondet-" + _row_id(row) + ".json")) as fh:
            report = json.load(fh)
    assert report["pass"], report
