# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warm cuDNN KDA submission and CUDA graph replay, not serving TTFT.

Run the same command on baseline/candidate checkouts in separate processes:
    python benchmarks/bench_cudnn_linear_attention_host.py --output kda.json

--tokens is per request. BF16 inputs/state, D128, caller-owned O, fused Q/K
normalization, beta sigmoid and safe gate (bound -5) are identical across runs.
Host timers exclude state reset and synchronization; completed wall includes
synchronization. Replay includes the identical state reset before every call.
Checks compare with the cuDNN adapter's separate-output-state contract, not an
independent mathematical reference. Inspect versions/routes before comparing.
"""

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from functools import partial
from pathlib import Path

import cudnn
import flashinfer
import torch
from flashinfer.collect_env import collect_env_info
from flashinfer.cudnn import linear_attention as la


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (
        ("tokens", 8192),
        ("heads", 16),
        ("batch", 1),
        ("iters", 100),
        ("unroll", 10),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.tokens, args.heads, args.batch, args.iters, args.unroll) <= 0:
        parser.error("shapes and iteration counts must be positive")
    torch.set_num_threads(1)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    torch.manual_seed(42)
    shape = (1, args.batch * args.tokens, args.heads, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k, v, gate = torch.randn_like(q), torch.randn_like(q), torch.randn_like(q)
    beta = torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16)
    seed = (
        torch.randn(args.batch, args.heads, 128, 128, device="cuda", dtype=q.dtype)
        * 0.1
    )
    state, out = torch.empty_like(seed), torch.empty_like(v)
    ref_state, ref_out = torch.empty_like(seed), torch.empty_like(v)
    kwargs = dict(
        A_log=torch.zeros(args.heads, device="cuda"),
        dt_bias=torch.zeros(args.heads * 128, device="cuda"),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        beta_is_logit=True,
        cu_seqlens=torch.arange(args.batch + 1, device="cuda", dtype=torch.int64)
        * args.tokens,
    )
    run = partial(
        flashinfer.recurrent_kda,
        q,
        k,
        v,
        gate,
        beta,
        initial_state=state,
        output=out,
        backend="cudnn",
        **kwargs,
    )
    reference = partial(
        la.cudnn_recurrent_kda,
        q,
        k,
        v,
        gate,
        beta,
        initial_state=seed,
        output=ref_out,
        output_state=ref_state,
        **kwargs,
    )

    def check(replay=None):
        saved_seed = seed.clone()
        reference()
        state.copy_(seed)
        out.fill_(float("nan"))
        if replay is None:
            result = run()
            assert result[1].data_ptr() == state.data_ptr()
        else:
            replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(seed, saved_seed, rtol=0, atol=0)
        results = {}
        for name, actual, expected in (
            ("O", out, ref_out),
            ("state", state, ref_state),
        ):
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
            torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)
            results[name] = dict(
                bitwise_equal=torch.equal(actual, expected),
                max_abs=(actual.float() - expected.float()).abs().max().item(),
            )
        return results

    checks = {"eager": check()}
    for _ in range(20):
        state.copy_(seed)
        run()
    torch.cuda.synchronize()
    captured = torch.cuda.CUDAGraph()
    with torch.cuda.graph(captured):
        for _ in range(args.unroll):
            state.copy_(seed)
            run()
    checks["capture"] = check(captured.replay)
    samples = {
        name: []
        for name in (
            "cpu_active_us",
            "enqueue_wall_us",
            "completed_wall_us",
            "replay_with_reset_us",
        )
    }
    was_gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(args.iters):
            state.copy_(seed)
            torch.cuda.synchronize()
            c0, t0 = time.thread_time_ns(), time.perf_counter_ns()
            run()
            t1, c1 = time.perf_counter_ns(), time.thread_time_ns()
            samples["cpu_active_us"].append((c1 - c0) / 1e3)
            samples["enqueue_wall_us"].append((t1 - t0) / 1e3)
            torch.cuda.synchronize()
            state.copy_(seed)
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            run()
            torch.cuda.synchronize()
            samples["completed_wall_us"].append((time.perf_counter_ns() - t0) / 1e3)
    finally:
        if was_gc_enabled:
            gc.enable()
    torch.testing.assert_close(out, ref_out, rtol=args.rtol, atol=args.atol)
    torch.testing.assert_close(state, ref_state, rtol=args.rtol, atol=args.atol)
    checks["after_host_timing"] = check()
    for _ in range(5):
        captured.replay()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    for _ in range(20):
        start.record()
        captured.replay()
        end.record()
        end.synchronize()
        samples["replay_with_reset_us"].append(
            start.elapsed_time(end) * 1000 / args.unroll
        )
    # Inspect the actual final timed replay before a fresh call can overwrite it.
    torch.testing.assert_close(out, ref_out, rtol=args.rtol, atol=args.atol)
    torch.testing.assert_close(state, ref_state, rtol=args.rtol, atol=args.atol)
    checks["after_replay_timing"] = check(captured.replay)
    old_out, old_state = out.clone(), state.clone()
    q.mul_(0.8).add_(0.04)
    k.mul_(0.9).sub_(0.03)
    v.add_(0.05)
    gate.add_(0.2)
    beta.sub_(0.1)
    seed.add_(0.15)
    checks["changed_eager"] = check()
    checks["changed_replay"] = check(captured.replay)
    assert not torch.equal(out, old_out) and not torch.equal(state, old_state)
    paths = {
        "benchmark": Path(__file__),
        "adapter": Path(la.__file__),
        "public_kda": Path(flashinfer.__file__).parent / "kda.py",
    }
    record = dict(
        args=vars(args),
        checks=checks,
        scope="public recurrent_kda backend=cudnn; caller O; warm single-call host clocks; excludes compilation and serving work; adapter state scratch/copy remains included when needed; CUDA replay includes state reset",
        affinity=sorted(os.sched_getaffinity(0)),
        versions=dict(
            flashinfer=flashinfer.__version__,
            frontend=cudnn.__version__,
            backend=cudnn.backend_version(),
            torch=torch.__version__,
        ),
        sources={
            name: dict(
                path=str(path.resolve()),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for name, path in paths.items()
        },
        environment=collect_env_info(),
        measurements={
            name: dict(median=statistics.median(values), samples=values)
            for name, values in samples.items()
        },
    )
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
