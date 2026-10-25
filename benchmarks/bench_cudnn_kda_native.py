# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure warm public KDA host enqueue cost; GPU completion is outside timing."""

import argparse
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import cudnn
import torch

from flashinfer import _kda_cudnn_fast as fast
from flashinfer.cudnn import linear_attention as la
from flashinfer.kda import recurrent_kda


def bind_chain(args):
    """Bind the same compiled chain without public validation or native checks."""
    from cudnn.frost.workspace import Workspace

    q, k, v, g, beta = (args[name].squeeze(0) for name in ("q", "k", "v", "g", "beta"))
    state = args["initial_state"]
    out = args["output"].squeeze(0)
    graph, _ = la._build_la_graph(
        "kda",
        q,
        k,
        v,
        g,
        beta,
        args["cu_seqlens"],
        out,
        a_log=args["A_log"],
        dt_bias=args["dt_bias"],
        initial_state=state,
        final_state=state,
        scale=1 / math.sqrt(128),
        use_qk_l2norm=True,
        use_beta_sigmoid=True,
        safe_gate=True,
        gate_lower_bound=-5.0,
        batch_invariant=False,
        overwrite_initial_state=True,
    )
    plan = graph._compiled_plans[graph._plan_index]
    compiled = plan.compiled
    assert graph._fi_la_overwrite and compiled.chain
    workspace = torch.empty(
        graph._fi_la_workspace_size, dtype=torch.uint8, device=q.device
    )
    buffers = (
        q,
        k,
        v,
        g,
        beta,
        args["cu_seqlens"],
        out,
        args["A_log"],
        args["dt_bias"],
        state,
        state,
    )
    pack = graph._normalize_ordered(
        buffers, graph._fi_la_uids, workspace, None, None, None
    )
    views = pack.operands(plan.indices)
    ws = Workspace.over(pack, compiled.workspace_size, type(compiled).__name__)
    operands = compiled.chain_buffers((*views, *ws.carve(compiled.carve), None))
    frame = (
        *compiled.chain_scalars,
        *operands,
        compiled.stream_type(torch.cuda.current_stream().cuda_stream),
    )
    owners = (graph, plan, compiled, workspace, pack, views, ws, operands)

    def run():
        assert owners
        compiled.chain_launch(*frame)
        torch.autograd.graph.increment_version(state)
        return args["output"], state

    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=61)
    ns = parser.parse_args()
    if ns.output.exists():
        raise FileExistsError(ns.output)
    if ns.repeats < 5:
        parser.error("--repeats must be at least 5")
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    torch.set_num_threads(1)
    torch.manual_seed(42)
    with torch.inference_mode():
        shape = (1, ns.tokens, ns.heads, 128)
        args = {
            name: torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            for name in ("q", "k", "v", "g")
        }
        args.update(
            beta=torch.randn(
                1, ns.tokens, ns.heads, device="cuda", dtype=torch.bfloat16
            ),
            A_log=torch.randn(ns.heads, device="cuda") * 0.1,
            dt_bias=torch.randn(ns.heads, 128, device="cuda") * 0.1,
            initial_state=torch.randn(
                1, ns.heads, 128, 128, device="cuda", dtype=torch.bfloat16
            )
            * 0.05,
            output=torch.empty_like(args["v"]),
            cu_seqlens=torch.tensor([0, ns.tokens], device="cuda", dtype=torch.int32),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            beta_is_logit=True,
            lower_bound=-5.0,
            backend="cudnn",
        )
        seed = args["initial_state"].clone()
        public = lambda: recurrent_kda(**args)
        fast._enabled = False
        for _ in range(5):
            public()
        torch.cuda.synchronize()
        fast.clear()
        fast._enabled = True
        start = time.perf_counter_ns()
        public()
        setup_us = (time.perf_counter_ns() - start) / 1000
        torch.cuda.synchronize()
        assert fast.stats()["entries"], "Native path unavailable"
        chain = bind_chain(args)
        calls = {
            "public_native_disabled": (public, False),
            "public_native_enabled": (public, True),
            "bound_compiled_chain": (chain, False),
        }
        for count in (1, 8):
            reference = None
            for fn, enabled in calls.values():
                fast._enabled = enabled
                args["initial_state"].copy_(seed)
                args["output"].fill_(float("nan"))
                for _ in range(count):
                    fn()
                torch.cuda.synchronize()
                actual = (args["output"].clone(), args["initial_state"].clone())
                assert all(torch.isfinite(x).all() for x in actual)
                if reference is None:
                    reference = actual
                else:
                    for x, y in zip(actual, reference, strict=True):
                        torch.testing.assert_close(x, y, rtol=0, atol=0)
        for fn, enabled in calls.values():
            fast._enabled = enabled
            for _ in range(30):
                fn()
        torch.cuda.synchronize()
        rng = random.Random(42)
        samples = []
        before = fast.stats()
        for count in (1, 8):
            for repeat in range(ns.repeats):
                order = list(calls)
                rng.shuffle(order)
                for name in order:
                    fn, fast._enabled = calls[name]
                    args["initial_state"].copy_(seed)
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()
                    for _ in range(count):
                        fn()
                    elapsed = (time.perf_counter_ns() - start) / count / 1000
                    torch.cuda.synchronize()
                    samples.append(
                        dict(stage=name, calls=count, repeat=repeat, enqueue_us=elapsed)
                    )
        assert fast.stats()["hits"] - before["hits"] == ns.repeats * 9
        summary = [
            dict(
                stage=name,
                calls=count,
                median_enqueue_us=statistics.median(
                    row["enqueue_us"]
                    for row in samples
                    if row["stage"] == name and row["calls"] == count
                ),
            )
            for count in (1, 8)
            for name in calls
        ]
        fast.clear()
    result = dict(
        summary=summary,
        samples=samples,
        correctness="exact output/state match for 1 and 8 updates",
        first_native_call_after_frontend_warmup_us=setup_us,
        workload=dict(
            tokens=ns.tokens,
            heads=ns.heads,
            dim=128,
            dtype="BF16",
            state="in-place; reset before each sample",
        ),
        environment=dict(
            torch=torch.__version__,
            frontend=cudnn.__version__,
            gpu=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),
            affinity=sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
        ),
        scope="Host enqueue only; compilation, preparation, synchronization and state reset excluded",
    )
    ns.output.parent.mkdir(parents=True, exist_ok=True)
    ns.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
