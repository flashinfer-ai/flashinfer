# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Mixed routed/shared computation over EP, including reusable graph storage."""

import pytest
import torch
import torch.distributed as dist

from flashinfer.moe_ep import PrequantizedMoEWeights
from flashinfer.moe_ep.backends.mega.kernel.sm107.shared_weights import (
    preprocess_shared_weights,
)
from tests.moe_ep.sm107_test_utils import (
    assert_reference,
    canonical_reference,
    quantize,
    weight_pack,
)
from tests.moe_ep.test_sm107_auto_multirank import rubin_bootstrap  # noqa: F401
from tests.moe_ep import test_moe_ep_sm107_block_scaled_mega_multirank as problem


@pytest.mark.gpu_2
@pytest.mark.arch_rubin
@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("schedule", ["grouped", "phase_interleave"])
@pytest.mark.parametrize("clusters", ["uniform", "mixed", "all_fallback"])
def test_shared_forward_replay(early, schedule, clusters, rubin_bootstrap, monkeypatch):  # noqa: F811
    import cutlass.cute as cute
    from flashinfer.moe_ep.kernel_src.sm107 import next_cutedsl_megamoe as pkg
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import (
        block_scaled,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm107.mxfp8_mxfp4_bf16_cutedsl import (
        staging,
        weights,
    )

    pkg.require_sm107_dsl()
    pkg.bootstrap_paths()
    import sources
    from sources.api import ProblemDesc

    rank, world = rubin_bootstrap.rank, rubin_bootstrap.world_size
    w13, w2, x, ids, scores = problem._global_problem(world)
    capacity = 256
    repeat = (capacity + problem.NUM_TOKENS - 1) // problem.NUM_TOKENS
    x, ids, scores = (t.repeat(1, repeat, 1)[:, :capacity] for t in (x, ids, scores))
    pack, _, _ = weight_pack(w13, w2, "mxfp4_mxfp8")
    count = problem.NUM_EXPERTS // world
    local = slice(rank * count, (rank + 1) * count)
    local_pack = PrequantizedMoEWeights(
        *(getattr(pack, name)[local] for name in ("w13", "w2", "w13_scale", "w2_scale"))
    )
    transformed = weights.preprocess_mega_weights(
        local_pack, hidden_size=problem.HIDDEN, intermediate_size=problem.INTERMEDIATE
    )
    shared_pack, _, _ = weight_pack(w13[:1], w2[:1], "mxfp8_e4m3")
    shared = preprocess_shared_weights(
        shared_pack.w13[0],
        shared_pack.w2[0],
        shared_pack.w13_scale[0],
        shared_pack.w2_scale[0],
        hidden_size=problem.HIDDEN,
        intermediate_size=problem.INTERMEDIATE,
    )
    # Exercise the kernel before exposing shared ownership in the public adapter.
    kernel_type = sources.RubinInferenceMegaMoE

    def shared_kernel(desc, impl):
        return kernel_type(
            ProblemDesc({**desc, "shared_quant_kind": "mxfp8_e4m3"}), impl
        )

    with monkeypatch.context() as patch:
        patch.setattr(sources, "RubinInferenceMegaMoE", shared_kernel)
        ws = pkg.get_symm_buffer_for_sm107_block_scaled_mega_moe(
            problem.NUM_EXPERTS,
            capacity,
            problem.TOP_K,
            problem.HIDDEN,
            problem.INTERMEDIATE,
            rank,
            world,
            quant_kind="mxfp4_mxfp8",
            schedule_policy=(schedule, None),
            work_id_mode="atomic_counter"
            if schedule == "phase_interleave"
            else "grid_stride",
            cluster_shape_mn=(4, 1) if clusters != "uniform" else (2, 1),
            fallback_cluster_shape_mn=(2, 1) if clusters != "uniform" else None,
            apply_routing_weights_before_fc2=early,
        )
    graphs = []
    try:
        prepared_shared = tuple(
            block_scaled._to_cute(tensor) for pair in shared for tensor in pair
        )

        def stage(live, step, shared_only):
            hidden = x[rank, :live] * (1.0 + step * 0.125)
            route_ids = ids[rank, :live].clone()
            route_scores = scores[rank, :live].clone()
            if live:
                route_ids[::3] = -1
                route_scores[::3] = float("nan")
                if step:
                    route_ids[1::3, 0] = -1
                    route_scores[1::3, 0] = float("inf")
                if shared_only:
                    route_ids.fill_(-1)
                    route_scores.fill_(float("inf"))
            staging.stage_mega_moe_inputs(
                hidden,
                route_scores,
                route_ids,
                ws.x,
                ws.x_sf,
                ws.topk_idx,
                ws.topk_weights,
                kind="mxfp4_mxfp8",
            )
            return hidden, route_ids, route_scores

        def reference(hidden, route_ids, route_scores):
            xq, xsf = quantize(hidden, "mxfp4_mxfp8")
            routed = canonical_reference(
                xq,
                xsf,
                route_ids,
                route_scores,
                pack,
                "mxfp4_mxfp8",
                early=early,
            )
            shared_ref = canonical_reference(
                xq,
                xsf,
                torch.zeros((hidden.shape[0], 1), dtype=torch.int32, device="cuda"),
                torch.ones((hidden.shape[0], 1), device="cuda"),
                shared_pack,
                "mxfp8_e4m3",
                early=False,
            )
            return (routed.float() + shared_ref.float()).to(torch.bfloat16)

        compiled = None
        for live, shared_only in (
            (129 - rank, False),
            (127 - rank, False),
            (128 - rank, False),
            (129 - rank, True),
            (1 if rank == 0 else 0, True),
        ):
            output = torch.empty(
                (live, problem.HIDDEN), device="cuda", dtype=torch.bfloat16
            )
            kwargs = ws._runtime_kwargs(*transformed, output=output)
            kwargs["shared_weights"] = prepared_shared
            if compiled is None:
                if clusters == "all_fallback":
                    from cutlass.cutlass_dsl.cutlass import KernelLauncher

                    original_launch = KernelLauncher.launch
                    forced = []

                    def force_fallback(launcher, *args, **launch_kwargs):
                        if "fallback_cluster" in launch_kwargs:
                            launch_kwargs["cluster"] = launch_kwargs["fallback_cluster"]
                            forced.append(True)
                        return original_launch(launcher, *args, **launch_kwargs)

                    # Exercise more physical fallback groups than the occupancy
                    # recipe predicts, without changing the kernel's work mapping.
                    with monkeypatch.context() as patch:
                        patch.setattr(KernelLauncher, "launch", force_fallback)
                        compiled = cute.compile(ws.kernel, **kwargs)
                    assert forced
                else:
                    compiled = cute.compile(ws.kernel, **kwargs)

            def launch():
                kwargs["stream"] = block_scaled._cu_stream(torch.cuda.current_stream())
                compiled(**kwargs)

            inputs = stage(live, 0, shared_only)
            launch()
            torch.cuda.synchronize()
            assert_reference(output, reference(*inputs), "mxfp4_mxfp8")
            graph = torch.cuda.CUDAGraph()
            graphs.append((graph, output, live, shared_only))
            dist.barrier()
            with torch.cuda.graph(graph):
                launch()
            for step in (1, 0, 2):
                inputs = stage(live, step, shared_only)
                graph.replay()
                torch.cuda.synchronize()
                assert_reference(output, reference(*inputs), "mxfp4_mxfp8")
        # Revisit captured counts after smaller and shared-only calls reset the workspace.
        for graph, output, live, shared_only in reversed(graphs):
            inputs = stage(live, 3, shared_only)
            graph.replay()
            torch.cuda.synchronize()
            assert_reference(output, reference(*inputs), "mxfp4_mxfp8")
    finally:
        for graph, _, _, _ in graphs:
            graph.reset()
        ws.destroy()
