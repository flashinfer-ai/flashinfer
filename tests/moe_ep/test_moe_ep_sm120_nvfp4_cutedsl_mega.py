"""SM120 NVFP4 x NVFP4 FlashInfer MegaMoE integration tests."""

from __future__ import annotations

import os

import pytest
import torch


def _packed_e2m1(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randint(
        0,
        256,
        shape,
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    ).view(torch.float4_e2m1fn_x2)


def _problem(
    rank: int,
    world_size: int,
    *,
    tokens: int,
    capacity: int,
    top_k: int = 2,
    hidden: int = 1024,
    intermediate: int = 1024,
    experts: int = 8,
):
    from flashinfer.moe_ep import MoEEpTensors, MoEWeightPack

    local_experts = experts // world_size
    generator = torch.Generator(device="cuda").manual_seed(91 + rank)
    weights = MoEWeightPack(
        _packed_e2m1((local_experts, 2 * intermediate, hidden // 2), generator),
        _packed_e2m1((local_experts, hidden, intermediate // 2), generator),
        torch.ones(
            (local_experts, 2 * intermediate, hidden // 16),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        ),
        torch.ones(
            (local_experts, hidden, intermediate // 16),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        ),
    )
    hidden_states = (
        torch.randn(
            (tokens, hidden),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.05
    )
    rows = torch.arange(tokens, device="cuda")
    slots = torch.arange(top_k, device="cuda")
    topk_ids = ((rows[:, None] * 3 + rank + slots) % experts).to(torch.int32)
    inputs = MoEEpTensors(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=torch.full(
            (tokens, top_k), 1.0 / top_k, dtype=torch.float32, device="cuda"
        ),
    )
    return {
        "capacity": capacity,
        "experts": experts,
        "hidden": hidden,
        "intermediate": intermediate,
        "top_k": top_k,
        "weights": weights,
        "inputs": inputs,
    }


def _make_layer(
    rank: int,
    world_size: int,
    problem: dict,
    *,
    knobs=None,
    input_norm_const=1.0,
    quantize_input=True,
):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    return MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=world_size, rank=rank),
        fleet_params=FleetParams(
            num_experts=problem["experts"],
            max_tokens_per_rank=problem["capacity"],
            token_hidden_size=problem["hidden"],
        ),
        weights=problem["weights"],
        backend=MegaConfig(
            megakernel=Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=problem["intermediate"],
                top_k=problem["top_k"],
                gate_up_clamp=10.0,
                input_norm_const=input_norm_const,
                knobs=knobs,
            ),
            quantize_input=quantize_input,
            preprocess_weights=True,
        ),
    )


@pytest.mark.arch_sm120
def test_sm120_nvfp4_single_rank_replay_and_outer_cuda_graph() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")

    problem = _problem(0, 1, tokens=16, capacity=16)
    layer = _make_layer(0, 1, problem)
    try:
        layer.warmup(problem["inputs"])
        eager0 = layer(problem["inputs"]).clone()
        eager1 = layer(problem["inputs"]).clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer(problem["inputs"])
        graph.replay()
        replay0 = captured.clone()
        graph.replay()
        replay1 = captured.clone()
        torch.cuda.synchronize()
        assert torch.isfinite(eager0).all()
        torch.testing.assert_close(eager0, eager1, atol=0.0, rtol=0.0)
        torch.testing.assert_close(eager0, replay0, atol=0.0, rtol=0.0)
        torch.testing.assert_close(replay0, replay1, atol=0.0, rtol=0.0)

        # Invalid capacity must fail before launching a collective, and must
        # not poison the next valid call or a captured graph's shared workspace.
        from flashinfer.moe_ep import MoEEpConfigError

        for tokens in (17, 64):
            oversized = _problem(0, 1, tokens=tokens, capacity=16)["inputs"]
            with pytest.raises(MoEEpConfigError, match="capacity"):
                layer(oversized)
            torch.testing.assert_close(
                layer(problem["inputs"]), eager0, atol=0.0, rtol=0.0
            )
            graph.replay()
            torch.testing.assert_close(captured, eager0, atol=0.0, rtol=0.0)
    finally:
        layer.destroy()


def _owner_grouped_reference(routes, ids, *, world_size, local_experts):
    partials = []
    for owner in range(world_size):
        value = torch.zeros_like(routes[:, 0], dtype=torch.float32)
        for slot in range(ids.shape[1]):
            selected = (ids[:, slot] // local_experts) == owner
            value += routes[:, slot].float() * selected[:, None]
        partials.append(value.bfloat16())
    total = torch.zeros_like(routes[:, 0], dtype=torch.float32)
    for partial in partials:
        total += partial.float()
    return total.bfloat16()


@pytest.mark.gpu_4
@pytest.mark.arch_sm120
@pytest.mark.parametrize("rank_local_combine", (False, True))
def test_sm120_nvfp4_rank_cache_and_combine_graph_epochs(rank_local_combine) -> None:
    import torch.distributed as dist

    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("requires exactly four ranks")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    tokens = (124, 63, 7, 0)[rank]
    bucket = 128
    problem = _problem(rank, 4, tokens=tokens, capacity=256, top_k=6)
    base_knobs = {"dispatch_rank_cache": False, "rank_local_combine": False}
    candidate_knobs = {
        "dispatch_rank_cache": True,
        "rank_local_combine": rank_local_combine,
    }
    baseline = _make_layer(rank, 4, problem, knobs=base_knobs)
    candidate = _make_layer(rank, 4, problem, knobs=candidate_knobs)
    second_layer = _make_layer(rank, 4, problem, knobs=candidate_knobs)
    try:
        inputs = problem["inputs"]
        for layer in (baseline, candidate, second_layer):
            layer.stage_inputs(inputs, compile_tokens_per_rank=bucket)
            layer.compute_staged(output=None)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            candidate.stage_inputs(inputs, compile_tokens_per_rank=bucket)
            captured0 = candidate.compute_staged(output=None).clone()
            second_layer.stage_inputs(inputs, compile_tokens_per_rank=bucket)
            captured1 = second_layer.compute_staged(output=None).clone()

        storage = candidate._workspace._storages[bucket]
        assert candidate._workspace is second_layer._workspace
        for epoch in range(3):
            # Change the source rows and omit an expert owner in epoch 1.
            active_rows = tokens if epoch != 1 else tokens // 2
            ids = inputs.topk_ids
            rows = torch.arange(tokens, device="cuda")[:, None]
            slots = torch.arange(problem["top_k"], device="cuda")[None, :]
            ids.copy_((rows * 3 + slots + rank + epoch) % (6 if epoch == 1 else 8))
            ids[active_rows:].fill_(-1)
            inputs.hidden_states.mul_(-0.5 if epoch == 1 else 1.25)
            inputs.topk_weights.mul_(0.75)
            torch.cuda.synchronize()
            dist.barrier()
            baseline.stage_inputs(inputs, compile_tokens_per_rank=bucket)
            expected_direct = baseline.compute_staged(output=None).clone()
            routes = (
                baseline._workspace._storages[bucket].combine_output[:tokens].clone()
            )
            expected = (
                _owner_grouped_reference(routes, ids, world_size=4, local_experts=2)
                if rank_local_combine
                else expected_direct
            )
            torch.cuda.synchronize()
            dist.barrier()
            initial_generation = (
                int(storage.rank_combine_ready[-1, 0]) if rank_local_combine else 0
            )
            for _ in range(40):
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    captured0[:active_rows], expected[:active_rows], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    captured1[:active_rows], expected[:active_rows], atol=0, rtol=0
                )
                assert torch.isfinite(captured0[:active_rows]).all()
            if rank_local_combine:
                assert int(storage.rank_combine_ready[-1, 0]) == initial_generation + 80
            dist.barrier()
    finally:
        second_layer.destroy()
        candidate.destroy()
        baseline.destroy()


@pytest.mark.gpu_4
@pytest.mark.arch_sm120
def test_sm120_nvfp4_four_rank_imbalanced_second_epoch() -> None:
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 4:
        pytest.skip("requires exactly four ranks")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    tokens_by_rank = (17, 16, 7, 1)
    problem = _problem(
        rank,
        world_size,
        tokens=tokens_by_rank[rank],
        capacity=32,
    )
    layer = _make_layer(rank, world_size, problem)
    try:
        outputs = []
        for _ in range(3):
            layer.stage_inputs(
                problem["inputs"], compile_tokens_per_rank=max(tokens_by_rank)
            )
            outputs.append(layer.compute_staged(output=None).clone())
        torch.cuda.synchronize()
        dist.barrier()
        assert torch.isfinite(outputs[0]).all()
        for output in outputs[1:]:
            torch.testing.assert_close(outputs[0], output, atol=0.0, rtol=0.0)
    finally:
        layer.destroy()


def _assert_green_context_bindings(layer, bucket):
    """Do not mistake a successful set-params call for a retained context."""
    from cuda.bindings import driver as cuda

    frontend = next(
        value
        for value in layer._workspace._frontends.values()
        if value.compile_bucket == bucket
    )
    graph = frontend._compiled.graph
    contexts = []
    for green in graph._green_contexts:
        error, context = cuda.cuCtxFromGreenCtx(green)
        assert int(error) == 0
        contexts.append(int(context))
    error, _, count = cuda.cuGraphGetNodes(graph._graph, 0)
    assert int(error) == 0
    error, nodes, _ = cuda.cuGraphGetNodes(graph._graph, count)
    assert int(error) == 0
    bound = set()
    for node in nodes:
        error, kind = cuda.cuGraphNodeGetType(node)
        assert int(error) == 0
        if kind != cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
            continue
        error, params = cuda.cuGraphKernelNodeGetParams(node)
        assert int(error) == 0
        if int(params.ctx) in contexts:
            bound.add(int(params.ctx))
    assert bound == set(contexts)


@pytest.mark.gpu_4
@pytest.mark.arch_sm120
def test_sm120_nvfp4_model_shape_shared_workspace_graph_replay():
    """DSv4-sized persistent grids must retain both Green Context bindings."""
    import torch.distributed as dist

    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("requires exactly four ranks")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    problem = _problem(
        rank,
        4,
        tokens=(8192, 4097, 1, 0)[rank],
        capacity=8192,
        top_k=6,
        hidden=4096,
        intermediate=2048,
        experts=256,
    )
    other = _problem(
        rank + 17,
        4,
        tokens=0,
        capacity=8192,
        top_k=6,
        hidden=4096,
        intermediate=2048,
        experts=256,
    )
    layers = [
        _make_layer(rank, 4, problem),
        _make_layer(rank, 4, other, input_norm_const=0.25),
    ]
    try:
        source = problem["inputs"]
        for requested in (7, 16, 32, 64, 128, 168, 256, 320, 384, 8192, 64, 0, 7):
            from flashinfer.moe_ep import MoEEpTensors

            bucket = max(7, requested)
            rows = min(source.hidden_states.shape[0], requested)
            inputs = MoEEpTensors(
                hidden_states=source.hidden_states[:rows],
                topk_ids=source.topk_ids[:rows],
                topk_weights=source.topk_weights[:rows],
            )
            expected = []
            for layer in layers:
                layer.stage_inputs(inputs, compile_tokens_per_rank=bucket)
                expected.append(layer.compute_staged(output=None).clone())
                torch.cuda.synchronize()
                _assert_green_context_bindings(layer, bucket)
            workspace = layers[0]._workspace
            assert workspace is layers[1]._workspace
            graphs = [f._compiled.graph for f in workspace._frontends.values()]
            resources = tuple(workspace._green_resources.values())
            assert len(resources) < len(graphs)
            for compiled_graph in graphs:
                assert not compiled_graph._owns_green_contexts
                assert any(
                    compiled_graph._green_contexts == pair.green_contexts
                    for pair in resources
                )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = []
                for layer in layers:
                    layer.stage_inputs(inputs, compile_tokens_per_rank=bucket)
                    captured.append(layer.compute_staged(output=None).clone())
            for _ in range(100):
                graph.replay()
            torch.cuda.synchronize()
            for actual, reference in zip(captured, expected):
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            graph.reset()
            dist.barrier()
    finally:
        resources = tuple(layers[0]._workspace._green_resources.values())
        for layer in reversed(layers):
            layer.destroy()
        assert all(pair._closed for pair in resources)


@pytest.mark.gpu_4
@pytest.mark.arch_sm120
def test_sm120_nvfp4_fused_stage_matches_cuda_prequantized():
    """Layer calibration and bucket shrink/grow must preserve the old path."""
    import torch.distributed as dist
    from flashinfer import fp4_quantize
    from flashinfer.moe_ep import MoEEpTensors

    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("requires exactly four ranks")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    problem = _problem(
        rank,
        4,
        tokens=(256, 170, 1, 0)[rank],
        capacity=320,
        top_k=6,
        hidden=4096,
        intermediate=2048,
        experts=256,
    )
    for norm in (0.001, 0.127, 0.25, 17.0):
        fused = _make_layer(rank, 4, problem, input_norm_const=norm)
        reference = _make_layer(
            rank, 4, problem, input_norm_const=norm, quantize_input=False
        )
        try:
            assert fused._workspace is reference._workspace
            source = problem["inputs"]
            for requested, bucket in ((128, 128), (53, 64), (129, 168), (0, 7), (7, 7)):
                rows = min(len(source.hidden_states), requested)
                inputs = MoEEpTensors(
                    hidden_states=source.hidden_states[:rows],
                    topk_ids=source.topk_ids[:rows],
                    topk_weights=source.topk_weights[:rows],
                )
                if rows:
                    x, sf = fp4_quantize(
                        inputs.hidden_states,
                        torch.tensor(norm, device="cuda"),
                        sf_vec_size=16,
                        is_sf_swizzled_layout=False,
                        backend="cuda",
                    )
                else:
                    x = torch.empty((0, 2048), dtype=torch.uint8, device="cuda")
                    sf = torch.empty((0, 256), dtype=torch.uint8, device="cuda")
                quantized = MoEEpTensors(
                    hidden_states=x.view(torch.float4_e2m1fn_x2),
                    scales=sf.view(torch.float8_e4m3fn),
                    topk_ids=inputs.topk_ids,
                    topk_weights=inputs.topk_weights,
                )
                reference.stage_inputs(quantized, compile_tokens_per_rank=bucket)
                expected = reference.compute_staged(output=None).clone()
                fused.stage_inputs(inputs, compile_tokens_per_rank=bucket)
                actual = fused.compute_staged(output=None).clone()
                torch.cuda.synchronize()
                assert torch.all(fused._workspace.topk_ids[rows:bucket] == -1)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fused.stage_inputs(inputs, compile_tokens_per_rank=bucket)
                    captured = fused.compute_staged(output=None).clone()
                for _ in range(10):
                    graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(captured, expected, atol=0, rtol=0)
                graph.reset()
                dist.barrier()
        finally:
            reference.destroy()
            fused.destroy()


@pytest.mark.arch_sm120
def test_sm120_nvfp4_staging_bucket_shrink_grow_padding():
    from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel import (
        stage_inputs,
    )

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    capacity, hidden = 320, 4096
    x = torch.empty((capacity, hidden // 2), dtype=torch.uint8, device="cuda")
    sf = torch.empty((capacity, hidden // 16), dtype=torch.float8_e4m3fn, device="cuda")
    ids = torch.full((capacity, 6), 123, dtype=torch.int64, device="cuda")
    weights = torch.empty((capacity, 6), dtype=torch.float32, device="cuda")
    # No reference staging between calls: a small view must not hide stale
    # routes above its end when the next selected compile bucket grows.
    for rows, bucket in (
        (128, 128),
        (53, 64),
        (129, 168),
        (200, 256),
        (1, 64),
        (130, 256),
        (0, 7),
    ):
        states = torch.ones((rows, hidden), dtype=torch.bfloat16, device="cuda")
        live_ids = torch.zeros((rows, 6), dtype=torch.int32, device="cuda")
        live_weights = torch.ones((rows, 6), dtype=torch.float32, device="cuda")
        stage_inputs(
            states,
            live_weights,
            live_ids,
            x[:bucket].view(torch.float4_e2m1fn_x2),
            sf[:bucket],
            ids[:bucket],
            weights[:bucket],
            quantize_input=True,
            scales=None,
        )
        torch.cuda.synchronize()
        assert torch.all(ids[:rows] == 0)
        assert torch.all(ids[rows:bucket] == -1)


@pytest.mark.arch_sm120
@pytest.mark.parametrize("hidden,offset", [(4096, False), (1056, False), (4096, True)])
def test_sm120_nvfp4_cuda_quantizer_boundary_parity(hidden, offset):
    from flashinfer import fp4_quantize
    from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel import (
        stage_inputs,
    )

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    n = 7
    storage = torch.linspace(
        -2, 2, n * hidden + int(offset), device="cuda", dtype=torch.bfloat16
    )
    h = storage[int(offset) :].view(n, hidden)
    x = torch.empty((n, hidden // 2), dtype=torch.uint8, device="cuda")
    sf = torch.empty(
        (n, ((hidden // 16 + 3) // 4) * 4), dtype=torch.float8_e4m3fn, device="cuda"
    )
    ids = torch.zeros((n, 6), dtype=torch.int32, device="cuda")
    weights = torch.ones((n, 6), dtype=torch.float32, device="cuda")
    staged_ids = torch.empty_like(ids, dtype=torch.int64)
    staged_weights = torch.empty_like(weights)
    for norm in (0.001, 0.127, 328.3053283691406, 1303.272705078125):
        qref, sref = fp4_quantize(
            h.clone(),
            torch.tensor(norm, device="cuda"),
            sf_vec_size=16,
            is_sf_swizzled_layout=False,
            backend="cuda",
        )

        def run():
            stage_inputs(
                h,
                weights,
                ids,
                x.view(torch.float4_e2m1fn_x2),
                sf,
                staged_ids,
                staged_weights,
                quantize_input=True,
                scales=None,
                norm_const=norm,
            )

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(x, qref.view(torch.uint8))
        assert torch.equal(
            sf[:, : hidden // 16].view(torch.uint8), sref.view(torch.uint8)
        )
        if hidden // 16 < sf.shape[1]:
            assert torch.all(sf[:, hidden // 16 :].float() == 0)
        graph.reset()


@pytest.mark.gpu_4
@pytest.mark.arch_sm120
def test_sm120_nvfp4_lazy_bucket_after_inflight_outer_graph() -> None:
    """Cold symmetric allocation must not race queued peer-dependent graphs."""
    import torch.distributed as dist

    from flashinfer.moe_ep import MoEEpTensors
    from flashinfer.moe_ep.kernel_src.sm120.nvfp4_split_cutedsl_megakernel.shim.runtime import (
        MegaMoESm120Nvfp4Frontend,
    )

    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("requires exactly four ranks")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    problem = _problem(
        rank,
        4,
        tokens=128,
        capacity=8192,
        top_k=6,
        hidden=4096,
        intermediate=2048,
        experts=256,
    )
    layer = _make_layer(rank, 4, problem)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def inputs(rows):
        source = problem["inputs"]
        return MoEEpTensors(
            hidden_states=source.hidden_states[:rows],
            topk_ids=source.topk_ids[:rows],
            topk_weights=source.topk_weights[:rows],
        )

    try:
        with torch.cuda.stream(stream):
            small = inputs((7, 4, 1, 0)[rank])
            layer.stage_inputs(small, compile_tokens_per_rank=16)
            workspace = layer._workspace
            # Exercise many per-layer graphs without duplicating large weights.
            outputs = [torch.empty_like(layer.output_buffer) for _ in range(43)]

            def forward(tensors, bucket):
                result = []
                for output in outputs:
                    layer.stage_inputs(tensors, compile_tokens_per_rank=bucket)
                    result.append(layer.compute_staged(output=output).clone())
                return result

            for bucket in (16, 7, 8192):
                expected_small = forward(small, bucket)
                torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured_small = forward(small, 8192)

            for bucket in (32, 64, 128):
                ragged = inputs((bucket - 3, 7, 1, 0)[rank])
                # Precompile staging and the CPU plan, but not execution storage.
                layer.stage_inputs(ragged, compile_tokens_per_rank=bucket)
                planner = MegaMoESm120Nvfp4Frontend(
                    workspace,
                    compile_bucket=bucket,
                    control_group=workspace._control_group,
                )
                planner._ensure_plan(bucket)
                assert bucket not in workspace._storages
                torch.cuda.synchronize()
                dist.barrier()
                if rank == 0:
                    torch.cuda._sleep(5_000_000_000)
                graph.replay()
                graph.replay()
                assert not stream.query()

                actual = forward(ragged, bucket)
                torch.cuda.synchronize()
                reference = forward(ragged, bucket)
                torch.cuda.synchronize()
                for value, expected in zip(actual, reference):
                    assert torch.isfinite(value).all()
                    torch.testing.assert_close(value, expected, atol=0, rtol=0)
                for value, expected in zip(captured_small, expected_small):
                    torch.testing.assert_close(value, expected, atol=0, rtol=0)
            graph.reset()
    finally:
        layer.destroy()
