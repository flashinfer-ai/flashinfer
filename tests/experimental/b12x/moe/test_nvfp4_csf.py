"""Compressed scales preserve native NVFP4 expert output under graph replay."""

import b12x.moe._shared.kernels.w4a16.kernel.fused_launch as w4a16_fused_launch


from dataclasses import replace

import numpy as np
import pytest
import torch

from b12x.moe import fused_moe as moe
from b12x.preparation import PreparationSession, PreparedCall
from .test_nvfp4_phase_kernels import _build_domain
from ..conftest import require_b12x


def compress_fixture(swizzled, rows, columns, raw=False):
    """Build valid byte-window records independently from GPU expansion."""
    e = swizzled.shape[0]
    logical = swizzled.view(torch.uint8).reshape(e, rows // 128, columns // 4, 32, 4, 4)
    logical = logical.permute(0, 1, 4, 3, 2, 5).contiguous().reshape(e, rows, columns)
    fixed, exceptions = [], []
    for source in logical.cpu().numpy():
        base = np.minimum(source.min(axis=1), 240).astype(np.uint8)
        offsets = source.astype(np.int16) - base[:, None]
        outside = offsets > 15
        offsets[outside] = 0
        offsets = offsets.astype(np.uint8)
        packed = offsets[:, ::2] | (offsets[:, 1::2] << 4)
        fixed.append(
            np.concatenate((base.reshape(-1, 16), packed.reshape(rows // 16, -1)), 1)
        )
        pos = np.flatnonzero(outside).astype(np.uint32)
        exceptions.append(pos | (source.ravel()[pos].astype(np.uint32) << 24))
    from b12x.loader._csf_blocks import export_csf_blocks

    return moe.CsfBlockPlanes(
        tuple(
            export_csf_blocks(f, x, rows=rows, columns=columns, quantization="nvfp4")
            for f, x in zip(fixed, exceptions, strict=True)
        )
    )


@pytest.mark.parametrize("raw", ["blocks"])
@pytest.mark.parametrize("tokens", [1, 17, 33])
@pytest.mark.parametrize("activation_mode", ["a4", "a16"])
@pytest.mark.parametrize("n", [64, 128, 192, 320])
def test_native_expert_output_and_shared_scratch_poisoned_replay(
    tokens,
    activation_mode,
    raw,
    n,
    dynamic_override=None,
    autotune=False,
    deterministic=True,
    tile_m=16,
    after_plan=None,
    inline_scales=False,
    mma_warps=None,
    execution_override=None,
):
    device = require_b12x()
    e, h, topk = 8, 256, 2
    domain = _build_domain(E=e, K=h, n=n, m=tokens, top_k=topk, seed=931)
    weight_plan = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=moe.ActivationSpec(
            mode=activation_mode,
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
            swiglu_limit=10.0,
        ),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=n),
    )
    s13 = domain["w13_sfb"].view(torch.float8_e4m3fn).view(e, 2 * n, h // 16)
    s2 = domain["w2_sfb"].view(torch.float8_e4m3fn).view(e, h, n // 16)
    one = torch.ones(e, device=device)
    packed = moe.PackedWeights(
        w13=domain["w13_packed"],
        w2=domain["w2_packed"],
        w13_block_scales=s13,
        w2_block_scales=s2,
        w13_global_scales=one,
        w2_global_scales=one,
        input_scale=one,
        intermediate_scale=one,
        immutable_input_scales=True,
    )
    buffers = (torch.empty_like(s13), torch.empty_like(s2))
    compressed = moe.Nvfp4CsfWeights(
        packed=replace(
            packed,
            w13=packed.w13.clone(),
            w2=packed.w2.clone(),
            w13_block_scales=buffers[0],
            w2_block_scales=buffers[1],
        ),
        w13_scales=compress_fixture(s13, 2 * n, h // 16, raw=raw),
        w2_scales=compress_fixture(s2, h, n // 16, raw=raw),
    )
    experts = [
        moe.prepare_weights(plan=weight_plan, weights=w) for w in (packed, compressed)
    ]
    if activation_mode == "a16":
        assert experts[1].plan._impl.w4a16_scale_format == "e4m3_k16"
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig

    override = (
        None
        if dynamic_override is None
        else MoeDecodeConfig(
            backend="dynamic",
            route_planner="internal",
            max_active_clusters=None,
            dynamic_tile_m=tile_m,
            dynamic_mma_warps=mma_warps,
            dynamic_route_mode="grouped",
        )
    )
    if inline_scales and activation_mode == "a16":
        override = MoeDecodeConfig(
            backend="w4a16",
            route_planner="internal",
            max_active_clusters=None,
            w4a16_route_mode="packed",
        )
    if execution_override is not None:
        override = execution_override
    expansion = (
        moe.plan_scale_expansion(experts[1], routes=tokens * topk)
        if raw == "blocks"
        else None
    )
    plans = [
        moe.plan_execution(
            experts=owner,
            capacity=moe.ExecutionCapacity(max_tokens=tokens, top_k=topk),
            invocation={"fast_math": False},
            routing=moe.RoutingSpec(deterministic_output=deterministic),
            override=(
                replace(override, csf_inline_scales=True)
                if inline_scales and i == 1
                else override
            ),
            scale_expansion=expansion if i == 1 else None,
        )
        for i, owner in enumerate(experts)
    ]
    if after_plan is not None:
        after_plan()
    source, ids, probabilities = domain["x"], domain["topk_ids"], domain["topk_weights"]
    activation_source, route_source = source.clone(), ids.clone()

    def prepare_expansion(state):
        active = torch.empty(e, dtype=torch.int32, device=device)
        binding = state.bind(ids=ids, out13=buffers[0], out2=buffers[1], scratch=active)
        return PreparedCall(
            run=binding.run,
            produce=lambda: ids.copy_(route_source),
            owners=(active, binding),
        )

    def prepare(state):
        scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=device)
            for s in state.scratch.scratch_specs()
        )
        output = torch.empty_like(source)
        binding = state.bind(
            a=source,
            topk_ids=ids,
            topk_weights=probabilities,
            output=output,
            scratch=scratch,
            input_scales_static=True,
        )
        return PreparedCall(
            run=lambda: state.run(binding),
            output=output,
            produce=lambda: source.copy_(activation_source),
            owners=(scratch, binding),
        )

    with PreparationSession(
        device=device, autotune=autotune, compile_workers=0
    ) as session:
        session.prepare(
            (
                *(
                    (
                        expansion.request(
                            name="scale-expansion",
                            prepare_call=prepare_expansion,
                            benchmark_call=prepare_expansion,
                        ),
                    )
                    if expansion is not None
                    else ()
                ),
                *tuple(
                    p.request(
                        name=f"nvfp4-scale-storage-{i}",
                        prepare_call=prepare,
                        benchmark_call=prepare,
                        dependencies=("scale-expansion",)
                        if i == 1 and expansion
                        else (),
                    )
                    for i, p in enumerate(plans)
                ),
            )
        )
        if expansion is not None:
            assert plans[
                1
            ].prepared.state.execution_experts.nvfp4_csf.first.groups_per_task == (
                expansion.selection.config.groups_per_task
            )
        owners, outputs, bindings, graphs = [], [], [], []
        for plan in plans:
            scratch = tuple(
                torch.empty(s.shape, dtype=s.dtype, device=device)
                for s in plan.scratch_specs()
            )
            output = torch.empty_like(source)
            binding = moe.bind(
                plan,
                a=source,
                topk_ids=ids,
                topk_weights=probabilities,
                output=output,
                scratch=scratch,
                input_scales_static=True,
            )
            owners.append(scratch)
            outputs.append(output)
            bindings.append(binding)
        if inline_scales:
            assert not moe.uses_expanded_nvfp4_scales(plans[1], num_tokens=tokens)
        session.freeze()
        for binding in bindings:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                moe.run(binding=binding)
            graphs.append(graph)
        for _ in range(3):
            source.neg_()
            ids.add_(1).remainder_(e)
            for buffer in buffers:
                buffer.view(torch.uint8).fill_(0x7F)
            for output in outputs:
                output.fill_(float("nan"))
            for binding in bindings:
                if binding.barrier_count is not None:
                    binding.barrier_count.fill_(123)
                    binding.barrier_epoch.fill_(-42)
            allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            assert (
                torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
            )
            assert torch.isfinite(outputs[0]).all() and torch.count_nonzero(outputs[0])
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            if inline_scales:
                assert all(torch.all(b.view(torch.uint8) == 0x7F) for b in buffers)
            else:
                assert not torch.all(buffers[0].view(torch.uint8) == 0x7F)
        for graph in graphs:
            graph.reset()


@pytest.mark.parametrize("tokens,n", [(1, 64), (17, 320)])
def test_scale_decoder_override_preserves_expert_graph_output(tokens, n):
    test_native_expert_output_and_shared_scratch_poisoned_replay(
        tokens, "a4", True, n, dynamic_override=True
    )


@pytest.mark.parametrize("tile_m", [16, 32, 64, 128])
def test_split_gate_scale_operands_preserve_native_output_for_consumer_counts(tile_m):
    test_native_expert_output_and_shared_scratch_poisoned_replay(
        33,
        "a4",
        True,
        320,
        dynamic_override=True,
        tile_m=tile_m,
    )


@pytest.mark.parametrize("deterministic", [False, True])
def test_indexed_scale_programs_are_declared_for_preparation(deterministic):
    test_native_expert_output_and_shared_scratch_poisoned_replay(
        1,
        "a4",
        True,
        128,
        dynamic_override=False,
        autotune=True,
        deterministic=deterministic,
    )


@pytest.mark.parametrize("n", [128, 320])
def test_large_a16_calls_expand_compressed_scales_per_call(n, monkeypatch):
    test_native_expert_output_and_shared_scratch_poisoned_replay(300, "a16", False, n)


def test_a16_expanded_scales_reuse_the_planned_launch_for_live_counts():
    """One capacity plan serves smaller live counts without resolving new kernels."""
    device = require_b12x()
    e, h, n, topk, capacity = 8, 256, 128, 2, 64
    domain = _build_domain(E=e, K=h, n=n, m=capacity, top_k=topk, seed=977)
    weight_plan = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=moe.ActivationSpec(
            mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=n),
    )
    s13 = domain["w13_sfb"].view(torch.float8_e4m3fn).view(e, 2 * n, h // 16)
    s2 = domain["w2_sfb"].view(torch.float8_e4m3fn).view(e, h, n // 16)
    one = torch.ones(e, device=device)
    packed = moe.PackedWeights(
        w13=domain["w13_packed"],
        w2=domain["w2_packed"],
        w13_block_scales=s13,
        w2_block_scales=s2,
        w13_global_scales=one,
        w2_global_scales=one,
        input_scale=one,
        intermediate_scale=one,
        immutable_input_scales=True,
    )
    buffers = (torch.empty_like(s13), torch.empty_like(s2))
    # Native preparation repacks its package in place: compress the source first.
    compressed = moe.Nvfp4CsfWeights(
        packed=replace(
            packed,
            w13=packed.w13.clone(),
            w2=packed.w2.clone(),
            w13_block_scales=buffers[0],
            w2_block_scales=buffers[1],
        ),
        w13_scales=compress_fixture(s13, 2 * n, h // 16),
        w2_scales=compress_fixture(s2, h, n // 16),
    )
    experts = (
        moe.prepare_weights(plan=weight_plan, weights=packed),
        moe.prepare_weights(plan=weight_plan, weights=compressed),
    )
    plans = tuple(
        moe.plan_execution(
            experts=owner,
            capacity=moe.ExecutionCapacity(max_tokens=capacity, top_k=topk),
            invocation={"fast_math": False},
            routing=moe.RoutingSpec(deterministic_output=True),
        )
        for owner in experts
    )
    x, ids, weights = domain["x"], domain["topk_ids"], domain["topk_weights"]

    def bind(plan, rows):
        scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=device)
            for s in plan.scratch_specs()
        )
        output = torch.empty_like(x[:rows])
        binding = moe.bind(
            plan,
            a=x[:rows],
            topk_ids=ids[:rows],
            topk_weights=weights[:rows],
            output=output,
            scratch=scratch,
            input_scales_static=True,
        )
        return binding, output, scratch

    def prepare(state):
        scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=device)
            for s in state.scratch.scratch_specs()
        )
        output = torch.empty_like(x)
        binding = state.bind(
            a=x,
            topk_ids=ids,
            topk_weights=weights,
            output=output,
            scratch=scratch,
            input_scales_static=True,
        )
        return PreparedCall(
            run=lambda: state.run(binding), output=output, owners=(scratch, binding)
        )

    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare(
            tuple(
                p.request(name=f"csf-live-{i}", prepare_call=prepare)
                for i, p in enumerate(plans)
            )
        )
        session.freeze()
        for rows in (1, 7, 33, capacity):
            native, compressed = (bind(plan, rows) for plan in plans)
            for buffer in buffers:
                buffer.view(torch.uint8).fill_(0x7F)
            moe.run(binding=native[0])
            moe.run(binding=compressed[0])
            torch.cuda.synchronize()
            assert torch.isfinite(native[1]).all() and torch.count_nonzero(native[1])
            torch.testing.assert_close(compressed[1], native[1], rtol=0, atol=0)
            assert all(not torch.all(buf.view(torch.uint8) == 0x7F) for buf in buffers)


@pytest.mark.parametrize(
    "activation_mode,backend,tokens",
    [("a16", None, 1600), ("a4", None, 1), ("a4", None, 33), ("a4", "dynamic", 33)],
)
def test_prefetched_scales_replace_the_per_call_expansion(
    activation_mode, backend, tokens
):
    """expand_scales() fills the shared scratch for a later scales_expanded call."""
    device = require_b12x()
    e, h, n, topk = 8, 256, 128, 2
    domain = _build_domain(E=e, K=h, n=n, m=tokens, top_k=topk, seed=937)
    weight_plan = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=moe.ActivationSpec(
            mode=activation_mode,
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
            swiglu_limit=10.0,
        ),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=n),
    )
    s13 = domain["w13_sfb"].view(torch.float8_e4m3fn).view(e, 2 * n, h // 16)
    s2 = domain["w2_sfb"].view(torch.float8_e4m3fn).view(e, h, n // 16)
    one = torch.ones(e, device=device)
    packed = moe.PackedWeights(
        w13=domain["w13_packed"],
        w2=domain["w2_packed"],
        w13_block_scales=s13,
        w2_block_scales=s2,
        w13_global_scales=one,
        w2_global_scales=one,
        input_scale=one,
        intermediate_scale=one,
        immutable_input_scales=True,
    )
    buffers = (torch.empty_like(s13), torch.empty_like(s2))
    compressed = moe.Nvfp4CsfWeights(
        packed=replace(
            packed,
            w13=packed.w13.clone(),
            w2=packed.w2.clone(),
            w13_block_scales=buffers[0],
            w2_block_scales=buffers[1],
        ),
        w13_scales=compress_fixture(s13, 2 * n, h // 16),
        w2_scales=compress_fixture(s2, h, n // 16),
    )
    experts = [
        moe.prepare_weights(plan=weight_plan, weights=w) for w in (packed, compressed)
    ]
    assert not moe.expand_scales(experts[0])
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig

    override = (
        None
        if backend is None
        else MoeDecodeConfig(
            backend="dynamic",
            route_planner="internal",
            max_active_clusters=None,
            dynamic_tile_m=16,
            dynamic_route_mode="grouped",
        )
    )
    plans = [
        moe.plan_execution(
            experts=owner,
            capacity=moe.ExecutionCapacity(max_tokens=tokens, top_k=topk),
            invocation={"fast_math": False},
            override=override,
            routing=moe.RoutingSpec(deterministic_output=True),
        )
        for owner in experts
    ]
    source, ids, probabilities = domain["x"], domain["topk_ids"], domain["topk_weights"]
    scratches = [
        tuple(
            torch.empty(s.shape, dtype=s.dtype, device=device)
            for s in p.scratch_specs()
        )
        for p in plans
    ]

    def call(index, **kwargs):
        output = torch.full_like(source, float("nan"))
        binding = moe.bind(
            plans[index],
            a=source,
            topk_ids=ids,
            topk_weights=probabilities,
            output=output,
            scratch=scratches[index],
            input_scales_static=True,
            **kwargs,
        )
        moe.run(binding=binding)
        torch.cuda.synchronize()
        return output

    def prepare(state):
        scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=device)
            for s in state.scratch.scratch_specs()
        )
        output = torch.empty_like(source)
        binding = state.bind(
            a=source,
            topk_ids=ids,
            topk_weights=probabilities,
            output=output,
            scratch=scratch,
            input_scales_static=True,
        )
        return PreparedCall(
            run=lambda: state.run(binding), output=output, owners=(scratch, binding)
        )

    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare(
            tuple(
                p.request(name=f"nvfp4-scale-prefetch-{i}", prepare_call=prepare)
                for i, p in enumerate(plans)
            )
        )
        reference = call(0)
        assert torch.isfinite(reference).all() and torch.count_nonzero(reference)
        poison = lambda: [b.view(torch.uint8).fill_(0x7F) for b in buffers]  # noqa: E731
        poison()
        stale = call(1, scales_expanded=True)
        if plans[1].selection.config.backend == "micro":
            # Small FP4-activation calls run the micro kernel, whose barrier
            # reset is fused into the expansion: they expand regardless.
            torch.testing.assert_close(stale, reference, rtol=0, atol=0)
        else:
            assert not torch.equal(stale, reference)
        # ... and a prefetch on a side stream restores the native result.
        poison()
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            assert moe.expand_scales(experts[1])
        torch.cuda.current_stream(device).wait_stream(side)
        torch.testing.assert_close(
            call(1, scales_expanded=True), reference, rtol=0, atol=0
        )
        # Without the flag the call still expands its own routed experts.
        poison()
        torch.testing.assert_close(call(1), reference, rtol=0, atol=0)

        if torch.cuda.device_count() > 1:
            poison()
            torch.cuda.synchronize(device)
            with torch.cuda.device(1):
                assert moe.expand_scales(experts[1])
                assert torch.cuda.current_device() == 1
            torch.cuda.synchronize(device)
            torch.testing.assert_close(
                call(1, scales_expanded=True), reference, rtol=0, atol=0
            )

        output = torch.empty_like(source)
        binding = moe.bind(
            plans[1],
            a=source,
            topk_ids=ids,
            topk_weights=probabilities,
            output=output,
            scratch=scratches[1],
            input_scales_static=True,
            scales_expanded=True,
        )
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            moe.expand_scales(experts[1])
            moe.run(binding=binding)
        for _ in range(3):
            poison()
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert (
                torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
            )
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
        graph.reset()


@pytest.mark.parametrize("tokens", [1, 33, 128])
@pytest.mark.parametrize("mode", ["a4", "a16"])
@pytest.mark.parametrize("width", [192, 320])
def test_block_scales_use_native_expert_math_and_shared_scratch(tokens, mode, width):
    test_native_expert_output_and_shared_scratch_poisoned_replay(
        tokens=tokens,
        activation_mode=mode,
        raw="blocks",
        n=width,
        autotune=(tokens == 33 and mode == "a4" and width == 192),
    )


@pytest.mark.parametrize("tile_m,mma_warps", [(16, None), (16, 4), (32, None)])
@pytest.mark.parametrize("n", [64, 320])
def test_inline_scales_bypass_expansion_under_frozen_graph_replay(tile_m, mma_warps, n):
    test_native_expert_output_and_shared_scratch_poisoned_replay(
        8,
        "a4",
        "blocks",
        n,
        dynamic_override=True,
        tile_m=tile_m,
        inline_scales=True,
        mma_warps=mma_warps,
    )


@pytest.mark.parametrize("tokens", [4, 32])
@pytest.mark.parametrize("n", [128, 320])
def test_w4a16_inline_scales_bypass_expansion_under_frozen_graph_replay(tokens, n):
    test_native_expert_output_and_shared_scratch_poisoned_replay(
        tokens, "a16", "blocks", n, inline_scales=True
    )


@pytest.mark.parametrize("tokens", [1, 4])
def test_w4a16_direct_inline_poisoned_graph(tokens):
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig

    test_native_expert_output_and_shared_scratch_poisoned_replay(
        tokens,
        "a16",
        "blocks",
        128,
        inline_scales=True,
        execution_override=MoeDecodeConfig(
            backend="w4a16",
            route_planner="internal",
            max_active_clusters=None,
            w4a16_route_mode="direct",
        ),
    )


def test_w4a16_wide_inline_poisoned_graph(monkeypatch):
    compile_moe = w4a16_fused_launch.compile_w4a16_fused_moe

    def compile_wide(*args, **kwargs):
        kwargs["force_tile_config"] = (64, 256, 64, 256)
        return compile_moe(*args, **kwargs)

    monkeypatch.setattr(w4a16_fused_launch, "compile_w4a16_fused_moe", compile_wide)
    test_native_expert_output_and_shared_scratch_poisoned_replay(
        32, "a16", "blocks", 256, inline_scales=True
    )
